import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_lambda_over_m_l711_71195

/-- Given vectors a and b, prove that the range of λ/m is [-6, 1] -/
theorem range_of_lambda_over_m (l m α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (l + 2, l^2 - Real.cos α ^ 2))
  (hb : b = (m, m / 2 + Real.sin α))
  (h_eq : a = (2 : ℝ) • b) :
  ∃ (x : ℝ), x = l / m ∧ -6 ≤ x ∧ x ≤ 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_lambda_over_m_l711_71195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l711_71198

-- Define the complex number z
noncomputable def z (b : ℝ) : ℂ := Complex.I * b

-- Theorem statement
theorem complex_number_properties
  (b : ℝ)
  (h1 : b ≠ 0)
  (h2 : ∃ (r : ℝ), (z b - 2) / (1 + Complex.I) = r) :
  (Complex.abs (z b) = 2) ∧
  (∀ m : ℝ, (Complex.re ((m + z b)^2) > 0 ∧ Complex.im ((m + z b)^2) < 0) ↔ m > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l711_71198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_chord_l711_71185

/-- Given a circle with center O, perpendicular diameters AB and CD, and chord DF intersecting AB at E,
    prove that if DE = 8 and EF = 4, then the area of the circle is 32π. -/
theorem circle_area_from_chord (O A B C D E F : ℝ × ℝ) : 
  let circle := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = (A.1 - O.1)^2 + (A.2 - O.2)^2}
  (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle) →
  ((A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)) →
  (F ∈ circle) →
  ((D.1 - E.1)^2 + (D.2 - E.2)^2 = 64) →
  ((E.1 - F.1)^2 + (E.2 - F.2)^2 = 16) →
  Real.pi * ((A.1 - O.1)^2 + (A.2 - O.2)^2) = 32 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_chord_l711_71185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_clean_time_is_4_hours_l711_71132

noncomputable def john_clean_time (nick_time : ℝ) : ℝ := (2/3) * nick_time

theorem john_clean_time_is_4_hours :
  ∀ (nick_time : ℝ),
  nick_time > 0 →
  john_clean_time nick_time > 0 →
  (1 / john_clean_time nick_time + 1 / nick_time = 1 / 3.6) →
  john_clean_time nick_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_clean_time_is_4_hours_l711_71132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l711_71149

noncomputable section

/-- The function f(x) = sin(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- The function g(x) = f(x) - 2cos²(πx/8) + 1 -/
noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := f ω φ x - 2 * (Real.cos (Real.pi * x / 8))^2 + 1

theorem function_and_range_proof 
  (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : |φ| ≤ Real.pi/2)
  (h_max : f ω φ (8/3) = 1)
  (h_zero : f ω φ (14/3) = 0) :
  (∀ x, f ω φ x = Real.sin (Real.pi * x / 4 - Real.pi / 6)) ∧ 
  (∀ x ∈ Set.Icc (2/3) 2, g ω φ x ∈ Set.Icc (-Real.sqrt 3 / 2) (Real.sqrt 3 / 2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l711_71149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_C_completes_in_four_days_l711_71128

/-- The time it takes for worker C to complete a piece of work -/
noncomputable def time_for_C (days_for_A : ℝ) (efficiency_B_vs_A : ℝ) (efficiency_C_vs_B : ℝ) : ℝ :=
  days_for_A / (efficiency_B_vs_A * efficiency_C_vs_B)

/-- Theorem stating that C completes the work in 4 days -/
theorem worker_C_completes_in_four_days :
  time_for_C 12 1.5 2 = 4 := by
  -- Unfold the definition of time_for_C
  unfold time_for_C
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_C_completes_in_four_days_l711_71128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_90_l711_71135

/-- The speed of a train in km/hr, given its length in meters and the time it takes to cross a pole. -/
noncomputable def train_speed (length_m : ℝ) (time_s : ℝ) : ℝ :=
  (length_m / 1000) / (time_s / 3600)

/-- Theorem stating that a train with a length of 250 meters crossing a pole in 10 seconds has a speed of 90 km/hr. -/
theorem train_speed_is_90 :
  train_speed 250 10 = 90 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp [div_div]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_90_l711_71135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flammable_ice_scientific_notation_l711_71155

/-- Expresses a number in scientific notation -/
noncomputable def scientific_notation (n : ℝ) : ℝ × ℤ :=
  let exp := Int.floor (Real.log n / Real.log 10)
  let coef := n / (10 : ℝ) ^ exp
  (coef, exp)

/-- The number of cubic meters of flammable ice reserves -/
def flammable_ice_reserves : ℝ := 19.4e9

theorem flammable_ice_scientific_notation :
  scientific_notation flammable_ice_reserves = (1.94, 10) := by
  sorry

#eval flammable_ice_reserves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flammable_ice_scientific_notation_l711_71155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_inequality_l711_71102

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Iic 0))
  (h_ineq : f a ≤ f 2) :
  a ≥ 2 ∨ a ≤ -2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_inequality_l711_71102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_sum_37_l711_71142

def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 37 ∧ 
  (n.digits 10).Nodup

def is_largest_valid_number (n : ℕ) : Prop :=
  is_valid_number n ∧
  ∀ m : ℕ, is_valid_number m → m ≤ n

theorem largest_number_with_sum_37 :
  is_largest_valid_number 976543210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_sum_37_l711_71142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_union_l711_71108

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 2*x + 8)

-- Define the domain of f
def domain : Set ℝ := { x | -4 ≤ x ∧ x ≤ 2 }

-- Define the range of f
def range : Set ℝ := { y | 0 ≤ y ∧ y ≤ 3 }

-- Theorem statement
theorem domain_range_union :
  domain ∪ range = Set.Icc (-4 : ℝ) 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_union_l711_71108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_and_perimeter_l711_71183

/-- A regular hexadecagon inscribed in a circle -/
structure RegularHexadecagon where
  radius : ℝ
  sides : Nat
  is_regular : sides = 16

/-- The area of a regular hexadecagon inscribed in a circle -/
noncomputable def area (h : RegularHexadecagon) : ℝ :=
  16 * (1/2 * h.radius^2 * Real.sin (22.5 * Real.pi / 180))

/-- The perimeter of a regular hexadecagon inscribed in a circle -/
noncomputable def perimeter (h : RegularHexadecagon) : ℝ :=
  16 * (2 * h.radius * Real.sin (11.25 * Real.pi / 180))

/-- Theorem stating the approximate area and perimeter of a regular hexadecagon -/
theorem hexadecagon_area_and_perimeter (h : RegularHexadecagon) :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (abs (area h - 3.0616 * h.radius^2) < ε) ∧ 
  (abs (perimeter h - 6.2432 * h.radius) < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_and_perimeter_l711_71183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l711_71184

/-- Represents the volume of a geometric shape in cubic centimeters -/
def Volume : Type := ℝ

/-- Calculates the volume of a cone given its radius and height -/
noncomputable def cone_volume (radius height : ℝ) : Volume :=
  (1 / 3) * Real.pi * radius^2 * height

/-- Calculates the volume of a cylinder given its radius and height -/
noncomputable def cylinder_volume (radius height : ℝ) : Volume :=
  Real.pi * radius^2 * height

/-- Represents the properties of the water container problem -/
structure WaterContainer where
  cone_radius : ℝ
  cone_height : ℝ
  cylinder_radius : ℝ

theorem water_height_in_cylinder (container : WaterContainer)
    (h_cone_radius : container.cone_radius = 15)
    (h_cone_height : container.cone_height = 20)
    (h_cylinder_radius : container.cylinder_radius = 10) :
    ∃ (cylinder_height : ℝ),
      cylinder_height = 15 ∧
      cone_volume container.cone_radius container.cone_height =
      cylinder_volume container.cylinder_radius cylinder_height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l711_71184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_39_l711_71173

/-- Definition of the sequence {aₙ} -/
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n + 2

/-- Theorem: The 20th term of the sequence is 39 -/
theorem a_20_equals_39 : a 19 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_39_l711_71173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_to_fill_large_bucket_l711_71191

-- Define the capacities and cost
def small_bucket_capacity : ℚ := 120
def large_bucket_capacity : ℚ := 800
def small_bucket_cost : ℚ := 3

-- Define the function to calculate the number of small buckets needed
def small_buckets_needed (large_capacity small_capacity : ℚ) : ℕ :=
  Int.toNat ⌈large_capacity / small_capacity⌉

-- Define the function to calculate the total cost
def total_cost (num_buckets : ℕ) (bucket_cost : ℚ) : ℚ :=
  (num_buckets : ℚ) * bucket_cost

-- Theorem statement
theorem min_cost_to_fill_large_bucket :
  total_cost (small_buckets_needed large_bucket_capacity small_bucket_capacity) small_bucket_cost = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_to_fill_large_bucket_l711_71191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l711_71182

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.cos (3 * x)) / (Real.cos x)^2

-- State the theorem
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧
  ∀ x, x ∈ Set.Icc 0 (Real.pi / 3) →
  f (Real.cos x) ≥ min :=
by
  -- We'll use -4 as our minimum value
  use -4
  constructor
  · -- Prove that min = -4
    rfl
  · -- Prove that f(cos x) ≥ -4 for all x in [0, π/3]
    intro x hx
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l711_71182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_fourth_quadrant_l711_71134

theorem sin_value_fourth_quadrant (x : ℝ) :
  Real.sin (π / 2 + x) = 5 / 13 →
  (π < x ∧ x < 3 * π / 2) →
  Real.sin x = -12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_fourth_quadrant_l711_71134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_side_larger_triangle_l711_71186

/-- Two similar triangles with specific properties -/
structure SimilarTriangles where
  area_small : ℕ
  area_large : ℕ
  side_small : ℝ
  perimeter_small : ℝ
  area_difference : area_large - area_small = 72
  area_ratio : area_large = 9 * area_small
  side_small_value : side_small = 5
  perimeter_small_value : perimeter_small = 15

/-- The theorem stating the corresponding side of the larger triangle -/
theorem corresponding_side_larger_triangle (t : SimilarTriangles) : 
  ∃ (side_large : ℝ), side_large = 15 ∧ side_large / t.side_small = Real.sqrt (t.area_large / t.area_small) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_side_larger_triangle_l711_71186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l711_71160

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the asymptotes
noncomputable def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 3

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → asymptotes x y) ∧
  eccentricity = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l711_71160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_symmetry_l711_71122

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem translation_symmetry (φ : ℝ) :
  (∃ k : ℤ, 2 * φ + Real.pi / 6 = k * Real.pi + Real.pi / 2) ∧
  (∃ l : ℤ, Real.pi / 6 - 2 * φ = l * Real.pi + Real.pi / 2) →
  (φ = Real.pi / 6 ∨ φ > Real.pi / 6) ∧ (φ = Real.pi / 3 ∨ φ > Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_symmetry_l711_71122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l711_71129

theorem solve_exponential_equation :
  ∃ y : ℝ, (5 : ℝ)^(y + 1) = 625 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l711_71129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l711_71154

-- Define the statements as axioms instead of definitions
axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom statement_D : Prop

-- Define axioms for the correctness of each statement
axiom A_correct : statement_A
axiom B_incorrect : ¬statement_B
axiom C_correct : statement_C
axiom D_correct : statement_D

-- Theorem stating which statements are correct
theorem correct_statements :
  statement_A ∧ ¬statement_B ∧ statement_C ∧ statement_D :=
by
  apply And.intro
  · exact A_correct
  · apply And.intro
    · exact B_incorrect
    · apply And.intro
      · exact C_correct
      · exact D_correct

#check correct_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l711_71154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_million_l711_71163

-- Define the table
def Table := Matrix (Fin 10) (Fin 10) ℝ

-- Define a frog position
def FrogPosition := Fin 10 × Fin 10

-- Define the state of the game
structure GameState where
  table : Table
  frog_positions : Finset FrogPosition
  visible_sum : ℝ

-- Define a valid jump
def valid_jump (f g : FrogPosition) : Prop :=
  (f.1 = g.1 ∧ (f.2 = g.2 + 1 ∨ f.2 = g.2 - 1)) ∨
  (f.2 = g.2 ∧ (f.1 = g.1 + 1 ∨ f.1 = g.1 - 1))

-- Define the theorem
theorem max_sum_is_million (initial_state : GameState) :
  initial_state.frog_positions.card = 5 →
  initial_state.visible_sum = 10 →
  (∀ n : ℕ, ∃ next_state : GameState,
    (∀ f ∈ initial_state.frog_positions, ∃ f' ∈ next_state.frog_positions, valid_jump f f') ∧
    next_state.visible_sum = initial_state.visible_sum * 10^n) →
  (∀ state : GameState,
    (∀ f ∈ initial_state.frog_positions, ∃ f' ∈ state.frog_positions, valid_jump f f') →
    state.visible_sum ≤ 10^6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_million_l711_71163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_walk_l711_71119

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction as a unit vector -/
structure Direction where
  dx : ℝ
  dy : ℝ

/-- Represents a movement in 2D space -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Calculates the final position after a series of movements -/
def finalPosition (start : Point) (movements : List Movement) : Point :=
  movements.foldl
    (fun pos mov => Point.mk
      (pos.x + mov.distance * mov.direction.dx)
      (pos.y + mov.distance * mov.direction.dy))
    start

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem sandy_walk :
  let start := Point.mk 0 0
  let south := Direction.mk 0 (-1)
  let east := Direction.mk 1 0
  let north := Direction.mk 0 1
  let movements := [
    Movement.mk 20 south,
    Movement.mk 20 east,
    Movement.mk 20 north,
    Movement.mk 10 east
  ]
  let final := finalPosition start movements
  distance start final = 30 ∧ final.x = 30 ∧ final.y = 0 := by
  sorry

#eval "Sandy_walk theorem is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_walk_l711_71119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_1010th_term_l711_71115

/-- An arithmetic sequence with specific first four terms -/
def ArithmeticSequence (p r : ℚ) : ℕ → ℚ
  | 0 => p + 2
  | 1 => 13
  | 2 => 4*p - r
  | 3 => 4*p + r
  | n+4 => ArithmeticSequence p r (n+3) + (ArithmeticSequence p r 1 - ArithmeticSequence p r 0)

/-- The 1010th term of the arithmetic sequence is 5691 -/
theorem arithmetic_sequence_1010th_term (p r : ℚ) :
  ArithmeticSequence p r 1009 = 5691 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_1010th_term_l711_71115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_fraction_equality_l711_71137

theorem tan_fraction_equality (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_fraction_equality_l711_71137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_passes_through_fixed_point_l711_71152

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Defines a point on the ellipse -/
def point_on_ellipse (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- Defines the eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - E.b^2 / E.a^2)

/-- Defines a point on the minor axis of the ellipse -/
def point_on_minor_axis (y : ℝ) : ℝ × ℝ :=
  (0, y)

/-- Theorem: The line AB passes through a fixed point for any choice of M and N -/
theorem line_AB_passes_through_fixed_point (E : Ellipse) 
  (h_point : point_on_ellipse E 2 1)
  (h_ecc : eccentricity E = Real.sqrt 3 / 2)
  (M N : ℝ × ℝ) 
  (h_M : M.1 = 0 ∧ M.2 ≠ E.b ∧ M.2 ≠ -E.b)
  (h_N : N.1 = 0 ∧ N.2 ≠ E.b ∧ N.2 ≠ -E.b)
  (h_MN : M.2 = -N.2) :
  ∃ (A B : ℝ × ℝ), 
    point_on_ellipse E A.1 A.2 ∧ 
    point_on_ellipse E B.1 B.2 ∧
    (∃ (k t : ℝ), 
      (A.2 - 1) / (A.1 - 2) = (M.2 - 1) / (-2) ∧
      (B.2 - 1) / (B.1 - 2) = (N.2 - 1) / (-2) ∧
      B.2 - A.2 = k * (B.1 - A.1) ∧
      A.2 = k * A.1 + t ∧
      B.2 = k * B.1 + t ∧
      t = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_passes_through_fixed_point_l711_71152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l711_71109

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x + 3 * Real.log x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 1 + 3 / x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- Define the slope at the point of tangency
noncomputable def m : ℝ := f' point.fst

-- Theorem: The equation of the tangent line is 4x - y - 3 = 0
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - point.fst) + point.snd ↔ 4 * x - y - 3 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l711_71109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_package_weight_theorem_l711_71105

/-- The weight of package B in grams -/
noncomputable def weight_B : ℝ := 6

/-- The weight of package A in grams -/
noncomputable def weight_A : ℝ := 4 * weight_B

/-- The initial ratio of package A to package B -/
noncomputable def initial_ratio : ℝ := weight_A / weight_B

/-- The new ratio after transferring 10 grams from A to B -/
noncomputable def new_ratio : ℝ := (weight_A - 10) / (weight_B + 10)

/-- The total weight of both packages in grams -/
noncomputable def total_weight : ℝ := weight_A + weight_B

theorem package_weight_theorem :
  initial_ratio = 4 ∧ new_ratio = 7/8 → total_weight = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_package_weight_theorem_l711_71105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l711_71127

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6)

noncomputable def g (x : ℝ) : ℝ := cos (π / 6 + 2 * x)

noncomputable def g_shift_right (x : ℝ) : ℝ := g (x - π / 4)

noncomputable def g_shift_left (x : ℝ) : ℝ := g (x + 3 * π / 4)

theorem shift_equivalence (x : ℝ) :
  f x = g_shift_right x ∧ f x = g_shift_left x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l711_71127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_l711_71143

/-- The repeating decimal 36.363636... -/
def repeating_decimal : ℚ := 36 + 36 / 99

/-- Rounding a rational number to the nearest hundredth -/
def round_to_hundredth (q : ℚ) : ℚ := 
  ⌊q * 100 + 1/2⌋ / 100

/-- Theorem: Rounding 36.363636... to the nearest hundredth equals 36.37 -/
theorem round_repeating_decimal : 
  round_to_hundredth repeating_decimal = 36 + 37 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_l711_71143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l711_71170

-- Define the radius and volume of the cone
noncomputable def cone_radius : ℝ := 12
noncomputable def cone_volume : ℝ := 432 * Real.pi

-- Define the theorem
theorem unused_sector_angle : 
  ∀ (circle_radius : ℝ) (cone_height : ℝ),
  -- Conditions
  cone_height > 0 ∧
  circle_radius > cone_radius ∧
  (1 / 3) * Real.pi * cone_radius^2 * cone_height = cone_volume →
  -- Conclusion
  2 * Real.pi - (2 * Real.pi * cone_radius / circle_radius) = Real.pi / 2.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l711_71170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_paths_l711_71138

/-- Definition of the number of paths from (0,0) to (n,m) on an n x m grid --/
def number_of_paths (n m : ℕ) : ℕ := 
  Nat.choose (n + m) m

/-- Theorem stating that the number of paths equals the binomial coefficient --/
theorem ant_paths (n m : ℕ) : 
  number_of_paths n m = Nat.choose (n + m) m :=
by
  -- The proof is trivial due to the definition of number_of_paths
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_paths_l711_71138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_two_max_values_l711_71139

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x / 3)

theorem min_t_for_two_max_values (t : ℕ) :
  (∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ t ∧
    (∀ x ∈ Set.Icc 0 t, f x ≤ f x₁) ∧
    (∀ x ∈ Set.Icc x₁ x₂, f x ≤ f x₂) ∧
    f x₁ = f x₂) →
  t ≥ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_two_max_values_l711_71139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_negative_l711_71157

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≤ -5 ∨ x ≥ 5

-- f is odd
axiom f_odd : ∀ x, domain x → f (-x) = -f x

-- f is monotonically decreasing on [5, +∞)
axiom f_decreasing_positive : ∀ x y, x ≥ 5 → y ≥ 5 → x < y → f x > f y

-- Theorem: f is monotonically decreasing on (-∞, -5]
theorem f_decreasing_negative : ∀ x y, x ≤ -5 → y ≤ -5 → x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_negative_l711_71157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_interval_l711_71116

noncomputable def f (x : ℝ) : ℝ := 5 / (x^2 - 2*x + 20)

theorem function_decreasing_interval (a : ℝ) :
  (∀ x ∈ Set.Icc (2*a) (2-a), StrictMonoOn f (Set.Icc (2*a) (2-a))) ↔ 
  a ∈ Set.Icc (1/2) (2/3) := by
  sorry

#check function_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_interval_l711_71116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_score_82_l711_71104

/-- Represents a team in the knowledge competition -/
structure Team where
  size : Nat
  probabilities : Fin size → ℝ

/-- Represents the competition settings -/
structure Competition where
  num_questions : Nat
  points_per_correct : Nat

/-- Calculates the probability of at least one team member answering correctly -/
def prob_at_least_one_correct (team : Team) : ℝ :=
  1 - (Finset.univ.prod fun i => 1 - team.probabilities i)

/-- Calculates the expected score for a team in the competition -/
def expected_score (team : Team) (comp : Competition) : ℝ :=
  (prob_at_least_one_correct team) * (comp.num_questions : ℝ) * (comp.points_per_correct : ℝ)

/-- The main theorem to prove -/
theorem expected_score_82 : 
  let team : Team := { 
    size := 3, 
    probabilities := fun i => match i with 
      | 0 => 0.4 
      | 1 => 0.4 
      | _ => 0.5 
  }
  let comp : Competition := { 
    num_questions := 10, 
    points_per_correct := 10 
  }
  expected_score team comp = 82 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_score_82_l711_71104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angle_adjustment_l711_71181

/-- Given two complementary angles with measures in the ratio 3:6, if the smaller angle
    is increased by 20%, prove that the larger angle must decrease by 10% for the angles
    to remain complementary. -/
theorem complementary_angle_adjustment (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 1 / 2 →  -- ratio of angles is 3:6, which simplifies to 1:2
  (b - (90 - a * 1.2)) / b = 1 / 10  -- larger angle decreased by 10%
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angle_adjustment_l711_71181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_a_minus_b_l711_71197

-- Define the angle α and variables a and b
variable (α a b : ℝ)

-- Define points A and B
def A : ℝ × ℝ := (1, a)
def B : ℝ × ℝ := (2, b)

-- State the conditions
axiom collinear : a = b / 2

-- We can't directly represent the geometric conditions in Lean,
-- so we'll use the trigonometric relationship instead
axiom cos_α : Real.cos α = 1 / Real.sqrt (1 + a^2)

axiom cos_2α : Real.cos (2 * α) = 2/3

-- State the theorem to be proved
theorem abs_a_minus_b : |a - b| = 3 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_a_minus_b_l711_71197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l711_71151

def z (x : ℝ) : ℂ := Complex.mk (x^2 - 2*x - 3) (x^2 + 3*x + 2)

theorem complex_number_properties (x : ℝ) :
  ((z x).im = 0 ↔ (x = -1 ∨ x = -2)) ∧
  ((z x).re < 0 ∧ (z x).im > 0 ↔ (-1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l711_71151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_store_theorem_l711_71158

/-- Represents the cheese selling scenario in a store -/
structure CheeseStore (initialQuantity : ℝ) where
  customersServed : Nat
  remainingCheese : ℝ
  hCustomers : customersServed ≤ 10
  hRemaining : 0 ≤ remainingCheese ∧ remainingCheese ≤ initialQuantity

/-- The average weight of cheese sold per customer after k sales -/
noncomputable def averageWeight (k : ℕ) : ℝ := 20 / (k + 10)

/-- The remaining cheese after k sales -/
noncomputable def remainingCheese (k : ℕ) : ℝ := 20 - k * (20 / (k + 10))

theorem cheese_store_theorem :
  (∀ k : ℕ, k ≤ 10 → remainingCheese k = 10 * averageWeight k) ∧
  (averageWeight 10 = 1) ∧
  (remainingCheese 10 = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_store_theorem_l711_71158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_necessary_not_sufficient_for_B_l711_71194

-- Define the propositions A and B
def proposition_A (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 1 > 0

def proposition_B (a : ℝ) : Prop := 
  ∀ x y : ℝ, x > 0 → y > 0 → x < y → Real.log x / Real.log (2*a - 1) > Real.log y / Real.log (2*a - 1)

-- Theorem statement
theorem A_necessary_not_sufficient_for_B :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  (∃ a : ℝ, proposition_A a ∧ ¬proposition_B a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_necessary_not_sufficient_for_B_l711_71194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_cost_l711_71106

/-- Proves that the cost of a bottle of orange juice is $0.70 given the conditions of the problem -/
theorem orange_juice_cost
  (total_bottles : ℕ)
  (total_cost : ℚ)
  (orange_bottles : ℕ)
  (apple_cost : ℚ)
  (h1 : total_bottles = 70)
  (h2 : total_cost = 46.20)
  (h3 : orange_bottles = 42)
  (h4 : apple_cost = 0.60) :
  (total_cost - ((total_bottles - orange_bottles) : ℚ) * apple_cost) / (orange_bottles : ℚ) = 0.70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_cost_l711_71106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_relief_selection_arrangement_girls_together_arrangement_no_girls_adjacent_l711_71159

-- Define the number of surgeons and physicians
def num_surgeons : ℕ := 5
def num_physicians : ℕ := 4

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 4

-- Define the total number of people to be selected
def num_selected : ℕ := 5

-- Theorem 1: Selection of people for earthquake relief
theorem earthquake_relief_selection :
  (Nat.choose num_surgeons 3 * Nat.choose num_physicians 2 +
   Nat.choose num_surgeons 4 * Nat.choose num_physicians 1 +
   Nat.choose num_surgeons 5) = 81 := by
  sorry

-- Theorem 2: Arrangement with girls together
theorem arrangement_girls_together :
  (Nat.factorial (num_boys + 1) * Nat.factorial num_girls) = 17280 := by
  sorry

-- Theorem 3: Arrangement with no girls adjacent
theorem arrangement_no_girls_adjacent :
  (Nat.factorial num_boys * Nat.choose (num_boys + 1) num_girls * Nat.factorial num_girls) = 43200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_relief_selection_arrangement_girls_together_arrangement_no_girls_adjacent_l711_71159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_iff_in_solution_set_l711_71113

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else -x + 2

-- Define the solution set
def solution_set : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem f_geq_x_squared_iff_in_solution_set :
  ∀ x : ℝ, f x ≥ x^2 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_iff_in_solution_set_l711_71113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_255_l711_71161

def my_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => 4 * my_sequence n + 3

theorem fifth_term_is_255 : my_sequence 4 = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_255_l711_71161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l711_71168

/-- The distance between the closest points of two circles with centers at (3,5) and (20,8), 
    both tangent to the x-axis, is equal to √298 - 13. -/
theorem closest_points_distance (c1 c2 : ℝ × ℝ) : 
  c1 = (3, 5) → c2 = (20, 8) → 
  (∃ (r1 r2 : ℝ), r1 = c1.2 ∧ r2 = c2.2) →
  Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - (c1.2 + c2.2) = Real.sqrt 298 - 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l711_71168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complexSequenceArithmetic_l711_71175

-- Define the complex sequence
noncomputable def complexSequence (θ : ℝ) (n : ℕ) : ℂ :=
  Complex.exp (n * Complex.I * θ)

-- Define what it means for a complex sequence to be arithmetic
def isArithmeticSequence (s : ℕ → ℂ) : Prop :=
  ∃ d : ℂ, ∀ n : ℕ, s (n + 1) - s n = d

-- Main theorem
theorem complexSequenceArithmetic (θ : ℝ) :
  isArithmeticSequence (complexSequence θ) ↔ ∃ k : ℤ, θ = 2 * k * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complexSequenceArithmetic_l711_71175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_36m_minus_5n_l711_71188

theorem smallest_value_36m_minus_5n :
  (∀ m n : ℕ+, |((36 : ℤ) ^ (m : ℕ)) - ((5 : ℤ) ^ (n : ℕ))| ≥ 11) ∧
  (∃ m n : ℕ+, |((36 : ℤ) ^ (m : ℕ)) - ((5 : ℤ) ^ (n : ℕ))| = 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_36m_minus_5n_l711_71188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_implies_k_value_l711_71130

open Set
open MeasureTheory
open Interval

/-- The parabola function -/
def f (x : ℝ) : ℝ := x + x^2

/-- The line function -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The region M -/
def M : Set (ℝ × ℝ) := {p | 0 ≤ p.2 ∧ p.2 ≤ f p.1 ∧ -1 ≤ p.1 ∧ p.1 ≤ 0}

/-- The region A -/
def A (k : ℝ) : Set (ℝ × ℝ) := {p | g k p.1 ≤ p.2 ∧ p.2 ≤ f p.1 ∧ 0 ≤ p.1 ∧ p.1 ≤ k - 1}

/-- The theorem statement -/
theorem probability_implies_k_value (k : ℝ) (hk : k > 0) :
  (volume (A k) / volume M = 8 / 27) → k = (3 + Real.sqrt 5) / 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_implies_k_value_l711_71130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_M_to_origin_l711_71140

/-- The line l₁ -/
def l₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 = 7}

/-- The line l₂ -/
def l₂ : Set (ℝ × ℝ) := {p | p.1 + p.2 = 5}

/-- Point A on l₁ -/
noncomputable def A : ℝ × ℝ := sorry

/-- Point B on l₂ -/
noncomputable def B : ℝ × ℝ := sorry

/-- Midpoint M of AB -/
noncomputable def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The origin point (0, 0) -/
def origin : ℝ × ℝ := (0, 0)

/-- Main theorem: The minimum distance from M to the origin is 3√2 -/
theorem min_distance_M_to_origin :
  ∀ (A B : ℝ × ℝ), A ∈ l₁ → B ∈ l₂ →
  (∀ p : ℝ × ℝ, distance M origin ≤ distance p origin) →
  distance M origin = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_M_to_origin_l711_71140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l711_71144

theorem angle_in_second_quadrant (θ : Real) : 
  Real.sin θ > Real.cos θ → Real.sin θ * Real.cos θ < 0 → 
  0 < θ ∧ θ < Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l711_71144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l711_71153

/-- The inequality function --/
noncomputable def f (a x : ℝ) : ℝ := (x^2 + (2*a^2 + 2)*x - a^2 + 4*a - 7) / (x^2 + (a^2 + 4*a - 5)*x - a^2 + 4*a - 7)

/-- The solution set of the inequality --/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x < 0}

/-- The condition that the solution set is a union of intervals --/
def is_union_of_intervals (s : Set ℝ) : Prop := sorry

/-- The sum of the lengths of the intervals --/
noncomputable def sum_of_lengths (s : Set ℝ) : ℝ := sorry

/-- The main theorem --/
theorem range_of_a :
  ∀ a : ℝ, (is_union_of_intervals (solution_set a) ∧ 
            sum_of_lengths (solution_set a) < 4) →
           (a ≤ 1 ∨ a ≥ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l711_71153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_logic_l711_71110

theorem proposition_logic :
  (∃ p q : Prop, (p ∨ q) ∧ ¬(p ∧ q)) ∧
  (¬(∀ a b : ℝ, a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ (∃ a b : ℝ, a > b ∧ (2 : ℝ)^a ≤ (2 : ℝ)^b - 1)) ∧
  (¬(∀ x : ℝ, x^2 + x ≥ 1) ↔ (∃ x : ℝ, x^2 + x < 1)) ∧
  ((∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_logic_l711_71110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_35_to_nearest_tenth_l711_71179

/-- Rounding function for decimals to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  if x - ⌊x * 10⌋ / 10 ≥ 0.05 then
    (⌊x * 10⌋ + 1) / 10
  else
    ⌊x * 10⌋ / 10

/-- Theorem stating that 4.35 rounded to the nearest tenth is 4.4 -/
theorem round_4_35_to_nearest_tenth :
  roundToNearestTenth 4.35 = 4.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_35_to_nearest_tenth_l711_71179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_95_factorial_l711_71117

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : Nat) : Nat := sorry

/-- 95! has 22 factors of 10 -/
axiom factors_of_ten : ∃ k : Nat, Nat.factorial 95 = k * (10^22)

theorem last_two_nonzero_digits_95_factorial :
  lastTwoNonzeroDigits 95 = 52 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_95_factorial_l711_71117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l711_71165

-- Define the ellipse C₁ and hyperbola C₂
def C₁ (m n x y : ℝ) : Prop := x^2 / (m + 2) - y^2 / n = 1
def C₂ (m n x y : ℝ) : Prop := x^2 / m + y^2 / n = 1

-- Define the eccentricity of C₁
noncomputable def e₁ (m n : ℝ) : ℝ := Real.sqrt (1 - 1 / (m + 2))

-- State the theorem
theorem ellipse_hyperbola_eccentricity 
  (m n : ℝ) 
  (hm : m > 0) 
  (hn : n < 0) 
  (h_foci : ∀ x y, C₁ m n x y ↔ C₂ m n x y) : 
  Real.sqrt 2 / 2 < e₁ m n ∧ e₁ m n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l711_71165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_width_is_16_5_l711_71111

/-- The width of a rectangular courtyard given its length, number of paving stones, and paving stone dimensions. -/
noncomputable def courtyard_width (length : ℝ) (num_stones : ℕ) (stone_length : ℝ) (stone_width : ℝ) : ℝ :=
  (num_stones : ℝ) * stone_length * stone_width / length

/-- Theorem stating that the width of the courtyard is 16.5 meters -/
theorem courtyard_width_is_16_5 :
  courtyard_width 70 231 (5/2) 2 = 16.5 := by
  -- Expand the definition of courtyard_width
  unfold courtyard_width
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The proof is complete
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_width_is_16_5_l711_71111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_l711_71196

/-- The speed of a boat relative to the water -/
def u : ℝ := sorry

/-- The speed of the current -/
def v : ℝ := sorry

/-- The time when the boats meet after departure -/
def meeting_time : ℝ := 3

/-- The distance the raft has drifted when the boats meet -/
def raft_distance : ℝ := 7.5

/-- Theorem stating that the speed of the current is 2.5 km/h -/
theorem current_speed : v = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_l711_71196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l711_71167

def f (x : ℝ) : ℝ := x^2 + 1

def g (x : ℝ) : ℝ := (f x)^2 + 1

def F (x : ℝ) (lambda : ℝ) : ℝ := g x - lambda * f x

theorem quadratic_function_properties :
  (∀ x y : ℝ, y = f x → g x = y^2 + 1) ∧
  (∃ lambda : ℝ, 
    (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -Real.sqrt 2 / 2 → F x₁ lambda > F x₂ lambda) ∧
    (∀ x₁ x₂ : ℝ, -Real.sqrt 2 / 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → F x₁ lambda < F x₂ lambda)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l711_71167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l711_71192

noncomputable def f (a b x : ℝ) : ℝ := (3 * x + b) / (a * x^2 + 4)

theorem function_properties :
  ∀ (a b : ℝ),
  (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f a b x = -f a b (-x)) →
  f a b 1 = 3/5 →
  (a = 1 ∧ b = 0) ∧
  (∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → f 1 0 x < f 1 0 y) ∧
  (∀ m : ℝ, f 1 0 (m^2 + 1) + f 1 0 (2*m - 2) > 0 ↔ m ∈ Set.Ioo (Real.sqrt 2 - 1) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l711_71192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l711_71148

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The line x + y + 2 = 0 -/
def line (p : Point) : Prop :=
  p.x + p.y + 2 = 0

/-- The circle x² + y² - 4x - 2y = 0 -/
def circleEq (p : Point) : Prop :=
  p.x^2 + p.y^2 - 4*p.x - 2*p.y = 0

/-- Point A is at (0,2) -/
def A : Point :=
  ⟨0, 2⟩

/-- Theorem: The minimum value of |PA| + |PQ| is 2√5 -/
theorem min_distance_sum :
  ∃ (P Q : Point),
    line P ∧ circleEq Q ∧
    ∀ (P' Q' : Point),
      line P' → circleEq Q' →
      distance A P + distance P Q ≤ distance A P' + distance P' Q' ∧
      distance A P + distance P Q = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l711_71148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_in_fund_x_l711_71172

/-- Represents the investment scenario with two funds --/
structure InvestmentScenario where
  total_investment : ℚ
  fund_x_rate : ℚ
  fund_y_rate : ℚ
  interest_difference : ℚ

/-- The specific investment scenario described in the problem --/
def problem_scenario : InvestmentScenario :=
  { total_investment := 100000
  , fund_x_rate := 23 / 100
  , fund_y_rate := 17 / 100
  , interest_difference := 200 }

/-- Calculates the amount invested in Fund X --/
def amount_in_fund_x (scenario : InvestmentScenario) : ℚ :=
  (scenario.total_investment * scenario.fund_y_rate - scenario.interest_difference) /
  (scenario.fund_x_rate + scenario.fund_y_rate)

/-- Theorem stating that the amount invested in Fund X is $42,000 --/
theorem investment_in_fund_x :
  amount_in_fund_x problem_scenario = 42000 := by
  -- Proof goes here
  sorry

#eval amount_in_fund_x problem_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_in_fund_x_l711_71172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l711_71107

theorem triangle_properties (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^2 + b^2 - c^2 = (1/5) * a * b →
  ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ (0 < B ∧ B < π) ∧ (0 < C ∧ C < π) ∧
    A + B + C = π ∧
    a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
    Real.cos C = 1/10 ∧
    (c = 3 * Real.sqrt 11 → 
      ∃ R, R = 5 ∧ R = c / (2 * Real.sin C)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l711_71107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_students_same_topic_l711_71100

/-- Represents a student -/
def Student : Type := ℕ

/-- Represents a topic -/
inductive Topic
| T1
| T2
| T3

/-- The total number of students -/
def num_students : ℕ := 17

/-- The function representing the topic discussed between two students -/
def discussed_topic : Student → Student → Topic := sorry

/-- The theorem to be proved -/
theorem exist_three_students_same_topic :
  ∃ (s1 s2 s3 : Student) (t : Topic),
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
    discussed_topic s1 s2 = t ∧
    discussed_topic s2 s3 = t ∧
    discussed_topic s1 s3 = t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_students_same_topic_l711_71100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformations_l711_71178

-- Define the triangle Γ
def Triangle : Set (ℝ × ℝ) :=
  {(0, 0), (4, 0), (0, 3)}

-- Define the transformations
inductive Transformation
| Rotate90 : Transformation
| Rotate180 : Transformation
| Rotate270 : Transformation
| ReflectX : Transformation
| ReflectY : Transformation

-- Define a sequence of three transformations
def TransformationSequence := Fin 3 → Transformation

-- Function to check if a sequence returns Γ to its original position
def returnsToOriginal (seq : TransformationSequence) : Prop :=
  sorry -- Implementation details omitted

-- Prove that TransformationSequence is finite
instance : Fintype TransformationSequence := 
  sorry -- Proof omitted

-- Prove that returnsToOriginal is decidable
instance (seq : TransformationSequence) : Decidable (returnsToOriginal seq) :=
  sorry -- Proof omitted

-- The main theorem
theorem triangle_transformations :
  (Finset.filter returnsToOriginal (Finset.univ : Finset TransformationSequence)).card = 12 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformations_l711_71178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l711_71103

def is_valid (n : ℕ) : Prop :=
  (∃ i ∈ ({n, n+1, n+2} : Set ℕ), i % 4 = 0) ∧
  (∃ i ∈ ({n, n+1, n+2} : Set ℕ), i % 9 = 0) ∧
  (∃ i ∈ ({n, n+1, n+2} : Set ℕ), i % 25 = 0)

theorem smallest_valid_n :
  (is_valid 475) ∧ (∀ m < 475, ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l711_71103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_formula_l711_71174

/-- 
N(n) is the maxima number of triples (a_i, b_i, c_i) of nonnegative integers
satisfying the following conditions:
1. a_i + b_i + c_i = n for all i = 1, ..., N(n)
2. If i ≠ j then a_i ≠ a_j, b_i ≠ b_j, and c_i ≠ c_j
-/
def N (n : ℕ) : ℕ := sorry

/-- The main theorem stating that N(n) = ⌊(2n/3)⌋ + 1 for all n ≥ 2 -/
theorem N_formula (n : ℕ) (h : n ≥ 2) : N n = (2 * n / 3 : ℚ).floor.toNat + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_formula_l711_71174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l711_71150

noncomputable def ceiling (m : ℝ) : ℤ := Int.ceil m

noncomputable def floor (m : ℝ) : ℤ := Int.floor m

theorem problem_solution (x : ℤ) : 3 * (ceiling (x : ℝ)) + 2 * (floor (x : ℝ)) = 23 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l711_71150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_l711_71112

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- Uri's purchase: 3 burgers and 2 sodas cost 380 cents -/
axiom uri_purchase : 3 * burger_cost + 2 * soda_cost = 380

/-- Gen's purchase: 2 burgers and 3 sodas cost 390 cents -/
axiom gen_purchase : 2 * burger_cost + 3 * soda_cost = 390

/-- Theorem: The cost of a soda is 82 cents -/
theorem soda_price : soda_cost = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_l711_71112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_propositions_l711_71193

/-- Represents a proposition -/
inductive Proposition
  | Original
  | Inverse
  | Converse
  | Contrapositive

/-- Represents the truth value of a set of related propositions -/
structure PropSet where
  original : Bool
  inverse : Bool
  converse : Bool
  contrapositive : Bool

/-- Counts the number of true propositions in a PropSet -/
def countTrue (ps : PropSet) : Nat :=
  (if ps.original then 1 else 0) +
  (if ps.inverse then 1 else 0) +
  (if ps.converse then 1 else 0) +
  (if ps.contrapositive then 1 else 0)

/-- Theorem: The maximum number of true propositions in any valid PropSet is 4 -/
theorem max_true_propositions :
  ∃ (ps : PropSet), countTrue ps = 4 ∧
  ∀ (ps' : PropSet), countTrue ps' ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_propositions_l711_71193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l711_71166

/-- Calculates compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time) - principal

/-- Theorem: The compound interest on $1200 for 4 years at 20% p.a., compounded yearly, is approximately $1288.32 -/
theorem compound_interest_calculation :
  let principal : ℝ := 1200
  let rate : ℝ := 0.20
  let time : ℝ := 4
  let frequency : ℝ := 1
  abs (compound_interest principal rate time frequency - 1288.32) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l711_71166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_l711_71101

/-- Represents the scale of the map -/
noncomputable def map_scale : ℝ := 745.9 / 45

/-- Conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.609

/-- Conversion factor from yards to meters -/
def yard_to_m : ℝ := 0.914

/-- Converts centimeters to miles based on the map scale -/
noncomputable def cm_to_miles (cm : ℝ) : ℝ := cm / map_scale

/-- Converts kilometers to miles -/
noncomputable def km_to_miles (km : ℝ) : ℝ := km / mile_to_km

/-- Converts yards to miles -/
noncomputable def yards_to_miles (yards : ℝ) : ℝ := 
  (yards * yard_to_m / 1000) / mile_to_km

/-- The total distance in miles -/
noncomputable def total_distance : ℝ := 
  cm_to_miles 254 + km_to_miles 18.7 + yards_to_miles 1500

theorem total_distance_approx : 
  ∃ ε > 0, |total_distance - 27.792| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_l711_71101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l711_71164

theorem least_subtraction : 
  (∀ d ∈ ({5, 7, 9} : Finset ℕ), (642 - 4) % d = 4) ∧ 
  (∀ x < 4, ∃ d ∈ ({5, 7, 9} : Finset ℕ), (642 - x) % d ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l711_71164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_to_line_l_l711_71125

/-- Line l in 2D space -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-8 + t, t / 2)

/-- Curve C in 2D space -/
noncomputable def curve_C (s : ℝ) : ℝ × ℝ := (2 * s^2, 2 * Real.sqrt 2 * s)

/-- Distance from a point to line l -/
noncomputable def distance_to_line_l (p : ℝ × ℝ) : ℝ :=
  abs (p.1 - 2 * p.2 + 8) / Real.sqrt 5

/-- Theorem: The minimum distance from any point on curve C to line l is 4√5/5 -/
theorem min_distance_curve_C_to_line_l :
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
  ∀ (s : ℝ), distance_to_line_l (curve_C s) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_to_line_l_l711_71125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_199_l711_71171

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => f (n + 1) + f n

theorem eleventh_term_is_199 : f 11 = 199 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_199_l711_71171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l711_71126

def y : ℕ → ℝ
  | 0 => 2  -- Define y(0) to be 2 as well, to cover all natural numbers
  | 1 => 2
  | k + 2 => y (k + 1) ^ 2 - y (k + 1)

theorem sum_reciprocal_y_plus_one :
  ∑' n, 1 / (y n + 1) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l711_71126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l711_71147

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

-- State the theorem
theorem problem_solution :
  (∀ a > 0, (∀ x > 1, ∀ y > 1, x < y → f a x < f a y)) →
  (∀ a > 0, a ≥ 1) ∧
  (∀ x ≥ 0, g x ≤ 0 ∧ g 0 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l711_71147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_radius_range_l711_71145

/-- The set of points (x, y) satisfying the given equation -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sin (p.1 + 2 * p.2) = Real.sin p.1 + Real.sin (2 * p.2)}

/-- A circle with center c and radius r -/
def Circle (c : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

/-- The theorem stating the range of valid radii -/
theorem valid_radius_range (c : ℝ × ℝ) (r : ℝ) :
    (∀ p ∈ M, p ∉ Circle c r) →
    0 < r ∧ r < (3 - Real.sqrt 5) / 2 * Real.pi :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_radius_range_l711_71145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l711_71190

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 4
def g (x : ℝ) : ℝ := x^2 - 6 * x + 10

-- Theorem stating the x-coordinates of intersection points
theorem parabola_intersection :
  ∃ (y₁ y₂ : ℝ),
    (f (3/2 + Real.sqrt 21/2) = g (3/2 + Real.sqrt 21/2) ∧ f (3/2 + Real.sqrt 21/2) = y₁) ∧
    (f (3/2 - Real.sqrt 21/2) = g (3/2 - Real.sqrt 21/2) ∧ f (3/2 - Real.sqrt 21/2) = y₂) ∧
    ∀ (x y : ℝ), f x = g x ∧ f x = y → x = 3/2 + Real.sqrt 21/2 ∨ x = 3/2 - Real.sqrt 21/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l711_71190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l711_71189

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a vector
noncomputable def vectorLength (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Define the dot product of two vectors
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem triangle_dot_product (t : Triangle) : 
  vectorLength (t.B.1 - t.A.1, t.B.2 - t.A.2) = 7 →
  vectorLength (t.C.1 - t.B.1, t.C.2 - t.B.2) = 5 →
  vectorLength (t.A.1 - t.C.1, t.A.2 - t.C.2) = 6 →
  dotProduct (t.A.1 - t.B.1, t.A.2 - t.B.2) (t.C.1 - t.B.1, t.C.2 - t.B.2) = 19 :=
by
  sorry

#check triangle_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l711_71189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l711_71146

/-- Calculates the annual interest rate given the principal, final amount, time, and compounding frequency -/
noncomputable def calculate_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) (compounds_per_year : ℝ) : ℝ :=
  ((final_amount / principal) ^ (1 / (time * compounds_per_year)) - 1) * compounds_per_year

/-- The annual interest rate for the given scenario -/
noncomputable def annual_interest_rate : ℝ := calculate_interest_rate 4000 4410 2 1

theorem interest_rate_is_five_percent : 
  annual_interest_rate = 0.05 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l711_71146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_l711_71120

-- Define the function f(x) = 6/x
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- State the theorem
theorem inverse_proportion_range :
  ∀ y : ℝ, (∃ x : ℝ, x > 3 ∧ f x = y) ↔ (0 < y ∧ y < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_l711_71120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_l711_71176

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the distance from a point to a line
def distanceToLine (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem orthocenter_circumcenter_distance (t : Triangle) :
  distance t.A (orthocenter t) = 2 * distanceToLine (circumcenter t) (t.B, t.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_l711_71176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_specific_l711_71124

/-- Represents the time it takes for a leak to empty a full tank, given the filling times with and without the leak -/
noncomputable def leak_empty_time (fill_time_no_leak : ℝ) (fill_time_with_leak : ℝ) : ℝ :=
  (fill_time_no_leak * fill_time_with_leak) / (fill_time_with_leak - fill_time_no_leak)

/-- Theorem stating that for a pipe that fills a tank in 6 hours without a leak
    and in 10 hours with a leak, the leak alone will empty the full tank in 15 hours -/
theorem leak_empty_time_specific :
  leak_empty_time 6 10 = 15 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_specific_l711_71124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l711_71133

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x + Real.cos x

theorem f_monotone_increasing_interval :
  ∀ (k : ℤ) (x : ℝ),
    (x ∈ Set.Icc (2 * k * Real.pi + 2 * Real.pi / 3) (2 * k * Real.pi + 5 * Real.pi / 3)) ↔
    (∀ y ∈ Set.Icc (2 * k * Real.pi + 2 * Real.pi / 3) x, f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l711_71133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_bradley_bill_change_l711_71156

theorem uncle_bradley_bill_change (total_amount : ℕ) (fraction_for_fifty : ℚ) 
  (fifty_bill_value : ℕ) (hundred_bill_value : ℕ) : 
  total_amount = 1000 →
  fraction_for_fifty = 3/10 →
  fifty_bill_value = 50 →
  hundred_bill_value = 100 →
  (↑total_amount * fraction_for_fifty / ↑fifty_bill_value).floor.toNat + 
  ((↑total_amount - ↑total_amount * fraction_for_fifty) / ↑hundred_bill_value).floor.toNat = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_bradley_bill_change_l711_71156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_around_block_l711_71121

/-- The difference in distance run around a square block with streets -/
theorem distance_difference_around_block (block_side : ℝ) (street_width : ℝ) : 
  block_side = 400 → street_width = 30 → 
  4 * (block_side + 2 * street_width) - 4 * block_side = 240 := by
  intros h1 h2
  rw [h1, h2]
  norm_num

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_around_block_l711_71121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_sine_cosine_relation_l711_71187

theorem right_angle_sine_cosine_relation (A B C : ℝ) (h_triangle : A + B + C = Real.pi) :
  A = Real.pi / 2 ↔ Real.sin C = Real.sin A * Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_sine_cosine_relation_l711_71187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weights_std_dev_l711_71141

noncomputable def apple_weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (weights : List ℝ) : ℝ :=
  weights.sum / weights.length

noncomputable def variance (weights : List ℝ) : ℝ :=
  let m := mean weights
  (weights.map (λ x => (x - m) ^ 2)).sum / weights.length

noncomputable def standard_deviation (weights : List ℝ) : ℝ :=
  Real.sqrt (variance weights)

theorem apple_weights_std_dev :
  standard_deviation apple_weights = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weights_std_dev_l711_71141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l711_71136

theorem inscribed_square_area (larger_square circle_diameter rectangle_length rectangle_width : Real) :
  larger_square = 4 ∧ 
  circle_diameter = 2 ∧ 
  rectangle_length = 4 ∧ 
  rectangle_width = 2 → 
  ∃ (inscribed_square : Real), 
    inscribed_square^2 = 8 - Real.pi ∧ 
    inscribed_square = Real.sqrt (8 - Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l711_71136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l711_71131

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the parallel lines 2x - y + 3 = 0 and -4x + 2y + 5 = 0 is 11√5/10 -/
theorem parallel_lines_distance :
  distance_between_parallel_lines 2 (-1) 3 (-5/2) = 11 * Real.sqrt 5 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l711_71131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_theorem_l711_71177

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a point is the foot of the perpendicular from the origin to a plane -/
def is_foot_of_perpendicular (p : Point) (plane : Plane) : Prop :=
  p.x = plane.A ∧ p.y = plane.B ∧ p.z = plane.C

/-- The main theorem -/
theorem plane_equation_theorem (p : Point) (h1 : p.x = 10 ∧ p.y = -2 ∧ p.z = 5) :
  ∃ (plane : Plane), 
    point_on_plane p plane ∧ 
    is_foot_of_perpendicular p plane ∧
    plane.A = 10 ∧ plane.B = -2 ∧ plane.C = 5 ∧ plane.D = -129 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_theorem_l711_71177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_3y_eq_seven_sixths_l711_71180

-- Define the space and plane
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (α : AffineSubspace ℝ V)

-- Define the points
variable (A B C D E : α.carrier)
variable (O : V)

-- Define variables for coefficients
variable (x y : ℝ)

-- Define the vector equations
axiom eq1 : (A : V) - O = (1/2 : ℝ) • ((B : V) - O) + x • ((C : V) - O) + y • ((D : V) - O)
axiom eq2 : (B : V) - O = (2*x : ℝ) • ((C : V) - O) + (1/3 : ℝ) • ((D : V) - O) + y • ((E : V) - O)

-- No three points are collinear
axiom not_collinear : ∀ (p q r : ℝ), p • ((B : V) - A) + q • ((C : V) - A) + r • ((D : V) - A) ≠ 0 ∨ (p = 0 ∧ q = 0 ∧ r = 0)

-- The theorem to prove
theorem x_plus_3y_eq_seven_sixths : x + 3 * y = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_3y_eq_seven_sixths_l711_71180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_three_l711_71169

/-- The area of a triangle using the three-oblique formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt ((1/4) * (a^2 * c^2 - ((a^2 + c^2 - b^2)/2)^2))

/-- Theorem: The area of triangle ABC is √3 -/
theorem triangle_area_is_sqrt_three
  (a b c : ℝ)
  (h1 : a^2 * Real.sin c = 4 * Real.sin a)
  (h2 : (a + c)^2 = 12 + b^2) :
  triangle_area a b c = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_three_l711_71169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_not_always_perpendicular_l711_71123

-- Define the basic types
structure Line3D where

structure Plane3D where

-- Define the relationships
def perpendicular (a b : Line3D) : Prop := sorry

def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

theorem perpendicular_parallel_not_always_perpendicular :
  ∃ (x y : Line3D) (z : Plane3D),
    perpendicular x y ∧
    parallel_line_plane y z ∧
    ¬perpendicular_line_plane x z := by
  sorry

#check perpendicular_parallel_not_always_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_not_always_perpendicular_l711_71123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_proof_l711_71118

/-- The distance between a child's home and school, given their walking speeds and arrival times -/
def school_distance (speed_late speed_early : ℝ) (time_late time_early : ℝ) : ℝ :=
  let time_on_time := 120 -- Calculated from the problem
  let distance := speed_late * (time_on_time + time_late)
  distance

theorem school_distance_proof (speed_late speed_early : ℝ) (time_late time_early : ℝ)
    (h1 : speed_late = 5) (h2 : speed_early = 7) (h3 : time_late = 6) (h4 : time_early = 30) :
    school_distance speed_late speed_early time_late time_early = 630 := by
  unfold school_distance
  rw [h1, h3]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval school_distance 5 7 6 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_proof_l711_71118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_bean_percentage_in_mixed_container_l711_71199

/-- Represents a bag of jelly beans -/
structure BagOfBeans where
  total : ℕ
  redRatio : ℚ

/-- Calculates the total number of beans in a list of bags -/
def totalBeans (bags : List BagOfBeans) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.total) 0

/-- Calculates the total number of red beans in a list of bags -/
def totalRedBeans (bags : List BagOfBeans) : ℚ :=
  bags.foldl (fun acc bag => acc + (bag.total : ℚ) * bag.redRatio) 0

/-- The main theorem statement -/
theorem red_bean_percentage_in_mixed_container : 
  let bags : List BagOfBeans := [
    { total := 24, redRatio := 2/5 },
    { total := 32, redRatio := 3/10 },
    { total := 36, redRatio := 1/4 },
    { total := 40, redRatio := 3/20 }
  ]
  let totalRed := totalRedBeans bags
  let total := totalBeans bags
  ∃ (ε : ℚ), |((totalRed / (total : ℚ)) * 100 - 27)| < ε ∧ ε < 1 := by
  sorry

#eval totalBeans [
  { total := 24, redRatio := 2/5 },
  { total := 32, redRatio := 3/10 },
  { total := 36, redRatio := 1/4 },
  { total := 40, redRatio := 3/20 }
]

#eval totalRedBeans [
  { total := 24, redRatio := 2/5 },
  { total := 32, redRatio := 3/10 },
  { total := 36, redRatio := 1/4 },
  { total := 40, redRatio := 3/20 }
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_bean_percentage_in_mixed_container_l711_71199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l711_71162

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + x^3 - 2

-- State the theorem
theorem unique_zero_in_interval :
  ∃! x, x ∈ Set.Ioo 0 1 ∧ f x = 0 :=
by
  sorry

#check unique_zero_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l711_71162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l711_71114

/-- Represents the sum of the first k terms of a geometric sequence. -/
noncomputable def geometricSum (a₁ : ℝ) (r : ℝ) (k : ℕ) : ℝ := a₁ * (1 - r^k) / (1 - r)

/-- Proves that for a geometric sequence with S_n = 2 and S_{3n} = 14, S_{4n} = 30. -/
theorem geometric_sequence_sum (a₁ r : ℝ) (n : ℕ) 
  (h_positive : a₁ > 0 ∧ r > 0)
  (h_Sn : geometricSum a₁ r n = 2)
  (h_S3n : geometricSum a₁ r (3*n) = 14) :
  geometricSum a₁ r (4*n) = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l711_71114
