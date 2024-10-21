import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_l1294_129422

/-- The Apollonian circle theorem -/
theorem apollonian_circle 
  (A B : ℝ × ℝ) -- Two points in the plane
  (k : ℝ) -- The ratio constant
  (h : k ≠ 1) -- Assumption that k is not equal to 1
  : 
  ∃ (C : ℝ × ℝ) (r : ℝ), -- There exists a center C and radius r
    (∃ t : ℝ, C.1 = A.1 + t * (B.1 - A.1) ∧ C.2 = A.2 + t * (B.2 - A.2)) ∧ -- C is on the line AB
    (∀ (M : ℝ × ℝ), dist M A / dist M B = k ↔ dist M C = r) -- M is on the circle iff |AM|/|MB| = k
  := by sorry

/-- Helper function to calculate Euclidean distance -/
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_l1294_129422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_twenty_equals_p_r_l1294_129454

theorem power_twenty_equals_p_r (m n : ℤ) (P R : ℕ) 
  (h_P : P = (2 : ℕ)^(m.natAbs)) (h_R : R = (5 : ℕ)^(n.natAbs)) : 
  (20 : ℕ)^(m*n).natAbs = P^(2*n).natAbs * R^m.natAbs := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_twenty_equals_p_r_l1294_129454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_f50_16_l1294_129440

/-- Number of positive integer divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- f₁(n) is three times the number of positive integer divisors of n -/
def f₁ (n : ℕ) : ℕ := 3 * d n

/-- fⱼ(n) for j ≥ 2 -/
def f (j : ℕ) (n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | 1 => f₁ n
  | j+2 => f₁ (f (j+1) n)

/-- Theorem stating that there is exactly one n ≤ 100 such that f₅₀(n) = 16 -/
theorem unique_n_for_f50_16 :
  ∃! (n : ℕ), n > 0 ∧ n ≤ 100 ∧ f 50 n = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_f50_16_l1294_129440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1294_129437

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the triangle is a right triangle --/
def Triangle.isRight (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2

/-- Calculates the area of the triangle --/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Calculates the semiperimeter of the triangle --/
noncomputable def Triangle.semiperimeter (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 2

/-- Calculates the inradius of the triangle --/
noncomputable def Triangle.inradius (t : Triangle) : ℝ :=
  t.area / t.semiperimeter

theorem triangle_properties (t : Triangle) 
    (h1 : t.a = 9) (h2 : t.b = 12) (h3 : t.c = 15) :
    t.isRight ∧ 
    t.area = 54 ∧ 
    t.semiperimeter = 18 ∧ 
    t.inradius = 3 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1294_129437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1294_129458

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then Real.log (-x) + 3 * x
  else Real.log x - 3 * x

-- State the theorem
theorem tangent_line_slope (h1 : ∀ x, f x = f (-x)) 
                            (h2 : f 1 = -3) : 
  HasDerivAt f (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1294_129458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_proof_l1294_129463

-- Define the point through which the terminal side of angle α passes
noncomputable def point : ℝ × ℝ := (2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5)

-- Define cos α
noncomputable def cos_alpha : ℝ := 2 * Real.sqrt 5 / 5

-- Theorem statement
theorem cos_alpha_proof (α : ℝ) : 
  (Real.cos α = cos_alpha) ↔ 
  (Real.cos α * Real.cos α + Real.sin α * Real.sin α = 1 ∧
   Real.cos α = point.1 ∧ 
   Real.sin α = point.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_proof_l1294_129463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_color_blocks_probability_l1294_129410

-- Define the number of chips for each color
def tan_chips : ℕ := 4
def pink_chips : ℕ := 3
def violet_chips : ℕ := 5
def total_chips : ℕ := tan_chips + pink_chips + violet_chips

-- Define the probability of the specific drawing arrangement
def specific_arrangement_probability : ℚ :=
  (Nat.factorial tan_chips * Nat.factorial pink_chips * Nat.factorial violet_chips * Nat.factorial 2) /
  Nat.factorial total_chips

-- Theorem statement
theorem consecutive_color_blocks_probability :
  specific_arrangement_probability = 1 / 13860 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_color_blocks_probability_l1294_129410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1294_129416

theorem trigonometric_identities 
  (α : ℝ) 
  (h1 : Real.cos α = -3/5) 
  (h2 : α ∈ Set.Ioo π (2*π)) 
  (β : ℝ)
  (h3 : Real.tan β = -1/3) : 
  Real.sin (2*α) = 24/25 ∧ Real.tan (α - β) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1294_129416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_on_line_l1294_129441

/-- Given a point P(cos α, sin α) on the line y = 2x, prove that sin 2α = 4/5 -/
theorem sin_2alpha_on_line (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_on_line_l1294_129441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_uncovered_area_of_three_frames_l1294_129439

/-- Represents a circular frame with a given diameter -/
structure Frame where
  diameter : ℝ

/-- Calculates the area of a circular frame -/
noncomputable def frameArea (f : Frame) : ℝ := Real.pi * (f.diameter / 2) ^ 2

/-- Calculates the uncovered area of a frame given a coverage percentage -/
noncomputable def uncoveredArea (f : Frame) (coveragePercentage : ℝ) : ℝ :=
  frameArea f * (1 - coveragePercentage)

theorem total_uncovered_area_of_three_frames :
  let frameA : Frame := ⟨16⟩
  let frameB : Frame := ⟨14⟩
  let frameC : Frame := ⟨12⟩
  let uncoveredA := uncoveredArea frameA 0.7
  let uncoveredB := uncoveredArea frameB 0.5
  let uncoveredC := frameArea frameC
  uncoveredA + uncoveredB + uncoveredC = 79.7 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_uncovered_area_of_three_frames_l1294_129439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1294_129453

-- Define the speed of the train in km/h
noncomputable def train_speed_kmh : ℝ := 72

-- Define the time taken to cross in seconds
noncomputable def crossing_time_s : ℝ := 9

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 5 / 18

-- Theorem statement
theorem train_length_proof :
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  let train_length_m := train_speed_ms * crossing_time_s
  train_length_m = 180 := by
  -- Unfold the definitions
  unfold train_speed_kmh crossing_time_s kmh_to_ms
  -- Simplify the expressions
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1294_129453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1294_129407

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a + abc.c = 8)
  (h2 : Real.cos abc.B = 1/4)
  (h3 : abc.a * abc.c * Real.cos abc.B = 4)
  (h4 : Real.sin abc.A = Real.sqrt 6 / 4) :
  (abc.b = 2 * Real.sqrt 6) ∧ 
  (Real.sin abc.C = 3 * Real.sqrt 6 / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1294_129407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1294_129413

theorem triangle_angle_relation (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle condition
  π/2 < C →  -- C is obtuse
  Real.cos A ^ 2 + Real.cos C ^ 2 + 2 * Real.sin A * Real.sin C * Real.cos B = 16/9 →
  Real.cos C ^ 2 + Real.cos B ^ 2 + 2 * Real.sin C * Real.sin B * Real.cos A = 17/10 →
  Real.cos B ^ 2 + Real.cos A ^ 2 + 2 * Real.sin B * Real.sin A * Real.cos C = (16 + Real.sqrt 315) / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1294_129413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_nonempty_solutions_l1294_129461

/-- A function f(x) = x^2 + ax + b cos(x) -/
noncomputable def f (a b x : ℝ) : ℝ := x^2 + a*x + b*(Real.cos x)

/-- The set of real solutions to f(x) = 0 -/
def SolutionSet (a b : ℝ) : Set ℝ := {x : ℝ | f a b x = 0}

/-- The set of real solutions to f(f(x)) = 0 -/
def CompositeSolutionSet (a b : ℝ) : Set ℝ := {x : ℝ | f a b (f a b x) = 0}

/-- The theorem stating the conditions for identical and non-empty solution sets -/
theorem identical_nonempty_solutions (a b : ℝ) :
  (SolutionSet a b = CompositeSolutionSet a b ∧ Set.Nonempty (SolutionSet a b)) ↔
  (0 ≤ a ∧ a < 4 ∧ b = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_nonempty_solutions_l1294_129461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1294_129434

-- Define the dimensions of the shapes
def rect1_length : ℝ := 11
def rect1_width : ℝ := 11
def rect2_length : ℝ := 5.5
def rect2_width : ℝ := 11
def triangle_side : ℝ := 6
def circle_diameter : ℝ := 4

-- Define the areas of the shapes
noncomputable def rect1_area : ℝ := rect1_length * rect1_width
noncomputable def rect2_area : ℝ := rect2_length * rect2_width
noncomputable def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
noncomputable def circle_area : ℝ := Real.pi * (circle_diameter / 2)^2

-- Define the combined areas
noncomputable def rectangles_area : ℝ := rect1_area + rect2_area
noncomputable def triangle_circle_area : ℝ := triangle_area + circle_area

-- Define the difference in areas
noncomputable def area_difference : ℝ := rectangles_area - triangle_circle_area

-- Theorem statement
theorem area_difference_approx :
  abs (area_difference - 153.35) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1294_129434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1294_129491

/-- The function f(x, y) as defined in the problem -/
noncomputable def f (x y : ℝ) : ℝ :=
  Real.sqrt (Real.cos (4 * x) + 7) + 
  Real.sqrt (Real.cos (4 * y) + 7) + 
  Real.sqrt (Real.cos (4 * x) + Real.cos (4 * y) - 8 * Real.sin x ^ 2 * Real.sin y ^ 2 + 6)

/-- The theorem stating that 6√2 is the maximum value of f(x, y) -/
theorem f_max_value :
  ∀ x y : ℝ, f x y ≤ 6 * Real.sqrt 2 ∧ ∃ x y : ℝ, f x y = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1294_129491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_placement_theorem_l1294_129475

/-- A soap piece with dimensions 1 × 2 × (n+1) -/
structure SoapPiece (n : ℕ) where
  length : Fin 3
  width : Fin 3
  height : Fin 3
  dim_check : (length = 0 ∧ width = 1 ∧ height = 2) ∨ 
              (length = 1 ∧ width = 0 ∧ height = 2) ∨ 
              (length = 2 ∧ width = 1 ∧ height = 0)

/-- A cubic box with edge 2n+1 -/
structure CubicBox (n : ℕ) where
  edge : ℕ
  edge_check : edge = 2*n + 1

/-- Predicate to check if a soap piece can be placed in the box -/
def canPlace (n : ℕ) (piece : SoapPiece n) (box : CubicBox n) : Prop :=
  piece.length.val < box.edge ∧ piece.width.val < box.edge ∧ piece.height.val < box.edge

/-- Predicate to check if all soap pieces can be placed in the box -/
def canPlaceAll (n : ℕ) : Prop :=
  ∃ (arrangement : Fin (2*n*(2*n+1)) → SoapPiece n),
    ∀ (i : Fin (2*n*(2*n+1))), canPlace n (arrangement i) (CubicBox.mk (2*n+1) rfl)

/-- The main theorem -/
theorem soap_placement_theorem (n : ℕ) :
  canPlaceAll n ↔ n % 2 = 0 ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_placement_theorem_l1294_129475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_staff_reduction_l1294_129478

/-- Proves that reducing clerical staff by 1/3 results in the specified percentage of remaining employees being clerical -/
theorem clerical_staff_reduction (total_employees : ℕ) (initial_clerical_fraction : ℚ) 
  (final_clerical_percentage : ℚ) (reduction_fraction : ℚ) : 
  total_employees = 3600 →
  initial_clerical_fraction = 1/6 →
  final_clerical_percentage = 11764705882352941/100000000000000000 →
  reduction_fraction = 1/3 →
  (let initial_clerical := (initial_clerical_fraction * total_employees : ℚ)
   let remaining_clerical := initial_clerical - (reduction_fraction * initial_clerical)
   let remaining_total := (total_employees : ℚ) - (reduction_fraction * initial_clerical)
   remaining_clerical / remaining_total = final_clerical_percentage) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_staff_reduction_l1294_129478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_implies_x_and_y_l1294_129464

-- Define the vectors
def a : ℝ × ℝ × ℝ := (2, -3, 5)
def b (x y : ℝ) : ℝ × ℝ × ℝ := (4, x, y)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = k • w ∨ w = k • v

-- Theorem statement
theorem parallel_vectors_implies_x_and_y (x y : ℝ) :
  parallel a (b x y) → x = -6 ∧ y = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_implies_x_and_y_l1294_129464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l1294_129401

/-- The function g(x) parameterized by d -/
noncomputable def g (d : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + d) / (x^2 - 5*x + 6)

/-- Predicate for g having exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (d : ℝ) : Prop :=
  (∃! x : ℝ, (x^2 - 5*x + 6 = 0 ∧ x^2 - 2*x + d ≠ 0)) ∧
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 → x^2 - 2*x + d = 0 → False)

/-- Theorem stating the condition for g to have exactly one vertical asymptote -/
theorem g_one_vertical_asymptote :
  ∀ d : ℝ, has_exactly_one_vertical_asymptote d ↔ (d = 0 ∨ d = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l1294_129401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l1294_129427

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (x - 2)^2 / 7^2 - (y - 10)^2 / 3^2 = 1

/-- The x-coordinate of the focus with larger x-coordinate -/
noncomputable def focus_x : ℝ := 2 + Real.sqrt 58

/-- The y-coordinate of the focus -/
def focus_y : ℝ := 10

/-- Theorem stating that (focus_x, focus_y) is the focus with larger x-coordinate -/
theorem focus_coordinates :
  ∀ x y : ℝ, hyperbola_eq x y → 
  (x ≠ 2 - Real.sqrt 58 ∨ y ≠ 10) →
  (x < focus_x ∨ (x = focus_x ∧ y = focus_y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l1294_129427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_27π_l1294_129455

/-- Represents the dimensions of a rectangle --/
structure RectangleDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of the shaded region in a rectangle with quarter circles at corners and a central circle --/
noncomputable def shadedArea (rect : RectangleDimensions) (centralRadius : ℝ) : ℝ :=
  let quarterCircleRadius := rect.width / 2
  let quarterCirclesArea := Real.pi * quarterCircleRadius^2
  let centralCircleArea := Real.pi * centralRadius^2
  quarterCirclesArea - centralCircleArea

/-- Theorem stating that the shaded area in the given configuration is 27π cm² --/
theorem shaded_area_is_27π :
  let rect : RectangleDimensions := ⟨15, 12⟩
  let centralRadius : ℝ := 3
  shadedArea rect centralRadius = 27 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_27π_l1294_129455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1294_129474

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2 * Real.sin (x + Real.pi/4) + 2*x^2 + x) / (2*x^2 + Real.cos x)

theorem sum_of_max_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ (M + m = 2) := by
  sorry

#check sum_of_max_min_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1294_129474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_range_is_600_cost_relationship_l1294_129486

/-- Represents the range of the vehicles in kilometers. -/
def range : ℝ := 600

/-- The cost difference per kilometer between fuel and new energy vehicles. -/
def cost_difference : ℝ := 0.54

/-- The fuel tank capacity of the fuel vehicle in liters. -/
def fuel_capacity : ℝ := 40

/-- The fuel price in yuan per liter. -/
def fuel_price : ℝ := 9

/-- The battery capacity of the new energy vehicle in kilowatt-hours. -/
def battery_capacity : ℝ := 60

/-- The electricity price in yuan per kilowatt-hour. -/
def electricity_price : ℝ := 0.6

/-- Theorem stating that the range of both vehicles is 600 km. -/
theorem vehicle_range_is_600 : range = 600 := by
  rfl

/-- Theorem proving the relationship between fuel and new energy vehicle costs. -/
theorem cost_relationship : 
  fuel_capacity * fuel_price / range = 
  battery_capacity * electricity_price / range + cost_difference := by
  simp [range, cost_difference, fuel_capacity, fuel_price, battery_capacity, electricity_price]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_range_is_600_cost_relationship_l1294_129486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1294_129487

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the original expression
noncomputable def original_expr : ℝ := 1 / (cubeRoot 5 - cubeRoot 3)

-- Define the rationalized expression
noncomputable def rationalized_expr : ℝ := (cubeRoot 25 + cubeRoot 15 + cubeRoot 9) / 2

-- Theorem statement
theorem rationalize_denominator :
  original_expr = rationalized_expr ∧ 25 + 15 + 9 + 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1294_129487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bulrush_reed_equal_height_l1294_129484

/-- The height of the bulrush after n days -/
noncomputable def bulrush_height (n : ℝ) : ℝ := 3 * (1 - (1/2)^n) / (1 - 1/2)

/-- The height of the reed after n days -/
noncomputable def reed_height (n : ℝ) : ℝ := (2^n - 1) / (2 - 1)

/-- The number of days for the bulrush and reed to have equal height -/
noncomputable def equal_height_days : ℝ := 
  Real.log 6 / Real.log 2

theorem bulrush_reed_equal_height :
  ∃ ε > 0, abs (equal_height_days - 2.6) < ε ∧
  bulrush_height equal_height_days = reed_height equal_height_days :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bulrush_reed_equal_height_l1294_129484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l1294_129426

-- Define constants
noncomputable def running_km : ℝ := 40
def swimming_miles : ℝ := 2.5
def time_limit : ℝ := 10

-- Define conversion rates
def km_to_miles : ℝ := 0.621371
def yard_to_miles : ℝ := 0.000568182
def foot_to_miles : ℝ := 0.000189394

-- Define exercise distances
noncomputable def running_miles : ℝ := running_km * km_to_miles
noncomputable def walking_miles : ℝ := 3/5 * running_miles
noncomputable def jogging_yards : ℝ := walking_miles * 1760 * 5
noncomputable def biking_feet : ℝ := jogging_yards / yard_to_miles * 5280 * 3

-- Theorem statement
theorem total_distance_theorem :
  running_miles + walking_miles + jogging_yards * yard_to_miles + 
  biking_feet * foot_to_miles + swimming_miles = 340.449562 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l1294_129426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_sequence_formula_l1294_129485

/-- A sequence where the nth term consists of n digits, all of which are 6 -/
def sequenceA (n : ℕ) : ℚ :=
  2/3 * ((10 : ℚ)^n - 1)

/-- The nth term of the sequence consists of n digits, all of which are 6 -/
theorem sequence_property (n : ℕ) : ∀ d : ℕ, d < n → (sequenceA n).num % 10 = 6 := by
  sorry

/-- The general formula for the nth term of the sequence -/
theorem sequence_formula (n : ℕ) : sequenceA n = 2/3 * ((10 : ℚ)^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_sequence_formula_l1294_129485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_sales_effect_l1294_129499

theorem tv_sales_effect (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  (0.9 * P * (1.85 * Q)) / (P * Q) - 1 = 0.665 := by
  sorry

#eval (0.9 * 1.85 - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_sales_effect_l1294_129499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_intervals_l1294_129480

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (π / 4 - 2 * x)

-- Define the interval endpoints
noncomputable def interval_start (k : ℤ) : ℝ := k * π + 3 * π / 8
noncomputable def interval_end (k : ℤ) : ℝ := k * π + 7 * π / 8

-- Theorem statement
theorem f_increasing_on_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (interval_start k) (interval_end k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_intervals_l1294_129480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_winners_l1294_129466

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_players : ℕ
  can_win : ℕ → ℕ → Prop

/-- The specific tournament described in the problem. -/
def ping_pong_tournament : Tournament where
  num_players := 2^2013
  can_win := λ x y ↦ x ≤ y + 3

/-- The maximum seed of a player who can possibly win the tournament. -/
def max_winning_seed (t : Tournament) : ℕ := sorry

/-- Theorem stating the maximum number of players who can win the tournament. -/
theorem max_winners (t : Tournament) (h : t = ping_pong_tournament) : 
  max_winning_seed t = 6038 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_winners_l1294_129466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr1_is_fraction_l1294_129433

-- Define the expressions
noncomputable def expr1 (x : ℝ) := x / (x + 1)
noncomputable def expr2 (x : ℝ) := x / 2
noncomputable def expr3 (x : ℝ) := x / 2 + 1
noncomputable def expr4 (x y : ℝ) := x * y / 3

-- Define what it means to be a fraction in this context
def is_fraction (f : ℝ → ℝ) : Prop :=
  ∃ (n d : ℝ → ℝ), ∀ x, f x = (n x) / (d x) ∧ d x ≠ 0

-- Theorem statement
theorem only_expr1_is_fraction :
  is_fraction expr1 ∧
  ¬is_fraction expr2 ∧
  ¬is_fraction expr3 ∧
  ¬is_fraction (λ x => expr4 x x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr1_is_fraction_l1294_129433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_zeros_of_h_zeros_count_by_a_l1294_129459

noncomputable section

-- Define the functions f, g, and h
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x - 1/4
def g (x : ℝ) : ℝ := Real.exp x - Real.exp 1
def h (a : ℝ) (x : ℝ) : ℝ := if f a x ≥ g x then f a x else g x

-- Theorem for the perpendicular tangent lines
theorem perpendicular_tangents :
  ∃ a : ℝ, (deriv (f a)) 0 * (deriv g) 0 = -1 ∧ a = -1 :=
sorry

-- Theorem for the number of zeros of h
theorem zeros_of_h (a : ℝ) :
  (∃ x₁ x₂ : ℝ, h a x₁ = 0 ∧ h a x₂ = 0 ∧ x₁ ≠ x₂ ∧
    ∀ x : ℝ, h a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃ : ℝ, h a x₁ = 0 ∧ h a x₂ = 0 ∧ h a x₃ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    ∀ x : ℝ, h a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄ : ℝ, h a x₁ = 0 ∧ h a x₂ = 0 ∧ h a x₃ = 0 ∧ h a x₄ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    ∀ x : ℝ, h a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
sorry

-- Theorem relating the number of zeros to the value of a
theorem zeros_count_by_a (a : ℝ) :
  (a < 3/4 ∨ a > 5/4 →
    ∃ x₁ x₂ : ℝ, h a x₁ = 0 ∧ h a x₂ = 0 ∧ x₁ ≠ x₂ ∧
      ∀ x : ℝ, h a x = 0 → x = x₁ ∨ x = x₂) ∧
  (a = 3/4 ∨ a = 5/4 →
    ∃ x₁ x₂ x₃ : ℝ, h a x₁ = 0 ∧ h a x₂ = 0 ∧ h a x₃ = 0 ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      ∀ x : ℝ, h a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (3/4 < a ∧ a < 5/4 →
    ∃ x₁ x₂ x₃ x₄ : ℝ, h a x₁ = 0 ∧ h a x₂ = 0 ∧ h a x₃ = 0 ∧ h a x₄ = 0 ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      ∀ x : ℝ, h a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_zeros_of_h_zeros_count_by_a_l1294_129459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_machine_rate_l1294_129494

/-- The rate at which the old machine makes bolts (in bolts per hour) -/
def old_rate : ℚ := 100

/-- The time both machines work (in hours) -/
def work_time : ℚ := 2

/-- The total number of bolts made by both machines -/
def total_bolts : ℕ := 500

/-- The rate at which the new machine makes bolts (in bolts per hour) -/
noncomputable def new_rate : ℚ := (total_bolts - old_rate * work_time) / work_time

/-- Theorem stating that the new machine's rate is 150 bolts per hour -/
theorem new_machine_rate : new_rate = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_machine_rate_l1294_129494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1294_129400

structure Ellipse where
  foci : ℝ × ℝ × ℝ × ℝ
  intersectionLine : Set (ℝ × ℝ)
  intersectionPoints : ℝ × ℝ × ℝ × ℝ
  trianglePerimeter : ℝ

def standardEllipseEquation (e : Ellipse) : Prop :=
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ (x, y) ∈ e.intersectionLine

theorem ellipse_equation (e : Ellipse) 
  (h1 : e.foci = (-1, 0, 1, 0))
  (h2 : (-1, 0) ∈ e.intersectionLine)
  (h3 : e.trianglePerimeter = 8) :
  standardEllipseEquation e :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1294_129400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_is_140_l1294_129482

-- Define the cost price and selling price
variable (C : ℝ) -- Cost price
variable (S : ℝ) -- Selling price

-- Define the condition that selling at 1/3 of the price results in a 20% loss
def condition (C S : ℝ) : Prop := (1/3) * S = 0.8 * C

-- Define the profit percent formula
noncomputable def profit_percent (C S : ℝ) : ℝ := ((S - C) / C) * 100

-- Theorem statement
theorem profit_percent_is_140 {C S : ℝ} (h : condition C S) : profit_percent C S = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_is_140_l1294_129482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1294_129429

noncomputable def f (x : Real) : Real := Real.sin (2 * x + Real.pi / 3)

theorem min_m_value :
  ∃ (m : Real),
    (∀ (x₁ x₂ x₃ : Real),
      0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ Real.pi →
      |f x₁ - f x₂| + |f x₂ - f x₃| ≤ m) ∧
    (∀ (m' : Real),
      (∀ (x₁ x₂ x₃ : Real),
        0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ Real.pi →
        |f x₁ - f x₂| + |f x₂ - f x₃| ≤ m') →
      m ≤ m') ∧
    m = 3 + Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1294_129429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_sum_l1294_129436

/-- Represents a rectangular box with given dimensions -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the triangle formed by center points of three faces meeting at a corner -/
noncomputable def triangleArea (b : Box) : ℝ :=
  let s1 := Real.sqrt ((b.height / 2) ^ 2 + (b.width / 2) ^ 2)
  let s2 := Real.sqrt ((b.height / 2) ^ 2 + (b.length / 2) ^ 2)
  let s3 := (b.width + b.length) / 2
  let s := (s1 + s2 + s3) / 2
  Real.sqrt (s * (s - s1) * (s - s2) * (s - s3))

/-- The main theorem to be proved -/
theorem box_dimensions_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (hcoprime : Nat.Coprime m n) :
  let b := Box.mk 15 20 (m / n : ℝ)
  triangleArea b = 35 → m + n = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_sum_l1294_129436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1294_129498

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 4)
  (h2 : (x : ℝ) / (y : ℝ) = 96.16) : 
  y = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1294_129498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2000_terms_eq_6101_l1294_129471

/-- Defines the n-th term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let k := ((8 * n.pred + 1 : ℕ).sqrt - 1) / 2
    if n = (k * (k + 1)) / 2 + 1 then 1 else 3

/-- The sum of the first n terms of the sequence -/
def sequenceSum (n : ℕ) : ℕ :=
  (List.range n).map sequenceTerm |>.sum

/-- The main theorem stating that the sum of the first 2000 terms is 6101 -/
theorem sum_2000_terms_eq_6101 : sequenceSum 2000 = 6101 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2000_terms_eq_6101_l1294_129471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l1294_129406

-- Define f and g as variables instead of functions
variable (f g : ℝ → ℝ)

-- Range of f is [-10, 1]
axiom f_range : ∀ x, -10 ≤ f x ∧ f x ≤ 1

-- Range of g is [-3, 2]
axiom g_range : ∀ x, -3 ≤ g x ∧ g x ≤ 2

-- Theorem: The maximum value of f(x) * g(x) is 30
theorem max_product_value : 
  ∃ x y : ℝ, f x * g y = 30 ∧ ∀ z w : ℝ, f z * g w ≤ 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l1294_129406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1294_129448

noncomputable section

open Real

-- Define f(α) as given in the problem
def f (α : ℝ) : ℝ :=
  (sqrt ((1 - sin α) / (1 + sin α)) + sqrt ((1 + sin α) / (1 - sin α))) * (cos α)^3 +
  2 * sin (π / 2 + α) * cos (3 * π / 2 - α)

-- Theorem for Part I
theorem part_one (α : ℝ) (h1 : tan α = 3) (h2 : π < α ∧ α < 3 * π / 2) :
  f α = -4/5 := by sorry

-- Theorem for Part II
theorem part_two (α : ℝ) (h1 : f α = 14/5 * cos α) (h2 : π < α ∧ α < 3 * π / 2) :
  tan α = 3/4 ∨ tan α = 4/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1294_129448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1294_129468

theorem polynomial_remainder (q : Polynomial ℝ) : 
  (∃ r₁ r₂ : Polynomial ℝ, q = (X - 2) * r₁ + 8 ∧ q = (X + 3) * r₂ - 10) →
  ∃ r : Polynomial ℝ, q = (X - 2) * (X + 3) * r + (18/5 * X + 4/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1294_129468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ton_conversion_liter_conversion_kilometer_conversion_l1294_129460

-- Define units as noncomputable
noncomputable def ton : ℝ := 1
noncomputable def kilogram : ℝ := 1 / 1000
noncomputable def gram : ℝ := 1 / 1000000
noncomputable def liter : ℝ := 1
noncomputable def milliliter : ℝ := 1 / 1000
noncomputable def kilometer : ℝ := 1
noncomputable def meter : ℝ := 1 / 1000
noncomputable def centimeter : ℝ := 1 / 100000

-- Define unit types
inductive UnitType
  | Mass
  | Volume
  | Length

-- Theorem statements
theorem ton_conversion :
  (ton = 1000 * kilogram) ∧ (ton = 1000000 * gram) ∧ (UnitType.Mass = UnitType.Mass) := by sorry

theorem liter_conversion :
  (liter = 1000 * milliliter) ∧ (UnitType.Volume = UnitType.Volume) := by sorry

theorem kilometer_conversion :
  (kilometer = 1000 * meter) ∧ (kilometer = 100000 * centimeter) ∧ (UnitType.Length = UnitType.Length) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ton_conversion_liter_conversion_kilometer_conversion_l1294_129460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethereum_investment_gain_l1294_129419

theorem ethereum_investment_gain (initial_investment : ℝ) (first_week_gain_percent : ℝ) (final_value : ℝ) : 
  initial_investment = 400 →
  first_week_gain_percent = 25 →
  final_value = 750 →
  let first_week_value := initial_investment * (1 + first_week_gain_percent / 100)
  let second_week_gain := final_value - first_week_value
  let second_week_gain_percent := (second_week_gain / first_week_value) * 100
  second_week_gain_percent = 50 := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check ethereum_investment_gain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethereum_investment_gain_l1294_129419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_in_room_l1294_129432

/-- Calculates the total number of people in a room given certain conditions. -/
theorem people_in_room (chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  chairs = 32 ∧ 
  seated_people = (5 : ℕ) * chairs / 6 ∧
  seated_people = (3 : ℕ) * total_people / 5 ∧
  8 = chairs / 4 ∧
  seated_people = 3 * (seated_people / 3) →
  total_people = 45 := by
  intro h
  sorry

#check people_in_room

end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_in_room_l1294_129432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_difference_l1294_129420

theorem multiple_difference (a b c d n : ℤ) :
  (n ∣ (a * d - b * c)) →
  (n ∣ (a - b)) →
  Int.gcd b.natAbs n.natAbs = 1 →
  (n ∣ (c - d)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_difference_l1294_129420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l1294_129411

theorem triangle_angle_ratio (A B C : ℝ) (a b c : ℝ) :
  A + B + C = Real.pi →
  0 < A ∧ 0 < B ∧ 0 < C →
  A < Real.pi ∧ B < Real.pi ∧ C < Real.pi →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  C = 2 * B →
  (Real.sin (3 * B)) / (Real.sin B) = a / b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l1294_129411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_equals_S_l1294_129483

-- Define the function f(x) = (x-1)^(-1/2)
noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ (-(1/2 : ℝ))

-- Define the set S = {x | x > 1}
def S : Set ℝ := {x | x > 1}

-- Theorem: The domain of f is equal to S
theorem domain_f_equals_S : {x : ℝ | ∃ y, f x = y} = S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_equals_S_l1294_129483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1294_129414

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hour_to_m_per_sec (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

/-- Calculates the time (in seconds) it takes for an object to travel a given distance at a given speed -/
noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_passing_time (train_length : ℝ) (train_speed_km_hr : ℝ) 
    (h1 : train_length = 180)
    (h2 : train_speed_km_hr = 36) :
  time_to_pass train_length (km_per_hour_to_m_per_sec train_speed_km_hr) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1294_129414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_on_l1294_129412

/-- Represents the state of a light (on or off) -/
inductive LightState
| On : LightState
| Off : LightState

/-- Represents a 3x4 grid of lights -/
def LightGrid : Type := Fin 3 → Fin 4 → LightState

/-- Returns the list of adjacent positions for a given position in the grid -/
def adjacentPositions (row : Fin 3) (col : Fin 4) : List (Fin 3 × Fin 4) :=
  sorry

/-- Toggles a single light state -/
def toggleLight (state : LightState) : LightState :=
  match state with
  | LightState.On => LightState.Off
  | LightState.Off => LightState.On

/-- Applies the toggle operation to a light and its adjacent lights -/
def pressLight (grid : LightGrid) (row : Fin 3) (col : Fin 4) : LightGrid :=
  sorry

/-- Counts the number of off lights in the grid -/
def countOffLights (grid : LightGrid) : Nat :=
  sorry

/-- Checks if all lights in the grid are on -/
def allLightsOn (grid : LightGrid) : Prop :=
  ∀ (row : Fin 3) (col : Fin 4), grid row col = LightState.On

/-- Main theorem: If the initial number of off lights is odd, it's impossible to turn all lights on -/
theorem impossible_all_on (initial_grid : LightGrid) :
  Odd (countOffLights initial_grid) →
  ¬∃ (sequence : List (Fin 3 × Fin 4)), allLightsOn (sequence.foldl (λ g (r, c) => pressLight g r c) initial_grid) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_on_l1294_129412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1294_129490

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/3 + y²/2 = 1 -/
def Ellipse : Set Point :=
  {p : Point | p.x^2 / 3 + p.y^2 / 2 = 1}

/-- The left focus of the ellipse -/
def leftFocus : Point :=
  ⟨-1, 0⟩

/-- Checks if two points are symmetric with respect to a line through the left focus -/
def isSymmetric (A B : Point) : Prop :=
  ∃ (k : ℝ), B.y = k * (B.x + 1) ∧ A.y = k * (A.x + 1)

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (O A B : Point) : ℝ :=
  abs (A.x * B.y - B.x * A.y) / 2

/-- Main theorem: The maximum area of triangle AOB is √6/2 -/
theorem max_triangle_area :
  ∀ (A B : Point), A ∈ Ellipse → B ∈ Ellipse → isSymmetric A B →
  triangleArea ⟨0, 0⟩ A B ≤ Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1294_129490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1294_129443

/-- Given a parabola C defined by y = 2x^2 with focus F, and a point P(1, y) on C,
    prove that the distance |PF| is equal to 17/8. -/
theorem parabola_focus_distance (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (P : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ y = 2 * x^2) →  -- Definition of parabola C
  (∃ (p : ℝ), p > 0 ∧ F = (0, p / 2)) →     -- F is the focus
  P.1 = 1 →                                 -- x-coordinate of P is 1
  P ∈ C →                                   -- P is on the parabola
  dist P F = 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1294_129443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_n_9_and_10_l1294_129447

/-- Represents the state of the game board -/
inductive BoardState
| Minuses (n : ℕ)  -- n consecutive minuses
| Split (l r : ℕ)  -- l minuses, some pluses, r minuses

/-- Represents a player's move -/
inductive Move
| ChangeOne
| ChangeTwo

/-- Defines a winning strategy for the first player -/
def has_winning_strategy (n : ℕ) : Prop :=
  ∃ (strategy : BoardState → Move),
    ∀ (opponent_strategy : BoardState → Move),
      let game := λ (state : BoardState) (player : Bool) =>
        if player then strategy state else opponent_strategy state
      ∃ (k : ℕ), game (BoardState.Minuses n) true = Move.ChangeOne ∧ 
                 game (BoardState.Minuses 0) false = Move.ChangeOne

theorem first_player_wins_n_9_and_10 :
  has_winning_strategy 9 ∧ has_winning_strategy 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_n_9_and_10_l1294_129447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_problem_l1294_129450

noncomputable section

-- Define the points
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (1, -1)
noncomputable def C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)
def O : ℝ × ℝ := (0, 0)

-- Define vectors
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define vector length
noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define scalar multiplication for vectors
def scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define the theorem
theorem geometric_problem (θ : ℝ) (m n : ℝ) :
  (vec_length (vec (C θ) B - vec (C θ) A) = Real.sqrt 2) →
  (vec_add (scalar_mul m (vec O A)) (scalar_mul n (vec O B)) = vec O (C θ)) →
  (Real.sin (2 * θ) = -1/2 ∧
   ∀ (m' n' : ℝ), vec_add (scalar_mul m' (vec O A)) (scalar_mul n' (vec O B)) = vec O (C θ) →
                   (m' - 3)^2 + n'^2 ≤ 16) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_problem_l1294_129450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1294_129403

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x) - 2 / (Real.exp x + 1)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (f x + f (2*x - 1) > -2) ↔ (x > 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1294_129403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sampling_plan_satisfies_conditions_l1294_129430

/-- Represents the different categories of staff members -/
inductive StaffCategory
| Administrative
| Teaching
| Logistics

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| Systematic

/-- Represents the sampling plan for a specific staff category -/
structure SamplingPlan where
  category : StaffCategory
  sampleSize : Nat
  method : SamplingMethod

/-- The total number of staff members -/
def totalStaff : Nat := 160

/-- The number of administrative staff -/
def administrativeStaff : Nat := 16

/-- The number of teachers -/
def teachers : Nat := 112

/-- The number of logistics personnel -/
def logisticsStaff : Nat := 32

/-- The desired total sample size -/
def desiredSampleSize : Nat := 20

/-- The correct sampling plan for the school -/
def correctSamplingPlan : List SamplingPlan := [
  { category := StaffCategory.Administrative, sampleSize := 2, method := SamplingMethod.Lottery },
  { category := StaffCategory.Teaching, sampleSize := 14, method := SamplingMethod.Systematic },
  { category := StaffCategory.Logistics, sampleSize := 4, method := SamplingMethod.Lottery }
]

/-- Helper function to check if a category is valid -/
def isValidCategory (c : StaffCategory) : Bool :=
  match c with
  | StaffCategory.Administrative => true
  | StaffCategory.Teaching => true
  | StaffCategory.Logistics => true

/-- Theorem stating that the correct sampling plan satisfies the given conditions -/
theorem correct_sampling_plan_satisfies_conditions :
  (List.sum (correctSamplingPlan.map (λ p => p.sampleSize)) = desiredSampleSize) ∧
  (List.length correctSamplingPlan = 3) ∧
  (correctSamplingPlan.all (λ p => p.sampleSize > 0)) ∧
  (correctSamplingPlan.all (λ p => isValidCategory p.category)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sampling_plan_satisfies_conditions_l1294_129430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_zeros_implies_a_positive_l1294_129465

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp 2 - a * x^2 + (a - 2 * Real.exp 1) * x

/-- Theorem stating that if f has three distinct zero points, then a is in (0, +∞) -/
theorem three_distinct_zeros_implies_a_positive (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_zeros_implies_a_positive_l1294_129465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1294_129425

noncomputable def x : ℕ → ℝ
  | 0 => 4  -- Add case for 0
  | 1 => 4
  | n + 2 => Real.sqrt (2 * x (n + 1) + 3)

theorem sequence_properties :
  (∀ n ≥ 2, |x n - 3| ≤ (2/3) * |x (n-1) - 3|) ∧
  (∀ n ≥ 1, 3 - (2/3)^(n-1) ≤ x n ∧ x n ≤ 3 + (2/3)^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1294_129425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1294_129469

/-- Calculates the speed of a train given its length, time to cross a man, and the man's speed -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * 3600 / 1000

/-- The speed of the train is approximately 63.0036 km/hr -/
theorem train_speed_calculation :
  let ε := 0.0001
  let calculated_speed := train_speed 700 41.9966402687785 3
  (63.0036 - ε ≤ calculated_speed) ∧ (calculated_speed ≤ 63.0036 + ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1294_129469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_two_pans_before_discount_l1294_129438

/-- Calculates the cost of 2 pans before discount given the following conditions:
  * 3 pots and 4 pans are purchased
  * 10% discount on the entire purchase
  * Total cost after discount is $100
  * Each pot costs $20 before discount
  * Discount rate is the same for both pots and pans
-/
theorem cost_of_two_pans_before_discount :
  let num_pots : ℕ := 3
  let num_pans : ℕ := 4
  let discount_rate : ℚ := 1/10
  let total_cost_after_discount : ℚ := 100
  let cost_per_pot_before_discount : ℚ := 20
  let total_cost_before_discount : ℚ := total_cost_after_discount / (1 - discount_rate)
  let cost_of_pots_before_discount : ℚ := num_pots * cost_per_pot_before_discount
  let cost_of_pans_before_discount : ℚ := total_cost_before_discount - cost_of_pots_before_discount
  let cost_per_pan_before_discount : ℚ := cost_of_pans_before_discount / num_pans
  ∃ (ε : ℚ), abs (cost_per_pan_before_discount * 2 - 25.56) < ε ∧ ε > 0 := by
  sorry

#eval (100 / (1 - 1/10) - 3 * 20) / 4 * 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_two_pans_before_discount_l1294_129438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_complex_probability_l1294_129428

/-- The probability that (m+ni)² is a pure imaginary number when m and n are from two dice throws -/
theorem dice_complex_probability : 
  let outcomes := Finset.product (Finset.range 6) (Finset.range 6)
  let favorable := outcomes.filter (fun p => p.1 = p.2)
  (favorable.card : ℚ) / outcomes.card = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_complex_probability_l1294_129428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_theorem_l1294_129496

/-- The slope of the best-fitting line for three points -/
noncomputable def best_fit_slope (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : ℝ :=
  (y₃ - y₁) / (x₃ - x₁)

/-- Theorem stating the slope of the best-fitting line for three points -/
theorem best_fit_slope_theorem (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h₁ : x₁ < x₂) (h₂ : x₂ < x₃) (h₃ : x₃ + x₁ = 2 * x₂) :
  best_fit_slope x₁ x₂ x₃ y₁ y₂ y₃ = 
    (y₃ - y₁) / (x₃ - x₁) :=
by
  -- Unfold the definition of best_fit_slope
  unfold best_fit_slope
  -- The definition matches the right-hand side exactly
  rfl

#check best_fit_slope_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_theorem_l1294_129496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_average_speed_l1294_129479

/-- Represents a segment of the car's journey -/
structure JourneySegment where
  duration : ℝ
  speed : ℝ
  elevation_change : ℝ
  wind_effect : ℝ

/-- Calculates the distance traveled in a journey segment -/
def distance_traveled (segment : JourneySegment) : ℝ :=
  segment.duration * (segment.speed + segment.wind_effect)

/-- Represents the entire car journey -/
def car_journey : List JourneySegment :=
  [
    { duration := 1, speed := 40, elevation_change := 500, wind_effect := 0 },
    { duration := 0.5, speed := 60, elevation_change := 0, wind_effect := -5 },
    { duration := 2, speed := 60, elevation_change := -1000, wind_effect := 0 }
  ]

/-- Calculates the total distance of the journey -/
noncomputable def total_distance : ℝ :=
  (car_journey.map distance_traveled).sum

/-- Calculates the total duration of the journey -/
noncomputable def total_duration : ℝ :=
  (car_journey.map (λ segment => segment.duration)).sum

/-- Calculates the average speed of the journey -/
noncomputable def average_speed : ℝ :=
  total_distance / total_duration

/-- Theorem stating that the average speed is approximately 53.57 km/h -/
theorem car_journey_average_speed :
  |average_speed - 53.57| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_average_speed_l1294_129479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wooden_box_height_is_6_l1294_129488

/-- The height of a wooden box that can hold a specific number of smaller boxes -/
noncomputable def wooden_box_height (wooden_length wooden_width : ℝ) 
                      (small_length small_width small_height : ℝ)
                      (max_boxes : ℕ) : ℝ :=
  (max_boxes : ℝ) * small_length * small_width * small_height / (wooden_length * wooden_width)

/-- Theorem: The height of the wooden box is 6 meters -/
theorem wooden_box_height_is_6 :
  wooden_box_height 8 7 0.04 0.07 0.06 2000000 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wooden_box_height_is_6_l1294_129488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l1294_129472

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 2

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem inverse_function_problem (a b : ℝ) :
  (∀ x, g x = (f a b).invFun x - 1) →
  7 * a + 3 * b = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l1294_129472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_iff_n_eq_four_l1294_129473

/-- The area of the resulting shape for a given rectangle and n -/
noncomputable def resulting_area (a b n : ℝ) : ℝ := a * b + 2 * (a^2 / n + b^2 / n)

/-- Two rectangles have equal perimeter -/
def equal_perimeter (a b c d : ℝ) : Prop := a + b = c + d

/-- Theorem stating that the resulting area is constant if and only if n = 4 -/
theorem constant_area_iff_n_eq_four (a b c d n : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ n > 0) 
  (h_perimeter : equal_perimeter a b c d) :
  resulting_area a b n = resulting_area c d n ↔ n = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_iff_n_eq_four_l1294_129473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_consecutive_integers_with_specific_degree_l1294_129424

def degree (n : ℕ) : ℕ := sorry

theorem existence_of_consecutive_integers_with_specific_degree :
  ∃ (start : ℕ), (Finset.filter (λ i ↦ degree (start + i) < 11) (Finset.range 2018)).card = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_consecutive_integers_with_specific_degree_l1294_129424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1294_129404

def a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | (n + 1) => a n / (3 * a n + 1)

theorem a_formula (n : ℕ) (h : n ≥ 1) : a n = 1 / (3 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1294_129404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_theorem_l1294_129477

/-- Rectangle ABCD with special points K and M -/
structure SpecialRectangle where
  /-- The length of side BC -/
  x : ℝ
  /-- Point K on diagonal AC such that CK = BC -/
  k : ℝ × ℝ
  /-- Point M on side BC such that KM = CM -/
  m : ℝ × ℝ
  /-- K is on diagonal AC -/
  k_on_diagonal : k.1^2 + k.2^2 = 1 + x^2
  /-- CK = BC -/
  ck_eq_bc : (1 - k.1)^2 + (x - k.2)^2 = x^2
  /-- M is on side BC -/
  m_on_bc : m.1 = 1 ∧ 0 ≤ m.2 ∧ m.2 ≤ x
  /-- KM = CM -/
  km_eq_cm : (k.1 - m.1)^2 + (k.2 - m.2)^2 = (1 - m.1)^2 + (x - m.2)^2

/-- The main theorem to prove -/
theorem special_rectangle_theorem (r : SpecialRectangle) :
  let a := (0, 0)
  let b := (1, 0)
  let c := (1, r.x)
  Real.sqrt ((r.k.1 - a.1)^2 + (r.k.2 - a.2)^2) +  -- AK
  Real.sqrt ((r.m.1 - b.1)^2 + (r.m.2 - b.2)^2) =  -- BM
  Real.sqrt ((c.1 - r.m.1)^2 + (c.2 - r.m.2)^2)    -- CM
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_theorem_l1294_129477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1294_129409

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4:ℝ)^x - 2^(x+1) - 3

-- State the theorem
theorem min_value_of_f (x : ℝ) (h : x * (Real.log 2 / Real.log 3) ≥ 1) :
  ∃ (min_val : ℝ), min_val = 0 ∧ ∀ y, f y ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1294_129409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1294_129444

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Asymptote of a hyperbola -/
noncomputable def asymptote (h : Hyperbola) : ℝ → ℝ := λ x ↦ (h.b / h.a) * x

/-- Symmetric point with respect to an asymptote -/
noncomputable def symmetric_point (p : Point) (h : Hyperbola) : Point :=
  { x := (h.b^2 - h.a^2) / p.x,
    y := -2 * h.a * h.b / p.x }

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A : Point)
    (h_F₁ : F₁.x < 0)
    (h_F₂ : F₂.x > 0)
    (h_foci : F₂.x = -F₁.x)
    (h_A : A = symmetric_point F₂ h)
    (h_left_branch : A.x < 0) :
    eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1294_129444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_for_given_range_l1294_129423

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem not_perfect_square_for_given_range : 
  ∀ n : ℕ, n ∈ ({19, 20, 21, 22, 23} : Set ℕ) → ¬(is_perfect_square ((n.factorial * (n + 1).factorial) / 2)) :=
by
  intro n hn
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_for_given_range_l1294_129423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_equals_four_l1294_129408

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angle A in radians (60 degrees = π/3 radians)
noncomputable def angle_A : ℝ := Real.pi / 3

-- Define the equation for b and c
def roots_equation (x : ℝ) : Prop := x^2 - 7*x + 11 = 0

-- Theorem statement
theorem side_a_equals_four :
  ∀ a b c : ℝ,
  triangle a b c →
  roots_equation b ∧ roots_equation c →
  a = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_equals_four_l1294_129408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_a_2008_nonzero_l1294_129462

-- Define the polynomial f with integer coefficients
def f (x : ℤ) : ℤ := sorry

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => 0
  | n + 1 => f (a n)

theorem sequence_property (i j : ℕ) (h : i < j) :
  ∃ k : ℤ, a (j + 1) - a j = k * (a (i + 1) - a i) := by
  sorry

theorem a_2008_nonzero : a 2008 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_a_2008_nonzero_l1294_129462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l1294_129449

noncomputable def original_equation (x : ℝ) : ℝ :=
  (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2

theorem roots_of_equation :
  ∃ (r₁ r₂ : ℝ), 
    (abs (r₁ - 2.76) < 0.01) ∧ 
    (abs (r₂ + 0.18) < 0.01) ∧ 
    (abs (original_equation r₁) < 0.01) ∧ 
    (abs (original_equation r₂) < 0.01) := by
  sorry

#check roots_of_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l1294_129449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l1294_129467

-- Define the basic types
structure Line
structure Plane

-- Define the relations
axiom parallel_planes : Plane → Plane → Prop
axiom perpendicular_planes : Plane → Plane → Prop
axiom line_parallel_plane : Line → Plane → Prop
axiom line_perpendicular_plane : Line → Plane → Prop
axiom line_in_plane : Line → Plane → Prop
axiom parallel_lines : Line → Line → Prop

-- Define the given conditions
axiom different_lines (m n : Line) : m ≠ n
axiom different_planes (α β γ : Plane) : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Define the propositions
def proposition_1 (α β γ : Plane) : Prop :=
  parallel_planes α β → parallel_planes α γ → parallel_planes β γ

def proposition_2 (α β : Plane) (m : Line) : Prop :=
  perpendicular_planes α β → line_parallel_plane m α → line_perpendicular_plane m β

def proposition_3 (α β : Plane) (m : Line) : Prop :=
  line_perpendicular_plane m α → line_parallel_plane m β → perpendicular_planes α β

def proposition_4 (α : Plane) (m n : Line) : Prop :=
  parallel_lines m n → line_in_plane n α → line_parallel_plane m α

-- State the theorem
theorem propositions_correctness (α β γ : Plane) (m n : Line) :
  (∀ α β γ, proposition_1 α β γ) ∧
  (∃ α β m, ¬proposition_2 α β m) ∧
  (∀ α β m, proposition_3 α β m) ∧
  (∃ α m n, ¬proposition_4 α m n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l1294_129467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_ratio_l1294_129492

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio :
  ∀ (v p : ℝ), v > 0 → p > 0 →
  (let brand_x_volume := 1.3 * v
   let brand_x_price := 0.8 * p
   let brand_x_unit_price := brand_x_price / brand_x_volume
   let brand_y_unit_price := p / v
   brand_x_unit_price / brand_y_unit_price) = 8 / 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_ratio_l1294_129492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1294_129431

/-- Given a quadratic function f(x) = x^2 + bx - 3 where f(-2) = f(0),
    prove that the solution set for f(x) ≤ 0 is [-3, 1]. -/
theorem quadratic_inequality_solution_set 
  (f : ℝ → ℝ) 
  (b : ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x - 3) 
  (h2 : f (-2) = f 0) :
  {x : ℝ | f x ≤ 0} = Set.Icc (-3) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1294_129431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l1294_129452

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n + 2 => a (n + 1) / (1 + 3 * a (n + 1))

/-- Theorem stating the closed form of a_n -/
theorem a_closed_form (n : ℕ) (h : n ≥ 1) : a n = 2 / (6 * n - 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l1294_129452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_store_optimization_l1294_129456

/-- Represents the number of pieces of clothing type A -/
def x : ℕ := sorry

/-- Represents the discount amount for type A clothing -/
def a : ℝ := sorry

/-- The cost of type A clothing -/
def cost_A : ℕ := 80

/-- The selling price of type A clothing -/
def price_A : ℕ := 120

/-- The cost of type B clothing -/
def cost_B : ℕ := 60

/-- The selling price of type B clothing -/
def price_B : ℕ := 90

/-- The total number of pieces to be purchased -/
def total_pieces : ℕ := 100

/-- The maximum allowed cost for all pieces -/
def max_cost : ℕ := 7500

/-- The minimum number of type A pieces required -/
def min_A : ℕ := 65

theorem clothing_store_optimization :
  (x ≥ min_A) →
  (x ≤ total_pieces) →
  (cost_A * x + cost_B * (total_pieces - x) ≤ max_cost) →
  (0 < a) →
  (a < 10) →
  (∀ y : ℕ, y ≥ min_A → y ≤ total_pieces → cost_A * y + cost_B * (total_pieces - y) ≤ max_cost → x ≥ y) →
  (x = 75 ∧ 
   ∀ y : ℕ, y ≥ min_A → y ≤ total_pieces → cost_A * y + cost_B * (total_pieces - y) ≤ max_cost →
   (price_A - cost_A - a) * x + (price_B - cost_B) * (total_pieces - x) ≥
   (price_A - cost_A - a) * y + (price_B - cost_B) * (total_pieces - y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_store_optimization_l1294_129456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_a_gt_ten_b_necessary_not_sufficient_for_log_a_gt_log_b_l1294_129470

theorem ten_a_gt_ten_b_necessary_not_sufficient_for_log_a_gt_log_b :
  ∃ (a b : ℝ), (Real.log a > Real.log b → 10*a > 10*b) ∧
               ¬(∀ (a b : ℝ), 10*a > 10*b → Real.log a > Real.log b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_a_gt_ten_b_necessary_not_sufficient_for_log_a_gt_log_b_l1294_129470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_one_two_zeros_condition_l1294_129497

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), (∀ x, f 1 x ≥ m) ∧ (∃ x, f 1 x = m) ∧ m = -1 := by
  sorry

-- Theorem 2: Condition for f to have exactly 2 zeros
theorem two_zeros_condition (a : ℝ) :
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ 
  (a ≥ 1/2 ∧ a < 1) ∨ (a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_one_two_zeros_condition_l1294_129497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_health_code_survey_is_comprehensive_l1294_129495

/-- Represents a person --/
structure Person where

/-- Represents a survey --/
structure Survey where
  population : Set Person
  is_feasible : Bool

/-- Represents a supermarket --/
structure Supermarket where
  personnel : Set Person

/-- Defines what makes a survey comprehensive --/
def is_comprehensive (s : Survey) : Prop :=
  s.is_feasible ∧ (∀ p : Person, p ∈ s.population)

/-- The survey of health codes in a supermarket --/
def health_code_survey (sm : Supermarket) : Survey where
  population := sm.personnel
  is_feasible := true

/-- Theorem stating that the health code survey is comprehensive --/
theorem health_code_survey_is_comprehensive (sm : Supermarket) :
  is_comprehensive (health_code_survey sm) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_health_code_survey_is_comprehensive_l1294_129495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l1294_129421

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - x - 2 else x^2 + 2*x

-- Theorem statement
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l1294_129421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_height_proof_l1294_129417

/-- The side length of the square base tin in cm -/
def square_side : ℝ := 8

/-- The diameter of the circular base tin in cm -/
def circle_diameter : ℝ := 8

/-- The volume difference between the square and circular base tins in cm³ -/
def volume_difference : ℝ := 192.28324559588634

/-- The height of both tins in cm -/
def tin_height : ℝ := 14.005

theorem tin_height_proof :
  let square_volume := square_side ^ 2 * tin_height
  let circle_volume := π * (circle_diameter / 2) ^ 2 * tin_height
  square_volume - circle_volume = volume_difference := by
  sorry

#eval tin_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_height_proof_l1294_129417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_measure_l1294_129405

/-- Two complementary angles with a ratio of 4:3 -/
structure ComplementaryAngles where
  larger : ℝ
  smaller : ℝ
  complementary : larger + smaller = 90
  ratio : larger / smaller = 4 / 3

/-- The measure of the smaller angle is approximately 38.57° -/
theorem smaller_angle_measure (angles : ComplementaryAngles) :
  abs (angles.smaller - 38.57) < 0.01 := by
  sorry

#check smaller_angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_measure_l1294_129405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1294_129415

theorem power_of_three (a b : ℝ) (h1 : (3 : ℝ)^a = 5) (h2 : (9 : ℝ)^b = 10) : 
  (3 : ℝ)^(a + 2*b) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1294_129415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_prove_max_b_value_l1294_129457

def is_lattice_point (x y : ℤ) : Prop := True

theorem max_b_value (b : ℚ) : Prop :=
  (b = 50 / 151) ∧
  ∀ m : ℚ, 1/3 < m → m < b →
    ∀ x y : ℤ, 0 < x → x ≤ 150 → is_lattice_point x y →
      ↑y ≠ m * ↑x + 3 ∧
  ∀ b' : ℚ, b < b' →
    ∃ m : ℚ, 1/3 < m ∧ m < b' ∧
      ∃ x y : ℤ, 0 < x ∧ x ≤ 150 ∧ is_lattice_point x y ∧
        ↑y = m * ↑x + 3

theorem prove_max_b_value : ∃ b : ℚ, max_b_value b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_prove_max_b_value_l1294_129457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_three_thirty_l1294_129435

/-- Calculates the angle between hour and minute hands of a clock -/
noncomputable def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  |60 * (hour % 12 : ℝ) - 11 * (minute : ℝ)| / 2

/-- The angle between hour and minute hands at 3:30 is 75° -/
theorem angle_at_three_thirty : clockAngle 3 30 = 75 := by
  -- Unfold the definition of clockAngle
  unfold clockAngle
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_three_thirty_l1294_129435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1294_129476

/-- An arithmetic sequence {a_n} with the given properties --/
def a : ℕ → ℝ := sorry

/-- The sum of the first n terms of sequence {b_n} --/
def S : ℕ → ℝ := sorry

/-- Sequence {b_n} --/
def b : ℕ → ℝ := sorry

/-- Sequence {c_n} --/
def c : ℕ → ℝ := sorry

/-- The sum of the first n terms of sequence {c_n} --/
def T : ℕ → ℝ := sorry

/-- Check if a sequence is arithmetic --/
def IsArithmetic (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n + 1) - f n = d

/-- The main theorem encapsulating the problem and its solution --/
theorem sequence_problem (h1 : a 1 + a 2 + a 3 = 9)
                         (h2 : a 2 + a 8 = 18)
                         (h3 : ∀ n, S n = 2 * b n - 2)
                         (h4 : ∀ n, c n = a n / b n)
                         (h5 : IsArithmetic a) :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, b n = 2^n) ∧
  (∀ n, T n = 3 - (2 * n + 3) / 2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1294_129476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_bound_l1294_129489

/-- A two-dimensional figure containing polygons -/
structure Figure where
  n : ℕ  -- Total number of boundary lines
  polygons : List (ℕ)  -- List of polygon sides

/-- 
Theorem: In a two-dimensional figure containing a p-sided polygon and a q-sided polygon, 
where n is the total number of boundary lines in the figure, p + q ≤ n + 4.
-/
theorem polygon_sides_bound (fig : Figure) (p q : ℕ) 
  (hp : p ∈ fig.polygons)
  (hq : q ∈ fig.polygons)
  : p + q ≤ fig.n + 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_bound_l1294_129489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1294_129445

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a < b ∧ b > 0

/-- The focus of the ellipse -/
noncomputable def focus (e : Ellipse) (ecc : ℝ) : ℝ × ℝ := (-e.a * ecc, 0)

/-- A point on the ellipse -/
def point_A (e : Ellipse) : ℝ × ℝ := (0, e.b)

/-- The line connecting the focus and point A -/
noncomputable def line_FA (e : Ellipse) (ecc : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ Real.sqrt (1 - ecc^2) * x - ecc * y + e.a * ecc * Real.sqrt (1 - ecc^2) = 0

/-- The distance from the origin to the line FA -/
noncomputable def distance_O_to_FA (e : Ellipse) : ℝ := Real.sqrt 2 / 2 * e.b

/-- The line of symmetry -/
def line_l : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y = 0

/-- The circle O -/
def circle_O : ℝ → ℝ → Prop := fun x y ↦ x^2 + y^2 = 4

/-- The main theorem -/
theorem ellipse_properties (e : Ellipse) (ecc : ℝ) (P : ℝ × ℝ) :
  (∀ x y, x^2 / e.a^2 + y^2 / e.b^2 = 1 → (x, y) ∈ Set.range (fun t ↦ (t, 0))) →
  (∃ x y, line_FA e ecc x y ∧ (x - 0)^2 + (y - 0)^2 = (distance_O_to_FA e)^2) →
  (∃ x y, line_l x y ∧ x = (P.1 + (focus e ecc).1) / 2 ∧ y = (P.2 + (focus e ecc).2) / 2) →
  circle_O P.1 P.2 →
  ecc = Real.sqrt 2 / 2 ∧
  (∀ x y, x^2 / 8 + y^2 / 4 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  P = (6/5, 8/5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1294_129445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_pyramid_inscribed_sphere_touches_six_l1294_129481

/-- The radius of a sphere inscribed in the center of a triangular pyramid arrangement of identical spheres -/
noncomputable def inscribed_sphere_radius (circumscribed_radius : ℝ) : ℝ :=
  Real.sqrt 2 - 1

/-- Theorem stating the radius of the inscribed sphere in a specific arrangement -/
theorem inscribed_sphere_radius_in_pyramid (circumscribed_radius : ℝ) 
  (h : circumscribed_radius = Real.sqrt 6 + 1) :
  inscribed_sphere_radius circumscribed_radius = Real.sqrt 2 - 1 := by
  sorry

/-- Verifies that the inscribed sphere touches six identical spheres -/
def touches_six_spheres (inscribed_radius : ℝ) : Prop :=
  sorry

/-- Theorem stating that the inscribed sphere touches six identical spheres -/
theorem inscribed_sphere_touches_six (circumscribed_radius : ℝ) 
  (h : circumscribed_radius = Real.sqrt 6 + 1) :
  touches_six_spheres (inscribed_sphere_radius circumscribed_radius) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_pyramid_inscribed_sphere_touches_six_l1294_129481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l1294_129446

-- Define the system of differential equations
def system (x y : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, (deriv x t = -2 * x t - 4 * y t + 1 + 4 * t) ∧
           (deriv y t = -x t + y t + 3/2 * t^2)

-- Define the proposed solution
noncomputable def x_solution (C₁ C₂ : ℝ) (t : ℝ) : ℝ :=
  -C₁ * Real.exp (2 * t) + 4 * C₂ * Real.exp (-3 * t) + t + t^2

noncomputable def y_solution (C₁ C₂ : ℝ) (t : ℝ) : ℝ :=
  C₁ * Real.exp (2 * t) + C₂ * Real.exp (-3 * t) - 1/2 + t^2

-- Theorem statement
theorem solution_satisfies_system :
  ∀ C₁ C₂ : ℝ, system (x_solution C₁ C₂) (y_solution C₁ C₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l1294_129446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_preserving_rigid_motions_l1294_129442

/-- Represents the types of shapes in the pattern -/
inductive Shape
| Circle
| Square

/-- Represents a point on the line ℓ -/
structure Point where
  x : ℝ

/-- Represents the figure pattern on line ℓ -/
def Pattern : List Shape := [Shape.Circle, Shape.Circle, Shape.Square]

/-- Represents rigid motion transformations -/
inductive RigidMotion
| Rotation (center : Point) (angle : ℝ)
| Translation (distance : ℝ)
| ReflectionAcross
| ReflectionPerpendicular (point : Point)

/-- Checks if a rigid motion preserves the pattern -/
def preservesPattern (m : RigidMotion) : Prop :=
  match m with
  | RigidMotion.Rotation center angle => angle = Real.pi
  | RigidMotion.Translation distance => ∃ k : ℤ, distance = k * (3 : ℝ)
  | _ => False

/-- The main theorem stating that exactly two rigid motions preserve the pattern -/
theorem two_preserving_rigid_motions :
  ∃! (preserving : List RigidMotion),
    preserving.length = 2 ∧
    ∀ m : RigidMotion, m ∈ preserving ↔ preservesPattern m :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_preserving_rigid_motions_l1294_129442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_as_sine_and_sin_2alpha_l1294_129451

noncomputable def f (x : Real) : Real := 
  Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2 - Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_as_sine_and_sin_2alpha (α : Real) :
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) ∧
  (f α = 1 / 7 ∧ 0 < 2 * α ∧ 2 * α < Real.pi / 2 → Real.sin (2 * α) = 5 * Real.sqrt 3 / 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_as_sine_and_sin_2alpha_l1294_129451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_gets_2000_l1294_129402

noncomputable def total_savings : ℚ := 10000

noncomputable def natalie_share (total : ℚ) : ℚ := total / 2

noncomputable def rick_share (remaining : ℚ) : ℚ := remaining * (6 / 10)

noncomputable def lucy_share (total : ℚ) (natalie : ℚ) (rick : ℚ) : ℚ := total - natalie - rick

theorem lucy_gets_2000 :
  lucy_share total_savings (natalie_share total_savings) (rick_share (total_savings - natalie_share total_savings)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_gets_2000_l1294_129402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_is_30_l1294_129493

/-- The speed of the bus in km/h -/
noncomputable def bus_speed : ℝ := 30

/-- The speed of the pedestrian in km/h -/
noncomputable def pedestrian_speed : ℝ := 5

/-- The time interval of observation in hours -/
noncomputable def observation_time : ℝ := 2

/-- The time between bus encounters in hours -/
noncomputable def encounter_interval : ℝ := 1 / 12

/-- The difference between oncoming and overtaking buses -/
def bus_difference : ℕ := 4

theorem bus_speed_is_30 :
  ∃ (oncoming overtaking : ℕ),
    (oncoming : ℝ) * (bus_speed + pedestrian_speed) * observation_time = 
    (overtaking : ℝ) * (bus_speed - pedestrian_speed) * observation_time ∧
    oncoming - overtaking = bus_difference ∧
    (oncoming : ℝ) + overtaking = observation_time / encounter_interval :=
by sorry

#check bus_speed_is_30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_is_30_l1294_129493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l1294_129418

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: In a rhombus with area 157.5 and one diagonal of 15, the other diagonal is 21 -/
theorem rhombus_other_diagonal :
  ∃ (d2 : ℝ), rhombusArea 15 d2 = 157.5 ∧ d2 = 21 := by
  use 21
  constructor
  · simp [rhombusArea]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l1294_129418
