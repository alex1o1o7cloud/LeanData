import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l364_36474

/-- Calculates the time required to mow a rectangular lawn -/
noncomputable def mowing_time (length width swath_width speed : ℝ) : ℝ :=
  let strips := width / swath_width
  let total_distance := strips * length
  total_distance / speed

/-- Proves that mowing the given lawn takes approximately 1.8 hours -/
theorem lawn_mowing_time :
  let length : ℝ := 120
  let width : ℝ := 180
  let swath_width : ℝ := 2  -- Effective swath width after accounting for overlap
  let speed : ℝ := 6000
  let time := mowing_time length width swath_width speed
  ∃ ε > 0, |time - 1.8| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l364_36474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_125_l364_36404

theorem cube_root_of_125 : (125 : ℝ) ^ (1/3 : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_125_l364_36404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_moved_under_dilation_l364_36481

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Distance moved by a point under a specific dilation -/
theorem distance_moved_under_dilation 
  (original : Circle)
  (dilated : Circle)
  (p : Point) :
  original.center = Point.mk 4 (-3) →
  original.radius = 4 →
  dilated.center = Point.mk (-2) 9 →
  dilated.radius = 6 →
  p = Point.mk 1 1 →
  ∃ (center : Point), 
    distance p (Point.mk (center.x + 0.5 * (p.x - center.x)) (center.y + 0.5 * (p.y - center.y))) = 0.5 * Real.sqrt 2533 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_moved_under_dilation_l364_36481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l364_36492

theorem calculation_proof :
  (Real.sqrt (25 / 9) - (8 / 27) ^ (1 / 3 : ℝ) - (Real.pi + Real.exp 1) ^ (0 : ℝ) + (1 / 4) ^ (-(1 / 2 : ℝ)) = 2) ∧
  (2 * Real.log 5 / Real.log 10 + Real.log 4 / Real.log 10 + Real.log (Real.sqrt (Real.exp 1)) = 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l364_36492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l364_36490

-- Define the slope of a line given its coefficients
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

-- Define perpendicularity condition for two lines
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the lines
def line1 (m : ℝ) : (ℝ → ℝ → Prop) := λ x y => x + (m^2 - m) * y = 4 * m - 1
def line2 : (ℝ → ℝ → Prop) := λ x y => 2 * x - y - 5 = 0

-- Theorem statement
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, (are_perpendicular (line_slope 1 (m^2 - m)) (line_slope 2 (-1))) → (m = -1 ∨ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_values_l364_36490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l364_36459

open Real

/-- Given a triangle ABC with circumradius 1 and the given condition, 
    its area is at most 3√3/4 -/
theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- sum of angles in a triangle
  a = 2 * sin A →  -- sine theorem
  b = 2 * sin B →  -- sine theorem
  c = 2 * sin C →  -- sine theorem
  (tan A) / (tan B) = (2 * c - b) / b →  -- given condition
  a^2 = b^2 + c^2 - 2*b*c*cos A →  -- cosine theorem
  (1/2) * b * c * sin A ≤ 3 * sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l364_36459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_49_7_fourth_l364_36425

theorem log_49_7_fourth (x : ℝ) : x = (1/8 : ℝ) ↔ (49 : ℝ)^x = 7^(1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_49_7_fourth_l364_36425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l364_36453

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (((1 + 2 * Complex.I) * (1 + a * Complex.I)).re = 0 ∧ 
   ((1 + 2 * Complex.I) * (1 + a * Complex.I)).im ≠ 0) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l364_36453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_negative_range_l364_36483

theorem sin_minus_cos_negative_range (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (Real.sin x - Real.cos x < 0 ↔ x ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Ioo (5 * Real.pi / 4) (2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_negative_range_l364_36483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l364_36410

noncomputable section

-- Define the curve
def y (a x : ℝ) : ℝ := 1 + (Real.log x) / (Real.log a)

-- Define the derivative of y with respect to x
def y_derivative (a x : ℝ) : ℝ := 1 / (x * Real.log a)

-- Define the tangent line equation
def tangent_line (a x : ℝ) : ℝ := 1 + y_derivative a 1 * (x - 1)

-- Main theorem
theorem tangent_through_origin (a : ℝ) :
  a > 0 ∧ a ≠ 1 ∧ tangent_line a 0 = 0 → a = Real.exp 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l364_36410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l364_36467

/-- Line l in 2D space -/
def line_l (t : ℝ) : ℝ × ℝ := (t, 4 - t)

/-- Circle C in 2D space -/
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Distance from a point to line l -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 + p.2 - 4| / Real.sqrt 2

theorem max_distance_to_line :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧
  ∀ (p : ℝ × ℝ), p ∈ circle_C →
    distance_to_line p ≤ d ∧
    ∃ (q : ℝ × ℝ), q ∈ circle_C ∧ distance_to_line q = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l364_36467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_cost_l364_36496

/-- The cost of items at a baseball game -/
structure BaseballCosts where
  hotdog : ℚ
  softdrink : ℚ

/-- The quantity of items purchased by a group -/
structure GroupPurchase where
  hotdogs : ℕ
  softdrinks : ℕ

/-- Calculate the total cost of a group's purchase -/
def total_cost (costs : BaseballCosts) (purchase : GroupPurchase) : ℚ :=
  costs.hotdog * purchase.hotdogs + costs.softdrink * purchase.softdrinks

/-- Theorem: The first group's purchase costs $7.5 -/
theorem first_group_cost (costs : BaseballCosts) (first_group : GroupPurchase) 
  (h1 : costs.hotdog = 1/2)
  (h2 : costs.softdrink = 1/2)
  (h3 : first_group.hotdogs = 10)
  (h4 : first_group.softdrinks = 5) :
  total_cost costs first_group = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_cost_l364_36496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_k_values_l364_36400

-- Define the ellipse parameters
variable (a b : ℝ)
variable (h : a > b ∧ b > 0)

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (2, 0)

-- Define the point the ellipse passes through
noncomputable def pass_through : ℝ × ℝ := (2 * Real.sqrt 3, Real.sqrt 3)

-- Define line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the slope of line FN
def slope_FN : ℝ := -1

-- Define the relationship between FN, MN, and angle FON
def relationship (FN MN : ℝ) (angle_FON : ℝ) : Prop :=
  FN / MN = (2 * Real.sqrt 2 / 3) * Real.sin angle_FON

-- State the theorem
theorem ellipse_and_k_values 
  (h1 : ellipse_eq a b (right_focus.1) (right_focus.2))
  (h2 : ellipse_eq a b (pass_through.1) (pass_through.2))
  (h3 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ ellipse_eq a b x y ∧ y = line_l k x)
  (h4 : ∃ (x y : ℝ), y = line_l k x ∧ y = -x + 2)
  (h5 : ∃ (FN MN angle_FON : ℝ), relationship FN MN angle_FON) :
  (a^2 = 16 ∧ b^2 = 12) ∧ (k = 3/2 ∨ k = 9/26) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_k_values_l364_36400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_second_oil_price_l364_36414

/-- Given two oils mixed together, calculate the price of the second oil -/
theorem calculate_second_oil_price 
  (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (mixture_price : ℝ)
  (h1 : volume1 = 10)
  (h2 : price1 = 50)
  (h3 : volume2 = 5)
  (h4 : mixture_price = 55.67)
  : ∃ (price2 : ℝ), (price2 ≥ 67.00 ∧ price2 ≤ 67.02) ∧ 
    (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = mixture_price :=
by sorry

-- The #eval command is not necessary for building and can be removed
-- #eval calculate_second_oil_price 10 50 5 55.67

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_second_oil_price_l364_36414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_run_square_field_40m_9kmph_l364_36411

/-- The time taken for a boy to run around a square field -/
noncomputable def time_to_run_around_square_field (side_length : ℝ) (speed_km_per_hr : ℝ) : ℝ :=
  (4 * side_length) / (speed_km_per_hr * 1000 / 3600)

/-- Theorem: The time taken for a boy to run around a square field with side length 40 meters at a speed of 9 km/hr is 64 seconds -/
theorem time_to_run_square_field_40m_9kmph :
  time_to_run_around_square_field 40 9 = 64 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_run_around_square_field 40 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_run_square_field_40m_9kmph_l364_36411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coverage_impossible_l364_36412

/-- Represents a grid structure --/
structure Grid where
  vertices : ℕ
  odd_degree_vertices : ℕ

/-- Represents a path in the grid --/
structure GridPath where
  length : ℕ

/-- Checks if a grid can be covered by a given number of paths --/
def can_cover_grid (g : Grid) (paths : List GridPath) : Prop :=
  g.odd_degree_vertices = 2 * paths.length

/-- The specific grid from the problem --/
def problem_grid : Grid :=
  { vertices := 25,  -- Assuming a 5x5 grid based on the figure
    odd_degree_vertices := 12 }

/-- The paths we're trying to use --/
def problem_paths : List GridPath :=
  List.replicate 5 { length := 8 }

/-- Theorem stating that the specific grid cannot be covered by the given paths --/
theorem grid_coverage_impossible :
  ¬ (can_cover_grid problem_grid problem_paths) := by
  sorry

#eval problem_paths.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coverage_impossible_l364_36412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_translated_is_odd_l364_36469

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) * Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/8) - 1/2

theorem f_translated_is_odd : 
  ∀ x : ℝ, g x = -g (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_translated_is_odd_l364_36469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l364_36440

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2*x + 1) / (x - 1)

-- State the theorem
theorem min_value_of_f :
  ∀ x ∈ Set.Icc 2 4, f x ≥ 3 ∧ ∃ y ∈ Set.Icc 2 4, f y = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l364_36440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_form_set_correct_answer_l364_36458

-- Define the properties of a set
def is_set (S : Type) : Prop :=
  (∀ x y : S, x = y ∨ x ≠ y) ∧ 
  (∀ x y : S, x = y → y = x)

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Theorem: The totality of equilateral triangles forms a set
theorem equilateral_triangles_form_set : 
  is_set EquilateralTriangle := by
  constructor
  · intro x y
    exact Classical.em (x = y)
  · intro x y h
    exact h.symm

-- The answer is option C
def answer : String := "C"

theorem correct_answer : answer = "C" := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_form_set_correct_answer_l364_36458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l364_36418

theorem power_equation_solution (w : ℝ) : (2 : ℝ)^(2*w) = (8 : ℝ)^(w-4) → w = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l364_36418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l364_36463

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x / 4) + Real.pi / 6

theorem period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = 8 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l364_36463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l364_36409

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f a * f (2 * b) = Real.exp 2) : 
  (∀ x y, x > 0 → y > 0 → f x * f (2 * y) = Real.exp 2 → 1 / x + 2 / y ≥ 9 / 2) ∧ 
  (∃ x y, x > 0 ∧ y > 0 ∧ f x * f (2 * y) = Real.exp 2 ∧ 1 / x + 2 / y = 9 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l364_36409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_to_circle_l364_36462

/-- The length of the tangent segment from a point to a circle -/
noncomputable def tangent_length (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℝ :=
  Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2 - radius^2)

theorem tangent_length_to_circle : 
  let circle_center : ℝ × ℝ := (1, 1)
  let circle_radius : ℝ := 1
  let point_p : ℝ × ℝ := (2, 3)
  tangent_length circle_center circle_radius point_p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_to_circle_l364_36462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relations_l364_36461

/-- Given three real numbers x, y, and z satisfying certain conditions, 
    prove that they are approximately equal to specific values. -/
theorem number_relations (x y z : ℝ) 
  (h1 : 0.25 * x = 1.35 * 0.45 * y)
  (h2 : 0.30 * z = 0.5 * (0.25 * x - 0.45 * y))
  (h3 : x + y = 1200)
  (h4 : y - z = 250) :
  (abs (x - 850.15) < 0.01) ∧ (abs (y - 349.85) < 0.01) ∧ (abs (z - 99.85) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relations_l364_36461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_power_tower_l364_36447

theorem last_two_digits_of_power_tower (n : ℕ) :
  9^(8^(Nat.iterate (λ x => 7^x) n 2)) ≡ 21 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_power_tower_l364_36447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_minimum_integer_a_l364_36484

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Theorem for part (1)
theorem monotonic_decrease_interval (a : ℝ) (h : f a 1 = 0) :
  ∃ (l u : ℝ), l = 1 ∧ u = Real.pi ∧ 
  ∀ x y, l < x ∧ x < y ∧ y < u → f a y < f a x := by
  sorry

-- Theorem for part (2)
theorem minimum_integer_a :
  ∃ (a : ℕ), (∀ x, x > 0 → f (a : ℝ) x ≤ a * x - 1) ∧
  (∀ b : ℕ, b < a → ∃ x, x > 0 ∧ f (b : ℝ) x > b * x - 1) ∧
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_minimum_integer_a_l364_36484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_third_quadrant_l364_36441

theorem cos_B_third_quadrant (B : Real) : 
  (B > π ∧ B < 3*π/2) →  -- B is in the third quadrant
  Real.sin B = -5/13 → 
  Real.cos B = -12/13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_third_quadrant_l364_36441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_sum_l364_36444

theorem divisibility_of_power_sum (a b c d : ℤ) (e : ℤ) (h1 : e = a - b + c - d) 
  (h2 : Odd e) (h3 : e ∣ a^2 - b^2 + c^2 - d^2) :
  ∀ n : ℕ, e ∣ a^n - b^n + c^n - d^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_sum_l364_36444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l364_36482

/-- Given a > 0, b is the y-intercept of the tangent line y = x + b to the curve y = a ln x -/
noncomputable def b (a : ℝ) : ℝ := a * Real.log a - a

/-- The minimum value of b for a > 0 is -1 -/
theorem min_b_value :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, x > 0 → b x ≥ b a) → b a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l364_36482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l364_36426

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_eccentricity (C : Ellipse) (A P Q : Point) (l : Line) :
  C.a^2 * P.x^2 + C.b^2 * P.y^2 = C.a^2 * C.b^2 →  -- P is on the ellipse
  A.x = C.a ∧ A.y = 0 →  -- A is the right vertex
  l.m * 0 + l.c = 0 →  -- l passes through origin
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = C.a^2 →  -- |PQ| = a
  (A.x - P.x) * (P.x - Q.x) + (A.y - P.y) * (P.y - Q.y) = 0 →  -- AP ⟂ PQ
  eccentricity C = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l364_36426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_probability_theorem_l364_36427

/-- Represents a box containing good and defective products -/
structure Box where
  good : ℕ
  defective : ℕ

/-- Calculates the probability of selecting 2 defective products from a box -/
def prob_two_defective (b : Box) : ℚ :=
  (b.defective.choose 2 : ℚ) / ((b.good + b.defective).choose 2 : ℚ)

/-- Calculates the probability of selecting a good product from a box after adding 2 random products from another box -/
def prob_good_after_transfer (b1 b2 : Box) : ℚ :=
  let total_from := b1.good + b1.defective
  let p_two_good := (b1.good.choose 2 : ℚ) / (total_from.choose 2 : ℚ)
  let p_one_good_one_def := (b1.good.choose 1 * b1.defective.choose 1 : ℚ) / (total_from.choose 2 : ℚ)
  let p_two_def := (b1.defective.choose 2 : ℚ) / (total_from.choose 2 : ℚ)
  
  let p_good_if_two_good := (b2.good + 2 : ℚ) / (b2.good + b2.defective + 2 : ℚ)
  let p_good_if_one_good := (b2.good + 1 : ℚ) / (b2.good + b2.defective + 2 : ℚ)
  let p_good_if_zero_good := (b2.good : ℚ) / (b2.good + b2.defective + 2 : ℚ)
  
  p_two_good * p_good_if_two_good + p_one_good_one_def * p_good_if_one_good + p_two_def * p_good_if_zero_good

theorem box_probability_theorem (box_a box_b : Box) 
    (ha : box_a.good = 5 ∧ box_a.defective = 3) 
    (hb : box_b.good = 4 ∧ box_b.defective = 3) : 
    prob_two_defective box_a = 3/28 ∧ 
    prob_good_after_transfer box_a box_b = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_probability_theorem_l364_36427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_pretzel_cost_is_six_l364_36432

/-- The cost of a small pretzel in cents -/
def small_pretzel_cost : ℚ := sorry

/-- The cost of a large pretzel in cents -/
def large_pretzel_cost : ℚ := sorry

/-- One large pretzel costs the same as three small pretzels -/
axiom large_small_relation : large_pretzel_cost = 3 * small_pretzel_cost

/-- Seven large pretzels and four small pretzels cost twelve cents more than four large pretzels and seven small pretzels -/
axiom cost_difference : 7 * large_pretzel_cost + 4 * small_pretzel_cost = 4 * large_pretzel_cost + 7 * small_pretzel_cost + 12

theorem large_pretzel_cost_is_six : large_pretzel_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_pretzel_cost_is_six_l364_36432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_system_initial_volume_l364_36415

/-- Represents the cooling system state -/
structure CoolingSystem where
  initialVolume : ℝ
  initialConcentration : ℝ
  finalConcentration : ℝ
  remainingOriginalVolume : ℝ
  replacementConcentration : ℝ

/-- The cooling system satisfies the problem conditions -/
def validCoolingSystem (cs : CoolingSystem) : Prop :=
  cs.initialConcentration = 0.30 ∧
  cs.finalConcentration = 0.50 ∧
  cs.remainingOriginalVolume = 7.6 ∧
  cs.replacementConcentration = 0.80

/-- The theorem stating the initial volume of coolant -/
theorem cooling_system_initial_volume (cs : CoolingSystem) 
  (h : validCoolingSystem cs) : 
  ∃ ε > 0, |cs.initialVolume - 10.13| < ε := by
  sorry

#check cooling_system_initial_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_system_initial_volume_l364_36415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l364_36428

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := sorry
noncomputable def F2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the property that A and B are on a line passing through F1
def on_line_through_F1 (P : ℝ × ℝ) : Prop := sorry

-- Define the distance function
noncomputable def my_dist (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem ellipse_triangle_perimeter :
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  on_line_through_F1 A →
  on_line_through_F1 B →
  my_dist A F1 + my_dist A F2 = 10 →
  my_dist B F1 + my_dist B F2 = 10 →
  my_dist A B + my_dist A F2 + my_dist B F2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l364_36428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_shifted_l364_36406

theorem cos_double_angle_from_shifted (α : ℝ) :
  Real.cos (π / 2 + α) = 1 / 3 → Real.cos (2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_shifted_l364_36406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l364_36431

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 6)

theorem function_properties (φ : ℝ) 
    (h1 : 0 < φ ∧ φ < Real.pi / 2)
    (h2 : ∀ x, f (Real.pi / 6) (x + Real.pi / 6) = f (Real.pi / 6) (-x + Real.pi / 6)) :
  (∀ x, f φ x = f (Real.pi / 6) x) ∧
  (∀ x, g x = 3 * Real.sin (x + Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 2), 
    g x ≤ 3 ∧ g x ≥ -3/2 ∧
    (g x = 3 ∨ g x = -3/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l364_36431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l364_36435

/-- A quadratic function with a negative leading coefficient -/
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The axis of symmetry of the quadratic function -/
noncomputable def axis_of_symmetry (b : ℝ) : ℝ := b / 2

theorem quadratic_inequality (b c : ℝ) (h : axis_of_symmetry b = 2) :
  f b c 2 > f b c 1 ∧ f b c 1 > f b c 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l364_36435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_parallel_condition_l364_36419

/-- Two lines are parallel if and only if their slopes are equal -/
theorem parallel_lines {m1 m2 : ℚ} : m1 = m2 ↔ m1 = m2 := by
  simp

/-- The slope of a line ax + by + c = 0 is -a/b -/
noncomputable def line_slope (a b : ℚ) : ℚ := -a / b

theorem parallel_condition (a : ℚ) :
  (a = 1) ↔ 
  line_slope a 2 = line_slope (a + 1) 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_parallel_condition_l364_36419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l364_36460

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The distance from a point to a focus -/
noncomputable def distance_to_focus (x y c : ℝ) : ℝ :=
  Real.sqrt ((x + c)^2 + y^2)

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (C : Ellipse) 
  (h3 : C.b^2 = 3) 
  (h4 : distance_to_focus 0 (Real.sqrt 3) (-Real.sqrt (C.a^2 - C.b^2)) + 
        distance_to_focus 0 (Real.sqrt 3) (Real.sqrt (C.a^2 - C.b^2)) = 4) :
  (C.a = 2 ∧ 
   C.b = Real.sqrt 3 ∧ 
   2 * C.b = 2 * Real.sqrt 3 ∧
   2 * Real.sqrt (C.a^2 - C.b^2) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l364_36460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_formed_by_circle_centers_l364_36488

noncomputable section

-- Define the radii of the circles
def R₁ : ℝ := 6
def R₂ : ℝ := 7
def R₃ : ℝ := 8

-- Define the sides of the triangle
def a : ℝ := R₁ + R₂
def b : ℝ := R₂ + R₃
def c : ℝ := R₁ + R₃

-- Define the semi-perimeter
def p : ℝ := (a + b + c) / 2

-- State the theorem
theorem area_of_triangle_formed_by_circle_centers : 
  Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 84 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_formed_by_circle_centers_l364_36488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_proof_l364_36423

/-- Given that z is a pure imaginary number and (2+i)z = 1+ai^3, prove that |a+z| = √5 --/
theorem complex_equation_proof (z : ℂ) (a : ℝ) 
  (h1 : z.re = 0)  -- z is pure imaginary
  (h2 : (2 + Complex.I) * z = 1 + a * Complex.I^3) : 
  Complex.abs (a + z) = Real.sqrt 5 := by
  sorry  -- Proof steps would go here

#check complex_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_proof_l364_36423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_sqrt7_and_sqrt77_l364_36498

theorem integers_between_sqrt7_and_sqrt77 :
  (∃! n : ℕ, (∀ k : ℤ, (Real.sqrt 7 : ℝ) < k ∧ k < (Real.sqrt 77 : ℝ) → k.natAbs ≤ n) ∧
             (∀ m : ℕ, m < n → ∃ k : ℤ, (Real.sqrt 7 : ℝ) < k ∧ k < (Real.sqrt 77 : ℝ) ∧ k.natAbs = m + 1)) ∧
  (∃ n : ℕ, n = 6 ∧ (∀ k : ℤ, (Real.sqrt 7 : ℝ) < k ∧ k < (Real.sqrt 77 : ℝ) → k.natAbs ≤ n) ∧
             (∀ m : ℕ, m < n → ∃ k : ℤ, (Real.sqrt 7 : ℝ) < k ∧ k < (Real.sqrt 77 : ℝ) ∧ k.natAbs = m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_sqrt7_and_sqrt77_l364_36498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_angle_l364_36421

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_slope_angle :
  ∀ (θ₁ θ₂ t₁ t₂ α : ℝ),
    let M := line_l t₁ α
    let N := line_l t₂ α
    curve_C θ₁ = M →
    curve_C θ₂ = N →
    distance M N = Real.sqrt 10 →
    Real.sin (2 * α) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_angle_l364_36421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_cos_2alpha_l364_36413

theorem parallel_vectors_cos_2alpha (α : ℝ) :
  let a : Fin 2 → ℝ := ![1/3, Real.tan α]
  let b : Fin 2 → ℝ := ![Real.cos α, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  Real.cos (2 * α) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_cos_2alpha_l364_36413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_fill_time_l364_36457

/-- Represents the time taken to fill a bucket completely -/
noncomputable def fill_time : ℚ := 3

/-- Represents the fraction of the bucket filled in 2 minutes -/
noncomputable def fraction_filled : ℚ := 2/3

/-- Represents the time taken to fill the given fraction of the bucket -/
noncomputable def partial_fill_time : ℚ := 2

theorem bucket_fill_time :
  fraction_filled * fill_time = partial_fill_time := by
  -- Convert fractions to decimals for exact calculation
  have h1 : (2:ℚ)/3 * 3 = 2
  · norm_num
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_fill_time_l364_36457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_correct_l364_36454

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_Wins
| B_Wins

/-- Represents the state of the match -/
structure MatchState :=
  (score_A : Nat)
  (score_B : Nat)
  (games_played : Nat)

/-- Determines if the match is over -/
def is_match_over (state : MatchState) : Bool :=
  (state.score_A ≥ state.score_B + 2) ∨ 
  (state.score_B ≥ state.score_A + 2) ∨ 
  (state.games_played = 6)

/-- Probability of A winning a single game -/
noncomputable def prob_A_wins : ℝ := 2/3

/-- Probability of B winning a single game -/
noncomputable def prob_B_wins : ℝ := 1/3

/-- Expected number of games played -/
noncomputable def expected_games : ℝ := 266/81

theorem expected_games_correct : 
  expected_games = 
    2 * (prob_A_wins^2 + prob_B_wins^2) + 
    4 * (2 * (prob_A_wins^3 * prob_B_wins + prob_B_wins^3 * prob_A_wins)) + 
    6 * (1 - (prob_A_wins^2 + prob_B_wins^2) - 2 * (prob_A_wins^3 * prob_B_wins + prob_B_wins^3 * prob_A_wins)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_correct_l364_36454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_translation_of_sine_l364_36468

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem symmetric_translation_of_sine (φ : ℝ) : 
  (0 ≤ φ ∧ φ ≤ π) →
  (∀ x, f φ (x - π/6) = f φ (-x - π/6)) →
  φ = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_translation_of_sine_l364_36468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_upper_bound_l364_36476

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (1/2)c^2 and ab = √2, then a^2 + b^2 + c^2 ≤ 4. -/
theorem triangle_side_sum_upper_bound (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_area : (1/2) * c^2 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))))
    (h_ab : a * b = Real.sqrt 2) :
    a^2 + b^2 + c^2 ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_upper_bound_l364_36476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_quarter_circles_area_l364_36480

/-- The area of the region inside a regular hexagon with side length 4 but outside
    all quarter circles drawn at each vertex with radius 4. -/
theorem hexagon_quarter_circles_area : 
  let hexagon_side : ℝ := 4
  let hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * hexagon_side ^ 2
  let quarter_circle_area : ℝ := Real.pi * hexagon_side ^ 2 / 4
  let total_quarter_circles_area : ℝ := 6 * quarter_circle_area
  hexagon_area - total_quarter_circles_area = 48 * Real.sqrt 3 - 24 * Real.pi :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_quarter_circles_area_l364_36480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_leftover_correct_l364_36438

def apple_leftover (total_greg_sarah : ℕ) (susan_multiplier : ℕ) (mark_difference : ℕ) 
  (emily_addition : ℚ) (mom_needed : ℕ) : ℕ :=
  let greg_apples := total_greg_sarah / 2
  let sarah_apples := total_greg_sarah / 2
  let susan_apples := greg_apples * susan_multiplier
  let mark_apples := susan_apples - mark_difference
  let emily_apples := (mark_apples : ℚ) + emily_addition
  let total_apples := greg_apples + sarah_apples + susan_apples + mark_apples + emily_apples.floor
  (total_apples - mom_needed).natAbs

#eval apple_leftover 18 2 5 (3/2) 40

theorem apple_leftover_correct : apple_leftover 18 2 5 (3/2) 40 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_leftover_correct_l364_36438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_translated_odd_l364_36499

/-- A function representing the cosine after translation -/
noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ - Real.pi / 3)

/-- The condition for f to be an odd function -/
def is_odd (φ : ℝ) : Prop := ∀ x, f x φ = -f (-x) φ

/-- Theorem stating that φ = 5π/6 makes f an odd function -/
theorem cos_translated_odd :
  is_odd (5 * Real.pi / 6) := by
  sorry

#check cos_translated_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_translated_odd_l364_36499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_weight_avg_distance_correlation_negative_l364_36416

/-- Represents the weight of a car -/
def car_weight : ℝ → ℝ := sorry

/-- Represents the average distance a car can travel per liter of fuel consumed -/
def avg_distance_per_liter : ℝ → ℝ := sorry

/-- Represents the correlation between two variables -/
def correlation : (ℝ → ℝ) → (ℝ → ℝ) → ℝ := sorry

/-- Theorem stating that the correlation between car weight and average distance per liter is negative -/
theorem car_weight_avg_distance_correlation_negative :
  correlation car_weight avg_distance_per_liter < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_weight_avg_distance_correlation_negative_l364_36416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_theorem_l364_36424

theorem frog_jump_theorem (p q : ℕ) (h_coprime : Nat.Coprime p q) : 
  ∃ (jumps : List ℤ), 
    (jumps.head? = some 0) ∧ 
    (jumps.getLast? = some 0) ∧
    (∀ (i : ℕ) (hi : i < jumps.length - 1), 
      (jumps.get ⟨i + 1, by sorry⟩ - jumps.get ⟨i, by sorry⟩ = p) ∨ 
      (jumps.get ⟨i + 1, by sorry⟩ - jumps.get ⟨i, by sorry⟩ = -q)) →
    ∀ (d : ℕ), d < p + q → 
      ∃ (i j : ℕ) (hi : i < jumps.length) (hj : j < jumps.length),
        Int.natAbs (jumps.get ⟨i, hi⟩ - jumps.get ⟨j, hj⟩) = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_theorem_l364_36424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_arc_length_l364_36464

noncomputable def polar_curve (φ : ℝ) : ℝ := 5 * Real.exp (5 * φ / 12)

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (f x ^ 2 + (deriv f x) ^ 2)

theorem polar_curve_arc_length :
  arc_length polar_curve 0 (π / 3) = 13 * (Real.exp (5 * π / 36) - 1) := by
  sorry

#check polar_curve_arc_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_arc_length_l364_36464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_theorem_l364_36451

def sequence_15 (n : ℕ) : ℕ := 15 * n

theorem multiples_of_15_theorem :
  ∃ (n : ℕ), sequence_15 n < 2016 ∧ 2016 < sequence_15 (n + 1) ∧
  sequence_15 (n + 1) - 2016 = 9 := by
  use 134
  have h1 : sequence_15 134 = 2010 := by rfl
  have h2 : sequence_15 135 = 2025 := by rfl
  refine ⟨?_, ?_, ?_⟩
  · exact Nat.lt_of_sub_eq_succ rfl
  · exact Nat.lt_of_sub_eq_succ rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_theorem_l364_36451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l364_36446

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := (x + y - 12)^2 + y = 1
def equation2 (x y : ℝ) : Prop := y = -abs x

-- Define the solution
def solution : ℝ × ℝ := (143, -143)

-- Theorem statement
theorem unique_solution :
  (∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2) ∧
  (equation1 solution.1 solution.2 ∧ equation2 solution.1 solution.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l364_36446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_intersecting_octahedra_l364_36430

/-- The region defined by the given inequalities -/
def Region : Set (Fin 3 → ℝ) :=
  {p | (|p 0| + |p 1| + |p 2| ≤ 2) ∧ (|p 0| + |p 1| + |p 2 - 2| ≤ 2)}

/-- The volume of the region -/
noncomputable def volume_of_region : ℝ := sorry

/-- Theorem stating that the volume of the region is 16/3 -/
theorem volume_of_intersecting_octahedra : volume_of_region = 16/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_intersecting_octahedra_l364_36430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l364_36434

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 1, and three congruent quadrilaterals between them, prove that the
    area of one quadrilateral is 35/3. -/
theorem quadrilateral_area (outer_triangle : Set (EuclideanSpace ℝ (Fin 2)))
  (inner_triangle : Set (EuclideanSpace ℝ (Fin 2)))
  (quadrilaterals : Fin 3 → Set (EuclideanSpace ℝ (Fin 2))) :
  MeasureTheory.volume outer_triangle = 36 →
  MeasureTheory.volume inner_triangle = 1 →
  (∀ i j : Fin 3, i ≠ j → quadrilaterals i ≃ᵐ quadrilaterals j) →
  (∀ i : Fin 3, MeasureTheory.volume (quadrilaterals i) = 35 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l364_36434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l364_36403

theorem quadratic_function_range
  (a b c : ℝ) 
  (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  let range := Set.range (fun x ↦ f x)
  let domain := Set.Icc 0 2
  Set.Icc (min ((-b^2) / (4*a) + c) (min c (4*a + 2*b + c))) (max c (4*a + 2*b + c)) = 
    {y | ∃ x ∈ domain, f x = y} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l364_36403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_is_2_to_3_l364_36489

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ+) : ℝ := f a n

-- Theorem statement
theorem a_range_is_2_to_3 :
  ∀ a : ℝ,
  (∀ n : ℕ+, a_n a n < a_n a (n + 1)) ∧
  (∀ x : ℝ, x > 7 → f a x > 0) ∧
  (∀ x : ℝ, x ≤ 7 → f a x > 0) ↔
  2 < a ∧ a < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_is_2_to_3_l364_36489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l364_36478

theorem max_true_statements (x y : ℝ) : 
  (∃ (s : Fin 5 → Bool), 
    (s 0 → 1/x > 1/y) ∧ 
    (s 1 → x^2 < y^2) ∧ 
    (s 2 → x > y) ∧ 
    (s 3 → x > 0) ∧ 
    (s 4 → y > 0) ∧ 
    ((Finset.sum Finset.univ (fun i => if s i then 1 else 0)) ≤ 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l364_36478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_sum_theorem_l364_36420

def weight_sum (weights : List ℝ) : ℝ := weights.sum

theorem weight_sum_theorem (weights : List ℝ) (h1 : weights.length = 20) 
  (h2 : ∀ w ∈ weights, 60 ≤ w ∧ w ≤ 90) 
  (h3 : (weight_sum weights + 47) / 21 = weight_sum weights / 20 - 5) : 
  weight_sum weights = 3040 := by
  sorry

#check weight_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_sum_theorem_l364_36420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_extrema_l364_36466

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := x + 1/x + 2

-- Define the symmetry condition
def symmetric_to_h (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ h (-x) = 2 - y

-- Main theorem
theorem symmetry_and_extrema (f : ℝ → ℝ) 
  (h_sym : symmetric_to_h f) :
  (∀ x, f x = x + 1/x) ∧ 
  (∀ x ∈ Set.Ioo 0 8, f x ≥ 2) ∧
  (f 1 = 2) ∧
  (∀ x ∈ Set.Ioo 0 8, f x ≤ 65/8) ∧
  (f 8 = 65/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_extrema_l364_36466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l364_36436

theorem sin_double_angle_special (α : ℝ) 
  (h1 : α > -π/2 ∧ α < π/2) 
  (h2 : Real.cos (α + π/6) = 1/5) : 
  Real.sin (2*α + π/3) = 4*Real.sqrt 6/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l364_36436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_reading_speed_reading_time_l364_36486

/-- Represents the reading speed and time for Juan's book reading scenario -/
structure ReadingScenario where
  S : ℝ  -- Speed of reading the first book in pages per hour
  T : ℝ  -- Time to read the first book in hours

/-- Calculates the total pages read in one hour given a ReadingScenario -/
noncomputable def totalPagesPerHour (scenario : ReadingScenario) : ℝ :=
  (4 * scenario.S + 3 * scenario.S) / 4

/-- Theorem stating that Juan reads 1.75S pages in total from both books in one hour -/
theorem juan_reading_speed (scenario : ReadingScenario) :
    totalPagesPerHour scenario = 1.75 * scenario.S := by
  unfold totalPagesPerHour
  -- The proof steps would go here, but we'll use sorry for now
  sorry

/-- Theorem stating that the time to read the first book is 4 hours -/
theorem reading_time (scenario : ReadingScenario) :
    scenario.T = 4 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_reading_speed_reading_time_l364_36486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_480_calories_l364_36450

/-- Represents a snack with its calorie content and cost -/
structure Snack where
  calories : ℕ
  cost : ℚ

/-- Calculates the total calories from a given number of snacks -/
def totalCalories (snack : Snack) (quantity : ℕ) : ℕ :=
  snack.calories * quantity

/-- Calculates the total cost from a given number of snacks -/
def totalCost (snack : Snack) (quantity : ℕ) : ℚ :=
  snack.cost * quantity

/-- Theorem: The minimal amount Peter needs to spend for 480 calories is $4 -/
theorem min_cost_for_480_calories : ∃ (min_cost : ℚ), min_cost = 4 := by
  let chip : Snack := { calories := 10, cost := 2 / 24 }
  let chocolate : Snack := { calories := 200, cost := 1 }
  let cookie : Snack := { calories := 50, cost := 1 / 2 }
  let target_calories : ℕ := 480
  let min_cost : ℚ := 4

  have h1 : ∀ c ch co : ℕ,
    totalCalories chip c + totalCalories chocolate ch + totalCalories cookie co = target_calories →
    totalCost chip c + totalCost chocolate ch + totalCost cookie co ≥ min_cost := by
    sorry

  have h2 : ∃ c ch co : ℕ,
    totalCalories chip c + totalCalories chocolate ch + totalCalories cookie co = target_calories ∧
    totalCost chip c + totalCost chocolate ch + totalCost cookie co = min_cost := by
    sorry

  exact ⟨min_cost, rfl⟩

#check min_cost_for_480_calories

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_480_calories_l364_36450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_payment_range_l364_36475

/-- Represents a person with a certain amount of Reos --/
structure Person where
  reos : ℕ

/-- Represents a group of three people --/
structure Group3 where
  people : Fin 3 → Person

/-- Defines the ability to pay within a specified range --/
def canPay (amount : ℕ) : Prop :=
  1 ≤ amount ∧ amount ≤ 15

/-- Defines the collective ability of a group to pay an amount --/
def groupCanPay (g : Group3) (amount : ℕ) : Prop :=
  ∃ (p1 p2 p3 : ℕ), p1 + p2 + p3 = amount ∧
    (g.people 0).reos + (g.people 1).reos + (g.people 2).reos ≥ amount

/-- The main theorem --/
theorem group_payment_range (g : Group3) 
    (h1 : ∀ i, (g.people i).reos = 60)
    (h2 : ∀ i j, i ≠ j → ∀ amount, canPay amount → 
           ∃ newI newJ, newI + newJ = (g.people i).reos + (g.people j).reos ∧ 
           (g.people i).reos - 15 ≤ newI ∧ newI ≤ (g.people i).reos + 15) :
    ∀ amount, 45 ≤ amount ∧ amount ≤ 135 → groupCanPay g amount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_payment_range_l364_36475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_circles_distance_l364_36494

/-- Given two externally tangent circles and their common external tangent,
    calculate the distance from the center of the larger circle to the
    tangent point on the smaller circle. -/
theorem externally_tangent_circles_distance 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ = 10) 
  (h₂ : r₂ = 4) 
  (h₃ : r₁ > r₂) : 
  Real.sqrt ((r₁ + r₂)^2 - (r₁ - r₂)^2 + r₁^2) = 2 * Real.sqrt 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_circles_distance_l364_36494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l364_36495

/-- Calculates the time (in seconds) it takes for a train to cross a man walking in the same direction --/
noncomputable def time_to_cross (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / (train_speed - man_speed)

/-- Converts speed from km/hr to m/s --/
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

theorem train_crossing_time :
  let train_length : ℝ := 500
  let train_speed_km_hr : ℝ := 75
  let man_speed_km_hr : ℝ := 3
  let train_speed_m_s := km_per_hr_to_m_per_s train_speed_km_hr
  let man_speed_m_s := km_per_hr_to_m_per_s man_speed_km_hr
  time_to_cross train_length train_speed_m_s man_speed_m_s = 25 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l364_36495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_included_rectangle_ratio_l364_36465

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

theorem included_rectangle_ratio 
  (larger_rect : Rectangle) 
  (squares : Finset Square) 
  (included_rect : Rectangle) :
  larger_rect.length = 3 * larger_rect.width →
  squares.card = 5 →
  (∀ s ∈ squares, s.side = larger_rect.width) →
  included_rect.width + (squares.card - 2) * larger_rect.width = larger_rect.width →
  included_rect.length = larger_rect.length →
  included_rect.length / included_rect.width = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_included_rectangle_ratio_l364_36465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_implies_a_value_l364_36487

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + Real.cos (2 * x)

theorem axis_of_symmetry_implies_a_value :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = f a (π / 6 - x)) → 
  a = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_implies_a_value_l364_36487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_8_is_8_or_64_l364_36470

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a_1_eq_1 : a 1 = 1
  is_geometric : (a 1) * (a 5) = (a 2)^2

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem sum_8_is_8_or_64 (seq : ArithmeticSequence) : 
  sum_n seq 8 = 8 ∨ sum_n seq 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_8_is_8_or_64_l364_36470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_juice_water_percentage_l364_36497

/-- Calculates the percentage of water in tomato juice given the initial volume of juice and final volume of puree -/
noncomputable def water_percentage (initial_volume : ℝ) (final_volume : ℝ) : ℝ :=
  ((initial_volume - final_volume) / initial_volume) * 100

/-- Theorem stating that the percentage of water in tomato juice is 87.5% when 20 litres of juice produces 2.5 litres of puree -/
theorem tomato_juice_water_percentage :
  water_percentage 20 2.5 = 87.5 := by
  -- Unfold the definition of water_percentage
  unfold water_percentage
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- The following line is commented out because it's not computable
-- #eval water_percentage 20 2.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_juice_water_percentage_l364_36497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l364_36407

theorem min_sin6_cos6 (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l364_36407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l364_36471

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = Real.sqrt 3) :
  z^(2010 : ℤ) + z^(-(2010 : ℤ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l364_36471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l364_36477

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
  N = !![24/7, 20/7; -49/14, 36/7] ∧
  N.mulVec ![4, -2] = ![8, -18] ∧
  N.mulVec ![2, 3] = ![14, 9] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l364_36477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_increase_l364_36448

theorem cube_edge_increase (a x : ℝ) (h : a > 0) :
  let new_surface_area := 6 * (a * (1 + x))^2
  let original_surface_area := 6 * a^2
  new_surface_area = original_surface_area * 2.25 →
  x = 0.5 := by
  intro h1
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_increase_l364_36448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_cosine_angle_l364_36405

noncomputable section

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-2, -4]

def c (k : ℝ) : Fin 2 → ℝ := fun i => a i + k * b i

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt (dot_product v v)

theorem perpendicular_vectors_k (k : ℝ) : 
  dot_product b (c k) = 0 → k = 7/10 := by sorry

theorem cosine_angle :
  dot_product a b / (magnitude a * magnitude b) = -7 * Real.sqrt 65 / 65 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_cosine_angle_l364_36405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l364_36485

/-- The function f(x) in the binomial expansion -/
noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

/-- The function g(x) in the binomial expansion -/
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

/-- The coefficient of x^2 in the binomial expansion of (f(x) + g(x))^5 -/
def coefficient : ℕ := 40

/-- Theorem stating that the coefficient of x^2 in the binomial expansion of (f(x) + g(x))^5 is 40 -/
theorem binomial_expansion_coefficient :
  coefficient = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l364_36485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l364_36452

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ p = Real.pi ∧ ∀ x : ℝ, f (x + p) = f x) ∧
  (∀ x : ℝ, f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) ∧
  (∀ x y : ℝ, -Real.pi / 6 < x ∧ x < y ∧ y < Real.pi / 3 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l364_36452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_variance_abc_l364_36417

-- Define the variance function as noncomputable
noncomputable def variance (a b c : ℝ) : ℝ :=
  let mean := (a + b + c) / 3
  ((a - mean)^2 + (b - mean)^2 + (c - mean)^2) / 3

-- State the theorem
theorem max_variance_abc :
  ∀ a b c : ℝ,
  a ≥ 0 → b ≥ 0 → c ≥ 0 →  -- non-negative conditions
  a > 0 →                  -- a is strictly positive
  a + b + c = 6 →          -- sum condition
  variance a b c ≤ 8 :=    -- maximum variance is 8
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_variance_abc_l364_36417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l364_36437

/-- An inverse proportion function with parameter m -/
noncomputable def inverse_proportion (m : ℝ) : ℝ → ℝ := fun x ↦ (m - 2) / x

/-- Predicate to check if a function's graph is in the first and third quadrants -/
def in_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

/-- 
If the graph of the inverse proportion function y = (m-2)/x is in the first and third quadrants,
then m > 2
-/
theorem inverse_proportion_quadrants (m : ℝ) :
  in_first_and_third_quadrants (inverse_proportion m) → m > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l364_36437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l364_36455

open BigOperators Nat

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem factorial_divisibility : 
  (is_divisible (factorial 24) (5^4)) ∧ 
  ¬(is_divisible (factorial 24) (5^5)) ∧ 
  ∀ m : ℕ, m > 24 → is_divisible (factorial m) (5^5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l364_36455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_equivalence_l364_36493

theorem log_inequality_equivalence (a x : ℝ) (h1 : a > 0) (h2 : a ≠ 1/2) :
  (0 < a ∧ a < 1/2 → (Real.log (a + 2*x - x^2) / Real.log (Real.sqrt (2*a)) < 2 ↔ x^2 - 2*x + a < 0)) ∧
  (a > 1/2 → (Real.log (a + 2*x - x^2) / Real.log (Real.sqrt (2*a)) < 2 ↔ x^2 - 2*x + a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_equivalence_l364_36493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_comparison_quadratic_comparison_l364_36491

-- Part I
theorem sqrt_comparison : Real.sqrt 11 + Real.sqrt 3 < Real.sqrt 9 + Real.sqrt 5 := by sorry

-- Part II
theorem quadratic_comparison (x : ℝ) :
  (x < -3 ∨ x > 9 → x^2 + 5*x + 16 < 2*x^2 - x - 11) ∧
  (-3 < x ∧ x < 9 → x^2 + 5*x + 16 > 2*x^2 - x - 11) ∧
  (x = -3 ∨ x = 9 → x^2 + 5*x + 16 = 2*x^2 - x - 11) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_comparison_quadratic_comparison_l364_36491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_c_value_l364_36472

-- Define the vector type
abbrev Vec2 := ℝ × ℝ

-- Define the given vectors
def a : Vec2 := (1, 2)
def b : Vec2 := (2, -3)

-- Define parallel and perpendicular operations
def parallel (v w : Vec2) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

def perpendicular (v w : Vec2) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem vector_c_value :
  ∀ c : Vec2,
    parallel (c.1 + a.1, c.2 + a.2) b ∧ perpendicular c (a.1 + b.1, a.2 + b.2) →
    c = (-7/9, -20/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_c_value_l364_36472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_distances_l364_36433

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define A and B as intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Statement of the theorem
theorem min_reciprocal_distances :
  ∃ (α : ℝ), 0 ≤ α ∧ α ≤ Real.pi ∧
  (∀ (β : ℝ), 0 ≤ β ∧ β ≤ Real.pi →
    1 / dist point_P A + 1 / dist point_P B ≥ 2 * Real.sqrt 7 / 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_distances_l364_36433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_digits_base5_625_l364_36479

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a digit is even in base 5 --/
def isEvenDigitBase5 (d : ℕ) : Bool :=
  d = 0 || d = 2 || d = 4

/-- Counts the number of even digits in a list of base 5 digits --/
def countEvenDigitsBase5 (digits : List ℕ) : ℕ :=
  digits.filter isEvenDigitBase5 |>.length

theorem even_digits_base5_625 :
  countEvenDigitsBase5 (toBase5 625) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_digits_base5_625_l364_36479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l364_36408

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * sin (π / 6 - 2 * x)

-- State the theorem
theorem f_increasing_interval :
  ∀ x ∈ Set.Icc (π / 3) (5 * π / 6),
    x ∈ Set.Icc 0 π →
    ∀ y ∈ Set.Icc (π / 3) (5 * π / 6),
      x < y → f x < f y :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l364_36408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_x_f_zeros_f_two_zeros_f_no_zeros_f_one_zero_l364_36445

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x - a

-- Theorem 1: f(x) > x when a = 0
theorem f_greater_than_x : ∀ x : ℝ, Real.exp x - x > x := by sorry

-- Theorem 2: Number of zeros of f(x)
theorem f_zeros (a : ℝ) :
  (∃! x y : ℝ, f a x = 0 ∧ f a y = 0 ∧ x ≠ y) ∨
  (∀ x : ℝ, f a x ≠ 0) ∨
  (∃! x : ℝ, f a x = 0) := by sorry

-- Theorem 2a: Exactly 2 zeros when a > 1
theorem f_two_zeros (a : ℝ) (h : a > 1) :
  ∃! x y : ℝ, f a x = 0 ∧ f a y = 0 ∧ x ≠ y := by sorry

-- Theorem 2b: No zeros when a < 1
theorem f_no_zeros (a : ℝ) (h : a < 1) :
  ∀ x : ℝ, f a x ≠ 0 := by sorry

-- Theorem 2c: Exactly 1 zero when a = 1
theorem f_one_zero (a : ℝ) (h : a = 1) :
  ∃! x : ℝ, f a x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_x_f_zeros_f_two_zeros_f_no_zeros_f_one_zero_l364_36445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_principal_l364_36439

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem total_interest_after_trebling_principal
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Interest rate (in percentage per annum)
  (h1 : simple_interest P R 10 = 1000) -- Interest after 10 years is 1000
  (h2 : P > 0) -- Principal is positive
  (h3 : R > 0) -- Interest rate is positive
  : simple_interest P R 5 + simple_interest (3 * P) R 5 = 650 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_principal_l364_36439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_equivalence_l364_36422

theorem angle_inequality_equivalence 
  (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : θ₁ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) 
  (h₂ : θ₂ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) 
  (h₃ : θ₃ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) 
  (h₄ : θ₄ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) :
  (∃ x : ℝ, 
    (Real.cos θ₁)^2 * (Real.cos θ₂)^2 - (Real.sin θ₁ * Real.sin θ₂ - x)^2 ≥ 0 ∧
    (Real.cos θ₃)^2 * (Real.cos θ₄)^2 - (Real.sin θ₃ * Real.sin θ₄ - x)^2 ≥ 0) ↔
  (Real.sin θ₁)^2 + (Real.sin θ₂)^2 + (Real.sin θ₃)^2 + (Real.sin θ₄)^2 ≤ 
    2 * (1 + Real.sin θ₁ * Real.sin θ₂ * Real.sin θ₃ * Real.sin θ₄ + 
    Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ * Real.cos θ₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_equivalence_l364_36422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_players_in_chess_club_l364_36449

structure ChessClub where
  players : Type
  takes_lessons : players → players → Prop
  distinct_triple_condition : 
    ∀ (a b c : players), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (takes_lessons a b ∧ ¬takes_lessons b c ∧ ¬takes_lessons c a) ∨
    (¬takes_lessons a b ∧ takes_lessons b c ∧ ¬takes_lessons c a) ∨
    (¬takes_lessons a b ∧ ¬takes_lessons b c ∧ takes_lessons c a)

theorem max_players_in_chess_club (c : ChessClub) [Fintype c.players] :
  Fintype.card c.players ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_players_in_chess_club_l364_36449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l364_36456

/-- Represents the speed of a train in km/h given its length in meters and time to cross a pole in seconds -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train with length 100.008 meters that crosses a pole in 4 seconds has a speed of 90.0072 km/h -/
theorem train_speed_calculation :
  train_speed 100.008 4 = 90.0072 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l364_36456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shells_formula_l364_36442

variable (x : ℚ)

def shells_first_hour := x
def shells_second_hour := x + 32
def total_shells := shells_first_hour x + shells_second_hour x

theorem total_shells_formula : total_shells x = 2*x + 32 := by
  unfold total_shells shells_first_hour shells_second_hour
  ring

#check total_shells_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shells_formula_l364_36442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_rectangle_area_l364_36429

/-- An isosceles right triangle with legs of length 1 -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ  -- vertex A (right angle)
  B : ℝ × ℝ  -- vertex B
  C : ℝ × ℝ  -- vertex C
  isRight : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0  -- right angle at A
  isIsosceles : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2  -- AB = AC
  legLength : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1  -- length of legs is 1

/-- A rectangle inscribed in an isosceles right triangle -/
structure InscribedRectangle (t : IsoscelesRightTriangle) where
  E : ℝ × ℝ  -- vertex E
  F : ℝ × ℝ  -- vertex F
  G : ℝ × ℝ  -- vertex G
  H : ℝ × ℝ  -- vertex H
  onAB : ∃ s : ℝ, E.1 = t.A.1 + s * (t.B.1 - t.A.1) ∧ E.2 = t.A.2 + s * (t.B.2 - t.A.2)
  onAC : ∃ r : ℝ, F.1 = t.A.1 + r * (t.C.1 - t.A.1) ∧ F.2 = t.A.2 + r * (t.C.2 - t.A.2)
  onBC : ∃ u v : ℝ, G.1 = t.B.1 + u * (t.C.1 - t.B.1) ∧ G.2 = t.B.2 + u * (t.C.2 - t.B.2) ∧
                   H.1 = t.B.1 + v * (t.C.1 - t.B.1) ∧ H.2 = t.B.2 + v * (t.C.2 - t.B.2)
  isRectangle : (E.1 - F.1) * (G.1 - F.1) + (E.2 - F.2) * (G.2 - F.2) = 0 ∧
                (E.1 - H.1) * (G.1 - H.1) + (E.2 - H.2) * (G.2 - H.2) = 0

/-- The area of an inscribed rectangle -/
def rectangleArea (t : IsoscelesRightTriangle) (r : InscribedRectangle t) : ℝ :=
  abs ((r.E.1 - r.F.1) * (r.G.2 - r.F.2) - (r.E.2 - r.F.2) * (r.G.1 - r.F.1))

/-- The maximum area of a rectangle inscribed in an isosceles right triangle with legs of length 1 is 1/4 -/
theorem max_inscribed_rectangle_area (t : IsoscelesRightTriangle) :
  (⨆ r : InscribedRectangle t, rectangleArea t r) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_rectangle_area_l364_36429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_in_rice_estimate_l364_36473

/-- The problem of estimating wheat in a rice batch -/
theorem wheat_in_rice_estimate 
  (total_grain : ℕ) 
  (sample_size : ℕ) 
  (wheat_in_sample : ℕ) 
  (total_grain_positive : 0 < total_grain)
  (sample_size_positive : 0 < sample_size)
  (wheat_in_sample_le_sample : wheat_in_sample ≤ sample_size) :
  let wheat_proportion : ℚ := (wheat_in_sample : ℚ) / (sample_size : ℚ)
  let estimated_wheat : ℕ := (((total_grain : ℚ) * wheat_proportion).floor : ℤ).toNat
  total_grain = 1512 ∧ 
  sample_size = 216 ∧ 
  wheat_in_sample = 27 →
  estimated_wheat = 189 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_in_rice_estimate_l364_36473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l364_36401

theorem function_identity (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2 - 1) :
  ∀ x : ℝ, f x = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l364_36401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l364_36402

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 10

theorem f_properties (a : ℝ) :
  (∀ x, (deriv (f a)) x = 3*x^2 + a) →
  deriv (f a) 2 = 0 →
  a = -12 ∧
  (∀ x ∈ Set.Icc (-3) 4, f a x ≥ -6) ∧
  (∀ x ∈ Set.Icc (-3) 4, f a x ≤ 26) ∧
  ∃ x ∈ Set.Icc (-3) 4, f a x = -6 ∧
  ∃ x ∈ Set.Icc (-3) 4, f a x = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l364_36402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l2_general_form_l364_36443

-- Define the slopes of lines l₁ and l₂
noncomputable def slope_l1 : ℝ := 1 / 2
noncomputable def slope_l2 (θ : ℝ) : ℝ := Real.tan (2 * θ)

-- Define the y-intercept of line l₂
def y_intercept_l2 : ℝ := -3

-- Define the general form coefficients of line l₂
def A : ℝ := 4
def B : ℝ := -3
def C : ℝ := -9

-- Theorem statement
theorem line_l2_general_form (θ : ℝ) (h : Real.tan θ = slope_l1) :
  ∀ x y : ℝ, y = slope_l2 θ * x + y_intercept_l2 ↔ A * x + B * y + C = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l2_general_form_l364_36443
