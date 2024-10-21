import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_equals_one_l585_58549

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
  (∀ x y, l1 x y ↔ a * x + b * y = 0) ∧
  (∀ x y, l2 x y ↔ c * x + d * y = 0) ∧
  a / b = c / d

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + (1 + m) * y = 2 - m

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * m * x + 4 * y = -16

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel_lines (l₁ m) (l₂ m) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_equals_one_l585_58549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l585_58557

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The addition problem structure -/
structure AdditionProblem where
  T : Digit
  W : Digit
  O : Digit
  F : Digit
  U : Digit
  R : Digit
  distinct : T ≠ W ∧ T ≠ O ∧ T ≠ F ∧ T ≠ U ∧ T ≠ R ∧
             W ≠ O ∧ W ≠ F ∧ W ≠ U ∧ W ≠ R ∧
             O ≠ F ∧ O ≠ U ∧ O ≠ R ∧
             F ≠ U ∧ F ≠ R ∧
             U ≠ R
  T_is_8 : T.val = 8
  O_is_even : O.val % 2 = 0
  addition_correct : 
    (100 * F.val + 10 * O.val + U.val + R.val) = 
    (100 * T.val + 10 * W.val + O.val) + (100 * T.val + 10 * W.val + O.val)

theorem unique_solution (p : AdditionProblem) : p.W.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l585_58557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l585_58592

/-- The time taken for two workers to complete a job together, given their individual completion times -/
noncomputable def time_together (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem stating that two workers with given individual completion times will take 6 hours together -/
theorem workers_completion_time :
  let time_a : ℝ := 10
  let time_b : ℝ := 15
  time_together time_a time_b = 6 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l585_58592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_diffs_l585_58528

-- Define the differentiation operations
noncomputable def diff1 (x : ℝ) : ℝ := (x - 1/x)
noncomputable def diff1_deriv (x : ℝ) : ℝ := (1 + x^2) / x^2

noncomputable def diff2 (x : ℝ) : ℝ := Real.log (2*x - 1)
noncomputable def diff2_deriv (x : ℝ) : ℝ := 2 / (2*x - 1)

noncomputable def diff3 (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def diff3_deriv (x : ℝ) : ℝ := 2*x * Real.exp x

noncomputable def diff4 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def diff4_deriv (x : ℝ) : ℝ := 1 / (x * Real.log 2)

-- Define a function to check if a differentiation operation is correct
def is_correct_diff (f : ℝ → ℝ) (f_deriv : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = some (f_deriv x)

-- Theorem stating that exactly 3 out of 4 operations are correct
theorem exactly_three_correct_diffs :
  (is_correct_diff diff1 diff1_deriv ∧
   is_correct_diff diff2 diff2_deriv ∧
   ¬is_correct_diff diff3 diff3_deriv ∧
   is_correct_diff diff4 diff4_deriv) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_diffs_l585_58528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l585_58567

noncomputable def f (a x : ℝ) : ℝ := x^2 - (a + 1/a)*x + 1

theorem f_inequality_solution (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1 → (∀ x, f a x ≤ 0 ↔ a ≤ x ∧ x ≤ 1/a)) ∧
  (a > 1 → (∀ x, f a x ≤ 0 ↔ 1/a ≤ x ∧ x ≤ a)) ∧
  (a = 1 → (∀ x, f a x ≤ 0 ↔ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l585_58567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_travel_time_l585_58572

/-- A problem about a boy's travel time to school. -/
theorem school_travel_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (late_time : ℝ) : 
  distance = 60 ∧ 
  speed_day1 = 10 ∧ 
  speed_day2 = 20 ∧ 
  late_time = 2 → 
  distance / speed_day2 = distance / (distance / (distance / speed_day1 - late_time)) - 1 := by
  intro h
  sorry

#check school_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_travel_time_l585_58572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C2_to_l_l585_58582

/-- The curve C2 -/
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

/-- The line l -/
noncomputable def l (x y : ℝ) : ℝ := x + y - 4 * Real.sqrt 5

/-- Distance from a point to the line l -/
noncomputable def distance_to_l (p : ℝ × ℝ) : ℝ :=
  |l p.1 p.2| / Real.sqrt 2

theorem max_distance_C2_to_l :
  (⨆ θ : ℝ, distance_to_l (C2 θ)) = (5 * Real.sqrt 10) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C2_to_l_l585_58582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l585_58538

/-- Represents a configuration of unit squares in the coordinate plane -/
structure SquareConfiguration where
  num_squares : ℕ
  lower_right : ℝ × ℝ

/-- Represents a line in the coordinate plane -/
structure Line where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Checks if a line divides a square configuration into two equal areas -/
def divides_equally (config : SquareConfiguration) (line : Line) : Prop :=
  let total_area := config.num_squares
  let triangle_area := triangle_area (line.endpoint.1 - line.start.1) line.endpoint.2
  2 * triangle_area = total_area

/-- The main theorem to be proved -/
theorem equal_area_division (config : SquareConfiguration) (b : ℝ) :
  config.num_squares = 7 ∧ 
  config.lower_right = (0, 0) ∧
  divides_equally config { start := (b, 0), endpoint := (5, 5) } ↔ 
  b = 18 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l585_58538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_function_range_l585_58540

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * (1/2)^x + (1/4)^x

/-- Theorem stating the range of a for which f is bounded on [-2, 1] with upper bound 3 -/
theorem bounded_function_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 1, |f a x| ≤ 3) →
  a ∈ Set.Icc (-4) (-7/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_function_range_l585_58540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_worth_is_correct_l585_58570

/-- The total worth of John's presents to his fiancee -/
noncomputable def total_worth : ℝ :=
  let ring_cost : ℝ := 4000
  let car_cost : ℝ := 2000
  let bracelet_cost : ℝ := 2 * ring_cost
  let gown_cost : ℝ := bracelet_cost / 2
  let jewelry_cost : ℝ := 1.2 * ring_cost
  let painting_cost_eur : ℝ := 3000
  let usd_to_eur : ℝ := 1.2
  let painting_cost_usd : ℝ := painting_cost_eur * usd_to_eur
  let honeymoon_cost_jpy : ℝ := 180000
  let usd_to_jpy : ℝ := 110
  let honeymoon_cost_usd : ℝ := honeymoon_cost_jpy / usd_to_jpy
  let watch_cost : ℝ := 5500
  ring_cost + car_cost + bracelet_cost + gown_cost + jewelry_cost + 
  painting_cost_usd + honeymoon_cost_usd + watch_cost

theorem total_worth_is_correct : 
  total_worth = 33536.36 := by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_worth_is_correct_l585_58570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_representation_l585_58512

def S : Set ℝ := {x | x > 0 ∧ x ≠ 2}

theorem set_representation : S = Set.Ioo 0 2 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_representation_l585_58512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_garden_area_l585_58509

/-- The optimal area of a rectangular garden with a perimeter of 200 feet,
    length at least 60 feet, and width at least 30 feet. -/
theorem optimal_garden_area :
  let perimeter := 200
  let min_length := 60
  let min_width := 30
  let area (l w : ℝ) := l * w
  let constraint (l w : ℝ) := 2 * l + 2 * w = perimeter ∧ l ≥ min_length ∧ w ≥ min_width
  ∃ l w : ℝ, constraint l w ∧
    ∀ l' w' : ℝ, constraint l' w' → area l w ≥ area l' w' :=
by
  sorry

#check optimal_garden_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_garden_area_l585_58509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l585_58586

theorem cos_sin_equation (x : ℝ) : 
  (Real.cos x - 4 * Real.sin x = 1) → 
  (Real.sin x + 4 * Real.cos x = 4 ∨ Real.sin x + 4 * Real.cos x = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l585_58586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_rectangles_area_l585_58554

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem overlapping_rectangles_area
  (pquv wstv : Rectangle)
  (h_pquv_width : pquv.width = 2)
  (h_pquv_height : pquv.height = 7)
  (h_wstv_width : wstv.width = 5)
  (h_wstv_height : wstv.height = 5)
  (h_overlap_height : ℝ) 
  (h_overlap_height_val : h_overlap_height = 2) :
  pquv.area + wstv.area - (h_overlap_height * wstv.width) = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_rectangles_area_l585_58554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l585_58564

/-- The perimeter of a hexagon with given side lengths is 58 inches. -/
theorem hexagon_perimeter 
  (side1 side2 side3 side4 side5 side6 : ℝ)
  (h1 : side1 = 7)
  (h2 : side2 = 10)
  (h3 : side3 = 8)
  (h4 : side4 = 13)
  (h5 : side5 = 11)
  (h6 : side6 = 9) :
  side1 + side2 + side3 + side4 + side5 + side6 = 58 := by
  rw [h1, h2, h3, h4, h5, h6]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l585_58564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_with_divisor_property_l585_58597

theorem unique_n_with_divisor_property : ∃! (n : ℕ),
  (∃ (k : ℕ) (d : Fin (k + 1) → ℕ),
    (∀ i : Fin (k + 1), d i ∣ n) ∧
    (∀ i j : Fin (k + 1), i < j → d i < d j) ∧
    (d 0 = 1) ∧
    (d k = n) ∧
    (∀ m : ℕ, m ∣ n → ∃ i : Fin (k + 1), d i = m) ∧
    (2 * n = (d 4)^2 + (d 5)^2 - 1)) ∧
  n = 272 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_with_divisor_property_l585_58597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_is_45_l585_58519

/-- The degree of each exterior angle of a regular octagon -/
def exterior_angle_regular_octagon : ℚ := 45

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The sum of exterior angles of any polygon in degrees -/
def sum_exterior_angles : ℚ := 360

theorem exterior_angle_regular_octagon_is_45 :
  exterior_angle_regular_octagon = sum_exterior_angles / octagon_sides :=
by
  -- Convert the goal to an equality of rational numbers
  rw [exterior_angle_regular_octagon, sum_exterior_angles, octagon_sides]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_is_45_l585_58519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_problem_l585_58550

theorem triangle_angle_problem (a b c A B C : ℝ) : 
  a = 1 →
  b = Real.sqrt 3 →
  A + C = 2 * B →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sin A / a = Real.sin B / b →
  Real.sin B / b = Real.sin C / c →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_problem_l585_58550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_analysis_l585_58552

/-- Motion of a point along the x-axis -/
def x (t : ℝ) : ℝ := 2 * t^3 - 4 * t^2 + 2 * t + 3

/-- Velocity function -/
def v (t : ℝ) : ℝ := 6 * t^2 - 8 * t + 2

/-- Acceleration function -/
def a (t : ℝ) : ℝ := 12 * t - 8

theorem motion_analysis :
  (∀ t, v t = deriv x t) ∧
  (∀ t, a t = deriv v t) ∧
  (v 0 = 2) ∧
  (v 3 = 32) ∧
  (v (1/3) = 0 ∧ v 1 = 0) ∧
  (x (1/3) = 3 + 8/27 ∧ x 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_analysis_l585_58552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_significant_digits_l585_58589

-- Define the original area of the square
def original_area : ℝ := 1.4455

-- Define a function to calculate the side length of a square given its area
noncomputable def side_length (area : ℝ) : ℝ := Real.sqrt area

-- Define a function to count significant digits in a real number
-- This is a placeholder function and needs to be implemented
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_length_significant_digits :
  count_significant_digits (side_length original_area) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_significant_digits_l585_58589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_subjects_intersection_l585_58574

theorem student_subjects_intersection (total : ℤ) 
  (math_min math_max phys_min phys_max : ℤ) : 
  total = 2500 →
  math_min = 1750 →
  math_max = 1875 →
  phys_min = 875 →
  phys_max = 1125 →
  let m := math_max + phys_max - total
  let M := math_min + phys_min - total
  M - m = -375 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_subjects_intersection_l585_58574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_ellipse_or_line_segment_l585_58501

/-- Definition of an ellipse in 2D space -/
def Ellipse (c : ℝ × ℝ) (r₁ r₂ : ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), p = (c.1 + r₁ * Real.cos θ, c.2 + r₂ * Real.sin θ)

/-- The trajectory of a point P satisfying the given condition is either an ellipse or a line segment -/
theorem trajectory_ellipse_or_line_segment 
  (F₁ F₂ P : ℝ × ℝ) 
  (h_F₁ : F₁ = (2, 0)) 
  (h_F₂ : F₂ = (-2, 0)) 
  (a : ℝ) 
  (h_a : a > 0) 
  (h_condition : dist P F₁ + dist P F₂ = 4*a + 1/a) :
  (∃ (c : ℝ × ℝ) (r₁ r₂ : ℝ), 
    Ellipse c r₁ r₂ P ∨ 
    (P.1 ≥ -2 ∧ P.1 ≤ 2 ∧ P.2 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_ellipse_or_line_segment_l585_58501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_constant_coefficient_polynomial_l585_58575

theorem no_constant_coefficient_polynomial :
  ¬ ∃ (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0),
  ∀ (n : ℕ) (h_n : n > 3), ∃ (P : Polynomial ℝ),
    (Polynomial.degree P = n) ∧
    (Polynomial.coeff P 2 = a) ∧
    (Polynomial.coeff P 1 = b) ∧
    (Polynomial.coeff P 0 = c) ∧
    (∃ (roots : Finset ℤ), roots.card = n ∧ ∀ (x : ℤ), x ∈ roots ↔ Polynomial.eval (↑x : ℝ) P = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_constant_coefficient_polynomial_l585_58575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l585_58508

/-- A set of points where every point is the midpoint of a segment with endpoints in the set is infinite. -/
theorem midpoint_set_infinite (S : Set ℝ) 
  (h : ∀ x ∈ S, ∃ a b, a ∈ S ∧ b ∈ S ∧ x = (a + b) / 2) : 
  Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l585_58508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_total_bill_l585_58530

-- Define constants for item prices
def shirt_price : ℚ := 12
def socks_price : ℚ := 5
def shorts_price : ℚ := 15
def swim_trunks_price : ℚ := 14

-- Define quantities
def num_shirts : ℕ := 3
def num_socks : ℕ := 1
def num_shorts : ℕ := 2
def num_swim_trunks : ℕ := 1

-- Define shipping rules
def flat_rate_shipping : ℚ := 5
def shipping_threshold : ℚ := 50
def shipping_rate : ℚ := 1/5

-- Calculate total purchase price
def total_purchase_price : ℚ :=
  shirt_price * num_shirts +
  socks_price * num_socks +
  shorts_price * num_shorts +
  swim_trunks_price * num_swim_trunks

-- Calculate shipping cost
noncomputable def shipping_cost : ℚ :=
  if total_purchase_price ≤ shipping_threshold then
    flat_rate_shipping
  else
    total_purchase_price * shipping_rate

-- Calculate total bill
noncomputable def total_bill : ℚ := total_purchase_price + shipping_cost

-- Theorem to prove
theorem thomas_total_bill : total_bill = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_total_bill_l585_58530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l585_58558

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the conditions
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 * v2.1 + v1.2 * v2.2 = 0)

noncomputable def length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem quadrilateral_perimeter (EFGH : Quadrilateral) :
  is_perpendicular (EFGH.F - EFGH.E) (EFGH.G - EFGH.F) →
  is_perpendicular (EFGH.H - EFGH.G) (EFGH.G - EFGH.F) →
  length (EFGH.F - EFGH.E) = 15 →
  length (EFGH.H - EFGH.G) = 5 →
  length (EFGH.G - EFGH.F) = 8 →
  length (EFGH.F - EFGH.E) + length (EFGH.G - EFGH.F) +
  length (EFGH.H - EFGH.G) + length (EFGH.E - EFGH.H) = 28 + 2 * Real.sqrt 41 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l585_58558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_AE_BE_l585_58595

-- Define the triangle ABC and points D, E, O
variable (A B C D E O : EuclideanSpace ℝ (Fin 2))

-- Define the areas of the triangles
variable (area_OBE area_OBC area_OCD : ℝ)

-- State the theorem
theorem ratio_AE_BE :
  D ∈ segment A C →
  E ∈ segment A B →
  O ∈ segment B D →
  O ∈ segment C E →
  area_OBE = 15 →
  area_OBC = 30 →
  area_OCD = 24 →
  (dist A E) / (dist B E) = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_AE_BE_l585_58595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_convex_quadrilateral_with_equal_diagonals_l585_58571

/-- A convex quadrilateral with equal diagonals -/
structure ConvexQuadrilateral where
  /-- Length of each diagonal -/
  e : ℝ
  /-- Sum of the two midlines -/
  s : ℝ
  /-- Length of one side -/
  a : ℝ
  /-- Length of an adjacent side -/
  b : ℝ
  /-- The quadrilateral is convex -/
  convex : 0 < e ∧ 0 < s ∧ 0 < a ∧ 0 < b
  /-- Condition for real diagonals -/
  diagonal_condition : s / Real.sqrt 2 ≤ e ∧ e < s

/-- The area of a convex quadrilateral with equal diagonals -/
noncomputable def area (q : ConvexQuadrilateral) : ℝ := (q.s^2 - q.e^2) / 4

/-- Theorem stating the area of the quadrilateral -/
theorem area_of_convex_quadrilateral_with_equal_diagonals (q : ConvexQuadrilateral) :
  area q = (q.s^2 - q.e^2) / 4 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_convex_quadrilateral_with_equal_diagonals_l585_58571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_area_l585_58523

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a trapezoid with given bases and height -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (t.base1 + t.base2) * t.height / 2

/-- Represents the folding operation that transforms a rectangle into a trapezoid -/
noncomputable def foldRectangleToTrapezoid (r : Rectangle) : Trapezoid :=
  { base1 := r.width
    base2 := r.width - 2 * 3  -- Assuming 3-4-5 triangle formation
    height := r.height }

/-- The main theorem stating that folding a 5x8 rectangle results in a trapezoid with area 55/2 -/
theorem folded_rectangle_area (r : Rectangle) 
    (h_width : r.width = 8) 
    (h_height : r.height = 5) : 
    trapezoidArea (foldRectangleToTrapezoid r) = 55 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_rectangle_area_l585_58523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_needed_to_fill_tanks_l585_58514

/-- Given a truck and a car with specified tank capacities and current fill levels,
    calculate the total amount of gas needed to fill both tanks completely. -/
theorem gas_needed_to_fill_tanks (truck_capacity car_capacity : ℚ) 
    (truck_fill_ratio car_fill_ratio : ℚ) 
    (h1 : truck_capacity = 20)
    (h2 : car_capacity = 12)
    (h3 : truck_fill_ratio = 1/2)
    (h4 : car_fill_ratio = 1/3)
    : truck_capacity * (1 - truck_fill_ratio) + car_capacity * (1 - car_fill_ratio) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_needed_to_fill_tanks_l585_58514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_l585_58551

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | n + 4 => (List.range n).foldl (λ acc i => acc * a (i + 1)) 1 - 1

def b : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 9
  | n + 4 => b (n + 3) + 3 * (n + 4) - 1

def product_a (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * a (i + 1)) 1

def sum_b (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc + b (i + 1)) 0

theorem sequence_difference (n : ℕ) : 
  product_a n - sum_b n = (List.range n).foldl (λ acc i => acc * a (i + 1)) 1 - (List.range n).foldl (λ acc i => acc + b (i + 1)) 0 := by
  sorry

#eval product_a 100 - sum_b 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_l585_58551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cup_orientation_l585_58579

-- Define the possible orientations of the cup
inductive CupOrientation
  | Up
  | Down

-- Define the initial state of the cup
def initial_state : CupOrientation := CupOrientation.Up

-- Define the function that determines the orientation after n flips
def orientation_after_flips (n : ℕ) : CupOrientation :=
  if n % 2 = 0 then initial_state else
    match initial_state with
    | CupOrientation.Up => CupOrientation.Down
    | CupOrientation.Down => CupOrientation.Up

-- Theorem to prove
theorem cup_orientation (n : ℕ) :
  (orientation_after_flips n = CupOrientation.Down ↔ n % 2 = 1) ∧
  (orientation_after_flips n = CupOrientation.Up ↔ n % 2 = 0) := by
  sorry

-- Examples for the specific cases mentioned in the problem
example : orientation_after_flips 19 = CupOrientation.Down := by
  simp [orientation_after_flips]
  rfl

example : orientation_after_flips 2008 = CupOrientation.Up := by
  simp [orientation_after_flips]
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cup_orientation_l585_58579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_slope_l585_58503

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The x-coordinate of the right focus of a hyperbola -/
noncomputable def right_focus_x (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: If the slope of the line connecting the right vertex to a point on the hyperbola
    whose projection on the x-axis coincides with the right focus is 3,
    then the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity_from_slope (h : Hyperbola)
  (slope_condition : (h.b^2 / h.a) / (right_focus_x h - h.a) = 3) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_slope_l585_58503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l585_58596

theorem congruence_solutions_count :
  let y_solutions := {y : ℕ | y > 0 ∧ y < 150 ∧ (y + 21) % 46 = 79 % 46}
  Finset.card (Finset.filter (λ y => y > 0 ∧ y < 150 ∧ (y + 21) % 46 = 79 % 46) (Finset.range 150)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l585_58596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l585_58520

theorem max_product_sum (x y z w : ℕ) :
  x ∈ ({7, 8, 9, 10} : Set ℕ) →
  y ∈ ({7, 8, 9, 10} : Set ℕ) →
  z ∈ ({7, 8, 9, 10} : Set ℕ) →
  w ∈ ({7, 8, 9, 10} : Set ℕ) →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x < y ∧ y < z ∧ z < w →
  (x * y + y * z + z * w + x * w) ≤ 288 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l585_58520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l585_58555

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = Real.pi / 6 ∧ b = 2 * Real.pi / 3 ∧
  (∀ x y, a < x ∧ x < y ∧ y < b → f y < f x) ∧
  (∀ ε > 0, ∃ x y, a - ε < x ∧ x < a ∧ b < y ∧ y < b + ε ∧ f x ≤ f y) :=
by
  sorry

#check monotonic_decreasing_interval_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l585_58555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l585_58562

noncomputable def present_value : ℝ := 51200
noncomputable def annual_increase_rate : ℝ := 1/8
def years : ℕ := 2

theorem amount_after_two_years :
  present_value * (1 + annual_increase_rate)^years = 64800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l585_58562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_new_matching_l585_58588

structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 40
  even_degree : ∀ v ∈ vertices, Even ((edges.filter (λ e => e.1 = v ∨ e.2 = v)).card)
  initial_matching : ∃ m : Finset (Nat × Nat), m.card = 20 ∧ (∀ e ∈ m, e ∈ edges) ∧
    (∀ v ∈ vertices, (m.filter (λ e => e.1 = v ∨ e.2 = v)).card = 1)

theorem no_new_matching (G : FriendshipGraph) :
  ∃ e ∈ G.edges, ¬∃ m : Finset (Nat × Nat),
    m.card = 20 ∧
    (∀ e' ∈ m, e' ∈ G.edges ∧ e' ≠ e) ∧
    (∀ v ∈ G.vertices, (m.filter (λ e' => e'.1 = v ∨ e'.2 = v)).card = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_new_matching_l585_58588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_expr_is_b_l585_58505

noncomputable def expr_a : ℝ := 6 + 5 + 4
noncomputable def expr_b : ℝ := 6 + 5 - 4
noncomputable def expr_c : ℝ := 6 + 5 * 4
noncomputable def expr_d : ℝ := 6 - 5 * 4
noncomputable def expr_e : ℝ := 6 * 5 / 4

def closest_to_zero (x y : ℝ) : Prop := abs x ≤ abs y

theorem closest_expr_is_b :
  closest_to_zero expr_b expr_a ∧
  closest_to_zero expr_b expr_c ∧
  closest_to_zero expr_b expr_d ∧
  closest_to_zero expr_b expr_e :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_expr_is_b_l585_58505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_sister_weight_ratio_l585_58506

/-- Proves that the ratio of a student's weight after losing 5 kg to his sister's weight is 2:1 --/
theorem student_sister_weight_ratio :
  ∀ (total_weight student_weight : ℝ),
  total_weight = 104 →
  student_weight = 71 →
  let sister_weight := total_weight - student_weight
  let student_new_weight := student_weight - 5
  (∃ (k : ℝ), student_new_weight = k * sister_weight) →
  (student_new_weight / sister_weight = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_sister_weight_ratio_l585_58506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l585_58565

-- Define the line C₁
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the circle C₂ in polar form
noncomputable def C₂ (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

-- State the theorem
theorem line_circle_intersection :
  -- General equation of C₁
  (∀ (x y : ℝ), (∃ t : ℝ, C₁ t = (x, y)) ↔ x - y + 1 = 0) ∧
  -- Polar coordinates of C₂'s center
  (∃ r θ : ℝ, r = 2 ∧ θ = 2 * Real.pi / 3 ∧
    r * Real.cos θ = -1 ∧ r * Real.sin θ = Real.sqrt 3) ∧
  -- Length of chord AB
  (∃ A B : ℝ × ℝ,
    (∃ t : ℝ, C₁ t = A) ∧
    (∃ θ : ℝ, C₂ θ * Real.cos θ = A.1 ∧ C₂ θ * Real.sin θ = A.2) ∧
    (∃ t : ℝ, C₁ t = B) ∧
    (∃ θ : ℝ, C₂ θ * Real.cos θ = B.1 ∧ C₂ θ * Real.sin θ = B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l585_58565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_is_seven_twelfths_l585_58531

structure Spinner where
  probabilities : List ℚ
  len_eq_five : probabilities.length = 5
  second_double_first : probabilities[1]! = 2 * probabilities[0]!
  third_is_quarter : probabilities[2]! = 1/4
  fifth_is_quarter : probabilities[4]! = 1/4
  sum_is_one : probabilities.sum = 1

theorem probability_of_odd_is_seven_twelfths (s : Spinner) :
  s.probabilities[0]! + s.probabilities[2]! + s.probabilities[4]! = 7/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_is_seven_twelfths_l585_58531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_ellipse_not_circle_not_parabola_not_hyperbola_l585_58590

/-- Definition of the distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Definition of the equation given in the problem -/
def equation (x y : ℝ) : Prop :=
  distance x y 2 (-2) + distance x y (-3) 4 = 14

/-- Theorem stating that the equation describes an ellipse -/
theorem equation_describes_ellipse : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

/-- Theorem stating that the equation does not describe a circle -/
theorem not_circle : ¬∃ (r : ℝ) (h k : ℝ), ∀ (x y : ℝ),
  equation x y ↔ ((x - h)^2 + (y - k)^2 = r^2) := by
  sorry

/-- Theorem stating that the equation does not describe a parabola -/
theorem not_parabola : ¬∃ (a p : ℝ) (h k : ℝ), ∀ (x y : ℝ),
  equation x y ↔ ((y - k)^2 = 4*a*(x - h)) := by
  sorry

/-- Theorem stating that the equation does not describe a hyperbola -/
theorem not_hyperbola : ¬∃ (a b : ℝ) (h k : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), equation x y ↔ ((x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_ellipse_not_circle_not_parabola_not_hyperbola_l585_58590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_of_arithmetic_sequence_l585_58593

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of the first n terms of an arithmetic sequence -/
def partial_sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem eighth_term_of_arithmetic_sequence 
  (seq : ArithmeticSequence) 
  (sum_15_eq_90 : partial_sum seq 15 = 90) :
  seq.a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_of_arithmetic_sequence_l585_58593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_needed_l585_58525

/-- Represents a query that can reveal the relative order of 50 numbers -/
def Query := Fin 100 → Fin 50

/-- Represents a permutation of the numbers 1 to 100 -/
def Permutation := Fin 100 ↪ Fin 100

/-- The result of applying a query to a permutation -/
def QueryResult (q : Query) (p : Permutation) : Fin 50 ↪ Fin 50 := sorry

/-- A set of queries is sufficient if it can uniquely determine any permutation -/
def IsSufficient (qs : Finset Query) : Prop :=
  ∀ p₁ p₂ : Permutation, (∀ q ∈ qs, QueryResult q p₁ = QueryResult q p₂) → p₁ = p₂

/-- The main theorem: The minimum number of queries needed is 5 -/
theorem min_queries_needed :
  (∃ qs : Finset Query, IsSufficient qs ∧ qs.card = 5) ∧
  (∀ qs : Finset Query, IsSufficient qs → qs.card ≥ 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_needed_l585_58525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_plus_one_l585_58581

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

-- State the theorem
theorem domain_of_log_plus_one :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_plus_one_l585_58581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_side_relation_l585_58532

theorem triangle_tangent_side_relation (α β γ a b c p q r : ℝ) :
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a > 0 → b > 0 → c > 0 →
  p > 0 → q > 0 → r > 0 →
  (Real.tan α) / (Real.tan β) = p / q →
  (Real.tan β) / (Real.tan γ) = q / r →
  (Real.tan γ) / (Real.tan α) = r / p →
  (a^2 : ℝ) / (b^2 : ℝ) = (1/q + 1/r) / (1/r + 1/p) ∧
  (b^2 : ℝ) / (c^2 : ℝ) = (1/r + 1/p) / (1/p + 1/q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_side_relation_l585_58532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l585_58594

theorem triangle_perimeter : ∀ x : ℝ,
  x > 0 ∧
  x * (x - 9) - 13 * (x - 9) = 0 ∧
  x + 3 > 8 ∧
  x + 8 > 3 ∧
  3 + 8 > x →
  x + 3 + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l585_58594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_negative_angle_l585_58537

theorem cosine_negative_angle (θ : ℝ) : Real.cos (-θ) = Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_negative_angle_l585_58537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l585_58510

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) + Real.sin x * Real.sin x

theorem f_range :
  Set.range f = Set.Icc ((1:ℝ)/2 - Real.sqrt 2 / 2) ((1:ℝ)/2 + Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l585_58510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_min_distance_min_max_distance_l585_58502

-- Manhattan distance definition
def manhattan_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + |y₁ - y₂|

-- Part 1
theorem range_of_x (x : ℝ) :
  (∀ y, y = 1 - x → manhattan_distance x y 0 0 ≤ 1) → x ∈ Set.Icc 0 1 := by sorry

-- Part 2
theorem min_distance :
  ∃ min_dist : ℝ, min_dist = 1/2 ∧ 
    ∀ x₁ x₂ : ℝ, manhattan_distance x₁ (2 * x₁ - 2) x₂ (x₂^2) ≥ min_dist := by sorry

-- Part 3
theorem min_max_distance :
  ∃ (min_max_dist : ℝ) (a b : ℝ),
    min_max_dist = 25/8 ∧
    (a, b) = (0, 23/8) ∧
    (∀ x ∈ Set.Icc (-2) 2, manhattan_distance a b x (x^2) ≤ min_max_dist) ∧
    (∀ a' b', ∃ x ∈ Set.Icc (-2) 2, manhattan_distance a' b' x (x^2) ≥ min_max_dist) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_min_distance_min_max_distance_l585_58502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_distance_sum_equal_l585_58573

/-- Represents a square on the chessboard -/
structure Square where
  x : Fin 8
  y : Fin 8
deriving Fintype

/-- The color of a square based on its coordinates -/
def squareColor (s : Square) : Bool :=
  (s.x.val + s.y.val) % 2 == 0

/-- The squared distance between two squares -/
def squaredDistance (s1 s2 : Square) : ℕ :=
  (s1.x.val - s2.x.val)^2 + (s1.y.val - s2.y.val)^2

/-- The sum of squared distances from a given square to all squares of a specific color -/
def sumSquaredDistances (start : Square) (color : Bool) : ℕ :=
  Finset.sum (Finset.filter (fun s => squareColor s = color) Finset.univ) (fun s => squaredDistance start s)

theorem chessboard_distance_sum_equal (start : Square) :
  sumSquaredDistances start true = sumSquaredDistances start false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_distance_sum_equal_l585_58573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l585_58526

theorem product_of_numbers_with_ratio (a b : ℝ) :
  (a - b) / (a + b) = 1 / 8 ∧ (a - b) / (a * b) = 1 / 40 →
  a * b = 6400 / 63 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l585_58526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l585_58534

def A : Set ℝ := {1, 3, 5, 7}
def B : Set ℝ := {x | x^2 - 2*x - 5 ≤ 0}

theorem A_intersect_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l585_58534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_triangle_area_l585_58578

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

/-- Two perpendicular lines in a 2D plane -/
structure PerpendicularLines where
  line1 : Line2D
  line2 : Line2D
  perpendicular : line1.slope * line2.slope = -1

/-- The y-intercept of a line -/
def yIntercept (l : Line2D) : Point2D :=
  { x := 0, y := l.yIntercept }

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point2D) : ℝ :=
  sorry

theorem perpendicular_lines_triangle_area :
  ∀ (lines : PerpendicularLines),
    let A : Point2D := { x := 4, y := 5 }
    let P := yIntercept lines.line1
    let Q := yIntercept lines.line2
    P.y + Q.y = 4 →
    lines.line1.slope * 4 + lines.line1.yIntercept = 5 →
    lines.line2.slope * 4 + lines.line2.yIntercept = 5 →
    triangleArea A P Q = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_triangle_area_l585_58578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_f_l585_58524

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := 
  Real.sin (k * x) * (Real.sin x) ^ k + Real.cos (k * x) * (Real.cos x) ^ k - (Real.cos (2 * x)) ^ k

theorem exists_constant_f : ∃ k : ℕ, ∀ x : ℝ, f k x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_f_l585_58524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_product_is_half_l585_58569

/-- The probability that the product of two independently and uniformly
    selected real numbers from the interval [-15, 15] is greater than zero -/
noncomputable def probability_positive_product : ℝ := 1/2

/-- The lower bound of the interval -/
def lower_bound : ℝ := -15

/-- The upper bound of the interval -/
def upper_bound : ℝ := 15

/-- Theorem stating that the probability of a positive product is 1/2 -/
theorem probability_positive_product_is_half :
  probability_positive_product = 1/2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_positive_product_is_half_l585_58569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_altitudes_right_triangle_l585_58561

-- Define a triangle in a coordinate system
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the extended point
def extendedPoint (P Q R : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  (P.1 + k * (R.2 - Q.2), P.2 + k * (Q.1 - R.1))

-- Define the condition for a right triangle
def isRightTriangle (P Q R : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

theorem extended_altitudes_right_triangle (ABC : Triangle) :
  (∃ k : ℝ, k > 0 ∧
    let A' := extendedPoint ABC.A ABC.B ABC.C k
    let B' := extendedPoint ABC.B ABC.C ABC.A k
    let C' := extendedPoint ABC.C ABC.A ABC.B k
    isRightTriangle A' B' C') →
  (∀ k : ℝ, k > 0 → 
    let A' := extendedPoint ABC.A ABC.B ABC.C k
    let B' := extendedPoint ABC.B ABC.C ABC.A k
    let C' := extendedPoint ABC.C ABC.A ABC.B k
    isRightTriangle A' B' C') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_altitudes_right_triangle_l585_58561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compounding_difference_approx_l585_58539

noncomputable def loan_amount : ℝ := 8000
noncomputable def annual_rate : ℝ := 0.08
def years : ℕ := 3

noncomputable def monthly_compound (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 12) ^ (time * 12)

noncomputable def semi_annual_compound (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 2) ^ (time * 2)

theorem compounding_difference_approx :
  ∃ ε > 0, abs ((monthly_compound loan_amount annual_rate years -
   semi_annual_compound loan_amount annual_rate years) - 23.36) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compounding_difference_approx_l585_58539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_theorem_solution_set_theorem_l585_58541

def is_valid_solution (cake piece : Nat) (n : Nat) : Prop :=
  cake = n * piece ∧
  cake ≥ 10000 ∧ cake < 100000 ∧
  piece ≥ 10000 ∧ piece < 100000 ∧
  ∀ d, d ∈ (Nat.digits 10 cake ∪ Nat.digits 10 piece) → 
    ((Nat.digits 10 cake ∪ Nat.digits 10 piece).count d = 1)

def max_pieces : Nat := 7

theorem max_pieces_theorem :
  ∃ (cake piece : Nat),
    is_valid_solution cake piece max_pieces ∧
    ∀ (n : Nat) (c p : Nat), n > max_pieces → ¬(is_valid_solution c p n) :=
by sorry

def solution_set : List (Nat × Nat) :=
  [(84357, 12051), (86457, 12351), (95207, 13601), (98357, 14051)]

theorem solution_set_theorem :
  ∀ (cake piece : Nat),
    is_valid_solution cake piece max_pieces ↔ (cake, piece) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_theorem_solution_set_theorem_l585_58541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_squared_div_area_bound_perimeter_squared_div_area_max_achievable_l585_58544

/-- A right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  right_angle : a^2 + b^2 = c^2
  coprime : Nat.Coprime a (Nat.gcd b c)

/-- The perimeter of a right triangle -/
def perimeter (t : RightTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The area of a right triangle -/
def area (t : RightTriangle) : ℕ :=
  t.a * t.b / 2

/-- The main theorem: P^2/A ≤ 24 for all right triangles with integer side lengths -/
theorem perimeter_squared_div_area_bound (t : RightTriangle) :
    (perimeter t)^2 / (area t) ≤ 24 := by
  sorry

/-- The maximum value of P^2/A is achievable -/
theorem perimeter_squared_div_area_max_achievable :
    ∃ t : RightTriangle, (perimeter t)^2 / (area t) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_squared_div_area_bound_perimeter_squared_div_area_max_achievable_l585_58544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_shop_solution_l585_58568

noncomputable def candy_shop_problem (total_sours : ℕ) 
  (cherry_lemon_ratio : ℚ) (orange_percentage : ℚ) : ℕ :=
  let cherry_sours := (1 - orange_percentage) * (total_sours : ℝ) * 
    (cherry_lemon_ratio / (1 + cherry_lemon_ratio))
  (Int.floor cherry_sours).toNat

theorem candy_shop_solution : 
  candy_shop_problem 96 (4/5) (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_shop_solution_l585_58568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l585_58517

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
    (X : Polynomial ℤ)^n + a • (X : Polynomial ℤ)^(n-1) + (p * q : ℤ) = g * h) ↔
  (Even n ∧ a = 1 + (p * q : ℤ)) ∨ (Odd n ∧ a = -1 - (p * q : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l585_58517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C2_to_line_l585_58559

/-- Curve C1 in polar coordinates -/
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (2 / (3 + Real.cos (2 * θ)))
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Curve C2 obtained from C1 by stretching x-coordinates and shortening y-coordinates -/
noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  let (x, y) := C1 θ
  (2 * x, y / 2)

/-- Distance from a point to the line x + y - 5 = 0 -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  abs (x + y - 5) / Real.sqrt 2

/-- The maximum distance from any point on C2 to the line x + y - 5 = 0 is 13√2 / 4 -/
theorem max_distance_C2_to_line :
  ∃ (θ : ℝ), ∀ (φ : ℝ), distanceToLine (C2 θ) ≥ distanceToLine (C2 φ) ∧
  distanceToLine (C2 θ) = 13 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C2_to_line_l585_58559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l585_58527

theorem smallest_sum_of_a_and_b : ∃ (a b : ℕ), 
  (3^8 * 5^2 : ℕ) = a^b ∧ 
  (∀ (c d : ℕ), (3^8 * 5^2 : ℕ) = c^d → a + b ≤ c + d) ∧
  a + b = 407 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l585_58527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l585_58536

-- Define the vector a
def a : ℝ × ℝ := (12, -5)

-- Define the properties of vector b
def b_opposite_direction (b : ℝ × ℝ) : Prop :=
  ∃ (lambda : ℝ), lambda < 0 ∧ b = (lambda * a.1, lambda * a.2)

-- Define the magnitude of vector b
def b_magnitude (b : ℝ × ℝ) : Prop :=
  b.1^2 + b.2^2 = 13^2

-- Theorem statement
theorem lambda_value :
  ∀ b : ℝ × ℝ,
  b_opposite_direction b →
  b_magnitude b →
  ∃ (lambda : ℝ), lambda = -1 ∧ b = (lambda * a.1, lambda * a.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l585_58536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l585_58513

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | (n + 1) => (1/16) * (1 + 4 * a n + Real.sqrt (1 + 24 * a n))

theorem a_closed_form (n : ℕ) (h : n ≥ 1) : 
  a n = (1/24) * (2^(2*n - 1) + 3 * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l585_58513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l585_58547

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3 / 4) : 
  Real.sqrt (1 + (b / a)^2) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l585_58547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seniority_ranking_l585_58583

-- Define the colleagues
inductive Colleague
| Dan
| Emma
| Fred

-- Define the seniority relation
def more_senior : Colleague → Colleague → Prop := sorry

-- Define the statements
def statement_I : Prop := ¬(more_senior Colleague.Dan Colleague.Emma ∧ more_senior Colleague.Dan Colleague.Fred)
def statement_II : Prop := more_senior Colleague.Emma Colleague.Dan ∧ more_senior Colleague.Emma Colleague.Fred
def statement_III : Prop := ¬(more_senior Colleague.Fred Colleague.Dan ∧ more_senior Colleague.Fred Colleague.Emma)

-- Theorem to prove
theorem seniority_ranking :
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III) →
  more_senior Colleague.Fred Colleague.Dan ∧ more_senior Colleague.Dan Colleague.Emma :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seniority_ranking_l585_58583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_carton_per_case_l585_58587

/-- The number of paper clips in a box -/
def clips_per_box : ℕ := 300

/-- The total number of paper clips in two cases -/
def total_clips : ℕ := 600

/-- The number of cases -/
def num_cases : ℕ := 2

/-- The number of boxes in a carton -/
def boxes_per_carton : ℕ := 1

/-- The number of cartons in a case -/
def cartons_per_case : ℕ := 1

theorem one_carton_per_case :
  cartons_per_case = 1 :=
by
  -- We'll use the given information to prove that there's one carton per case
  have h1 : total_clips = num_cases * cartons_per_case * boxes_per_carton * clips_per_box :=
    by sorry -- This step would require algebraic manipulation
  
  -- Substitute the known values
  have h2 : 600 = 2 * cartons_per_case * 1 * 300 :=
    by sorry -- This step would substitute the defined values
  
  -- Simplify the equation
  have h3 : cartons_per_case = 1 :=
    by sorry -- This step would solve the equation for cartons_per_case
  
  -- Conclude the proof
  exact h3

#check one_carton_per_case

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_carton_per_case_l585_58587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_avoiding_circle_l585_58591

/-- The shortest path from (0,0) to (16,12) avoiding a circle -/
theorem shortest_path_avoiding_circle : 
  ∃ (path : ℝ → ℝ × ℝ),
    (path 0 = (0, 0)) ∧ 
    (path 1 = (16, 12)) ∧ 
    (∀ t ∈ Set.Icc 0 1, ((path t).1 - 8)^2 + ((path t).2 - 6)^2 ≥ 36) ∧
    (∀ other_path : ℝ → ℝ × ℝ, 
      (other_path 0 = (0, 0)) ∧ 
      (other_path 1 = (16, 12)) ∧ 
      (∀ t ∈ Set.Icc 0 1, ((other_path t).1 - 8)^2 + ((other_path t).2 - 6)^2 ≥ 36) →
      (∃ length : ℝ, length = 16 + 2.23 * Real.pi ∧
        ∀ other_length : ℝ, length ≤ other_length)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_avoiding_circle_l585_58591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l585_58542

/-- The function f(x) = 3^x + 3x - 8 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3) + 3*x - 8

theorem zero_in_interval :
  Continuous f ∧ StrictMono f →
  ∃! x, x ∈ Set.Ioo 1 (3/2) ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l585_58542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_transformed_plane_l585_58543

noncomputable section

/-- The plane equation before transformation -/
def original_plane (x y z : ℝ) : Prop :=
  6 * x - 5 * y + 3 * z - 4 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := -3/4

/-- The plane equation after transformation -/
def transformed_plane (x y z : ℝ) : Prop :=
  6 * x - 5 * y + 3 * z + 3 = 0

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (0, 1, -1)

/-- Theorem stating that point A does not belong to the transformed plane -/
theorem point_not_on_transformed_plane :
  ¬ transformed_plane point_A.1 point_A.2.1 point_A.2.2 :=
by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_transformed_plane_l585_58543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_digit_pairs_count_l585_58518

/-- A pair of two-digit numbers (A, B) where A > B -/
structure DigitPair where
  A : ℕ
  B : ℕ
  h1 : 10 ≤ A ∧ A < 100
  h2 : 10 ≤ B ∧ B < 100
  h3 : A > B

/-- The sum of a DigitPair -/
def sum (p : DigitPair) : ℕ := p.A + p.B

/-- The difference of a DigitPair -/
def diff (p : DigitPair) : ℕ := p.A - p.B

/-- Check if two numbers have reversed digits -/
def has_reversed_digits (a b : ℕ) : Prop :=
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

/-- The set of all DigitPairs satisfying the reverse digit property -/
def reverse_digit_pairs : Set DigitPair :=
  {p : DigitPair | has_reversed_digits (sum p) (diff p)}

/-- Proof that the set of reverse digit pairs is finite -/
instance : Fintype reverse_digit_pairs := by
  sorry

/-- The count of reverse digit pairs is 9 -/
theorem reverse_digit_pairs_count : Fintype.card reverse_digit_pairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_digit_pairs_count_l585_58518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_squared_l585_58563

/-- Given that x varies inversely as the square of y, prove that if x = 16 when y = 4, then x = 64 when y = -2 -/
theorem inverse_variation_squared (x y : ℝ) (h : ∃ k : ℝ, ∀ y ≠ 0, x * y^2 = k) 
  (h_initial : x = 16 ∧ y = 4) : 
  (fun y ↦ (16 * 4^2) / y^2) (-2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_squared_l585_58563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_set_contains_all_positive_l585_58585

/-- A set of positive real numbers with specific properties -/
def PositiveSetA (A : Set ℝ) : Prop :=
  (∀ x, x ∈ A → x > 0) ∧ 
  (∀ x y, x ∈ A → y ∈ A → x + y ∈ A) ∧
  (∀ a b : ℝ, 0 < a → a < b → ∃ c d, a ≤ c ∧ d ≤ b ∧ c < d ∧ Set.Icc c d ⊆ A)

/-- Theorem stating that a set A with the given properties contains all positive real numbers -/
theorem positive_set_contains_all_positive (A : Set ℝ) (hA : PositiveSetA A) : 
  ∀ x : ℝ, x > 0 → x ∈ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_set_contains_all_positive_l585_58585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l585_58599

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sqrt 3 * Real.tan x - 3)

-- Define the domain set
def domain_set : Set ℝ := {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

-- Theorem statement
theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f x = y} = domain_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l585_58599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_magnitude_l585_58533

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as given in the problem
noncomputable def z : ℂ := 1 / (1 + i) + i

-- Theorem statement
theorem z_magnitude : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_magnitude_l585_58533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owen_initial_turtles_owen_final_count_l585_58535

/-- The number of turtles Owen initially bred -/
def owen_initial : ℕ := 21

/-- The number of turtles Johanna initially had -/
def johanna_initial : ℕ := owen_initial - 5

/-- Owen's turtle count after doubling -/
def owen_doubled : ℕ := 2 * owen_initial

/-- Johanna's turtle count after losing half -/
def johanna_halved : ℕ := johanna_initial / 2

/-- Owen's final turtle count -/
def owen_final : ℕ := owen_doubled + johanna_halved

theorem owen_initial_turtles : owen_initial = 21 :=
  by
    -- We define owen_initial directly as 21, so this is trivially true
    rfl

theorem owen_final_count : owen_final = 50 :=
  by
    -- Expand the definitions and perform the calculation
    unfold owen_final owen_doubled johanna_halved johanna_initial owen_initial
    -- The rest of the proof would go here, but we'll use sorry for now
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_owen_initial_turtles_owen_final_count_l585_58535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_twelve_valid_sequences_l585_58566

/-- Represents a house on Hamilton Avenue -/
inductive House : Type
| one : House
| two : House
| three : House
| four : House
| five : House
| six : House
| seven : House
| eight : House
deriving Inhabited

/-- Determines if a house is odd-numbered -/
def isOdd (h : House) : Prop :=
  h = House.one ∨ h = House.three ∨ h = House.five ∨ h = House.seven

/-- Determines if a house is even-numbered -/
def isEven (h : House) : Prop :=
  h = House.two ∨ h = House.four ∨ h = House.six ∨ h = House.eight

/-- Determines if two houses are directly opposite each other -/
def isOpposite (h1 h2 : House) : Prop :=
  (h1 = House.one ∧ h2 = House.two) ∨
  (h1 = House.three ∧ h2 = House.four) ∨
  (h1 = House.five ∧ h2 = House.six) ∨
  (h1 = House.seven ∧ h2 = House.eight) ∨
  (h2 = House.one ∧ h1 = House.two) ∨
  (h2 = House.three ∧ h1 = House.four) ∨
  (h2 = House.five ∧ h1 = House.six) ∨
  (h2 = House.seven ∧ h1 = House.eight)

/-- Represents a valid delivery sequence -/
def ValidDeliverySequence (seq : List House) : Prop :=
  seq.length = 8 ∧
  seq.head? = some House.one ∧
  seq.getLast? = some House.one ∧
  (∀ i, i > 0 ∧ i < 7 → isOdd (seq[i]!) ↔ isEven (seq[i+1]!)) ∧
  (∀ i j, i ≠ j ∧ i > 0 ∧ i < 7 ∧ j > 0 ∧ j < 7 → seq[i]! ≠ seq[j]!) ∧
  (∀ i, i > 0 ∧ i < 7 → ¬isOpposite (seq[i]!) (seq[i+1]!))

/-- The main theorem stating that there are exactly 12 valid delivery sequences -/
theorem exactly_twelve_valid_sequences :
  ∃! (sequences : List (List House)), 
    sequences.length = 12 ∧ 
    (∀ seq, seq ∈ sequences ↔ ValidDeliverySequence seq) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_twelve_valid_sequences_l585_58566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l585_58515

-- Define the circles
def Circle (center : ℝ × ℝ) (radius : ℝ) : Type :=
  {point : ℝ × ℝ | (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2}

-- Define the tangent line
def TangentLine (m : ℝ) : Type := {point : ℝ × ℝ | point.2 = m * point.1}

-- Define the property of being tangent
def IsTangent (line : Type) (circle : Type) : Prop := sorry

-- Define the intersection of two circles
def Intersect (C₁ C₂ : Type) (point : ℝ × ℝ) : Prop := sorry

-- Define the property of m being in the form a√b/c
def IsInForm (m : ℝ) (a b c : ℕ) : Prop := 
  m = (a : ℝ) * Real.sqrt (b : ℝ) / (c : ℝ) ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ b)) ∧
  Nat.Coprime a c

theorem circle_intersection_theorem 
  (C₁ C₂ : Type) 
  (r₁ r₂ : ℝ) 
  (m : ℝ) 
  (a b c : ℕ) :
  Intersect C₁ C₂ (9, 6) →
  r₁ * r₂ = 68 →
  IsTangent (TangentLine 0) C₁ →
  IsTangent (TangentLine 0) C₂ →
  IsTangent (TangentLine m) C₁ →
  IsTangent (TangentLine m) C₂ →
  m > 0 →
  IsInForm m a b c →
  a + b + c = 282 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l585_58515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_angles_theorem_l585_58500

/-- A parallelogram with a rhombus inscribed as described in the problem. -/
structure RhombusInParallelogram where
  a : ℝ
  b : ℝ
  α : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_lt_b : a < b
  h_α_acute : 0 < α ∧ α < Real.pi / 2

/-- The angles of the inscribed rhombus. -/
noncomputable def rhombusAngles (rip : RhombusInParallelogram) : ℝ × ℝ :=
  let θ := 2 * Real.arctan (rip.a / (rip.b * Real.sin rip.α))
  (θ, Real.pi - θ)

/-- The theorem stating the angles of the inscribed rhombus. -/
theorem rhombus_angles_theorem (rip : RhombusInParallelogram) :
  rhombusAngles rip = (2 * Real.arctan (rip.a / (rip.b * Real.sin rip.α)),
                       Real.pi - 2 * Real.arctan (rip.a / (rip.b * Real.sin rip.α))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_angles_theorem_l585_58500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l585_58576

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def ArithmeticSequenceSum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_10 (a : ℕ → ℚ) :
  ArithmeticSequence a →
  a 4^2 + a 7^2 + 2 * a 4 * a 7 = 9 →
  (ArithmeticSequenceSum a 10 = 15 ∨ ArithmeticSequenceSum a 10 = -15) :=
by
  sorry

#check arithmetic_sequence_sum_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l585_58576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_inequality_solution_l585_58580

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Conditions
axiom func_property : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom positive_property : ∀ x : ℝ, x > 0 → f x > 1
axiom f_4 : f 4 = 3

-- Theorem statements
theorem f_monotone_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

theorem inequality_solution : 
  ∀ m : ℝ, f (3 * m^2 - m - 2) < 2 ↔ -1 < m ∧ m < 4/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_inequality_solution_l585_58580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_find_f_4_l585_58553

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + 1

-- Theorem 1: Given f(-2) = 0, prove a = 2
theorem find_a (a : ℝ) (h : f a (-2) = 0) : a = 2 := by
  sorry

-- Theorem 2: Given a = 6, prove f(4) = 5/2
theorem find_f_4 : f 6 4 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_find_f_4_l585_58553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_non_training_days_l585_58507

/-- Calculates the number of days per week James does not train given his training schedule --/
theorem james_non_training_days (hours_per_session : ℕ) (sessions_per_day : ℕ) (total_hours_per_year : ℕ) : 
  hours_per_session = 4 →
  sessions_per_day = 2 →
  total_hours_per_year = 2080 →
  7 - (total_hours_per_year / 52) / (hours_per_session * sessions_per_day) = 2 := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check james_non_training_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_non_training_days_l585_58507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_inclusion_probabilities_l585_58529

/-- Given K laws, N ministers, and probability p that a minister knows a law,
    prove the probability of M laws being included and the expected number of laws. -/
theorem law_inclusion_probabilities 
  (K N M : ℕ) (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N
  ∃ (prob_M : ℝ), 
    prob_M = Nat.choose K M * q^M * (1 - q)^(K - M) ∧
    0 ≤ prob_M ∧ prob_M ≤ 1 ∧
  ∃ (expected : ℝ),
    expected = K * q ∧
    0 ≤ expected ∧ expected ≤ K := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_inclusion_probabilities_l585_58529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_is_2_level_iff_phi_is_all_t_level_iff_l585_58516

noncomputable def φ (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a / (x^2 + 1))

def is_t_level_distribution (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + t) = f x₀ * f t

-- Theorem 1: φ is a 2-level distribution function iff a ∈ [15-10√2, 15+10√2]
theorem phi_is_2_level_iff (a : ℝ) (h : a > 0) :
  is_t_level_distribution (φ a) 2 ↔ 
  15 - 10 * Real.sqrt 2 ≤ a ∧ a ≤ 15 + 10 * Real.sqrt 2 := by
  sorry

-- Theorem 2: φ is a t-level distribution function for all real t iff a = 1
theorem phi_is_all_t_level_iff (a : ℝ) (h : a > 0) :
  (∀ t : ℝ, is_t_level_distribution (φ a) t) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_is_2_level_iff_phi_is_all_t_level_iff_l585_58516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_third_quadrant_l585_58584

/-- An angle is in the third quadrant if its sine and cosine are both negative -/
def ThirdQuadrant (α : Real) : Prop := Real.sin α < 0 ∧ Real.cos α < 0

theorem sin_cos_sum_third_quadrant (α : Real) (h : ThirdQuadrant α) :
  -1 < Real.sin α + Real.cos α ∧ Real.sin α + Real.cos α ≤ -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_third_quadrant_l585_58584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_triangle_l585_58545

noncomputable section

-- Define the triangle
def triangle_side1 : ℝ := 6
def triangle_side2 : ℝ := 8

-- Define the hypotenuse using Pythagorean theorem
noncomputable def hypotenuse : ℝ := Real.sqrt (triangle_side1^2 + triangle_side2^2)

-- Define the radius of the cone's base
noncomputable def cone_radius : ℝ := hypotenuse / 2

-- Define the volume of a cone
noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem statement
theorem cone_volume_from_triangle :
  (cone_volume cone_radius triangle_side1 = 50 * Real.pi) ∨
  (cone_volume cone_radius triangle_side2 = 200/3 * Real.pi) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_triangle_l585_58545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_distance_half_initial_l585_58546

/-- Represents the state of two ships P and Q during a chase. -/
structure ShipChase where
  -- Distance between ships P and Q
  distance : ℝ
  -- Angle between PQ and Q's direction
  angle : ℝ
  -- Speed of both ships (can change over time)
  speed : ℝ

/-- The initial state of the ship chase -/
noncomputable def initialState : ShipChase :=
  { distance := 10,  -- 10 nautical miles
    angle := Real.pi / 2,  -- Q moves perpendicular to PQ
    speed := 1  -- Arbitrary initial speed
  }

/-- The final state of the ship chase after a long time -/
noncomputable def finalState (initial : ShipChase) : ShipChase :=
  { distance := initial.distance / 2,
    angle := 0,  -- P and Q moving in the same direction
    speed := initial.speed  -- Speed is irrelevant for final state
  }

/-- Theorem stating that the final distance between P and Q is half the initial distance -/
theorem final_distance_half_initial
  (initial : ShipChase)
  (h_initial : initial = initialState)
  (final : ShipChase)
  (h_final : final = finalState initial) :
  final.distance = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_distance_half_initial_l585_58546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_count_after_predation_l585_58560

/-- Calculate the number of birds remaining after predation --/
def birds_remaining (initial_count : ℕ) (predation_percentage : ℚ) : ℕ :=
  (initial_count : ℤ) - Int.floor (predation_percentage * initial_count) |>.toNat

/-- Theorem stating the final bird counts after predation --/
theorem bird_count_after_predation 
  (peregrine_falcons crows pigeons sparrows : ℕ)
  (pigeon_chicks crow_chicks sparrow_chicks : ℕ)
  (pigeon_predation crow_predation sparrow_predation : ℚ)
  (h1 : peregrine_falcons = 12)
  (h2 : pigeons = 80)
  (h3 : crows = 25)
  (h4 : sparrows = 15)
  (h5 : pigeon_chicks = 8)
  (h6 : crow_chicks = 5)
  (h7 : sparrow_chicks = 3)
  (h8 : pigeon_predation = 2/5)
  (h9 : crow_predation = 1/4)
  (h10 : sparrow_predation = 1/10) :
  (peregrine_falcons,
   birds_remaining pigeons pigeon_predation,
   birds_remaining crows crow_predation,
   birds_remaining sparrows sparrow_predation) = (12, 48, 19, 14) := by
  sorry

#check bird_count_after_predation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_count_after_predation_l585_58560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_relation_l585_58521

noncomputable def g (t : ℝ) : ℝ := t / (1 + t)

theorem inverse_g_relation (x z : ℝ) (hx : x ≠ -1) (hz : z ≠ 1) :
  z = g x → x = -g (-z) := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_relation_l585_58521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_proof_l585_58598

noncomputable def container_capacity : ℝ := 60
noncomputable def initial_fullness : ℝ := 0.3
noncomputable def final_fullness : ℝ := 3/4

theorem water_added_proof :
  let initial_water := initial_fullness * container_capacity
  let final_water := final_fullness * container_capacity
  final_water - initial_water = 27 := by
  -- Unfold the definitions
  unfold container_capacity initial_fullness final_fullness
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_proof_l585_58598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_straight_line_l585_58548

-- Define the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | distance p M - distance p N = 4}

-- Theorem statement
theorem locus_is_straight_line :
  ∃ (a b c : ℝ), ∀ p : ℝ × ℝ, p ∈ S ↔ a * p.1 + b * p.2 + c = 0 := by
  sorry

#check locus_is_straight_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_straight_line_l585_58548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l585_58556

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- Sum of angles in a triangle
  A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of sines
  a / sin A = b / sin B ∧ b / sin B = c / sin C →
  -- Given condition
  tan A = (sin C + sin B) / (cos C + cos B) →
  -- Conclusions
  A = π/3 ∧ -1/3 < (b^2 - b*c) / a^2 ∧ (b^2 - b*c) / a^2 < 2/3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l585_58556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_in_cube_correct_solution_l585_58504

structure Cube where
  edgeLength : ℝ

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflectionPoint (c : Cube) : Point3D :=
  { x := 6, y := 3, z := c.edgeLength }

noncomputable def lightPathLength (c : Cube) : ℝ :=
  10 * Real.sqrt 145

theorem light_path_in_cube (c : Cube) (h : c.edgeLength = 10) :
  lightPathLength c = 10 * Real.sqrt 145 := by
  sorry

def solution : ℕ := 155

theorem correct_solution :
  solution = 155 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_in_cube_correct_solution_l585_58504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l585_58577

noncomputable section

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f(x+1) = f(x-1) for all x
def periodic_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

-- For 1 ≤ x ≤ 2, f(x) = log_a(x) where a > 0 and a ≠ 1
def log_property (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (a > 0 ∧ a ≠ 1) ∧ ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = Real.log x / Real.log a

-- The theorem to prove
theorem f_expression (h_even : even_function f) (h_periodic : periodic_property f) 
  (h_log : log_property f a) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → 
    f x = if x ≤ 0 then Real.log (x + 2) / Real.log a else Real.log (2 - x) / Real.log a := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l585_58577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_discount_for_suit_l585_58522

/-- Calculates the coupon discount percentage given the original price, price increase percentage, and final price after coupon. -/
noncomputable def coupon_discount_percentage (original_price : ℝ) (price_increase_percent : ℝ) (final_price : ℝ) : ℝ :=
  let increased_price := original_price * (1 + price_increase_percent / 100)
  100 * (1 - final_price / increased_price)

/-- Theorem stating that for a suit with original price $150, 20% price increase, and final price $144 after coupon, the coupon discount is 20%. -/
theorem coupon_discount_for_suit : 
  coupon_discount_percentage 150 20 144 = 20 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_discount_for_suit_l585_58522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rates_calculation_l585_58511

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rates_calculation (principal_B principal_C time_B time_C interest_B interest_C : ℝ) 
  (h1 : principal_B = 5000)
  (h2 : principal_C = 3000)
  (h3 : time_B = 5)
  (h4 : time_C = 7)
  (h5 : interest_B = 2200)
  (h6 : interest_C = 2730) :
  ∃ (rate_B rate_C : ℝ),
    simple_interest principal_B rate_B time_B = interest_B ∧
    simple_interest principal_C rate_C time_C = interest_C ∧
    rate_B = 8.8 ∧
    rate_C = 13 := by
  sorry

#check interest_rates_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rates_calculation_l585_58511
