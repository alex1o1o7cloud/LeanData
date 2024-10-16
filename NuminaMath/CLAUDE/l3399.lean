import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_point_yOz_l3399_339945

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the yOz plane
def yOz_plane (p : Point3D) : Prop := p.x = 0

-- Define symmetry with respect to the yOz plane
def symmetric_to_yOz (a b : Point3D) : Prop :=
  b.x = -a.x ∧ b.y = a.y ∧ b.z = a.z

theorem symmetry_point_yOz :
  let a := Point3D.mk (-2) 4 3
  let b := Point3D.mk 2 4 3
  symmetric_to_yOz a b := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_yOz_l3399_339945


namespace NUMINAMATH_CALUDE_min_cuts_to_touch_coin_l3399_339933

/-- Represents a circular object with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a straight cut on the pancake -/
structure Cut where
  width : ℝ

/-- The pancake -/
def pancake : Circle := { radius := 10 }

/-- The coin -/
def coin : Circle := { radius := 1 }

/-- The width of the area covered by a single cut -/
def cut_width : ℝ := 2

/-- The minimum number of cuts needed -/
def min_cuts : ℕ := 10

theorem min_cuts_to_touch_coin : 
  ∀ (cuts : ℕ), 
    cuts < min_cuts → 
    ∃ (coin_position : ℝ × ℝ), 
      coin_position.1^2 + coin_position.2^2 ≤ pancake.radius^2 ∧ 
      ∀ (cut : Cut), cut.width = cut_width → 
        ∃ (d : ℝ), d > coin.radius ∧ 
          ∀ (p : ℝ × ℝ), p.1^2 + p.2^2 ≤ coin.radius^2 → 
            (p.1 - coin_position.1)^2 + (p.2 - coin_position.2)^2 ≤ d^2 := by
  sorry

#check min_cuts_to_touch_coin

end NUMINAMATH_CALUDE_min_cuts_to_touch_coin_l3399_339933


namespace NUMINAMATH_CALUDE_xy_max_value_l3399_339969

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 16) :
  x * y ≤ 32 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 16 ∧ x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l3399_339969


namespace NUMINAMATH_CALUDE_inverse_f_sum_l3399_339932

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3*x - x^2

theorem inverse_f_sum : ∃ y₁ y₂ y₃ : ℝ, 
  f y₁ = -4 ∧ f y₂ = 1 ∧ f y₃ = 4 ∧ y₁ + y₂ + y₃ = 5 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l3399_339932


namespace NUMINAMATH_CALUDE_stream_speed_l3399_339902

/-- Proves that the speed of a stream is 5 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 22 →
  distance = 135 →
  time = 5 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l3399_339902


namespace NUMINAMATH_CALUDE_no_consecutive_tails_probability_l3399_339912

/-- Represents the number of ways to toss n coins without getting two consecutive tails -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => a (n + 1) + a n

/-- The probability of not getting two consecutive tails when tossing five fair coins -/
theorem no_consecutive_tails_probability : 
  (a 5 : ℚ) / (2^5 : ℚ) = 13 / 32 := by sorry

end NUMINAMATH_CALUDE_no_consecutive_tails_probability_l3399_339912


namespace NUMINAMATH_CALUDE_kittens_left_tim_kittens_left_l3399_339928

/-- Given an initial number of kittens and the number of kittens given to two people,
    calculate the number of kittens left. -/
theorem kittens_left (initial : ℕ) (given_to_jessica : ℕ) (given_to_sara : ℕ) :
  initial - (given_to_jessica + given_to_sara) = initial - given_to_jessica - given_to_sara :=
by sorry

/-- Prove that Tim has 9 kittens left after giving away some kittens. -/
theorem tim_kittens_left :
  let initial := 18
  let given_to_jessica := 3
  let given_to_sara := 6
  initial - (given_to_jessica + given_to_sara) = 9 :=
by sorry

end NUMINAMATH_CALUDE_kittens_left_tim_kittens_left_l3399_339928


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3399_339978

/-- Given two positive real numbers a and b between 4 and 16,
    if (4, a, b) and (a, b, 16) are both in geometric progression,
    then a + b = 4(∛4 + 2∛2) -/
theorem inserted_numbers_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (hab : 4 < a ∧ a < b ∧ b < 16) 
    (h_geo1 : ∃ r : ℝ, r > 0 ∧ a = 4 * r ∧ b = 4 * r^2)
    (h_geo2 : ∃ s : ℝ, s > 0 ∧ b = a * s ∧ 16 = b * s) : 
  a + b = 4 * (Real.rpow 4 (1/3) + 2 * Real.rpow 2 (1/3)) := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3399_339978


namespace NUMINAMATH_CALUDE_triangle_properties_l3399_339947

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle A is π/3 and the area is 4√3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sin C / (Real.sin A + Real.sin B) + b / (a + c) = 1 →
  |b - a| = 4 →
  Real.cos B + Real.cos C = 1 →
  A = π / 3 ∧ a * c * (Real.sin B) / 2 = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3399_339947


namespace NUMINAMATH_CALUDE_three_distinct_volumes_l3399_339930

/-- A triangular pyramid with specific face conditions -/
structure TriangularPyramid where
  /-- Two lateral faces are isosceles right triangles -/
  has_two_isosceles_right_faces : Bool
  /-- One face is an equilateral triangle with side length 1 -/
  has_equilateral_face : Bool
  /-- The side length of the equilateral face -/
  equilateral_side_length : ℝ

/-- The volume of a triangular pyramid -/
def volume (pyramid : TriangularPyramid) : ℝ := sorry

/-- The set of all possible volumes for triangular pyramids satisfying the conditions -/
def possible_volumes : Set ℝ := sorry

/-- Theorem stating that there are exactly three distinct volumes -/
theorem three_distinct_volumes :
  ∃ (v₁ v₂ v₃ : ℝ), v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧
  possible_volumes = {v₁, v₂, v₃} :=
sorry

end NUMINAMATH_CALUDE_three_distinct_volumes_l3399_339930


namespace NUMINAMATH_CALUDE_grid_value_bound_l3399_339980

/-- The value of a square in the grid -/
def square_value (is_filled : Bool) (filled_neighbors : Nat) : Nat :=
  if is_filled then 0 else filled_neighbors

/-- The maximum number of neighbors a square can have -/
def max_neighbors : Nat := 8

/-- The function f(m,n) representing the largest total value of squares in the grid -/
noncomputable def f (m n : Nat) : Nat :=
  sorry  -- Definition of f(m,n) is complex and depends on optimal grid configuration

/-- The theorem stating that 2 is the minimal constant C such that f(m,n) / (m*n) ≤ C -/
theorem grid_value_bound (m n : Nat) (hm : m > 0) (hn : n > 0) :
  (f m n : ℝ) / (m * n : ℝ) ≤ 2 ∧ ∀ C : ℝ, (∀ m' n' : Nat, m' > 0 → n' > 0 → (f m' n' : ℝ) / (m' * n' : ℝ) ≤ C) → C ≥ 2 :=
  sorry


end NUMINAMATH_CALUDE_grid_value_bound_l3399_339980


namespace NUMINAMATH_CALUDE_order_of_numbers_l3399_339909

theorem order_of_numbers (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) :
  a > -b ∧ -b > b ∧ b > -a := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3399_339909


namespace NUMINAMATH_CALUDE_f_is_odd_l3399_339996

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 / (1 - x)) - 1) / Real.log 10

theorem f_is_odd : ∀ x : ℝ, x ≠ 1 → f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l3399_339996


namespace NUMINAMATH_CALUDE_quadratic_and_related_function_properties_l3399_339979

/-- Given a quadratic function f and its derivative, prove properties about its coefficients and a related function g --/
theorem quadratic_and_related_function_properties
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ x, f x = a * x^2 + b * x + 3)
  (h₂ : a ≠ 0)
  (h₃ : ∀ x, deriv f x = 2 * x - 8)
  (g : ℝ → ℝ)
  (h₄ : ∀ x, g x = Real.exp x * Real.sin x + f x) :
  a = 1 ∧ b = -8 ∧
  (∃ m c : ℝ, m = 7 ∧ c = -3 ∧ ∀ x y, y = deriv g 0 * (x - 0) + g 0 ↔ m * x + y + c = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_and_related_function_properties_l3399_339979


namespace NUMINAMATH_CALUDE_roses_cut_equality_l3399_339900

/-- Represents the number of roses in various states --/
structure RoseCount where
  initial : ℕ
  thrown : ℕ
  given : ℕ
  final : ℕ

/-- Calculates the number of roses cut --/
def rosesCut (r : RoseCount) : ℕ :=
  r.final - r.initial + r.thrown + r.given

/-- Theorem stating that the number of roses cut is equal to the sum of
    the difference between final and initial roses, roses thrown away, and roses given away --/
theorem roses_cut_equality (r : RoseCount) :
  rosesCut r = r.final - r.initial + r.thrown + r.given :=
by sorry

end NUMINAMATH_CALUDE_roses_cut_equality_l3399_339900


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l3399_339991

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l3399_339991


namespace NUMINAMATH_CALUDE_smallest_y_value_l3399_339973

theorem smallest_y_value : ∃ y : ℝ, 
  (∀ z : ℝ, 3 * z^2 + 21 * z + 18 = z * (2 * z + 12) → y ≤ z) ∧
  (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧
  y = -6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_value_l3399_339973


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3399_339958

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    with eccentricity e = √7/2, and a point P on the right branch of the hyperbola
    such that PF₂ ⊥ F₁F₂ and PF₂ = 9/2, prove that the length of the conjugate axis
    is 6√3. -/
theorem hyperbola_conjugate_axis_length
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (he : Real.sqrt 7 / 2 = Real.sqrt (1 + b^2 / a^2))
  (hP : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0)
  (hPF2 : b^2 / a = 9 / 2) :
  2 * b = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3399_339958


namespace NUMINAMATH_CALUDE_f_derivative_l3399_339965

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- State the theorem
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l3399_339965


namespace NUMINAMATH_CALUDE_star_three_four_l3399_339959

-- Define the star operation
def star (a b : ℝ) : ℝ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_three_four : star 3 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l3399_339959


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3399_339976

/-- Time for a train to cross a bridge with another train coming from the opposite direction -/
theorem train_bridge_crossing_time
  (train1_length : ℝ)
  (train1_speed : ℝ)
  (bridge_length : ℝ)
  (train2_length : ℝ)
  (train2_speed : ℝ)
  (h1 : train1_length = 110)
  (h2 : train1_speed = 60)
  (h3 : bridge_length = 170)
  (h4 : train2_length = 90)
  (h5 : train2_speed = 45)
  : ∃ (time : ℝ), abs (time - 280 / (60 * 1000 / 3600 + 45 * 1000 / 3600)) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3399_339976


namespace NUMINAMATH_CALUDE_arithmetic_progression_quartic_l3399_339994

theorem arithmetic_progression_quartic (q : ℝ) : 
  (∃ (a d : ℝ), ∀ (x : ℝ), x^4 - 40*x^2 + q = 0 ↔ 
    (x = a - 3*d/2 ∨ x = a - d/2 ∨ x = a + d/2 ∨ x = a + 3*d/2)) → 
  q = 144 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_progression_quartic_l3399_339994


namespace NUMINAMATH_CALUDE_canoe_capacity_ratio_l3399_339924

def canoe_capacity : ℕ := 6
def person_weight : ℕ := 140
def dog_weight_ratio : ℚ := 1/4
def total_weight_with_dog : ℕ := 595

theorem canoe_capacity_ratio :
  let people_with_dog := (total_weight_with_dog - person_weight * dog_weight_ratio) / person_weight
  (people_with_dog : ℚ) / canoe_capacity = 2/3 := by sorry

end NUMINAMATH_CALUDE_canoe_capacity_ratio_l3399_339924


namespace NUMINAMATH_CALUDE_expression_simplification_l3399_339905

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3399_339905


namespace NUMINAMATH_CALUDE_right_triangle_area_l3399_339954

theorem right_triangle_area (a c : ℝ) (h1 : a = 40) (h2 : c = 41) :
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3399_339954


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3399_339916

theorem angle_ABC_measure :
  ∀ (angle_ABC angle_ABD angle_CBD : ℝ),
  angle_CBD = 90 →
  angle_ABC + angle_ABD + angle_CBD = 270 →
  angle_ABD = 100 →
  angle_ABC = 80 := by
sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3399_339916


namespace NUMINAMATH_CALUDE_implicit_derivative_l3399_339919

noncomputable section

open Real

-- Define the implicit function
def F (x y : ℝ) : ℝ := log (sqrt (x^2 + y^2)) - arctan (y / x)

-- State the theorem
theorem implicit_derivative (x y : ℝ) (h1 : x ≠ 0) (h2 : x ≠ y) :
  let y' := (x + y) / (x - y)
  (∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |F (x + h) (y + y' * h) - F x y| ≤ ε * |h|) :=
sorry

end

end NUMINAMATH_CALUDE_implicit_derivative_l3399_339919


namespace NUMINAMATH_CALUDE_sqrt_31_plus_3_tan_56_approx_7_l3399_339988

/-- Prove that the absolute difference between √31 + 3tan(56°) and 7.00 is less than 0.005 -/
theorem sqrt_31_plus_3_tan_56_approx_7 :
  |Real.sqrt 31 + 3 * Real.tan (56 * π / 180) - 7| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_plus_3_tan_56_approx_7_l3399_339988


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l3399_339908

theorem square_of_linear_expression (x : ℝ) :
  x = -2 → (3*x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l3399_339908


namespace NUMINAMATH_CALUDE_fraction_equality_l3399_339914

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) :
  m / q = 3 / 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3399_339914


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3399_339935

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant 3 (-4) :=
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3399_339935


namespace NUMINAMATH_CALUDE_solution_system_equations_l3399_339950

theorem solution_system_equations :
  ∀ x y z : ℝ,
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4 →
  ((x = 1/3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2)) :=
by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3399_339950


namespace NUMINAMATH_CALUDE_solution_count_condition_condition_implies_solution_count_l3399_339999

/-- The system of equations has three or two solutions if and only if a = ±1 or a = ±√2 -/
theorem solution_count_condition (a : ℝ) : 
  (∃ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1 → 
    (x = 0 ∨ x ≠ 0) ∧ (y = 0 ∨ y ≠ 0)) →
  (a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

/-- If a = ±1 or a = ±√2, then the system has three or two solutions -/
theorem condition_implies_solution_count (a : ℝ) 
  (h : a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :
  (∃ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1 → 
    (x = 0 ∨ x ≠ 0) ∧ (y = 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_count_condition_condition_implies_solution_count_l3399_339999


namespace NUMINAMATH_CALUDE_contrapositive_divisibility_l3399_339964

theorem contrapositive_divisibility (n : ℤ) : 
  (∀ m : ℤ, m % 6 = 0 → m % 2 = 0) ↔ 
  (∀ k : ℤ, k % 2 ≠ 0 → k % 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_divisibility_l3399_339964


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3399_339923

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ a b : ℚ, y = a * (b ^ (1/2 : ℝ))) → (∃ c d : ℚ, x = c * (d ^ (1/2 : ℝ))) → False

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (26 ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (8 ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical ((1/3 : ℝ) ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (2 / (6 ^ (1/2 : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3399_339923


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3399_339907

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 22

theorem total_cost_calculation :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 941.6) :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3399_339907


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3399_339936

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b + l1.b * l2.a = 0

-- Define a point being on a line
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- The main theorem
theorem perpendicular_line_through_point :
  let P : Point2D := ⟨1, -2⟩
  let given_line : Line2D := ⟨1, -3, 2⟩
  let result_line : Line2D := ⟨3, 1, -1⟩
  perpendicular given_line result_line ∧ point_on_line P result_line := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3399_339936


namespace NUMINAMATH_CALUDE_napkin_division_l3399_339971

structure Napkin :=
  (is_square : Bool)
  (folds : Nat)
  (cut_type : String)

def can_divide (n : Napkin) (parts : Nat) : Prop :=
  n.is_square ∧ n.folds = 2 ∧ n.cut_type = "straight" ∧ 
  ((parts = 2 ∨ parts = 3 ∨ parts = 4) ∨ parts ≠ 5)

theorem napkin_division (n : Napkin) (parts : Nat) :
  can_divide n parts ↔ (parts = 2 ∨ parts = 3 ∨ parts = 4) :=
sorry

end NUMINAMATH_CALUDE_napkin_division_l3399_339971


namespace NUMINAMATH_CALUDE_smallest_urn_satisfying_condition_l3399_339938

/-- An urn contains marbles of five colors: red, white, blue, green, and yellow. -/
structure Urn :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- The total number of marbles in the urn -/
def Urn.total (u : Urn) : ℕ := u.red + u.white + u.blue + u.green + u.yellow

/-- The probability of drawing five red marbles -/
def Urn.prob_five_red (u : Urn) : ℚ :=
  (u.red.choose 5 : ℚ) / (u.total.choose 5)

/-- The probability of drawing one white, one blue, and three red marbles -/
def Urn.prob_one_white_one_blue_three_red (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.red.choose 3) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one white, one blue, one green, and two red marbles -/
def Urn.prob_one_white_one_blue_one_green_two_red (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 2) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one marble of each color except yellow -/
def Urn.prob_one_each_except_yellow (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 1) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one marble of each color -/
def Urn.prob_one_each (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 1) * (u.yellow.choose 1) : ℚ) / (u.total.choose 5)

/-- The urn satisfies the equal probability condition -/
def Urn.satisfies_condition (u : Urn) : Prop :=
  u.prob_five_red = u.prob_one_white_one_blue_three_red ∧
  u.prob_five_red = u.prob_one_white_one_blue_one_green_two_red ∧
  u.prob_five_red = u.prob_one_each_except_yellow ∧
  u.prob_five_red = u.prob_one_each

theorem smallest_urn_satisfying_condition :
  ∃ (u : Urn), u.satisfies_condition ∧ u.total = 14 ∧ ∀ (v : Urn), v.satisfies_condition → u.total ≤ v.total :=
sorry

end NUMINAMATH_CALUDE_smallest_urn_satisfying_condition_l3399_339938


namespace NUMINAMATH_CALUDE_set_equality_l3399_339981

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define sets M and N
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {1, 3, 6}

-- Theorem statement
theorem set_equality : (U \ M) ∩ N = {1, 6} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3399_339981


namespace NUMINAMATH_CALUDE_binary_to_decimal_1100101_l3399_339956

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1100101₂ -/
def binary_number : List Bool := [true, false, true, false, false, true, true]

/-- Theorem: The decimal equivalent of 1100101₂ is 101 -/
theorem binary_to_decimal_1100101 :
  binary_to_decimal binary_number = 101 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_1100101_l3399_339956


namespace NUMINAMATH_CALUDE_season_games_count_l3399_339953

/-- The number of teams in the conference -/
def num_teams : ℕ := 10

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := num_teams * (num_teams - 1) + num_teams * non_conference_games

theorem season_games_count :
  total_games = 150 :=
by sorry

end NUMINAMATH_CALUDE_season_games_count_l3399_339953


namespace NUMINAMATH_CALUDE_negation_equivalence_l3399_339943

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3399_339943


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_factors_l3399_339951

-- Define a function to count the number of factors of a natural number
def count_factors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is five digits
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_five_digit_multiple_with_16_factors : 
  ∀ n : ℕ, is_five_digit n → n % 2014 = 0 → 
  count_factors (n % 1000) = 16 → n ≥ 24168 := by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_factors_l3399_339951


namespace NUMINAMATH_CALUDE_chess_game_probability_l3399_339941

theorem chess_game_probability (p_win p_not_lose : ℝ) :
  p_win = 0.3 → p_not_lose = 0.8 → p_win + (p_not_lose - p_win) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3399_339941


namespace NUMINAMATH_CALUDE_clock_angles_at_3_and_6_l3399_339990

/-- The angle between the hour hand and minute hand of a clock at a given time -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

theorem clock_angles_at_3_and_6 :
  (clock_angle 3 0 = 90) ∧ (clock_angle 6 0 = 180) := by
  sorry

end NUMINAMATH_CALUDE_clock_angles_at_3_and_6_l3399_339990


namespace NUMINAMATH_CALUDE_lamp_cost_l3399_339986

/-- Proves the cost of the lamp given Daria's furniture purchase scenario -/
theorem lamp_cost (savings : ℕ) (couch_cost table_cost remaining_debt : ℕ) : 
  savings = 500 → 
  couch_cost = 750 → 
  table_cost = 100 → 
  remaining_debt = 400 → 
  ∃ (lamp_cost : ℕ), 
    lamp_cost = remaining_debt - (couch_cost + table_cost - savings) ∧ 
    lamp_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_lamp_cost_l3399_339986


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3399_339946

/-- Calculate compound interest for a fixed deposit -/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℕ) 
  (h1 : principal = 50000) 
  (h2 : rate = 0.04) 
  (h3 : time = 3) : 
  (principal * (1 + rate)^time - principal) = (5 * (1 + 0.04)^3 - 5) * 10000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l3399_339946


namespace NUMINAMATH_CALUDE_rectangle_length_l3399_339993

theorem rectangle_length (P b l A : ℝ) : 
  P / b = 5 → 
  A = 216 → 
  P = 2 * (l + b) → 
  A = l * b → 
  l = 18 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_l3399_339993


namespace NUMINAMATH_CALUDE_min_values_xy_l3399_339917

/-- Given positive real numbers x and y satisfying 2xy = x + 4y + a,
    prove the minimum values for xy and x + y + 2/x + 1/(2y) for different values of a. -/
theorem min_values_xy (x y a : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x * y = x + 4 * y + a) :
  (a = 16 → x * y ≥ 16) ∧
  (a = 0 → x + y + 2 / x + 1 / (2 * y) ≥ 11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_l3399_339917


namespace NUMINAMATH_CALUDE_number_of_schools_l3399_339910

def students_per_school : ℕ := 247
def total_students : ℕ := 6175

theorem number_of_schools : (total_students / students_per_school : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_schools_l3399_339910


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3399_339977

/-- The perimeter of a semicircle with radius r is π * r + 2r -/
theorem semicircle_perimeter (r : ℝ) (h : r = 6.5) :
  ∃ P : ℝ, P = π * r + 2 * r := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3399_339977


namespace NUMINAMATH_CALUDE_probability_ratio_l3399_339960

-- Define the total number of slips
def total_slips : ℕ := 30

-- Define the number of different numbers on the slips
def num_options : ℕ := 6

-- Define the number of slips for each number
def slips_per_number : ℕ := 5

-- Define the number of slips drawn
def drawn_slips : ℕ := 4

-- Define the probability of drawing four slips with the same number
def p : ℚ := (num_options * slips_per_number) / Nat.choose total_slips drawn_slips

-- Define the probability of drawing two pairs of slips with different numbers
def q : ℚ := (Nat.choose num_options 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

-- Theorem statement
theorem probability_ratio : q / p = 50 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l3399_339960


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l3399_339995

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_corners : Nat)

/-- The number of unpainted unit cubes in a cube with painted corners -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - c.painted_corners

/-- Theorem stating the number of unpainted cubes in the specific 4x4x4 cube -/
theorem unpainted_cubes_in_4x4x4_cube :
  ∃ (c : Cube), c.size = 4 ∧ c.total_units = 64 ∧ c.painted_corners = 8 ∧ unpainted_cubes c = 56 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l3399_339995


namespace NUMINAMATH_CALUDE_earth_moon_distance_scientific_notation_l3399_339989

/-- Represents the distance from Earth to Moon in kilometers -/
def earth_moon_distance : ℝ := 384401

/-- Converts a real number to scientific notation with given significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem earth_moon_distance_scientific_notation :
  to_scientific_notation earth_moon_distance 3 = (3.84, 5) :=
sorry

end NUMINAMATH_CALUDE_earth_moon_distance_scientific_notation_l3399_339989


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3399_339931

/-- Proves that when the cost price of 30 books equals the selling price of 40 books, the loss percentage is 25% -/
theorem book_sale_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h : 30 * cost_price = 40 * selling_price) : 
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3399_339931


namespace NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l3399_339974

/-- A parabola with equation y^2 = 4x and two points on it -/
structure Parabola where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁^2 = 4 * x₁
  eq₂ : y₂^2 = 4 * x₂
  dist : (x₁ - x₂)^2 + (y₁ - y₂)^2 = 49  -- |AB|^2 = 7^2

/-- The distance from the midpoint of AB to the directrix of the parabola is 7/2 -/
theorem parabola_midpoint_to_directrix (p : Parabola) : 
  (p.x₁ + p.x₂) / 2 + 1 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l3399_339974


namespace NUMINAMATH_CALUDE_triangle_side_length_l3399_339966

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  A = π / 3 →  -- 60 degrees in radians
  B = π / 4 →  -- 45 degrees in radians
  b = Real.sqrt 6 →
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3399_339966


namespace NUMINAMATH_CALUDE_confucius_wine_consumption_l3399_339901

theorem confucius_wine_consumption :
  let wine_sequence : List ℚ := [1, 1, 1/2, 1/4, 1/8, 1/16]
  List.sum wine_sequence = 47/16 := by
  sorry

end NUMINAMATH_CALUDE_confucius_wine_consumption_l3399_339901


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3399_339903

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3399_339903


namespace NUMINAMATH_CALUDE_sum_reciprocals_geq_nine_l3399_339975

theorem sum_reciprocals_geq_nine (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  1/x + 1/y + 1/z ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_geq_nine_l3399_339975


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3399_339997

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3399_339997


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l3399_339992

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_sum_of_digits (A B C D : ℕ) : 
  is_digit A → is_digit B → is_digit C → is_digit D →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (C + D) % 2 = 0 →
  (A + B) % (C + D) = 0 →
  A + B ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l3399_339992


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_of_3_pow_6_minus_1_l3399_339920

theorem sum_of_prime_factors_of_3_pow_6_minus_1 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range ((3^6 - 1) + 1))) id) = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_of_3_pow_6_minus_1_l3399_339920


namespace NUMINAMATH_CALUDE_cost_price_is_four_l3399_339911

/-- The cost price of a bag of popcorn -/
def cost_price : ℝ := sorry

/-- The selling price of a bag of popcorn -/
def selling_price : ℝ := 8

/-- The number of bags sold -/
def bags_sold : ℝ := 30

/-- The total profit -/
def total_profit : ℝ := 120

/-- Theorem: The cost price of each bag of popcorn is $4 -/
theorem cost_price_is_four :
  cost_price = 4 :=
by
  have h1 : total_profit = bags_sold * (selling_price - cost_price) :=
    sorry
  sorry

end NUMINAMATH_CALUDE_cost_price_is_four_l3399_339911


namespace NUMINAMATH_CALUDE_sandy_total_marks_l3399_339904

/-- Sandy's marking system and attempt results -/
structure SandyAttempt where
  correct_marks : ℕ  -- Marks for each correct sum
  incorrect_penalty : ℕ  -- Marks lost for each incorrect sum
  total_attempts : ℕ  -- Total number of sums attempted
  correct_attempts : ℕ  -- Number of correct sums

/-- Calculate Sandy's total marks -/
def calculate_total_marks (s : SandyAttempt) : ℤ :=
  (s.correct_attempts * s.correct_marks : ℤ) -
  ((s.total_attempts - s.correct_attempts) * s.incorrect_penalty : ℤ)

/-- Theorem stating that Sandy's total marks is 65 -/
theorem sandy_total_marks :
  let s : SandyAttempt := {
    correct_marks := 3,
    incorrect_penalty := 2,
    total_attempts := 30,
    correct_attempts := 25
  }
  calculate_total_marks s = 65 := by
  sorry

end NUMINAMATH_CALUDE_sandy_total_marks_l3399_339904


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3399_339927

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), (2*x + 3*y = (6-x) + (6-3*y)) ∧ (x - 2*y = (x-2) - (y+2)) ∧ x = -4 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3399_339927


namespace NUMINAMATH_CALUDE_prime_between_squares_l3399_339922

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ a : ℕ, p = a^2 + 5 ∧ p = (a+1)^2 - 8 :=
by
  sorry

end NUMINAMATH_CALUDE_prime_between_squares_l3399_339922


namespace NUMINAMATH_CALUDE_joshua_toy_cars_l3399_339949

theorem joshua_toy_cars : 
  ∀ (box1 box2 box3 box4 box5 : ℕ),
    box1 = 21 →
    box2 = 31 →
    box3 = 19 →
    box4 = 45 →
    box5 = 27 →
    box1 + box2 + box3 + box4 + box5 = 143 :=
by
  sorry

end NUMINAMATH_CALUDE_joshua_toy_cars_l3399_339949


namespace NUMINAMATH_CALUDE_peters_situps_l3399_339952

theorem peters_situps (greg_situps : ℕ) (ratio : ℚ) : 
  greg_situps = 32 →
  ratio = 3 / 4 →
  ∃ peter_situps : ℕ, peter_situps * 4 = greg_situps * 3 ∧ peter_situps = 24 :=
by sorry

end NUMINAMATH_CALUDE_peters_situps_l3399_339952


namespace NUMINAMATH_CALUDE_total_baseball_cards_l3399_339961

-- Define the number of people
def num_people : Nat := 4

-- Define the number of cards each person has
def cards_per_person : Nat := 3

-- Theorem to prove
theorem total_baseball_cards : num_people * cards_per_person = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l3399_339961


namespace NUMINAMATH_CALUDE_hike_time_calculation_l3399_339906

theorem hike_time_calculation (distance : ℝ) (pace_up : ℝ) (pace_down : ℝ) 
  (h1 : distance = 12)
  (h2 : pace_up = 4)
  (h3 : pace_down = 6) :
  distance / pace_up + distance / pace_down = 5 := by
  sorry

end NUMINAMATH_CALUDE_hike_time_calculation_l3399_339906


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3399_339934

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 2 ↔ (x = 1 ∧ y = 3) ∨ (x = -1 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3399_339934


namespace NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l3399_339955

theorem not_all_zero_equiv_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l3399_339955


namespace NUMINAMATH_CALUDE_eulers_formula_l3399_339915

/-- A polyhedron with S vertices, A edges, and F faces, where no four vertices are coplanar. -/
structure Polyhedron where
  S : ℕ  -- number of vertices
  A : ℕ  -- number of edges
  F : ℕ  -- number of faces
  no_four_coplanar : True  -- represents the condition that no four vertices are coplanar

/-- Euler's formula for polyhedra -/
theorem eulers_formula (p : Polyhedron) : p.S + p.F = p.A + 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l3399_339915


namespace NUMINAMATH_CALUDE_sequence_integer_condition_l3399_339921

def sequence_condition (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))

def infinitely_many_integers (x : ℕ → ℝ) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ ∃ k : ℤ, x n = k

theorem sequence_integer_condition (x : ℕ → ℝ) :
  (∀ n : ℕ, x n ≠ 0) →
  sequence_condition x →
  (infinitely_many_integers x ↔ ∃ k : ℤ, k ≠ 0 ∧ x 1 = k ∧ x 2 = k) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_condition_l3399_339921


namespace NUMINAMATH_CALUDE_andys_basketball_team_size_l3399_339987

/-- The number of cookies Andy had initially -/
def initial_cookies : ℕ := 72

/-- The number of cookies Andy ate -/
def cookies_eaten : ℕ := 3

/-- The number of cookies Andy gave to his little brother -/
def cookies_given : ℕ := 5

/-- Calculate the remaining cookies after Andy ate some and gave some to his brother -/
def remaining_cookies : ℕ := initial_cookies - (cookies_eaten + cookies_given)

/-- Function to calculate the sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := n * n

theorem andys_basketball_team_size :
  ∃ (team_size : ℕ), team_size > 0 ∧ sum_odd_numbers team_size = remaining_cookies :=
sorry

end NUMINAMATH_CALUDE_andys_basketball_team_size_l3399_339987


namespace NUMINAMATH_CALUDE_sequence_sixth_term_l3399_339939

theorem sequence_sixth_term (a : ℕ+ → ℤ) (S : ℕ+ → ℤ) : 
  (∀ n : ℕ+, S n = n^2 - 3*n) → 
  (∀ n : ℕ+, a n = S n - S (n-1)) → 
  a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sixth_term_l3399_339939


namespace NUMINAMATH_CALUDE_student_arrangements_l3399_339926

def num_students : ℕ := 6

-- Condition 1: A not at head, B not at tail
def condition1 (arrangements : ℕ) : Prop :=
  arrangements = 504

-- Condition 2: A, B, and C not adjacent
def condition2 (arrangements : ℕ) : Prop :=
  arrangements = 144

-- Condition 3: A and B adjacent, C and D adjacent
def condition3 (arrangements : ℕ) : Prop :=
  arrangements = 96

-- Condition 4: Neither A nor B adjacent to C
def condition4 (arrangements : ℕ) : Prop :=
  arrangements = 288

theorem student_arrangements :
  ∃ (arr1 arr2 arr3 arr4 : ℕ),
    condition1 arr1 ∧
    condition2 arr2 ∧
    condition3 arr3 ∧
    condition4 arr4 :=
  by sorry

end NUMINAMATH_CALUDE_student_arrangements_l3399_339926


namespace NUMINAMATH_CALUDE_area_triangle_abc_is_ten_l3399_339940

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Area of triangle ABC is 10 -/
theorem area_triangle_abc_is_ten
  (M₁ : Parabola)
  (M₂ : Parabola)
  (A : Point)
  (B : Point)
  (C : Point)
  (h₁ : M₂.b = -2 * M₂.a) -- M₂ is a horizontal translation of M₁
  (h₂ : A.y = M₂.a * A.x^2 + M₂.b * A.x + M₂.c) -- A is on M₂
  (h₃ : B.x = C.x) -- B and C are on the axis of symmetry of M₂
  (h₄ : C.x = 2 ∧ C.y = M₁.c - 5) -- Coordinates of C
  (h₅ : B.y = M₁.a * B.x^2 + M₁.c) -- B is on M₁
  (h₆ : C.y = M₂.a * C.x^2 + M₂.b * C.x + M₂.c) -- C is on M₂
  : (1/2 : ℝ) * |C.x - A.x| * |C.y - B.y| = 10 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_abc_is_ten_l3399_339940


namespace NUMINAMATH_CALUDE_total_items_sold_l3399_339957

/-- The total revenue from all items sold -/
def total_revenue : ℝ := 2550

/-- The average price of a pair of ping pong rackets -/
def ping_pong_price : ℝ := 9.8

/-- The average price of a tennis racquet -/
def tennis_price : ℝ := 35

/-- The average price of a badminton racket -/
def badminton_price : ℝ := 15

/-- The number of each type of equipment sold -/
def items_per_type : ℕ := 42

theorem total_items_sold :
  3 * items_per_type = 126 ∧
  (ping_pong_price + tennis_price + badminton_price) * items_per_type = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_total_items_sold_l3399_339957


namespace NUMINAMATH_CALUDE_inequality_proof_l3399_339972

theorem inequality_proof (a b : ℝ) (h : a > b) : 2 - a < 2 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3399_339972


namespace NUMINAMATH_CALUDE_windows_preference_l3399_339918

theorem windows_preference (total_students : ℕ) (mac_preference : ℕ) (no_preference : ℕ) 
  (h1 : total_students = 210)
  (h2 : mac_preference = 60)
  (h3 : no_preference = 90) :
  total_students - (mac_preference + mac_preference / 3 + no_preference) = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_l3399_339918


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3399_339983

theorem sum_of_three_numbers : 4.75 + 0.303 + 0.432 = 5.485 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3399_339983


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_thirds_l3399_339944

theorem no_solution_implies_a_leq_two_thirds (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| < 4*x - 1 ∧ x < a)) → a ≤ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_thirds_l3399_339944


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3399_339937

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time_years simple_interest : ℝ) :
  principal = 10000 →
  time_years = 1 →
  simple_interest = 400 →
  (simple_interest / (principal * time_years)) * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3399_339937


namespace NUMINAMATH_CALUDE_length_BI_approx_l3399_339963

/-- Triangle ABC with given side lengths --/
structure Triangle where
  ab : ℝ
  ac : ℝ
  bc : ℝ

/-- The incenter of a triangle --/
def Incenter (t : Triangle) : Point := sorry

/-- The distance between two points --/
def distance (p q : Point) : ℝ := sorry

/-- The given triangle --/
def triangle_ABC : Triangle := { ab := 31, ac := 29, bc := 30 }

/-- Theorem: The length of BI in the given triangle is approximately 17.22 --/
theorem length_BI_approx (ε : ℝ) (h : ε > 0) : 
  ∃ (B I : Point), I = Incenter triangle_ABC ∧ 
    |distance B I - 17.22| < ε := by sorry

end NUMINAMATH_CALUDE_length_BI_approx_l3399_339963


namespace NUMINAMATH_CALUDE_point_on_linear_function_l3399_339929

/-- Given that point P(a, b) is on the graph of y = -2x + 3, prove that 2a + b - 2 = 1 -/
theorem point_on_linear_function (a b : ℝ) (h : b = -2 * a + 3) : 2 * a + b - 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_function_l3399_339929


namespace NUMINAMATH_CALUDE_factorization_equality_l3399_339982

theorem factorization_equality (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3399_339982


namespace NUMINAMATH_CALUDE_count_positive_rationals_l3399_339962

def numbers : List ℚ := [-2023, 1/100, 3/2, 0, 1/5]

theorem count_positive_rationals : 
  (numbers.filter (λ x => x > 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_positive_rationals_l3399_339962


namespace NUMINAMATH_CALUDE_problem_solution_l3399_339985

theorem problem_solution (x y : ℝ) 
  (h1 : 2*x + 3*y = 9) 
  (h2 : x*y = -12) : 
  4*x^2 + 9*y^2 = 225 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3399_339985


namespace NUMINAMATH_CALUDE_kaleb_summer_earnings_l3399_339913

/-- Kaleb's lawn mowing business earnings --/
theorem kaleb_summer_earnings 
  (spring_earnings : ℕ) 
  (supplies_cost : ℕ) 
  (total_amount : ℕ) 
  (h1 : spring_earnings = 4)
  (h2 : supplies_cost = 4)
  (h3 : total_amount = 50)
  : ℕ := by
  sorry

#check kaleb_summer_earnings

end NUMINAMATH_CALUDE_kaleb_summer_earnings_l3399_339913


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l3399_339948

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define two different planes
variable (α β : Plane)
variable (h_diff : α ≠ β)

-- Define a line m in plane α
variable (m : Line)
variable (h_m_in_α : line_in_plane m α)

-- Theorem statement
theorem perp_planes_necessary_not_sufficient :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l3399_339948


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l3399_339968

theorem units_digit_of_sum_of_cubes : ∃ n : ℕ, n < 10 ∧ (41^3 + 23^3) % 10 = n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l3399_339968


namespace NUMINAMATH_CALUDE_strawberry_division_l3399_339984

def strawberry_problem (brother_baskets : ℕ) (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) (parents_difference : ℕ) (family_size : ℕ) : Prop :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := kimberly_multiplier * brother_strawberries
  let parents_strawberries := kimberly_strawberries - parents_difference
  let total_strawberries := kimberly_strawberries + brother_strawberries + parents_strawberries
  (total_strawberries / family_size = 168)

theorem strawberry_division :
  strawberry_problem 3 15 8 93 4 :=
by
  sorry

#check strawberry_division

end NUMINAMATH_CALUDE_strawberry_division_l3399_339984


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l3399_339942

theorem least_positive_integer_for_multiple_of_five : 
  ∀ n : ℕ, n > 0 → (725 + n) % 5 = 0 → n ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l3399_339942


namespace NUMINAMATH_CALUDE_no_valid_right_triangle_with_prime_angles_l3399_339970

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_valid_right_triangle_with_prime_angles :
  ¬ ∃ (x : ℕ), 
    x > 0 ∧ 
    3 * x < 90 ∧ 
    x + 3 * x = 90 ∧ 
    is_prime x ∧ 
    is_prime (3 * x) :=
sorry

end NUMINAMATH_CALUDE_no_valid_right_triangle_with_prime_angles_l3399_339970


namespace NUMINAMATH_CALUDE_function_divisibility_property_l3399_339998

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem function_divisibility_property 
  (f : ℤ → ℕ) 
  (h : ∀ (m n : ℤ), is_divisible (f (m - n)) (f m - f n)) :
  ∀ (m n : ℤ), is_divisible (f m) (f n) → is_divisible (f m) (f n) :=
sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l3399_339998


namespace NUMINAMATH_CALUDE_symmetric_points_ratio_l3399_339925

/-- Given two points A and B that are symmetric about a line ax + y - b = 0,
    prove that a/b = 1/3 -/
theorem symmetric_points_ratio (a b : ℝ) : 
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (3, 5)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (a * midpoint.1 + midpoint.2 - b = 0) →  -- midpoint lies on the line
  ((B.2 - A.2) / (B.1 - A.1) * (-a) = -1) →  -- AB is perpendicular to the line
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_ratio_l3399_339925


namespace NUMINAMATH_CALUDE_mittens_per_box_l3399_339967

/-- Given the conditions of Chloe's winter clothing boxes, prove the number of mittens per box -/
theorem mittens_per_box 
  (num_boxes : ℕ) 
  (scarves_per_box : ℕ) 
  (total_pieces : ℕ) 
  (h1 : num_boxes = 4) 
  (h2 : scarves_per_box = 2) 
  (h3 : total_pieces = 32) : 
  (total_pieces - num_boxes * scarves_per_box) / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_mittens_per_box_l3399_339967
