import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_nine_l553_55361

/-- The area enclosed by y=2x and y=4-2x^2 is 9 -/
theorem enclosed_area_is_nine : 
  ∃ (A : ℝ), A = 9 ∧ 
  A = ∫ x in Set.Icc (-2 : ℝ) (1 : ℝ), (4 - 2*x^2 - 2*x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_nine_l553_55361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_changs_garden_sour_apple_price_l553_55373

/-- Proves that the price of a sour apple is $0.1 given the conditions of Chang's Garden --/
theorem changs_garden_sour_apple_price 
  (total_apples : ℕ) 
  (sweet_percentage : ℚ)
  (sweet_price : ℚ)
  (total_earnings : ℚ) :
  total_apples = 100 →
  sweet_percentage = 3/4 →
  sweet_price = 1/2 →
  total_earnings = 40 →
  (total_earnings - (sweet_percentage * (total_apples : ℚ) * sweet_price)) / ((1 - sweet_percentage) * (total_apples : ℚ)) = 1/10 := by
  sorry

#check changs_garden_sour_apple_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_changs_garden_sour_apple_price_l553_55373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_distance_l553_55339

noncomputable section

/-- Race parameters --/
def total_distance : ℝ := 14
def uphill_distance : ℝ := 8
def downhill_distance : ℝ := 6
def john_head_start : ℝ := 0.2  -- 12 minutes in hours
def john_uphill_speed : ℝ := 10
def john_downhill_speed : ℝ := 15
def jane_uphill_speed : ℝ := 12
def jane_downhill_speed : ℝ := 18

/-- Time for John to reach the top of the hill --/
noncomputable def john_uphill_time : ℝ := uphill_distance / john_uphill_speed

/-- Time for Jane to reach the top of the hill --/
noncomputable def jane_uphill_time : ℝ := uphill_distance / jane_uphill_speed

/-- John's position as a function of time --/
noncomputable def john_position (t : ℝ) : ℝ :=
  if t ≤ john_uphill_time
  then john_uphill_speed * t
  else uphill_distance - john_downhill_speed * (t - john_uphill_time)

/-- Jane's position as a function of time --/
noncomputable def jane_position (t : ℝ) : ℝ :=
  jane_uphill_speed * (t - john_head_start)

/-- The theorem to prove --/
theorem runners_meet_distance :
  ∃ t : ℝ, t > john_head_start ∧ 
    john_position t = jane_position t ∧ 
    uphill_distance - john_position t = 24 / 27 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_distance_l553_55339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l553_55359

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 3*x + 4) + 8/9

-- Define the composition of floor and f
noncomputable def floor_f (x : ℝ) : ℤ := floor (f x)

-- Theorem statement
theorem range_of_floor_f :
  Set.range floor_f = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l553_55359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l553_55306

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 3*x else ((-x)^2 - 3*(-x))

-- State the theorem
theorem f_inequality : f (Real.tan (70 * π / 180)) > f 1.4 ∧ f 1.4 > f (-1.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l553_55306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l553_55332

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 9^x - 2 * 3^x + 4

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 2

-- Theorem statement
theorem f_extrema :
  ∃ (min max : ℝ), min = 3 ∧ max = 67 ∧
  (∀ x ∈ I, f x ≥ min) ∧
  (∃ x₁ ∈ I, f x₁ = min) ∧
  (∀ x ∈ I, f x ≤ max) ∧
  (∃ x₂ ∈ I, f x₂ = max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l553_55332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_approx_l553_55318

noncomputable def article1_cost : ℝ := 100
noncomputable def article1_gain_percent : ℝ := 15

noncomputable def article2_cost : ℝ := 150
noncomputable def article2_gain_percent : ℝ := 20

noncomputable def article3_cost : ℝ := 200
noncomputable def article3_loss_percent : ℝ := 10

noncomputable def article4_cost : ℝ := 250
noncomputable def article4_gain_percent : ℝ := 12

noncomputable def total_cost : ℝ := article1_cost + article2_cost + article3_cost + article4_cost

noncomputable def selling_price (cost : ℝ) (percent : ℝ) : ℝ := cost * (1 + percent / 100)

noncomputable def total_selling_price : ℝ :=
  selling_price article1_cost article1_gain_percent +
  selling_price article2_cost article2_gain_percent +
  selling_price article3_cost (-article3_loss_percent) +
  selling_price article4_cost article4_gain_percent

noncomputable def overall_gain_percent : ℝ := (total_selling_price - total_cost) / total_cost * 100

theorem overall_gain_percentage_approx :
  abs (overall_gain_percent - 7.86) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_approx_l553_55318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_two_sectors_l553_55352

-- Define the radius of the circle
def radius : ℝ := 15

-- Define the angle of each sector in radians
noncomputable def sector_angle : ℝ := Real.pi / 4

-- Define the number of sectors
def num_sectors : ℕ := 2

-- Statement to prove
theorem area_of_two_sectors :
  (num_sectors : ℝ) * (1 / 2 * radius^2 * sector_angle) = 225 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_two_sectors_l553_55352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l553_55360

noncomputable section

theorem triangle_tangent_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  c / b + b / c = 5 * Real.cos A / 2 →
  Real.tan A / Real.tan B + Real.tan A / Real.tan C = 1 / 2 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l553_55360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angle_existence_l553_55380

open Real Set

theorem complementary_angle_existence :
  (∃ α, α ∈ Ioo (π/3) (π/2) ∧
    ∃ β₁ β₂, β₁ ∈ Icc 0 (2*π) ∧ β₂ ∈ Icc 0 (2*π) ∧
    β₁ ≠ β₂ ∧ cos (α + β₁) = cos α + cos β₁ ∧ cos (α + β₂) = cos α + cos β₂) ∧
  (∃ α, α ∈ Ioo 0 (π/3) ∧
    ∃ β, β ∈ Icc 0 (2*π) ∧ cos (α + β) = cos α + cos β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angle_existence_l553_55380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_always_positive_derivative_l553_55327

open Real

-- Define the functions
noncomputable def f (x : ℝ) := sin x
noncomputable def g (x : ℝ) := cos x
def h (x : ℝ) := x^2
noncomputable def k (x : ℝ) := exp x

-- State the theorem
theorem exp_always_positive_derivative :
  (∀ x : ℝ, ∃ y : ℝ, y ≤ 0 ∧ (deriv f x = y ∨ deriv g x = y ∨ deriv h x = y)) ∧
  (∀ x : ℝ, deriv k x > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_always_positive_derivative_l553_55327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_locker_opened_l553_55366

/-- Represents the state of a locker (open or closed) -/
inductive LockerState where
  | Open
  | Closed
deriving BEq, Repr

/-- Simulates the process of opening lockers -/
def openLockers (n : Nat) : Nat :=
  let rec loop (lockers : List LockerState) (currentPass : Nat) (remaining : Nat) : Nat :=
    if remaining = 0 then
      n - lockers.length
    else
      let newLockers := lockers.enum.map fun (i, state) =>
        if state == LockerState.Closed && (i + 1) % (2 ^ currentPass + 1) == 0 then
          LockerState.Open
        else
          state
      loop newLockers (currentPass + 1) (lockers.filter (· == LockerState.Closed)).length
  loop (List.replicate n LockerState.Closed) 0 n
termination_by _ => remaining

/-- The main theorem stating that the last locker opened is number 494 -/
theorem last_locker_opened (n : Nat) (h : n = 500) : openLockers n = 494 := by
  sorry

#eval openLockers 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_locker_opened_l553_55366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rhombus_from_quartic_roots_l553_55394

open Complex

/-- The quartic equation whose roots form the vertices of the rhombus -/
noncomputable def quartic (z : ℂ) : ℂ := z^4 + 4*I*z^3 + (5 + 5*I)*z^2 + (10 - I)*z + (1 - 4*I)

/-- The area of the rhombus formed by the roots of the quartic equation -/
noncomputable def rhombus_area : ℝ := 2 * Real.sqrt 10

/-- Function to calculate the area of a quadrilateral given its vertices -/
noncomputable def area_of_quadrilateral (w x y z : ℂ) : ℝ := sorry

theorem area_of_rhombus_from_quartic_roots :
  ∃ (w x y z : ℂ), quartic w = 0 ∧ quartic x = 0 ∧ quartic y = 0 ∧ quartic z = 0 ∧
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    area_of_quadrilateral w x y z = rhombus_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rhombus_from_quartic_roots_l553_55394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_two_l553_55369

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is inside a circle -/
def isInside (c : Circle) (p : Point) : Prop :=
  distance c.center p < c.radius

/-- Checks if a point is outside a circle -/
def isOutside (c : Circle) (p : Point) : Prop :=
  distance c.center p > c.radius

/-- Checks if a point is on a circle -/
def isOn (c : Circle) (p : Point) : Prop :=
  distance c.center p = c.radius

/-- Checks if a circle is tangent to the y-axis at a given point -/
def isTangentToYAxis (c : Circle) (p : Point) : Prop :=
  p.x = 0 ∧ distance c.center p = c.radius

/-- Checks if a triangle is isosceles -/
def isIsosceles (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = distance p1 p3

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem circle_radius_is_two (c : Circle) 
  (A B C D E : Point)
  (h1 : A = Point.mk (-2) (-3))
  (h2 : B = Point.mk (-2) 3)
  (h3 : C = Point.mk 6 (-3))
  (h4 : D = Point.mk 6 3)
  (h5 : E = Point.mk 0 (-3))
  (h6 : c.center = A)
  (h7 : isInside c B)
  (h8 : isOutside c C)
  (h9 : isOn c D)
  (h10 : isIsosceles A C D)
  (h11 : isTangentToYAxis c E)
  (h12 : ∃ n : ℕ, c.radius = n)
  (h13 : ∃ m : ℕ, triangleArea A C D = m) :
  c.radius = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_two_l553_55369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_zero_minimum_a_value_l553_55358

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a * Real.exp x - x) * Real.exp x

-- Part 1
theorem monotonicity_when_a_zero :
  let f₀ := f 0
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -1 → f₀ x₁ < f₀ x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ → f₀ x₁ > f₀ x₂) :=
by sorry

-- Part 2
theorem minimum_a_value (a : ℝ) :
  (∀ x, f a x + 1/a ≤ 0) → a ≥ -Real.exp 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_zero_minimum_a_value_l553_55358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_decreasing_cosine_minus_sine_l553_55313

theorem max_a_for_decreasing_cosine_minus_sine (a : ℝ) : 
  (∀ x ∈ Set.Icc (-a) a, ∀ y ∈ Set.Icc (-a) a, x < y → (Real.cos x - Real.sin x) > (Real.cos y - Real.sin y)) → 
  a ≤ π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_decreasing_cosine_minus_sine_l553_55313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l553_55377

theorem calculate_expression : (-1/3 : ℝ)⁻¹ - Real.sqrt 8 - (5 - Real.pi)^0 + 4 * Real.cos (45 * π / 180) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l553_55377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l553_55383

/-- Represents the rate of interest (in percentage) for a loan -/
noncomputable def rate_of_interest (principal : ℝ) (interest : ℝ) : ℝ :=
  Real.sqrt (100 * interest / principal)

/-- Theorem: Given the conditions, the rate of interest is approximately 6% -/
theorem interest_rate_approximation (principal interest : ℝ) 
  (h1 : principal = 1800)
  (h2 : interest = 632) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |rate_of_interest principal interest - 6| < ε := by
  sorry

#check interest_rate_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l553_55383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_49_minus_cbrt_27_equals_4_l553_55317

theorem sqrt_49_minus_cbrt_27_equals_4 : Real.sqrt 49 - (27 : Real).rpow (1/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_49_minus_cbrt_27_equals_4_l553_55317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_all_x_nonpositive_range_of_a_for_some_x_nonpositive_l553_55381

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

/-- Theorem 1: Range of a when f(x) ≤ 0 for all x ∈ ℝ -/
theorem range_of_a_for_all_x_nonpositive (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) ↔ 0 < a ∧ a ≤ 1 := by sorry

/-- Theorem 2: Range of a when a ≥ 2 and there exists x ∈ ℝ such that f(x) ≤ 0 -/
theorem range_of_a_for_some_x_nonpositive (a : ℝ) (h : a ≥ 2) :
  (∃ x : ℝ, f a x ≤ 0) ↔ 2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_all_x_nonpositive_range_of_a_for_some_x_nonpositive_l553_55381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_consumption_l553_55350

/-- The percentage of fiber that koalas absorb from their diet -/
noncomputable def koala_absorption_rate : ℝ := 0.40

/-- The amount of fiber (in ounces) that the koala absorbed in one day -/
noncomputable def absorbed_fiber : ℝ := 8

/-- The total amount of fiber (in ounces) that the koala consumed in one day -/
noncomputable def total_fiber_consumed : ℝ := absorbed_fiber / koala_absorption_rate

theorem koala_fiber_consumption :
  total_fiber_consumed = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_consumption_l553_55350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_a_p_l553_55384

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | 2 => 2
  | 3 => 3
  | n + 4 => a (n + 2) + a (n + 1)

/-- Theorem: For any prime p, p divides a_p -/
theorem prime_divides_a_p (p : ℕ) (hp : Nat.Prime p) : (p : ℤ) ∣ a p := by
  sorry

#eval a 5  -- You can add this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_a_p_l553_55384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisibility_l553_55375

def f (m n : ℕ) : ℤ :=
  (m : ℤ)^(3^(4*n)+6) - (m : ℤ)^(3^(4*n)+4) - (m : ℤ)^5 + (m : ℤ)^3

theorem f_divisibility (n : ℕ) : 
  (∀ m : ℕ+, 1992 ∣ f m.val n) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisibility_l553_55375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_college_entrance_exam_score_college_entrance_exam_score_proof_l553_55336

theorem college_entrance_exam_score 
  (total_questions : ℕ) 
  (questions_answered : ℕ) 
  (correct_answers : ℕ) 
  (correct_points : ℚ) 
  (incorrect_penalty : ℚ) 
  (h1 : total_questions = 85)
  (h2 : questions_answered = 82)
  (h3 : correct_answers = 70)
  (h4 : correct_points = 1)
  (h5 : incorrect_penalty = 1/4)
  : ℚ := by
  let incorrect_answers := questions_answered - correct_answers
  let unanswered := total_questions - questions_answered
  let raw_score := correct_answers * correct_points - incorrect_answers * incorrect_penalty
  exact 67

theorem college_entrance_exam_score_proof 
  (total_questions : ℕ) 
  (questions_answered : ℕ) 
  (correct_answers : ℕ) 
  (correct_points : ℚ) 
  (incorrect_penalty : ℚ) 
  (h1 : total_questions = 85)
  (h2 : questions_answered = 82)
  (h3 : correct_answers = 70)
  (h4 : correct_points = 1)
  (h5 : incorrect_penalty = 1/4)
  : college_entrance_exam_score total_questions questions_answered correct_answers correct_points incorrect_penalty h1 h2 h3 h4 h5 = 67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_college_entrance_exam_score_college_entrance_exam_score_proof_l553_55336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reachable_area_theorem_l553_55307

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex -/
noncomputable def dogReachableArea (sideLength : ℝ) (ropeLength : ℝ) : ℝ :=
  3 * Real.pi

/-- Theorem stating the area the dog can reach outside the hexagonal doghouse -/
theorem dog_reachable_area_theorem (sideLength ropeLength : ℝ) 
    (h1 : sideLength = 1.5)
    (h2 : ropeLength = 3) :
    dogReachableArea sideLength ropeLength = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reachable_area_theorem_l553_55307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l553_55340

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

-- Main theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- f is odd
  (∀ x, f a (-x) = -f a x) ∧
  -- f is decreasing when 0 < a < 1
  ((0 < a ∧ a < 1) → (∀ x y, x < y → f a x > f a y)) ∧
  -- f is increasing when a > 1
  (a > 1 → (∀ x y, x < y → f a x < f a y)) ∧
  -- Given f(m-1) + f(m) < 0 for x ∈ (-1, 1)
  (∀ m, (∀ x, -1 < x ∧ x < 1 → f a (m-1) + f a m < 0) →
    -- When 0 < a < 1, 1/2 < m < 1
    ((0 < a ∧ a < 1) → (1/2 < m ∧ m < 1)) ∧
    -- When a > 1, 0 < m < 1/2
    (a > 1 → (0 < m ∧ m < 1/2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l553_55340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l553_55348

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Predicate for an isosceles triangle formed by the foci and a point on the ellipse -/
def isIsoscelesTriangle (e : Ellipse) (p : PointOnEllipse e) : Prop :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  (p.x + c)^2 + p.y^2 = (p.x - c)^2 + p.y^2 ∨
  2 * ((p.x + c)^2 + p.y^2) = e.a^2 ∨
  2 * ((p.x - c)^2 + p.y^2) = e.a^2

/-- The main theorem -/
theorem ellipse_eccentricity_range (e : Ellipse)
  (h : ∃! (s : Finset (PointOnEllipse e)), s.card = 6 ∧ ∀ p ∈ s, isIsoscelesTriangle e p) :
  1/3 < e.eccentricity ∧ e.eccentricity < 1 ∧ e.eccentricity ≠ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l553_55348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_half_integer_point_prove_exists_half_integer_point_l553_55378

/-- A point in the xy-plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Represents a sequence of 1993 points in the xy-plane -/
def PointSequence : Type := Fin 1993 → IntPoint

/-- Checks if a point is on the line segment between two other points -/
def isOnSegment (p q r : IntPoint) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧
    q.x = p.x + ⌊t * (r.x - p.x)⌋ ∧
    q.y = p.y + ⌊t * (r.y - p.y)⌋

/-- Represents a point with rational coordinates -/
structure RatPoint where
  x : ℚ
  y : ℚ

/-- The main theorem -/
theorem exists_half_integer_point (points : PointSequence) : Prop :=
  (∀ i : Fin 1993, points i ≠ points ((i + 1) % 1993)) →
  (∀ i : Fin 1993, ∀ q : IntPoint, 
    isOnSegment (points i) q (points ((i + 1) % 1993)) → 
    q = points i ∨ q = points ((i + 1) % 1993)) →
  ∃ i : Fin 1993, ∃ q : RatPoint,
    (∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧
      q.x = (points i).x + t * ((points ((i + 1) % 1993)).x - (points i).x) ∧
      q.y = (points i).y + t * ((points ((i + 1) % 1993)).y - (points i).y)) ∧
    Odd (2 * q.x).num ∧ Odd (2 * q.y).num

/-- Proof of the theorem -/
theorem prove_exists_half_integer_point (points : PointSequence) :
  exists_half_integer_point points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_half_integer_point_prove_exists_half_integer_point_l553_55378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_underdetermined_system_no_unique_sale_prices_l553_55311

/-- Represents the cost price of an item -/
structure CostPrice where
  value : ℚ
  pos : value > 0

/-- Represents the sale price of an item -/
structure SalePrice where
  value : ℚ
  pos : value > 0

/-- Calculates the profit percentage given cost and sale prices -/
noncomputable def profitPercentage (cost : CostPrice) (sale : SalePrice) : ℚ :=
  (sale.value - cost.value) / cost.value * 100

/-- Theorem stating that the given system is underdetermined -/
theorem underdetermined_system
  (P1 P2 P3 : CostPrice)
  (h1 : 872 - P1.value = P2.value - 448)
  (h2 : 650 - P3.value = P1.value - 550) :
  ∃ (Q1 Q2 Q3 : CostPrice),
    Q1 ≠ P1 ∧ Q2 ≠ P2 ∧ Q3 ≠ P3 ∧
    872 - Q1.value = Q2.value - 448 ∧
    650 - Q3.value = Q1.value - 550 :=
by sorry

/-- Theorem stating that unique sale prices for 50% profit cannot be determined -/
theorem no_unique_sale_prices
  (P1 P2 P3 : CostPrice)
  (h1 : 872 - P1.value = P2.value - 448)
  (h2 : 650 - P3.value = P1.value - 550) :
  ¬∃! (S1 S2 S3 : SalePrice),
    profitPercentage P1 S1 = 50 ∧
    profitPercentage P2 S2 = 50 ∧
    profitPercentage P3 S3 = 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_underdetermined_system_no_unique_sale_prices_l553_55311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l553_55310

/-- The distance from the origin to a line ax + by + c = 0 is |c| / √(a² + b²) -/
noncomputable def distanceFromOriginToLine (a b c : ℝ) : ℝ :=
  |c| / Real.sqrt (a^2 + b^2)

/-- The line equation x - 2y + 3 = 0 -/
def lineEquation (x y : ℝ) : Prop :=
  x - 2*y + 3 = 0

theorem distance_to_line :
  distanceFromOriginToLine 1 (-2) 3 = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l553_55310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l553_55379

-- Define the Δ operation
noncomputable def delta (c d : ℝ) : ℝ := (c + d) / (1 + c * d)

-- Theorem statement
theorem delta_calculation :
  delta (delta 1 2 + 1) 3 = 5 / 7 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l553_55379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_subset_existence_l553_55357

def T : Set (ℤ × ℤ × ℤ) := Set.univ

def neighbors (p q : ℤ × ℤ × ℤ) : Prop :=
  let (x, y, z) := p
  let (u, v, w) := q
  |x - u| + |y - v| + |z - w| = 1

theorem lattice_subset_existence :
  ∃ (S : Set (ℤ × ℤ × ℤ)), S ⊆ T ∧
    ∀ p ∈ T, ∃! q, q ∈ ({p} ∪ {n | neighbors p n}) ∧ q ∈ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_subset_existence_l553_55357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cost_is_7000_l553_55399

noncomputable def total_machines : ℕ := 5
noncomputable def faulty_machines : ℕ := 2
noncomputable def cost_per_test : ℝ := 2000

noncomputable def probability_four_tests : ℝ := (faulty_machines : ℝ) / total_machines * 1 / (total_machines - 1)
noncomputable def probability_six_tests : ℝ := 
  (faulty_machines : ℝ) / total_machines * (total_machines - faulty_machines : ℝ) / (total_machines - 1) * 1 / (total_machines - 2) +
  (total_machines - faulty_machines : ℝ) / total_machines * (faulty_machines : ℝ) / (total_machines - 1) * 1 / (total_machines - 2) +
  (total_machines - faulty_machines : ℝ) / total_machines * (faulty_machines : ℝ) / (total_machines - 1) * 1 / (total_machines - 2)
noncomputable def probability_eight_tests : ℝ := 1 - probability_four_tests - probability_six_tests

theorem expected_cost_is_7000 :
  probability_four_tests * (2 * cost_per_test) +
  probability_six_tests * (3 * cost_per_test) +
  probability_eight_tests * (4 * cost_per_test) = 7000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cost_is_7000_l553_55399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l553_55342

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (2*x - 5)*(x - 3)/x

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x ≤ 5/2} ∪ {x | x ≥ 3}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ≠ 0 → (g x ≥ 0 ↔ x ∈ solution_set) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l553_55342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l553_55337

theorem vector_sum_magnitude (a b : Fin 2 → ℝ) :
  ‖a‖ = 3 →
  ‖b‖ = 4 →
  a • b = -6 →
  ‖a + 2 • b‖ = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l553_55337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_tenth_finger_l553_55305

-- Define the function g based on the graph
def g : ℕ → ℕ
| 0 => 0
| 1 => 8
| 2 => 1
| 3 => 6
| 4 => 3
| 5 => 4
| 6 => 4
| 7 => 3
| 8 => 2
| 9 => 1
| _ => 0  -- For any other input, return 0

-- Define the repeated application of g
def repeat_g : ℕ → ℕ → ℕ
| 0, x => x
| (n + 1), x => repeat_g n (g x)

-- Theorem statement
theorem mindy_tenth_finger :
  repeat_g 9 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_tenth_finger_l553_55305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l553_55324

/-- Represents a pyramid with a square base drawn using the oblique method -/
structure ObliquePyramid where
  height : ℝ
  base_side : ℝ

/-- Calculates the volume of an oblique pyramid -/
noncomputable def volume (p : ObliquePyramid) : ℝ :=
  (1 / 3) * p.height * (p.base_side * (2 * Real.sqrt 2))

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  let p := ObliquePyramid.mk 3 1
  volume p = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l553_55324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_avg_cost_at_eight_floors_l553_55323

/-- Represents the housing project parameters -/
structure HousingProject where
  landCost : ℚ             -- Land purchase cost in yuan
  numBuildings : ℕ         -- Number of buildings
  floorArea : ℚ            -- Construction area per floor in m²
  kValue : ℚ               -- Constant k in the construction cost formula
  initialFloors : ℕ        -- Initial number of floors used to calculate k
  initialAvgCost : ℚ       -- Initial average comprehensive cost per m²

/-- Calculates the average comprehensive cost per square meter -/
def avgComprehensiveCost (project : HousingProject) (floors : ℕ) : ℚ :=
  let totalArea := project.numBuildings * project.floorArea * floors
  let constructionCost := (floors * (project.kValue * (floors + 1) / 2 + 800)) * project.numBuildings * project.floorArea
  (project.landCost + constructionCost) / totalArea

/-- States that the minimum average comprehensive cost is achieved at 8 floors -/
theorem min_avg_cost_at_eight_floors (project : HousingProject) :
  project.landCost = 16000000 ∧
  project.numBuildings = 10 ∧
  project.floorArea = 1000 ∧
  project.initialFloors = 5 ∧
  project.initialAvgCost = 1270 ∧
  project.kValue = 50 →
  (∀ n : ℕ, n > 0 → avgComprehensiveCost project 8 ≤ avgComprehensiveCost project n) ∧
  avgComprehensiveCost project 8 = 1225 := by
  sorry

#check min_avg_cost_at_eight_floors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_avg_cost_at_eight_floors_l553_55323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_formula_l553_55349

theorem sin_plus_cos_formula (θ : Real) (b : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2)  -- θ is an acute angle
  (h2 : Real.cos (2 * θ) = b)    -- cos 2θ = b
  : Real.sin θ + Real.cos θ = Real.sqrt (1 + Real.sqrt ((1 + b) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_formula_l553_55349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l553_55370

theorem trigonometric_inequalities (α₁ α₂ α₃ : ℝ) :
  (α₁ ∈ Set.Icc 0 π ∧ α₂ ∈ Set.Icc 0 π ∧ α₃ ∈ Set.Icc 0 π →
    Real.sin α₁ + Real.sin α₂ + Real.sin α₃ ≤ 3 * Real.sin ((α₁ + α₂ + α₃) / 3)) ∧
  (α₁ ∈ Set.Icc (-π/2) (π/2) ∧ α₂ ∈ Set.Icc (-π/2) (π/2) ∧ α₃ ∈ Set.Icc (-π/2) (π/2) →
    Real.cos α₁ + Real.cos α₂ + Real.cos α₃ ≤ 3 * Real.cos ((α₁ + α₂ + α₃) / 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l553_55370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_opposite_vertices_differ_by_one_l553_55387

-- Define the cube structure
structure Cube where
  vertices : Fin 8 → ℤ
  adjacent : Fin 8 → Fin 8 → Prop

-- Define the property of adjacent vertices differing by at most 1
def adjacent_differ_by_one (c : Cube) : Prop :=
  ∀ (i j : Fin 8), c.adjacent i j → |c.vertices i - c.vertices j| ≤ 1

-- Define diametrically opposite vertices
def diametrically_opposite : Fin 8 → Fin 8 → Prop
| 0, 6 | 6, 0 | 1, 7 | 7, 1 | 2, 4 | 4, 2 | 3, 5 | 5, 3 => True
| _, _ => False

-- Theorem statement
theorem cube_opposite_vertices_differ_by_one (c : Cube) 
  (h : adjacent_differ_by_one c) : 
  ∃ (i j : Fin 8), diametrically_opposite i j ∧ |c.vertices i - c.vertices j| ≤ 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_opposite_vertices_differ_by_one_l553_55387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_of_Q_l553_55320

-- Define the cubic root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define our special number
noncomputable def specialNumber : ℝ := cubeRoot 5 + cubeRoot 45

-- Define the polynomial of least degree with rational coefficients having specialNumber as a root
noncomputable def Q : Polynomial ℚ := Polynomial.X^3 - 320

-- State the theorem
theorem product_of_roots_of_Q : (Q.roots.prod) = -320 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_of_Q_l553_55320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_negative_two_l553_55338

noncomputable section

-- Define the circle's center
def center : ℝ × ℝ := (2, 1)

-- Define the point of tangency
def tangentPoint : ℝ × ℝ := (6, 3)

-- Define the slope of the radius
def radiusSlope : ℝ := (tangentPoint.2 - center.2) / (tangentPoint.1 - center.1)

-- Define the slope of the tangent line
def tangentSlope : ℝ := -1 / radiusSlope

-- Theorem statement
theorem tangent_slope_is_negative_two :
  tangentSlope = -2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_negative_two_l553_55338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_consumption_proof_l553_55346

/-- The original amount of oil in kilograms -/
noncomputable def original_amount : ℝ := 75

/-- The amount of oil remaining after the first consumption -/
noncomputable def after_first_consumption : ℝ := original_amount / 2

/-- The amount of oil remaining after the second consumption -/
noncomputable def after_second_consumption : ℝ := after_first_consumption * (4/5)

theorem oil_consumption_proof :
  after_second_consumption = 30 ∧ original_amount = 75 := by
  sorry

#check oil_consumption_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_consumption_proof_l553_55346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_has_value_l553_55376

/-- Represents the various aspects of mathematics' value --/
inductive MathValue
  | foundational
  | widelyApplicable
  | logicalThinking
  | culturalLiteracy
  | scientificSpirit

/-- Theorem stating that mathematics has various valuable aspects --/
theorem math_has_value : ∃ (aspects : List MathValue), aspects.length > 1 := by
  let aspects := [
    MathValue.foundational,
    MathValue.widelyApplicable,
    MathValue.logicalThinking,
    MathValue.culturalLiteracy,
    MathValue.scientificSpirit
  ]
  exists aspects
  simp
  -- The proof is trivial as we've explicitly listed more than one aspect
  sorry

#check math_has_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_has_value_l553_55376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_theorem_l553_55374

noncomputable def tangent_circle_circumference (arc_length : ℝ) : ℝ :=
  let r := arc_length * 3 / Real.pi
  let r₂ := (180 * Real.pi - 2025) / (4 * Real.pi^2)
  2 * Real.pi * r₂

theorem tangent_circle_theorem (arc_length : ℝ) (h : arc_length = 15) :
  abs (tangent_circle_circumference arc_length - 93.94) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_theorem_l553_55374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l553_55365

open Real

theorem trigonometric_identity (α β γ : ℝ) : 
  tan α + tan β + tan γ - sin (α + β + γ) / (cos α * cos β * cos γ) = tan α * tan β * tan γ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l553_55365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_smallest_factorable_b_l553_55385

/-- 
Given a positive integer b, this function returns true if x^2 + bx + 2016
can be factored into a product of two binomials with integer coefficients.
-/
def is_factorable (b : ℕ+) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), (x^2 : ℤ) + (b : ℤ)*x + 2016 = (x + r) * (x + s)

/-- 
Theorem stating that 92 is the smallest positive integer b for which
x^2 + bx + 2016 can be factored into a product of two binomials
with integer coefficients.
-/
theorem smallest_factorable_b : 
  (is_factorable 92) ∧ (∀ b : ℕ+, b < 92 → ¬(is_factorable b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_smallest_factorable_b_l553_55385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eight_degrees_l553_55368

theorem cos_eight_degrees (m : ℝ) (h : Real.sin (74 * π / 180) = m) :
  Real.cos (8 * π / 180) = Real.sqrt ((1 + m) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eight_degrees_l553_55368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_l553_55334

theorem gcd_power_minus_one (n m : ℕ) (a : ℕ) (ha : a ≥ 1) :
  Int.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd n m) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_l553_55334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_axis_l553_55330

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)
noncomputable def g (x : ℝ) : ℝ := 2^(1 - x)

theorem symmetry_about_y_axis : ∀ x : ℝ, f (-x) = g x := by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_axis_l553_55330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l553_55392

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 + 1) / (4 * x^2 + 6 * x + 3)

def is_vertical_asymptote (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε

theorem vertical_asymptotes_sum (p q : ℝ) 
  (hp : is_vertical_asymptote f p) 
  (hq : is_vertical_asymptote f q) 
  (h_distinct : p ≠ q) :
  p + q = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l553_55392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l553_55354

theorem smallest_sum_of_a_and_b : ∃ (a b : ℕ), 
  (2^6 * 3^5 : ℕ) = a^b ∧ 
  (∀ (c d : ℕ), (2^6 * 3^5 : ℕ) = c^d → a + b ≤ c + d) ∧
  a + b = 1946 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l553_55354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uba_capital_suv_count_l553_55372

/-- Calculates the number of SUVs purchased by UBA Capital --/
def uba_capital_suvs (total_vehicles : ℕ) (toyota_ratio honda_ratio nissan_ratio : ℕ) 
  (toyota_suv_percent honda_suv_percent nissan_suv_percent : ℚ) : ℕ :=
  let total_ratio := toyota_ratio + honda_ratio + nissan_ratio
  let toyota_count := (total_vehicles * toyota_ratio) / total_ratio
  let honda_count := (total_vehicles * honda_ratio) / total_ratio
  let nissan_count := (total_vehicles * nissan_ratio) / total_ratio
  let toyota_suvs := Int.floor (toyota_count * toyota_suv_percent)
  let honda_suvs := Int.floor (honda_count * honda_suv_percent)
  let nissan_suvs := Int.floor (nissan_count * nissan_suv_percent)
  (toyota_suvs + honda_suvs + nissan_suvs).toNat

/-- Proves that UBA Capital purchased 13 SUVs --/
theorem uba_capital_suv_count : 
  uba_capital_suvs 45 8 4 3 (30/100) (20/100) (40/100) = 13 := by
  -- Unfold the definition and simplify
  unfold uba_capital_suvs
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uba_capital_suv_count_l553_55372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_omega_range_l553_55398

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

-- State the theorem
theorem four_roots_omega_range (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, 0 < x ∧ x < Real.pi ∧ f ω x = -1) :
  7/2 < ω ∧ ω ≤ 25/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_omega_range_l553_55398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_A_length_l553_55367

-- Define the given parameters
noncomputable def speed_A : ℝ := 54 -- km/hr
noncomputable def speed_B : ℝ := 36 -- km/hr
noncomputable def time_to_cross : ℝ := 11 -- seconds
noncomputable def length_B : ℝ := 150 -- meters

-- Define the conversion factor from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

-- Define the function to calculate the length of train A
noncomputable def length_A : ℝ :=
  (speed_A * km_per_hr_to_m_per_s + speed_B * km_per_hr_to_m_per_s) * time_to_cross - length_B

-- State the theorem
theorem train_A_length :
  length_A = 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_A_length_l553_55367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l553_55393

noncomputable def f (x : ℝ) := Real.sin (Real.pi / 2 - 2 * x)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ ε > 0, ∃ p > 0, p ≤ Real.pi ∧ ∀ x, f (x + p) = f x) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l553_55393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_translation_l553_55315

theorem cos_graph_translation (φ : ℝ) : 
  -π ≤ φ ∧ φ ≤ π →
  (∀ x : ℝ, Real.cos (2*(x - π/2) + φ) = Real.sin (2*x + π/3)) →
  φ = 5*π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_translation_l553_55315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l553_55304

theorem square_root_problem (x y : ℝ) 
  (h1 : 2 = Real.sqrt (x - 2)) 
  (h2 : 2^3 = 2*x - y + 1) : 
  Real.sqrt (x^2 - 4*y) = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l553_55304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_5000_l553_55303

/-- Represents the water flow rates and timings of pipes in a tank system -/
structure PipeSystem where
  pipeA_rate : ℚ
  pipeB_rate : ℚ
  pipeC_rate : ℚ
  pipeA_time : ℚ
  pipeB_time : ℚ
  pipeC_time : ℚ
  total_time : ℚ

/-- Calculates the capacity of a tank given a PipeSystem -/
def calculate_tank_capacity (system : PipeSystem) : ℚ :=
  let cycle_time := system.pipeA_time + system.pipeB_time + system.pipeC_time
  let water_per_cycle := 
    system.pipeA_rate * system.pipeA_time +
    system.pipeB_rate * system.pipeB_time -
    system.pipeC_rate * system.pipeC_time
  (system.total_time / cycle_time) * water_per_cycle

/-- The main theorem stating that the tank capacity is 5000 liters -/
theorem tank_capacity_is_5000 (system : PipeSystem)
  (h1 : system.pipeA_rate = 200)
  (h2 : system.pipeB_rate = 50)
  (h3 : system.pipeC_rate = 25)
  (h4 : system.pipeA_time = 1)
  (h5 : system.pipeB_time = 2)
  (h6 : system.pipeC_time = 2)
  (h7 : system.total_time = 100) :
  calculate_tank_capacity system = 5000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_5000_l553_55303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_profit_calculation_l553_55321

/-- Calculates the profit from selling a sack of rice given specific conditions. -/
theorem rice_profit_calculation 
  (weight : ℝ) 
  (cost : ℝ) 
  (price_per_kg : ℝ) 
  (tax_rate : ℝ) 
  (discount_rate : ℝ) 
  (h1 : weight = 50)
  (h2 : cost = 50)
  (h3 : price_per_kg = 1.2)
  (h4 : tax_rate = 0.12)
  (h5 : discount_rate = 0.05) :
  (weight * price_per_kg * (1 - discount_rate) * (1 + tax_rate) - cost) = 13.84 := by
  sorry

#eval (50 : ℝ) * 1.2 * (1 - 0.05) * (1 + 0.12) - 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_profit_calculation_l553_55321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_crew_loading_fraction_l553_55344

theorem day_crew_loading_fraction :
  ∀ (D W : ℚ),
  D > 0 → W > 0 →
  let night_worker_load : ℚ := 3/4 * D;
  let night_worker_count : ℚ := 4/9 * W;
  let day_total : ℚ := D * W;
  let night_total : ℚ := night_worker_load * night_worker_count;
  day_total / (day_total + night_total) = 3/4 :=
by
  intros D W hD hW
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_crew_loading_fraction_l553_55344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l553_55329

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3/2 * x^2 + 2 * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 3 * x + 2

theorem min_value_of_f (a : ℝ) :
  (f' a 1 = 0) →  -- Tangent line at (1, f(1)) is parallel to x-axis
  (∃ x₀ ∈ Set.Ioo 1 3, ∀ x ∈ Set.Ioo 1 3, f a x₀ ≤ f a x) →  -- Minimum exists in (1,3)
  (∃ x₀ ∈ Set.Ioo 1 3, f a x₀ = 5/3 ∧ ∀ x ∈ Set.Ioo 1 3, f a x₀ ≤ f a x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l553_55329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_point_l553_55355

theorem terminal_side_point (θ : Real) (t : Real) : 
  (∃ P : Real × Real, P = (-2, t) ∧ 
   Real.sin θ = t / Real.sqrt (4 + t^2) ∧ 
   Real.cos θ = -2 / Real.sqrt (4 + t^2) ∧
   Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) → 
  t = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_point_l553_55355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_angle_measure_l553_55382

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an Angle type
structure Angle where
  measure : ℝ

-- Define a Triangle type
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define IsIsosceles predicate
def IsIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 ∨
  (t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∨
  (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2 = (t.C.x - t.B.x)^2 + (t.C.y - t.B.y)^2

-- Define AngleMeasure function
def AngleMeasure (a : Angle) : ℝ := a.measure

-- Define IsInside predicate
def IsInside (P : Point) (t : Triangle) : Prop :=
  -- Placeholder definition, actual implementation would be more complex
  true

-- Main theorem
theorem isosceles_triangles_angle_measure :
  ∀ (X Y Z W : Point),
  let t1 := Triangle.mk X Y Z
  let t2 := Triangle.mk X W Z
  IsIsosceles t1 ∧ IsIsosceles t2 ∧
  IsInside W t1 ∧
  AngleMeasure (Angle.mk 60) = 60 ∧
  AngleMeasure (Angle.mk 100) = 100 →
  AngleMeasure (Angle.mk 20) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_angle_measure_l553_55382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_angle_bisector_l553_55314

-- Define a symmetrical figure
structure SymmetricalFigure where
  -- Add any necessary properties for a symmetrical figure
  is_symmetrical : Bool

-- Define an axis of symmetry
structure AxisOfSymmetry where
  -- The axis divides the figure into two mirror-image halves
  divides_into_mirror_halves : Bool

-- Define an angle bisector
structure AngleBisector where
  -- Add any necessary properties for an angle bisector
  is_angle_bisector : Bool

-- Define a function to check if an AxisOfSymmetry is also an AngleBisector
def is_axis_and_bisector (axis : AxisOfSymmetry) (bisector : AngleBisector) : Prop :=
  axis.divides_into_mirror_halves ∧ bisector.is_angle_bisector

-- Theorem statement
theorem axis_of_symmetry_is_angle_bisector 
  (figure : SymmetricalFigure) 
  (axis : AxisOfSymmetry) 
  (bisector : AngleBisector) : 
  figure.is_symmetrical → 
  is_axis_and_bisector axis bisector := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_angle_bisector_l553_55314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l553_55345

theorem triangle_side_range (a b c : ℝ) (A C : ℝ) :
  a = 4 →
  4 < b →
  b < 6 →
  Real.sin (2 * A) = Real.sin C →
  4 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l553_55345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swap_repetitive_implies_drab_l553_55302

/-- A word is a finite sequence of letters from some alphabet. -/
def Word := List Char

/-- A word is repetitive if it is a concatenation of at least two identical subwords. -/
def isRepetitive (w : Word) : Prop :=
  ∃ (subword : Word) (n : Nat), n ≥ 2 ∧ w = List.join (List.replicate n subword)

/-- Swap two adjacent letters in a word. -/
def swapAdjacent (w : Word) (i : Nat) : Word :=
  if i + 1 < w.length then
    w.take i ++ [w.get! (i+1), w.get! i] ++ w.drop (i+2)
  else
    w

/-- A word has the swap-repetitive property if swapping any two adjacent letters makes it repetitive. -/
def hasSwapRepetitiveProperty (w : Word) : Prop :=
  ∀ i, i + 1 < w.length → isRepetitive (swapAdjacent w i)

/-- A word is drab if it consists of repetitions of a single letter. -/
def isDrab (w : Word) : Prop :=
  ∃ (c : Char) (n : Nat), w = List.replicate n c

theorem swap_repetitive_implies_drab (w : Word) :
  hasSwapRepetitiveProperty w → isDrab w :=
by sorry

#check swap_repetitive_implies_drab

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swap_repetitive_implies_drab_l553_55302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_at_pi_over_eight_l553_55326

/-- Given a function f(x) = sin(2x) + a*cos(2x), if |f(x)| ≤ f(π/8) for all x in ℝ, then a = 1. -/
theorem function_max_at_pi_over_eight (a : ℝ) : 
  (∀ x : ℝ, |Real.sin (2*x) + a * Real.cos (2*x)| ≤ Real.sin (2*(Real.pi/8)) + a * Real.cos (2*(Real.pi/8))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_at_pi_over_eight_l553_55326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_plus_reward_is_five_l553_55356

/-- Represents the amount of money promised for a B+ grade -/
def b_plus_reward : ℚ := sorry

/-- The number of courses in Paul's scorecard -/
def num_courses : ℕ := 10

/-- The reward for an A grade is twice the reward for a B+ grade -/
def a_reward : ℚ := 2 * b_plus_reward

/-- The flat reward for an A+ grade -/
def a_plus_reward : ℚ := 15

/-- The maximum amount Paul could receive -/
def max_reward : ℚ := 190

/-- Theorem stating that the B+ reward is $5 given the conditions -/
theorem b_plus_reward_is_five :
  b_plus_reward = 5 ∧
  num_courses = 10 ∧
  a_reward = 2 * b_plus_reward ∧
  max_reward = 2 * a_plus_reward + 8 * (2 * a_reward) ∧
  max_reward = 190 →
  b_plus_reward = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_plus_reward_is_five_l553_55356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l553_55391

noncomputable def coefficient (n : ℕ) (f : ℝ → ℝ) : ℝ :=
  (1 / (n.factorial : ℝ)) * (deriv^[n] f) 0

theorem coefficient_x_cubed_expansion : 
  coefficient 3 (fun x => (1 - x^3) * (1 + x)^10) = 119 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l553_55391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_l553_55397

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity : ℝ := Real.sqrt 5 / 2

/-- The equation of the given ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The equation of the hyperbola we want to prove -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- Theorem stating that the hyperbola with the given eccentricity and sharing a focus with the ellipse has the equation x²/4 - y² = 1 -/
theorem hyperbola_equation_proof :
  ∃ (a b c : ℝ), 
    (c / a = eccentricity) ∧ 
    (∃ (x₀ y₀ : ℝ), ellipse_equation x₀ y₀ ∧ (x₀ - c)^2 + y₀^2 = a^2) →
    (∀ (x y : ℝ), hyperbola_equation x y ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by
  sorry

#check hyperbola_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_l553_55397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_l553_55386

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 2) - a (n + 1) = q * (a (n + 1) - a n)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 2 = 2 →
  a 3 = 5 →
  ∀ n : ℕ, n ≥ 1 → a n = (3^(n-1) / 2) + (1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_l553_55386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l553_55331

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 6)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (14, 0)

-- Define the midpoint P of AB
noncomputable def P : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the point Q on BC
def Q : ℝ × ℝ := (10, 0)

-- Define the center O of the circle
def O : ℝ × ℝ := (7, 1)

-- Theorem statement
theorem triangle_abc_properties :
  -- 1. Equation of line perpendicular to AB passing through P
  (∀ x y : ℝ, x + 3 * y = 10 ↔ (x - P.1) * (A.1 - B.1) + (y - P.2) * (A.2 - B.2) = 0) ∧
  -- 2. Length of AQ
  Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) = 10 ∧
  -- 3. Radius of the circle
  Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) = 5 * Real.sqrt 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l553_55331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l553_55363

theorem constant_term_of_expansion (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (a * x + 1 / x) * (2 * x - 1 / x)^5 = 2) →
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → 
    ∃ p q r : ℝ, (a * x + 1 / x) * (2 * x - 1 / x)^5 = c + p * x + q * (1 / x) + r) ∧
  c = 40 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l553_55363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_dormitory_location_l553_55362

-- Define the cost function
noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 5)^2 + 1000 / (x + 5)

-- State the theorem
theorem optimal_dormitory_location :
  ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 8 ∧
  f x = 150 ∧
  ∀ y, 2 ≤ y ∧ y ≤ 8 → f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_dormitory_location_l553_55362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isoperimetric_quotient_comparison_l553_55341

/-- Isoperimetric quotient of a figure -/
noncomputable def isoperimetric_quotient (area perimeter : ℝ) : ℝ := area / (perimeter ^ 2)

/-- A point is in the interior of a triangle -/
def interior_point (A B C P : ℝ × ℝ) : Prop := sorry

/-- Theorem: Isoperimetric quotient comparison for nested triangles -/
theorem isoperimetric_quotient_comparison
  (A B C A₁ A₂ : ℝ × ℝ)
  (h_equilateral : sorry)  -- ABC is an equilateral triangle
  (h_interior_A₁ : interior_point A B C A₁)
  (h_interior_A₂ : interior_point A₁ B C A₂)
  (area_A₁BC perimeter_A₁BC : ℝ)
  (area_A₂BC perimeter_A₂BC : ℝ) :
  isoperimetric_quotient area_A₁BC perimeter_A₁BC >
  isoperimetric_quotient area_A₂BC perimeter_A₂BC := by
  sorry

#check isoperimetric_quotient_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isoperimetric_quotient_comparison_l553_55341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l553_55395

/-- Curve A in the xy-plane -/
def curve_A (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Curve B in the xy-plane with parameter t -/
def curve_B (x y t : ℝ) : Prop := x = -1 + (Real.sqrt 2/2)*t ∧ y = 1 + (Real.sqrt 2/2)*t

/-- Point P on curve B -/
def point_P : ℝ × ℝ := (-1, 1)

/-- Distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    curve_A M.1 M.2 ∧ curve_A N.1 N.2 ∧
    (∃ t1 t2 : ℝ, curve_B M.1 M.2 t1 ∧ curve_B N.1 N.2 t2) ∧
    distance point_P M + distance point_P N = 12 * Real.sqrt 2 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l553_55395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l553_55389

/-- The point P -/
def P : ℝ × ℝ := (1, 1)

/-- The circular region -/
def circleRegion (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

/-- The line that maximizes the area difference -/
def max_diff_line (x y : ℝ) : Prop := x + y - 2 = 0

/-- Theorem stating that the line x + y - 2 = 0 maximizes the area difference -/
theorem max_area_difference :
  ∀ (l : ℝ → ℝ → Prop),
  (l P.1 P.2) →  -- The line passes through point P
  (∀ x y, l x y → circleRegion x y) →  -- The line intersects the circle
  (∃ x y, l x y ∧ ¬(max_diff_line x y)) →  -- The line is different from x + y - 2 = 0
  ∃ a1 a2 : ℝ, 
    (a1 ≥ 0) ∧ 
    (a2 ≥ 0) ∧ 
    (|a1 - a2| < |4 - 2*π|) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l553_55389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_rectangle_ratio_min_l553_55319

/-- Given a rectangle inscribed in a unit circle, we extend it to an octagon by adding
    the intersection points of the perpendicular bisectors of the rectangle's sides with the circle.
    This function represents the ratio of the area of this octagon to the area of the rectangle. -/
noncomputable def octagon_rectangle_area_ratio (a b : ℝ) : ℝ := (a + b) / (2 * a * b)

/-- The theorem states that the minimum value of the octagon to rectangle area ratio
    is √2, and this occurs when the rectangle is a square (i.e., when a = b). -/
theorem octagon_rectangle_ratio_min :
  (∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a^2 + b^2 = 1 →
    octagon_rectangle_area_ratio a b ≥ Real.sqrt 2) ∧
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a^2 + b^2 = 1 ∧
    octagon_rectangle_area_ratio a b = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_rectangle_ratio_min_l553_55319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_perimeter_l553_55335

/-- The perimeter of a regular pentagon with side length 2 cm is 10 cm. -/
theorem regular_pentagon_perimeter (side_length : ℝ) : 
  side_length = 2 → 5 * side_length = 10 := by
  intro h
  rw [h]
  norm_num

#check regular_pentagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_perimeter_l553_55335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_eight_thirty_l553_55390

-- Define the angle between hour and minute hands at a given time
noncomputable def angle_between_hands (hours : ℕ) (minutes : ℕ) : ℝ :=
  (30 * (hours % 12 : ℝ) + 0.5 * (minutes : ℝ)) - (6 * (minutes : ℝ))

-- Theorem statement
theorem angle_at_eight_thirty :
  angle_between_hands 8 30 = 75 := by
  -- Unfold the definition of angle_between_hands
  unfold angle_between_hands
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_eight_thirty_l553_55390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l553_55388

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law holds for the triangle -/
axiom sine_law {t : Triangle} : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The area formula for the triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle)
  (h1 : t.b * Real.sin t.A = 2 * t.a * Real.sin t.B)
  (h2 : t.a = Real.sqrt 7)
  (h3 : 2 * t.b - t.c = 4) :
  t.A = π/3 ∧ triangle_area t = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l553_55388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_max_not_greater_than_min_exists_function_zero_derivative_not_extreme_exists_function_without_max_and_min_extreme_value_in_interval_extreme_value_point_and_value_l553_55328

-- Statement 1
theorem exists_function_max_not_greater_than_min : 
  ∃ f : ℝ → ℝ, ∃ a b : ℝ, (∀ x, f x ≤ f a) ∧ (∀ x, f x ≥ f b) ∧ f a ≤ f b :=
sorry

-- Statement 2
theorem exists_function_zero_derivative_not_extreme : 
  ∃ f : ℝ → ℝ, ∃ c : ℝ, (HasDerivAt f 0 c) ∧ ¬(IsLocalMax f c ∨ IsLocalMin f c) :=
sorry

-- Statement 3
theorem exists_function_without_max_and_min : 
  ∃ f : ℝ → ℝ, (¬∃ a : ℝ, ∀ x, f x ≤ f a) ∧ (¬∃ b : ℝ, ∀ x, f x ≥ f b) :=
sorry

-- Statement 4
theorem extreme_value_in_interval {f : ℝ → ℝ} {a b : ℝ} (h : a < b) :
  (∃ c, a ≤ c ∧ c ≤ b ∧ (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f c)) →
  (∃ c, a < c ∧ c < b ∧ (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f c)) :=
sorry

-- Statement 5
theorem extreme_value_point_and_value {f : ℝ → ℝ} {a : ℝ} :
  (IsLocalMax f a ∨ IsLocalMin f a) ↔ 
  (∃ ε > 0, ∀ x, |x - a| < ε → f x ≤ f a) ∨ 
  (∃ ε > 0, ∀ x, |x - a| < ε → f x ≥ f a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_max_not_greater_than_min_exists_function_zero_derivative_not_extreme_exists_function_without_max_and_min_extreme_value_in_interval_extreme_value_point_and_value_l553_55328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coincides_with_ellipse_focus_theorem_l553_55316

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with 
    the right focus of the ellipse x^2/5 + y^2 = 1 -/
def parabola_focus_coincides_with_ellipse_focus : ℚ := 4

/-- The x-coordinate of the right focus of the ellipse x^2/5 + y^2 = 1 -/
def ellipse_right_focus_x : ℚ := 2

/-- The y-coordinate of the right focus of the ellipse x^2/5 + y^2 = 1 -/
def ellipse_right_focus_y : ℚ := 0

/-- The x-coordinate of the focus of the parabola y^2 = 2px -/
def parabola_focus_x (p : ℚ) : ℚ := p / 2

/-- The y-coordinate of the focus of the parabola y^2 = 2px -/
def parabola_focus_y : ℚ := 0

theorem parabola_focus_coincides_with_ellipse_focus_theorem :
  parabola_focus_x parabola_focus_coincides_with_ellipse_focus = ellipse_right_focus_x ∧
  parabola_focus_y = ellipse_right_focus_y := by
  sorry

#eval parabola_focus_x parabola_focus_coincides_with_ellipse_focus
#eval ellipse_right_focus_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coincides_with_ellipse_focus_theorem_l553_55316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_product_gt_12_l553_55300

/-- A ball with a number from 1 to 5 -/
def Ball := Fin 5

/-- The set of all possible outcomes when drawing two balls -/
def AllOutcomes := Ball × Ball

/-- Predicate for a pair of balls satisfying the condition -/
def SatisfiesCondition (pair : AllOutcomes) : Prop :=
  let (a, b) := pair
  Odd ((a.val + 1) * (b.val + 1)) ∧ ((a.val + 1) * (b.val + 1) : ℕ) > 12

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 3

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 25

theorem probability_odd_product_gt_12 :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_product_gt_12_l553_55300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_calculation_l553_55351

noncomputable section

-- Define the radius and central angle
def radius : ℝ := 3
def centralAngle : ℝ := 120 * (Real.pi / 180)

-- Define the perimeter of the monster
def monsterPerimeter : ℝ := 
  (2 * Real.pi * radius * (1 - centralAngle / (2 * Real.pi))) + (2 * radius * Real.sin (centralAngle / 2))

-- Theorem statement
theorem monster_perimeter_calculation :
  monsterPerimeter = 4 * Real.pi + 3 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_calculation_l553_55351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_on_unit_circle_l553_55333

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The vector from one point to another -/
def vector (p q : Point) : Point :=
  ⟨q.x - p.x, q.y - p.y⟩

/-- The magnitude of a vector -/
noncomputable def magnitude (v : Point) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Check if a point is on the unit circle -/
def onUnitCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 1

/-- Check if three points form a right-angled triangle -/
def isRightAngled (a b c : Point) : Prop :=
  (distance a b)^2 + (distance b c)^2 = (distance a c)^2 ∨
  (distance b c)^2 + (distance c a)^2 = (distance b a)^2 ∨
  (distance c a)^2 + (distance a b)^2 = (distance c b)^2

/-- Addition of Points -/
instance : Add Point where
  add (p q : Point) := ⟨p.x + q.x, p.y + q.y⟩

theorem max_vector_sum_on_unit_circle (a b c : Point) (m : Point) :
  isRightAngled a b c →
  onUnitCircle a →
  onUnitCircle b →
  onUnitCircle c →
  m = ⟨1/2, 1/2⟩ →
  (∃ (v : Point), magnitude v ≤ (3 * Real.sqrt 2) / 2 + 1 ∧
    ∀ (a' b' c' : Point),
      isRightAngled a' b' c' →
      onUnitCircle a' →
      onUnitCircle b' →
      onUnitCircle c' →
      magnitude (vector m a' + vector m b' + vector m c') ≤ magnitude v) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_on_unit_circle_l553_55333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l553_55309

noncomputable def f (a : ℝ) (x : ℝ) := (2*a - 6)^x

def g (a : ℝ) (x : ℝ) := x^2 - 3*a*x + 2*a^2 + 1

def p (a : ℝ) := ∀ x y, x < y → f a y < f a x

def q (a : ℝ) := ∀ x, g a x = 0 → x > 3

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a > 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l553_55309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_is_yellow_l553_55312

/-- Represents the colors of beads in the bracelet -/
inductive BeadColor
  | Red
  | Orange
  | Yellow
  | Green
  | Blue
deriving Repr

/-- The repeating pattern of beads -/
def pattern : List BeadColor :=
  [BeadColor.Red, BeadColor.Red, BeadColor.Orange, 
   BeadColor.Yellow, BeadColor.Yellow, BeadColor.Yellow,
   BeadColor.Green, BeadColor.Green, BeadColor.Blue]

/-- The total number of beads in the bracelet -/
def totalBeads : Nat := 85

/-- The color of the bead at a given position in the pattern -/
def colorAtPosition (n : Nat) : BeadColor :=
  pattern[n % pattern.length]'(by
    apply Nat.mod_lt
    exact Nat.zero_lt_of_ne_zero (by decide))

/-- Theorem stating that the last bead is yellow -/
theorem last_bead_is_yellow : 
  colorAtPosition (totalBeads - 1) = BeadColor.Yellow := by
  sorry

#eval colorAtPosition (totalBeads - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_is_yellow_l553_55312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l553_55353

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 1 = 0

-- Define the distance function between two parallel lines
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_between_lines :
  distance_parallel_lines 6 8 (-4) 1 = 1/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l553_55353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_to_asymptotes_distance_l553_55347

/-- Represents a hyperbola with equation x²/8 - y² = 1 -/
structure Hyperbola where
  equation : ∀ x y : ℝ, x^2 / 8 - y^2 = 1

/-- Represents the foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(3, 0), (-3, 0)}

/-- Represents the asymptotes of the hyperbola -/
def asymptotes : Set (ℝ → ℝ) :=
  {(λ x => (Real.sqrt 2 / 4) * x), (λ x => -(Real.sqrt 2 / 4) * x)}

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
  sorry

/-- Theorem: The distance from the foci of the hyperbola to its asymptotes is 1 -/
theorem foci_to_asymptotes_distance :
  ∀ f ∈ foci, ∀ a ∈ asymptotes, distancePointToLine f a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_to_asymptotes_distance_l553_55347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inverse_of_itself_l553_55396

noncomputable def f (a b c d x : ℝ) : ℝ := (2*a*x + b) / (c*x - 3*d)

theorem function_inverse_of_itself
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h : ∀ x, x ∈ Set.univ \ {x | c * x = 3 * d} → f a b c d (f a b c d x) = x) :
  a - 3*d = 0 := by
  sorry

#check function_inverse_of_itself

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inverse_of_itself_l553_55396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rhombus_diagonal_distinction_l553_55301

-- Define the basic shapes
class Quadrilateral (α : Type*) where
  -- We'll leave this empty for now, as we don't need specific properties for this proof

class Rectangle (α : Type*) extends Quadrilateral α where
  -- Rectangle-specific properties could be added here

class Rhombus (α : Type*) extends Quadrilateral α where
  -- Rhombus-specific properties could be added here

-- Define the property of equal diagonals
def has_equal_diagonals {α : Type*} [Quadrilateral α] (q : α) : Prop := 
  ∃ (d1 d2 : ℝ), d1 = d2 ∧ (d1 > 0) ∧ (d2 > 0)

-- State the theorem
theorem rectangle_rhombus_diagonal_distinction :
  (∀ {α : Type*} [Rectangle α] (R : α), has_equal_diagonals R) ∧
  ¬(∀ {β : Type*} [Rhombus β] (S : β), has_equal_diagonals S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rhombus_diagonal_distinction_l553_55301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_parabola_l553_55322

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 6 * Real.sin θ * (1 / Real.cos θ)

-- Define the Cartesian equation of a parabola
def parabola_equation (x y : ℝ) : Prop :=
  x^2 = 6 * y

-- Theorem stating the equivalence of the polar equation to the Cartesian parabola equation
theorem polar_to_parabola :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    parabola_equation x y :=
by
  sorry

#check polar_to_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_parabola_l553_55322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_characterization_l553_55308

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that P(a+b) - P(b) is a multiple of P(a) for all integers a and b -/
def HasPropertyP (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, ∃ k : ℤ, P.eval (a + b) - P.eval b = k * P.eval a

/-- The main theorem -/
theorem polynomial_property_characterization (P : IntPolynomial) :
  HasPropertyP P → (∃ c : ℤ, P = c • (Polynomial.X : IntPolynomial)) ∨ (∃ c : ℤ, P = Polynomial.C c) := by
  sorry

#check polynomial_property_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_characterization_l553_55308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_comparison_l553_55325

/-- Calculates the cost at Store A given a purchase amount x -/
noncomputable def costA (x : ℝ) : ℝ :=
  if x ≤ 100 then x else 100 + 0.9 * (x - 100)

/-- Calculates the cost at Store B given a purchase amount x -/
noncomputable def costB (x : ℝ) : ℝ :=
  if x ≤ 50 then x else 50 + 0.95 * (x - 50)

theorem store_comparison (x : ℝ) :
  (x ≤ 50 → costA x = costB x) ∧
  (50 < x ∧ x < 150 → costB x < costA x) ∧
  (x > 150 → costA x < costB x) ∧
  (x = 150 → costA x = costB x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_comparison_l553_55325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_diff_l553_55364

theorem max_value_of_exponential_diff :
  ∃ (x : ℝ), ∀ (y : ℝ), (3:ℝ)^x - (9:ℝ)^x ≥ (3:ℝ)^y - (9:ℝ)^y ∧ (3:ℝ)^x - (9:ℝ)^x = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_diff_l553_55364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mold_cost_is_three_l553_55343

/-- Represents the cost of molds given the popsicle-making scenario -/
noncomputable def cost_of_molds (total_budget : ℚ) (stick_pack_cost : ℚ) (sticks_per_pack : ℕ)
  (juice_bottle_cost : ℚ) (popsicles_per_bottle : ℕ) (unused_sticks : ℕ) : ℚ :=
  let used_sticks := sticks_per_pack - unused_sticks
  let bottles_used := (used_sticks : ℚ) / popsicles_per_bottle
  let juice_cost := bottles_used * juice_bottle_cost
  let total_cost_without_molds := stick_pack_cost + juice_cost
  total_budget - total_cost_without_molds

/-- The cost of molds is $3 given the specified conditions -/
theorem mold_cost_is_three :
  cost_of_molds 10 1 100 2 20 40 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mold_cost_is_three_l553_55343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_equality_l553_55371

theorem trigonometric_sum_equality : 
  Real.sin (5 * π / 24) ^ 4 + Real.cos (7 * π / 24) ^ 4 + 
  Real.sin (17 * π / 24) ^ 4 + Real.cos (19 * π / 24) ^ 4 = 
  3/2 - Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_equality_l553_55371
