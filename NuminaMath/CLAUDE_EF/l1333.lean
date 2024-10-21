import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_equals_five_sixths_l1333_133373

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x^2
  else if 1 ≤ x ∧ x ≤ 2 then 2 - x
  else 0  -- Define the function for all real numbers

-- State the theorem
theorem integral_of_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_equals_five_sixths_l1333_133373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_pear_equivalence_l1333_133361

-- Define the cost of fruits in terms of an arbitrary unit
def cost_banana : ℚ → ℚ := sorry
def cost_apple : ℚ → ℚ := sorry
def cost_pear : ℚ → ℚ := sorry

-- Define the given conditions
axiom banana_apple_ratio : cost_banana 4 = cost_apple 3
axiom apple_pear_ratio : cost_apple 9 = cost_pear 6

-- State the theorem to be proved
theorem banana_pear_equivalence : cost_banana 24 = cost_pear 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_pear_equivalence_l1333_133361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pipes_fill_tank_l1333_133308

/-- Represents the time taken to fill the tank using all pipes -/
noncomputable def total_time : ℝ := 5

/-- Represents the time taken by pipe a alone to fill the tank -/
noncomputable def time_a : ℝ := 35

/-- Represents the relative speed of pipe b compared to pipe a -/
noncomputable def speed_ratio_b_a : ℝ := 2

/-- Represents the relative speed of pipe c compared to pipe b -/
noncomputable def speed_ratio_c_b : ℝ := 2

/-- Calculates the rate at which a pipe fills the tank given its fill time -/
noncomputable def rate (time : ℝ) : ℝ := 1 / time

/-- Represents the number of pipes used -/
def num_pipes : ℕ := 3

/-- Theorem stating that given the conditions, 3 pipes are used to fill the tank -/
theorem three_pipes_fill_tank : 
  rate total_time = rate time_a + speed_ratio_b_a * rate time_a + 
    speed_ratio_c_b * speed_ratio_b_a * rate time_a → num_pipes = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pipes_fill_tank_l1333_133308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_garage_time_l1333_133333

theorem parking_garage_time (
  total_floors : ℕ) 
  (gate_frequency : ℕ) 
  (gate_wait_time : ℕ) 
  (floor_distance : ℕ) 
  (driving_speed : ℕ) : 
  total_floors = 12 →
  gate_frequency = 3 →
  gate_wait_time = 2 →
  floor_distance = 800 →
  driving_speed = 10 →
  (total_floors / gate_frequency * gate_wait_time * 60 + 
   floor_distance / driving_speed * total_floors) = 1440 := by
  intros h1 h2 h3 h4 h5
  sorry

#eval 12 / 3 * 2 * 60 + 800 / 10 * 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_garage_time_l1333_133333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_properties_l1333_133372

theorem regular_polygon_properties (exterior_angle : ℝ) 
  (h1 : exterior_angle = 18) : 
  ∃ (num_sides : ℕ) (sum_interior_angles : ℝ),
    num_sides = 20 ∧ 
    sum_interior_angles = 3240 ∧
    (↑num_sides : ℝ) = 360 / exterior_angle ∧
    sum_interior_angles = 180 * (↑num_sides - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_properties_l1333_133372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadratic_real_roots_l1333_133366

def S : Finset Int := {-3, -2, -1, 0, 1, 2, 3}

theorem count_quadratic_real_roots : 
  (Finset.filter (fun p : Int × Int => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1^2 - 4*p.2 ≥ 0) 
    (Finset.product S S)).card = 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadratic_real_roots_l1333_133366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1333_133339

-- Define the polar coordinate system
structure PolarCoordinate where
  ρ : ℝ
  θ : ℝ

-- Define the circle in polar coordinates
def circleEquation (p : PolarCoordinate) : Prop :=
  p.ρ = 2 * Real.cos p.θ

-- Define the line in polar coordinates
def lineEquation (p : PolarCoordinate) : Prop :=
  p.ρ * Real.cos p.θ = 4

-- Define the distance function
noncomputable def distance (p1 p2 : PolarCoordinate) : ℝ :=
  Real.sqrt ((p1.ρ * Real.cos p1.θ - p2.ρ * Real.cos p2.θ)^2 + 
             (p1.ρ * Real.sin p1.θ - p2.ρ * Real.sin p2.θ)^2)

-- Theorem statement
theorem distance_from_circle_center_to_line :
  ∃ (center : PolarCoordinate), 
    (∀ p, circleEquation p → distance center p ≤ distance center (PolarCoordinate.mk 2 0)) ∧
    (∀ p, lineEquation p → distance center p = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1333_133339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_proof_l1333_133314

/-- Represents the number of days it takes for one man to complete the work -/
noncomputable def days_for_one_man : ℝ := 100

/-- Represents the total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- Represents the rate at which one man completes the work per day -/
noncomputable def man_rate : ℝ := total_work / days_for_one_man

/-- Represents the rate at which one woman completes the work per day -/
noncomputable def woman_rate : ℝ := total_work / 225

theorem work_completion_proof :
  (10 * man_rate + 15 * woman_rate) * 6 = total_work ∧
  woman_rate * 225 = total_work ∧
  man_rate * days_for_one_man = total_work :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_proof_l1333_133314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1333_133330

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + (1/4)*x - 5

-- Theorem statement
theorem f_has_zero_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1333_133330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1333_133317

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem triangle_angle_range (a b c A B C : ℝ) : 
  0 < A ∧ A < Real.pi / 2 ∧   -- A is acute
  0 < B ∧ B < Real.pi / 2 ∧   -- B is acute
  0 < C ∧ C < Real.pi / 2 ∧   -- C is acute
  A + B + C = Real.pi ∧       -- Sum of angles in a triangle
  (a^2 + c^2 - b^2) / c = (a^2 + b^2 - c^2) / (2*a - c) →
  ∀ y ∈ Set.Ioo 1 2, ∃ A', A' ∈ Set.Ioo (Real.pi/6) (Real.pi/2) ∧ f A' = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1333_133317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pyramid_volume_l1333_133360

/-- The volume of a cone in cubic centimeters -/
noncomputable def cone_volume : ℝ := 20

/-- The angles of the base triangle of the inscribed pyramid in radians -/
noncomputable def α : ℝ := 0.449106
noncomputable def β : ℝ := 1.434573
noncomputable def γ : ℝ := 1.258131

/-- The volume of the inscribed pyramid -/
noncomputable def pyramid_volume : ℝ := (2 * cone_volume / Real.pi) * Real.sin α * Real.sin β * Real.sin γ

/-- Theorem stating that the volume of the inscribed pyramid is approximately 5.21 cm³ -/
theorem inscribed_pyramid_volume :
  abs (pyramid_volume - 5.21) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pyramid_volume_l1333_133360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_three_l1333_133355

noncomputable def f (x : ℝ) := Real.exp (-5 * x) + 2

theorem tangent_line_at_zero_three :
  let tangent_line (x : ℝ) := -5 * x + 3
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| → |x| < δ → |(tangent_line x - f x) / x| < ε) ∧
  tangent_line 0 = f 0 := by
  sorry

#check tangent_line_at_zero_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_three_l1333_133355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_l1333_133388

/-- The sum of five complex exponentials -/
noncomputable def complex_sum : ℂ := 
  Complex.exp (11 * Real.pi * Complex.I / 60) + 
  Complex.exp (31 * Real.pi * Complex.I / 60) + 
  Complex.exp (51 * Real.pi * Complex.I / 60) + 
  Complex.exp (71 * Real.pi * Complex.I / 60) + 
  Complex.exp (91 * Real.pi * Complex.I / 60)

/-- The theorem stating that the argument of the complex sum is 17π/20 -/
theorem complex_sum_arg :
  Complex.arg complex_sum = 17 * Real.pi / 20 ∧ 
  0 ≤ Complex.arg complex_sum ∧ 
  Complex.arg complex_sum < 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_l1333_133388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_ten_l1333_133301

theorem divisible_by_ten (a b c d e : ℤ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (prod_div_ten : ∀ x y z : ℤ, x ∈ ({a, b, c, d, e} : Set ℤ) → y ∈ ({a, b, c, d, e} : Set ℤ) → z ∈ ({a, b, c, d, e} : Set ℤ) →
                  x ≠ y ∧ y ≠ z ∧ x ≠ z → (x * y * z) % 10 = 0) :
  ∃ x : ℤ, x ∈ ({a, b, c, d, e} : Set ℤ) ∧ x % 10 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_ten_l1333_133301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l1333_133332

noncomputable section

/-- The line equation y = 2x - 4 --/
def line_equation (x : ℝ) : ℝ := 2 * x - 4

/-- The external point (3, -1) --/
def external_point : ℝ × ℝ := (3, -1)

/-- The claimed closest point (9/5, -2/5) --/
def closest_point : ℝ × ℝ := (9/5, -2/5)

/-- Theorem stating that the closest_point is on the line and is the closest to the external_point --/
theorem closest_point_is_correct :
  (line_equation closest_point.1 = closest_point.2) ∧
  (∀ p : ℝ × ℝ, line_equation p.1 = p.2 →
    (closest_point.1 - external_point.1)^2 + (closest_point.2 - external_point.2)^2 ≤
    (p.1 - external_point.1)^2 + (p.2 - external_point.2)^2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l1333_133332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_books_count_l1333_133331

def books_loaned : ℝ := 40.00000000000001
def return_rate : ℝ := 0.8
def final_books : ℕ := 67

theorem initial_books_count :
  ∃ (initial_books : ℕ),
    (initial_books : ℝ) - (books_loaned - books_loaned * return_rate) = final_books ∧
    initial_books = 75 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_books_count_l1333_133331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_price_calculation_l1333_133325

/-- Given a video recorder with a wholesale cost and markup percentage,
    calculate the price an employee pays after applying their discount. -/
theorem employee_price_calculation
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_discount_percentage : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_discount_percentage = 10) :
  wholesale_cost * (1 + markup_percentage / 100) * (1 - employee_discount_percentage / 100) = 216 := by
  -- Calculate retail price
  have retail_price := wholesale_cost * (1 + markup_percentage / 100)
  -- Calculate employee price
  have employee_price := retail_price * (1 - employee_discount_percentage / 100)
  -- Prove that employee_price equals 216
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_price_calculation_l1333_133325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_not_dividing_power_minus_self_l1333_133376

theorem prime_not_dividing_power_minus_self (p : ℕ) (h_p : Nat.Prime p) :
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬(q ∣ n^p - p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_not_dividing_power_minus_self_l1333_133376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_eight_equation_l1333_133329

theorem power_eight_equation (y : ℝ) (h : (8 : ℝ)^(5*y) = 512) : (8 : ℝ)^(5*y - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_eight_equation_l1333_133329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1333_133340

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  
/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The sine of an angle -/
def sin (θ : ℝ) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : area t = (t.a + t.b)^2 - t.c^2) 
  (h2 : t.a + t.b = 4) : 
  (sin t.C = 8/17) ∧ 
  ((t.a^2 - t.b^2) / t.c^2 = sin (t.A - t.B) / sin t.C) ∧ 
  (t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * area t) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1333_133340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_veranda_area_l1333_133334

theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
  room_length * room_width = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_veranda_area_l1333_133334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_equal_f_values_l1333_133348

noncomputable def f (x : ℝ) : ℝ := |Real.log x - 1/2|

theorem product_of_equal_f_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hf : f a = f b) :
  a * b = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_equal_f_values_l1333_133348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_frequency_is_point_two_l1333_133358

def sample : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

noncomputable def inRange (x : ℝ) : Bool :=
  11.5 ≤ x ∧ x ≤ 13.5

noncomputable def countInRange (xs : List ℝ) : Nat :=
  xs.filter inRange |>.length

theorem range_frequency_is_point_two :
  (countInRange sample : ℝ) / sample.length = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_frequency_is_point_two_l1333_133358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1333_133370

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define the line in polar coordinates
def line (p : PolarPoint) : Prop :=
  p.ρ * Real.cos (p.θ - Real.pi/4) = Real.sqrt 2

-- Define the circle in polar coordinates
def circle' (p : PolarPoint) : Prop :=
  p.ρ = Real.sqrt 2

-- Theorem statement
theorem line_circle_intersection :
  ∃! p : PolarPoint, line p ∧ circle' p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1333_133370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l1333_133357

-- Define the curve C
noncomputable def curve_C (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def line_l_polar (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Define the line l in Cartesian form
def line_l_cartesian (x y m : ℝ) : Prop :=
  Real.sqrt 3 * x + y + 2 * m = 0

-- Theorem statement
theorem curve_line_intersection (m : ℝ) :
  (∃ t, ∃ x y, curve_C t = (x, y) ∧ line_l_cartesian x y m) ↔ 
  -19/12 ≤ m ∧ m ≤ 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l1333_133357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l1333_133362

/-- A point in the XY-plane with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- A right triangle ABC in the XY-plane -/
structure RightTriangle where
  A : Point
  B : Point
  C : Point

/-- Check if a point is within the given range -/
def inRange (p : Point) : Prop :=
  -8 ≤ p.x ∧ p.x ≤ 2 ∧ 4 ≤ p.y ∧ p.y ≤ 9

/-- Check if a triangle satisfies the given conditions -/
def isValidTriangle (t : RightTriangle) : Prop :=
  inRange t.A ∧ inRange t.B ∧ inRange t.C ∧
  t.A.x = t.B.x ∧ t.A.y = t.C.y ∧
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.B ≠ t.C

/-- The set of all valid right triangles -/
def validTriangles : Set RightTriangle :=
  { t : RightTriangle | isValidTriangle t }

/-- Finite type instance for validTriangles -/
instance : Fintype validTriangles := sorry

theorem count_valid_triangles :
  Fintype.card validTriangles = 3300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l1333_133362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_of_m_l1333_133384

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

theorem range_of_m :
  {m : ℝ | f (2*m - 1) > f (1 - m)} = Set.Ioo (2/3) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_of_m_l1333_133384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1333_133326

def M : Set ℤ := {x : ℤ | x^2 ≤ 1}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

def M_real : Set ℝ := {x : ℝ | ∃ y : ℤ, y ∈ M ∧ x = ↑y}

theorem intersection_M_N : M_real ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1333_133326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1333_133319

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the main theorem
theorem triangle_theorem (t : Triangle) : 
  (Real.sin t.C = Real.sin t.A * Real.cos t.B + (Real.sqrt 2 / 2) * Real.sin (t.A + t.C)) →
  (t.A = Real.pi / 4) ∧
  (t.a = Real.sqrt 5 ∧ t.b = Real.sqrt 2) →
  ∃! (h : Real), h = (3 * Real.sqrt 5) / 5 ∧ 
    h * t.a = t.b * t.c * Real.sin t.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1333_133319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1333_133304

open Set Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) :
  {x : ℝ | (exp 1) * f x > f 1 * exp x} = {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1333_133304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1333_133344

/-- 
Given an arithmetic sequence where 3 is the first term and 38 is the last term,
prove that the sum of the fourth-to-last and third-to-last terms is 61.
-/
theorem arithmetic_sequence_sum (seq : List ℕ) : 
  seq.head? = some 3 →
  seq.getLast? = some 38 →
  seq.length ≥ 6 →
  (∀ i : ℕ, i + 1 < seq.length → seq[i+1]! - seq[i]! = seq[1]! - seq[0]!) →
  seq[seq.length - 4]! + seq[seq.length - 3]! = 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1333_133344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_problem_ribbon_length_problem_cm_l1333_133389

theorem ribbon_length_problem (original_length : ℝ) (remaining_length : ℝ) (num_pieces : ℕ) 
  (h1 : original_length = 51)
  (h2 : remaining_length = 36)
  (h3 : num_pieces = 100) : 
  (original_length - remaining_length) / num_pieces * 100 = 15 := by
  -- Replace real numbers with their values
  rw [h1, h2, h3]
  
  -- Perform the calculation
  norm_num
  
  -- The result is automatically simplified to 15

theorem ribbon_length_problem_cm : 
  ∃ (original_length remaining_length : ℝ) (num_pieces : ℕ),
  original_length = 51 ∧
  remaining_length = 36 ∧
  num_pieces = 100 ∧
  (original_length - remaining_length) / num_pieces * 100 = 15 := by
  use 51, 36, 100
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  · norm_num

#check ribbon_length_problem
#check ribbon_length_problem_cm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_problem_ribbon_length_problem_cm_l1333_133389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_min_distance_curve_C_to_line_l_l1333_133320

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define point P
def point_P : ℝ × ℝ := (0, 4)

-- Define the distance function from a point to the line l
noncomputable def distance_to_line_l (x y : ℝ) : ℝ :=
  abs (x - y + 4) / Real.sqrt 2

theorem point_P_on_line_l :
  line_l point_P.1 point_P.2 := by sorry

theorem min_distance_curve_C_to_line_l :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (α : ℝ), distance_to_line_l (curve_C α).1 (curve_C α).2 ≥ d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_min_distance_curve_C_to_line_l_l1333_133320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_data_l1333_133369

open Set

theorem inconsistent_data :
  ¬ ∃ (A : Finset Nat) (A₁ A₂ A₃ : Finset Nat),
    A₁ ⊆ A ∧ A₂ ⊆ A ∧ A₃ ⊆ A ∧
    A.card = 1000 ∧
    A₁.card = 265 ∧
    A₂.card = 51 ∧
    A₃.card = 803 ∧
    (A₁ ∪ A₂).card = 287 ∧
    (A₂ ∪ A₃).card = 843 ∧
    (A₁ ∪ A₃).card = 919 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_data_l1333_133369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_l1333_133321

/-- A quadratic function with specific properties -/
def special_quadratic (a b c : ℝ) : Prop :=
  let f := fun x => a * x^2 + b * x + c
  (f (-4) = 0)
  ∧ (f 2 = 0)
  ∧ (∃ x, f x = 2 * x ∧ (a * 2 * x + b) = 0)

/-- The theorem stating that the quadratic function with given properties
    is uniquely determined -/
theorem unique_quadratic :
  ∀ a b c : ℝ, special_quadratic a b c →
  a = 2/9 ∧ b = 4/9 ∧ c = -16/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_l1333_133321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_approx_l1333_133309

/-- Represents the time in minutes for Danny to reach Steve's house -/
noncomputable def danny_time : ℝ := 31

/-- Represents the time in minutes for Steve to reach Danny's house -/
noncomputable def steve_time : ℝ := 2 * danny_time

/-- Represents the wind speed factor for Danny (speed increase) -/
noncomputable def danny_wind_factor : ℝ := 1.1

/-- Represents the wind speed factor for Steve (speed decrease) -/
noncomputable def steve_wind_factor : ℝ := 0.9

/-- Calculates the time difference between Steve and Danny reaching the halfway point -/
noncomputable def time_difference : ℝ :=
  (steve_time / (2 * steve_wind_factor)) - (danny_time / (2 * danny_wind_factor))

theorem time_difference_approx :
  ∃ ε > 0, |time_difference - 20.35| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_approx_l1333_133309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pairwise_coprime_l1333_133386

def f (x : ℤ) : ℤ := x^2 - x + 1

def sequenceM (m : ℤ) : ℕ → ℤ
  | 0 => m
  | n + 1 => f (sequenceM m n)

theorem sequence_pairwise_coprime (m : ℤ) (h : m > 1) :
  ∀ i j : ℕ, i ≠ j → Int.gcd (sequenceM m i) (sequenceM m j) = 1 := by
  sorry

#check sequence_pairwise_coprime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pairwise_coprime_l1333_133386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_approx_l1333_133336

-- Define the first value
noncomputable def first_value : ℝ := Real.sqrt (0.4 * 60)

-- Define the second value
noncomputable def second_value : ℝ := (4/5) * (25^3)

-- Define the difference
noncomputable def difference : ℝ := first_value - second_value

-- Theorem statement
theorem difference_approx :
  abs (difference - (-12495.10102)) < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_approx_l1333_133336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_point_fahrenheit_l1333_133343

/-- Converts Celsius temperature to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

/-- The temperature at which water boils in Celsius -/
def boiling_point_celsius : ℝ := 100

/-- The temperature at which water boils in Fahrenheit -/
def boiling_point_fahrenheit : ℝ := 212

/-- The temperature at which ice melts in Celsius -/
def melting_point_celsius : ℝ := 0

/-- Temperature of a pot of water in Celsius -/
def pot_temperature_celsius : ℝ := 55

/-- Temperature of a pot of water in Fahrenheit -/
def pot_temperature_fahrenheit : ℝ := 131

/-- Theorem stating that the melting point of ice in Fahrenheit is 32°F -/
theorem melting_point_fahrenheit :
  celsius_to_fahrenheit melting_point_celsius = 32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_point_fahrenheit_l1333_133343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_existence_l1333_133341

-- Define the four functions
noncomputable def f_A (x : ℝ) := x^2 + x
noncomputable def f_B (x : ℝ) := x^3 + Real.exp x
noncomputable def f_C (x : ℝ) := Real.log x + x^2 / 2
noncomputable def f_D (x : ℝ) := Real.sqrt x + 2*x

-- Define their derivatives
noncomputable def f_A' (x : ℝ) := 2*x + 1
noncomputable def f_B' (x : ℝ) := 3*x^2 + Real.exp x
noncomputable def f_C' (x : ℝ) := 1/x + x
noncomputable def f_D' (x : ℝ) := 1/(2*Real.sqrt x) + 2

-- Theorem statement
theorem tangent_line_existence :
  (∃ x : ℝ, f_A' x = 2) ∧
  (∃ x : ℝ, f_B' x = 2) ∧
  (∃ x : ℝ, f_C' x = 2) ∧
  (∀ x : ℝ, x > 0 → f_D' x > 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_existence_l1333_133341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_square_sequence_2018_l1333_133328

/-- A function that returns true if a number is a perfect square, false otherwise -/
def is_perfect_square (n : ℕ) : Bool :=
  match Nat.sqrt n with
  | m => m * m == n

/-- The sequence of natural numbers with perfect squares removed -/
def non_square_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let prev := non_square_sequence n
    let next := prev + 1
    if is_perfect_square next then non_square_sequence (n + 1) else next
termination_by non_square_sequence n => n

/-- The 2018th term of the non-square sequence is 2063 -/
theorem non_square_sequence_2018 : non_square_sequence 2017 = 2063 := by
  sorry

#eval non_square_sequence 2017

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_square_sequence_2018_l1333_133328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_sixth_power_l1333_133396

noncomputable def z : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

theorem z_sixth_power : z^6 = (1 : ℂ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_sixth_power_l1333_133396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l1333_133322

noncomputable def g (z : ℂ) : ℂ := ((-2 + 2 * Complex.I * Real.sqrt 3) * z + (-3 * Real.sqrt 3 - 27 * Complex.I)) / 3

noncomputable def d : ℂ := -69 * Real.sqrt 3 / 37 - 141 * Complex.I / 37

theorem rotation_fixed_point : g d = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l1333_133322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_before_increase_l1333_133311

theorem soda_price_before_increase (current_price : ℝ) (increase_percentage : ℝ) 
  (h1 : current_price = 6)
  (h2 : increase_percentage = 50) :
  current_price / (1 + increase_percentage / 100) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_before_increase_l1333_133311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_boats_l1333_133371

-- Define the problem parameters
noncomputable def island_height : ℝ := 300
noncomputable def depression_angle_B : ℝ := 30 * Real.pi / 180
noncomputable def depression_angle_C : ℝ := 45 * Real.pi / 180

-- Theorem statement
theorem distance_between_boats :
  let AB := island_height / Real.tan depression_angle_B
  let AC := island_height / Real.tan depression_angle_C
  Real.sqrt (AB^2 + AC^2) = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_boats_l1333_133371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_diagonals_is_ten_l1333_133303

/-- Represents a regular pentagonal prism -/
structure RegularPentagonalPrism where
  -- Add any necessary properties here
  dummy : Unit

/-- A diagonal in a regular pentagonal prism -/
def diagonal (prism : RegularPentagonalPrism) : Type :=
  Unit -- Placeholder definition

/-- The total number of diagonals in a regular pentagonal prism -/
def total_diagonals (prism : RegularPentagonalPrism) : ℕ :=
  10 -- Placeholder definition

/-- Theorem stating that the total number of diagonals in a regular pentagonal prism is 10 -/
theorem total_diagonals_is_ten (prism : RegularPentagonalPrism) :
  total_diagonals prism = 10 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_diagonals_is_ten_l1333_133303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1333_133347

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10*x
  else 51*x + 1000/x - 1450

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ :=
  50*x - C x - 250

-- State the theorem
theorem max_profit_at_100 :
  ∀ x > 0, L x ≤ L 100 := by
  sorry

#check max_profit_at_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1333_133347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_problem_l1333_133375

theorem cookie_problem (r_large r_small : ℝ) (n_small : ℕ) (r_scrap : ℝ) :
  r_large = 4 →
  r_small = 1 →
  n_small = 6 →
  π * r_large^2 = π * (n_small : ℝ) * r_small^2 + 2 * π * r_scrap^2 →
  r_scrap = Real.sqrt 5 :=
by
  intros h_large h_small h_n h_area
  -- The proof steps would go here
  sorry

#check cookie_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_problem_l1333_133375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l1333_133356

theorem triangle_cosine_sum (a b c A B C : ℝ) :
  0 < A → 0 < B → 0 < C →
  A + B + C = π →
  C = π/3 →
  c = 2 →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  a * Real.cos B + b * Real.cos A = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_l1333_133356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stoichiometry_and_methane_required_l1333_133365

-- Define the chemical species
inductive Species
| CH4 | O2 | CO2 | H2O | CO

-- Define a reaction as a map from species to their stoichiometric coefficients
def Reaction := Species → ℤ

-- Define the two reactions
def reaction1 : Reaction := sorry
def reaction2 : Reaction := sorry

-- Known quantities from the problem
axiom known_oxygen : reaction1 Species.O2 = 2
axiom known_carbon_dioxide : reaction1 Species.CO2 = 1
axiom known_water : reaction1 Species.H2O = 2

-- Conservation of atoms in each reaction
axiom conservation_C_1 : reaction1 Species.CH4 = reaction1 Species.CO2
axiom conservation_H_1 : 4 * reaction1 Species.CH4 = 2 * reaction1 Species.H2O
axiom conservation_O_1 : 2 * reaction1 Species.O2 = 2 * reaction1 Species.CO2 + reaction1 Species.H2O

axiom conservation_C_2 : reaction2 Species.CH4 = reaction2 Species.CO
axiom conservation_H_2 : 4 * reaction2 Species.CH4 = 2 * reaction2 Species.H2O
axiom conservation_O_2 : 2 * reaction2 Species.O2 = reaction2 Species.CO + reaction2 Species.H2O

-- Theorem stating the stoichiometric coefficients and total methane required
theorem stoichiometry_and_methane_required :
  reaction1 Species.CH4 = 1 ∧
  reaction1 Species.O2 = 2 ∧
  reaction1 Species.CO2 = 1 ∧
  reaction1 Species.H2O = 2 ∧
  reaction2 Species.CH4 = 1 ∧
  reaction2 Species.O2 = 2 ∧
  reaction2 Species.CO = 1 ∧
  reaction2 Species.H2O = 2 ∧
  reaction1 Species.CH4 + reaction2 Species.CH4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stoichiometry_and_methane_required_l1333_133365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_equals_g_l1333_133346

-- Define the parameter φ as a variable
variable (φ : ℝ)

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi + x) * Real.sin (x + Real.pi / 3 + φ)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x - φ)

-- State the theorem
theorem f_shift_equals_g (h1 : φ ∈ Set.Ioo 0 Real.pi) 
  (h2 : ∀ x, f φ x = f φ (-x)) : 
  ∀ x, f φ (x - Real.pi / 3) = g φ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_equals_g_l1333_133346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_two_maxima_omega_range_l1333_133380

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

def has_two_zeros_two_maxima (ω : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ),
    Real.pi / 6 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < Real.pi / 4 ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧
    ∀ x ∈ (Set.Ioo (Real.pi / 6) (Real.pi / 4)), f ω x ≤ f ω x₃ ∧ f ω x ≤ f ω x₄ ∧
    ∀ y ∈ (Set.Ioo x₃ x₄), f ω y < f ω x₃ ∧ f ω y < f ω x₄

theorem f_two_zeros_two_maxima_omega_range (ω : ℝ) :
  ω > 0 ∧ has_two_zeros_two_maxima ω →
  ω ∈ Set.Ioo 25 (51 / 2) ∪ Set.Icc (69 / 2) 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_two_maxima_omega_range_l1333_133380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_constraints_unique_zero_point_constraints_l1333_133367

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a^x + b else -x^2 - 1

-- Part I
theorem monotonic_constraints (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) → a > 1 ∧ b ≥ -2 := by
  sorry

-- Part II
theorem unique_zero_point_constraints (b : ℝ) :
  (∃! x : ℝ, f 2 b x = 0) → b ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_constraints_unique_zero_point_constraints_l1333_133367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unilluminated_fraction_is_correct_unilluminated_fraction_approx_l1333_133310

/-- Represents a rectangular room with a light source and mirror -/
structure IlluminatedRoom where
  L : ℝ  -- Length of the room
  W : ℝ  -- Width of the room
  H : ℝ  -- Height of the room
  w : ℝ  -- Width of the mirror
  positive_dimensions : L > 0 ∧ W > 0 ∧ H > 0 ∧ w > 0
  mirror_fits : w ≤ W

/-- Calculates the fraction of unilluminated wall area in the room -/
noncomputable def unilluminated_fraction (room : IlluminatedRoom) : ℝ :=
  21.5 / 32

/-- Theorem stating that the unilluminated fraction is correct -/
theorem unilluminated_fraction_is_correct (room : IlluminatedRoom) :
  unilluminated_fraction room = 21.5 / 32 := by
  rfl

/-- Theorem stating that the unilluminated fraction is approximately 0.67 -/
theorem unilluminated_fraction_approx (room : IlluminatedRoom) :
  ∃ ε > 0, |unilluminated_fraction room - 0.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unilluminated_fraction_is_correct_unilluminated_fraction_approx_l1333_133310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1333_133363

noncomputable section

-- Define the sequence a_n
def a (n : ℕ) : ℝ := 4 * n - 2

-- Define S_n as the sum of the first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := (n * (4 * n - 2)) / 2

-- Define b_n
noncomputable def b (n : ℕ) : ℝ := 4 / (a n * a (n + 1))

-- Define T_n as the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := n / (2 * n + 1)

theorem sequence_properties :
  (a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 10) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2 * Real.sqrt (2 * S n) - 2) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (2 * n + 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1333_133363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_A_in_terms_of_B_l1333_133302

noncomputable section

variable (f g : ℝ → ℝ)

theorem determine_A_in_terms_of_B (B : ℝ) (hB : B ≠ 0) :
  ∃ A : ℝ, (∀ x, f x = A * x - 3 * B^2) ∧
           (∀ x, g x = B * x^2) ∧
           (f (g 2) = 0) →
           A = 3 * B / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_A_in_terms_of_B_l1333_133302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_rounded_to_two_decimal_places_l1333_133313

/-- Rounds a real number to the specified number of decimal places -/
noncomputable def round_to_decimal_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋) / 10^n

/-- The fraction 8/11 -/
def fraction : ℚ := 8/11

theorem fraction_rounded_to_two_decimal_places :
  round_to_decimal_places (fraction : ℝ) 2 = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_rounded_to_two_decimal_places_l1333_133313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l1333_133349

theorem eigenvalues_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]
  ∃ (k₁ k₂ : ℝ), k₁ = 3 + 2 * Real.sqrt 6 ∧ k₂ = 3 - 2 * Real.sqrt 6 ∧
    ∀ (k : ℝ), (∃ (v : Fin 2 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v) ↔ (k = k₁ ∨ k = k₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l1333_133349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1333_133399

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sine_rule : b * Real.sin A = a * Real.sin B
  condition : b * Real.sin A = a * Real.cos (B - π/6)
  side_a : a = 2
  side_c : c = 3

-- State the theorem
theorem triangle_properties (t : Triangle) : 
  t.B = π/3 ∧ 
  t.b = Real.sqrt 7 ∧ 
  Real.sin (2 * t.A - t.B) = (3 * Real.sqrt 3) / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1333_133399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_theorem_l1333_133390

/-- A rook placement is valid if every two rooks threaten at least one empty square. -/
def is_valid_placement (n : ℕ) (placement : Finset (ℕ × ℕ)) : Prop :=
  ∀ r1 r2, r1 ∈ placement → r2 ∈ placement → r1 ≠ r2 → ∃ x y, (x, y) ∉ placement ∧ 
    ((x = r1.1 ∧ y = r2.2) ∨ (x = r2.1 ∧ y = r1.2))

/-- The maximum number of rooks that can be placed on an n × n chessboard
    with a valid placement. -/
def max_rooks (n : ℕ) : ℕ :=
  if n % 2 = 0 then (3 * n - 2) / 2 else (3 * n - 1) / 2

/-- Theorem stating the maximum number of rooks for a valid placement. -/
theorem max_rooks_theorem (n : ℕ) (h : n > 0) :
  ∃ (placement : Finset (ℕ × ℕ)), 
    placement.card = max_rooks n ∧ 
    is_valid_placement n placement ∧
    ∀ (other_placement : Finset (ℕ × ℕ)), 
      is_valid_placement n other_placement → 
      other_placement.card ≤ max_rooks n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_theorem_l1333_133390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_with_three_inversions_l1333_133391

/-- The number of permutations with exactly k inversions for a set of size n -/
def num_permutations_with_inversions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- A permutation has an inversion if there exists a pair (i, j) where i < j and π[i] > π[j] -/
def has_inversion (π : Fin n → Fin n) : Prop := 
  ∃ (i j : Fin n), i < j ∧ π i > π j

/-- The number of inversions in a permutation -/
def num_inversions (π : Fin n → Fin n) : ℕ := 
  (Finset.univ.filter (λ i => ∃ j, i < j ∧ π i > π j)).card

theorem permutations_with_three_inversions (n : ℕ) (h : n ≥ 3) :
  num_permutations_with_inversions n 3 = (n * (n^2 - 7)) / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_with_three_inversions_l1333_133391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_line_equation_l1333_133378

-- Define the parabola
def Parabola (p : ℝ) := {z : ℝ × ℝ | z.2^2 = 2*p*z.1 ∧ p > 0}

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the point that the line l passes through
def PointOnL : ℝ × ℝ := (-1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem for part (I)
theorem parabola_equation (p : ℝ) :
  p = 2 ↔ ∀ (z : ℝ × ℝ), z ∈ Parabola p ↔ z.2^2 = 4*z.1 := by sorry

-- Theorem for part (II)
theorem line_equation (p : ℝ) 
  (A B : ℝ × ℝ) (hA : A ∈ Parabola p) (hB : B ∈ Parabola p) 
  (hl : ∃ (k : ℝ), A.2 = k * (A.1 + 1) ∧ B.2 = k * (B.1 + 1)) 
  (hd : distance Focus A = 2 * distance Focus B) :
  ∃ (k : ℝ), (k = 2*Real.sqrt 2/3 ∨ k = -2*Real.sqrt 2/3) ∧
    ∀ (x y : ℝ), y = k * (x + 1) ↔ (x, y) ∈ {z | ∃ (t : ℝ), z.1 = (1-t)*(-1) + t*A.1 ∧ z.2 = (1-t)*0 + t*A.2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_line_equation_l1333_133378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_range_m_for_two_roots_l1333_133335

-- Define the functions
noncomputable def f (x m : ℝ) : ℝ := x^2 - (m+1)*x + 4
noncomputable def F (x m : ℝ) : ℝ := x^2 - 2*m*x + 4
noncomputable def G (x m : ℝ) : ℝ := 2^(f x m)

-- Part I
theorem min_value_F (m : ℝ) (hm : m > 0) :
  ∃ (min_F : ℝ), ∀ x ∈ Set.Ioo 0 1, F x m ≥ min_F ∧
  (m > 1 → min_F = 5 - 2*m) ∧
  (m ≤ 1 → min_F = 4 - m^2) := by
  sorry

-- Part II
theorem range_m_for_two_roots :
  ∃ a b : ℝ, a = 3 ∧ b = 10/3 ∧
  ∀ m : ℝ, (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 ∧
    f x₁ m = 0 ∧ f x₂ m = 0) ↔ m ∈ Set.Ioo a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_range_m_for_two_roots_l1333_133335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l1333_133315

open Real

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / sin A = b / sin B ∧ b / sin B = c / sin C →
  2 * a = Real.sqrt 3 * c * sin A - a * cos C →
  c = Real.sqrt 3 →
  ∃ (S : ℝ), S = 1/2 * a * b * sin C ∧ S ≤ Real.sqrt 3/4 ∧ ∃ (a' b' : ℝ), S = Real.sqrt 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l1333_133315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_product_equality_l1333_133392

theorem inverse_sum_product_equality : 
  10 * (1/2 + 1/5 + 1/10 : ℝ)⁻¹ = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_product_equality_l1333_133392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1333_133398

noncomputable def f (x : ℝ) := Real.log (x + 1) / Real.sqrt (3^x - 27)

theorem domain_of_f : 
  Set.range f = {y | ∃ x > 3, y = f x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1333_133398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_puzzle_solution_exists_l1333_133377

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Predicate to check if a number is even -/
def isEven (n : Nat) : Prop := ∃ k, n = 2 * k

/-- Theorem stating the existence of a valid digit assignment for the division puzzle -/
theorem division_puzzle_solution_exists :
  ∃ (R S M D V : Digit),
    (R = ⟨5, by norm_num⟩) ∧
    (S.val ≠ R.val) ∧
    (M.val ≠ R.val) ∧ (M.val ≠ S.val) ∧
    (D.val ≠ R.val) ∧ (D.val ≠ S.val) ∧ (D.val ≠ M.val) ∧
    (V.val ≠ R.val) ∧ (V.val ≠ S.val) ∧ (V.val ≠ M.val) ∧ (V.val ≠ D.val) ∧
    isEven S.val ∧
    (M.val = 1 ∨ M.val = 2 ∨ M.val = 3 ∨ M.val = 4) ∧
    (D = ⟨0, by norm_num⟩) ∧
    ((R.val * 10 + S.val) / M.val = V.val) ∧
    ((R.val * 10 + S.val) % M.val = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_puzzle_solution_exists_l1333_133377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_condition_l1333_133307

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex expression -/
noncomputable def complexExpr (a : ℝ) : ℂ := (3 + i) * (a + 2*i) / (1 + i)

theorem real_part_condition (a : ℝ) : 
  (complexExpr a).im = 0 → a = 4 := by
  sorry

#check real_part_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_condition_l1333_133307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_fouls_per_game_l1333_133359

/-- The number of times John gets fouled per game -/
noncomputable def fouls_per_game (
  free_throw_accuracy : ℝ)
  (shots_per_foul : ℕ)
  (games_played_percentage : ℝ)
  (total_games : ℕ)
  (total_free_throws : ℕ) : ℝ :=
  (total_free_throws : ℝ) / ((games_played_percentage * (total_games : ℝ)) * (shots_per_foul : ℝ))

/-- Theorem stating that John gets fouled 3.5 times per game -/
theorem john_fouls_per_game :
  fouls_per_game 0.7 2 0.8 20 112 = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_fouls_per_game_l1333_133359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_passes_through_solution_parallel_and_equal_intercepts_solution_l1333_133393

/-- Two lines in the plane -/
structure Lines where
  l₁ : ℝ → ℝ → ℝ
  l₂ : ℝ → ℝ → ℝ

/-- The lines are perpendicular -/
def perpendicular (L : Lines) : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ, L.l₁ x y = (a * x - b * y + 4) ∧
             L.l₂ x y = ((a - 1) * x + y + b) ∧
             a * (a - 1) + (-b) * 1 = 0

/-- The first line passes through (-3, -1) -/
def passes_through (L : Lines) : Prop :=
  L.l₁ (-3) (-1) = 0

/-- The lines are parallel -/
def parallel (L : Lines) : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ, L.l₁ x y = (a * x - b * y + 4) ∧
             L.l₂ x y = ((a - 1) * x + y + b) ∧
             a * 1 - (-b) * (a - 1) = 0

/-- The first line has equal intercepts on both axes -/
def equal_intercepts (L : Lines) : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ, L.l₁ x y = (a * x - b * y + 4) ∧
             (4 / b = -4 / a)

theorem perpendicular_and_passes_through_solution (L : Lines) :
  perpendicular L ∧ passes_through L → ∃ a b : ℝ, a = 2 ∧ b = 2 := by sorry

theorem parallel_and_equal_intercepts_solution (L : Lines) :
  parallel L ∧ equal_intercepts L → ∃ a b : ℝ, a = 2 ∧ b = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_passes_through_solution_parallel_and_equal_intercepts_solution_l1333_133393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_time_theorem_l1333_133312

/-- Represents the escalator with given properties -/
structure Escalator where
  total_length : ℝ
  speed1 : ℝ
  speed2 : ℝ
  speed3 : ℝ
  stop_interval : ℝ
  stop_duration : ℝ

/-- Represents a person walking on the escalator -/
structure Person where
  walking_speed : ℝ
  resume_time : ℝ

/-- Calculates the time taken to cover a segment of the escalator -/
noncomputable def time_for_segment (e : Escalator) (p : Person) (segment_length : ℝ) (escalator_speed : ℝ) : ℝ :=
  let combined_speed := escalator_speed + p.walking_speed
  let movement_time := segment_length / combined_speed
  let num_stops := segment_length / e.stop_interval
  let stop_time := num_stops * (e.stop_duration + p.resume_time)
  movement_time + stop_time

/-- Calculates the total time taken to cover the entire escalator -/
noncomputable def total_time (e : Escalator) (p : Person) : ℝ :=
  time_for_segment e p 100 e.speed1 +
  time_for_segment e p 100 e.speed2 +
  time_for_segment e p 100 e.speed3

/-- The main theorem stating the time taken is approximately 49.83 seconds -/
theorem escalator_time_theorem (e : Escalator) (p : Person)
  (h1 : e.total_length = 300)
  (h2 : e.speed1 = 30)
  (h3 : e.speed2 = 20)
  (h4 : e.speed3 = 40)
  (h5 : e.stop_interval = 50)
  (h6 : e.stop_duration = 5)
  (h7 : p.walking_speed = 10)
  (h8 : p.resume_time = 2) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |total_time e p - 49.83| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_time_theorem_l1333_133312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_double_l1333_133397

/-- The period of tan(2x) is π/2. -/
theorem period_of_tan_double (x : ℝ) : 
  ∃ p : ℝ, p > 0 ∧ ∀ t : ℝ, Real.tan (2 * (x + p)) = Real.tan (2 * x) ∧ 
  ∀ q : ℝ, 0 < q ∧ q < p → ∃ t : ℝ, Real.tan (2 * (x + q)) ≠ Real.tan (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_double_l1333_133397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1333_133353

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*x - 3 else Real.log x

-- State the theorem
theorem f_inequality_range (a : ℝ) :
  (∀ x, |f x| + 1 ≥ a * x) ↔ -8 ≤ a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1333_133353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_bounds_l1333_133385

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

-- State the theorem
theorem zeros_sum_bounds (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ = 0 → f a x₂ = 0 →
  2 < x₁ + x₂ ∧ x₁ + x₂ < 3 * Real.exp (a - 1) - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_bounds_l1333_133385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_december_sales_fraction_is_seven_eighteenths_l1333_133316

/-- Represents the sales data for a department store -/
structure DepartmentStore where
  avg_monthly_sales : ℚ  -- Average monthly sales from Jan-Nov
  dec_sales_multiplier : ℚ  -- Multiplier for December sales

/-- Calculates the fraction of December sales out of total yearly sales -/
def december_sales_fraction (store : DepartmentStore) : ℚ :=
  let jan_nov_sales := 11 * store.avg_monthly_sales
  let dec_sales := store.dec_sales_multiplier * store.avg_monthly_sales
  let total_sales := jan_nov_sales + dec_sales
  dec_sales / total_sales

/-- Theorem: For a department store where December sales are 7 times the average
    monthly sales from January through November, the fraction of December sales
    out of total yearly sales is 7/18 -/
theorem december_sales_fraction_is_seven_eighteenths
  (store : DepartmentStore)
  (h : store.dec_sales_multiplier = 7) :
  december_sales_fraction store = 7 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_december_sales_fraction_is_seven_eighteenths_l1333_133316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1333_133350

theorem complex_number_problem (α : ℂ) 
  (h1 : α ≠ 1)
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^3 - 1) = 8 * Complex.abs (α - 1))
  (h4 : Complex.arg α = π / 6) :
  α = Complex.ofReal (Real.sqrt 3 / 2) + Complex.I * (1 / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1333_133350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_multiple_of_180_l1333_133342

/-- The number of natural-number factors of 2^12 * 3^15 * 5^9 that are multiples of 180 -/
def factors_multiple_of_180 : ℕ := 1386

/-- The prime factorization of the given number n -/
def n : ℕ := 2^12 * 3^15 * 5^9

/-- Theorem stating that the number of natural-number factors of n that are multiples of 180 is equal to factors_multiple_of_180 -/
theorem count_factors_multiple_of_180 :
  (Finset.filter (λ x ↦ x ∣ n ∧ 180 ∣ x) (Finset.range (n + 1))).card = factors_multiple_of_180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_multiple_of_180_l1333_133342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_flight_distance_specific_max_distance_l1333_133383

/-- Represents the total number of planes -/
def total_planes : ℕ := 100

/-- Represents the distance a single plane can fly with a full tank (in km) -/
def single_plane_range : ℕ := 1000

/-- Theorem stating the maximum distance a single plane can fly given the conditions -/
theorem max_flight_distance :
  ∃ (max_distance : ℕ), 
    max_distance = total_planes * single_plane_range ∧
    ∀ (d : ℕ), d > max_distance → 
      ¬(∃ (strategy : Unit), d ≤ total_planes * single_plane_range) :=
by sorry

/-- Corollary stating the specific maximum distance of 100,000 km -/
theorem specific_max_distance :
  ∃ (max_distance : ℕ), max_distance = 100000 ∧
    ∀ (d : ℕ), d > 100000 → 
      ¬(∃ (strategy : Unit), d ≤ total_planes * single_plane_range) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_flight_distance_specific_max_distance_l1333_133383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_distance_l1333_133345

/-- Represents the distance to the park -/
noncomputable def initial_distance : ℝ := 3

/-- Represents the fraction of remaining distance walked each time -/
noncomputable def walk_fraction : ℝ := 2/3

/-- Calculates the position after walking towards the park -/
noncomputable def walk_to_park (prev_pos : ℝ) : ℝ :=
  prev_pos + walk_fraction * (initial_distance - prev_pos)

/-- Calculates the position after walking towards the apartment -/
noncomputable def walk_to_apartment (prev_pos : ℝ) : ℝ :=
  prev_pos - walk_fraction * prev_pos

/-- Represents the point closest to the apartment where oscillation occurs -/
noncomputable def point_C : ℝ := 5/4

/-- Represents the point closest to the park where oscillation occurs -/
noncomputable def point_D : ℝ := 11/4

/-- Theorem stating the distance between oscillation points -/
theorem oscillation_distance :
  |point_D - point_C| = 3/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_distance_l1333_133345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetric_about_5pi_4_l1333_133351

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem not_symmetric_about_5pi_4 :
  ¬ (∀ (x : ℝ), f (5 * Real.pi / 4 + x) = f (5 * Real.pi / 4 - x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetric_about_5pi_4_l1333_133351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l1333_133382

/-- A quadratic function -/
def quadratic_function (a b c : ℚ) : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c

/-- The specific quadratic function we're proving about -/
def specific_quadratic : ℚ → ℚ := λ x ↦ -(1/2) * (x+1)^2 + 9/2

theorem quadratic_function_proof :
  ∃ a b c : ℚ, 
    a ≠ 0 ∧
    quadratic_function a b c (-3) = 5/2 ∧
    quadratic_function a b c (-2) = 4 ∧
    quadratic_function a b c (-1) = 9/2 ∧
    quadratic_function a b c 0 = 4 ∧
    quadratic_function a b c 1 = 5/2 ∧
    ∀ x : ℚ, quadratic_function a b c x = specific_quadratic x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l1333_133382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andre_caught_eight_ladybugs_on_monday_l1333_133379

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := sorry

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots for all ladybugs -/
def total_dots : ℕ := 78

/-- Theorem stating that Andre caught 8 ladybugs on Monday -/
theorem andre_caught_eight_ladybugs_on_monday :
  monday_ladybugs = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andre_caught_eight_ladybugs_on_monday_l1333_133379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womens_tshirt_interval_l1333_133364

/-- Represents the shop's T-shirt sales and revenue --/
structure TShirtShop where
  womens_price : ℚ
  mens_price : ℚ
  mens_interval : ℚ
  daily_open_minutes : ℚ
  weekly_revenue : ℚ

/-- Calculates the weekly revenue for the shop given the women's T-shirt sale interval --/
def weekly_revenue (shop : TShirtShop) (womens_interval : ℚ) : ℚ :=
  7 * (shop.womens_price * (shop.daily_open_minutes / womens_interval) +
       shop.mens_price * (shop.daily_open_minutes / shop.mens_interval))

/-- Theorem stating that women's T-shirts are sold every 30 minutes --/
theorem womens_tshirt_interval (shop : TShirtShop) 
  (h1 : shop.womens_price = 18)
  (h2 : shop.mens_price = 15)
  (h3 : shop.mens_interval = 40)
  (h4 : shop.daily_open_minutes = 720)
  (h5 : shop.weekly_revenue = 4914) :
  ∃ x : ℚ, x = 30 ∧ weekly_revenue shop x = shop.weekly_revenue := by
  sorry

#check womens_tshirt_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_womens_tshirt_interval_l1333_133364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1333_133338

/-- The time it takes for worker B to complete the work independently -/
noncomputable def time_B : ℝ := 12

/-- The time it takes for worker A to complete the work independently -/
noncomputable def time_A : ℝ := time_B / 2

/-- The time it takes for workers A and B to complete the work together -/
noncomputable def time_AB : ℝ := 4

theorem work_completion_time :
  (time_A = time_B / 2) →
  (1 / time_A + 1 / time_B) * time_AB = 1 →
  time_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1333_133338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_when_g_has_solution_range_of_a_when_f_equals_g_l1333_133352

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x^2 + 3)
noncomputable def g (x a : ℝ) : ℝ := (1/2) * x^2 - Real.log x - a

-- Theorem 1: Range of f(x)
theorem range_of_f :
  ∀ y ∈ Set.range f,
  (0 < y ∧ y ≤ 2/3) ∧
  ∀ x ∈ Set.Ioo 0 2, ∃ y' ∈ Set.Ioo 0 (2/3), f x = y' :=
by sorry

-- Theorem 2: Range of a when g(x) = 0 has a solution
theorem range_of_a_when_g_has_solution :
  ∀ a : ℝ,
  (∃ x ∈ Set.Icc 1 2, g x a = 0) →
  (1/2 ≤ a ∧ a ≤ 2 - Real.log 2) :=
by sorry

-- Theorem 3: Range of a when f(x1) = g(x2) for any x1 and some x2
theorem range_of_a_when_f_equals_g :
  ∀ a : ℝ,
  (∀ x1 ∈ Set.Ioo 0 2, ∃ x2 ∈ Set.Icc 1 2, f x1 = g x2 a) →
  (1/2 ≤ a ∧ a ≤ 4/3 - Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_when_g_has_solution_range_of_a_when_f_equals_g_l1333_133352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l1333_133387

-- Define the circle
noncomputable def circle_center : ℝ × ℝ := (Real.sqrt 3, 0)
def circle_radius : ℝ := 2

-- Define the line
noncomputable def line_slope : ℝ := Real.sqrt 3 / 3

-- Theorem statement
theorem intersection_chord_length :
  let center := circle_center
  let radius := circle_radius
  let m := line_slope
  let d := abs (center.2 - m * center.1) / Real.sqrt (1 + m^2)
  2 * Real.sqrt (radius^2 - d^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l1333_133387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projected_pentagon_sides_l1333_133306

/-- A planar convex pentagon that is an orthogonal projection of a regular pentagon -/
structure ProjectedPentagon where
  -- Three consecutive sides
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- Condition that it's a projection of a regular pentagon
  is_projection : Bool

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Theorem about the remaining sides of the projected pentagon -/
theorem projected_pentagon_sides (p : ProjectedPentagon) 
  (h1 : p.side1 = 1) 
  (h2 : p.side2 = 2) 
  (h3 : p.is_projection = true) :
  ∃ (side4 side5 : ℝ),
    side4 = (Real.sqrt 5 - 1) / 4 * Real.sqrt (14 + 10 * Real.sqrt 5 - 2 * (Real.sqrt 5 + 1) * p.side3^2) ∧
    side5 = (Real.sqrt 5 - 1) / 4 * Real.sqrt (p.side3^2 * (6 + 2 * Real.sqrt 5) + (6 * Real.sqrt 5 + 1)) ∧
    Real.sqrt 5 - 2 < p.side3 ∧ p.side3 < Real.sqrt 5 :=
by sorry

#check projected_pentagon_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projected_pentagon_sides_l1333_133306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1333_133327

/-- Two circles in a plane -/
structure TwoCircles where
  m : ℝ
  O₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 5}
  O₂ : Set (ℝ × ℝ) := {p | (p.1 + m)^2 + p.2^2 = 20}

/-- The intersection points of two circles -/
def intersection (c : TwoCircles) : Set (ℝ × ℝ) :=
  c.O₁ ∩ c.O₂

/-- The condition that tangents at an intersection point are perpendicular -/
def perpendicular_tangents (c : TwoCircles) : Prop :=
  ∃ A : ℝ × ℝ, A ∈ intersection c ∧ 
    let O₁ : ℝ × ℝ := (0, 0)
    let O₂ : ℝ × ℝ := (-c.m, 0)
    (A.1 * (A.1 + c.m) + A.2 * A.2 = 0)

/-- The theorem to be proved -/
theorem intersection_distance (c : TwoCircles) 
  (h : perpendicular_tangents c) : 
  ∃ A B : ℝ × ℝ, A ∈ intersection c ∧ B ∈ intersection c ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1333_133327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_is_fraction_l1333_133318

def sequence_a : ℕ → ℚ
  | 0 => 2
  | 1 => 7/5
  | (n+2) => (sequence_a n * sequence_a (n+1)) / (3 * sequence_a n - sequence_a (n+1))

theorem a_100_is_fraction :
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ Nat.Coprime p q ∧ sequence_a 99 = p / q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_is_fraction_l1333_133318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_four_l1333_133374

/-- A three-digit positive integer with a ones digit of 7 -/
def ThreeDigitIntegerEndingIn7 : Type :=
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 7 }

/-- The probability that a ThreeDigitIntegerEndingIn7 is divisible by 4 -/
theorem probability_divisible_by_four :
  (Finset.filter (fun n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 7 ∧ n % 4 = 0) (Finset.range 1000)).card /
  (Finset.filter (fun n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 7) (Finset.range 1000)).card = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_four_l1333_133374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_lengths_l1333_133395

-- Define the points A, B, C as real numbers
def Point : Type := ℝ

-- Define the distance function
def distance (p q : ℝ) : ℝ := |p - q|

-- State the theorem
theorem line_segment_lengths 
  (A B C : ℝ) 
  (h_line : ∃ (t : ℝ), C = A + t * (B - A)) -- A, B, C are collinear
  (h_AB : distance A B = 3)
  (h_BC : distance B C = 5) :
  distance A C = 2 ∨ distance A C = 8 :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_lengths_l1333_133395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_speed_B_to_C_l1333_133324

/-- Represents the journey of a motorcyclist between highway markers --/
structure Journey where
  distanceAB : ℝ
  distanceBC : ℝ
  timeAB : ℝ
  timeBC : ℝ
  avgSpeedTotal : ℝ

/-- Calculates the average speed between two points given distance and time --/
noncomputable def avgSpeed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem stating the average speed from B to C given the journey conditions --/
theorem avg_speed_B_to_C (j : Journey)
    (h1 : j.distanceAB = 120)
    (h2 : j.distanceBC = j.distanceAB / 2)
    (h3 : j.timeAB = 3 * j.timeBC)
    (h4 : j.avgSpeedTotal = 30)
    (h5 : j.avgSpeedTotal = (j.distanceAB + j.distanceBC) / (j.timeAB + j.timeBC)) :
    avgSpeed j.distanceBC j.timeBC = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_speed_B_to_C_l1333_133324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1333_133323

noncomputable def function (a x : ℝ) : ℝ := a^(x-1) + 1

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_ne : a ≠ 1) (hmn : m * n > 0) :
  function a 1 = 2 →
  2 * m + n * 2 - 4 = 0 →
  (∀ m' n', m' * n' > 0 → 2 * m' + n' * 2 - 4 = 0 → 4 / m' + 2 / n' ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ m' n', m' * n' > 0 ∧ 2 * m' + n' * 2 - 4 = 0 ∧ 4 / m' + 2 / n' = 3 + 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1333_133323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_amount_is_40_l1333_133354

/-- The lower amount for renting a room --/
def lower_amount : ℕ := sorry

/-- The number of rooms that were rented for $60 but could have been rented for the lower amount --/
def num_rooms : ℕ := 10

/-- The higher rental price --/
def higher_price : ℕ := 60

/-- The total rent charged for that night --/
def total_rent : ℕ := 2000

/-- The percentage reduction in total rent if the rooms were rented at the lower amount --/
def reduction_percentage : ℚ := 1 / 10

theorem lower_amount_is_40 :
  (num_rooms * (higher_price - lower_amount) = total_rent * reduction_percentage) →
  lower_amount = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_amount_is_40_l1333_133354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_max_area_l1333_133394

/-- The semiperimeter of a triangle -/
noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

/-- The area of a triangle using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: Among all triangles with a fixed semiperimeter, 
    the equilateral triangle has the largest area -/
theorem equilateral_triangle_max_area (p : ℝ) (hp : p > 0) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → semiperimeter a b c = p →
    triangle_area a b c ≤ p^2 / (3 * Real.sqrt 3) ∧
    (triangle_area a b c = p^2 / (3 * Real.sqrt 3) ↔ a = b ∧ b = c) := by
  sorry

#check equilateral_triangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_max_area_l1333_133394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_valid_l1333_133300

-- Define the differential equation
def diff_eq (x y : ℝ) (dx dy : ℝ) : Prop :=
  2 * x * y * Real.log y * dx + (x^2 + y^2 * Real.sqrt (y^2 + 1)) * dy = 0

-- Define the solution function
noncomputable def F (x y : ℝ) : ℝ :=
  3 * x^2 * Real.log y + Real.sqrt ((y^2 + 1)^3)

-- Theorem statement
theorem solution_valid (x y : ℝ) (h : y > 0) :
  ∃ (C : ℝ), F x y = C ∧ 
  ∀ (dx dy : ℝ), diff_eq x y dx dy → 
    (deriv (fun x => F x y) x) * dx + (deriv (fun y => F x y) y) * dy = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_valid_l1333_133300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_is_14_60_l1333_133337

/-- Calculates the total savings for Trip and his friends by choosing the earlier movie time. -/
def calculate_total_savings (
  evening_ticket_cost : ℚ)
  (large_popcorn_drink_cost : ℚ)
  (medium_nachos_drink_cost : ℚ)
  (hotdog_drink_cost : ℚ)
  (ticket_discount : ℚ)
  (large_popcorn_drink_discount : ℚ)
  (medium_nachos_drink_discount : ℚ)
  (hotdog_drink_discount : ℚ)
  (group_size : ℕ) : ℚ :=
  let ticket_savings := evening_ticket_cost * (group_size : ℚ) * ticket_discount
  let large_popcorn_drink_savings := large_popcorn_drink_cost * large_popcorn_drink_discount
  let medium_nachos_drink_savings := medium_nachos_drink_cost * medium_nachos_drink_discount
  let hotdog_drink_savings := hotdog_drink_cost * hotdog_drink_discount
  ticket_savings + large_popcorn_drink_savings + medium_nachos_drink_savings + hotdog_drink_savings

/-- Theorem stating that the total savings for Trip and his friends is $14.60. -/
theorem total_savings_is_14_60 :
  calculate_total_savings 10 10 8 6 (1/5) (1/2) (3/10) (1/5) 3 = 73/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_is_14_60_l1333_133337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_even_l1333_133368

noncomputable section

variable (a : ℝ)

axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1

variable (F : ℝ → ℝ)
axiom F_odd : ∀ x, F (-x) = -F x

def G (x : ℝ) : ℝ := F x * (1 / (a^x - 1) + 1/2)

theorem G_is_even : ∀ x, G (-x) = G x := by
  intro x
  -- The proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_even_l1333_133368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_narrower_width_better_fit_stronger_correlation_comparison_smaller_R_squared_worse_fit_probability_calculation_l1333_133381

-- Define the residual distribution width and model fitting effect
noncomputable def residual_width : ℝ → ℝ := sorry
noncomputable def fitting_effect : ℝ → ℝ := sorry

-- Define the correlation coefficient and its absolute value
noncomputable def correlation_coefficient : ℝ → ℝ := sorry
noncomputable def abs_correlation : ℝ → ℝ := sorry

-- Define the coefficient of determination (R²)
noncomputable def R_squared : ℝ → ℝ := sorry

-- Define the probability calculation function
noncomputable def probability_one_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ := sorry

-- Statement 1
theorem narrower_width_better_fit (w1 w2 : ℝ) :
  w1 < w2 → fitting_effect (residual_width w1) > fitting_effect (residual_width w2) := by
  sorry

-- Statement 2
theorem stronger_correlation_comparison :
  abs_correlation (correlation_coefficient (-0.99)) > abs_correlation (correlation_coefficient 0.97) := by
  sorry

-- Statement 3
theorem smaller_R_squared_worse_fit (r1 r2 : ℝ) :
  R_squared r1 < R_squared r2 → fitting_effect r1 < fitting_effect r2 := by
  sorry

-- Statement 4
theorem probability_calculation :
  probability_one_defective 10 3 2 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_narrower_width_better_fit_stronger_correlation_comparison_smaller_R_squared_worse_fit_probability_calculation_l1333_133381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_position_1500_l1333_133305

/-- Andy's position after a certain number of turns -/
structure Position where
  x : Int
  y : Int

/-- Andy's initial position -/
def initial_position : Position := ⟨-30, 25⟩

/-- Calculate Andy's position after n turns -/
def andy_position (n : Nat) : Position :=
  let k := (n - 1) / 4
  let x := initial_position.x - (4 * k + 1) * k / 2
  let y := initial_position.y + (4 * k - 1) * k / 2
  let final_adjustment : Position := 
    match n % 4 with
    | 0 => ⟨0, n⟩
    | 1 => ⟨n, 0⟩
    | 2 => ⟨0, -n⟩
    | _ => ⟨-n, 0⟩
  ⟨x + final_adjustment.x, y + final_adjustment.y⟩

/-- Theorem: Andy's position after 1500 turns -/
theorem andy_position_1500 : andy_position 1500 = ⟨-280141, 280060⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_position_1500_l1333_133305
