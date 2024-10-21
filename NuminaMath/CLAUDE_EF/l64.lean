import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_46_l64_6473

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ :=
  2 * (x^2 - 2*x^3 + x) + 4 * (x + 3*x^3 - 2*x^2 + 2*x^5 + 2*x^3) - 6 * (2 + x - 5*x^3 - x^2)

-- Theorem stating that the coefficient of x^3 in the polynomial is 46
theorem coefficient_of_x_cubed_is_46 :
  ∃ (a b c d e : ℝ), ∀ x, polynomial x = a*x^5 + b*x^4 + 46*x^3 + c*x^2 + d*x + e :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_46_l64_6473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l64_6498

/-- Function representing y₁ which is directly proportional to (x+1) -/
noncomputable def y₁ (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)

/-- Function representing y₂ which is inversely proportional to (x+1) -/
noncomputable def y₂ (b : ℝ) (x : ℝ) : ℝ := b / (x + 1)

/-- Function representing y as the sum of y₁ and y₂ -/
noncomputable def y (a b x : ℝ) : ℝ := y₁ a x + y₂ b x

theorem solve_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (y a b 0 = -5) →
  (y a b 2 = -7) →
  (∃ x : ℝ, y a b x = 5 ∧ (x = -2 ∨ x = -5/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l64_6498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l64_6486

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x - 4 * a else a * x

-- Define what it means for f to be increasing on ℝ
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  isIncreasing (f a) → a ∈ Set.Icc (1/3) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l64_6486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_nine_equals_plus_minus_three_l64_6475

theorem square_root_nine_equals_plus_minus_three :
  (∀ x, x^2 = 9 → x = 3 ∨ x = -3) ∧
  ¬(Real.sqrt ((-2)^2) = -2) ∧
  ¬(-Real.sqrt (3^2) = 3) ∧
  (∃ x, x^3 = -9 ∧ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_nine_equals_plus_minus_three_l64_6475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_to_exceed_ten_billion_forty_six_is_minimum_l64_6457

/-- The growth function for the number of cells over time -/
noncomputable def cell_growth (t : ℕ) : ℝ := 100 * (3/2)^t

/-- The theorem stating that 46 hours is the minimum time to exceed 10^10 cells -/
theorem min_time_to_exceed_ten_billion :
  ∀ t : ℕ, t < 46 → cell_growth t ≤ 10^10 ∧
  cell_growth 46 > 10^10 := by
  sorry

/-- The theorem stating that 46 is indeed the minimum integer satisfying the condition -/
theorem forty_six_is_minimum :
  ∀ t : ℕ, cell_growth t > 10^10 → t ≥ 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_to_exceed_ten_billion_forty_six_is_minimum_l64_6457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_10_5_l64_6436

/-- Represents the running/walking data for Cheenu at different ages --/
structure CheenuData where
  young_miles : ℚ
  young_hours : ℚ
  old_miles : ℚ
  old_hours : ℚ
  old_rest_minutes : ℚ

/-- Calculates the difference in minutes per mile between older and younger Cheenu --/
def timeDifference (data : CheenuData) : ℚ :=
  let young_time_per_mile := (data.young_hours * 60) / data.young_miles
  let old_effective_minutes := data.old_hours * 60 - data.old_rest_minutes
  let old_time_per_mile := old_effective_minutes / data.old_miles
  old_time_per_mile - young_time_per_mile

/-- Theorem stating the time difference is 10.5 minutes --/
theorem time_difference_is_10_5 (data : CheenuData) 
  (h1 : data.young_miles = 20)
  (h2 : data.young_hours = 4)
  (h3 : data.old_miles = 12)
  (h4 : data.old_hours = 5)
  (h5 : data.old_rest_minutes = 30) :
  timeDifference data = 21/2 := by
  sorry

#eval timeDifference { young_miles := 20, young_hours := 4, old_miles := 12, old_hours := 5, old_rest_minutes := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_10_5_l64_6436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_trip_speed_l64_6454

/-- Proves that given a 250 km trip where the first 100 km is traveled at 20 km/h
    and the average speed for the entire trip is 16.67 km/h,
    the speed for the remainder of the distance is 15 km/h. -/
theorem bicycle_trip_speed (total_distance : ℝ) (first_part_distance : ℝ) 
  (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 250 →
  first_part_distance = 100 →
  first_part_speed = 20 →
  average_speed = 16.67 →
  (let remainder_distance := total_distance - first_part_distance
   let total_time := total_distance / average_speed
   let first_part_time := first_part_distance / first_part_speed
   let remainder_time := total_time - first_part_time
   let remainder_speed := remainder_distance / remainder_time
   remainder_speed) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_trip_speed_l64_6454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_period_l64_6487

/-- The minimum positive period of the cosine function with coefficient 2/5 -/
theorem cosine_period : ∃ (p : ℝ), p > 0 ∧ 
  (∀ x t : ℝ, 2 * Real.cos ((2/5 : ℝ) * (x + p) - π/3) = 2 * Real.cos ((2/5 : ℝ) * x - π/3)) ∧ 
  (∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, 2 * Real.cos ((2/5 : ℝ) * (x + q) - π/3) ≠ 2 * Real.cos ((2/5 : ℝ) * x - π/3)) ∧
  p = 5 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_period_l64_6487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_2015_l64_6449

/-- The units digit of 2^n - 1 for positive integer n -/
def unitsDigit (n : ℕ) : Fin 10 :=
  match n % 4 with
  | 0 => 5
  | 1 => 1
  | 2 => 3
  | 3 => 7
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

theorem units_digit_2015 : unitsDigit 2015 = 7 := by
  -- Proof goes here
  sorry

#eval unitsDigit 2015  -- This will evaluate to 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_2015_l64_6449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l64_6472

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line
def my_line (x y : ℝ) (a : ℝ) : Prop := 3*x + y + a = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop := ∀ x y : ℝ, my_circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 2*h - 4*k)

-- Theorem statement
theorem line_through_circle_center (a : ℝ) : 
  (∃ h k : ℝ, is_center h k ∧ my_line h k a) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l64_6472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alan_cd_purchase_cost_l64_6424

/-- Represents the total cost of CDs Alan buys -/
noncomputable def total_cost (price_avn : ℝ) (num_dark : ℕ) (num_avn : ℕ) (num_90s : ℕ) : ℝ :=
  let price_dark := 2 * price_avn
  let cost_dark_avn := num_dark * price_dark + num_avn * price_avn
  let cost_90s := (2/5) * cost_dark_avn
  cost_dark_avn + cost_90s

/-- Theorem stating that the total cost of Alan's CD purchase is $84 -/
theorem alan_cd_purchase_cost :
  total_cost 12 2 1 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alan_cd_purchase_cost_l64_6424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l64_6413

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6)

theorem symmetry_axis_of_g :
  ∀ x : ℝ, g (Real.pi / 3 + x) = g (Real.pi / 3 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l64_6413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l64_6462

/-- The quadratic equation x² - mx + 2m = 0 has two roots, 
    with one root greater than 3 and the other less than 3, 
    if and only if m > 9 -/
theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - m*x₁ + 2*m = 0 ∧ 
    x₂^2 - m*x₂ + 2*m = 0 ∧ 
    x₁ > 3 ∧ x₂ < 3) ↔ 
  m > 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l64_6462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l64_6442

noncomputable def triangle_ABC (A B C D : ℝ × ℝ) : Prop :=
  let BC := C.1 - B.1
  let AD := D.1 - A.1
  3 * (D.1 - B.1) = 2 * (C.1 - D.1) ∧ 
  AD * BC = 0 ∧
  (1/2) * BC * AD = (3/5) * BC^2

noncomputable def angle_BAC (A B C : ℝ × ℝ) : ℝ := sorry

noncomputable def y (x : ℝ) (A B C : ℝ × ℝ) : ℝ :=
  2 * Real.sin x + Real.sqrt 2 * Real.cos (x + angle_BAC A B C)

theorem range_of_y (A B C D: ℝ × ℝ) :
  triangle_ABC A B C D →
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
  1 ≤ y x A B C ∧ y x A B C ≤ Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l64_6442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l64_6471

noncomputable def f (x : ℝ) := (9 : ℝ)^x - 3^(x+1) - 1

theorem f_max_min (x : ℝ) 
  (h1 : (1/2 : ℝ)^x ≤ 4) 
  (h2 : Real.log x / Real.log (Real.sqrt 3) ≤ 2) : 
  (∃ y, f y = 647 ∧ ∀ z, f z ≤ f y) ∧ 
  (∃ y, f y = -13/4 ∧ ∀ z, f z ≥ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l64_6471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l64_6416

noncomputable def f (x : ℝ) := 1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 3))

theorem inequality_solution :
  ∀ x : ℝ, f x < 1/4 ↔ x ∈ Set.union (Set.Iio (-3)) (Set.union (Set.Ioo (-1) 0) (Set.Ioi 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l64_6416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_for_eccentricity_3_l64_6461

noncomputable section

/-- The equation of the asymptote of a hyperbola -/
def asymptote_equation (a b : ℝ) (x : ℝ) : ℝ := (a / b) * x

/-- The eccentricity of a hyperbola -/
def eccentricity (c a : ℝ) : ℝ := c / a

theorem hyperbola_asymptote_for_eccentricity_3 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (eccentricity (Real.sqrt (a^2 + b^2)) a = 3) →
  (∃ k : ℝ, k = Real.sqrt 2 / 4 ∧ 
    ∀ x : ℝ, asymptote_equation a b x = k * x ∨ asymptote_equation a b x = -k * x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_for_eccentricity_3_l64_6461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_is_line_l64_6401

/-- A curve in the xy-plane -/
structure Curve where
  -- The predicate that defines the curve
  contains : ℝ × ℝ → Prop

/-- The angle between a vector and the positive y-axis -/
noncomputable def angle_with_y_axis (p : ℝ × ℝ) : ℝ :=
  Real.arctan (p.1 / p.2)

/-- A curve defined by a constant angle with the y-axis -/
def constant_angle_curve (θ : ℝ) : Curve where
  contains := λ p => angle_with_y_axis p = θ

/-- A straight line through the origin -/
def line_through_origin (m : ℝ) : Curve where
  contains := λ p => p.2 = m * p.1

/-- Theorem: A curve defined by a constant angle with the y-axis is a straight line through the origin -/
theorem constant_angle_is_line (θ : ℝ) :
  ∃ m : ℝ, constant_angle_curve θ = line_through_origin m := by
  sorry

#check constant_angle_is_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_is_line_l64_6401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_u_plus_v_sixth_power_l64_6478

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of i -/
theorem i_squared : i^2 = -1 := Complex.I_sq

/-- Definition of u -/
noncomputable def u : ℂ := (-1 + i * Real.sqrt 7) / 2

/-- Definition of v -/
noncomputable def v : ℂ := (-1 - i * Real.sqrt 7) / 2

/-- Theorem: u^6 + v^6 = 2 -/
theorem u_plus_v_sixth_power : u^6 + v^6 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_u_plus_v_sixth_power_l64_6478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l64_6421

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 243) : (3 : ℝ)^(y+3) = 6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l64_6421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_specific_triangle_l64_6431

/-- Given a triangle PQR with side lengths p, q, r, this function calculates
    the coefficients u, v, w for expressing the incenter J as a linear
    combination of the position vectors of P, Q, and R. -/
noncomputable def incenter_coefficients (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

/-- The theorem states that for a triangle with side lengths 8, 10, and 6,
    the incenter can be expressed as J = (1/3)*P + (5/12)*Q + (1/4)*R. -/
theorem incenter_specific_triangle :
  let (u, v, w) := incenter_coefficients 8 10 6
  u = 1/3 ∧ v = 5/12 ∧ w = 1/4 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_specific_triangle_l64_6431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_with_inclination_l64_6430

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the inclination angle
noncomputable def angle : ℝ := Real.pi / 4  -- 45° in radians

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem line_through_P_with_inclination : 
  (line_equation P.1 P.2) ∧ 
  (∀ (x y : ℝ), line_equation x y → (y - P.2) = (Real.tan angle) * (x - P.1)) := by
  sorry

#check line_through_P_with_inclination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_with_inclination_l64_6430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_arrangements_l64_6422

/-- A seating arrangement for 3 boys and 3 girls in a 2x3 grid. -/
structure SeatingArrangement :=
  (grid : Fin 2 → Fin 3 → Bool)

/-- Two seats are neighbors if they are directly adjacent horizontally or vertically. -/
def are_neighbors (r1 c1 r2 c2 : Nat) : Bool :=
  (r1 = r2 ∧ (c1 = c2 + 1 ∨ c2 = c1 + 1)) ∨ 
  (c1 = c2 ∧ (r1 = r2 + 1 ∨ r2 = r1 + 1))

/-- Checks if all boys are seated next to each other in the given arrangement. -/
def all_boys_adjacent (arr : SeatingArrangement) : Bool :=
  sorry

/-- The set of all valid seating arrangements. -/
def valid_arrangements : Finset SeatingArrangement :=
  sorry

/-- The main theorem stating the number of valid seating arrangements. -/
theorem num_valid_arrangements : 
  Finset.card valid_arrangements = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_arrangements_l64_6422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_two_years_l64_6402

/-- Sam's current age -/
def s : ℕ := sorry

/-- Anna's current age -/
def a : ℕ := sorry

/-- The number of years until the ratio of their ages is 3/2 -/
def x : ℕ := sorry

/-- Sam's age two years ago was twice Anna's age two years ago -/
axiom past_condition_1 : s - 2 = 2 * (a - 2)

/-- Sam's age four years ago was three times Anna's age four years ago -/
axiom past_condition_2 : s - 4 = 3 * (a - 4)

/-- The ratio of their ages will be 3/2 after x years -/
axiom future_ratio : (s + x) / (a + x) = 3 / 2

theorem age_ratio_in_two_years :
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_two_years_l64_6402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l64_6492

/-- Given two vectors a and b in ℝ², prove that the projection of b onto a is equal to the expected result. -/
theorem projection_vector (a b : ℝ × ℝ) (ha : a = (2, -1)) (hb : b = (6, 2)) :
  let proj := ((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a
  proj = (4, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l64_6492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_equals_odd_product_l64_6437

theorem factorial_ratio_equals_odd_product (m : ℕ) :
  (Nat.factorial (2 * m)) / (Nat.factorial m * 2^(2 * m)) =
  Finset.prod (Finset.range m) (λ k => (2 * k + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_equals_odd_product_l64_6437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_golden_ratio_l64_6485

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem statement -/
theorem ellipse_eccentricity_golden_ratio (C : Ellipse) (F₁ F₂ M A : Point) :
  M.x > 0 → M.y > 0 →  -- M is in the first quadrant
  (M.x / C.a)^2 + (M.y / C.b)^2 = 1 →  -- M is on the ellipse
  (F₁.x)^2 + (F₁.y)^2 = (F₂.x)^2 + (F₂.y)^2 →  -- F₁ and F₂ are symmetrical
  (M.x - F₁.x)^2 + (M.y - F₁.y)^2 = (F₂.x - F₁.x)^2 + (F₂.y - F₁.y)^2 →  -- |MF₁| = |F₁F₂|
  A.x = 0 →  -- A is on the y-axis
  (A.y - F₁.y) / (A.x - F₁.x) = (M.y - F₁.y) / (M.x - F₁.x) →  -- F₁M passes through A
  (M.x - F₂.x) * (A.x - F₂.x) + (M.y - F₂.y) * (A.y - F₂.y) = 0 →  -- F₂A bisects ∠MF₂F₁
  eccentricity C = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_golden_ratio_l64_6485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_number_value_l64_6406

theorem sixth_number_value (numbers : List ℝ) : 
  numbers.length = 11 →
  numbers.sum / numbers.length = 60 →
  (numbers.take 6).sum / 6 = 78 →
  (numbers.drop 5).sum / 6 = 75 →
  numbers[5]! = 129 := by
  intro h_length h_avg h_first_six h_last_six
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_number_value_l64_6406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_pairs_l64_6433

def isValidPair (x y : ℕ) : Prop := x = 2 * y + 2

def CardSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

def ValidPairs : Set (ℕ × ℕ) :=
  {pair : ℕ × ℕ | pair.1 ∈ CardSet ∧ pair.2 ∈ CardSet ∧ isValidPair pair.1 pair.2}

theorem max_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), S.toSet ⊆ ValidPairs ∧ 
    (∀ (x y : ℕ), (x, y) ∈ S → (y, x) ∉ S) ∧
    (∀ (a b c d : ℕ), (a, b) ∈ S → (c, d) ∈ S → a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d) ∧
    (∀ (T : Finset (ℕ × ℕ)), T.toSet ⊆ ValidPairs →
      (∀ (x y : ℕ), (x, y) ∈ T → (y, x) ∉ T) →
      (∀ (a b c d : ℕ), (a, b) ∈ T → (c, d) ∈ T → a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d) →
      (T.biUnion (λ (x, y) => {x, y})).card ≤ 12) ∧
    (S.biUnion (λ (x, y) => {x, y})).card = 12 :=
by
  sorry

#check max_valid_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_pairs_l64_6433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_second_quadrant_l64_6412

noncomputable def Z : ℂ := (13 * Complex.I) / (3 - Complex.I) + (1 + Complex.I)

theorem Z_in_second_quadrant : (Z.re < 0) ∧ (Z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_second_quadrant_l64_6412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_voltage_approx_516mV_l64_6414

/-- Represents the parameters of an LC circuit --/
structure LCCircuit where
  C : ℝ  -- Capacitance in μF
  L₁ : ℝ -- Inductance in mH
  L₂ : ℝ -- Inductance in mH
  Imax : ℝ -- Maximum current in mA

/-- Calculates the maximum voltage on the capacitor after switch closure --/
noncomputable def maxVoltage (circuit : LCCircuit) : ℝ :=
  let I := circuit.L₂ * circuit.Imax / (circuit.L₁ + circuit.L₂)
  let energyBefore := 0.5 * circuit.L₂ * (circuit.Imax * circuit.Imax)
  let energyInductorsAfter := 0.5 * (circuit.L₁ + circuit.L₂) * (I * I)
  Real.sqrt (2 * (energyBefore - energyInductorsAfter) / circuit.C)

/-- Theorem stating that the maximum voltage is approximately 516 mV --/
theorem max_voltage_approx_516mV (circuit : LCCircuit)
  (hC : circuit.C = 1)
  (hL₁ : circuit.L₁ = 4)
  (hL₂ : circuit.L₂ = 2)
  (hImax : circuit.Imax = 10) :
  ∃ ε > 0, |maxVoltage circuit - 516| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_voltage_approx_516mV_l64_6414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_one_sixth_opposite_of_negative_one_sixth_equals_positive_one_sixth_l64_6467

theorem opposite_of_negative_one_sixth :
  (-(1/6 : ℚ)) = -1/6 := by
  rfl

theorem opposite_of_negative_one_sixth_equals_positive_one_sixth :
  (-(1/6 : ℚ)).neg = 1/6 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_one_sixth_opposite_of_negative_one_sixth_equals_positive_one_sixth_l64_6467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_l64_6465

theorem election_votes_calculation (winning_percentage : ℚ) (majority : ℕ) : 
  winning_percentage = 84 / 100 → 
  majority = 476 → 
  ∃ total_votes : ℕ, 
    (winning_percentage * total_votes).floor = 
      ((1 - winning_percentage) * total_votes).floor + majority ∧ 
    total_votes = 700 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_l64_6465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_average_l64_6426

theorem cookie_average (total_packages : ℕ) (counted_packages : ℕ) (cookie_counts : List ℕ) : 
  total_packages = 10 →
  counted_packages = 8 →
  cookie_counts = [9, 11, 12, 14, 16, 17, 18, 21] →
  let total_counted := cookie_counts.sum
  let avg_counted := (total_counted : ℚ) / counted_packages
  let total_cookies := total_counted + (avg_counted * (total_packages - counted_packages)).floor
  (total_cookies : ℚ) / total_packages = 147.5 / 10 := by
sorry

#eval [9, 11, 12, 14, 16, 17, 18, 21].sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_average_l64_6426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_midpoint_equations_l64_6481

-- Define the circle C
noncomputable def circle_center : ℝ × ℝ := (1, Real.pi/4)
def circle_radius : ℝ := 1

-- Define the polar coordinates of a point on the circle
def polar_coords (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos (θ - Real.pi/4)

-- Define the Cartesian coordinates of the midpoint Q
def midpoint_coords (x y : ℝ) : Prop :=
  (x - Real.sqrt 2 / 4)^2 + (y - Real.sqrt 2 / 4)^2 = 1/4

-- Theorem statement
theorem circle_and_midpoint_equations :
  ∀ (ρ θ x y : ℝ),
  (polar_coords ρ θ → midpoint_coords x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_midpoint_equations_l64_6481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_january_bill_amount_l64_6484

/-- Represents the oil bill for a month -/
structure OilBill where
  amount : ℚ
  deriving Repr

/-- Represents the oil bills for three months -/
structure ThreeMonthBills where
  january : OilBill
  february : OilBill
  march : OilBill
  deriving Repr

/-- The conditions of the problem -/
def satisfiesConditions (bills : ThreeMonthBills) : Prop :=
  let j := bills.january.amount
  let f := bills.february.amount
  let m := bills.march.amount
  f / j = 3 / 2 ∧
  f / m = 4 / 5 ∧
  (f + 20) / j = 5 / 3 ∧
  (f + 20) / m = 2 / 3

/-- The theorem to prove -/
theorem january_bill_amount (bills : ThreeMonthBills) :
  satisfiesConditions bills → bills.january.amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_january_bill_amount_l64_6484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_segment_property_l64_6407

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  midpoint_ratio : (b + 75) / (b + 150) = 3 / 4  -- Condition for midpoint division ratio
  b_positive : b > 0
  h_positive : h > 0

/-- The length of the segment that divides the trapezoid into equal areas -/
noncomputable def equal_area_segment (t : Trapezoid) : ℝ :=
  (150 + Real.sqrt (22500 + 180000)) / 2

/-- The theorem to be proved -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^2 / 150⌋ = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_segment_property_l64_6407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_line_through_trisection_point_l64_6448

-- Define the points
noncomputable def P : ℝ × ℝ := (1, 2)
noncomputable def A : ℝ × ℝ := (2, 3)
noncomputable def B : ℝ × ℝ := (-3, 0)

-- Define the trisection points
noncomputable def trisection_point_1 : ℝ × ℝ := (1/3, 2)
noncomputable def trisection_point_2 : ℝ × ℝ := (-4/3, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + 7 * y = 17

-- Define the line segment
def line_segment (P Q : ℝ × ℝ) (R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))

-- Theorem statement
theorem one_line_through_trisection_point :
  ∃ (T : ℝ × ℝ), (T = trisection_point_1 ∨ T = trisection_point_2) ∧
  (∀ (x y : ℝ), line_segment P T (x, y) → line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_line_through_trisection_point_l64_6448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_change_largest_l64_6432

def original_number : ℚ := 0.246810

def change_digit (n : ℚ) (position : Fin 6) : ℚ :=
  let digits : List ℕ := [2, 4, 6, 8, 1, 0]
  let new_digits := digits.set position.val 7
  new_digits.enum.foldl (λ acc (i, d) => acc + d / (10 ^ (i + 1))) 0

theorem first_digit_change_largest :
  ∀ position : Fin 6, 
    change_digit original_number ⟨0, by simp⟩ ≥ change_digit original_number position :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_change_largest_l64_6432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_two_dice_l64_6409

/-- Represents an 8-sided die -/
def Die8 : Type := Fin 8

/-- The probability of rolling an even number on a single 8-sided die -/
def prob_even_single : ℚ := 1/2

/-- The probability of rolling an odd number on a single 8-sided die -/
def prob_odd_single : ℚ := 1/2

/-- The sum of two dice rolls is even if both are even or both are odd -/
def sum_is_even (d1 d2 : Die8) : Prop := 
  (d1.val % 2 = 0 ∧ d2.val % 2 = 0) ∨ (d1.val % 2 ≠ 0 ∧ d2.val % 2 ≠ 0)

/-- The probability of rolling an even sum with two 8-sided dice -/
theorem prob_even_sum_two_dice : 
  (prob_even_single * prob_even_single + prob_odd_single * prob_odd_single : ℚ) = 1/2 := by
  -- Expand the definition of prob_even_single and prob_odd_single
  simp [prob_even_single, prob_odd_single]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_two_dice_l64_6409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_in_triangle_l64_6453

/-- Given a triangle with area T and its sides divided into m > 2, n > 2, and p > 2 equal parts,
    the area t of the convex hexagon formed by connecting the first and last division points
    on each side is t = T * (1 - (m + n + p) / (m * n * p)). -/
theorem hexagon_area_in_triangle (T m n p : ℝ) (hm : m > 2) (hn : n > 2) (hp : p > 2) :
  T * (1 - (m + n + p) / (m * n * p)) = T * (1 - (m + n + p) / (m * n * p)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_in_triangle_l64_6453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shirts_to_remove_is_correct_l64_6445

/-- Represents a shirt color -/
inductive ShirtColor
| White
| Purple
deriving BEq, Repr

/-- Represents the arrangement of shirts in a row -/
def ShirtArrangement := List ShirtColor

/-- Checks if a list of shirts has all shirts of the same color consecutively -/
def isConsecutive (arrangement : ShirtArrangement) : Prop := sorry

/-- Checks if it's possible to remove k shirts of each color to make the remaining shirts consecutive -/
def canMakeConsecutive (k : Nat) (arrangement : ShirtArrangement) : Prop := sorry

/-- The total number of shirts -/
def totalShirts : Nat := 42

/-- The number of shirts of each color -/
def shirtsPerColor : Nat := 21

/-- The minimum number of shirts to remove -/
def minShirtsToRemove : Nat := 10

theorem min_shirts_to_remove_is_correct :
  ∀ (arrangement : ShirtArrangement),
    (arrangement.length = totalShirts) →
    (arrangement.count ShirtColor.White = shirtsPerColor) →
    (arrangement.count ShirtColor.Purple = shirtsPerColor) →
    (∀ (k : Nat), k < minShirtsToRemove → ¬(canMakeConsecutive k arrangement)) ∧
    (canMakeConsecutive minShirtsToRemove arrangement) := by
  sorry

#check min_shirts_to_remove_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shirts_to_remove_is_correct_l64_6445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_balance_l64_6483

theorem savings_account_balance (starting_balance increase_rate decrease_rate : ℝ) : 
  starting_balance = 125 →
  increase_rate = 0.25 →
  decrease_rate = 0.20 →
  (starting_balance * (1 + increase_rate) * (1 - decrease_rate)) / starting_balance = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_balance_l64_6483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l64_6435

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 2

-- State the theorem
theorem root_in_interval :
  (∀ x y, x < y → f x < f y) →  -- f is monotonically increasing
  f 1 < 0 →
  f 2 > 0 →
  ∃ r, r ∈ Set.Ioo 1 2 ∧ f r = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l64_6435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l64_6480

/-- The ratio of speeds between the fastest and slowest cyclists -/
noncomputable def speed_ratio (v_fast v_slow : ℝ) : ℝ := v_fast / v_slow

/-- The distance between cyclists -/
def initial_distance : ℝ := 8

/-- Time taken for fastest and slowest to meet when traveling in same direction -/
def same_direction_time : ℝ := 4

/-- Time taken for fastest and slowest to meet when traveling towards each other -/
def opposite_direction_time : ℝ := 1

theorem cyclist_speed_ratio (v_fast v_slow : ℝ) 
  (h_same : initial_distance = (v_fast - v_slow) * same_direction_time)
  (h_opposite : initial_distance = (v_fast + v_slow) * opposite_direction_time)
  (h_positive : v_fast > 0 ∧ v_slow > 0) :
  speed_ratio v_fast v_slow = 5/3 := by
  sorry

#check cyclist_speed_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l64_6480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l64_6423

/-- The slope of a line passing through (0,2) and tangent to the ellipse x²/7 + y²/2 = 1 -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ (x y : ℝ), x^2/7 + y^2/2 = 1 ∧ y = k*x + 2 ∧ 
   ∀ (x' y' : ℝ), x'^2/7 + y'^2/2 = 1 ∧ y' = k*x' + 2 → (x', y') = (x, y)) →
  k = Real.sqrt 14 / 7 ∨ k = -(Real.sqrt 14 / 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l64_6423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l64_6434

/-- Given a triangle ABC with sides a, b, c opposite to internal angles A, B, C respectively,
    if sin B = 1/4, sin A = 2/3, and b = 3, then a = 8 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  Real.sin B = 1/4 → Real.sin A = 2/3 → b = 3 → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l64_6434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pay_cut_percentage_first_pay_cut_percentage_proof_l64_6419

theorem first_pay_cut_percentage 
  (second_cut : ℝ) 
  (third_cut : ℝ) 
  (overall_decrease : ℝ) 
  (first_cut : ℝ) : Prop :=
  second_cut = 14 ∧
  third_cut = 18 ∧
  overall_decrease = 35.1216 ∧
  first_cut = 8.04 ∧
  (1 - first_cut / 100) * (1 - second_cut / 100) * (1 - third_cut / 100) = 1 - overall_decrease / 100

theorem first_pay_cut_percentage_proof : 
  ∃ (second_cut third_cut overall_decrease first_cut : ℝ),
  first_pay_cut_percentage second_cut third_cut overall_decrease first_cut := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pay_cut_percentage_first_pay_cut_percentage_proof_l64_6419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_linear_l64_6488

noncomputable section

/-- Represents a binary operation on real numbers -/
def BinaryOp := ℝ → ℝ → ℝ

/-- Represents an equation with two variables -/
structure Equation :=
  (lhs rhs : BinaryOp)

/-- Checks if a given equation is linear in two variables -/
def is_linear (eq : Equation) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), eq.lhs x y = a * x + b * y ∧ eq.rhs x y = c

/-- Equation A: x/3 - 2/y = x -/
def eq_A : Equation :=
  { lhs := λ x y => x / 3 - 2 / y,
    rhs := λ x _ => x }

/-- Equation B: 3x = 2y -/
def eq_B : Equation :=
  { lhs := λ x _ => 3 * x,
    rhs := λ _ y => 2 * y }

/-- Equation C: x - y^2 = 0 -/
def eq_C : Equation :=
  { lhs := λ x y => x - y^2,
    rhs := λ _ _ => 0 }

/-- Equation D: 2x - 3y = xy -/
def eq_D : Equation :=
  { lhs := λ x y => 2 * x - 3 * y,
    rhs := λ x y => x * y }

/-- Theorem: Only equation B is linear in two variables -/
theorem only_B_is_linear :
  is_linear eq_B ∧ ¬is_linear eq_A ∧ ¬is_linear eq_C ∧ ¬is_linear eq_D := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_linear_l64_6488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_percentage_l64_6410

def max_marks : ℕ := 500
def obtained_marks : ℕ := 125
def failing_margin : ℕ := 40

theorem passing_percentage :
  (((obtained_marks + failing_margin : ℚ) / max_marks) * 100).floor = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_percentage_l64_6410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l64_6464

/-- Given a geometric progression with the first three terms as specified,
    prove that the fourth term is 1. -/
theorem fourth_term_of_geometric_progression (a : ℝ) (r : ℝ) :
  a = Real.sqrt 3 ∧ 
  a * r = (3 : ℝ) ^ (1/3) ∧ 
  a * r^2 = (3 : ℝ) ^ (1/6) →
  a * r^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l64_6464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l64_6476

/-- Represents a circular swimming pool with a cylindrical part and a hemispherical bottom. -/
structure CircularPool where
  diameter : ℝ
  cylinderDepth : ℝ

/-- Calculates the volume of a CircularPool in cubic feet. -/
noncomputable def poolVolume (pool : CircularPool) : ℝ :=
  let radius := pool.diameter / 2
  let cylinderVolume := Real.pi * radius^2 * pool.cylinderDepth
  let hemisphereVolume := (2/3) * Real.pi * radius^3
  cylinderVolume + hemisphereVolume

/-- Theorem stating that the volume of the specified pool is (3500/3)π cubic feet. -/
theorem pool_volume_calculation :
  let pool : CircularPool := ⟨20, 5⟩
  poolVolume pool = (3500/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l64_6476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_copper_ratio_for_alloy_l64_6440

/-- Represents the weight of a substance relative to water -/
structure RelativeWeight where
  value : ℝ
  pos : value > 0

/-- The relative weight of gold -/
def gold_weight : RelativeWeight := ⟨11, by norm_num⟩

/-- The relative weight of copper -/
def copper_weight : RelativeWeight := ⟨5, by norm_num⟩

/-- The desired relative weight of the alloy -/
def alloy_weight : RelativeWeight := ⟨8, by norm_num⟩

/-- Theorem stating that the ratio of gold to copper should be 1:1 for the desired alloy -/
theorem gold_copper_ratio_for_alloy :
  ∀ (g c : ℝ), g > 0 → c > 0 →
  (gold_weight.value * g + copper_weight.value * c) / (g + c) = alloy_weight.value →
  g = c := by
  sorry

#check gold_copper_ratio_for_alloy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_copper_ratio_for_alloy_l64_6440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_l64_6470

open BigOperators

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, k ≥ 1 → a k + a (k + 1) = (1 / 3) ^ k) →
  a 1 = 1 →
  (∀ k, S k = ∑ i in Finset.range k, (3 ^ i) * a (i + 1)) →
  4 * S n - (3 ^ n) * a n = n :=
by
  intros h_rec h_base h_S
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_l64_6470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l64_6477

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x + a * x^2 - 3 * x

-- State the theorem
theorem function_properties (a : ℝ) :
  -- The tangent line to g(x) at (1, g(1)) is parallel to the x-axis
  (∃ a, deriv (g a) 1 = 0) →
  -- Claim 1: The value of a is 1
  a = 1 ∧
  -- Claim 2: The minimum value of g(x) is -2
  (∃ x₀ > 0, ∀ x > 0, g a x ≥ g a x₀ ∧ g a x₀ = -2) ∧
  -- Claim 3: For a line with slope k intersecting f(x) at two points
  (∀ k x₁ x₂ y₁ y₂ : ℝ,
    0 < x₁ ∧ x₁ < x₂ ∧
    f x₁ = y₁ ∧ f x₂ = y₂ ∧
    k = (y₂ - y₁) / (x₂ - x₁) →
    1 / x₂ < k ∧ k < 1 / x₁) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l64_6477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bees_direction_at_15_feet_l64_6429

/-- Represents the direction of bee movement --/
inductive Direction
  | North
  | South
  | East
  | West
  | Up
  | Down

/-- Represents the position of a bee in 3D space --/
structure Position where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a bee's movement pattern --/
structure BeePattern where
  step1 : Direction
  step2 : Direction
  step3 : Direction

/-- Updates the position based on the given direction --/
def updatePosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.North => { pos with y := pos.y + 1 }
  | Direction.South => { pos with y := pos.y - 1 }
  | Direction.East => { pos with x := pos.x + 1 }
  | Direction.West => { pos with x := pos.x - 1 }
  | Direction.Up => { pos with z := pos.z + 1 }
  | Direction.Down => { pos with z := pos.z - 1 }

/-- Calculates the distance between two positions --/
noncomputable def distance (pos1 pos2 : Position) : ℝ :=
  Real.sqrt ((pos1.x - pos2.x)^2 + (pos1.y - pos2.y)^2 + (pos1.z - pos2.z)^2)

/-- Theorem: When the bees are exactly 15 feet apart, one is moving up and the other down --/
theorem bees_direction_at_15_feet 
  (beeA_pattern : BeePattern)
  (beeB_pattern : BeePattern)
  (beeA_pos beeB_pos : Position)
  (h1 : beeA_pattern.step1 = Direction.North ∧ 
        beeA_pattern.step2 = Direction.East ∧ 
        beeA_pattern.step3 = Direction.Up)
  (h2 : beeB_pattern.step1 = Direction.South ∧ 
        beeB_pattern.step2 = Direction.West ∧ 
        beeB_pattern.step3 = Direction.Down)
  (h3 : distance beeA_pos beeB_pos = 15) :
  (beeA_pattern.step3 = Direction.Up ∧ beeB_pattern.step3 = Direction.Down) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bees_direction_at_15_feet_l64_6429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_max_price_max_profit_value_l64_6474

/-- Represents the selling price in yuan -/
def x : ℝ := sorry

/-- Represents the daily sales in items -/
def y : ℝ := sorry

/-- Represents the daily sales profit in yuan -/
def w : ℝ := sorry

/-- The cost price of each item in yuan -/
def cost_price : ℝ := 6

/-- The maximum allowed selling price in yuan -/
def max_price : ℝ := 12

/-- The relationship between daily sales and selling price -/
axiom sales_price_relation : y = -10 * x + 280

/-- The relationship between daily profit and selling price -/
axiom profit_price_relation : w = -10 * (x - 11)^2 + 1210

/-- The selling price is constrained between cost price and maximum price -/
axiom price_constraint : cost_price ≤ x ∧ x ≤ max_price

/-- The theorem stating that the maximum daily sales profit occurs at the maximum allowed price -/
theorem max_profit_at_max_price :
  ∃ (max_profit : ℝ), (x = max_price ∧ w = max_profit) ∧
  ∀ (x' : ℝ), cost_price ≤ x' ∧ x' ≤ max_price → w ≤ max_profit := by
  sorry

/-- The theorem stating that the maximum profit is 960 yuan -/
theorem max_profit_value :
  ∃ (max_profit : ℝ), (x = max_price ∧ w = max_profit) ∧ max_profit = 960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_max_price_max_profit_value_l64_6474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_change_l64_6405

/-- Represents a solution with volume and acid concentration -/
structure Solution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of acid in a solution -/
noncomputable def acidAmount (s : Solution) : ℝ := s.volume * s.concentration

/-- Represents the process of removing and replacing part of a solution -/
noncomputable def replacePartOfSolution (initial : Solution) (removeAmount : ℝ) (newConcentration : ℝ) : Solution :=
  { volume := initial.volume,
    concentration := (acidAmount initial - removeAmount * initial.concentration + removeAmount * newConcentration) / initial.volume }

theorem acid_concentration_change (initialVolume finalVolume : ℝ) (initialConcentration finalConcentration replacementConcentration : ℝ) 
    (hVolume : initialVolume = 80 ∧ finalVolume = 80)
    (hConcentrations : initialConcentration = 0.2 ∧ finalConcentration = 0.4 ∧ replacementConcentration = 1) :
  ∃ (removeAmount : ℝ), removeAmount = 16 ∧ 
    replacePartOfSolution 
      { volume := initialVolume, concentration := initialConcentration } 
      removeAmount 
      replacementConcentration = 
    { volume := finalVolume, concentration := finalConcentration } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_change_l64_6405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_fifth_terms_l64_6450

/-- Given two arithmetic sequences {a_n} and {b_n} -/
def a : ℕ → ℚ := sorry
def b : ℕ → ℚ := sorry

/-- S_n is the sum of the first n terms of sequence a -/
def S (n : ℕ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- S'_n is the sum of the first n terms of sequence b -/
def S' (n : ℕ) : ℚ := (n : ℚ) * (b 1 + b n) / 2

/-- The ratio of the sum of the first n terms -/
axiom ratio_sums (n : ℕ) : S n / S' n = (5 * n + 3) / (2 * n + 7)

/-- The main theorem to prove -/
theorem ratio_fifth_terms : a 5 / b 5 = 48 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_fifth_terms_l64_6450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l64_6490

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) : 
  (∃ (f : ℝ → ℝ), f = λ x => Real.sin (ω * x + π / 6)) →
  ((π / 12 : ℝ) ∈ {x : ℝ | ∀ y, (λ x => Real.sin (ω * x + π / 6)) (x - y) = (λ x => Real.sin (ω * x + π / 6)) (x + y)}) →
  4 ≤ ω :=
by
  intro h_exists h_symmetry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l64_6490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axes_of_symmetry_intersect_l64_6420

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  -- Define the structure of a polygon
  -- (We'll leave this abstract for now)

/-- An axis of symmetry is a line that divides a polygon into two congruent halves. -/
structure AxisOfSymmetry (P : Polygon) where
  -- Define the structure of an axis of symmetry
  -- (We'll leave this abstract for now)

/-- A point in the 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on an axis of symmetry -/
def PointOnAxis (P : Polygon) (p : Point) (axis : AxisOfSymmetry P) : Prop :=
  sorry  -- We'll leave the implementation abstract for now

/-- Theorem: All axes of symmetry of a polygon intersect at a single point. -/
theorem axes_of_symmetry_intersect (P : Polygon) 
  (axes : Set (AxisOfSymmetry P)) 
  (h : axes.Nonempty) : 
  ∃ (p : Point), ∀ (axis : AxisOfSymmetry P), axis ∈ axes → PointOnAxis P p axis :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axes_of_symmetry_intersect_l64_6420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_and_odd_l64_6493

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - a^(-x)

-- State the theorem
theorem f_monotonic_and_odd (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y, x < y → f a x < f a y ∨ ∀ x y, x < y → f a x > f a y) ∧
  (∀ x, f a (-x) = -(f a x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_and_odd_l64_6493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_theorem_l64_6455

theorem sum_difference_theorem : 
  let n_odd : ℤ := (2049 - 1) / 2 + 1
  let n_even : ℤ := (2048 - 2) / 2 + 1
  let n_multiple_3 : ℤ := (2046 - 3) / 3 + 1
  (n_odd ^ 2) - (n_even * (n_even + 1)) - (n_multiple_3 * (3 + 2046) / 2) = -694684 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_theorem_l64_6455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l64_6425

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ (x : ℝ), f x = 2 * Real.cos (2 * (x - π / 12))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l64_6425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l64_6468

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points closer to (0,0) than to (4,2)
def closerToOrigin : Set (ℝ × ℝ) :=
  {p ∈ rectangle | distance p (0, 0) < distance p (4, 2)}

-- State the theorem
theorem probability_closer_to_origin :
  MeasureTheory.volume closerToOrigin / MeasureTheory.volume rectangle = 5 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l64_6468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_factors_l64_6403

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 2310 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
  A + B + C ≤ 52 ∧ (∃ (A' B' C' : ℕ+), A' ≠ B' ∧ B' ≠ C' ∧ A' ≠ C' ∧ A' * B' * C' = 2310 ∧ A' + B' + C' = 52) := by
  sorry

#check max_sum_of_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_factors_l64_6403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l64_6408

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - π) = 2/3) 
  (h2 : α ∈ Set.Ioo (-π/2) 0) : 
  Real.tan α = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l64_6408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_stand_profit_l64_6418

/-- Calculates the total profit for Sarah's lemonade stand --/
theorem lemonade_stand_profit : 
  (let total_days : ℕ := 10
  let hot_days : ℕ := 4
  let cups_per_day : ℕ := 32
  let cost_per_cup : ℚ := 75 / 100
  let hot_day_price : ℚ := 2095170454545454600 / 1000000000000000000
  let regular_day_price : ℚ := hot_day_price / (1 + 25 / 100)
  let hot_day_revenue : ℚ := hot_day_price * cups_per_day * hot_days
  let regular_day_revenue : ℚ := regular_day_price * cups_per_day * (total_days - hot_days)
  let total_revenue : ℚ := hot_day_revenue + regular_day_revenue
  let total_cost : ℚ := cost_per_cup * cups_per_day * total_days
  let total_profit : ℚ := total_revenue - total_cost
  total_profit) = 34935102 / 100000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_stand_profit_l64_6418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_run_average_speed_l64_6411

/-- Represents Emily's run with given distances, speeds, and break times -/
structure EmilyRun where
  total_distance : ℝ
  uphill1_distance : ℝ
  uphill1_speed : ℝ
  downhill1_distance : ℝ
  downhill1_speed : ℝ
  flat_distance : ℝ
  flat_speed : ℝ
  uphill2_distance : ℝ
  uphill2_speed : ℝ
  downhill2_distance : ℝ
  downhill2_speed : ℝ
  break1_time : ℝ
  break2_time : ℝ
  break3_time : ℝ

/-- Calculates the average speed of Emily's run -/
noncomputable def averageSpeed (run : EmilyRun) : ℝ :=
  let total_time := run.uphill1_distance / run.uphill1_speed +
                    run.downhill1_distance / run.downhill1_speed +
                    run.flat_distance / run.flat_speed +
                    run.uphill2_distance / run.uphill2_speed +
                    run.downhill2_distance / run.downhill2_speed +
                    run.break1_time / 60 + run.break2_time / 60 + run.break3_time / 60
  run.total_distance / total_time

/-- Theorem stating that Emily's average speed is approximately 4.36 mph -/
theorem emily_run_average_speed :
  let run : EmilyRun := {
    total_distance := 10,
    uphill1_distance := 2,
    uphill1_speed := 4,
    downhill1_distance := 1,
    downhill1_speed := 6,
    flat_distance := 3,
    flat_speed := 5,
    uphill2_distance := 2,
    uphill2_speed := 4.5,
    downhill2_distance := 2,
    downhill2_speed := 6,
    break1_time := 5,
    break2_time := 7,
    break3_time := 3
  }
  abs (averageSpeed run - 4.36) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_run_average_speed_l64_6411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l64_6460

-- Define the ellipse
def Ellipse (f1 f2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p f1 + dist p f2 = dist f1 f2}

-- Define the major axis length
def majorAxisLength (f1 f2 : ℝ × ℝ) : ℝ :=
  dist f1 f2

-- Define the distance function
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem ellipse_major_axis_length :
  let f1 : ℝ × ℝ := (9, 20)
  let f2 : ℝ × ℝ := (49, 55)
  let e := Ellipse f1 f2
  (∃ x : ℝ, (x, 0) ∈ e) →
  majorAxisLength f1 f2 = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l64_6460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_max_value_condition_max_set_nonempty_l64_6427

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3 - |2 * x - 1|

-- Theorem for part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 2} = Set.Iic 0 ∪ Set.Ici 2 := by sorry

-- Theorem for part II
theorem max_value_condition (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) ↔ -2 ≤ a ∧ a ≤ 2 := by sorry

-- Define the set of x where f(x) attains its maximum value
def max_set (a : ℝ) : Set ℝ := {x : ℝ | ∀ y : ℝ, f a y ≤ f a x}

-- Theorem stating that the maximum set is non-empty when -2 ≤ a ≤ 2
theorem max_set_nonempty (a : ℝ) (h : -2 ≤ a ∧ a ≤ 2) :
  (max_set a).Nonempty := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_max_value_condition_max_set_nonempty_l64_6427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_lines_exist_l64_6439

-- Define the ellipse (trajectory of circle C's center)
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Define the line (modified to use ℝ instead of ℤ)
def line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the condition for intersection points
def intersectionCondition (x1 x2 x3 x4 : ℝ) : Prop :=
  x4 - x2 + x3 - x1 = 0

-- Main theorem
theorem nine_lines_exist :
  ∃ (S : Finset (ℤ × ℤ)),
    (∀ (k m : ℤ), (k, m) ∈ S ↔
      (∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
        ellipse x1 y1 ∧ ellipse x2 y2 ∧
        hyperbola x3 y3 ∧ hyperbola x4 y4 ∧
        line (k : ℝ) (m : ℝ) x1 y1 ∧ line (k : ℝ) (m : ℝ) x2 y2 ∧
        line (k : ℝ) (m : ℝ) x3 y3 ∧ line (k : ℝ) (m : ℝ) x4 y4 ∧
        x1 ≠ x2 ∧ x3 ≠ x4 ∧
        intersectionCondition x1 x2 x3 x4)) ∧
    Finset.card S = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_lines_exist_l64_6439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kyungsoo_second_place_l64_6496

noncomputable def kyungsoo_jump : ℝ := 2.3
noncomputable def younghee_jump : ℝ := 9/10
noncomputable def jinju_jump : ℝ := 1.8
noncomputable def chanho_jump : ℝ := 2.5

theorem kyungsoo_second_place :
  kyungsoo_jump > jinju_jump ∧
  kyungsoo_jump > younghee_jump ∧
  kyungsoo_jump < chanho_jump :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kyungsoo_second_place_l64_6496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_foci_max_diff_distances_to_foci_max_distance_M_to_P_l64_6441

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the foci
noncomputable def foci_distance : ℝ := Real.sqrt 2

-- Define the point M
def M : ℝ × ℝ := (0, 2)

-- Theorem 1: Sum of distances from P to foci is constant
theorem sum_distances_to_foci (x y : ℝ) (h : is_on_ellipse x y) :
  ∃ (F₁ F₂ : ℝ × ℝ), 
    (F₁.1 = -foci_distance ∧ F₁.2 = 0) ∧
    (F₂.1 = foci_distance ∧ F₂.2 = 0) ∧
    Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) + Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 4 := by
  sorry

-- Theorem 2: Maximum difference of distances from P to foci
theorem max_diff_distances_to_foci :
  ∃ (x y : ℝ) (h : is_on_ellipse x y),
    ∀ (x' y' : ℝ) (h' : is_on_ellipse x' y'),
      abs (Real.sqrt ((x - (-foci_distance))^2 + y^2) - Real.sqrt ((x - foci_distance)^2 + y^2)) ≥
      abs (Real.sqrt ((x' - (-foci_distance))^2 + y'^2) - Real.sqrt ((x' - foci_distance)^2 + y'^2)) ∧
      abs (Real.sqrt ((x - (-foci_distance))^2 + y^2) - Real.sqrt ((x - foci_distance)^2 + y^2)) = 2 * Real.sqrt 2 := by
  sorry

-- Theorem 3: Maximum distance from M to any point P on the ellipse
theorem max_distance_M_to_P :
  ∃ (x y : ℝ) (h : is_on_ellipse x y),
    ∀ (x' y' : ℝ) (h' : is_on_ellipse x' y'),
      Real.sqrt ((x - M.1)^2 + (y - M.2)^2) ≥ Real.sqrt ((x' - M.1)^2 + (y' - M.2)^2) ∧
      Real.sqrt ((x - M.1)^2 + (y - M.2)^2) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_foci_max_diff_distances_to_foci_max_distance_M_to_P_l64_6441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_approximately_8_85_l64_6499

-- Define the expenses and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 5200
def amount_saved : ℕ := 2300

-- Calculate total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Calculate total monthly salary
def total_monthly_salary : ℕ := total_expenses + amount_saved

-- Define the percentage saved
noncomputable def percentage_saved : ℚ := (amount_saved : ℚ) / (total_monthly_salary : ℚ) * 100

-- Theorem to prove
theorem savings_percentage_approximately_8_85 :
  |percentage_saved - 8.85| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_approximately_8_85_l64_6499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_sufficient_not_necessary_l64_6479

-- Define the basic structures
structure Plane : Type
structure Line : Type

-- Define the relationships
axiom intersect : Plane → Plane → Line → Prop
axiom contains : Plane → Line → Prop
axiom perpendicular : Line → Line → Prop
axiom perpendicular_planes : Plane → Plane → Prop

-- Define the problem setup
def problem_setup (α β : Plane) (m a b : Line) : Prop :=
  intersect α β m ∧ contains α a ∧ contains β b ∧ perpendicular b m

-- Define the theorem
theorem perpendicular_planes_sufficient_not_necessary 
  (α β : Plane) (m a b : Line) 
  (h : problem_setup α β m a b) : 
  (perpendicular_planes α β → perpendicular a b) ∧ 
  ∃ α' β' m' a' b', problem_setup α' β' m' a' b' ∧ 
                    perpendicular a' b' ∧ 
                    ¬perpendicular_planes α' β' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_sufficient_not_necessary_l64_6479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_second_half_speed_l64_6463

/-- Given a journey with total distance, total time, and speed for the first half,
    calculate the speed for the second half of the journey. -/
noncomputable def second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let first_half_time := half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  half_distance / second_half_time

/-- Theorem stating that for the given journey conditions, 
    the speed for the second half is 100 km/h. -/
theorem journey_second_half_speed :
  second_half_speed 400 10 25 = 100 := by
  -- Unfold the definition of second_half_speed
  unfold second_half_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_second_half_speed_l64_6463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l64_6456

noncomputable def x (t : ℝ) : ℝ := t - Real.sin t
noncomputable def y (t : ℝ) : ℝ := 2 - Real.cos t

theorem second_derivative_parametric_function (t : ℝ) :
  let x' := fun s => deriv x s
  let y' := fun s => deriv y s
  let y_x' := fun s => y' s / x' s
  let y_xx'' := fun s => deriv y_x' s / x' s
  y_xx'' t = -1 / (1 - Real.cos t)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l64_6456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circ_triple_solution_l64_6491

def circ (a b : ℤ) : ℤ := a + b - a * b

theorem circ_triple_solution :
  ∀ x y z : ℤ, 
    (circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0) ↔ 
    ((x = 0 ∧ y = 0 ∧ z = 2) ∨ 
     (x = 0 ∧ y = 2 ∧ z = 0) ∨ 
     (x = 2 ∧ y = 0 ∧ z = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circ_triple_solution_l64_6491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l64_6404

/-- Rounds a real number to the specified number of decimal places -/
noncomputable def round_to_decimal_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋) / 10^n

/-- Proves that 8/11 rounded to 3 decimal places equals 0.727 -/
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 3 = 0.727 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l64_6404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_census_for_important_accurate_surveys_l64_6400

/-- Represents different survey methods -/
inductive SurveyMethod
| Census
| SampleSurvey
| Other

/-- Represents the characteristics of a survey -/
structure SurveyCharacteristics where
  requiresHighAccuracy : Bool
  isOfGreatImportance : Bool

/-- Determines the best survey method based on given characteristics -/
def bestSurveyMethod (chars : SurveyCharacteristics) : SurveyMethod :=
  if chars.requiresHighAccuracy && chars.isOfGreatImportance then
    SurveyMethod.Census
  else
    SurveyMethod.Other

/-- Theorem: For a survey requiring high accuracy and of great importance, 
    the best method is a census -/
theorem census_for_important_accurate_surveys 
  (chars : SurveyCharacteristics)
  (h1 : chars.requiresHighAccuracy = true)
  (h2 : chars.isOfGreatImportance = true) :
  bestSurveyMethod chars = SurveyMethod.Census := by
  simp [bestSurveyMethod, h1, h2]

#check census_for_important_accurate_surveys

end NUMINAMATH_CALUDE_ERRORFEEDBACK_census_for_important_accurate_surveys_l64_6400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_distance_100_l64_6428

noncomputable def θ : ℂ := Complex.exp (Complex.I * Real.pi / 4)

noncomputable def bee_position : ℕ → ℂ
| 0 => 0
| 1 => 2
| (n + 2) => bee_position (n + 1) + (2 * (n + 2) - 1 : ℝ) * θ^(n + 1)

theorem bee_distance_100 :
  Complex.abs (bee_position 100) = (Real.sqrt (20410 + 10205 * Real.sqrt 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_distance_100_l64_6428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l64_6494

/-- The volume of a cone given its slant height and height --/
noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let radius := Real.sqrt (slant_height^2 - height^2)
  (1/3) * Real.pi * radius^2 * height

/-- Theorem: The volume of a cone with slant height 17 cm and height 15 cm is 320π cubic centimeters --/
theorem cone_volume_specific : cone_volume 17 15 = 320 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l64_6494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l64_6482

theorem remainder_problem (x : ℕ) (h : 7 * x % 31 = 1) : (14 + x) % 31 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l64_6482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circumference_approximation_l64_6438

/-- Ramanujan's approximation for ellipse circumference -/
noncomputable def ellipseCircumference (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

theorem ellipse_circumference_approximation :
  let majorAxis : ℝ := 25
  let minorAxis : ℝ := 15
  let a : ℝ := majorAxis / 2
  let b : ℝ := minorAxis / 2
  abs (ellipseCircumference a b - 129.7883) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circumference_approximation_l64_6438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l64_6459

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2 + Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q)) ∧
  (∀ y : ℝ, y ∈ Set.Ioo (-1/2) (5/2) ↔ ∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l64_6459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_growth_l64_6452

-- Define the growth rate function v(x)
noncomputable def v (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then 2
  else if 4 < x ∧ x ≤ 20 then -1/8 * x + 5/2
  else 0

-- Define the annual growth function f(x)
noncomputable def f (x : ℝ) : ℝ := x * v x

-- Theorem statement
theorem max_annual_growth :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 20 ∧
  f x = 12.5 ∧
  ∀ (y : ℝ), 0 < y ∧ y ≤ 20 → f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_growth_l64_6452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_scheduling_arrangements_l64_6469

/-- The number of staff members -/
def num_staff : ℕ := 7

/-- The number of days to schedule -/
def num_days : ℕ := 7

/-- The number of staff members who cannot work on the first two days -/
def num_restricted : ℕ := 2

/-- The number of ways to arrange the staff schedule -/
def num_arrangements : ℕ := 2400

/-- Function to calculate factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem staff_scheduling_arrangements :
  factorial (num_staff - num_restricted) * factorial (num_staff - num_restricted) = num_arrangements :=
by
  -- Replace this with the actual proof when ready
  sorry

#eval factorial (num_staff - num_restricted) * factorial (num_staff - num_restricted)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_scheduling_arrangements_l64_6469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l64_6458

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define point A
def A : ℝ × ℝ := (-2, 2)

-- Define the left focus F
def F : ℝ × ℝ := (-3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the objective function to be minimized
noncomputable def objective (B : ℝ × ℝ) : ℝ :=
  distance A B + (5/3) * distance B F

-- State the theorem
theorem min_distance_point :
  ∃ (B : ℝ × ℝ), 
    ellipse B.1 B.2 ∧ 
    (∀ (C : ℝ × ℝ), ellipse C.1 C.2 → objective B ≤ objective C) ∧
    B = (-5 * Real.sqrt 3 / 2, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l64_6458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_constant_l64_6443

-- Define the polynomial and constant polynomial properties
def is_polynomial {α : Type*} [CommRing α] (p : α → α) : Prop := sorry

def is_constant_polynomial {α : Type*} [CommRing α] (p : α → α) : Prop := sorry

-- Main theorem
theorem polynomial_sum_constant {α : Type*} [CommRing α] 
  (f g h : α → α)
  (hh : is_polynomial h ∧ ¬is_constant_polynomial h)
  (hfg : f ≠ g)
  (hcomp : ∀ x, h (f x) = h (g x))
  : is_constant_polynomial (λ x ↦ f x + g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_constant_l64_6443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_quantifier_l64_6495

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x^3 + 2*x ≥ 0) ↔ (∃ x : ℝ, x^3 + 2*x < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_quantifier_l64_6495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangements_l64_6446

theorem student_arrangements (n m : ℕ) : 
  n = 6 → m = 2 → (Nat.factorial (n - m)) * (Nat.choose (n - m + 1) m) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangements_l64_6446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_ratio_constant_l64_6415

/-- A regular hexagon inscribed in a circle -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ i j : Fin 6, dist (vertices i) (vertices ((i + 1) % 6)) = dist (vertices j) (vertices ((j + 1) % 6))
  is_inscribed : ∃ center : ℝ × ℝ, ∃ radius : ℝ, ∀ i : Fin 6, dist center (vertices i) = radius

/-- A point on the circumcircle of a regular hexagon -/
def PointOnCircumcircle (h : RegularHexagon) :=
  { p : ℝ × ℝ // ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    (∀ i : Fin 6, dist center (h.vertices i) = radius) ∧
    dist center p = radius }

/-- The theorem to be proved -/
theorem regular_hexagon_ratio_constant (h : RegularHexagon) (p : PointOnCircumcircle h)
  (h_p_on_arc : p.val ∈ Set.uIcc (h.vertices 4) (h.vertices 5)) :
  (dist p.val (h.vertices 0) + dist p.val (h.vertices 1) + dist p.val (h.vertices 2) + dist p.val (h.vertices 3)) /
  (dist p.val (h.vertices 4) + dist p.val (h.vertices 5)) = 3 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_ratio_constant_l64_6415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_f_l64_6489

/-- The function f(x) = A sin(ωx + π/6) - 1 -/
noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 6) - 1

/-- Theorem stating the maximum difference of f on the interval [0, π/2] -/
theorem max_diff_f (A ω : ℝ) (hA : A > 0) (hω : ω > 0)
  (h_symmetry : Real.pi / ω = Real.pi / 2)
  (h_f_pi_6 : f A ω (Real.pi / 6) = 1) :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧
    f A ω x₁ - f A ω x₂ = 3 ∧
    ∀ (y₁ y₂ : ℝ), y₁ ∈ Set.Icc 0 (Real.pi / 2) → y₂ ∈ Set.Icc 0 (Real.pi / 2) →
      f A ω y₁ - f A ω y₂ ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_f_l64_6489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_zero_one_no_solution_in_one_two_no_solution_in_two_three_no_solution_in_three_four_l64_6497

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem solution_in_zero_one :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
by
  sorry

-- Additional theorems to show that the solution is not in other intervals
theorem no_solution_in_one_two :
  ¬∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

theorem no_solution_in_two_three :
  ¬∃ x ∈ Set.Ioo 2 3, f x = 0 :=
by
  sorry

theorem no_solution_in_three_four :
  ¬∃ x ∈ Set.Ioo 3 4, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_zero_one_no_solution_in_one_two_no_solution_in_two_three_no_solution_in_three_four_l64_6497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_works_four_hours_l64_6444

/-- Calculates the time James spends on chores given the following conditions:
    - There are 3 bedrooms, 1 living room, and 2 bathrooms to clean.
    - Each bedroom takes 20 minutes to clean.
    - The living room takes as long as the 3 bedrooms combined.
    - Each bathroom takes twice as long as the living room.
    - Cleaning the outside takes twice as long as cleaning the house.
    - The chores are split among James and his 2 siblings, who are equally fast. -/
noncomputable def james_chore_time : ℚ :=
  let bedroom_count : ℕ := 3
  let bathroom_count : ℕ := 2
  let bedroom_time : ℚ := 20 / 60  -- 20 minutes in hours
  let living_room_time : ℚ := bedroom_count * bedroom_time
  let bathroom_time : ℚ := 2 * living_room_time
  let inside_time : ℚ := bedroom_count * bedroom_time + living_room_time + bathroom_count * bathroom_time
  let outside_time : ℚ := 2 * inside_time
  let total_time : ℚ := inside_time + outside_time
  let sibling_count : ℕ := 3
  total_time / sibling_count

/-- Proves that James works for 4 hours given the conditions of the problem. -/
theorem james_works_four_hours : james_chore_time = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_works_four_hours_l64_6444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l64_6466

/-- Calculates the percentage increase between two amounts -/
noncomputable def percentageIncrease (originalAmount newAmount : ℝ) : ℝ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem johns_earnings_increase :
  let originalEarnings : ℝ := 60
  let newEarnings : ℝ := 70
  ∃ ε > 0, |percentageIncrease originalEarnings newEarnings - 16.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l64_6466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_sqrt_three_l64_6447

/-- Given two vectors a and e in a real inner product space, 
    with |a| = 2, |e| = 1, and the angle between them π/3,
    prove that the projection of a + e onto a - e equals √3. -/
theorem projection_equals_sqrt_three 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a e : V) :
  ‖a‖ = 2 →
  ‖e‖ = 1 →
  inner a e = ‖a‖ * ‖e‖ * Real.cos (π / 3) →
  inner (a + e) (a - e) / ‖a - e‖ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_sqrt_three_l64_6447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_views_imply_cone_l64_6417

/-- A geometric solid. -/
class Solid :=
  (front_view : Type)
  (side_view : Type)

/-- An isosceles triangle. -/
class IsoscelesTriangle

/-- A cone. -/
class Cone extends Solid

/-- The property that a solid has isosceles triangle views. -/
def has_isosceles_triangle_views (S : Solid) : Prop :=
  S.front_view = IsoscelesTriangle ∧ S.side_view = IsoscelesTriangle

/-- Theorem stating that if a solid has isosceles triangle views, it is a cone. -/
theorem isosceles_views_imply_cone :
  ∀ (S : Solid), has_isosceles_triangle_views S → Cone :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_views_imply_cone_l64_6417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendan_tax_rate_l64_6451

/-- Brendan's hourly wage as a waiter -/
def hourly_wage : ℚ := 6

/-- Number of 8-hour shifts Brendan works -/
def eight_hour_shifts : ℕ := 2

/-- Number of 12-hour shifts Brendan works -/
def twelve_hour_shifts : ℕ := 1

/-- Brendan's average hourly tips -/
def hourly_tips : ℚ := 12

/-- Fraction of tips Brendan reports to the IRS -/
def reported_tip_fraction : ℚ := 1/3

/-- Amount Brendan pays in taxes each week -/
def weekly_taxes : ℚ := 56

/-- Total hours worked in a week -/
def total_hours : ℚ := 8 * eight_hour_shifts + 12 * twelve_hour_shifts

/-- Brendan's total income (wage + tips) -/
noncomputable def total_income : ℚ := total_hours * (hourly_wage + hourly_tips)

/-- Brendan's reported income -/
noncomputable def reported_income : ℚ := total_hours * hourly_wage + total_hours * hourly_tips * reported_tip_fraction

/-- Theorem: Brendan's tax rate is 20% of his reported income -/
theorem brendan_tax_rate : weekly_taxes / reported_income = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendan_tax_rate_l64_6451
