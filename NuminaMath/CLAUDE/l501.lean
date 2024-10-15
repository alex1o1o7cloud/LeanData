import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_eq_one_l501_50113

theorem unique_solution_ab_minus_a_minus_b_eq_one :
  ∃! (a b : ℕ), a * b - a - b = 1 ∧ a > b ∧ b > 0 ∧ a = 3 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_ab_minus_a_minus_b_eq_one_l501_50113


namespace NUMINAMATH_CALUDE_roberts_initial_balls_prove_roberts_initial_balls_l501_50114

theorem roberts_initial_balls (tim_balls : ℕ) (robert_final : ℕ) : ℕ :=
  let tim_gave := tim_balls / 2
  let robert_initial := robert_final - tim_gave
  robert_initial

theorem prove_roberts_initial_balls :
  roberts_initial_balls 40 45 = 25 := by
  sorry

end NUMINAMATH_CALUDE_roberts_initial_balls_prove_roberts_initial_balls_l501_50114


namespace NUMINAMATH_CALUDE_son_age_l501_50121

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_l501_50121


namespace NUMINAMATH_CALUDE_horner_v2_value_l501_50175

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ :=
  v * x + a

theorem horner_v2_value :
  let f := fun x => 8 * x^4 + 5 * x^3 + 3 * x^2 + 2 * x + 1
  let coeffs := [8, 5, 3, 2, 1]
  let x := 2
  let v₀ := coeffs.head!
  let v₁ := horner_step v₀ x (coeffs.get! 1)
  let v₂ := horner_step v₁ x (coeffs.get! 2)
  v₂ = 45 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l501_50175


namespace NUMINAMATH_CALUDE_circle_equation_l501_50194

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points M and N
def M : ℝ × ℝ := (-2, 2)
def N : ℝ × ℝ := (-1, -1)

-- Define the line equation x - y - 1 = 0
def LineEquation (p : ℝ × ℝ) : Prop := p.1 - p.2 - 1 = 0

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    LineEquation center ∧
    M ∈ Circle center radius ∧
    N ∈ Circle center radius ∧
    center = (3, 2) ∧
    radius = 5 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l501_50194


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l501_50192

theorem smallest_multiple_of_6_and_15 : ∃ (a : ℕ), a > 0 ∧ 6 ∣ a ∧ 15 ∣ a ∧ ∀ (b : ℕ), b > 0 → 6 ∣ b → 15 ∣ b → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l501_50192


namespace NUMINAMATH_CALUDE_not_prime_sum_products_l501_50188

theorem not_prime_sum_products (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬ Prime (a * b + c * d) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_products_l501_50188


namespace NUMINAMATH_CALUDE_max_product_sum_l501_50180

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ X' Y' Z' : ℕ, X' + Y' + Z' = 15 → 
    X' * Y' * Z' + X' * Y' + Y' * Z' + Z' * X' ≤ X * Y * Z + X * Y + Y * Z + Z * X) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_l501_50180


namespace NUMINAMATH_CALUDE_median_divides_triangle_equally_l501_50110

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- A median of a triangle -/
def median (t : Triangle) (vertex : Point) : Point := sorry

/-- Theorem: A median of a triangle divides the triangle into two triangles of equal area -/
theorem median_divides_triangle_equally (t : Triangle) (vertex : Point) :
  let m := median t vertex
  triangleArea ⟨t.A, t.B, m⟩ = triangleArea ⟨t.A, t.C, m⟩ ∨
  triangleArea ⟨t.B, t.C, m⟩ = triangleArea ⟨t.A, t.C, m⟩ ∨
  triangleArea ⟨t.A, t.B, m⟩ = triangleArea ⟨t.B, t.C, m⟩ :=
sorry

end NUMINAMATH_CALUDE_median_divides_triangle_equally_l501_50110


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l501_50164

theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) :
  value_last_year = 20000 :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_last_year_l501_50164


namespace NUMINAMATH_CALUDE_triangle_theorem_l501_50171

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that if b cos C + c cos B = 2a cos A and AB · AC = √3 in a triangle ABC,
    then the measure of angle A is π/3 and the area of the triangle is 3/2. -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.A)
  (h2 : t.a * t.c * Real.cos t.A = Real.sqrt 3) : 
  t.A = π / 3 ∧ (1 / 2 * t.a * t.c * Real.sin t.A = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l501_50171


namespace NUMINAMATH_CALUDE_expression_evaluation_l501_50123

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/3
  (4*x^2 - 2*x*y + y^2) - 3*(x^2 - x*y + 5*y^2) = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l501_50123


namespace NUMINAMATH_CALUDE_total_flowers_is_105_l501_50131

/-- The total number of hibiscus, chrysanthemums, and dandelions -/
def total_flowers (h c d : ℕ) : ℕ := h + c + d

/-- Theorem: The total number of flowers is 105 -/
theorem total_flowers_is_105 
  (h : ℕ) 
  (c : ℕ) 
  (d : ℕ) 
  (h_count : h = 34)
  (h_vs_c : h = c - 13)
  (c_vs_d : c = d + 23) : 
  total_flowers h c d = 105 := by
  sorry

#check total_flowers_is_105

end NUMINAMATH_CALUDE_total_flowers_is_105_l501_50131


namespace NUMINAMATH_CALUDE_shift_function_unit_shift_l501_50158

/-- A function satisfying specific inequalities for shifts of 24 and 77 -/
def ShiftFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 24) ≤ f x + 24) ∧ (∀ x : ℝ, f (x + 77) ≥ f x + 77)

/-- Theorem stating that a ShiftFunction satisfies f(x+1) = f(x)+1 for all real x -/
theorem shift_function_unit_shift (f : ℝ → ℝ) (hf : ShiftFunction f) :
  ∀ x : ℝ, f (x + 1) = f x + 1 := by
  sorry

end NUMINAMATH_CALUDE_shift_function_unit_shift_l501_50158


namespace NUMINAMATH_CALUDE_same_remainder_implies_specific_remainder_l501_50181

theorem same_remainder_implies_specific_remainder 
  (m : ℕ) 
  (h1 : m ≠ 1) 
  (h2 : ∃ r : ℕ, 69 % m = r ∧ 90 % m = r ∧ 125 % m = r) : 
  86 % m = 2 := by
sorry

end NUMINAMATH_CALUDE_same_remainder_implies_specific_remainder_l501_50181


namespace NUMINAMATH_CALUDE_correct_machines_in_first_scenario_l501_50118

/-- The number of machines in the first scenario -/
def machines_in_first_scenario : ℕ := 5

/-- The number of units produced in the first scenario -/
def units_first_scenario : ℕ := 20

/-- The number of hours in the first scenario -/
def hours_first_scenario : ℕ := 10

/-- The number of machines in the second scenario -/
def machines_second_scenario : ℕ := 10

/-- The number of units produced in the second scenario -/
def units_second_scenario : ℕ := 100

/-- The number of hours in the second scenario -/
def hours_second_scenario : ℕ := 25

/-- The production rate per machine is constant across both scenarios -/
axiom production_rate_constant : 
  (units_first_scenario : ℚ) / (machines_in_first_scenario * hours_first_scenario) = 
  (units_second_scenario : ℚ) / (machines_second_scenario * hours_second_scenario)

theorem correct_machines_in_first_scenario : 
  machines_in_first_scenario = 5 := by sorry

end NUMINAMATH_CALUDE_correct_machines_in_first_scenario_l501_50118


namespace NUMINAMATH_CALUDE_fifth_derivative_y_l501_50101

noncomputable def y (x : ℝ) : ℝ := (4 * x + 3) * (2 : ℝ)^(-x)

theorem fifth_derivative_y (x : ℝ) :
  (deriv^[5] y) x = (-Real.log 2^5 * (4 * x + 3) + 20 * Real.log 2^4) * (2 : ℝ)^(-x) :=
by sorry

end NUMINAMATH_CALUDE_fifth_derivative_y_l501_50101


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l501_50183

theorem complex_number_in_first_quadrant : let z : ℂ := (Complex.I) / (Complex.I + 1)
  (0 < z.re) ∧ (0 < z.im) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l501_50183


namespace NUMINAMATH_CALUDE_first_load_pieces_l501_50161

/-- The number of pieces of clothing in the first load -/
def first_load (total : ℕ) (num_small_loads : ℕ) (pieces_per_small_load : ℕ) : ℕ :=
  total - (num_small_loads * pieces_per_small_load)

/-- Theorem stating that the number of pieces of clothing in the first load is 17 -/
theorem first_load_pieces : first_load 47 5 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_first_load_pieces_l501_50161


namespace NUMINAMATH_CALUDE_ellipse_tangent_circle_radius_l501_50139

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a circle centered at one of its foci and tangent to the ellipse,
    prove that the radius of the circle is √((a^2 - b^2)/2). -/
theorem ellipse_tangent_circle_radius 
  (a b : ℝ) 
  (h_a : a = 6) 
  (h_b : b = 3) : 
  let c := Real.sqrt (a^2 - b^2)
  let r := Real.sqrt ((a^2 - b^2)/2)
  ∀ x y : ℝ,
  (x^2 / a^2 + y^2 / b^2 = 1) →
  ((x - c)^2 + y^2 = r^2) →
  r = Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_tangent_circle_radius_l501_50139


namespace NUMINAMATH_CALUDE_unique_solution_l501_50190

def f (d : ℝ) (x : ℝ) : ℝ := 4 * x^3 - d * x

def g (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_solution :
  ∃! (a b c d : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f d x| ≤ 1) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |g a b c x| ≤ 1) ∧
    a = 0 ∧ b = -3 ∧ c = 0 ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l501_50190


namespace NUMINAMATH_CALUDE_point_division_theorem_l501_50116

/-- Given points A and B, if there exists a point C on the line y=x that divides AB in the ratio 2:1, then the y-coordinate of B is 4. -/
theorem point_division_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, a)
  ∃ (C : ℝ × ℝ), 
    (C.1 = C.2) ∧  -- C is on the line y = x
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) ∧  -- C is on line segment AB
    (C - A = 2 • (B - C))  -- AC = 2CB
    → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_division_theorem_l501_50116


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l501_50193

/-- The probability of selecting two non-defective pens from a box of 8 pens, where 2 are defective -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) :
  total_pens = 8 →
  defective_pens = 2 →
  selected_pens = 2 →
  (total_pens - defective_pens : ℚ) / total_pens *
  ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1)) = 15 / 28 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l501_50193


namespace NUMINAMATH_CALUDE_vector_combination_l501_50133

/-- Given three points A, B, and C in a plane, prove that the coordinates of 1/2 * AC - 1/4 * BC are (-3, 6) -/
theorem vector_combination (A B C : ℝ × ℝ) (h1 : A = (2, -4)) (h2 : B = (0, 6)) (h3 : C = (-8, 10)) :
  (1 / 2 : ℝ) • (C - A) - (1 / 4 : ℝ) • (C - B) = (-3, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_l501_50133


namespace NUMINAMATH_CALUDE_tom_catch_equals_16_l501_50138

def melanie_catch : ℕ := 8

def tom_catch_multiplier : ℕ := 2

def tom_catch : ℕ := tom_catch_multiplier * melanie_catch

theorem tom_catch_equals_16 : tom_catch = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_catch_equals_16_l501_50138


namespace NUMINAMATH_CALUDE_regular_2000_pointed_stars_count_l501_50103

theorem regular_2000_pointed_stars_count : ℕ :=
  let n : ℕ := 2000
  let φ : ℕ → ℕ := fun m => Nat.totient m
  let non_similar_count : ℕ := (φ n - 2) / 2
  399

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_regular_2000_pointed_stars_count_l501_50103


namespace NUMINAMATH_CALUDE_balloon_ratio_is_seven_l501_50148

-- Define the number of balloons for Dan and Tim
def dans_balloons : ℕ := 29
def tims_balloons : ℕ := 203

-- Define the ratio of Tim's balloons to Dan's balloons
def balloon_ratio : ℚ := tims_balloons / dans_balloons

-- Theorem stating that the ratio is 7
theorem balloon_ratio_is_seven : balloon_ratio = 7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_is_seven_l501_50148


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l501_50165

theorem cube_volume_from_surface_area :
  ∀ (side : ℝ), 
    side > 0 →
    6 * side^2 = 486 →
    side^3 = 729 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l501_50165


namespace NUMINAMATH_CALUDE_watch_cost_price_l501_50167

/-- The cost price of a watch given specific selling conditions -/
theorem watch_cost_price (C : ℚ) : 
  (0.9 * C = C - 0.1 * C) →  -- Selling price at 10% loss
  (1.04 * C = C + 0.04 * C) →  -- Selling price at 4% gain
  (1.04 * C - 0.9 * C = 200) →  -- Difference between selling prices
  C = 10000 / 7 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l501_50167


namespace NUMINAMATH_CALUDE_distinct_convex_polygons_l501_50186

/-- Represents a triangle with side lengths --/
structure Triangle :=
  (side1 side2 side3 : ℝ)

/-- Represents a convex polygon --/
structure ConvexPolygon :=
  (vertices : List (ℝ × ℝ))

/-- Checks if a polygon is convex --/
def isConvex (p : ConvexPolygon) : Prop :=
  sorry

/-- Counts the number of distinct convex polygons that can be formed --/
def countConvexPolygons (triangles : List Triangle) : ℕ :=
  sorry

/-- The main theorem --/
theorem distinct_convex_polygons :
  let triangles : List Triangle := [
    ⟨3, 3, 3⟩, ⟨3, 3, 3⟩,  -- Two equilateral triangles
    ⟨3, 4, 5⟩, ⟨3, 4, 5⟩   -- Two scalene triangles
  ]
  countConvexPolygons triangles = 16 := by
  sorry

end NUMINAMATH_CALUDE_distinct_convex_polygons_l501_50186


namespace NUMINAMATH_CALUDE_ladder_slide_l501_50111

theorem ladder_slide (initial_length initial_base_distance slip_distance : ℝ) 
  (h1 : initial_length = 30)
  (h2 : initial_base_distance = 6)
  (h3 : slip_distance = 5) :
  let initial_height := Real.sqrt (initial_length ^ 2 - initial_base_distance ^ 2)
  let new_height := initial_height - slip_distance
  let new_base_distance := Real.sqrt (initial_length ^ 2 - new_height ^ 2)
  new_base_distance - initial_base_distance = Real.sqrt (11 + 120 * Real.sqrt 6) - 6 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slide_l501_50111


namespace NUMINAMATH_CALUDE_spinner_probability_l501_50129

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/2 →
  p_B = 1/8 →
  p_C = p_D →
  p_A + p_B + p_C + p_D = 1 →
  p_C = 3/16 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l501_50129


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_18_30_45_l501_50132

/-- The sum of the greatest common factor and least common multiple of 18, 30, and 45 is 93 -/
theorem gcd_lcm_sum_18_30_45 : 
  (Nat.gcd 18 (Nat.gcd 30 45) + Nat.lcm 18 (Nat.lcm 30 45)) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_18_30_45_l501_50132


namespace NUMINAMATH_CALUDE_last_passenger_correct_seat_prob_l501_50105

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  tickets : Fin n → Seat

/-- Represents a passenger -/
inductive Passenger
| scientist
| regular (id : ℕ)

/-- Represents a seat -/
def Seat := ℕ

/-- The seating process for the bus -/
def seatingProcess (b : Bus n) : Bus n := sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerInCorrectSeat (b : Bus n) : ℚ := sorry

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 -/
theorem last_passenger_correct_seat_prob (n : ℕ) (b : Bus n) :
  lastPassengerInCorrectSeat (seatingProcess b) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_last_passenger_correct_seat_prob_l501_50105


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_unit_circle_sum_one_implies_diff_sqrt_three_l501_50177

variables (z₁ z₂ : ℂ)

-- Statement B
theorem product_zero_implies_factor_zero : z₁ * z₂ = 0 → z₁ = 0 ∨ z₂ = 0 := by sorry

-- Statement D
theorem unit_circle_sum_one_implies_diff_sqrt_three : 
  Complex.abs z₁ = 1 → Complex.abs z₂ = 1 → z₁ + z₂ = 1 → Complex.abs (z₁ - z₂) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_unit_circle_sum_one_implies_diff_sqrt_three_l501_50177


namespace NUMINAMATH_CALUDE_area_of_JKLMNO_l501_50155

/-- Represents a polygon with 6 vertices -/
structure Hexagon :=
  (J K L M N O : ℝ × ℝ)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Calculate the area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The given polygon JKLMNO -/
def polygon : Hexagon := sorry

/-- The intersection point P -/
def P : Point := sorry

/-- Theorem: The area of polygon JKLMNO is 62 square units -/
theorem area_of_JKLMNO : 
  let JK : ℝ := 8
  let KL : ℝ := 10
  let OP : ℝ := 6
  let PM : ℝ := 3
  let area_JKLMNP := rectangleArea JK KL
  let area_PMNO := rectangleArea PM OP
  area_JKLMNP - area_PMNO = 62 := by sorry

end NUMINAMATH_CALUDE_area_of_JKLMNO_l501_50155


namespace NUMINAMATH_CALUDE_labourerPayCorrect_l501_50125

/-- Calculates the total amount received by a labourer given the engagement conditions and absence -/
def labourerPay (totalDays : ℕ) (payRate : ℚ) (fineRate : ℚ) (absentDays : ℕ) : ℚ :=
  let workedDays := totalDays - absentDays
  let totalEarned := (workedDays : ℚ) * payRate
  let totalFine := (absentDays : ℚ) * fineRate
  totalEarned - totalFine

/-- The labourer's pay calculation is correct for the given conditions -/
theorem labourerPayCorrect :
  labourerPay 25 2 0.5 5 = 37.5 := by
  sorry

#eval labourerPay 25 2 0.5 5

end NUMINAMATH_CALUDE_labourerPayCorrect_l501_50125


namespace NUMINAMATH_CALUDE_count_integers_in_range_l501_50119

theorem count_integers_in_range : 
  (Finset.filter (fun x => 30 < x^2 + 8*x + 16 ∧ x^2 + 8*x + 16 < 60) (Finset.range 100)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l501_50119


namespace NUMINAMATH_CALUDE_janet_dresses_pockets_l501_50117

theorem janet_dresses_pockets :
  -- Total number of dresses
  ∀ total_dresses : ℕ,
  -- Number of dresses with pockets
  ∀ dresses_with_pockets : ℕ,
  -- Number of dresses with 2 pockets
  ∀ dresses_with_two_pockets : ℕ,
  -- Total number of pockets
  ∀ total_pockets : ℕ,
  -- Conditions
  total_dresses = 24 →
  dresses_with_pockets = total_dresses / 2 →
  dresses_with_two_pockets = dresses_with_pockets / 3 →
  total_pockets = 32 →
  -- Conclusion
  (total_pockets - 2 * dresses_with_two_pockets) / (dresses_with_pockets - dresses_with_two_pockets) = 3 :=
by sorry

end NUMINAMATH_CALUDE_janet_dresses_pockets_l501_50117


namespace NUMINAMATH_CALUDE_pizza_order_count_l501_50140

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) :
  total_slices / slices_per_pizza = 21 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_count_l501_50140


namespace NUMINAMATH_CALUDE_quadratic_roots_at_minimum_l501_50141

/-- Given a quadratic function y = ax² + bx + c with a ≠ 0 and its lowest point at (1, -1),
    the roots of ax² + bx + c = -1 are both equal to 1. -/
theorem quadratic_roots_at_minimum (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ a * 1^2 + b * 1 + c) →
  (a * 1^2 + b * 1 + c = -1) →
  (∀ x, a * x^2 + b * x + c = -1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_at_minimum_l501_50141


namespace NUMINAMATH_CALUDE_integer_pair_conditions_l501_50187

theorem integer_pair_conditions (a b : ℕ+) : 
  (∃ k : ℕ, a^3 = k * b^2) ∧ 
  (∃ m : ℕ, b - 1 = m * (a - 1)) → 
  (a = b) ∨ (b = 1) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_conditions_l501_50187


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l501_50122

/-- The sum of the lengths of quarter-circles approaches a value between the diameter and semi-circumference -/
theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∃ (L : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |2 * n * (π * D / (8 * n)) - L| < ε) ∧
             D < L ∧ L < π * D / 2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l501_50122


namespace NUMINAMATH_CALUDE_square_division_theorem_l501_50199

theorem square_division_theorem :
  ∃ (s : ℝ) (a b : ℝ) (n m : ℕ),
    s > 0 ∧ a > 0 ∧ b > 0 ∧
    b / a ≤ 1.25 ∧
    n + m = 40 ∧
    s * s = n * a * a + m * b * b :=
by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l501_50199


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l501_50166

theorem simplify_algebraic_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = 2 * a * (a - b) / b :=
sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l501_50166


namespace NUMINAMATH_CALUDE_perpendicular_lines_direction_vectors_l501_50126

theorem perpendicular_lines_direction_vectors (b : ℝ) :
  let v1 : Fin 2 → ℝ := ![- 5, 11]
  let v2 : Fin 2 → ℝ := ![b, 3]
  (∀ i : Fin 2, (v1 • v2) = 0) → b = 33 / 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_direction_vectors_l501_50126


namespace NUMINAMATH_CALUDE_probability_of_drawing_specific_balls_l501_50178

theorem probability_of_drawing_specific_balls (red white blue black : ℕ) : 
  red = 5 → white = 4 → blue = 3 → black = 6 →
  (red * white * blue : ℚ) / ((red + white + blue + black) * (red + white + blue + black - 1) * (red + white + blue + black - 2)) = 5 / 408 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_specific_balls_l501_50178


namespace NUMINAMATH_CALUDE_lcm_of_25_35_50_l501_50157

theorem lcm_of_25_35_50 : Nat.lcm 25 (Nat.lcm 35 50) = 350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_25_35_50_l501_50157


namespace NUMINAMATH_CALUDE_largest_even_number_with_sum_20_l501_50160

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- The theorem stating that 86420 is the largest even number with all different digits whose digits add up to 20 -/
theorem largest_even_number_with_sum_20 : 
  ∀ n : ℕ, 
    n % 2 = 0 ∧ 
    has_different_digits n ∧ 
    sum_of_digits n = 20 → 
    n ≤ 86420 := by sorry

end NUMINAMATH_CALUDE_largest_even_number_with_sum_20_l501_50160


namespace NUMINAMATH_CALUDE_quadratic_reducible_conditions_l501_50128

def is_quadratic_or_reducible (a b : ℚ) : Prop :=
  ∃ (p q r : ℚ), ∀ x : ℚ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
    (a / (1 - x) - 2 / (2 - x) + 3 / (3 - x) - 4 / (4 - x) + b / (5 - x) = 0) ↔
    (p * x^2 + q * x + r = 0)

theorem quadratic_reducible_conditions :
  ∀ a b : ℚ, is_quadratic_or_reducible a b ↔
    ((a, b) = (1, 2) ∨
     (a, b) = (13/48, 178/48) ∨
     (a, b) = (9/14, 5/2) ∨
     (a, b) = (1/2, 5/2) ∨
     (a, b) = (0, 0)) := by sorry

end NUMINAMATH_CALUDE_quadratic_reducible_conditions_l501_50128


namespace NUMINAMATH_CALUDE_gain_percentage_calculation_l501_50159

def cost_price : ℝ := 180
def selling_price : ℝ := 216

theorem gain_percentage_calculation : 
  let gain_percentage := (selling_price / cost_price - 1) * 100
  gain_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_gain_percentage_calculation_l501_50159


namespace NUMINAMATH_CALUDE_tan_75_deg_l501_50162

/-- Proves that tan 75° = 2 + √3 given tan 60° and tan 15° -/
theorem tan_75_deg (tan_60_deg : Real.tan (60 * π / 180) = Real.sqrt 3)
                   (tan_15_deg : Real.tan (15 * π / 180) = 2 - Real.sqrt 3) :
  Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_deg_l501_50162


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l501_50174

theorem consecutive_integers_product (a b c d e : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (e = d + 1) →
  (a * b * c * d * e = 15120) →
  (e = 9) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l501_50174


namespace NUMINAMATH_CALUDE_blue_to_yellow_ratio_l501_50136

/-- Represents the number of fish of each color in the aquarium -/
structure FishCount where
  yellow : ℕ
  blue : ℕ
  green : ℕ
  other : ℕ

/-- The conditions of the aquarium -/
def aquariumConditions (f : FishCount) : Prop :=
  f.yellow = 12 ∧
  f.green = 2 * f.yellow ∧
  f.yellow + f.blue + f.green + f.other = 42

/-- The theorem stating the ratio of blue to yellow fish -/
theorem blue_to_yellow_ratio (f : FishCount) 
  (h : aquariumConditions f) : 
  f.blue * 2 = f.yellow := by sorry

end NUMINAMATH_CALUDE_blue_to_yellow_ratio_l501_50136


namespace NUMINAMATH_CALUDE_factorization_problem_l501_50184

theorem factorization_problem (C D : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 76 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l501_50184


namespace NUMINAMATH_CALUDE_bottle_caps_problem_l501_50189

theorem bottle_caps_problem (sammy janine billie : ℕ) 
  (h1 : sammy = 8)
  (h2 : sammy = janine + 2)
  (h3 : janine = 3 * billie) :
  billie = 2 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_problem_l501_50189


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l501_50196

theorem factor_implies_m_value (x y m : ℝ) : 
  (∃ k : ℝ, (1 - 2*x + y) * k = 4*x*y - 4*x^2 - y^2 - m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l501_50196


namespace NUMINAMATH_CALUDE_at_least_one_passes_l501_50176

def exam_pool : ℕ := 10
def A_correct : ℕ := 6
def B_correct : ℕ := 8
def test_questions : ℕ := 3
def passing_threshold : ℕ := 2

def prob_A_pass : ℚ := (Nat.choose A_correct 2 * Nat.choose (exam_pool - A_correct) 1 + Nat.choose A_correct 3) / Nat.choose exam_pool test_questions

def prob_B_pass : ℚ := (Nat.choose B_correct 2 * Nat.choose (exam_pool - B_correct) 1 + Nat.choose B_correct 3) / Nat.choose exam_pool test_questions

theorem at_least_one_passes : 
  1 - (1 - prob_A_pass) * (1 - prob_B_pass) = 44 / 45 := by sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l501_50176


namespace NUMINAMATH_CALUDE_three_roots_implies_a_plus_minus_four_l501_50145

theorem three_roots_implies_a_plus_minus_four (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + a*x| = 4 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  a = 4 ∨ a = -4 :=
by sorry

end NUMINAMATH_CALUDE_three_roots_implies_a_plus_minus_four_l501_50145


namespace NUMINAMATH_CALUDE_same_problem_probability_l501_50112

/-- The probability of two students choosing the same problem out of three options --/
theorem same_problem_probability : 
  let num_problems : ℕ := 3
  let num_students : ℕ := 2
  let total_outcomes : ℕ := num_problems ^ num_students
  let favorable_outcomes : ℕ := num_problems
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_same_problem_probability_l501_50112


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l501_50195

theorem quadratic_equation_solution (a b : ℕ+) :
  (∃ x : ℝ, x^2 + 14*x = 24 ∧ x > 0 ∧ x = Real.sqrt a - b) →
  a + b = 80 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l501_50195


namespace NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l501_50163

def fair_12_sided_die : Finset ℕ := Finset.range 12

theorem expected_value_fair_12_sided_die : 
  (fair_12_sided_die.sum (λ x => (x + 1) * (1 : ℚ)) / 12) = (13 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l501_50163


namespace NUMINAMATH_CALUDE_cannot_reach_2003_l501_50182

/-- The set of numbers that can appear on the board -/
def BoardNumbers : Set ℕ :=
  {n : ℕ | ∃ (k : ℕ), n ≡ 5 [ZMOD 5] ∨ n ≡ 7 [ZMOD 5] ∨ n ≡ 9 [ZMOD 5]}

/-- The transformation rule -/
def Transform (a b : ℕ) : ℕ := 5 * a - 4 * b

/-- Theorem stating that 2003 cannot appear on the board -/
theorem cannot_reach_2003 : 2003 ∉ BoardNumbers := by
  sorry

/-- Lemma: The transformation preserves the set of possible remainders modulo 5 -/
lemma transform_preserves_remainders (a b : ℕ) (h : a ∈ BoardNumbers) (h' : b ∈ BoardNumbers) :
  Transform a b ∈ BoardNumbers := by
  sorry

end NUMINAMATH_CALUDE_cannot_reach_2003_l501_50182


namespace NUMINAMATH_CALUDE_circle_equation_l501_50102

/-- Given a circle passing through points A(0,-6) and B(1,-5), with its center lying on the line x-y+1=0,
    prove that the standard equation of the circle is (x+3)^2 + (y+2)^2 = 25. -/
theorem circle_equation (C : ℝ × ℝ) : 
  (C.1 - C.2 + 1 = 0) →  -- Center lies on the line x-y+1=0
  ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 = ((1 : ℝ) - C.1)^2 + ((-5 : ℝ) - C.2)^2 →  -- Circle passes through A and B
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l501_50102


namespace NUMINAMATH_CALUDE_complex_root_coefficients_l501_50108

theorem complex_root_coefficients :
  ∀ (b c : ℝ),
  (∃ (z : ℂ), z = 1 + Complex.I * Real.sqrt 2 ∧ z^2 + b*z + c = 0) →
  b = -2 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_root_coefficients_l501_50108


namespace NUMINAMATH_CALUDE_expression_evaluation_l501_50172

theorem expression_evaluation (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 2) / (3 * x^3))^2) = 
  (Real.sqrt ((x^6 + 4) * (x^6 + 1))) / (3 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l501_50172


namespace NUMINAMATH_CALUDE_power_function_quadrant_propositions_l501_50146

-- Define a power function
def is_power_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

-- Define the property of not passing through the fourth quadrant
def not_in_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y < 0)

-- The main theorem
theorem power_function_quadrant_propositions :
  let P : (ℝ → ℝ) → Prop := λ f => is_power_function f → not_in_fourth_quadrant f
  let contrapositive : (ℝ → ℝ) → Prop := λ f => ¬(not_in_fourth_quadrant f) → ¬(is_power_function f)
  let converse : (ℝ → ℝ) → Prop := λ f => not_in_fourth_quadrant f → is_power_function f
  let inverse : (ℝ → ℝ) → Prop := λ f => ¬(is_power_function f) → ¬(not_in_fourth_quadrant f)
  (∀ f : ℝ → ℝ, P f) ∧
  (∀ f : ℝ → ℝ, contrapositive f) ∧
  ¬(∀ f : ℝ → ℝ, converse f) ∧
  ¬(∀ f : ℝ → ℝ, inverse f) :=
by sorry

end NUMINAMATH_CALUDE_power_function_quadrant_propositions_l501_50146


namespace NUMINAMATH_CALUDE_book_cost_solution_l501_50191

/-- Represents the cost of books problem --/
def BookCostProblem (initial_budget : ℚ) (books_per_series : ℕ) (series_bought : ℕ) (money_left : ℚ) (tax_rate : ℚ) : Prop :=
  let total_books := books_per_series * series_bought
  let money_spent := initial_budget - money_left
  let pre_tax_total := money_spent / (1 + tax_rate)
  let book_cost := pre_tax_total / total_books
  book_cost = 60 / 11

/-- Theorem stating the solution to the book cost problem --/
theorem book_cost_solution :
  BookCostProblem 200 8 3 56 (1/10) :=
sorry

end NUMINAMATH_CALUDE_book_cost_solution_l501_50191


namespace NUMINAMATH_CALUDE_max_value_of_f_l501_50106

def f (x : ℝ) : ℝ := -4 * x^2 + 10 * x

theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l501_50106


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l501_50144

/-- The number of ways to arrange 6 rings out of 10 on 5 fingers -/
def ring_arrangements : ℕ := sorry

/-- The number of ways to choose 6 rings out of 10 -/
def choose_rings : ℕ := sorry

/-- The number of ways to order 6 rings -/
def order_rings : ℕ := sorry

/-- The number of ways to distribute 6 rings among 5 fingers -/
def distribute_rings : ℕ := sorry

theorem ring_arrangement_count :
  ring_arrangements = choose_rings * order_rings * distribute_rings ∧
  ring_arrangements = 31752000 :=
sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l501_50144


namespace NUMINAMATH_CALUDE_percentage_not_sold_is_62_14_l501_50151

/-- Represents the book inventory and sales data for a bookshop --/
structure BookshopData where
  initial_fiction : ℕ
  initial_nonfiction : ℕ
  fiction_sold : ℕ
  nonfiction_sold : ℕ
  fiction_returned : ℕ
  nonfiction_returned : ℕ

/-- Calculates the percentage of books not sold --/
def percentage_not_sold (data : BookshopData) : ℚ :=
  let total_initial := data.initial_fiction + data.initial_nonfiction
  let net_fiction_sold := data.fiction_sold - data.fiction_returned
  let net_nonfiction_sold := data.nonfiction_sold - data.nonfiction_returned
  let total_sold := net_fiction_sold + net_nonfiction_sold
  let not_sold := total_initial - total_sold
  (not_sold : ℚ) / (total_initial : ℚ) * 100

/-- The main theorem stating the percentage of books not sold --/
theorem percentage_not_sold_is_62_14 (data : BookshopData)
  (h1 : data.initial_fiction = 400)
  (h2 : data.initial_nonfiction = 300)
  (h3 : data.fiction_sold = 150)
  (h4 : data.nonfiction_sold = 160)
  (h5 : data.fiction_returned = 30)
  (h6 : data.nonfiction_returned = 15) :
  percentage_not_sold data = 62.14 := by
  sorry

#eval percentage_not_sold {
  initial_fiction := 400,
  initial_nonfiction := 300,
  fiction_sold := 150,
  nonfiction_sold := 160,
  fiction_returned := 30,
  nonfiction_returned := 15
}

end NUMINAMATH_CALUDE_percentage_not_sold_is_62_14_l501_50151


namespace NUMINAMATH_CALUDE_john_completion_time_l501_50142

/-- Represents the time it takes to complete a task -/
structure TaskTime where
  days : ℝ
  time_positive : days > 0

/-- Represents a person's ability to complete a task -/
structure Worker where
  time_to_complete : TaskTime

/-- Represents two people working together on a task -/
structure TeamWork where
  worker1 : Worker
  worker2 : Worker
  time_to_complete : TaskTime
  jane_leaves_early : ℝ
  jane_leaves_early_positive : jane_leaves_early > 0
  jane_leaves_early_less_than_total : jane_leaves_early < time_to_complete.days

theorem john_completion_time 
  (john : Worker) 
  (jane : Worker) 
  (team : TeamWork) :
  team.worker1 = john →
  team.worker2 = jane →
  jane.time_to_complete.days = 12 →
  team.time_to_complete.days = 10 →
  team.jane_leaves_early = 4 →
  john.time_to_complete.days = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_completion_time_l501_50142


namespace NUMINAMATH_CALUDE_cube_root_monotone_l501_50152

theorem cube_root_monotone (a b : ℝ) : a ≤ b → (a ^ (1/3 : ℝ)) ≤ (b ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_monotone_l501_50152


namespace NUMINAMATH_CALUDE_min_chicken_hits_l501_50134

def ring_toss (chicken monkey dog : ℕ) : Prop :=
  chicken * 9 + monkey * 5 + dog * 2 = 61 ∧
  chicken + monkey + dog = 10 ∧
  chicken ≥ 1 ∧ monkey ≥ 1 ∧ dog ≥ 1

theorem min_chicken_hits :
  ∀ chicken monkey dog : ℕ,
    ring_toss chicken monkey dog →
    chicken ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_min_chicken_hits_l501_50134


namespace NUMINAMATH_CALUDE_divisibility_pairs_l501_50156

def satisfies_condition (a b : ℕ) : Prop :=
  (a + 1) % b = 0 ∧ (b + 1) % a = 0

theorem divisibility_pairs :
  ∀ a b : ℕ, satisfies_condition a b ↔ ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l501_50156


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l501_50185

theorem intersection_and_union_of_sets (x : ℝ) 
  (A : Set ℝ) (B : Set ℝ)
  (hA : A = {-3, x^2, x+1})
  (hB : B = {x-3, 2*x-1, x^2+1})
  (hIntersection : A ∩ B = {-3}) :
  x = -1 ∧ A ∪ B = {-4, -3, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l501_50185


namespace NUMINAMATH_CALUDE_complex_equation_solution_l501_50130

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 4 →
  c * d = x - 3 * Complex.I →
  x > 0 →
  x = 3 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l501_50130


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l501_50107

-- Define the hyperbola and its properties
def Hyperbola (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2

-- Define the point P on the right branch of the hyperbola
def PointOnHyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (x, y) := P
  x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0

-- Define the right focus F₂
def RightFocus (F₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  F₂ = (c, 0)

-- Define the midpoint M of PF₂
def Midpoint (M P F₂ : ℝ × ℝ) : Prop :=
  M = ((P.1 + F₂.1) / 2, (P.2 + F₂.2) / 2)

-- Define the property |OF₂| = |F₂M|
def EqualDistances (O F₂ M : ℝ × ℝ) : Prop :=
  (F₂.1 - O.1)^2 + (F₂.2 - O.2)^2 = (M.1 - F₂.1)^2 + (M.2 - F₂.2)^2

-- Define the dot product property
def DotProductProperty (O F₂ M : ℝ × ℝ) (c : ℝ) : Prop :=
  (F₂.1 - O.1) * (M.1 - F₂.1) + (F₂.2 - O.2) * (M.2 - F₂.2) = c^2 / 2

-- The main theorem
theorem hyperbola_eccentricity 
  (a b c : ℝ) (O P F₂ M : ℝ × ℝ) 
  (h1 : Hyperbola a b c)
  (h2 : PointOnHyperbola P a b)
  (h3 : RightFocus F₂ c)
  (h4 : Midpoint M P F₂)
  (h5 : EqualDistances O F₂ M)
  (h6 : DotProductProperty O F₂ M c)
  (h7 : O = (0, 0)) :
  c / a = (Real.sqrt 3 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l501_50107


namespace NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l501_50168

open Real

/-- An angle is in the third quadrant if it's between π and 3π/2 -/
def in_third_quadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3*π/2

/-- An angle is in the second quadrant if it's between π/2 and π -/
def in_second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

theorem half_angle_in_second_quadrant (θ : ℝ) 
  (h1 : in_third_quadrant θ) 
  (h2 : |cos θ| = -cos (θ/2)) : 
  in_second_quadrant (θ/2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l501_50168


namespace NUMINAMATH_CALUDE_regina_farm_correct_l501_50153

/-- Represents the farm animals and their selling prices -/
structure Farm where
  cows : ℕ
  pigs : ℕ
  cow_price : ℕ
  pig_price : ℕ

/-- Regina's farm satisfying the given conditions -/
def regina_farm : Farm where
  cows := 20  -- We'll prove this is correct
  pigs := 80  -- Four times the number of cows
  cow_price := 800
  pig_price := 400

/-- The total sale value of all animals on the farm -/
def total_sale_value (f : Farm) : ℕ :=
  f.cows * f.cow_price + f.pigs * f.pig_price

theorem regina_farm_correct :
  regina_farm.pigs = 4 * regina_farm.cows ∧
  total_sale_value regina_farm = 48000 := by
  sorry

#eval regina_farm.cows  -- Should output 20

end NUMINAMATH_CALUDE_regina_farm_correct_l501_50153


namespace NUMINAMATH_CALUDE_problem_statement_l501_50104

theorem problem_statement :
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 → x^3 + 1/x^3 - 3 = 15) ∧
  (∀ x a b c : ℝ, a = 1/20*x + 20 ∧ b = 1/20*x + 19 ∧ c = 1/20*x + 21 →
    a^2 + b^2 + c^2 - a*b - b*c - a*c = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l501_50104


namespace NUMINAMATH_CALUDE_last_three_digits_sum_l501_50198

theorem last_three_digits_sum (n : ℕ) : 9^15 + 15^15 ≡ 24 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_sum_l501_50198


namespace NUMINAMATH_CALUDE_max_draw_without_pair_is_four_l501_50197

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (white : Nat)
  (blue : Nat)
  (red : Nat)

/-- Represents the maximum number of socks that can be drawn without guaranteeing a pair -/
def maxDrawWithoutPair (drawer : SockDrawer) : Nat :=
  4

/-- Theorem stating that for the given sock drawer, the maximum number of socks
    that can be drawn without guaranteeing a pair is 4 -/
theorem max_draw_without_pair_is_four (drawer : SockDrawer) 
  (h1 : drawer.white = 16) 
  (h2 : drawer.blue = 3) 
  (h3 : drawer.red = 6) : 
  maxDrawWithoutPair drawer = 4 := by
  sorry

#eval maxDrawWithoutPair { white := 16, blue := 3, red := 6 }

end NUMINAMATH_CALUDE_max_draw_without_pair_is_four_l501_50197


namespace NUMINAMATH_CALUDE_sum_of_digits_45_40_l501_50143

def product_45_40 : Nat := 45 * 40

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_45_40 : sum_of_digits product_45_40 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_45_40_l501_50143


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l501_50150

theorem min_value_trig_expression (α β : Real) :
  ∃ (min : Real),
    (∀ (α' β' : Real), (3 * Real.cos α' + 4 * Real.sin β' - 7)^2 + (3 * Real.sin α' + 4 * Real.cos β' - 12)^2 ≥ min) ∧
    ((3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = min) ∧
    min = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l501_50150


namespace NUMINAMATH_CALUDE_triangle_area_equation_l501_50179

theorem triangle_area_equation : ∃! (x : ℝ), x > 3 ∧ (1/2 : ℝ) * (x - 3) * (3*x + 7) = 12*x - 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_equation_l501_50179


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l501_50100

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (on_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (planes_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_condition 
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : on_plane m α)
  (h2 : on_plane n α)
  (h3 : m ≠ n)
  (h4 : on_plane l₁ β)
  (h5 : on_plane l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : parallel m l₁)
  (h8 : parallel n l₂) :
  planes_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l501_50100


namespace NUMINAMATH_CALUDE_xy_less_18_implies_x_less_2_or_y_less_9_l501_50135

theorem xy_less_18_implies_x_less_2_or_y_less_9 :
  ∀ x y : ℝ, x * y < 18 → x < 2 ∨ y < 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_less_18_implies_x_less_2_or_y_less_9_l501_50135


namespace NUMINAMATH_CALUDE_linear_function_unique_l501_50147

/-- A function f: ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem linear_function_unique
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (f x) = 4 * x + 6)
  (h2 : Increasing f) :
  ∀ x, f x = 2 * x + 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_unique_l501_50147


namespace NUMINAMATH_CALUDE_square_roots_theorem_l501_50154

theorem square_roots_theorem (a : ℝ) :
  (3 - a) ^ 2 = (2 * a + 1) ^ 2 → (3 - a) ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l501_50154


namespace NUMINAMATH_CALUDE_product_of_primes_in_final_positions_l501_50124

-- Define the colors
inductive Color
| Red
| Yellow
| Green
| Blue

-- Define the positions in a 2x2 grid
inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

-- Define the transformation function
def transform (c : Color) : Position → Position
| Position.TopLeft => 
    match c with
    | Color.Red => Position.TopRight
    | Color.Yellow => Position.TopRight
    | Color.Green => Position.BottomLeft
    | Color.Blue => Position.BottomRight
| Position.TopRight => 
    match c with
    | Color.Red => Position.TopRight
    | Color.Yellow => Position.TopRight
    | Color.Green => Position.BottomRight
    | Color.Blue => Position.BottomRight
| Position.BottomLeft => 
    match c with
    | Color.Red => Position.TopLeft
    | Color.Yellow => Position.TopLeft
    | Color.Green => Position.BottomLeft
    | Color.Blue => Position.BottomLeft
| Position.BottomRight => 
    match c with
    | Color.Red => Position.TopLeft
    | Color.Yellow => Position.TopLeft
    | Color.Green => Position.BottomRight
    | Color.Blue => Position.BottomRight

-- Define the numbers in Figure 4
def figure4 (p : Position) : Nat :=
  match p with
  | Position.TopLeft => 6
  | Position.TopRight => 7
  | Position.BottomLeft => 5
  | Position.BottomRight => 8

-- Define primality
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem product_of_primes_in_final_positions : 
  let finalRedPosition := transform Color.Red (transform Color.Red Position.TopLeft)
  let finalYellowPosition := transform Color.Yellow (transform Color.Yellow Position.TopRight)
  (isPrime (figure4 finalRedPosition) ∧ isPrime (figure4 finalYellowPosition)) →
  figure4 finalRedPosition * figure4 finalYellowPosition = 55 := by
  sorry


end NUMINAMATH_CALUDE_product_of_primes_in_final_positions_l501_50124


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l501_50149

/-- Slope of the line tangent to a circle -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (3, 2) → point = (5, 5) → 
  (let radius_slope := (point.2 - center.2) / (point.1 - center.1);
   -1 / radius_slope) = -2/3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l501_50149


namespace NUMINAMATH_CALUDE_sum_of_cubes_l501_50137

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l501_50137


namespace NUMINAMATH_CALUDE_outfits_count_l501_50120

/-- The number of possible outfits given the number of shirts, pants, and shoes. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (shoes : ℕ) : ℕ :=
  shirts * pants * shoes

/-- Theorem stating that the number of outfits from 4 shirts, 5 pants, and 2 shoes is 40. -/
theorem outfits_count : number_of_outfits 4 5 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l501_50120


namespace NUMINAMATH_CALUDE_power_of_power_l501_50169

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l501_50169


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l501_50170

theorem shaded_area_of_concentric_circles 
  (outer_circle_area : ℝ)
  (inner_circle_radius : ℝ)
  (h1 : outer_circle_area = 81 * Real.pi)
  (h2 : inner_circle_radius = 4.5)
  : ∃ (shaded_area : ℝ), shaded_area = 54 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l501_50170


namespace NUMINAMATH_CALUDE_cubic_fraction_inequality_l501_50115

theorem cubic_fraction_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) +
  (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) +
  (d^3 + a^3 + b^3) / (d + a + b) ≥
  a^2 + b^2 + c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_inequality_l501_50115


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l501_50127

theorem sum_of_x_and_y (x y : ℝ) : 
  |x - 2*y - 3| + (y - 2*x)^2 = 0 → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l501_50127


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_l501_50109

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric -/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Checks if a polygon is inside a triangle -/
def isInsideTriangle (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Checks if a polygon is a hexagon -/
def isHexagon (p : Polygon) : Prop := sorry

/-- Checks if the vertices of a polygon divide the sides of a triangle into three equal parts -/
def verticesDivideSides (p : Polygon) (t : Triangle) : Prop := sorry

theorem largest_centrally_symmetric_polygon (t : Triangle) :
  ∃ (p : Polygon),
    isCentrallySymmetric p ∧
    isInsideTriangle p t ∧
    isHexagon p ∧
    verticesDivideSides p t ∧
    area p = (2/3) * triangleArea t ∧
    ∀ (q : Polygon),
      isCentrallySymmetric q → isInsideTriangle q t →
      area q ≤ area p :=
sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_l501_50109


namespace NUMINAMATH_CALUDE_marj_wallet_remaining_l501_50173

/-- Calculates the remaining money in Marj's wallet after expenses --/
def remaining_money (initial_usd : ℚ) (initial_euro : ℚ) (initial_pound : ℚ) 
  (euro_to_usd : ℚ) (pound_to_usd : ℚ) (cake_cost : ℚ) (gift_cost : ℚ) (donation : ℚ) : ℚ :=
  initial_usd + initial_euro * euro_to_usd + initial_pound * pound_to_usd - cake_cost - gift_cost - donation

/-- Theorem stating that Marj will have $64.40 left in her wallet after expenses --/
theorem marj_wallet_remaining : 
  remaining_money 81.5 10 5 1.18 1.32 17.5 12.7 5.3 = 64.4 := by
  sorry

end NUMINAMATH_CALUDE_marj_wallet_remaining_l501_50173
