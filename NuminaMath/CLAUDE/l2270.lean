import Mathlib

namespace NUMINAMATH_CALUDE_calculate_running_speed_l2270_227046

/-- Given a swimming speed and an average speed for swimming and running,
    calculate the running speed. -/
theorem calculate_running_speed
  (swimming_speed : ℝ)
  (average_speed : ℝ)
  (h1 : swimming_speed = 1)
  (h2 : average_speed = 4.5)
  : (2 * average_speed - swimming_speed) = 8 := by
  sorry

#check calculate_running_speed

end NUMINAMATH_CALUDE_calculate_running_speed_l2270_227046


namespace NUMINAMATH_CALUDE_triangle_line_equation_l2270_227083

/-- A line passing through a point and forming a triangle with coordinate axes -/
structure TriangleLine where
  -- Coefficients of the line equation ax + by = c
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the point (-2, 2)
  passes_through_point : a * (-2) + b * 2 = c
  -- The line forms a triangle with area 1
  triangle_area : |a * b| / 2 = 1

/-- The equation of the line is either x + 2y - 2 = 0 or 2x + y + 2 = 0 -/
theorem triangle_line_equation (l : TriangleLine) : 
  (l.a = 1 ∧ l.b = 2 ∧ l.c = 2) ∨ (l.a = 2 ∧ l.b = 1 ∧ l.c = -2) :=
sorry

end NUMINAMATH_CALUDE_triangle_line_equation_l2270_227083


namespace NUMINAMATH_CALUDE_min_value_theorem_l2270_227031

theorem min_value_theorem (x : ℝ) (h1 : x > 0) (h2 : Real.log x + 1 ≤ x) :
  (x^2 - Real.log x + x) / x ≥ 2 ∧
  (∃ y > 0, (y^2 - Real.log y + y) / y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2270_227031


namespace NUMINAMATH_CALUDE_cow_plus_cow_equals_milk_l2270_227073

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents an assignment of digits to letters -/
structure LetterAssignment where
  C : Digit
  O : Digit
  W : Digit
  M : Digit
  I : Digit
  L : Digit
  K : Digit
  all_different : C ≠ O ∧ C ≠ W ∧ C ≠ M ∧ C ≠ I ∧ C ≠ L ∧ C ≠ K ∧
                  O ≠ W ∧ O ≠ M ∧ O ≠ I ∧ O ≠ L ∧ O ≠ K ∧
                  W ≠ M ∧ W ≠ I ∧ W ≠ L ∧ W ≠ K ∧
                  M ≠ I ∧ M ≠ L ∧ M ≠ K ∧
                  I ≠ L ∧ I ≠ K ∧
                  L ≠ K

/-- Converts a LetterAssignment to the numeric value of COW -/
def cow_value (assignment : LetterAssignment) : ℕ :=
  100 * assignment.C.val + 10 * assignment.O.val + assignment.W.val

/-- Converts a LetterAssignment to the numeric value of MILK -/
def milk_value (assignment : LetterAssignment) : ℕ :=
  1000 * assignment.M.val + 100 * assignment.I.val + 10 * assignment.L.val + assignment.K.val

/-- The main theorem stating that there are exactly three solutions to the puzzle -/
theorem cow_plus_cow_equals_milk :
  ∃! (solutions : Finset LetterAssignment),
    solutions.card = 3 ∧
    (∀ assignment ∈ solutions, 2 * cow_value assignment = milk_value assignment) :=
sorry

end NUMINAMATH_CALUDE_cow_plus_cow_equals_milk_l2270_227073


namespace NUMINAMATH_CALUDE_sum_even_implies_one_even_l2270_227004

theorem sum_even_implies_one_even (a b c : ℕ) :
  Even (a + b + c) → (Even a ∨ Even b ∨ Even c) :=
sorry

end NUMINAMATH_CALUDE_sum_even_implies_one_even_l2270_227004


namespace NUMINAMATH_CALUDE_car_speed_proof_l2270_227012

/-- The speed of a car in km/h -/
def car_speed : ℝ := 48

/-- The reference speed in km/h -/
def reference_speed : ℝ := 60

/-- The additional time taken in seconds -/
def additional_time : ℝ := 15

/-- The distance traveled in km -/
def distance : ℝ := 1

theorem car_speed_proof :
  (distance / car_speed) * 3600 = (distance / reference_speed) * 3600 + additional_time :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2270_227012


namespace NUMINAMATH_CALUDE_parabola_intersection_and_area_minimization_l2270_227077

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the line passing through M and intersecting the parabola
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the dot product of vectors OA and OB
def dot_product (x1 x2 : ℝ) : ℝ := x1 * x2 + (x1^2) * (x2^2)

theorem parabola_intersection_and_area_minimization 
  (k m : ℝ) -- Parameters of the line
  (x1 x2 : ℝ) -- x-coordinates of intersection points A and B
  (h1 : parabola x1 = line k m x1) -- A is on both parabola and line
  (h2 : parabola x2 = line k m x2) -- B is on both parabola and line
  (h3 : dot_product x1 x2 = 2) -- Given condition
  (h4 : m = 2) -- Line passes through (0, 2)
  : 
  (∃ (x : ℝ), line k m x = 2) ∧ -- Line passes through (0, 2)
  (∃ (area : ℝ), area = 3 ∧ 
    ∀ (x : ℝ), x > 0 → x + 9/(4*x) ≥ area) -- Minimum area is 3
  := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_and_area_minimization_l2270_227077


namespace NUMINAMATH_CALUDE_undefined_fraction_l2270_227005

theorem undefined_fraction (x : ℝ) : x = 1 → ¬∃y : ℝ, y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l2270_227005


namespace NUMINAMATH_CALUDE_square_diagonal_and_inscribed_circle_area_l2270_227096

/-- Given a square with side length 40√3 cm, this theorem proves the length of its diagonal
    and the area of its inscribed circle. -/
theorem square_diagonal_and_inscribed_circle_area 
  (side_length : ℝ) 
  (h_side : side_length = 40 * Real.sqrt 3) :
  ∃ (diagonal_length : ℝ) (inscribed_circle_area : ℝ),
    diagonal_length = 40 * Real.sqrt 6 ∧
    inscribed_circle_area = 1200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_square_diagonal_and_inscribed_circle_area_l2270_227096


namespace NUMINAMATH_CALUDE_cos_difference_special_l2270_227095

theorem cos_difference_special (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_special_l2270_227095


namespace NUMINAMATH_CALUDE_square_circle_radius_l2270_227079

theorem square_circle_radius (r : ℝ) (h : r > 0) :
  4 * r * Real.sqrt 2 = π * r^2 → r = 4 * Real.sqrt 2 / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_radius_l2270_227079


namespace NUMINAMATH_CALUDE_students_taking_both_languages_l2270_227055

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ) :
  total = 94 →
  french = 41 →
  german = 22 →
  neither = 40 →
  ∃ (both : ℕ), both = 9 ∧ total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_taking_both_languages_l2270_227055


namespace NUMINAMATH_CALUDE_perimeter_approx_l2270_227034

/-- A right triangle with area 150 and one leg 15 units longer than the other -/
structure RightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  area_eq : (1/2) * shorter_leg * longer_leg = 150
  leg_diff : longer_leg = shorter_leg + 15
  pythagorean : shorter_leg^2 + longer_leg^2 = hypotenuse^2

/-- The perimeter of the triangle -/
def perimeter (t : RightTriangle) : ℝ :=
  t.shorter_leg + t.longer_leg + t.hypotenuse

/-- Theorem stating that the perimeter is approximately 66.47 -/
theorem perimeter_approx (t : RightTriangle) :
  abs (perimeter t - 66.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_approx_l2270_227034


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2270_227068

theorem regular_polygon_sides (exterior_angle : ℝ) : 
  exterior_angle = 45 → (360 : ℝ) / exterior_angle = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2270_227068


namespace NUMINAMATH_CALUDE_president_vice_selection_ways_l2270_227060

/-- The number of ways to choose a president and vice-president from a club with the given conditions -/
def choose_president_and_vice (total_members boys girls : ℕ) : ℕ :=
  (boys * (boys - 1)) + (girls * (girls - 1))

/-- Theorem stating the number of ways to choose a president and vice-president under the given conditions -/
theorem president_vice_selection_ways :
  let total_members : ℕ := 30
  let boys : ℕ := 18
  let girls : ℕ := 12
  choose_president_and_vice total_members boys girls = 438 := by
  sorry

#eval choose_president_and_vice 30 18 12

end NUMINAMATH_CALUDE_president_vice_selection_ways_l2270_227060


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l2270_227094

def f (x : ℝ) : ℝ := (x - 2)^2

-- Property 1: f(x+2) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f (-x + 2)

-- Property 2: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
def is_decreasing_then_increasing (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x < y ∧ y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x ∧ x < y → f x < f y)

theorem f_satisfies_properties : 
  is_even_shifted f ∧ is_decreasing_then_increasing f :=
sorry

end NUMINAMATH_CALUDE_f_satisfies_properties_l2270_227094


namespace NUMINAMATH_CALUDE_wanda_blocks_count_l2270_227028

/-- The number of blocks Wanda has initially -/
def initial_blocks : ℕ := 4

/-- The number of blocks Theresa gives to Wanda -/
def additional_blocks : ℕ := 79

/-- The total number of blocks Wanda has after receiving blocks from Theresa -/
def total_blocks : ℕ := initial_blocks + additional_blocks

theorem wanda_blocks_count : total_blocks = 83 := by
  sorry

end NUMINAMATH_CALUDE_wanda_blocks_count_l2270_227028


namespace NUMINAMATH_CALUDE_custom_operation_result_l2270_227039

def custom_operation (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y * (x + y)}

theorem custom_operation_result :
  let A : Set ℕ := {0, 1}
  let B : Set ℕ := {2, 3}
  custom_operation A B = {0, 6, 12} := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l2270_227039


namespace NUMINAMATH_CALUDE_interest_problem_solution_l2270_227084

/-- Given conditions for the interest problem -/
structure InterestProblem where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate (as a decimal)
  t : ℝ  -- Time period in years
  diff : ℝ  -- Difference between compound and simple interest

/-- Theorem statement for the interest problem -/
theorem interest_problem_solution (prob : InterestProblem) 
  (h1 : prob.r = 0.1)  -- 10% interest rate
  (h2 : prob.t = 2)  -- 2 years time period
  (h3 : prob.diff = 631)  -- Difference between compound and simple interest is $631
  : prob.P = 63100 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_solution_l2270_227084


namespace NUMINAMATH_CALUDE_product_of_numbers_l2270_227023

theorem product_of_numbers (x y : ℝ) : x + y = 24 → x - y = 8 → x * y = 128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2270_227023


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2270_227035

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c => 
    (a = 9 ∧ b = 9 ∧ c = 4) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
    (a = b) →  -- Isosceles condition
    (a + b + c = 22)  -- Perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 9 9 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2270_227035


namespace NUMINAMATH_CALUDE_probability_is_one_third_l2270_227080

/-- Line represented by slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The region of interest in the first quadrant -/
def Region (p q r : Line) : Set Point :=
  {pt : Point | 0 ≤ pt.x ∧ 0 ≤ pt.y ∧ 
                pt.y ≤ p.slope * pt.x + p.intercept ∧
                r.intercept < pt.y ∧ 
                q.slope * pt.x + q.intercept < pt.y ∧ 
                pt.y < p.slope * pt.x + p.intercept}

/-- The area of the region of interest -/
noncomputable def areaOfRegion (p q r : Line) : ℝ := sorry

/-- The total area under line p and above x-axis in the first quadrant -/
noncomputable def totalArea (p : Line) : ℝ := sorry

/-- The main theorem stating the probability -/
theorem probability_is_one_third 
  (p : Line) 
  (q : Line) 
  (r : Line) 
  (hp : p.slope = -2 ∧ p.intercept = 8) 
  (hq : q.slope = -3 ∧ q.intercept = 8) 
  (hr : r.slope = 0 ∧ r.intercept = 4) : 
  areaOfRegion p q r / totalArea p = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l2270_227080


namespace NUMINAMATH_CALUDE_rotate_A_180_l2270_227086

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Rotates a point 180 degrees clockwise about the origin -/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The original point A -/
def A : Point := { x := -4, y := 1 }

/-- The expected result after rotation -/
def A_rotated : Point := { x := 4, y := -1 }

/-- Theorem stating that rotating A 180 degrees clockwise about the origin results in A_rotated -/
theorem rotate_A_180 : rotate180 A = A_rotated := by sorry

end NUMINAMATH_CALUDE_rotate_A_180_l2270_227086


namespace NUMINAMATH_CALUDE_product_one_inequality_l2270_227075

theorem product_one_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_one_inequality_l2270_227075


namespace NUMINAMATH_CALUDE_intuitive_diagram_area_l2270_227082

/-- The area of the intuitive diagram of a square in oblique axonometric drawing -/
theorem intuitive_diagram_area (a : ℝ) (h : a > 0) :
  let planar_area := a^2
  let ratio := 2 * Real.sqrt 2
  let intuitive_area := planar_area / ratio
  intuitive_area = (Real.sqrt 2 / 4) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_intuitive_diagram_area_l2270_227082


namespace NUMINAMATH_CALUDE_machines_working_time_l2270_227066

theorem machines_working_time (x : ℝ) : 
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_machines_working_time_l2270_227066


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2270_227030

theorem max_sum_of_squares (m n : ℕ) : 
  1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2270_227030


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2270_227065

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2270_227065


namespace NUMINAMATH_CALUDE_a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4_l2270_227098

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n-1)

theorem a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4 :
  ∃ (a q : ℝ), 
    let seq := geometric_sequence a q
    (seq 1 < seq 3 ∧ seq 2 ≥ seq 4) ∧
    (seq 2 < seq 4 ∧ seq 1 ≥ seq 3) :=
sorry

end NUMINAMATH_CALUDE_a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4_l2270_227098


namespace NUMINAMATH_CALUDE_longer_subsegment_length_l2270_227081

/-- Triangle with sides in ratio 3:4:5 -/
structure Triangle :=
  (a b c : ℝ)
  (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5)

/-- Angle bisector theorem -/
axiom angle_bisector_theorem {t : Triangle} (d : ℝ) :
  d / (t.c - d) = t.a / t.b

/-- Main theorem -/
theorem longer_subsegment_length (t : Triangle) (h : t.c = 15) :
  let d := t.c * (t.a / (t.a + t.b))
  d = 75 / 8 := by sorry

end NUMINAMATH_CALUDE_longer_subsegment_length_l2270_227081


namespace NUMINAMATH_CALUDE_inequality_proof_l2270_227015

variable (x y z : ℝ)

def condition (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z + x * y + y * z + z * x = x + y + z + 1

theorem inequality_proof (h : condition x y z) :
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + 
             Real.sqrt ((1 + y^2) / (1 + y)) + 
             Real.sqrt ((1 + z^2) / (1 + z))) ≤ ((x + y + z) / 3) ^ (5/8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2270_227015


namespace NUMINAMATH_CALUDE_log_difference_equals_eight_l2270_227049

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_difference_equals_eight :
  log 3 243 - log 3 (1/27) = 8 := by sorry

end NUMINAMATH_CALUDE_log_difference_equals_eight_l2270_227049


namespace NUMINAMATH_CALUDE_efficient_methods_l2270_227040

-- Define the types of calculation methods
inductive CalculationMethod
  | Mental
  | Written
  | Calculator

-- Define a function to determine the most efficient method for a given calculation
def most_efficient_method (calculation : ℕ → ℕ → ℕ) : CalculationMethod :=
  sorry

-- Define the specific calculations
def calc1 : ℕ → ℕ → ℕ := λ x y ↦ (x - y) / 5
def calc2 : ℕ → ℕ → ℕ := λ x _ ↦ x * x

-- State the theorem
theorem efficient_methods :
  (most_efficient_method calc1 = CalculationMethod.Calculator) ∧
  (most_efficient_method calc2 = CalculationMethod.Mental) :=
sorry

end NUMINAMATH_CALUDE_efficient_methods_l2270_227040


namespace NUMINAMATH_CALUDE_double_after_increase_decrease_l2270_227045

theorem double_after_increase_decrease (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) :
  N * (1 + r / 100) * (1 - s / 100) = 2 * N ↔ 
  r = (10000 + 100 * s) / (100 - s) :=
by sorry

end NUMINAMATH_CALUDE_double_after_increase_decrease_l2270_227045


namespace NUMINAMATH_CALUDE_jack_emails_l2270_227003

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The total number of emails Jack received in the day -/
def total_emails : ℕ := morning_emails + afternoon_emails

theorem jack_emails : total_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_emails_l2270_227003


namespace NUMINAMATH_CALUDE_farm_heads_count_l2270_227009

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads on the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem stating that a farm with 140 feet and 22 hens has 46 heads -/
theorem farm_heads_count (f : Farm) 
  (feet_count : totalFeet f = 140) 
  (hen_count : f.hens = 22) : 
  totalHeads f = 46 := by
  sorry

end NUMINAMATH_CALUDE_farm_heads_count_l2270_227009


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2270_227016

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6,7,8}

-- Define set A
def A : Finset Nat := {1,4,6}

-- Define set B
def B : Finset Nat := {4,5,7}

-- Theorem statement
theorem complement_intersection_equals_set :
  (U \ A) ∩ (U \ B) = {2,3,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2270_227016


namespace NUMINAMATH_CALUDE_fourth_power_difference_l2270_227051

theorem fourth_power_difference (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_l2270_227051


namespace NUMINAMATH_CALUDE_inequality_range_l2270_227054

theorem inequality_range (P : ℝ) (h : 0 ≤ P ∧ P ≤ 4) :
  (∀ x : ℝ, x^2 + P*x > 4*x + P - 3) ↔ (∀ x : ℝ, x < -1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2270_227054


namespace NUMINAMATH_CALUDE_total_liquid_proof_l2270_227090

/-- The amount of oil used per dish in cups -/
def oil_per_dish : ℝ := 0.17

/-- The amount of water used per dish in cups -/
def water_per_dish : ℝ := 1.17

/-- The number of times the dish is prepared -/
def num_preparations : ℕ := 12

/-- The total amount of liquid used for all preparations -/
def total_liquid : ℝ := (oil_per_dish + water_per_dish) * num_preparations

theorem total_liquid_proof : total_liquid = 16.08 := by
  sorry

end NUMINAMATH_CALUDE_total_liquid_proof_l2270_227090


namespace NUMINAMATH_CALUDE_cost_of_16_pencils_10_notebooks_l2270_227044

/-- The cost of pencils and notebooks given specific quantities -/
def cost_of_items (pencil_price notebook_price : ℚ) (num_pencils num_notebooks : ℕ) : ℚ :=
  pencil_price * num_pencils + notebook_price * num_notebooks

/-- The theorem stating the cost of 16 pencils and 10 notebooks -/
theorem cost_of_16_pencils_10_notebooks :
  ∀ (pencil_price notebook_price : ℚ),
    cost_of_items pencil_price notebook_price 7 8 = 415/100 →
    cost_of_items pencil_price notebook_price 5 3 = 177/100 →
    cost_of_items pencil_price notebook_price 16 10 = 584/100 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_16_pencils_10_notebooks_l2270_227044


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2270_227059

theorem simplify_and_evaluate (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
  (x / (x + 1) - 3 * x / (x - 1)) / (x / (x^2 - 1)) = -8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2270_227059


namespace NUMINAMATH_CALUDE_sqrt_five_power_calculation_l2270_227017

theorem sqrt_five_power_calculation :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 78125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_power_calculation_l2270_227017


namespace NUMINAMATH_CALUDE_midpoint_x_sum_l2270_227057

/-- Given a triangle in the Cartesian plane where the sum of x-coordinates of its vertices is 15,
    the sum of x-coordinates of the midpoints of its sides is also 15. -/
theorem midpoint_x_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_x_sum_l2270_227057


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2270_227087

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  geometric_sequence (a 1) (a 2) (a 5) →
  d = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2270_227087


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2270_227010

theorem quadratic_inequality_solution :
  {z : ℝ | z^2 - 40*z + 340 ≤ 4} = Set.Icc 12 28 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2270_227010


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2270_227041

noncomputable section

/-- Two circles with common center and a point configuration --/
structure CircleConfig where
  a : ℝ
  b : ℝ
  h_ab : a > b

variable (cfg : CircleConfig)

/-- The locus of points Si --/
def locus (cfg : CircleConfig) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ cfg.b^2 / cfg.a ∧
    p.1 = t * cfg.a^2 / cfg.b^2 ∧
    p.2^2 = cfg.b^2 - (t * cfg.a / cfg.b)^2}

/-- The ellipse with major axis 2a and minor axis 2b --/
def ellipse (cfg : CircleConfig) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / cfg.a^2 + p.2^2 / cfg.b^2 = 1 ∧ p.1 ≥ 0}

/-- The main theorem --/
theorem locus_is_ellipse (cfg : CircleConfig) :
  locus cfg = ellipse cfg := by sorry

end

end NUMINAMATH_CALUDE_locus_is_ellipse_l2270_227041


namespace NUMINAMATH_CALUDE_right_triangle_legs_from_altitude_areas_l2270_227037

/-- Given a right-angled triangle ABC with right angle at C, and altitude CD to hypotenuse AB
    dividing the triangle into two triangles with areas Q and q, 
    the legs of the triangle are √(2(q + Q)√(q/Q)) and √(2(q + Q)√(Q/q)). -/
theorem right_triangle_legs_from_altitude_areas (Q q : ℝ) (hQ : Q > 0) (hq : q > 0) :
  ∃ (AC BC : ℝ),
    AC = Real.sqrt (2 * (q + Q) * Real.sqrt (q / Q)) ∧
    BC = Real.sqrt (2 * (q + Q) * Real.sqrt (Q / q)) ∧
    AC^2 + BC^2 = (AC * BC)^2 / (Q + q) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_from_altitude_areas_l2270_227037


namespace NUMINAMATH_CALUDE_rice_division_l2270_227042

/-- 
Given an arithmetic sequence of three terms (a, b, c) where:
- The sum of the terms is 180
- The difference between the first and third term is 36
This theorem proves that the middle term (b) is equal to 60.
-/
theorem rice_division (a b c : ℕ) : 
  a + b + c = 180 →
  a - c = 36 →
  b = 60 := by
  sorry


end NUMINAMATH_CALUDE_rice_division_l2270_227042


namespace NUMINAMATH_CALUDE_nine_times_polygon_properties_l2270_227033

/-- A polygon with interior angles 9 times the exterior angles -/
structure NineTimesPolygon where
  n : ℕ -- number of sides
  interior_angles : Fin n → ℝ
  exterior_angles : Fin n → ℝ
  h_positive : ∀ i, interior_angles i > 0 ∧ exterior_angles i > 0
  h_relation : ∀ i, interior_angles i = 9 * exterior_angles i
  h_exterior_sum : (Finset.univ.sum exterior_angles) = 360

theorem nine_times_polygon_properties (Q : NineTimesPolygon) :
  (Finset.univ.sum Q.interior_angles = 3240) ∧
  (∃ (i j : Fin Q.n), Q.interior_angles i ≠ Q.interior_angles j ∨ Q.interior_angles i = Q.interior_angles j) :=
by sorry

end NUMINAMATH_CALUDE_nine_times_polygon_properties_l2270_227033


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l2270_227089

/-- Represents the number of people in each age group -/
structure PopulationGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Represents the number of people to be sampled from each age group -/
structure SampleGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the total population -/
def totalPopulation (p : PopulationGroups) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the proportion of each group in the sample -/
def sampleProportion (p : PopulationGroups) (sampleSize : ℕ) : SampleGroups :=
  let total := totalPopulation p
  { elderly := (p.elderly * sampleSize + total - 1) / total,
    middleAged := (p.middleAged * sampleSize + total - 1) / total,
    young := (p.young * sampleSize + total - 1) / total }

/-- The main theorem to prove -/
theorem stratified_sampling_correct 
  (population : PopulationGroups)
  (sampleSize : ℕ) :
  population.elderly = 28 →
  population.middleAged = 54 →
  population.young = 81 →
  sampleSize = 36 →
  sampleProportion population sampleSize = { elderly := 6, middleAged := 12, young := 18 } := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_correct_l2270_227089


namespace NUMINAMATH_CALUDE_soccer_team_girls_l2270_227024

theorem soccer_team_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  present = (2 / 3 : ℚ) * boys + girls →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_girls_l2270_227024


namespace NUMINAMATH_CALUDE_gear_rotation_l2270_227070

/-- Represents a gear in the system -/
structure Gear where
  angle : Real

/-- Represents a system of two meshed gears -/
structure GearSystem where
  left : Gear
  right : Gear

/-- Rotates the left gear by a given angle -/
def rotateLeft (system : GearSystem) (θ : Real) : GearSystem :=
  { left := { angle := system.left.angle + θ },
    right := { angle := system.right.angle - θ } }

/-- Theorem stating that rotating the left gear by θ results in the right gear rotating by -θ -/
theorem gear_rotation (system : GearSystem) (θ : Real) :
  (rotateLeft system θ).right.angle = system.right.angle - θ :=
by sorry

end NUMINAMATH_CALUDE_gear_rotation_l2270_227070


namespace NUMINAMATH_CALUDE_chair_price_l2270_227011

theorem chair_price (num_tables : ℕ) (num_chairs : ℕ) (total_cost : ℕ) 
  (h1 : num_tables = 2)
  (h2 : num_chairs = 3)
  (h3 : total_cost = 110)
  (h4 : ∀ (chair_price : ℕ), num_tables * (4 * chair_price) + num_chairs * chair_price = total_cost) :
  ∃ (chair_price : ℕ), chair_price = 10 ∧ 
    num_tables * (4 * chair_price) + num_chairs * chair_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_chair_price_l2270_227011


namespace NUMINAMATH_CALUDE_fish_pond_estimation_l2270_227002

theorem fish_pond_estimation (marked_initial : ℕ) (second_sample : ℕ) (marked_in_sample : ℕ) :
  marked_initial = 40 →
  second_sample = 300 →
  marked_in_sample = 8 →
  (marked_initial * second_sample) / marked_in_sample = 1500 :=
by
  sorry

#check fish_pond_estimation

end NUMINAMATH_CALUDE_fish_pond_estimation_l2270_227002


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l2270_227058

/-- Given two 2D vectors a and b, if a + b is parallel to a - b, 
    then the first component of a is -4/3. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = 2 ∧ b = (2, -3) ∧ 
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) →
  m = -4/3 := by sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l2270_227058


namespace NUMINAMATH_CALUDE_f_properties_l2270_227076

def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x > 2 ∧ y > x → f x < f y) ∧ 
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2270_227076


namespace NUMINAMATH_CALUDE_min_tablets_extracted_l2270_227074

/-- Represents the number of tablets of each medicine type in the box -/
structure MedicineBox where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least three of each type -/
def minTablets (box : MedicineBox) : Nat :=
  (box.a + box.b + box.c) - min (box.a - 3) 0 - min (box.b - 3) 0 - min (box.c - 3) 0

/-- Theorem: The minimum number of tablets to extract from the given box is 48 -/
theorem min_tablets_extracted (box : MedicineBox) 
  (ha : box.a = 20) (hb : box.b = 25) (hc : box.c = 15) : 
  minTablets box = 48 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extracted_l2270_227074


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2270_227036

theorem arithmetic_computation : -9 * 5 - (-7 * -2) + (-11 * -6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2270_227036


namespace NUMINAMATH_CALUDE_min_value_expression_l2270_227013

theorem min_value_expression (x : ℝ) (h : x > 0) : 
  4 * x + 1 / x^2 ≥ 5 ∧ ∃ y > 0, 4 * y + 1 / y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2270_227013


namespace NUMINAMATH_CALUDE_quadratic_intersection_range_l2270_227020

/-- For a quadratic function y = 2mx^2 + (8m+1)x + 8m that intersects the x-axis, 
    the range of m is [m ≥ -1/16 and m ≠ 0] -/
theorem quadratic_intersection_range (m : ℝ) : 
  (∃ x, 2*m*x^2 + (8*m + 1)*x + 8*m = 0) → 
  (m ≥ -1/16 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_range_l2270_227020


namespace NUMINAMATH_CALUDE_parallel_line_through_circle_center_l2270_227027

/-- Given a circle C and a line l₁, prove that the line passing through the center of C
    and parallel to l₁ has the equation 2x - 3y - 8 = 0 -/
theorem parallel_line_through_circle_center 
  (C : Set (ℝ × ℝ))
  (l₁ : Set (ℝ × ℝ))
  (h_C : C = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = 5})
  (h_l₁ : l₁ = {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 + 6 = 0}) :
  ∃ l : Set (ℝ × ℝ), 
    (∀ p ∈ l, 2 * p.1 - 3 * p.2 - 8 = 0) ∧ 
    (∃ c ∈ C, c ∈ l) ∧
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (p.2 - q.2) / (p.1 - q.1) = (2 : ℝ) / 3) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_circle_center_l2270_227027


namespace NUMINAMATH_CALUDE_intersection_symmetry_implies_k_minus_m_eq_four_l2270_227097

/-- The line equation y = kx + 1 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

/-- The circle equation x² + y² + kx + my - 4 = 0 -/
def circle_equation (k m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + k*x + m*y - 4 = 0

/-- The symmetry line equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric with respect to the line x + y - 1 = 0 -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 + (y₁ + y₂) / 2 - 1 = 0

theorem intersection_symmetry_implies_k_minus_m_eq_four (k m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    circle_equation k m x₁ y₁ ∧
    circle_equation k m x₂ y₂ ∧
    symmetric_points x₁ y₁ x₂ y₂) →
  k - m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_symmetry_implies_k_minus_m_eq_four_l2270_227097


namespace NUMINAMATH_CALUDE_problem_solution_l2270_227048

theorem problem_solution (x y z : ℝ) 
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x*(y + z) - y*(x - y)) :
  (y^2 + z^2 - x^2) / (2*y*z) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2270_227048


namespace NUMINAMATH_CALUDE_michelle_boutique_two_ties_probability_l2270_227018

/-- Represents the probability of selecting 2 ties from a boutique with given items. -/
def probability_two_ties (shirts pants ties : ℕ) : ℚ :=
  let total := shirts + pants + ties
  (ties : ℚ) / total * ((ties - 1) : ℚ) / (total - 1)

/-- Theorem stating the probability of selecting 2 ties from Michelle's boutique. -/
theorem michelle_boutique_two_ties_probability : 
  probability_two_ties 4 8 18 = 51 / 145 := by
  sorry

end NUMINAMATH_CALUDE_michelle_boutique_two_ties_probability_l2270_227018


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2270_227064

def i : ℂ := Complex.I

theorem modulus_of_complex_fraction : 
  Complex.abs ((1 + 3 * i) / (1 - 2 * i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2270_227064


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l2270_227063

theorem choose_three_from_eight :
  Nat.choose 8 3 = 56 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l2270_227063


namespace NUMINAMATH_CALUDE_complex_power_sum_l2270_227053

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^97 + z^98 * z^99 * z^100 + z^101 + z^102 + z^103 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2270_227053


namespace NUMINAMATH_CALUDE_divisibility_by_1946_l2270_227043

theorem divisibility_by_1946 (n : ℕ) (hn : n ≤ 1945) :
  ∃ k : ℤ, 1492^n - 1770^n - 1863^n + 2141^n = 1946 * k := by
  sorry


end NUMINAMATH_CALUDE_divisibility_by_1946_l2270_227043


namespace NUMINAMATH_CALUDE_problem_solution_l2270_227069

/-- Represents the box of electronic products -/
structure Box where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

/-- The probability of drawing a first-class product only on the third draw without replacement -/
def prob_first_class_third_draw (b : Box) : ℚ :=
  (b.second_class : ℚ) / b.total *
  ((b.second_class - 1) : ℚ) / (b.total - 1) *
  (b.first_class : ℚ) / (b.total - 2)

/-- The expected number of first-class products in n draws with replacement -/
def expected_first_class (b : Box) (n : ℕ) : ℚ :=
  (n : ℚ) * (b.first_class : ℚ) / b.total

/-- The box described in the problem -/
def problem_box : Box := { total := 5, first_class := 3, second_class := 2 }

theorem problem_solution :
  prob_first_class_third_draw problem_box = 1 / 10 ∧
  expected_first_class problem_box 10 = 6 := by
  sorry

#eval prob_first_class_third_draw problem_box
#eval expected_first_class problem_box 10

end NUMINAMATH_CALUDE_problem_solution_l2270_227069


namespace NUMINAMATH_CALUDE_circus_ticket_ratio_l2270_227067

theorem circus_ticket_ratio : 
  ∀ (num_kids num_adults : ℕ) 
    (total_cost kid_ticket_cost : ℚ),
  num_kids = 6 →
  num_adults = 2 →
  total_cost = 50 →
  kid_ticket_cost = 5 →
  (kid_ticket_cost / ((total_cost - num_kids * kid_ticket_cost) / num_adults)) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_ratio_l2270_227067


namespace NUMINAMATH_CALUDE_arc_length_for_given_circle_l2270_227025

theorem arc_length_for_given_circle (r : ℝ) (θ : ℝ) (arc_length : ℝ) : 
  r = 2 → θ = π / 7 → arc_length = r * θ → arc_length = 2 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_given_circle_l2270_227025


namespace NUMINAMATH_CALUDE_quadratic_shift_l2270_227062

/-- Represents a quadratic function of the form y = -(x+a)^2 + b -/
def QuadraticFunction (a b : ℝ) := λ x : ℝ => -(x + a)^2 + b

/-- Represents a horizontal shift of a function -/
def HorizontalShift (f : ℝ → ℝ) (shift : ℝ) := λ x : ℝ => f (x - shift)

/-- Theorem: Shifting the graph of y = -(x+2)^2 + 1 by 1 unit to the right 
    results in the function y = -(x+1)^2 + 1 -/
theorem quadratic_shift :
  HorizontalShift (QuadraticFunction 2 1) 1 = QuadraticFunction 1 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l2270_227062


namespace NUMINAMATH_CALUDE_min_value_expression_l2270_227001

theorem min_value_expression (a b c k m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hk : k > 0) (hm : m > 0) (hn : n > 0) : 
  (k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a ≥ 6 * k ∧
  ((k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a = 6 * k ↔ 
   k = m ∧ m = n ∧ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2270_227001


namespace NUMINAMATH_CALUDE_math_team_probability_l2270_227014

theorem math_team_probability : 
  let team_sizes : List Nat := [6, 8, 9]
  let num_teams : Nat := 3
  let num_cocaptains : Nat := 3
  let prob_select_team : Rat := 1 / num_teams
  let prob_select_cocaptains (n : Nat) : Rat := 6 / (n * (n - 1) * (n - 2))
  (prob_select_team * (team_sizes.map prob_select_cocaptains).sum : Rat) = 1 / 70 := by
  sorry

end NUMINAMATH_CALUDE_math_team_probability_l2270_227014


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_60_kmph_l2270_227007

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph

/-- The speed of the train is 60 kmph given the specified conditions -/
theorem train_speed_is_60_kmph : 
  train_speed 55 3 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_60_kmph_l2270_227007


namespace NUMINAMATH_CALUDE_hockey_games_played_total_games_played_l2270_227099

/-- Calculates the total number of hockey games played in a season -/
theorem hockey_games_played 
  (season_duration : ℕ) 
  (games_per_month : ℕ) 
  (cancelled_games : ℕ) 
  (postponed_games : ℕ) : ℕ :=
  season_duration * games_per_month - cancelled_games

/-- Proves that the total number of hockey games played is 172 -/
theorem total_games_played : 
  hockey_games_played 14 13 10 5 = 172 := by
  sorry

end NUMINAMATH_CALUDE_hockey_games_played_total_games_played_l2270_227099


namespace NUMINAMATH_CALUDE_novel_pages_count_l2270_227026

theorem novel_pages_count (x : ℝ) : 
  let day1_read := x / 6 + 10
  let day1_remaining := x - day1_read
  let day2_read := day1_remaining / 5 + 14
  let day2_remaining := day1_remaining - day2_read
  let day3_read := day2_remaining / 4 + 16
  let day3_remaining := day2_remaining - day3_read
  day3_remaining = 48 → x = 161 := by
sorry

end NUMINAMATH_CALUDE_novel_pages_count_l2270_227026


namespace NUMINAMATH_CALUDE_work_completion_time_l2270_227061

/-- Given a work that can be completed by person a in 15 days and by person b in 30 days,
    prove that a and b together can complete the work in 10 days. -/
theorem work_completion_time (work : ℝ) (a b : ℝ) 
    (ha : a * 15 = work) 
    (hb : b * 30 = work) : 
    (a + b) * 10 = work := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2270_227061


namespace NUMINAMATH_CALUDE_students_per_table_unchanged_l2270_227008

/-- Proves that the number of students per table remains the same when evenly dividing the total number of students across all tables. -/
theorem students_per_table_unchanged 
  (initial_students_per_table : ℝ) 
  (num_tables : ℝ) 
  (h1 : initial_students_per_table = 6.0)
  (h2 : num_tables = 34.0) :
  let total_students := initial_students_per_table * num_tables
  total_students / num_tables = initial_students_per_table := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_unchanged_l2270_227008


namespace NUMINAMATH_CALUDE_right_triangle_one_two_sqrt_three_l2270_227000

theorem right_triangle_one_two_sqrt_three :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 1 ∧ b = Real.sqrt 3 ∧ c = 2 ∧
  a^2 + b^2 = c^2 ∧
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_one_two_sqrt_three_l2270_227000


namespace NUMINAMATH_CALUDE_custom_polynomial_value_l2270_227072

/-- Custom multiplication operation -/
def star_mult (x y : ℕ) : ℕ := (x + 1) * (y + 1)

/-- Custom squaring operation -/
def star_square (x : ℕ) : ℕ := star_mult x x

/-- The main theorem to prove -/
theorem custom_polynomial_value :
  3 * (star_square 2) - 2 * 2 + 1 = 32 := by sorry

end NUMINAMATH_CALUDE_custom_polynomial_value_l2270_227072


namespace NUMINAMATH_CALUDE_raymonds_dimes_proof_l2270_227029

/-- The number of dimes Raymond has left after spending at the arcade -/
def raymonds_remaining_dimes : ℕ :=
  let initial_amount : ℕ := 250 -- $2.50 in cents
  let petes_spent : ℕ := 20 -- 4 nickels * 5 cents
  let total_spent : ℕ := 200
  let raymonds_spent : ℕ := total_spent - petes_spent
  let raymonds_remaining : ℕ := initial_amount - raymonds_spent
  raymonds_remaining / 10 -- divide by 10 cents per dime

theorem raymonds_dimes_proof :
  raymonds_remaining_dimes = 7 := by
  sorry

end NUMINAMATH_CALUDE_raymonds_dimes_proof_l2270_227029


namespace NUMINAMATH_CALUDE_geometric_relations_l2270_227093

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (contains : Plane → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (plane_through_point_perp_to_line : Point → Line → Plane)
variable (line_perp_to_plane : Point → Plane → Line)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem geometric_relations 
  (α β : Plane) (l : Line) (P : Point) 
  (h1 : perpendicular α β)
  (h2 : intersection α β = l)
  (h3 : contains α P)
  (h4 : ¬ on_line P l) :
  (perpendicular (plane_through_point_perp_to_line P l) β) ∧ 
  (parallel (line_perp_to_plane P α) β) ∧
  (line_in_plane (line_perp_to_plane P β) α) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l2270_227093


namespace NUMINAMATH_CALUDE_teal_color_perception_l2270_227022

theorem teal_color_perception (total : ℕ) (kinda_blue : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  kinda_blue = 90 →
  both = 35 →
  neither = 25 →
  ∃ kinda_green : ℕ, kinda_green = 70 ∧ 
    kinda_green + kinda_blue - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_perception_l2270_227022


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l2270_227050

def total_marbles : ℕ := 6
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 3

theorem one_of_each_color_probability :
  (red_marbles * blue_marbles * green_marbles) / Nat.choose total_marbles selected_marbles = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l2270_227050


namespace NUMINAMATH_CALUDE_trig_identity_l2270_227091

theorem trig_identity (α : ℝ) : 
  2 * (Real.sin (3 * Real.pi - 2 * α))^2 * (Real.cos (5 * Real.pi + 2 * α))^2 = 
  1/4 - 1/4 * Real.sin (5 * Real.pi / 2 - 8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2270_227091


namespace NUMINAMATH_CALUDE_hotel_elevator_cubic_at_15_l2270_227047

/-- The hotel elevator cubic polynomial -/
def hotel_elevator_cubic (P : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, P x = a*x^3 + b*x^2 + c*x + d) ∧
  P 11 = 11 ∧ P 12 = 12 ∧ P 13 = 14 ∧ P 14 = 15

theorem hotel_elevator_cubic_at_15 (P : ℝ → ℝ) (h : hotel_elevator_cubic P) : P 15 = 13 := by
  sorry

end NUMINAMATH_CALUDE_hotel_elevator_cubic_at_15_l2270_227047


namespace NUMINAMATH_CALUDE_min_value_theorem_l2270_227006

theorem min_value_theorem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 20 * b = 2) (h2 : c + 20 * d = 2) :
  (1 / a + 1 / (b * c * d)) ≥ 441 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2270_227006


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l2270_227092

/-- Given a point P(x,6) on the terminal side of angle θ with cos θ = -4/5, prove that x = -8 -/
theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (x, 6) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) →
  Real.cos θ = -4/5 →
  x = -8 :=
by sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l2270_227092


namespace NUMINAMATH_CALUDE_exponential_inequality_l2270_227085

theorem exponential_inequality (x y a b : ℝ) 
  (hxy : x > y ∧ y > 1) 
  (hab : 0 < a ∧ a < b ∧ b < 1) : 
  a^x < b^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2270_227085


namespace NUMINAMATH_CALUDE_smallest_sum_with_real_roots_l2270_227078

theorem smallest_sum_with_real_roots (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + 3*b = 0) → 
  (∃ x : ℝ, x^2 + 3*b*x + a = 0) → 
  a + b ≥ 7 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃ x : ℝ, x^2 + a₀*x + 3*b₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 3*b₀*x + a₀ = 0) ∧ 
    a₀ + b₀ = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_real_roots_l2270_227078


namespace NUMINAMATH_CALUDE_least_number_with_remainder_five_l2270_227038

def is_valid_number (n : ℕ) : Prop :=
  ∃ (S : Set ℕ), 15 ∈ S ∧ ∀ m ∈ S, m > 0 ∧ n % m = 5

theorem least_number_with_remainder_five :
  is_valid_number 125 ∧ ∀ k < 125, ¬(is_valid_number k) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_five_l2270_227038


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2270_227056

/-- Given a quadratic equation x^2 - 9x + 18 = 0, if its roots represent the base and legs
    of an isosceles triangle, then the perimeter of the triangle is 15. -/
theorem isosceles_triangle_perimeter (x : ℝ) : 
  x^2 - 9*x + 18 = 0 →
  ∃ (base leg : ℝ), 
    (x = base ∨ x = leg) ∧ 
    (base > 0 ∧ leg > 0) ∧
    (2*leg > base) ∧
    (base + 2*leg = 15) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2270_227056


namespace NUMINAMATH_CALUDE_final_time_sum_l2270_227021

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the time after a given duration -/
def timeAfter (initial : Time) (duration : Time) : Time :=
  sorry

/-- Converts a Time to its representation on a 12-hour clock -/
def to12HourClock (t : Time) : Time :=
  sorry

theorem final_time_sum (initial : Time) (duration : Time) : 
  initial.hours = 15 ∧ initial.minutes = 0 ∧ initial.seconds = 0 →
  duration.hours = 158 ∧ duration.minutes = 55 ∧ duration.seconds = 32 →
  let finalTime := to12HourClock (timeAfter initial duration)
  finalTime.hours + finalTime.minutes + finalTime.seconds = 92 :=
sorry

end NUMINAMATH_CALUDE_final_time_sum_l2270_227021


namespace NUMINAMATH_CALUDE_final_statue_count_statue_count_increases_l2270_227019

/-- Represents the number of statues on Grandma Molly's lawn over four years -/
def statue_count : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => 7  -- After year 2: 4 + 7 - 4
| 2 => 9  -- After year 3: 7 + 9 - 7
| 3 => 13 -- After year 4: 9 + 4
| _ => 13 -- Any year after 4

/-- The final number of statues after four years is 13 -/
theorem final_statue_count : statue_count 3 = 13 := by
  sorry

/-- The number of statues increases over the years -/
theorem statue_count_increases (n : ℕ) : n < 3 → statue_count n < statue_count (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_final_statue_count_statue_count_increases_l2270_227019


namespace NUMINAMATH_CALUDE_combined_machine_time_l2270_227032

theorem combined_machine_time (t1 t2 : ℝ) (h1 : t1 = 20) (h2 : t2 = 30) :
  1 / (1 / t1 + 1 / t2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_combined_machine_time_l2270_227032


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_approx_l2270_227052

-- Define the given conditions
def total_runs : ℕ := 134
def boundaries : ℕ := 12
def sixes : ℕ := 2

-- Define the runs per boundary and six
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

-- Calculate runs made by running between wickets
def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

-- Define the percentage of runs made by running
def percentage_runs_by_running : ℚ := (runs_by_running : ℚ) / (total_runs : ℚ) * 100

-- Theorem to prove
theorem percentage_runs_by_running_approx :
  abs (percentage_runs_by_running - 55.22) < 0.01 := by sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_approx_l2270_227052


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2270_227088

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 2, 6]
  Matrix.det A = 24 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2270_227088


namespace NUMINAMATH_CALUDE_total_shingle_area_l2270_227071

/-- Calculate the total square footage of shingles required for a house with a main roof and a porch roof. -/
theorem total_shingle_area (main_roof_base main_roof_height porch_roof_length porch_roof_upper_base porch_roof_lower_base porch_roof_height : ℝ) : 
  main_roof_base = 20.5 →
  main_roof_height = 25 →
  porch_roof_length = 6 →
  porch_roof_upper_base = 2.5 →
  porch_roof_lower_base = 4.5 →
  porch_roof_height = 3 →
  (main_roof_base * main_roof_height + (porch_roof_upper_base + porch_roof_lower_base) * porch_roof_height * 2) = 554.5 := by
  sorry

#check total_shingle_area

end NUMINAMATH_CALUDE_total_shingle_area_l2270_227071
