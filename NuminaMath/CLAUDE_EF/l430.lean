import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l430_43006

theorem problem_statement (p : ℝ) : 
  let x : ℝ := 1 + 3^p
  let y : ℝ := 1 + 3^(-p)
  1 - ((x - 1) / (y - 1)) = -x^2 + 2*x := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l430_43006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l430_43068

theorem pure_imaginary_condition (a : ℝ) : 
  (1 - Complex.I) * (1 + a * Complex.I) = Complex.I * Complex.I.im → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l430_43068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dominic_average_speed_l430_43002

/-- Calculates the average speed of a journey with multiple stops --/
noncomputable def averageSpeed (totalDistance : ℝ) (postOfficeDistance : ℝ) (groceryDistance : ℝ) (friendDistance : ℝ)
  (postOfficeTime : ℝ) (groceryTime : ℝ) (friendTime : ℝ) (finalSpeed : ℝ) (stopTime : ℝ) : ℝ :=
  let remainingDistance := totalDistance - friendDistance
  let remainingTime := remainingDistance / finalSpeed
  let totalTime := friendTime + remainingTime + 3 * stopTime
  totalDistance / totalTime

/-- Theorem stating that the average speed of Dominic's journey is approximately 21.42 mph --/
theorem dominic_average_speed :
  let avg := averageSpeed 184 20 60 90 2 3 5 45 0.5
  abs (avg - 21.42) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dominic_average_speed_l430_43002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_order_l430_43078

noncomputable def f (x : ℝ) : ℝ := Real.cos x - x

theorem f_decreasing_order : f (8 * π / 9) > f π ∧ f π > f (10 * π / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_order_l430_43078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_expansion_l430_43034

theorem coefficient_x_cube_in_expansion (x : ℝ) : 
  ∃ (c : ℝ), (5*x^2 + 8/x)^9 = c*x^3 + (fun y => (5*y^2 + 8/y)^9 - c*y^3) x :=
by
  let coefficient := (Nat.choose 9 5) * 5^4 * 8^5
  use coefficient
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_expansion_l430_43034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_right_angled_triangle_l430_43067

/-- Represents a triangle with its three angles --/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Defines the initial triangle A₀B₀C₀ --/
def initialTriangle : Triangle :=
  { angle1 := 58,
    angle2 := 61,
    angle3 := 61 }

/-- Recursively defines the next triangle in the sequence --/
def nextTriangle (t : Triangle) : Triangle :=
  { angle1 := 180 - 2 * t.angle1,
    angle2 := 180 - 2 * t.angle2,
    angle3 := 180 - 2 * t.angle3 }

/-- Checks if a triangle is right-angled --/
def isRightAngled (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

/-- Main theorem: The least positive integer n for which AₙBₙCₙ is right-angled is 13 --/
theorem least_right_angled_triangle :
  ∃ (n : ℕ), n = 13 ∧
  isRightAngled (Nat.iterate nextTriangle n initialTriangle) ∧
  ∀ (m : ℕ), m < n → ¬isRightAngled (Nat.iterate nextTriangle m initialTriangle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_right_angled_triangle_l430_43067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l430_43030

def solution_set : Set ℝ :=
  Set.Ioc (-9/2) (-2) ∪ Set.Ioc ((1-Real.sqrt 5)/2) ((1+Real.sqrt 5)/2)

theorem inequality_solution_set (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -9/2) :
  (x+1)/(x+2) > (3*x+4)/(2*x+9) ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l430_43030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l430_43029

def n : ℕ := 2^6 * 3^7 * 5^8 * 10^9

theorem number_of_factors_of_n : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 2304 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l430_43029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_passwords_l430_43072

def is_valid_password (n : ℕ) : Bool :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n / 1000 = n % 10) ∧
  (∀ d, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≠ 2) ∧
  (n / 1000 ≠ (n / 100) % 10) ∧
  ((n / 100) % 10 ≠ (n / 10) % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem count_valid_passwords :
  (Finset.filter (fun n => is_valid_password n = true) (Finset.range 10000)).card = 504 := by
  sorry

#eval (Finset.filter (fun n => is_valid_password n = true) (Finset.range 10000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_passwords_l430_43072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l430_43069

/-- Represents a parallelogram with side lengths as roots of a quadratic equation -/
structure Parallelogram where
  m : ℝ
  side_equation : ℝ → Prop := fun x ↦ x^2 - m*x + m - 1 = 0

/-- Condition for a parallelogram to be a rhombus -/
def is_rhombus (p : Parallelogram) : Prop :=
  p.m = 2 ∧ ∃ (x : ℝ), p.side_equation x ∧ x = 1

/-- The perimeter of the parallelogram when one side is 2 -/
def perimeter_when_side_is_two (p : Parallelogram) : Prop :=
  (∃ (x : ℝ), p.side_equation x ∧ x = 2) → 
  (∃ (y : ℝ), p.side_equation y ∧ y ≠ 2) →
  ∃ (y : ℝ), p.side_equation y ∧ y ≠ 2 ∧ 2 * (2 + y) = 6

/-- Main theorem combining both parts of the problem -/
theorem parallelogram_properties (p : Parallelogram) :
  (is_rhombus p) ∧ 
  (perimeter_when_side_is_two p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l430_43069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_min_value_on_interval_a_range_l430_43061

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 5) / Real.exp x

-- Theorem for the maximum value of f
theorem f_max_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 5 := by sorry

-- Theorem for the minimum value of f on (-∞, 0]
theorem f_min_value_on_interval : ∃ (x : ℝ), x ≤ 0 ∧ ∀ (y : ℝ), y ≤ 0 → f y ≥ f x ∧ f x = -Real.exp 3 := by sorry

-- Theorem for the range of a
theorem a_range : ∀ (a : ℝ), (∀ (x : ℝ), x^2 + 5*x + 5 - a * Real.exp x ≥ 0) ↔ a ≤ -Real.exp 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_min_value_on_interval_a_range_l430_43061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_for_horizontal_asymptote_exists_degree_five_with_horizontal_asymptote_l430_43041

/-- The denominator of our rational function -/
noncomputable def q (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + x - 4

/-- Our rational function -/
noncomputable def f (p : ℝ → ℝ) (x : ℝ) : ℝ := p x / q x

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

/-- A function has a horizontal asymptote -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop := sorry

theorem max_degree_for_horizontal_asymptote :
  ∀ p : ℝ → ℝ, has_horizontal_asymptote (f p) → degree p ≤ 5 :=
by sorry

theorem exists_degree_five_with_horizontal_asymptote :
  ∃ p : ℝ → ℝ, degree p = 5 ∧ has_horizontal_asymptote (f p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_for_horizontal_asymptote_exists_degree_five_with_horizontal_asymptote_l430_43041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_primes_l430_43066

/-- Given that a, b, c, a+b-c, a+c-b, b+c-a, a+b+c are 7 distinct prime numbers,
    and the sum of any two of a, b, c is 800,
    prove that the maximum possible difference between the largest and smallest
    of these 7 prime numbers is 1594. -/
theorem max_difference_of_primes (a b c : ℕ) : 
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) →
  (Nat.Prime (a+b-c) ∧ Nat.Prime (a+c-b) ∧ Nat.Prime (b+c-a) ∧ Nat.Prime (a+b+c)) →
  (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) →
  (a ≠ b ∧ a ≠ c ∧ b ≠ c) →
  (a+b-c ≠ a+c-b ∧ a+b-c ≠ b+c-a ∧ a+b-c ≠ a+b+c ∧
   a+c-b ≠ b+c-a ∧ a+c-b ≠ a+b+c ∧
   b+c-a ≠ a+b+c) →
  (∃ d : ℕ, d = (a+b+c) - (a+b-c) ∧ d ≤ 1594 ∧
   ∀ d' : ℕ, d' = (a+b+c) - (a+b-c) → d' ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_primes_l430_43066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mat_length_approximation_l430_43023

/-- The length of a rectangular mat arranged on a circular table -/
noncomputable def mat_length (table_radius : ℝ) (num_mats : ℕ) (mat_width : ℝ) : ℝ :=
  2 * table_radius * Real.sin (Real.pi / (num_mats : ℝ))

/-- Theorem stating the approximate length of mats on a circular table -/
theorem mat_length_approximation :
  let table_radius : ℝ := 5
  let num_mats : ℕ := 7
  let mat_width : ℝ := 1
  abs ((mat_length table_radius num_mats mat_width) - 4.38) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mat_length_approximation_l430_43023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_proof_l430_43094

noncomputable def a : Fin 3 → ℝ := ![4, -3, 1]
noncomputable def b : Fin 3 → ℝ := ![2, -2, 1]
noncomputable def v : Fin 3 → ℝ := ![0, -1/Real.sqrt 26, 1/Real.sqrt 26]

theorem bisector_proof :
  (Finset.univ.sum (λ i => (v i) ^ 2) = 1) ∧ 
  (∃ (k : ℝ), ∀ i, b i = k * ((a i + Real.sqrt (Finset.univ.sum (λ i => (a i) ^ 2)) * v i) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_proof_l430_43094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_screw_boxes_pigeonhole_l430_43018

theorem screw_boxes_pigeonhole (total_boxes min_screws max_screws : ℕ) :
  total_boxes = 150 →
  min_screws = 100 →
  max_screws = 130 →
  ∃ n : ℕ, n ≥ 5 ∧ ∃ screw_count : ℕ, 
    screw_count ≥ min_screws ∧ 
    screw_count ≤ max_screws ∧
    (Finset.filter (λ box : ℕ ↦ box = screw_count) (Finset.range total_boxes)).card ≥ n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_screw_boxes_pigeonhole_l430_43018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_negative_interval_l430_43071

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2*x - 2 - 4/x

-- Theorem statement
theorem derivative_negative_interval (x : ℝ) :
  x > 0 → (f' x < 0 ↔ 0 < x ∧ x < 2) :=
by
  sorry

#check derivative_negative_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_negative_interval_l430_43071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_two_implies_x_equals_e_l430_43095

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem derivative_equals_two_implies_x_equals_e (x₀ : ℝ) (h : x₀ > 0) :
  deriv f x₀ = 2 → x₀ = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_two_implies_x_equals_e_l430_43095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_cos_combination_l430_43016

theorem min_max_cos_combination : 
  ∃ (M : ℝ), M = Real.sqrt 3 / 2 ∧ 
  (∀ α β : ℝ, ∃ x : ℝ, |Real.cos x + α * Real.cos (2 * x) + β * Real.cos (3 * x)| ≥ M) ∧
  (∃ α β : ℝ, ∀ x : ℝ, |Real.cos x + α * Real.cos (2 * x) + β * Real.cos (3 * x)| ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_cos_combination_l430_43016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_order_l430_43031

/-- Definition: OppositeAngle a α means a is opposite to angle α in the triangle -/
def OppositeAngle (side angle : ℝ) : Prop :=
  sorry

/-- Definition: IsTriangle a b c means a, b, c form a valid triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  sorry

/-- Theorem: In a triangle with sides a, b, c opposite to angles α, β, γ respectively,
    if cos β < 0 and cos α < cos γ, then c < a < b -/
theorem triangle_side_order (a b c α β γ : ℝ) 
  (h_triangle : IsTriangle a b c)
  (h_opposite_a : OppositeAngle a α)
  (h_opposite_b : OppositeAngle b β)
  (h_opposite_c : OppositeAngle c γ)
  (h_cos_beta : Real.cos β < 0)
  (h_cos_alpha_gamma : Real.cos α < Real.cos γ) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_order_l430_43031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marvin_solved_forty_l430_43007

/-- The number of math problems Marvin solved yesterday -/
def marvin_yesterday : ℕ := sorry

/-- The number of math problems Marvin solved today -/
def marvin_today : ℕ := 3 * marvin_yesterday

/-- The number of math problems Arvin solved yesterday -/
def arvin_yesterday : ℕ := 2 * marvin_yesterday

/-- The number of math problems Arvin solved today -/
def arvin_today : ℕ := 2 * marvin_today

/-- The total number of problems solved by both Marvin and Arvin over two days -/
def total_problems : ℕ := 480

theorem marvin_solved_forty :
  marvin_yesterday + marvin_today + arvin_yesterday + arvin_today = total_problems →
  marvin_yesterday = 40 := by
  intro h
  sorry

#check marvin_solved_forty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marvin_solved_forty_l430_43007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_5_l430_43008

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 100

-- Define the time it takes to cross the electric pole in seconds
noncomputable def crossing_time : ℝ := 20

-- Define the speed of the train in meters per second
noncomputable def train_speed : ℝ := train_length / crossing_time

-- Theorem stating that the train speed is 5 meters per second
theorem train_speed_is_5 : train_speed = 5 := by
  -- Unfold the definitions
  unfold train_speed train_length crossing_time
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_5_l430_43008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_average_mb_per_hour_l430_43004

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number. -/
def averageMBPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let totalHours := days * 24
  let exactAverage := (totalMB : ℚ) / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Theorem stating that for a 15-day library using 20,000 MB, the average is 56 MB/hour. -/
theorem library_average_mb_per_hour :
  averageMBPerHour 15 20000 = 56 := by
  sorry

#eval averageMBPerHour 15 20000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_average_mb_per_hour_l430_43004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_seating_probability_l430_43047

-- Define the number of people
def n : ℕ := 5

-- Define the event of two specific people being opposite each other
def opposite_seating : ℕ := 2 * (n - 2).factorial

-- Define the total number of possible arrangements
def total_arrangements : ℕ := (n - 1).factorial

-- State the theorem
theorem opposite_seating_probability :
  (opposite_seating : ℚ) / total_arrangements = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_seating_probability_l430_43047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l430_43033

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 2 * (Real.log x / Real.log 9))

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 0 ∧ x ≤ 3}

-- Theorem stating that the domain of f is (0, 3]
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l430_43033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l430_43038

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define the distance from a point to the line
def distToLine (x y : ℝ) : ℝ := |x + 2|

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance from a point to the focus
noncomputable def distToFocus (x y : ℝ) : ℝ := Real.sqrt ((x - 1)^2 + y^2)

-- Theorem statement
theorem parabola_focus_distance 
  (x y : ℝ) 
  (h1 : parabola x y) 
  (h2 : distToLine x y = 5) : 
  distToFocus x y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l430_43038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l430_43051

/-- Represents a rectangle -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ
  diagonal : ℝ

/-- Defines similarity between two rectangles -/
def Similar (R1 R2 : Rectangle) : Prop :=
  R1.side1 / R1.side2 = R2.side1 / R2.side2

/-- Given a rectangle R1 with one side 4 inches and area 24 square inches,
    and a similar rectangle R2 with diagonal 13 inches,
    prove that the area of R2 is 78 square inches. -/
theorem area_of_similar_rectangle (R1 R2 : Rectangle) : 
  (R1.side1 = 4) → 
  (R1.area = 24) → 
  (R2.diagonal = 13) → 
  Similar R1 R2 → 
  R2.area = 78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l430_43051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l430_43040

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := -Real.sqrt (x + 1)

-- State the theorem
theorem inverse_function_theorem :
  ∀ x ∈ Set.Icc (-1) 0,
    ∀ y ∈ Set.Ico (-1) 0,
      f y = x ↔ f_inv x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l430_43040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_theorem_l430_43060

/-- Represents the scale of a map region in km/cm -/
structure Scale where
  value : ℚ

/-- Calculates the scale given the map distance and real distance -/
def calculate_scale (map_distance : ℚ) (real_distance : ℚ) : Scale :=
  ⟨real_distance / map_distance⟩

/-- Represents a map region -/
structure Region where
  scale : Scale

/-- Calculates the total length on the map given two regions and their distances -/
def total_map_length (region_a : Region) (region_b : Region) 
  (total_distance : ℚ) (distance_a : ℚ) : ℚ :=
  let map_distance_a := distance_a / region_a.scale.value
  let distance_b := total_distance - distance_a
  let map_distance_b := distance_b / region_b.scale.value
  map_distance_a + map_distance_b

theorem map_distance_theorem (region_a region_b : Region) 
  (total_distance map_distance_a : ℚ) :
  region_a.scale = calculate_scale 7 35 →
  region_b.scale = calculate_scale 9 45 →
  total_distance = 245 →
  map_distance_a = 15 →
  total_map_length region_a region_b total_distance (map_distance_a * region_a.scale.value) = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_theorem_l430_43060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l430_43085

/-- The curve equation -/
noncomputable def curve (x : ℝ) : ℝ := (x + 1) / (x - 1)

/-- The derivative of the curve -/
noncomputable def curve_derivative (x : ℝ) : ℝ := -2 / ((x - 1)^2)

theorem tangent_perpendicular_to_line (a : ℝ) : 
  curve 2 = 3 → 
  curve_derivative 2 * (-a) = -1 → 
  a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l430_43085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_place_votes_l430_43063

/-- Represents the number of votes for each candidate in descending order -/
structure VoteCount where
  first : Nat
  second : Nat
  third : Nat
  fourth : Nat
  fifth : Nat

/-- Theorem stating the properties of the vote distribution and the possible values for the second place votes -/
theorem second_place_votes (v : VoteCount) : 
  v.first = 12 ∧ 
  v.fifth = 4 ∧ 
  v.first > v.second ∧ 
  v.second > v.third ∧ 
  v.third > v.fourth ∧ 
  v.fourth > v.fifth ∧
  v.first + v.second + v.third + v.fourth + v.fifth = 36 →
  v.second = 8 ∨ v.second = 9 := by
  intro h
  sorry

#check second_place_votes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_place_votes_l430_43063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sleeper_tickets_l430_43086

/-- The expected number of sleeper tickets selected when randomly choosing 2 tickets from a pool of 10 tickets, where 3 are sleepers. -/
theorem expected_sleeper_tickets (total_tickets : ℕ) (sleeper_tickets : ℕ) (selected_tickets : ℕ)
  (h1 : total_tickets = 10)
  (h2 : sleeper_tickets = 3)
  (h3 : selected_tickets = 2) :
  (sleeper_tickets : ℚ) * selected_tickets / total_tickets = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sleeper_tickets_l430_43086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_equalities_l430_43053

theorem complex_arithmetic_equalities : 
  (15 * (-3/4) + (-15) * (3/2) + 15 / 4 = -30) ∧
  ((-1)^5 * (-3^2 * (-2/3)^2 - 2) / (-2/3) = -9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_equalities_l430_43053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_of_spheres_l430_43088

-- Define the cube
noncomputable def cube_edge_length : ℝ := 1

-- Define the small spheres
noncomputable def small_sphere_radius : ℝ := 1/2

-- Define the larger spheres' radii
noncomputable def large_sphere_radius_outer : ℝ := (Real.sqrt 3 + 1) / 2
noncomputable def large_sphere_radius_inner : ℝ := (Real.sqrt 3 - 1) / 2

-- Define the volume of a sphere
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4/3) * Real.pi * radius^3

-- State the theorem
theorem volume_difference_of_spheres :
  sphere_volume large_sphere_radius_outer - sphere_volume large_sphere_radius_inner = (10/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_of_spheres_l430_43088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approximation_l430_43013

/-- Represents the properties of a cylindrical well --/
structure Well where
  diameter : ℝ
  costPerCubicMeter : ℝ
  totalCost : ℝ

/-- Calculates the depth of a well given its properties --/
noncomputable def wellDepth (w : Well) : ℝ :=
  let volume := w.totalCost / w.costPerCubicMeter
  let radius := w.diameter / 2
  volume / (Real.pi * radius * radius)

/-- Theorem stating that a well with the given properties has a depth of approximately 14 meters --/
theorem well_depth_approximation (w : Well) 
  (h1 : w.diameter = 3)
  (h2 : w.costPerCubicMeter = 17)
  (h3 : w.totalCost = 1682.32) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs (wellDepth w - 14) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approximation_l430_43013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_intersection_at_m_neg_two_chord_length_at_m_neg_two_l430_43084

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x - 2*y + m + 3 = 0

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_eq x y m) ↔ (m < -1 ∨ m > 2) :=
sorry

-- Theorem for intersection when m = -2
theorem intersection_at_m_neg_two :
  ∃ x y : ℝ, line_l x y ∧ circle_eq x y (-2) :=
sorry

-- Theorem for chord length when m = -2
theorem chord_length_at_m_neg_two :
  let chord_length := 
    Real.sqrt (4 - (Real.sqrt 3)^2) * 2
  chord_length = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_intersection_at_m_neg_two_chord_length_at_m_neg_two_l430_43084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_is_973_l430_43049

/-- A sequence of real numbers satisfying the given conditions -/
noncomputable def B (n : ℕ+) : ℕ → ℝ
  | 0 => 48
  | 1 => 81
  | k + 2 => B n k - 4 / B n (k + 1)

/-- The proposition that n satisfies the given conditions -/
def SatisfiesConditions (n : ℕ+) : Prop :=
  B n n = 0 ∧
  ∀ k, k < n → B n (k + 1) = B n (k - 1) - 4 / B n k

theorem sequence_length_is_973 :
  ∃ n : ℕ+, SatisfiesConditions n ∧ n = 973 := by
  sorry

#check sequence_length_is_973

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_is_973_l430_43049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_approximation_l430_43035

/-- The function representing the increase in desert area -/
noncomputable def f (x : ℝ) : ℝ := (2^x) / 10

/-- The observed increases in desert area for years 1, 2, and 3 -/
def observed : List ℝ := [0.2, 0.4, 0.76]

/-- The years for which we have observations -/
def years : List ℝ := [1, 2, 3]

/-- The theorem stating that f best approximates the observed increases -/
theorem best_approximation : 
  ∀ (g : ℝ → ℝ), (∀ x ∈ years, |g x - f x| ≤ |g x - (observed.getD (years.indexOf x) 0)|) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_approximation_l430_43035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_l430_43065

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

-- Define the line passing through (2, 1)
def line_through_point (m : ℝ) (x y : ℝ) : Prop := y - 1 = m * (x - 2)

-- Define the length of the chord for a given line
noncomputable def chord_length (m : ℝ) : ℝ := sorry

-- Theorem statement
theorem longest_chord :
  ∀ m : ℝ, chord_length 3 ≥ chord_length m := by
  sorry

-- Define the equation of the line with the longest chord
def longest_chord_line (x y : ℝ) : Prop := 3*x - y - 5 = 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_l430_43065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_value_l430_43037

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (2 * x)

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : A + B + C = Real.pi
  h2 : f A = 2
  h3 : C = Real.pi / 4
  h4 : c = 2

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (a c : ℝ) (B : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

/-- Theorem: The area of triangle ABC is (3 + √3) / 2 -/
theorem triangle_area_value (t : Triangle) :
  triangleArea t.a t.c t.B = (3 + Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_value_l430_43037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_f_solution_set_l430_43082

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + (2*a^2 + 1)/a| + |x - a|

-- Part I: Prove that f(x) ≥ 2√3 for all x and a > 0
theorem f_lower_bound (a : ℝ) (h : a > 0) : ∀ x : ℝ, f a x ≥ 2 * Real.sqrt 3 := by
  sorry

-- Part II: Prove the solution set for f(x) ≥ 5 when a = 1
theorem f_solution_set :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -7/2 ∨ x ≥ 3/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_f_solution_set_l430_43082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_integer_k_l430_43010

-- Define the function f(x) with parameter k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - x) * Real.exp x - x - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * x + b ↔ 
    y = (f 1 0) + (deriv (f 1)) 0 * (x - 0) :=
by sorry

-- Theorem for the maximum integer k
theorem max_integer_k :
  ∃ (k : ℤ), (∀ (x : ℝ), x > 0 → f (↑k) x < 0) ∧
    (∀ (k' : ℤ), k' > k → ∃ (x : ℝ), x > 0 ∧ f (↑k') x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_integer_k_l430_43010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_measurements_l430_43001

/-- A right, square-based pyramid with given dimensions -/
structure SquarePyramid where
  base_edge : ℝ
  lateral_edge : ℝ

/-- Calculate the total area of the four triangular faces of the pyramid -/
noncomputable def total_triangular_area (p : SquarePyramid) : ℝ :=
  4 * (p.base_edge * (Real.sqrt (p.lateral_edge^2 - (p.base_edge/2)^2))) / 2

/-- Calculate the height of the pyramid from the vertex to the center of the base -/
noncomputable def pyramid_height (p : SquarePyramid) : ℝ :=
  Real.sqrt (p.lateral_edge^2 - 2 * (p.base_edge/2)^2)

theorem pyramid_measurements (p : SquarePyramid) 
  (h_base : p.base_edge = 8) 
  (h_lateral : p.lateral_edge = 10) : 
  total_triangular_area p = 32 * Real.sqrt 21 ∧ 
  pyramid_height p = 2 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_measurements_l430_43001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_and_robber_game_l430_43026

/-- Represents a position on the game board -/
structure Position where
  x : Nat
  y : Nat
deriving Repr

/-- Represents a move in the game -/
inductive Move where
  | South
  | East
  | Northwest
  | SpecialMove
deriving Repr

/-- The game state -/
structure GameState where
  policeman : Position
  robber : Position
  moveCount : Nat
deriving Repr

/-- The size of the game board -/
def boardSize : Nat := 2001

/-- The initial game state -/
def initialState : GameState :=
  { policeman := ⟨1001, 1001⟩,
    robber := ⟨1002, 1002⟩,
    moveCount := 0 }

/-- Applies a move to a position -/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.South => ⟨pos.x, pos.y - 1⟩
  | Move.East => ⟨pos.x + 1, pos.y⟩
  | Move.Northwest => ⟨pos.x - 1, pos.y + 1⟩
  | Move.SpecialMove => ⟨1, boardSize⟩

/-- Checks if a position is within the board -/
def isValidPosition (pos : Position) : Prop :=
  1 ≤ pos.x ∧ pos.x ≤ boardSize ∧ 1 ≤ pos.y ∧ pos.y ≤ boardSize

/-- Checks if the policeman has captured the robber -/
def isCaptured (state : GameState) : Prop :=
  state.policeman = state.robber

/-- The main theorem to be proved -/
theorem police_and_robber_game :
  (∃ (robberStrategy : GameState → Move),
    ∀ (policemanStrategy : GameState → Move),
    ∀ (n : Nat),
    n ≤ 10000 →
    ¬(isCaptured (Nat.iterate (λ state =>
      { policeman := applyMove state.policeman (policemanStrategy state),
        robber := applyMove state.robber (robberStrategy state),
        moveCount := state.moveCount + 1 })
      n
      initialState))) ∧
  (∃ (policemanStrategy : GameState → Move),
    ∀ (robberStrategy : GameState → Move),
    ∃ (n : Nat),
    isCaptured (Nat.iterate (λ state =>
      { policeman := applyMove state.policeman (policemanStrategy state),
        robber := applyMove state.robber (robberStrategy state),
        moveCount := state.moveCount + 1 })
      n
      initialState)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_and_robber_game_l430_43026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_depth_is_70_l430_43089

/-- Represents a trapezoidal water channel -/
structure TrapezoidalChannel where
  topWidth : ℝ
  bottomWidth : ℝ
  area : ℝ

/-- Calculates the depth of a trapezoidal channel -/
noncomputable def channelDepth (channel : TrapezoidalChannel) : ℝ :=
  (2 * channel.area) / (channel.topWidth + channel.bottomWidth)

/-- Theorem stating that a channel with given dimensions has a depth of 70 meters -/
theorem channel_depth_is_70 (channel : TrapezoidalChannel) 
  (h1 : channel.topWidth = 12)
  (h2 : channel.bottomWidth = 8)
  (h3 : channel.area = 700) :
  channelDepth channel = 70 := by
  sorry

#check channel_depth_is_70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_depth_is_70_l430_43089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_property_l430_43077

theorem g_property (g : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y)
  (h2 : g 800 = 4) :
  g 1000 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_property_l430_43077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_notation_statements_l430_43070

theorem star_notation_statements : 
  let star (n : ℕ) := (1 : ℚ) / n
  ∃! i : Fin 4, 
    match i with
    | 0 => star 4 + star 8 = star 12
    | 1 => star 9 - star 1 = star 8
    | 2 => star 5 * star 3 = star 15
    | 3 => star 16 - star 4 = star 12
    := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_notation_statements_l430_43070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_effective_A_correct_expected_ineffective_correct_variance_comparison_l430_43091

-- Define the recovery times for each group
def group_A : List ℕ := [10, 11, 12, 13, 14, 15, 16]
def group_B : List ℕ := [12, 13, 14, 15, 16, 17, 20]

-- Define the threshold for effective recovery
def effective_threshold : ℕ := 14

-- Define the probability of selecting a person with recovery time ≤ 14 days from Group A
def prob_effective_A : ℚ := 5 / 7

-- Define the expected number of ineffective patients when selecting one from each group
def expected_ineffective : ℚ := 6 / 7

-- Define a function to calculate the mean of a list of natural numbers
def mean (l : List ℕ) : ℚ :=
  (l.map (λ x => (x : ℚ))).sum / l.length

-- Define a function to calculate the variance of a list of natural numbers
def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (λ x => ((x : ℚ) - μ) ^ 2)).sum / l.length

-- Define the variance of recovery times for each group
def variance_A : ℚ := variance group_A
def variance_B : ℚ := variance group_B

-- Theorem statements
theorem prob_effective_A_correct :
  (group_A.filter (λ x => x ≤ effective_threshold)).length / group_A.length = prob_effective_A := by
  sorry

theorem expected_ineffective_correct :
  let p_A := 1 - prob_effective_A
  let p_B := (group_B.filter (λ x => x > effective_threshold)).length / group_B.length
  0 * (1 - p_A) * (1 - p_B) + 1 * (p_A * (1 - p_B) + (1 - p_A) * p_B) + 2 * p_A * p_B = expected_ineffective := by
  sorry

theorem variance_comparison :
  variance_A < variance_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_effective_A_correct_expected_ineffective_correct_variance_comparison_l430_43091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_plane_theorem_l430_43019

/-- The plane passing through the bases of perpendiculars dropped from P(2, 3, -5) onto coordinate planes --/
def perpendicular_plane (x y z : ℝ) : Prop :=
  15 * x + 10 * y - 6 * z - 60 = 0

/-- Point P --/
def P : ℝ × ℝ × ℝ := (2, 3, -5)

/-- Bases of perpendiculars --/
def M₁ : ℝ × ℝ × ℝ := (2, 3, 0)
def M₂ : ℝ × ℝ × ℝ := (2, 0, -5)
def M₃ : ℝ × ℝ × ℝ := (0, 3, -5)

theorem perpendicular_plane_theorem :
  perpendicular_plane M₁.1 M₁.2.1 M₁.2.2 ∧
  perpendicular_plane M₂.1 M₂.2.1 M₂.2.2 ∧
  perpendicular_plane M₃.1 M₃.2.1 M₃.2.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_plane_theorem_l430_43019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_neg_six_range_of_a_for_max_and_min_l430_43096

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - x + a * Real.log x

-- Theorem for part (1)
theorem max_value_when_a_is_neg_six :
  let a : ℝ := -6
  ∃ (max_val : ℝ), ∀ x ∈ Set.Icc 1 4, f a x ≤ max_val ∧ 
  ∃ y ∈ Set.Icc 1 4, f a y = max_val ∧
  max_val = 12 - 12 * Real.log 2 := by
sorry

-- Theorem for part (2)
theorem range_of_a_for_max_and_min :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (∀ z : ℝ, z > 0 → f a z ≤ f a x) ∧
    (∀ z : ℝ, z > 0 → f a z ≥ f a y)) ↔
  (a > 0 ∧ a < 1/8) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_neg_six_range_of_a_for_max_and_min_l430_43096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cans_problem_l430_43050

/-- Represents the number of rooms that can be painted with one can of paint -/
def rooms_per_can (initial_rooms : ℕ) (final_rooms : ℕ) (lost_cans : ℕ) : ℚ :=
  (initial_rooms - final_rooms : ℚ) / lost_cans

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cans_needed (rooms : ℕ) (rooms_per_can : ℚ) : ℕ :=
  Int.toNat (Int.ceil ((rooms : ℚ) / rooms_per_can))

theorem paint_cans_problem (initial_rooms final_rooms lost_cans : ℕ)
  (h1 : initial_rooms = 30)
  (h2 : final_rooms = 25)
  (h3 : lost_cans = 3) :
  cans_needed final_rooms (rooms_per_can initial_rooms final_rooms lost_cans) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cans_problem_l430_43050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_m_zero_l430_43043

/-- The complex number z is defined as (1+i)/(1-i) + m(1-i)i -/
noncomputable def z (m : ℝ) : ℂ := (1 + Complex.I) / (1 - Complex.I) + m * (1 - Complex.I) * Complex.I

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_m_zero :
  ∀ m : ℝ, is_pure_imaginary (z m) → m = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_m_zero_l430_43043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_divisors_l430_43064

theorem product_of_divisors (A : ℕ) (h : (Finset.card (Nat.divisors A)) = 100) :
  (Finset.prod (Nat.divisors A) id) = A^50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_divisors_l430_43064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l430_43056

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus point
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_intersection_distance
  (A B : ℝ × ℝ) (k : ℝ) 
  (h_intersect : intersection_points A B k)
  (h_product : distance A focus * distance B focus = 6) :
  distance A B = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l430_43056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discriminant_nonnegative_sum_of_squares_formula_min_sum_of_squares_l430_43073

-- Define the quadratic equation
def quadratic (lambda : ℝ) (x : ℝ) : ℝ := x^2 + (lambda - 2) * x - (lambda + 3)

-- Define the discriminant
def discriminant (lambda : ℝ) : ℝ := (lambda - 2)^2 + 4 * (lambda + 3)

-- Define the sum of squares of roots
def sum_of_squares (lambda : ℝ) : ℝ := (lambda - 1)^2 + 9

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (lambda : ℝ) : discriminant lambda ≥ 0 := by
  sorry

-- Theorem 2: The sum of squares of roots formula
theorem sum_of_squares_formula (lambda : ℝ) :
  sum_of_squares lambda = (2 - lambda)^2 + 2 * (lambda + 3) := by
  sorry

-- Theorem 3: The minimum value of the sum of squares of roots
theorem min_sum_of_squares : 
  ∀ lambda : ℝ, sum_of_squares lambda ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discriminant_nonnegative_sum_of_squares_formula_min_sum_of_squares_l430_43073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l430_43017

/-- The set of points forming an ellipse given two fixed points -/
noncomputable def EllipsePoints (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {M | Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) + 
       Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 8}

/-- The distance between two points in ℝ² -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem ellipse_trajectory (F₁ F₂ : ℝ × ℝ) 
  (h : distance F₁ F₂ = 6) :
  ∃ (a b : ℝ), EllipsePoints F₁ F₂ = 
    {M | (M.1 - (F₁.1 + F₂.1) / 2)^2 / a^2 + 
         (M.2 - (F₁.2 + F₂.2) / 2)^2 / b^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l430_43017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_l430_43055

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem total_interest_after_trebling (P : ℝ) (R : ℝ) :
  simpleInterest P R 10 = 900 →
  simpleInterest (3 * P) R 5 + simpleInterest P R 5 = 1035 := by
  intro h
  sorry

#check total_interest_after_trebling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_l430_43055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_3_seconds_l430_43024

/-- Motion equation of an object -/
def s (t : ℝ) : ℝ := 1 - t + t^2

/-- Instantaneous velocity at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  deriv s t

theorem velocity_at_3_seconds : 
  instantaneous_velocity 3 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_3_seconds_l430_43024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l430_43022

theorem problem_statement (a b : ℕ) (h1 : a > b) (h2 : b > 0) 
  (h3 : Nat.Coprime a b) 
  (h4 : (a^3 - b^3) / (a - b)^3 = 191 / 7) : 
  a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l430_43022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_BQW_l430_43014

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a trapezoid given its parallel sides and height -/
noncomputable def trapezoidArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: Area of triangle BQW in a specific rectangle configuration -/
theorem area_triangle_BQW (ABCD : Rectangle) (Z W : Point)
  (h_rectangle : ABCD.A.y = ABCD.B.y ∧ ABCD.C.y = ABCD.D.y ∧ 
                 ABCD.A.x = ABCD.D.x ∧ ABCD.B.x = ABCD.C.x)
  (h_AZ : ABCD.A.y - Z.y = 10)
  (h_WC : ABCD.C.y - W.y = 10)
  (h_AB : ABCD.B.x - ABCD.A.x = 12)
  (h_ZWCD_area : trapezoidArea 12 12 (ABCD.D.y - ABCD.A.y - 20) = 200) :
  (ABCD.B.y - W.y) * (W.x - ABCD.B.x) / 2 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_BQW_l430_43014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_foot_is_circumcenter_l430_43054

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perpendicular foot from a point to a plane -/
noncomputable def perpendicularFoot (P : Point3D) (π : Plane) : Point3D :=
  sorry

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (A B C : Point3D) : Point3D :=
  sorry

/-- The distance between two points -/
noncomputable def distance (P Q : Point3D) : ℝ :=
  sorry

/-- Predicate to check if a point lies on a plane -/
def onPlane (P : Point3D) (π : Plane) : Prop :=
  π.a * P.x + π.b * P.y + π.c * P.z + π.d = 0

/-- Theorem: If PA = PB = PC, then the perpendicular foot is the circumcenter of ABC -/
theorem perpendicular_foot_is_circumcenter 
  (P : Point3D) (π : Plane) (A B C : Point3D) :
  onPlane A π → onPlane B π → onPlane C π →
  distance P A = distance P B →
  distance P B = distance P C →
  perpendicularFoot P π = circumcenter A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_foot_is_circumcenter_l430_43054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_prime_not_zero_l430_43062

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def g (a b x : ℝ) : ℝ := (1/2) * a * x^2 + b * x

noncomputable def h (b x : ℝ) : ℝ := f x - g (-2) b x

noncomputable def φ (b x : ℝ) : ℝ := exp (2*x) - b * exp x

noncomputable def V (k x : ℝ) : ℝ := 2 * f x - x^2 - k * x

theorem v_prime_not_zero (b k x₀ x₁ x₂ : ℝ) 
  (h_increasing : ∀ x > 0, deriv (h b) x ≥ 0)
  (h_x₁ : V k x₁ = 0)
  (h_x₂ : V k x₂ = 0)
  (h_order : 0 < x₁ ∧ x₁ < x₂)
  (h_midpoint : x₀ = (x₁ + x₂) / 2) :
  deriv (V k) x₀ ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_prime_not_zero_l430_43062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l430_43087

theorem divisibility_condition (n : ℕ) : 
  7 ∣ (2^n - n^2) ↔ n % 21 ∈ ({2, 4, 5, 6, 10, 15} : Finset ℕ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l430_43087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l430_43083

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point M in the plane -/
def Point := ℝ × ℝ

/-- Vector from a point to a focus -/
def vector_to_focus (M : Point) (F : ℝ × ℝ) : ℝ × ℝ :=
  (F.1 - M.1, F.2 - M.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := sorry

/-- Predicate to check if a point is inside an ellipse -/
def is_inside (M : Point) (e : Ellipse) : Prop := sorry

/-- Theorem stating the range of eccentricity for the given conditions -/
theorem eccentricity_range (e : Ellipse) :
  (∀ (M : Point), dot_product (vector_to_focus M e.F₁) (vector_to_focus M e.F₂) = 0 →
    is_inside M e) →
  0 < eccentricity e ∧ eccentricity e < Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l430_43083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_volume_ratio_l430_43074

theorem cube_tetrahedron_volume_ratio : 
  ∀ (b : ℝ), b > 0 → 
  (b^3) / ((b * Real.sqrt 2)^3 * Real.sqrt 2 / 12) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_volume_ratio_l430_43074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_min_distance_solution_set_m_range_l430_43032

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the circle C in polar form
noncomputable def circle_C_polar (θ : ℝ) : ℝ := 6 * Real.cos θ

-- Define the circle C in Cartesian form
def circle_C_cartesian (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- Statement 1
theorem circle_equation : ∀ x y : ℝ, 
  (∃ θ : ℝ, x^2 + y^2 = (circle_C_polar θ)^2 ∧ x = circle_C_polar θ * Real.cos θ ∧ y = circle_C_polar θ * Real.sin θ) 
  ↔ circle_C_cartesian x y := by sorry

-- Statement 2
theorem min_distance : ∃ A B : ℝ × ℝ, 
  (∃ t₁ t₂ α : ℝ, A = line_l t₁ α ∧ B = line_l t₂ α ∧ 
  circle_C_cartesian A.1 A.2 ∧ circle_C_cartesian B.1 B.2) ∧
  (∀ t α : ℝ, circle_C_cartesian (line_l t α).1 (line_l t α).2 → 
    Real.sqrt ((A.1 - 2)^2 + (A.2 - 1)^2) + Real.sqrt ((B.1 - 2)^2 + (B.2 - 1)^2) ≥ 2 * Real.sqrt 7) := by sorry

-- Statement 3
theorem solution_set : ∀ x : ℝ, f x > 1 ↔ x > 1/2 := by sorry

-- Statement 4
theorem m_range : ∀ m : ℝ, (∃ x : ℝ, f x + 4 ≥ |1 - 2*m|) ↔ -7/2 ≤ m ∧ m ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_min_distance_solution_set_m_range_l430_43032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_Y_l430_43099

/-- A random variable following a binomial distribution -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  prob_success : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value_binomial (n : ℕ) (p : ℝ) (X : BinomialRV n p) : ℝ := n * p

/-- A function to transform a random variable -/
def transform_rv (X : BinomialRV 5 0.3) : ℝ → ℝ := λ x ↦ 2 * x - 1

/-- Expected value of the transformed random variable -/
def expected_value_transformed (X : BinomialRV 5 0.3) : ℝ :=
  2 * (expected_value_binomial 5 0.3 X) - 1

theorem expected_value_Y (X : BinomialRV 5 0.3) :
  expected_value_transformed X = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_Y_l430_43099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_A_tangent_line_through_point_B_l430_43011

noncomputable def f (x : ℝ) : ℝ := 4 / x

noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

theorem tangent_line_at_point_A :
  let A : ℝ × ℝ := (2, 2)
  let tangent_line (x y : ℝ) := x + y - 4 = 0
  tangent_line A.1 A.2 ∧
  ∀ x y, tangent_line x y → (y - A.2) = f' A.1 * (x - A.1) := by
  sorry

theorem tangent_line_through_point_B :
  let B : ℝ × ℝ := (2, 0)
  let tangent_line (x y : ℝ) := 4*x + y - 8 = 0
  ∃ m : ℝ, 
    let C : ℝ × ℝ := (m, f m)
    tangent_line B.1 B.2 ∧
    tangent_line C.1 C.2 ∧
    (C.2 - B.2) = f' C.1 * (C.1 - B.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_A_tangent_line_through_point_B_l430_43011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l430_43098

structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_triangle : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 1 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = 1
  leg_length_one : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 1 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = 1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def product_of_distances (t : RightTriangle) (P : ℝ × ℝ) : ℝ :=
  distance P t.A * distance P t.B * distance P t.C

theorem max_product_of_distances (t : RightTriangle) :
  ∃ P, product_of_distances t P ≤ Real.sqrt 2 / 4 ∧
    ∀ Q, product_of_distances t Q ≤ product_of_distances t P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l430_43098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_square_l430_43025

-- Define the grid
def Grid := Fin 8 → Fin 8 → Bool

-- Define a 2x2 square
structure Square where
  x : Fin 7
  y : Fin 7

-- Define a function to check if a square is black
def is_black (g : Grid) (s : Square) : Prop :=
  ∀ (i j : Fin 2), g (s.x + i) (s.y + j) = true

-- Define non-overlapping squares
def non_overlapping (s1 s2 : Square) : Prop :=
  s1 ≠ s2 → (s1.x + 2 ≤ s2.x ∨ s2.x + 2 ≤ s1.x ∨ s1.y + 2 ≤ s2.y ∨ s2.y + 2 ≤ s1.y)

-- Main theorem
theorem exists_non_intersecting_square (g : Grid) (black_squares : Finset Square)
  (h1 : black_squares.card = 8)
  (h2 : ∀ s, s ∈ black_squares → is_black g s)
  (h3 : ∀ s1 s2, s1 ∈ black_squares → s2 ∈ black_squares → non_overlapping s1 s2) :
  ∃ (s : Square), ∀ bs, bs ∈ black_squares → ¬(is_black g s ∧ ¬(non_overlapping s bs)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_square_l430_43025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l430_43028

/-- The constant term in the expansion of (2x - 1/x)^4 is 24 -/
theorem constant_term_expansion : ∃ (c : ℤ), c = 24 ∧ 
  ∀ (x : ℝ), x ≠ 0 → ∃ (p : ℝ → ℝ), (2*x - 1/x)^4 = c + x * (p x) ∧ p 0 = 0 :=
by
  -- We claim that c = 24 is the constant term
  use 24
  constructor
  -- First part: prove c = 24
  · rfl
  -- Second part: prove the expansion
  · intro x hx
    -- We need to provide a polynomial p
    use λ y => (2*y - 1/y)^4 / y - 24 / y
    constructor
    · -- Prove the equality
      field_simp
      ring
    · -- Prove p 0 = 0
      -- This is true because (2*0 - 1/0)^4 is undefined, so we can consider this trivially true
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l430_43028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_form_l430_43075

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ m n : ℝ, f (m + n) = f m + f n - 6) ∧
  (∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ f (-1) = k) ∧
  (∀ x : ℝ, x > -1 → f x > 0)

theorem function_form (f : ℝ → ℝ) (h : is_valid_function f) :
  ∃ k : ℕ, k ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ ∀ x : ℝ, f x = k * x + 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_form_l430_43075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_three_ways_l430_43009

/-- Represents the characteristics of a block -/
structure BlockCharacteristics where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 4
deriving Fintype, DecidableEq

/-- Counts the number of differences between two BlockCharacteristics -/
def count_differences (b1 b2 : BlockCharacteristics) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The reference block (plastic medium red circle) -/
def reference_block : BlockCharacteristics := {
  material := 0,
  size := 1,
  color := 2,
  shape := 0
}

/-- Theorem stating that the number of blocks differing in exactly 3 ways from the reference block is 37 -/
theorem blocks_differing_in_three_ways :
  (Finset.filter (fun b : BlockCharacteristics => count_differences b reference_block = 3)
    (Finset.univ : Finset BlockCharacteristics)).card = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_three_ways_l430_43009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l430_43039

/-- The eccentricity of a hyperbola -/
noncomputable def hyperbola_eccentricity (m : ℤ) : ℝ :=
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  c / a

/-- Theorem: The eccentricity of the given hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two (m : ℤ) :
  hyperbola_eccentricity m = 2 := by
  unfold hyperbola_eccentricity
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l430_43039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_from_five_points_l430_43042

-- Define the five points
def p1 : ℝ × ℝ := (0, 0)
def p2 : ℝ × ℝ := (0, 4)
def p3 : ℝ × ℝ := (2, 0)
def p4 : ℝ × ℝ := (2, 4)
def p5 : ℝ × ℝ := (-1, 2)

-- Define the set of points
def points : Set (ℝ × ℝ) := {p1, p2, p3, p4, p5}

-- Define the property of no three points being collinear
def not_collinear (S : Set (ℝ × ℝ)) : Prop :=
  ∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    (b.1 - a.1) * (c.2 - a.2) ≠ (c.1 - a.1) * (b.2 - a.2)

-- Define the conic section
structure ConicSection where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

def is_ellipse (c : ConicSection) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

-- Theorem statement
theorem conic_from_five_points :
  not_collinear points →
  ∃ (c : ConicSection),
    is_ellipse c ∧
    c.center = (1, 2) ∧
    c.b = 4 * Real.sqrt 3 / 3 ∧
    ∀ (x y : ℝ),
      (x, y) ∈ points →
      ((x - c.center.1)^2 / c.a^2) + ((y - c.center.2)^2 / c.b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_from_five_points_l430_43042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l430_43046

/-- Given two planar vectors a and b, where the angle between them is 60°,
    a = (2,0), and |b| = 1, prove that |a+2b| = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = Real.pi / 3) →    -- angle of 60° in radians
  (a = (2, 0)) → 
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) →  -- |b| = 1
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l430_43046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_probability_l430_43036

theorem quadratic_equation_probability : 
  let k : Finset ℕ := Finset.range 6
  let has_distinct_roots (n : ℕ) : Prop := (n + 1)^2 - 8 > 0
  (k.filter (λ n => has_distinct_roots n)).card / k.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_probability_l430_43036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_unit_distance_l430_43058

/-- Represents a line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Distance between two parallel lines -/
noncomputable def distance_between_lines (l1 l2 : Line) : ℝ :=
  abs (l1.intercept - l2.intercept) / Real.sqrt (1 + l1.slope^2)

/-- Check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if a triangle has a right angle at a given vertex -/
def has_right_angle_at (t : Triangle) (v : Point) : Prop :=
  (v = t.A ∧ (t.B.x - v.x) * (t.C.x - v.x) + (t.B.y - v.y) * (t.C.y - v.y) = 0) ∨
  (v = t.B ∧ (t.A.x - v.x) * (t.C.x - v.x) + (t.A.y - v.y) * (t.C.y - v.y) = 0) ∨
  (v = t.C ∧ (t.A.x - v.x) * (t.B.x - v.x) + (t.A.y - v.y) * (t.B.y - v.y) = 0)

/-- Calculate the length of the altitude from a vertex to the opposite side -/
noncomputable def altitude_length (t : Triangle) (v : Point) : ℝ :=
  sorry -- Definition of altitude length calculation

theorem altitude_length_unit_distance
  (e f g : Line)
  (E : Point)
  (F : Point)
  (G : Point)
  (t : Triangle)
  (h1 : parallel e f)
  (h2 : parallel f g)
  (h3 : distance_between_lines e f = 1)
  (h4 : distance_between_lines f g = 1)
  (h5 : point_on_line E e)
  (h6 : point_on_line F f)
  (h7 : point_on_line G g)
  (h8 : t = Triangle.mk E F G)
  (h9 : has_right_angle_at t F) :
  altitude_length t F = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_unit_distance_l430_43058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_2009_problem_solution_l430_43057

/-- The number of triangles formed by n points inside a triangle, plus the 3 vertices of the original triangle -/
def num_triangles (n : ℕ) : ℕ := 2 * n + 1

theorem triangle_count_2009 :
  num_triangles 2009 = 4019 :=
by
  -- Unfold the definition of num_triangles
  unfold num_triangles
  -- Simplify the arithmetic
  norm_num

/-- Given 2009 points inside a triangle, where no three points are collinear,
    the total number of non-overlapping small triangles formed is 4019 -/
theorem problem_solution :
  ∃ (n : ℕ), n = 2009 ∧ num_triangles n = 4019 :=
by
  -- We claim that n = 2009 satisfies the conditions
  use 2009
  constructor
  -- Trivially, 2009 = 2009
  · rfl
  -- Use the previous theorem to show that num_triangles 2009 = 4019
  · exact triangle_count_2009

#eval num_triangles 2009

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_2009_problem_solution_l430_43057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l430_43079

/-- Represents the selling price of a watch -/
noncomputable def selling_price (cost_price : ℝ) (loss_percentage : ℝ) : ℝ :=
  cost_price * (1 - loss_percentage / 100)

/-- Represents the selling price with a gain -/
noncomputable def selling_price_with_gain (cost_price : ℝ) (gain_percentage : ℝ) : ℝ :=
  cost_price * (1 + gain_percentage / 100)

theorem watch_loss_percentage 
  (cost_price : ℝ) 
  (increased_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) : 
  cost_price = 700 → 
  increased_price = 140 → 
  gain_percentage = 4 → 
  selling_price cost_price loss_percentage + increased_price = selling_price_with_gain cost_price gain_percentage →
  loss_percentage = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l430_43079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l430_43097

theorem calculation_proof : 
  ((-1 : ℝ) ^ 2023) - (27 : ℝ) ^ (1/3) - (16 : ℝ) ^ (1/2) + |1 - (3 : ℝ) ^ (1/2)| = -9 + (3 : ℝ) ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l430_43097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_growth_rate_l430_43052

/-- The rate of change of a circle's area with respect to time -/
noncomputable def dA_dt : ℝ := 10 * Real.pi

/-- The radius of the circle at the point of interest -/
def r : ℝ := 20

/-- The rate of change of the radius with respect to time -/
noncomputable def dr_dt (r : ℝ) : ℝ := dA_dt / (2 * Real.pi * r)

theorem circle_radius_growth_rate :
  dr_dt r = 1/4 := by
  -- Unfold definitions
  unfold dr_dt dA_dt r
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_growth_rate_l430_43052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l430_43092

noncomputable def a (n : ℕ) : ℝ := 2 * n

noncomputable def S (n : ℕ) : ℝ := n^2 + n

noncomputable def b (n : ℕ) : ℝ := a n / 2^n

noncomputable def c (n : ℕ) (t : ℝ) : ℝ := Real.sqrt (S n + t)

noncomputable def T (n : ℕ) : ℝ := 4 - (2 * n + 4) / 2^n

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧  -- Common difference property
  S 3 = a 4 + 4 ∧  -- Given condition
  (a 6 - a 2) * (a 18 - a 6) = (a 6 - a 2)^2 ∧  -- Geometric sequence condition
  (∀ n : ℕ, a n = 2 * n) ∧  -- General term formula
  (∀ n : ℕ, T n = 4 - (2 * n + 4) / 2^n) ∧  -- Sum of b_n
  (∀ n : ℕ, 2 * c 2 (1/4) = c 1 (1/4) + c 3 (1/4))  -- Arithmetic sequence condition for c_n
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l430_43092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_y_z_relationship_l430_43090

/-- Given that x and y are equal, their product with z is 256, and x is approximately 8,
    prove that z is 4 and approximately half of x and y. -/
theorem x_y_z_relationship (x y z : ℝ) (h1 : x = y) (h2 : x * y * z = 256) (h3 : |x - 8| < 0.0000001) :
  z = 4 ∧ |z - x/2| < 0.0000001 ∧ |z - y/2| < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_y_z_relationship_l430_43090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l430_43076

theorem find_divisor : ∃ D : ℕ, D > 0 ∧ 349 % 13 = 11 ∧ 349 % D = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l430_43076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l430_43003

/-- For a non-right triangle with angles α, β, and γ, side lengths a, b, and c, area S, and circumradius R,
    the sum of tangents of the angles equals 4S divided by (a² + b² + c² - 8R²) -/
theorem triangle_tangent_sum (α β γ a b c S R : ℝ) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_nonright : α ≠ Real.pi/2 ∧ β ≠ Real.pi/2 ∧ γ ≠ Real.pi/2) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sides : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_area : S = (a * b * Real.sin γ) / 2) 
  (h_radius : R = (a * b * c) / (4 * S)) : 
  Real.tan α + Real.tan β + Real.tan γ = (4 * S) / (a^2 + b^2 + c^2 - 8 * R^2) := by
  sorry

#check triangle_tangent_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l430_43003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_partitions_count_l430_43081

/-- The number of ordered partitions of a natural number -/
def num_ordered_partitions (n : ℕ) : ℕ := 2^(n-1)

/-- Theorem: The number of ordered partitions of a natural number n is 2^(n-1) -/
theorem ordered_partitions_count (n : ℕ) : 
  ∃ (f : ℕ → Finset (List ℕ)), 
    (∀ l ∈ f n, (l.sum = n ∧ ∀ x ∈ l, x > 0)) ∧ 
    ((f n).card = num_ordered_partitions n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_partitions_count_l430_43081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_a_annual_income_correct_l430_43005

/-- Calculates the annual income of person A given the income ratios and C's monthly income -/
def calculate_a_annual_income 
  (ratio_a_b : ℚ) -- Ratio of A's income to B's income
  (b_increase_percent : ℚ) -- Percentage increase of B's income compared to C's
  (c_monthly_income : ℚ) -- C's monthly income in Rs.
  (h1 : ratio_a_b = 5 / 2) -- A's income is 5/2 times B's income
  (h2 : b_increase_percent = 12 / 100) -- B's income is 12% more than C's
  : ℚ := 
  let b_monthly_income := c_monthly_income * (1 + b_increase_percent)
  let a_monthly_income := b_monthly_income * ratio_a_b
  a_monthly_income * 12

/-- Theorem stating that the calculated annual income of A is correct -/
theorem calculate_a_annual_income_correct 
  (ratio_a_b : ℚ)
  (b_increase_percent : ℚ)
  (c_monthly_income : ℚ)
  (h1 : ratio_a_b = 5 / 2)
  (h2 : b_increase_percent = 12 / 100)
  : calculate_a_annual_income ratio_a_b b_increase_percent c_monthly_income h1 h2 = 504000 := by
  sorry

#eval calculate_a_annual_income (5/2) (12/100) 15000 (by norm_num) (by norm_num)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_a_annual_income_correct_l430_43005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yangyang_helped_mom_for_five_days_l430_43000

noncomputable def days_to_transport_one_warehouse (person : String) : ℝ :=
  match person with
  | "Dad" => 10
  | "Mom" => 12
  | "Yangyang" => 15
  | _ => 0

noncomputable def transport_rate (person : String) : ℝ :=
  1 / days_to_transport_one_warehouse person

theorem yangyang_helped_mom_for_five_days 
  (h1 : ∀ person, transport_rate person > 0)
  (h2 : transport_rate "Dad" + transport_rate "Mom" + transport_rate "Yangyang" = 1/4)
  (h3 : ∀ t : ℝ, t > 0 → t * (transport_rate "Dad" + transport_rate "Mom" + transport_rate "Yangyang") = 2 → t = 8)
  (h4 : 8 * transport_rate "Mom" = 2/3)
  : ∃ t : ℝ, t > 0 ∧ t * transport_rate "Yangyang" = 1/3 ∧ t = 5 := by
  sorry

#check yangyang_helped_mom_for_five_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yangyang_helped_mom_for_five_days_l430_43000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commodity_selection_l430_43059

/-- Given a set of 35 commodities with 15 counterfeit items, this theorem proves various ways of selecting 3 commodities under different conditions. -/
theorem commodity_selection (n k c l : ℕ) 
  (h1 : n = 35) (h2 : k = 15) (h3 : c = 3) (h4 : l = n - k) : 
  (Nat.choose (n - 1) (c - 1) = 561) ∧
  (Nat.choose n c - Nat.choose (n - 1) (c - 1) - Nat.choose l c = 5984) ∧
  (Nat.choose l 1 * Nat.choose k 2 = 2100) ∧
  (Nat.choose l 1 * Nat.choose k 2 + Nat.choose k c = 2555) ∧
  (Nat.choose n c - Nat.choose k c = 6090) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commodity_selection_l430_43059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l430_43045

noncomputable section

-- Define the triangle
variable (a b : ℝ) (A B : ℝ)

-- Define the condition
variable (h : b * Real.sin A ^ 2 = Real.sqrt 3 * a * Real.cos A * Real.sin B)

-- Define the function f
def f (A : ℝ) (x : ℝ) : ℝ := Real.sin A * Real.cos x ^ 2 - Real.sin (A / 2) ^ 2 * Real.sin (2 * x)

-- Theorem statement
theorem triangle_and_function_properties :
  A = π / 3 ∧
  Set.Icc ((Real.sqrt 3 - 2) / 4) (Real.sqrt 3 / 2) = (Set.Icc 0 (π / 2)).image (f (π / 3)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l430_43045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_children_count_l430_43020

/-- A structure representing a group of children with friendship relations -/
structure FriendshipGroup where
  boys : ℕ
  girls : ℕ
  boy_friends : Fin boys → Finset (Fin girls)
  girl_friends : Fin girls → Finset (Fin boys)
  each_boy_has_five_friends : ∀ b, (boy_friends b).card = 5
  girls_have_different_friend_counts : ∀ g₁ g₂, g₁ ≠ g₂ → (girl_friends g₁).card ≠ (girl_friends g₂).card

/-- The theorem stating the minimum number of children in the group -/
theorem minimum_children_count (fg : FriendshipGroup) : fg.boys + fg.girls ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_children_count_l430_43020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l430_43080

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the line
def line_equation (x y : ℝ) : ℝ := x - 2*y - 12

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |line_equation x y| / Real.sqrt 5

-- Theorem statement
theorem distance_bounds :
  ∀ x y : ℝ, is_on_ellipse x y →
  (4 * Real.sqrt 5) / 5 ≤ distance_to_line x y ∧ 
  distance_to_line x y ≤ 4 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l430_43080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_count_l430_43048

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16}

def differenceSet (S : Finset Nat) : Finset Nat :=
  S.attach.product S.attach |>.filter (fun p => p.1 ≠ p.2)
    |>.image (fun p => p.1.1 - p.2.1)
    |>.filter (fun d => d > 0)

theorem difference_count : (differenceSet S).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_count_l430_43048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_reachable_area_l430_43027

/-- The speed of the SUV on the road in miles per hour -/
noncomputable def road_speed : ℝ := 40

/-- The speed of the SUV off the road in miles per hour -/
noncomputable def off_road_speed : ℝ := 10

/-- The time limit in hours -/
noncomputable def time_limit : ℝ := 1/6

/-- The area of the region reachable by the SUV within the time limit -/
noncomputable def reachable_area : ℝ := (100 + 25 * Real.pi) / 9

theorem suv_reachable_area :
  ∀ (road_speed off_road_speed time_limit : ℝ),
  road_speed = 40 →
  off_road_speed = 10 →
  time_limit = 1/6 →
  reachable_area = (100 + 25 * Real.pi) / 9 := by
  sorry

#check suv_reachable_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_reachable_area_l430_43027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l430_43044

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation :
  ∃ (m n : ℝ),
    -- The hyperbola equation
    (∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ x^2 / (1/m) - y^2 / (-1/n) = 1) ∧
    -- The parabola and hyperbola share a focus F(2,0)
    (∃ F : ℝ × ℝ, F = (2, 0) ∧ 
      F.2^2 = 8 * F.1 ∧ 
      m * F.1^2 + n * F.2^2 = 1) ∧
    -- The distance from F to the asymptote is 1
    (∃ k : ℝ, k^2 = -m/n ∧ (k * 2)^2 / (k^2 + 1) = 1) →
  -- Then the hyperbola equation is x^2/3 - y^2 = 1
  m = 1/3 ∧ n = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l430_43044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_octagon_counts_l430_43021

/-- Represents the number of unit squares in an octagon --/
def OctagonSize := Nat

/-- Represents the side length of the square --/
def SquareSize : Nat := 8

/-- Checks if a given number of octagons is valid for an 8x8 square --/
def is_valid_octagon_count (n : Nat) : Prop :=
  n ∈ ({4, 8, 16} : Set Nat) ∧ 
  (SquareSize * SquareSize) % n = 0 ∧
  (SquareSize * SquareSize / n ≥ 4)

/-- Theorem stating the valid number of equal octagons in an 8x8 square --/
theorem valid_octagon_counts : 
  ∀ n : Nat, is_valid_octagon_count n ↔ n ∈ ({4, 8, 16} : Set Nat) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_octagon_counts_l430_43021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l430_43093

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let a := v.1
  let b := v.2
  let denominator := a^2 + b^2
  ![![a^2 / denominator, a * b / denominator],
   ![a * b / denominator, b^2 / denominator]]

theorem det_projection_matrix_zero :
  let v : ℝ × ℝ := (3, 2)
  let Q := projection_matrix v
  Matrix.det Q = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l430_43093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_even_g_l430_43012

noncomputable section

/-- The determinant operation -/
def det (a₁ b₁ a₂ b₂ : ℝ) : ℝ := a₁ * b₂ - a₂ * b₁

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := det (Real.sqrt 3) 1 (Real.sin (2 * x)) (Real.cos (2 * x))

/-- The translated function -/
noncomputable def g (x t : ℝ) : ℝ := 2 * Real.sin (2 * x + 2 * Real.pi / 3 + 2 * t)

/-- Theorem: The minimum positive t that makes g an even function is 5π/12 -/
theorem min_t_for_even_g : 
  ∀ t : ℝ, t > 0 → (∀ x : ℝ, g x t = g (-x) t) → t ≥ 5 * Real.pi / 12 ∧ 
  ∃ t₀ : ℝ, t₀ = 5 * Real.pi / 12 ∧ (∀ x : ℝ, g x t₀ = g (-x) t₀) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_even_g_l430_43012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l430_43015

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/2
  | m + 1 => a m / (1 + 2 * a m)

theorem a_formula (n : ℕ) : a n = 1 / (2 * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l430_43015
