import Mathlib

namespace NUMINAMATH_CALUDE_prime_divisibility_l1693_169389

theorem prime_divisibility (p a b : ℤ) : 
  Prime p → 
  ∃ k : ℤ, p = 4 * k + 3 → 
  p ∣ (a^2 + b^2) → 
  p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1693_169389


namespace NUMINAMATH_CALUDE_barkley_buried_bones_l1693_169314

/-- Calculates the number of bones Barkley has buried given the conditions -/
def bones_buried (bones_per_month : ℕ) (months_passed : ℕ) (available_bones : ℕ) : ℕ :=
  bones_per_month * months_passed - available_bones

/-- Theorem stating that Barkley has buried 42 bones under the given conditions -/
theorem barkley_buried_bones : 
  bones_buried 10 5 8 = 42 := by
  sorry

end NUMINAMATH_CALUDE_barkley_buried_bones_l1693_169314


namespace NUMINAMATH_CALUDE_no_periodic_sum_with_periods_one_and_pi_l1693_169301

/-- A function is periodic if it takes at least two different values and there exists a positive real number p such that f(x + p) = f(x) for all x. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- p is a period of f if f(x + p) = f(x) for all x. -/
def IsPeriodOf (p : ℝ) (f : ℝ → ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem no_periodic_sum_with_periods_one_and_pi :
  ¬ ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ IsPeriodic h ∧
    IsPeriodOf 1 g ∧ IsPeriodOf Real.pi h ∧
    IsPeriodic (g + h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sum_with_periods_one_and_pi_l1693_169301


namespace NUMINAMATH_CALUDE_distinct_shapes_count_is_31_l1693_169346

/-- Represents a convex-shaped paper made of four 1×1 squares -/
structure ConvexPaper :=
  (squares : Fin 4 → (Fin 1 × Fin 1))

/-- Represents a 5×6 grid paper -/
structure GridPaper :=
  (grid : Fin 5 → Fin 6 → Bool)

/-- Represents a placement of the convex paper on the grid paper -/
structure Placement :=
  (position : Fin 5 × Fin 6)
  (orientation : Fin 4)

/-- Checks if a placement is valid (all squares of convex paper overlap with grid squares) -/
def isValidPlacement (cp : ConvexPaper) (gp : GridPaper) (p : Placement) : Prop :=
  sorry

/-- Checks if two placements are rotationally equivalent -/
def areRotationallyEquivalent (p1 p2 : Placement) : Prop :=
  sorry

/-- The number of distinct shapes that can be formed -/
def distinctShapesCount (cp : ConvexPaper) (gp : GridPaper) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct shapes is 31 -/
theorem distinct_shapes_count_is_31 (cp : ConvexPaper) (gp : GridPaper) :
  distinctShapesCount cp gp = 31 :=
  sorry

end NUMINAMATH_CALUDE_distinct_shapes_count_is_31_l1693_169346


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1693_169353

/-- Given that i² = -1, prove that (3 + 2i) / (4 - 5i) = 2/41 + (23/41)i -/
theorem complex_fraction_simplification :
  (Complex.I : ℂ)^2 = -1 →
  (3 + 2 * Complex.I) / (4 - 5 * Complex.I) = (2 : ℂ) / 41 + (23 : ℂ) / 41 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1693_169353


namespace NUMINAMATH_CALUDE_no_solution_condition_l1693_169326

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, (x + m) / (4 - x^2) + x / (x - 2) ≠ 1) ↔ (m = 2 ∨ m = 6) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1693_169326


namespace NUMINAMATH_CALUDE_processing_400_parts_l1693_169386

/-- Linear regression function for processing time -/
def processingTime (x : ℝ) : ℝ := 0.2 * x + 3

/-- Theorem: Processing 400 parts takes 83 hours -/
theorem processing_400_parts : processingTime 400 = 83 := by
  sorry

end NUMINAMATH_CALUDE_processing_400_parts_l1693_169386


namespace NUMINAMATH_CALUDE_tan_37_5_deg_identity_l1693_169385

theorem tan_37_5_deg_identity : 
  (Real.tan (37.5 * π / 180)) / (1 - (Real.tan (37.5 * π / 180))^2) = 1 + (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_37_5_deg_identity_l1693_169385


namespace NUMINAMATH_CALUDE_sum_equals_two_thirds_l1693_169355

theorem sum_equals_two_thirds : 
  let original_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
  let remaining_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/18
  remaining_sum = 2/3 := by sorry

end NUMINAMATH_CALUDE_sum_equals_two_thirds_l1693_169355


namespace NUMINAMATH_CALUDE_commute_days_l1693_169382

theorem commute_days (x : ℕ) 
  (h1 : x > 0)
  (h2 : 2 * x = 9 + 8 + 15) : 
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_commute_days_l1693_169382


namespace NUMINAMATH_CALUDE_grandpa_water_distribution_l1693_169395

/-- The number of water bottles Grandpa has -/
def num_bottles : ℕ := 12

/-- The volume of each water bottle in liters -/
def bottle_volume : ℚ := 3

/-- The volume of water to be distributed to each student in liters -/
def student_share : ℚ := 3/4

/-- The number of students Grandpa can share water with -/
def num_students : ℕ := 48

theorem grandpa_water_distribution :
  (↑num_bottles * bottle_volume) / student_share = num_students := by
  sorry

end NUMINAMATH_CALUDE_grandpa_water_distribution_l1693_169395


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1693_169311

/-- 
Given a right triangle with hypotenuse h, legs a and b, and an inscribed circle with radius r,
prove that the ratio of the area of the inscribed circle to the area of the triangle is πr / (h + r).
-/
theorem inscribed_circle_area_ratio (h a b r : ℝ) (h_positive : h > 0) (r_positive : r > 0) 
  (right_triangle : a^2 + b^2 = h^2) (inscribed_circle : r = (a + b - h) / 2) : 
  (π * r^2) / ((1/2) * a * b) = π * r / (h + r) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1693_169311


namespace NUMINAMATH_CALUDE_OPRQ_shapes_l1693_169333

-- Define the points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral OPRQ
structure Quadrilateral where
  O : Point2D
  P : Point2D
  R : Point2D
  Q : Point2D

-- Define the conditions for parallelogram, rectangle, and rhombus
def is_parallelogram (quad : Quadrilateral) : Prop :=
  ∃ k l : ℝ, k ≠ 0 ∧ l ≠ 0 ∧
  quad.R.x = k * quad.P.x + l * quad.Q.x ∧
  quad.R.y = k * quad.P.y + l * quad.Q.y

def is_rectangle (quad : Quadrilateral) : Prop :=
  is_parallelogram quad ∧
  quad.P.x * quad.Q.x + quad.P.y * quad.Q.y = 0

def is_rhombus (quad : Quadrilateral) : Prop :=
  is_parallelogram quad ∧
  quad.P.x^2 + quad.P.y^2 = quad.Q.x^2 + quad.Q.y^2

-- Main theorem
theorem OPRQ_shapes (P Q : Point2D) (h : P ≠ Q) :
  ∃ (R : Point2D) (quad : Quadrilateral),
    quad.O = ⟨0, 0⟩ ∧ quad.P = P ∧ quad.Q = Q ∧ quad.R = R ∧
    (is_parallelogram quad ∨ is_rectangle quad ∨ is_rhombus quad) :=
  sorry

end NUMINAMATH_CALUDE_OPRQ_shapes_l1693_169333


namespace NUMINAMATH_CALUDE_complex_power_difference_l1693_169302

theorem complex_power_difference (x : ℂ) (h : x - 1/x = 2*I) : x^8 - 1/x^8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1693_169302


namespace NUMINAMATH_CALUDE_floor_of_pi_l1693_169378

theorem floor_of_pi : ⌊Real.pi⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_pi_l1693_169378


namespace NUMINAMATH_CALUDE_quadratic_roots_and_specific_case_l1693_169368

/-- The quadratic equation x^2 - (m-1)x = 3 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (m-1)*x = 3

theorem quadratic_roots_and_specific_case :
  (∀ m : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y) ∧
  (∃ m : ℝ, quadratic_equation m 2 ∧ quadratic_equation m (-3/2) ∧ m = 5/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_specific_case_l1693_169368


namespace NUMINAMATH_CALUDE_union_of_sets_l1693_169351

def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {a + 2, 5}

theorem union_of_sets (a : ℕ) (h : A ∩ B a = {3}) : A ∪ B a = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1693_169351


namespace NUMINAMATH_CALUDE_bangles_per_box_l1693_169337

def total_pairs : ℕ := 240
def num_boxes : ℕ := 20

theorem bangles_per_box :
  (total_pairs * 2) / num_boxes = 24 := by
  sorry

end NUMINAMATH_CALUDE_bangles_per_box_l1693_169337


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1693_169312

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_x2y2 (a b : ℕ) : ℕ := 
  binomial_coefficient a 2 * binomial_coefficient b 2

theorem expansion_coefficient : 
  coefficient_x2y2 3 4 = 18 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1693_169312


namespace NUMINAMATH_CALUDE_difference_of_squares_75_35_l1693_169397

theorem difference_of_squares_75_35 : 75^2 - 35^2 = 4400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_35_l1693_169397


namespace NUMINAMATH_CALUDE_loaves_delivered_correct_evening_delivery_l1693_169324

/-- Given the initial number of loaves, the number of loaves sold, and the final number of loaves,
    calculate the number of loaves delivered. -/
theorem loaves_delivered (initial : ℕ) (sold : ℕ) (final : ℕ) :
  final - (initial - sold) = final - initial + sold :=
by sorry

/-- The number of loaves delivered in the evening -/
def evening_delivery : ℕ := 2215 - (2355 - 629)

theorem correct_evening_delivery : evening_delivery = 489 :=
by sorry

end NUMINAMATH_CALUDE_loaves_delivered_correct_evening_delivery_l1693_169324


namespace NUMINAMATH_CALUDE_two_amoebas_fill_time_l1693_169380

/-- The time (in minutes) it takes for amoebas to fill a bottle -/
def fill_time (initial_count : ℕ) : ℕ → ℕ
| 60 => 1  -- One amoeba fills the bottle in 60 minutes
| t => initial_count * 2^(t / 3)  -- Amoeba count at time t

/-- Theorem stating that two amoebas fill the bottle in 57 minutes -/
theorem two_amoebas_fill_time : fill_time 2 57 = fill_time 1 60 := by
  sorry

end NUMINAMATH_CALUDE_two_amoebas_fill_time_l1693_169380


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1693_169310

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 5/4 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), (y = (4/3) * x ∨ y = -(4/3) * x) ↔ (y = (a/b) * x ∨ y = -(a/b) * x)) →
  (∀ (x y : ℝ), y^2/a^2 - x^2/b^2 = 1 → x = 0 → ∃ (c : ℝ), y^2 = a^2 + c^2) →
  e = (a^2 + b^2).sqrt / a := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1693_169310


namespace NUMINAMATH_CALUDE_joans_savings_l1693_169349

/-- The number of quarters Joan has saved --/
def num_quarters : ℕ := 6

/-- The value of one quarter in cents --/
def cents_per_quarter : ℕ := 25

/-- Theorem: The total value of Joan's quarters in cents --/
theorem joans_savings : num_quarters * cents_per_quarter = 150 := by
  sorry

end NUMINAMATH_CALUDE_joans_savings_l1693_169349


namespace NUMINAMATH_CALUDE_stick_cutting_theorem_l1693_169343

/-- Represents a marked stick with cuts -/
structure MarkedStick :=
  (length : ℕ)
  (left_interval : ℕ)
  (right_interval : ℕ)

/-- Counts the number of segments of a given length in a marked stick -/
def count_segments (stick : MarkedStick) (segment_length : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that a 240 cm stick marked as described yields 12 pieces of 3 cm -/
theorem stick_cutting_theorem :
  let stick : MarkedStick := ⟨240, 7, 6⟩
  count_segments stick 3 = 12 := by sorry

end NUMINAMATH_CALUDE_stick_cutting_theorem_l1693_169343


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l1693_169317

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  curve a (-1) = a + 2 →
  curve_derivative a (-1) = 8 →
  a = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l1693_169317


namespace NUMINAMATH_CALUDE_max_profundity_eq_fib_l1693_169360

/-- The dog dictionary consists of words made from letters A and U -/
inductive DogLetter
| A
| U

/-- A word in the dog dictionary is a list of DogLetters -/
def DogWord := List DogLetter

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- The profundity of a word is the number of its subwords -/
def profundity (w : DogWord) : ℕ := sorry

/-- The maximum profundity for words of length n -/
def max_profundity (n : ℕ) : ℕ := sorry

/-- The main theorem: maximum profundity equals F_{n+3} - 3 -/
theorem max_profundity_eq_fib (n : ℕ) :
  max_profundity n = fib (n + 3) - 3 := by sorry

end NUMINAMATH_CALUDE_max_profundity_eq_fib_l1693_169360


namespace NUMINAMATH_CALUDE_four_possible_values_for_D_l1693_169329

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem four_possible_values_for_D :
  ∀ (A B C D : ℕ),
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    A < 10 → B < 10 → C < 10 → D < 10 →
    is_even A →
    is_odd B →
    A + B = D →
    C + D = D →
    (∃ (S : Finset ℕ), S.card = 4 ∧ ∀ d, d ∈ S ↔ (∃ a b, is_even a ∧ is_odd b ∧ a + b = d ∧ a < 10 ∧ b < 10 ∧ a ≠ b)) :=
by sorry

end NUMINAMATH_CALUDE_four_possible_values_for_D_l1693_169329


namespace NUMINAMATH_CALUDE_root_product_equals_27_l1693_169396

theorem root_product_equals_27 : 
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l1693_169396


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l1693_169394

def hulk_jump (n : ℕ) : ℕ := 2^(n-1)

theorem hulk_jump_exceeds_1000 :
  ∃ n : ℕ, n > 0 ∧ hulk_jump n > 1000 ∧ ∀ m : ℕ, m > 0 ∧ m < n → hulk_jump m ≤ 1000 :=
by
  use 11
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l1693_169394


namespace NUMINAMATH_CALUDE_defective_pens_l1693_169373

theorem defective_pens (total : ℕ) (prob : ℚ) (defective : ℕ) : 
  total = 8 →
  prob = 15/28 →
  (total - defective : ℚ) / total * ((total - defective - 1) : ℚ) / (total - 1) = prob →
  defective = 2 :=
sorry

end NUMINAMATH_CALUDE_defective_pens_l1693_169373


namespace NUMINAMATH_CALUDE_volume_sin_squared_rotation_l1693_169339

theorem volume_sin_squared_rotation (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x ^ 2) :
  ∫ x in (0)..(Real.pi / 2), π * (f x)^2 = (3 * Real.pi^2) / 16 := by
  sorry

end NUMINAMATH_CALUDE_volume_sin_squared_rotation_l1693_169339


namespace NUMINAMATH_CALUDE_hamburger_combinations_l1693_169348

-- Define the number of patty options
def patty_options : Nat := 4

-- Define the number of condiments
def num_condiments : Nat := 9

-- Theorem statement
theorem hamburger_combinations :
  (patty_options * 2^num_condiments) = 2048 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l1693_169348


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1693_169307

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) > a n) →  -- increasing sequence
  (a 1 + a 2 + a 3 = 12) →  -- sum of first three terms
  ((a 3)^2 = a 2 * (a 4 + 1)) →  -- geometric sequence condition
  (∃ d : ℝ, ∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∃ d : ℝ, (∀ n, a (n + 1) - a n = d) ∧ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1693_169307


namespace NUMINAMATH_CALUDE_ratio_of_squares_l1693_169322

theorem ratio_of_squares : (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_l1693_169322


namespace NUMINAMATH_CALUDE_group_b_sample_size_l1693_169325

/-- Represents the number of cities to be selected from a group in stratified sampling -/
def stratifiedSample (totalCities : ℕ) (groupCities : ℕ) (totalSample : ℕ) : ℕ :=
  (groupCities * totalSample) / totalCities

/-- Theorem stating the correct number of cities to be selected from Group B -/
theorem group_b_sample_size :
  let totalCities : ℕ := 36
  let groupBCities : ℕ := 12
  let totalSample : ℕ := 12
  stratifiedSample totalCities groupBCities totalSample = 4 := by
  sorry

end NUMINAMATH_CALUDE_group_b_sample_size_l1693_169325


namespace NUMINAMATH_CALUDE_derivative_zero_at_origin_l1693_169361

theorem derivative_zero_at_origin (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f (-x) = f x) : 
  deriv f 0 = 0 := by
sorry

end NUMINAMATH_CALUDE_derivative_zero_at_origin_l1693_169361


namespace NUMINAMATH_CALUDE_positive_integer_solution_for_exponential_equation_l1693_169313

theorem positive_integer_solution_for_exponential_equation :
  ∀ (n a b c : ℕ), 
    n > 1 → a > 0 → b > 0 → c > 0 →
    n^a + n^b = n^c →
    (n = 2 ∧ b = a ∧ c = a + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_for_exponential_equation_l1693_169313


namespace NUMINAMATH_CALUDE_pencil_sharpening_mean_l1693_169365

def pencil_sharpening_data : List ℕ := [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

theorem pencil_sharpening_mean :
  (pencil_sharpening_data.sum : ℚ) / pencil_sharpening_data.length = 543 / 30 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_mean_l1693_169365


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l1693_169305

theorem max_value_x_plus_y (x y : ℝ) (h : x - Real.sqrt (x + 1) = Real.sqrt (y + 3) - y) :
  ∃ (M : ℝ), M = 4 ∧ x + y ≤ M ∧ ∀ (N : ℝ), (x + y ≤ N) → (M ≤ N) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l1693_169305


namespace NUMINAMATH_CALUDE_system_is_linear_l1693_169300

-- Define a linear equation in two variables
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

-- Define the system of equations
def equation1 (x y : ℝ) : ℝ := x + y - 2
def equation2 (x y : ℝ) : ℝ := x - 2 * y

-- Theorem statement
theorem system_is_linear :
  is_linear_equation equation1 ∧ is_linear_equation equation2 :=
sorry

end NUMINAMATH_CALUDE_system_is_linear_l1693_169300


namespace NUMINAMATH_CALUDE_twentieth_stage_toothpicks_l1693_169398

/-- Number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 3 + 3 * (n - 1)

/-- The 20th stage of the toothpick pattern has 60 toothpicks -/
theorem twentieth_stage_toothpicks : toothpicks 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_stage_toothpicks_l1693_169398


namespace NUMINAMATH_CALUDE_herd_size_l1693_169399

theorem herd_size (herd : ℕ) : 
  (herd / 3 + herd / 6 + herd / 8 + herd / 24 + 15 = herd) → 
  herd = 45 := by
  sorry

end NUMINAMATH_CALUDE_herd_size_l1693_169399


namespace NUMINAMATH_CALUDE_enclosed_area_is_nine_l1693_169323

-- Define the line function
def line (x : ℝ) : ℝ := 2 * x

-- Define the curve function
def curve (x : ℝ) : ℝ := 4 - 2 * x^2

-- Define the intersection points
def x1 : ℝ := -2
def x2 : ℝ := 1

-- Theorem statement
theorem enclosed_area_is_nine :
  ∫ x in x1..x2, (curve x - line x) = 9 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_nine_l1693_169323


namespace NUMINAMATH_CALUDE_common_factor_of_2a2_and_4ab_l1693_169383

theorem common_factor_of_2a2_and_4ab :
  ∀ (a b : ℤ), ∃ (k₁ k₂ : ℤ), 2 * a^2 = (2 * a) * k₁ ∧ 4 * a * b = (2 * a) * k₂ ∧
  (∀ (d : ℤ), (∃ (m₁ m₂ : ℤ), 2 * a^2 = d * m₁ ∧ 4 * a * b = d * m₂) → d ∣ (2 * a)) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_2a2_and_4ab_l1693_169383


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_right_triangle_l1693_169374

theorem quadratic_roots_imply_right_triangle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (hroots : ∃ x : ℝ, x^2 - (a + b + c)*x + (a*b + b*c + c*a) = 0 ∧ 
    ∀ y : ℝ, y^2 - (a + b + c)*y + (a*b + b*c + c*a) = 0 → y = x) :
  ∃ p q r : ℝ, p^4 = a ∧ q^4 = b ∧ r^4 = c ∧ p^2 = q^2 + r^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_right_triangle_l1693_169374


namespace NUMINAMATH_CALUDE_largest_b_for_divisibility_by_three_l1693_169327

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_b_for_divisibility_by_three :
  ∀ b : ℕ, b ≤ 9 →
    (is_divisible_by_three (500000 + 100000 * b + 6584) ↔ is_divisible_by_three (b + 28)) ∧
    (∀ k : ℕ, k ≤ 9 ∧ k > b → ¬is_divisible_by_three (500000 + 100000 * k + 6584)) →
    b = 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_for_divisibility_by_three_l1693_169327


namespace NUMINAMATH_CALUDE_prime_iff_binomial_divisible_l1693_169328

theorem prime_iff_binomial_divisible (n : ℕ) (h : n > 1) : 
  Nat.Prime n ↔ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → n ∣ Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_binomial_divisible_l1693_169328


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1693_169331

theorem trigonometric_identity (θ : ℝ) (h : Real.sin (3 * π / 2 + θ) = 1 / 4) :
  (Real.cos (π + θ)) / (Real.cos θ * (Real.cos (π + θ) - 1)) +
  (Real.cos (θ - 2 * π)) / (Real.cos (θ + 2 * π) * Real.cos (θ + π) + Real.cos (-θ)) = 32 / 15 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1693_169331


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1693_169308

theorem quadratic_equal_roots (k : ℝ) (A : ℝ) : 
  (∃ x : ℝ, A * x^2 + 6 * k * x + 2 = 0 ∧ 
   ∀ y : ℝ, A * y^2 + 6 * k * y + 2 = 0 → y = x) ∧ 
  k = 0.4444444444444444 → 
  A = 9 * k^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1693_169308


namespace NUMINAMATH_CALUDE_special_number_value_l1693_169372

/-- A number with specified digits in certain decimal places -/
def SpecialNumber : ℝ :=
  60 + 0.06

/-- Proof that the SpecialNumber is equal to 60.06 -/
theorem special_number_value : SpecialNumber = 60.06 := by
  sorry

#check special_number_value

end NUMINAMATH_CALUDE_special_number_value_l1693_169372


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l1693_169356

theorem cricket_bat_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 234) :
  ∃ (cost_price_A : ℝ), cost_price_A = 156 ∧
    price_C = (1 + profit_B_to_C) * ((1 + profit_A_to_B) * cost_price_A) := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l1693_169356


namespace NUMINAMATH_CALUDE_weight_sum_determination_l1693_169335

/-- Given the weights of four people in pairs, prove that the sum of the weights of two specific people can be determined. -/
theorem weight_sum_determination (a b c d : ℝ) 
  (h1 : a + b = 280)
  (h2 : b + c = 230)
  (h3 : c + d = 250)
  (h4 : a + d = 300) :
  a + c = 250 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_determination_l1693_169335


namespace NUMINAMATH_CALUDE_garden_area_calculation_l1693_169366

/-- The total area of Mancino's and Marquita's gardens -/
def total_garden_area (mancino_garden_length mancino_garden_width mancino_garden_count
                       marquita_garden_length marquita_garden_width marquita_garden_count : ℕ) : ℕ :=
  (mancino_garden_length * mancino_garden_width * mancino_garden_count) +
  (marquita_garden_length * marquita_garden_width * marquita_garden_count)

/-- Theorem stating that the total area of Mancino's and Marquita's gardens is 304 square feet -/
theorem garden_area_calculation :
  total_garden_area 16 5 3 8 4 2 = 304 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_calculation_l1693_169366


namespace NUMINAMATH_CALUDE_distinct_polynomials_differ_l1693_169336

-- Define the set X inductively
inductive X : (ℝ → ℝ) → Prop
  | base : X (λ x => x)
  | mul {r} : X r → X (λ x => x * r x)
  | add {r} : X r → X (λ x => x + (1 - x) * r x)

-- Define the theorem
theorem distinct_polynomials_differ (r s : ℝ → ℝ) (hr : X r) (hs : X s) (h_distinct : r ≠ s) :
  ∀ x, 0 < x → x < 1 → r x ≠ s x :=
sorry

end NUMINAMATH_CALUDE_distinct_polynomials_differ_l1693_169336


namespace NUMINAMATH_CALUDE_remainder_property_l1693_169369

/-- A polynomial of the form Dx^6 + Ex^4 + Fx^2 + 7 -/
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^6 + E * x^4 + F * x^2 + 7

/-- The remainder theorem -/
def remainder_theorem (p : ℝ → ℝ) (a : ℝ) : ℝ := p a

theorem remainder_property (D E F : ℝ) :
  remainder_theorem (q D E F) 2 = 17 →
  remainder_theorem (q D E F) (-2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l1693_169369


namespace NUMINAMATH_CALUDE_power_of_product_l1693_169303

theorem power_of_product (a b : ℝ) : ((-3 * a^2 * b^3)^2) = 9 * a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1693_169303


namespace NUMINAMATH_CALUDE_derivative_of_f_l1693_169344

noncomputable def f (x : ℝ) : ℝ := (2^x * (Real.sin x + Real.cos x * Real.log 2)) / (1 + Real.log 2 ^ 2)

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1693_169344


namespace NUMINAMATH_CALUDE_max_value_of_f_l1693_169306

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1693_169306


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l1693_169377

theorem stamp_collection_problem : ∃! x : ℕ, 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 5 = 3 ∧ 
  x % 9 = 7 ∧ 
  150 < x ∧ 
  x ≤ 300 ∧ 
  x = 223 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l1693_169377


namespace NUMINAMATH_CALUDE_time_addition_theorem_l1693_169318

/-- Represents time in a 12-hour format -/
structure Time12 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  isPM : Bool

/-- Represents a duration in hours, minutes, and seconds -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a time and returns the resulting time -/
def addDuration (t : Time12) (d : Duration) : Time12 := sorry

/-- Computes the sum of hours, minutes, and seconds for a given time -/
def sumComponents (t : Time12) : Nat := sorry

theorem time_addition_theorem (initialTime : Time12) (duration : Duration) :
  initialTime = Time12.mk 3 0 0 true →
  duration = Duration.mk 300 55 30 →
  (addDuration initialTime duration = Time12.mk 3 55 30 true) ∧
  (sumComponents (addDuration initialTime duration) = 88) := by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l1693_169318


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_fifth_l1693_169320

theorem sqrt_expression_equals_one_fifth :
  (Real.sqrt 3 + Real.sqrt 2) ^ (2 * (Real.log (Real.sqrt 5) / Real.log (Real.sqrt 3 - Real.sqrt 2))) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_fifth_l1693_169320


namespace NUMINAMATH_CALUDE_solve_equation_l1693_169352

theorem solve_equation : ∃ x : ℝ, (3 * x) / 4 = 15 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1693_169352


namespace NUMINAMATH_CALUDE_garden_area_l1693_169359

theorem garden_area (total_posts : ℕ) (post_spacing : ℕ) 
  (h1 : total_posts = 20)
  (h2 : post_spacing = 4)
  (h3 : ∃ (short_posts long_posts : ℕ), 
    short_posts > 1 ∧ 
    long_posts > 1 ∧ 
    short_posts + long_posts = total_posts / 2 + 2 ∧ 
    long_posts = 2 * short_posts) :
  ∃ (width length : ℕ), 
    width * length = 336 ∧ 
    width = post_spacing * (short_posts - 1) ∧ 
    length = post_spacing * (long_posts - 1) :=
by sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l1693_169359


namespace NUMINAMATH_CALUDE_license_plate_count_l1693_169375

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in a license plate -/
def num_plate_letters : ℕ := 2

/-- The number of possible positions for the letter block -/
def num_letter_block_positions : ℕ := num_plate_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := num_digits ^ num_plate_digits * (num_letters ^ num_plate_letters) * num_letter_block_positions

theorem license_plate_count :
  total_license_plates = 40560000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1693_169375


namespace NUMINAMATH_CALUDE_smallest_positive_period_monotonically_increasing_intervals_l1693_169341

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x * Real.cos x - 5 * Real.sqrt 3 * (Real.cos x)^2 + 5 * Real.sqrt 3 / 2

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧ 
  T = Real.pi :=
sorry

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (- Real.pi / 12 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_monotonically_increasing_intervals_l1693_169341


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l1693_169347

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line L passing through M(0, -1/3)
def line (x y k : ℝ) : Prop := y = k*x - 1/3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the intersection points A and B
def intersection_points (x1 y1 x2 y2 k : ℝ) : Prop :=
  point_on_ellipse x1 y1 ∧ point_on_ellipse x2 y2 ∧
  line x1 y1 k ∧ line x2 y2 k

-- Define the circle with diameter AB
def circle_AB (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (x - (x1 + x2)/2)^2 + (y - (y1 + y2)/2)^2 = ((x1 - x2)^2 + (y1 - y2)^2)/4

-- Theorem statement
theorem fixed_point_on_circle (k : ℝ) :
  ∀ x1 y1 x2 y2,
  intersection_points x1 y1 x2 y2 k →
  circle_AB 0 1 x1 y1 x2 y2 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l1693_169347


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1693_169362

theorem quadratic_factorization (x : ℝ) : 
  (x^2 - 4*x + 3 = (x-1)*(x-3)) ∧ 
  (4*x^2 + 12*x - 7 = (2*x+7)*(2*x-1)) := by
  sorry

#check quadratic_factorization

end NUMINAMATH_CALUDE_quadratic_factorization_l1693_169362


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1693_169354

-- Problem 1
theorem problem_1 (t : ℝ) : 
  (∀ x : ℝ, |2*x + t| - t ≤ 8 ↔ -5 ≤ x ∧ x ≤ 4) → t = 1 :=
sorry

-- Problem 2
theorem problem_2 (x y z : ℝ) :
  x^2 + (1/4)*y^2 + (1/9)*z^2 = 2 →
  x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1693_169354


namespace NUMINAMATH_CALUDE_triangle_preserves_triangle_parallelogram_preserves_parallelogram_square_not_always_square_rhombus_not_always_rhombus_l1693_169364

-- Define the oblique projection method
structure ObliqueProjection where
  -- Add necessary fields for oblique projection

-- Define geometric shapes
structure Triangle where
  -- Add necessary fields for triangle

structure Parallelogram where
  -- Add necessary fields for parallelogram

structure Square where
  -- Add necessary fields for square

structure Rhombus where
  -- Add necessary fields for rhombus

-- Define the intuitive diagram function
def intuitiveDiagram (op : ObliqueProjection) (shape : Type) : Type :=
  sorry

-- Theorem statements
theorem triangle_preserves_triangle (op : ObliqueProjection) (t : Triangle) :
  intuitiveDiagram op Triangle = Triangle :=
sorry

theorem parallelogram_preserves_parallelogram (op : ObliqueProjection) (p : Parallelogram) :
  intuitiveDiagram op Parallelogram = Parallelogram :=
sorry

theorem square_not_always_square (op : ObliqueProjection) :
  ¬(∀ (s : Square), intuitiveDiagram op Square = Square) :=
sorry

theorem rhombus_not_always_rhombus (op : ObliqueProjection) :
  ¬(∀ (r : Rhombus), intuitiveDiagram op Rhombus = Rhombus) :=
sorry

end NUMINAMATH_CALUDE_triangle_preserves_triangle_parallelogram_preserves_parallelogram_square_not_always_square_rhombus_not_always_rhombus_l1693_169364


namespace NUMINAMATH_CALUDE_dress_price_difference_l1693_169363

/-- Proves that the final price of a dress is $2.3531875 more than the original price
    given specific discounts, increases, and taxes. -/
theorem dress_price_difference (original_price : ℝ) : 
  original_price * 0.85 = 78.2 →
  let price_after_sale := 78.2
  let price_after_increase := price_after_sale * 1.25
  let price_after_coupon := price_after_increase * 0.9
  let final_price := price_after_coupon * 1.0725
  final_price - original_price = 2.3531875 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_difference_l1693_169363


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l1693_169338

theorem angle_sum_pi_half (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_eq : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) :
  α + β = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l1693_169338


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l1693_169319

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Property that g satisfies the given conditions -/
def satisfiesConditions (g : ThirdDegreePolynomial) : Prop :=
  (∀ i ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ), |g i| = 18) ∧
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d)

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : satisfiesConditions g) : |g 0| = 162 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l1693_169319


namespace NUMINAMATH_CALUDE_katie_mp3_songs_l1693_169387

theorem katie_mp3_songs (initial : ℕ) (deleted : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 7 → added = 24 → final = initial - deleted + added → final = 28 := by
  sorry

end NUMINAMATH_CALUDE_katie_mp3_songs_l1693_169387


namespace NUMINAMATH_CALUDE_petes_number_l1693_169379

theorem petes_number : ∃ x : ℝ, 4 * (2 * x + 20) = 200 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l1693_169379


namespace NUMINAMATH_CALUDE_elena_pen_purchase_l1693_169371

/-- The number of brand X pens Elena purchased -/
def brand_x_pens : ℕ := 9

/-- The number of brand Y pens Elena purchased -/
def brand_y_pens : ℕ := 12 - brand_x_pens

/-- The cost of a single brand X pen -/
def cost_x : ℚ := 4

/-- The cost of a single brand Y pen -/
def cost_y : ℚ := 2.2

/-- The total cost of all pens -/
def total_cost : ℚ := 42

theorem elena_pen_purchase :
  (brand_x_pens : ℚ) * cost_x + (brand_y_pens : ℚ) * cost_y = total_cost ∧
  brand_x_pens + brand_y_pens = 12 :=
by sorry

end NUMINAMATH_CALUDE_elena_pen_purchase_l1693_169371


namespace NUMINAMATH_CALUDE_remainder_2007_div_81_l1693_169316

theorem remainder_2007_div_81 : 2007 % 81 = 63 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2007_div_81_l1693_169316


namespace NUMINAMATH_CALUDE_function_inequality_l1693_169340

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 →
    |f a x₁ - f a x₂| ≤ a - 1) →
  a ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1693_169340


namespace NUMINAMATH_CALUDE_smallest_factor_for_square_l1693_169350

theorem smallest_factor_for_square (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 10 → ¬∃ k : ℕ, 4410 * m = k * k) ∧ 
  (∃ k : ℕ, 4410 * 10 = k * k) := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_square_l1693_169350


namespace NUMINAMATH_CALUDE_correct_calculation_l1693_169376

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1693_169376


namespace NUMINAMATH_CALUDE_percentage_difference_l1693_169342

theorem percentage_difference : 
  (60 * (50 / 100) * (40 / 100)) - (70 * (60 / 100) * (50 / 100)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1693_169342


namespace NUMINAMATH_CALUDE_total_selections_exactly_three_girls_at_most_three_girls_both_boys_and_girls_l1693_169330

-- Define the number of boys and girls
def num_boys : ℕ := 8
def num_girls : ℕ := 5
def total_people : ℕ := num_boys + num_girls
def selection_size : ℕ := 6

-- (1) Total number of ways to select 6 people
theorem total_selections : Nat.choose total_people selection_size = 1716 := by sorry

-- (2) Number of ways to select exactly 3 girls
theorem exactly_three_girls : 
  Nat.choose num_girls 3 * Nat.choose num_boys 3 = 560 := by sorry

-- (3) Number of ways to select at most 3 girls
theorem at_most_three_girls : 
  Nat.choose num_boys 6 + 
  Nat.choose num_boys 5 * Nat.choose num_girls 1 + 
  Nat.choose num_boys 4 * Nat.choose num_girls 2 + 
  Nat.choose num_boys 3 * Nat.choose num_girls 3 = 1568 := by sorry

-- (4) Number of ways to select both boys and girls
theorem both_boys_and_girls : 
  Nat.choose total_people selection_size - Nat.choose num_boys selection_size = 1688 := by sorry

end NUMINAMATH_CALUDE_total_selections_exactly_three_girls_at_most_three_girls_both_boys_and_girls_l1693_169330


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l1693_169367

-- Define a function f over the reals
variable (f : ℝ → ℝ)

-- State the theorem
theorem symmetry_about_x_equals_one (x : ℝ) : f (x - 1) = f (-(x - 2) + 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l1693_169367


namespace NUMINAMATH_CALUDE_jack_baseball_cards_l1693_169384

theorem jack_baseball_cards :
  ∀ (total_cards baseball_cards football_cards : ℕ),
    total_cards = 125 →
    baseball_cards = 3 * football_cards + 5 →
    total_cards = baseball_cards + football_cards →
    baseball_cards = 95 := by
  sorry

end NUMINAMATH_CALUDE_jack_baseball_cards_l1693_169384


namespace NUMINAMATH_CALUDE_expression_value_l1693_169358

theorem expression_value (x : ℝ) (h : x^2 + 2*x = 2) :
  (x - 1)^2 + (x + 3)*(x - 3) - (x - 3)*(x - 1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1693_169358


namespace NUMINAMATH_CALUDE_line_through_origin_l1693_169315

variable (m n p : ℝ)
variable (h : m ≠ 0 ∨ n ≠ 0 ∨ p ≠ 0)

def line_set : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | ∃ (k : ℝ), x = k * m ∧ y = k * n ∧ z = k * p}

theorem line_through_origin (m n p : ℝ) (h : m ≠ 0 ∨ n ≠ 0 ∨ p ≠ 0) :
  ∃ (a b c : ℝ), line_set m n p = {(x, y, z) | a * x + b * y + c * z = 0} ∧
  (0, 0, 0) ∈ line_set m n p :=
sorry

end NUMINAMATH_CALUDE_line_through_origin_l1693_169315


namespace NUMINAMATH_CALUDE_last_divisor_problem_l1693_169388

theorem last_divisor_problem (initial : ℚ) (div1 div2 mult last_div : ℚ) (result : ℚ) : 
  initial = 377 →
  div1 = 13 →
  div2 = 29 →
  mult = 1/4 →
  result = 0.125 →
  (((initial / div1) / div2) * mult) / last_div = result →
  last_div = 2 :=
by sorry

end NUMINAMATH_CALUDE_last_divisor_problem_l1693_169388


namespace NUMINAMATH_CALUDE_bella_stamps_l1693_169345

theorem bella_stamps (snowflake : ℕ) (truck : ℕ) (rose : ℕ) (butterfly : ℕ) 
  (h1 : snowflake = 15)
  (h2 : truck = snowflake + 11)
  (h3 : rose = truck - 17)
  (h4 : butterfly = 2 * rose) :
  snowflake + truck + rose + butterfly = 68 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamps_l1693_169345


namespace NUMINAMATH_CALUDE_not_perfect_square_l1693_169332

theorem not_perfect_square (n : ℕ+) : ¬∃ (m : ℕ), 4 * n^2 + 4 * n + 4 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1693_169332


namespace NUMINAMATH_CALUDE_arcsin_sufficient_not_necessary_l1693_169321

theorem arcsin_sufficient_not_necessary :
  (∃ α : ℝ, α = Real.arcsin (1/3) ∧ Real.sin α = 1/3) ∧
  (∃ β : ℝ, Real.sin β = 1/3 ∧ β ≠ Real.arcsin (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sufficient_not_necessary_l1693_169321


namespace NUMINAMATH_CALUDE_point_division_l1693_169381

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:5,
    prove that P can be expressed as a linear combination of A and B with coefficients 5/8 and 3/8 respectively. -/
theorem point_division (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) →  -- P is on line segment AB
  (dist A P) / (dist P B) = 3 / 5 →                     -- AP:PB = 3:5
  P = (5/8) • A + (3/8) • B :=                          -- P = (5/8)A + (3/8)B
by sorry

end NUMINAMATH_CALUDE_point_division_l1693_169381


namespace NUMINAMATH_CALUDE_incorrect_expression_l1693_169370

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : 
  (x - 2 * y) / y ≠ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1693_169370


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1693_169309

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1693_169309


namespace NUMINAMATH_CALUDE_continuous_stripe_theorem_l1693_169391

/-- A regular tetrahedron with painted stripes on each face -/
structure StripedTetrahedron where
  /-- Each face has a stripe from one edge center to the opposite edge center -/
  faces : Fin 4 → Bool

/-- The probability of a continuous stripe encircling the tetrahedron -/
def continuous_stripe_probability : ℚ :=
  1 / 8

/-- 
Theorem: The probability of a continuous stripe encircling a regular tetrahedron, 
given that each face has a randomly and independently painted stripe from the 
center of one edge to the center of the opposite edge, is 1/8.
-/
theorem continuous_stripe_theorem : 
  continuous_stripe_probability = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_theorem_l1693_169391


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l1693_169392

theorem inverse_sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x⁻¹ + y⁻¹)⁻¹ = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l1693_169392


namespace NUMINAMATH_CALUDE_impossibleToAchieveTwoHundreds_l1693_169304

/-- Represents the score changes that can be applied to exam scores. -/
inductive ScoreChange
  | AddOneToAll
  | DecreaseOneIncreaseTwo

/-- Represents the scores for three exams. -/
structure ExamScores where
  russian : ℕ
  physics : ℕ
  mathematics : ℕ

/-- Applies a score change to the exam scores. -/
def applyScoreChange (scores : ExamScores) (change : ScoreChange) : ExamScores :=
  match change with
  | ScoreChange.AddOneToAll =>
      { russian := scores.russian + 1,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }
  | ScoreChange.DecreaseOneIncreaseTwo =>
      { russian := scores.russian - 3,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }

/-- Checks if at least two scores are equal to 100. -/
def atLeastTwoEqual100 (scores : ExamScores) : Prop :=
  (scores.russian = 100 ∧ scores.physics = 100) ∨
  (scores.russian = 100 ∧ scores.mathematics = 100) ∨
  (scores.physics = 100 ∧ scores.mathematics = 100)

/-- Theorem stating the impossibility of achieving at least two scores of 100. -/
theorem impossibleToAchieveTwoHundreds (initialScores : ExamScores)
  (hRussian : initialScores.russian = initialScores.physics - 5)
  (hPhysics : initialScores.physics = initialScores.mathematics - 9)
  (hMaxScore : ∀ scores : ExamScores, scores.russian ≤ 100 ∧ scores.physics ≤ 100 ∧ scores.mathematics ≤ 100) :
  ¬∃ (changes : List ScoreChange), atLeastTwoEqual100 (changes.foldl applyScoreChange initialScores) :=
sorry

end NUMINAMATH_CALUDE_impossibleToAchieveTwoHundreds_l1693_169304


namespace NUMINAMATH_CALUDE_number_puzzle_l1693_169393

theorem number_puzzle : ∃ x : ℝ, 3 * (x + 2) = 24 + x ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1693_169393


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l1693_169357

theorem complex_in_second_quadrant (m : ℝ) :
  let z : ℂ := (2 + m * I) / (4 - 5 * I)
  (z.re < 0 ∧ z.im > 0) ↔ m > 8/5 := by sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l1693_169357


namespace NUMINAMATH_CALUDE_probability_both_types_selected_l1693_169334

def num_type_a : ℕ := 3
def num_type_b : ℕ := 2
def total_tvs : ℕ := num_type_a + num_type_b
def num_selected : ℕ := 3

theorem probability_both_types_selected :
  (Nat.choose num_type_a 2 * Nat.choose num_type_b 1 +
   Nat.choose num_type_a 1 * Nat.choose num_type_b 2) /
  Nat.choose total_tvs num_selected = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_both_types_selected_l1693_169334


namespace NUMINAMATH_CALUDE_upstream_distance_l1693_169390

/-- Proves that a man swimming downstream 16 km in 2 hours and upstream for 2 hours,
    with a speed of 6.5 km/h in still water, swims 10 km upstream. -/
theorem upstream_distance
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (still_water_speed : ℝ)
  (h_downstream_distance : downstream_distance = 16)
  (h_downstream_time : downstream_time = 2)
  (h_upstream_time : upstream_time = 2)
  (h_still_water_speed : still_water_speed = 6.5)
  : ∃ upstream_distance : ℝ,
    upstream_distance = 10 ∧
    upstream_distance = still_water_speed * upstream_time - 
      (downstream_distance / downstream_time - still_water_speed) * upstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_upstream_distance_l1693_169390
