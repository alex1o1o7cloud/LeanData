import Mathlib

namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l3400_340025

/-- Calculates the total number of ice cream cones sold in a week based on given sales pattern -/
def total_ice_cream_sales (monday : ℕ) (tuesday : ℕ) : ℕ :=
  let wednesday := 2 * tuesday
  let thursday := (3 * wednesday) / 2
  let friday := (3 * thursday) / 4
  let weekend := 2 * friday
  monday + tuesday + wednesday + thursday + friday + weekend

/-- Theorem stating that the total ice cream sales for the week is 163,000 -/
theorem ice_cream_sales_theorem : total_ice_cream_sales 10000 12000 = 163000 := by
  sorry

#eval total_ice_cream_sales 10000 12000

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l3400_340025


namespace NUMINAMATH_CALUDE_f_is_linear_function_l3400_340072

/-- A linear function is of the form y = kx + b, where k and b are constants, and k ≠ 0 -/
structure LinearFunction (α : Type*) [Ring α] where
  k : α
  b : α
  k_nonzero : k ≠ 0

/-- The function y = -3x + 1 -/
def f (x : ℝ) : ℝ := -3 * x + 1

/-- Theorem: f is a linear function -/
theorem f_is_linear_function : ∃ (lf : LinearFunction ℝ), ∀ x, f x = lf.k * x + lf.b :=
  sorry

end NUMINAMATH_CALUDE_f_is_linear_function_l3400_340072


namespace NUMINAMATH_CALUDE_pells_equation_unique_solution_l3400_340065

-- Define the fundamental solution
def fundamental_solution (x₀ y₀ : ℤ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1 ∧ x₀ > 0 ∧ y₀ > 0

-- Define the property that all prime factors of x divide x₀
def all_prime_factors_divide (x x₀ : ℤ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p : ℤ) ∣ x → (p : ℤ) ∣ x₀

-- Main theorem
theorem pells_equation_unique_solution
  (x₀ y₀ x y : ℤ)
  (h_fund : fundamental_solution x₀ y₀)
  (h_sol : x^2 - 2003 * y^2 = 1)
  (h_pos : x > 0 ∧ y > 0)
  (h_divide : all_prime_factors_divide x x₀) :
  x = x₀ ∧ y = y₀ :=
sorry

end NUMINAMATH_CALUDE_pells_equation_unique_solution_l3400_340065


namespace NUMINAMATH_CALUDE_tree_spacing_l3400_340063

/-- Given a sidewalk of length 148 feet where 8 trees are to be planted, 
    and each tree occupies 1 square foot, the space between each tree is 20 feet. -/
theorem tree_spacing (sidewalk_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) :
  sidewalk_length = 148 →
  num_trees = 8 →
  tree_space = 1 →
  (sidewalk_length - num_trees * tree_space) / (num_trees - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l3400_340063


namespace NUMINAMATH_CALUDE_remainder_13_pow_2000_mod_1000_l3400_340062

theorem remainder_13_pow_2000_mod_1000 : 13^2000 % 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2000_mod_1000_l3400_340062


namespace NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l3400_340081

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1 -/
theorem even_quadratic_implies_k_eq_one (k : ℝ) : IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l3400_340081


namespace NUMINAMATH_CALUDE_shekar_math_marks_l3400_340082

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates Shekar's marks in mathematics -/
def calculateMathMarks (marks : StudentMarks) (average : ℕ) (totalSubjects : ℕ) : ℕ :=
  average * totalSubjects - (marks.science + marks.socialStudies + marks.english + marks.biology)

/-- Theorem stating that Shekar's marks in mathematics are 76 -/
theorem shekar_math_marks :
  let marks : StudentMarks := {
    science := 65,
    socialStudies := 82,
    english := 67,
    biology := 95
  }
  let average : ℕ := 77
  let totalSubjects : ℕ := 5
  calculateMathMarks marks average totalSubjects = 76 := by
  sorry

end NUMINAMATH_CALUDE_shekar_math_marks_l3400_340082


namespace NUMINAMATH_CALUDE_sum_base5_equals_2112_l3400_340008

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The sum of 1234₅, 234₅, and 34₅ in base 5 is equal to 2112₅ -/
theorem sum_base5_equals_2112 :
  let a := base5ToDecimal [1, 2, 3, 4]
  let b := base5ToDecimal [2, 3, 4]
  let c := base5ToDecimal [3, 4]
  decimalToBase5 (a + b + c) = [2, 1, 1, 2] := by
  sorry


end NUMINAMATH_CALUDE_sum_base5_equals_2112_l3400_340008


namespace NUMINAMATH_CALUDE_boys_circle_distance_l3400_340087

/-- The least total distance traveled by 8 boys on a circle -/
theorem boys_circle_distance (n : ℕ) (r : ℝ) (h_n : n = 8) (h_r : r = 30) :
  let chord_length := 2 * r * Real.sqrt ((2 : ℝ) + Real.sqrt 2) / 2
  let non_adjacent_count := n - 3
  let total_distance := n * non_adjacent_count * chord_length
  total_distance = 1200 * Real.sqrt ((2 : ℝ) + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_boys_circle_distance_l3400_340087


namespace NUMINAMATH_CALUDE_vowel_soup_combinations_l3400_340090

/-- The number of vowels available -/
def num_vowels : ℕ := 5

/-- The length of the words to be formed -/
def word_length : ℕ := 6

/-- The number of times each vowel appears in the bowl -/
def vowel_count : ℕ := 7

/-- The total number of six-letter words that can be formed -/
def total_combinations : ℕ := num_vowels ^ word_length

theorem vowel_soup_combinations :
  total_combinations = 15625 :=
sorry

end NUMINAMATH_CALUDE_vowel_soup_combinations_l3400_340090


namespace NUMINAMATH_CALUDE_min_perimeter_exists_l3400_340029

def min_perimeter_isosceles_triangles (a b x : ℤ) : Prop :=
  let triangle1_base := 20 * x
  let triangle2_base := 25 * x
  let triangle1_perimeter := 2 * a + triangle1_base
  let triangle2_perimeter := 2 * b + triangle2_base
  let triangle1_height := (a^2 - (triangle1_base / 2)^2).sqrt
  let triangle2_height := (b^2 - (triangle2_base / 2)^2).sqrt
  let triangle1_area := (triangle1_base * triangle1_height) / 2
  let triangle2_area := (triangle2_base * triangle2_height) / 2
  (triangle1_perimeter = triangle2_perimeter) ∧
  (triangle1_area = triangle2_area) ∧
  (triangle1_base * 4 = triangle2_base * 5) ∧
  (x = 2 * (a - b)) ∧
  (a ≠ b)

theorem min_perimeter_exists :
  ∃ (a b x : ℤ), min_perimeter_isosceles_triangles a b x ∧
  ∀ (a' b' x' : ℤ), min_perimeter_isosceles_triangles a' b' x' →
  (2 * a + 20 * x ≤ 2 * a' + 20 * x') :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_exists_l3400_340029


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3400_340057

theorem double_inequality_solution (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9 * x + 3) ↔ 
  (x > 3 + 2 * Real.sqrt 2 ∧ x < 5.5 + Real.sqrt 32.25) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3400_340057


namespace NUMINAMATH_CALUDE_train_crossing_time_l3400_340080

/-- Given a train and platform with specific dimensions and time to pass,
    prove the time it takes for the train to cross a point (tree) -/
theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 700)
  (h3 : time_to_pass_platform = 190)
  : (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3400_340080


namespace NUMINAMATH_CALUDE_paperbacks_count_l3400_340032

/-- The number of books on the shelf -/
def total_books : ℕ := 8

/-- The number of hardback books on the shelf -/
def hardbacks : ℕ := 6

/-- The number of possible selections of 3 books that include at least one paperback -/
def selections_with_paperback : ℕ := 36

/-- The number of paperbacks on the shelf -/
def paperbacks : ℕ := total_books - hardbacks

/-- Theorem stating that the number of paperbacks is 2 -/
theorem paperbacks_count : paperbacks = 2 := by sorry

end NUMINAMATH_CALUDE_paperbacks_count_l3400_340032


namespace NUMINAMATH_CALUDE_heat_of_neutralization_instruments_l3400_340042

-- Define the set of available instruments
inductive Instrument
  | Balance
  | MeasuringCylinder
  | Beaker
  | Burette
  | Thermometer
  | TestTube
  | AlcoholLamp

-- Define the requirements for the heat of neutralization experiment
structure ExperimentRequirements where
  needsWeighing : Bool
  needsHeating : Bool
  reactionContainer : Instrument
  volumeMeasurementTool : Instrument
  temperatureMeasurementTool : Instrument

-- Define the correct set of instruments
def correctInstruments : Set Instrument :=
  {Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer}

-- Define the heat of neutralization experiment requirements
def heatOfNeutralizationRequirements : ExperimentRequirements :=
  { needsWeighing := false
  , needsHeating := false
  , reactionContainer := Instrument.Beaker
  , volumeMeasurementTool := Instrument.MeasuringCylinder
  , temperatureMeasurementTool := Instrument.Thermometer
  }

-- Theorem statement
theorem heat_of_neutralization_instruments :
  correctInstruments = 
    { i : Instrument | i = heatOfNeutralizationRequirements.volumeMeasurementTool ∨
                       i = heatOfNeutralizationRequirements.reactionContainer ∨
                       i = heatOfNeutralizationRequirements.temperatureMeasurementTool } :=
by sorry

end NUMINAMATH_CALUDE_heat_of_neutralization_instruments_l3400_340042


namespace NUMINAMATH_CALUDE_exam_mean_score_l3400_340083

/-- Given an exam where a score of 58 is 2 standard deviations below the mean
    and a score of 98 is 3 standard deviations above the mean,
    prove that the mean score is 74. -/
theorem exam_mean_score (mean sd : ℝ) 
    (h1 : 58 = mean - 2 * sd)
    (h2 : 98 = mean + 3 * sd) : 
  mean = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3400_340083


namespace NUMINAMATH_CALUDE_age_difference_l3400_340011

-- Define variables for ages
variable (a b c d : ℕ)

-- Define the conditions
def condition1 : Prop := a + b = b + c + 15
def condition2 : Prop := a + d = c + d + 12
def condition3 : Prop := a = d + 3

-- Theorem statement
theorem age_difference (h1 : condition1 a b c) (h2 : condition2 a c d) (h3 : condition3 a d) :
  a - c = 12 := by sorry

end NUMINAMATH_CALUDE_age_difference_l3400_340011


namespace NUMINAMATH_CALUDE_least_possible_difference_l3400_340002

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → 
  Odd y → 
  Odd z → 
  (∀ d : ℤ, d = z - x → d ≥ 9) ∧ (∃ x' y' z' : ℤ, x' < y' ∧ y' < z' ∧ y' - x' > 5 ∧ Even x' ∧ Odd y' ∧ Odd z' ∧ z' - x' = 9) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l3400_340002


namespace NUMINAMATH_CALUDE_time_addition_theorem_l3400_340049

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time, wrapping around in 12-hour format -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Calculates the sum of hour, minute, and second components of a time -/
def sumComponents (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_addition_theorem :
  let startTime := Time.mk 3 0 0
  let finalTime := addTime startTime 315 58 36
  finalTime = Time.mk 6 58 36 ∧ sumComponents finalTime = 100 := by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l3400_340049


namespace NUMINAMATH_CALUDE_xy_sum_over_five_l3400_340010

theorem xy_sum_over_five (x y : ℝ) (h1 : x * y > 0) (h2 : 1 / x + 1 / y = 15) (h3 : 1 / (x * y) = 5) :
  (x + y) / 5 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_over_five_l3400_340010


namespace NUMINAMATH_CALUDE_line_equations_l3400_340061

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y - 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Define the line passing through the origin
def line_through_origin (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the line parallel to 2x + y + 5 = 0
def parallel_line (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the line perpendicular to 2x + y + 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 5 = 0

theorem line_equations :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = M) →
  (line_through_origin M.1 M.2) ∧
  (parallel_line M.1 M.2) ∧
  (perpendicular_line M.1 M.2) := by sorry

end NUMINAMATH_CALUDE_line_equations_l3400_340061


namespace NUMINAMATH_CALUDE_least_common_multiple_of_first_ten_l3400_340067

/-- The least positive integer divisible by each of the first ten positive integers -/
def leastCommonMultiple : ℕ := 2520

/-- The set of the first ten positive integers -/
def firstTenIntegers : Finset ℕ := Finset.range 10

theorem least_common_multiple_of_first_ten :
  (∀ n ∈ firstTenIntegers, leastCommonMultiple % (n + 1) = 0) ∧
  (∀ m : ℕ, m > 0 → m < leastCommonMultiple →
    ∃ k ∈ firstTenIntegers, m % (k + 1) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_first_ten_l3400_340067


namespace NUMINAMATH_CALUDE_two_points_determine_line_l3400_340018

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem: Two distinct points determine a unique line
theorem two_points_determine_line (p1 p2 : Point2D) (h : p1 ≠ p2) :
  ∃! l : Line2D, pointOnLine p1 l ∧ pointOnLine p2 l :=
sorry

end NUMINAMATH_CALUDE_two_points_determine_line_l3400_340018


namespace NUMINAMATH_CALUDE_smallest_regiment_size_exact_smallest_regiment_size_new_uniforms_condition_l3400_340033

theorem smallest_regiment_size (m n : ℕ) (h1 : m ≥ 40) (h2 : n ≥ 30) : m * n ≥ 1200 := by
  sorry

theorem exact_smallest_regiment_size : ∃ m n : ℕ, m ≥ 40 ∧ n ≥ 30 ∧ m * n = 1200 := by
  sorry

theorem new_uniforms_condition (m n : ℕ) (h1 : m ≥ 40) (h2 : n ≥ 30) :
  (m * n : ℚ) / 100 ≥ (0.3 : ℚ) * m ∧ (m * n : ℚ) / 100 ≥ (0.4 : ℚ) * n := by
  sorry

end NUMINAMATH_CALUDE_smallest_regiment_size_exact_smallest_regiment_size_new_uniforms_condition_l3400_340033


namespace NUMINAMATH_CALUDE_gcd_of_319_377_116_l3400_340070

theorem gcd_of_319_377_116 : Nat.gcd 319 (Nat.gcd 377 116) = 29 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_319_377_116_l3400_340070


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3400_340059

/-- An ellipse with foci at (7, 15) and (53, 65) that is tangent to the y-axis has a major axis of length 68. -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ),
    F₁ = (7, 15) →
    F₂ = (53, 65) →
    (∃ (y : ℝ), (0, y) ∈ E) →
    (∀ (P : ℝ × ℝ), P ∈ E ↔ 
      ∃ (k : ℝ), dist P F₁ + dist P F₂ = k ∧ 
      ∀ (Q : ℝ × ℝ), dist Q F₁ + dist Q F₂ ≤ k) →
    ∃ (a : ℝ), a = 68 ∧ 
      ∀ (P : ℝ × ℝ), P ∈ E → dist P F₁ + dist P F₂ = a :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3400_340059


namespace NUMINAMATH_CALUDE_cricket_runs_l3400_340043

theorem cricket_runs (a b c : ℕ) (h1 : 3 * a = b) (h2 : 5 * b = c) (h3 : a + b + c = 95) :
  c = 75 := by
  sorry

end NUMINAMATH_CALUDE_cricket_runs_l3400_340043


namespace NUMINAMATH_CALUDE_inequality_proof_l3400_340074

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 3) :
  (b * c / (1 + a^4)) + (c * a / (1 + b^4)) + (a * b / (1 + c^4)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3400_340074


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3400_340058

theorem angle_ABC_measure :
  ∀ (ABC ABD CBD : ℝ),
  CBD = 90 →
  ABC + ABD + CBD = 180 →
  ABD = 60 →
  ABC = 30 := by
sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3400_340058


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l3400_340099

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x + 1 -/
def reflect_y_eq_x_plus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let p'' := (p'.2, p'.1)   -- Reflect across y = x
  (p''.1, p''.2 + 1)        -- Translate back up by 1

/-- The main theorem -/
theorem double_reflection_of_D (D : ℝ × ℝ) (h : D = (4, 1)) :
  reflect_y_eq_x_plus_1 (reflect_x D) = (-2, 5) := by
  sorry


end NUMINAMATH_CALUDE_double_reflection_of_D_l3400_340099


namespace NUMINAMATH_CALUDE_triangle_with_ratio_1_2_3_is_right_triangle_l3400_340073

-- Define a triangle type
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

-- Define the properties of a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧ t.angle1 + t.angle2 + t.angle3 = 180

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Define a triangle with angles in the ratio 1:2:3
def triangle_with_ratio_1_2_3 (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t.angle1 = k ∧ t.angle2 = 2*k ∧ t.angle3 = 3*k

-- Theorem statement
theorem triangle_with_ratio_1_2_3_is_right_triangle (t : Triangle) :
  is_valid_triangle t → triangle_with_ratio_1_2_3 t → is_right_triangle t :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_ratio_1_2_3_is_right_triangle_l3400_340073


namespace NUMINAMATH_CALUDE_landscape_ratio_l3400_340051

/-- Proves that the ratio of breadth to length of a rectangular landscape is 8:1 given specific conditions --/
theorem landscape_ratio (B L : ℝ) (n : ℝ) (h1 : B = n * L) (h2 : B = 480) 
  (h3 : B * L = 3200 * 9) : B / L = 8 := by
  sorry

end NUMINAMATH_CALUDE_landscape_ratio_l3400_340051


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3400_340031

/-- Definition of a hyperbola with equation x^2 - y^2/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- Definition of the asymptotes y = ±2x -/
def asymptotes (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- Theorem stating that the given hyperbola has the specified asymptotes -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ 0 ∧ asymptotes x' y') :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3400_340031


namespace NUMINAMATH_CALUDE_octagon_arc_length_l3400_340071

/-- The arc length intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (side_length : ℝ) (h : side_length = 4) :
  let radius : ℝ := side_length
  let circumference : ℝ := 2 * π * radius
  let central_angle : ℝ := π / 4  -- 45 degrees in radians
  let arc_length : ℝ := (central_angle / (2 * π)) * circumference
  arc_length = π :=
by sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l3400_340071


namespace NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_proof_l3400_340014

/-- A function that checks if a natural number only contains digits 0 and 1 in base 10 -/
def only_zero_one_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The smallest natural number with only 0 and 1 digits divisible by 225 -/
def smallest_binary_divisible_by_225 : ℕ := 11111111100

theorem smallest_binary_divisible_by_225_proof :
  (smallest_binary_divisible_by_225 % 225 = 0) ∧
  only_zero_one_digits smallest_binary_divisible_by_225 ∧
  ∀ n : ℕ, n < smallest_binary_divisible_by_225 →
    ¬(n % 225 = 0 ∧ only_zero_one_digits n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_proof_l3400_340014


namespace NUMINAMATH_CALUDE_kenny_mushroom_pieces_l3400_340026

/-- The number of mushroom pieces Kenny used on his pizza -/
def kenny_pieces (total_mushrooms : ℕ) (pieces_per_mushroom : ℕ) (karla_pieces : ℕ) (remaining_pieces : ℕ) : ℕ :=
  total_mushrooms * pieces_per_mushroom - (karla_pieces + remaining_pieces)

/-- Theorem stating the number of mushroom pieces Kenny used -/
theorem kenny_mushroom_pieces :
  kenny_pieces 22 4 42 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_kenny_mushroom_pieces_l3400_340026


namespace NUMINAMATH_CALUDE_sixth_root_unity_product_l3400_340036

theorem sixth_root_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_unity_product_l3400_340036


namespace NUMINAMATH_CALUDE_women_who_left_l3400_340004

theorem women_who_left (initial_men initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  ∃ (left : ℕ), 2 * (initial_women - left) = 24 ∧ left = 3 :=
by sorry

end NUMINAMATH_CALUDE_women_who_left_l3400_340004


namespace NUMINAMATH_CALUDE_right_triangle_area_l3400_340030

/-- The area of a right triangle with base 30 and height 24 is 360 -/
theorem right_triangle_area :
  let base : ℝ := 30
  let height : ℝ := 24
  (1 / 2 : ℝ) * base * height = 360 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3400_340030


namespace NUMINAMATH_CALUDE_solve_system_l3400_340048

theorem solve_system (x y b : ℚ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 7 * y = 3 * b) → 
  (y = 3) → 
  (b = 22 / 3) := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3400_340048


namespace NUMINAMATH_CALUDE_alice_survey_l3400_340047

theorem alice_survey (total_students : ℕ) 
  (malfunction_believers : ℕ) 
  (password_believers : ℕ) :
  (malfunction_believers : ℚ) / (total_students : ℚ) = 723/1000 →
  (password_believers : ℚ) / (malfunction_believers : ℚ) = 346/1000 →
  password_believers = 18 →
  total_students = 72 := by
sorry

end NUMINAMATH_CALUDE_alice_survey_l3400_340047


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l3400_340035

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : 
  binomial_probability 6 4 0.5 = 0.234375 := by sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l3400_340035


namespace NUMINAMATH_CALUDE_complex_square_l3400_340095

/-- Given that i^2 = -1, prove that (3 - 4i)^2 = 5 - 24i -/
theorem complex_square (i : ℂ) (h : i^2 = -1) : (3 - 4*i)^2 = 5 - 24*i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l3400_340095


namespace NUMINAMATH_CALUDE_blocks_per_friend_l3400_340009

theorem blocks_per_friend (total_blocks : ℕ) (num_friends : ℕ) (blocks_per_friend : ℕ) : 
  total_blocks = 28 → num_friends = 4 → blocks_per_friend = total_blocks / num_friends → blocks_per_friend = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_friend_l3400_340009


namespace NUMINAMATH_CALUDE_product_of_two_primes_not_prime_l3400_340041

/-- A number is prime if it's greater than 1 and its only positive divisors are 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem product_of_two_primes_not_prime (a b : ℤ) :
  isPrime (Int.natAbs (a * b)) → ¬(isPrime (Int.natAbs a) ∧ isPrime (Int.natAbs b)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_primes_not_prime_l3400_340041


namespace NUMINAMATH_CALUDE_least_difference_l3400_340086

theorem least_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 5 ∧
  Even x ∧
  Nat.Prime y.toNat ∧ Odd y ∧
  Odd z ∧ z % 3 = 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∀ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧
    y' - x' > 5 ∧
    Even x' ∧
    Nat.Prime y'.toNat ∧ Odd y' ∧
    Odd z' ∧ z' % 3 = 0 ∧
    x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' →
    z - x ≤ z' - x' ∧
    z - x = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_difference_l3400_340086


namespace NUMINAMATH_CALUDE_bobs_grade_is_35_l3400_340027

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℚ := jason_grade / 2

theorem bobs_grade_is_35 : bob_grade = 35 := by sorry

end NUMINAMATH_CALUDE_bobs_grade_is_35_l3400_340027


namespace NUMINAMATH_CALUDE_circle_center_l3400_340060

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- The center of a circle -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 9

/-- Theorem: The center of the circle is (1, 3) -/
theorem circle_center : is_center 1 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3400_340060


namespace NUMINAMATH_CALUDE_magic_square_solution_l3400_340091

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℚ)
  (row_sum : ℚ)
  (magic_property : 
    a11 + a12 + a13 = row_sum ∧
    a21 + a22 + a23 = row_sum ∧
    a31 + a32 + a33 = row_sum ∧
    a11 + a21 + a31 = row_sum ∧
    a12 + a22 + a32 = row_sum ∧
    a13 + a23 + a33 = row_sum ∧
    a11 + a22 + a33 = row_sum ∧
    a13 + a22 + a31 = row_sum)

/-- The theorem stating the solution to the magic square problem -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.a12 = 25)
  (h2 : ms.a13 = 64)
  (h3 : ms.a21 = 3) :
  ms.a11 = 272 / 3 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l3400_340091


namespace NUMINAMATH_CALUDE_triangle_properties_l3400_340064

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: If b - (1/2)c = a cos C, 4(b + c) = 3bc, and a = 2√3 in a triangle ABC,
    then angle A = 60° and the area of the triangle is 2√3 --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b - (1/2) * t.c = t.a * Real.cos t.C)
  (h2 : 4 * (t.b + t.c) = 3 * t.b * t.c)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3400_340064


namespace NUMINAMATH_CALUDE_bookshelf_count_l3400_340022

theorem bookshelf_count (books_per_shelf : ℕ) (total_books : ℕ) (shelf_count : ℕ) : 
  books_per_shelf = 15 → 
  total_books = 2250 → 
  shelf_count * books_per_shelf = total_books → 
  shelf_count = 150 := by
sorry

end NUMINAMATH_CALUDE_bookshelf_count_l3400_340022


namespace NUMINAMATH_CALUDE_sine_law_application_l3400_340046

/-- Given a triangle ABC with sides a and b opposite to angles A and B respectively,
    if a = 2√2, b = 3, and sin A = √2/6, then sin B = 1/4 -/
theorem sine_law_application (a b : ℝ) (A B : ℝ) :
  a = 2 * Real.sqrt 2 →
  b = 3 →
  Real.sin A = Real.sqrt 2 / 6 →
  Real.sin B = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_sine_law_application_l3400_340046


namespace NUMINAMATH_CALUDE_unique_solution_E_l3400_340023

/-- Definition of the function E -/
def E (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that 3/8 is the unique solution to E(a, 3, 12) = E(a, 5, 6) -/
theorem unique_solution_E :
  ∃! a : ℝ, E a 3 12 = E a 5 6 ∧ a = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_E_l3400_340023


namespace NUMINAMATH_CALUDE_prince_total_spent_prince_total_spent_proof_l3400_340056

-- Define the total number of CDs
def total_cds : ℕ := 200

-- Define the percentage of CDs that cost $10
def percentage_expensive : ℚ := 40 / 100

-- Define the cost of expensive CDs
def cost_expensive : ℕ := 10

-- Define the cost of cheap CDs
def cost_cheap : ℕ := 5

-- Define the fraction of expensive CDs Prince bought
def fraction_bought : ℚ := 1 / 2

-- Theorem to prove
theorem prince_total_spent (total_cds : ℕ) (percentage_expensive : ℚ) 
  (cost_expensive cost_cheap : ℕ) (fraction_bought : ℚ) : ℕ :=
  -- The total amount Prince spent on CDs
  1000

-- Proof of the theorem
theorem prince_total_spent_proof :
  prince_total_spent total_cds percentage_expensive cost_expensive cost_cheap fraction_bought = 1000 := by
  sorry

end NUMINAMATH_CALUDE_prince_total_spent_prince_total_spent_proof_l3400_340056


namespace NUMINAMATH_CALUDE_system_solution_l3400_340077

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 7 * y = -14) ∧ 
    (5 * x + 3 * y = -13) ∧ 
    (x = -133 / 47) ∧ 
    (y = 18 / 47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3400_340077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3400_340054

/-- 
Theorem: The number of terms in an arithmetic sequence 
starting with 2, ending with 2014, and having a common difference of 4 
is equal to 504.
-/
theorem arithmetic_sequence_length : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 4   -- Common difference
  let aₙ : ℕ := 2014  -- Last term
  ∃ n : ℕ, n > 0 ∧ aₙ = a₁ + (n - 1) * d ∧ n = 504
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3400_340054


namespace NUMINAMATH_CALUDE_divisible_by_five_not_ending_in_five_l3400_340015

theorem divisible_by_five_not_ending_in_five : ∃ n : ℕ, 5 ∣ n ∧ n % 10 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_not_ending_in_five_l3400_340015


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3400_340028

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3400_340028


namespace NUMINAMATH_CALUDE_square_triangle_apothem_ratio_l3400_340019

theorem square_triangle_apothem_ratio :
  ∀ (s t : ℝ),
  s > 0 → t > 0 →
  s * Real.sqrt 2 = 9 * t →  -- diagonal of square = 3 * perimeter of triangle
  s * s = 2 * s →           -- apothem of square = area of square
  (s / 2) / ((Real.sqrt 3 / 2 * t) / 3) = 9 * Real.sqrt 6 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_triangle_apothem_ratio_l3400_340019


namespace NUMINAMATH_CALUDE_simplify_absolute_value_expression_l3400_340050

theorem simplify_absolute_value_expression
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y < 0)
  (hz : z < 0)
  (hxy : |x| > |y|)
  (hzx : |z| > |x|) :
  |x + z| - |y + z| - |x + y| = -2 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_expression_l3400_340050


namespace NUMINAMATH_CALUDE_intersection_tangent_line_l3400_340094

theorem intersection_tangent_line (x₀ : ℝ) (hx₀ : x₀ ≠ 0) (h : Real.tan x₀ = -x₀) :
  (x₀^2 + 1) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_tangent_line_l3400_340094


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3400_340097

-- Define the number of diagonals in a regular polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Theorem statement
theorem regular_polygon_sides :
  ∀ n : ℕ, n ≥ 3 →
  (num_diagonals n + 2 * n = n^2) → n = 3 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3400_340097


namespace NUMINAMATH_CALUDE_p_minus_q_plus_r_equals_two_thirds_l3400_340076

theorem p_minus_q_plus_r_equals_two_thirds
  (p q r : ℚ)
  (hp : 3 / p = 6)
  (hq : 3 / q = 18)
  (hr : 5 / r = 15) :
  p - q + r = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_p_minus_q_plus_r_equals_two_thirds_l3400_340076


namespace NUMINAMATH_CALUDE_cylinder_increase_equality_l3400_340003

theorem cylinder_increase_equality (x : ℝ) : 
  x > 0 → 
  π * (8 + x)^2 * 3 = π * 8^2 * (3 + x) → 
  x = 16/3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_increase_equality_l3400_340003


namespace NUMINAMATH_CALUDE_min_white_pairs_problem_solution_l3400_340085

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Nat)

/-- Calculates the total number of adjacent cell pairs in a square grid -/
def total_pairs (g : Grid) : Nat :=
  2 * g.size * (g.size - 1)

/-- Calculates the maximum number of central cell pairs -/
def max_central_pairs (g : Grid) : Nat :=
  ((g.size - 2) * (g.size - 2)) / 2

/-- Theorem: Given an 8x8 grid with 20 black cells, the minimum number of pairs of adjacent white cells is 34 -/
theorem min_white_pairs (g : Grid) (h1 : g.size = 8) (h2 : g.black_cells = 20) :
  total_pairs g - (60 + min g.black_cells (max_central_pairs g)) = 34 := by
  sorry

/-- Main theorem stating the result for the specific problem -/
theorem problem_solution : 
  ∃ (g : Grid), g.size = 8 ∧ g.black_cells = 20 ∧ 
  (total_pairs g - (60 + min g.black_cells (max_central_pairs g)) = 34) := by
  sorry

end NUMINAMATH_CALUDE_min_white_pairs_problem_solution_l3400_340085


namespace NUMINAMATH_CALUDE_no_intersection_points_l3400_340066

theorem no_intersection_points : 
  ¬∃ (z : ℂ), z^4 + z = 1 ∧ Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_no_intersection_points_l3400_340066


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l3400_340037

-- Define the structure of a rectangle divided into four parts
structure DividedRectangle where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

-- Define the property that the given areas are 24, 12, and 8
def has_given_areas (r : DividedRectangle) : Prop :=
  (r.area1 = 24 ∧ r.area2 = 12 ∧ r.area3 = 8) ∨
  (r.area1 = 24 ∧ r.area2 = 12 ∧ r.area4 = 8) ∨
  (r.area1 = 24 ∧ r.area3 = 12 ∧ r.area4 = 8) ∨
  (r.area2 = 24 ∧ r.area3 = 12 ∧ r.area4 = 8)

-- Theorem statement
theorem fourth_rectangle_area (r : DividedRectangle) :
  has_given_areas r → r.area1 = 16 ∨ r.area2 = 16 ∨ r.area3 = 16 ∨ r.area4 = 16 :=
by sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l3400_340037


namespace NUMINAMATH_CALUDE_statistics_test_probability_l3400_340000

def word : String := "STATISTICS"
def test_word : String := "TEST"

def letter_count (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem statistics_test_probability :
  let total_tiles := word.length
  let overlapping_tiles := (test_word.toList.eraseDups.filter (λ c => word.contains c))
                            |>.map (λ c => letter_count word c)
                            |>.sum
  (↑overlapping_tiles : ℚ) / total_tiles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_statistics_test_probability_l3400_340000


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3400_340055

def is_solution (x y z : ℕ+) : Prop :=
  (x + y) * (y + z) * (z + x) = x * y * z * (x + y + z) ∧
  Nat.gcd x.val y.val = 1 ∧ Nat.gcd y.val z.val = 1 ∧ Nat.gcd z.val x.val = 1

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+, is_solution x y z ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
     (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
     (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3400_340055


namespace NUMINAMATH_CALUDE_parametric_eq_line_l3400_340069

/-- Prove that the parametric equations x = t - 1 and y = 2t - 1 represent the line y = 2x + 1 for all real values of t. -/
theorem parametric_eq_line (t : ℝ) : 
  let x := t - 1
  let y := 2*t - 1
  y = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_eq_line_l3400_340069


namespace NUMINAMATH_CALUDE_soda_duration_problem_l3400_340038

/-- Given the number of soda and water bottles, and the daily consumption ratio,
    calculate the number of days the soda bottles will last. -/
def sodaDuration (sodaCount waterCount : ℕ) (sodaRatio waterRatio : ℕ) : ℕ :=
  min (sodaCount / sodaRatio) (waterCount / waterRatio)

/-- Theorem stating that with 360 soda bottles and 162 water bottles,
    consumed in a 3:2 ratio, the soda bottles will last for 81 days. -/
theorem soda_duration_problem :
  sodaDuration 360 162 3 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_soda_duration_problem_l3400_340038


namespace NUMINAMATH_CALUDE_big_sale_commission_l3400_340024

def commission_problem (new_average : ℝ) (num_sales : ℕ) (average_increase : ℝ) : Prop :=
  let old_average := new_average - average_increase
  let old_total := old_average * (num_sales - 1 : ℝ)
  let new_total := new_average * num_sales
  new_total - old_total = 1150

theorem big_sale_commission : 
  commission_problem 400 6 150 := by sorry

end NUMINAMATH_CALUDE_big_sale_commission_l3400_340024


namespace NUMINAMATH_CALUDE_three_conclusions_correct_l3400_340096

-- Define the "heap" for natural numbers
def heap (r : Nat) : Set Nat := {n : Nat | ∃ k : Nat, n = 3 * k + r}

-- Define the four conclusions
def conclusion1 : Prop := 2011 ∈ heap 1
def conclusion2 : Prop := ∀ a b : Nat, a ∈ heap 1 → b ∈ heap 2 → (a + b) ∈ heap 0
def conclusion3 : Prop := (heap 0) ∪ (heap 1) ∪ (heap 2) = Set.univ
def conclusion4 : Prop := ∀ r : Fin 3, ∀ a b : Nat, a ∈ heap r → b ∈ heap r → (a - b) ∉ heap r

-- Theorem stating that exactly 3 out of 4 conclusions are correct
theorem three_conclusions_correct :
  (conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬conclusion4) ∨
  (conclusion1 ∧ conclusion2 ∧ ¬conclusion3 ∧ conclusion4) ∨
  (conclusion1 ∧ ¬conclusion2 ∧ conclusion3 ∧ conclusion4) ∨
  (¬conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ conclusion4) :=
sorry

end NUMINAMATH_CALUDE_three_conclusions_correct_l3400_340096


namespace NUMINAMATH_CALUDE_carla_bug_collection_l3400_340079

theorem carla_bug_collection (leaves : ℕ) (days : ℕ) (items_per_day : ℕ) 
  (h1 : leaves = 30)
  (h2 : days = 10)
  (h3 : items_per_day = 5) :
  let total_items := days * items_per_day
  let bugs := total_items - leaves
  bugs = 20 := by sorry

end NUMINAMATH_CALUDE_carla_bug_collection_l3400_340079


namespace NUMINAMATH_CALUDE_curve_decomposition_l3400_340044

-- Define the curve
def curve (x y : ℝ) : Prop := (x + y - 1) * Real.sqrt (x - 1) = 0

-- Define the line x = 1
def line (x y : ℝ) : Prop := x = 1

-- Define the ray x + y - 1 = 0 where x ≥ 1
def ray (x y : ℝ) : Prop := x + y - 1 = 0 ∧ x ≥ 1

-- Theorem statement
theorem curve_decomposition :
  ∀ x y : ℝ, x ≥ 1 → (curve x y ↔ line x y ∨ ray x y) :=
sorry

end NUMINAMATH_CALUDE_curve_decomposition_l3400_340044


namespace NUMINAMATH_CALUDE_banana_milk_distribution_l3400_340039

/-- The amount of banana milk Hyeonju drinks in milliliters -/
def hyeonju_amount : ℕ := 1000

/-- The amount of banana milk Jinsol drinks in milliliters -/
def jinsol_amount : ℕ := hyeonju_amount + 200

/-- The amount of banana milk Changhyeok drinks in milliliters -/
def changhyeok_amount : ℕ := hyeonju_amount - 200

/-- The total amount of banana milk in milliliters -/
def total_amount : ℕ := 3000

theorem banana_milk_distribution :
  hyeonju_amount + jinsol_amount + changhyeok_amount = total_amount ∧
  jinsol_amount = hyeonju_amount + 200 ∧
  hyeonju_amount = changhyeok_amount + 200 := by
  sorry

end NUMINAMATH_CALUDE_banana_milk_distribution_l3400_340039


namespace NUMINAMATH_CALUDE_incorrect_inequality_for_all_reals_l3400_340012

theorem incorrect_inequality_for_all_reals : 
  ¬(∀ x : ℝ, x + (1 / x) ≥ 2 * Real.sqrt (x * (1 / x))) :=
sorry

end NUMINAMATH_CALUDE_incorrect_inequality_for_all_reals_l3400_340012


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_sqrt_l3400_340093

theorem fourth_power_of_nested_sqrt (y : ℝ) :
  y = Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 5)) →
  y^4 = 12 + 6 * Real.sqrt (3 + Real.sqrt 5) + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_sqrt_l3400_340093


namespace NUMINAMATH_CALUDE_unique_seven_l3400_340078

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if all numbers in the grid are unique and between 1 and 9 -/
def valid_numbers (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Check if the sum of each row, column, and diagonal is 18 -/
def sum_18 (g : Grid) : Prop :=
  (∀ i, g i 0 + g i 1 + g i 2 = 18) ∧  -- rows
  (∀ j, g 0 j + g 1 j + g 2 j = 18) ∧  -- columns
  (g 0 0 + g 1 1 + g 2 2 = 18) ∧       -- main diagonal
  (g 0 2 + g 1 1 + g 2 0 = 18)         -- other diagonal

/-- The main theorem -/
theorem unique_seven (g : Grid) 
  (h1 : valid_numbers g) 
  (h2 : sum_18 g) 
  (h3 : g 0 0 = 6) 
  (h4 : g 2 2 = 1) : 
  ∃! (i j : Fin 3), g i j = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_seven_l3400_340078


namespace NUMINAMATH_CALUDE_martha_cards_total_l3400_340021

/-- Given that Martha starts with 3 cards and receives 76 more cards,
    prove that she ends up with 79 cards in total. -/
theorem martha_cards_total : 
  let initial_cards : ℕ := 3
  let received_cards : ℕ := 76
  initial_cards + received_cards = 79 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_total_l3400_340021


namespace NUMINAMATH_CALUDE_negation_abc_zero_l3400_340034

theorem negation_abc_zero (a b c : ℝ) : (a = 0 ∨ b = 0 ∨ c = 0) → a * b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_abc_zero_l3400_340034


namespace NUMINAMATH_CALUDE_weight_of_other_new_member_l3400_340068

/-- Given the initial and final average weights of a group, the number of initial members,
    and the weight of one new member, calculate the weight of the other new member. -/
theorem weight_of_other_new_member
  (initial_average : ℝ)
  (final_average : ℝ)
  (initial_members : ℕ)
  (weight_of_one_new_member : ℝ)
  (h1 : initial_average = 48)
  (h2 : final_average = 51)
  (h3 : initial_members = 23)
  (h4 : weight_of_one_new_member = 78) :
  (initial_members + 2) * final_average - initial_members * initial_average - weight_of_one_new_member = 93 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_other_new_member_l3400_340068


namespace NUMINAMATH_CALUDE_min_value_of_f_l3400_340098

/-- The function f(x) = -(x-1)³ + 12x + a - 1 -/
def f (x a : ℝ) : ℝ := -(x-1)^3 + 12*x + a - 1

/-- The interval [a, b] -/
def closed_interval (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}

theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ closed_interval (-2) 2, ∀ y ∈ closed_interval (-2) 2, f y a ≤ f x a) ∧
  (∃ x ∈ closed_interval (-2) 2, f x a = 20) →
  (∃ x ∈ closed_interval (-2) 2, f x a = -7 ∧ ∀ y ∈ closed_interval (-2) 2, -7 ≤ f y a) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3400_340098


namespace NUMINAMATH_CALUDE_chair_arrangement_count_l3400_340007

/-- The number of ways to arrange 45 chairs in a rectangular array with at least 3 chairs in each row and column -/
def rectangular_chair_arrangements : ℕ := 4

/-- The total number of chairs -/
def total_chairs : ℕ := 45

/-- The minimum number of chairs required in each row and column -/
def min_chairs_per_line : ℕ := 3

theorem chair_arrangement_count :
  ∀ (arrangement : ℕ × ℕ),
    arrangement.1 * arrangement.2 = total_chairs ∧
    arrangement.1 ≥ min_chairs_per_line ∧
    arrangement.2 ≥ min_chairs_per_line →
    rectangular_chair_arrangements = (Finset.filter
      (λ arr : ℕ × ℕ => arr.1 * arr.2 = total_chairs ∧
                        arr.1 ≥ min_chairs_per_line ∧
                        arr.2 ≥ min_chairs_per_line)
      (Finset.product (Finset.range (total_chairs + 1)) (Finset.range (total_chairs + 1)))).card :=
by sorry

end NUMINAMATH_CALUDE_chair_arrangement_count_l3400_340007


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l3400_340005

theorem triangle_tangent_ratio (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute triangle condition
  A + B + C = π →  -- Triangle angle sum
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / (2 * Real.sin A) = b / (2 * Real.sin B) →  -- Law of sines
  a / (2 * Real.sin A) = c / (2 * Real.sin C) →  -- Law of sines
  a / b + b / a = 6 * Real.cos C →  -- Given condition
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l3400_340005


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l3400_340089

theorem number_satisfying_condition : ∃ x : ℝ, (0.1 * x = 0.2 * 650 + 190) ∧ (x = 3200) := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l3400_340089


namespace NUMINAMATH_CALUDE_abs_x_squared_plus_abs_x_minus_six_roots_sum_l3400_340040

theorem abs_x_squared_plus_abs_x_minus_six_roots_sum (x : ℝ) :
  (|x|^2 + |x| - 6 = 0) → (∃ a b : ℝ, a + b = 0 ∧ |a|^2 + |a| - 6 = 0 ∧ |b|^2 + |b| - 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_squared_plus_abs_x_minus_six_roots_sum_l3400_340040


namespace NUMINAMATH_CALUDE_lucas_age_in_three_years_l3400_340006

/-- Proves Lucas's age in three years given the relationships between Gladys, Billy, and Lucas's ages -/
theorem lucas_age_in_three_years (gladys billy lucas : ℕ) : 
  gladys = 2 * (billy + lucas) → 
  gladys = 3 * billy → 
  gladys = 30 → 
  lucas + 3 = 8 := by
  sorry

#check lucas_age_in_three_years

end NUMINAMATH_CALUDE_lucas_age_in_three_years_l3400_340006


namespace NUMINAMATH_CALUDE_permutation_and_exponent_inequalities_l3400_340017

theorem permutation_and_exponent_inequalities 
  (i m n : ℕ) 
  (h1 : 1 < i) 
  (h2 : i ≤ m) 
  (h3 : m < n) : 
  n * (m.factorial / (m - i).factorial) < m * (n.factorial / (n - i).factorial) ∧ 
  (1 + m : ℝ) ^ n > (1 + n : ℝ) ^ m := by
  sorry

end NUMINAMATH_CALUDE_permutation_and_exponent_inequalities_l3400_340017


namespace NUMINAMATH_CALUDE_annual_subscription_cost_is_96_l3400_340053

/-- The cost of a monthly newspaper subscription in dollars. -/
def monthly_cost : ℝ := 10

/-- The discount rate for an annual subscription. -/
def discount_rate : ℝ := 0.2

/-- The number of months in a year. -/
def months_per_year : ℕ := 12

/-- The cost of an annual newspaper subscription with a discount. -/
def annual_subscription_cost : ℝ :=
  monthly_cost * months_per_year * (1 - discount_rate)

/-- Theorem stating that the annual subscription cost is $96. -/
theorem annual_subscription_cost_is_96 :
  annual_subscription_cost = 96 := by
  sorry


end NUMINAMATH_CALUDE_annual_subscription_cost_is_96_l3400_340053


namespace NUMINAMATH_CALUDE_game_ends_in_one_round_l3400_340045

/-- Represents a player in the game -/
inductive Player : Type
  | A | B | C | D

/-- The state of the game, containing the token count for each player -/
structure GameState :=
  (tokens : Player → Nat)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := fun p => match p with
    | Player.A => 8
    | Player.B => 9
    | Player.C => 10
    | Player.D => 11 }

/-- Determines if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p, state.tokens p = 0

/-- Determines the player with the most tokens -/
def playerWithMostTokens (state : GameState) : Player :=
  sorry

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Theorem: The game ends after 1 round -/
theorem game_ends_in_one_round :
  gameEnded (playRound initialState) :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_one_round_l3400_340045


namespace NUMINAMATH_CALUDE_range_of_f_l3400_340001

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x < 0 then x^2 + 6*x
  else 0  -- We define f as 0 outside the given intervals

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-9 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3400_340001


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3400_340016

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem unique_two_digit_number :
  ∃! n : ℕ, is_two_digit n ∧
    tens_digit n + 2 = ones_digit n ∧
    3 * (tens_digit n * ones_digit n) = n ∧
    n = 24 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3400_340016


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3400_340088

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d

-- Define a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b / a = r ∧ c / b = r

-- Theorem statement
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : geometric_sequence (a 2) (a 3) (a 6)) : 
  ∃ r : ℝ, r = 5 / 3 ∧ (a 3) / (a 2) = r ∧ (a 6) / (a 3) = r :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3400_340088


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3400_340020

theorem unique_solution_for_exponential_equation :
  ∀ a b p : ℕ+,
  p.val.Prime →
  (2 : ℕ)^(a : ℕ) + (p : ℕ)^(b : ℕ) = 19^(a : ℕ) →
  a = 1 ∧ b = 1 ∧ p = 17 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3400_340020


namespace NUMINAMATH_CALUDE_nine_fifteen_div_fifty_four_five_l3400_340092

theorem nine_fifteen_div_fifty_four_five :
  (9 : ℝ)^15 / 54^5 = 1594323 * (3 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_nine_fifteen_div_fifty_four_five_l3400_340092


namespace NUMINAMATH_CALUDE_fencing_requirement_l3400_340052

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area = uncovered_side * (area / uncovered_side) →
  uncovered_side + 2 * (area / uncovered_side) = 25 := by
  sorry

#check fencing_requirement

end NUMINAMATH_CALUDE_fencing_requirement_l3400_340052


namespace NUMINAMATH_CALUDE_percent_composition_l3400_340075

theorem percent_composition (y : ℝ) : (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_composition_l3400_340075


namespace NUMINAMATH_CALUDE_money_division_l3400_340084

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 406)
  (h_sum : a + b + c = total)
  (h_a_half_b : a = (1/2) * b)
  (h_b_half_c : b = (1/2) * c) :
  c = 232 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3400_340084


namespace NUMINAMATH_CALUDE_nine_people_four_houses_l3400_340013

-- Define the relationship between people, houses, and time
def paint_time (people : ℕ) (houses : ℕ) : ℚ :=
  let rate := (8 : ℚ) * 12 / 3  -- Rate derived from the given condition
  rate * houses / people

-- Theorem statement
theorem nine_people_four_houses :
  paint_time 9 4 = 128 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_people_four_houses_l3400_340013
