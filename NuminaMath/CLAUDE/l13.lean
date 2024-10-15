import Mathlib

namespace NUMINAMATH_CALUDE_sin_kpi_minus_x_is_odd_cos_squared_when_tan_pi_minus_x_is_two_cos_2x_plus_pi_third_symmetry_l13_1325

-- Statement 1
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sin_kpi_minus_x_is_odd (k : ℤ) :
  is_odd_function (λ x => Real.sin (k * Real.pi - x)) :=
sorry

-- Statement 2
theorem cos_squared_when_tan_pi_minus_x_is_two :
  ∀ x, Real.tan (Real.pi - x) = 2 → Real.cos x ^ 2 = 1/5 :=
sorry

-- Statement 3
def is_line_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem cos_2x_plus_pi_third_symmetry :
  is_line_of_symmetry (λ x => Real.cos (2*x + Real.pi/3)) (-2*Real.pi/3) :=
sorry

end NUMINAMATH_CALUDE_sin_kpi_minus_x_is_odd_cos_squared_when_tan_pi_minus_x_is_two_cos_2x_plus_pi_third_symmetry_l13_1325


namespace NUMINAMATH_CALUDE_simplify_fraction_l13_1378

theorem simplify_fraction : 20 * (9 / 14) * (1 / 18) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l13_1378


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_l13_1343

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The fiscal year starts on Tuesday, February 1 -/
def fiscalYearStart : Date := { day := 1, month := 2 }

/-- The day of the week for the fiscal year start -/
def fiscalYearStartDay : DayOfWeek := DayOfWeek.Tuesday

/-- Function to determine if a given date is a Terrific Tuesday -/
def isTerrificTuesday (d : Date) : Prop := sorry

/-- The first Terrific Tuesday after the fiscal year starts -/
def firstTerrificTuesday : Date := { day := 29, month := 3 }

/-- Theorem stating that the first Terrific Tuesday after the fiscal year starts is March 29 -/
theorem first_terrific_tuesday :
  isTerrificTuesday firstTerrificTuesday ∧
  (∀ d : Date, d.month < firstTerrificTuesday.month ∨ 
    (d.month = firstTerrificTuesday.month ∧ d.day < firstTerrificTuesday.day) → 
    ¬isTerrificTuesday d) :=
by sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_l13_1343


namespace NUMINAMATH_CALUDE_time_between_periods_l13_1377

theorem time_between_periods 
  (total_time : ℕ)
  (num_periods : ℕ)
  (period_duration : ℕ)
  (h1 : total_time = 220)
  (h2 : num_periods = 5)
  (h3 : period_duration = 40) :
  (total_time - num_periods * period_duration) / (num_periods - 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_time_between_periods_l13_1377


namespace NUMINAMATH_CALUDE_number_system_existence_l13_1392

/-- Represents a number in a given base --/
def BaseNumber (base : ℕ) (value : ℕ) : Prop :=
  value < base

/-- Addition in a given base --/
def BaseAdd (base : ℕ) (a b c : ℕ) : Prop :=
  BaseNumber base a ∧ BaseNumber base b ∧ BaseNumber base c ∧
  (a + b) % base = c

/-- Multiplication in a given base --/
def BaseMult (base : ℕ) (a b c : ℕ) : Prop :=
  BaseNumber base a ∧ BaseNumber base b ∧ BaseNumber base c ∧
  (a * b) % base = c

theorem number_system_existence :
  (∃ b : ℕ, BaseAdd b 3 4 10 ∧ BaseMult b 3 4 15) ∧
  (¬ ∃ b : ℕ, BaseAdd b 2 3 5 ∧ BaseMult b 2 3 11) := by
  sorry

end NUMINAMATH_CALUDE_number_system_existence_l13_1392


namespace NUMINAMATH_CALUDE_min_ratio_logarithmic_intersections_l13_1397

theorem min_ratio_logarithmic_intersections (m : ℝ) (h : m > 0) :
  let f (m : ℝ) := (2^m - 2^(8/(2*m+1))) / (2^(-m) - 2^(-8/(2*m+1)))
  ∀ x > 0, f m ≥ 8 * Real.sqrt 2 ∧ ∃ m₀ > 0, f m₀ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_logarithmic_intersections_l13_1397


namespace NUMINAMATH_CALUDE_right_triangle_arctans_l13_1306

theorem right_triangle_arctans (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_angle : a^2 = b^2 + c^2) : 
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctans_l13_1306


namespace NUMINAMATH_CALUDE_sin_300_degrees_l13_1380

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l13_1380


namespace NUMINAMATH_CALUDE_division_multiplication_order_l13_1385

theorem division_multiplication_order : (120 / 6) / 2 * 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_order_l13_1385


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l13_1323

/-- Given the cost of pencils and pens, prove the cost of one pencil and two pens -/
theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 4 * q = 3.20)
  (h2 : 2 * p + 3 * q = 2.50) :
  p + 2 * q = 1.80 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l13_1323


namespace NUMINAMATH_CALUDE_kenny_contribution_percentage_l13_1366

/-- Represents the contributions and total cost for house painting -/
structure PaintingContributions where
  total_cost : ℕ
  judson_contribution : ℕ
  kenny_contribution : ℕ
  camilo_contribution : ℕ

/-- Defines the conditions for the house painting contributions -/
def valid_contributions (c : PaintingContributions) : Prop :=
  c.total_cost = 1900 ∧
  c.judson_contribution = 500 ∧
  c.kenny_contribution > c.judson_contribution ∧
  c.camilo_contribution = c.kenny_contribution + 200 ∧
  c.judson_contribution + c.kenny_contribution + c.camilo_contribution = c.total_cost

/-- Calculates the percentage difference between Kenny's and Judson's contributions -/
def percentage_difference (c : PaintingContributions) : ℚ :=
  (c.kenny_contribution - c.judson_contribution : ℚ) / c.judson_contribution * 100

/-- Theorem stating that Kenny contributed 20% more than Judson -/
theorem kenny_contribution_percentage (c : PaintingContributions)
  (h : valid_contributions c) : percentage_difference c = 20 := by
  sorry


end NUMINAMATH_CALUDE_kenny_contribution_percentage_l13_1366


namespace NUMINAMATH_CALUDE_jean_grandchildren_l13_1324

/-- The number of cards Jean buys for each grandchild per year -/
def cards_per_grandchild : ℕ := 2

/-- The amount of money Jean puts in each card -/
def money_per_card : ℕ := 80

/-- The total amount of money Jean gives away to her grandchildren per year -/
def total_money_given : ℕ := 480

/-- The number of grandchildren Jean has -/
def num_grandchildren : ℕ := total_money_given / (cards_per_grandchild * money_per_card)

theorem jean_grandchildren :
  num_grandchildren = 3 :=
sorry

end NUMINAMATH_CALUDE_jean_grandchildren_l13_1324


namespace NUMINAMATH_CALUDE_range_of_a_for_zero_in_interval_l13_1362

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x - 3 * a - 1

-- State the theorem
theorem range_of_a_for_zero_in_interval :
  ∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ [-1, 1] ∧ f a x₀ = 0) → a ∈ [-1/2, -1/3] :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_zero_in_interval_l13_1362


namespace NUMINAMATH_CALUDE_quadratic_function_range_l13_1370

def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x - 6

theorem quadratic_function_range (a : ℝ) :
  (∃ y₁ y₂ y₃ y₄ : ℝ,
    quadratic_function a (-4) = y₁ ∧
    quadratic_function a (-3) = y₂ ∧
    quadratic_function a 0 = y₃ ∧
    quadratic_function a 2 = y₄ ∧
    (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ > 0) ∨
    (y₁ ≤ 0 ∧ y₂ > 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
    (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ > 0 ∧ y₄ ≤ 0) ∨
    (y₁ > 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0)) →
  a < -2 ∨ a > 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l13_1370


namespace NUMINAMATH_CALUDE_math_question_probability_l13_1338

/-- The probability of drawing a math question in a quiz -/
theorem math_question_probability :
  let chinese_questions : ℕ := 2
  let math_questions : ℕ := 3
  let comprehensive_questions : ℕ := 4
  let total_questions : ℕ := chinese_questions + math_questions + comprehensive_questions
  (math_questions : ℚ) / (total_questions : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_math_question_probability_l13_1338


namespace NUMINAMATH_CALUDE_parabola_intersection_l13_1308

theorem parabola_intersection :
  let f (x : ℝ) := 4 * x^2 + 3 * x - 4
  let g (x : ℝ) := 2 * x^2 + 15
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
    x₁ = -19/2 ∧ x₂ = 5/2 ∧
    f x₁ = 195.5 ∧ f x₂ = 27.5 ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l13_1308


namespace NUMINAMATH_CALUDE_eventually_one_first_l13_1340

/-- Represents a permutation of integers from 1 to 1993 -/
def Permutation := Fin 1993 → Fin 1993

/-- The reversal operation on a permutation -/
def reverseOperation (p : Permutation) : Permutation :=
  sorry

/-- Predicate to check if 1 is the first element in the permutation -/
def isOneFirst (p : Permutation) : Prop :=
  p 0 = 0

/-- Main theorem: The reversal operation will eventually make 1 the first element -/
theorem eventually_one_first (p : Permutation) : 
  ∃ n : ℕ, isOneFirst (n.iterate reverseOperation p) :=
sorry

end NUMINAMATH_CALUDE_eventually_one_first_l13_1340


namespace NUMINAMATH_CALUDE_positive_2x2_square_exists_l13_1322

/-- Represents a 50 by 50 grid of integers -/
def Grid := Fin 50 → Fin 50 → ℤ

/-- Represents a configuration G of 8 squares obtained by taking a 3 by 3 grid and removing the central square -/
def G (grid : Grid) (i j : Fin 48) : ℤ :=
  (grid i j) + (grid i (j+1)) + (grid i (j+2)) +
  (grid (i+1) j) + (grid (i+1) (j+2)) +
  (grid (i+2) j) + (grid (i+2) (j+1)) + (grid (i+2) (j+2))

/-- Represents a 2 by 2 square in the grid -/
def Square2x2 (grid : Grid) (i j : Fin 49) : ℤ :=
  (grid i j) + (grid i (j+1)) + (grid (i+1) j) + (grid (i+1) (j+1))

/-- The main theorem -/
theorem positive_2x2_square_exists (grid : Grid) 
  (h : ∀ i j : Fin 48, G grid i j > 0) :
  ∃ i j : Fin 49, Square2x2 grid i j > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_2x2_square_exists_l13_1322


namespace NUMINAMATH_CALUDE_height_difference_l13_1361

theorem height_difference (height_B : ℝ) (height_A : ℝ) 
  (h : height_A = height_B * 1.25) : 
  (height_B - height_A) / height_A * 100 = -20 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l13_1361


namespace NUMINAMATH_CALUDE_red_balls_count_l13_1368

/-- The number of red balls in a bag with given conditions -/
theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 60 →
  white = 22 →
  green = 18 →
  yellow = 8 →
  purple = 7 →
  prob_not_red_purple = 4/5 →
  (white + green + yellow : ℚ) / total = prob_not_red_purple →
  total - (white + green + yellow + purple) = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l13_1368


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l13_1383

theorem binomial_coefficient_equality (n : ℕ+) :
  (Nat.choose 9 (n + 1) = Nat.choose 9 (2 * n - 1)) → (n = 2 ∨ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l13_1383


namespace NUMINAMATH_CALUDE_saree_price_calculation_l13_1355

theorem saree_price_calculation (final_price : ℝ) 
  (h1 : final_price = 227.70) 
  (first_discount : ℝ) (second_discount : ℝ)
  (h2 : first_discount = 0.12)
  (h3 : second_discount = 0.25) : ∃ P : ℝ, 
  P * (1 - first_discount) * (1 - second_discount) = final_price ∧ P = 345 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l13_1355


namespace NUMINAMATH_CALUDE_staircase_dissection_l13_1321

/-- Represents an n-staircase polyomino -/
structure Staircase (n : ℕ+) where
  cells : Fin (n * (n + 1) / 2) → Bool

/-- Predicate to check if a staircase is valid -/
def is_valid_staircase (n : ℕ+) (s : Staircase n) : Prop :=
  ∀ i : Fin n, ∃ j : Fin (i + 1), s.cells ⟨i * (i + 1) / 2 + j, sorry⟩ = true

/-- Predicate to check if a staircase can be dissected into smaller staircases -/
def can_be_dissected (n : ℕ+) (s : Staircase n) : Prop :=
  ∃ (m : ℕ) (smaller_staircases : Fin m → Staircase n),
    (∀ i : Fin m, is_valid_staircase n (smaller_staircases i)) ∧
    (∀ i : Fin m, Staircase.cells (smaller_staircases i) ≠ s.cells) ∧
    (∀ cell : Fin (n * (n + 1) / 2), 
      s.cells cell = true ↔ ∃ i : Fin m, (smaller_staircases i).cells cell = true)

/-- Theorem stating that any n-staircase can be dissected into strictly smaller n-staircases -/
theorem staircase_dissection (n : ℕ+) (s : Staircase n) 
  (h : is_valid_staircase n s) : can_be_dissected n s :=
sorry

end NUMINAMATH_CALUDE_staircase_dissection_l13_1321


namespace NUMINAMATH_CALUDE_purely_imaginary_trajectory_l13_1301

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def trajectory (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ y

theorem purely_imaginary_trajectory (x y : ℝ) :
  is_purely_imaginary ((x^2 + y^2 - 4 : ℝ) + (x - y) * I) ↔ trajectory x y :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_trajectory_l13_1301


namespace NUMINAMATH_CALUDE_expression_simplification_l13_1341

theorem expression_simplification : 
  ((0.2 * 0.4 - (0.3 / 0.5)) + ((0.6 * 0.8 + (0.1 / 0.2)) - (0.9 * (0.3 - 0.2 * 0.4)))^2) * (1 - (0.4^2 / (0.2 * 0.8))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l13_1341


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l13_1365

/-- For a quadratic equation kx^2 + 2x - 1 = 0 to have two equal real roots, k must equal -1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0 ∧ 
   ∀ y : ℝ, k * y^2 + 2 * y - 1 = 0 → y = x) → 
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l13_1365


namespace NUMINAMATH_CALUDE_final_bird_count_and_ratio_l13_1373

/-- Represents the number of birds in the park -/
structure BirdCount where
  blackbirds : ℕ
  magpies : ℕ
  blueJays : ℕ
  robins : ℕ

/-- Calculates the initial bird count based on given conditions -/
def initialBirdCount : BirdCount :=
  { blackbirds := 3 * 7,
    magpies := 13,
    blueJays := 2 * 5,
    robins := 4 }

/-- Calculates the final bird count after changes -/
def finalBirdCount : BirdCount :=
  { blackbirds := initialBirdCount.blackbirds - 6,
    magpies := initialBirdCount.magpies + 8,
    blueJays := initialBirdCount.blueJays + 3,
    robins := initialBirdCount.robins }

/-- Calculates the total number of birds -/
def totalBirds (count : BirdCount) : ℕ :=
  count.blackbirds + count.magpies + count.blueJays + count.robins

/-- Theorem: The final number of birds is 53 and the ratio is 15:21:13:4 -/
theorem final_bird_count_and_ratio :
  totalBirds finalBirdCount = 53 ∧
  finalBirdCount.blackbirds = 15 ∧
  finalBirdCount.magpies = 21 ∧
  finalBirdCount.blueJays = 13 ∧
  finalBirdCount.robins = 4 := by
  sorry


end NUMINAMATH_CALUDE_final_bird_count_and_ratio_l13_1373


namespace NUMINAMATH_CALUDE_monotonic_iff_m_geq_one_third_l13_1345

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- State the theorem
theorem monotonic_iff_m_geq_one_third :
  ∀ m : ℝ, (∀ x : ℝ, Monotone (f m)) ↔ m ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_monotonic_iff_m_geq_one_third_l13_1345


namespace NUMINAMATH_CALUDE_function_properties_l13_1320

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : f (Real.pi / 12) φ = f (Real.pi / 4) φ) :
  (φ = Real.pi / 6) ∧ 
  (∀ x, f x φ = f (-x - Real.pi / 6) φ) ∧
  (∀ x ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 6), 
    ∀ y ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 6), 
    x < y → f x φ < f y φ) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l13_1320


namespace NUMINAMATH_CALUDE_hayley_initial_meatballs_hayley_initial_meatballs_proof_l13_1364

theorem hayley_initial_meatballs : ℕ → ℕ → ℕ → Prop :=
  fun initial_meatballs stolen_meatballs remaining_meatballs =>
    (stolen_meatballs = 14) →
    (remaining_meatballs = 11) →
    (initial_meatballs = stolen_meatballs + remaining_meatballs) →
    (initial_meatballs = 25)

-- Proof
theorem hayley_initial_meatballs_proof :
  hayley_initial_meatballs 25 14 11 := by
  sorry

end NUMINAMATH_CALUDE_hayley_initial_meatballs_hayley_initial_meatballs_proof_l13_1364


namespace NUMINAMATH_CALUDE_initial_student_count_l13_1333

/-- Given the initial average weight and the new average weight after admitting a new student,
    prove that the initial number of students is 29. -/
theorem initial_student_count
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (new_student_weight : ℝ)
  (h1 : initial_avg = 28)
  (h2 : new_avg = 27.1)
  (h3 : new_student_weight = 1)
  : ∃ n : ℕ, n = 29 ∧ 
    n * initial_avg + new_student_weight = (n + 1) * new_avg :=
by
  sorry


end NUMINAMATH_CALUDE_initial_student_count_l13_1333


namespace NUMINAMATH_CALUDE_cylinder_height_relation_l13_1302

theorem cylinder_height_relation (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relation_l13_1302


namespace NUMINAMATH_CALUDE_min_value_expression_l13_1313

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (a + 1/b) * (a + 1/b - 1024) + (b + 1/a) * (b + 1/a - 1024) = -524288 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l13_1313


namespace NUMINAMATH_CALUDE_point_on_line_segment_l13_1391

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def OnSegment (D A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D.x = A.x + t * (B.x - A.x) ∧ D.y = A.y + t * (B.y - A.y)

theorem point_on_line_segment (A B C D : Point) :
  Triangle A B C →
  A = Point.mk 1 2 →
  B = Point.mk 4 6 →
  C = Point.mk 6 3 →
  OnSegment D A B →
  D.y = (4/3) * D.x - (2/3) →
  ∃ t : ℝ, 1 ≤ t ∧ t ≤ 4 ∧ D = Point.mk t ((4/3) * t - (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_segment_l13_1391


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l13_1326

theorem cubic_equation_solutions :
  let x₁ : ℂ := 4
  let x₂ : ℂ := -2 + 2 * Complex.I * Real.sqrt 3
  let x₃ : ℂ := -2 - 2 * Complex.I * Real.sqrt 3
  (∀ x : ℂ, 2 * x^3 = 128 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l13_1326


namespace NUMINAMATH_CALUDE_function_solution_l13_1311

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x

/-- The theorem stating that the function satisfying the equation has a specific form -/
theorem function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := by
  sorry

end NUMINAMATH_CALUDE_function_solution_l13_1311


namespace NUMINAMATH_CALUDE_circle_properties_l13_1346

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 36 = -y^2 + 12*x + 16

def is_center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = (2 * Real.sqrt 23)^2

theorem circle_properties :
  ∃ a b : ℝ,
    is_center a b ∧
    a = 6 ∧
    b = 2 ∧
    a + b + 2 * Real.sqrt 23 = 8 + 2 * Real.sqrt 23 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l13_1346


namespace NUMINAMATH_CALUDE_sphere_intersection_radius_l13_1374

-- Define the sphere
def sphere_center : ℝ × ℝ × ℝ := (3, 5, -9)

-- Define the intersection circles
def xy_circle_center : ℝ × ℝ × ℝ := (3, 5, 0)
def xy_circle_radius : ℝ := 2

def xz_circle_center : ℝ × ℝ × ℝ := (0, 5, -9)

-- Theorem statement
theorem sphere_intersection_radius : 
  let s := Real.sqrt ((Real.sqrt 85)^2 - 3^2)
  s = Real.sqrt 76 :=
sorry

end NUMINAMATH_CALUDE_sphere_intersection_radius_l13_1374


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_value_l13_1386

theorem quadratic_roots_imply_a_value (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 3) * x^2 + 5 * x - 2 = 0 ↔ (x = 1/2 ∨ x = 2)) →
  (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_value_l13_1386


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l13_1359

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 6*x + 8 = 0 ∧ x ≠ 0 → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l13_1359


namespace NUMINAMATH_CALUDE_horner_method_f_3_l13_1376

/-- Horner's method for evaluating polynomials -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ :=
  horner [7, 6, 5, 4, 3, 2, 1, 0] x

theorem horner_method_f_3 :
  f 3 = 21324 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_3_l13_1376


namespace NUMINAMATH_CALUDE_mrs_hilt_pie_arrangement_l13_1309

/-- Given the number of pecan pies, apple pies, and rows, 
    calculate the number of pies in each row -/
def piesPerRow (pecanPies applePies rows : ℕ) : ℕ :=
  (pecanPies + applePies) / rows

/-- Theorem: Given 16 pecan pies, 14 apple pies, and 30 rows,
    the number of pies in each row is 1 -/
theorem mrs_hilt_pie_arrangement :
  piesPerRow 16 14 30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pie_arrangement_l13_1309


namespace NUMINAMATH_CALUDE_building_height_calculation_l13_1329

/-- Given a flagstaff and a building casting shadows under the same sun angle, 
    calculate the height of the building. -/
theorem building_height_calculation 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ) 
  (building_shadow : ℝ) 
  (h_flagstaff : flagstaff_height = 17.5) 
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  (flagstaff_height * building_shadow) / flagstaff_shadow = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_building_height_calculation_l13_1329


namespace NUMINAMATH_CALUDE_triangular_array_sum_l13_1350

theorem triangular_array_sum (N : ℕ) : 
  (N * (N + 1)) / 2 = 3003 → (N / 10 + N % 10) = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_l13_1350


namespace NUMINAMATH_CALUDE_square_difference_1989_l13_1356

theorem square_difference_1989 :
  {(a, b) : ℕ × ℕ | a > b ∧ a ^ 2 - b ^ 2 = 1989} =
  {(995, 994), (333, 330), (115, 106), (83, 70), (67, 50), (45, 6)} :=
by sorry

end NUMINAMATH_CALUDE_square_difference_1989_l13_1356


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l13_1395

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the bag of balls -/
def Bag : Multiset BallColor :=
  Multiset.replicate 3 BallColor.Red + Multiset.replicate 2 BallColor.Black

/-- Represents a draw of two balls from the bag -/
def Draw : Type := Fin 2 → BallColor

/-- The event of drawing at least one black ball -/
def AtLeastOneBlack (draw : Draw) : Prop :=
  ∃ i : Fin 2, draw i = BallColor.Black

/-- The event of drawing all red balls -/
def AllRed (draw : Draw) : Prop :=
  ∀ i : Fin 2, draw i = BallColor.Red

/-- The theorem stating that AtLeastOneBlack and AllRed are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ (draw : Draw), ¬(AtLeastOneBlack draw ∧ AllRed draw) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l13_1395


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l13_1351

theorem solution_set_of_equation (x y : ℝ) : 
  (Real.sqrt (3 * x - 1) + abs (2 * y + 2) = 0) ↔ (x = 1/3 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l13_1351


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l13_1398

theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 5*x₁ + 6 = 0 ∧ x₂^2 - 5*x₂ + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l13_1398


namespace NUMINAMATH_CALUDE_square_gt_one_vs_cube_gt_one_l13_1372

theorem square_gt_one_vs_cube_gt_one :
  {a : ℝ | a^2 > 1} ⊃ {a : ℝ | a^3 > 1} ∧ {a : ℝ | a^2 > 1} ≠ {a : ℝ | a^3 > 1} :=
by sorry

end NUMINAMATH_CALUDE_square_gt_one_vs_cube_gt_one_l13_1372


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l13_1336

theorem point_in_fourth_quadrant (α : Real) (h : -π/2 < α ∧ α < 0) :
  let P : ℝ × ℝ := (Real.tan α, Real.cos α)
  P.1 < 0 ∧ P.2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l13_1336


namespace NUMINAMATH_CALUDE_david_fewer_crunches_l13_1349

/-- Represents the number of exercises done by a person -/
structure ExerciseCount where
  pushups : ℕ
  crunches : ℕ

/-- Given the exercise counts for David and Zachary, proves that David did 17 fewer crunches than Zachary -/
theorem david_fewer_crunches (david zachary : ExerciseCount) 
  (h1 : david.pushups = zachary.pushups + 40)
  (h2 : david.crunches < zachary.crunches)
  (h3 : zachary.pushups = 34)
  (h4 : zachary.crunches = 62)
  (h5 : david.crunches = 45) :
  zachary.crunches - david.crunches = 17 := by
  sorry


end NUMINAMATH_CALUDE_david_fewer_crunches_l13_1349


namespace NUMINAMATH_CALUDE_real_part_of_z_l13_1303

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l13_1303


namespace NUMINAMATH_CALUDE_consecutive_four_product_ending_l13_1339

theorem consecutive_four_product_ending (n : ℕ) :
  ∃ (k : ℕ), (n * (n + 1) * (n + 2) * (n + 3) % 1000 = 24 ∧ k = n * (n + 1) * (n + 2) * (n + 3) / 1000) ∨
              (n * (n + 1) * (n + 2) * (n + 3) % 10 = 0 ∧ (k = n * (n + 1) * (n + 2) * (n + 3) / 10) ∧ k % 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_four_product_ending_l13_1339


namespace NUMINAMATH_CALUDE_pole_length_theorem_l13_1348

/-- The length of a pole after two cuts -/
def pole_length_after_cuts (initial_length : ℝ) (first_cut_percentage : ℝ) (second_cut_percentage : ℝ) : ℝ :=
  initial_length * (1 - first_cut_percentage) * (1 - second_cut_percentage)

/-- Theorem stating that a 20-meter pole, after cuts of 30% and 25%, will be 10.5 meters long -/
theorem pole_length_theorem :
  pole_length_after_cuts 20 0.3 0.25 = 10.5 := by
  sorry

#eval pole_length_after_cuts 20 0.3 0.25

end NUMINAMATH_CALUDE_pole_length_theorem_l13_1348


namespace NUMINAMATH_CALUDE_profit_minimum_at_radius_one_l13_1315

noncomputable def profit_function (r : ℝ) : ℝ :=
  0.2 * (4/3) * Real.pi * r^3 - 0.8 * Real.pi * r^2

theorem profit_minimum_at_radius_one :
  ∀ r : ℝ, 0 < r → r ≤ 6 →
  profit_function r ≥ profit_function 1 :=
sorry

end NUMINAMATH_CALUDE_profit_minimum_at_radius_one_l13_1315


namespace NUMINAMATH_CALUDE_min_ratio_of_circles_l13_1357

/-- The locus M of point A -/
def locus_M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ y ≠ 0

/-- Point B -/
def B : ℝ × ℝ := (-1, 0)

/-- Point C -/
def C : ℝ × ℝ := (1, 0)

/-- The area of the inscribed circle of triangle PBC -/
noncomputable def S₁ (P : ℝ × ℝ) : ℝ := sorry

/-- The area of the circumscribed circle of triangle PBC -/
noncomputable def S₂ (P : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_ratio_of_circles :
  ∀ P : ℝ × ℝ, locus_M P.1 P.2 → S₂ P / S₁ P ≥ 4 ∧ ∃ Q : ℝ × ℝ, locus_M Q.1 Q.2 ∧ S₂ Q / S₁ Q = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_of_circles_l13_1357


namespace NUMINAMATH_CALUDE_population_decrease_rate_l13_1310

theorem population_decrease_rate (initial_population : ℕ) (population_after_2_years : ℕ) 
  (h1 : initial_population = 30000)
  (h2 : population_after_2_years = 19200) :
  ∃ (r : ℝ), r = 0.2 ∧ (1 - r)^2 * initial_population = population_after_2_years :=
by sorry

end NUMINAMATH_CALUDE_population_decrease_rate_l13_1310


namespace NUMINAMATH_CALUDE_min_value_parabola_vectors_l13_1300

/-- Given a parabola y² = 2px where p > 0, prove that the minimum value of 
    |⃗OA + ⃗OB|² - |⃗AB|² for any two distinct points A and B on the parabola is -4p² -/
theorem min_value_parabola_vectors (p : ℝ) (hp : p > 0) :
  ∃ (min : ℝ), min = -4 * p^2 ∧
  ∀ (A B : ℝ × ℝ), A ≠ B →
  (A.2)^2 = 2 * p * A.1 →
  (B.2)^2 = 2 * p * B.1 →
  (A.1 + B.1)^2 + (A.2 + B.2)^2 - ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min :=
by sorry


end NUMINAMATH_CALUDE_min_value_parabola_vectors_l13_1300


namespace NUMINAMATH_CALUDE_extra_page_number_l13_1331

/-- Given a book with 77 pages, if one page number is included three times
    instead of once, resulting in a sum of 3028, then the page number
    that was added extra times is 25. -/
theorem extra_page_number :
  let n : ℕ := 77
  let correct_sum := n * (n + 1) / 2
  let incorrect_sum := 3028
  ∃ k : ℕ, k ≤ n ∧ correct_sum + 2 * k = incorrect_sum ∧ k = 25 := by
  sorry

#check extra_page_number

end NUMINAMATH_CALUDE_extra_page_number_l13_1331


namespace NUMINAMATH_CALUDE_plane_existence_and_uniqueness_l13_1317

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The first bisector plane -/
def firstBisectorPlane : Plane3D := sorry

/-- Check if a point lies in a plane -/
def pointInPlane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- First angle of projection of a plane -/
def firstProjectionAngle (plane : Plane3D) : ℝ := sorry

/-- Angle between first and second trace lines of a plane -/
def traceLinesAngle (plane : Plane3D) : ℝ := sorry

/-- Theorem: Existence and uniqueness of a plane with given properties -/
theorem plane_existence_and_uniqueness 
  (P : Point3D) 
  (α β : ℝ) 
  (h_P : pointInPlane P firstBisectorPlane) :
  ∃! s : Plane3D, 
    pointInPlane P s ∧ 
    firstProjectionAngle s = α ∧ 
    traceLinesAngle s = β := by
  sorry

end NUMINAMATH_CALUDE_plane_existence_and_uniqueness_l13_1317


namespace NUMINAMATH_CALUDE_percentage_less_than_l13_1334

theorem percentage_less_than (x y z : ℝ) :
  x = 1.2 * y →
  x = 0.84 * z →
  y = 0.7 * z :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_l13_1334


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l13_1367

theorem quadratic_roots_sum (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 3*m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l13_1367


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l13_1314

theorem complex_number_in_first_quadrant (z : ℂ) (h : z * (4 + I) = 3 + I) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l13_1314


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l13_1330

/-- The focus of a parabola y = ax^2 + c is at (0, 1/(4a) + c) -/
theorem parabola_focus (a c : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1/(4*a) + c)
  ∀ x y : ℝ, y = a * x^2 + c → (x - f.1)^2 + (y - f.2)^2 = (y - c + 1/(4*a))^2 :=
by sorry

/-- The focus of the parabola y = 9x^2 - 5 is at (0, -179/36) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, -179/36)
  ∀ x y : ℝ, y = 9 * x^2 - 5 → (x - f.1)^2 + (y - f.2)^2 = (y + 5 + 1/36)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l13_1330


namespace NUMINAMATH_CALUDE_square_area_of_adjacent_corners_l13_1358

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (5, 6)

-- Define the square area function
def squareArea (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  (dx * dx + dy * dy)

-- Theorem statement
theorem square_area_of_adjacent_corners :
  squareArea A B = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_of_adjacent_corners_l13_1358


namespace NUMINAMATH_CALUDE_prob_diff_suits_one_heart_correct_l13_1371

/-- The number of cards in a standard deck --/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- The number of cards of each suit in a standard deck --/
def cards_per_suit : ℕ := standard_deck_size / num_suits

/-- The total number of cards in two combined standard decks --/
def total_cards : ℕ := 2 * standard_deck_size

/-- The probability of selecting two cards from two combined standard 52-card decks,
    where the cards are of different suits and at least one is a heart --/
def prob_diff_suits_one_heart : ℚ := 91467 / 276044

theorem prob_diff_suits_one_heart_correct :
  let total_combinations := total_cards.choose 2
  let diff_suit_prob := (total_cards - cards_per_suit) / (total_cards - 1)
  let at_least_one_heart := total_combinations - (total_cards - 2 * cards_per_suit).choose 2
  diff_suit_prob * (at_least_one_heart / total_combinations) = prob_diff_suits_one_heart := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_suits_one_heart_correct_l13_1371


namespace NUMINAMATH_CALUDE_system_solution_l13_1388

-- Define the system of equations
def equation1 (x y p : ℝ) : Prop := (x - p)^2 = 16 * (y - 3 + p)
def equation2 (x y : ℝ) : Prop := y^2 + ((x - 3) / (|x| - 3))^2 = 1

-- Define the solution set
def is_solution (x y p : ℝ) : Prop :=
  equation1 x y p ∧ equation2 x y

-- Define the valid range for p
def valid_p (p : ℝ) : Prop :=
  (p > 3 ∧ p ≤ 4) ∨ (p > 12 ∧ p ≠ 19) ∨ (p > 19)

-- Theorem statement
theorem system_solution :
  ∀ p : ℝ, valid_p p →
    ∃ x y : ℝ, is_solution x y p ∧
      x = p + 4 * Real.sqrt (p - 3) ∧
      y = 0 :=
sorry

end NUMINAMATH_CALUDE_system_solution_l13_1388


namespace NUMINAMATH_CALUDE_gary_initial_amount_l13_1318

/-- Gary's initial amount of money -/
def initial_amount : ℕ := sorry

/-- Amount Gary spent on the snake -/
def spent_amount : ℕ := 55

/-- Amount Gary has left -/
def remaining_amount : ℕ := 18

/-- Theorem: Gary's initial amount equals the sum of spent and remaining amounts -/
theorem gary_initial_amount : initial_amount = spent_amount + remaining_amount := by sorry

end NUMINAMATH_CALUDE_gary_initial_amount_l13_1318


namespace NUMINAMATH_CALUDE_max_element_bound_l13_1399

/-- A set of 5 different positive integers -/
def IntegerSet : Type := Fin 5 → ℕ+

/-- The mean of the set is 20 -/
def hasMean20 (s : IntegerSet) : Prop :=
  (s 0 + s 1 + s 2 + s 3 + s 4 : ℚ) / 5 = 20

/-- The median of the set is 18 -/
def hasMedian18 (s : IntegerSet) : Prop :=
  s 2 = 18

/-- The elements of the set are distinct -/
def isDistinct (s : IntegerSet) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j

/-- The elements are in ascending order -/
def isAscending (s : IntegerSet) : Prop :=
  ∀ i j, i < j → s i < s j

theorem max_element_bound (s : IntegerSet) 
  (h_mean : hasMean20 s)
  (h_median : hasMedian18 s)
  (h_distinct : isDistinct s)
  (h_ascending : isAscending s) :
  s 4 ≤ 60 :=
sorry

end NUMINAMATH_CALUDE_max_element_bound_l13_1399


namespace NUMINAMATH_CALUDE_slope_of_right_triangle_l13_1332

/-- Given a right triangle ABC in the x-y plane where:
  * ∠B = 90°
  * AC = 225
  * AB = 180
  Prove that the slope of line segment AC is 4/3 -/
theorem slope_of_right_triangle (A B C : ℝ × ℝ) :
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 180^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 225^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 - (B.1 - A.1)^2 - (B.2 - A.2)^2 →
  (C.2 - A.2) / (C.1 - A.1) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_right_triangle_l13_1332


namespace NUMINAMATH_CALUDE_combined_data_mode_l13_1307

/-- Given two sets of data with specified averages, proves that the mode of the combined set is 8 -/
theorem combined_data_mode (x y : ℝ) : 
  (3 + x + 2*y + 5) / 4 = 6 →
  (x + 6 + y) / 3 = 6 →
  let combined_set := [3, x, 2*y, 5, x, 6, y]
  ∃ (mode : ℝ), mode = 8 ∧ 
    (∀ z ∈ combined_set, (combined_set.filter (λ t => t = z)).length ≤ 
                         (combined_set.filter (λ t => t = mode)).length) :=
by sorry

end NUMINAMATH_CALUDE_combined_data_mode_l13_1307


namespace NUMINAMATH_CALUDE_set_intersection_range_l13_1396

theorem set_intersection_range (a : ℝ) : 
  let A : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
  let B : Set ℝ := {x | 0 < x ∧ x < 1}
  A ∩ B = ∅ → (a ≤ -1/2 ∨ a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_range_l13_1396


namespace NUMINAMATH_CALUDE_inequality_proof_l13_1305

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_ineq : x + y + z ≥ x*y + y*z + z*x) : 
  x/(y*z) + y/(z*x) + z/(x*y) ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l13_1305


namespace NUMINAMATH_CALUDE_largest_divisible_sum_fourth_powers_l13_1352

/-- A set of n prime numbers greater than 10 -/
def PrimeSet (n : ℕ) := { S : Finset ℕ | S.card = n ∧ ∀ p ∈ S, Nat.Prime p ∧ p > 10 }

/-- The sum of fourth powers of elements in a finite set -/
def SumFourthPowers (S : Finset ℕ) : ℕ := S.sum (λ x => x^4)

/-- The main theorem statement -/
theorem largest_divisible_sum_fourth_powers :
  ∀ n > 240, ∃ S ∈ PrimeSet n, ¬ (n ∣ SumFourthPowers S) ∧
  ∀ m ≤ 240, ∀ T ∈ PrimeSet m, m ∣ SumFourthPowers T :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_sum_fourth_powers_l13_1352


namespace NUMINAMATH_CALUDE_banana_permutations_count_l13_1335

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of times 'A' appears in "BANANA" -/
def count_A : ℕ := 3

/-- The number of times 'N' appears in "BANANA" -/
def count_N : ℕ := 2

/-- Theorem stating that the number of distinct permutations of the letters in "BANANA" is 60 -/
theorem banana_permutations_count : 
  banana_permutations = (Nat.factorial total_letters) / ((Nat.factorial count_A) * (Nat.factorial count_N)) :=
by sorry

end NUMINAMATH_CALUDE_banana_permutations_count_l13_1335


namespace NUMINAMATH_CALUDE_tangent_length_to_given_circle_l13_1382

/-- The circle passing through three given points -/
structure Circle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The length of the tangent segment from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

/-- The origin point (0,0) -/
def origin : ℝ × ℝ := (0, 0)

/-- The circle passing through (4,3), (8,6), and (9,12) -/
def givenCircle : Circle :=
  { point1 := (4, 3)
    point2 := (8, 6)
    point3 := (9, 12) }

theorem tangent_length_to_given_circle :
  tangentLength origin givenCircle = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_to_given_circle_l13_1382


namespace NUMINAMATH_CALUDE_jack_son_birth_time_l13_1360

def jack_lifetime : ℝ := 84

theorem jack_son_birth_time (adolescence : ℝ) (facial_hair : ℝ) (marriage : ℝ) (son_lifetime : ℝ) :
  adolescence = jack_lifetime / 6 →
  facial_hair = jack_lifetime / 6 + jack_lifetime / 12 →
  marriage = jack_lifetime / 6 + jack_lifetime / 12 + jack_lifetime / 7 →
  son_lifetime = jack_lifetime / 2 →
  jack_lifetime - (marriage + (jack_lifetime - son_lifetime - 4)) = 5 := by
sorry

end NUMINAMATH_CALUDE_jack_son_birth_time_l13_1360


namespace NUMINAMATH_CALUDE_range_of_a_l13_1304

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem range_of_a (h : A ⊆ B a) : a ∈ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l13_1304


namespace NUMINAMATH_CALUDE_restricted_arrangements_five_students_l13_1369

/-- The number of ways to arrange n students in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a row with one specific student not in the front. -/
def arrangementsWithRestriction (n : ℕ) : ℕ := (n - 1) * arrangements (n - 1)

/-- Theorem stating that for 5 students, there are 96 ways to arrange them with one specific student not in the front. -/
theorem restricted_arrangements_five_students :
  arrangementsWithRestriction 5 = 96 := by
  sorry

#eval arrangementsWithRestriction 5  -- This should output 96

end NUMINAMATH_CALUDE_restricted_arrangements_five_students_l13_1369


namespace NUMINAMATH_CALUDE_min_photos_theorem_l13_1347

/-- Represents a photo of two children -/
structure Photo where
  child1 : Nat
  child2 : Nat
  deriving Repr

/-- The set of all possible photos -/
def AllPhotos : Set Photo := sorry

/-- The set of photos with two boys -/
def BoyBoyPhotos : Set Photo := sorry

/-- The set of photos with two girls -/
def GirlGirlPhotos : Set Photo := sorry

/-- Predicate to check if two photos are the same -/
def SamePhoto (p1 p2 : Photo) : Prop := sorry

theorem min_photos_theorem (n : Nat) (photos : Fin n → Photo) :
  (∀ i : Fin n, photos i ∈ AllPhotos) →
  (n ≥ 33) →
  (∃ i : Fin n, photos i ∈ BoyBoyPhotos) ∨
  (∃ i : Fin n, photos i ∈ GirlGirlPhotos) ∨
  (∃ i j : Fin n, i ≠ j ∧ SamePhoto (photos i) (photos j)) := by
  sorry

#check min_photos_theorem

end NUMINAMATH_CALUDE_min_photos_theorem_l13_1347


namespace NUMINAMATH_CALUDE_probability_three_quarters_l13_1379

/-- A diamond-shaped checkerboard formed by an 8x8 grid -/
structure DiamondCheckerboard where
  total_squares : ℕ
  squares_per_vertex : ℕ
  num_vertices : ℕ

/-- The probability that a randomly chosen unit square does not touch a vertex of the diamond -/
def probability_not_touching_vertex (board : DiamondCheckerboard) : ℚ :=
  1 - (board.squares_per_vertex * board.num_vertices : ℚ) / board.total_squares

/-- Theorem stating that the probability of not touching a vertex is 3/4 -/
theorem probability_three_quarters (board : DiamondCheckerboard) 
  (h1 : board.total_squares = 64)
  (h2 : board.squares_per_vertex = 4)
  (h3 : board.num_vertices = 4) : 
  probability_not_touching_vertex board = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_quarters_l13_1379


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l13_1328

theorem consecutive_integers_sum (n : ℕ) (h1 : n > 0) 
  (h2 : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070) :
  n + 5 = 347 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l13_1328


namespace NUMINAMATH_CALUDE_election_win_condition_l13_1342

theorem election_win_condition 
  (total_students : ℕ) 
  (boy_percentage : ℚ) 
  (girl_percentage : ℚ) 
  (male_vote_percentage : ℚ) 
  (h1 : total_students = 200)
  (h2 : boy_percentage = 3/5)
  (h3 : girl_percentage = 2/5)
  (h4 : boy_percentage + girl_percentage = 1)
  (h5 : male_vote_percentage = 27/40)
  : ∃ (female_vote_percentage : ℚ),
    female_vote_percentage ≥ 1/4 ∧
    (boy_percentage * male_vote_percentage + girl_percentage * female_vote_percentage) * total_students > total_students / 2 ∧
    ∀ (x : ℚ), x < female_vote_percentage →
      (boy_percentage * male_vote_percentage + girl_percentage * x) * total_students ≤ total_students / 2 :=
by sorry

end NUMINAMATH_CALUDE_election_win_condition_l13_1342


namespace NUMINAMATH_CALUDE_min_value_theorem_l13_1353

theorem min_value_theorem (a b c d : ℝ) 
  (h : |b - Real.log a / a| + |c - d + 2| = 0) : 
  ∃ (min_val : ℝ), min_val = 9/2 ∧ 
    ∀ (x y : ℝ), (x - y)^2 + (Real.log x / x - (y + 2))^2 ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l13_1353


namespace NUMINAMATH_CALUDE_dance_step_time_ratio_l13_1327

/-- Proves that the ratio of time spent on the third dance step to the combined time
    spent on the first and second steps is 1:1, given the specified conditions. -/
theorem dance_step_time_ratio :
  ∀ (time_step1 time_step2 time_step3 total_time : ℕ),
  time_step1 = 30 →
  time_step2 = time_step1 / 2 →
  total_time = 90 →
  total_time = time_step1 + time_step2 + time_step3 →
  time_step3 = time_step1 + time_step2 :=
by
  sorry

#check dance_step_time_ratio

end NUMINAMATH_CALUDE_dance_step_time_ratio_l13_1327


namespace NUMINAMATH_CALUDE_odd_function_implies_a_eq_two_l13_1375

/-- The function f(x) = (x + a - 2)(2x² + a - 1) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a - 2) * (2 * x^2 + a - 1)

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_implies_a_eq_two :
  ∀ a : ℝ, is_odd_function (f a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_eq_two_l13_1375


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l13_1384

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 3 * x₁ - 5 = 0) →
  (2 * x₂^2 + 3 * x₂ - 5 = 0) →
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 29/4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l13_1384


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l13_1394

open Set

def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

theorem union_of_M_and_N :
  M ∪ N = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l13_1394


namespace NUMINAMATH_CALUDE_odd_divisors_of_180_l13_1319

/-- The number of positive divisors of 180 that are not divisible by 2 -/
def count_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ ¬ 2 ∣ d) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 180 not divisible by 2 is 6 -/
theorem odd_divisors_of_180 : count_odd_divisors 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_of_180_l13_1319


namespace NUMINAMATH_CALUDE_solve_for_a_l13_1389

theorem solve_for_a (x a : ℚ) (h1 : x - 2 * a + 5 = 0) (h2 : x = -2) : a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l13_1389


namespace NUMINAMATH_CALUDE_g_shifted_l13_1316

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem statement
theorem g_shifted (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_g_shifted_l13_1316


namespace NUMINAMATH_CALUDE_metro_line_stations_l13_1381

theorem metro_line_stations (x : ℕ) (h : x * (x - 1) = 1482) :
  x * (x - 1) = 1482 ∧ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_metro_line_stations_l13_1381


namespace NUMINAMATH_CALUDE_pencil_distribution_l13_1387

/-- Given an initial number of pencils, number of containers, and additional pencils,
    calculate the number of pencils that can be evenly distributed per container. -/
def evenlyDistributedPencils (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) : ℕ :=
  (initialPencils + additionalPencils) / containers

theorem pencil_distribution (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) 
    (h1 : initialPencils = 150)
    (h2 : containers = 5)
    (h3 : additionalPencils = 30) :
  evenlyDistributedPencils initialPencils containers additionalPencils = 36 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l13_1387


namespace NUMINAMATH_CALUDE_billiard_ball_trajectory_l13_1354

theorem billiard_ball_trajectory :
  ∀ (x y : ℚ),
    (x ≥ 0 ∧ y ≥ 0) →  -- Restricting to first quadrant
    (y = x / Real.sqrt 2) →  -- Line equation
    (¬ ∃ (m n : ℤ), (x = ↑m ∧ y = ↑n)) :=  -- No integer coordinate intersection
by sorry

end NUMINAMATH_CALUDE_billiard_ball_trajectory_l13_1354


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l13_1344

/-- Given a line in vector form, prove it's equivalent to a specific slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), (-2 : ℝ) * (x - 5) + 4 * (y + 6) = 0 ↔ y = (1/2 : ℝ) * x - (17/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l13_1344


namespace NUMINAMATH_CALUDE_circle_center_coordinate_product_l13_1393

/-- Given two points as endpoints of a circle's diameter, 
    calculate the product of the coordinates of the circle's center -/
theorem circle_center_coordinate_product 
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) 
  (h1 : p1 = (7, -8)) 
  (h2 : p2 = (-2, 3)) : 
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (center.1 * center.2) = -25/4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_product_l13_1393


namespace NUMINAMATH_CALUDE_compound_interest_problem_l13_1363

theorem compound_interest_problem :
  ∃ (P r : ℝ), P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8840 ∧
  P * (1 + r)^3 = 9261 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l13_1363


namespace NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l13_1390

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_to_parallel_planes 
  (l : Line) (α β : Plane) :
  perp l β → para α β → perp l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l13_1390


namespace NUMINAMATH_CALUDE_square_addition_l13_1337

theorem square_addition (b : ℝ) : b^2 + b^2 = 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_addition_l13_1337


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l13_1312

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_property (n : ℕ) : Prop :=
  is_two_digit n ∧ (n + reverse_digits n) % 13 = 0

theorem count_numbers_with_property :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_property n) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l13_1312
