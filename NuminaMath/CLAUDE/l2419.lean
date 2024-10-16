import Mathlib

namespace NUMINAMATH_CALUDE_seven_twelfths_decimal_l2419_241947

theorem seven_twelfths_decimal : 
  (7 : ℚ) / 12 = 0.5833333333333333 := by sorry

end NUMINAMATH_CALUDE_seven_twelfths_decimal_l2419_241947


namespace NUMINAMATH_CALUDE_parabola_directrix_l2419_241922

/-- The equation of the directrix of the parabola y = x^2 is y = -1/4 -/
theorem parabola_directrix : ∃ (k : ℝ), k = -1/4 ∧
  ∀ (x y : ℝ), y = x^2 → (x = 0 ∨ (x^2 + (y - k)^2) / (2 * (y - k)) = k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2419_241922


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2419_241969

/-- An ellipse with foci at (4, -4 + 3√2) and (4, -4 - 3√2) -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  h1 : focus1 = (4, -4 + 3 * Real.sqrt 2)
  h2 : focus2 = (4, -4 - 3 * Real.sqrt 2)

/-- The ellipse is tangent to both x-axis and y-axis -/
def is_tangent_to_axes (e : Ellipse) : Prop := sorry

/-- The length of the major axis of the ellipse -/
def major_axis_length (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 8 -/
theorem ellipse_major_axis_length (e : Ellipse) (h : is_tangent_to_axes e) : 
  major_axis_length e = 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2419_241969


namespace NUMINAMATH_CALUDE_reading_time_calculation_l2419_241904

def total_time : ℕ := 120
def piano_time : ℕ := 30
def writing_time : ℕ := 25
def exerciser_time : ℕ := 27

theorem reading_time_calculation :
  total_time - piano_time - writing_time - exerciser_time = 38 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l2419_241904


namespace NUMINAMATH_CALUDE_max_digit_sum_is_37_l2419_241983

/-- Represents a two-digit display --/
structure TwoDigitDisplay where
  tens : Nat
  ones : Nat
  valid : tens ≤ 9 ∧ ones ≤ 9

/-- Represents a time display in 12-hour format --/
structure TimeDisplay where
  hours : TwoDigitDisplay
  minutes : TwoDigitDisplay
  seconds : TwoDigitDisplay
  valid_hours : hours.tens * 10 + hours.ones ≥ 1 ∧ hours.tens * 10 + hours.ones ≤ 12
  valid_minutes : minutes.tens * 10 + minutes.ones ≤ 59
  valid_seconds : seconds.tens * 10 + seconds.ones ≤ 59

/-- Calculates the sum of digits in a TwoDigitDisplay --/
def digitSum (d : TwoDigitDisplay) : Nat :=
  d.tens + d.ones

/-- Calculates the total sum of digits in a TimeDisplay --/
def totalDigitSum (t : TimeDisplay) : Nat :=
  digitSum t.hours + digitSum t.minutes + digitSum t.seconds

/-- The maximum possible sum of digits in a 12-hour format digital watch display --/
def maxDigitSum : Nat := 37

/-- Theorem: The maximum sum of digits in a 12-hour format digital watch display is 37 --/
theorem max_digit_sum_is_37 :
  ∀ t : TimeDisplay, totalDigitSum t ≤ maxDigitSum :=
by
  sorry  -- The proof would go here

#check max_digit_sum_is_37

end NUMINAMATH_CALUDE_max_digit_sum_is_37_l2419_241983


namespace NUMINAMATH_CALUDE_simplify_expression_l2419_241915

theorem simplify_expression (x y z : ℝ) : (x - (y - z)) - ((x - y) - z) = 2 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2419_241915


namespace NUMINAMATH_CALUDE_commodity_price_difference_l2419_241920

theorem commodity_price_difference (total_cost first_price : ℕ) 
  (h1 : total_cost = 827)
  (h2 : first_price = 477)
  (h3 : first_price > total_cost - first_price) : 
  first_price - (total_cost - first_price) = 127 := by
  sorry

end NUMINAMATH_CALUDE_commodity_price_difference_l2419_241920


namespace NUMINAMATH_CALUDE_scott_total_earnings_l2419_241971

/-- 
Proves that the total money Scott made from selling smoothies and cakes is $156, 
given the prices and quantities of items sold.
-/
theorem scott_total_earnings : 
  let smoothie_price : ℕ := 3
  let cake_price : ℕ := 2
  let smoothies_sold : ℕ := 40
  let cakes_sold : ℕ := 18
  
  smoothie_price * smoothies_sold + cake_price * cakes_sold = 156 := by
  sorry

end NUMINAMATH_CALUDE_scott_total_earnings_l2419_241971


namespace NUMINAMATH_CALUDE_part_one_part_two_l2419_241938

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Part I
theorem part_one : ∃ (m n : ℝ), a = (m • b.1 + n • c.1, m • b.2 + n • c.2) := by sorry

-- Part II
theorem part_two : 
  ∃ (d : ℝ × ℝ), 
    (∃ (k : ℝ), (d.1 - c.1, d.2 - c.2) = k • (a.1 + b.1, a.2 + b.2)) ∧ 
    (d.1 - c.1)^2 + (d.2 - c.2)^2 = 5 ∧
    (d = (3, -1) ∨ d = (5, 3)) := by sorry


end NUMINAMATH_CALUDE_part_one_part_two_l2419_241938


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2419_241980

/-- The number of ways to arrange 9 distinct objects in a row with specific conditions -/
def arrangement_count : ℕ := 2880

/-- The total number of objects -/
def total_objects : ℕ := 9

/-- The number of objects that must be at the ends -/
def end_objects : ℕ := 2

/-- The number of objects that must be adjacent -/
def adjacent_objects : ℕ := 2

/-- The number of remaining objects -/
def remaining_objects : ℕ := total_objects - end_objects - adjacent_objects

theorem arrangement_theorem :
  arrangement_count = 
    2 * -- ways to arrange end objects
    (remaining_objects + 1) * -- ways to place adjacent objects
    2 * -- ways to arrange adjacent objects
    remaining_objects! -- ways to arrange remaining objects
  := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2419_241980


namespace NUMINAMATH_CALUDE_tank_length_proof_l2419_241973

/-- Proves that a tank with given dimensions and plastering cost has a specific length -/
theorem tank_length_proof (width depth L : ℝ) (plastering_rate : ℝ) (total_cost : ℝ) : 
  width = 12 →
  depth = 6 →
  plastering_rate = 75 / 100 →
  total_cost = 558 →
  (2 * depth * L + 2 * depth * width + width * L) * plastering_rate = total_cost →
  L = 25 := by
sorry


end NUMINAMATH_CALUDE_tank_length_proof_l2419_241973


namespace NUMINAMATH_CALUDE_symmetric_functions_properties_l2419_241979

/-- Given a > 1, f(x) is symmetric to g(x) = 4 - a^|x-2| - 2*a^(x-2) w.r.t (1, 2) -/
def SymmetricFunctions (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a > 1 ∧ ∀ x y, f x = y ↔ 4 - a^|2-x| - 2*a^(2-x) = 4 - y

theorem symmetric_functions_properties {a : ℝ} {f : ℝ → ℝ} 
  (h : SymmetricFunctions a f) :
  (∀ x, f x = a^|x| + 2*a^(-x)) ∧ 
  (∀ m, (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f x₁ = m ∧ f x₂ = m) ↔ 
    2*(2:ℝ)^(1/2) < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_functions_properties_l2419_241979


namespace NUMINAMATH_CALUDE_product_of_four_integers_l2419_241943

theorem product_of_four_integers (P Q R S : ℕ+) : 
  P + Q + R + S = 100 →
  (P : ℚ) + 5 = (Q : ℚ) - 5 →
  (P : ℚ) + 5 = (R : ℚ) * 2 →
  (P : ℚ) + 5 = (S : ℚ) / 2 →
  (P : ℚ) * (Q : ℚ) * (R : ℚ) * (S : ℚ) = 1509400000 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_integers_l2419_241943


namespace NUMINAMATH_CALUDE_missing_chess_pieces_l2419_241940

theorem missing_chess_pieces (total_pieces : Nat) (present_pieces : Nat) : 
  total_pieces = 32 → present_pieces = 22 → total_pieces - present_pieces = 10 := by
  sorry

end NUMINAMATH_CALUDE_missing_chess_pieces_l2419_241940


namespace NUMINAMATH_CALUDE_phi_value_l2419_241955

theorem phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin φ - Real.cos φ) : 
  φ = 25 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l2419_241955


namespace NUMINAMATH_CALUDE_average_height_problem_l2419_241995

/-- Given a class of girls with specific average heights, prove the average height of a subgroup -/
theorem average_height_problem (total_girls : ℕ) (subgroup_girls : ℕ) (remaining_girls : ℕ)
  (subgroup_avg_height : ℝ) (remaining_avg_height : ℝ) (total_avg_height : ℝ)
  (h1 : total_girls = subgroup_girls + remaining_girls)
  (h2 : total_girls = 40)
  (h3 : subgroup_girls = 30)
  (h4 : remaining_avg_height = 156)
  (h5 : total_avg_height = 159) :
  subgroup_avg_height = 160 := by
sorry


end NUMINAMATH_CALUDE_average_height_problem_l2419_241995


namespace NUMINAMATH_CALUDE_total_toys_l2419_241952

theorem total_toys (num_dolls : ℕ) (h1 : num_dolls = 18) : ℕ :=
  let total := 4 * num_dolls / 3
  have h2 : total = 24 := by sorry
  total

#check total_toys

end NUMINAMATH_CALUDE_total_toys_l2419_241952


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2419_241976

theorem smaller_number_in_ratio (x y d : ℝ) : 
  x > 0 → y > 0 → x / y = 2 / 3 → 2 * x + 3 * y = d → min x y = 2 * d / 13 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2419_241976


namespace NUMINAMATH_CALUDE_expression_value_l2419_241916

theorem expression_value : 
  let x : ℝ := 2
  2 * x^2 + 3 * x^2 = 20 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2419_241916


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l2419_241981

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, a^2 * x^2 + a * x - 1 = 0 ∧ x^2 - a * x - a^2 = 0) →
  (a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
   a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l2419_241981


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2419_241908

theorem simple_interest_problem (P r : ℝ) 
  (h1 : P * (1 + 0.02 * r) = 600)
  (h2 : P * (1 + 0.07 * r) = 850) : 
  P = 500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2419_241908


namespace NUMINAMATH_CALUDE_characterize_N_l2419_241991

def StrictlyIncreasing (s : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < j → s i < s j

def IsPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ ∀ i : ℕ, a (i + m) = a i

def SatisfiesConditions (s : ℕ → ℕ) (N : ℕ) : Prop :=
  StrictlyIncreasing s ∧
  IsPeriodic (fun i => s (i + 1) - s i) ∧
  ∀ n : ℕ, n > 0 → s (s n) - s (s (n - 1)) ≤ N ∧ N < s (1 + s n) - s (s (n - 1))

theorem characterize_N :
  ∀ N : ℕ, (∃ s : ℕ → ℕ, SatisfiesConditions s N) ↔
    (∃ k : ℕ, k > 0 ∧ k^2 ≤ N ∧ N < k^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_characterize_N_l2419_241991


namespace NUMINAMATH_CALUDE_product_of_ratios_l2419_241998

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2030) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2029)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2030) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2029)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2030) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2029)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/1015 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l2419_241998


namespace NUMINAMATH_CALUDE_circle_symmetry_l2419_241957

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  original_circle x y ∧ symmetry_line x y →
  ∃ (x' y' : ℝ), symmetric_circle x' y' :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2419_241957


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2419_241963

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → abs a = 1) ∧ ¬(abs a = 1 → a = 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2419_241963


namespace NUMINAMATH_CALUDE_xyz_value_l2419_241946

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8) :
  x * y * z = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2419_241946


namespace NUMINAMATH_CALUDE_total_guppies_l2419_241974

/-- The number of guppies owned by each person -/
structure GuppyOwners where
  haylee : ℕ
  jose : ℕ
  charliz : ℕ
  nicolai : ℕ
  alice : ℕ
  bob : ℕ
  cameron : ℕ

/-- The conditions of guppy ownership as described in the problem -/
def guppy_conditions (g : GuppyOwners) : Prop :=
  g.haylee = 3 * 12 ∧
  g.jose = g.haylee / 2 ∧
  g.charliz = g.jose / 3 ∧
  g.nicolai = 4 * g.charliz ∧
  g.alice = g.nicolai + 5 ∧
  g.bob = (g.jose + g.charliz) / 2 ∧
  g.cameron = 2^3

/-- The theorem stating that the total number of guppies is 133 -/
theorem total_guppies (g : GuppyOwners) (h : guppy_conditions g) :
  g.haylee + g.jose + g.charliz + g.nicolai + g.alice + g.bob + g.cameron = 133 := by
  sorry


end NUMINAMATH_CALUDE_total_guppies_l2419_241974


namespace NUMINAMATH_CALUDE_expected_digits_is_nineteen_twelfths_l2419_241909

/-- Die numbers -/
def die_numbers : List ℕ := List.range 12 |>.map (· + 5)

/-- Count of digits in a number -/
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- Expected value calculation -/
def expected_digits : ℚ :=
  (die_numbers.map digit_count).sum / die_numbers.length

/-- Theorem: Expected number of digits is 19/12 -/
theorem expected_digits_is_nineteen_twelfths :
  expected_digits = 19 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_nineteen_twelfths_l2419_241909


namespace NUMINAMATH_CALUDE_max_value_of_function_l2419_241986

theorem max_value_of_function (x : ℝ) : 
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x) ≤ Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2419_241986


namespace NUMINAMATH_CALUDE_problem_solution_l2419_241918

def problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) : Prop :=
  (∀ x, g_inv (g x) = x) ∧
  (∀ y, g (g_inv y) = y) ∧
  g 4 = 6 ∧
  g 6 = 2 ∧
  g 3 = 7 ∧
  g_inv (g_inv 6 + g_inv 7) = 3

theorem problem_solution :
  ∃ (g : ℝ → ℝ) (g_inv : ℝ → ℝ), problem g g_inv :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2419_241918


namespace NUMINAMATH_CALUDE_rectangular_field_posts_l2419_241903

/-- Calculates the number of posts needed for a rectangular fence -/
def num_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let num_sections := perimeter / post_spacing
  num_sections

theorem rectangular_field_posts :
  num_posts 6 8 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_posts_l2419_241903


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2419_241945

theorem arithmetic_square_root_of_16 :
  Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2419_241945


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l2419_241927

/-- Proves that the cost of each bar of chocolate is $5 given the conditions of the problem. -/
theorem chocolate_bar_cost
  (num_bars : ℕ)
  (total_selling_price : ℚ)
  (packaging_cost_per_bar : ℚ)
  (total_profit : ℚ)
  (h1 : num_bars = 5)
  (h2 : total_selling_price = 90)
  (h3 : packaging_cost_per_bar = 2)
  (h4 : total_profit = 55) :
  ∃ (cost_per_bar : ℚ), cost_per_bar = 5 ∧
    total_selling_price = num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + total_profit :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l2419_241927


namespace NUMINAMATH_CALUDE_equilateral_triangle_inscribed_circle_radius_l2419_241905

/-- Given an equilateral triangle inscribed in a circle with area 81 cm²,
    prove that the radius of the circle is 6 * (3^(1/4)) cm. -/
theorem equilateral_triangle_inscribed_circle_radius 
  (S : ℝ) (r : ℝ) (h1 : S = 81) :
  r = 6 * (3 : ℝ)^(1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_inscribed_circle_radius_l2419_241905


namespace NUMINAMATH_CALUDE_cuboid_volume_l2419_241921

/-- The volume of a cuboid with edges 2 cm, 5 cm, and 3 cm is 30 cubic centimeters. -/
theorem cuboid_volume : 
  ∀ (length width height : ℝ), 
    length = 2 → width = 5 → height = 3 → 
    length * width * height = 30 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l2419_241921


namespace NUMINAMATH_CALUDE_age_system_properties_l2419_241972

/-- Represents the ages and aging rates of four people -/
structure AgeSystem where
  a : ℝ  -- Age of person A
  b : ℝ  -- Age of person B
  c : ℝ  -- Age of person C
  d : ℝ  -- Age of person D
  x : ℝ  -- Age difference between A and C
  y : ℝ  -- Number of years passed
  rA : ℝ  -- Aging rate of A relative to C
  rB : ℝ  -- Aging rate of B relative to C
  rD : ℝ  -- Aging rate of D relative to C

/-- The age system satisfies the given conditions -/
def satisfiesConditions (s : AgeSystem) : Prop :=
  s.a + s.b = 13 + (s.b + s.c) ∧
  s.c = s.a - s.x ∧
  s.a + s.d = 2 * (s.b + s.c) ∧
  (s.a + s.rA * s.y) + (s.b + s.rB * s.y) = 25 + (s.b + s.rB * s.y) + (s.c + s.y)

/-- Theorem stating the properties of the age system -/
theorem age_system_properties (s : AgeSystem) 
  (h : satisfiesConditions s) : 
  s.x = 13 ∧ s.d = 2 * s.b + s.a - 26 ∧ s.rA * s.y = 12 + s.y := by
  sorry


end NUMINAMATH_CALUDE_age_system_properties_l2419_241972


namespace NUMINAMATH_CALUDE_pencils_given_l2419_241949

theorem pencils_given (initial_pencils total_pencils : ℕ) 
  (h1 : initial_pencils = 9)
  (h2 : total_pencils = 65) :
  total_pencils - initial_pencils = 56 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_l2419_241949


namespace NUMINAMATH_CALUDE_multiplier_value_l2419_241926

theorem multiplier_value (x n : ℚ) : 
  x = 40 → (x / 4) * n + 10 - 12 = 48 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l2419_241926


namespace NUMINAMATH_CALUDE_joyful_joan_calculation_l2419_241977

theorem joyful_joan_calculation (a b c d e : ℚ) : 
  a = 2 → b = 3 → c = 4 → d = 5 →
  (a + b + c + d + e = a + (b + (c - (d * e)))) →
  e = -5/6 := by sorry

end NUMINAMATH_CALUDE_joyful_joan_calculation_l2419_241977


namespace NUMINAMATH_CALUDE_clothes_washing_time_l2419_241930

/-- Represents the time in minutes for washing different types of laundry -/
structure LaundryTime where
  clothes : ℕ
  towels : ℕ
  sheets : ℕ

/-- Defines the conditions for the laundry washing problem -/
def valid_laundry_time (t : LaundryTime) : Prop :=
  t.towels = 2 * t.clothes ∧
  t.sheets = t.towels - 15 ∧
  t.clothes + t.towels + t.sheets = 135

/-- Theorem stating that the time to wash clothes is 30 minutes -/
theorem clothes_washing_time (t : LaundryTime) :
  valid_laundry_time t → t.clothes = 30 := by
  sorry

end NUMINAMATH_CALUDE_clothes_washing_time_l2419_241930


namespace NUMINAMATH_CALUDE_survey_participants_survey_participants_proof_l2419_241941

theorem survey_participants (total_participants : ℕ) 
  (first_myth_percentage : ℚ) 
  (second_myth_percentage : ℚ) 
  (both_myths_count : ℕ) : Prop :=
  first_myth_percentage = 923 / 1000 ∧ 
  second_myth_percentage = 382 / 1000 ∧
  both_myths_count = 29 →
  total_participants = 83

-- The proof of the theorem
theorem survey_participants_proof : 
  ∃ (total_participants : ℕ), 
    survey_participants total_participants (923 / 1000) (382 / 1000) 29 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_participants_survey_participants_proof_l2419_241941


namespace NUMINAMATH_CALUDE_plugged_handle_pressure_l2419_241924

/-- The gauge pressure at the bottom of a jug with a plugged handle -/
theorem plugged_handle_pressure
  (ρ g h H P : ℝ)
  (h_pos : h > 0)
  (H_pos : H > 0)
  (H_gt_h : H > h)
  (ρ_pos : ρ > 0)
  (g_pos : g > 0) :
  ρ * g * H < P ∧ P < ρ * g * h :=
sorry

end NUMINAMATH_CALUDE_plugged_handle_pressure_l2419_241924


namespace NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_five_l2419_241985

def is_abundant (n : ℕ) : Prop :=
  n > 0 ∧ (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id > n)

def is_multiple_of_five (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_abundant_not_multiple_of_five : 
  (∀ k : ℕ, k < 12 → ¬(is_abundant k ∧ ¬is_multiple_of_five k)) ∧ 
  (is_abundant 12 ∧ ¬is_multiple_of_five 12) := by
  sorry

end NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_five_l2419_241985


namespace NUMINAMATH_CALUDE_sphere_radius_from_perpendicular_chords_l2419_241992

/-- Given a sphere with three mutually perpendicular chords APB, CPD, and EPF passing through
    a common point P, where AP = 2a, BP = 2b, CP = 2c, DP = 2d, EP = 2e, and FP = 2f,
    the radius R of the sphere is √(a² + b² + c² + d² + e² + f² - 2ab - 2cd - 2ef). -/
theorem sphere_radius_from_perpendicular_chords
  (a b c d e f : ℝ) : ∃ (R : ℝ),
  R = Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 - 2*a*b - 2*c*d - 2*e*f) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_perpendicular_chords_l2419_241992


namespace NUMINAMATH_CALUDE_pascal_high_school_students_l2419_241906

/-- The number of students at Pascal High School -/
def total_students : ℕ := sorry

/-- The number of students who went on the first trip -/
def first_trip : ℕ := sorry

/-- The number of students who went on the second trip -/
def second_trip : ℕ := sorry

/-- The number of students who went on the third trip -/
def third_trip : ℕ := sorry

/-- The number of students who went on all three trips -/
def all_three_trips : ℕ := 160

theorem pascal_high_school_students :
  (first_trip = total_students / 2) ∧
  (second_trip = (total_students * 4) / 5) ∧
  (third_trip = (total_students * 9) / 10) ∧
  (all_three_trips = 160) ∧
  (∀ s, s ∈ Finset.range total_students →
    (s ∈ Finset.range first_trip ∧ s ∈ Finset.range second_trip) ∨
    (s ∈ Finset.range first_trip ∧ s ∈ Finset.range third_trip) ∨
    (s ∈ Finset.range second_trip ∧ s ∈ Finset.range third_trip) ∨
    (s ∈ Finset.range all_three_trips)) →
  total_students = 800 := by sorry

end NUMINAMATH_CALUDE_pascal_high_school_students_l2419_241906


namespace NUMINAMATH_CALUDE_m_range_l2419_241939

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the range of m
def range_m (m : ℝ) : Prop := m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3

-- Theorem statement
theorem m_range : 
  ∀ m : ℝ, (¬(P m ∧ Q m) ∧ (P m ∨ Q m)) → range_m m :=
sorry

end NUMINAMATH_CALUDE_m_range_l2419_241939


namespace NUMINAMATH_CALUDE_monogram_count_is_66_l2419_241951

/-- The number of letters available for the first two initials -/
def n : ℕ := 12

/-- The number of initials to choose (first and middle) -/
def k : ℕ := 2

/-- The number of ways to choose k distinct letters from n letters in alphabetical order -/
def monogram_count : ℕ := Nat.choose n k

/-- Theorem stating that the number of possible monograms is 66 -/
theorem monogram_count_is_66 : monogram_count = 66 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_is_66_l2419_241951


namespace NUMINAMATH_CALUDE_albert_track_runs_l2419_241990

theorem albert_track_runs : 
  ∀ (total_distance track_length additional_laps current_laps : ℕ),
    total_distance = 99 →
    track_length = 9 →
    additional_laps = 5 →
    total_distance = track_length * (current_laps + additional_laps) →
    current_laps = 6 := by
  sorry

end NUMINAMATH_CALUDE_albert_track_runs_l2419_241990


namespace NUMINAMATH_CALUDE_line_slope_l2419_241954

/-- The slope of the line x + √3y + 2 = 0 is -1/√3 -/
theorem line_slope (x y : ℝ) : x + Real.sqrt 3 * y + 2 = 0 → 
  (y - (-2/Real.sqrt 3)) / (x - 0) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2419_241954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2419_241978

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = -2) 
  (h_7 : a 7 = -10) : 
  ∀ n : ℕ, n > 0 → a n = -2 * n + 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2419_241978


namespace NUMINAMATH_CALUDE_parabola_vertex_l2419_241956

/-- A quadratic function f(x) = -x^2 + ax + b where f(x) ≤ 0 
    has the solution (-∞,-3] ∪ [5,∞) -/
def f (a b x : ℝ) : ℝ := -x^2 + a*x + b

/-- The solution set of f(x) ≤ 0 -/
def solution_set (a b : ℝ) : Set ℝ :=
  {x | x ≤ -3 ∨ x ≥ 5}

/-- The vertex of the parabola -/
def vertex (a b : ℝ) : ℝ × ℝ := (1, 16)

theorem parabola_vertex (a b : ℝ) 
  (h : ∀ x, f a b x ≤ 0 ↔ x ∈ solution_set a b) :
  vertex a b = (1, 16) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2419_241956


namespace NUMINAMATH_CALUDE_area_to_paint_l2419_241999

-- Define the wall dimensions
def wall_height : ℝ := 10
def wall_width : ℝ := 15

-- Define the unpainted area dimensions
def unpainted_height : ℝ := 3
def unpainted_width : ℝ := 5

-- Theorem to prove
theorem area_to_paint : 
  wall_height * wall_width - unpainted_height * unpainted_width = 135 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_l2419_241999


namespace NUMINAMATH_CALUDE_sum_of_triangles_is_26_l2419_241958

-- Define the triangle operation
def triangleOp (a b c : ℚ) : ℚ := a * b / c

-- Define the sum of two triangle operations
def sumTriangleOps (a1 b1 c1 a2 b2 c2 : ℚ) : ℚ :=
  triangleOp a1 b1 c1 + triangleOp a2 b2 c2

-- Theorem statement
theorem sum_of_triangles_is_26 :
  sumTriangleOps 4 8 2 5 10 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_is_26_l2419_241958


namespace NUMINAMATH_CALUDE_university_theater_ticket_sales_l2419_241993

theorem university_theater_ticket_sales 
  (total_tickets : ℕ) 
  (adult_price senior_price : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_university_theater_ticket_sales_l2419_241993


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2419_241935

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 7 - a 3 = 20) :
  a 2008 - a 2000 = 40 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2419_241935


namespace NUMINAMATH_CALUDE_dandelion_picking_average_l2419_241961

/-- Represents the number of dandelions picked by Billy and George -/
structure DandelionPicks where
  billy_initial : ℕ
  george_initial : ℕ
  billy_additional : ℕ
  george_additional : ℕ

/-- Calculates the average number of dandelions picked -/
def average_picks (d : DandelionPicks) : ℚ :=
  (d.billy_initial + d.george_initial + d.billy_additional + d.george_additional : ℚ) / 2

/-- Theorem stating the average number of dandelions picked by Billy and George -/
theorem dandelion_picking_average :
  ∃ d : DandelionPicks,
    d.billy_initial = 36 ∧
    d.george_initial = (2 * d.billy_initial) / 5 ∧
    d.billy_additional = (5 * d.billy_initial) / 3 ∧
    d.george_additional = (7 * d.george_initial) / 2 ∧
    average_picks d = 79.5 :=
by
  sorry


end NUMINAMATH_CALUDE_dandelion_picking_average_l2419_241961


namespace NUMINAMATH_CALUDE_shaded_area_of_rotated_diameters_l2419_241964

theorem shaded_area_of_rotated_diameters (r : ℝ) (h : r = 6) :
  let circle_area := π * r^2
  let quadrant_area := circle_area / 4
  let triangle_area := r^2
  2 * quadrant_area + 2 * triangle_area = 72 + 9 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_rotated_diameters_l2419_241964


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2419_241994

/-- The eccentricity of an ellipse with a focus shared with the parabola y^2 = x -/
theorem ellipse_eccentricity (a : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = x}
  let ellipse := {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / 3) = 1}
  let parabola_focus : ℝ × ℝ := (1/4, 0)
  (parabola_focus ∈ ellipse) →
  (∃ c b : ℝ, c^2 + b^2 = a^2 ∧ c = 1/4 ∧ b^2 = 3) →
  (c / a = 1/7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2419_241994


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2419_241932

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2419_241932


namespace NUMINAMATH_CALUDE_last_k_digits_power_l2419_241928

theorem last_k_digits_power (k n : ℕ) (A B : ℤ) 
  (h : A ≡ B [ZMOD 10^k]) : 
  A^n ≡ B^n [ZMOD 10^k] := by sorry

end NUMINAMATH_CALUDE_last_k_digits_power_l2419_241928


namespace NUMINAMATH_CALUDE_expression_evaluation_l2419_241936

theorem expression_evaluation :
  (2^1000 + 5^1001)^2 - (2^1000 - 5^1001)^2 = 20 * 10^1000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2419_241936


namespace NUMINAMATH_CALUDE_carol_stereo_savings_l2419_241917

theorem carol_stereo_savings : 
  ∀ (stereo_fraction : ℚ),
  (stereo_fraction + (1/3) * stereo_fraction = 1/4) →
  stereo_fraction = 3/16 := by
sorry

end NUMINAMATH_CALUDE_carol_stereo_savings_l2419_241917


namespace NUMINAMATH_CALUDE_binomial_12_9_l2419_241968

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_9_l2419_241968


namespace NUMINAMATH_CALUDE_kims_class_hours_l2419_241987

/-- Calculates the total class hours after dropping a class -/
def total_class_hours_after_drop (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

/-- Proves that Kim's total class hours after dropping a class is 6 -/
theorem kims_class_hours : total_class_hours_after_drop 4 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kims_class_hours_l2419_241987


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_l2419_241913

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, -m)

theorem perpendicular_vectors_sum (m : ℝ) 
  (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (3 * a.1 + 2 * (b m).1, 3 * a.2 + 2 * (b m).2) = (7, 4) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_l2419_241913


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l2419_241975

theorem relationship_between_x_and_y (x y : ℝ) 
  (h1 : x - y > x + 2) 
  (h2 : x + y + 3 < y - 1) : 
  x < -4 ∧ y < -2 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l2419_241975


namespace NUMINAMATH_CALUDE_complex_number_equality_l2419_241959

theorem complex_number_equality : ∀ z : ℂ, z = (Complex.I ^ 3) / (1 + Complex.I) → z = (-1 - Complex.I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2419_241959


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l2419_241900

/-- Calculates the remaining money after grocery shopping --/
def remaining_money (initial_amount : ℝ) (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
  (pasta_price : ℝ) (pasta_quantity : ℝ) (sauce_price : ℝ) (sauce_quantity : ℝ) : ℝ :=
  initial_amount - (mustard_oil_price * mustard_oil_quantity + pasta_price * pasta_quantity + sauce_price * sauce_quantity)

/-- Theorem stating that Jerry will have $7 after shopping --/
theorem jerry_remaining_money :
  remaining_money 50 13 2 4 3 5 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l2419_241900


namespace NUMINAMATH_CALUDE_max_sum_abs_on_circle_l2419_241962

theorem max_sum_abs_on_circle : 
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ 
  (∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = M) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_circle_l2419_241962


namespace NUMINAMATH_CALUDE_repeating_digit_equality_l2419_241914

/-- Represents a repeating digit number -/
def repeatingDigit (d : ℕ) (n : ℕ) : ℕ := d * (10^n - 1) / 9

/-- The main theorem -/
theorem repeating_digit_equality (x y z : ℕ) (h : x < 10 ∧ y < 10 ∧ z < 10) :
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧
    (repeatingDigit x (2 * n₁) - repeatingDigit y n₁).sqrt = repeatingDigit z n₁ ∧
    (repeatingDigit x (2 * n₂) - repeatingDigit y n₂).sqrt = repeatingDigit z n₂) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 9 ∧ y = 8 ∧ z = 9)) ∧
  (∀ n : ℕ, (repeatingDigit x (2 * n) - repeatingDigit y n).sqrt = repeatingDigit z n) :=
sorry

end NUMINAMATH_CALUDE_repeating_digit_equality_l2419_241914


namespace NUMINAMATH_CALUDE_polygon_is_trapezoid_l2419_241948

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a trapezoid: a quadrilateral with at least one pair of parallel sides -/
def is_trapezoid (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (l1 l2 l3 l4 : Line),
    (p1.y = l1.slope * p1.x + l1.intercept) ∧
    (p2.y = l2.slope * p2.x + l2.intercept) ∧
    (p3.y = l3.slope * p3.x + l3.intercept) ∧
    (p4.y = l4.slope * p4.x + l4.intercept) ∧
    ((l1.slope = l2.slope ∧ l1.slope ≠ l3.slope ∧ l1.slope ≠ l4.slope) ∨
     (l1.slope = l3.slope ∧ l1.slope ≠ l2.slope ∧ l1.slope ≠ l4.slope) ∨
     (l1.slope = l4.slope ∧ l1.slope ≠ l2.slope ∧ l1.slope ≠ l3.slope) ∨
     (l2.slope = l3.slope ∧ l2.slope ≠ l1.slope ∧ l2.slope ≠ l4.slope) ∨
     (l2.slope = l4.slope ∧ l2.slope ≠ l1.slope ∧ l2.slope ≠ l3.slope) ∨
     (l3.slope = l4.slope ∧ l3.slope ≠ l1.slope ∧ l3.slope ≠ l2.slope))

theorem polygon_is_trapezoid :
  let l1 : Line := ⟨2, 3⟩
  let l2 : Line := ⟨-2, 3⟩
  let l3 : Line := ⟨2, -1⟩
  let l4 : Line := ⟨0, -1⟩
  ∃ (p1 p2 p3 p4 : Point),
    (p1.y = l1.slope * p1.x + l1.intercept ∨ p1.y = l2.slope * p1.x + l2.intercept ∨
     p1.y = l3.slope * p1.x + l3.intercept ∨ p1.y = l4.slope * p1.x + l4.intercept) ∧
    (p2.y = l1.slope * p2.x + l1.intercept ∨ p2.y = l2.slope * p2.x + l2.intercept ∨
     p2.y = l3.slope * p2.x + l3.intercept ∨ p2.y = l4.slope * p2.x + l4.intercept) ∧
    (p3.y = l1.slope * p3.x + l1.intercept ∨ p3.y = l2.slope * p3.x + l2.intercept ∨
     p3.y = l3.slope * p3.x + l3.intercept ∨ p3.y = l4.slope * p3.x + l4.intercept) ∧
    (p4.y = l1.slope * p4.x + l1.intercept ∨ p4.y = l2.slope * p4.x + l2.intercept ∨
     p4.y = l3.slope * p4.x + l3.intercept ∨ p4.y = l4.slope * p4.x + l4.intercept) ∧
    is_trapezoid p1 p2 p3 p4 :=
by sorry

end NUMINAMATH_CALUDE_polygon_is_trapezoid_l2419_241948


namespace NUMINAMATH_CALUDE_container_volume_scaling_l2419_241934

theorem container_volume_scaling (original_volume : ℝ) :
  let scale_factor : ℝ := 2
  let new_volume : ℝ := original_volume * scale_factor^3
  new_volume = 8 * original_volume := by sorry

end NUMINAMATH_CALUDE_container_volume_scaling_l2419_241934


namespace NUMINAMATH_CALUDE_intersection_collinearity_l2419_241933

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Check if three points are collinear -/
def collinear (P Q R : Point) : Prop :=
  (Q.y - P.y) * (R.x - P.x) = (R.y - P.y) * (Q.x - P.x)

/-- The main theorem -/
theorem intersection_collinearity 
  (ABCD : Quadrilateral) 
  (P Q : Point) 
  (l : Line) 
  (E F : Point) 
  (R S T : Point) :
  (∃ (l1 : Line), l1.a * ABCD.A.x + l1.b * ABCD.A.y + l1.c = 0 ∧ 
                  l1.a * ABCD.B.x + l1.b * ABCD.B.y + l1.c = 0 ∧ 
                  l1.a * P.x + l1.b * P.y + l1.c = 0) →  -- AB extended through P
  (∃ (l2 : Line), l2.a * ABCD.C.x + l2.b * ABCD.C.y + l2.c = 0 ∧ 
                  l2.a * ABCD.D.x + l2.b * ABCD.D.y + l2.c = 0 ∧ 
                  l2.a * P.x + l2.b * P.y + l2.c = 0) →  -- CD extended through P
  (∃ (l3 : Line), l3.a * ABCD.B.x + l3.b * ABCD.B.y + l3.c = 0 ∧ 
                  l3.a * ABCD.C.x + l3.b * ABCD.C.y + l3.c = 0 ∧ 
                  l3.a * Q.x + l3.b * Q.y + l3.c = 0) →  -- BC extended through Q
  (∃ (l4 : Line), l4.a * ABCD.A.x + l4.b * ABCD.A.y + l4.c = 0 ∧ 
                  l4.a * ABCD.D.x + l4.b * ABCD.D.y + l4.c = 0 ∧ 
                  l4.a * Q.x + l4.b * Q.y + l4.c = 0) →  -- AD extended through Q
  (l.a * P.x + l.b * P.y + l.c = 0) →  -- P is on line l
  (l.a * E.x + l.b * E.y + l.c = 0) →  -- E is on line l
  (l.a * F.x + l.b * F.y + l.c = 0) →  -- F is on line l
  (∃ (l5 l6 : Line), l5.a * ABCD.A.x + l5.b * ABCD.A.y + l5.c = 0 ∧ 
                     l5.a * ABCD.C.x + l5.b * ABCD.C.y + l5.c = 0 ∧ 
                     l6.a * ABCD.B.x + l6.b * ABCD.B.y + l6.c = 0 ∧ 
                     l6.a * ABCD.D.x + l6.b * ABCD.D.y + l6.c = 0 ∧ 
                     l5.a * R.x + l5.b * R.y + l5.c = 0 ∧ 
                     l6.a * R.x + l6.b * R.y + l6.c = 0) →  -- R is intersection of AC and BD
  (∃ (l7 l8 : Line), l7.a * ABCD.A.x + l7.b * ABCD.A.y + l7.c = 0 ∧ 
                     l7.a * E.x + l7.b * E.y + l7.c = 0 ∧ 
                     l8.a * ABCD.B.x + l8.b * ABCD.B.y + l8.c = 0 ∧ 
                     l8.a * F.x + l8.b * F.y + l8.c = 0 ∧ 
                     l7.a * S.x + l7.b * S.y + l7.c = 0 ∧ 
                     l8.a * S.x + l8.b * S.y + l8.c = 0) →  -- S is intersection of AE and BF
  (∃ (l9 l10 : Line), l9.a * ABCD.C.x + l9.b * ABCD.C.y + l9.c = 0 ∧ 
                      l9.a * F.x + l9.b * F.y + l9.c = 0 ∧ 
                      l10.a * ABCD.D.x + l10.b * ABCD.D.y + l10.c = 0 ∧ 
                      l10.a * E.x + l10.b * E.y + l10.c = 0 ∧ 
                      l9.a * T.x + l9.b * T.y + l9.c = 0 ∧ 
                      l10.a * T.x + l10.b * T.y + l10.c = 0) →  -- T is intersection of CF and DE
  collinear R S T ∧ collinear R S Q :=
by sorry

end NUMINAMATH_CALUDE_intersection_collinearity_l2419_241933


namespace NUMINAMATH_CALUDE_boxes_given_to_mother_l2419_241965

theorem boxes_given_to_mother (initial_boxes : ℕ) (final_boxes : ℕ) : 
  initial_boxes = 9 →
  final_boxes = 4 →
  final_boxes * 2 = initial_boxes - (initial_boxes - final_boxes * 2) :=
by
  sorry

end NUMINAMATH_CALUDE_boxes_given_to_mother_l2419_241965


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2419_241925

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 200 * p - 5 = 0) →
  (3 * q^3 - 4 * q^2 + 200 * q - 5 = 0) →
  (3 * r^3 - 4 * r^2 + 200 * r - 5 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 403 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2419_241925


namespace NUMINAMATH_CALUDE_probability_of_not_losing_l2419_241997

theorem probability_of_not_losing (prob_draw prob_win : ℚ) 
  (h1 : prob_draw = 1/2) 
  (h2 : prob_win = 1/3) : 
  prob_draw + prob_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_not_losing_l2419_241997


namespace NUMINAMATH_CALUDE_age_problem_l2419_241907

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - The total of their ages is 52
  Prove that b is 20 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 52) :
  b = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2419_241907


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2419_241931

theorem girls_to_boys_ratio :
  ∀ (total girls boys : ℕ),
  total = 36 →
  girls = boys + 6 →
  girls + boys = total →
  (girls : ℚ) / (boys : ℚ) = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2419_241931


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2419_241982

theorem quadratic_roots_sum_product (α β : ℝ) : 
  α^2 + α - 1 = 0 → β^2 + β - 1 = 0 → α ≠ β → α*β + α + β = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2419_241982


namespace NUMINAMATH_CALUDE_cookie_milk_calculation_l2419_241960

/-- Given that 12 cookies require 2 quarts of milk and 1 quart equals 2 pints,
    prove that 3 cookies require 1 pint of milk. -/
theorem cookie_milk_calculation 
  (cookies_per_recipe : ℕ := 12)
  (quarts_per_recipe : ℚ := 2)
  (pints_per_quart : ℕ := 2)
  (target_cookies : ℕ := 3) :
  let pints_per_recipe := quarts_per_recipe * pints_per_quart
  let pints_per_cookie := pints_per_recipe / cookies_per_recipe
  target_cookies * pints_per_cookie = 1 := by
sorry

end NUMINAMATH_CALUDE_cookie_milk_calculation_l2419_241960


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2419_241944

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (x - 1) / x > 1 ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2419_241944


namespace NUMINAMATH_CALUDE_expression_simplification_l2419_241966

theorem expression_simplification (a b : ℝ) (h1 : 0 < a) (h2 : a < 2*b) :
  1.15 * (Real.sqrt (a^2 - 4*a*b + 4*b^2) / Real.sqrt (a^2 + 4*a*b + 4*b^2)) - 
  (8*a*b / (a^2 - 4*b^2)) + (2*b / (a - 2*b)) = a / (2*b - a) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2419_241966


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2419_241942

structure Point where
  x : ℝ
  y : ℝ

def translate_left (p : Point) (d : ℝ) : Point :=
  ⟨p.x - d, p.y⟩

def symmetric_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let A : Point := ⟨1, 2⟩
  let B : Point := translate_left A 2
  let C : Point := symmetric_origin B
  C = ⟨1, -2⟩ := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2419_241942


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l2419_241911

theorem quadratic_always_negative (k : ℝ) :
  (∀ x : ℝ, (5 - k) * x^2 - 2 * (1 - k) * x + (2 - 2 * k) < 0) ↔ k > 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l2419_241911


namespace NUMINAMATH_CALUDE_absolute_value_and_roots_calculation_l2419_241910

theorem absolute_value_and_roots_calculation : 
  |(-3)| + (1/2)^0 - Real.sqrt 8 * Real.sqrt 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_roots_calculation_l2419_241910


namespace NUMINAMATH_CALUDE_pizza_problem_l2419_241996

/-- Calculates the total number of pizza pieces carried by children -/
def total_pizza_pieces (num_children : ℕ) (pizzas_per_child : ℕ) (pieces_per_pizza : ℕ) : ℕ :=
  num_children * pizzas_per_child * pieces_per_pizza

/-- Proves that 10 children buying 20 pizzas each, with 6 pieces per pizza, carry 1200 pieces total -/
theorem pizza_problem : total_pizza_pieces 10 20 6 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l2419_241996


namespace NUMINAMATH_CALUDE_inequality_theorem_l2419_241950

theorem inequality_theorem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2419_241950


namespace NUMINAMATH_CALUDE_at_least_one_girl_selection_l2419_241967

theorem at_least_one_girl_selection (n_boys n_girls k : ℕ) 
  (h_boys : n_boys = 3) 
  (h_girls : n_girls = 4) 
  (h_select : k = 3) : 
  Nat.choose (n_boys + n_girls) k - Nat.choose n_boys k = 34 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_girl_selection_l2419_241967


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l2419_241923

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 9 else sorry

def sequence_sum (n : ℕ) : ℚ :=
  sorry

theorem smallest_n_satisfying_inequality : 
  (∀ n : ℕ, n > 0 → 3 * sequence_a (n + 1) + sequence_a n = 4) →
  sequence_a 1 = 9 →
  (∀ n : ℕ, n > 0 → |sequence_sum n - n - 6| < 1 / 125 → n ≥ 7) ∧
  |sequence_sum 7 - 7 - 6| < 1 / 125 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l2419_241923


namespace NUMINAMATH_CALUDE_games_before_third_l2419_241953

theorem games_before_third (average_score : ℝ) (third_game_score : ℝ) (points_needed : ℝ) :
  average_score = 61.5 →
  third_game_score = 47 →
  points_needed = 330 →
  (∃ n : ℕ, n * average_score + third_game_score + points_needed = 500 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_games_before_third_l2419_241953


namespace NUMINAMATH_CALUDE_max_value_4tau_minus_n_l2419_241937

/-- τ(n) is the number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- The maximum value of 4τ(n) - n over all positive integers n is 12 -/
theorem max_value_4tau_minus_n :
  (∀ n : ℕ+, (4 * τ n : ℤ) - n.val ≤ 12) ∧
  (∃ n : ℕ+, (4 * τ n : ℤ) - n.val = 12) :=
sorry

end NUMINAMATH_CALUDE_max_value_4tau_minus_n_l2419_241937


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2419_241929

theorem linear_function_quadrants (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x + b ∧ x > 0 ∧ y > 0) ∧  -- First quadrant
  (∃ x y : ℝ, y = a * x + b ∧ x < 0 ∧ y > 0) ∧  -- Second quadrant
  (∃ x y : ℝ, y = a * x + b ∧ x > 0 ∧ y < 0) →  -- Fourth quadrant
  ¬(∃ x y : ℝ, y = b * x - a ∧ x > 0 ∧ y < 0)   -- Not in fourth quadrant
:= by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l2419_241929


namespace NUMINAMATH_CALUDE_combined_flock_size_is_300_l2419_241919

/-- Calculates the combined flock size after a given number of years -/
def combinedFlockSize (initialSize birthRate deathRate years additionalFlockSize : ℕ) : ℕ :=
  initialSize + (birthRate - deathRate) * years + additionalFlockSize

/-- Theorem: The combined flock size after 5 years is 300 ducks -/
theorem combined_flock_size_is_300 :
  combinedFlockSize 100 30 20 5 150 = 300 := by
  sorry

#eval combinedFlockSize 100 30 20 5 150

end NUMINAMATH_CALUDE_combined_flock_size_is_300_l2419_241919


namespace NUMINAMATH_CALUDE_mean_of_numbers_l2419_241989

def number_set : List ℝ := [1, 22, 23, 24, 25, 26, 27, 2]

theorem mean_of_numbers : (number_set.sum / number_set.length : ℝ) = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_numbers_l2419_241989


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2419_241902

theorem complex_magnitude_product (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : Complex.abs (a + b + c) = 1)
  (h5 : Complex.abs (a - b) = Complex.abs (a - c))
  (h6 : b ≠ c) :
  Complex.abs (a + b) * Complex.abs (a + c) = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2419_241902


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l2419_241970

/-- The polynomial f(x) = x^3 - x^2 + 4x + 2 -/
def f (x : ℝ) : ℝ := x^3 - x^2 + 4*x + 2

/-- The polynomial g(x) = x^3 + bx^2 + cx + d -/
def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_relation :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∃ b c d : ℝ, ∀ x : ℝ, f x = 0 → g (x^3) b c d = 0) →
  ∃ b c d : ℝ, b = -1 ∧ c = 72 ∧ d = 8 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l2419_241970


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l2419_241984

theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℝ)
  (excluded_count : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 30)
  (h2 : all_average = 80)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 92)
  : (total_students * all_average - (total_students - excluded_count) * remaining_average) / excluded_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l2419_241984


namespace NUMINAMATH_CALUDE_bananas_permutations_l2419_241912

/-- The number of distinct permutations of a word with repeated letters -/
def permutationsWithRepeats (total : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial total / (repeats.map Nat.factorial).prod

/-- The word "BANANAS" has 7 letters -/
def totalLetters : ℕ := 7

/-- The repetition pattern of letters in "BANANAS" -/
def letterRepeats : List ℕ := [3, 2]  -- 3 'A's and 2 'N's

theorem bananas_permutations :
  permutationsWithRepeats totalLetters letterRepeats = 420 := by
  sorry


end NUMINAMATH_CALUDE_bananas_permutations_l2419_241912


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2419_241901

open Real

theorem min_distance_between_curves : 
  ∀ (m n : ℝ), 
  2 * (m + 1) = n + log n → 
  |m - n| ≥ 3/2 ∧ 
  ∃ (m₀ n₀ : ℝ), 2 * (m₀ + 1) = n₀ + log n₀ ∧ |m₀ - n₀| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2419_241901


namespace NUMINAMATH_CALUDE_binomial_8_4_l2419_241988

theorem binomial_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_4_l2419_241988
