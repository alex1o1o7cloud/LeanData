import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_decreasing_l131_13161

-- Define the quadratic function
def y (x m : ℝ) : ℝ := (x - m)^2 - 1

-- State the theorem
theorem quadratic_function_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≤ 3 ∧ x₂ ≤ 3 ∧ x₁ < x₂ → y x₁ m > y x₂ m) →
  m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_decreasing_l131_13161


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l131_13187

theorem smallest_four_digit_multiple_of_112 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 112 ∣ n → 1008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l131_13187


namespace NUMINAMATH_CALUDE_brian_stones_l131_13180

theorem brian_stones (total : ℕ) (grey : ℕ) (green : ℕ) (white : ℕ) (black : ℕ) : 
  total = 100 →
  grey = 40 →
  green = 60 →
  grey + green = total →
  white + black = total →
  (white : ℚ) / total = (green : ℚ) / total →
  white > black →
  white = 60 := by
sorry

end NUMINAMATH_CALUDE_brian_stones_l131_13180


namespace NUMINAMATH_CALUDE_probability_of_winning_more_than_4000_l131_13143

/-- Represents the number of boxes and keys -/
def num_boxes : ℕ := 3

/-- Represents the total number of ways to assign keys to boxes -/
def total_assignments : ℕ := Nat.factorial num_boxes

/-- Represents the number of ways to correctly assign keys to both the second and third boxes -/
def correct_assignments : ℕ := 1

/-- Theorem stating the probability of correctly assigning keys to both the second and third boxes -/
theorem probability_of_winning_more_than_4000 :
  (correct_assignments : ℚ) / total_assignments = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_more_than_4000_l131_13143


namespace NUMINAMATH_CALUDE_rectangle_ratio_l131_13103

/-- A configuration of squares and a rectangle forming a large square -/
structure SquareConfiguration where
  /-- Side length of each small square -/
  s : ℝ
  /-- Side length of the large square -/
  largeSide : ℝ
  /-- Length of the rectangle -/
  rectLength : ℝ
  /-- Width of the rectangle -/
  rectWidth : ℝ
  /-- The large square's side is 3 times the small square's side -/
  large_square : largeSide = 3 * s
  /-- The rectangle's length is 3 times the small square's side -/
  rect_length : rectLength = 3 * s
  /-- The rectangle's width is 2 times the small square's side -/
  rect_width : rectWidth = 2 * s

/-- The ratio of the rectangle's length to its width is 3/2 -/
theorem rectangle_ratio (config : SquareConfiguration) :
  config.rectLength / config.rectWidth = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l131_13103


namespace NUMINAMATH_CALUDE_right_triangle_area_l131_13115

theorem right_triangle_area (a b : ℝ) (h1 : a = 24) (h2 : b = 30) : 
  (1/2 : ℝ) * a * b = 360 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l131_13115


namespace NUMINAMATH_CALUDE_f_one_when_m_three_max_value_when_even_max_value_attained_when_even_l131_13135

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 2

-- Theorem 1: When m = 3, f(1) = 4
theorem f_one_when_m_three : f 3 1 = 4 := by sorry

-- Define what it means for f to be an even function
def is_even_function (m : ℝ) : Prop := ∀ x, f m (-x) = f m x

-- Theorem 2: If f is an even function, its maximum value is 2
theorem max_value_when_even :
  ∀ m, is_even_function m → ∀ x, f m x ≤ 2 := by sorry

-- Theorem 3: The maximum value 2 is attained when f is an even function
theorem max_value_attained_when_even :
  ∃ m, is_even_function m ∧ ∃ x, f m x = 2 := by sorry

end NUMINAMATH_CALUDE_f_one_when_m_three_max_value_when_even_max_value_attained_when_even_l131_13135


namespace NUMINAMATH_CALUDE_greatest_x_value_l131_13125

theorem greatest_x_value (x : ℤ) : 
  (2.134 * (10 : ℝ) ^ (x : ℝ) < 240000) ↔ x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l131_13125


namespace NUMINAMATH_CALUDE_store_dvds_count_l131_13189

def total_dvds : ℕ := 10
def online_dvds : ℕ := 2

theorem store_dvds_count : total_dvds - online_dvds = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_dvds_count_l131_13189


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l131_13124

-- First expression
theorem simplify_first_expression (a b : ℝ) :
  2*a + 3*b + 6*a + 9*b - 8*a - 5 = 12*b - 5 := by sorry

-- Second expression
theorem simplify_second_expression (x : ℝ) :
  2*(3*x + 1) - (4 - x - x^2) = x^2 + 7*x - 2 := by sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l131_13124


namespace NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l131_13121

/-- Represents a conic section of the form mx^2 + ny^2 = 1 -/
structure Conic (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate to check if a conic is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_axis (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0

theorem hyperbola_y_axis_condition (m n : ℝ) :
  (∃ (c : Conic m n), is_hyperbola_y_axis m n) → m * n < 0 ∧
  ∃ (m' n' : ℝ), m' * n' < 0 ∧ ¬∃ (c : Conic m' n'), is_hyperbola_y_axis m' n' :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l131_13121


namespace NUMINAMATH_CALUDE_geometric_sequence_303rd_term_l131_13119

/-- Represents a geometric sequence -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) := fun (n : ℕ) => a₁ * r ^ (n - 1)

theorem geometric_sequence_303rd_term :
  let seq := GeometricSequence 5 (-2)
  seq 303 = 5 * 2^302 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_303rd_term_l131_13119


namespace NUMINAMATH_CALUDE_polar_equation_of_line_l131_13173

/-- The polar equation of a line passing through (5,0) and perpendicular to α = π/4 -/
theorem polar_equation_of_line (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = 5 ∧ y = 0 ∧ ρ * (Real.cos θ) = x ∧ ρ * (Real.sin θ) = y) →
  (∀ (α : ℝ), α = π/4 → (Real.tan α) * (Real.tan (α + π/2)) = -1) →
  ρ * Real.sin (π/4 + θ) = 5 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_of_line_l131_13173


namespace NUMINAMATH_CALUDE_factorization_2m_squared_minus_8_factorization_perfect_square_trinomial_l131_13149

-- Part 1
theorem factorization_2m_squared_minus_8 (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by sorry

-- Part 2
theorem factorization_perfect_square_trinomial (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_2m_squared_minus_8_factorization_perfect_square_trinomial_l131_13149


namespace NUMINAMATH_CALUDE_total_swordfish_catch_l131_13118

/-- The number of times Shelly and Sam go fishing -/
def fishing_trips : ℕ := 5

/-- The number of swordfish Shelly catches each time -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches each time -/
def sam_catch : ℕ := shelly_catch - 1

/-- The total number of swordfish caught by Shelly and Sam after their fishing trips -/
def total_catch : ℕ := fishing_trips * (shelly_catch + sam_catch)

theorem total_swordfish_catch : total_catch = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_swordfish_catch_l131_13118


namespace NUMINAMATH_CALUDE_jovanas_shells_l131_13126

/-- The total amount of shells Jovana has after her friends add to her collection -/
def total_shells (initial : ℕ) (friend1 : ℕ) (friend2 : ℕ) : ℕ :=
  initial + friend1 + friend2

/-- Theorem stating that Jovana's total shells equal 37 pounds -/
theorem jovanas_shells :
  total_shells 5 15 17 = 37 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l131_13126


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_solution_l131_13106

theorem sum_of_squares_quadratic_solution : 
  ∀ (s₁ s₂ : ℝ), s₁^2 - 10*s₁ + 7 = 0 → s₂^2 - 10*s₂ + 7 = 0 → s₁^2 + s₂^2 = 86 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_solution_l131_13106


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_l131_13109

theorem mixed_number_multiplication : 
  (39 + 18 / 19) * (18 + 19 / 20) = 757 + 1 / 380 := by sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_l131_13109


namespace NUMINAMATH_CALUDE_survey_selection_theorem_l131_13141

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 2

-- Define the total number of students to be selected
def num_selected : ℕ := 4

-- Define the function to calculate the number of ways to select students
def num_ways_to_select : ℕ := (num_boys + num_girls).choose num_selected - num_boys.choose num_selected

-- Theorem statement
theorem survey_selection_theorem : num_ways_to_select = 14 := by
  sorry

end NUMINAMATH_CALUDE_survey_selection_theorem_l131_13141


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l131_13170

theorem missing_digit_divisible_by_three (x : Nat) :
  x < 10 →
  (1357 * 10 + x) * 10 + 2 % 3 = 0 →
  x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l131_13170


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l131_13123

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (2 + Real.sqrt 2) / 2
  let x₂ : ℝ := (2 - Real.sqrt 2) / 2
  2 * x₁^2 = 4 * x₁ - 1 ∧ 2 * x₂^2 = 4 * x₂ - 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l131_13123


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l131_13151

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    5 * x - 3 = (x^2 - 3*x - 18) * (C / (x - 6) + D / (x + 3))) →
  C = 3 ∧ D = 2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l131_13151


namespace NUMINAMATH_CALUDE_mass_percentage_Ca_in_mixture_l131_13116

/-- Molar mass of calcium in g/mol -/
def molar_mass_Ca : ℝ := 40.08

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℝ := 32.07

/-- Molar mass of calcium oxide (CaO) in g/mol -/
def molar_mass_CaO : ℝ := molar_mass_Ca + molar_mass_O

/-- Molar mass of calcium carbonate (CaCO₃) in g/mol -/
def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

/-- Molar mass of calcium sulfate (CaSO₄) in g/mol -/
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

/-- Percentage of CaO in the mixed compound -/
def percent_CaO : ℝ := 40

/-- Percentage of CaCO₃ in the mixed compound -/
def percent_CaCO3 : ℝ := 30

/-- Percentage of CaSO₄ in the mixed compound -/
def percent_CaSO4 : ℝ := 30

/-- Theorem: The mass percentage of Ca in the mixed compound is approximately 49.432% -/
theorem mass_percentage_Ca_in_mixture : 
  ∃ (x : ℝ), abs (x - 49.432) < 0.001 ∧ 
  x = (percent_CaO / 100 * (molar_mass_Ca / molar_mass_CaO * 100)) +
      (percent_CaCO3 / 100 * (molar_mass_Ca / molar_mass_CaCO3 * 100)) +
      (percent_CaSO4 / 100 * (molar_mass_Ca / molar_mass_CaSO4 * 100)) :=
by sorry

end NUMINAMATH_CALUDE_mass_percentage_Ca_in_mixture_l131_13116


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l131_13169

theorem simplify_trig_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l131_13169


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l131_13186

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 12)
  (h2 : current_speed = 2) :
  speed_with_current - 2 * current_speed = 8 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l131_13186


namespace NUMINAMATH_CALUDE_remainder_of_sum_l131_13191

theorem remainder_of_sum (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l131_13191


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l131_13171

/-- A rectangle with a square inside it -/
structure RectangleWithSquare where
  square_side : ℝ
  rect_width : ℝ
  rect_length : ℝ
  width_to_side_ratio : rect_width = 3 * square_side
  length_to_width_ratio : rect_length = 2 * rect_width

/-- The theorem stating that the area of the square is 1/18 of the area of the rectangle -/
theorem square_to_rectangle_area_ratio (r : RectangleWithSquare) :
  (r.square_side ^ 2) / (r.rect_width * r.rect_length) = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l131_13171


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l131_13132

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((37 * x) / (73 * y)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt x / Real.sqrt y = 1281 / 94 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l131_13132


namespace NUMINAMATH_CALUDE_whittlesford_brass_band_max_members_l131_13196

theorem whittlesford_brass_band_max_members 
  (k : ℕ) 
  (h1 : 45 * k % 37 = 28) 
  (h2 : 45 * k < 1500) : 
  45 * k ≤ 945 ∧ ∃ (k : ℕ), 45 * k = 945 ∧ 45 * k % 37 = 28 ∧ 45 * k < 1500 :=
by sorry

end NUMINAMATH_CALUDE_whittlesford_brass_band_max_members_l131_13196


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l131_13133

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  white : ℕ
  black : ℕ

/-- The minimum number of balls needed to guarantee a specific count of one color -/
def minBallsForGuarantee (counts : BallCounts) (targetCount : ℕ) : ℕ :=
  (min counts.red (targetCount - 1)) +
  (min counts.green (targetCount - 1)) +
  (min counts.yellow (targetCount - 1)) +
  (min counts.blue (targetCount - 1)) +
  (min counts.white (targetCount - 1)) +
  (min counts.black (targetCount - 1)) + 1

/-- Theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_to_draw (counts : BallCounts)
    (h_red : counts.red = 35)
    (h_green : counts.green = 25)
    (h_yellow : counts.yellow = 22)
    (h_blue : counts.blue = 15)
    (h_white : counts.white = 14)
    (h_black : counts.black = 12) :
    minBallsForGuarantee counts 18 = 93 := by
  sorry


end NUMINAMATH_CALUDE_min_balls_to_draw_l131_13133


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l131_13146

theorem circle_area_with_diameter_10 :
  ∀ (d : ℝ) (A : ℝ), 
    d = 10 →
    A = π * (d / 2)^2 →
    A = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l131_13146


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l131_13137

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisibility_by_nine (n : ℕ) : 
  (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l131_13137


namespace NUMINAMATH_CALUDE_negation_of_odd_function_implication_l131_13100

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_odd_function_implication :
  (¬ (IsOdd f → IsOdd (fun x ↦ f (-x)))) ↔ (¬ IsOdd f → ¬ IsOdd (fun x ↦ f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_odd_function_implication_l131_13100


namespace NUMINAMATH_CALUDE_expression_factorization_l131_13128

theorem expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l131_13128


namespace NUMINAMATH_CALUDE_infinitely_many_terms_greater_than_position_l131_13156

/-- A sequence of natural numbers excluding 1 -/
def NatSequenceExcluding1 := ℕ → {n : ℕ // n ≠ 1}

/-- The proposition that for any sequence of natural numbers excluding 1,
    there are infinitely many terms greater than their positions -/
theorem infinitely_many_terms_greater_than_position
  (seq : NatSequenceExcluding1) :
  ∀ N : ℕ, ∃ n > N, (seq n).val > n := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_terms_greater_than_position_l131_13156


namespace NUMINAMATH_CALUDE_inequality_proof_l131_13185

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l131_13185


namespace NUMINAMATH_CALUDE_centerpiece_roses_count_l131_13160

/-- The number of centerpieces --/
def num_centerpieces : ℕ := 6

/-- The number of lilies in each centerpiece --/
def lilies_per_centerpiece : ℕ := 6

/-- The total cost of all centerpieces in cents --/
def total_cost : ℕ := 270000

/-- The cost of each flower in cents --/
def flower_cost : ℕ := 1500

/-- The number of roses in each centerpiece --/
def roses_per_centerpiece : ℕ := 8

theorem centerpiece_roses_count :
  ∃ (r : ℕ),
    r = roses_per_centerpiece ∧
    num_centerpieces * flower_cost * (3 * r + lilies_per_centerpiece) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_centerpiece_roses_count_l131_13160


namespace NUMINAMATH_CALUDE_circle_properties_l131_13162

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ k : ℝ, k = Real.sqrt 3 ∧ 
    (∀ t : ℝ, x ≠ 0 → y / x ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ / x₀ = k)) ∧
  (∃ k : ℝ, k = -Real.sqrt 3 ∧ 
    (∀ t : ℝ, x ≠ 0 → k ≤ y / x) ∧
    (∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ / x₀ = k)) ∧
  (∃ k : ℝ, k = -2 + Real.sqrt 6 ∧ 
    (∀ t : ℝ, y - x ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = k)) ∧
  (∃ k : ℝ, k = -2 - Real.sqrt 6 ∧ 
    (∀ t : ℝ, k ≤ y - x) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = k)) ∧
  (∃ k : ℝ, k = 7 + 4 * Real.sqrt 3 ∧ 
    (∀ t : ℝ, x^2 + y^2 ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = k)) ∧
  (∃ k : ℝ, k = 7 - 4 * Real.sqrt 3 ∧ 
    (∀ t : ℝ, k ≤ x^2 + y^2) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = k)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l131_13162


namespace NUMINAMATH_CALUDE_square_garden_area_l131_13101

theorem square_garden_area (s : ℝ) (h1 : s > 0) : 
  (4 * s = 40) → (s^2 = 2 * (4 * s) + 20) → s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_l131_13101


namespace NUMINAMATH_CALUDE_ratio_change_after_subtraction_l131_13163

theorem ratio_change_after_subtraction (a b : ℕ) (h1 : a * 5 = b * 6) (h2 : a > 5 ∧ b > 5) 
  (h3 : (a - 5) - (b - 5) = 5) : (a - 5) * 4 = (b - 5) * 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_after_subtraction_l131_13163


namespace NUMINAMATH_CALUDE_kids_stayed_home_l131_13179

theorem kids_stayed_home (camp_kids : ℕ) (additional_home_kids : ℕ) 
  (h1 : camp_kids = 202958)
  (h2 : additional_home_kids = 574664) :
  camp_kids + additional_home_kids = 777622 := by
  sorry

end NUMINAMATH_CALUDE_kids_stayed_home_l131_13179


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l131_13182

theorem bernoulli_inequality (p : ℝ) (k : ℚ) (hp : p > 0) (hk : k > 1) :
  (1 + p)^(k : ℝ) > 1 + p * k := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l131_13182


namespace NUMINAMATH_CALUDE_unique_solution_l131_13181

/-- Two functions f and g from ℝ to ℝ satisfying the given functional equation -/
def SatisfyEquation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - g y) = (g x)^2 - y

/-- The theorem stating that the only functions satisfying the equation are the identity function -/
theorem unique_solution {f g : ℝ → ℝ} (h : SatisfyEquation f g) :
    (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l131_13181


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l131_13130

-- Define sets A and B as functions of a
def A (a : ℝ) : Set ℝ := {4, a^2}
def B (a : ℝ) : Set ℝ := {a-6, 1+a, 9}

-- Theorem statement
theorem union_of_A_and_B :
  ∃ a : ℝ, (A a ∩ B a = {9}) ∧ (A a ∪ B a = {-9, -2, 4, 9}) :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l131_13130


namespace NUMINAMATH_CALUDE_third_group_frequency_l131_13148

theorem third_group_frequency
  (total_groups : Nat)
  (sum_first_three : Nat)
  (sum_last_three : Nat)
  (frequency_third : Real)
  (h1 : total_groups = 5)
  (h2 : sum_first_three = 160)
  (h3 : sum_last_three = 260)
  (h4 : frequency_third = 0.20) :
  ∃ (third_group : Nat),
    third_group = 70 ∧
    (third_group : Real) / (sum_first_three + sum_last_three - third_group) = frequency_third :=
by sorry

end NUMINAMATH_CALUDE_third_group_frequency_l131_13148


namespace NUMINAMATH_CALUDE_football_cost_l131_13188

theorem football_cost (total_cost marbles_cost baseball_cost : ℚ)
  (h1 : total_cost = 20.52)
  (h2 : marbles_cost = 9.05)
  (h3 : baseball_cost = 6.52) :
  total_cost - marbles_cost - baseball_cost = 4.95 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_l131_13188


namespace NUMINAMATH_CALUDE_newspaper_conference_attendees_l131_13154

/-- The minimum number of people attending the newspaper conference -/
def min_attendees : ℕ := 126

/-- The number of writers at the conference -/
def writers : ℕ := 35

/-- The minimum number of editors at the conference -/
def min_editors : ℕ := 39

/-- The maximum number of people who are both writers and editors -/
def max_both : ℕ := 26

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * max_both

theorem newspaper_conference_attendees :
  ∀ N : ℕ,
  (N ≥ writers + min_editors - max_both + neither) →
  (N ≥ min_attendees) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_attendees_l131_13154


namespace NUMINAMATH_CALUDE_point_transformation_to_polar_coordinates_l131_13107

theorem point_transformation_to_polar_coordinates :
  ∀ (x y : ℝ),
    2 * x = 6 ∧ Real.sqrt 3 * y = -3 →
    ∃ (ρ θ : ℝ),
      ρ = 2 * Real.sqrt 3 ∧
      θ = 11 * π / 6 ∧
      ρ > 0 ∧
      0 ≤ θ ∧ θ < 2 * π ∧
      x = ρ * Real.cos θ ∧
      y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_point_transformation_to_polar_coordinates_l131_13107


namespace NUMINAMATH_CALUDE_lawnmower_value_calculation_l131_13197

/-- Calculates the final value of a lawnmower after three consecutive price drops -/
theorem lawnmower_value_calculation (initial_value : ℝ) 
  (drop1 drop2 drop3 : ℝ) (final_value : ℝ) : 
  initial_value = 100 ∧ 
  drop1 = 0.25 ∧ 
  drop2 = 0.20 ∧ 
  drop3 = 0.15 ∧ 
  final_value = initial_value * (1 - drop1) * (1 - drop2) * (1 - drop3) →
  final_value = 51 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_value_calculation_l131_13197


namespace NUMINAMATH_CALUDE_opposite_of_2023_l131_13198

theorem opposite_of_2023 :
  ∃ y : ℤ, (2023 : ℤ) + y = 0 ∧ y = -2023 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l131_13198


namespace NUMINAMATH_CALUDE_special_quadrilateral_AD_length_special_quadrilateral_AD_length_is_30_l131_13112

/-- A quadrilateral with specific side lengths and angle properties -/
structure SpecialQuadrilateral where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- Angle properties
  B_obtuse : ℝ
  C_obtuse : ℝ
  sin_C : ℝ
  cos_B : ℝ
  -- Conditions
  AB_eq : AB = 6
  BC_eq : BC = 8
  CD_eq : CD = 15
  B_obtuse_cond : B_obtuse > π / 2
  C_obtuse_cond : C_obtuse > π / 2
  sin_C_eq : sin_C = 4 / 5
  cos_B_eq : cos_B = -4 / 5

/-- The length of side AD in the special quadrilateral is 30 -/
theorem special_quadrilateral_AD_length (q : SpecialQuadrilateral) : ℝ :=
  30

/-- The main theorem stating that for any special quadrilateral, 
    the length of side AD is 30 -/
theorem special_quadrilateral_AD_length_is_30 (q : SpecialQuadrilateral) :
  special_quadrilateral_AD_length q = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_AD_length_special_quadrilateral_AD_length_is_30_l131_13112


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l131_13159

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_n_satisfying_conditions :
  ∃! n : ℕ, n ≥ 10 ∧ 
            is_prime (n + 6) ∧ 
            is_perfect_square (9*n + 7) ∧
            ∀ m : ℕ, m ≥ 10 → 
                     is_prime (m + 6) → 
                     is_perfect_square (9*m + 7) → 
                     n ≤ m ∧
            n = 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l131_13159


namespace NUMINAMATH_CALUDE_crayons_per_child_l131_13120

theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_crayons = 50) : 
  total_crayons / total_children = 5 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_child_l131_13120


namespace NUMINAMATH_CALUDE_carla_cards_theorem_l131_13192

/-- A structure representing a card with two numbers -/
structure Card where
  visible : ℕ
  hidden : ℕ

/-- The setup of Carla's cards -/
def carla_cards : Card × Card :=
  ⟨⟨37, 0⟩, ⟨53, 0⟩⟩

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Theorem stating the properties of Carla's card setup and the result -/
theorem carla_cards_theorem (cards : Card × Card) : 
  cards = carla_cards →
  (∃ p₁ p₂ : ℕ, 
    is_prime p₁ ∧ 
    is_prime p₂ ∧ 
    p₁ ≠ p₂ ∧
    cards.1.visible + p₁ = cards.2.visible + p₂ ∧
    (p₁ + p₂) / 2 = 11) := by
  sorry

#check carla_cards_theorem

end NUMINAMATH_CALUDE_carla_cards_theorem_l131_13192


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l131_13199

/-- Given that 3x^(2m)y^3 and -2x^2y^n are like terms, prove that m + n = 4 -/
theorem like_terms_exponent_sum (m n : ℕ) : 
  (∀ x y : ℝ, 3 * x^(2*m) * y^3 = -2 * x^2 * y^n) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l131_13199


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l131_13114

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its division -/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Represents a way of cutting the plywood -/
structure CutMethod where
  piece : Rectangle
  is_valid : Bool

theorem plywood_cut_perimeter_difference 
  (p : Plywood) 
  (h1 : p.length = 10 ∧ p.width = 5)
  (h2 : p.num_pieces = 5) :
  ∃ (m1 m2 : CutMethod), 
    m1.is_valid ∧ m2.is_valid ∧ 
    perimeter m1.piece - perimeter m2.piece = 8 ∧
    ∀ (m : CutMethod), m.is_valid → 
      perimeter m.piece ≤ perimeter m1.piece ∧
      perimeter m.piece ≥ perimeter m2.piece := by
  sorry


end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l131_13114


namespace NUMINAMATH_CALUDE_ratio_of_segments_l131_13193

/-- Given five consecutive points on a straight line, prove the ratio of two segments --/
theorem ratio_of_segments (a b c d e : ℝ) : 
  (b < c) ∧ (c < d) ∧  -- Consecutive points
  (e - d = 8) ∧        -- de = 8
  (b - a = 5) ∧        -- ab = 5
  (c - a = 11) ∧       -- ac = 11
  (e - a = 21)         -- ae = 21
  → (c - b) / (d - c) = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l131_13193


namespace NUMINAMATH_CALUDE_shortest_chain_no_self_intersections_l131_13113

/-- A polygonal chain in a plane -/
structure PolygonalChain (n : ℕ) where
  points : Fin n → ℝ × ℝ
  
/-- The length of a polygonal chain -/
def length (chain : PolygonalChain n) : ℝ := sorry

/-- A polygonal chain has self-intersections -/
def has_self_intersections (chain : PolygonalChain n) : Prop := sorry

/-- A polygonal chain is the shortest among all chains connecting the same points -/
def is_shortest (chain : PolygonalChain n) : Prop := 
  ∀ other : PolygonalChain n, chain.points = other.points → length chain ≤ length other

/-- The shortest polygonal chain connecting n points in a plane has no self-intersections -/
theorem shortest_chain_no_self_intersections (n : ℕ) (chain : PolygonalChain n) :
  is_shortest chain → ¬ has_self_intersections chain :=
sorry

end NUMINAMATH_CALUDE_shortest_chain_no_self_intersections_l131_13113


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l131_13145

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ),
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
      (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 5 / (x - 5)) ∧
    C + D = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l131_13145


namespace NUMINAMATH_CALUDE_furniture_sale_price_l131_13157

theorem furniture_sale_price (wholesale_price : ℝ) 
  (sticker_price : ℝ) (sale_price : ℝ) :
  sticker_price = wholesale_price * 1.4 →
  sale_price = sticker_price * 0.65 →
  sale_price = wholesale_price * 0.91 := by
sorry

end NUMINAMATH_CALUDE_furniture_sale_price_l131_13157


namespace NUMINAMATH_CALUDE_ball_returns_after_15_throws_l131_13177

/-- Represents the number of girls to skip in each throw -/
def skip_pattern : ℕ → ℕ
  | n => if n % 2 = 0 then 3 else 4

/-- Calculates the position of the girl who receives the ball after n throws -/
def ball_position (n : ℕ) : Fin 15 :=
  (List.range n).foldl (fun pos _ => 
    (pos + skip_pattern pos + 1 : Fin 15)) 0

theorem ball_returns_after_15_throws :
  ball_position 15 = 0 := by sorry

end NUMINAMATH_CALUDE_ball_returns_after_15_throws_l131_13177


namespace NUMINAMATH_CALUDE_min_value_product_l131_13165

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 3) ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l131_13165


namespace NUMINAMATH_CALUDE_paper_flipping_difference_l131_13174

theorem paper_flipping_difference (Y G : ℕ) : 
  Y - 152 = G + 152 + 346 → Y - G = 650 := by sorry

end NUMINAMATH_CALUDE_paper_flipping_difference_l131_13174


namespace NUMINAMATH_CALUDE_root_ratio_sum_squared_l131_13139

theorem root_ratio_sum_squared (k₁ k₂ : ℝ) (a b : ℝ) : 
  (∀ x, k₁ * (x^2 - x) + x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x, k₂ * (x^2 - x) + x + 3 = 0 → (x = a ∨ x = b)) →
  a / b + b / a = 2 →
  k₁^2 + k₂^2 = 194 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_squared_l131_13139


namespace NUMINAMATH_CALUDE_pencil_distribution_l131_13183

/-- Given 1204 pens and an unknown number of pencils distributed equally among 28 students,
    prove that the total number of pencils must be a multiple of 28. -/
theorem pencil_distribution (total_pencils : ℕ) : 
  (1204 % 28 = 0) → 
  (∃ (pencils_per_student : ℕ), total_pencils = 28 * pencils_per_student) :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l131_13183


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_specific_l131_13142

def arithmetic_series_sum (a₁ aₙ d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_specific :
  arithmetic_series_sum 12 50 (1/10) = 11811 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_specific_l131_13142


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l131_13155

theorem ac_squared_gt_bc_squared (a b c : ℝ) : a > b → a * c^2 > b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l131_13155


namespace NUMINAMATH_CALUDE_problem_solution_l131_13122

theorem problem_solution (x y : ℝ) 
  (h1 : x = 51) 
  (h2 : x^3*y - 2*x^2*y + x*y = 127500) : 
  y = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l131_13122


namespace NUMINAMATH_CALUDE_lcm_equation_solution_l131_13172

theorem lcm_equation_solution :
  ∀ x y : ℕ, 
    x > 0 ∧ y > 0 → 
    Nat.lcm x y = 1 + 2*x + 3*y ↔ (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_lcm_equation_solution_l131_13172


namespace NUMINAMATH_CALUDE_marco_card_trade_ratio_l131_13175

theorem marco_card_trade_ratio : 
  ∀ (total_cards duplicates_traded new_cards : ℕ),
    total_cards = 500 →
    duplicates_traded = new_cards →
    new_cards = 25 →
    (duplicates_traded : ℚ) / (total_cards / 4 : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_marco_card_trade_ratio_l131_13175


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l131_13129

/-- Given two intersecting circles, prove that the equation of the line 
    containing their intersection points can be found by subtracting 
    the equations of the circles. -/
theorem intersection_line_of_circles 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 10) 
  (h2 : (x-1)^2 + (y-3)^2 = 20) : 
  x + 3*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l131_13129


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l131_13104

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 294) :
  let s := Real.sqrt (S / 6)
  s ^ 3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l131_13104


namespace NUMINAMATH_CALUDE_minimum_c_value_l131_13108

-- Define the curve
def on_curve (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the inequality condition
def inequality_holds (c : ℝ) : Prop := ∀ x y : ℝ, on_curve x y → x + y + c ≥ 0

-- State the theorem
theorem minimum_c_value : 
  (∃ c_min : ℝ, (∀ c : ℝ, c ≥ c_min ↔ inequality_holds c) ∧ c_min = Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_minimum_c_value_l131_13108


namespace NUMINAMATH_CALUDE_expensive_gimbap_count_l131_13167

def basic_gimbap : ℕ := 2000
def tuna_gimbap : ℕ := 3500
def red_pepper_gimbap : ℕ := 3000
def beef_gimbap : ℕ := 4000
def rice_gimbap : ℕ := 3500

def gimbap_prices : List ℕ := [basic_gimbap, tuna_gimbap, red_pepper_gimbap, beef_gimbap, rice_gimbap]

def count_expensive_gimbap (prices : List ℕ) : ℕ :=
  (prices.filter (λ price => price ≥ 3500)).length

theorem expensive_gimbap_count : count_expensive_gimbap gimbap_prices = 3 := by
  sorry

end NUMINAMATH_CALUDE_expensive_gimbap_count_l131_13167


namespace NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l131_13127

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_even_digits_proof :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
    n ≤ largest_eight_digit_with_even_digits :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_even_digits_proof_l131_13127


namespace NUMINAMATH_CALUDE_recipe_ratio_change_l131_13136

/-- Represents the ratio of ingredients in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  { flour := 7, water := 2, sugar := 1 }

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  { flour := 7, water := 1, sugar := 2 }

/-- The amount of water in the new recipe --/
def new_water_amount : ℚ := 2

/-- The amount of sugar in the new recipe --/
def new_sugar_amount : ℚ := 4

theorem recipe_ratio_change :
  (new_ratio.water - original_ratio.water) = -1 :=
sorry

end NUMINAMATH_CALUDE_recipe_ratio_change_l131_13136


namespace NUMINAMATH_CALUDE_expression_nonpositive_l131_13111

theorem expression_nonpositive (x : ℝ) : (6 * x - 1) / 4 - 2 * x ≤ 0 ↔ x ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonpositive_l131_13111


namespace NUMINAMATH_CALUDE_seven_mult_three_equals_sixteen_l131_13184

-- Define the custom operation *
def custom_mult (a b : ℤ) : ℤ := 4*a + 3*b - a*b

-- State the theorem
theorem seven_mult_three_equals_sixteen : custom_mult 7 3 = 16 := by sorry

end NUMINAMATH_CALUDE_seven_mult_three_equals_sixteen_l131_13184


namespace NUMINAMATH_CALUDE_shaded_area_l131_13178

/-- The shaded area in a geometric configuration --/
theorem shaded_area (AB BC : ℝ) (h1 : AB = Real.sqrt ((8 + Real.sqrt (64 - π^2)) / π))
  (h2 : BC = Real.sqrt ((8 - Real.sqrt (64 - π^2)) / π)) :
  (π / 4) * (AB^2 + BC^2) - AB * BC = 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_l131_13178


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l131_13153

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- Shorter leg length
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l131_13153


namespace NUMINAMATH_CALUDE_acceptable_quality_probability_l131_13105

theorem acceptable_quality_probability (p1 p2 : ℝ) 
  (h1 : p1 = 0.01) 
  (h2 : p2 = 0.03) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = 0.960 := by
  sorry

end NUMINAMATH_CALUDE_acceptable_quality_probability_l131_13105


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_81_l131_13138

theorem right_triangle_arithmetic_progression_81 :
  ∃ (a d : ℕ), 
    (a > 0) ∧ (d > 0) ∧
    (a - d)^2 + a^2 = (a + d)^2 ∧
    (81 = a - d ∨ 81 = a ∨ 81 = a + d) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_81_l131_13138


namespace NUMINAMATH_CALUDE_task_completion_ways_l131_13195

theorem task_completion_ways (method1_count method2_count : ℕ) :
  method1_count + method2_count = 
  (number_of_ways_to_choose_person : ℕ) :=
by sorry

#check task_completion_ways 5 4

end NUMINAMATH_CALUDE_task_completion_ways_l131_13195


namespace NUMINAMATH_CALUDE_expression_value_l131_13134

theorem expression_value (x y : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : y ≠ 0) :
  (7 * x + 4 * y) / (x - 2 * y) = 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l131_13134


namespace NUMINAMATH_CALUDE_allison_not_lowest_prob_l131_13117

/-- Represents a 6-sided cube with specific face values -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube with all faces showing 3 -/
def allison_cube : Cube :=
  ⟨λ _ => 3⟩

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  ⟨λ i => i.val + 1⟩

/-- Noah's cube with three faces showing 1 and three faces showing 4 -/
def noah_cube : Cube :=
  ⟨λ i => if i.val < 3 then 1 else 4⟩

/-- The probability of rolling a value less than or equal to 3 on Brian's cube -/
def brian_prob_le_3 : ℚ :=
  1/2

/-- The probability of rolling a 4 on Noah's cube -/
def noah_prob_4 : ℚ :=
  1/2

/-- The probability of both Brian and Noah rolling lower than Allison -/
def prob_both_lower : ℚ :=
  1/6

theorem allison_not_lowest_prob :
  1 - prob_both_lower = 5/6 :=
sorry

end NUMINAMATH_CALUDE_allison_not_lowest_prob_l131_13117


namespace NUMINAMATH_CALUDE_rectangle_width_equals_eight_l131_13176

theorem rectangle_width_equals_eight (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ)
  (h1 : square_side = 4)
  (h2 : rect_length = 2)
  (h3 : square_side * square_side = rect_length * rect_width) :
  rect_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_eight_l131_13176


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l131_13166

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_percentage : ℝ)
  (defective_shipped_percentage : ℝ)
  (h1 : defective_percentage = 8)
  (h2 : defective_shipped_percentage = 0.4) :
  (defective_shipped_percentage / defective_percentage) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l131_13166


namespace NUMINAMATH_CALUDE_correct_subtraction_l131_13164

/-- Given a two-digit number XY and another number Z, prove that the correct subtraction result is 49 -/
theorem correct_subtraction (X Y Z : ℕ) : 
  X = 2 → 
  Y = 4 → 
  Z - 59 = 14 → 
  Z - (10 * X + Y) = 49 := by
sorry

end NUMINAMATH_CALUDE_correct_subtraction_l131_13164


namespace NUMINAMATH_CALUDE_nice_function_property_l131_13152

def nice (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, f^[a] b = f (a + b - 1)

theorem nice_function_property (g : ℕ → ℕ) (A : ℕ) 
  (hg : nice g)
  (hA : g (A + 2018) = g A + 1)
  (hA1 : g (A + 1) ≠ g (A + 1 + 2017^2017)) :
  ∀ n < A, g n = n + 1 := by sorry

end NUMINAMATH_CALUDE_nice_function_property_l131_13152


namespace NUMINAMATH_CALUDE_green_peaches_per_basket_l131_13158

/-- Proves the number of green peaches in each basket -/
theorem green_peaches_per_basket 
  (num_baskets : ℕ) 
  (red_per_basket : ℕ) 
  (total_peaches : ℕ) 
  (h1 : num_baskets = 2)
  (h2 : red_per_basket = 4)
  (h3 : total_peaches = 12) :
  (total_peaches - num_baskets * red_per_basket) / num_baskets = 2 := by
sorry

end NUMINAMATH_CALUDE_green_peaches_per_basket_l131_13158


namespace NUMINAMATH_CALUDE_banana_purchase_cost_l131_13150

/-- The cost of bananas in dollars per three pounds -/
def banana_rate : ℚ := 3

/-- The amount of bananas in pounds to be purchased -/
def banana_amount : ℚ := 18

/-- The cost of purchasing the given amount of bananas -/
def banana_cost : ℚ := banana_amount * (banana_rate / 3)

theorem banana_purchase_cost : banana_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_banana_purchase_cost_l131_13150


namespace NUMINAMATH_CALUDE_sphere_volume_from_inscribed_box_l131_13190

/-- The volume of a sphere given an inscribed rectangular box --/
theorem sphere_volume_from_inscribed_box (AB BC AA₁ : ℝ) (h1 : AB = 2) (h2 : BC = 2) (h3 : AA₁ = 2 * Real.sqrt 2) :
  let box_diagonal := Real.sqrt (AB^2 + BC^2 + AA₁^2)
  let sphere_radius := box_diagonal / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = (32 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_inscribed_box_l131_13190


namespace NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l131_13194

/-- Given that 9 bananas weigh the same as 6 pears, prove that 36 bananas weigh the same as 24 pears. -/
theorem banana_pear_weight_equivalence (banana_weight pear_weight : ℝ) 
  (h : 9 * banana_weight = 6 * pear_weight) :
  36 * banana_weight = 24 * pear_weight := by
  sorry

end NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l131_13194


namespace NUMINAMATH_CALUDE_theme_parks_calculation_l131_13110

/-- The number of theme parks in three towns -/
def total_theme_parks (jamestown venice marina_del_ray : ℕ) : ℕ :=
  jamestown + venice + marina_del_ray

/-- Theorem stating the total number of theme parks in the three towns -/
theorem theme_parks_calculation :
  ∃ (jamestown venice marina_del_ray : ℕ),
    jamestown = 20 ∧
    venice = jamestown + 25 ∧
    marina_del_ray = jamestown + 50 ∧
    total_theme_parks jamestown venice marina_del_ray = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_theme_parks_calculation_l131_13110


namespace NUMINAMATH_CALUDE_adam_caramel_boxes_l131_13131

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes (chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies - chocolate_boxes * pieces_per_box) / pieces_per_box

/-- Proof that Adam bought 5 boxes of caramel candy -/
theorem adam_caramel_boxes : 
  caramel_boxes 2 4 28 = 5 := by
  sorry

end NUMINAMATH_CALUDE_adam_caramel_boxes_l131_13131


namespace NUMINAMATH_CALUDE_track_length_track_length_is_200_l131_13102

/-- The length of a circular track given specific meeting conditions of two runners -/
theorem track_length : ℝ → Prop :=
  fun track_length =>
    ∀ (brenda_speed sally_speed : ℝ),
      brenda_speed > 0 ∧ sally_speed > 0 →
      ∃ (first_meeting_time second_meeting_time : ℝ),
        first_meeting_time > 0 ∧ second_meeting_time > first_meeting_time ∧
        brenda_speed * first_meeting_time = 120 ∧
        brenda_speed * (second_meeting_time - first_meeting_time) = 200 ∧
        (brenda_speed * first_meeting_time + sally_speed * first_meeting_time = track_length / 2) ∧
        (brenda_speed * second_meeting_time + sally_speed * second_meeting_time = 
          track_length + track_length / 2) →
        track_length = 200

/-- The track length is 200 meters -/
theorem track_length_is_200 : track_length 200 := by
  sorry

end NUMINAMATH_CALUDE_track_length_track_length_is_200_l131_13102


namespace NUMINAMATH_CALUDE_park_orchid_bushes_after_planting_l131_13168

/-- The number of orchid bushes in the park after planting -/
def total_orchid_bushes (current : ℕ) (newly_planted : ℕ) : ℕ :=
  current + newly_planted

/-- Theorem: The park will have 35 orchid bushes after planting -/
theorem park_orchid_bushes_after_planting :
  total_orchid_bushes 22 13 = 35 := by
  sorry

end NUMINAMATH_CALUDE_park_orchid_bushes_after_planting_l131_13168


namespace NUMINAMATH_CALUDE_train_speed_l131_13147

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 360) (h2 : time = 30) :
  length / time = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l131_13147


namespace NUMINAMATH_CALUDE_right_triangle_angle_l131_13140

theorem right_triangle_angle (α β : ℝ) : 
  α + β + 90 = 180 → β = 70 → α = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_l131_13140


namespace NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l131_13144

/-- The probability that Xavier and Yvonne solve a problem but Zelda does not,
    given their individual success probabilities -/
theorem xavier_yvonne_not_zelda_probability 
  (p_xavier : ℚ) (p_yvonne : ℚ) (p_zelda : ℚ)
  (h_xavier : p_xavier = 1/5)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8)
  (h_independent : True) -- Assumption of independence
  : p_xavier * p_yvonne * (1 - p_zelda) = 3/80 := by
  sorry

end NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l131_13144
