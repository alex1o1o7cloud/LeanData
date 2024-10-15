import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l872_87200

theorem negation_of_absolute_value_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| ≥ 3) ↔ (∃ x ∈ S, |x| < 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l872_87200


namespace NUMINAMATH_CALUDE_only_zero_solution_l872_87280

theorem only_zero_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_solution_l872_87280


namespace NUMINAMATH_CALUDE_rational_equation_implies_c_zero_l872_87254

theorem rational_equation_implies_c_zero (a b c : ℚ) 
  (h : (a + b + c) * (a + b - c) = 2 * c^2) : c = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_implies_c_zero_l872_87254


namespace NUMINAMATH_CALUDE_expression_equality_l872_87247

theorem expression_equality : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l872_87247


namespace NUMINAMATH_CALUDE_cube_side_length_l872_87227

theorem cube_side_length (surface_area : ℝ) (h : surface_area = 600) :
  ∃ (side_length : ℝ), side_length > 0 ∧ 6 * side_length^2 = surface_area ∧ side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l872_87227


namespace NUMINAMATH_CALUDE_mikes_marbles_l872_87267

/-- Given that Mike initially has 8 orange marbles and gives 4 to Sam,
    prove that Mike now has 4 orange marbles. -/
theorem mikes_marbles (initial_marbles : ℕ) (marbles_given : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 8 →
  marbles_given = 4 →
  remaining_marbles = initial_marbles - marbles_given →
  remaining_marbles = 4 := by
  sorry

end NUMINAMATH_CALUDE_mikes_marbles_l872_87267


namespace NUMINAMATH_CALUDE_triangle_formation_l872_87246

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_formation :
  can_form_triangle 13 12 20 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 ∧
  ¬ can_form_triangle 3 4 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l872_87246


namespace NUMINAMATH_CALUDE_customer_difference_l872_87216

theorem customer_difference (initial_customers remaining_customers : ℕ) 
  (h1 : initial_customers = 19) 
  (h2 : remaining_customers = 4) : 
  initial_customers - remaining_customers = 15 := by
sorry

end NUMINAMATH_CALUDE_customer_difference_l872_87216


namespace NUMINAMATH_CALUDE_vector_operation_proof_l872_87275

def v1 : ℝ × ℝ × ℝ := (-3, 2, -5)
def v2 : ℝ × ℝ × ℝ := (1, 7, -3)

theorem vector_operation_proof :
  v1 + (2 : ℝ) • v2 = (-1, 16, -11) := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l872_87275


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l872_87266

theorem sufficient_not_necessary (x y : ℝ) :
  (((x < 0 ∧ y < 0) → (x + y - 4 < 0)) ∧
   ∃ x y : ℝ, (x + y - 4 < 0) ∧ ¬(x < 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l872_87266


namespace NUMINAMATH_CALUDE_average_speed_calculation_l872_87219

-- Define the variables
def distance_day1 : ℝ := 240
def distance_day2 : ℝ := 420
def time_difference : ℝ := 3

-- Define the theorem
theorem average_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧
  distance_day2 / v = distance_day1 / v + time_difference ∧
  v = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l872_87219


namespace NUMINAMATH_CALUDE_find_V_l872_87279

-- Define the relationship between R, V, and W
def relationship (R V W : ℚ) : Prop :=
  ∃ c : ℚ, c ≠ 0 ∧ R * W = c * V

-- State the theorem
theorem find_V : 
  (∃ R₀ V₀ W₀ : ℚ, R₀ = 6 ∧ V₀ = 2 ∧ W₀ = 3 ∧ relationship R₀ V₀ W₀) →
  (∃ R₁ V₁ W₁ : ℚ, R₁ = 25 ∧ W₁ = 5 ∧ relationship R₁ V₁ W₁ ∧ V₁ = 125 / 9) :=
by sorry

end NUMINAMATH_CALUDE_find_V_l872_87279


namespace NUMINAMATH_CALUDE_taxi_speed_theorem_l872_87237

/-- The speed of the taxi in mph -/
def taxi_speed : ℝ := 60

/-- The speed of the bus in mph -/
def bus_speed : ℝ := taxi_speed - 30

/-- The time difference between the taxi and bus departure in hours -/
def time_difference : ℝ := 3

/-- The time it takes for the taxi to overtake the bus in hours -/
def overtake_time : ℝ := 3

theorem taxi_speed_theorem :
  taxi_speed * overtake_time = (taxi_speed - 30) * (overtake_time + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_taxi_speed_theorem_l872_87237


namespace NUMINAMATH_CALUDE_number_of_pumice_rocks_l872_87281

/-- The number of slate rocks -/
def slate_rocks : ℕ := 10

/-- The number of granite rocks -/
def granite_rocks : ℕ := 4

/-- The probability of choosing 2 slate rocks at random without replacement -/
def prob_two_slate : ℚ := 15/100

/-- The number of pumice rocks -/
def pumice_rocks : ℕ := 11

theorem number_of_pumice_rocks :
  (slate_rocks : ℚ) * (slate_rocks - 1) / 
  ((slate_rocks + pumice_rocks + granite_rocks) * (slate_rocks + pumice_rocks + granite_rocks - 1)) = 
  prob_two_slate := by sorry

end NUMINAMATH_CALUDE_number_of_pumice_rocks_l872_87281


namespace NUMINAMATH_CALUDE_shifted_parabola_vertex_l872_87236

/-- The vertex of a parabola y = 3x^2 shifted 2 units left and 3 units up is at (-2,3) -/
theorem shifted_parabola_vertex :
  let f (x : ℝ) := 3 * (x + 2)^2 + 3
  ∃! (a b : ℝ), (∀ x, f x ≥ f a) ∧ f a = b ∧ a = -2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_vertex_l872_87236


namespace NUMINAMATH_CALUDE_pages_left_to_read_l872_87282

theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read_day1 : ℕ) 
  (pages_read_day2 : ℕ) 
  (h1 : total_pages = 95) 
  (h2 : pages_read_day1 = 18) 
  (h3 : pages_read_day2 = 58) : 
  total_pages - (pages_read_day1 + pages_read_day2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l872_87282


namespace NUMINAMATH_CALUDE_words_with_vowel_count_l872_87265

/-- The set of all letters used to construct words -/
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of vowels -/
def vowels : Finset Char := {'A', 'E'}

/-- The set of consonants -/
def consonants : Finset Char := letters \ vowels

/-- The length of words we're considering -/
def wordLength : Nat := 5

/-- The number of 5-letter words with at least one vowel -/
def numWordsWithVowel : Nat :=
  letters.card ^ wordLength - consonants.card ^ wordLength

theorem words_with_vowel_count :
  numWordsWithVowel = 6752 := by
  sorry

end NUMINAMATH_CALUDE_words_with_vowel_count_l872_87265


namespace NUMINAMATH_CALUDE_clark_bought_seven_parts_l872_87278

/-- The number of parts Clark bought -/
def n : ℕ := sorry

/-- The original price of each part in dollars -/
def original_price : ℕ := 80

/-- The total amount Clark paid in dollars -/
def total_paid : ℕ := 439

/-- The total discount in dollars -/
def total_discount : ℕ := 121

theorem clark_bought_seven_parts : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_clark_bought_seven_parts_l872_87278


namespace NUMINAMATH_CALUDE_fiftieth_day_previous_year_is_wednesday_l872_87244

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : Int
  dayNumber : Nat

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

theorem fiftieth_day_previous_year_is_wednesday
  (N : Int)
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Friday)
  (h2 : dayOfWeek ⟨N + 1, 150⟩ = DayOfWeek.Friday) :
  dayOfWeek ⟨N - 1, 50⟩ = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_fiftieth_day_previous_year_is_wednesday_l872_87244


namespace NUMINAMATH_CALUDE_triangle_folding_theorem_l872_87277

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Represents a folding method for a triangle -/
structure FoldingMethod where
  apply : Triangle → ℕ

/-- Represents the result of applying a folding method to a triangle -/
structure FoldedTriangle where
  original : Triangle
  method : FoldingMethod
  layers : ℕ

/-- A folded triangle has uniform thickness if all points have the same number of layers -/
def hasUniformThickness (ft : FoldedTriangle) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ ft.original.a :: ft.original.b :: ft.original.c :: [] → 
    ft.method.apply ft.original = ft.layers

theorem triangle_folding_theorem :
  ∀ t : Triangle, ∃ fm : FoldingMethod, 
    let ft := FoldedTriangle.mk t fm 2020
    hasUniformThickness ft ∧ ft.layers = 2020 := by
  sorry

end NUMINAMATH_CALUDE_triangle_folding_theorem_l872_87277


namespace NUMINAMATH_CALUDE_f_iterated_four_times_l872_87257

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^2 else z^2

theorem f_iterated_four_times :
  f (f (f (f (1 + 2*I)))) = 165633 - 112896*I := by sorry

end NUMINAMATH_CALUDE_f_iterated_four_times_l872_87257


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_one_l872_87297

theorem inequality_solution_implies_a_less_than_one :
  ∀ a : ℝ, (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_one_l872_87297


namespace NUMINAMATH_CALUDE_indian_teepee_proportion_l872_87288

/-- Represents the fraction of drawings with a specific combination of person and dwelling -/
structure DrawingFraction :=
  (eskimo_teepee : ℚ)
  (eskimo_igloo : ℚ)
  (indian_igloo : ℚ)
  (indian_teepee : ℚ)

/-- The conditions given in the problem -/
def problem_conditions (df : DrawingFraction) : Prop :=
  df.eskimo_teepee + df.eskimo_igloo + df.indian_igloo + df.indian_teepee = 1 ∧
  df.indian_teepee + df.indian_igloo = 2 * (df.eskimo_teepee + df.eskimo_igloo) ∧
  df.indian_igloo = df.eskimo_teepee ∧
  df.eskimo_igloo = 3 * df.eskimo_teepee

/-- The theorem to be proved -/
theorem indian_teepee_proportion (df : DrawingFraction) :
  problem_conditions df →
  df.indian_teepee / (df.indian_teepee + df.eskimo_teepee) = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_indian_teepee_proportion_l872_87288


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l872_87296

def f (x : ℝ) := x^2 - 4*x + 3
def g (x : ℝ) := -3*x + 3

theorem quadratic_function_properties :
  (∃ (x : ℝ), g x = 0 ∧ f x = 0) ∧
  (g 0 = f 0) ∧
  (∀ (x : ℝ), f x ≥ -1) ∧
  (∃ (x : ℝ), f x = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l872_87296


namespace NUMINAMATH_CALUDE_sum_of_a_and_d_l872_87224

theorem sum_of_a_and_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_d_l872_87224


namespace NUMINAMATH_CALUDE_max_xy_collinear_vectors_l872_87222

def vector_a (x : ℝ) : ℝ × ℝ := (1, x^2)
def vector_b (y : ℝ) : ℝ × ℝ := (-2, y^2 - 2)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

theorem max_xy_collinear_vectors (x y : ℝ) :
  collinear (vector_a x) (vector_b y) →
  x * y ≤ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_xy_collinear_vectors_l872_87222


namespace NUMINAMATH_CALUDE_first_equation_is_root_multiplying_root_multiplying_with_root_two_l872_87214

/-- A quadratic equation ax^2 + bx + c = 0 is root-multiplying if it has two real roots and one root is twice the other -/
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

/-- The first part of the theorem -/
theorem first_equation_is_root_multiplying :
  is_root_multiplying 1 (-3) 2 :=
sorry

/-- The second part of the theorem -/
theorem root_multiplying_with_root_two (a b : ℝ) :
  is_root_multiplying a b (-6) ∧ (∃ x : ℝ, a * x^2 + b * x - 6 = 0 ∧ x = 2) →
  (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_root_multiplying_root_multiplying_with_root_two_l872_87214


namespace NUMINAMATH_CALUDE_lassis_and_smoothies_count_l872_87272

/-- Represents the number of lassis that can be made from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (15 * mangoes) / 3

/-- Represents the number of smoothies that can be made from given numbers of mangoes and bananas -/
def smoothies_from_ingredients (mangoes bananas : ℕ) : ℕ :=
  min mangoes (bananas / 2)

/-- Theorem stating the number of lassis and smoothies that can be made -/
theorem lassis_and_smoothies_count :
  lassis_from_mangoes 18 = 90 ∧ smoothies_from_ingredients 18 36 = 18 :=
by sorry

end NUMINAMATH_CALUDE_lassis_and_smoothies_count_l872_87272


namespace NUMINAMATH_CALUDE_solutions_of_equation_l872_87223

theorem solutions_of_equation : 
  {z : ℂ | z^6 - 9*z^3 + 8 = 0} = {2, 1} := by sorry

end NUMINAMATH_CALUDE_solutions_of_equation_l872_87223


namespace NUMINAMATH_CALUDE_game_draw_probability_l872_87230

theorem game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) :
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_game_draw_probability_l872_87230


namespace NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_approx_42_l872_87228

/-- The diameter of a circular field given the fencing cost per meter and total fencing cost -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- Proof that the diameter of the circular field is approximately 42 meters -/
theorem circular_field_diameter_approx_42 :
  ∃ ε > 0, |circular_field_diameter 5 659.73 - 42| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_approx_42_l872_87228


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l872_87215

/-- Given a quadratic function f(x) = x^2 + ax + b - 3 that passes through the point (2,0),
    the minimum value of a^2 + b^2 is 1. -/
theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b - 3 = 0 → x = 2) → 
  (∃ m : ℝ, ∀ a' b' : ℝ, (∀ x : ℝ, x^2 + a'*x + b' - 3 = 0 → x = 2) → a'^2 + b'^2 ≥ m) ∧
  (a^2 + b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l872_87215


namespace NUMINAMATH_CALUDE_inequality_proof_l872_87205

theorem inequality_proof (a b c x : ℝ) :
  (a + c) / 2 - (1 / 2) * Real.sqrt ((a - c)^2 + b^2) ≤ 
  a * (Real.cos x)^2 + b * Real.cos x * Real.sin x + c * (Real.sin x)^2 ∧
  a * (Real.cos x)^2 + b * Real.cos x * Real.sin x + c * (Real.sin x)^2 ≤ 
  (a + c) / 2 + (1 / 2) * Real.sqrt ((a - c)^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l872_87205


namespace NUMINAMATH_CALUDE_exists_decreasing_then_increasing_not_exists_increasing_then_decreasing_l872_87239

-- Define the sequence type
def PowerSumSequence (originalNumbers : List ℝ) : ℕ → ℝ :=
  λ n => (originalNumbers.map (λ x => x ^ n)).sum

-- Theorem for part (a)
theorem exists_decreasing_then_increasing :
  ∃ (originalNumbers : List ℝ),
    (∀ x ∈ originalNumbers, x > 0) ∧
    (let a := PowerSumSequence originalNumbers
     a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 > a 5 ∧
     ∀ n ≥ 5, a n < a (n + 1)) := by
  sorry

-- Theorem for part (b)
theorem not_exists_increasing_then_decreasing :
  ¬ ∃ (originalNumbers : List ℝ),
    (∀ x ∈ originalNumbers, x > 0) ∧
    (let a := PowerSumSequence originalNumbers
     a 1 < a 2 ∧ a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5 ∧
     ∀ n ≥ 5, a n > a (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_decreasing_then_increasing_not_exists_increasing_then_decreasing_l872_87239


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l872_87283

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 588 →
  width = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l872_87283


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l872_87284

def complex (a b : ℝ) := a + b * Complex.I

theorem condition_necessary_not_sufficient :
  ∃ a b : ℝ, (complex a b)^2 = 2 * Complex.I ∧ (a ≠ 1 ∨ b ≠ 1) ∧
  ∀ a b : ℝ, (complex a b)^2 = 2 * Complex.I → a = 1 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l872_87284


namespace NUMINAMATH_CALUDE_curve_equation_k_value_l872_87229

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) = 4

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define the perpendicularity condition
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem for the equation of C
theorem curve_equation :
  ∀ x y : ℝ, C x y ↔ x^2 + y^2/4 = 1 :=
sorry

-- Theorem for the value of k
theorem k_value (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    C x₁ y₁ ∧ C x₂ y₂ ∧
    Line k x₁ y₁ ∧ Line k x₂ y₂ ∧
    Perpendicular x₁ y₁ x₂ y₂) →
  k = 1/2 ∨ k = -1/2 :=
sorry

end NUMINAMATH_CALUDE_curve_equation_k_value_l872_87229


namespace NUMINAMATH_CALUDE_integer_decimal_parts_of_2_plus_sqrt_6_l872_87276

theorem integer_decimal_parts_of_2_plus_sqrt_6 :
  let x := Int.floor (2 + Real.sqrt 6)
  let y := (2 + Real.sqrt 6) - x
  (x = 4 ∧ y = Real.sqrt 6 - 2 ∧ Real.sqrt (x - 1) = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_of_2_plus_sqrt_6_l872_87276


namespace NUMINAMATH_CALUDE_min_value_expression_l872_87270

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) ≥ 24 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 6 ∧ (9 / a₀ + 16 / b₀ + 25 / c₀) = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l872_87270


namespace NUMINAMATH_CALUDE_second_division_divisor_l872_87293

theorem second_division_divisor (x y : ℕ) (h1 : x > 0) (h2 : x % 10 = 3) (h3 : x / 10 = y)
  (h4 : ∃ k : ℕ, (2 * x) % k = 1 ∧ (2 * x) / k = 3 * y) (h5 : 11 * y - x = 2) :
  ∃ k : ℕ, (2 * x) % k = 1 ∧ (2 * x) / k = 3 * y ∧ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_second_division_divisor_l872_87293


namespace NUMINAMATH_CALUDE_root_in_interval_l872_87273

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  ∃ (root : ℝ), root ∈ Set.Icc 2 2.5 ∧ f root = 0 :=
by
  have h1 : f 2 < 0 := by sorry
  have h2 : f 2.5 > 0 := by sorry
  have h3 : f 3 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l872_87273


namespace NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l872_87225

/-- The number of color options for shorts -/
def shorts_colors : ℕ := 3

/-- The number of color options for jerseys -/
def jersey_colors : ℕ := 4

/-- The total number of possible combinations of shorts and jerseys -/
def total_combinations : ℕ := shorts_colors * jersey_colors

/-- The number of combinations where shorts and jerseys have different colors -/
def different_color_combinations : ℕ := shorts_colors * (jersey_colors - 1)

/-- The probability of choosing different colors for shorts and jersey -/
def prob_different_colors : ℚ := different_color_combinations / total_combinations

/-- Theorem stating that the probability of choosing different colors for shorts and jersey is 3/4 -/
theorem prob_different_colors_is_three_fourths : prob_different_colors = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l872_87225


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l872_87240

/-- Given a geometric sequence with first term 2 and fifth term 18, the third term is 6 -/
theorem geometric_sequence_third_term :
  ∀ (x y z : ℝ), 
  (∃ q : ℝ, q ≠ 0 ∧ x = 2 * q ∧ y = 2 * q^2 ∧ z = 2 * q^3 ∧ 18 = 2 * q^4) →
  y = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l872_87240


namespace NUMINAMATH_CALUDE_mark_kate_difference_l872_87221

/-- Represents the hours charged by each person on the project -/
structure ProjectHours where
  kate : ℝ
  pat : ℝ
  ravi : ℝ
  sarah : ℝ
  mark : ℝ

/-- Defines the conditions of the project hours problem -/
def validProjectHours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.ravi = 1.5 * h.kate ∧
  h.sarah = 4 * h.ravi ∧
  h.sarah = 2/3 * h.mark ∧
  h.kate + h.pat + h.ravi + h.sarah + h.mark = 310

/-- Theorem stating the difference between Mark's and Kate's hours -/
theorem mark_kate_difference (h : ProjectHours) (hvalid : validProjectHours h) :
  ∃ ε > 0, |h.mark - h.kate - 127.2| < ε :=
sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l872_87221


namespace NUMINAMATH_CALUDE_optimal_rate_maximizes_income_l872_87262

/-- Represents the hotel's room pricing and occupancy model -/
structure HotelModel where
  totalRooms : ℕ
  baseRate : ℕ
  occupancyDecrease : ℕ
  rateIncrease : ℕ

/-- Calculates the number of occupied rooms based on the new rate -/
def occupiedRooms (model : HotelModel) (newRate : ℕ) : ℤ :=
  model.totalRooms - (newRate - model.baseRate) / model.rateIncrease * model.occupancyDecrease

/-- Calculates the total daily income based on the new rate -/
def dailyIncome (model : HotelModel) (newRate : ℕ) : ℕ :=
  newRate * (occupiedRooms model newRate).toNat

/-- The optimal rate that maximizes daily income -/
def optimalRate (model : HotelModel) : ℕ := model.baseRate + model.rateIncrease * (model.totalRooms / model.occupancyDecrease) / 2

/-- Theorem stating that the optimal rate maximizes daily income -/
theorem optimal_rate_maximizes_income (model : HotelModel) :
  model.totalRooms = 300 →
  model.baseRate = 200 →
  model.occupancyDecrease = 10 →
  model.rateIncrease = 20 →
  ∀ rate, dailyIncome model (optimalRate model) ≥ dailyIncome model rate := by
  sorry

#eval optimalRate { totalRooms := 300, baseRate := 200, occupancyDecrease := 10, rateIncrease := 20 }
#eval dailyIncome { totalRooms := 300, baseRate := 200, occupancyDecrease := 10, rateIncrease := 20 } 400

end NUMINAMATH_CALUDE_optimal_rate_maximizes_income_l872_87262


namespace NUMINAMATH_CALUDE_equation_solution_l872_87269

theorem equation_solution : 
  ∃! x : ℚ, (x - 17) / 3 = (3 * x + 4) / 8 ∧ x = -148 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l872_87269


namespace NUMINAMATH_CALUDE_hexagon_side_count_l872_87271

/-- A convex hexagon with two distinct side lengths -/
structure ConvexHexagon where
  side1 : ℕ  -- Length of the first type of side
  side2 : ℕ  -- Length of the second type of side
  count1 : ℕ -- Number of sides with length side1
  count2 : ℕ -- Number of sides with length side2
  distinct : side1 ≠ side2
  total_sides : count1 + count2 = 6
  perimeter : side1 * count1 + side2 * count2 = 38

theorem hexagon_side_count (h : ConvexHexagon) (h_side1 : h.side1 = 7) (h_side2 : h.side2 = 4) :
  h.count2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_count_l872_87271


namespace NUMINAMATH_CALUDE_polynomial_expansion_l872_87298

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l872_87298


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l872_87206

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define the points on the graph
def y₁ : ℝ := f 1
def y₂ : ℝ := f 2
def y₃ : ℝ := f (-2)

-- Theorem statement
theorem quadratic_points_relationship : y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l872_87206


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l872_87285

/-- A quadratic function f(x) = x^2 + 2ax - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 3

/-- The condition a > -1 is sufficient but not necessary for f to be monotonically increasing on (1, +∞) -/
theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > -1 → ∀ x y, 1 < x → x < y → f a x < f a y) ∧
  ¬(∀ x y, 1 < x → x < y → f a x < f a y → a > -1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l872_87285


namespace NUMINAMATH_CALUDE_power_sum_inequality_l872_87231

theorem power_sum_inequality (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l872_87231


namespace NUMINAMATH_CALUDE_ice_cream_sales_for_games_l872_87243

/-- The number of ice creams needed to be sold to buy two games -/
def ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) : ℕ :=
  2 * game_cost / ice_cream_price

/-- Proof that 24 ice creams are needed to buy two $60 games when each ice cream is $5 -/
theorem ice_cream_sales_for_games : ice_creams_needed 60 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_for_games_l872_87243


namespace NUMINAMATH_CALUDE_closest_point_is_correct_l872_87220

/-- The point on the line y = -4x - 8 that is closest to (3, 6) -/
def closest_point : ℚ × ℚ := (-53/17, 76/17)

/-- The line y = -4x - 8 -/
def mouse_trajectory (x : ℚ) : ℚ := -4 * x - 8

theorem closest_point_is_correct :
  let (a, b) := closest_point
  -- The point is on the line
  (mouse_trajectory a = b) ∧
  -- It's the closest point to (3, 6)
  (∀ x y, mouse_trajectory x = y →
    (x - 3)^2 + (y - 6)^2 ≥ (a - 3)^2 + (b - 6)^2) ∧
  -- The sum of its coordinates is 23/17
  (a + b = 23/17) := by sorry


end NUMINAMATH_CALUDE_closest_point_is_correct_l872_87220


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l872_87250

theorem oak_trees_in_park (initial_trees : ℕ) (planted_trees : ℕ) : initial_trees = 5 → planted_trees = 4 → initial_trees + planted_trees = 9 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l872_87250


namespace NUMINAMATH_CALUDE_resulting_angle_25_2_5_turns_l872_87256

/-- Given an initial angle and a number of clockwise turns, calculate the resulting angle -/
def resulting_angle (initial_angle : ℝ) (clockwise_turns : ℝ) : ℝ :=
  initial_angle - 360 * clockwise_turns

/-- Theorem: The resulting angle after rotating 25° clockwise by 2.5 turns is -875° -/
theorem resulting_angle_25_2_5_turns :
  resulting_angle 25 2.5 = -875 := by
  sorry

end NUMINAMATH_CALUDE_resulting_angle_25_2_5_turns_l872_87256


namespace NUMINAMATH_CALUDE_wage_increase_hours_reduction_l872_87234

/-- Proves that when an employee's hourly wage increases by 10% and they want to maintain
    the same total weekly income, the percent reduction in hours worked is (1 - 1/1.10) * 100% -/
theorem wage_increase_hours_reduction (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let new_wage := 1.1 * w
  let new_hours := h * w / new_wage
  let percent_reduction := (h - new_hours) / h * 100
  percent_reduction = (1 - 1 / 1.1) * 100 := by
  sorry


end NUMINAMATH_CALUDE_wage_increase_hours_reduction_l872_87234


namespace NUMINAMATH_CALUDE_even_iff_mod_two_eq_zero_l872_87294

theorem even_iff_mod_two_eq_zero (x : Int) : Even x ↔ x % 2 = 0 := by sorry

end NUMINAMATH_CALUDE_even_iff_mod_two_eq_zero_l872_87294


namespace NUMINAMATH_CALUDE_all_natural_numbers_have_P_structure_l872_87212

/-- The set of all squares of positive integers -/
def P : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^2}

/-- A number n has a P structure if it can be expressed as a sum of some distinct elements from P -/
def has_P_structure (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ s ∈ S, s ∈ P) ∧ (S.sum id = n)

/-- Every natural number has a P structure -/
theorem all_natural_numbers_have_P_structure :
  ∀ n : ℕ, has_P_structure n :=
sorry

end NUMINAMATH_CALUDE_all_natural_numbers_have_P_structure_l872_87212


namespace NUMINAMATH_CALUDE_betty_strawberries_l872_87241

/-- Proves that Betty picked 16 strawberries given the conditions of the problem -/
theorem betty_strawberries : ∃ (B N : ℕ),
  let M := B + 20
  let total_strawberries := B + M + N
  let jars := 40 / 4
  let strawberries_per_jar := 7
  B + 20 = 2 * N ∧
  total_strawberries = jars * strawberries_per_jar ∧
  B = 16 := by
  sorry


end NUMINAMATH_CALUDE_betty_strawberries_l872_87241


namespace NUMINAMATH_CALUDE_driver_weekly_distance_l872_87289

def weekday_distance (speed1 speed2 speed3 time1 time2 time3 : ℕ) : ℕ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3

def sunday_distance (speed time : ℕ) : ℕ :=
  speed * time

def weekly_distance (weekday_dist sunday_dist days_per_week : ℕ) : ℕ :=
  weekday_dist * days_per_week + sunday_dist

theorem driver_weekly_distance :
  let weekday_dist := weekday_distance 30 25 40 3 4 2
  let sunday_dist := sunday_distance 35 5
  weekly_distance weekday_dist sunday_dist 6 = 1795 := by sorry

end NUMINAMATH_CALUDE_driver_weekly_distance_l872_87289


namespace NUMINAMATH_CALUDE_inequality_solution_l872_87268

def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then Set.univ
  else if m > 0 then {x | -3/m < x ∧ x < 1/m}
  else {x | 1/m < x ∧ x < -3/m}

theorem inequality_solution (m : ℝ) :
  {x : ℝ | m^2 * x^2 + 2*m*x - 3 < 0} = solution_set m :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l872_87268


namespace NUMINAMATH_CALUDE_candy_bar_count_l872_87204

theorem candy_bar_count (bags : ℕ) (candy_per_bag : ℕ) (h1 : bags = 5) (h2 : candy_per_bag = 3) :
  bags * candy_per_bag = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_count_l872_87204


namespace NUMINAMATH_CALUDE_side_length_is_seven_l872_87202

noncomputable def triangle_side_length (a c : ℝ) (B : ℝ) : ℝ :=
  Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))

theorem side_length_is_seven :
  let a : ℝ := 3 * Real.sqrt 3
  let c : ℝ := 2
  let B : ℝ := 150 * π / 180
  triangle_side_length a c B = 7 := by
sorry

end NUMINAMATH_CALUDE_side_length_is_seven_l872_87202


namespace NUMINAMATH_CALUDE_feathers_count_l872_87274

/-- The number of animals in the first group -/
def group1_animals : ℕ := 934

/-- The number of feathers in crowns for the first group -/
def group1_feathers : ℕ := 7

/-- The number of animals in the second group -/
def group2_animals : ℕ := 425

/-- The number of colored feathers in crowns for the second group -/
def group2_colored_feathers : ℕ := 7

/-- The number of golden feathers in crowns for the second group -/
def group2_golden_feathers : ℕ := 5

/-- The number of animals in the third group -/
def group3_animals : ℕ := 289

/-- The number of colored feathers in crowns for the third group -/
def group3_colored_feathers : ℕ := 4

/-- The number of golden feathers in crowns for the third group -/
def group3_golden_feathers : ℕ := 10

/-- The total number of feathers needed for all animals -/
def total_feathers : ℕ := 15684

theorem feathers_count :
  group1_animals * group1_feathers +
  group2_animals * (group2_colored_feathers + group2_golden_feathers) +
  group3_animals * (group3_colored_feathers + group3_golden_feathers) =
  total_feathers := by
  sorry

end NUMINAMATH_CALUDE_feathers_count_l872_87274


namespace NUMINAMATH_CALUDE_no_overlap_in_intervals_l872_87290

theorem no_overlap_in_intervals (x : ℝ) : 
  50 ≤ x ∧ x ≤ 150 ∧ Int.floor (Real.sqrt x) = 11 → 
  Int.floor (Real.sqrt (50 * x)) ≠ 110 := by
sorry

end NUMINAMATH_CALUDE_no_overlap_in_intervals_l872_87290


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l872_87252

theorem triangle_angle_sum 
  (A B C : ℝ) 
  (h_acute_A : 0 < A ∧ A < π/2) 
  (h_acute_B : 0 < B ∧ B < π/2)
  (h_sin_A : Real.sin A = Real.sqrt 5 / 5)
  (h_sin_B : Real.sin B = Real.sqrt 10 / 10)
  : Real.cos (A + B) = Real.sqrt 2 / 2 ∧ C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l872_87252


namespace NUMINAMATH_CALUDE_correct_operation_l872_87259

theorem correct_operation (a : ℝ) : 3 * a^2 - 4 * a^2 = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l872_87259


namespace NUMINAMATH_CALUDE_f_is_odd_l872_87242

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Theorem: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l872_87242


namespace NUMINAMATH_CALUDE_symmetrical_line_sum_l872_87245

/-- Given a line y = mx + b that is symmetrical to the line x - 3y + 11 = 0
    with respect to the x-axis, prove that m + b = -4 -/
theorem symmetrical_line_sum (m b : ℝ) : 
  (∀ x y, y = m * x + b ↔ x + 3 * y + 11 = 0) → m + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_line_sum_l872_87245


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l872_87238

/-- Calculates the pass percentage for a group of students -/
def passPercentage (totalStudents : ℕ) (passedStudents : ℕ) : ℚ :=
  (passedStudents : ℚ) / (totalStudents : ℚ) * 100

theorem exam_pass_percentage :
  let set1 := 40
  let set2 := 50
  let set3 := 60
  let pass1 := 40  -- 100% of 40
  let pass2 := 45  -- 90% of 50
  let pass3 := 48  -- 80% of 60
  let totalStudents := set1 + set2 + set3
  let totalPassed := pass1 + pass2 + pass3
  abs (passPercentage totalStudents totalPassed - 88.67) < 0.01 := by
  sorry

#eval passPercentage (40 + 50 + 60) (40 + 45 + 48)

end NUMINAMATH_CALUDE_exam_pass_percentage_l872_87238


namespace NUMINAMATH_CALUDE_calculation_proof_l872_87253

theorem calculation_proof : 5^2 * 3 + (7 * 2 - 15) / 3 = 74 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l872_87253


namespace NUMINAMATH_CALUDE_largest_difference_l872_87203

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 1005^1006)
  (hB : B = 1005^1006)
  (hC : C = 1004 * 1005^1005)
  (hD : D = 3 * 1005^1005)
  (hE : E = 1005^1005)
  (hF : F = 1005^1004) :
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l872_87203


namespace NUMINAMATH_CALUDE_square_sum_eq_841_times_product_plus_one_l872_87248

theorem square_sum_eq_841_times_product_plus_one :
  ∀ a b : ℕ, a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_eq_841_times_product_plus_one_l872_87248


namespace NUMINAMATH_CALUDE_prob_at_least_as_many_females_l872_87232

/-- The probability of selecting at least as many females as males when randomly choosing 2 students from a group of 5 students with 2 females and 3 males is 7/10. -/
theorem prob_at_least_as_many_females (total : ℕ) (females : ℕ) (males : ℕ) :
  total = 5 →
  females = 2 →
  males = 3 →
  females + males = total →
  (Nat.choose total 2 : ℚ) ≠ 0 →
  (Nat.choose females 2 + Nat.choose females 1 * Nat.choose males 1 : ℚ) / Nat.choose total 2 = 7 / 10 := by
  sorry

#check prob_at_least_as_many_females

end NUMINAMATH_CALUDE_prob_at_least_as_many_females_l872_87232


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_not_p_or_not_q_l872_87261

theorem not_p_and_q_implies_not_p_or_not_q (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_not_p_or_not_q_l872_87261


namespace NUMINAMATH_CALUDE_trajectory_E_equation_max_area_AMBN_l872_87258

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P on circle C
def P (x y : ℝ) : Prop := C x y

-- Define the point H as the foot of the perpendicular from P to x-axis
def H (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point Q
def Q (x y : ℝ) : Prop := ∃ (px py : ℝ), P px py ∧ x = (px + (H px).1) / 2 ∧ y = (py + (H px).2) / 2

-- Define the trajectory E
def E (x y : ℝ) : Prop := Q x y

-- Define the line y = kx
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x ∧ k > 0

-- Theorem for the equation of trajectory E
theorem trajectory_E_equation : ∀ x y : ℝ, E x y ↔ x^2/4 + y^2 = 1 :=
sorry

-- Theorem for the maximum area of quadrilateral AMBN
theorem max_area_AMBN : ∃ (max_area : ℝ), 
  (∀ k x1 y1 x2 y2 : ℝ, E x1 y1 ∧ E x2 y2 ∧ Line k x1 y1 ∧ Line k x2 y2 → 
    abs (x1 * y2 - x2 * y1) ≤ max_area) ∧
  max_area = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_E_equation_max_area_AMBN_l872_87258


namespace NUMINAMATH_CALUDE_square_pyramid_volume_l872_87211

/-- The volume of a regular square pyramid with base edge length 1 and height 3 is 1 -/
theorem square_pyramid_volume :
  let base_edge : ℝ := 1
  let height : ℝ := 3
  let base_area : ℝ := base_edge ^ 2
  let volume : ℝ := (1 / 3) * base_area * height
  volume = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_volume_l872_87211


namespace NUMINAMATH_CALUDE_trigonometric_equality_l872_87217

theorem trigonometric_equality : 2 * Real.tan (π / 3) + Real.tan (π / 4) - 4 * Real.cos (π / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l872_87217


namespace NUMINAMATH_CALUDE_series_end_probability_l872_87260

/-- Probability of Mathletes winning a single game -/
def p : ℚ := 2/3

/-- Probability of the opponent winning a single game -/
def q : ℚ := 1 - p

/-- Number of games in the series before the final game -/
def n : ℕ := 6

/-- Number of wins required to end the series -/
def k : ℕ := 5

/-- Probability of the series ending in exactly 7 games -/
def prob_series_end_7 : ℚ := 
  (Nat.choose n (k-1)) * (p^(k-1) * q^(n-(k-1)) * p + p^(n-(k-1)) * q^(k-1) * q)

theorem series_end_probability :
  prob_series_end_7 = 20/81 := by
  sorry

end NUMINAMATH_CALUDE_series_end_probability_l872_87260


namespace NUMINAMATH_CALUDE_pauline_car_count_l872_87291

/-- Represents the total number of matchbox cars Pauline has. -/
def total_cars : ℕ := 125

/-- Represents the number of convertible cars Pauline has. -/
def convertibles : ℕ := 35

/-- Represents the percentage of regular cars as a rational number. -/
def regular_cars_percent : ℚ := 64 / 100

/-- Represents the percentage of trucks as a rational number. -/
def trucks_percent : ℚ := 8 / 100

/-- Theorem stating that given the conditions, Pauline has 125 matchbox cars in total. -/
theorem pauline_car_count : 
  (regular_cars_percent + trucks_percent) * total_cars + convertibles = total_cars :=
sorry

end NUMINAMATH_CALUDE_pauline_car_count_l872_87291


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l872_87201

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 998 ∧ 
    100 ≤ n ∧ n < 1000 ∧ 
    (70 * n) % 350 = 210 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ (70 * m) % 350 = 210 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l872_87201


namespace NUMINAMATH_CALUDE_dice_sum_not_23_l872_87264

theorem dice_sum_not_23 (a b c d e : ℕ) : 
  a ≥ 1 ∧ a ≤ 6 ∧
  b ≥ 1 ∧ b ≤ 6 ∧
  c ≥ 1 ∧ c ≤ 6 ∧
  d ≥ 1 ∧ d ≤ 6 ∧
  e ≥ 1 ∧ e ≤ 6 ∧
  a * b * c * d * e = 720 →
  a + b + c + d + e ≠ 23 :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_not_23_l872_87264


namespace NUMINAMATH_CALUDE_sum_of_numbers_l872_87207

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 375) (h4 : 1 / x + 1 / y = 0.10666666666666667) : 
  x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l872_87207


namespace NUMINAMATH_CALUDE_probability_of_seven_in_three_elevenths_l872_87210

-- Define the fraction
def fraction : ℚ := 3 / 11

-- Define the decimal representation as a sequence of digits
def decimal_representation : ℕ → ℕ
  | 0 => 0  -- The digit before the decimal point
  | (n + 1) => if n % 2 = 0 then 2 else 7  -- The repeating pattern 27

-- Define the probability of selecting a 7
def probability_of_seven : ℚ := 1 / 2

-- Theorem statement
theorem probability_of_seven_in_three_elevenths :
  (∃ (n : ℕ), decimal_representation n = 7) ∧ 
  (∀ (m : ℕ), m ≠ 0 → decimal_representation m = decimal_representation (m + 2)) →
  probability_of_seven = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_seven_in_three_elevenths_l872_87210


namespace NUMINAMATH_CALUDE_stating_circle_implies_a_eq_neg_one_l872_87226

/-- 
A function that represents the equation of a potential circle.
-/
def potential_circle (a : ℝ) (x y : ℝ) : ℝ :=
  x^2 + (a + 2) * y^2 + 2 * a * x + a

/-- 
A predicate that determines if an equation represents a circle.
This is a simplified representation and may need to be adjusted based on the specific criteria for a circle.
-/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), f x y = (x - h)^2 + (y - k)^2 - r^2

/-- 
Theorem stating that if the given equation represents a circle, then a = -1.
-/
theorem circle_implies_a_eq_neg_one :
  is_circle (potential_circle a) → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_stating_circle_implies_a_eq_neg_one_l872_87226


namespace NUMINAMATH_CALUDE_worker_problem_l872_87208

/-- The number of workers in the problem -/
def num_workers : ℕ := 30

/-- The total amount of money -/
def total_money : ℤ := 5 * num_workers + 30

theorem worker_problem :
  (total_money - 5 * num_workers = 30) ∧
  (total_money - 7 * num_workers = -30) :=
by sorry

end NUMINAMATH_CALUDE_worker_problem_l872_87208


namespace NUMINAMATH_CALUDE_jack_marbles_remaining_l872_87233

/-- 
Given that Jack starts with a certain number of marbles and shares some with Rebecca,
this theorem proves how many marbles Jack ends up with.
-/
theorem jack_marbles_remaining (initial : ℕ) (shared : ℕ) (remaining : ℕ) 
  (h1 : initial = 62)
  (h2 : shared = 33)
  (h3 : remaining = initial - shared) : 
  remaining = 29 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_remaining_l872_87233


namespace NUMINAMATH_CALUDE_circle_line_intersection_l872_87249

theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 16 ∧ p.1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l872_87249


namespace NUMINAMATH_CALUDE_work_completion_l872_87213

/-- The number of men in the first group that can complete a work in 18 days, 
    working 7 hours a day, given that 12 men can complete the same work in 12 days,
    also working 7 hours a day. -/
def number_of_men : ℕ := 8

theorem work_completion :
  ∀ (hours_per_day : ℕ) (days_first_group : ℕ) (days_second_group : ℕ),
    hours_per_day > 0 →
    days_first_group > 0 →
    days_second_group > 0 →
    number_of_men * hours_per_day * days_first_group = 12 * hours_per_day * days_second_group →
    hours_per_day = 7 →
    days_first_group = 18 →
    days_second_group = 12 →
    number_of_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l872_87213


namespace NUMINAMATH_CALUDE_right_triangle_area_l872_87286

theorem right_triangle_area (a b c : ℝ) (h1 : b = (2/3) * a) (h2 : b = (2/3) * c) 
  (h3 : a^2 + b^2 = c^2) : (1/2) * a * b = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l872_87286


namespace NUMINAMATH_CALUDE_difference_between_B_and_C_l872_87209

theorem difference_between_B_and_C (A B C : ℤ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h2 : C - 8 = 1)
  (h3 : A + 5 = B)
  (h4 : A = 9 - 4) :
  B - C = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_B_and_C_l872_87209


namespace NUMINAMATH_CALUDE_sphere_volume_from_intersection_l872_87263

/-- Given a sphere intersected by a plane at distance 1 from its center,
    creating a cross-sectional area of π, prove that its volume is (8√2π)/3. -/
theorem sphere_volume_from_intersection (r : ℝ) : 
  (r^2 - 1^2 = 1^2) →   -- Pythagorean theorem for the right triangle
  (π * 1^2 = π) →       -- Cross-sectional area is π
  ((4/3) * π * r^3 = (8 * Real.sqrt 2 * π) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_intersection_l872_87263


namespace NUMINAMATH_CALUDE_remaining_amount_is_15_60_l872_87295

/-- Calculate the remaining amount for a trip given expenses and gifts --/
def calculate_remaining_amount (initial_amount gas_cost lunch_cost gift_cost_per_person num_people extra_gift_cost grandma_gift toll_fee ice_cream_cost : ℚ) : ℚ :=
  let total_spent := gas_cost + lunch_cost + (gift_cost_per_person * num_people) + extra_gift_cost
  let total_received := initial_amount + (grandma_gift * num_people)
  let amount_before_return := total_received - total_spent
  amount_before_return - (toll_fee + ice_cream_cost)

/-- Theorem stating that the remaining amount for the return trip is $15.60 --/
theorem remaining_amount_is_15_60 :
  calculate_remaining_amount 60 12 23.40 5 3 7 10 8 9 = 15.60 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_is_15_60_l872_87295


namespace NUMINAMATH_CALUDE_tangent_line_slope_l872_87287

/-- A line passing through the origin and tangent to the circle (x - √3)² + (y - 1)² = 1 has a slope of either 0 or √3 -/
theorem tangent_line_slope :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = k * x ∧ (x - Real.sqrt 3)^2 + (y - 1)^2 = 1) →
  (k = 0 ∨ k = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l872_87287


namespace NUMINAMATH_CALUDE_james_cryptocurrency_investment_l872_87235

theorem james_cryptocurrency_investment (C : ℕ) : 
  (C * 15 = 12 * (15 + 15 * (2/3))) → C = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_cryptocurrency_investment_l872_87235


namespace NUMINAMATH_CALUDE_expression_simplification_l872_87251

theorem expression_simplification (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l872_87251


namespace NUMINAMATH_CALUDE_fraction_problem_l872_87292

theorem fraction_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l872_87292


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l872_87218

/-- Represents a trapezoid ABCD with side lengths AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem stating that if the area ratio of triangles ABC to ADC is 8:2
    and AB + CD = 250, then AB = 200 -/
theorem trapezoid_segment_length (t : Trapezoid) 
    (h1 : (t.AB / t.CD) = 4)  -- Ratio of areas is equivalent to ratio of bases
    (h2 : t.AB + t.CD = 250) : 
  t.AB = 200 := by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l872_87218


namespace NUMINAMATH_CALUDE_train_speed_calculation_l872_87255

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length1 length2 speed1 time_to_cross : ℝ) :
  length1 = 280 →
  length2 = 220.04 →
  speed1 = 120 →
  time_to_cross = 9 →
  ∃ speed2 : ℝ, 
    (length1 + length2) = (speed1 + speed2) * (5 / 18) * time_to_cross ∧ 
    abs (speed2 - 80.016) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l872_87255


namespace NUMINAMATH_CALUDE_september_first_is_wednesday_l872_87299

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The number of lessons for each day of the week -/
def lessonsPerDay (d : DayOfWeek) : Nat :=
  match d with
  | .Monday => 1
  | .Tuesday => 2
  | .Wednesday => 3
  | .Thursday => 4
  | .Friday => 5
  | .Saturday => 0
  | .Sunday => 0

/-- The total number of lessons in a week -/
def lessonsPerWeek : Nat :=
  (lessonsPerDay .Monday) +
  (lessonsPerDay .Tuesday) +
  (lessonsPerDay .Wednesday) +
  (lessonsPerDay .Thursday) +
  (lessonsPerDay .Friday) +
  (lessonsPerDay .Saturday) +
  (lessonsPerDay .Sunday)

/-- The function to determine the day of the week for September 1 -/
def septemberFirstDay (totalLessons : Nat) : DayOfWeek :=
  sorry

/-- The theorem stating that September 1 falls on a Wednesday -/
theorem september_first_is_wednesday :
  septemberFirstDay 64 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_september_first_is_wednesday_l872_87299
