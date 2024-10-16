import Mathlib

namespace NUMINAMATH_CALUDE_log_inequality_range_l4082_408271

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the range set
def range_set : Set ℝ := {x | x ∈ Set.Ioc 0 (1/8) ∪ Set.Ici 8}

-- State the theorem
theorem log_inequality_range :
  ∀ x > 0, Complex.abs (log_half x - (0 : ℝ) + 4*Complex.I) ≥ Complex.abs (3 + 4*Complex.I) ↔ x ∈ range_set :=
sorry

end NUMINAMATH_CALUDE_log_inequality_range_l4082_408271


namespace NUMINAMATH_CALUDE_total_interest_is_1800_l4082_408249

/-- Calculates the total interest over 10 years when the principal is trebled after 5 years -/
def totalInterest (P R : ℚ) : ℚ :=
  let firstHalfInterest := (P * R * 5) / 100
  let secondHalfInterest := (3 * P * R * 5) / 100
  firstHalfInterest + secondHalfInterest

/-- Theorem stating that the total interest is 1800 given the problem conditions -/
theorem total_interest_is_1800 (P R : ℚ) 
    (h : (P * R * 10) / 100 = 900) : totalInterest P R = 1800 := by
  sorry

#eval totalInterest 1000 9  -- This should evaluate to 1800

end NUMINAMATH_CALUDE_total_interest_is_1800_l4082_408249


namespace NUMINAMATH_CALUDE_store_a_more_cost_effective_for_large_x_l4082_408269

/-- Represents the cost of purchasing table tennis rackets from Store A -/
def cost_store_a (x : ℕ) : ℚ :=
  if x ≤ 10 then 30 * x else 300 + 21 * (x - 10)

/-- Represents the cost of purchasing table tennis rackets from Store B -/
def cost_store_b (x : ℕ) : ℚ := 25.5 * x

/-- Theorem stating that Store A is more cost-effective than Store B for x > 20 -/
theorem store_a_more_cost_effective_for_large_x :
  ∀ x : ℕ, x > 20 → cost_store_a x < cost_store_b x :=
by
  sorry

/-- Helper lemma to show that cost_store_a simplifies to 21x + 90 for x > 10 -/
lemma cost_store_a_simplification (x : ℕ) (h : x > 10) :
  cost_store_a x = 21 * x + 90 :=
by
  sorry

end NUMINAMATH_CALUDE_store_a_more_cost_effective_for_large_x_l4082_408269


namespace NUMINAMATH_CALUDE_garden_perimeter_is_60_l4082_408230

/-- A rectangular garden with given diagonal and area -/
structure RectangularGarden where
  width : ℝ
  height : ℝ
  diagonal_sq : width^2 + height^2 = 26^2
  area : width * height = 120

/-- The perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ := 2 * (g.width + g.height)

/-- Theorem: The perimeter of the given rectangular garden is 60 meters -/
theorem garden_perimeter_is_60 (g : RectangularGarden) : perimeter g = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_60_l4082_408230


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l4082_408284

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l4082_408284


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4082_408276

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ α < x ∧ x < β) 
  (h2 : α > 0) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ x < 1/β ∨ x > 1/α :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4082_408276


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l4082_408278

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 6*x + 5 ≤ 0}

-- State the theorem
theorem complement_of_M_in_U :
  (U \ M) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l4082_408278


namespace NUMINAMATH_CALUDE_a_minus_b_equals_four_l4082_408260

theorem a_minus_b_equals_four :
  ∀ (A B : ℕ),
    (A ≥ 10 ∧ A ≤ 99) →  -- A is a two-digit number
    (B ≥ 10 ∧ B ≤ 99) →  -- B is a two-digit number
    A = 23 - 8 →         -- A is 8 less than 23
    B + 7 = 18 →         -- The number that is 7 greater than B is 18
    A - B = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_four_l4082_408260


namespace NUMINAMATH_CALUDE_first_day_sale_l4082_408293

theorem first_day_sale (total_days : ℕ) (average_sale : ℕ) (known_days_sales : List ℕ) :
  total_days = 5 →
  average_sale = 625 →
  known_days_sales = [927, 855, 230, 562] →
  (total_days * average_sale) - known_days_sales.sum = 551 := by
  sorry

end NUMINAMATH_CALUDE_first_day_sale_l4082_408293


namespace NUMINAMATH_CALUDE_suraj_average_increase_l4082_408270

/-- Represents a cricket player's innings record -/
structure InningsRecord where
  initial_innings : ℕ
  new_innings_score : ℕ
  new_average : ℚ

/-- Calculates the increase in average for a given innings record -/
def average_increase (record : InningsRecord) : ℚ :=
  record.new_average - (record.new_average * (record.initial_innings + 1) - record.new_innings_score) / record.initial_innings

/-- Theorem stating that Suraj's average increased by 6 runs -/
theorem suraj_average_increase :
  let suraj_record : InningsRecord := {
    initial_innings := 16,
    new_innings_score := 112,
    new_average := 16
  }
  average_increase suraj_record = 6 := by sorry

end NUMINAMATH_CALUDE_suraj_average_increase_l4082_408270


namespace NUMINAMATH_CALUDE_shaded_area_formula_l4082_408231

/-- An equilateral triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- The triangle is inscribed in a circle -/
  inscribed : Bool
  /-- Two vertices of the triangle are endpoints of a circle diameter -/
  diameter_endpoints : Bool

/-- The shaded area outside the triangle but inside the circle -/
def shaded_area (t : InscribedTriangle) : ℝ := sorry

/-- Theorem stating the shaded area for a specific inscribed triangle -/
theorem shaded_area_formula (t : InscribedTriangle) 
  (h1 : t.side_length = 10)
  (h2 : t.inscribed = true)
  (h3 : t.diameter_endpoints = true) :
  shaded_area t = (50 * Real.pi / 3) - 25 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_formula_l4082_408231


namespace NUMINAMATH_CALUDE_rational_criterion_l4082_408273

/-- The number of different digit sequences of length n in the decimal expansion of a real number -/
def num_digit_sequences (a : ℝ) (n : ℕ) : ℕ := sorry

/-- A real number is rational if there exists a natural number n such that 
    the number of different digit sequences of length n in its decimal expansion 
    is less than or equal to n + 8 -/
theorem rational_criterion (a : ℝ) : 
  (∃ n : ℕ, num_digit_sequences a n ≤ n + 8) → ∃ q : ℚ, a = ↑q := by sorry

end NUMINAMATH_CALUDE_rational_criterion_l4082_408273


namespace NUMINAMATH_CALUDE_seating_arrangements_l4082_408274

/-- Represents a row of seats -/
structure Row :=
  (total : ℕ)
  (available : ℕ)

/-- Calculates the number of seating arrangements for two people in a single row -/
def arrangementsInRow (row : Row) : ℕ :=
  row.available * (row.available - 1)

/-- Calculates the number of seating arrangements for two people in different rows -/
def arrangementsAcrossRows (row1 row2 : Row) : ℕ :=
  row1.available * row2.available * 2

/-- The main theorem stating the total number of seating arrangements -/
theorem seating_arrangements :
  let frontRow : Row := ⟨11, 8⟩
  let backRow : Row := ⟨12, 12⟩
  arrangementsInRow frontRow + arrangementsInRow backRow + arrangementsAcrossRows frontRow backRow = 334 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l4082_408274


namespace NUMINAMATH_CALUDE_greatest_integer_solution_two_satisfies_inequality_three_exceeds_inequality_greatest_integer_value_l4082_408238

theorem greatest_integer_solution (x : ℤ) : x^2 + 5*x < 30 → x ≤ 2 :=
by
  sorry

theorem two_satisfies_inequality : 2^2 + 5*2 < 30 :=
by
  sorry

theorem three_exceeds_inequality : ¬(3^2 + 5*3 < 30) :=
by
  sorry

theorem greatest_integer_value : ∃ (x : ℤ), x^2 + 5*x < 30 ∧ ∀ (y : ℤ), y^2 + 5*y < 30 → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_two_satisfies_inequality_three_exceeds_inequality_greatest_integer_value_l4082_408238


namespace NUMINAMATH_CALUDE_max_value_on_curve_l4082_408259

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := 3*x + y

-- Theorem statement
theorem max_value_on_curve :
  ∃ (M : ℝ), M = 2 * Real.sqrt 3 ∧
  (∀ (x y : ℝ), C x y → f x y ≤ M) ∧
  (∃ (x y : ℝ), C x y ∧ f x y = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l4082_408259


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l4082_408280

theorem complex_number_imaginary_part (i : ℂ) (a : ℝ) :
  i * i = -1 →
  let z := (1 - a * i) / (1 + i)
  Complex.im z = -3 →
  a = 5 := by sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l4082_408280


namespace NUMINAMATH_CALUDE_pet_store_birds_l4082_408252

theorem pet_store_birds (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) (dogs : ℕ) :
  total_animals = 180 →
  talking_birds = 64 →
  non_talking_birds = 13 →
  dogs = 40 →
  talking_birds = 4 * ((total_animals - (talking_birds + non_talking_birds + dogs)) / 4) →
  talking_birds + non_talking_birds = 124 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_birds_l4082_408252


namespace NUMINAMATH_CALUDE_david_score_l4082_408226

/-- Calculates the score of a player in a Scrabble game given the opponent's initial lead,
    the opponent's play, and the opponent's final lead. -/
def calculate_score (initial_lead : ℕ) (opponent_play : ℕ) (final_lead : ℕ) : ℕ :=
  initial_lead + opponent_play - final_lead

/-- Theorem stating that David's score in the Scrabble game is 32 points. -/
theorem david_score :
  calculate_score 22 15 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_david_score_l4082_408226


namespace NUMINAMATH_CALUDE_quadratic_function_m_l4082_408296

/-- A quadratic function g(x) with integer coefficients -/
def g (d e f : ℤ) (x : ℤ) : ℤ := d * x^2 + e * x + f

/-- The theorem stating that under given conditions, m = -1 -/
theorem quadratic_function_m (d e f m : ℤ) : 
  g d e f 2 = 0 ∧ 
  60 < g d e f 6 ∧ g d e f 6 < 70 ∧
  80 < g d e f 9 ∧ g d e f 9 < 90 ∧
  10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_l4082_408296


namespace NUMINAMATH_CALUDE_spelling_contest_questions_l4082_408236

theorem spelling_contest_questions (drew_correct drew_wrong carla_correct : ℕ) 
  (h1 : drew_correct = 20)
  (h2 : drew_wrong = 6)
  (h3 : carla_correct = 14)
  (h4 : carla_correct + 2 * drew_wrong = drew_correct + drew_wrong) :
  drew_correct + drew_wrong = 26 :=
by sorry

end NUMINAMATH_CALUDE_spelling_contest_questions_l4082_408236


namespace NUMINAMATH_CALUDE_expression_evaluation_l4082_408251

theorem expression_evaluation (a b : ℝ) 
  (h : |a - 2| + (b - 1/2)^2 = 0) : 
  2*(a^2*b - 3*a*b^2) - (5*a^2*b - 3*(2*a*b^2 - a^2*b) - 2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4082_408251


namespace NUMINAMATH_CALUDE_star_equation_solution_l4082_408294

/-- Custom operation  defined for integers -/
def star (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if x star 9 = 160, then x = 21 -/
theorem star_equation_solution :
  ∀ x : ℤ, star x 9 = 160 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l4082_408294


namespace NUMINAMATH_CALUDE_angus_patrick_diff_l4082_408225

/-- The number of fish caught by Ollie -/
def ollie_catch : ℕ := 5

/-- The number of fish caught by Patrick -/
def patrick_catch : ℕ := 8

/-- The difference between Angus and Ollie's catch -/
def angus_ollie_diff : ℕ := 7

/-- The number of fish caught by Angus -/
def angus_catch : ℕ := ollie_catch + angus_ollie_diff

/-- Theorem: The difference between Angus and Patrick's fish catch is 4 -/
theorem angus_patrick_diff : angus_catch - patrick_catch = 4 := by
  sorry

end NUMINAMATH_CALUDE_angus_patrick_diff_l4082_408225


namespace NUMINAMATH_CALUDE_F_simplification_and_range_l4082_408204

noncomputable def f (t : ℝ) : ℝ := Real.sqrt ((1 - t) / (1 + t))

noncomputable def F (x : ℝ) : ℝ := Real.sin x * f (Real.cos x) + Real.cos x * f (Real.sin x)

theorem F_simplification_and_range (x : ℝ) (h : π < x ∧ x < 3 * π / 2) :
  F x = Real.sqrt 2 * Real.sin (x + π / 4) - 2 ∧
  ∃ y ∈ Set.Icc (-2 - Real.sqrt 2) (-3), F x = y :=
sorry

end NUMINAMATH_CALUDE_F_simplification_and_range_l4082_408204


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l4082_408279

/-- Define ω as a complex number satisfying the properties of a cube root of unity -/
def ω : ℂ := sorry

/-- ω is a cube root of unity -/
axiom ω_cubed : ω^3 = 1

/-- ω satisfies the equation ω^2 + ω + 1 = 0 -/
axiom ω_sum : ω^2 + ω + 1 = 0

/-- Theorem: ω^9 + (ω^2)^9 = 2 -/
theorem cube_root_unity_sum : ω^9 + (ω^2)^9 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l4082_408279


namespace NUMINAMATH_CALUDE_min_value_of_function_l4082_408282

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  x + 4 / x^2 ≥ 3 ∧ ∀ ε > 0, ∃ x₀ > 0, x₀ + 4 / x₀^2 < 3 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4082_408282


namespace NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l4082_408265

-- Define the quadratic function
def f (x : ℝ) : ℝ := 5 * (x - 3)^2 + 2

-- Theorem statement
theorem quadratic_decreasing_before_vertex :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 3 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l4082_408265


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l4082_408224

theorem arithmetic_mean_of_4_and_16 (x : ℝ) :
  x = (4 + 16) / 2 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l4082_408224


namespace NUMINAMATH_CALUDE_euler_minus_i_pi_is_real_l4082_408290

theorem euler_minus_i_pi_is_real : Complex.im (Complex.exp (-Complex.I * Real.pi)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_euler_minus_i_pi_is_real_l4082_408290


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l4082_408221

/-- In a pentagon with angles 104°, 97°, x°, 2x°, and R°, where the sum of all angles is 540°, 
    the measure of angle R is 204°. -/
theorem pentagon_angle_measure (x : ℝ) (R : ℝ) : 
  104 + 97 + x + 2*x + R = 540 → R = 204 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l4082_408221


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l4082_408218

def f (x : ℝ) : ℝ := x^4 + x^2 + 7*x

theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l4082_408218


namespace NUMINAMATH_CALUDE_minimize_fuel_consumption_l4082_408200

theorem minimize_fuel_consumption
  (total_cargo : ℕ)
  (large_capacity small_capacity : ℕ)
  (large_fuel small_fuel : ℕ)
  (h1 : total_cargo = 157)
  (h2 : large_capacity = 5)
  (h3 : small_capacity = 2)
  (h4 : large_fuel = 20)
  (h5 : small_fuel = 10) :
  ∃ (large_trucks small_trucks : ℕ),
    large_trucks * large_capacity + small_trucks * small_capacity ≥ total_cargo ∧
    ∀ (x y : ℕ),
      x * large_capacity + y * small_capacity ≥ total_cargo →
      x * large_fuel + y * small_fuel ≥ large_trucks * large_fuel + small_trucks * small_fuel →
      x = large_trucks ∧ y = small_trucks :=
by sorry

end NUMINAMATH_CALUDE_minimize_fuel_consumption_l4082_408200


namespace NUMINAMATH_CALUDE_arithmetic_progression_condition_l4082_408205

theorem arithmetic_progression_condition (a b c : ℝ) : 
  (∃ d : ℝ, ∃ n k p : ℤ, b = a + d * (k - n) ∧ c = a + d * (p - n)) →
  (∃ A B : ℤ, (b - a) / (c - b) = A / B) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_condition_l4082_408205


namespace NUMINAMATH_CALUDE_sum_minimized_at_Q₅_l4082_408250

/-- A type representing points on a line -/
structure PointOnLine where
  position : ℝ

/-- The distance between two points on a line -/
def distance (p q : PointOnLine) : ℝ := |p.position - q.position|

/-- The sum of distances from a point Q to points Q₁, ..., Q₉ -/
def sumOfDistances (Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : PointOnLine) : ℝ :=
  distance Q Q₁ + distance Q Q₂ + distance Q Q₃ + distance Q Q₄ + 
  distance Q Q₅ + distance Q Q₆ + distance Q Q₇ + distance Q Q₈ + distance Q Q₉

/-- The theorem stating that the sum of distances is minimized when Q is at Q₅ -/
theorem sum_minimized_at_Q₅ 
  (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : PointOnLine) 
  (h : Q₁.position < Q₂.position ∧ Q₂.position < Q₃.position ∧ 
       Q₃.position < Q₄.position ∧ Q₄.position < Q₅.position ∧ 
       Q₅.position < Q₆.position ∧ Q₆.position < Q₇.position ∧ 
       Q₇.position < Q₈.position ∧ Q₈.position < Q₉.position) :
  ∀ Q : PointOnLine, sumOfDistances Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ ≥ 
                     sumOfDistances Q₅ Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ :=
sorry

end NUMINAMATH_CALUDE_sum_minimized_at_Q₅_l4082_408250


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l4082_408283

/-- A right triangle with one leg of prime length n and other sides of natural number lengths has perimeter n + n^2 -/
theorem right_triangle_perimeter (n : ℕ) (h_prime : Nat.Prime n) :
  ∃ (x y : ℕ), x^2 + n^2 = y^2 ∧ x + y + n = n + n^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l4082_408283


namespace NUMINAMATH_CALUDE_sin_cos_225_degrees_l4082_408277

theorem sin_cos_225_degrees :
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 ∧
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_225_degrees_l4082_408277


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4082_408233

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 1 = 0

-- Define the roots of the equation
def roots_of_equation (x₁ x₂ : ℝ) : Prop :=
  quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂

-- Theorem statement
theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) :
  roots_of_equation x₁ x₂ → 1/x₁ + 1/x₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4082_408233


namespace NUMINAMATH_CALUDE_exists_number_not_exceeding_kr_l4082_408214

/-- The operation that replaces a number with two new numbers -/
def replace_operation (x : ℝ) : ℝ × ℝ :=
  sorry

/-- Perform the operation k^2 - 1 times -/
def perform_operations (r : ℝ) (k : ℕ) : List ℝ :=
  sorry

theorem exists_number_not_exceeding_kr (r : ℝ) (k : ℕ) (h_r : r > 0) :
  ∃ x ∈ perform_operations r k, x ≤ k * r :=
sorry

end NUMINAMATH_CALUDE_exists_number_not_exceeding_kr_l4082_408214


namespace NUMINAMATH_CALUDE_investment_sum_l4082_408248

theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 240 → P = 2000 := by sorry

end NUMINAMATH_CALUDE_investment_sum_l4082_408248


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l4082_408211

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (2 * z + 3 * w) = 19)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l4082_408211


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4082_408229

def y (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties :
  ∀ (a b : ℝ),
  (∀ x : ℝ, y a b x < 0 ↔ 1 < x ∧ x < 3) →
  a > 0 →
  b = -2 * a →
  (a = 1 ∧ b = -2) ∧
  (∀ x : ℝ,
    y a b x ≤ -1 ↔
      ((0 < a ∧ a < 1 → 2 ≤ x ∧ x ≤ 2/a) ∧
       (a = 1 → x = 2) ∧
       (a > 1 → 2/a ≤ x ∧ x ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4082_408229


namespace NUMINAMATH_CALUDE_equation_solution_l4082_408262

theorem equation_solution (x : ℝ) : 
  (x^2 + x - 2)^3 + (2*x^2 - x - 1)^3 = 27*(x^2 - 1)^3 ↔ 
  x = 1 ∨ x = -1 ∨ x = -2 ∨ x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4082_408262


namespace NUMINAMATH_CALUDE_second_number_calculation_second_number_is_190_l4082_408235

theorem second_number_calculation : ℝ → Prop :=
  fun x =>
    let first_number : ℝ := 1280
    let twenty_percent_of_650 : ℝ := 0.2 * 650
    let twenty_five_percent_of_first : ℝ := 0.25 * first_number
    x = twenty_five_percent_of_first - twenty_percent_of_650 → x = 190

-- The proof is omitted
theorem second_number_is_190 : ∃ x : ℝ, second_number_calculation x :=
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_second_number_is_190_l4082_408235


namespace NUMINAMATH_CALUDE_squats_on_fourth_day_l4082_408246

/-- Calculates the number of squats on a given day, given the initial number of squats and daily increase. -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Theorem stating that on the fourth day, the number of squats will be 45, given the initial conditions. -/
theorem squats_on_fourth_day :
  squats_on_day 30 5 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_squats_on_fourth_day_l4082_408246


namespace NUMINAMATH_CALUDE_three_digit_numbers_problem_l4082_408266

theorem three_digit_numbers_problem : ∃ (a b : ℕ), 
  (100 ≤ a ∧ a < 1000) ∧ 
  (100 ≤ b ∧ b < 1000) ∧ 
  (a / 100 = b % 10) ∧ 
  (b / 100 = a % 10) ∧ 
  (a > b → a - b = 297) ∧ 
  (b > a → b - a = 297) ∧ 
  ((a < b → (a / 100 + (a / 10) % 10 + a % 10 = 23)) ∧ 
   (b < a → (b / 100 + (b / 10) % 10 + b % 10 = 23))) ∧ 
  ((a = 986 ∧ b = 689) ∨ (a = 689 ∧ b = 986)) := by
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_problem_l4082_408266


namespace NUMINAMATH_CALUDE_oil_measurement_l4082_408257

theorem oil_measurement (initial_oil : ℚ) (added_oil : ℚ) :
  initial_oil = 17/100 → added_oil = 67/100 → initial_oil + added_oil = 84/100 := by
  sorry

end NUMINAMATH_CALUDE_oil_measurement_l4082_408257


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4082_408286

theorem complex_modulus_problem (z : ℂ) (h : z - 2*Complex.I = z*Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4082_408286


namespace NUMINAMATH_CALUDE_b_invests_after_six_months_l4082_408241

/-- A partnership model with three partners --/
structure Partnership where
  x : ℝ  -- A's investment
  m : ℝ  -- Months after which B invests
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The investment-time products for each partner --/
def investment_time (p : Partnership) : ℝ × ℝ × ℝ :=
  (p.x * 12, 2 * p.x * (12 - p.m), 3 * p.x * 4)

/-- The total investment-time product --/
def total_investment_time (p : Partnership) : ℝ :=
  let (a, b, c) := investment_time p
  a + b + c

/-- Theorem stating that B invests after 6 months --/
theorem b_invests_after_six_months (p : Partnership) 
  (h1 : p.total_gain = 12000)
  (h2 : p.a_share = 4000)
  (h3 : p.a_share / p.total_gain = 1 / 3)
  (h4 : p.x * 12 = (1 / 3) * total_investment_time p) :
  p.m = 6 := by
  sorry


end NUMINAMATH_CALUDE_b_invests_after_six_months_l4082_408241


namespace NUMINAMATH_CALUDE_other_radius_length_l4082_408207

/-- A circle is defined by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- All radii of a circle have the same length -/
axiom circle_radii_equal (c : Circle) (p q : ℝ × ℝ) :
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 →
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 →
  ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2).sqrt =
  ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2).sqrt

theorem other_radius_length (c : Circle) (p q : ℝ × ℝ) 
    (hp : (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2)
    (hq : (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2)
    (h_radius : ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2).sqrt = 2) :
    ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2).sqrt = 2 := by
  sorry

end NUMINAMATH_CALUDE_other_radius_length_l4082_408207


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equality_l4082_408223

theorem arithmetic_sequence_equality (n : ℕ) (a b : Fin n → ℕ) :
  n ≥ 2018 →
  (∀ i : Fin n, a i ≤ 5 * n ∧ b i ≤ 5 * n) →
  (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j) →
  (∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = d * (i - j)) →
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equality_l4082_408223


namespace NUMINAMATH_CALUDE_window_width_window_width_is_six_l4082_408216

/-- The width of a window in a bedroom, given the room dimensions and areas of doors and windows. -/
theorem window_width : ℝ :=
  let room_width : ℝ := 20
  let room_length : ℝ := 20
  let room_height : ℝ := 8
  let door1_width : ℝ := 3
  let door1_height : ℝ := 7
  let door2_width : ℝ := 5
  let door2_height : ℝ := 7
  let window_height : ℝ := 4
  let total_paint_area : ℝ := 560
  let total_wall_area : ℝ := 4 * room_width * room_height
  let door1_area : ℝ := door1_width * door1_height
  let door2_area : ℝ := door2_width * door2_height
  let window_width : ℝ := (total_wall_area - door1_area - door2_area - total_paint_area) / window_height
  window_width

/-- Proof that the window width is 6 feet. -/
theorem window_width_is_six : window_width = 6 := by
  sorry

end NUMINAMATH_CALUDE_window_width_window_width_is_six_l4082_408216


namespace NUMINAMATH_CALUDE_frame_price_increase_l4082_408228

theorem frame_price_increase (budget : ℝ) (remaining : ℝ) (ratio : ℝ) : 
  budget = 60 → 
  remaining = 6 → 
  ratio = 3/4 → 
  let smaller_frame_price := budget - remaining
  let initial_frame_price := smaller_frame_price / ratio
  (initial_frame_price - budget) / budget * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_frame_price_increase_l4082_408228


namespace NUMINAMATH_CALUDE_expansion_properties_l4082_408288

/-- Given n, returns the sum of the binomial coefficients of the last three terms in (1-3x)^n -/
def sumLastThreeCoefficients (n : ℕ) : ℕ :=
  Nat.choose n (n-2) + Nat.choose n (n-1) + Nat.choose n n

/-- Returns the coefficient of the (r+1)-th term in the expansion of (1-3x)^n -/
def coefficientOfTerm (n : ℕ) (r : ℕ) : ℤ :=
  (Nat.choose n r : ℤ) * (-3) ^ r

/-- Returns the absolute value of the coefficient of the (r+1)-th term in the expansion of (1-3x)^n -/
def absCoefficient (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r * 3 ^ r

/-- The main theorem about the expansion of (1-3x)^n -/
theorem expansion_properties (n : ℕ) (h : sumLastThreeCoefficients n = 121) :
  (∃ r : ℕ, r = 12 ∧ ∀ k : ℕ, absCoefficient n k ≤ absCoefficient n r) ∧
  (∀ k : ℕ, Nat.choose n k ≤ Nat.choose n 7 ∧ Nat.choose n k ≤ Nat.choose n 8) :=
sorry

end NUMINAMATH_CALUDE_expansion_properties_l4082_408288


namespace NUMINAMATH_CALUDE_red_beads_count_l4082_408247

theorem red_beads_count (green : ℕ) (brown : ℕ) (taken : ℕ) (left : ℕ) : 
  green = 1 → brown = 2 → taken = 2 → left = 4 → 
  ∃ (red : ℕ), red = (green + brown + taken + left) - (green + brown) :=
by sorry

end NUMINAMATH_CALUDE_red_beads_count_l4082_408247


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4082_408256

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | Real.log (x + 1) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4082_408256


namespace NUMINAMATH_CALUDE_largest_angle_60_degrees_l4082_408237

/-- 
Given a triangle ABC with side lengths a, b, and c satisfying the equation
a^2 + b^2 = c^2 - ab, the largest interior angle of the triangle is 60°.
-/
theorem largest_angle_60_degrees 
  (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (eq : a^2 + b^2 = c^2 - a*b) : 
  ∃ θ : ℝ, θ ≤ 60 * π / 180 ∧ 
    θ = Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧
    θ = Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) ∧
    θ = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) :=
sorry

end NUMINAMATH_CALUDE_largest_angle_60_degrees_l4082_408237


namespace NUMINAMATH_CALUDE_max_table_height_for_specific_triangle_l4082_408258

/-- Triangle ABC with sides a, b, c -/
structure Triangle :=
  (a b c : ℝ)

/-- The maximum height of the table constructed from the triangle -/
def maxTableHeight (t : Triangle) : ℝ := sorry

/-- The theorem to be proved -/
theorem max_table_height_for_specific_triangle :
  let t := Triangle.mk 23 27 30
  maxTableHeight t = (40 * Real.sqrt 221) / 57 := by sorry

end NUMINAMATH_CALUDE_max_table_height_for_specific_triangle_l4082_408258


namespace NUMINAMATH_CALUDE_remainder_of_base_12_num_div_9_l4082_408245

-- Define the base-12 number 1742₁₂
def base_12_num : ℕ := 1 * 12^3 + 7 * 12^2 + 4 * 12 + 2

-- Theorem statement
theorem remainder_of_base_12_num_div_9 :
  base_12_num % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_base_12_num_div_9_l4082_408245


namespace NUMINAMATH_CALUDE_cubes_volume_percentage_l4082_408298

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : Cube) : ℕ :=
  cube.sideLength * cube.sideLength * cube.sideLength

/-- Calculates the number of cubes that can fit along a given dimension -/
def cubesFitInDimension (dimension : ℕ) (cube : Cube) : ℕ :=
  dimension / cube.sideLength

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubesFit (box : BoxDimensions) (cube : Cube) : ℕ :=
  (cubesFitInDimension box.length cube) *
  (cubesFitInDimension box.width cube) *
  (cubesFitInDimension box.height cube)

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a 8x6x12 inch box is 66.67% -/
theorem cubes_volume_percentage :
  let box := BoxDimensions.mk 8 6 12
  let cube := Cube.mk 4
  let cubesVolume := (totalCubesFit box cube) * (cubeVolume cube)
  let totalVolume := boxVolume box
  let percentage := (cubesVolume : ℚ) / (totalVolume : ℚ) * 100
  percentage = 200/3 := by
  sorry

end NUMINAMATH_CALUDE_cubes_volume_percentage_l4082_408298


namespace NUMINAMATH_CALUDE_circle_radius_increase_l4082_408267

theorem circle_radius_increase (r n : ℝ) (h : r > 0) (h_n : n > 0) :
  π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l4082_408267


namespace NUMINAMATH_CALUDE_bucket_problem_l4082_408232

/-- Represents the state of the two buckets -/
structure BucketState :=
  (large : ℕ)  -- Amount in 7-liter bucket
  (small : ℕ)  -- Amount in 3-liter bucket

/-- Represents a single operation on the buckets -/
inductive BucketOperation
  | FillLarge
  | FillSmall
  | EmptyLarge
  | EmptySmall
  | PourLargeToSmall
  | PourSmallToLarge

/-- Applies a single operation to a bucket state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillLarge => { large := 7, small := state.small }
  | BucketOperation.FillSmall => { large := state.large, small := 3 }
  | BucketOperation.EmptyLarge => { large := 0, small := state.small }
  | BucketOperation.EmptySmall => { large := state.large, small := 0 }
  | BucketOperation.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      { large := state.large - amount, small := state.small + amount }
  | BucketOperation.PourSmallToLarge =>
      let amount := min state.small (7 - state.large)
      { large := state.large + amount, small := state.small - amount }

/-- Applies a sequence of operations to an initial state -/
def applyOperations (initial : BucketState) (ops : List BucketOperation) : BucketState :=
  ops.foldl applyOperation initial

/-- Checks if a specific amount can be measured using a sequence of operations -/
def canMeasure (amount : ℕ) : Prop :=
  ∃ (ops : List BucketOperation),
    (applyOperations { large := 0, small := 0 } ops).large = amount ∨
    (applyOperations { large := 0, small := 0 } ops).small = amount

theorem bucket_problem :
  canMeasure 1 ∧ canMeasure 2 ∧ canMeasure 4 ∧ canMeasure 5 ∧ canMeasure 6 :=
sorry

end NUMINAMATH_CALUDE_bucket_problem_l4082_408232


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l4082_408215

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x*y - z*t) + f (x*t + y*z)

/-- The main theorem stating the possible solutions. -/
theorem functional_equation_solutions (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = 1/2) ∨ (∀ x, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l4082_408215


namespace NUMINAMATH_CALUDE_tangent_lines_range_l4082_408255

/-- The range of k values for which two tangent lines can be drawn from (1, 2) to the circle x^2 + y^2 + kx + 2y + k^2 - 15 = 0 -/
theorem tangent_lines_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0 ∧ 
   ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
     (∀ (x' y' : ℝ), (y' - 2 = m₁ * (x' - 1) ∨ y' - 2 = m₂ * (x' - 1)) →
       (x'^2 + y'^2 + k*x' + 2*y' + k^2 - 15 = 0 → x' = 1 ∧ y' = 2))) ↔ 
  (k ∈ Set.Ioo (-8 * Real.sqrt 3 / 3) (-3) ∪ Set.Ioo 2 (8 * Real.sqrt 3 / 3)) :=
by sorry


end NUMINAMATH_CALUDE_tangent_lines_range_l4082_408255


namespace NUMINAMATH_CALUDE_election_win_margin_l4082_408263

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) :
  (winner_votes : ℚ) / total_votes = 62 / 100 →
  winner_votes = 868 →
  winner_votes - (total_votes - winner_votes) = 336 :=
by sorry

end NUMINAMATH_CALUDE_election_win_margin_l4082_408263


namespace NUMINAMATH_CALUDE_cost_difference_formula_option1_more_cost_effective_at_50_l4082_408220

/-- Represents the cost difference between Option 2 and Option 1 for a customer
    buying 20 water dispensers and x water dispenser barrels, where x > 20. -/
def cost_difference (x : ℝ) : ℝ :=
  (45 * x + 6300) - (50 * x + 6000)

/-- Theorem stating that the cost difference between Option 2 and Option 1
    is always 300 - 5x yuan, for x > 20. -/
theorem cost_difference_formula (x : ℝ) (h : x > 20) :
  cost_difference x = 300 - 5 * x := by
  sorry

/-- Corollary stating that Option 1 is more cost-effective when x = 50. -/
theorem option1_more_cost_effective_at_50 :
  cost_difference 50 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_formula_option1_more_cost_effective_at_50_l4082_408220


namespace NUMINAMATH_CALUDE_order_of_abc_l4082_408285

theorem order_of_abc (a b c : ℝ) 
  (ha : a = (1.1 : ℝ)^10)
  (hb : (5 : ℝ)^b = 3^a + 4^a)
  (hc : c = Real.exp a - a) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l4082_408285


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4082_408210

theorem quadratic_inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4082_408210


namespace NUMINAMATH_CALUDE_age_ratio_change_l4082_408239

theorem age_ratio_change (father_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  father_age = 60 → 
  man_age = (2 * father_age) / 5 → 
  (man_age + years) * 2 = father_age + years → 
  years = 12 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_change_l4082_408239


namespace NUMINAMATH_CALUDE_marys_age_l4082_408201

theorem marys_age :
  ∃! x : ℕ, 
    (∃ n : ℕ, x - 2 = n^2) ∧ 
    (∃ m : ℕ, x + 2 = m^3) ∧ 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_marys_age_l4082_408201


namespace NUMINAMATH_CALUDE_hash_six_eight_l4082_408275

-- Define the # operation
def hash (a b : ℤ) : ℤ := 3*a - 3*b + 4

-- Theorem statement
theorem hash_six_eight : hash 6 8 = -2 := by
  sorry

end NUMINAMATH_CALUDE_hash_six_eight_l4082_408275


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l4082_408202

theorem necessary_not_sufficient (a b c d : ℝ) (h : c > d) :
  (∀ a b, (a - c > b - d) → (a > b)) ∧
  (∃ a b, (a > b) ∧ ¬(a - c > b - d)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l4082_408202


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4082_408222

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + y) = f x + f y) : 
  ∃ a : ℤ, ∀ x : ℤ, f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4082_408222


namespace NUMINAMATH_CALUDE_music_program_band_members_l4082_408295

theorem music_program_band_members :
  ∀ (total_students : ℕ) 
    (band_percentage : ℚ) 
    (chorus_percentage : ℚ) 
    (band_members : ℕ) 
    (chorus_members : ℕ),
  total_students = 36 →
  band_percentage = 1/5 →
  chorus_percentage = 1/4 →
  band_members + chorus_members = total_students →
  (band_percentage * band_members : ℚ) = (chorus_percentage * chorus_members : ℚ) →
  band_members = 16 := by
sorry

end NUMINAMATH_CALUDE_music_program_band_members_l4082_408295


namespace NUMINAMATH_CALUDE_house_people_count_l4082_408268

theorem house_people_count (total_pizza : ℕ) (eaten_pizza : ℕ) (pizza_per_person : ℕ) 
  (pizza_eater_ratio : ℚ) (remaining_pizza : ℕ) :
  total_pizza = 50 →
  eaten_pizza = total_pizza - remaining_pizza →
  pizza_per_person = 4 →
  pizza_eater_ratio = 3 / 5 →
  remaining_pizza = 14 →
  ∃ (people : ℕ), people = 15 ∧ 
    (pizza_eater_ratio * people).num * pizza_per_person = eaten_pizza * pizza_eater_ratio.den :=
by
  sorry

end NUMINAMATH_CALUDE_house_people_count_l4082_408268


namespace NUMINAMATH_CALUDE_intersection_S_complement_T_l4082_408208

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}

-- Define set S
def S : Set ℕ := {1, 2, 4, 5}

-- Define set T
def T : Set ℕ := {3, 4, 5, 7}

-- Theorem statement
theorem intersection_S_complement_T : S ∩ (U \ T) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_S_complement_T_l4082_408208


namespace NUMINAMATH_CALUDE_total_fish_caught_l4082_408253

theorem total_fish_caught (pikes sturgeon herring : ℕ) 
  (h1 : pikes = 30) 
  (h2 : sturgeon = 40) 
  (h3 : herring = 75) : 
  pikes + sturgeon + herring = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l4082_408253


namespace NUMINAMATH_CALUDE_dice_roll_probability_l4082_408242

def total_outcomes : ℕ := 6^6

def ways_to_choose_numbers : ℕ := Nat.choose 6 2

def ways_to_arrange_dice : ℕ := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1)

def successful_outcomes : ℕ := ways_to_choose_numbers * ways_to_arrange_dice

theorem dice_roll_probability :
  (successful_outcomes : ℚ) / total_outcomes = 25 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l4082_408242


namespace NUMINAMATH_CALUDE_travelers_meet_on_day_three_l4082_408219

/-- Distance traveled by the first traveler on day n -/
def d1 (n : ℕ) : ℕ := 3 * n - 1

/-- Distance traveled by the second traveler on day n -/
def d2 (n : ℕ) : ℕ := 2 * n + 1

/-- Total distance traveled by the first traveler after n days -/
def D1 (n : ℕ) : ℕ := (3 * n^2 + n) / 2

/-- Total distance traveled by the second traveler after n days -/
def D2 (n : ℕ) : ℕ := n^2 + 2 * n

theorem travelers_meet_on_day_three :
  ∃ n : ℕ, n > 0 ∧ D1 n = D2 n ∧ ∀ m : ℕ, 0 < m ∧ m < n → D1 m < D2 m :=
sorry

end NUMINAMATH_CALUDE_travelers_meet_on_day_three_l4082_408219


namespace NUMINAMATH_CALUDE_sequence_sum_properties_l4082_408289

/-- Given a sequence {a_n} with sum of first n terms S_n = n^2 - 3,
    prove the first term and general term. -/
theorem sequence_sum_properties (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n, S n = n^2 - 3) :
  (a 1 = -2) ∧ 
  (∀ n ≥ 2, a n = 2*n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_properties_l4082_408289


namespace NUMINAMATH_CALUDE_sodium_reduction_proof_l4082_408240

def salt_teaspoons : ℕ := 2
def initial_parmesan_ounces : ℕ := 8
def sodium_per_salt_teaspoon : ℕ := 50
def sodium_per_parmesan_ounce : ℕ := 25
def reduction_factor : ℚ := 1/3

def total_sodium (parmesan_ounces : ℕ) : ℕ :=
  salt_teaspoons * sodium_per_salt_teaspoon + parmesan_ounces * sodium_per_parmesan_ounce

def reduced_parmesan_ounces : ℕ := initial_parmesan_ounces - 4

theorem sodium_reduction_proof :
  (total_sodium initial_parmesan_ounces : ℚ) * (1 - reduction_factor) =
  (total_sodium reduced_parmesan_ounces : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sodium_reduction_proof_l4082_408240


namespace NUMINAMATH_CALUDE_complex_modulus_l4082_408297

theorem complex_modulus (z : ℂ) (h : (3 + 2*I) * z = 5 - I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l4082_408297


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l4082_408217

/-- Given a right triangle with legs of lengths 6 and 8, and semicircles constructed
    on all its sides as diameters lying outside the triangle, the radius of the circle
    tangent to these semicircles is 144/23. -/
theorem tangent_circle_radius (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_a : a = 6) (h_b : b = 8) : ∃ r : ℝ, r = 144 / 23 ∧ 
  r > 0 ∧
  (∃ x y z : ℝ, x^2 + y^2 = (r + a/2)^2 ∧
               y^2 + z^2 = (r + b/2)^2 ∧
               z^2 + x^2 = (r + c/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l4082_408217


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l4082_408281

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 12 * (5 / 18))  -- Convert 12 km/hr to m/s
  (h2 : train_speed = 60 * (5 / 18))   -- Convert 60 km/hr to m/s
  (h3 : train_length = 300)
  (h4 : initial_distance = 300) :
  (train_length + initial_distance) / (train_speed - jogger_speed) = 15 := by
  sorry

#eval Float.ofScientific 15 0 1  -- Output: 15.0

end NUMINAMATH_CALUDE_train_passing_jogger_time_l4082_408281


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l4082_408254

theorem arctan_equation_solution :
  ∀ x : ℝ, 2 * Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 → x = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l4082_408254


namespace NUMINAMATH_CALUDE_cloud_nine_total_amount_l4082_408213

/-- Cloud 9 Diving Company's pricing structure and bookings -/
structure CloudNineDiving where
  individual_bookings : ℝ
  early_bird_individual : ℝ
  group_a : ℝ
  group_b : ℝ
  group_c : ℝ
  individual_refunds : ℝ
  group_refunds : ℝ

/-- Calculate the total amount after discounts and refunds -/
def calculate_total (c : CloudNineDiving) : ℝ :=
  let individual_after_discount := c.individual_bookings - (c.early_bird_individual * 0.03)
  let group_a_after_discount := c.group_a * (1 - 0.05) * (1 - 0.03)
  let group_b_after_discount := c.group_b * (1 - 0.10)
  let group_c_after_discount := c.group_c * (1 - 0.15) * (1 - 0.03)
  let total_before_refunds := individual_after_discount + group_a_after_discount + group_b_after_discount + group_c_after_discount
  total_before_refunds - c.individual_refunds - c.group_refunds

/-- Theorem: The total amount taken by Cloud 9 Diving Company is $35,006.50 -/
theorem cloud_nine_total_amount (c : CloudNineDiving) 
  (h1 : c.individual_bookings = 12000)
  (h2 : c.early_bird_individual = 3000)
  (h3 : c.group_a = 6000)
  (h4 : c.group_b = 9000)
  (h5 : c.group_c = 15000)
  (h6 : c.individual_refunds = 2100)
  (h7 : c.group_refunds = 800) :
  calculate_total c = 35006.50 := by
  sorry

end NUMINAMATH_CALUDE_cloud_nine_total_amount_l4082_408213


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l4082_408299

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -7 + 5*I
  let z₂ : ℂ := 5 - 9*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -1 - 2*I := by sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l4082_408299


namespace NUMINAMATH_CALUDE_expand_product_l4082_408287

theorem expand_product (x : ℝ) : (2*x + 3) * (4*x - 5) = 8*x^2 + 2*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4082_408287


namespace NUMINAMATH_CALUDE_prism_volume_l4082_408272

-- Define the prism dimensions
variable (a b c : ℝ)

-- Define the conditions
axiom face_area_1 : a * b = 30
axiom face_area_2 : b * c = 72
axiom face_area_3 : c * a = 45

-- State the theorem
theorem prism_volume : a * b * c = 180 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l4082_408272


namespace NUMINAMATH_CALUDE_train_speed_problem_l4082_408264

/-- Proves that given two trains of specified lengths running in opposite directions,
    where one train has a known speed and the time to cross each other is known,
    the speed of the other train can be determined. -/
theorem train_speed_problem (length1 length2 known_speed crossing_time : ℝ) :
  length1 = 140 ∧
  length2 = 190 ∧
  known_speed = 40 ∧
  crossing_time = 11.879049676025918 →
  ∃ other_speed : ℝ,
    other_speed = 60 ∧
    (length1 + length2) / crossing_time * 3.6 = known_speed + other_speed :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l4082_408264


namespace NUMINAMATH_CALUDE_azalea_sheep_count_l4082_408243

/-- The number of sheep Azalea sheared -/
def num_sheep : ℕ := 200

/-- The amount paid to the shearer -/
def shearer_payment : ℕ := 2000

/-- The amount of wool produced by each sheep in pounds -/
def wool_per_sheep : ℕ := 10

/-- The price of wool per pound -/
def wool_price : ℕ := 20

/-- The profit made by Azalea -/
def profit : ℕ := 38000

/-- Theorem stating that the number of sheep Azalea sheared is 200 -/
theorem azalea_sheep_count :
  num_sheep = (profit + shearer_payment) / (wool_per_sheep * wool_price) :=
by sorry

end NUMINAMATH_CALUDE_azalea_sheep_count_l4082_408243


namespace NUMINAMATH_CALUDE_carA_distance_at_2016th_meeting_l4082_408292

/-- Represents a car with its current speed and direction -/
structure Car where
  speed : ℝ
  direction : Bool

/-- Represents the state of the system at any given time -/
structure State where
  carA : Car
  carB : Car
  positionA : ℝ
  positionB : ℝ
  meetingCount : ℕ
  distanceTraveledA : ℝ

/-- The distance between points A and B -/
def distance : ℝ := 900

/-- Function to update the state after each meeting -/
def updateState (s : State) : State :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the total distance traveled by Car A at the 2016th meeting -/
theorem carA_distance_at_2016th_meeting :
  ∃ (finalState : State),
    finalState.meetingCount = 2016 ∧
    finalState.distanceTraveledA = 1813900 :=
by
  sorry

end NUMINAMATH_CALUDE_carA_distance_at_2016th_meeting_l4082_408292


namespace NUMINAMATH_CALUDE_no_common_solution_l4082_408234

theorem no_common_solution : ¬∃ x : ℝ, (5*x - 2) / (6*x - 6) = 3/4 ∧ x^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l4082_408234


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l4082_408209

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ x : ℂ, x^6 - 12*x^5 + A*x^4 + (-162)*x^3 + C*x^2 + D*x + 36 = 
      (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l4082_408209


namespace NUMINAMATH_CALUDE_number_equal_to_its_opposite_l4082_408203

theorem number_equal_to_its_opposite : ∀ x : ℝ, x = -x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_number_equal_to_its_opposite_l4082_408203


namespace NUMINAMATH_CALUDE_f_unique_zero_l4082_408227

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a + 1) * x + a * Real.log x

theorem f_unique_zero (a : ℝ) (h : a > 0) : 
  ∃! x : ℝ, x > 0 ∧ f a x = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_unique_zero_l4082_408227


namespace NUMINAMATH_CALUDE_right_triangle_cos_c_l4082_408244

theorem right_triangle_cos_c (A B C : Real) (sinB : Real) :
  -- Triangle ABC exists
  -- Angle A is a right angle (90 degrees)
  A + B + C = Real.pi →
  A = Real.pi / 2 →
  -- sin B is given as 3/5
  sinB = 3 / 5 →
  -- Prove that cos C = 3/5
  Real.cos C = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_c_l4082_408244


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_9_l4082_408206

theorem smallest_common_multiple_of_6_and_9 : 
  ∃ n : ℕ+, (∀ m : ℕ+, 6 ∣ m ∧ 9 ∣ m → n ≤ m) ∧ 6 ∣ n ∧ 9 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_9_l4082_408206


namespace NUMINAMATH_CALUDE_juan_running_time_l4082_408291

theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 80) (h2 : speed = 10) :
  distance / speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_time_l4082_408291


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l4082_408261

/-- Converts a natural number from base 2 to base 10 --/
def base2ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a natural number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- The base 2 representation of the number --/
def base2Number : ℕ := 101101100

/-- The expected base 4 representation of the number --/
def expectedBase4Number : ℕ := 23110

theorem base2_to_base4_conversion :
  base10ToBase4 (base2ToBase10 base2Number) = expectedBase4Number := by sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l4082_408261


namespace NUMINAMATH_CALUDE_point_M_properties_l4082_408212

def M (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

theorem point_M_properties (m : ℝ) :
  (((M m).1 = (M m).2 ∨ (M m).1 = -(M m).2) → (m = -2/3 ∨ m = -4)) ∧
  (abs (M m).2 = 1 → (m = -1 ∨ m = -2)) := by
  sorry

end NUMINAMATH_CALUDE_point_M_properties_l4082_408212
