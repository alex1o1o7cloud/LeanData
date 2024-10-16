import Mathlib

namespace NUMINAMATH_CALUDE_awards_distribution_l1363_136343

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The condition that each student receives at least one award -/
def at_least_one_award (distribution : List ℕ) : Prop :=
  sorry

theorem awards_distribution :
  ∃ (d : List ℕ),
    d.length = 4 ∧
    d.sum = 6 ∧
    at_least_one_award d ∧
    distribute_awards 6 4 = 1560 :=
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l1363_136343


namespace NUMINAMATH_CALUDE_power_sum_unique_solution_l1363_136364

theorem power_sum_unique_solution (k : ℕ+) :
  (∃ (n : ℕ) (m : ℕ), m > 1 ∧ 3^k.val + 5^k.val = n^m) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_unique_solution_l1363_136364


namespace NUMINAMATH_CALUDE_hyperbolic_matrix_det_is_one_cosh_sq_sub_sinh_sq_l1363_136398

open Matrix Real

/-- The determinant of a specific 3x3 matrix involving hyperbolic functions is 1 -/
theorem hyperbolic_matrix_det_is_one (α β : ℝ) : 
  det !![cosh α * cosh β, cosh α * sinh β, -sinh α;
         -sinh β, cosh β, 0;
         sinh α * cosh β, sinh α * sinh β, cosh α] = 1 := by
  sorry

/-- The fundamental hyperbolic identity -/
theorem cosh_sq_sub_sinh_sq (x : ℝ) : cosh x * cosh x - sinh x * sinh x = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_matrix_det_is_one_cosh_sq_sub_sinh_sq_l1363_136398


namespace NUMINAMATH_CALUDE_remainder_of_3_power_20_l1363_136391

theorem remainder_of_3_power_20 (a : ℕ) : 
  a = (1 + 2)^20 → a % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_power_20_l1363_136391


namespace NUMINAMATH_CALUDE_test_score_calculation_l1363_136389

/-- The average test score for a portion of the class -/
def average_score (portion : ℝ) (score : ℝ) : ℝ := portion * score

/-- The overall class average -/
def class_average (score1 : ℝ) (score2 : ℝ) (score3 : ℝ) : ℝ :=
  average_score 0.45 0.95 + average_score 0.50 score2 + average_score 0.05 0.60

theorem test_score_calculation (score2 : ℝ) :
  class_average 0.95 score2 0.60 = 0.8475 → score2 = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l1363_136389


namespace NUMINAMATH_CALUDE_triangle_area_l1363_136366

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) :
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1363_136366


namespace NUMINAMATH_CALUDE_first_investment_interest_rate_l1363_136385

/-- Proves that the annual simple interest rate of the first investment is 8.5% -/
theorem first_investment_interest_rate 
  (total_income : ℝ) 
  (total_invested : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (second_rate : ℝ) :
  total_income = 575 →
  total_invested = 8000 →
  first_investment = 3000 →
  second_investment = 5000 →
  second_rate = 0.064 →
  ∃ (first_rate : ℝ), 
    first_rate = 0.085 ∧ 
    total_income = first_investment * first_rate + second_investment * second_rate :=
by
  sorry

end NUMINAMATH_CALUDE_first_investment_interest_rate_l1363_136385


namespace NUMINAMATH_CALUDE_mingming_calculation_correction_l1363_136352

theorem mingming_calculation_correction : 
  (-4 - 2/3) - (1 + 5/6) - (-18 - 1/2) + (-13 - 3/4) = -7/4 := by sorry

end NUMINAMATH_CALUDE_mingming_calculation_correction_l1363_136352


namespace NUMINAMATH_CALUDE_kathy_happy_probability_kathy_probability_sum_l1363_136335

def total_cards : ℕ := 10
def cards_laid_out : ℕ := 5
def red_cards : ℕ := 5
def green_cards : ℕ := 5

def happy_configurations : ℕ := 62
def total_configurations : ℕ := 30240

def probability_numerator : ℕ := 31
def probability_denominator : ℕ := 15120

theorem kathy_happy_probability :
  (happy_configurations : ℚ) / total_configurations = probability_numerator / probability_denominator :=
sorry

theorem kathy_probability_sum :
  probability_numerator + probability_denominator = 15151 :=
sorry

end NUMINAMATH_CALUDE_kathy_happy_probability_kathy_probability_sum_l1363_136335


namespace NUMINAMATH_CALUDE_perimeter_of_square_C_l1363_136384

/-- Given squares A, B, and C with specified properties, prove that the perimeter of square C is 64 units -/
theorem perimeter_of_square_C (a b c : ℝ) : 
  (4 * a = 16) →  -- Perimeter of square A is 16 units
  (4 * b = 48) →  -- Perimeter of square B is 48 units
  (c = a + b) →   -- Side length of C is sum of side lengths of A and B
  (4 * c = 64) :=  -- Perimeter of square C is 64 units
by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_square_C_l1363_136384


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1363_136353

theorem complex_fraction_equality : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1363_136353


namespace NUMINAMATH_CALUDE_walk_a_thon_earnings_l1363_136346

theorem walk_a_thon_earnings (last_year_rate last_year_miles this_year_rate : ℝ) 
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : this_year_rate * (last_year_miles + 5) = last_year_rate * last_year_miles) :
  last_year_rate * last_year_miles = 44 := by
  sorry

end NUMINAMATH_CALUDE_walk_a_thon_earnings_l1363_136346


namespace NUMINAMATH_CALUDE_quadratic_solution_l1363_136350

theorem quadratic_solution (x : ℚ) : 
  (63 * x^2 - 100 * x + 45 = 0) → 
  (63 * (5/7)^2 - 100 * (5/7) + 45 = 0) → 
  (63 * 1^2 - 100 * 1 + 45 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1363_136350


namespace NUMINAMATH_CALUDE_distance_covered_proof_l1363_136300

/-- Calculates the distance covered given a fuel-to-distance ratio and fuel consumption -/
def distance_covered (fuel_ratio : ℚ) (distance_ratio : ℚ) (fuel_consumed : ℚ) : ℚ :=
  (distance_ratio / fuel_ratio) * fuel_consumed

/-- Proves that given a fuel-to-distance ratio of 4:7 and fuel consumption of 44 gallons, 
    the distance covered is 77 miles -/
theorem distance_covered_proof :
  let fuel_ratio : ℚ := 4
  let distance_ratio : ℚ := 7
  let fuel_consumed : ℚ := 44
  distance_covered fuel_ratio distance_ratio fuel_consumed = 77 := by
sorry

end NUMINAMATH_CALUDE_distance_covered_proof_l1363_136300


namespace NUMINAMATH_CALUDE_solution_value_l1363_136304

theorem solution_value (x a : ℝ) (h : 2 * 2 + a = 3) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1363_136304


namespace NUMINAMATH_CALUDE_no_solution_for_absolute_value_equation_l1363_136370

theorem no_solution_for_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 4| = x^2 + 6*x + 8 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_absolute_value_equation_l1363_136370


namespace NUMINAMATH_CALUDE_max_value_g_range_of_k_l1363_136315

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (log x + k) / exp x

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f k x * exp x - x

-- Part 1
theorem max_value_g (k : ℝ) :
  (∃ x > 0, deriv (f k) x = 0) →
  (∃ x₀ > 0, ∀ x > 0, g k x ≤ g k x₀) ∧ 
  (∃ x₁ > 0, g k x₁ = 0) :=
sorry

-- Part 2
theorem range_of_k :
  (∃ k : ℝ, ∃ x ∈ Set.Ioo 0 1, deriv (f k) x = 0) →
  (∀ k : ℝ, (∃ x ∈ Set.Ioo 0 1, deriv (f k) x = 0) → k ≥ 1) ∧
  (∀ k ≥ 1, ∃ x ∈ Set.Ioo 0 1, deriv (f k) x = 0) :=
sorry

end NUMINAMATH_CALUDE_max_value_g_range_of_k_l1363_136315


namespace NUMINAMATH_CALUDE_polygon_properties_l1363_136314

theorem polygon_properties :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    -- Condition: Each interior angle is 30° more than four times its adjacent exterior angle
    (180 : ℝ) = exterior_angle + 4 * exterior_angle + 30 →
    -- Condition: Sum of exterior angles is always 360°
    (n : ℝ) * exterior_angle = 360 →
    -- Conclusions
    n = 12 ∧
    (n - 2 : ℝ) * 180 = 1800 ∧
    n * (n - 3) / 2 = 54 :=
by
  sorry


end NUMINAMATH_CALUDE_polygon_properties_l1363_136314


namespace NUMINAMATH_CALUDE_necklace_beads_l1363_136386

theorem necklace_beads (total blue red white silver : ℕ) : 
  total = 40 →
  blue = 5 →
  red = 2 * blue →
  white = blue + red →
  total = blue + red + white + silver →
  silver = 10 := by
sorry

end NUMINAMATH_CALUDE_necklace_beads_l1363_136386


namespace NUMINAMATH_CALUDE_A_and_D_independent_l1363_136303

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A
def A : Set Ω := {ω | ω.1 = 0}

-- Define event D
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- Theorem statement
theorem A_and_D_independent :
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_and_D_independent_l1363_136303


namespace NUMINAMATH_CALUDE_florist_bouquet_problem_l1363_136376

theorem florist_bouquet_problem (narcissus : ℕ) (chrysanthemums : ℕ) (total_bouquets : ℕ) :
  narcissus = 75 →
  chrysanthemums = 90 →
  total_bouquets = 33 →
  (narcissus + chrysanthemums) % total_bouquets = 0 →
  (narcissus + chrysanthemums) / total_bouquets = 5 :=
by sorry

end NUMINAMATH_CALUDE_florist_bouquet_problem_l1363_136376


namespace NUMINAMATH_CALUDE_no_trapezoid_solutions_l1363_136356

theorem no_trapezoid_solutions : ¬∃ (b₁ b₂ : ℤ),
  (1800 = (40 * (b₁ + b₂)) / 2) ∧
  (∃ (k : ℤ), b₁ = 2 * k + 1) ∧
  (∃ (m : ℤ), b₁ = 5 * m) ∧
  (∃ (n : ℤ), b₂ = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_no_trapezoid_solutions_l1363_136356


namespace NUMINAMATH_CALUDE_triangle_inequality_l1363_136318

theorem triangle_inequality (a b c : ℝ) : 
  (a + b + c = 2) → 
  (a > 0) → (b > 0) → (c > 0) →
  (a + b ≥ c) → (b + c ≥ a) → (c + a ≥ b) →
  abc + 1/27 ≥ ab + bc + ca - 1 ∧ ab + bc + ca - 1 ≥ abc := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1363_136318


namespace NUMINAMATH_CALUDE_apple_distribution_l1363_136317

theorem apple_distribution (total_apples : ℕ) (additional_people : ℕ) (apple_reduction : ℕ) :
  total_apples = 10000 →
  additional_people = 100 →
  apple_reduction = 15 →
  ∃ X : ℕ,
    (X * (total_apples / X) = total_apples) ∧
    ((X + additional_people) * (total_apples / X - apple_reduction) = total_apples) ∧
    X = 213 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1363_136317


namespace NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l1363_136354

theorem science_club_neither_math_nor_physics 
  (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 120)
  (h2 : math = 75)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 10 :=
by sorry

end NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l1363_136354


namespace NUMINAMATH_CALUDE_time_after_1550_minutes_l1363_136368

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime (midnight on January 1, 2011) -/
def startTime : DateTime :=
  { day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 1550

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { day := 2, hour := 1, minute := 50 }

/-- Theorem stating that adding 1550 minutes to midnight on January 1
    results in 1:50 AM on January 2 -/
theorem time_after_1550_minutes :
  addMinutes startTime minutesToAdd = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_time_after_1550_minutes_l1363_136368


namespace NUMINAMATH_CALUDE_negation_of_existence_equiv_forall_l1363_136380

theorem negation_of_existence_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_equiv_forall_l1363_136380


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1363_136313

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a (a : ℝ) : A ∪ B a = A → a ∈ ({0, 1/3, -1/2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1363_136313


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1363_136334

-- Define the properties of the parallelogram
def parallelogram_area : ℝ := 200
def parallelogram_height : ℝ := 20

-- Theorem statement
theorem parallelogram_base_length :
  ∃ (base : ℝ), base * parallelogram_height = parallelogram_area ∧ base = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1363_136334


namespace NUMINAMATH_CALUDE_triangle_theorem_l1363_136301

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific acute triangle -/
theorem triangle_theorem (t : Triangle) 
  (h_acute : 0 < t.A ∧ t.A < π/2 ∧ 0 < t.B ∧ t.B < π/2 ∧ 0 < t.C ∧ t.C < π/2)
  (h_cos : Real.cos t.A / Real.cos t.C = t.a / (2 * t.b - t.c))
  (h_a : t.a = Real.sqrt 7)
  (h_c : t.c = 3)
  (h_D : ∃ D : ℝ × ℝ, D = ((t.b + t.c)/2, 0)) :
  t.A = π/3 ∧ Real.sqrt ((t.b^2 + t.c^2 + 2*t.b*t.c*Real.cos t.A) / 4) = Real.sqrt 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1363_136301


namespace NUMINAMATH_CALUDE_not_divisible_1998_1000_l1363_136345

theorem not_divisible_1998_1000 (m : ℕ) : ¬(1000^m - 1 ∣ 1998^m - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_1998_1000_l1363_136345


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1363_136367

/-- The quadratic function f(x) = (x - 2)² - 3 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, -3)

theorem vertex_of_quadratic :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1363_136367


namespace NUMINAMATH_CALUDE_g_value_at_one_l1363_136309

def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - 2 * x * y

theorem g_value_at_one (g : ℝ → ℝ) (h : g_property g) : g 1 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_one_l1363_136309


namespace NUMINAMATH_CALUDE_f_properties_l1363_136326

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/4)^x - 8 * (1/2)^x - 1
  else if x = 0 then 0
  else -4^x + 8 * 2^x + 1

theorem f_properties :
  (∀ x, f x + f (-x) = 0) →
  (∀ x < 0, f x = (1/4)^x - 8 * (1/2)^x - 1) →
  (∀ x > 0, f x = -4^x + 8 * 2^x + 1) ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f y ≤ f x) ∧
  f 2 = 17 ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f x ≤ f y) ∧
  f 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1363_136326


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1363_136302

theorem quadratic_solution_sum (a b : ℝ) : 
  (a^2 - 6*a + 11 = 23) → 
  (b^2 - 6*b + 11 = 23) → 
  (a ≥ b) → 
  (a + 3*b = 12 - 2*Real.sqrt 21) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1363_136302


namespace NUMINAMATH_CALUDE_expand_and_simplify_simplify_complex_fraction_l1363_136327

-- Problem 1
theorem expand_and_simplify (x : ℝ) :
  (2*x - 1)*(2*x - 3) - (1 - 2*x)*(2 - x) = 2*x^2 - 3*x + 1 := by sorry

-- Problem 2
theorem simplify_complex_fraction (a : ℝ) (ha : a ≠ 0) (ha1 : a ≠ 1) :
  (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_simplify_complex_fraction_l1363_136327


namespace NUMINAMATH_CALUDE_integer_solution_exists_l1363_136332

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_exists_l1363_136332


namespace NUMINAMATH_CALUDE_right_and_obtuse_angles_in_clerts_l1363_136365

-- Define the number of clerts in a full Martian circle
def martian_full_circle : ℕ := 600

-- Define Earth angles in degrees
def earth_right_angle : ℕ := 90
def earth_obtuse_angle : ℕ := 135
def earth_full_circle : ℕ := 360

-- Define the conversion function from Earth degrees to Martian clerts
def earth_to_martian (earth_angle : ℕ) : ℕ :=
  (earth_angle * martian_full_circle) / earth_full_circle

-- Theorem statement
theorem right_and_obtuse_angles_in_clerts :
  earth_to_martian earth_right_angle = 150 ∧
  earth_to_martian earth_obtuse_angle = 225 := by
  sorry


end NUMINAMATH_CALUDE_right_and_obtuse_angles_in_clerts_l1363_136365


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1363_136316

theorem arithmetic_geometric_sequence_ratio (d : ℚ) (q : ℚ) (a : ℕ → ℚ) (b : ℕ → ℚ) :
  d ≠ 0 →
  0 < q →
  q < 1 →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∀ n : ℕ, b (n + 1) = b n * q) →
  a 1 = d →
  b 1 = d^2 →
  ∃ m : ℕ+, (a 1^2 + a 2^2 + a 3^2) / (b 1 + b 2 + b 3) = m →
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1363_136316


namespace NUMINAMATH_CALUDE_binomial_sum_l1363_136357

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  ((1 + 2 * x) ^ 5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry


end NUMINAMATH_CALUDE_binomial_sum_l1363_136357


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l1363_136342

/-- The units digit of m^2 + 3^m is 5, where m = 2023^2 + 3^2023 -/
theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ) : 
  m = 2023^2 + 3^2023 → (m^2 + 3^m) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l1363_136342


namespace NUMINAMATH_CALUDE_line_equation_problem_l1363_136312

/-- Two lines are the same if their coefficients are proportional -/
def same_line (a b c : ℝ) (d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ k * a = d ∧ k * b = e ∧ k * c = f

/-- The problem statement -/
theorem line_equation_problem (p q : ℝ) :
  same_line p 2 7 3 q 5 → p = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_problem_l1363_136312


namespace NUMINAMATH_CALUDE_sum_odd_implies_difference_odd_l1363_136306

theorem sum_odd_implies_difference_odd (a b : ℤ) : 
  Odd (a + b) → Odd (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_implies_difference_odd_l1363_136306


namespace NUMINAMATH_CALUDE_angle_relation_l1363_136351

theorem angle_relation (α β : Real) : 
  π / 2 < α ∧ α < π ∧
  π / 2 < β ∧ β < π ∧
  (1 - Real.cos (2 * α)) * (1 + Real.sin β) = Real.sin (2 * α) * Real.cos β →
  2 * α + β = 5 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_relation_l1363_136351


namespace NUMINAMATH_CALUDE_problem_solution_l1363_136387

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + 2*x + 2*y = 7) :
  x^2*y + x*y^2 = 245/121 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1363_136387


namespace NUMINAMATH_CALUDE_min_value_of_function_l1363_136321

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < π) :
  ∃ (y : ℝ), y = (2 - Real.cos x) / Real.sin x ∧
  (∀ (z : ℝ), z = (2 - Real.cos x) / Real.sin x → y ≤ z) ∧
  y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1363_136321


namespace NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_12_15_18_l1363_136361

theorem greatest_five_digit_divisible_by_12_15_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n → n ≤ 99900 := by
  sorry

#check greatest_five_digit_divisible_by_12_15_18

end NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_12_15_18_l1363_136361


namespace NUMINAMATH_CALUDE_candidate_x_win_percentage_l1363_136331

theorem candidate_x_win_percentage :
  ∀ (total_voters : ℕ) (republican_ratio democrat_ratio : ℚ) 
    (republican_for_x democrat_for_x : ℚ),
  republican_ratio / democrat_ratio = 3 / 2 →
  republican_for_x = 70 / 100 →
  democrat_for_x = 25 / 100 →
  let republicans := (republican_ratio / (republican_ratio + democrat_ratio)) * total_voters
  let democrats := (democrat_ratio / (republican_ratio + democrat_ratio)) * total_voters
  let votes_for_x := republican_for_x * republicans + democrat_for_x * democrats
  let votes_for_y := total_voters - votes_for_x
  (votes_for_x - votes_for_y) / total_voters = 4 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_x_win_percentage_l1363_136331


namespace NUMINAMATH_CALUDE_second_year_associates_percentage_l1363_136323

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  notFirstYear : ℝ
  moreThanTwoYears : ℝ

/-- Calculates the percentage of second-year associates -/
def secondYearPercentage (p : AssociatePercentages) : ℝ :=
  p.notFirstYear - p.moreThanTwoYears

/-- Theorem stating the percentage of second-year associates -/
theorem second_year_associates_percentage 
  (p : AssociatePercentages)
  (h1 : p.notFirstYear = 75)
  (h2 : p.moreThanTwoYears = 50) :
  secondYearPercentage p = 25 := by
  sorry

#check second_year_associates_percentage

end NUMINAMATH_CALUDE_second_year_associates_percentage_l1363_136323


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1363_136359

theorem polynomial_simplification (q : ℝ) : 
  (4 * q^3 - 7 * q^2 + 3 * q + 8) + (5 - 3 * q^3 + 9 * q^2 - 2 * q) = q^3 + 2 * q^2 + q + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1363_136359


namespace NUMINAMATH_CALUDE_similar_transformation_l1363_136375

structure Square where
  diagonal : ℝ

structure Transformation where
  area_after : ℝ
  is_similar : Bool

def original_square : Square := { diagonal := 2 }

def transformation : Transformation := { area_after := 4, is_similar := true }

theorem similar_transformation (s : Square) (t : Transformation) :
  s.diagonal = 2 ∧ t.area_after = 4 → t.is_similar = true := by
  sorry

end NUMINAMATH_CALUDE_similar_transformation_l1363_136375


namespace NUMINAMATH_CALUDE_quadratic_crosses_origin_l1363_136328

/-- Given a quadratic function g(x) = ax^2 + bx where a ≠ 0 and b ≠ 0,
    the graph crosses the x-axis at the origin. -/
theorem quadratic_crosses_origin (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ a * x^2 + b * x
  (g 0 = 0) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε \ {0}, g x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_crosses_origin_l1363_136328


namespace NUMINAMATH_CALUDE_impossibleToGetAllPlus_l1363_136338

/-- Represents a 4x4 grid of signs -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Flips all signs in a given row -/
def flipRow (g : Grid) (row : Fin 4) : Grid := sorry

/-- Flips all signs in a given column -/
def flipColumn (g : Grid) (col : Fin 4) : Grid := sorry

/-- The initial grid configuration -/
def initialGrid : Grid := 
  ![![true,  false, true,  true],
    ![true,  true,  true,  true],
    ![true,  true,  true,  true],
    ![true,  false, true,  true]]

/-- Checks if all cells in the grid are true ("+") -/
def allPlus (g : Grid) : Prop := ∀ i j, g i j = true

/-- Represents a sequence of row and column flipping operations -/
inductive FlipSequence : Type
  | empty : FlipSequence
  | flipRow : FlipSequence → Fin 4 → FlipSequence
  | flipColumn : FlipSequence → Fin 4 → FlipSequence

/-- Applies a sequence of flipping operations to a grid -/
def applyFlips : Grid → FlipSequence → Grid
  | g, FlipSequence.empty => g
  | g, FlipSequence.flipRow s i => applyFlips (flipRow g i) s
  | g, FlipSequence.flipColumn s j => applyFlips (flipColumn g j) s

theorem impossibleToGetAllPlus : 
  ¬∃ (s : FlipSequence), allPlus (applyFlips initialGrid s) := by
  sorry

end NUMINAMATH_CALUDE_impossibleToGetAllPlus_l1363_136338


namespace NUMINAMATH_CALUDE_extreme_points_condition_l1363_136399

/-- The function f(x) = ln x + ax^2 - 2x has two distinct extreme points
    if and only if 0 < a < 1/2, where x > 0 -/
theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → (((1 : ℝ) / x + 2 * a * x - 2 = 0) ↔ (x = x₁ ∨ x = x₂))))
  ↔ (0 < a ∧ a < (1 : ℝ) / 2) :=
by sorry


end NUMINAMATH_CALUDE_extreme_points_condition_l1363_136399


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1363_136382

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 125 * r = b ∧ b * r = 60 / 49) → 
  b = 50 * Real.sqrt 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1363_136382


namespace NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l1363_136395

theorem tan_eleven_pi_sixths : Real.tan (11 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l1363_136395


namespace NUMINAMATH_CALUDE_max_diagonals_theorem_l1363_136377

/-- The maximum number of non-intersecting or perpendicular diagonals in a regular n-gon -/
def max_diagonals (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 2 else n - 3

/-- Theorem stating the maximum number of non-intersecting or perpendicular diagonals in a regular n-gon -/
theorem max_diagonals_theorem (n : ℕ) (h : n ≥ 3) :
  max_diagonals n = if n % 2 = 0 then n - 2 else n - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_diagonals_theorem_l1363_136377


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l1363_136324

theorem parking_lot_spaces (total_spaces motorcycle_spaces ev_spaces : ℕ)
  (full_size_ratio compact_ratio : ℕ) :
  total_spaces = 750 →
  motorcycle_spaces = 50 →
  ev_spaces = 30 →
  full_size_ratio = 11 →
  compact_ratio = 4 →
  ∃ (full_size_spaces : ℕ),
    full_size_spaces = 489 ∧
    full_size_spaces * compact_ratio = (total_spaces - motorcycle_spaces - ev_spaces - full_size_spaces) * full_size_ratio :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_spaces_l1363_136324


namespace NUMINAMATH_CALUDE_root_location_l1363_136337

theorem root_location (a b : ℝ) (n : ℤ) : 
  (2 : ℝ)^a = 3 → 
  (3 : ℝ)^b = 2 → 
  (∃ x_b : ℝ, x_b ∈ Set.Ioo (n : ℝ) (n + 1) ∧ a^x_b + x_b - b = 0) → 
  n = -1 := by
sorry

end NUMINAMATH_CALUDE_root_location_l1363_136337


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1363_136305

/-- Two lines are parallel but not coincident -/
def parallel_not_coincident (a : ℝ) : Prop :=
  (a * 3 - 3 * (a - 1) = 0) ∧ (a * (a - 7) - 3 * (3 * a) ≠ 0)

/-- The condition a = 3 or a = -2 -/
def condition (a : ℝ) : Prop := a = 3 ∨ a = -2

theorem parallel_lines_condition :
  (∀ a : ℝ, parallel_not_coincident a → condition a) ∧
  ¬(∀ a : ℝ, condition a → parallel_not_coincident a) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1363_136305


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1363_136330

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/144
  let r : ℚ := -10/21
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1363_136330


namespace NUMINAMATH_CALUDE_quadratic_properties_l1363_136373

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - 4

-- Theorem stating the properties of the function
theorem quadratic_properties :
  ∃ a : ℝ, 
    (f a 0 = -3) ∧ 
    (∀ x, f 1 x = (x - 1)^2 - 4) ∧
    (∀ x > 1, ∀ y > x, f 1 y > f 1 x) ∧
    (f 1 (-1) = 0 ∧ f 1 3 = 0) ∧
    (∀ x, f 1 x = 0 → x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1363_136373


namespace NUMINAMATH_CALUDE_reciprocal_equation_l1363_136349

theorem reciprocal_equation (x : ℝ) : 
  (3 + 1 / (2 - x) = 2 * (1 / (2 - x))) → x = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l1363_136349


namespace NUMINAMATH_CALUDE_sum_max_at_5_l1363_136325

/-- An arithmetic sequence with its first term and sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum : ℕ → ℝ
  sum_def : ∀ n : ℕ, sum n = (n : ℝ) / 2 * (a 1 + a n)
  sum_9_positive : sum 9 > 0
  sum_10_negative : sum 10 < 0

/-- The sum of the arithmetic sequence is maximized at n = 5 -/
theorem sum_max_at_5 (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), seq.sum m ≤ seq.sum n ∧ n = 5 :=
sorry

end NUMINAMATH_CALUDE_sum_max_at_5_l1363_136325


namespace NUMINAMATH_CALUDE_p_20_equals_657_l1363_136348

/-- A polynomial p(x) = 3x^2 + kx + 117 where k is a constant such that p(1) = p(10) -/
def p (k : ℚ) (x : ℚ) : ℚ := 3 * x^2 + k * x + 117

/-- The theorem stating that for the polynomial p(x) with the given properties, p(20) = 657 -/
theorem p_20_equals_657 :
  ∃ k : ℚ, (p k 1 = p k 10) ∧ (p k 20 = 657) := by sorry

end NUMINAMATH_CALUDE_p_20_equals_657_l1363_136348


namespace NUMINAMATH_CALUDE_b_equals_three_l1363_136393

/-- The function f(x) -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1

/-- f(x) is monotonically increasing in the interval (1, 2) -/
def monotone_increasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y

/-- f(x) is monotonically decreasing in the interval (2, 3) -/
def monotone_decreasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 2 < x ∧ x < y ∧ y < 3 → f x > f y

/-- Main theorem: b equals 3 -/
theorem b_equals_three :
  ∃ b : ℝ, 
    (monotone_increasing_in_interval (f b)) ∧ 
    (monotone_decreasing_in_interval (f b)) → 
    b = 3 := by sorry

end NUMINAMATH_CALUDE_b_equals_three_l1363_136393


namespace NUMINAMATH_CALUDE_pebble_collection_proof_l1363_136374

def initial_pebbles : ℕ := 3
def collection_days : ℕ := 15
def first_day_collection : ℕ := 2
def daily_increase : ℕ := 1

def total_pebbles : ℕ := initial_pebbles + (collection_days * (2 * first_day_collection + (collection_days - 1) * daily_increase)) / 2

theorem pebble_collection_proof :
  total_pebbles = 138 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_proof_l1363_136374


namespace NUMINAMATH_CALUDE_quadrilateral_congruence_l1363_136388

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The median line of a quadrilateral -/
def median_line (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Two quadrilaterals are equal if their corresponding sides and median lines are equal -/
theorem quadrilateral_congruence (q1 q2 : Quadrilateral) :
  (q1.A = q2.A ∧ q1.B = q2.B ∧ q1.C = q2.C ∧ q1.D = q2.D) →
  median_line q1 = median_line q2 →
  q1 = q2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_congruence_l1363_136388


namespace NUMINAMATH_CALUDE_joes_investment_rate_l1363_136378

/-- Represents a simple interest bond investment -/
structure SimpleInterestBond where
  initialValue : ℝ
  interestRate : ℝ

/-- Calculates the value of a simple interest bond after a given number of years -/
def bondValue (bond : SimpleInterestBond) (years : ℝ) : ℝ :=
  bond.initialValue * (1 + bond.interestRate * years)

/-- Theorem: Given the conditions of Joe's investment, the interest rate is 1/13 -/
theorem joes_investment_rate : ∃ (bond : SimpleInterestBond),
  bondValue bond 3 = 260 ∧
  bondValue bond 8 = 360 ∧
  bond.interestRate = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_joes_investment_rate_l1363_136378


namespace NUMINAMATH_CALUDE_permutation_preserves_lines_l1363_136339

-- Define a type for points in a plane
variable {Point : Type*}

-- Define a permutation of points
variable (f : Point → Point)

-- Define what it means for three points to be collinear
def collinear (A B C : Point) : Prop := sorry

-- Define what it means for three points to lie on a circle
def on_circle (A B C : Point) : Prop := sorry

-- State the theorem
theorem permutation_preserves_lines 
  (h : ∀ A B C : Point, on_circle A B C → on_circle (f A) (f B) (f C)) :
  (∀ A B C : Point, collinear A B C ↔ collinear (f A) (f B) (f C)) :=
sorry

end NUMINAMATH_CALUDE_permutation_preserves_lines_l1363_136339


namespace NUMINAMATH_CALUDE_second_round_score_l1363_136308

/-- Represents the points scored in a round of darts --/
structure DartScore :=
  (points : ℕ)

/-- Represents the scores for three rounds of darts --/
structure ThreeRoundScores :=
  (round1 : DartScore)
  (round2 : DartScore)
  (round3 : DartScore)

/-- Defines the relationship between scores in three rounds --/
def validScores (scores : ThreeRoundScores) : Prop :=
  scores.round2.points = 2 * scores.round1.points ∧
  scores.round3.points = (3 * scores.round1.points : ℕ)

/-- Theorem: Given the conditions, the score in the second round is 48 --/
theorem second_round_score (scores : ThreeRoundScores) 
  (h : validScores scores) : scores.round2.points = 48 := by
  sorry

#check second_round_score

end NUMINAMATH_CALUDE_second_round_score_l1363_136308


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1363_136394

/-- The slopes of the asymptotes of a hyperbola -/
def asymptote_slopes (a b : ℝ) : Set ℝ :=
  {m : ℝ | m = b / a ∨ m = -b / a}

/-- Theorem: The slopes of the asymptotes of the hyperbola (x^2/16) - (y^2/25) = 1 are ±5/4 -/
theorem hyperbola_asymptote_slopes :
  asymptote_slopes 4 5 = {5/4, -5/4} := by
  sorry

#check hyperbola_asymptote_slopes

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1363_136394


namespace NUMINAMATH_CALUDE_lemonade_problem_l1363_136344

theorem lemonade_problem (x : ℝ) :
  x > 0 ∧
  (x + (x / 8 + 2) = (3 / 2 * x) - (x / 8 + 2)) →
  x + (3 / 2 * x) = 40 := by
sorry

end NUMINAMATH_CALUDE_lemonade_problem_l1363_136344


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l1363_136319

theorem angle_sum_at_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l1363_136319


namespace NUMINAMATH_CALUDE_stem_and_leaf_update_l1363_136307

/-- Represents a stem-and-leaf diagram --/
structure StemAndLeaf :=
  (stem : List ℕ)
  (leaf : List (List ℕ))

/-- The initial stem-and-leaf diagram --/
def initial_diagram : StemAndLeaf := {
  stem := [0, 1, 2, 3, 4],
  leaf := [[], [0, 0, 1, 2, 2, 3], [1, 5, 6], [0, 2, 4, 6], [1, 6]]
}

/-- Function to update ages in the diagram --/
def update_ages (d : StemAndLeaf) (years : ℕ) : StemAndLeaf :=
  sorry

/-- Theorem stating the time passed and the reconstruction of the new diagram --/
theorem stem_and_leaf_update :
  ∃ (years : ℕ) (new_diagram : StemAndLeaf),
    years = 6 ∧
    new_diagram = update_ages initial_diagram years ∧
    new_diagram.stem = [0, 1, 2, 3, 4] ∧
    new_diagram.leaf = [[],
                        [5, 5],
                        [1, 5, 6],
                        [0, 2, 4, 6],
                        [1, 6]] :=
  sorry

end NUMINAMATH_CALUDE_stem_and_leaf_update_l1363_136307


namespace NUMINAMATH_CALUDE_division_problem_l1363_136336

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1390) 
  (h2 : a = 1650) 
  (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1363_136336


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l1363_136363

-- Define the probabilities of hitting for Person A and Person B
def prob_A_hits : ℝ := 0.6
def prob_B_hits : ℝ := 0.5

-- Define the event of the plane being hit
def plane_hit (prob_A prob_B : ℝ) : Prop :=
  1 - (1 - prob_A) * (1 - prob_B) = 0.8

-- State the theorem
theorem enemy_plane_hit_probability :
  plane_hit prob_A_hits prob_B_hits :=
by sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l1363_136363


namespace NUMINAMATH_CALUDE_tetrahedron_cross_section_perimeter_bounds_l1363_136371

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A quadrilateral cross-section of a regular tetrahedron -/
structure TetrahedronCrossSection (t : RegularTetrahedron) where
  perimeter : ℝ
  is_quadrilateral : True  -- This is a placeholder for the quadrilateral property

/-- The perimeter of a quadrilateral cross-section of a regular tetrahedron 
    is between 2a and 3a, where a is the edge length of the tetrahedron -/
theorem tetrahedron_cross_section_perimeter_bounds 
  (t : RegularTetrahedron) (c : TetrahedronCrossSection t) : 
  2 * t.edge_length ≤ c.perimeter ∧ c.perimeter ≤ 3 * t.edge_length :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_cross_section_perimeter_bounds_l1363_136371


namespace NUMINAMATH_CALUDE_range_of_a_l1363_136310

/-- Proposition p -/
def p (x : ℝ) : Prop := (2*x - 1) / (x - 1) < 0

/-- Proposition q -/
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) : 
  (∀ x, p x ↔ q x a) → 0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1363_136310


namespace NUMINAMATH_CALUDE_total_cubes_after_distribution_l1363_136372

/-- Represents the number of cubes a person has -/
structure CubeCount where
  red : ℕ
  blue : ℕ

def Grady : CubeCount := { red := 20, blue := 15 }
def Gage : CubeCount := { red := 10, blue := 12 }
def Harper : CubeCount := { red := 8, blue := 10 }

def giveToGage (c : CubeCount) : CubeCount :=
  { red := c.red * 2 / 5, blue := c.blue / 3 }

def remainingAfterGage (initial : CubeCount) (given : CubeCount) : CubeCount :=
  { red := initial.red - given.red, blue := initial.blue - given.blue }

def giveToHarper (c : CubeCount) : CubeCount :=
  { red := c.red / 4, blue := c.blue / 2 }

def totalCubes (c : CubeCount) : ℕ := c.red + c.blue

theorem total_cubes_after_distribution :
  let gageGiven := giveToGage Grady
  let harperGiven := giveToHarper (remainingAfterGage Grady gageGiven)
  let finalGage := { red := Gage.red + gageGiven.red, blue := Gage.blue + gageGiven.blue }
  let finalHarper := { red := Harper.red + harperGiven.red, blue := Harper.blue + harperGiven.blue }
  totalCubes finalGage + totalCubes finalHarper = 61 := by
  sorry


end NUMINAMATH_CALUDE_total_cubes_after_distribution_l1363_136372


namespace NUMINAMATH_CALUDE_constant_product_percentage_change_l1363_136390

theorem constant_product_percentage_change (x y : ℝ) (C : ℝ) (h : x * y = C) :
  x * (1 + 0.2) * (y * (1 - 1/6)) = C := by sorry

end NUMINAMATH_CALUDE_constant_product_percentage_change_l1363_136390


namespace NUMINAMATH_CALUDE_seokjin_paper_count_prove_seokjin_paper_count_l1363_136381

theorem seokjin_paper_count : ℕ → ℕ → ℕ → Prop :=
  fun jimin_count seokjin_count difference =>
    (jimin_count = 41) →
    (seokjin_count = jimin_count - difference) →
    (difference = 1) →
    (seokjin_count = 40)

#check seokjin_paper_count

theorem prove_seokjin_paper_count :
  seokjin_paper_count 41 40 1 := by sorry

end NUMINAMATH_CALUDE_seokjin_paper_count_prove_seokjin_paper_count_l1363_136381


namespace NUMINAMATH_CALUDE_tim_total_sleep_l1363_136320

/-- Tim's weekly sleep schedule -/
structure SleepSchedule where
  weekdays : Nat -- Number of weekdays
  weekdaySleep : Nat -- Hours of sleep on weekdays
  weekends : Nat -- Number of weekend days
  weekendSleep : Nat -- Hours of sleep on weekends

/-- Calculate total sleep based on a sleep schedule -/
def totalSleep (schedule : SleepSchedule) : Nat :=
  schedule.weekdays * schedule.weekdaySleep + schedule.weekends * schedule.weekendSleep

/-- Tim's actual sleep schedule -/
def timSchedule : SleepSchedule :=
  { weekdays := 5
    weekdaySleep := 6
    weekends := 2
    weekendSleep := 10 }

/-- Theorem: Tim's total sleep per week is 50 hours -/
theorem tim_total_sleep : totalSleep timSchedule = 50 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_sleep_l1363_136320


namespace NUMINAMATH_CALUDE_screen_width_calculation_l1363_136397

theorem screen_width_calculation (height width diagonal : ℝ) : 
  height / width = 3 / 4 →
  height^2 + width^2 = diagonal^2 →
  diagonal = 36 →
  width = 28.8 :=
by sorry

end NUMINAMATH_CALUDE_screen_width_calculation_l1363_136397


namespace NUMINAMATH_CALUDE_forgotten_lawns_l1363_136396

/-- Proves the number of forgotten lawns given Henry's lawn mowing situation -/
theorem forgotten_lawns (dollars_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) : 
  dollars_per_lawn = 5 → 
  total_lawns = 12 → 
  actual_earnings = 25 → 
  total_lawns - (actual_earnings / dollars_per_lawn) = 7 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_lawns_l1363_136396


namespace NUMINAMATH_CALUDE_product_equality_l1363_136341

theorem product_equality (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^12 + 21 * x^8 * y^3 + 21 * x^4 * y^6 + 49 * y^9) =
  81 * x^16 - 2401 * y^12 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1363_136341


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l1363_136322

/-- Proves that for a point P(-√3, y) on the terminal side of angle β, 
    where sin β = √13/13, the value of y is 1/2. -/
theorem point_on_terminal_side (β : ℝ) (y : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = -Real.sqrt 3 ∧ P.2 = y ∧ 
    Real.sin β = Real.sqrt 13 / 13 ∧ 
    (P.1 ≥ 0 ∨ (P.1 < 0 ∧ P.2 > 0))) → 
  y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l1363_136322


namespace NUMINAMATH_CALUDE_equations_not_equivalent_l1363_136369

theorem equations_not_equivalent : 
  ¬(∀ x : ℝ, (Real.sqrt (x^2 + x - 5) = Real.sqrt (x - 1)) ↔ (x^2 + x - 5 = x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equations_not_equivalent_l1363_136369


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l1363_136347

def red_eggs : ℕ := 15
def blue_eggs : ℕ := 30
def min_eggs_per_basket : ℕ := 3

def is_valid_distribution (eggs_per_basket : ℕ) : Prop :=
  eggs_per_basket ≥ min_eggs_per_basket ∧
  red_eggs % eggs_per_basket = 0 ∧
  blue_eggs % eggs_per_basket = 0

theorem max_eggs_per_basket :
  ∃ (max : ℕ), is_valid_distribution max ∧
    ∀ (n : ℕ), is_valid_distribution n → n ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l1363_136347


namespace NUMINAMATH_CALUDE_intersection_M_N_l1363_136358

-- Define set M
def M : Set ℝ := {x | |x| < 1}

-- Define set N
def N : Set ℝ := {y | ∃ x ∈ M, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1363_136358


namespace NUMINAMATH_CALUDE_divisors_of_prime_products_l1363_136311

theorem divisors_of_prime_products (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  let num_divisors := fun x => (Nat.divisors x).card
  (num_divisors (p * q) = 4) ∧
  (num_divisors (p^2 * q) = 6) ∧
  (num_divisors (p^2 * q^2) = 9) ∧
  (num_divisors (p^m * q^n) = (m + 1) * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_prime_products_l1363_136311


namespace NUMINAMATH_CALUDE_employee_salary_proof_l1363_136392

/-- Given two employees with a total weekly salary and a salary ratio, prove the salary of one employee. -/
theorem employee_salary_proof (total : ℚ) (ratio : ℚ) (n_salary : ℚ) : 
  total = 583 →
  ratio = 1.2 →
  n_salary + ratio * n_salary = total →
  n_salary = 265 := by
sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l1363_136392


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1363_136355

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (z₁ : ℂ), z₁ = m * (m - 1) + (m - 1) * Complex.I ∧ z₁.re = 0 ∧ z₁.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1363_136355


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1363_136333

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1363_136333


namespace NUMINAMATH_CALUDE_equation_solution_l1363_136362

theorem equation_solution : ∃ t : ℝ, t = 1.5 ∧ 4 * (4 : ℝ)^t + Real.sqrt (16 * 16^t) = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1363_136362


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l1363_136383

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ := 
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers : total_stocking_stuffers = 21 := by
  sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l1363_136383


namespace NUMINAMATH_CALUDE_inverse_of_singular_matrix_l1363_136360

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 10, 6]

theorem inverse_of_singular_matrix :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_singular_matrix_l1363_136360


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1363_136329

def complex_number : ℂ := 2 - Complex.I

theorem point_in_fourth_quadrant (z : ℂ) (h : z = complex_number) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1363_136329


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1363_136379

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x - Real.sqrt 2)^3 * (x + Real.sqrt 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1363_136379


namespace NUMINAMATH_CALUDE_polynomial_identity_l1363_136340

/-- Given a polynomial function f such that f(x^2 + 1) = x^4 + 4x^2 for all x,
    prove that f(x^2 - 1) = x^4 - 4 for all x. -/
theorem polynomial_identity (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 4*x^2) :
  ∀ x : ℝ, f (x^2 - 1) = x^4 - 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1363_136340
