import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_identity_l2083_208383

-- Define g as a polynomial function
variable (g : ℝ → ℝ)

-- State the theorem
theorem polynomial_identity 
  (h : ∀ x, g (x^2 + 2) = x^4 + 6*x^2 + 8) :
  ∀ x, g (x^2 - 1) = x^4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2083_208383


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2083_208333

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_3 : a 3 = 16)
  (h_9 : a 9 = 80) :
  a 6 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2083_208333


namespace NUMINAMATH_CALUDE_tangent_circles_slope_l2083_208322

/-- Definition of circle w1 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 20*y - 77 = 0

/-- Definition of circle w2 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 20*y + 193 = 0

/-- Definition of a line y = mx -/
def line (m x y : ℝ) : Prop := y = m * x

/-- Definition of internal tangency -/
def internallyTangent (x y r : ℝ) : Prop := (x - 5)^2 + (y - 10)^2 = (8 - r)^2

/-- Definition of external tangency -/
def externallyTangent (x y r : ℝ) : Prop := (x + 5)^2 + (y - 10)^2 = (r + 12)^2

/-- Main theorem -/
theorem tangent_circles_slope : 
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (m' : ℝ), m' > 0 → 
    (∃ (x y r : ℝ), line m' x y ∧ internallyTangent x y r ∧ externallyTangent x y r) 
    → m' ≥ m) ∧ 
  m^2 = 81/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_slope_l2083_208322


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2083_208345

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℝ, (x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) ∧ 
  (∀ x : ℤ, x < 2 → ¬((x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2))) →
  -3 < m ∧ m ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2083_208345


namespace NUMINAMATH_CALUDE_group_purchase_equations_l2083_208327

theorem group_purchase_equations (x y : ℤ) : 
  (∀ (z : ℤ), z * x - y = 5 → z = 9) ∧ 
  (∀ (w : ℤ), y - w * x = 4 → w = 6) → 
  (9 * x - 5 = y ∧ 6 * x + 4 = y) := by
  sorry

end NUMINAMATH_CALUDE_group_purchase_equations_l2083_208327


namespace NUMINAMATH_CALUDE_sum_25_36_in_base3_l2083_208326

/-- Converts a natural number from base 10 to base 3 -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def fromBase3 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_25_36_in_base3 :
  toBase3 (25 + 36) = [2, 0, 2, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_25_36_in_base3_l2083_208326


namespace NUMINAMATH_CALUDE_relay_team_orders_l2083_208361

/-- The number of permutations of n elements -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of different orders for a relay team of 6 runners,
    where one specific runner is fixed to run the last lap -/
def relay_orders : ℕ := factorial 5

theorem relay_team_orders :
  relay_orders = 120 := by sorry

end NUMINAMATH_CALUDE_relay_team_orders_l2083_208361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2083_208342

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_4 + a_6 + a_8 = 12,
    prove that a_8 - (1/2)a_10 = 2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 4 + a 6 + a 8 = 12) :
  a 8 - (1/2) * a 10 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2083_208342


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l2083_208341

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 2 * x^2 - 10 * x

-- State the theorem
theorem quadratic_function_unique :
  (∀ x, f x < 0 ↔ 0 < x ∧ x < 5) →
  (∀ x ∈ Set.Icc (-1) 4, f x ≤ 12) →
  (∃ x ∈ Set.Icc (-1) 4, f x = 12) →
  (∀ x, f x = 2 * x^2 - 10 * x) :=
by sorry

-- Note: The condition a < 0 is not used in this theorem as it's only relevant for part II of the original problem

end NUMINAMATH_CALUDE_quadratic_function_unique_l2083_208341


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_54_l2083_208301

/-- A triangle with side lengths 9, 12, and 15 units has an area of 54 square units. -/
theorem triangle_area : ℝ :=
  let a := 9
  let b := 12
  let c := 15
  let s := (a + b + c) / 2  -- semi-perimeter
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))  -- Heron's formula
  54

/-- The theorem statement -/
theorem triangle_area_is_54 : triangle_area = 54 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_54_l2083_208301


namespace NUMINAMATH_CALUDE_circle_number_assignment_l2083_208373

theorem circle_number_assignment :
  let valid_assignment (a b c : ℕ) : Prop :=
    14 * 4 * a = 14 * 6 * c ∧
    14 * 4 * a = a * b * c ∧
    a > 0 ∧ b > 0 ∧ c > 0
  ∃! (solutions : Finset (ℕ × ℕ × ℕ)),
    solutions.card = 6 ∧
    ∀ (abc : ℕ × ℕ × ℕ), abc ∈ solutions ↔ valid_assignment abc.1 abc.2.1 abc.2.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_number_assignment_l2083_208373


namespace NUMINAMATH_CALUDE_new_light_wattage_l2083_208390

/-- Given a light with a rating of 60 watts, a new light with 12% higher wattage will have 67.2 watts. -/
theorem new_light_wattage :
  let original_wattage : ℝ := 60
  let increase_percentage : ℝ := 12
  let new_wattage : ℝ := original_wattage * (1 + increase_percentage / 100)
  new_wattage = 67.2 := by
  sorry

end NUMINAMATH_CALUDE_new_light_wattage_l2083_208390


namespace NUMINAMATH_CALUDE_min_value_theorem_l2083_208323

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 4) :
  (1 / x + 4 / y + 9 / z) ≥ 9 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
  x' + y' + z' = 4 ∧ 1 / x' + 4 / y' + 9 / z' = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2083_208323


namespace NUMINAMATH_CALUDE_robert_nickel_difference_l2083_208337

/-- Represents the number of chocolates eaten by each person -/
structure Chocolates where
  sarah : ℕ
  nickel : ℕ
  robert : ℕ

/-- The chocolate eating scenario -/
def chocolate_scenario : Chocolates :=
  { sarah := 15,
    nickel := 15 - 5,
    robert := 2 * (15 - 5) }

/-- Theorem stating the difference between Robert's and Nickel's chocolates -/
theorem robert_nickel_difference :
  chocolate_scenario.robert - chocolate_scenario.nickel = 10 := by
  sorry

end NUMINAMATH_CALUDE_robert_nickel_difference_l2083_208337


namespace NUMINAMATH_CALUDE_angle_between_specific_vectors_l2083_208358

/-- The angle between two 3D vectors given by their coordinates --/
def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem: The angle between vectors (3, -2, 2) and (2, 2, -1) is 90° --/
theorem angle_between_specific_vectors :
  angle_between_vectors (3, -2, 2) (2, 2, -1) = 90 := by sorry

end NUMINAMATH_CALUDE_angle_between_specific_vectors_l2083_208358


namespace NUMINAMATH_CALUDE_NaHCO3_moles_equal_H2O_moles_NaHCO3_moles_proof_l2083_208334

-- Define the molar masses and quantities
def molar_mass_H2O : ℝ := 18
def HNO3_moles : ℝ := 2
def H2O_grams : ℝ := 36

-- Define the reaction stoichiometry
def HNO3_to_H2O_ratio : ℝ := 1
def NaHCO3_to_H2O_ratio : ℝ := 1

-- Theorem statement
theorem NaHCO3_moles_equal_H2O_moles : ℝ → Prop :=
  fun NaHCO3_moles =>
    let H2O_moles := H2O_grams / molar_mass_H2O
    NaHCO3_moles = H2O_moles ∧ NaHCO3_moles = HNO3_moles

-- Proof (skipped)
theorem NaHCO3_moles_proof : ∃ (x : ℝ), NaHCO3_moles_equal_H2O_moles x :=
sorry

end NUMINAMATH_CALUDE_NaHCO3_moles_equal_H2O_moles_NaHCO3_moles_proof_l2083_208334


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l2083_208328

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, x^2 - x*y + 4*y^2 ≥ 0 ∧ (x^2 - x*y + 4*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l2083_208328


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l2083_208398

/-- The average price of books bought by Sandy -/
def average_price (shop1_books shop2_books : ℕ) (shop1_cost shop2_cost : ℚ) : ℚ :=
  (shop1_cost + shop2_cost) / (shop1_books + shop2_books)

/-- Theorem: The average price Sandy paid per book is $16 -/
theorem sandy_average_book_price :
  average_price 65 55 1080 840 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sandy_average_book_price_l2083_208398


namespace NUMINAMATH_CALUDE_min_sum_squares_l2083_208367

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 2*z = 6) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + 2*c = 6 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             m = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2083_208367


namespace NUMINAMATH_CALUDE_eldest_age_is_fifteen_l2083_208336

/-- The ages of three grandchildren satisfying specific conditions -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ
  age_difference : middle - youngest = 3
  eldest_triple_youngest : eldest = 3 * youngest
  eldest_sum_plus_two : eldest = youngest + middle + 2

/-- The age of the eldest grandchild is 15 -/
theorem eldest_age_is_fifteen (ages : GrandchildrenAges) : ages.eldest = 15 := by
  sorry

end NUMINAMATH_CALUDE_eldest_age_is_fifteen_l2083_208336


namespace NUMINAMATH_CALUDE_binomial_threshold_l2083_208378

theorem binomial_threshold (n : ℕ) : 
  (n ≥ 82 → Nat.choose (2*n) n < 4^(n-2)) ∧ 
  (n ≥ 1305 → Nat.choose (2*n) n < 4^(n-3)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_threshold_l2083_208378


namespace NUMINAMATH_CALUDE_system_solution_and_arithmetic_progression_l2083_208365

-- Define the system of equations
def system (m a b c x y z : ℝ) : Prop :=
  x + y + m*z = a ∧ x + m*y + z = b ∧ m*x + y + z = c

-- Theorem statement
theorem system_solution_and_arithmetic_progression
  (m a b c : ℝ) :
  (∃! (x y z : ℝ), system m a b c x y z) ↔ 
    (m ≠ -2 ∧ m ≠ 1) ∧
  (∀ (x y z : ℝ), system m a b c x y z → (2*y = x + z) ↔ a + c = b) :=
sorry

end NUMINAMATH_CALUDE_system_solution_and_arithmetic_progression_l2083_208365


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2083_208399

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  cylinder_height : ℝ
  h_cylinder_diameter_height : cylinder_height = 2 * cylinder_radius
  h_axes_coincide : True

/-- The theorem stating the radius of the inscribed cylinder -/
theorem inscribed_cylinder_radius 
  (c : InscribedCylinder) 
  (h_cone_diameter : c.cone_diameter = 16)
  (h_cone_altitude : c.cone_altitude = 20) :
  c.cylinder_radius = 40 / 9 := by
  sorry

#check inscribed_cylinder_radius

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2083_208399


namespace NUMINAMATH_CALUDE_range_of_a1_l2083_208370

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = 2 * (|a n| - 1)

/-- The sequence is bounded by some positive constant M -/
def BoundedSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ+, |a n| ≤ M

/-- The main theorem stating the range of a₁ -/
theorem range_of_a1 (a : ℕ+ → ℝ) 
    (h1 : RecurrenceSequence a) 
    (h2 : BoundedSequence a) : 
    -2 ≤ a 1 ∧ a 1 ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a1_l2083_208370


namespace NUMINAMATH_CALUDE_integer_fraction_conditions_l2083_208359

theorem integer_fraction_conditions (p a b : ℕ) : 
  Prime p → 
  a > 0 → 
  b > 0 → 
  (∃ k : ℤ, (4 * a + p : ℤ) / b + (4 * b + p : ℤ) / a = k) → 
  (∃ m : ℤ, (a^2 : ℤ) / b + (b^2 : ℤ) / a = m) → 
  a = b ∨ a = p * b :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_conditions_l2083_208359


namespace NUMINAMATH_CALUDE_inequality_solution_l2083_208300

-- Define the inequality function
def f (y : ℝ) : Prop :=
  y^3 / (y + 2) ≥ 3 / (y - 2) + 9/4

-- Define the solution set
def S : Set ℝ :=
  {y | y ∈ Set.Ioo (-2) 2 ∨ y ∈ Set.Ici 3}

-- State the theorem
theorem inequality_solution :
  {y : ℝ | f y} = S :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2083_208300


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_286_l2083_208329

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | is_prime p ∧ p ∣ n}

theorem sum_two_smallest_prime_factors_of_286 :
  let factors := prime_factors 286
  ∃ p q : ℕ, p ∈ factors ∧ q ∈ factors ∧ p ≠ q ∧
    (∀ r ∈ factors, r ≠ p → r ≠ q → p ≤ r ∧ q ≤ r) ∧
    p + q = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_286_l2083_208329


namespace NUMINAMATH_CALUDE_table_height_l2083_208388

/-- Represents the configuration of two identical blocks on a table -/
structure BlockConfiguration where
  l : ℝ  -- length of each block
  w : ℝ  -- width of each block
  h : ℝ  -- height of the table

/-- The length measurement in Figure 1 -/
def figure1_length (config : BlockConfiguration) : ℝ :=
  config.l + config.h - config.w

/-- The length measurement in Figure 2 -/
def figure2_length (config : BlockConfiguration) : ℝ :=
  config.w + config.h - config.l

/-- The main theorem stating the height of the table -/
theorem table_height (config : BlockConfiguration) 
  (h1 : figure1_length config = 32)
  (h2 : figure2_length config = 28) : 
  config.h = 30 := by
  sorry


end NUMINAMATH_CALUDE_table_height_l2083_208388


namespace NUMINAMATH_CALUDE_distribution_schemes_correct_l2083_208321

/-- The number of ways to distribute 5 volunteers to 3 different Olympic venues,
    with at least one volunteer assigned to each venue. -/
def distributionSchemes : ℕ := 150

/-- Theorem stating that the number of distribution schemes is correct. -/
theorem distribution_schemes_correct : distributionSchemes = 150 := by sorry

end NUMINAMATH_CALUDE_distribution_schemes_correct_l2083_208321


namespace NUMINAMATH_CALUDE_stanley_tire_cost_l2083_208379

/-- The total cost of tires purchased by Stanley -/
def total_cost (num_tires : ℕ) (cost_per_tire : ℚ) : ℚ :=
  num_tires * cost_per_tire

/-- Proof that Stanley's total cost for tires is $240.00 -/
theorem stanley_tire_cost :
  let num_tires : ℕ := 4
  let cost_per_tire : ℚ := 60
  total_cost num_tires cost_per_tire = 240 := by
  sorry

#eval total_cost 4 60

end NUMINAMATH_CALUDE_stanley_tire_cost_l2083_208379


namespace NUMINAMATH_CALUDE_remainder_sum_mod15_l2083_208344

theorem remainder_sum_mod15 (p q : ℤ) 
  (hp : p % 60 = 53) 
  (hq : q % 75 = 24) : 
  (p + q) % 15 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_mod15_l2083_208344


namespace NUMINAMATH_CALUDE_max_distance_F₂_to_l_max_value_PF₂_QF₂_range_F₁P_dot_F₁Q_l2083_208371

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define a line passing through F₁
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define the intersection points P and Q
def Intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ Ellipse x y ∧ Line k x y}

-- State the theorems
theorem max_distance_F₂_to_l :
  ∃ (k : ℝ), ∀ (l : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), l x y ↔ Line k x y) →
    (∃ (d : ℝ), d = 4 ∧ ∀ (p : ℝ × ℝ), l p.1 p.2 → dist F₂ p ≤ d) :=
sorry

theorem max_value_PF₂_QF₂ :
  ∃ (P Q : ℝ × ℝ), P ∈ Intersection k ∧ Q ∈ Intersection k ∧
    ∀ (P' Q' : ℝ × ℝ), P' ∈ Intersection k → Q' ∈ Intersection k →
      dist P' F₂ + dist Q' F₂ ≤ 26/3 :=
sorry

theorem range_F₁P_dot_F₁Q :
  ∀ (k : ℝ), ∀ (P Q : ℝ × ℝ),
    P ∈ Intersection k → Q ∈ Intersection k →
    -5 ≤ (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) ∧
    (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) ≤ -25/9 :=
sorry

end NUMINAMATH_CALUDE_max_distance_F₂_to_l_max_value_PF₂_QF₂_range_F₁P_dot_F₁Q_l2083_208371


namespace NUMINAMATH_CALUDE_absolute_difference_of_numbers_l2083_208352

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (product_eq : x * y = 396) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_numbers_l2083_208352


namespace NUMINAMATH_CALUDE_complex_simplification_l2083_208324

theorem complex_simplification :
  3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - Complex.I) + 2 * (1 + 2 * Complex.I) = 10 - 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2083_208324


namespace NUMINAMATH_CALUDE_hexagon_five_layers_dots_l2083_208302

/-- Calculates the number of dots in a hexagonal layer -/
def dots_in_layer (n : ℕ) : ℕ := 6 * n

/-- Calculates the total number of dots up to and including a given layer -/
def total_dots (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => total_dots m + dots_in_layer (m + 1)

theorem hexagon_five_layers_dots :
  total_dots 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_five_layers_dots_l2083_208302


namespace NUMINAMATH_CALUDE_unscreened_percentage_l2083_208350

/-- Calculates the percentage of unscreened part of a TV -/
theorem unscreened_percentage (tv_length tv_width screen_length screen_width : ℕ) 
  (h1 : tv_length = 6) (h2 : tv_width = 5) (h3 : screen_length = 5) (h4 : screen_width = 4) :
  (1 : ℚ) / 3 * 100 = 
    (tv_length * tv_width - screen_length * screen_width : ℚ) / (tv_length * tv_width) * 100 := by
  sorry

end NUMINAMATH_CALUDE_unscreened_percentage_l2083_208350


namespace NUMINAMATH_CALUDE_exam_scores_l2083_208393

theorem exam_scores (full_marks : ℝ) (a b c d : ℝ) : 
  full_marks = 500 →
  a = b * 0.9 →
  b = c * 1.25 →
  c = d * 0.8 →
  a = 360 →
  d / full_marks = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_l2083_208393


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2083_208382

def polynomial (x : ℤ) : ℤ := x^3 + 2*x^2 - 5*x + 30

def is_root (x : ℤ) : Prop := polynomial x = 0

def divisors_of_30 : Set ℤ := {x : ℤ | x ∣ 30 ∨ x ∣ -30}

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = divisors_of_30 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2083_208382


namespace NUMINAMATH_CALUDE_result_units_digit_is_seven_l2083_208347

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- The original three-digit number satisfying the condition -/
def original : ThreeDigitNumber := sorry

/-- The condition that the hundreds digit is 3 more than the units digit -/
axiom hundreds_units_relation : original.hundreds = original.units + 3

/-- The reversed number -/
def reversed : ThreeDigitNumber := sorry

/-- The result of subtracting the reversed number from the original number -/
def result : Nat := 
  (100 * original.hundreds + 10 * original.tens + original.units) - 
  (100 * reversed.hundreds + 10 * reversed.tens + reversed.units)

/-- The theorem stating that the units digit of the result is 7 -/
theorem result_units_digit_is_seven : result % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_result_units_digit_is_seven_l2083_208347


namespace NUMINAMATH_CALUDE_phone_price_increase_l2083_208311

/-- Proves that the percentage increase in the phone's price was 40% given the auction conditions --/
theorem phone_price_increase (tv_initial_price : ℝ) (tv_price_increase_ratio : ℝ) 
  (phone_initial_price : ℝ) (total_amount : ℝ) :
  tv_initial_price = 500 →
  tv_price_increase_ratio = 2 / 5 →
  phone_initial_price = 400 →
  total_amount = 1260 →
  let tv_final_price := tv_initial_price * (1 + tv_price_increase_ratio)
  let phone_final_price := total_amount - tv_final_price
  let phone_price_increase := (phone_final_price - phone_initial_price) / phone_initial_price
  phone_price_increase = 0.4 := by
sorry

end NUMINAMATH_CALUDE_phone_price_increase_l2083_208311


namespace NUMINAMATH_CALUDE_inequality_solution_l2083_208312

theorem inequality_solution (a : ℝ) (x : ℝ) : 
  (x + 1) * ((a - 1) * x - 1) > 0 ↔ 
    (a < 0 ∧ -1 < x ∧ x < 1 / (a - 1)) ∨
    (0 < a ∧ a < 1 ∧ 1 / (a - 1) < x ∧ x < -1) ∨
    (a = 1 ∧ x < -1) ∨
    (a > 1 ∧ (x < -1 ∨ x > 1 / (a - 1))) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2083_208312


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2083_208330

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 5 * x^2 + 10 * x + 15 → y ≥ min_y ∧ ∃ (x₀ : ℝ), 5 * x₀^2 + 10 * x₀ + 15 = min_y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2083_208330


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2083_208354

theorem solve_exponential_equation :
  ∃ x : ℝ, (2 : ℝ) ^ (x - 3) = 4 ^ (x + 1) ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2083_208354


namespace NUMINAMATH_CALUDE_total_cost_is_100_l2083_208315

-- Define the number of shirts
def num_shirts : ℕ := 10

-- Define the number of pants as half the number of shirts
def num_pants : ℕ := num_shirts / 2

-- Define the cost of each shirt
def cost_per_shirt : ℕ := 6

-- Define the cost of each pair of pants
def cost_per_pants : ℕ := 8

-- Theorem to prove the total cost
theorem total_cost_is_100 :
  num_shirts * cost_per_shirt + num_pants * cost_per_pants = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_100_l2083_208315


namespace NUMINAMATH_CALUDE_sum_of_digits_in_period_of_one_over_98_squared_l2083_208377

/-- The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) -/
def sum_of_digits_in_period (n : ℕ) : ℕ :=
  sorry

/-- The period length of the repeating decimal expansion of 1/(98^2) -/
def period_length : ℕ := 196

/-- Theorem: The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) is 882 -/
theorem sum_of_digits_in_period_of_one_over_98_squared :
  sum_of_digits_in_period period_length = 882 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_period_of_one_over_98_squared_l2083_208377


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_l2083_208332

theorem triangle_angle_cosine (A B C : ℝ) (a b c : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) 
  (h4 : B = 2 * C) (h5 : A + B + C = π) : Real.cos C = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_l2083_208332


namespace NUMINAMATH_CALUDE_derivative_at_two_l2083_208376

theorem derivative_at_two (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h_eq : ∀ x, f x = 2 * f (2 - x) - x^2 + 8*x - 8) : 
  deriv f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_two_l2083_208376


namespace NUMINAMATH_CALUDE_friday_ice_cream_amount_l2083_208360

/-- The amount of ice cream eaten on Friday night -/
def friday_ice_cream : ℝ := 3.5 - 0.25

/-- The total amount of ice cream eaten over two nights -/
def total_ice_cream : ℝ := 3.5

/-- The amount of ice cream eaten on Saturday night -/
def saturday_ice_cream : ℝ := 0.25

/-- Proof that the amount of ice cream eaten on Friday night is 3.25 pints -/
theorem friday_ice_cream_amount : friday_ice_cream = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_friday_ice_cream_amount_l2083_208360


namespace NUMINAMATH_CALUDE_line_passes_through_P_x_coordinate_range_l2083_208395

-- Define the line l
def line_l (θ : ℝ) (x y : ℝ) : Prop :=
  (Real.cos θ)^2 * x + Real.cos (2*θ) * y - 1 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Theorem 1: Line l always passes through point P
theorem line_passes_through_P :
  ∀ θ : ℝ, line_l θ (point_P.1) (point_P.2) :=
sorry

-- Define the range for x-coordinate of M
def x_range (x : ℝ) : Prop :=
  (2 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ 4/5

-- Theorem 2: The x-coordinate of M is in the specified range
theorem x_coordinate_range :
  ∀ θ x y xm : ℝ,
  line_l θ x y →
  circle_C x y →
  -- Additional conditions for point M would be defined here
  x_range xm :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_x_coordinate_range_l2083_208395


namespace NUMINAMATH_CALUDE_artist_painting_difference_l2083_208349

/-- Given an artist's painting schedule over three months, prove that the difference
    between the number of pictures painted in July and June is zero. -/
theorem artist_painting_difference (june july august total : ℕ) 
    (h_june : june = 2)
    (h_august : august = 9)
    (h_total : total = 13)
    (h_sum : june + july + august = total) : july - june = 0 := by
  sorry

end NUMINAMATH_CALUDE_artist_painting_difference_l2083_208349


namespace NUMINAMATH_CALUDE_angle_sum_pi_over_two_l2083_208306

theorem angle_sum_pi_over_two (a b : Real) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (eq1 : 2 * Real.sin a ^ 3 + 3 * Real.sin b ^ 2 = 1)
  (eq2 : 2 * Real.sin (3 * a) - 3 * Real.sin (3 * b) = 0) :
  a + 3 * b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_pi_over_two_l2083_208306


namespace NUMINAMATH_CALUDE_count_quadrilaterals_with_equidistant_point_l2083_208389

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A point is equidistant from all vertices of a quadrilateral -/
def has_equidistant_point (q : Quadrilateral) : Prop :=
  ∃ p : ℝ × ℝ, ∀ i : Fin 4, dist p (q.vertices i) = dist p (q.vertices 0)

/-- A kite with two consecutive right angles -/
def is_kite_with_two_right_angles (q : Quadrilateral) : Prop :=
  sorry

/-- A rectangle with sides in the ratio 3:1 -/
def is_rectangle_3_1 (q : Quadrilateral) : Prop :=
  sorry

/-- A rhombus with an angle of 120 degrees -/
def is_rhombus_120 (q : Quadrilateral) : Prop :=
  sorry

/-- A general quadrilateral with perpendicular diagonals -/
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

/-- An isosceles trapezoid where the non-parallel sides are equal in length -/
def is_isosceles_trapezoid (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem count_quadrilaterals_with_equidistant_point :
  ∃ (q1 q2 q3 : Quadrilateral),
    (is_kite_with_two_right_angles q1 ∨ 
     is_rectangle_3_1 q1 ∨ 
     is_rhombus_120 q1 ∨ 
     has_perpendicular_diagonals q1 ∨ 
     is_isosceles_trapezoid q1) ∧
    (is_kite_with_two_right_angles q2 ∨ 
     is_rectangle_3_1 q2 ∨ 
     is_rhombus_120 q2 ∨ 
     has_perpendicular_diagonals q2 ∨ 
     is_isosceles_trapezoid q2) ∧
    (is_kite_with_two_right_angles q3 ∨ 
     is_rectangle_3_1 q3 ∨ 
     is_rhombus_120 q3 ∨ 
     has_perpendicular_diagonals q3 ∨ 
     is_isosceles_trapezoid q3) ∧
    has_equidistant_point q1 ∧
    has_equidistant_point q2 ∧
    has_equidistant_point q3 ∧
    (∀ q : Quadrilateral, 
      (is_kite_with_two_right_angles q ∨ 
       is_rectangle_3_1 q ∨ 
       is_rhombus_120 q ∨ 
       has_perpendicular_diagonals q ∨ 
       is_isosceles_trapezoid q) →
      has_equidistant_point q →
      (q = q1 ∨ q = q2 ∨ q = q3)) :=
by
  sorry

end NUMINAMATH_CALUDE_count_quadrilaterals_with_equidistant_point_l2083_208389


namespace NUMINAMATH_CALUDE_peters_pants_purchase_l2083_208392

theorem peters_pants_purchase (shirt_price : ℕ) (pants_price : ℕ) (total_cost : ℕ) :
  shirt_price * 2 = 20 →
  pants_price = 6 →
  ∃ (num_pants : ℕ), shirt_price * 5 + pants_price * num_pants = 62 →
  num_pants = 2 := by
sorry

end NUMINAMATH_CALUDE_peters_pants_purchase_l2083_208392


namespace NUMINAMATH_CALUDE_cube_surface_area_difference_l2083_208308

theorem cube_surface_area_difference : 
  let large_cube_volume : ℝ := 343
  let small_cube_count : ℕ := 343
  let small_cube_volume : ℝ := 1
  let large_cube_side : ℝ := large_cube_volume ^ (1/3)
  let large_cube_surface_area : ℝ := 6 * large_cube_side ^ 2
  let small_cube_side : ℝ := small_cube_volume ^ (1/3)
  let small_cube_surface_area : ℝ := 6 * small_cube_side ^ 2
  let total_small_cubes_surface_area : ℝ := small_cube_count * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 1764 := by
sorry


end NUMINAMATH_CALUDE_cube_surface_area_difference_l2083_208308


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2083_208366

theorem min_value_squared_sum (x y a : ℝ) : 
  x + y ≥ a →
  x - y ≤ a →
  y ≤ a →
  a > 0 →
  (∀ x' y' : ℝ, x' + y' ≥ a → x' - y' ≤ a → y' ≤ a → x'^2 + y'^2 ≥ 2) →
  (∃ x'' y'' : ℝ, x'' + y'' ≥ a ∧ x'' - y'' ≤ a ∧ y'' ≤ a ∧ x''^2 + y''^2 = 2) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2083_208366


namespace NUMINAMATH_CALUDE_molecular_properties_l2083_208396

structure MolecularSystem where
  surface_distance : ℝ
  internal_distance : ℝ
  surface_attraction : Bool

structure IdealGas where
  temperature : ℝ
  collision_frequency : ℝ

structure OleicAcid where
  diameter : ℝ
  molar_volume : ℝ

def surface_tension (ms : MolecularSystem) : Prop :=
  ms.surface_distance > ms.internal_distance ∧ ms.surface_attraction

def gas_collision_frequency (ig : IdealGas) : Prop :=
  ig.collision_frequency = ig.temperature

def avogadro_estimation (oa : OleicAcid) : Prop :=
  oa.diameter > 0 ∧ oa.molar_volume > 0

theorem molecular_properties 
  (ms : MolecularSystem) 
  (ig : IdealGas) 
  (oa : OleicAcid) : 
  surface_tension ms ∧ 
  gas_collision_frequency ig ∧ 
  avogadro_estimation oa :=
sorry

end NUMINAMATH_CALUDE_molecular_properties_l2083_208396


namespace NUMINAMATH_CALUDE_square_difference_equality_l2083_208339

theorem square_difference_equality : (25 + 15 + 8)^2 - (25^2 + 15^2 + 8^2) = 1390 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2083_208339


namespace NUMINAMATH_CALUDE_simplify_expression_l2083_208305

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 + b*(2*b+1) = 9*b^3 + 6*b^2 + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2083_208305


namespace NUMINAMATH_CALUDE_vector_triangle_l2083_208353

/-- Given a triangle ABC, a point D on BC such that BD = 3DC, and E the midpoint of AC,
    prove that ED = 1/4 AB + 1/4 AC -/
theorem vector_triangle (A B C D E : ℝ × ℝ) : 
  (∃ (t : ℝ), D = B + t • (C - B) ∧ t = 3/4) →  -- D is on BC with BD = 3DC
  E = A + (1/2 : ℝ) • (C - A) →                 -- E is midpoint of AC
  E - D = (1/4 : ℝ) • (B - A) + (1/4 : ℝ) • (C - A) := by
sorry

end NUMINAMATH_CALUDE_vector_triangle_l2083_208353


namespace NUMINAMATH_CALUDE_min_n_value_l2083_208343

theorem min_n_value (m : ℝ) :
  (∀ x : ℝ, |x - m| ≤ 2 → -1 ≤ x ∧ x ≤ 3) ∧
  ¬(∀ x : ℝ, |x - m| ≤ 2 → -1 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_min_n_value_l2083_208343


namespace NUMINAMATH_CALUDE_nabla_computation_l2083_208338

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_computation : nabla (nabla 1 3) 2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_nabla_computation_l2083_208338


namespace NUMINAMATH_CALUDE_line_equation_through_bisecting_point_l2083_208386

/-- Given a parabola and a line with specific properties, prove the equation of the line -/
theorem line_equation_through_bisecting_point (x y : ℝ) :
  (∀ x y, y^2 = 16*x) → -- parabola equation
  (∃ x1 y1 x2 y2 : ℝ, 
    y1^2 = 16*x1 ∧ y2^2 = 16*x2 ∧ -- intersection points on parabola
    (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 1) → -- midpoint is (2, 1)
  (8*x - y - 15 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_line_equation_through_bisecting_point_l2083_208386


namespace NUMINAMATH_CALUDE_triangles_congruent_l2083_208356

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- Define area and perimeter functions
def area (t : Triangle) : ℝ := sorry
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangles_congruent (t1 t2 : Triangle) 
  (h_area : area t1 = area t2)
  (h_perimeter : perimeter t1 = perimeter t2)
  (h_side : t1.a = t2.a) :
  t1 = t2 := by sorry

end NUMINAMATH_CALUDE_triangles_congruent_l2083_208356


namespace NUMINAMATH_CALUDE_james_glasses_cost_l2083_208368

/-- The total cost of James' new pair of glasses -/
def total_cost (frame_cost lens_cost insurance_coverage frame_discount : ℚ) : ℚ :=
  (frame_cost - frame_discount) + (lens_cost * (1 - insurance_coverage))

/-- Theorem stating the total cost for James' new pair of glasses -/
theorem james_glasses_cost :
  let frame_cost : ℚ := 200
  let lens_cost : ℚ := 500
  let insurance_coverage : ℚ := 0.8
  let frame_discount : ℚ := 50
  total_cost frame_cost lens_cost insurance_coverage frame_discount = 250 := by
sorry

end NUMINAMATH_CALUDE_james_glasses_cost_l2083_208368


namespace NUMINAMATH_CALUDE_sphere_speed_l2083_208304

-- Define constants
def Q : Real := -20e-6
def q : Real := 50e-6
def AB : Real := 2
def AC : Real := 3
def m : Real := 0.2
def g : Real := 10
def k : Real := 9e9

-- Define the theorem
theorem sphere_speed (BC : Real) (v : Real) 
  (h1 : BC^2 = AC^2 - AB^2)  -- Pythagorean theorem
  (h2 : v^2 = (2/m) * (k*Q*q * (1/AB - 1/BC) + m*g*AB)) : -- Energy conservation
  v = 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_speed_l2083_208304


namespace NUMINAMATH_CALUDE_triangle_side_length_l2083_208309

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = Real.sqrt 7 →
  B = π / 3 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2083_208309


namespace NUMINAMATH_CALUDE_points_to_office_theorem_l2083_208355

/-- The number of points needed to be sent to the office -/
def points_to_office : ℕ := 100

/-- Points for interrupting -/
def interrupt_points : ℕ := 5

/-- Points for insulting classmates -/
def insult_points : ℕ := 10

/-- Points for throwing things -/
def throw_points : ℕ := 25

/-- Number of times Jerry interrupted -/
def jerry_interrupts : ℕ := 2

/-- Number of times Jerry insulted classmates -/
def jerry_insults : ℕ := 4

/-- Number of times Jerry can throw things before being sent to office -/
def jerry_throws_left : ℕ := 2

/-- Theorem stating the number of points needed to be sent to the office -/
theorem points_to_office_theorem :
  points_to_office = 
    jerry_interrupts * interrupt_points +
    jerry_insults * insult_points +
    jerry_throws_left * throw_points :=
by sorry

end NUMINAMATH_CALUDE_points_to_office_theorem_l2083_208355


namespace NUMINAMATH_CALUDE_c_absolute_value_l2083_208313

def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem c_absolute_value (a b c : ℤ) :
  (∀ d : ℤ, d ≠ 1 → d ∣ a → d ∣ b → d ∣ c → False) →
  g a b c (3 + I) = 0 →
  |c| = 142 := by sorry

end NUMINAMATH_CALUDE_c_absolute_value_l2083_208313


namespace NUMINAMATH_CALUDE_quadrant_function_m_range_l2083_208318

/-- A proportional function passing through the second and fourth quadrants -/
structure QuadrantFunction where
  m : ℝ
  passes_through_second_fourth : (∀ x y, y = (1 - m) * x → 
    ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)))

/-- The range of m for a QuadrantFunction -/
theorem quadrant_function_m_range (f : QuadrantFunction) : f.m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_function_m_range_l2083_208318


namespace NUMINAMATH_CALUDE_comic_book_stacks_result_l2083_208385

/-- The number of ways to stack comic books with given constraints -/
def comic_book_stacks (spiderman_count archie_count garfield_count : ℕ) : ℕ :=
  spiderman_count.factorial * archie_count.factorial * garfield_count.factorial * 2

/-- Theorem stating the number of ways to stack the comic books -/
theorem comic_book_stacks_result : comic_book_stacks 7 6 5 = 91612800 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacks_result_l2083_208385


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2083_208310

/-- The perimeter of a rectangle -/
def perimeter (x y : ℝ) : ℝ := 2 * (x + y)

/-- The area of a rectangle -/
def area (x y : ℝ) : ℝ := x * y

/-- Theorem: For a rectangle with a fixed perimeter, the area is maximized when length equals width -/
theorem max_area_rectangle (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ perimeter x y = p ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → perimeter a b = p → area a b ≤ area x y) ∧
  x = p / 4 ∧ y = p / 4 := by
  sorry

#check max_area_rectangle

end NUMINAMATH_CALUDE_max_area_rectangle_l2083_208310


namespace NUMINAMATH_CALUDE_room_breadth_is_ten_l2083_208307

/-- Proves that the breadth of a room is 10 feet given specific conditions -/
theorem room_breadth_is_ten (room_length : ℝ) (tile_size : ℝ) (blue_tile_count : ℕ) : 
  room_length = 20 ∧ 
  tile_size = 2 ∧ 
  blue_tile_count = 16 →
  ∃ (room_breadth : ℝ),
    room_breadth = 10 ∧
    (room_length - 2 * tile_size) * (room_breadth - 2 * tile_size) * (2/3) = 
      (blue_tile_count : ℝ) * tile_size^2 :=
by sorry


end NUMINAMATH_CALUDE_room_breadth_is_ten_l2083_208307


namespace NUMINAMATH_CALUDE_cats_needed_to_reach_goal_l2083_208319

theorem cats_needed_to_reach_goal (current_cats goal_cats : ℕ) : 
  current_cats = 11 → goal_cats = 43 → goal_cats - current_cats = 32 := by
sorry

end NUMINAMATH_CALUDE_cats_needed_to_reach_goal_l2083_208319


namespace NUMINAMATH_CALUDE_pen_pencil_difference_l2083_208320

theorem pen_pencil_difference (pen_count : ℕ) (pencil_count : ℕ) : 
  pencil_count = 48 →
  pen_count * 6 = pencil_count * 5 →
  pencil_count > pen_count →
  pencil_count - pen_count = 8 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_difference_l2083_208320


namespace NUMINAMATH_CALUDE_journey_equation_correct_l2083_208314

/-- Represents a journey with a stop in between -/
structure Journey where
  preBrakeSpeed : ℝ
  postBrakeSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  brakeTime : ℝ

/-- Checks if the given equation correctly represents the journey -/
def isCorrectEquation (j : Journey) (equation : ℝ → Prop) : Prop :=
  ∀ t, equation t ↔ 
    j.preBrakeSpeed * t + j.postBrakeSpeed * (j.totalTime - j.brakeTime - t) = j.totalDistance

theorem journey_equation_correct (j : Journey) 
    (h1 : j.preBrakeSpeed = 60)
    (h2 : j.postBrakeSpeed = 80)
    (h3 : j.totalDistance = 220)
    (h4 : j.totalTime = 4)
    (h5 : j.brakeTime = 2/3) :
    isCorrectEquation j (fun t ↦ 60 * t + 80 * (10/3 - t) = 220) := by
  sorry

#check journey_equation_correct

end NUMINAMATH_CALUDE_journey_equation_correct_l2083_208314


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2083_208394

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2083_208394


namespace NUMINAMATH_CALUDE_product_sum_relation_l2083_208384

theorem product_sum_relation (a b c : ℕ+) (h1 : c = 2 * a + b) 
  (h2 : a * b * c = 8 * (a + b + c)) : a * b * c = 136 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2083_208384


namespace NUMINAMATH_CALUDE_sum_of_sequences_l2083_208303

def sequence1 : List Nat := [2, 12, 22, 32, 42]
def sequence2 : List Nat := [10, 20, 30, 40, 50]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum = 260) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l2083_208303


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l2083_208364

theorem factorial_prime_factorization :
  let x : ℕ := Finset.prod (Finset.range 15) (fun i => i + 1)
  ∀ (i k m p q r : ℕ),
    (i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0) →
    x = 2^i * 3^k * 5^m * 7^p * 11^q * 13^r →
    i + k + m + p + q + r = 29 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l2083_208364


namespace NUMINAMATH_CALUDE_triangle_cos_C_eq_neg_one_fourth_l2083_208351

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to these sides respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law of sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The law of cosines for a triangle -/
axiom law_of_cosines (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_cos_C_eq_neg_one_fourth (t : Triangle) 
  (ha : t.a = 2)
  (hc : t.c = 4)
  (h_sin : 3 * Real.sin t.A = 2 * Real.sin t.B) :
  Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_C_eq_neg_one_fourth_l2083_208351


namespace NUMINAMATH_CALUDE_area_ratio_quadrilateral_triangle_l2083_208346

-- Define the types for points and shapes
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Quadrilateral : Type)
variable (Triangle : Type)

-- Define functions for area calculation
variable (area : Quadrilateral → ℝ)
variable (area_triangle : Triangle → ℝ)

-- Define a function to create a quadrilateral from four points
variable (make_quadrilateral : Point → Point → Point → Point → Quadrilateral)

-- Define a function to create a triangle from three points
variable (make_triangle : Point → Point → Point → Triangle)

-- Define a function to get the midpoint of two points
variable (midpoint : Point → Point → Point)

-- Define a function to extend two line segments to their intersection
variable (extend_to_intersection : Point → Point → Point → Point → Point)

-- Theorem statement
theorem area_ratio_quadrilateral_triangle 
  (A B C D : Point) 
  (ABCD : Quadrilateral) 
  (E : Point) 
  (H G : Point) 
  (EHG : Triangle) :
  ABCD = make_quadrilateral A B C D →
  E = extend_to_intersection A D B C →
  H = midpoint B D →
  G = midpoint A C →
  EHG = make_triangle E H G →
  (area_triangle EHG) / (area ABCD) = 1/4 := by sorry

end NUMINAMATH_CALUDE_area_ratio_quadrilateral_triangle_l2083_208346


namespace NUMINAMATH_CALUDE_fruit_vendor_problem_l2083_208363

-- Define the parameters
def total_boxes : ℕ := 60
def strawberry_price : ℕ := 60
def apple_price : ℕ := 40
def total_spent : ℕ := 3100
def profit_strawberry_A : ℕ := 15
def profit_apple_A : ℕ := 20
def profit_strawberry_B : ℕ := 12
def profit_apple_B : ℕ := 16
def profit_A : ℕ := 600

-- Define the theorem
theorem fruit_vendor_problem :
  ∃ (strawberry_boxes apple_boxes : ℕ),
    strawberry_boxes + apple_boxes = total_boxes ∧
    strawberry_boxes * strawberry_price + apple_boxes * apple_price = total_spent ∧
    strawberry_boxes = 35 ∧
    apple_boxes = 25 ∧
    (∃ (a b : ℕ),
      a + b ≤ total_boxes ∧
      a * profit_strawberry_A + b * profit_apple_A = profit_A ∧
      (strawberry_boxes - a) * profit_strawberry_B + (apple_boxes - b) * profit_apple_B = 340 ∧
      (a + b = 52 ∨ a + b = 53)) :=
sorry

end NUMINAMATH_CALUDE_fruit_vendor_problem_l2083_208363


namespace NUMINAMATH_CALUDE_park_wheels_count_l2083_208381

/-- The total number of wheels on bikes in a park -/
def total_wheels (regular_bikes children_bikes tandem_4_wheels tandem_6_wheels : ℕ) : ℕ :=
  regular_bikes * 2 + children_bikes * 4 + tandem_4_wheels * 4 + tandem_6_wheels * 6

/-- Theorem: The total number of wheels in the park is 96 -/
theorem park_wheels_count :
  total_wheels 7 11 5 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_park_wheels_count_l2083_208381


namespace NUMINAMATH_CALUDE_constant_function_theorem_l2083_208391

/-- The set of all points in the plane -/
def S : Type := ℝ × ℝ

/-- A function from the plane to real numbers -/
def PlaneFunction : Type := S → ℝ

/-- Predicate for a nondegenerate triangle -/
def NonDegenerateTriangle (A B C : S) : Prop := sorry

/-- The orthocenter of a triangle -/
def Orthocenter (A B C : S) : S := sorry

/-- The property that the function satisfies for all nondegenerate triangles -/
def SatisfiesTriangleProperty (f : PlaneFunction) : Prop :=
  ∀ A B C : S, NonDegenerateTriangle A B C →
    let H := Orthocenter A B C
    (f A ≤ f B ∧ f B ≤ f C) → f A + f C = f B + f H

/-- The main theorem: if a function satisfies the triangle property, it must be constant -/
theorem constant_function_theorem (f : PlaneFunction) 
  (h : SatisfiesTriangleProperty f) : 
  ∀ x y : S, f x = f y := sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l2083_208391


namespace NUMINAMATH_CALUDE_evaluate_expression_l2083_208325

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 2) : y * (y - 3 * x) = -14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2083_208325


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2083_208317

/-- The interval of segmentation for systematic sampling -/
def intervalOfSegmentation (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The interval of segmentation for a population of 2000 and sample size of 40 is 50 -/
theorem systematic_sampling_interval :
  intervalOfSegmentation 2000 40 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2083_208317


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2083_208348

/-- A geometric sequence is defined by its first term and common ratio. -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- The nth term of a geometric sequence. -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * seq.r ^ (n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 8 and the 9th term is 128, the 12th term is 1024. -/
theorem geometric_sequence_12th_term
  (seq : GeometricSequence)
  (h5 : seq.nthTerm 5 = 8)
  (h9 : seq.nthTerm 9 = 128) :
  seq.nthTerm 12 = 1024 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2083_208348


namespace NUMINAMATH_CALUDE_license_plate_ratio_l2083_208340

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate. -/
def old_letters : ℕ := 2

/-- The number of digits in an old license plate. -/
def old_digits : ℕ := 5

/-- The number of letters in a new license plate. -/
def new_letters : ℕ := 4

/-- The number of digits in a new license plate. -/
def new_digits : ℕ := 2

/-- The ratio of new possible license plates to old possible license plates. -/
theorem license_plate_ratio :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) =
  (num_letters ^ 2 : ℚ) / (num_digits ^ 3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_license_plate_ratio_l2083_208340


namespace NUMINAMATH_CALUDE_no_real_b_for_single_solution_l2083_208369

-- Define the quadratic function g(x) with parameter b
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 3*b*x + 4*b

-- Theorem stating that no real b exists such that g(x) has its vertex at y = 5
theorem no_real_b_for_single_solution :
  ¬ ∃ b : ℝ, ∃ x : ℝ, g b x = 5 ∧ ∀ y : ℝ, g b y ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_no_real_b_for_single_solution_l2083_208369


namespace NUMINAMATH_CALUDE_sin_negative_1920_degrees_l2083_208397

theorem sin_negative_1920_degrees : 
  Real.sin ((-1920 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_1920_degrees_l2083_208397


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2083_208374

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = 3 + 2 * Real.sqrt 2 ∧ x2 = 3 - 2 * Real.sqrt 2 ∧
    x1^2 - 6*x1 + 1 = 0 ∧ x2^2 - 6*x2 + 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -1 ∧ x2 = 1/2 ∧
    2*(x1+1)^2 = 3*(x1+1) ∧ 2*(x2+1)^2 = 3*(x2+1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2083_208374


namespace NUMINAMATH_CALUDE_largest_gcd_sum_1008_l2083_208362

theorem largest_gcd_sum_1008 :
  ∃ (a b : ℕ+), a + b = 1008 ∧ 
  ∀ (c d : ℕ+), c + d = 1008 → Nat.gcd a b ≥ Nat.gcd c d ∧
  Nat.gcd a b = 504 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_sum_1008_l2083_208362


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2083_208335

theorem complex_equation_solution (i : ℂ) (h : i * i = -1) :
  ∃ z : ℂ, (2 + i) * z = 5 ∧ z = 2 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2083_208335


namespace NUMINAMATH_CALUDE_triangle_inconsistency_l2083_208375

theorem triangle_inconsistency : ¬ ∃ (a b c : ℝ),
  (a = 40 ∧ b = 50 ∧ c = 2 * (a + b) ∧ a + b + c = 160) ∧
  (a + b > c ∧ b + c > a ∧ a + c > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inconsistency_l2083_208375


namespace NUMINAMATH_CALUDE_min_values_and_corresponding_points_l2083_208380

theorem min_values_and_corresponding_points (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 2) : 
  (∃ (min_ab : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → x*y ≥ min_ab ∧ a*b = min_ab) ∧
  (∃ (min_sum : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → x + 2*y ≥ min_sum ∧ a + 2*b = min_sum) ∧
  a = (1 + Real.sqrt 2) / 2 ∧ b = (2 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_values_and_corresponding_points_l2083_208380


namespace NUMINAMATH_CALUDE_midpoint_locus_of_square_l2083_208387

/-- The locus of the midpoint of a square with side length 2a, where two consecutive vertices
    are always on the x- and y-axes respectively in the first quadrant, is a circle with
    radius a centered at the origin. -/
theorem midpoint_locus_of_square (a : ℝ) (h : a > 0) :
  ∃ (C : ℝ × ℝ), (∀ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ x^2 + y^2 = (2*a)^2 →
    C = (x/2, y/2) ∧ C.1^2 + C.2^2 = a^2) :=
sorry

end NUMINAMATH_CALUDE_midpoint_locus_of_square_l2083_208387


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2083_208316

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches -/
theorem cylinder_surface_area : 
  ∀ (h r : ℝ), 
  h = 8 → 
  r = 3 → 
  2 * π * r * h + 2 * π * r^2 = 66 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2083_208316


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2083_208331

def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry 
  (a b c : ℝ) 
  (h_symmetry : ∀ x, p a b c x = p a b c (30 - x))
  (h_p25 : p a b c 25 = 9)
  (h_p0 : p a b c 0 = 1) :
  p a b c 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2083_208331


namespace NUMINAMATH_CALUDE_sum_equals_210_l2083_208357

theorem sum_equals_210 : 145 + 35 + 25 + 5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_210_l2083_208357


namespace NUMINAMATH_CALUDE_complex_product_real_l2083_208372

theorem complex_product_real (t : ℝ) : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 3 + 4 * i
  let z₂ : ℂ := t + i
  (z₁ * z₂).im = 0 → t = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l2083_208372
