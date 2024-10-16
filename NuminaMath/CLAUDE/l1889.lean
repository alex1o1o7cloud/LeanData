import Mathlib

namespace NUMINAMATH_CALUDE_toys_after_game_purchase_l1889_188930

theorem toys_after_game_purchase (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : 
  initial_amount = 57 → game_cost = 27 → toy_cost = 6 → 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_after_game_purchase_l1889_188930


namespace NUMINAMATH_CALUDE_graduating_class_size_l1889_188910

theorem graduating_class_size (boys : ℕ) (girls : ℕ) (h1 : boys = 127) (h2 : girls = boys + 212) :
  boys + girls = 466 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_size_l1889_188910


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_min_people_for_second_caterer_cheaper_l1889_188971

/-- Represents the pricing model of a caterer -/
structure CatererPrice where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total price for a given number of people -/
def totalPrice (price : CatererPrice) (people : ℕ) : ℕ :=
  price.basicFee + price.perPersonFee * people

/-- The first caterer's pricing model -/
def caterer1 : CatererPrice :=
  { basicFee := 150, perPersonFee := 18 }

/-- The second caterer's pricing model -/
def caterer2 : CatererPrice :=
  { basicFee := 250, perPersonFee := 15 }

/-- Theorem stating that 34 is the minimum number of people for which
    the second caterer becomes cheaper than the first caterer -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalPrice caterer1 n ≤ totalPrice caterer2 n) ∧
  (totalPrice caterer1 34 > totalPrice caterer2 34) :=
by sorry

/-- Theorem stating that 34 is indeed the minimum such number -/
theorem min_people_for_second_caterer_cheaper :
  ∀ m : ℕ, m < 34 → ¬(∀ n : ℕ, n < m → totalPrice caterer1 n ≤ totalPrice caterer2 n) ∧
                    (totalPrice caterer1 m > totalPrice caterer2 m) :=
by sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_min_people_for_second_caterer_cheaper_l1889_188971


namespace NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_one_l1889_188908

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ : ℝ) (m₂ n₂ : ℝ) : Prop := m₁ * n₂ = m₂ * n₁

/-- The statement that a = 1 is necessary and sufficient for the lines to be parallel -/
theorem parallel_lines_iff_a_eq_one :
  ∀ a : ℝ, are_parallel a 1 3 (a + 2) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_one_l1889_188908


namespace NUMINAMATH_CALUDE_evaluate_expression_l1889_188966

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1889_188966


namespace NUMINAMATH_CALUDE_fertilizer_amounts_l1889_188958

def petunia_flats : ℕ := 4
def petunias_per_flat : ℕ := 8
def rose_flats : ℕ := 3
def roses_per_flat : ℕ := 6
def sunflower_flats : ℕ := 5
def sunflowers_per_flat : ℕ := 10
def orchid_flats : ℕ := 2
def orchids_per_flat : ℕ := 4
def venus_flytraps : ℕ := 2

def petunia_fertilizer_A : ℕ := 8
def rose_fertilizer_B : ℕ := 3
def sunflower_fertilizer_B : ℕ := 6
def orchid_fertilizer_A : ℕ := 4
def orchid_fertilizer_B : ℕ := 4
def venus_flytrap_fertilizer_C : ℕ := 2

theorem fertilizer_amounts :
  let total_fertilizer_A := petunia_flats * petunias_per_flat * petunia_fertilizer_A +
                            orchid_flats * orchids_per_flat * orchid_fertilizer_A
  let total_fertilizer_B := rose_flats * roses_per_flat * rose_fertilizer_B +
                            sunflower_flats * sunflowers_per_flat * sunflower_fertilizer_B +
                            orchid_flats * orchids_per_flat * orchid_fertilizer_B
  let total_fertilizer_C := venus_flytraps * venus_flytrap_fertilizer_C
  total_fertilizer_A = 288 ∧
  total_fertilizer_B = 386 ∧
  total_fertilizer_C = 4 :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_amounts_l1889_188958


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l1889_188972

theorem snow_leopard_arrangement (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (k.factorial) * ((n - k).factorial) = 30240 :=
sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l1889_188972


namespace NUMINAMATH_CALUDE_only_one_four_cell_piece_l1889_188943

/-- Represents a piece on the board -/
structure Piece where
  size : Nat
  deriving Repr

/-- Represents the board configuration -/
structure Board where
  size : Nat
  pieces : List Piece
  deriving Repr

/-- Checks if a board configuration is valid -/
def isValidBoard (b : Board) : Prop :=
  b.size = 7 ∧ 
  b.pieces.all (λ p => p.size = 4) ∧
  b.pieces.length ≤ 3 ∧
  (b.pieces.map (λ p => p.size)).sum = b.size * b.size

/-- Theorem: Only one four-cell piece can be used in a valid 7x7 board configuration -/
theorem only_one_four_cell_piece (b : Board) :
  isValidBoard b → (b.pieces.filter (λ p => p.size = 4)).length = 1 := by
  sorry

#check only_one_four_cell_piece

end NUMINAMATH_CALUDE_only_one_four_cell_piece_l1889_188943


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1889_188941

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

/-- Definition of the line passing through (0, 2) with slope 1 -/
def line (x y : ℝ) : Prop :=
  y = x + 2

/-- Intersection points of the ellipse and the line -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

theorem ellipse_and_line_intersection :
  ∃ (A B : ℝ × ℝ),
    intersection_points A B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (6 * Real.sqrt 3 / 5)^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1889_188941


namespace NUMINAMATH_CALUDE_simplify_expression_l1889_188997

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = 5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1889_188997


namespace NUMINAMATH_CALUDE_degree_not_determined_by_characteristic_l1889_188993

/-- A type representing a characteristic of a polynomial -/
def PolynomialCharacteristic := Type

/-- A function that computes a characteristic of a polynomial -/
noncomputable def compute_characteristic (P : Polynomial ℝ) : PolynomialCharacteristic :=
  sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from its characteristic -/
theorem degree_not_determined_by_characteristic :
  ∃ (P1 P2 : Polynomial ℝ), 
    P1.degree ≠ P2.degree ∧ 
    compute_characteristic P1 = compute_characteristic P2 := by
  sorry

end NUMINAMATH_CALUDE_degree_not_determined_by_characteristic_l1889_188993


namespace NUMINAMATH_CALUDE_floyd_books_theorem_l1889_188990

def total_books : ℕ := 89
def mcgregor_books : ℕ := 34
def unread_books : ℕ := 23

theorem floyd_books_theorem : 
  total_books - mcgregor_books - unread_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_floyd_books_theorem_l1889_188990


namespace NUMINAMATH_CALUDE_coefficient_sum_l1889_188989

variables (a b c d e : ℝ)

/-- The polynomial equation -/
def polynomial (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- Theorem stating the relationship between the coefficients of the polynomial -/
theorem coefficient_sum (h1 : a ≠ 0)
  (h2 : polynomial 5 = 0)
  (h3 : polynomial (-3) = 0)
  (h4 : polynomial 1 = 0) :
  (b + c + d) / a = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1889_188989


namespace NUMINAMATH_CALUDE_sin_X_in_right_triangle_l1889_188961

-- Define the right triangle XYZ
def RightTriangle (X Y Z : ℝ) : Prop :=
  0 < X ∧ 0 < Y ∧ 0 < Z ∧ X^2 + Y^2 = Z^2

-- State the theorem
theorem sin_X_in_right_triangle :
  ∀ X Y Z : ℝ,
  RightTriangle X Y Z →
  X = 8 →
  Z = 17 →
  Real.sin (Real.arcsin (X / Z)) = 8 / 17 :=
by sorry

end NUMINAMATH_CALUDE_sin_X_in_right_triangle_l1889_188961


namespace NUMINAMATH_CALUDE_square_perimeter_l1889_188991

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 900) (h2 : side * side = area) :
  4 * side = 120 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1889_188991


namespace NUMINAMATH_CALUDE_relationship_abc_l1889_188939

-- Define a, b, and c
def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

-- Theorem stating the relationship between a, b, and c
theorem relationship_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1889_188939


namespace NUMINAMATH_CALUDE_equation_solution_l1889_188979

theorem equation_solution : ∃ x : ℚ, (x - 7) / 3 - (1 + x) / 2 = 1 ∧ x = -23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1889_188979


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_l1889_188974

-- Define the total number of employees
variable (E : ℝ)

-- Define the percentage of fair-haired employees who are women
def fair_haired_women_ratio : ℝ := 0.4

-- Define the percentage of employees who have fair hair
def fair_haired_ratio : ℝ := 0.8

-- Define the percentage of employees who are women with fair hair
def women_fair_hair_ratio : ℝ := fair_haired_women_ratio * fair_haired_ratio

-- Theorem statement
theorem women_fair_hair_percentage :
  women_fair_hair_ratio = 0.32 :=
sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_l1889_188974


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_m_bound_l1889_188947

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = 4 * x + m

/-- Two points are symmetric with respect to a line -/
def symmetric_points (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 ∧ y₂ - y₁ = -4 * (x₂ - x₁)

/-- The theorem statement -/
theorem ellipse_symmetric_points_m_bound :
  ∀ m : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∃ x₀ y₀ : ℝ,
      line x₀ y₀ m ∧
      symmetric_points x₁ y₁ x₂ y₂ x₀ y₀)) →
  -2 * Real.sqrt 3 / 13 < m ∧ m < 2 * Real.sqrt 3 / 13 :=
sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_m_bound_l1889_188947


namespace NUMINAMATH_CALUDE_problem_solution_l1889_188964

theorem problem_solution (x y : ℝ) (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 6) :
  3 * x^2 + 5 * x * y + 3 * y^2 = 99 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1889_188964


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1889_188938

theorem inequality_solution_set (a : ℝ) : 
  ((3 - a) / 2 - 2 = 2) → 
  {x : ℝ | (2 - a / 5) < (1 / 3) * x} = {x : ℝ | x > 9} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1889_188938


namespace NUMINAMATH_CALUDE_function_inequality_l1889_188965

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x < (deriv^[2] f) x) : 
  f 1 > ℯ * f 0 ∧ f 2019 > ℯ^2019 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1889_188965


namespace NUMINAMATH_CALUDE_expand_expression_l1889_188902

theorem expand_expression (y : ℝ) : 5 * (y + 3) * (y - 2) * (y + 1) = 5 * y^3 + 10 * y^2 - 25 * y - 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1889_188902


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1889_188962

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 5)) ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1889_188962


namespace NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l1889_188948

/-- Given a cylinder and a sphere with equal radii, if the ratio of their surface areas is m:n,
    then the ratio of their volumes is (6m - 3n) : 4n. -/
theorem cylinder_sphere_volume_ratio (R : ℝ) (H : ℝ) (m n : ℝ) (h_positive : R > 0 ∧ H > 0 ∧ m > 0 ∧ n > 0) :
  (2 * π * R^2 + 2 * π * R * H) / (4 * π * R^2) = m / n →
  (π * R^2 * H) / ((4/3) * π * R^3) = (6 * m - 3 * n) / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l1889_188948


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l1889_188923

theorem cistern_emptying_time (fill_time : ℝ) (combined_time : ℝ) : 
  fill_time = 7 → combined_time = 31.5 → 
  (fill_time⁻¹ - (fill_time⁻¹ - combined_time⁻¹)⁻¹) = 9⁻¹ :=
by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l1889_188923


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_six_l1889_188907

theorem no_solution_iff_k_eq_six (k : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 7 → (x - 1) / (x - 2) ≠ (x - k) / (x - 7)) ↔ k = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_six_l1889_188907


namespace NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l1889_188932

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem f_of_2_eq_neg_2 : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l1889_188932


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1889_188945

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {y | y > 2}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M) ∩ N = {x : ℝ | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1889_188945


namespace NUMINAMATH_CALUDE_expected_replanted_seeds_l1889_188927

/-- The expected number of replanted seeds when sowing 1000 seeds with a 0.9 germination probability -/
theorem expected_replanted_seeds :
  let germination_prob : ℝ := 0.9
  let total_seeds : ℕ := 1000
  let replant_per_fail : ℕ := 2
  let expected_non_germinating : ℝ := total_seeds * (1 - germination_prob)
  expected_non_germinating * replant_per_fail = 200 := by sorry

end NUMINAMATH_CALUDE_expected_replanted_seeds_l1889_188927


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l1889_188926

theorem sum_of_reciprocals_of_quadratic_roots :
  ∀ p q : ℝ, p^2 - 10*p + 3 = 0 → q^2 - 10*q + 3 = 0 → p ≠ q →
  1/p + 1/q = 10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l1889_188926


namespace NUMINAMATH_CALUDE_question_1_question_2_question_3_l1889_188992

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 8 * x^2 + 16 * x - k
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x

-- Define the interval [-3, 3]
def I : Set ℝ := Set.Icc (-3) 3

-- Statement for question 1
theorem question_1 (k : ℝ) : 
  (∀ x ∈ I, f k x ≤ g x) ↔ k ≥ 45 := by sorry

-- Statement for question 2
theorem question_2 (k : ℝ) : 
  (∃ x ∈ I, f k x ≤ g x) ↔ k ≥ -7 := by sorry

-- Statement for question 3
theorem question_3 (k : ℝ) : 
  (∀ x₁ ∈ I, ∀ x₂ ∈ I, f k x₁ ≤ g x₂) ↔ k ≥ 141 := by sorry

end NUMINAMATH_CALUDE_question_1_question_2_question_3_l1889_188992


namespace NUMINAMATH_CALUDE_ponderosa_price_calculation_l1889_188919

/-- The price of each ponderosa pine tree -/
def ponderosa_price : ℕ := 225

/-- The total number of trees -/
def total_trees : ℕ := 850

/-- The number of trees bought of one kind -/
def trees_of_one_kind : ℕ := 350

/-- The price of each Douglas fir tree -/
def douglas_price : ℕ := 300

/-- The total amount paid for all trees -/
def total_paid : ℕ := 217500

theorem ponderosa_price_calculation :
  ponderosa_price = 225 ∧
  total_trees = 850 ∧
  trees_of_one_kind = 350 ∧
  douglas_price = 300 ∧
  total_paid = 217500 →
  ∃ (douglas_count ponderosa_count : ℕ),
    douglas_count + ponderosa_count = total_trees ∧
    (douglas_count = trees_of_one_kind ∨ ponderosa_count = trees_of_one_kind) ∧
    douglas_count * douglas_price + ponderosa_count * ponderosa_price = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ponderosa_price_calculation_l1889_188919


namespace NUMINAMATH_CALUDE_joseph_driving_time_l1889_188980

theorem joseph_driving_time :
  let joseph_speed : ℝ := 50
  let kyle_speed : ℝ := 62
  let kyle_time : ℝ := 2
  let distance_difference : ℝ := 1
  let joseph_distance : ℝ := kyle_speed * kyle_time + distance_difference
  joseph_distance / joseph_speed = 2.5 := by sorry

end NUMINAMATH_CALUDE_joseph_driving_time_l1889_188980


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l1889_188913

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  wednesday_hours : ℕ
  friday_hours : ℕ
  tuesday_hours : ℕ
  thursday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := 
    3 * schedule.monday_hours + 
    2 * schedule.tuesday_hours
  schedule.weekly_earnings / total_hours

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule := {
  monday_hours := 8
  wednesday_hours := 8
  friday_hours := 8
  tuesday_hours := 6
  thursday_hours := 6
  weekly_earnings := 504
}

/-- Theorem: Sheila's hourly wage is $14 --/
theorem sheila_hourly_wage : 
  hourly_wage sheila_schedule = 14 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l1889_188913


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_210_l1889_188901

theorem last_three_digits_of_7_to_210 : 7^210 % 1000 = 599 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_210_l1889_188901


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1889_188917

/-- Given a triangle with vertices at (0, 0), (x, 3x), and (x, 0), where x > 0,
    if the area of the triangle is 81 square units, then x = 3√6. -/
theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 81 → x = 3 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1889_188917


namespace NUMINAMATH_CALUDE_complex_trajectory_l1889_188900

theorem complex_trajectory (x y : ℝ) (h1 : x ≥ (1/2 : ℝ)) (z : ℂ) 
  (h2 : z = Complex.mk x y) (h3 : Complex.abs (z - 1) = x) : 
  y^2 = 2*x - 1 := by
sorry

end NUMINAMATH_CALUDE_complex_trajectory_l1889_188900


namespace NUMINAMATH_CALUDE_square_product_equality_l1889_188914

theorem square_product_equality : (15 : ℕ)^2 * 9^2 * 356 = 6489300 := by
  sorry

end NUMINAMATH_CALUDE_square_product_equality_l1889_188914


namespace NUMINAMATH_CALUDE_price_difference_per_can_l1889_188933

/-- Proves that the difference in price per can between the grocery store and bulk warehouse is 25 cents -/
theorem price_difference_per_can (bulk_price bulk_quantity grocery_price grocery_quantity : ℚ) : 
  bulk_price = 12 →
  bulk_quantity = 48 →
  grocery_price = 6 →
  grocery_quantity = 12 →
  (grocery_price / grocery_quantity - bulk_price / bulk_quantity) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_per_can_l1889_188933


namespace NUMINAMATH_CALUDE_prime_dates_february_2024_l1889_188911

/-- A natural number is prime if it's greater than 1 and has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The number of days in February during a leap year. -/
def februaryDaysInLeapYear : ℕ := 29

/-- The month number for February. -/
def februaryMonth : ℕ := 2

/-- A prime date occurs when both the month and day are prime numbers. -/
def isPrimeDate (month day : ℕ) : Prop := isPrime month ∧ isPrime day

/-- The number of prime dates in February of a leap year. -/
def primeDatesInFebruaryLeapYear : ℕ := 10

/-- Theorem stating that the number of prime dates in February 2024 is 10. -/
theorem prime_dates_february_2024 :
  isPrime februaryMonth →
  (∀ d : ℕ, d ≤ februaryDaysInLeapYear → isPrimeDate februaryMonth d ↔ isPrime d) →
  (∃ dates : Finset ℕ, dates.card = primeDatesInFebruaryLeapYear ∧
    ∀ d ∈ dates, d ≤ februaryDaysInLeapYear ∧ isPrime d) :=
by sorry

end NUMINAMATH_CALUDE_prime_dates_february_2024_l1889_188911


namespace NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l1889_188955

theorem unique_real_sqrt_negative_square : ∃! x : ℝ, ∃ y : ℝ, y ^ 2 = -(2 * x - 3) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l1889_188955


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_correct_tangent_points_are_correct_l1889_188925

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

-- Theorem to verify the y-coordinates
theorem y_coordinates_correct :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

-- Main theorem combining both conditions
theorem tangent_points_are_correct :
  ∀ x y : ℝ, (f' x = 4 ∧ f x = y) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_correct_tangent_points_are_correct_l1889_188925


namespace NUMINAMATH_CALUDE_expression_simplification_l1889_188934

theorem expression_simplification (x y m n : ℝ) : 
  (2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2) ∧
  (9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1889_188934


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l1889_188951

theorem fraction_sum_equals_one (a : ℝ) (h : a ≠ -2) :
  (a + 1) / (a + 2) + 1 / (a + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l1889_188951


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1889_188912

theorem x_squared_plus_y_squared (x y : ℝ) :
  |x - 1/2| + (2*y + 1)^2 = 0 → x^2 + y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1889_188912


namespace NUMINAMATH_CALUDE_river_flow_volume_l1889_188996

/-- Given a river with specified dimensions and flow rate, calculate the volume of water flowing per minute -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 2) 
  (h_width : width = 45) 
  (h_flow_rate : flow_rate_kmph = 3) : 
  depth * width * (flow_rate_kmph * 1000 / 60) = 9000 := by
  sorry

end NUMINAMATH_CALUDE_river_flow_volume_l1889_188996


namespace NUMINAMATH_CALUDE_a_values_in_A_l1889_188936

def A : Set ℝ := {2, 4, 6}

theorem a_values_in_A : {a : ℝ | a ∈ A ∧ (6 - a) ∈ A} = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_a_values_in_A_l1889_188936


namespace NUMINAMATH_CALUDE_equilateral_cone_central_angle_l1889_188922

/-- An equilateral cone is a cone whose cross-section is an equilateral triangle -/
structure EquilateralCone where
  radius : ℝ
  slant_height : ℝ
  slant_height_eq : slant_height = 2 * radius

/-- The central angle of the sector of an equilateral cone is π radians -/
theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  (2 * π * cone.radius) / cone.slant_height = π :=
sorry

end NUMINAMATH_CALUDE_equilateral_cone_central_angle_l1889_188922


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l1889_188928

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg actual_num : ℝ) :
  n = 10 ∧ 
  initial_avg = 14 ∧ 
  correct_avg = 15 ∧ 
  actual_num = 36 →
  ∃ wrong_num : ℝ, 
    n * correct_avg - n * initial_avg = actual_num - wrong_num ∧ 
    wrong_num = 26 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l1889_188928


namespace NUMINAMATH_CALUDE_completing_square_transform_l1889_188946

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 2*x = 9) ↔ ((x - 1)^2 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_transform_l1889_188946


namespace NUMINAMATH_CALUDE_magazine_purchase_ways_l1889_188954

def magazine_count : ℕ := 11
def expensive_count : ℕ := 8
def cheap_count : ℕ := 3
def expensive_price : ℕ := 2
def cheap_price : ℕ := 1
def total_money : ℕ := 10

def ways_to_buy : ℕ := 266

theorem magazine_purchase_ways :
  magazine_count = expensive_count + cheap_count ∧
  expensive_count = 8 ∧
  cheap_count = 3 ∧
  expensive_price = 2 ∧
  cheap_price = 1 ∧
  total_money = 10 →
  ways_to_buy = (Nat.choose cheap_count 2 * Nat.choose expensive_count 4) +
                 Nat.choose expensive_count 5 :=
by sorry

end NUMINAMATH_CALUDE_magazine_purchase_ways_l1889_188954


namespace NUMINAMATH_CALUDE_astrophysics_degrees_calculation_l1889_188915

def microphotonics_percent : ℝ := 9
def home_electronics_percent : ℝ := 14
def food_additives_percent : ℝ := 10
def genetically_modified_microorganisms_percent : ℝ := 29
def industrial_lubricants_percent : ℝ := 8

def total_circle_degrees : ℝ := 360

def other_sectors_percent : ℝ :=
  microphotonics_percent + home_electronics_percent + food_additives_percent +
  genetically_modified_microorganisms_percent + industrial_lubricants_percent

def astrophysics_percent : ℝ := 100 - other_sectors_percent

theorem astrophysics_degrees_calculation :
  (astrophysics_percent / 100) * total_circle_degrees = 108 := by
  sorry

end NUMINAMATH_CALUDE_astrophysics_degrees_calculation_l1889_188915


namespace NUMINAMATH_CALUDE_postcard_width_is_six_l1889_188950

/-- Represents a rectangular postcard -/
structure Postcard where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangular postcard -/
def perimeter (p : Postcard) : ℝ := 2 * (p.width + p.height)

theorem postcard_width_is_six :
  ∀ p : Postcard,
  p.height = 4 →
  perimeter p = 20 →
  p.width = 6 := by
sorry

end NUMINAMATH_CALUDE_postcard_width_is_six_l1889_188950


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1889_188903

theorem polynomial_factorization (x : ℝ) : x^4 + 16 = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1889_188903


namespace NUMINAMATH_CALUDE_digit_2009_is_zero_l1889_188986

/-- The function that returns the nth digit in the sequence formed by 
    writing successive natural numbers without spaces -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the 2009th digit in the sequence is 0 -/
theorem digit_2009_is_zero : nthDigit 2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_2009_is_zero_l1889_188986


namespace NUMINAMATH_CALUDE_cube_sum_equals_sum_l1889_188957

theorem cube_sum_equals_sum (a b : ℝ) : 
  (a / (1 + b) + b / (1 + a) = 1) → a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_sum_l1889_188957


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_condition_l1889_188940

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_condition :
  ∃ (n : ℕ) (a : ℕ),
    n = 2329089562800 ∧
    a ≥ 1 ∧ a < 30 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → (k = a ∨ k = a + 1 ∨ is_divisible n k)) ∧
    consecutive_pair a (a + 1) ∧
    ¬(is_divisible n a) ∧
    ¬(is_divisible n (a + 1)) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ b : ℕ, b ≥ 1 ∧ b < 30 ∧
        (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → (k = b ∨ k = b + 1 ∨ is_divisible m k)) ∧
        consecutive_pair b (b + 1) ∧
        ¬(is_divisible m b) ∧
        ¬(is_divisible m (b + 1)))) :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_condition_l1889_188940


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1889_188967

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1889_188967


namespace NUMINAMATH_CALUDE_quadratic_shift_roots_l1889_188975

/-- Given a quadratic function f(x) = (x-m)^2 + n that intersects the x-axis at (-1,0) and (3,0),
    prove that the solutions to (x-m+2)^2 + n = 0 are -3 and 1. -/
theorem quadratic_shift_roots (m n : ℝ) : 
  (∀ x, (x - m)^2 + n = 0 ↔ x = -1 ∨ x = 3) →
  (∀ x, (x - m + 2)^2 + n = 0 ↔ x = -3 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_shift_roots_l1889_188975


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1889_188985

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 8 / x₀ + 1 / y₀ = 1 ∧ x₀ + 2 * y₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1889_188985


namespace NUMINAMATH_CALUDE_hyperbola_distance_theorem_l1889_188918

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- A point is on a hyperbola if the absolute difference of its distances to the foci is constant -/
def IsOnHyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), |distance p h.F₁ - distance p h.F₂| = k

theorem hyperbola_distance_theorem (h : Hyperbola) (p : ℝ × ℝ) :
  IsOnHyperbola h p → distance p h.F₁ = 12 →
  distance p h.F₂ = 22 ∨ distance p h.F₂ = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_distance_theorem_l1889_188918


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l1889_188956

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the removal of a smaller prism from a larger solid -/
structure PrismRemoval where
  original : RectangularSolid
  removed : RectangularSolid
  flushFaces : ℕ

/-- Theorem stating that the surface area remains unchanged after removal -/
theorem surface_area_unchanged (removal : PrismRemoval) :
  removal.original = RectangularSolid.mk 4 3 2 →
  removal.removed = RectangularSolid.mk 1 1 2 →
  removal.flushFaces = 2 →
  surfaceArea removal.original = surfaceArea removal.original - surfaceArea removal.removed + 2 * removal.removed.length * removal.removed.width :=
by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l1889_188956


namespace NUMINAMATH_CALUDE_stone_division_impossibility_l1889_188981

theorem stone_division_impossibility :
  ¬ ∃ (n : ℕ), n > 0 ∧ 3 * n = 1001 - (n - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_stone_division_impossibility_l1889_188981


namespace NUMINAMATH_CALUDE_twelve_ways_to_choose_l1889_188937

/-- The number of ways to choose one female student from a group of 4
    and one male student from a group of 3 -/
def waysToChoose (female_count male_count : ℕ) : ℕ :=
  female_count * male_count

/-- Theorem stating that there are 12 ways to choose one female student
    from a group of 4 and one male student from a group of 3 -/
theorem twelve_ways_to_choose :
  waysToChoose 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_ways_to_choose_l1889_188937


namespace NUMINAMATH_CALUDE_wheel_distance_l1889_188988

/-- Proves that a wheel rotating 20 times per minute and moving 35 cm per rotation will travel 420 meters in one hour -/
theorem wheel_distance (rotations_per_minute : ℕ) (distance_per_rotation_cm : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  (rotations_per_minute * 60 * distance_per_rotation_cm : ℚ) / 100 = 420 := by
  sorry

#check wheel_distance

end NUMINAMATH_CALUDE_wheel_distance_l1889_188988


namespace NUMINAMATH_CALUDE_marie_bike_distance_l1889_188982

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that Marie biked 31 miles -/
theorem marie_bike_distance :
  let speed := 12.0
  let time := 2.583333333
  distance speed time = 31 := by
sorry

end NUMINAMATH_CALUDE_marie_bike_distance_l1889_188982


namespace NUMINAMATH_CALUDE_optimal_group_division_l1889_188973

theorem optimal_group_division (total_members : ℕ) (large_group_size : ℕ) (small_group_size : ℕ) 
  (h1 : total_members = 90)
  (h2 : large_group_size = 7)
  (h3 : small_group_size = 3) :
  ∃ (large_groups small_groups : ℕ),
    large_groups * large_group_size + small_groups * small_group_size = total_members ∧
    large_groups = 12 ∧
    ∀ (lg sg : ℕ), lg * large_group_size + sg * small_group_size = total_members → lg ≤ large_groups :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_group_division_l1889_188973


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1889_188920

theorem decimal_point_problem :
  ∃! (x : ℝ), x > 0 ∧ 100000 * x = 5 * (1 / x) ∧ x = Real.sqrt 2 / 200 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1889_188920


namespace NUMINAMATH_CALUDE_rental_hours_proof_l1889_188931

/-- Represents a bike rental service with a base cost and hourly rate. -/
structure BikeRental where
  baseCost : ℕ
  hourlyRate : ℕ

/-- Calculates the total cost for a given number of hours. -/
def totalCost (rental : BikeRental) (hours : ℕ) : ℕ :=
  rental.baseCost + rental.hourlyRate * hours

/-- Proves that for the given bike rental conditions and total cost, the number of hours rented is 9. -/
theorem rental_hours_proof (rental : BikeRental) 
    (h1 : rental.baseCost = 17)
    (h2 : rental.hourlyRate = 7)
    (h3 : totalCost rental 9 = 80) : 
  ∃ (hours : ℕ), totalCost rental hours = 80 ∧ hours = 9 := by
  sorry

#check rental_hours_proof

end NUMINAMATH_CALUDE_rental_hours_proof_l1889_188931


namespace NUMINAMATH_CALUDE_ryan_recruitment_count_l1889_188924

def total_funding_required : ℕ := 1000
def ryan_initial_funds : ℕ := 200
def average_funding_per_person : ℕ := 10

theorem ryan_recruitment_count :
  (total_funding_required - ryan_initial_funds) / average_funding_per_person = 80 := by
  sorry

end NUMINAMATH_CALUDE_ryan_recruitment_count_l1889_188924


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_i_l1889_188959

theorem imaginary_part_of_reciprocal_i : 
  Complex.im (1 / Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_i_l1889_188959


namespace NUMINAMATH_CALUDE_person_c_start_time_l1889_188909

/-- Represents a point on the line AB -/
inductive Point : Type
| A : Point
| C : Point
| D : Point
| B : Point

/-- Represents a person walking on the line AB -/
structure Person where
  name : String
  startTime : Nat
  startPoint : Point
  endPoint : Point
  speed : Nat

/-- Represents the problem setup -/
structure ProblemSetup where
  personA : Person
  personB : Person
  personC : Person
  meetingTimeAB : Nat
  meetingTimeAC : Nat

/-- The theorem to prove -/
theorem person_c_start_time (setup : ProblemSetup) : setup.personC.startTime = 16 :=
  by sorry

end NUMINAMATH_CALUDE_person_c_start_time_l1889_188909


namespace NUMINAMATH_CALUDE_florist_fertilizer_l1889_188960

def fertilizer_problem (daily_amount : ℕ) (regular_days : ℕ) (extra_amount : ℕ) : Prop :=
  let regular_total := daily_amount * regular_days
  let final_day_amount := daily_amount + extra_amount
  let total_amount := regular_total + final_day_amount
  total_amount = 45

theorem florist_fertilizer :
  fertilizer_problem 3 12 6 := by
  sorry

end NUMINAMATH_CALUDE_florist_fertilizer_l1889_188960


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1889_188935

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 112 →
    pond_side = 8 →
    (pond_side^2) / (field_length * field_width) = 1 / 98 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1889_188935


namespace NUMINAMATH_CALUDE_password_matches_stored_sequence_l1889_188983

/-- Represents a 32-letter alphabet where each letter is encoded as a pair of digits. -/
def Alphabet : Type := Fin 32

/-- Converts a letter to its ordinal number representation. -/
def toOrdinal (a : Alphabet) : Fin 100 := sorry

/-- Represents the remainder when dividing by 10. -/
def r10 (x : ℕ) : Fin 10 := sorry

/-- Generates the x_i sequence based on the given recurrence relation. -/
def genX (a b : ℕ) : ℕ → Fin 10
  | 0 => sorry
  | n + 1 => r10 (a * (genX a b n).val + b)

/-- Generates the c_i sequence based on x_i and y_i. -/
def genC (x : ℕ → Fin 10) (y : ℕ → Fin 100) : ℕ → Fin 10 :=
  fun i => r10 (x i + (y i).val)

/-- Converts a string to a sequence of ordinal numbers. -/
def stringToOrdinals (s : String) : ℕ → Fin 100 := sorry

/-- The stored sequence c_i. -/
def storedSequence : List (Fin 10) :=
  [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7, 5]

/-- The password in lowercase letters. -/
def password : String := "яхта"

theorem password_matches_stored_sequence :
  ∃ (a b : ℕ),
    (∀ i, i ≥ 10 → genX a b i = genX a b (i - 10)) ∧
    genC (genX a b) (stringToOrdinals (password ++ password)) = fun i =>
      if h : i < storedSequence.length then
        storedSequence[i]'h
      else
        0 := by sorry

end NUMINAMATH_CALUDE_password_matches_stored_sequence_l1889_188983


namespace NUMINAMATH_CALUDE_fertility_rate_not_valid_indicator_other_indicators_are_valid_l1889_188916

-- Define the type for population growth indicators
inductive PopulationGrowthIndicator
  | BirthRate
  | MortalityRate
  | NaturalIncreaseRate
  | FertilityRate

-- Define the set of valid indicators
def validIndicators : Set PopulationGrowthIndicator :=
  {PopulationGrowthIndicator.BirthRate,
   PopulationGrowthIndicator.MortalityRate,
   PopulationGrowthIndicator.NaturalIncreaseRate}

-- Theorem: Fertility rate is not a valid indicator
theorem fertility_rate_not_valid_indicator :
  PopulationGrowthIndicator.FertilityRate ∉ validIndicators :=
by
  sorry

-- Theorem: All other indicators are valid
theorem other_indicators_are_valid :
  PopulationGrowthIndicator.BirthRate ∈ validIndicators ∧
  PopulationGrowthIndicator.MortalityRate ∈ validIndicators ∧
  PopulationGrowthIndicator.NaturalIncreaseRate ∈ validIndicators :=
by
  sorry

end NUMINAMATH_CALUDE_fertility_rate_not_valid_indicator_other_indicators_are_valid_l1889_188916


namespace NUMINAMATH_CALUDE_peters_initial_money_l1889_188998

/-- The cost of Peter's glasses purchase --/
def glasses_purchase (small_cost large_cost : ℕ) (small_count large_count : ℕ) (change : ℕ) : Prop :=
  ∃ (initial_amount : ℕ),
    initial_amount = small_cost * small_count + large_cost * large_count + change

/-- Theorem stating Peter's initial amount of money --/
theorem peters_initial_money :
  glasses_purchase 3 5 8 5 1 → ∃ (initial_amount : ℕ), initial_amount = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_initial_money_l1889_188998


namespace NUMINAMATH_CALUDE_min_value_theorem_l1889_188994

-- Define the condition function
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x * x ≥ 0

-- State the theorem
theorem min_value_theorem (a b : ℝ) (h : condition a b) : 
  ∃ (min : ℝ), min = 1 ∧ ∀ (c : ℝ), b / a ≥ c := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1889_188994


namespace NUMINAMATH_CALUDE_min_ice_cost_l1889_188942

/-- Represents the ice purchasing options --/
inductive IcePackType
  | OnePound
  | FivePound

/-- Calculates the cost of ice for a given pack type and number of packs --/
def calculateCost (packType : IcePackType) (numPacks : ℕ) : ℚ :=
  match packType with
  | IcePackType.OnePound => 
      if numPacks > 20 
      then (6 * numPacks : ℚ) * 0.9
      else 6 * numPacks
  | IcePackType.FivePound => 
      if numPacks > 20 
      then (25 * numPacks : ℚ) * 0.9
      else 25 * numPacks

/-- Calculates the number of packs needed for a given pack type and total ice needed --/
def calculatePacks (packType : IcePackType) (totalIce : ℕ) : ℕ :=
  match packType with
  | IcePackType.OnePound => (totalIce + 9) / 10
  | IcePackType.FivePound => (totalIce + 49) / 50

/-- Theorem: The minimum cost for ice is $100.00 --/
theorem min_ice_cost : 
  let totalPeople : ℕ := 50
  let icePerPerson : ℕ := 4
  let totalIce : ℕ := totalPeople * icePerPerson
  let onePoundCost := calculateCost IcePackType.OnePound (calculatePacks IcePackType.OnePound totalIce)
  let fivePoundCost := calculateCost IcePackType.FivePound (calculatePacks IcePackType.FivePound totalIce)
  min onePoundCost fivePoundCost = 100 := by
  sorry

end NUMINAMATH_CALUDE_min_ice_cost_l1889_188942


namespace NUMINAMATH_CALUDE_angle_sum_inequality_l1889_188969

theorem angle_sum_inequality (α β γ x y z : ℝ) 
  (h_angles : α + β + γ = Real.pi)
  (h_sum : x + y + z = 0) :
  y * z * Real.sin α ^ 2 + z * x * Real.sin β ^ 2 + x * y * Real.sin γ ^ 2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_inequality_l1889_188969


namespace NUMINAMATH_CALUDE_smallest_duck_count_is_975_l1889_188976

/-- Represents the number of birds in a flock for each type --/
structure FlockSize where
  ducks : Nat
  cranes : Nat
  herons : Nat

/-- Represents the number of flocks for each type of bird --/
structure FlockCount where
  ducks : Nat
  cranes : Nat
  herons : Nat

/-- The smallest number of ducks that satisfies the given conditions --/
def smallest_duck_count (fs : FlockSize) (fc : FlockCount) : Nat :=
  fs.ducks * fc.ducks

/-- Theorem stating the smallest number of ducks observed --/
theorem smallest_duck_count_is_975 (fs : FlockSize) (fc : FlockCount) :
  fs.ducks = 13 →
  fs.cranes = 17 →
  fs.herons = 11 →
  fs.ducks * fc.ducks + fs.cranes * fc.cranes = 15 * fs.herons * fc.herons →
  5 * fc.cranes = 3 * fc.ducks →
  smallest_duck_count fs fc = 975 := by
  sorry

end NUMINAMATH_CALUDE_smallest_duck_count_is_975_l1889_188976


namespace NUMINAMATH_CALUDE_solve_for_c_l1889_188970

/-- Given two functions p and q, where p(x) = 3x - 9 and q(x) = 4x - c,
    prove that c = 4 when p(q(3)) = 15 -/
theorem solve_for_c (p q : ℝ → ℝ) (c : ℝ) 
    (hp : ∀ x, p x = 3 * x - 9)
    (hq : ∀ x, q x = 4 * x - c)
    (h_eq : p (q 3) = 15) : 
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l1889_188970


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l1889_188929

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line :=
  { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line_through_point :
  ∃ (b : Line), 
    parallel b givenLine ∧ 
    pointOnLine b 4 (-2) ∧ 
    b.yIntercept = 10 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l1889_188929


namespace NUMINAMATH_CALUDE_population_ratio_A_to_F_l1889_188953

/-- Represents the population of a city -/
structure CityPopulation where
  population : ℕ

/-- Represents the relationship between populations of different cities -/
structure PopulationRelationship where
  A : CityPopulation
  B : CityPopulation
  C : CityPopulation
  D : CityPopulation
  E : CityPopulation
  F : CityPopulation
  A_to_B : A.population = 5 * B.population
  B_to_C : B.population = 3 * C.population
  C_to_D : C.population = 8 * D.population
  D_to_E : D.population = 2 * E.population
  E_to_F : E.population = 6 * F.population

/-- The theorem stating the ratio of populations between City A and City F -/
theorem population_ratio_A_to_F (r : PopulationRelationship) :
  r.A.population = 1440 * r.F.population := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_A_to_F_l1889_188953


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_answer_is_valid_l1889_188968

def is_valid_number (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.length = 10 ∧ 
    digits.toFinset = Finset.range 10 ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem largest_multiple_of_12 :
  ∀ n : ℕ, 
    is_valid_number n → 
    is_multiple_of_12 n → 
    n ≤ 9876543120 :=
by sorry

theorem answer_is_valid :
  is_valid_number 9876543120 ∧ is_multiple_of_12 9876543120 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_answer_is_valid_l1889_188968


namespace NUMINAMATH_CALUDE_abc_divisibility_theorem_l1889_188944

theorem abc_divisibility_theorem (a b c : ℕ+) 
  (h1 : a * b ∣ c * (c^2 - c + 1)) 
  (h2 : (c^2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c^2 - c + 1) ∨ (a = c^2 - c + 1 ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_abc_divisibility_theorem_l1889_188944


namespace NUMINAMATH_CALUDE_part_one_part_two_l1889_188977

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := 4 - |x + 1|

-- Part I
theorem part_one : 
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≥ 4 ∨ x ≤ -2} :=
by sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x, f x - g x ≥ a^2 - 3*a} = {a : ℝ | 1 ≤ a ∧ a ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1889_188977


namespace NUMINAMATH_CALUDE_pig_count_l1889_188906

theorem pig_count (P1 P2 : ℕ) (h1 : P1 = 64) (h2 : P1 + P2 = 86) : P2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l1889_188906


namespace NUMINAMATH_CALUDE_inventory_problem_l1889_188963

/-- The inventory problem -/
theorem inventory_problem
  (ties : ℕ) (belts : ℕ) (black_shirts : ℕ) (white_shirts : ℕ) (hats : ℕ) (socks : ℕ)
  (h_ties : ties = 34)
  (h_belts : belts = 40)
  (h_black_shirts : black_shirts = 63)
  (h_white_shirts : white_shirts = 42)
  (h_hats : hats = 25)
  (h_socks : socks = 80)
  : let jeans := (2 * (black_shirts + white_shirts)) / 3
    let scarves := (ties + belts) / 2
    let jackets := hats + hats / 5
    jeans - (scarves + jackets) = 3 := by
  sorry

end NUMINAMATH_CALUDE_inventory_problem_l1889_188963


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l1889_188984

theorem binomial_12_choose_6 : Nat.choose 12 6 = 1848 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l1889_188984


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1889_188999

theorem inequality_not_always_true (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬ (∀ a b, a > b ∧ b > 0 → a + b < 2 * Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1889_188999


namespace NUMINAMATH_CALUDE_teacher_budget_shortfall_l1889_188949

def euro_to_usd_rate : ℝ := 1.2
def last_year_budget : ℝ := 6
def this_year_allocation : ℝ := 50
def charity_grant : ℝ := 20
def gift_card : ℝ := 10

def textbooks_price : ℝ := 45
def textbooks_discount : ℝ := 0.15
def textbooks_tax : ℝ := 0.08

def notebooks_price : ℝ := 18
def notebooks_discount : ℝ := 0.10
def notebooks_tax : ℝ := 0.05

def pens_price : ℝ := 27
def pens_discount : ℝ := 0.05
def pens_tax : ℝ := 0.06

def art_supplies_price : ℝ := 35
def art_supplies_tax : ℝ := 0.07

def folders_price : ℝ := 15
def folders_voucher : ℝ := 5
def folders_tax : ℝ := 0.04

theorem teacher_budget_shortfall :
  let converted_budget := last_year_budget * euro_to_usd_rate
  let total_budget := converted_budget + this_year_allocation + charity_grant + gift_card
  
  let textbooks_cost := textbooks_price * (1 - textbooks_discount) * (1 + textbooks_tax)
  let notebooks_cost := notebooks_price * (1 - notebooks_discount) * (1 + notebooks_tax)
  let pens_cost := pens_price * (1 - pens_discount) * (1 + pens_tax)
  let art_supplies_cost := art_supplies_price * (1 + art_supplies_tax)
  let folders_cost := (folders_price - folders_voucher) * (1 + folders_tax)
  
  let total_cost := textbooks_cost + notebooks_cost + pens_cost + art_supplies_cost + folders_cost - gift_card
  
  total_budget - total_cost = -36.16 :=
by sorry

end NUMINAMATH_CALUDE_teacher_budget_shortfall_l1889_188949


namespace NUMINAMATH_CALUDE_olivias_initial_money_l1889_188987

theorem olivias_initial_money (initial_money : ℕ) 
  (atm_collection : ℕ) (supermarket_spending : ℕ) (money_left : ℕ) :
  atm_collection = 91 →
  money_left = 14 →
  supermarket_spending = atm_collection + 39 →
  initial_money + atm_collection - supermarket_spending = money_left →
  initial_money = 53 := by
sorry

end NUMINAMATH_CALUDE_olivias_initial_money_l1889_188987


namespace NUMINAMATH_CALUDE_inequality_proof_l1889_188921

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1889_188921


namespace NUMINAMATH_CALUDE_total_miles_run_l1889_188952

/-- Given that Sam runs 12 miles and Harvey runs 8 miles more than Sam,
    prove that the total distance run by both friends is 32 miles. -/
theorem total_miles_run (sam_miles harvey_miles total_miles : ℕ) : 
  sam_miles = 12 →
  harvey_miles = sam_miles + 8 →
  total_miles = sam_miles + harvey_miles →
  total_miles = 32 := by
sorry

end NUMINAMATH_CALUDE_total_miles_run_l1889_188952


namespace NUMINAMATH_CALUDE_john_typing_duration_l1889_188904

/-- The time John typed before Jack took over -/
def john_typing_time (
  john_total_time : ℝ)
  (jack_rate_ratio : ℝ)
  (jack_completion_time : ℝ) : ℝ :=
  3

/-- Theorem stating that John typed for 3 hours before Jack took over -/
theorem john_typing_duration :
  john_typing_time 5 (2/5) 4.999999999999999 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_typing_duration_l1889_188904


namespace NUMINAMATH_CALUDE_count_numbers_with_6_or_8_is_452_l1889_188995

/-- The count of three-digit whole numbers containing at least one digit 6 or at least one digit 8 -/
def count_numbers_with_6_or_8 : ℕ :=
  let total_three_digit_numbers := 999 - 100 + 1
  let digits_without_6_or_8 := 8  -- 0-5, 7, 9
  let first_digit_choices := 7    -- 1-5, 7, 9
  let numbers_without_6_or_8 := first_digit_choices * digits_without_6_or_8 * digits_without_6_or_8
  total_three_digit_numbers - numbers_without_6_or_8

theorem count_numbers_with_6_or_8_is_452 : count_numbers_with_6_or_8 = 452 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_6_or_8_is_452_l1889_188995


namespace NUMINAMATH_CALUDE_measure_15_minutes_l1889_188978

/-- Represents an hourglass with a given duration in minutes -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses -/
structure MeasurementState where
  time_elapsed : ℕ
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Checks if it's possible to measure the target time using two hourglasses -/
def can_measure_time (target : ℕ) (h1 : Hourglass) (h2 : Hourglass) : Prop :=
  ∃ (steps : ℕ) (final_state : MeasurementState),
    final_state.time_elapsed = target ∧
    final_state.hourglass1 = h1 ∧
    final_state.hourglass2 = h2

theorem measure_15_minutes :
  can_measure_time 15 (Hourglass.mk 7) (Hourglass.mk 11) :=
sorry

end NUMINAMATH_CALUDE_measure_15_minutes_l1889_188978


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l1889_188905

/-- Conversion factor from feet to inches -/
def inches_per_foot : ℕ := 12

/-- Cubic inches in one cubic foot -/
def cubic_inches_per_cubic_foot : ℕ := inches_per_foot ^ 3

theorem cubic_foot_to_cubic_inches :
  cubic_inches_per_cubic_foot = 1728 :=
sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l1889_188905
