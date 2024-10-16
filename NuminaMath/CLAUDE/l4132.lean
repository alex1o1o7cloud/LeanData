import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_for_f_squared_minimum_value_of_g_l4132_413282

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a (2*x + a) + 2 * f a x

-- Part 1
theorem solution_set_for_f_squared (x : ℝ) :
  f 1 x ^ 2 ≤ 2 ↔ 1 - Real.sqrt 2 ≤ x ∧ x ≤ 1 + Real.sqrt 2 :=
sorry

-- Part 2
theorem minimum_value_of_g (a : ℝ) (h : a > 0) :
  (∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), g a x ≥ m) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_f_squared_minimum_value_of_g_l4132_413282


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l4132_413237

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) : ℕ → ℤ := fun n => a + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
def term50 : ℤ := arithmeticSequence 3 2 50

/-- Theorem: The 50th term of the arithmetic sequence with first term 3 and common difference 2 is 101 -/
theorem arithmetic_sequence_50th_term : term50 = 101 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l4132_413237


namespace NUMINAMATH_CALUDE_not_all_prime_in_sequence_l4132_413248

/-- Given a sequence defined by x₀, a, b ∈ ℕ and xₙ₊₁ = xₙ · a + b,
    there exists at least one n ∈ ℕ such that xₙ is not prime. -/
theorem not_all_prime_in_sequence (x₀ a b : ℕ) :
  ∃ n : ℕ, ¬ Prime (Nat.iterate (λ x => x * a + b) n x₀) := by
  sorry

end NUMINAMATH_CALUDE_not_all_prime_in_sequence_l4132_413248


namespace NUMINAMATH_CALUDE_arrange_algebra_and_calculus_books_l4132_413231

/-- The number of ways to arrange books on a shelf --/
def arrange_books (algebra_copies : ℕ) (calculus_copies : ℕ) : ℕ :=
  Nat.choose (algebra_copies + calculus_copies) algebra_copies

/-- Theorem: Arranging 4 algebra books and 5 calculus books yields 126 possibilities --/
theorem arrange_algebra_and_calculus_books :
  arrange_books 4 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_arrange_algebra_and_calculus_books_l4132_413231


namespace NUMINAMATH_CALUDE_bottle_production_l4132_413272

theorem bottle_production (machines_base : ℕ) (bottles_per_minute : ℕ) 
  (machines_target : ℕ) (minutes : ℕ) : 
  machines_base = 6 → 
  bottles_per_minute = 270 → 
  machines_target = 20 → 
  minutes = 4 → 
  (machines_target * bottles_per_minute * minutes) / machines_base = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l4132_413272


namespace NUMINAMATH_CALUDE_max_value_quadratic_max_value_quadratic_achievable_l4132_413261

theorem max_value_quadratic (p : ℝ) : -3 * p^2 + 54 * p - 30 ≤ 213 := by sorry

theorem max_value_quadratic_achievable : ∃ p : ℝ, -3 * p^2 + 54 * p - 30 = 213 := by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_max_value_quadratic_achievable_l4132_413261


namespace NUMINAMATH_CALUDE_third_function_symmetry_l4132_413205

-- Define the type for our functions
def RealFunction := ℝ → ℝ

-- State the theorem
theorem third_function_symmetry 
  (f : RealFunction) 
  (f_inv : RealFunction) 
  (h : RealFunction) 
  (h1 : ∀ x, f_inv (f x) = x) -- f_inv is the inverse of f
  (h2 : ∀ x y, h y = -x ↔ f_inv (-y) = -x) -- h is symmetric to f_inv w.r.t. x + y = 0
  : ∀ x, h x = -f (-x) := by
  sorry


end NUMINAMATH_CALUDE_third_function_symmetry_l4132_413205


namespace NUMINAMATH_CALUDE_estate_distribution_l4132_413265

/-- Represents the estate distribution problem --/
theorem estate_distribution (total : ℚ) 
  (daughter_share : ℚ) (son_share : ℚ) (husband_share : ℚ) (gardener_share : ℚ) : 
  daughter_share + son_share = (3 : ℚ) / 5 * total →
  daughter_share = (3 : ℚ) / 5 * (daughter_share + son_share) →
  husband_share = 3 * son_share →
  gardener_share = 600 →
  total = daughter_share + son_share + husband_share + gardener_share →
  total = 1875 := by
  sorry

end NUMINAMATH_CALUDE_estate_distribution_l4132_413265


namespace NUMINAMATH_CALUDE_complex_equality_problem_l4132_413222

theorem complex_equality_problem (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 2 - Complex.I
  Complex.abs z₁ = Complex.abs z₂ → (a = 2 ∨ a = -2) := by
sorry

end NUMINAMATH_CALUDE_complex_equality_problem_l4132_413222


namespace NUMINAMATH_CALUDE_tech_gadget_cost_conversion_l4132_413215

/-- Proves that a tech gadget costing 160 Namibian dollars is equivalent to 100 Indian rupees given the exchange rates. -/
theorem tech_gadget_cost_conversion :
  -- Define the exchange rates
  let usd_to_namibian : ℚ := 8
  let usd_to_indian : ℚ := 5
  -- Define the cost in Namibian dollars
  let cost_namibian : ℚ := 160
  -- Define the function to convert Namibian dollars to Indian rupees
  let namibian_to_indian (n : ℚ) : ℚ := n / usd_to_namibian * usd_to_indian
  -- State the theorem
  namibian_to_indian cost_namibian = 100 := by
  sorry

end NUMINAMATH_CALUDE_tech_gadget_cost_conversion_l4132_413215


namespace NUMINAMATH_CALUDE_standard_colony_conditions_l4132_413244

/-- Represents the type of culture medium -/
inductive CultureMedium
| Liquid
| Solid

/-- Represents a bacterial colony -/
structure BacterialColony where
  initialBacteria : ℕ
  medium : CultureMedium

/-- Defines what constitutes a standard bacterial colony -/
def isStandardColony (colony : BacterialColony) : Prop :=
  colony.initialBacteria = 1 ∧ colony.medium = CultureMedium.Solid

/-- Theorem stating the conditions for a standard bacterial colony -/
theorem standard_colony_conditions :
  ∀ (colony : BacterialColony),
    isStandardColony colony ↔
      colony.initialBacteria = 1 ∧ colony.medium = CultureMedium.Solid :=
by
  sorry

end NUMINAMATH_CALUDE_standard_colony_conditions_l4132_413244


namespace NUMINAMATH_CALUDE_kim_easy_round_answers_l4132_413232

/-- Represents the number of points for each round in the math contest -/
structure ContestPoints where
  easy : ℕ
  average : ℕ
  hard : ℕ

/-- Represents the number of correct answers for each round -/
structure ContestAnswers where
  easy : ℕ
  average : ℕ
  hard : ℕ

def totalPoints (points : ContestPoints) (answers : ContestAnswers) : ℕ :=
  points.easy * answers.easy + points.average * answers.average + points.hard * answers.hard

theorem kim_easy_round_answers 
  (points : ContestPoints) 
  (answers : ContestAnswers) 
  (h1 : points.easy = 2) 
  (h2 : points.average = 3) 
  (h3 : points.hard = 5)
  (h4 : answers.average = 2)
  (h5 : answers.hard = 4)
  (h6 : totalPoints points answers = 38) : 
  answers.easy = 6 := by
sorry

end NUMINAMATH_CALUDE_kim_easy_round_answers_l4132_413232


namespace NUMINAMATH_CALUDE_sum_of_digits_a_l4132_413202

def a : ℕ := 10^101 - 100

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_a : sum_of_digits a = 891 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_a_l4132_413202


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l4132_413242

/-- Represents the price reduction scenario for a shirt -/
def price_reduction_scenario (original_price final_price : ℝ) (x : ℝ) : Prop :=
  original_price * (1 - x)^2 = final_price

/-- The theorem stating that the given equation correctly represents the price reduction scenario -/
theorem correct_price_reduction_equation :
  price_reduction_scenario 400 200 x ↔ 400 * (1 - x)^2 = 200 :=
by sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l4132_413242


namespace NUMINAMATH_CALUDE_base9_to_base10_3956_l4132_413233

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (a b c d : ℕ) : ℕ :=
  a * 9^3 + b * 9^2 + c * 9^1 + d * 9^0

/-- Theorem: The base-9 number 3956₉ is equal to 2967 in base-10 --/
theorem base9_to_base10_3956 : base9ToBase10 3 9 5 6 = 2967 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base10_3956_l4132_413233


namespace NUMINAMATH_CALUDE_division_problem_l4132_413299

theorem division_problem :
  ∃ x : ℝ, x > 0 ∧ 
    2 * x + (100 - 2 * x) = 100 ∧
    (300 - 6 * x) + x = 100 ∧
    x = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4132_413299


namespace NUMINAMATH_CALUDE_divides_n_squared_plus_one_l4132_413291

theorem divides_n_squared_plus_one (n : ℕ) : 
  (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divides_n_squared_plus_one_l4132_413291


namespace NUMINAMATH_CALUDE_frog_return_probability_l4132_413252

/-- Represents the probability of the frog being at position (x, y) after n hops -/
def prob (n : ℕ) (x y : ℕ) : ℚ :=
  sorry

/-- The grid size is 2x2 -/
def grid_size : ℕ := 2

/-- The number of hops the frog makes -/
def num_hops : ℕ := 3

/-- The probability of each possible movement (right, down, or stay) -/
def move_prob : ℚ := 1 / 3

theorem frog_return_probability :
  prob num_hops 0 0 = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_frog_return_probability_l4132_413252


namespace NUMINAMATH_CALUDE_work_completion_time_l4132_413278

/-- Given a work that can be completed by X in 40 days, prove that if X works for 8 days
    and Y finishes the remaining work in 32 days, then Y would take 40 days to complete
    the entire work alone. -/
theorem work_completion_time (x_total_days y_completion_days : ℕ) 
    (x_worked_days : ℕ) (h1 : x_total_days = 40) (h2 : x_worked_days = 8) 
    (h3 : y_completion_days = 32) : 
    (x_total_days : ℚ) * y_completion_days / (x_total_days - x_worked_days) = 40 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4132_413278


namespace NUMINAMATH_CALUDE_car_speed_l4132_413238

/-- Theorem: Given a car traveling for 5 hours and covering a distance of 800 km, its speed is 160 km/hour. -/
theorem car_speed (time : ℝ) (distance : ℝ) (speed : ℝ) : 
  time = 5 → distance = 800 → speed = distance / time → speed = 160 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_l4132_413238


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4132_413287

theorem fraction_to_decimal : (7 : ℚ) / 32 = 0.21875 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4132_413287


namespace NUMINAMATH_CALUDE_simplify_expression_l4132_413295

theorem simplify_expression (x : ℝ) :
  3*x + 4*x^2 + 2 - (5 - 3*x - 5*x^2 + x^3) = -x^3 + 9*x^2 + 6*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4132_413295


namespace NUMINAMATH_CALUDE_one_fifth_of_seven_x_plus_three_l4132_413289

theorem one_fifth_of_seven_x_plus_three (x : ℝ) : 
  (1 / 5) * (7 * x + 3) = (7 / 5) * x + 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_seven_x_plus_three_l4132_413289


namespace NUMINAMATH_CALUDE_acute_angle_is_40_isosceles_trapezoid_acute_angle_l4132_413204

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The length of the diagonal -/
  diagonal : ℝ
  /-- The angle between the diagonal and the leg -/
  angle_diagonal_leg : ℝ
  /-- The angle between the diagonal and the base -/
  angle_diagonal_base : ℝ
  /-- The area is √3 -/
  area_eq : area = Real.sqrt 3
  /-- The diagonal length is 2 -/
  diagonal_eq : diagonal = 2
  /-- The angle between the diagonal and the base is 20° greater than the angle between the diagonal and the leg -/
  angle_relation : angle_diagonal_base = angle_diagonal_leg + 20

/-- The acute angle of the trapezoid is 40° -/
theorem acute_angle_is_40 (t : IsoscelesTrapezoid) : ℝ :=
  40

/-- The main theorem: proving that the acute angle of the trapezoid is 40° -/
theorem isosceles_trapezoid_acute_angle 
  (t : IsoscelesTrapezoid) : acute_angle_is_40 t = 40 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_is_40_isosceles_trapezoid_acute_angle_l4132_413204


namespace NUMINAMATH_CALUDE_C_power_50_l4132_413275

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l4132_413275


namespace NUMINAMATH_CALUDE_tower_surface_area_l4132_413221

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the surface area of a cube given its visible faces -/
def surfaceArea (cube : Cube) (visibleFaces : ℕ) : ℕ :=
  visibleFaces * cube.sideLength * cube.sideLength

/-- Represents a tower of cubes -/
def CubeTower := List Cube

/-- Calculates the total surface area of a tower of cubes -/
def totalSurfaceArea (tower : CubeTower) : ℕ :=
  match tower with
  | [] => 0
  | [c] => surfaceArea c 6  -- Top cube has all 6 faces visible
  | c :: rest => surfaceArea c 5 + (rest.map (surfaceArea · 4)).sum

theorem tower_surface_area :
  let tower : CubeTower := [
    { sideLength := 1 },
    { sideLength := 2 },
    { sideLength := 3 },
    { sideLength := 4 },
    { sideLength := 5 },
    { sideLength := 6 },
    { sideLength := 7 }
  ]
  totalSurfaceArea tower = 610 := by sorry

end NUMINAMATH_CALUDE_tower_surface_area_l4132_413221


namespace NUMINAMATH_CALUDE_diamond_calculation_l4132_413218

/-- The diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 2 ◇ (3 ◇ (1 ◇ 4)) = -46652 -/
theorem diamond_calculation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l4132_413218


namespace NUMINAMATH_CALUDE_ap_terms_count_l4132_413250

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Even n →
  n / 2 * (2 * a + (n - 2) * d) = 18 →
  n / 2 * (2 * a + 2 * d + (n - 2) * d) = 36 →
  a + (n - 1) * d - a = 7 →
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_ap_terms_count_l4132_413250


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l4132_413209

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_y_intercept :
  let m := (3 : ℝ) * P.1^2  -- Slope of the tangent line
  let b := P.2 - m * P.1    -- y-intercept of the tangent line
  b = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l4132_413209


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l4132_413230

/-- Given a man's speed with the current and the speed of the current,
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Proves that given the specified conditions, the man's speed against the current is 8.6 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 3.2 = 8.6 := by
  sorry

#eval speed_against_current 15 3.2

end NUMINAMATH_CALUDE_mans_speed_against_current_l4132_413230


namespace NUMINAMATH_CALUDE_balloon_permutations_count_l4132_413257

/-- The number of distinct permutations of the letters in "BALLOON" -/
def balloon_permutations : ℕ := 1260

/-- The total number of letters in "BALLOON" -/
def total_letters : ℕ := 7

/-- The number of repeated 'L's in "BALLOON" -/
def repeated_L : ℕ := 2

/-- The number of repeated 'O's in "BALLOON" -/
def repeated_O : ℕ := 2

/-- Theorem stating that the number of distinct permutations of the letters in "BALLOON" is 1260 -/
theorem balloon_permutations_count :
  balloon_permutations = Nat.factorial total_letters / (Nat.factorial repeated_L * Nat.factorial repeated_O) :=
by sorry

end NUMINAMATH_CALUDE_balloon_permutations_count_l4132_413257


namespace NUMINAMATH_CALUDE_stock_sale_before_brokerage_l4132_413277

/-- Calculates the total amount before brokerage given the cash realized and brokerage rate -/
def totalBeforeBrokerage (cashRealized : ℚ) (brokerageRate : ℚ) : ℚ :=
  cashRealized / (1 - brokerageRate)

/-- Theorem stating that for a stock sale with cash realization of 106.25 
    after a 1/4% brokerage fee, the total amount before brokerage is approximately 106.515 -/
theorem stock_sale_before_brokerage :
  let cashRealized : ℚ := 106.25
  let brokerageRate : ℚ := 1 / 400
  let result := totalBeforeBrokerage cashRealized brokerageRate
  ⌊result * 1000⌋ / 1000 = 106515 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_before_brokerage_l4132_413277


namespace NUMINAMATH_CALUDE_company_employees_l4132_413264

theorem company_employees (december_employees : ℕ) (january_employees : ℕ) 
  (h1 : december_employees = 460) 
  (h2 : december_employees = january_employees + (january_employees * 15 / 100)) : 
  january_employees = 400 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l4132_413264


namespace NUMINAMATH_CALUDE_borrowed_amount_l4132_413281

/-- Calculates the total interest paid over 9 years given the principal amount and interest rates -/
def totalInterest (principal : ℝ) : ℝ :=
  principal * (0.06 * 2 + 0.09 * 3 + 0.14 * 4)

/-- Theorem stating that given the interest rates and total interest paid, the principal amount borrowed is 12000 -/
theorem borrowed_amount (totalInterestPaid : ℝ) 
  (h : totalInterestPaid = 11400) : 
  ∃ principal : ℝ, totalInterest principal = totalInterestPaid ∧ principal = 12000 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l4132_413281


namespace NUMINAMATH_CALUDE_inequality_range_l4132_413229

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) → 
  (a ≤ -1 ∨ a ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l4132_413229


namespace NUMINAMATH_CALUDE_ten_digit_square_plus_one_has_identical_digits_l4132_413201

-- Define a function to check if a number is ten digits
def isTenDigits (n : ℕ) : Prop := 1000000000 ≤ n ∧ n < 10000000000

-- Define a function to check if a number has at least two identical digits
def hasAtLeastTwoIdenticalDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (∃ i j : ℕ, i ≠ j ∧ (n / 10^i % 10 = d) ∧ (n / 10^j % 10 = d))

theorem ten_digit_square_plus_one_has_identical_digits (n : ℕ) :
  isTenDigits (n^2 + 1) → hasAtLeastTwoIdenticalDigits (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ten_digit_square_plus_one_has_identical_digits_l4132_413201


namespace NUMINAMATH_CALUDE_mobius_gauss_formula_pentagon_l4132_413207

/-- Given a pentagon ABCDE with triangle areas α, β, γ, θ, δ, 
    the total area S satisfies the Möbius-Gauss formula. -/
theorem mobius_gauss_formula_pentagon (α β γ θ δ S : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ θ > 0 ∧ δ > 0) 
  (h_area : S > 0) 
  (h_sum : S > (α + β + γ + θ + δ) / 2) :
  S^2 - (α + β + γ + θ + δ) * S + (α*β + β*γ + γ*θ + θ*δ + δ*α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_mobius_gauss_formula_pentagon_l4132_413207


namespace NUMINAMATH_CALUDE_spade_calculation_l4132_413211

-- Define the ◆ operation
def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

-- State the theorem
theorem spade_calculation : spade 4 (spade 5 (-2)) = -425 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l4132_413211


namespace NUMINAMATH_CALUDE_no_solution_exists_l4132_413210

/-- Represents the number of books for each subject -/
structure BookCounts where
  math : ℕ
  history : ℕ
  science : ℕ
  literature : ℕ

/-- The problem constraints -/
def satisfiesConstraints (books : BookCounts) : Prop :=
  books.math + books.history + books.science + books.literature = 80 ∧
  4 * books.math + 5 * books.history + 6 * books.science + 7 * books.literature = 520 ∧
  3 * books.history = 2 * books.math ∧
  2 * books.science = books.math ∧
  4 * books.literature = books.math

theorem no_solution_exists : ¬∃ (books : BookCounts), satisfiesConstraints books := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4132_413210


namespace NUMINAMATH_CALUDE_solution_set_correct_l4132_413216

/-- The solution set for the system of equations:
    x + y + z = 2
    (x+y)(y+z) + (y+z)(z+x) + (z+x)(x+y) = 1
    x²(y+z) + y²(z+x) + z²(x+y) = -6 -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 3, -1), (0, -1, 3), (3, 0, -1), (3, -1, 0), (-1, 0, 3), (-1, 3, 0)}

/-- The system of equations -/
def satisfies_equations (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧
  (x+y)*(y+z) + (y+z)*(z+x) + (z+x)*(x+y) = 1 ∧
  x^2*(y+z) + y^2*(z+x) + z^2*(x+y) = -6

theorem solution_set_correct :
  ∀ (x y z : ℝ), (x, y, z) ∈ solution_set ↔ satisfies_equations x y z :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_l4132_413216


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l4132_413267

/-- A circle C tangent to the y-axis at point (0,2) and tangent to the line 4x-3y+9=0 -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the y-axis at (0,2) -/
  tangent_y_axis : center.1 = radius
  /-- The circle's center is on the line y=2 -/
  center_on_line : center.2 = 2
  /-- The circle is tangent to the line 4x-3y+9=0 -/
  tangent_line : |4 * center.1 - 3 * center.2 + 9| / Real.sqrt 25 = radius

/-- The standard equation of the circle is either (x-3)^2+(y-2)^2=9 or (x+1/3)^2+(y-2)^2=1/9 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (c.center = (3, 2) ∧ c.radius = 3) ∨ (c.center = (-1/3, 2) ∧ c.radius = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l4132_413267


namespace NUMINAMATH_CALUDE_circle_with_same_center_and_radius_2_l4132_413223

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the center of a circle
def center (f : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- Define a new circle with given center and radius
def new_circle (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = r^2

-- Theorem statement
theorem circle_with_same_center_and_radius_2 :
  ∀ (x y : ℝ),
  new_circle (center given_circle) 2 x y ↔ (x + 1)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_with_same_center_and_radius_2_l4132_413223


namespace NUMINAMATH_CALUDE_baxter_peanut_purchase_l4132_413279

/-- Calculates the pounds of peanuts purchased over the minimum -/
def peanuts_over_minimum (cost_per_pound : ℚ) (minimum_pounds : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent / cost_per_pound) - minimum_pounds

theorem baxter_peanut_purchase : 
  let cost_per_pound : ℚ := 3
  let minimum_pounds : ℚ := 15
  let total_spent : ℚ := 105
  peanuts_over_minimum cost_per_pound minimum_pounds total_spent = 20 := by
sorry

end NUMINAMATH_CALUDE_baxter_peanut_purchase_l4132_413279


namespace NUMINAMATH_CALUDE_complex_exponent_l4132_413260

theorem complex_exponent (x : ℂ) (h : x - 1/x = 2*I) : x^729 - 1/x^729 = -4*I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponent_l4132_413260


namespace NUMINAMATH_CALUDE_smallest_cycle_length_cycle_length_21_smallest_b_is_21_l4132_413254

def g (x : ℤ) : ℤ :=
  if x % 5 = 0 ∧ x % 7 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iterate (n : ℕ) (x : ℤ) : ℤ :=
  match n with
  | 0 => x
  | n + 1 => g (g_iterate n x)

theorem smallest_cycle_length :
  ∀ b : ℕ, b > 1 → g_iterate b 3 = g 3 → b ≥ 21 :=
by sorry

theorem cycle_length_21 : g_iterate 21 3 = g 3 :=
by sorry

theorem smallest_b_is_21 :
  ∃! b : ℕ, b > 1 ∧ g_iterate b 3 = g 3 ∧ ∀ k : ℕ, k > 1 → g_iterate k 3 = g 3 → k ≥ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_cycle_length_cycle_length_21_smallest_b_is_21_l4132_413254


namespace NUMINAMATH_CALUDE_fundraising_problem_l4132_413240

/-- The number of workers in a fundraising scenario -/
def number_of_workers : ℕ := sorry

/-- The original contribution amount per worker -/
def original_contribution : ℕ := sorry

theorem fundraising_problem :
  (number_of_workers * original_contribution = 300000) ∧
  (number_of_workers * (original_contribution + 50) = 350000) →
  number_of_workers = 1000 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_problem_l4132_413240


namespace NUMINAMATH_CALUDE_commission_percentage_proof_l4132_413284

def commission_percentage_below_threshold (total_sales : ℚ) (threshold : ℚ) (remitted_amount : ℚ) (commission_percentage_above_threshold : ℚ) : ℚ :=
  let sales_above_threshold := total_sales - threshold
  let commission_above_threshold := sales_above_threshold * commission_percentage_above_threshold / 100
  let total_commission := total_sales - remitted_amount
  let commission_below_threshold := total_commission - commission_above_threshold
  commission_below_threshold / threshold * 100

theorem commission_percentage_proof :
  let total_sales : ℚ := 32500
  let threshold : ℚ := 10000
  let remitted_amount : ℚ := 31100
  let commission_percentage_above_threshold : ℚ := 4
  commission_percentage_below_threshold total_sales threshold remitted_amount commission_percentage_above_threshold = 5 := by
  sorry

#eval commission_percentage_below_threshold 32500 10000 31100 4

end NUMINAMATH_CALUDE_commission_percentage_proof_l4132_413284


namespace NUMINAMATH_CALUDE_square_difference_equality_l4132_413251

theorem square_difference_equality : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l4132_413251


namespace NUMINAMATH_CALUDE_all_terms_even_l4132_413293

theorem all_terms_even (m n : ℤ) (hm : Even m) (hn : Even n) :
  ∀ k : Fin 9, Even ((Finset.range 9).sum (λ i => (Nat.choose 8 i : ℤ) * m^(8 - i) * n^i)) := by
  sorry

end NUMINAMATH_CALUDE_all_terms_even_l4132_413293


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l4132_413292

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (h : x ≥ -1) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l4132_413292


namespace NUMINAMATH_CALUDE_base_r_palindrome_square_l4132_413253

theorem base_r_palindrome_square (r : ℕ) (h1 : r % 2 = 0) (h2 : r ≥ 18) : 
  let x := 5*r^3 + 5*r^2 + 5*r + 5
  let squared := x^2
  let a := squared / r^7 % r
  let b := squared / r^6 % r
  let c := squared / r^5 % r
  let d := squared / r^4 % r
  (squared = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a) ∧ 
  (d - c = 2) →
  r = 24 := by sorry

end NUMINAMATH_CALUDE_base_r_palindrome_square_l4132_413253


namespace NUMINAMATH_CALUDE_max_factors_bound_l4132_413236

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of factors for a^m where 1 ≤ a ≤ 20 and 1 ≤ m ≤ 10 is 231 -/
theorem max_factors_bound :
  ∀ a m : ℕ, 1 ≤ a → a ≤ 20 → 1 ≤ m → m ≤ 10 → num_factors (a^m) ≤ 231 := by
  sorry

end NUMINAMATH_CALUDE_max_factors_bound_l4132_413236


namespace NUMINAMATH_CALUDE_unique_solution_l4132_413213

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  lg x + lg (x - 2) = lg 3 + lg (x + 2)

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x > 2 ∧ equation x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4132_413213


namespace NUMINAMATH_CALUDE_congruence_solution_l4132_413228

theorem congruence_solution (n : ℕ) : n < 47 → (13 * n) % 47 = 9 % 47 ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l4132_413228


namespace NUMINAMATH_CALUDE_shiela_paper_stars_l4132_413271

/-- The number of classmates Shiela has -/
def num_classmates : ℕ := 9

/-- The number of stars Shiela places in each bottle -/
def stars_per_bottle : ℕ := 5

/-- The total number of paper stars Shiela prepared -/
def total_stars : ℕ := num_classmates * stars_per_bottle

/-- Theorem stating that the total number of paper stars Shiela prepared is 45 -/
theorem shiela_paper_stars : total_stars = 45 := by
  sorry

end NUMINAMATH_CALUDE_shiela_paper_stars_l4132_413271


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l4132_413274

/-- A biased coin with probability of heads and tails, and corresponding gains/losses -/
structure BiasedCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ

/-- The expected value of a coin flip -/
def expected_value (c : BiasedCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails)

/-- Theorem: The expected value of the specific biased coin is 0 -/
theorem biased_coin_expected_value :
  let c : BiasedCoin := {
    prob_heads := 2/3,
    prob_tails := 1/3,
    gain_heads := 5,
    loss_tails := 10
  }
  expected_value c = 0 := by sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l4132_413274


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l4132_413255

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l4132_413255


namespace NUMINAMATH_CALUDE_max_queens_8x8_l4132_413270

/-- Represents a chessboard configuration -/
def ChessBoard := Fin 8 → Fin 8

/-- Checks if two positions are on the same diagonal -/
def onSameDiagonal (p1 p2 : Fin 8 × Fin 8) : Prop :=
  (p1.1 : ℤ) - (p2.1 : ℤ) = (p1.2 : ℤ) - (p2.2 : ℤ) ∨
  (p1.1 : ℤ) - (p2.1 : ℤ) = (p2.2 : ℤ) - (p1.2 : ℤ)

/-- Checks if a chessboard configuration is valid (no queens attack each other) -/
def isValidConfiguration (board : ChessBoard) : Prop :=
  ∀ i j : Fin 8, i ≠ j →
    board i ≠ board j ∧
    ¬onSameDiagonal (i, board i) (j, board j)

/-- The theorem stating that the maximum number of non-attacking queens on an 8x8 chessboard is 8 -/
theorem max_queens_8x8 :
  (∃ (board : ChessBoard), isValidConfiguration board) ∧
  (∀ (n : ℕ) (f : Fin n → Fin 8 × Fin 8),
    (∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ ¬onSameDiagonal (f i) (f j)) →
    n ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_queens_8x8_l4132_413270


namespace NUMINAMATH_CALUDE_prob_two_heads_in_four_tosses_l4132_413212

/-- The probability of getting exactly 2 heads when tossing a fair coin 4 times -/
theorem prob_two_heads_in_four_tosses : 
  let n : ℕ := 4  -- number of tosses
  let k : ℕ := 2  -- number of desired heads
  let p : ℚ := 1/2  -- probability of getting heads in a single toss
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 :=
sorry

end NUMINAMATH_CALUDE_prob_two_heads_in_four_tosses_l4132_413212


namespace NUMINAMATH_CALUDE_f_properties_l4132_413269

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x ^ 2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + Real.cos x ^ 2

theorem f_properties :
  (∀ x : ℝ, f x ≤ 5/4) ∧
  (∀ x : ℝ, f x = 5/4 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) ∧
  (∀ x : ℝ, f x = (1/2) * Real.sin (2 * x + Real.pi / 6) + 3/4) := by sorry

end NUMINAMATH_CALUDE_f_properties_l4132_413269


namespace NUMINAMATH_CALUDE_egg_collection_sum_l4132_413258

theorem egg_collection_sum (n : ℕ) (a₁ : ℕ) (d : ℕ) (S : ℕ) : 
  n = 12 → a₁ = 25 → d = 5 → S = n * (2 * a₁ + (n - 1) * d) / 2 → S = 630 := by sorry

end NUMINAMATH_CALUDE_egg_collection_sum_l4132_413258


namespace NUMINAMATH_CALUDE_xiaolongs_dad_age_l4132_413298

theorem xiaolongs_dad_age (xiaolong_age : ℕ) : 
  xiaolong_age > 0 →
  (9 * xiaolong_age = 9 * xiaolong_age) →  -- Mom's age this year
  (9 * xiaolong_age + 3 = 9 * xiaolong_age + 3) →  -- Dad's age this year
  (9 * xiaolong_age + 4 = 8 * (xiaolong_age + 1)) →  -- Dad's age next year = 8 * Xiaolong's age next year
  9 * xiaolong_age + 3 = 39 := by
sorry

end NUMINAMATH_CALUDE_xiaolongs_dad_age_l4132_413298


namespace NUMINAMATH_CALUDE_number_of_students_in_class_l4132_413288

theorem number_of_students_in_class : 
  ∀ (N : ℕ) (avg_age_all avg_age_5 avg_age_9 last_student_age : ℚ),
    avg_age_all = 15 →
    avg_age_5 = 13 →
    avg_age_9 = 16 →
    last_student_age = 16 →
    N * avg_age_all = 5 * avg_age_5 + 9 * avg_age_9 + last_student_age →
    N = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_in_class_l4132_413288


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l4132_413226

/-- A circle with 10 points and chords connecting every pair of points. -/
structure CircleWithChords where
  num_points : ℕ
  no_triple_intersections : Bool

/-- The number of triangles formed by chord intersections inside the circle. -/
def num_triangles (c : CircleWithChords) : ℕ := sorry

/-- Theorem stating that the number of triangles is 210 for a circle with 10 points. -/
theorem ten_point_circle_triangles (c : CircleWithChords) : 
  c.num_points = 10 → c.no_triple_intersections → num_triangles c = 210 := by
  sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l4132_413226


namespace NUMINAMATH_CALUDE_steps_climbed_fraction_l4132_413296

/-- Proves that climbing 25 steps in a 6-floor building with 12 steps between floors
    is equivalent to climbing 5/12 of the total steps. -/
theorem steps_climbed_fraction (total_floors : Nat) (steps_per_floor : Nat) (steps_climbed : Nat) :
  total_floors = 6 →
  steps_per_floor = 12 →
  steps_climbed = 25 →
  (steps_climbed : Rat) / ((total_floors - 1) * steps_per_floor) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_steps_climbed_fraction_l4132_413296


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_120_by_75_percent_l4132_413246

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_120_by_75_percent :
  120 * (1 + 75 / 100) = 210 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_120_by_75_percent_l4132_413246


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l4132_413283

/-- The number of candies Susan ate during the week -/
def candies_eaten (bought_tuesday bought_thursday bought_friday remaining : ℕ) : ℕ :=
  bought_tuesday + bought_thursday + bought_friday - remaining

/-- Theorem: Susan ate 6 candies during the week -/
theorem susan_ate_six_candies : candies_eaten 3 5 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l4132_413283


namespace NUMINAMATH_CALUDE_students_count_l4132_413243

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 2

/-- The number of buses needed for the trip -/
def buses_needed : ℕ := 7

/-- The number of students going on the field trip -/
def students_on_trip : ℕ := seats_per_bus * buses_needed

theorem students_count : students_on_trip = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_count_l4132_413243


namespace NUMINAMATH_CALUDE_cookies_per_batch_l4132_413217

/-- Proves that each batch of cookies must produce 4 dozens to meet the required total --/
theorem cookies_per_batch 
  (classmates : ℕ) 
  (cookies_per_student : ℕ) 
  (total_batches : ℕ) 
  (h1 : classmates = 24) 
  (h2 : cookies_per_student = 10) 
  (h3 : total_batches = 5) : 
  (classmates * cookies_per_student) / (total_batches * 12) = 4 := by
  sorry

#check cookies_per_batch

end NUMINAMATH_CALUDE_cookies_per_batch_l4132_413217


namespace NUMINAMATH_CALUDE_locus_of_parabola_vertices_l4132_413285

/-- The locus of vertices of parabolas y = x^2 + tx + 1 is y = 1 - x^2 -/
theorem locus_of_parabola_vertices :
  ∀ (t : ℝ), 
  let f (x : ℝ) := x^2 + t*x + 1
  let vertex := (- t/2, f (- t/2))
  (vertex.1)^2 + vertex.2 = 1 := by
sorry

end NUMINAMATH_CALUDE_locus_of_parabola_vertices_l4132_413285


namespace NUMINAMATH_CALUDE_sequence_range_l4132_413294

-- Define the sequence a_n
def a (n : ℕ+) (p : ℝ) : ℝ := 2 * (n : ℝ)^2 + p * (n : ℝ)

-- State the theorem
theorem sequence_range (p : ℝ) :
  (∀ n : ℕ+, a n p < a (n + 1) p) ↔ p > -6 :=
sorry

end NUMINAMATH_CALUDE_sequence_range_l4132_413294


namespace NUMINAMATH_CALUDE_cereal_eating_time_l4132_413266

/-- The time it takes for three people to collectively eat a certain amount of cereal -/
def eating_time (fat_rate thin_rate medium_rate total_pounds : ℚ) : ℚ :=
  total_pounds / (1 / fat_rate + 1 / thin_rate + 1 / medium_rate)

/-- Theorem stating that given the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium,
    the time required for them to collectively eat 5 pounds of cereal is 100/3 minutes -/
theorem cereal_eating_time :
  eating_time 20 30 15 5 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l4132_413266


namespace NUMINAMATH_CALUDE_at_least_one_composite_l4132_413262

theorem at_least_one_composite (a b c : ℕ) (h1 : c ≥ 2) (h2 : 1 / a + 1 / b = 1 / c) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (a + c = x * y ∨ b + c = x * y)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_composite_l4132_413262


namespace NUMINAMATH_CALUDE_equation_proof_l4132_413290

theorem equation_proof : 289 + 2 * 17 * 4 + 16 = 441 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l4132_413290


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4132_413224

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - 2*i) / i) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4132_413224


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l4132_413203

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  -1 / (x + 1) - (a + 1) * log (x + 1) + a * x + Real.exp 1 - 2

theorem f_has_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    x₁ > -1 ∧ x₂ > -1 ∧ x₃ > -1 ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, x > -1 → f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔
  a > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l4132_413203


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l4132_413276

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l4132_413276


namespace NUMINAMATH_CALUDE_amount_per_friend_is_correct_l4132_413214

/-- The amount each friend pays when 6 friends split a $400 bill equally after applying a 5% discount -/
def amount_per_friend : ℚ :=
  let total_bill : ℚ := 400
  let discount_rate : ℚ := 5 / 100
  let num_friends : ℕ := 6
  let discounted_bill : ℚ := total_bill * (1 - discount_rate)
  discounted_bill / num_friends

/-- Theorem stating that the amount each friend pays is $63.33 (repeating) -/
theorem amount_per_friend_is_correct :
  amount_per_friend = 190 / 3 := by
  sorry

#eval amount_per_friend

end NUMINAMATH_CALUDE_amount_per_friend_is_correct_l4132_413214


namespace NUMINAMATH_CALUDE_f_composition_value_l4132_413208

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then
    if z.re = 0 then 2 * z else -z^2 + 1
  else z^2 + 1

theorem f_composition_value : f (f (f (f (1 + I)))) = 378 + 336 * I := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l4132_413208


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l4132_413286

theorem arithmetic_sequence_inequality (a b c : ℝ) (h : ∃ d : ℝ, d ≠ 0 ∧ b - a = d ∧ c - b = d) :
  ¬ (∀ a b c : ℝ, a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l4132_413286


namespace NUMINAMATH_CALUDE_at_least_eight_composite_l4132_413297

theorem at_least_eight_composite (n : ℕ) (h : n > 1000) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ 
  (∀ x ∈ s, x ≥ n ∧ x < n + 12) ∧
  (∃ (t : Finset ℕ), t ⊆ s ∧ t.card ≥ 8 ∧ ∀ y ∈ t, ¬ Nat.Prime y) := by
  sorry

end NUMINAMATH_CALUDE_at_least_eight_composite_l4132_413297


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l4132_413241

theorem sine_cosine_inequality : 
  Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ 
  Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l4132_413241


namespace NUMINAMATH_CALUDE_rachel_total_problems_l4132_413234

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes_solved : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes_solved + problems_next_day

/-- Proof that Rachel solved 76 math problems in total -/
theorem rachel_total_problems :
  total_problems 5 12 16 = 76 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_problems_l4132_413234


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4132_413259

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + 2 * Complex.I) = 3 - Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4132_413259


namespace NUMINAMATH_CALUDE_max_candies_after_20_hours_l4132_413239

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Next state of candies after one hour -/
def nextState (n : ℕ) : ℕ := n + sumOfDigits n

/-- State of candies after t hours, starting from 1 -/
def candyState (t : ℕ) : ℕ :=
  match t with
  | 0 => 1
  | t + 1 => nextState (candyState t)

theorem max_candies_after_20_hours :
  candyState 20 = 148 := by sorry

end NUMINAMATH_CALUDE_max_candies_after_20_hours_l4132_413239


namespace NUMINAMATH_CALUDE_multiply_subtract_equation_l4132_413280

theorem multiply_subtract_equation : ∃ x : ℝ, 12 * x - 3 = (12 - 7) * 9 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_equation_l4132_413280


namespace NUMINAMATH_CALUDE_product_equality_l4132_413249

theorem product_equality (x y : ℤ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l4132_413249


namespace NUMINAMATH_CALUDE_goldfish_percentage_l4132_413206

theorem goldfish_percentage (surface : ℕ) (below : ℕ) : 
  surface = 15 → below = 45 → (surface : ℚ) / (surface + below : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_percentage_l4132_413206


namespace NUMINAMATH_CALUDE_two_true_propositions_l4132_413235

-- Define the original proposition
def original_prop (a b c : ℝ) : Prop :=
  a > b → a * c^2 > b * c^2

-- Define the converse of the original proposition
def converse_prop (a b c : ℝ) : Prop :=
  a * c^2 > b * c^2 → a > b

-- Define the negation of the original proposition
def negation_prop (a b c : ℝ) : Prop :=
  ¬(a > b → a * c^2 > b * c^2)

-- Theorem statement
theorem two_true_propositions :
  ∃ (p q : Prop) (r : Prop),
    (p = ∀ a b c : ℝ, original_prop a b c) ∧
    (q = ∀ a b c : ℝ, converse_prop a b c) ∧
    (r = ∀ a b c : ℝ, negation_prop a b c) ∧
    ((¬p ∧ q ∧ r) ∨ (p ∧ ¬q ∧ r) ∨ (p ∧ q ∧ ¬r)) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l4132_413235


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l4132_413225

theorem prime_power_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  let n := (2^(2*p) - 1) / 3
  n ∣ (2^n - 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l4132_413225


namespace NUMINAMATH_CALUDE_initial_birds_count_l4132_413268

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := sorry

/-- The number of birds that landed on the fence -/
def landed_birds : ℕ := 8

/-- The total number of birds after more birds landed -/
def total_birds : ℕ := 20

/-- Theorem stating that the initial number of birds is 12 -/
theorem initial_birds_count : initial_birds = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l4132_413268


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l4132_413263

/-- Given two perpendicular lines, prove that the distance from (m, 1) to the y-axis is 0 or 5 -/
theorem distance_to_y_axis (m : ℝ) : 
  (∃ x y, mx - (x + 2) * y + 2 = 0 ∧ 3 * x - m * y - 1 = 0) →  -- Lines exist
  (∀ x₁ y₁ x₂ y₂, mx₁ - (x₁ + 2) * y₁ + 2 = 0 ∧ 3 * x₂ - m * y₂ - 1 = 0 → 
    (m * 3 + m * (m + 2) = 0)) →  -- Lines are perpendicular
  (abs m = 0 ∨ abs m = 5) :=  -- Distance from (m, 1) to y-axis is 0 or 5
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l4132_413263


namespace NUMINAMATH_CALUDE_cauchy_functional_equation_l4132_413220

-- Define the property of the function
def IsCauchyFunctional (f : ℚ → ℝ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

-- State the theorem
theorem cauchy_functional_equation :
  ∀ f : ℚ → ℝ, IsCauchyFunctional f →
  ∃ a : ℝ, ∀ q : ℚ, f q = a * q :=
sorry

end NUMINAMATH_CALUDE_cauchy_functional_equation_l4132_413220


namespace NUMINAMATH_CALUDE_modular_home_cost_modular_home_cost_proof_l4132_413245

/-- Calculates the total cost of a modular home given specific conditions. -/
theorem modular_home_cost (kitchen_area : ℕ) (kitchen_cost : ℕ) 
  (bathroom_area : ℕ) (bathroom_cost : ℕ) (other_cost_per_sqft : ℕ) 
  (total_area : ℕ) (num_bathrooms : ℕ) : ℕ :=
  let total_module_area := kitchen_area + num_bathrooms * bathroom_area
  let remaining_area := total_area - total_module_area
  kitchen_cost + num_bathrooms * bathroom_cost + remaining_area * other_cost_per_sqft

/-- Proves that the total cost of the specified modular home is $174,000. -/
theorem modular_home_cost_proof : 
  modular_home_cost 400 20000 150 12000 100 2000 2 = 174000 := by
  sorry

end NUMINAMATH_CALUDE_modular_home_cost_modular_home_cost_proof_l4132_413245


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l4132_413219

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b) + b / (6 * c) + c / (9 * a)) = 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l4132_413219


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4132_413200

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + 2 * x^4 + x + 15) - (x^6 + 4 * x^5 + x^4 - x^3 + 20) =
  x^6 - x^5 + x^4 + x^3 + x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4132_413200


namespace NUMINAMATH_CALUDE_line_up_ways_l4132_413227

def number_of_people : ℕ := 5

theorem line_up_ways (n : ℕ) (h : n = number_of_people) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 :=
sorry

end NUMINAMATH_CALUDE_line_up_ways_l4132_413227


namespace NUMINAMATH_CALUDE_inequality_implies_absolute_value_order_l4132_413256

theorem inequality_implies_absolute_value_order 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h : a^2 / (b^2 + c^2) < b^2 / (c^2 + a^2) ∧ b^2 / (c^2 + a^2) < c^2 / (a^2 + b^2)) : 
  |a| < |b| ∧ |b| < |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_absolute_value_order_l4132_413256


namespace NUMINAMATH_CALUDE_rectangle_roots_product_l4132_413247

/-- Given a rectangle ABCD with perimeter 12 and area 5, where the lengths of AB and BC
    are the roots of the equation x^2 + mx + n = 0, prove that mn = -30. -/
theorem rectangle_roots_product (m n : ℝ) : 
  ∃ (AB BC : ℝ),
    AB > 0 ∧ BC > 0 ∧
    AB + BC = 6 ∧
    AB * BC = 5 ∧
    AB^2 + m*AB + n = 0 ∧
    BC^2 + m*BC + n = 0 →
    m * n = -30 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_roots_product_l4132_413247


namespace NUMINAMATH_CALUDE_bob_corn_harvest_l4132_413273

/-- Calculates the number of bushels of corn harvested given the number of rows, 
    corn stalks per row, and corn stalks per bushel. -/
def corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) : ℕ :=
  (rows * stalks_per_row) / stalks_per_bushel

/-- Proves that Bob's corn harvest yields 50 bushels. -/
theorem bob_corn_harvest : 
  corn_harvest 5 80 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bob_corn_harvest_l4132_413273
