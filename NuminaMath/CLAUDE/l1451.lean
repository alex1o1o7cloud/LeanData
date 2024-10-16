import Mathlib

namespace NUMINAMATH_CALUDE_nine_integer_chords_l1451_145175

/-- A circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- The number of integer-length chords through P -/
def integerChordCount (c : CircleWithPoint) : ℕ :=
  sorry

theorem nine_integer_chords 
  (c : CircleWithPoint) 
  (h1 : c.radius = 20) 
  (h2 : c.distanceFromCenter = 12) : 
  integerChordCount c = 9 := by sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l1451_145175


namespace NUMINAMATH_CALUDE_smallest_N_property_l1451_145167

/-- The smallest natural number N such that N × 999 consists entirely of the digit seven in its decimal representation -/
def smallest_N : ℕ := 778556334111889667445223

/-- Predicate to check if a natural number consists entirely of the digit seven -/
def all_sevens (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

theorem smallest_N_property :
  (all_sevens (smallest_N * 999)) ∧
  (∀ m : ℕ, m < smallest_N → ¬(all_sevens (m * 999))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_property_l1451_145167


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1451_145151

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 4*x + 4 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1451_145151


namespace NUMINAMATH_CALUDE_container_capacity_l1451_145183

theorem container_capacity : ∀ (C : ℝ),
  (C > 0) →  -- Ensure the capacity is positive
  (0.35 * C + 48 = 0.75 * C) →
  C = 120 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l1451_145183


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1451_145170

theorem fraction_sum_equals_decimal : (1 : ℚ) / 20 + 2 / 10 + 4 / 40 = (35 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1451_145170


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_23_l1451_145112

theorem sum_of_divisors_of_23 (h : Nat.Prime 23) : (Finset.sum (Nat.divisors 23) id) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_23_l1451_145112


namespace NUMINAMATH_CALUDE_car_lot_total_l1451_145184

theorem car_lot_total (air_bags : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : air_bags = 45)
  (h2 : power_windows = 30)
  (h3 : both = 12)
  (h4 : neither = 2) :
  air_bags + power_windows - both + neither = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_lot_total_l1451_145184


namespace NUMINAMATH_CALUDE_calculate_sales_professionals_l1451_145105

/-- Calculates the number of sales professionals needed to sell a given number of cars
    over a specified period, with each professional selling a fixed number of cars per month. -/
theorem calculate_sales_professionals
  (total_cars : ℕ)
  (cars_per_salesperson_per_month : ℕ)
  (months_to_sell_all : ℕ)
  (h_total_cars : total_cars = 500)
  (h_cars_per_salesperson : cars_per_salesperson_per_month = 10)
  (h_months_to_sell : months_to_sell_all = 5)
  : (total_cars / months_to_sell_all) / cars_per_salesperson_per_month = 10 := by
  sorry

#check calculate_sales_professionals

end NUMINAMATH_CALUDE_calculate_sales_professionals_l1451_145105


namespace NUMINAMATH_CALUDE_set_intersection_empty_l1451_145137

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- State the theorem
theorem set_intersection_empty (a : ℝ) : 
  (A a ∩ B = ∅) ↔ ((1/2 ≤ a ∧ a ≤ 2) ∨ a > 3) := by sorry

end NUMINAMATH_CALUDE_set_intersection_empty_l1451_145137


namespace NUMINAMATH_CALUDE_max_d_value_l1451_145126

/-- Represents a 7-digit number of the form 5d5,22e1 -/
def SevenDigitNumber (d e : Nat) : Nat :=
  5000000 + d * 100000 + 500000 + 22000 + e * 10 + 1

/-- Checks if a number is divisible by 33 -/
def isDivisibleBy33 (n : Nat) : Prop :=
  n % 33 = 0

/-- Checks if a natural number is a single digit (0-9) -/
def isSingleDigit (n : Nat) : Prop :=
  n ≤ 9

theorem max_d_value :
  ∀ d e : Nat,
    isSingleDigit d →
    isSingleDigit e →
    isDivisibleBy33 (SevenDigitNumber d e) →
    d ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l1451_145126


namespace NUMINAMATH_CALUDE_max_distance_for_A_l1451_145135

/-- Represents a member of the expedition team -/
structure Member where
  name : String
  supplies : Nat

/-- Represents the expedition team -/
structure Team where
  members : List Member
  daily_distance : Nat

/-- Calculates the maximum distance a member can travel -/
def max_distance (team : Team) : Nat :=
  sorry

/-- Main theorem: The maximum distance A can travel is 900 kilometers -/
theorem max_distance_for_A (team : Team) :
  team.members.length = 3 ∧
  team.members.all (λ m => m.supplies = 36) ∧
  team.daily_distance = 30 →
  max_distance team = 900 :=
sorry

end NUMINAMATH_CALUDE_max_distance_for_A_l1451_145135


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_5985_l1451_145174

theorem largest_prime_factor_of_5985 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 5985 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 5985 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_5985_l1451_145174


namespace NUMINAMATH_CALUDE_min_value_sum_l1451_145121

theorem min_value_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + a*b + a*c + b*c = 4) :
  ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + x*y + x*z + y*z = 4 →
  2*a + b + c ≤ 2*x + y + z ∧ 2*a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1451_145121


namespace NUMINAMATH_CALUDE_blue_paint_cans_l1451_145127

def blue_to_green_ratio : ℚ := 4 / 3
def total_cans : ℕ := 35

theorem blue_paint_cans : ℕ := by
  -- The number of cans of blue paint is 20
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l1451_145127


namespace NUMINAMATH_CALUDE_min_value_theorem_l1451_145162

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min_val : ℝ), min_val = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 →
    1/(x-1) + 9/(y-1) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1451_145162


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1451_145154

/-- Proposition p: x = 1 and y = 1 -/
def p (x y : ℝ) : Prop := x = 1 ∧ y = 1

/-- Proposition q: x + y = 2 -/
def q (x y : ℝ) : Prop := x + y = 2

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary :
  (∀ x y : ℝ, p x y → q x y) ∧
  (∃ x y : ℝ, q x y ∧ ¬p x y) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1451_145154


namespace NUMINAMATH_CALUDE_binomial_linear_transform_l1451_145147

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  (p_nonneg : 0 ≤ p)
  (p_le_one : p ≤ 1)

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV n p) : ℝ :=
  n * p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ :=
  n * p * (1 - p)

/-- Theorem: Expected value and variance of η = 5ξ, where ξ ~ B(5, 0.5) -/
theorem binomial_linear_transform :
  ∀ (ξ : BinomialRV 5 (1/2)) (η : ℝ),
  η = 5 * (expected_value ξ) →
  expected_value ξ = 5/2 ∧
  variance ξ = 5/4 ∧
  η = 25/2 ∧
  25 * (variance ξ) = 125/4 :=
sorry

end NUMINAMATH_CALUDE_binomial_linear_transform_l1451_145147


namespace NUMINAMATH_CALUDE_provisions_problem_l1451_145182

/-- The number of days the provisions last for the initial group -/
def initial_days : ℕ := 20

/-- The number of additional men that join the group -/
def additional_men : ℕ := 200

/-- The number of days the provisions last after the additional men join -/
def final_days : ℕ := 16

/-- The initial number of men in the group -/
def initial_men : ℕ := 800

theorem provisions_problem :
  initial_men * initial_days = (initial_men + additional_men) * final_days :=
by sorry

end NUMINAMATH_CALUDE_provisions_problem_l1451_145182


namespace NUMINAMATH_CALUDE_division_multiplication_error_percentage_l1451_145145

theorem division_multiplication_error_percentage (x : ℝ) (h : x > 0) :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.5 ∧
  (|(x / 8 - 8 * x)| / (8 * x)) * 100 = 98 + ε := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_error_percentage_l1451_145145


namespace NUMINAMATH_CALUDE_ellipse_properties_l1451_145198

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 9 * x^2 + y^2 = 81

-- Define the major axis length
def major_axis_length : ℝ := 18

-- Define the foci coordinates
def foci_coordinates : Set (ℝ × ℝ) := {(0, 6*Real.sqrt 2), (0, -6*Real.sqrt 2)}

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 36

-- Theorem statement
theorem ellipse_properties :
  (∀ x y, ellipse x y → 
    (major_axis_length = 18 ∧ 
     (x, y) ∈ foci_coordinates → 
     (x = 0 ∧ (y = 6*Real.sqrt 2 ∨ y = -6*Real.sqrt 2)))) ∧
  (∀ x y, hyperbola x y → 
    (∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ 
                  c = 6*Real.sqrt 2 ∧ 
                  c/a = Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1451_145198


namespace NUMINAMATH_CALUDE_game_lives_proof_l1451_145186

/-- Calculates the total number of lives for all players in a game --/
def totalLives (initialPlayers newPlayers livesPerPlayer : ℕ) : ℕ :=
  (initialPlayers + newPlayers) * livesPerPlayer

/-- Proves that the total number of lives is 24 given the specified conditions --/
theorem game_lives_proof :
  let initialPlayers : ℕ := 2
  let newPlayers : ℕ := 2
  let livesPerPlayer : ℕ := 6
  totalLives initialPlayers newPlayers livesPerPlayer = 24 := by
  sorry


end NUMINAMATH_CALUDE_game_lives_proof_l1451_145186


namespace NUMINAMATH_CALUDE_square_free_divisibility_l1451_145153

theorem square_free_divisibility (n : ℕ) (h1 : n > 1) (h2 : Squarefree n) :
  ∃ (p m : ℕ), Prime p ∧ p ∣ n ∧ n ∣ p^2 + p * m^p :=
sorry

end NUMINAMATH_CALUDE_square_free_divisibility_l1451_145153


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_main_theorem_l1451_145101

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + x - 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 1

-- Theorem statement
theorem tangent_line_perpendicular (a : ℝ) : 
  (f' a 1 = 2) → a = 1 := by
  sorry

-- Main theorem
theorem main_theorem (a : ℝ) : 
  (∃ (k : ℝ), f' a 1 = k ∧ k * (-1/2) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_main_theorem_l1451_145101


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1451_145188

theorem quadratic_factorization (c d : ℕ) (h1 : c > d) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - c)*(x - d)) : c - 2*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1451_145188


namespace NUMINAMATH_CALUDE_paper_pallet_ratio_l1451_145197

theorem paper_pallet_ratio (total : ℕ) (towels tissues cups plates : ℕ) : 
  total = 20 → 
  towels = total / 2 → 
  tissues = total / 4 → 
  cups = 1 → 
  plates = total - (towels + tissues + cups) → 
  (plates : ℚ) / total = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_paper_pallet_ratio_l1451_145197


namespace NUMINAMATH_CALUDE_percentage_subtracted_l1451_145128

theorem percentage_subtracted (a : ℝ) (h : ∃ p : ℝ, a - p * a = 0.97 * a) : 
  ∃ p : ℝ, p = 0.03 ∧ a - p * a = 0.97 * a :=
sorry

end NUMINAMATH_CALUDE_percentage_subtracted_l1451_145128


namespace NUMINAMATH_CALUDE_day2_to_day1_rain_ratio_l1451_145165

/-- Represents the rainfall data and conditions for a 4-day storm --/
structure RainfallData where
  capacity : ℝ  -- Capacity in inches
  drainRate : ℝ  -- Drain rate in inches per day
  day1Rain : ℝ  -- Rainfall on day 1 in inches
  day3Increase : ℝ  -- Percentage increase of day 3 rain compared to day 2
  day4Rain : ℝ  -- Rainfall on day 4 in inches

/-- Theorem stating the ratio of day 2 rain to day 1 rain --/
theorem day2_to_day1_rain_ratio (data : RainfallData) 
  (h1 : data.capacity = 72) -- 6 feet = 72 inches
  (h2 : data.drainRate = 3)
  (h3 : data.day1Rain = 10)
  (h4 : data.day3Increase = 1.5) -- 50% more
  (h5 : data.day4Rain = 21) :
  ∃ (x : ℝ), x = 2 ∧ 
    data.day1Rain + x * data.day1Rain + data.day3Increase * x * data.day1Rain + data.day4Rain = 
    data.capacity + 3 * data.drainRate := by
  sorry

#check day2_to_day1_rain_ratio

end NUMINAMATH_CALUDE_day2_to_day1_rain_ratio_l1451_145165


namespace NUMINAMATH_CALUDE_f_triple_3_l1451_145155

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_triple_3 : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_triple_3_l1451_145155


namespace NUMINAMATH_CALUDE_equation_has_real_solution_l1451_145120

theorem equation_has_real_solution (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  ∃ x : ℝ, (a * b^x)^(x + 1) = c := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_solution_l1451_145120


namespace NUMINAMATH_CALUDE_square_rhombus_diagonal_distinction_l1451_145144

/-- A quadrilateral with four equal sides -/
structure Rhombus :=
  (side_length : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)

/-- A square is a rhombus with equal diagonals -/
structure Square extends Rhombus :=
  (diagonals_equal : diagonal1 = diagonal2)

/-- Theorem stating that squares have equal diagonals, but rhombuses don't necessarily have this property -/
theorem square_rhombus_diagonal_distinction :
  ∃ (s : Square) (r : Rhombus), s.diagonal1 = s.diagonal2 ∧ r.diagonal1 ≠ r.diagonal2 :=
sorry

end NUMINAMATH_CALUDE_square_rhombus_diagonal_distinction_l1451_145144


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1451_145146

theorem polynomial_remainder (x : ℝ) : (x^15 + 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1451_145146


namespace NUMINAMATH_CALUDE_rational_function_decomposition_l1451_145169

theorem rational_function_decomposition :
  ∃ (P Q R : ℝ), 
    (∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
      (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = P/x^2 + (Q*x + R)/(x^2 + 1)) ∧
    P = 3 ∧ Q = -1 ∧ R = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_decomposition_l1451_145169


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1451_145124

theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (students_not_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (18 * total_students) / 100)
  (h2 : students_not_more_than_100 = (75 * (students_more_than_100 + students_not_more_than_100)) / 100) :
  (students_more_than_100 + students_not_more_than_100) * 100 / total_students = 72 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1451_145124


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1451_145133

-- Define the square
def square_area : ℝ := 24

-- Define the rectangle's side ratio
def rectangle_ratio : ℝ := 3

-- Theorem statement
theorem inscribed_rectangle_area :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    y = rectangle_ratio * x ∧
    x * y = 18 ∧
    x^2 + y^2 = square_area := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1451_145133


namespace NUMINAMATH_CALUDE_correct_sample_size_l1451_145104

-- Define the population size
def population_size : ℕ := 5000

-- Define the number of sampled students
def sampled_students : ℕ := 450

-- Define what sample size means in this context
def sample_size (n : ℕ) : Prop := n = sampled_students

-- Theorem stating that the sample size is 450
theorem correct_sample_size : sample_size 450 := by sorry

end NUMINAMATH_CALUDE_correct_sample_size_l1451_145104


namespace NUMINAMATH_CALUDE_acidic_solution_concentration_l1451_145148

/-- Proves that the initial volume of a 40% acidic solution is 27 liters
    when it becomes 60% acidic after removing 9 liters of water. -/
theorem acidic_solution_concentration (initial_volume : ℝ) : 
  initial_volume > 0 →
  (0.4 * initial_volume) / (initial_volume - 9) = 0.6 →
  initial_volume = 27 := by
  sorry

end NUMINAMATH_CALUDE_acidic_solution_concentration_l1451_145148


namespace NUMINAMATH_CALUDE_population_growth_l1451_145157

theorem population_growth (P : ℝ) : 
  P * 1.1 * 1.2 = 1320 → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l1451_145157


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1451_145171

/-- A parabola is defined by the equation x² = 8y -/
def is_parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  is_parabola x y → f = (0, 2)

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry -/
def is_directrix (y : ℝ) : Prop :=
  ∀ x, is_parabola x y → y = -2

/-- Theorem: For the parabola x² = 8y, the focus is at (0, 2) and the directrix is y = -2 -/
theorem parabola_focus_and_directrix :
  (∀ x y, is_focus (0, 2) x y) ∧ is_directrix (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1451_145171


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1451_145142

theorem contrapositive_equivalence (a b : ℝ) :
  (ab = 0 → a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0 → ab ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1451_145142


namespace NUMINAMATH_CALUDE_quadratic_condition_l1451_145179

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a given equation is quadratic --/
def isQuadratic (eq : QuadraticEquation) : Prop :=
  eq.a ≠ 0

/-- The equation mx^2 + 3x - 4 = 3x^2 rearranged to standard form --/
def equationOfInterest (m : ℝ) : QuadraticEquation :=
  ⟨m - 3, 3, -4⟩

/-- Theorem stating that for the equation to be quadratic, m must not equal 3 --/
theorem quadratic_condition (m : ℝ) :
  isQuadratic (equationOfInterest m) ↔ m ≠ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l1451_145179


namespace NUMINAMATH_CALUDE_new_recipe_water_amount_l1451_145187

/-- Represents a recipe ratio --/
structure RecipeRatio :=
  (flour : ℕ)
  (water : ℕ)
  (sugar : ℕ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  ⟨8, 4, 3⟩

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  ⟨4, 1, 3⟩

/-- Amount of sugar in the new recipe (in cups) --/
def new_sugar_amount : ℕ := 6

/-- Calculates the amount of water in the new recipe --/
def calculate_water_amount (r : RecipeRatio) (sugar_amount : ℕ) : ℚ :=
  (r.water : ℚ) * sugar_amount / r.sugar

/-- Theorem stating that the new recipe calls for 2 cups of water --/
theorem new_recipe_water_amount :
  calculate_water_amount new_ratio new_sugar_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_recipe_water_amount_l1451_145187


namespace NUMINAMATH_CALUDE_sum_1_to_12_mod_9_l1451_145181

theorem sum_1_to_12_mod_9 : (List.range 12).sum % 9 = 6 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_12_mod_9_l1451_145181


namespace NUMINAMATH_CALUDE_linear_function_m_greater_than_one_l1451_145108

/-- A linear function y = (m+1)x + (m-1) whose graph passes through the first, second, and third quadrants -/
structure LinearFunction (m : ℝ) :=
  (passes_through_quadrants : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (x₂ < 0 ∧ y₂ > 0) ∧  -- Second quadrant
    (x₃ < 0 ∧ y₃ < 0) ∧  -- Third quadrant
    y₁ = (m + 1) * x₁ + (m - 1) ∧
    y₂ = (m + 1) * x₂ + (m - 1) ∧
    y₃ = (m + 1) * x₃ + (m - 1))

/-- Theorem: If a linear function y = (m+1)x + (m-1) has a graph that passes through
    the first, second, and third quadrants, then m > 1 -/
theorem linear_function_m_greater_than_one (m : ℝ) (f : LinearFunction m) : m > 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_m_greater_than_one_l1451_145108


namespace NUMINAMATH_CALUDE_ladder_problem_l1451_145199

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : ℝ, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1451_145199


namespace NUMINAMATH_CALUDE_complex_number_problem_l1451_145111

theorem complex_number_problem : ∃ (z : ℂ), 
  z.im = (3 * Complex.I).re ∧ 
  z.re = (-3 + Complex.I).im ∧ 
  z = 3 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1451_145111


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1451_145102

/-- A geometric sequence with a_2 = 2 and a_4 = 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 2 = 2 ∧ a 4 = 4

/-- In a geometric sequence with a_2 = 2 and a_4 = 4, a_6 = 8 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h : geometric_sequence a) : a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1451_145102


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1451_145118

/-- Represents the time (in hours) it takes to fill a cistern when three pipes are opened simultaneously. -/
def fill_time (rate_A rate_B rate_C : ℚ) : ℚ :=
  1 / (rate_A + rate_B + rate_C)

/-- Theorem stating that given the specific fill/empty rates of pipes A, B, and C,
    the cistern will be filled in 12 hours when all pipes are opened simultaneously. -/
theorem cistern_fill_time :
  fill_time (1/10) (1/15) (-1/12) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1451_145118


namespace NUMINAMATH_CALUDE_problem_statement_l1451_145194

theorem problem_statement :
  (∀ a : ℝ, a < (3/2) → 2*a + 4/(2*a - 3) + 3 ≤ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3*y = 2*x*y → x + 3*y ≥ 6) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1451_145194


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1451_145119

theorem fraction_equation_solution :
  ∃! x : ℚ, (x - 4) / (x + 3) = (x + 2) / (x - 1) ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1451_145119


namespace NUMINAMATH_CALUDE_initial_carrots_count_l1451_145116

/- Given conditions -/
def total_weight : ℝ := 5.94 * 1000  -- in grams
def removed_carrots : ℕ := 3
def avg_weight_remaining : ℝ := 200  -- in grams
def avg_weight_removed : ℝ := 180    -- in grams

/- Theorem to prove -/
theorem initial_carrots_count : 
  ∃ n : ℕ, 
    (n : ℝ) * avg_weight_remaining - removed_carrots * (avg_weight_remaining - avg_weight_removed) = total_weight ∧ 
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_carrots_count_l1451_145116


namespace NUMINAMATH_CALUDE_count_specific_divisors_l1451_145158

/-- The number of positive integer divisors of 2016^2016 that are divisible by exactly 2016 positive integers -/
def divisors_with_2016_divisors : ℕ :=
  let base := 2016
  let exponent := 2016
  let target_divisors := 2016
  -- Definition of the function to count the divisors
  sorry

/-- The main theorem stating that the number of such divisors is 126 -/
theorem count_specific_divisors :
  divisors_with_2016_divisors = 126 := by sorry

end NUMINAMATH_CALUDE_count_specific_divisors_l1451_145158


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1451_145149

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.4 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l1451_145149


namespace NUMINAMATH_CALUDE_square_area_is_56_l1451_145125

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 24 * x + 8 * y + 36

-- Define the property that the circle is inscribed in a square with sides parallel to axes
def inscribed_in_square (center_x center_y radius : ℝ) : Prop :=
  ∃ (side_length : ℝ), side_length = 2 * radius

-- Theorem statement
theorem square_area_is_56 :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    inscribed_in_square center_x center_y radius ∧
    4 * radius^2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_56_l1451_145125


namespace NUMINAMATH_CALUDE_anthony_pencils_l1451_145156

theorem anthony_pencils (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 245 → received = 758 → total = initial + received → total = 1003 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_l1451_145156


namespace NUMINAMATH_CALUDE_average_salary_calculation_l1451_145132

/-- Average salary calculation problem -/
theorem average_salary_calculation (n : ℕ) 
  (avg_all : ℕ) 
  (avg_feb_may : ℕ) 
  (salary_may : ℕ) 
  (salary_jan : ℕ) 
  (h1 : avg_all = 8000)
  (h2 : avg_feb_may = 8700)
  (h3 : salary_may = 6500)
  (h4 : salary_jan = 3700) :
  (salary_jan + (4 * avg_feb_may - salary_may)) / 4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_calculation_l1451_145132


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1451_145164

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (10 * bowling_ball_weight = 4 * canoe_weight) →
    (canoe_weight = 35) →
    (bowling_ball_weight = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1451_145164


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1451_145173

theorem power_tower_mod_500 : 2^(2^(2^2)) % 500 = 536 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1451_145173


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1451_145189

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 5

-- State the theorem
theorem quadratic_minimum :
  ∃ (x_min : ℝ), (∀ (x : ℝ), f x ≥ f x_min) ∧ (x_min = 2) ∧ (f x_min = 13) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1451_145189


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l1451_145134

theorem quadratic_inequalities :
  (∀ x : ℝ, 2 * x^2 + x + 1 > 0) ∧
  (∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x + 2 > 0 ↔ -1/2 < x ∧ x < 2) ∧ a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l1451_145134


namespace NUMINAMATH_CALUDE_butanoic_acid_nine_moles_weight_l1451_145117

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in Butanoic acid -/
def carbon_count : ℕ := 4

/-- The number of Hydrogen atoms in Butanoic acid -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in Butanoic acid -/
def oxygen_count : ℕ := 2

/-- The number of moles of Butanoic acid -/
def moles : ℝ := 9

/-- The molecular weight of Butanoic acid in g/mol -/
def butanoic_acid_weight : ℝ := 
  carbon_weight * carbon_count + 
  hydrogen_weight * hydrogen_count + 
  oxygen_weight * oxygen_count

/-- Theorem: The molecular weight of 9 moles of Butanoic acid is 792.936 grams -/
theorem butanoic_acid_nine_moles_weight : 
  butanoic_acid_weight * moles = 792.936 := by sorry

end NUMINAMATH_CALUDE_butanoic_acid_nine_moles_weight_l1451_145117


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1451_145110

open Real

/-- The function f(x) = 3^x - 3^(-x) is odd and increasing on ℝ -/
theorem f_odd_and_increasing :
  let f : ℝ → ℝ := fun x ↦ 3^x - 3^(-x)
  (∀ x, f (-x) = -f x) ∧ StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1451_145110


namespace NUMINAMATH_CALUDE_tangent_chord_fixed_point_l1451_145172

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A line represented by two points -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- Determines if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Determines if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Determines if a line is perpendicular to another line -/
def isPerpendicular (l1 l2 : Line) : Prop := sorry

/-- Determines if a point is outside a circle -/
def isOutside (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem tangent_chord_fixed_point 
  (O : ℝ × ℝ) (r : ℝ) (l : Line) (H : ℝ × ℝ) :
  let c : Circle := ⟨O, r⟩
  isOutside H c →
  pointOnLine H l →
  isPerpendicular (Line.mk O H) l →
  ∃ P : ℝ × ℝ, ∀ A : ℝ × ℝ, 
    pointOnLine A l →
    ∃ B C : ℝ × ℝ,
      isTangent (Line.mk A B) c ∧
      isTangent (Line.mk A C) c ∧
      pointOnLine P (Line.mk B C) ∧
      pointOnLine P (Line.mk O H) :=
sorry

end NUMINAMATH_CALUDE_tangent_chord_fixed_point_l1451_145172


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l1451_145115

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

theorem downstream_speed_calculation (s : RowingSpeed)
  (h1 : s.upstream = 25)
  (h2 : s.stillWater = 31) :
  downstreamSpeed s = 37 := by sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l1451_145115


namespace NUMINAMATH_CALUDE_unique_solution_two_and_five_l1451_145168

theorem unique_solution_two_and_five (x : ℝ) : (x - 2) * (x - 5) = 0 ↔ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_two_and_five_l1451_145168


namespace NUMINAMATH_CALUDE_triangle_side_values_l1451_145106

theorem triangle_side_values (y : ℕ+) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 11 ∧ c = y.val ^ 2 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l1451_145106


namespace NUMINAMATH_CALUDE_hearty_beads_count_l1451_145141

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each red package -/
def beads_per_red_package : ℕ := 40

/-- The number of beads in each blue package is twice the number in each red package -/
def beads_per_blue_package : ℕ := 2 * beads_per_red_package

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * beads_per_blue_package + red_packages * beads_per_red_package

theorem hearty_beads_count : total_beads = 440 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l1451_145141


namespace NUMINAMATH_CALUDE_other_bill_value_l1451_145185

/-- Represents the class fund with two types of bills -/
structure ClassFund where
  total_amount : ℕ
  num_other_bills : ℕ
  value_ten_dollar_bill : ℕ

/-- Theorem stating the value of the other type of bills -/
theorem other_bill_value (fund : ClassFund)
  (h1 : fund.total_amount = 120)
  (h2 : fund.num_other_bills = 3)
  (h3 : fund.value_ten_dollar_bill = 10)
  (h4 : 2 * fund.num_other_bills = (fund.total_amount - fund.num_other_bills * (fund.total_amount / fund.num_other_bills)) / fund.value_ten_dollar_bill) :
  fund.total_amount / fund.num_other_bills = 40 := by
sorry

end NUMINAMATH_CALUDE_other_bill_value_l1451_145185


namespace NUMINAMATH_CALUDE_units_digit_of_8421_to_1287_l1451_145180

theorem units_digit_of_8421_to_1287 : ∃ n : ℕ, 8421^1287 = 10 * n + 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_8421_to_1287_l1451_145180


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1451_145100

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  ∀ (circle_area : ℝ → ℝ) (pi : ℝ),
  (∀ r, circle_area r = pi * r^2) →
  circle_area 5 = 25 * pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1451_145100


namespace NUMINAMATH_CALUDE_total_salary_is_7600_l1451_145129

/-- Represents the weekly working hours for each employee -/
structure WeeklyHours where
  fiona : ℕ
  john : ℕ
  jeremy : ℕ

/-- Represents the hourly wage -/
def hourlyWage : ℚ := 20

/-- Represents the number of weeks in a month -/
def weeksPerMonth : ℕ := 4

/-- Calculates the monthly salary for an employee -/
def monthlySalary (hours : ℕ) : ℚ :=
  hours * hourlyWage * weeksPerMonth

/-- Calculates the total monthly expenditure on salaries -/
def totalMonthlyExpenditure (hours : WeeklyHours) : ℚ :=
  monthlySalary hours.fiona + monthlySalary hours.john + monthlySalary hours.jeremy

/-- Theorem stating that the total monthly expenditure on salaries is $7600 -/
theorem total_salary_is_7600 (hours : WeeklyHours)
    (h1 : hours.fiona = 40)
    (h2 : hours.john = 30)
    (h3 : hours.jeremy = 25) :
    totalMonthlyExpenditure hours = 7600 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_is_7600_l1451_145129


namespace NUMINAMATH_CALUDE_line_and_circle_tangent_l1451_145139

-- Define the lines and circle
def l₁ (x y : ℝ) : Prop := 2 * x - y = 1
def l₂ (x y : ℝ) : Prop := x + 2 * y = 3
def l₃ (x y : ℝ) : Prop := x - y + 1 = 0
def C (x y a : ℝ) : Prop := (x - a)^2 + y^2 = 8

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Main theorem
theorem line_and_circle_tangent :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = P) →  -- P is the intersection of l₁ and l₂
  (∀ x y : ℝ, l x y → l₃ (x + 1) (y + 1)) →  -- l is perpendicular to l₃
  ∃ a : ℝ, a > 0 ∧
    (∀ x y : ℝ, l x y → 
      (∃ t : ℝ, C x y a ∧ 
        (∀ x' y', C x' y' a → (x - x')^2 + (y - y')^2 ≥ t^2) ∧
        (∃ x' y', C x' y' a ∧ (x - x')^2 + (y - y')^2 = t^2))) →
  (∀ x y : ℝ, l x y ↔ x + y - 2 = 0) ∧ a = 6 :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_tangent_l1451_145139


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l1451_145195

/-- Given a cylinder with original volume of 15 cubic feet, proves that tripling its radius and halving its height results in a new volume of 67.5 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) (h3 : π * r^2 * h = 15) :
  π * (3*r)^2 * (h/2) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l1451_145195


namespace NUMINAMATH_CALUDE_zoo_visitors_l1451_145190

theorem zoo_visitors (visitors_saturday : ℕ) (visitors_that_day : ℕ) : 
  visitors_saturday = 3750 →
  visitors_saturday = 3 * visitors_that_day →
  visitors_that_day = 1250 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1451_145190


namespace NUMINAMATH_CALUDE_cube_of_fraction_l1451_145193

theorem cube_of_fraction (x y : ℝ) : 
  (-2/3 * x * y^2)^3 = -8/27 * x^3 * y^6 := by
sorry

end NUMINAMATH_CALUDE_cube_of_fraction_l1451_145193


namespace NUMINAMATH_CALUDE_carpet_dimensions_l1451_145122

/-- Represents a rectangular carpet -/
structure Carpet where
  length : ℕ
  width : ℕ

/-- Represents a rectangular room -/
structure Room where
  length : ℕ
  width : ℕ

/-- Check if a carpet fits perfectly in a room -/
def fits_perfectly (c : Carpet) (r : Room) : Prop :=
  c.length * c.length + c.width * c.width = r.length * r.length + r.width * r.width

/-- The main theorem -/
theorem carpet_dimensions (c : Carpet) (r1 r2 : Room) (h1 : r1.length = r2.length)
  (h2 : r1.width = 38) (h3 : r2.width = 50) (h4 : fits_perfectly c r1) (h5 : fits_perfectly c r2) :
  c.length = 25 ∧ c.width = 50 := by
  sorry

#check carpet_dimensions

end NUMINAMATH_CALUDE_carpet_dimensions_l1451_145122


namespace NUMINAMATH_CALUDE_both_activities_count_l1451_145113

/-- Represents a group of people with preferences for reading books and listening to songs -/
structure GroupPreferences where
  total : ℕ
  book_lovers : ℕ
  song_lovers : ℕ
  both_lovers : ℕ

/-- The principle of inclusion-exclusion for two sets -/
def inclusion_exclusion (g : GroupPreferences) : Prop :=
  g.total = g.book_lovers + g.song_lovers - g.both_lovers

/-- Theorem stating the number of people who like both activities -/
theorem both_activities_count (g : GroupPreferences) 
  (h1 : g.total = 100)
  (h2 : g.book_lovers = 50)
  (h3 : g.song_lovers = 70)
  (h4 : inclusion_exclusion g) : 
  g.both_lovers = 20 := by
  sorry


end NUMINAMATH_CALUDE_both_activities_count_l1451_145113


namespace NUMINAMATH_CALUDE_interest_rate_difference_l1451_145160

/-- Proves that the difference in interest rates is 3% given the problem conditions -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_difference : ℝ)
  (h_principal : principal = 5000)
  (h_time : time = 2)
  (h_interest_diff : interest_difference = 300)
  : ∃ (r dr : ℝ),
    principal * (r + dr) / 100 * time - principal * r / 100 * time = interest_difference ∧
    dr = 3 := by
  sorry

#check interest_rate_difference

end NUMINAMATH_CALUDE_interest_rate_difference_l1451_145160


namespace NUMINAMATH_CALUDE_unbounded_solution_set_l1451_145114

/-- The set of points (x, y) satisfying the given system of inequalities is unbounded -/
theorem unbounded_solution_set :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ 
      ((abs x + x)^2 + (abs y + y)^2 ≤ 4 ∧ 3*y + x ≤ 0)) ∧
    ¬(∃ (M : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → ‖p‖ ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_unbounded_solution_set_l1451_145114


namespace NUMINAMATH_CALUDE_daisy_germination_rate_l1451_145150

/-- Proves that the germination rate of daisy seeds is 60% given the problem conditions --/
theorem daisy_germination_rate :
  let daisy_seeds : ℕ := 25
  let sunflower_seeds : ℕ := 25
  let sunflower_germination_rate : ℚ := 80 / 100
  let flower_production_rate : ℚ := 80 / 100
  let total_flowering_plants : ℕ := 28
  ∃ (daisy_germination_rate : ℚ),
    daisy_germination_rate = 60 / 100 ∧
    (↑daisy_seeds * daisy_germination_rate * flower_production_rate +
     ↑sunflower_seeds * sunflower_germination_rate * flower_production_rate : ℚ) = total_flowering_plants :=
by sorry

end NUMINAMATH_CALUDE_daisy_germination_rate_l1451_145150


namespace NUMINAMATH_CALUDE_tangent_line_equations_l1451_145103

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 2)*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 2)

-- Theorem statement
theorem tangent_line_equations (a : ℝ) 
  (h1 : ∀ x, f' a x = f' a (-x)) -- f' is an even function
  : (∃ x₀ y₀, x₀ ≠ 1 ∧ f a x₀ = y₀ ∧ 
    (y₀ - (-2)) / (x₀ - 1) = f' a x₀ ∧
    (2 * x + y = 0 ∨ 19 * x - 4 * y - 27 = 0)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l1451_145103


namespace NUMINAMATH_CALUDE_distribution_schemes_l1451_145152

def math_teachers : ℕ := 3
def chinese_teachers : ℕ := 6
def schools : ℕ := 3
def math_teachers_per_school : ℕ := 1
def chinese_teachers_per_school : ℕ := 2

theorem distribution_schemes :
  (math_teachers.factorial) *
  (chinese_teachers.choose chinese_teachers_per_school) *
  ((chinese_teachers - chinese_teachers_per_school).choose chinese_teachers_per_school) = 540 :=
sorry

end NUMINAMATH_CALUDE_distribution_schemes_l1451_145152


namespace NUMINAMATH_CALUDE_intersection_point_parallel_through_point_perpendicular_with_y_intercept_l1451_145196

-- Define the lines l₁ and l₂
def l₁ (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l₂ (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Scenario 1: l₁ and l₂ intersect at point P(m, 1)
theorem intersection_point (m n : ℝ) : 
  (l₁ m n m 1 ∧ l₂ m m 1) → (m = 1/3 ∧ n = -73/9) := by sorry

-- Scenario 2: l₁ is parallel to l₂ and passes through (3, -1)
theorem parallel_through_point (m n : ℝ) :
  (∀ x y : ℝ, l₁ m n x y ↔ l₂ m x y) ∧ l₁ m n 3 (-1) → 
  ((m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20)) := by sorry

-- Scenario 3: l₁ is perpendicular to l₂ and y-intercept of l₁ is -1
theorem perpendicular_with_y_intercept (m n : ℝ) :
  (∀ x y : ℝ, l₁ m n x y → l₂ m x y → m * m = -1) ∧ l₁ m n 0 (-1) →
  (m = 0 ∧ n = 8) := by sorry

end NUMINAMATH_CALUDE_intersection_point_parallel_through_point_perpendicular_with_y_intercept_l1451_145196


namespace NUMINAMATH_CALUDE_inequality_theorem_l1451_145107

theorem inequality_theorem (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : 1/p + 1/q^2 = 1) : 
  1/(p*(p+2)) + 1/(q*(q+2)) ≥ (21*Real.sqrt 21 - 71)/80 ∧
  (1/(p*(p+2)) + 1/(q*(q+2)) = (21*Real.sqrt 21 - 71)/80 ↔ 
    p = 2 + 2*Real.sqrt (7/3) ∧ q = (Real.sqrt 21 + 1)/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1451_145107


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1451_145131

theorem smallest_factorization_coefficient : ∃ (r s : ℕ+), 
  (r : ℤ) * s = 1620 ∧ 
  r + s = 84 ∧ 
  (∀ (r' s' : ℕ+), (r' : ℤ) * s' = 1620 → r' + s' ≥ 84) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1451_145131


namespace NUMINAMATH_CALUDE_power_comparison_l1451_145192

theorem power_comparison : 2^51 > 4^25 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l1451_145192


namespace NUMINAMATH_CALUDE_banana_apple_sales_l1451_145143

/-- Proves that if the revenue from selling apples and bananas with reversed prices
    is $1 more than the revenue with original prices, then the number of bananas
    sold is 10 more than the number of apples sold. -/
theorem banana_apple_sales
  (apple_price : ℚ)
  (banana_price : ℚ)
  (apple_count : ℕ)
  (banana_count : ℕ)
  (h1 : apple_price = 0.5)
  (h2 : banana_price = 0.4)
  (h3 : banana_price * apple_count + apple_price * banana_count =
        apple_price * apple_count + banana_price * banana_count + 1) :
  banana_count = apple_count + 10 := by
sorry

end NUMINAMATH_CALUDE_banana_apple_sales_l1451_145143


namespace NUMINAMATH_CALUDE_range_of_b_l1451_145140

/-- A region in the xy-plane defined by y ≤ 3x + b -/
def region (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ 3 * p.1 + b}

/-- The theorem stating the range of b given the conditions -/
theorem range_of_b :
  ∀ b : ℝ,
  (¬ ((3, 4) ∈ region b) ∧ ((4, 4) ∈ region b)) ↔
  (-8 ≤ b ∧ b < -5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1451_145140


namespace NUMINAMATH_CALUDE_number_problem_l1451_145109

theorem number_problem (N : ℝ) : 
  1.15 * ((1/4) * (1/3) * (2/5) * N) = 23 → 0.5 * N = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1451_145109


namespace NUMINAMATH_CALUDE_jose_share_correct_l1451_145159

/-- Calculates an investor's share of the profit based on their investment amount, duration, and the total profit, given the investments and durations of all participants. -/
def calculate_share (tom_investment : ℕ) (tom_duration : ℕ) (jose_investment : ℕ) (jose_duration : ℕ) (maria_investment : ℕ) (maria_duration : ℕ) (total_profit : ℕ) : ℚ :=
  let total_capital_months : ℕ := tom_investment * tom_duration + jose_investment * jose_duration + maria_investment * maria_duration
  (jose_investment * jose_duration : ℚ) / total_capital_months * total_profit

/-- Proves that Jose's share of the profit is correct given the specific investments and durations. -/
theorem jose_share_correct (total_profit : ℕ) : 
  calculate_share 30000 12 45000 10 60000 8 total_profit = 
  (45000 * 10 : ℚ) / (30000 * 12 + 45000 * 10 + 60000 * 8) * total_profit :=
by sorry

end NUMINAMATH_CALUDE_jose_share_correct_l1451_145159


namespace NUMINAMATH_CALUDE_tens_digit_of_6_to_18_l1451_145178

/-- The tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The theorem stating that the tens digit of 6^18 is 1 -/
theorem tens_digit_of_6_to_18 : tens_digit (6^18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_to_18_l1451_145178


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_equality_l1451_145123

theorem consecutive_squares_sum_equality :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = (n+3)^2 + (n+4)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_equality_l1451_145123


namespace NUMINAMATH_CALUDE_vector_parallel_implies_m_value_l1451_145161

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem vector_parallel_implies_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2, 1 + m)
  let b : ℝ × ℝ := (3, m)
  parallel a b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_m_value_l1451_145161


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l1451_145191

theorem sin_2x_derivative (x : ℝ) : 
  deriv (fun x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l1451_145191


namespace NUMINAMATH_CALUDE_gold_silver_coin_values_l1451_145136

theorem gold_silver_coin_values :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset ℕ, S.card = n ∧
    ∀ x ∈ S, x > 0 ∧
    ∃ y : ℕ, y > 0 ∧ y < 100 ∧
    (100 + x) * (100 - y) = 10000) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_gold_silver_coin_values_l1451_145136


namespace NUMINAMATH_CALUDE_father_son_age_difference_l1451_145138

theorem father_son_age_difference :
  ∀ (f s : ℕ+),
  f * s = 2015 →
  f > s →
  f - s = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l1451_145138


namespace NUMINAMATH_CALUDE_average_player_minutes_is_two_l1451_145166

/-- Represents the highlight film about Patricia's basketball team. -/
structure HighlightFilm where
  /-- Footage duration for each player in seconds -/
  point_guard : ℕ
  shooting_guard : ℕ
  small_forward : ℕ
  power_forward : ℕ
  center : ℕ
  /-- Additional content durations in seconds -/
  game_footage : ℕ
  interviews : ℕ
  opening_closing : ℕ
  /-- Pause duration between segments in seconds -/
  pause_duration : ℕ

/-- Calculates the average number of minutes attributed to each player's footage -/
def averagePlayerMinutes (film : HighlightFilm) : ℚ :=
  let total_player_footage := film.point_guard + film.shooting_guard + film.small_forward + 
                              film.power_forward + film.center
  let total_additional_content := film.game_footage + film.interviews + film.opening_closing
  let total_pause_time := film.pause_duration * 8
  let total_film_time := total_player_footage + total_additional_content + total_pause_time
  (total_player_footage : ℚ) / (5 * 60)

/-- Theorem stating that the average number of minutes attributed to each player's footage is 2 minutes -/
theorem average_player_minutes_is_two (film : HighlightFilm) 
  (h1 : film.point_guard = 130)
  (h2 : film.shooting_guard = 145)
  (h3 : film.small_forward = 85)
  (h4 : film.power_forward = 60)
  (h5 : film.center = 180)
  (h6 : film.game_footage = 120)
  (h7 : film.interviews = 90)
  (h8 : film.opening_closing = 30)
  (h9 : film.pause_duration = 15) :
  averagePlayerMinutes film = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_player_minutes_is_two_l1451_145166


namespace NUMINAMATH_CALUDE_standard_deviation_of_random_variable_l1451_145130

def random_variable (ξ : ℝ → ℝ) : Prop :=
  (ξ 1 = 0.4) ∧ (ξ 3 = 0.1) ∧ (∃ x, ξ 5 = x) ∧ (ξ 1 + ξ 3 + ξ 5 = 1)

def expected_value (ξ : ℝ → ℝ) : ℝ :=
  1 * ξ 1 + 3 * ξ 3 + 5 * ξ 5

def variance (ξ : ℝ → ℝ) : ℝ :=
  (1 - expected_value ξ)^2 * ξ 1 + 
  (3 - expected_value ξ)^2 * ξ 3 + 
  (5 - expected_value ξ)^2 * ξ 5

theorem standard_deviation_of_random_variable (ξ : ℝ → ℝ) :
  random_variable ξ → Real.sqrt (variance ξ) = Real.sqrt 3.56 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_random_variable_l1451_145130


namespace NUMINAMATH_CALUDE_expression_values_l1451_145163

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -3 :=
sorry

end NUMINAMATH_CALUDE_expression_values_l1451_145163


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l1451_145177

theorem decimal_fraction_equality (b : ℕ) : 
  b > 0 ∧ (5 * b + 22 : ℚ) / (7 * b + 15) = 87 / 100 → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l1451_145177


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1451_145176

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 20/21
  let a₃ : ℚ := 100/63
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = (4/7) * (5/3)^(n-1)) →
  r = 5/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1451_145176
