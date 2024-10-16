import Mathlib

namespace NUMINAMATH_CALUDE_optimal_rental_income_l3115_311566

/-- Represents a travel agency's room rental scenario -/
structure RentalScenario where
  initialRooms : ℕ
  initialRate : ℕ
  rateIncrement : ℕ
  occupancyDecrease : ℕ

/-- Calculates the total daily rental income for a given rate increase -/
def totalIncome (scenario : RentalScenario) (rateIncrease : ℕ) : ℕ :=
  let newRate := scenario.initialRate + rateIncrease
  let newOccupancy := scenario.initialRooms - (rateIncrease / scenario.rateIncrement) * scenario.occupancyDecrease
  newRate * newOccupancy

/-- Finds the optimal rate increase to maximize total daily rental income -/
def optimalRateIncrease (scenario : RentalScenario) : ℕ :=
  sorry

/-- Calculates the increase in total daily rental income -/
def incomeIncrease (scenario : RentalScenario) : ℕ :=
  totalIncome scenario (optimalRateIncrease scenario) - totalIncome scenario 0

/-- Theorem stating the optimal rate increase and income increase for the given scenario -/
theorem optimal_rental_income (scenario : RentalScenario) 
  (h1 : scenario.initialRooms = 120)
  (h2 : scenario.initialRate = 50)
  (h3 : scenario.rateIncrement = 5)
  (h4 : scenario.occupancyDecrease = 6) :
  optimalRateIncrease scenario = 25 ∧ incomeIncrease scenario = 750 := by
  sorry

end NUMINAMATH_CALUDE_optimal_rental_income_l3115_311566


namespace NUMINAMATH_CALUDE_train_length_l3115_311580

theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 870)
  (h2 : platform_time = 39)
  (h3 : pole_time = 10) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by sorry

end NUMINAMATH_CALUDE_train_length_l3115_311580


namespace NUMINAMATH_CALUDE_inscribed_rectangle_width_l3115_311501

/-- Given a right-angled triangle with legs a and b, and a rectangle inscribed
    such that its width d satisfies d(d - (a + b)) = 0, prove that d = a + b -/
theorem inscribed_rectangle_width (a b d : ℝ) (h : d * (d - (a + b)) = 0) :
  d = a + b := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_width_l3115_311501


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l3115_311525

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l3115_311525


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3115_311500

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 5)^2 = 36) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3115_311500


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3115_311592

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a with a₁ = 2 and a₃ = 4, prove that a₇ = 16 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 4) : 
  a 7 = 16 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l3115_311592


namespace NUMINAMATH_CALUDE_janet_gained_lives_l3115_311540

/-- Calculates the number of lives Janet gained in a video game level -/
def lives_gained (initial : ℕ) (lost : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost)

/-- Proves that Janet gained 32 lives in the next level -/
theorem janet_gained_lives : lives_gained 38 16 54 = 32 := by
  sorry

end NUMINAMATH_CALUDE_janet_gained_lives_l3115_311540


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l3115_311536

/-- The number of legs a centipede has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid orders for a centipede to put on socks and shoes -/
def valid_orders : ℕ := Nat.factorial total_items / (2 ^ num_legs)

/-- Theorem stating the number of valid orders for a centipede to put on socks and shoes -/
theorem centipede_sock_shoe_orders :
  valid_orders = Nat.factorial total_items / (2 ^ num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l3115_311536


namespace NUMINAMATH_CALUDE_english_alphabet_is_set_l3115_311516

-- Define the type for English alphabet letters
inductive EnglishLetter
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

-- Define the properties of set elements
def isDefinite (x : Type) : Prop := sorry
def isDistinct (x : Type) : Prop := sorry
def isUnordered (x : Type) : Prop := sorry

-- Define what it means to be a valid set
def isValidSet (x : Type) : Prop :=
  isDefinite x ∧ isDistinct x ∧ isUnordered x

-- Theorem stating that the English alphabet forms a set
theorem english_alphabet_is_set :
  isValidSet EnglishLetter :=
sorry

end NUMINAMATH_CALUDE_english_alphabet_is_set_l3115_311516


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3115_311508

def arithmetic_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℚ)
  (h_seq : arithmetic_geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3115_311508


namespace NUMINAMATH_CALUDE_mike_changed_tires_on_ten_cars_l3115_311519

/-- The number of cars Mike changed tires on -/
def num_cars (total_tires num_motorcycles tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  (total_tires - num_motorcycles * tires_per_motorcycle) / tires_per_car

theorem mike_changed_tires_on_ten_cars :
  num_cars 64 12 2 4 = 10 := by sorry

end NUMINAMATH_CALUDE_mike_changed_tires_on_ten_cars_l3115_311519


namespace NUMINAMATH_CALUDE_square_roots_problem_l3115_311559

theorem square_roots_problem (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ (m - 3)^2 = x ∧ (m - 7)^2 = x) :
  (∃ (x : ℝ), x > 0 ∧ (m - 3)^2 = x ∧ (m - 7)^2 = x) →
  (m - 3)^2 = 4 ∧ (m^2 + 2)^(1/3) = 3 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3115_311559


namespace NUMINAMATH_CALUDE_selling_price_when_profit_equals_loss_l3115_311547

/-- The selling price of an article when the profit is equal to the loss -/
def selling_price : ℕ := 57

/-- The cost price of the article -/
def cost_price : ℕ := 50

/-- The price at which the article is sold at a loss -/
def loss_price : ℕ := 43

/-- Theorem: The selling price of the article when the profit is equal to the loss is 57 -/
theorem selling_price_when_profit_equals_loss :
  (selling_price - cost_price = cost_price - loss_price) ∧
  (selling_price = 57) := by
  sorry

end NUMINAMATH_CALUDE_selling_price_when_profit_equals_loss_l3115_311547


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3115_311544

theorem coin_flip_probability (p : ℝ) (h : p = 1 / 2) :
  p * (1 - p) * (1 - p) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3115_311544


namespace NUMINAMATH_CALUDE_cat_kittens_count_l3115_311584

/-- The number of kittens born to a cat, given specific weight conditions -/
def number_of_kittens (weight_two_lightest weight_four_heaviest total_weight : ℕ) : ℕ :=
  2 + 4 + (total_weight - weight_two_lightest - weight_four_heaviest) / ((weight_four_heaviest / 4 + weight_two_lightest / 2) / 2)

/-- Theorem stating that under the given conditions, the cat gave birth to 11 kittens -/
theorem cat_kittens_count :
  number_of_kittens 80 200 500 = 11 :=
by
  sorry

#eval number_of_kittens 80 200 500

end NUMINAMATH_CALUDE_cat_kittens_count_l3115_311584


namespace NUMINAMATH_CALUDE_domain_of_g_l3115_311575

-- Define the function f with domain (-1, 0)
def f : Set ℝ := { x : ℝ | -1 < x ∧ x < 0 }

-- Define the function g(x) = f(2x+1)
def g : Set ℝ := { x : ℝ | (2*x + 1) ∈ f }

-- Theorem statement
theorem domain_of_g : g = { x : ℝ | -1 < x ∧ x < -1/2 } := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l3115_311575


namespace NUMINAMATH_CALUDE_area_triangle_OPQ_l3115_311561

/-- Given complex numbers z₁ and z₂ corresponding to points P and Q on the complex plane,
    with |z₂| = 4 and 4z₁² - 2z₁z₂ + z₂² = 0, prove that the area of triangle OPQ is 2√3. -/
theorem area_triangle_OPQ (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₂ = 4)
  (h₂ : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) :
  (1/2 : ℝ) * Complex.abs z₁ * Complex.abs z₂ * Real.sin (π/3) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_OPQ_l3115_311561


namespace NUMINAMATH_CALUDE_albert_and_allison_marbles_albert_and_allison_marbles_proof_l3115_311505

/-- Proves that Albert and Allison have a total of 136 marbles given the conditions of the problem -/
theorem albert_and_allison_marbles : ℕ → ℕ → ℕ → Prop :=
  fun allison_marbles angela_marbles albert_marbles =>
    allison_marbles = 28 ∧
    angela_marbles = allison_marbles + 8 ∧
    albert_marbles = 3 * angela_marbles →
    albert_marbles + allison_marbles = 136

/-- Proof of the theorem -/
theorem albert_and_allison_marbles_proof :
  ∃ (allison_marbles angela_marbles albert_marbles : ℕ),
    albert_and_allison_marbles allison_marbles angela_marbles albert_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_albert_and_allison_marbles_albert_and_allison_marbles_proof_l3115_311505


namespace NUMINAMATH_CALUDE_largest_covered_range_l3115_311578

def is_monic_quadratic (p : ℤ → ℤ) : Prop :=
  ∃ a b : ℤ, ∀ x, p x = x^2 + a*x + b

def covers_range (p₁ p₂ p₃ : ℤ → ℤ) (n : ℕ) : Prop :=
  ∀ i ∈ Finset.range n, ∃ j ∈ [1, 2, 3], ∃ m : ℤ, 
    (if j = 1 then p₁ else if j = 2 then p₂ else p₃) m = i + 1

theorem largest_covered_range : 
  (∃ p₁ p₂ p₃ : ℤ → ℤ, 
    is_monic_quadratic p₁ ∧ 
    is_monic_quadratic p₂ ∧ 
    is_monic_quadratic p₃ ∧ 
    covers_range p₁ p₂ p₃ 9) ∧ 
  (∀ n > 9, ¬∃ p₁ p₂ p₃ : ℤ → ℤ, 
    is_monic_quadratic p₁ ∧ 
    is_monic_quadratic p₂ ∧ 
    is_monic_quadratic p₃ ∧ 
    covers_range p₁ p₂ p₃ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_covered_range_l3115_311578


namespace NUMINAMATH_CALUDE_expression_evaluation_l3115_311558

theorem expression_evaluation (a b c : ℕ) (ha : a = 3) (hb : b = 2) (hc : c = 4) :
  ((a^b)^a - (b^a)^b) * c = 2660 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3115_311558


namespace NUMINAMATH_CALUDE_lisa_phone_expenses_l3115_311509

/-- Calculate the total cost of Lisa's phone and related expenses after three years -/
theorem lisa_phone_expenses :
  let iphone_cost : ℝ := 1000
  let monthly_contract : ℝ := 200
  let case_cost : ℝ := 0.2 * iphone_cost
  let headphones_cost : ℝ := 0.5 * case_cost
  let charger_cost : ℝ := 60
  let warranty_cost : ℝ := 150
  let discount_rate : ℝ := 0.1
  let years : ℝ := 3

  let discounted_case_cost : ℝ := case_cost * (1 - discount_rate)
  let discounted_headphones_cost : ℝ := headphones_cost * (1 - discount_rate)
  let total_contract_cost : ℝ := monthly_contract * 12 * years

  let total_cost : ℝ := iphone_cost + total_contract_cost + discounted_case_cost + 
                        discounted_headphones_cost + charger_cost + warranty_cost

  total_cost = 8680 := by sorry

end NUMINAMATH_CALUDE_lisa_phone_expenses_l3115_311509


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3115_311543

/-- Given a circle with equation x^2 + y^2 = 1 and a line of symmetry x - y - 2 = 0,
    the equation of the symmetric circle is (x-2)^2 + (y+2)^2 = 1 -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 = 1) →
  (x - y - 2 = 0) →
  (∃ (x' y' : ℝ), (x' - 2)^2 + (y' + 2)^2 = 1 ∧
    (∀ (p q : ℝ), (p - q - 2 = 0) → 
      ((x - p)^2 + (y - q)^2 = (x' - p)^2 + (y' - q)^2))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3115_311543


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3115_311570

theorem square_perimeters_sum (x y : ℝ) (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  4*x + 4*y = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3115_311570


namespace NUMINAMATH_CALUDE_sum_ge_sum_sqrt_products_l3115_311562

theorem sum_ge_sum_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_ge_sum_sqrt_products_l3115_311562


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3115_311556

/-- Proves that the initial percentage of water in a 40-liter mixture is 10%
    given that adding 5 liters of water results in a 20% water mixture. -/
theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_water = 5)
  (h3 : final_water_percentage = 20)
  (h4 : (initial_volume * x / 100 + added_water) / (initial_volume + added_water) * 100 = final_water_percentage) :
  x = 10 := by
  sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l3115_311556


namespace NUMINAMATH_CALUDE_correct_point_satisfies_conditions_l3115_311531

def point_satisfies_conditions (x y : ℝ) : Prop :=
  -2 < x ∧ x < 0 ∧ 2 < y ∧ y < 4

theorem correct_point_satisfies_conditions :
  point_satisfies_conditions (-1) 3 ∧
  ¬ point_satisfies_conditions 1 3 ∧
  ¬ point_satisfies_conditions 1 (-3) ∧
  ¬ point_satisfies_conditions (-3) 1 ∧
  ¬ point_satisfies_conditions 3 (-1) :=
by sorry

end NUMINAMATH_CALUDE_correct_point_satisfies_conditions_l3115_311531


namespace NUMINAMATH_CALUDE_unique_diametric_circle_l3115_311557

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ (i j : Fin 3), i ≠ j → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 
    Real.sqrt ((vertices 0).1 - (vertices 1).1)^2 + ((vertices 0).2 - (vertices 1).2)^2

/-- A circle defined by two points as its diameter -/
structure DiametricCircle (T : EquilateralTriangle) where
  endpoint1 : Fin 3
  endpoint2 : Fin 3
  is_diameter : endpoint1 ≠ endpoint2

/-- The theorem stating that there's only one unique diametric circle for an equilateral triangle -/
theorem unique_diametric_circle (T : EquilateralTriangle) : 
  ∃! (c : DiametricCircle T), True := by sorry

end NUMINAMATH_CALUDE_unique_diametric_circle_l3115_311557


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3115_311555

theorem rectangle_y_value (y : ℝ) : 
  y > 0 → -- y is positive
  (6 - (-2)) * (y - 2) = 64 → -- area of rectangle is 64
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3115_311555


namespace NUMINAMATH_CALUDE_waiter_initial_customers_l3115_311574

/-- Calculates the initial number of customers in a waiter's section --/
def initial_customers (tables : ℕ) (people_per_table : ℕ) (left_customers : ℕ) : ℕ :=
  tables * people_per_table + left_customers

/-- Theorem: The initial number of customers in the waiter's section was 62 --/
theorem waiter_initial_customers :
  initial_customers 5 9 17 = 62 := by
  sorry

end NUMINAMATH_CALUDE_waiter_initial_customers_l3115_311574


namespace NUMINAMATH_CALUDE_condition_relationship_l3115_311594

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3115_311594


namespace NUMINAMATH_CALUDE_nigella_sold_three_houses_l3115_311550

/-- Represents a house with its cost -/
structure House where
  cost : ℝ

/-- Represents a realtor's earnings -/
structure RealtorEarnings where
  baseSalary : ℝ
  commissionRate : ℝ
  totalEarnings : ℝ

def calculateCommission (house : House) (commissionRate : ℝ) : ℝ :=
  house.cost * commissionRate

def nigellaEarnings : RealtorEarnings := {
  baseSalary := 3000
  commissionRate := 0.02
  totalEarnings := 8000
}

def houseA : House := { cost := 60000 }
def houseB : House := { cost := 3 * houseA.cost }
def houseC : House := { cost := 2 * houseA.cost - 110000 }

theorem nigella_sold_three_houses :
  let commission := calculateCommission houseA nigellaEarnings.commissionRate +
                    calculateCommission houseB nigellaEarnings.commissionRate +
                    calculateCommission houseC nigellaEarnings.commissionRate
  nigellaEarnings.totalEarnings = nigellaEarnings.baseSalary + commission ∧
  (houseA.cost > 0 ∧ houseB.cost > 0 ∧ houseC.cost > 0) →
  3 = 3 := by
  sorry

#check nigella_sold_three_houses

end NUMINAMATH_CALUDE_nigella_sold_three_houses_l3115_311550


namespace NUMINAMATH_CALUDE_sine_sqrt_equality_l3115_311511

theorem sine_sqrt_equality (a : ℝ) (h1 : a ≥ 0) :
  (∀ x : ℝ, x ≥ 0 → Real.sin (Real.sqrt (x + a)) = Real.sin (Real.sqrt x)) →
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_sqrt_equality_l3115_311511


namespace NUMINAMATH_CALUDE_number_puzzle_l3115_311581

theorem number_puzzle (N : ℚ) : 
  (N / (4/5)) = ((4/5) * N + 18) → N = 40 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_l3115_311581


namespace NUMINAMATH_CALUDE_books_sold_in_three_days_l3115_311545

/-- The total number of books sold over three days -/
def total_books_sold (tuesday_sales wednesday_sales thursday_sales : ℕ) : ℕ :=
  tuesday_sales + wednesday_sales + thursday_sales

/-- Theorem stating the total number of books sold over three days -/
theorem books_sold_in_three_days :
  ∃ (tuesday_sales wednesday_sales thursday_sales : ℕ),
    tuesday_sales = 7 ∧
    wednesday_sales = 3 * tuesday_sales ∧
    thursday_sales = 3 * wednesday_sales ∧
    total_books_sold tuesday_sales wednesday_sales thursday_sales = 91 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_in_three_days_l3115_311545


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3115_311551

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  perimeter : ℝ
  is_isosceles : side ≥ 0 ∧ base ≥ 0 ∧ perimeter = 2 * side + base

-- Theorem statement
theorem isosceles_triangle_base_length 
  (t : IsoscelesTriangle) 
  (h_perimeter : t.perimeter = 26) 
  (h_side : t.side = 11 ∨ t.base = 11) : 
  t.base = 11 ∨ t.base = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3115_311551


namespace NUMINAMATH_CALUDE_g_composition_result_l3115_311513

/-- Definition of the function g for complex numbers -/
noncomputable def g (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 - 1 else z^3 + 1

/-- Theorem stating the result of g(g(g(g(2+i)))) -/
theorem g_composition_result :
  g (g (g (g (2 + Complex.I)))) = (-64555 + 70232 * Complex.I)^3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_result_l3115_311513


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3115_311527

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (a 1 + a n) / 2
  sum_13 : sum 13 = 104
  a_6 : a 6 = 5

/-- The common difference of the arithmetic sequence is 3 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) : seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3115_311527


namespace NUMINAMATH_CALUDE_intersection_A_B_l3115_311554

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3115_311554


namespace NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l3115_311530

theorem right_triangle_and_modular_inverse : 
  (80^2 + 150^2 = 170^2) ∧ 
  (320 * 642 % 2879 = 1) := by sorry

end NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l3115_311530


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3115_311586

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + b^2 = 64 →
  (1/2) * a * b = 10 →
  c^2 + d^2 = (5*8)^2 →
  (1/2) * c * d = 250 →
  c + d = 51 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3115_311586


namespace NUMINAMATH_CALUDE_money_distribution_l3115_311563

theorem money_distribution (x : ℝ) (h : x > 0) :
  let moe_original := 6 * x
  let loki_original := 5 * x
  let kai_original := 2 * x
  let total_original := moe_original + loki_original + kai_original
  let ott_received := 3 * x
  ott_received / total_original = 3 / 13 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3115_311563


namespace NUMINAMATH_CALUDE_power_twelve_half_l3115_311553

theorem power_twelve_half : (12 : ℕ) ^ ((12 : ℕ) / 2) = 2985984 := by sorry

end NUMINAMATH_CALUDE_power_twelve_half_l3115_311553


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3115_311564

theorem inequality_system_solution (x : ℝ) : 
  2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3115_311564


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l3115_311518

theorem dvd_rental_cost (total_dvds : ℕ) (total_cost : ℝ) (known_dvds : ℕ) (known_cost : ℝ) : 
  total_dvds = 7 → 
  total_cost = 12.6 → 
  known_dvds = 3 → 
  known_cost = 1.5 → 
  total_cost - (known_dvds * known_cost) = 8.1 :=
by sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l3115_311518


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l3115_311552

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (h1 : 5 ∣ (x - 2)) 
  (h2 : 5 ∣ (y + 4)) : 
  (∃ n : ℕ, n > 0 ∧ 5 ∣ (x^2 + x^2*y + y^2 + n) ∧ 
    ∀ m : ℕ, (m > 0 ∧ m < n) → ¬(5 ∣ (x^2 + x^2*y + y^2 + m))) → 
  (∃ n : ℕ, n = 1 ∧ 5 ∣ (x^2 + x^2*y + y^2 + n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l3115_311552


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l3115_311576

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem largest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (factorial 6 + factorial 7) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (factorial 6 + factorial 7) → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l3115_311576


namespace NUMINAMATH_CALUDE_count_special_primes_l3115_311507

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def swap_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

def is_special_prime (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ is_prime (swap_digits n)

theorem count_special_primes : 
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_special_prime n) ∧ s.card = 9 ∧ 
   (∀ m : ℕ, is_special_prime m → m ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_count_special_primes_l3115_311507


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_l3115_311539

/-- Represents different types of statistical graphs -/
inductive StatisticalGraph
| BarGraph
| PieChart
| LineGraph
| FrequencyDistributionGraph

/-- Characteristics of a statistical graph -/
structure GraphCharacteristics where
  showsTrend : Bool
  showsTimeProgression : Bool
  comparesCategories : Bool
  showsProportions : Bool
  showsFrequency : Bool

/-- Define the characteristics of each graph type -/
def graphProperties : StatisticalGraph → GraphCharacteristics
| StatisticalGraph.BarGraph => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := true,
    showsProportions := false,
    showsFrequency := false
  }
| StatisticalGraph.PieChart => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := false,
    showsProportions := true,
    showsFrequency := false
  }
| StatisticalGraph.LineGraph => {
    showsTrend := true,
    showsTimeProgression := true,
    comparesCategories := false,
    showsProportions := false,
    showsFrequency := false
  }
| StatisticalGraph.FrequencyDistributionGraph => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := false,
    showsProportions := false,
    showsFrequency := true
  }

/-- Defines the requirements for a graph to show temperature trends over a week -/
def suitableForTemperatureTrend (g : GraphCharacteristics) : Prop :=
  g.showsTrend ∧ g.showsTimeProgression

/-- Theorem stating that a line graph is the most suitable for showing temperature trends over a week -/
theorem line_graph_most_suitable :
  ∀ (g : StatisticalGraph), 
    suitableForTemperatureTrend (graphProperties g) → g = StatisticalGraph.LineGraph := by
  sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_l3115_311539


namespace NUMINAMATH_CALUDE_right_triangle_sin_cos_relation_l3115_311572

theorem right_triangle_sin_cos_relation (A B C : ℝ) (h1 : 0 < A) (h2 : A < π / 2) :
  Real.cos B = 0 → 3 * Real.sin A = 4 * Real.cos A → Real.sin A = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_cos_relation_l3115_311572


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3115_311567

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_ninth_term :
  ∀ n : ℕ, arithmeticSequence 1 (-2) n = -15 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3115_311567


namespace NUMINAMATH_CALUDE_positive_sixth_root_of_64_l3115_311515

theorem positive_sixth_root_of_64 (y : ℝ) (h1 : y > 0) (h2 : y^6 = 64) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_sixth_root_of_64_l3115_311515


namespace NUMINAMATH_CALUDE_m_range_l3115_311590

-- Define the plane region
def plane_region (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |2 * p.1 + p.2 + m| < 3}

-- Define the theorem
theorem m_range (m : ℝ) :
  ((0, 0) ∈ plane_region m) ∧ ((-1, 1) ∈ plane_region m) →
  -2 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3115_311590


namespace NUMINAMATH_CALUDE_min_integer_solution_l3115_311533

def is_solution (x : ℤ) : Prop :=
  (3 - x > 0) ∧ ((4 * x : ℚ) / 3 + 3 / 2 > -x / 6)

theorem min_integer_solution :
  is_solution 0 ∧ ∀ y : ℤ, y < 0 → ¬is_solution y :=
sorry

end NUMINAMATH_CALUDE_min_integer_solution_l3115_311533


namespace NUMINAMATH_CALUDE_enclosed_area_circular_arcs_l3115_311537

/-- The area enclosed by a curve composed of 9 congruent circular arcs -/
theorem enclosed_area_circular_arcs (n : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) 
  (h1 : n = 9)
  (h2 : arc_length = 5 * π / 6)
  (h3 : hexagon_side = 3) :
  ∃ (area : ℝ), area = 13.5 * Real.sqrt 3 + 375 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_circular_arcs_l3115_311537


namespace NUMINAMATH_CALUDE_power_sum_divisibility_l3115_311512

theorem power_sum_divisibility (k : ℕ) :
  7 ∣ (2^k + 3^k) ↔ k % 6 = 3 := by sorry

end NUMINAMATH_CALUDE_power_sum_divisibility_l3115_311512


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3115_311571

theorem solve_exponential_equation (x : ℝ) (h : x ≠ 0) :
  x^(-(2/3) : ℝ) = 4 ↔ x = 1/8 ∨ x = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3115_311571


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3115_311520

/-- The quadratic function f(x) with parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + t*x - t

/-- f has a root for a given t -/
def has_root (t : ℝ) : Prop := ∃ x, f t x = 0

theorem sufficient_not_necessary_condition :
  (∀ t ≥ 0, has_root t) ∧ (∃ t < 0, has_root t) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3115_311520


namespace NUMINAMATH_CALUDE_study_group_probability_l3115_311529

/-- Given a study group where 70% of members are women and 40% of women are lawyers,
    the probability of randomly selecting a woman lawyer is 0.28. -/
theorem study_group_probability (total : ℕ) (women : ℕ) (women_lawyers : ℕ)
    (h1 : women = (70 : ℕ) * total / 100)
    (h2 : women_lawyers = (40 : ℕ) * women / 100) :
    (women_lawyers : ℚ) / total = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_study_group_probability_l3115_311529


namespace NUMINAMATH_CALUDE_avg_days_before_trial_is_four_l3115_311597

/-- The average number of days spent in jail before trial -/
def avg_days_before_trial (num_cities num_days arrests_per_day total_weeks : ℕ) : ℚ :=
  let total_arrests := num_cities * num_days * arrests_per_day
  let total_jail_days := total_weeks * 7
  let days_after_trial := 7
  (total_jail_days / total_arrests : ℚ) - days_after_trial

theorem avg_days_before_trial_is_four :
  avg_days_before_trial 21 30 10 9900 = 4 := by
  sorry

end NUMINAMATH_CALUDE_avg_days_before_trial_is_four_l3115_311597


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3115_311535

theorem ab_positive_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, a * b > 0 ∧ b / a + a / b ≤ 2) ∧
  (∀ a b : ℝ, b / a + a / b > 2 → a * b > 0) :=
sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3115_311535


namespace NUMINAMATH_CALUDE_f_properties_l3115_311598

def f (x : ℝ) : ℝ := x^2 + x - 6

theorem f_properties :
  (f 0 = -6) ∧ (∀ x : ℝ, f x = 0 → x = -3 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3115_311598


namespace NUMINAMATH_CALUDE_lemonade_third_intermission_l3115_311506

theorem lemonade_third_intermission 
  (total : ℝ) 
  (first : ℝ) 
  (second : ℝ) 
  (h1 : total = 0.9166666666666666) 
  (h2 : first = 0.25) 
  (h3 : second = 0.4166666666666667) :
  total - (first + second) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_lemonade_third_intermission_l3115_311506


namespace NUMINAMATH_CALUDE_dealer_profit_is_25_percent_l3115_311541

/-- Represents a dishonest dealer's selling strategy -/
structure DishonestDealer where
  weight_reduction : ℝ  -- Percentage reduction in weight
  impurity_addition : ℝ  -- Percentage of impurities added
  
/-- Calculates the net profit percentage for a dishonest dealer -/
def net_profit_percentage (dealer : DishonestDealer) : ℝ :=
  sorry

/-- Theorem stating that the net profit percentage is 25% for the given conditions -/
theorem dealer_profit_is_25_percent :
  let dealer : DishonestDealer := { weight_reduction := 0.20, impurity_addition := 0.25 }
  net_profit_percentage dealer = 0.25 := by sorry

end NUMINAMATH_CALUDE_dealer_profit_is_25_percent_l3115_311541


namespace NUMINAMATH_CALUDE_complex_fraction_equals_point_l3115_311589

theorem complex_fraction_equals_point : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_point_l3115_311589


namespace NUMINAMATH_CALUDE_tim_website_earnings_l3115_311588

/-- Calculates Tim's earnings from his website for a week -/
def website_earnings (
  daily_visitors : ℕ)  -- Number of visitors per day for the first 6 days
  (days : ℕ)            -- Number of days with constant visitors
  (last_day_multiplier : ℕ)  -- Multiplier for visitors on the last day
  (earnings_per_visit : ℚ)  -- Earnings per visit in dollars
  : ℚ :=
  let first_days_visitors := daily_visitors * days
  let last_day_visitors := first_days_visitors * last_day_multiplier
  let total_visitors := first_days_visitors + last_day_visitors
  (total_visitors : ℚ) * earnings_per_visit

/-- Theorem stating Tim's earnings for the week -/
theorem tim_website_earnings :
  website_earnings 100 6 2 (1/100) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tim_website_earnings_l3115_311588


namespace NUMINAMATH_CALUDE_real_roots_iff_m_leq_quarter_m_eq_neg_one_when_sum_and_product_condition_l3115_311548

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2*m - 1)^2 - 4*1*m^2

-- Theorem for the range of m for real roots
theorem real_roots_iff_m_leq_quarter (m : ℝ) :
  (∃ x : ℝ, quadratic m x = 0) ↔ m ≤ 1/4 := by sorry

-- Theorem for the value of m when x₁x₂ + x₁ + x₂ = 4
theorem m_eq_neg_one_when_sum_and_product_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁*x₂ + x₁ + x₂ = 4) →
  m = -1 := by sorry

end NUMINAMATH_CALUDE_real_roots_iff_m_leq_quarter_m_eq_neg_one_when_sum_and_product_condition_l3115_311548


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3115_311596

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 0 ∧ exterior_angle = 15 → n * exterior_angle = 360 → n = 24 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3115_311596


namespace NUMINAMATH_CALUDE_shouzhudaitu_only_random_event_l3115_311560

-- Define the type for idioms
inductive Idiom
  | HaiKuShiLan
  | ShouZhuDaiTu
  | HuaBingChongJi
  | GuaShuDiLuo

-- Define a property for idioms that describe a random event
def describes_random_event (i : Idiom) : Prop :=
  match i with
  | Idiom.ShouZhuDaiTu => True
  | _ => False

-- Theorem statement
theorem shouzhudaitu_only_random_event :
  ∀ (i : Idiom), describes_random_event i ↔ i = Idiom.ShouZhuDaiTu :=
by
  sorry


end NUMINAMATH_CALUDE_shouzhudaitu_only_random_event_l3115_311560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3115_311577

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 2011 = 10) →
  (a 1 * a 2011 = 16) →
  a 2 + a 1006 + a 2010 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3115_311577


namespace NUMINAMATH_CALUDE_dilute_herbal_essence_l3115_311582

/-- Proves that adding 7.5 ounces of water to a 15-ounce solution containing 60% essence
    results in a new solution with 40% essence -/
theorem dilute_herbal_essence :
  let initial_weight : ℝ := 15
  let initial_concentration : ℝ := 0.6
  let final_concentration : ℝ := 0.4
  let water_added : ℝ := 7.5
  let essence_amount : ℝ := initial_weight * initial_concentration
  let final_weight : ℝ := initial_weight + water_added
  essence_amount / final_weight = final_concentration := by sorry

end NUMINAMATH_CALUDE_dilute_herbal_essence_l3115_311582


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3115_311521

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 5
  ∃ x₁ x₂ : ℝ, x₁ = 5 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3115_311521


namespace NUMINAMATH_CALUDE_overlapping_circles_common_chord_l3115_311528

theorem overlapping_circles_common_chord 
  (r : ℝ) 
  (h : r = 12) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_circles_common_chord_l3115_311528


namespace NUMINAMATH_CALUDE_monomial_sum_implies_m_plus_n_eq_3_l3115_311538

/-- Two algebraic expressions form a monomial when added together if they have the same powers for each variable -/
def forms_monomial (expr1 expr2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), expr1 x y ≠ 0 ∧ expr2 x y ≠ 0 → x = y

/-- The first algebraic expression: 3a^m * b^2 -/
def expr1 (m : ℕ) (a b : ℕ) : ℚ := 3 * (a^m) * (b^2)

/-- The second algebraic expression: -2a^2 * b^(n+1) -/
def expr2 (n : ℕ) (a b : ℕ) : ℚ := -2 * (a^2) * (b^(n+1))

theorem monomial_sum_implies_m_plus_n_eq_3 (m n : ℕ) :
  forms_monomial (expr1 m) (expr2 n) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_m_plus_n_eq_3_l3115_311538


namespace NUMINAMATH_CALUDE_burger_share_inches_l3115_311569

/-- The length of a foot in inches -/
def foot_in_inches : ℝ := 12

/-- The length of the burger in feet -/
def burger_length_feet : ℝ := 1

/-- The number of people sharing the burger -/
def num_people : ℕ := 2

/-- Theorem: Each person's share of a foot-long burger is 6 inches when shared equally between two people -/
theorem burger_share_inches : 
  (burger_length_feet * foot_in_inches) / num_people = 6 := by sorry

end NUMINAMATH_CALUDE_burger_share_inches_l3115_311569


namespace NUMINAMATH_CALUDE_shifted_cosine_to_sine_l3115_311587

/-- Given a cosine function shifted to create an odd function, 
    prove the value at a specific point. -/
theorem shifted_cosine_to_sine (f g : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.cos (x / 2 - π / 3)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, g x = f (x - φ)) →
  (∀ x, g x + g (-x) = 0) →
  g (2 * φ + π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_cosine_to_sine_l3115_311587


namespace NUMINAMATH_CALUDE_raised_beds_planks_l3115_311534

/-- Calculates the number of 8-foot long planks needed for raised beds --/
def planks_needed (num_beds : ℕ) (bed_height : ℕ) (bed_width : ℕ) (bed_length : ℕ) (plank_width : ℕ) (plank_length : ℕ) : ℕ :=
  let long_sides := 2 * bed_height
  let short_sides := 2 * bed_height * bed_width / plank_length
  let planks_per_bed := long_sides + short_sides
  num_beds * planks_per_bed

theorem raised_beds_planks :
  planks_needed 10 2 2 8 1 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_raised_beds_planks_l3115_311534


namespace NUMINAMATH_CALUDE_roots_equation_sum_l3115_311573

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 4*α - 1 = 0 → β^2 - 4*β - 1 = 0 → 3*α^3 + 4*β^2 = 80 + 35*α := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l3115_311573


namespace NUMINAMATH_CALUDE_magazines_per_bookshelf_l3115_311542

theorem magazines_per_bookshelf
  (num_books : ℕ)
  (num_bookshelves : ℕ)
  (total_items : ℕ)
  (h1 : num_books = 23)
  (h2 : num_bookshelves = 29)
  (h3 : total_items = 2436) :
  (total_items - num_books) / num_bookshelves = 83 := by
sorry

end NUMINAMATH_CALUDE_magazines_per_bookshelf_l3115_311542


namespace NUMINAMATH_CALUDE_price_decrease_l3115_311522

theorem price_decrease (original_price : ℝ) (decreased_price : ℝ) : 
  decreased_price = original_price * (1 - 0.24) ∧ decreased_price = 608 → 
  original_price = 800 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_l3115_311522


namespace NUMINAMATH_CALUDE_women_average_age_l3115_311599

theorem women_average_age (n : ℕ) (A : ℝ) (W₁ W₂ : ℝ) :
  n = 7 ∧ 
  (n * A - 26 - 30 + W₁ + W₂) / n = A + 4 →
  (W₁ + W₂) / 2 = 42 := by
sorry

end NUMINAMATH_CALUDE_women_average_age_l3115_311599


namespace NUMINAMATH_CALUDE_class_size_proof_l3115_311583

theorem class_size_proof (S : ℕ) : 
  S / 2 + S / 3 + 4 = S → S = 24 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l3115_311583


namespace NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_09_l3115_311502

/-- The molecular weight of calcium carbonate (CaCO3) -/
def molecular_weight_CaCO3 : ℝ :=
  let calcium_weight : ℝ := 40.08
  let carbon_weight : ℝ := 12.01
  let oxygen_weight : ℝ := 16.00
  calcium_weight + carbon_weight + 3 * oxygen_weight

/-- Theorem stating that the molecular weight of CaCO3 is approximately 100.09 -/
theorem molecular_weight_CaCO3_is_100_09 :
  ∃ ε > 0, |molecular_weight_CaCO3 - 100.09| < ε :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_09_l3115_311502


namespace NUMINAMATH_CALUDE_rectangle_width_from_square_l3115_311565

theorem rectangle_width_from_square (square_side : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_length = 18 →
  4 * square_side = 2 * (rect_length + (4 * square_side - 2 * rect_length) / 2) →
  (4 * square_side - 2 * rect_length) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_from_square_l3115_311565


namespace NUMINAMATH_CALUDE_min_length_AB_l3115_311510

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the line y = 2
def line_y_2 (x y : ℝ) : Prop := y = 2

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- State the theorem
theorem min_length_AB :
  ∀ (x_A y_A x_B y_B : ℝ),
  line_y_2 x_A y_A →
  ellipse_C x_B y_B →
  perpendicular x_A y_A x_B y_B →
  ∀ (x y : ℝ),
  line_y_2 x y →
  ellipse_C x y →
  perpendicular x y x_B y_B →
  (x_A - x_B)^2 + (y_A - y_B)^2 ≤ (x - x_B)^2 + (y - y_B)^2 :=
sorry

end NUMINAMATH_CALUDE_min_length_AB_l3115_311510


namespace NUMINAMATH_CALUDE_roberts_trip_l3115_311524

/-- Proves that given the conditions of Robert's trip, the return trip takes 2.5 hours -/
theorem roberts_trip (distance : ℝ) (outbound_time : ℝ) (saved_time : ℝ) (avg_speed : ℝ) :
  distance = 180 →
  outbound_time = 3 →
  saved_time = 0.5 →
  avg_speed = 80 →
  (2 * distance) / (outbound_time + (outbound_time + saved_time - 2 * saved_time) - 2 * saved_time) = avg_speed →
  outbound_time + saved_time - 2 * saved_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_roberts_trip_l3115_311524


namespace NUMINAMATH_CALUDE_ladder_geometric_sequence_a10_l3115_311526

/-- A sequence {a_n} is an m-th order ladder geometric sequence if it satisfies
    a_{n+m}^2 = a_n × a_{n+2m} for any positive integers n and m. -/
def is_ladder_geometric (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∀ n : ℕ, (a (n + m))^2 = a n * a (n + 2*m)

theorem ladder_geometric_sequence_a10 (a : ℕ → ℝ) :
  is_ladder_geometric a 3 → a 1 = 1 → a 4 = 2 → a 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ladder_geometric_sequence_a10_l3115_311526


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_in_right_triangle_l3115_311523

/-- For a right-angled triangle with perimeter k, the radius r of the largest inscribed circle
    is given by r = k/2 * (3 - 2√2). -/
theorem largest_inscribed_circle_in_right_triangle (k : ℝ) (h : k > 0) :
  ∃ (r : ℝ), r = k / 2 * (3 - 2 * Real.sqrt 2) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a + b + c = k →   -- perimeter condition
  2 * (a * b) / (a + b + c) ≤ r :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_in_right_triangle_l3115_311523


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l3115_311568

theorem expansion_coefficient_sum (a : ℤ) (n : ℕ) : 
  (2^n = 64) → 
  ((1 + a)^6 = 729) → 
  (a = -4 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l3115_311568


namespace NUMINAMATH_CALUDE_board_covering_impossible_l3115_311546

/-- Represents a rectangular board -/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Represents a rectangular piece -/
structure Piece where
  width : ℕ
  height : ℕ

/-- Determines if a board can be completely covered by given pieces without overlapping or sticking out -/
def can_cover (b : Board) (p : Piece) : Prop :=
  ∃ (arrangement : ℕ), 
    arrangement * (p.width * p.height) = b.rows * b.cols ∧ 
    b.rows % p.height = 0 ∧ 
    b.cols % p.width = 0

theorem board_covering_impossible : 
  ¬(can_cover ⟨6, 6⟩ ⟨1, 4⟩) ∧ ¬(can_cover ⟨12, 9⟩ ⟨2, 2⟩) := by
  sorry


end NUMINAMATH_CALUDE_board_covering_impossible_l3115_311546


namespace NUMINAMATH_CALUDE_supercomputer_multiplications_l3115_311549

/-- The number of multiplications a supercomputer can perform in half a day -/
def multiplications_in_half_day (multiplications_per_second : ℕ) : ℕ :=
  multiplications_per_second * (12 * 3600)

/-- Theorem stating that a supercomputer performing 80,000 multiplications per second
    will execute 3,456,000,000 multiplications in half a day -/
theorem supercomputer_multiplications :
  multiplications_in_half_day 80000 = 3456000000 := by
  sorry

#eval multiplications_in_half_day 80000

end NUMINAMATH_CALUDE_supercomputer_multiplications_l3115_311549


namespace NUMINAMATH_CALUDE_constant_width_from_circle_sum_l3115_311595

/-- A curve in 2D space -/
structure Curve where
  -- Add necessary fields here
  convex : Bool

/-- Rotation of a curve by 180 degrees -/
def rotate180 (K : Curve) : Curve :=
  sorry

/-- Sum of two curves -/
def curveSum (K1 K2 : Curve) : Curve :=
  sorry

/-- Check if a curve is a circle -/
def isCircle (K : Curve) : Prop :=
  sorry

/-- Check if a curve has constant width -/
def hasConstantWidth (K : Curve) : Prop :=
  sorry

/-- Main theorem -/
theorem constant_width_from_circle_sum (K : Curve) (h : K.convex) :
  let K' := rotate180 K
  let K_star := curveSum K K'
  isCircle K_star → hasConstantWidth K :=
by
  sorry

end NUMINAMATH_CALUDE_constant_width_from_circle_sum_l3115_311595


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3115_311579

theorem inequality_equivalence (x : ℝ) : -9 < 2*x - 1 ∧ 2*x - 1 ≤ 6 → -4 < x ∧ x ≤ 3.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3115_311579


namespace NUMINAMATH_CALUDE_pole_height_l3115_311514

/-- Represents the geometry of a telephone pole with a supporting cable -/
structure TelephonePole where
  /-- Height of the pole in meters -/
  height : ℝ
  /-- Distance from the base of the pole to where the cable touches the ground, in meters -/
  cable_ground_distance : ℝ
  /-- Height of a person touching the cable, in meters -/
  person_height : ℝ
  /-- Distance from the base of the pole to where the person stands, in meters -/
  person_distance : ℝ

/-- Theorem stating the height of the telephone pole -/
theorem pole_height (pole : TelephonePole) 
  (h1 : pole.cable_ground_distance = 3)
  (h2 : pole.person_height = 1.5)
  (h3 : pole.person_distance = 2.5)
  : pole.height = 9 := by
  sorry

/-- Main statement combining the structure and theorem -/
def main : Prop :=
  ∃ pole : TelephonePole, 
    pole.cable_ground_distance = 3 ∧
    pole.person_height = 1.5 ∧
    pole.person_distance = 2.5 ∧
    pole.height = 9

end NUMINAMATH_CALUDE_pole_height_l3115_311514


namespace NUMINAMATH_CALUDE_francie_allowance_problem_l3115_311591

/-- Represents the number of weeks Francie received her increased allowance -/
def weeks_of_increased_allowance : ℕ := 6

/-- Initial savings from the first 8 weeks -/
def initial_savings : ℕ := 5 * 8

/-- Increased allowance per week -/
def increased_allowance : ℕ := 6

/-- Total savings including both initial and increased allowance periods -/
def total_savings (x : ℕ) : ℕ := initial_savings + increased_allowance * x

theorem francie_allowance_problem :
  total_savings weeks_of_increased_allowance / 2 = 35 + 3 ∧
  weeks_of_increased_allowance * increased_allowance = total_savings weeks_of_increased_allowance - initial_savings :=
by sorry

end NUMINAMATH_CALUDE_francie_allowance_problem_l3115_311591


namespace NUMINAMATH_CALUDE_probability_of_green_ball_l3115_311503

theorem probability_of_green_ball (total_balls : ℕ) (green_balls : ℕ) (red_balls : ℕ)
  (h1 : total_balls = 10)
  (h2 : green_balls = 7)
  (h3 : red_balls = 3)
  (h4 : total_balls = green_balls + red_balls) :
  (green_balls : ℚ) / total_balls = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_green_ball_l3115_311503


namespace NUMINAMATH_CALUDE_min_area_quadrilateral_on_parabola_l3115_311532

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4*x

/-- Point on a parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : par.eq x y

/-- Chord of a parabola -/
structure Chord (par : Parabola) where
  p1 : PointOnParabola par
  p2 : PointOnParabola par

/-- Theorem: Minimum area of quadrilateral ABCD -/
theorem min_area_quadrilateral_on_parabola (par : Parabola)
  (A B C D : PointOnParabola par) 
  (chord_AC chord_BD : Chord par)
  (perp : chord_AC.p1.x = A.x ∧ chord_AC.p1.y = A.y ∧ 
          chord_AC.p2.x = C.x ∧ chord_AC.p2.y = C.y ∧
          chord_BD.p1.x = B.x ∧ chord_BD.p1.y = B.y ∧
          chord_BD.p2.x = D.x ∧ chord_BD.p2.y = D.y ∧
          (chord_AC.p2.y - chord_AC.p1.y) * (chord_BD.p2.y - chord_BD.p1.y) = 
          -(chord_AC.p2.x - chord_AC.p1.x) * (chord_BD.p2.x - chord_BD.p1.x))
  (through_focus : ∃ t : ℝ, 
    chord_AC.p1.x + t * (chord_AC.p2.x - chord_AC.p1.x) = par.p / 2 ∧
    chord_AC.p1.y + t * (chord_AC.p2.y - chord_AC.p1.y) = 0 ∧
    chord_BD.p1.x + t * (chord_BD.p2.x - chord_BD.p1.x) = par.p / 2 ∧
    chord_BD.p1.y + t * (chord_BD.p2.y - chord_BD.p1.y) = 0) :
  ∃ area : ℝ, area ≥ 32 ∧ 
    area = (1/2) * Real.sqrt ((A.x - C.x)^2 + (A.y - C.y)^2) * 
                    Real.sqrt ((B.x - D.x)^2 + (B.y - D.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_min_area_quadrilateral_on_parabola_l3115_311532


namespace NUMINAMATH_CALUDE_johns_zoo_l3115_311585

theorem johns_zoo (snakes : ℕ) (monkeys : ℕ) (lions : ℕ) (pandas : ℕ) (dogs : ℕ) :
  snakes = 15 ∧
  monkeys = 2 * snakes ∧
  lions = monkeys - 5 ∧
  pandas = lions + 8 ∧
  dogs = pandas / 3 →
  snakes + monkeys + lions + pandas + dogs = 114 := by
sorry

end NUMINAMATH_CALUDE_johns_zoo_l3115_311585


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3115_311504

-- Problem 1
theorem problem_1 : |(-12)| - (-6) + 5 - 10 = 13 := by sorry

-- Problem 2
theorem problem_2 : 64.83 - 5 * (18/19) + 35.17 - 44 * (1/19) = 50 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3115_311504


namespace NUMINAMATH_CALUDE_largest_t_value_l3115_311517

theorem largest_t_value : ∃ t_max : ℚ, 
  (∀ t : ℚ, (16 * t^2 - 40 * t + 15) / (4 * t - 3) + 7 * t = 5 * t + 2 → t ≤ t_max) ∧ 
  (16 * t_max^2 - 40 * t_max + 15) / (4 * t_max - 3) + 7 * t_max = 5 * t_max + 2 ∧
  t_max = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_largest_t_value_l3115_311517


namespace NUMINAMATH_CALUDE_work_completion_l3115_311593

/-- Given that 8 men complete a work in 80 days, prove that 20 men will complete the same work in 32 days. -/
theorem work_completion (work : ℕ) : 
  (8 * 80 = work) → (20 * 32 = work) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3115_311593
