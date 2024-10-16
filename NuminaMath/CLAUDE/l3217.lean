import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_distribution_l3217_321781

/-- The number of students -/
def num_students : ℕ := 211

/-- The number of possible combinations of chocolate choices -/
def num_combinations : ℕ := 35

/-- The minimum number of students in the largest group -/
def min_largest_group : ℕ := 7

theorem chocolate_distribution :
  ∃ (group : Finset (Fin num_students)),
    group.card ≥ min_largest_group ∧
    ∀ (s₁ s₂ : Fin num_students),
      s₁ ∈ group → s₂ ∈ group →
      ∃ (c : Fin num_combinations), true :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3217_321781


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3217_321715

theorem quadratic_distinct_roots (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  (n < -6 ∨ n > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3217_321715


namespace NUMINAMATH_CALUDE_puppy_adoption_cost_is_96_l3217_321777

/-- Calculates the total cost of adopting a puppy and buying necessary supplies with a discount --/
def puppy_adoption_cost (adoption_fee : ℝ) (dog_food : ℝ) (treats_price : ℝ) (treats_quantity : ℕ)
  (toys : ℝ) (crate_bed_price : ℝ) (collar_leash : ℝ) (discount_rate : ℝ) : ℝ :=
  let supplies_cost := dog_food + treats_price * treats_quantity + toys + 2 * crate_bed_price + collar_leash
  let discounted_supplies := supplies_cost * (1 - discount_rate)
  adoption_fee + discounted_supplies

/-- Theorem stating that the total cost of adopting a puppy and buying supplies is $96.00 --/
theorem puppy_adoption_cost_is_96 :
  puppy_adoption_cost 20 20 2.5 2 15 20 15 0.2 = 96 := by
  sorry


end NUMINAMATH_CALUDE_puppy_adoption_cost_is_96_l3217_321777


namespace NUMINAMATH_CALUDE_garden_width_is_ten_l3217_321767

/-- Represents a rectangular garden with specific properties. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 60
  area_eq : width * length = 200
  length_twice_width : length = 2 * width

/-- Theorem stating that a rectangular garden with the given properties has a width of 10 meters. -/
theorem garden_width_is_ten (garden : RectangularGarden) : garden.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_is_ten_l3217_321767


namespace NUMINAMATH_CALUDE_age_half_in_ten_years_l3217_321775

def mother_age : ℕ := 50

def person_age : ℕ := (2 * mother_age) / 5

def years_until_half (y : ℕ) : Prop :=
  2 * (person_age + y) = mother_age + y

theorem age_half_in_ten_years :
  ∃ y : ℕ, years_until_half y ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_age_half_in_ten_years_l3217_321775


namespace NUMINAMATH_CALUDE_perimeter_difference_l3217_321763

/-- Represents a rectangle with length and height -/
structure Rectangle where
  length : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.height)

/-- Theorem stating the difference in perimeters of two rectangles -/
theorem perimeter_difference (inner outer : Rectangle) 
  (h1 : outer.length = 7)
  (h2 : outer.height = 5) :
  perimeter outer - perimeter inner = 24 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_difference_l3217_321763


namespace NUMINAMATH_CALUDE_sum_square_bound_l3217_321704

/-- The sum of integers from 1 to n -/
def sum_to (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for a natural number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem sum_square_bound :
  ∀ K : ℕ, K > 0 →
    (is_perfect_square (sum_to K) ∧
     ∃ N : ℕ, sum_to K = N * N ∧ N + K < 120) ↔
    (K = 1 ∨ K = 8 ∨ K = 49) :=
sorry

end NUMINAMATH_CALUDE_sum_square_bound_l3217_321704


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l3217_321799

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 - 3*x^11 + 4*x^9 - 2*x^8) =
  15*x^13 - 19*x^12 + 6*x^11 + 12*x^10 - 14*x^9 - 4*x^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l3217_321799


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l3217_321785

/-- The exchange rate from U.S. dollars to British pounds -/
def exchange_rate : ℚ := 8 / 5

/-- The amount spent in British pounds -/
def amount_spent : ℚ := 72

/-- The remaining amount in British pounds as a function of initial U.S. dollars -/
def remaining (d : ℚ) : ℚ := 4 * d

theorem currency_exchange_problem (d : ℚ) : 
  (exchange_rate * d - amount_spent = remaining d) → d = -30 := by
  sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l3217_321785


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l3217_321778

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 : ℚ) / 3 = 27 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l3217_321778


namespace NUMINAMATH_CALUDE_sphere_minus_cylinder_volume_l3217_321792

/-- The volume of space inside a sphere but outside an inscribed right cylinder -/
theorem sphere_minus_cylinder_volume (r_sphere : ℝ) (r_cylinder : ℝ) : 
  r_sphere = 6 → r_cylinder = 4 → 
  (4/3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * Real.sqrt (r_sphere^2 - r_cylinder^2)) = 
  (288 - 64 * Real.sqrt 5) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_minus_cylinder_volume_l3217_321792


namespace NUMINAMATH_CALUDE_reaction_masses_l3217_321790

-- Define the molar masses
def molar_mass_HCl : ℝ := 36.46
def molar_mass_AgNO3 : ℝ := 169.87
def molar_mass_AgCl : ℝ := 143.32

-- Define the number of moles of AgNO3
def moles_AgNO3 : ℝ := 3

-- Define the reaction stoichiometry
def stoichiometry : ℝ := 1

-- Theorem statement
theorem reaction_masses :
  let mass_HCl := moles_AgNO3 * molar_mass_HCl * stoichiometry
  let mass_AgNO3 := moles_AgNO3 * molar_mass_AgNO3
  let mass_AgCl := moles_AgNO3 * molar_mass_AgCl * stoichiometry
  (mass_HCl = 109.38) ∧ (mass_AgNO3 = 509.61) ∧ (mass_AgCl = 429.96) := by
  sorry

end NUMINAMATH_CALUDE_reaction_masses_l3217_321790


namespace NUMINAMATH_CALUDE_domain_of_sqrt_tan_plus_sqrt_neg_cos_l3217_321717

theorem domain_of_sqrt_tan_plus_sqrt_neg_cos (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ y, y = Real.sqrt (Real.tan x) + Real.sqrt (-Real.cos x)) ↔
  x ∈ Set.Ico Real.pi (3 * Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_tan_plus_sqrt_neg_cos_l3217_321717


namespace NUMINAMATH_CALUDE_max_sum_is_24_l3217_321722

/-- Represents the grid configuration -/
structure Grid :=
  (a b c d e : ℕ)

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {5, 8, 11, 14}

/-- Checks if the grid contains only the available numbers -/
def Grid.isValid (g : Grid) : Prop :=
  {g.a, g.b, g.c, g.d, g.e} ⊆ availableNumbers

/-- Calculates the horizontal sum -/
def Grid.horizontalSum (g : Grid) : ℕ := g.a + g.b + g.e

/-- Calculates the vertical sum -/
def Grid.verticalSum (g : Grid) : ℕ := g.a + g.c + 2 * g.e

/-- Checks if the grid satisfies the sum condition -/
def Grid.satisfiesSumCondition (g : Grid) : Prop :=
  g.horizontalSum = g.verticalSum

theorem max_sum_is_24 :
  ∃ (g : Grid), g.isValid ∧ g.satisfiesSumCondition ∧
  (∀ (h : Grid), h.isValid → h.satisfiesSumCondition →
    g.horizontalSum ≥ h.horizontalSum ∧ g.verticalSum ≥ h.verticalSum) ∧
  g.horizontalSum = 24 ∧ g.verticalSum = 24 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_24_l3217_321722


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3217_321782

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the asymptote equation
def asymptote (m x y : ℝ) : Prop := y = m * x ∨ y = -m * x

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, hyperbola x y → asymptote m x y) ∧ m = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3217_321782


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l3217_321750

theorem second_pipe_fill_time (pipe1_rate : ℝ) (pipe2_rate : ℝ) (pipe3_rate : ℝ) 
  (combined_fill_time : ℝ) :
  pipe1_rate = 1 / 10 →
  pipe3_rate = -1 / 20 →
  combined_fill_time = 7.5 →
  pipe1_rate + pipe2_rate + pipe3_rate = 1 / combined_fill_time →
  1 / pipe2_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l3217_321750


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l3217_321758

theorem dinner_bill_proof (total_friends : ℕ) (paying_friends : ℕ) (extra_payment : ℚ) : 
  total_friends = 10 → 
  paying_friends = 9 → 
  extra_payment = 3 → 
  ∃ (bill : ℚ), bill = 270 ∧ 
    paying_friends * (bill / total_friends + extra_payment) = bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l3217_321758


namespace NUMINAMATH_CALUDE_water_in_tank_after_40_days_l3217_321708

/-- Calculates the final amount of water in a tank given initial conditions and events. -/
def finalWaterAmount (initialWater : ℝ) (evaporationRate : ℝ) (daysBeforeAddition : ℕ) 
  (addedWater : ℝ) (remainingDays : ℕ) : ℝ :=
  let waterAfterFirstEvaporation := initialWater - evaporationRate * daysBeforeAddition
  let waterAfterAddition := waterAfterFirstEvaporation + addedWater
  waterAfterAddition - evaporationRate * remainingDays

/-- The final amount of water in the tank is 520 liters. -/
theorem water_in_tank_after_40_days :
  finalWaterAmount 500 2 15 100 25 = 520 := by
  sorry

end NUMINAMATH_CALUDE_water_in_tank_after_40_days_l3217_321708


namespace NUMINAMATH_CALUDE_angle_sum_in_polygon_l3217_321732

theorem angle_sum_in_polygon (D E F p q : ℝ) : 
  D = 38 → E = 58 → F = 36 → 
  D + E + (360 - p) + 90 + (126 - q) = 540 → 
  p + q = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_polygon_l3217_321732


namespace NUMINAMATH_CALUDE_value_k_std_dev_below_mean_two_std_dev_below_mean_l3217_321742

/-- For a normal distribution with mean μ and standard deviation σ,
    the value that is exactly k standard deviations less than the mean is μ - k * σ -/
theorem value_k_std_dev_below_mean (μ σ k : ℝ) :
  μ - k * σ = μ - k * σ := by sorry

/-- For a normal distribution with mean 12 and standard deviation 1.2,
    the value that is exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_below_mean :
  let μ : ℝ := 12  -- mean
  let σ : ℝ := 1.2 -- standard deviation
  let k : ℝ := 2   -- number of standard deviations below mean
  μ - k * σ = 9.6 := by sorry

end NUMINAMATH_CALUDE_value_k_std_dev_below_mean_two_std_dev_below_mean_l3217_321742


namespace NUMINAMATH_CALUDE_problem_statement_l3217_321754

theorem problem_statement : 4 * Real.sqrt (1/2) + 3 * Real.sqrt (1/3) - Real.sqrt 8 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3217_321754


namespace NUMINAMATH_CALUDE_tan_105_degrees_l3217_321766

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l3217_321766


namespace NUMINAMATH_CALUDE_payment_function_correct_l3217_321795

/-- Represents the payment function for book purchases with a discount. -/
def payment_function (x : ℝ) : ℝ :=
  20 * x + 100

/-- Theorem stating the correctness of the payment function. -/
theorem payment_function_correct (x : ℝ) (h : x > 20) :
  payment_function x = (x - 20) * (25 * 0.8) + 20 * 25 := by
  sorry

#check payment_function_correct

end NUMINAMATH_CALUDE_payment_function_correct_l3217_321795


namespace NUMINAMATH_CALUDE_sum_of_pentagon_angles_l3217_321713

theorem sum_of_pentagon_angles : ∀ (A B C D E : ℝ),
  A + B + C + D + E = 180 * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_pentagon_angles_l3217_321713


namespace NUMINAMATH_CALUDE_triangle_shape_l3217_321759

theorem triangle_shape (a b : ℝ) (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < a ∧ 0 < b) (h_condition : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_shape_l3217_321759


namespace NUMINAMATH_CALUDE_representable_as_product_of_three_l3217_321714

theorem representable_as_product_of_three : ∃ (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ 2^58 + 1 = a * b * c := by
  sorry

end NUMINAMATH_CALUDE_representable_as_product_of_three_l3217_321714


namespace NUMINAMATH_CALUDE_farmer_rabbit_problem_l3217_321726

theorem farmer_rabbit_problem :
  ∀ (initial_rabbits : ℕ),
    (∃ (rabbits_per_cage : ℕ),
      initial_rabbits + 6 = 17 * rabbits_per_cage) →
    initial_rabbits = 28 := by
  sorry

end NUMINAMATH_CALUDE_farmer_rabbit_problem_l3217_321726


namespace NUMINAMATH_CALUDE_problem_solution_l3217_321712

theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 9)
  (g : ℝ → ℝ) (hg : ∀ x, g x = x^2 - 5)
  (h2 : f (g a) = 25) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3217_321712


namespace NUMINAMATH_CALUDE_arrangement_count_l3217_321719

/-- The number of ways to arrange 4 boys and 3 girls in a row -/
def total_arrangements : ℕ := Nat.factorial 7

/-- The number of ways to arrange 4 boys and 3 girls where all 3 girls are adjacent -/
def three_girls_adjacent : ℕ := Nat.factorial 5 * Nat.factorial 3

/-- The number of ways to arrange 4 boys and 3 girls where exactly 2 girls are adjacent -/
def two_girls_adjacent : ℕ := Nat.factorial 6 * Nat.factorial 2 * 3

/-- The number of valid arrangements -/
def valid_arrangements : ℕ := two_girls_adjacent - three_girls_adjacent

theorem arrangement_count : valid_arrangements = 3600 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3217_321719


namespace NUMINAMATH_CALUDE_profit_calculation_l3217_321744

-- Define the number of items bought and the price paid
def items_bought : ℕ := 60
def price_paid : ℕ := 46

-- Define the discount rate
def discount_rate : ℚ := 1 / 100

-- Define a function to calculate the profit percent
def profit_percent (items : ℕ) (price : ℕ) (discount : ℚ) : ℚ :=
  let cost_per_item : ℚ := price / items
  let selling_price : ℚ := 1 - discount
  let profit_per_item : ℚ := selling_price - cost_per_item
  (profit_per_item / cost_per_item) * 100

-- State the theorem
theorem profit_calculation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (profit_percent items_bought price_paid discount_rate - 2911/100) < ε :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l3217_321744


namespace NUMINAMATH_CALUDE_cat_weight_sum_l3217_321793

/-- The combined weight of three cats -/
def combined_weight (w1 w2 w3 : ℕ) : ℕ := w1 + w2 + w3

/-- Theorem: The combined weight of cats weighing 2, 7, and 4 pounds is 13 pounds -/
theorem cat_weight_sum : combined_weight 2 7 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_sum_l3217_321793


namespace NUMINAMATH_CALUDE_job_completion_time_l3217_321723

theorem job_completion_time 
  (T : ℝ) -- Time for P to complete the job alone
  (h1 : T > 0) -- Ensure T is positive
  (h2 : 3 * (1/T + 1/20) + 0.4 * (1/T) = 1) -- Equation from working together and P finishing
  : T = 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3217_321723


namespace NUMINAMATH_CALUDE_det_max_value_l3217_321774

open Real Matrix

theorem det_max_value (θ : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1, 1 + sin θ, 1; 1, 1, 1 + cos θ]
  ∀ φ : ℝ, det A ≤ det (!![1, 1, 1; 1, 1 + sin φ, 1; 1, 1, 1 + cos φ]) → det A ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_det_max_value_l3217_321774


namespace NUMINAMATH_CALUDE_root_transformation_l3217_321702

theorem root_transformation (α β : ℝ) : 
  (2 * α^2 - 5 * α + 3 = 0) → 
  (2 * β^2 - 5 * β + 3 = 0) → 
  ((2 * α - 7)^2 + 9 * (2 * α - 7) + 20 = 0) ∧
  ((2 * β - 7)^2 + 9 * (2 * β - 7) + 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l3217_321702


namespace NUMINAMATH_CALUDE_multiplication_and_exponentiation_l3217_321768

theorem multiplication_and_exponentiation : 121 * (5^4) = 75625 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_exponentiation_l3217_321768


namespace NUMINAMATH_CALUDE_dana_wins_l3217_321739

/-- Represents a player in the game -/
inductive Player
  | Carl
  | Dana
  | Leah

/-- Represents the state of the game -/
structure GameState where
  chosenNumbers : List ℝ
  currentPlayer : Player

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ 10 ∧
  ∀ n ∈ state.chosenNumbers, |move - n| ≥ 2

/-- Defines the next player in the turn order -/
def nextPlayer : Player → Player
  | Player.Carl => Player.Dana
  | Player.Dana => Player.Leah
  | Player.Leah => Player.Carl

/-- Represents a winning strategy for a player -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ initialState : GameState,
    initialState.currentPlayer = player →
    ∃ (strategy : GameState → ℝ),
      ∀ gameSequence : List ℝ,
        (∀ move ∈ gameSequence, isValidMove initialState move) →
        (∃ finalState : GameState,
          finalState.chosenNumbers = initialState.chosenNumbers ++ gameSequence ∧
          finalState.currentPlayer = player ∧
          ¬∃ move, isValidMove finalState move)

/-- The main theorem stating that Dana has a winning strategy -/
theorem dana_wins : hasWinningStrategy Player.Dana := by
  sorry

end NUMINAMATH_CALUDE_dana_wins_l3217_321739


namespace NUMINAMATH_CALUDE_circle_equation_l3217_321710

-- Define the center and radius of the circle
def center : ℝ × ℝ := (2, -1)
def radius : ℝ := 4

-- State the theorem
theorem circle_equation :
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
  (x - 2)^2 + (y + 1)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3217_321710


namespace NUMINAMATH_CALUDE_ones_digit_of_6_pow_45_l3217_321780

theorem ones_digit_of_6_pow_45 : ∃ n : ℕ, 6^45 ≡ 6 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_6_pow_45_l3217_321780


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_a_and_b_l3217_321729

theorem arithmetic_mean_of_a_and_b (a b : ℝ) : 
  a = Real.sqrt 3 + Real.sqrt 2 → 
  b = Real.sqrt 3 - Real.sqrt 2 → 
  (a + b) / 2 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_a_and_b_l3217_321729


namespace NUMINAMATH_CALUDE_power_sum_difference_l3217_321733

theorem power_sum_difference : 2^(0+1+2) - (2^0 + 2^1 + 2^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l3217_321733


namespace NUMINAMATH_CALUDE_average_daily_high_temp_l3217_321738

def daily_highs : List ℝ := [51, 63, 59, 56, 47, 64, 52]

theorem average_daily_high_temp : 
  (daily_highs.sum / daily_highs.length : ℝ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_high_temp_l3217_321738


namespace NUMINAMATH_CALUDE_min_value_of_t_l3217_321797

theorem min_value_of_t (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_t_l3217_321797


namespace NUMINAMATH_CALUDE_mark_spent_40_dollars_l3217_321731

/-- The total amount Mark spent on tomatoes and apples -/
def total_spent (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) : ℝ :=
  tomato_price * tomato_weight + apple_price * apple_weight

/-- Proof that Mark spent $40 in total -/
theorem mark_spent_40_dollars : total_spent 5 2 6 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_mark_spent_40_dollars_l3217_321731


namespace NUMINAMATH_CALUDE_carols_weight_l3217_321735

theorem carols_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 240)
  (h2 : carol_weight - alice_weight = 2/3 * carol_weight) : 
  carol_weight = 180 := by
sorry

end NUMINAMATH_CALUDE_carols_weight_l3217_321735


namespace NUMINAMATH_CALUDE_B_power_100_l3217_321753

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_100 : B ^ 100 = B := by sorry

end NUMINAMATH_CALUDE_B_power_100_l3217_321753


namespace NUMINAMATH_CALUDE_triangle_point_distance_l3217_321779

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_point_distance (ABC : Triangle) (D E : ℝ × ℝ) :
  -- Given conditions
  (ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2 = 17^2 →
  (ABC.B.1 - ABC.C.1)^2 + (ABC.B.2 - ABC.C.2)^2 = 19^2 →
  (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 = 16^2 →
  PointOnSegment D ABC.B ABC.C →
  PointOnSegment E ABC.B ABC.C →
  (D.1 - ABC.B.1)^2 + (D.2 - ABC.B.2)^2 = 7^2 →
  Angle ABC.B ABC.A E = Angle ABC.C ABC.A D →
  -- Conclusion
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = (-251/41)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_distance_l3217_321779


namespace NUMINAMATH_CALUDE_select_medical_team_eq_630_l3217_321756

/-- The number of ways to select a medical team for earthquake relief. -/
def select_medical_team : ℕ :=
  let orthopedic : ℕ := 3
  let neurosurgeon : ℕ := 4
  let internist : ℕ := 5
  let team_size : ℕ := 5
  
  -- Combinations for each possible selection scenario
  let scenario1 := Nat.choose orthopedic 3 * Nat.choose neurosurgeon 1 * Nat.choose internist 1
  let scenario2 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 3 * Nat.choose internist 1
  let scenario3 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 1 * Nat.choose internist 3
  let scenario4 := Nat.choose orthopedic 2 * Nat.choose neurosurgeon 2 * Nat.choose internist 1
  let scenario5 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 2 * Nat.choose internist 2
  let scenario6 := Nat.choose orthopedic 2 * Nat.choose neurosurgeon 1 * Nat.choose internist 2

  -- Sum of all scenarios
  scenario1 + scenario2 + scenario3 + scenario4 + scenario5 + scenario6

/-- Theorem stating that the number of ways to select the medical team is 630. -/
theorem select_medical_team_eq_630 : select_medical_team = 630 := by
  sorry

end NUMINAMATH_CALUDE_select_medical_team_eq_630_l3217_321756


namespace NUMINAMATH_CALUDE_problem_solution_l3217_321798

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem problem_solution (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (3*x + 10) = f (3*x + 1))
  (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3217_321798


namespace NUMINAMATH_CALUDE_fraction_simplification_l3217_321772

theorem fraction_simplification : 
  (45 : ℚ) / 28 * 49 / 75 * 100 / 63 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3217_321772


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3217_321769

theorem sqrt_equation_solution :
  ∃! x : ℝ, 4 * x - 3 ≥ 0 ∧ Real.sqrt (4 * x - 3) + 16 / Real.sqrt (4 * x - 3) = 8 :=
by
  -- The unique solution is x = 19/4
  use 19/4
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3217_321769


namespace NUMINAMATH_CALUDE_complex_power_30_150_deg_l3217_321707

theorem complex_power_30_150_deg : (Complex.exp (Complex.I * Real.pi * (5/6)))^30 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_30_150_deg_l3217_321707


namespace NUMINAMATH_CALUDE_fliers_calculation_l3217_321736

theorem fliers_calculation (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (4/5) * (3/4) = 1800 → initial_fliers = 3000 := by
  sorry

end NUMINAMATH_CALUDE_fliers_calculation_l3217_321736


namespace NUMINAMATH_CALUDE_greater_a_than_c_l3217_321701

theorem greater_a_than_c (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : (a - b) * (b - c) * (c - a) > 0) : 
  a > c := by
  sorry

end NUMINAMATH_CALUDE_greater_a_than_c_l3217_321701


namespace NUMINAMATH_CALUDE_complex_power_of_four_l3217_321794

theorem complex_power_of_four : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_power_of_four_l3217_321794


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3217_321748

theorem condition_necessary_not_sufficient (a b : ℝ) :
  (a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  ¬(a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3217_321748


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3217_321755

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x : ℝ, a * (x - 1)^2 + 3 = a * x^2 + b * x + c) →
  a * 0^2 + b * 0 + c = 1 →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3217_321755


namespace NUMINAMATH_CALUDE_van_distance_proof_l3217_321760

theorem van_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 5 →
  new_speed = 58 →
  ∃ (distance : ℝ),
    distance = new_speed * (3/2 * initial_time) ∧
    distance = 435 :=
by sorry

end NUMINAMATH_CALUDE_van_distance_proof_l3217_321760


namespace NUMINAMATH_CALUDE_right_triangle_coordinate_l3217_321709

/-- Given a right triangle ABC with vertices A(0, 0), B(0, 4a - 2), and C(x, 4a - 2),
    if the area of the triangle is 63, then the x-coordinate of point C is 126 / (4a - 2). -/
theorem right_triangle_coordinate (a : ℝ) (x : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 4 * a - 2)
  let C : ℝ × ℝ := (x, 4 * a - 2)
  (4 * a - 2 ≠ 0) →
  (1 / 2 : ℝ) * x * (4 * a - 2) = 63 →
  x = 126 / (4 * a - 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_coordinate_l3217_321709


namespace NUMINAMATH_CALUDE_solve_for_m_l3217_321741

theorem solve_for_m : ∃ m : ℕ, (2022^2 - 4) * (2021^2 - 4) = 2024 * 2020 * 2019 * m ∧ m = 2023 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3217_321741


namespace NUMINAMATH_CALUDE_square_root_of_25_squared_l3217_321745

theorem square_root_of_25_squared : Real.sqrt 25 ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_squared_l3217_321745


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_plus_abs_inequality_l3217_321737

def f (x : ℝ) := |3 * x + 1| - |x - 4|

theorem f_inequality_solution (x : ℝ) :
  f x < 0 ↔ -5/2 < x ∧ x < 3/4 := by sorry

theorem f_plus_abs_inequality (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) ↔ m < 15 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_plus_abs_inequality_l3217_321737


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3217_321743

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (Real.cos x)) / x else 0

theorem derivative_f_at_zero :
  deriv f 0 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3217_321743


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3217_321776

def M : Set Int := {-1, 3, 5}
def N : Set Int := {-1, 0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3217_321776


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3217_321791

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-3, 4)
  let reflected := reflect_x initial_center
  let final_center := translate_up reflected 5
  final_center = (-3, 1) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3217_321791


namespace NUMINAMATH_CALUDE_push_ups_total_l3217_321789

theorem push_ups_total (david_pushups : ℕ) (difference : ℕ) : 
  david_pushups = 51 → difference = 49 → 
  david_pushups + (david_pushups - difference) = 53 := by
  sorry

end NUMINAMATH_CALUDE_push_ups_total_l3217_321789


namespace NUMINAMATH_CALUDE_power_seven_700_mod_100_l3217_321727

theorem power_seven_700_mod_100 : 7^700 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_700_mod_100_l3217_321727


namespace NUMINAMATH_CALUDE_linear_implies_constant_derivative_constant_derivative_not_sufficient_for_linear_l3217_321728

-- Define a linear function
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define a constant derivative
def has_constant_derivative (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x, deriv f x = c

theorem linear_implies_constant_derivative :
  ∀ f : ℝ → ℝ, is_linear f → has_constant_derivative f :=
sorry

theorem constant_derivative_not_sufficient_for_linear :
  ∃ f : ℝ → ℝ, has_constant_derivative f ∧ ¬is_linear f :=
sorry

end NUMINAMATH_CALUDE_linear_implies_constant_derivative_constant_derivative_not_sufficient_for_linear_l3217_321728


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l3217_321746

theorem smallest_n_for_exact_tax : ∃ (x : ℕ+), (104 * x : ℚ) / 10000 = 13 ∧
  ∀ (n : ℕ+), n < 13 → ¬∃ (y : ℕ+), (104 * y : ℚ) / 10000 = n := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l3217_321746


namespace NUMINAMATH_CALUDE_min_max_values_l3217_321771

theorem min_max_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → Real.sqrt x + Real.sqrt y ≥ Real.sqrt a + Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l3217_321771


namespace NUMINAMATH_CALUDE_division_remainder_l3217_321725

theorem division_remainder (dividend quotient divisor remainder : ℕ) : 
  dividend = 507 → 
  quotient = 61 → 
  divisor = 8 → 
  dividend = divisor * quotient + remainder → 
  remainder = 19 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3217_321725


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3217_321749

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 4 * y + 4 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -3)

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y + 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
    (∃ (m : ℝ), perpendicular_line x y ∧
      (∀ (x' y' : ℝ), given_line x' y' → (y - point.2 = m * (x - point.1))) ∧
      (m * (4 / 3) = -1)) →
    perpendicular_line x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3217_321749


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_five_l3217_321788

theorem sum_of_roots_equals_five : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, x ≠ 3 ∧ (x^3 - 3*x^2 - 12*x) / (x - 3) = 6) ∧ 
    (∀ x ∉ S, x = 3 ∨ (x^3 - 3*x^2 - 12*x) / (x - 3) ≠ 6) ∧
    (Finset.sum S id = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_five_l3217_321788


namespace NUMINAMATH_CALUDE_sqrt_64_equals_8_l3217_321752

theorem sqrt_64_equals_8 : Real.sqrt 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_equals_8_l3217_321752


namespace NUMINAMATH_CALUDE_sticker_remainder_l3217_321720

theorem sticker_remainder (a b c : ℤ) 
  (ha : a % 5 = 1)
  (hb : b % 5 = 4)
  (hc : c % 5 = 3) : 
  (a + b + c) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sticker_remainder_l3217_321720


namespace NUMINAMATH_CALUDE_bert_equals_kameron_in_40_days_l3217_321703

/-- The number of days required for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos : ℕ) (bert_kangaroos : ℕ) (bert_buying_rate : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_buying_rate

/-- Proof that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem bert_equals_kameron_in_40_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bert_equals_kameron_in_40_days_l3217_321703


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3217_321762

/-- Proves that for a hyperbola with equation x²/a² - y²/b² = 1, where a > b, 
    if the angle between the asymptotes is 45°, then a/b = 1/(-1 + √2). -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 4 = Real.arctan ((2 * b / a) / (1 - (b / a)^2))) →
  a / b = 1 / (-1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3217_321762


namespace NUMINAMATH_CALUDE_inequality_proof_l3217_321765

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_prod : a * b * c = 1) : 
  (a + 1/b)^2 + (b + 1/c)^2 + (c + 1/a)^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3217_321765


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3217_321786

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  9 * x^2 + 36 * x + 4 * y^2 - 8 * y + 20 = 0

/-- The distance between the foci of the ellipse -/
def foci_distance : ℝ := 0

/-- Theorem stating that the distance between the foci of the given ellipse is 0 -/
theorem ellipse_foci_distance :
  ∀ x y : ℝ, ellipse_equation x y → foci_distance = 0 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3217_321786


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l3217_321718

theorem ceiling_product_equation : ∃! x : ℝ, ⌈x⌉ * x = 168 ∧ x = 168 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l3217_321718


namespace NUMINAMATH_CALUDE_dining_bill_share_l3217_321757

def total_bill : ℝ := 211.00
def num_people : ℕ := 9
def tip_percentage : ℝ := 0.15

theorem dining_bill_share :
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / num_people
  ∃ ε > 0, |share_per_person - 26.96| < ε :=
by sorry

end NUMINAMATH_CALUDE_dining_bill_share_l3217_321757


namespace NUMINAMATH_CALUDE_lily_newspaper_collection_l3217_321761

/-- Given that Chris collected 42 newspapers and the total number of newspapers
    collected by Chris and Lily is 65, prove that Lily collected 23 newspapers. -/
theorem lily_newspaper_collection (chris_newspapers lily_newspapers total_newspapers : ℕ) :
  chris_newspapers = 42 →
  total_newspapers = 65 →
  total_newspapers = chris_newspapers + lily_newspapers →
  lily_newspapers = 23 := by
sorry

end NUMINAMATH_CALUDE_lily_newspaper_collection_l3217_321761


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l3217_321721

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C

/-- The theorem stating that A = π/3 given the condition -/
theorem angle_A_value (t : Triangle) (h : triangle_condition t) : t.A = π / 3 :=
sorry

/-- The theorem stating the maximum area when a = 4 -/
theorem max_area (t : Triangle) (h : triangle_condition t) (ha : t.a = 4) :
  (∀ t' : Triangle, triangle_condition t' → t'.a = 4 → 
    t'.b * t'.c * Real.sin t'.A / 2 ≤ 4 * Real.sqrt 3) ∧
  (∃ t' : Triangle, triangle_condition t' ∧ t'.a = 4 ∧ 
    t'.b * t'.c * Real.sin t'.A / 2 = 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l3217_321721


namespace NUMINAMATH_CALUDE_subset_condition_l3217_321730

def P : Set ℝ := {x | x^2 ≠ 4}
def Q (a : ℝ) : Set ℝ := {x | a * x = 4}

theorem subset_condition (a : ℝ) : Q a ⊆ P ↔ a = 0 ∨ a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3217_321730


namespace NUMINAMATH_CALUDE_tangent_line_angle_l3217_321734

open Real

theorem tangent_line_angle (n : ℤ) : 
  let M : ℝ × ℝ := (7, 1)
  let O : ℝ × ℝ := (4, 4)
  let r : ℝ := 2
  let MO : ℝ × ℝ := (O.1 - M.1, O.2 - M.2)
  let MO_length : ℝ := Real.sqrt ((MO.1)^2 + (MO.2)^2)
  let MO_angle : ℝ := Real.arctan (MO.2 / MO.1) + π
  let φ : ℝ := Real.arcsin (r / MO_length)
  ∃ (a : ℝ), a = MO_angle - φ + n * π ∨ a = MO_angle + φ + n * π := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_angle_l3217_321734


namespace NUMINAMATH_CALUDE_repair_time_30_workers_l3217_321770

/-- Represents the time taken to complete a road repair job given the number of workers -/
def repair_time (num_workers : ℕ) : ℚ :=
  3 * 45 / num_workers

/-- Proves that 30 workers would take 4.5 days to complete the road repair -/
theorem repair_time_30_workers :
  repair_time 30 = 4.5 := by sorry

end NUMINAMATH_CALUDE_repair_time_30_workers_l3217_321770


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3217_321787

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a*b + b*c + c*a = 72) : 
  a + b + c = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3217_321787


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_l3217_321784

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by (a b : ℕ) : Prop := b ∣ a

theorem largest_three_digit_divisible_by_6 :
  ∀ n : ℕ, is_three_digit n → divisible_by n 6 → n ≤ 996 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_l3217_321784


namespace NUMINAMATH_CALUDE_lcm_16_24_l3217_321796

theorem lcm_16_24 : Nat.lcm 16 24 = 48 := by sorry

end NUMINAMATH_CALUDE_lcm_16_24_l3217_321796


namespace NUMINAMATH_CALUDE_james_original_weight_l3217_321747

/-- Proves that given the conditions of James's weight gain, his original weight was 120 kg -/
theorem james_original_weight :
  ∀ W : ℝ,
  W > 0 →
  let muscle_gain := 0.20 * W
  let fat_gain := 0.25 * muscle_gain
  let final_weight := W + muscle_gain + fat_gain
  final_weight = 150 →
  W = 120 := by
sorry

end NUMINAMATH_CALUDE_james_original_weight_l3217_321747


namespace NUMINAMATH_CALUDE_d_equals_four_l3217_321724

/-- A nine-digit number with specific properties -/
structure NineDigitNumber where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ
  G : ℕ
  first_three_sum : 6 + A + B = 13
  second_three_sum : A + B + C = 13
  third_three_sum : B + C + D = 13
  fourth_three_sum : C + D + E = 13
  fifth_three_sum : D + E + F = 13
  sixth_three_sum : E + F + G = 13
  last_three_sum : F + G + 3 = 13

/-- The digit D in the number must be 4 -/
theorem d_equals_four (n : NineDigitNumber) : n.D = 4 := by
  sorry

end NUMINAMATH_CALUDE_d_equals_four_l3217_321724


namespace NUMINAMATH_CALUDE_at_most_two_distinct_values_l3217_321783

theorem at_most_two_distinct_values (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (sum_squares_eq : a^2 + b^2 = c^2 + d^2) : 
  ∃ (x y : ℝ), (a = x ∨ a = y) ∧ (b = x ∨ b = y) ∧ (c = x ∨ c = y) ∧ (d = x ∨ d = y) :=
by sorry

end NUMINAMATH_CALUDE_at_most_two_distinct_values_l3217_321783


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3217_321711

theorem quadratic_factorization_sum (a b c d : ℤ) : 
  (∀ x : ℚ, 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) →
  |a| + |b| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3217_321711


namespace NUMINAMATH_CALUDE_water_speed_calculation_l3217_321706

/-- A person's swimming speed in still water (in km/h) -/
def still_water_speed : ℝ := 4

/-- The time taken to swim against the current (in hours) -/
def time_against_current : ℝ := 3

/-- The distance swum against the current (in km) -/
def distance_against_current : ℝ := 6

/-- The speed of the water (in km/h) -/
def water_speed : ℝ := 2

theorem water_speed_calculation :
  (distance_against_current = (still_water_speed - water_speed) * time_against_current) →
  water_speed = 2 := by
sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l3217_321706


namespace NUMINAMATH_CALUDE_symmetry_about_origin_l3217_321705

/-- A point on the graph of y = 3^x -/
structure PointOn3x where
  x : ℝ
  y : ℝ
  h : y = 3^x

/-- A point on the graph of y = -3^(-x) -/
structure PointOnNeg3NegX where
  x : ℝ
  y : ℝ
  h : y = -3^(-x)

/-- The condition given in the problem -/
axiom symmetry_condition {p : PointOn3x} :
  ∃ (q : PointOnNeg3NegX), q.x = -p.x ∧ q.y = -p.y

/-- The theorem to be proved -/
theorem symmetry_about_origin :
  ∀ (p : PointOn3x), ∃ (q : PointOnNeg3NegX), q.x = -p.x ∧ q.y = -p.y :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_origin_l3217_321705


namespace NUMINAMATH_CALUDE_first_worker_time_l3217_321700

/-- Given two workers loading a truck, prove that the first worker's time is 5 hours. -/
theorem first_worker_time (T : ℝ) : 
  T > 0 →  -- The time must be positive
  (1 / T + 1 / 4 : ℝ) = 1 / 2.2222222222222223 → 
  T = 5 := by 
sorry

end NUMINAMATH_CALUDE_first_worker_time_l3217_321700


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3217_321751

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  n % 17 = 0 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∀ m : ℕ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3217_321751


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3217_321740

theorem geometric_progression_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 15)
  (h2 : sum_first_two = 10) :
  ∃ (a : ℝ), (a = (15 * (Real.sqrt 3 - 1)) / Real.sqrt 3 ∨
              a = (15 * (Real.sqrt 3 + 1)) / Real.sqrt 3) ∧
             (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3217_321740


namespace NUMINAMATH_CALUDE_s_equality_l3217_321764

theorem s_equality (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_s_equality_l3217_321764


namespace NUMINAMATH_CALUDE_shorter_base_length_l3217_321716

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_segment : ℝ

/-- The trapezoid satisfies the given conditions -/
def trapezoid_conditions (t : Trapezoid) : Prop :=
  t.long_base = 125 ∧ t.midpoint_segment = 5

/-- Theorem: In a trapezoid satisfying the given conditions, the shorter base is 115 -/
theorem shorter_base_length (t : Trapezoid) (h : trapezoid_conditions t) : 
  t.short_base = 115 := by
  sorry

#check shorter_base_length

end NUMINAMATH_CALUDE_shorter_base_length_l3217_321716


namespace NUMINAMATH_CALUDE_nell_ace_cards_l3217_321773

/-- The number of baseball cards Nell has now -/
def baseball_cards : ℕ := 178

/-- The difference between baseball cards and Ace cards Nell has now -/
def difference : ℕ := 123

/-- Theorem: The number of Ace cards Nell has now is 55 -/
theorem nell_ace_cards : 
  ∃ (ace_cards : ℕ), ace_cards = baseball_cards - difference ∧ ace_cards = 55 := by
  sorry

end NUMINAMATH_CALUDE_nell_ace_cards_l3217_321773
