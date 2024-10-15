import Mathlib

namespace NUMINAMATH_CALUDE_pencil_gain_percentage_l3018_301877

/-- Represents the cost price of a single pencil in rupees -/
def cost_price_per_pencil : ℚ := 1 / 12

/-- Represents the selling price of 15 pencils in rupees -/
def selling_price_15 : ℚ := 1

/-- Represents the selling price of 10 pencils in rupees -/
def selling_price_10 : ℚ := 1

/-- The loss percentage when selling 15 pencils for a rupee -/
def loss_percentage : ℚ := 20 / 100

theorem pencil_gain_percentage :
  let cost_15 := 15 * cost_price_per_pencil
  let cost_10 := 10 * cost_price_per_pencil
  selling_price_15 = (1 - loss_percentage) * cost_15 →
  (selling_price_10 - cost_10) / cost_10 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_gain_percentage_l3018_301877


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3018_301853

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection_ways : choose 12 5 = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3018_301853


namespace NUMINAMATH_CALUDE_xyz_equation_solutions_l3018_301885

theorem xyz_equation_solutions :
  ∀ (x y z : ℕ), x * y * z = x + y → ((x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2)) :=
by sorry

end NUMINAMATH_CALUDE_xyz_equation_solutions_l3018_301885


namespace NUMINAMATH_CALUDE_total_nails_needed_l3018_301839

/-- The total number of nails needed is equal to the sum of initial nails, 
    found nails, and nails to buy. -/
theorem total_nails_needed 
  (initial_nails : ℕ) 
  (found_nails : ℕ) 
  (nails_to_buy : ℕ) : 
  initial_nails + found_nails + nails_to_buy = 
  initial_nails + found_nails + nails_to_buy := by
  sorry

#eval 247 + 144 + 109

end NUMINAMATH_CALUDE_total_nails_needed_l3018_301839


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3018_301851

theorem modulus_of_complex_fraction :
  let z : ℂ := (-3 + Complex.I) / (2 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3018_301851


namespace NUMINAMATH_CALUDE_chemistry_is_other_subject_l3018_301886

/-- Represents the scores in three subjects -/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The conditions of the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.physics = 110 ∧
  (s.physics + s.chemistry + s.mathematics) / 3 = 70 ∧
  (s.physics + s.mathematics) / 2 = 90 ∧
  (s.physics + s.chemistry) / 2 = 70

/-- The theorem to be proved -/
theorem chemistry_is_other_subject (s : Scores) :
  satisfiesConditions s → (s.physics + s.chemistry) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_is_other_subject_l3018_301886


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3018_301822

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 8 * x + k = 0) ↔ k = 16/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3018_301822


namespace NUMINAMATH_CALUDE_distinct_integers_count_l3018_301835

def odd_squares_list : List ℤ :=
  (List.range 500).map (fun k => ⌊((2*k + 1)^2 : ℚ) / 500⌋)

theorem distinct_integers_count : (odd_squares_list.eraseDups).length = 469 := by
  sorry

end NUMINAMATH_CALUDE_distinct_integers_count_l3018_301835


namespace NUMINAMATH_CALUDE_beach_waders_l3018_301841

/-- Proves that 3 people from the first row got up to wade in the water, given the conditions of the beach scenario. -/
theorem beach_waders (first_row : ℕ) (second_row : ℕ) (third_row : ℕ) 
  (h1 : first_row = 24)
  (h2 : second_row = 20)
  (h3 : third_row = 18)
  (h4 : ∃ x : ℕ, first_row - x + (second_row - 5) + third_row = 54) :
  ∃ x : ℕ, x = 3 ∧ first_row - x + (second_row - 5) + third_row = 54 :=
by sorry

end NUMINAMATH_CALUDE_beach_waders_l3018_301841


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3018_301898

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = -5) → 
  (3 : ℝ) ∈ {x : ℝ | 3 * x^2 + k * x = -5} → 
  (5/9 : ℝ) ∈ {x : ℝ | 3 * x^2 + k * x = -5} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3018_301898


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l3018_301843

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  a : ℝ  -- First diagonal of the base
  b : ℝ  -- Second diagonal of the base
  sphere_inscribed : Bool  -- Indicator that a sphere is inscribed

-- Define the total surface area function
def total_surface_area (p : RectangularParallelepiped) : ℝ :=
  3 * p.a * p.b

-- Theorem statement
theorem parallelepiped_surface_area 
  (p : RectangularParallelepiped) 
  (h : p.sphere_inscribed = true) :
  total_surface_area p = 3 * p.a * p.b :=
by
  sorry


end NUMINAMATH_CALUDE_parallelepiped_surface_area_l3018_301843


namespace NUMINAMATH_CALUDE_contractor_payment_proof_l3018_301809

/-- Calculates the total amount received by a contractor given the contract terms and absence information. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  (total_days - absent_days : ℕ) * payment_per_day - absent_days * fine_per_day

/-- Proves that the contractor receives Rs. 555 given the specified conditions. -/
theorem contractor_payment_proof :
  contractor_payment 30 25 (15/2) 6 = 555 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_proof_l3018_301809


namespace NUMINAMATH_CALUDE_jerry_water_usage_l3018_301816

/-- Calculates the total water usage for Jerry's household in July --/
def total_water_usage (drinking_cooking : ℕ) (shower_usage : ℕ) (num_showers : ℕ) 
  (pool_length : ℕ) (pool_width : ℕ) (pool_height : ℕ) : ℕ :=
  drinking_cooking + (shower_usage * num_showers) + (pool_length * pool_width * pool_height)

theorem jerry_water_usage :
  total_water_usage 100 20 15 10 10 6 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jerry_water_usage_l3018_301816


namespace NUMINAMATH_CALUDE_minimize_sqrt_difference_l3018_301836

theorem minimize_sqrt_difference (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (x y : ℕ), 
    (x > 0 ∧ y > 0) ∧
    (x ≤ y) ∧
    (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0) ∧
    (∀ (a b : ℕ), (a > 0 ∧ b > 0) → (a ≤ b) → 
      (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) →
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b)) ∧
    (x = (p - 1) / 2) ∧
    (y = (p + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_minimize_sqrt_difference_l3018_301836


namespace NUMINAMATH_CALUDE_pig_price_calculation_l3018_301840

/-- Given the total cost of 3 pigs and 10 hens, and the average price of a hen,
    calculate the average price of a pig. -/
theorem pig_price_calculation (total_cost hen_price : ℚ) 
    (h1 : total_cost = 1200)
    (h2 : hen_price = 30) : 
    (total_cost - 10 * hen_price) / 3 = 300 :=
by sorry

end NUMINAMATH_CALUDE_pig_price_calculation_l3018_301840


namespace NUMINAMATH_CALUDE_food_fraction_is_one_fifth_l3018_301821

def salary : ℚ := 150000.00000000003
def house_rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5
def amount_left : ℚ := 15000

theorem food_fraction_is_one_fifth :
  let food_fraction := 1 - house_rent_fraction - clothes_fraction - amount_left / salary
  food_fraction = 1/5 := by sorry

end NUMINAMATH_CALUDE_food_fraction_is_one_fifth_l3018_301821


namespace NUMINAMATH_CALUDE_jellybean_distribution_l3018_301862

theorem jellybean_distribution (total_jellybeans : ℕ) (nephews : ℕ) (nieces : ℕ) 
  (h1 : total_jellybeans = 70)
  (h2 : nephews = 3)
  (h3 : nieces = 2) :
  total_jellybeans / (nephews + nieces) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_distribution_l3018_301862


namespace NUMINAMATH_CALUDE_fruit_basket_combinations_l3018_301856

def num_apple_options : ℕ := 7
def num_orange_options : ℕ := 13

def total_combinations : ℕ := num_apple_options * num_orange_options

theorem fruit_basket_combinations :
  total_combinations - 1 = 90 := by sorry

end NUMINAMATH_CALUDE_fruit_basket_combinations_l3018_301856


namespace NUMINAMATH_CALUDE_horner_method_f_2_l3018_301892

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_method_f_2 : f 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l3018_301892


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3018_301871

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3018_301871


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l3018_301812

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → -12 < y ∧ y < -11 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l3018_301812


namespace NUMINAMATH_CALUDE_jenny_recycling_money_is_160_l3018_301883

/-- Calculates the money Jenny makes from recycling cans and bottles -/
def jenny_recycling_money : ℕ :=
let bottle_weight : ℕ := 6
let can_weight : ℕ := 2
let total_capacity : ℕ := 100
let cans_collected : ℕ := 20
let bottle_price : ℕ := 10
let can_price : ℕ := 3
let remaining_capacity : ℕ := total_capacity - (can_weight * cans_collected)
let bottles_collected : ℕ := remaining_capacity / bottle_weight
bottles_collected * bottle_price + cans_collected * can_price

theorem jenny_recycling_money_is_160 :
  jenny_recycling_money = 160 := by
sorry

end NUMINAMATH_CALUDE_jenny_recycling_money_is_160_l3018_301883


namespace NUMINAMATH_CALUDE_height_is_four_l3018_301895

/-- The configuration of squares with a small square of area 1 -/
structure SquareConfiguration where
  /-- The side length of the second square -/
  a : ℝ
  /-- The height to be determined -/
  h : ℝ
  /-- The small square has area 1 -/
  small_square_area : 1 = 1
  /-- The equation relating the squares and height -/
  square_relation : 1 + a + 3 = a + h

/-- The theorem stating that h = 4 in the given square configuration -/
theorem height_is_four (config : SquareConfiguration) : config.h = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_is_four_l3018_301895


namespace NUMINAMATH_CALUDE_unique_solution_linear_equation_l3018_301845

theorem unique_solution_linear_equation (a b c : ℝ) (h1 : c ≠ 0) (h2 : b ≠ 2) :
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_linear_equation_l3018_301845


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_fraction_l3018_301819

theorem simplify_and_evaluate_fraction (a : ℝ) (h : a = 5) :
  (a^2 - 4) / a^2 / (1 - 2/a) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_fraction_l3018_301819


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3018_301830

/-- Theorem: Movie Ticket Cost
  Given:
  - Movie tickets cost M on Monday
  - Wednesday tickets cost 2M
  - Saturday tickets cost 5M
  - Total cost for Wednesday and Saturday is $35
  Prove: M = 5
-/
theorem movie_ticket_cost (M : ℚ) : 2 * M + 5 * M = 35 → M = 5 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l3018_301830


namespace NUMINAMATH_CALUDE_boatworks_production_l3018_301847

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boatworks_production : geometric_sum 5 3 6 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_boatworks_production_l3018_301847


namespace NUMINAMATH_CALUDE_hannah_easter_eggs_l3018_301868

theorem hannah_easter_eggs 
  (total : ℕ) 
  (helen : ℕ) 
  (hannah : ℕ) 
  (h1 : total = 63)
  (h2 : hannah = 2 * helen)
  (h3 : total = helen + hannah) : 
  hannah = 42 := by
sorry

end NUMINAMATH_CALUDE_hannah_easter_eggs_l3018_301868


namespace NUMINAMATH_CALUDE_sequence_formula_l3018_301820

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 1 - n * a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 1 - n * a n) : 
  ∀ n : ℕ+, a n = 1 / (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3018_301820


namespace NUMINAMATH_CALUDE_sum_of_squares_l3018_301873

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 3 → 
  (a - 1)^3 + (b - 1)^3 + (c - 1)^3 = 0 → 
  a = 2 → 
  a^2 + b^2 + c^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3018_301873


namespace NUMINAMATH_CALUDE_map_scale_conversion_l3018_301849

/-- Given a map scale where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale_conversion (scale : ℝ → ℝ) (h1 : scale 15 = 90) : scale 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l3018_301849


namespace NUMINAMATH_CALUDE_double_volume_double_capacity_l3018_301858

/-- Represents the capacity of a container in number of marbles -/
def ContainerCapacity (volume : ℝ) : ℝ := sorry

theorem double_volume_double_capacity :
  let v₁ : ℝ := 36
  let v₂ : ℝ := 72
  let c₁ : ℝ := 120
  ContainerCapacity v₁ = c₁ →
  ContainerCapacity v₂ = 2 * c₁ :=
by sorry

end NUMINAMATH_CALUDE_double_volume_double_capacity_l3018_301858


namespace NUMINAMATH_CALUDE_inverse_proposition_correct_l3018_301814

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define what it means for two lines to be parallel
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Define what it means for angles to be supplementary
def supplementary_angles (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

-- Define the original proposition
def original_proposition (l₁ l₂ : Line) (θ₁ θ₂ : ℝ) : Prop :=
  parallel l₁ l₂ → supplementary_angles θ₁ θ₂

-- Define the inverse proposition
def inverse_proposition (l₁ l₂ : Line) (θ₁ θ₂ : ℝ) : Prop :=
  supplementary_angles θ₁ θ₂ → parallel l₁ l₂

-- Theorem stating that the inverse proposition is correct
theorem inverse_proposition_correct :
  ∀ (l₁ l₂ : Line) (θ₁ θ₂ : ℝ),
    inverse_proposition l₁ l₂ θ₁ θ₂ =
    (supplementary_angles θ₁ θ₂ → parallel l₁ l₂) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_correct_l3018_301814


namespace NUMINAMATH_CALUDE_no_real_solutions_l3018_301876

theorem no_real_solutions :
  ¬∃ (x : ℝ), x ≠ -9 ∧ (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3018_301876


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3018_301803

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 4 ↔ (x : ℚ) / 4 + 3 / 5 < 7 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3018_301803


namespace NUMINAMATH_CALUDE_sum_inverse_max_min_S_l3018_301882

/-- Given real numbers x and y satisfying 4x^2 - 5xy + 4y^2 = 5,
    and S defined as x^2 + y^2, prove that the maximum and minimum
    values of S exist, and 1/S_max + 1/S_min = 8/5. -/
theorem sum_inverse_max_min_S :
  ∃ (S_max S_min : ℝ),
    (∀ x y : ℝ, 4 * x^2 - 5 * x * y + 4 * y^2 = 5 →
      let S := x^2 + y^2
      S ≤ S_max ∧ S_min ≤ S) ∧
    1 / S_max + 1 / S_min = 8 / 5 := by
  sorry

#check sum_inverse_max_min_S

end NUMINAMATH_CALUDE_sum_inverse_max_min_S_l3018_301882


namespace NUMINAMATH_CALUDE_vector_b_values_l3018_301833

/-- Given two vectors a and b in ℝ², where a = (2,1), |b| = 2√5, and a is parallel to b,
    prove that b is either (-4,-2) or (4,2) -/
theorem vector_b_values (a b : ℝ × ℝ) : 
  a = (2, 1) → 
  ‖b‖ = 2 * Real.sqrt 5 →
  ∃ (k : ℝ), b = k • a →
  b = (-4, -2) ∨ b = (4, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_b_values_l3018_301833


namespace NUMINAMATH_CALUDE_relay_race_time_l3018_301854

/-- Represents the time taken by each runner in the relay race -/
structure RelayTimes where
  mary : ℕ
  susan : ℕ
  jen : ℕ
  tiffany : ℕ

/-- Calculates the total time of the relay race -/
def total_time (times : RelayTimes) : ℕ :=
  times.mary + times.susan + times.jen + times.tiffany

/-- Theorem stating that the total time of the relay race is 223 seconds -/
theorem relay_race_time : ∃ (times : RelayTimes), 
  times.mary = 2 * times.susan ∧
  times.susan = times.jen + 10 ∧
  times.jen = 30 ∧
  times.tiffany = times.mary - 7 ∧
  total_time times = 223 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_time_l3018_301854


namespace NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_thirds_l3018_301887

theorem no_solution_iff_m_geq_two_thirds (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2*m < 0 ∧ x + m > 2)) ↔ m ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_thirds_l3018_301887


namespace NUMINAMATH_CALUDE_compare_expressions_l3018_301865

theorem compare_expressions : (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l3018_301865


namespace NUMINAMATH_CALUDE_prob_sum_7_or_11_l3018_301861

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The set of possible sums we're interested in -/
def target_sums : Set ℕ := {7, 11}

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of ways to get a sum of 7 or 11 -/
def favorable_outcomes : ℕ := 8

/-- The probability of rolling a sum of 7 or 11 with two dice -/
def probability_sum_7_or_11 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_7_or_11 : probability_sum_7_or_11 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_7_or_11_l3018_301861


namespace NUMINAMATH_CALUDE_candle_length_correct_l3018_301857

/-- Represents the remaining length of a burning candle after t hours. -/
def candle_length (t : ℝ) : ℝ := 20 - 5 * t

theorem candle_length_correct (t : ℝ) (h : 0 ≤ t ∧ t ≤ 4) : 
  candle_length t = 20 - 5 * t ∧ candle_length t ≥ 0 := by
  sorry

#check candle_length_correct

end NUMINAMATH_CALUDE_candle_length_correct_l3018_301857


namespace NUMINAMATH_CALUDE_ap_terms_count_l3018_301863

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  n % 2 = 0 ∧ 
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 32 ∧ 
  (n / 2 : ℚ) * (2 * a + 2 * d + (n - 2) * d) = 40 ∧ 
  a + (n - 1) * d - a = 8 → 
  n = 16 := by sorry

end NUMINAMATH_CALUDE_ap_terms_count_l3018_301863


namespace NUMINAMATH_CALUDE_solve_transportation_problem_l3018_301891

/-- Represents the daily transportation problem for building materials -/
structure TransportationProblem where
  daily_requirement : ℕ
  max_supply_A : ℕ
  max_supply_B : ℕ
  cost_scenario1 : ℕ
  cost_scenario2 : ℕ

/-- Represents the solution to the transportation problem -/
structure TransportationSolution where
  cost_per_ton_A : ℕ
  cost_per_ton_B : ℕ
  min_total_cost : ℕ
  optimal_tons_A : ℕ
  optimal_tons_B : ℕ

/-- Theorem stating the solution to the transportation problem -/
theorem solve_transportation_problem (p : TransportationProblem) 
  (h1 : p.daily_requirement = 120)
  (h2 : p.max_supply_A = 80)
  (h3 : p.max_supply_B = 90)
  (h4 : p.cost_scenario1 = 26000)
  (h5 : p.cost_scenario2 = 27000) :
  ∃ (s : TransportationSolution),
    s.cost_per_ton_A = 240 ∧
    s.cost_per_ton_B = 200 ∧
    s.min_total_cost = 25200 ∧
    s.optimal_tons_A = 30 ∧
    s.optimal_tons_B = 90 ∧
    s.optimal_tons_A + s.optimal_tons_B = p.daily_requirement ∧
    s.optimal_tons_A ≤ p.max_supply_A ∧
    s.optimal_tons_B ≤ p.max_supply_B ∧
    s.min_total_cost = s.cost_per_ton_A * s.optimal_tons_A + s.cost_per_ton_B * s.optimal_tons_B :=
by
  sorry


end NUMINAMATH_CALUDE_solve_transportation_problem_l3018_301891


namespace NUMINAMATH_CALUDE_arithmetic_square_root_sum_l3018_301834

theorem arithmetic_square_root_sum (a b c : ℝ) : 
  a^(1/3) = 2 → 
  b = ⌊Real.sqrt 5⌋ → 
  c^2 = 16 → 
  (Real.sqrt (a + b + c) = Real.sqrt 14) ∨ (Real.sqrt (a + b + c) = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_sum_l3018_301834


namespace NUMINAMATH_CALUDE_zoo_trip_remainder_is_24_l3018_301831

/-- Calculates the amount left for lunch and snacks after a zoo trip for two people -/
def zoo_trip_remainder (zoo_ticket_price : ℚ) (bus_fare_one_way : ℚ) (total_money : ℚ) : ℚ :=
  let zoo_cost := 2 * zoo_ticket_price
  let bus_cost := 2 * 2 * bus_fare_one_way
  total_money - (zoo_cost + bus_cost)

/-- Theorem: Given the specified prices and total money, the remainder for lunch and snacks is $24 -/
theorem zoo_trip_remainder_is_24 :
  zoo_trip_remainder 5 1.5 40 = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_remainder_is_24_l3018_301831


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l3018_301804

-- Define the function f
def f (x a : ℝ) : ℝ := |x^2 - 4*x| - a

-- State the theorem
theorem three_zeros_implies_a_equals_four :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l3018_301804


namespace NUMINAMATH_CALUDE_square_of_sqrt_17_l3018_301810

theorem square_of_sqrt_17 : (Real.sqrt 17) ^ 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sqrt_17_l3018_301810


namespace NUMINAMATH_CALUDE_total_income_calculation_l3018_301842

def original_cupcake_price : ℚ := 3
def original_cookie_price : ℚ := 2
def cupcake_discount : ℚ := 0.3
def cookie_discount : ℚ := 0.45
def cupcakes_sold : ℕ := 25
def cookies_sold : ℕ := 18

theorem total_income_calculation :
  let new_cupcake_price := original_cupcake_price * (1 - cupcake_discount)
  let new_cookie_price := original_cookie_price * (1 - cookie_discount)
  let total_income := (new_cupcake_price * cupcakes_sold) + (new_cookie_price * cookies_sold)
  total_income = 72.3 := by sorry

end NUMINAMATH_CALUDE_total_income_calculation_l3018_301842


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_equals_three_l3018_301890

theorem cubic_sum_over_product_equals_three
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_equals_three_l3018_301890


namespace NUMINAMATH_CALUDE_rational_power_floor_theorem_l3018_301838

theorem rational_power_floor_theorem (x : ℚ) : 
  (∃ (a : ℤ), a ≥ 1 ∧ x^(⌊x⌋) = a / 2) ↔ (∃ (n : ℤ), x = n) ∨ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_power_floor_theorem_l3018_301838


namespace NUMINAMATH_CALUDE_no_half_parallel_diagonals_l3018_301888

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  h : n > 2

/-- The number of diagonals in a polygon -/
def numDiagonals (p : RegularPolygon) : ℕ :=
  p.n * (p.n - 3) / 2

/-- The number of diagonals parallel to sides in a polygon -/
def numParallelDiagonals (p : RegularPolygon) : ℕ :=
  if p.n % 2 = 1 then numDiagonals p else (p.n / 2) - 1

/-- Theorem: No regular polygon has exactly half of its diagonals parallel to its sides -/
theorem no_half_parallel_diagonals (p : RegularPolygon) :
  2 * numParallelDiagonals p ≠ numDiagonals p :=
sorry

end NUMINAMATH_CALUDE_no_half_parallel_diagonals_l3018_301888


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3018_301817

theorem sin_cos_identity : 
  Real.sin (110 * π / 180) * Real.cos (40 * π / 180) - 
  Real.cos (70 * π / 180) * Real.sin (40 * π / 180) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3018_301817


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3018_301827

/-- The number of ways to choose two committees from a club -/
def choose_committees (total_members : ℕ) (exec_size : ℕ) (aux_size : ℕ) : ℕ :=
  Nat.choose total_members exec_size * Nat.choose (total_members - exec_size) aux_size

/-- Theorem stating the number of ways to choose committees from a 30-member club -/
theorem committee_selection_ways :
  choose_committees 30 5 3 = 327764800 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3018_301827


namespace NUMINAMATH_CALUDE_work_completion_time_l3018_301806

/-- The time taken for two workers to complete three times a piece of work -/
def time_to_complete_work (aarti_rate : ℚ) (bina_rate : ℚ) : ℚ :=
  3 / (aarti_rate + bina_rate)

/-- Theorem stating that Aarti and Bina will take approximately 9.23 days to complete three times the work -/
theorem work_completion_time :
  let aarti_rate : ℚ := 1 / 5
  let bina_rate : ℚ := 1 / 8
  abs (time_to_complete_work aarti_rate bina_rate - 9.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3018_301806


namespace NUMINAMATH_CALUDE_square_of_difference_negative_first_l3018_301852

theorem square_of_difference_negative_first (x y : ℝ) : (-x + y)^2 = x^2 - 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_negative_first_l3018_301852


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3018_301859

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : 1/x - 1/y = -3) :
  x + y = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3018_301859


namespace NUMINAMATH_CALUDE_expression_evaluation_l3018_301870

theorem expression_evaluation : 121 + 2 * 11 * 4 + 16 + 7 = 232 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3018_301870


namespace NUMINAMATH_CALUDE_total_birds_after_breeding_l3018_301808

/-- Represents the types of birds on the farm -/
inductive BirdType
  | Hen
  | Duck
  | Goose
  | Pigeon

/-- Represents the count and breeding information for each bird type -/
structure BirdInfo where
  count : ℕ
  maleRatio : ℚ
  femaleRatio : ℚ
  offspringPerFemale : ℕ
  breedingSuccessRate : ℚ

/-- Calculates the total number of birds after the breeding season -/
def totalBirdsAfterBreeding (birdCounts : BirdType → BirdInfo) (pigeonHatchRate : ℚ) : ℕ :=
  sorry

/-- The main theorem stating the total number of birds after breeding -/
theorem total_birds_after_breeding :
  let birdCounts : BirdType → BirdInfo
    | BirdType.Hen => ⟨40, 2/9, 7/9, 7, 85/100⟩
    | BirdType.Duck => ⟨20, 1/4, 3/4, 9, 75/100⟩
    | BirdType.Goose => ⟨10, 3/11, 8/11, 5, 90/100⟩
    | BirdType.Pigeon => ⟨30, 1/2, 1/2, 2, 80/100⟩
  totalBirdsAfterBreeding birdCounts (80/100) = 442 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_after_breeding_l3018_301808


namespace NUMINAMATH_CALUDE_correct_operation_l3018_301875

theorem correct_operation (a b : ℝ) : 3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3018_301875


namespace NUMINAMATH_CALUDE_log_equation_sum_l3018_301825

theorem log_equation_sum (A B C : ℕ+) 
  (h_coprime : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : 
  A + B + C = 5 := by
sorry

end NUMINAMATH_CALUDE_log_equation_sum_l3018_301825


namespace NUMINAMATH_CALUDE_spade_nested_calc_l3018_301889

-- Define the spade operation
def spade (x y : ℚ) : ℚ := x - 1 / y

-- Theorem statement
theorem spade_nested_calc : spade 3 (spade 3 (3/2)) = 18/7 := by sorry

end NUMINAMATH_CALUDE_spade_nested_calc_l3018_301889


namespace NUMINAMATH_CALUDE_rose_flowers_l3018_301879

/-- The number of flowers Rose bought -/
def total_flowers : ℕ := 12

/-- The number of daisies -/
def daisies : ℕ := 2

/-- The number of sunflowers -/
def sunflowers : ℕ := 4

/-- The number of tulips -/
def tulips : ℕ := (3 * (total_flowers - daisies)) / 5

theorem rose_flowers :
  total_flowers = daisies + tulips + sunflowers ∧
  tulips = (3 * (total_flowers - daisies)) / 5 ∧
  sunflowers = (2 * (total_flowers - daisies)) / 5 :=
by sorry

end NUMINAMATH_CALUDE_rose_flowers_l3018_301879


namespace NUMINAMATH_CALUDE_absent_present_probability_l3018_301837

theorem absent_present_probability (course_days : ℕ) (avg_absent_days : ℕ) : 
  course_days = 40 → 
  avg_absent_days = 1 → 
  (39 : ℚ) / 800 = (course_days - avg_absent_days) / (course_days^2) * 2 := by
  sorry

end NUMINAMATH_CALUDE_absent_present_probability_l3018_301837


namespace NUMINAMATH_CALUDE_figure_perimeter_l3018_301805

/-- The figure in the coordinate plane defined by |x + y| + |x - y| = 8 -/
def Figure := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| = 8}

/-- The perimeter of a set in ℝ² -/
noncomputable def perimeter (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem figure_perimeter : perimeter Figure = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_figure_perimeter_l3018_301805


namespace NUMINAMATH_CALUDE_rectangle_with_perpendicular_diagonals_is_square_l3018_301899

-- Define a rectangle
structure Rectangle :=
  (a b : ℝ)
  (a_positive : a > 0)
  (b_positive : b > 0)

-- Define a property for perpendicular diagonals
def has_perpendicular_diagonals (r : Rectangle) : Prop :=
  r.a^2 = r.b^2

-- Define a square as a special case of rectangle
def is_square (r : Rectangle) : Prop :=
  r.a = r.b

-- Theorem statement
theorem rectangle_with_perpendicular_diagonals_is_square 
  (r : Rectangle) (h : has_perpendicular_diagonals r) : 
  is_square r :=
sorry

end NUMINAMATH_CALUDE_rectangle_with_perpendicular_diagonals_is_square_l3018_301899


namespace NUMINAMATH_CALUDE_garden_length_l3018_301860

/-- Given a rectangular garden with perimeter 1200 m and breadth 240 m, prove its length is 360 m -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ)
  (h1 : perimeter = 1200)
  (h2 : breadth = 240)
  (h3 : perimeter = 2 * length + 2 * breadth) :
  length = 360 :=
by sorry

end NUMINAMATH_CALUDE_garden_length_l3018_301860


namespace NUMINAMATH_CALUDE_max_m_value_max_m_is_75_l3018_301855

theorem max_m_value (m n : ℕ+) (h : 8 * m + 9 * n = m * n + 6) : 
  ∀ k : ℕ+, 8 * k + 9 * n = k * n + 6 → k ≤ m :=
sorry

theorem max_m_is_75 : ∃ m n : ℕ+, 8 * m + 9 * n = m * n + 6 ∧ m = 75 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_max_m_is_75_l3018_301855


namespace NUMINAMATH_CALUDE_days_at_sisters_house_l3018_301869

/-- Calculates the number of days spent at the sister's house during a vacation --/
theorem days_at_sisters_house (total_vacation_days : ℕ) 
  (days_to_grandparents days_at_grandparents days_to_brother days_at_brother 
   days_to_sister days_from_sister : ℕ) : 
  total_vacation_days = 21 →
  days_to_grandparents = 1 →
  days_at_grandparents = 5 →
  days_to_brother = 1 →
  days_at_brother = 5 →
  days_to_sister = 2 →
  days_from_sister = 2 →
  total_vacation_days - (days_to_grandparents + days_at_grandparents + 
    days_to_brother + days_at_brother + days_to_sister + days_from_sister) = 5 := by
  sorry

end NUMINAMATH_CALUDE_days_at_sisters_house_l3018_301869


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3018_301818

theorem fraction_evaluation : (3 : ℚ) / (2 - 4 / (-5)) = 15 / 14 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3018_301818


namespace NUMINAMATH_CALUDE_largest_not_expressible_l3018_301874

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ k c, k > 0 ∧ is_composite c ∧ n = 37 * k + c

theorem largest_not_expressible :
  (∀ n > 66, is_expressible n) ∧ ¬is_expressible 66 :=
sorry

end NUMINAMATH_CALUDE_largest_not_expressible_l3018_301874


namespace NUMINAMATH_CALUDE_cone_slant_height_l3018_301894

/-- The slant height of a cone given its base radius and curved surface area -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 5) (h2 : csa = 157.07963267948966) :
  csa / (Real.pi * r) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3018_301894


namespace NUMINAMATH_CALUDE_probability_theorem_l3018_301893

def total_marbles : ℕ := 8
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 4

def probability_one_each_plus_red : ℚ :=
  (red_marbles.choose 2 * blue_marbles.choose 1 * green_marbles.choose 1) /
  total_marbles.choose selected_marbles

theorem probability_theorem :
  probability_one_each_plus_red = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3018_301893


namespace NUMINAMATH_CALUDE_ln_third_derivative_value_l3018_301880

open Real

theorem ln_third_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) 
  (h1 : ∀ x, f x = log x)
  (h2 : deriv (deriv (deriv f)) x₀ = 1 / x₀^2) :
  x₀ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ln_third_derivative_value_l3018_301880


namespace NUMINAMATH_CALUDE_sqrt_less_than_2x_iff_x_greater_than_quarter_l3018_301897

theorem sqrt_less_than_2x_iff_x_greater_than_quarter (x : ℝ) (hx : x > 0) :
  Real.sqrt x < 2 * x ↔ x > (1 / 4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_sqrt_less_than_2x_iff_x_greater_than_quarter_l3018_301897


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l3018_301823

-- Define the triangle vertices
def P : ℝ × ℝ := (-8, 2)
def Q : ℝ × ℝ := (-10, -10)
def R : ℝ × ℝ := (2, -4)

-- Define the angle bisector equation coefficients
def b : ℝ := sorry
def d : ℝ := sorry

-- State the theorem
theorem angle_bisector_sum (h : ∀ (x y : ℝ), b * x + 2 * y + d = 0 ↔ 
  (y - P.2) = (y - P.2) / (x - P.1) * (x - P.1)) : 
  abs (b + d + 64.226) < 0.001 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l3018_301823


namespace NUMINAMATH_CALUDE_cindy_calculation_l3018_301801

theorem cindy_calculation (h : 50^2 = 2500) : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l3018_301801


namespace NUMINAMATH_CALUDE_coconut_price_l3018_301824

/-- The price of a coconut given the yield per tree, total money needed, and number of trees to harvest. -/
theorem coconut_price
  (yield_per_tree : ℕ)  -- Number of coconuts per tree
  (total_money : ℕ)     -- Total money needed in dollars
  (trees_to_harvest : ℕ) -- Number of trees to harvest
  (h1 : yield_per_tree = 5)
  (h2 : total_money = 90)
  (h3 : trees_to_harvest = 6) :
  total_money / (yield_per_tree * trees_to_harvest) = 3 :=
by sorry


end NUMINAMATH_CALUDE_coconut_price_l3018_301824


namespace NUMINAMATH_CALUDE_dani_pants_reward_l3018_301896

/-- The number of pairs of pants Dani gets each year -/
def pants_per_year (initial_pants : ℕ) (pants_after_5_years : ℕ) : ℕ :=
  ((pants_after_5_years - initial_pants) / 5) / 2

/-- Theorem stating that Dani gets 4 pairs of pants each year -/
theorem dani_pants_reward (initial_pants : ℕ) (pants_after_5_years : ℕ) 
  (h1 : initial_pants = 50) 
  (h2 : pants_after_5_years = 90) : 
  pants_per_year initial_pants pants_after_5_years = 4 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_reward_l3018_301896


namespace NUMINAMATH_CALUDE_drone_image_trees_l3018_301848

theorem drone_image_trees (T : ℕ) (h1 : T ≥ 100) (h2 : T ≥ 90) (h3 : T ≥ 82) : 
  (T - 82) + (T - 82) = 26 := by
sorry

end NUMINAMATH_CALUDE_drone_image_trees_l3018_301848


namespace NUMINAMATH_CALUDE_physics_score_l3018_301828

/-- Represents the scores in physics, chemistry, and mathematics -/
structure Scores where
  physics : ℕ
  chemistry : ℕ
  mathematics : ℕ

/-- The average score of all three subjects is 60 -/
def average_all (s : Scores) : Prop :=
  (s.physics + s.chemistry + s.mathematics) / 3 = 60

/-- The average score of physics and mathematics is 90 -/
def average_physics_math (s : Scores) : Prop :=
  (s.physics + s.mathematics) / 2 = 90

/-- The average score of physics and chemistry is 70 -/
def average_physics_chem (s : Scores) : Prop :=
  (s.physics + s.chemistry) / 2 = 70

/-- Theorem stating that given the conditions, the physics score is 140 -/
theorem physics_score (s : Scores) 
  (h1 : average_all s)
  (h2 : average_physics_math s)
  (h3 : average_physics_chem s) :
  s.physics = 140 := by
  sorry

end NUMINAMATH_CALUDE_physics_score_l3018_301828


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l3018_301815

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l3018_301815


namespace NUMINAMATH_CALUDE_insect_meeting_point_l3018_301800

/-- Triangle PQR with given side lengths -/
structure Triangle (PQ QR PR : ℝ) where
  positive : 0 < PQ ∧ 0 < QR ∧ 0 < PR
  triangle_inequality : PQ + QR > PR ∧ QR + PR > PQ ∧ PR + PQ > QR

/-- Point S where insects meet -/
def MeetingPoint (t : Triangle PQ QR PR) := 
  {S : ℝ // 0 ≤ S ∧ S ≤ QR}

/-- Theorem stating that QS = 5 under given conditions -/
theorem insect_meeting_point 
  (t : Triangle 7 8 9) 
  (S : MeetingPoint t) : 
  S.val = 5 := by sorry

end NUMINAMATH_CALUDE_insect_meeting_point_l3018_301800


namespace NUMINAMATH_CALUDE_jerry_vote_difference_l3018_301872

def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375

theorem jerry_vote_difference : 
  jerry_votes - (total_votes - jerry_votes) = 20196 := by
  sorry

end NUMINAMATH_CALUDE_jerry_vote_difference_l3018_301872


namespace NUMINAMATH_CALUDE_cuboid_volume_l3018_301813

/-- Represents a cuboid with length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def Cuboid.volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Calculates the surface area of a cuboid -/
def Cuboid.surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Theorem: The volume of the cuboid is 180 cm³ -/
theorem cuboid_volume (c : Cuboid) :
  (∀ (c' : Cuboid), c'.length = c.length ∧ c'.width = c.width ∧ c'.height = c.height + 1 →
    c'.length = c'.width ∧ c'.width = c'.height) →
  (∃ (c' : Cuboid), c'.length = c.length ∧ c'.width = c.width ∧ c'.height = c.height + 1 ∧
    c'.surfaceArea = c.surfaceArea + 24) →
  c.volume = 180 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l3018_301813


namespace NUMINAMATH_CALUDE_die_roll_probability_l3018_301867

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def roll_twice : Finset (ℕ × ℕ) :=
  standard_die.product standard_die

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 3), (2, 6)}

theorem die_roll_probability :
  (favorable_outcomes.card : ℚ) / roll_twice.card = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3018_301867


namespace NUMINAMATH_CALUDE_impossibleColoring_l3018_301881

/-- Represents a color in the grid -/
inductive Color
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10

/-- Represents the grid of colors -/
def Grid := Fin 99 → Fin 99 → Color

/-- Checks if a 3x3 subgrid centered at (i, j) has exactly one match with the center -/
def validSubgrid (g : Grid) (i j : Fin 99) : Prop :=
  let center := g i j
  (∃! x y, x ∈ [i-1, i, i+1] ∧ y ∈ [j-1, j, j+1] ∧ (x, y) ≠ (i, j) ∧ g x y = center)

/-- The main theorem stating the impossibility of the described coloring -/
theorem impossibleColoring : ¬∃ g : Grid, ∀ i j : Fin 99, validSubgrid g i j := by
  sorry


end NUMINAMATH_CALUDE_impossibleColoring_l3018_301881


namespace NUMINAMATH_CALUDE_worm_distance_after_15_days_l3018_301844

/-- Represents the daily movement of a worm -/
structure WormMovement where
  forward : ℝ
  backward : ℝ

/-- Calculates the net daily distance traveled by the worm -/
def net_daily_distance (movement : WormMovement) : ℝ :=
  movement.forward - movement.backward

/-- Calculates the total distance traveled over a number of days -/
def total_distance (movement : WormMovement) (days : ℕ) : ℝ :=
  (net_daily_distance movement) * days

/-- The theorem to be proved -/
theorem worm_distance_after_15_days (worm_movement : WormMovement)
    (h1 : worm_movement.forward = 5)
    (h2 : worm_movement.backward = 3)
    : total_distance worm_movement 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_worm_distance_after_15_days_l3018_301844


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l3018_301802

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x-1)

theorem function_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 →
    f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l3018_301802


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3018_301846

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3018_301846


namespace NUMINAMATH_CALUDE_perfect_square_triples_l3018_301832

theorem perfect_square_triples :
  ∀ (a b c : ℕ),
    (∃ (x : ℕ), a^2 + 2*b + c = x^2) ∧
    (∃ (y : ℕ), b^2 + 2*c + a = y^2) ∧
    (∃ (z : ℕ), c^2 + 2*a + b = z^2) →
    ((a, b, c) = (0, 0, 0) ∨
     (a, b, c) = (1, 1, 1) ∨
     (a, b, c) = (127, 106, 43) ∨
     (a, b, c) = (106, 43, 127) ∨
     (a, b, c) = (43, 127, 106)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_triples_l3018_301832


namespace NUMINAMATH_CALUDE_exponential_inverse_existence_uniqueness_l3018_301878

theorem exponential_inverse_existence_uniqueness (a x : ℝ) (ha : 0 < a) (ha_neq : a ≠ 1) (hx : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by sorry

end NUMINAMATH_CALUDE_exponential_inverse_existence_uniqueness_l3018_301878


namespace NUMINAMATH_CALUDE_shielas_neighbors_l3018_301811

theorem shielas_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) (h3 : drawings_per_neighbor > 0) :
  total_drawings / drawings_per_neighbor = 6 := by
  sorry

end NUMINAMATH_CALUDE_shielas_neighbors_l3018_301811


namespace NUMINAMATH_CALUDE_student_assignment_l3018_301866

theorem student_assignment (n : ℕ) (m : ℕ) (h1 : n = 4) (h2 : m = 3) :
  (Nat.choose n 2) * (Nat.factorial m) = 36 := by
  sorry

end NUMINAMATH_CALUDE_student_assignment_l3018_301866


namespace NUMINAMATH_CALUDE_knight_reachability_l3018_301829

/-- Represents a position on an infinite chessboard -/
structure Position where
  x : Int
  y : Int

/-- Represents a knight's move -/
inductive KnightMove (n : Nat)
  | horizontal : KnightMove n
  | vertical   : KnightMove n

/-- Applies a knight's move to a position -/
def applyMove (n : Nat) (p : Position) (m : KnightMove n) : Position :=
  match m with
  | KnightMove.horizontal => ⟨p.x + n, p.y + 1⟩
  | KnightMove.vertical   => ⟨p.x + 1, p.y + n⟩

/-- Defines reachability for a knight -/
def isReachable (n : Nat) (start finish : Position) : Prop :=
  ∃ (moves : List (KnightMove n)), finish = moves.foldl (applyMove n) start

/-- The main theorem: A knight can reach any position iff n is even -/
theorem knight_reachability (n : Nat) :
  (∀ (start finish : Position), isReachable n start finish) ↔ Even n := by
  sorry


end NUMINAMATH_CALUDE_knight_reachability_l3018_301829


namespace NUMINAMATH_CALUDE_website_earnings_l3018_301807

/-- Calculates daily earnings for a website given monthly visits, days in a month, and earnings per visit -/
def daily_earnings (monthly_visits : ℕ) (days_in_month : ℕ) (earnings_per_visit : ℚ) : ℚ :=
  (monthly_visits : ℚ) / (days_in_month : ℚ) * earnings_per_visit

/-- Proves that given 30000 monthly visits in a 30-day month with $0.01 earnings per visit, daily earnings are $10 -/
theorem website_earnings : daily_earnings 30000 30 (1/100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_website_earnings_l3018_301807


namespace NUMINAMATH_CALUDE_distinct_power_tower_values_l3018_301850

def power_tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (power_tower base n)

def parenthesized_expressions (base : ℕ) (height : ℕ) : Finset ℕ :=
  sorry

theorem distinct_power_tower_values :
  (parenthesized_expressions 3 4).card = 5 :=
sorry

end NUMINAMATH_CALUDE_distinct_power_tower_values_l3018_301850


namespace NUMINAMATH_CALUDE_no_solution_iff_m_special_l3018_301884

/-- The equation has no solution if and only if m is -4, 6, or 1 -/
theorem no_solution_iff_m_special (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → 2 / (x - 2) + m * x / (x^2 - 4) ≠ 3 / (x + 2)) ↔ 
  (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_special_l3018_301884


namespace NUMINAMATH_CALUDE_total_precious_stones_l3018_301826

theorem total_precious_stones (agate olivine diamond : ℕ) : 
  olivine = agate + 5 →
  diamond = olivine + 11 →
  agate = 30 →
  agate + olivine + diamond = 111 := by
sorry

end NUMINAMATH_CALUDE_total_precious_stones_l3018_301826


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3018_301864

def divisors : List ℕ := [12, 16, 18, 21, 28, 35, 40, 45, 55]

theorem smallest_number_divisible (n : ℕ) : 
  (∀ d ∈ divisors, (n - 10) % d = 0) →
  (∀ m < n, ∃ d ∈ divisors, (m - 10) % d ≠ 0) →
  n = 55450 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l3018_301864
