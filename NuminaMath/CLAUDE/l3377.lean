import Mathlib

namespace NUMINAMATH_CALUDE_city_male_population_l3377_337747

theorem city_male_population (total_population : ℕ) (num_parts : ℕ) (male_parts : ℕ) :
  total_population = 1000 →
  num_parts = 5 →
  male_parts = 2 →
  (total_population / num_parts) * male_parts = 400 := by
sorry

end NUMINAMATH_CALUDE_city_male_population_l3377_337747


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l3377_337777

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 3 * x - 9/4 = 0) ↔ (k > -1 ∨ k < -1) ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l3377_337777


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3377_337732

theorem equation_solution_exists : ∃ (x y : ℕ), x^9 = 2013 * y^10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3377_337732


namespace NUMINAMATH_CALUDE_min_diagonal_pairs_l3377_337766

/-- Represents a triangle of cells arranged in rows -/
structure CellTriangle where
  rows : ℕ

/-- Calculates the total number of cells in the triangle -/
def totalCells (t : CellTriangle) : ℕ :=
  t.rows * (t.rows + 1) / 2

/-- Calculates the number of rows with an odd number of cells -/
def oddRows (t : CellTriangle) : ℕ :=
  t.rows / 2

/-- Theorem: The minimum number of diagonal pairs in a cell triangle
    with 5784 rows is equal to the number of rows with an odd number of cells -/
theorem min_diagonal_pairs (t : CellTriangle) (h : t.rows = 5784) :
  oddRows t = 2892 := by sorry

end NUMINAMATH_CALUDE_min_diagonal_pairs_l3377_337766


namespace NUMINAMATH_CALUDE_evaluate_expression_l3377_337726

theorem evaluate_expression (a b : ℕ) (h1 : a = 2009) (h2 : b = 2010) :
  2 * (b^3 - a*b^2 - a^2*b + a^3) = 24240542 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3377_337726


namespace NUMINAMATH_CALUDE_system_solution_l3377_337712

theorem system_solution (a b c k x y z : ℝ) 
  (h1 : a * x + b * y + c * z = k)
  (h2 : a^2 * x + b^2 * y + c^2 * z = k^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = k^3)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3377_337712


namespace NUMINAMATH_CALUDE_complex_power_sum_l3377_337779

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^2021 + 1/z^2021 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3377_337779


namespace NUMINAMATH_CALUDE_largest_fraction_l3377_337797

theorem largest_fraction
  (a b c d e : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (d + e) / (a + b) > (a + c) / (b + d) ∧
  (d + e) / (a + b) > (b + e) / (c + d) ∧
  (d + e) / (a + b) > (c + d) / (a + e) ∧
  (d + e) / (a + b) > (e + a) / (b + c) :=
by sorry


end NUMINAMATH_CALUDE_largest_fraction_l3377_337797


namespace NUMINAMATH_CALUDE_sphere_radii_ratio_l3377_337744

/-- Given four spheres arranged such that each sphere touches three others and a plane,
    with two spheres having radius R and two spheres having radius r,
    prove that the ratio of the larger radius to the smaller radius is 2 + √3. -/
theorem sphere_radii_ratio (R r : ℝ) (h1 : R > 0) (h2 : r > 0)
  (h3 : R^2 + r^2 = 4*R*r) : R/r = 2 + Real.sqrt 3 ∨ r/R = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radii_ratio_l3377_337744


namespace NUMINAMATH_CALUDE_food_weight_l3377_337730

/-- Given a bowl with 14 pieces of food, prove that each piece weighs 0.76 kg -/
theorem food_weight (total_weight : ℝ) (empty_bowl_weight : ℝ) (num_pieces : ℕ) :
  total_weight = 11.14 ∧ 
  empty_bowl_weight = 0.5 ∧ 
  num_pieces = 14 →
  (total_weight - empty_bowl_weight) / num_pieces = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_food_weight_l3377_337730


namespace NUMINAMATH_CALUDE_ice_pack_price_is_three_l3377_337740

/-- The price of a pack of 10 bags of ice for Chad's BBQ -/
def ice_pack_price (total_people : ℕ) (ice_per_person : ℕ) (bags_per_pack : ℕ) (total_spent : ℚ) : ℚ :=
  let total_ice := total_people * ice_per_person
  total_spent / (total_ice / bags_per_pack)

/-- Theorem: The price of a pack of 10 bags of ice is $3 -/
theorem ice_pack_price_is_three :
  ice_pack_price 15 2 10 9 = 3 := by
  sorry

#eval ice_pack_price 15 2 10 9

end NUMINAMATH_CALUDE_ice_pack_price_is_three_l3377_337740


namespace NUMINAMATH_CALUDE_additional_cars_needed_l3377_337738

def cars_per_row : ℕ := 8
def current_cars : ℕ := 37

theorem additional_cars_needed : 
  ∃ (n : ℕ), (n > 0) ∧ (cars_per_row * n ≥ current_cars) ∧ 
  (cars_per_row * n - current_cars = 3) ∧
  (∀ m : ℕ, m > 0 → cars_per_row * m ≥ current_cars → 
    cars_per_row * m - current_cars ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l3377_337738


namespace NUMINAMATH_CALUDE_games_comparison_l3377_337749

/-- Given Henry's and Neil's initial game counts and the number of games Henry gave to Neil,
    calculate how many times more games Henry has than Neil after the transfer. -/
theorem games_comparison (henry_initial : ℕ) (neil_initial : ℕ) (games_given : ℕ) : 
  henry_initial = 33 →
  neil_initial = 2 →
  games_given = 5 →
  (henry_initial - games_given) / (neil_initial + games_given) = 4 := by
sorry

end NUMINAMATH_CALUDE_games_comparison_l3377_337749


namespace NUMINAMATH_CALUDE_division_problem_l3377_337798

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1360)
  (h2 : a = 1614)
  (h3 : a = b * q + 15) :
  q = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3377_337798


namespace NUMINAMATH_CALUDE_megan_carrots_l3377_337719

theorem megan_carrots (initial_carrots thrown_out_carrots next_day_carrots : ℕ) :
  initial_carrots ≥ thrown_out_carrots →
  initial_carrots - thrown_out_carrots + next_day_carrots =
    initial_carrots + next_day_carrots - thrown_out_carrots :=
by sorry

end NUMINAMATH_CALUDE_megan_carrots_l3377_337719


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3377_337742

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (a + Complex.I) / (1 + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3377_337742


namespace NUMINAMATH_CALUDE_function_translation_transformation_result_l3377_337721

-- Define the original function
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3

-- Define the transformed function
def g (x : ℝ) : ℝ := 2 * x^2

-- Theorem stating that g is the result of translating f
theorem function_translation (x : ℝ) : 
  g x = f (x - 1) + 3 := by
  sorry

-- Prove that the transformation results in g
theorem transformation_result : 
  ∀ x, g x = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_translation_transformation_result_l3377_337721


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_l3377_337761

theorem arithmetic_geometric_mean_product : ∃ (a b : ℝ), 
  (a = (1 + 2) / 2) ∧ 
  (b^2 = (-1) * (-16)) ∧ 
  (ab = 6 ∨ ab = -6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_l3377_337761


namespace NUMINAMATH_CALUDE_females_in_coach_class_l3377_337764

theorem females_in_coach_class 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (male_first_class_fraction : ℚ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 30 / 100)
  (h3 : first_class_percentage = 10 / 100)
  (h4 : male_first_class_fraction = 1 / 3) :
  ↑((total_passengers : ℚ) * female_percentage - 
    (total_passengers : ℚ) * first_class_percentage * (1 - male_first_class_fraction)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_females_in_coach_class_l3377_337764


namespace NUMINAMATH_CALUDE_min_distance_of_sine_extrema_l3377_337701

open Real

theorem min_distance_of_sine_extrema :
  ∀ (f : ℝ → ℝ) (x₁ x₂ : ℝ),
  (∀ x, f x = sin (π * x)) →
  (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ (d : ℝ), d > 0 ∧ ∀ (y₁ y₂ : ℝ), (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ d) →
  (∀ (y₁ y₂ : ℝ), (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ 1) →
  |x₁ - x₂| = 1 := by
sorry

end NUMINAMATH_CALUDE_min_distance_of_sine_extrema_l3377_337701


namespace NUMINAMATH_CALUDE_eggs_per_plate_count_l3377_337700

def breakfast_plate (num_customers : ℕ) (total_bacon : ℕ) : ℕ → Prop :=
  λ eggs_per_plate : ℕ =>
    eggs_per_plate > 0 ∧
    2 * eggs_per_plate * num_customers = total_bacon

theorem eggs_per_plate_count (num_customers : ℕ) (total_bacon : ℕ) 
    (h1 : num_customers = 14) (h2 : total_bacon = 56) :
    ∃ eggs_per_plate : ℕ, breakfast_plate num_customers total_bacon eggs_per_plate ∧ 
    eggs_per_plate = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_plate_count_l3377_337700


namespace NUMINAMATH_CALUDE_solve_parking_problem_l3377_337776

def parking_problem (initial_balance : ℝ) (first_ticket_cost : ℝ) (num_full_cost_tickets : ℕ) (third_ticket_fraction : ℝ) (roommate_share : ℝ) : Prop :=
  let total_cost := first_ticket_cost * num_full_cost_tickets + first_ticket_cost * third_ticket_fraction
  let james_share := total_cost * (1 - roommate_share)
  initial_balance - james_share = 325

theorem solve_parking_problem :
  parking_problem 500 150 2 (1/3) (1/2) :=
by
  sorry

#check solve_parking_problem

end NUMINAMATH_CALUDE_solve_parking_problem_l3377_337776


namespace NUMINAMATH_CALUDE_madhav_rank_l3377_337707

theorem madhav_rank (total_students : ℕ) (rank_from_last : ℕ) (rank_from_start : ℕ) : 
  total_students = 31 →
  rank_from_last = 15 →
  rank_from_start = total_students - (rank_from_last - 1) →
  rank_from_start = 17 := by
sorry

end NUMINAMATH_CALUDE_madhav_rank_l3377_337707


namespace NUMINAMATH_CALUDE_parallel_line_through_point_A_l3377_337773

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-1, 0)

-- Define the parallel line passing through point A
def parallel_line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Theorem statement
theorem parallel_line_through_point_A :
  (∀ x y : ℝ, given_line x y ↔ 2 * x - y + 1 = 0) →
  parallel_line point_A.1 point_A.2 ∧
  ∀ x y : ℝ, parallel_line x y → 
    ∃ k : ℝ, y - point_A.2 = k * (x - point_A.1) ∧
             2 = k * 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_A_l3377_337773


namespace NUMINAMATH_CALUDE_total_gas_consumption_is_18_gallons_l3377_337785

/-- Represents the number of cuts for a lawn in a given month. -/
structure MonthlyCuts where
  regular : Nat  -- Number of cuts in regular months
  peak : Nat     -- Number of cuts in peak months

/-- Represents the gas consumption pattern for a lawn. -/
structure GasConsumption where
  gallons : Nat  -- Number of gallons consumed
  frequency : Nat  -- Frequency of consumption (every nth cut)

/-- Calculates the total number of cuts for a lawn over the season. -/
def totalCuts (cuts : MonthlyCuts) : Nat :=
  4 * cuts.regular + 4 * cuts.peak

/-- Calculates the gas consumed for a lawn over the season. -/
def gasConsumed (cuts : Nat) (consumption : GasConsumption) : Nat :=
  (cuts / consumption.frequency) * consumption.gallons

/-- Theorem stating that the total gas consumption is 18 gallons. -/
theorem total_gas_consumption_is_18_gallons 
  (large_lawn_cuts : MonthlyCuts)
  (small_lawn_cuts : MonthlyCuts)
  (large_lawn_gas : GasConsumption)
  (small_lawn_gas : GasConsumption)
  (h1 : large_lawn_cuts = { regular := 1, peak := 3 })
  (h2 : small_lawn_cuts = { regular := 2, peak := 2 })
  (h3 : large_lawn_gas = { gallons := 2, frequency := 3 })
  (h4 : small_lawn_gas = { gallons := 1, frequency := 2 })
  : gasConsumed (totalCuts large_lawn_cuts) large_lawn_gas + 
    gasConsumed (totalCuts small_lawn_cuts) small_lawn_gas = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_gas_consumption_is_18_gallons_l3377_337785


namespace NUMINAMATH_CALUDE_north_village_conscripts_l3377_337714

/-- The number of people to be conscripted from a village, given its population and the total population and conscription numbers. -/
def conscriptsFromVillage (villagePopulation totalPopulation totalConscripts : ℕ) : ℕ :=
  (villagePopulation * totalConscripts) / totalPopulation

/-- Theorem stating that given the specific village populations and total conscripts, 
    the number of conscripts from the north village is 108. -/
theorem north_village_conscripts :
  let northPopulation : ℕ := 8100
  let westPopulation : ℕ := 7488
  let southPopulation : ℕ := 6912
  let totalConscripts : ℕ := 300
  let totalPopulation : ℕ := northPopulation + westPopulation + southPopulation
  conscriptsFromVillage northPopulation totalPopulation totalConscripts = 108 := by
  sorry

end NUMINAMATH_CALUDE_north_village_conscripts_l3377_337714


namespace NUMINAMATH_CALUDE_seven_point_circle_triangles_l3377_337727

/-- The number of triangles formed by intersections of chords in a circle -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose (Nat.choose n 4) 3

/-- Theorem: Given 7 points on a circle with the specified conditions, 
    the number of triangles formed is 6545 -/
theorem seven_point_circle_triangles : num_triangles 7 = 6545 := by
  sorry

end NUMINAMATH_CALUDE_seven_point_circle_triangles_l3377_337727


namespace NUMINAMATH_CALUDE_unique_number_meeting_conditions_l3377_337772

theorem unique_number_meeting_conditions : ∃! n : ℕ+, 
  (((n < 12) ∨ (¬ 7 ∣ n) ∨ (5 * n < 70)) ∧ 
   ¬((n < 12) ∧ (¬ 7 ∣ n) ∧ (5 * n < 70))) ∧
  (((12 * n > 1000) ∨ (10 ∣ n) ∨ (n > 100)) ∧ 
   ¬((12 * n > 1000) ∧ (10 ∣ n) ∧ (n > 100))) ∧
  (((4 ∣ n) ∨ (11 * n < 1000) ∨ (9 ∣ n)) ∧ 
   ¬((4 ∣ n) ∧ (11 * n < 1000) ∧ (9 ∣ n))) ∧
  (((n < 20) ∨ Nat.Prime n ∨ (7 ∣ n)) ∧ 
   ¬((n < 20) ∧ Nat.Prime n ∧ (7 ∣ n))) ∧
  n = 89 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_meeting_conditions_l3377_337772


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l3377_337765

theorem sunglasses_cap_probability (total_sunglasses : ℕ) (total_caps : ℕ) 
  (prob_sunglasses_given_cap : ℚ) :
  total_sunglasses = 70 →
  total_caps = 45 →
  prob_sunglasses_given_cap = 3/9 →
  (prob_sunglasses_given_cap * total_caps : ℚ) / total_sunglasses = 3/14 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l3377_337765


namespace NUMINAMATH_CALUDE_root_implies_m_value_l3377_337715

theorem root_implies_m_value (m : ℚ) : 
  (∃ x : ℚ, x^2 - 6*x - 3*m - 5 = 0) ∧ 
  ((-1 : ℚ)^2 - 6*(-1) - 3*m - 5 = 0) → 
  m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l3377_337715


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3377_337717

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 12019 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3377_337717


namespace NUMINAMATH_CALUDE_f_15_equals_227_l3377_337754

/-- Given a function f(n) = n^2 - n + 17, prove that f(15) = 227 -/
theorem f_15_equals_227 (f : ℕ → ℕ) (h : ∀ n, f n = n^2 - n + 17) : f 15 = 227 := by
  sorry

end NUMINAMATH_CALUDE_f_15_equals_227_l3377_337754


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3377_337724

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 1 ≤ |x + 2| ∧ |x + 2| ≤ 5} = {x : ℝ | (-7 ≤ x ∧ x ≤ -3) ∨ (-1 ≤ x ∧ x ≤ 3)} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3377_337724


namespace NUMINAMATH_CALUDE_significant_difference_l3377_337778

/-- Represents the distribution of X, where X is the number of specified two mice
    assigned to the control group out of 40 total mice --/
def distribution_X : Fin 3 → ℚ
| 0 => 19/78
| 1 => 20/39
| 2 => 19/78

/-- The expectation of X --/
def E_X : ℚ := 1

/-- The median of the increase in body weight of all 40 mice --/
def median_weight : ℝ := 23.4

/-- The contingency table of mice counts below and above median --/
def contingency_table : Fin 2 → Fin 2 → ℕ
| 0, 0 => 6  -- Control group, below median
| 0, 1 => 14 -- Control group, above or equal to median
| 1, 0 => 14 -- Experimental group, below median
| 1, 1 => 6  -- Experimental group, above or equal to median

/-- The K² statistic --/
def K_squared : ℝ := 6.400

/-- The critical value for 95% confidence level --/
def critical_value_95 : ℝ := 3.841

/-- Theorem stating that the K² value is greater than the critical value,
    indicating a significant difference between groups --/
theorem significant_difference : K_squared > critical_value_95 := by sorry

end NUMINAMATH_CALUDE_significant_difference_l3377_337778


namespace NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_l3377_337762

theorem x_lt_5_necessary_not_sufficient :
  (∀ x : ℝ, -2 < x ∧ x < 4 → x < 5) ∧
  ¬(∀ x : ℝ, x < 5 → -2 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_l3377_337762


namespace NUMINAMATH_CALUDE_correct_number_probability_l3377_337769

-- Define the number of options for the first three digits
def first_three_options : ℕ := 3

-- Define the number of digits used in the last five digits
def last_five_digits : ℕ := 5

-- Theorem statement
theorem correct_number_probability :
  (1 : ℚ) / (first_three_options * Nat.factorial last_five_digits) = (1 : ℚ) / 360 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_probability_l3377_337769


namespace NUMINAMATH_CALUDE_cos_x_plus_7pi_12_l3377_337767

theorem cos_x_plus_7pi_12 (x : ℝ) (h : Real.sin (x + π / 12) = 1 / 3) :
  Real.cos (x + 7 * π / 12) = - 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_7pi_12_l3377_337767


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3377_337763

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrange_books (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: Arranging 5 and 6 indistinguishable objects in 11 positions yields 462 ways -/
theorem book_arrangement_theorem :
  arrange_books 5 6 = 462 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3377_337763


namespace NUMINAMATH_CALUDE_f_value_theorem_l3377_337751

-- Define the polynomial equation
def polynomial_equation (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a1*x^3 + a2*x^2 + a3*x + a4 = (x+1)^4 + b1*(x+1)^3 + b2*(x+1)^2 + b3*(x+1) + b4

-- Define the mapping f
def f (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : ℝ := b1 - b2 + b3 - b4

-- Theorem statement
theorem f_value_theorem :
  ∀ b1 b2 b3 b4 : ℝ, polynomial_equation 2 0 1 6 b1 b2 b3 b4 → f 2 0 1 6 b1 b2 b3 b4 = -3 :=
by sorry

end NUMINAMATH_CALUDE_f_value_theorem_l3377_337751


namespace NUMINAMATH_CALUDE_min_players_sum_divisible_by_10_l3377_337702

/-- Represents a 3x9 grid of distinct non-negative integers -/
def Grid := Matrix (Fin 3) (Fin 9) ℕ

/-- Predicate to check if all elements in a grid are distinct -/
def all_distinct (g : Grid) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Predicate to check if a sum is divisible by 10 -/
def sum_divisible_by_10 (a b : ℕ) : Prop :=
  (a + b) % 10 = 0

/-- Main theorem statement -/
theorem min_players_sum_divisible_by_10 (g : Grid) (h : all_distinct g) :
  ∃ i j i' j', sum_divisible_by_10 (g i j) (g i' j') :=
sorry

end NUMINAMATH_CALUDE_min_players_sum_divisible_by_10_l3377_337702


namespace NUMINAMATH_CALUDE_average_licks_to_center_l3377_337760

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_people : ℕ := 5

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  total_licks / total_people = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_licks_to_center_l3377_337760


namespace NUMINAMATH_CALUDE_pizzeria_sales_l3377_337745

theorem pizzeria_sales (small_price large_price total_sales small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_sales = 40)
  (h4 : small_count = 8) : 
  ∃ large_count : ℕ, 
    large_count = 3 ∧ 
    small_price * small_count + large_price * large_count = total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l3377_337745


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3377_337743

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 2 + a 3 = 6) :
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3377_337743


namespace NUMINAMATH_CALUDE_intersection_point_value_l3377_337734

theorem intersection_point_value (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a*x - 2) ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_value_l3377_337734


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3377_337791

theorem coefficient_of_x (x : ℝ) : 
  let expr := 4*(x - 5) + 3*(2 - 3*x^2 + 6*x) - 10*(3*x - 2)
  ∃ (a b c : ℝ), expr = a*x^2 + (-8)*x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3377_337791


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3377_337731

theorem fourth_number_proof (n : ℝ) (h1 : n = 27) : 
  let numbers : List ℝ := [3, 16, 33, n + 1]
  (numbers.sum / numbers.length = 20) → (n + 1 = 28) := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3377_337731


namespace NUMINAMATH_CALUDE_b_completion_time_l3377_337792

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 2
def work_rate_B : ℚ := 1 / 6

-- Define the total work as 1
def total_work : ℚ := 1

-- Define the work done in one day by both A and B
def work_done_together : ℚ := work_rate_A + work_rate_B

-- Define the remaining work after one day
def remaining_work : ℚ := total_work - work_done_together

-- Theorem to prove
theorem b_completion_time :
  remaining_work / work_rate_B = 2 := by sorry

end NUMINAMATH_CALUDE_b_completion_time_l3377_337792


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3377_337786

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4702 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3377_337786


namespace NUMINAMATH_CALUDE_smallest_c_in_arithmetic_progression_l3377_337780

theorem smallest_c_in_arithmetic_progression (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →
  a * b * c * d = 256 →
  ∀ x : ℝ, (∃ a' b' d' : ℝ, 
    0 < a' ∧ 0 < b' ∧ 0 < x ∧ 0 < d' ∧
    (∃ r' : ℝ, b' = a' + r' ∧ x = b' + r' ∧ d' = x + r') ∧
    a' * b' * x * d' = 256) →
  x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_in_arithmetic_progression_l3377_337780


namespace NUMINAMATH_CALUDE_jasons_library_visits_l3377_337756

/-- Jason's library visits in 4 weeks -/
def jasons_visits (williams_weekly_visits : ℕ) (jasons_multiplier : ℕ) (weeks : ℕ) : ℕ :=
  williams_weekly_visits * jasons_multiplier * weeks

/-- Theorem: Jason's library visits in 4 weeks -/
theorem jasons_library_visits :
  jasons_visits 2 4 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_jasons_library_visits_l3377_337756


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3377_337729

theorem sufficient_not_necessary :
  (∃ x : ℝ, x < -1 ∧ x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3377_337729


namespace NUMINAMATH_CALUDE_equation_implies_fraction_value_l3377_337746

theorem equation_implies_fraction_value (a b : ℝ) :
  a^2 + b^2 - 4*a - 2*b + 5 = 0 →
  (Real.sqrt a + b) / (2 * Real.sqrt a + b + 1) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_fraction_value_l3377_337746


namespace NUMINAMATH_CALUDE_john_annual_profit_l3377_337733

/-- Calculates John's annual profit from subletting his apartment -/
def annual_profit (tenant_a_rent tenant_b_rent tenant_c_rent john_rent utilities maintenance : ℕ) : ℕ :=
  let monthly_income := tenant_a_rent + tenant_b_rent + tenant_c_rent
  let monthly_expenses := john_rent + utilities + maintenance
  let monthly_profit := monthly_income - monthly_expenses
  12 * monthly_profit

/-- Theorem stating John's annual profit given his rental income and expenses -/
theorem john_annual_profit :
  annual_profit 350 400 450 900 100 50 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_profit_l3377_337733


namespace NUMINAMATH_CALUDE_farm_hens_count_l3377_337710

/-- Represents the number of animals on a farm. -/
structure FarmAnimals where
  hens : ℕ
  cows : ℕ
  goats : ℕ

/-- Calculates the total number of heads for all animals on the farm. -/
def totalHeads (farm : FarmAnimals) : ℕ :=
  farm.hens + farm.cows + farm.goats

/-- Calculates the total number of feet for all animals on the farm. -/
def totalFeet (farm : FarmAnimals) : ℕ :=
  2 * farm.hens + 4 * farm.cows + 4 * farm.goats

/-- Theorem stating that given the conditions, there are 66 hens on the farm. -/
theorem farm_hens_count (farm : FarmAnimals) 
  (head_count : totalHeads farm = 120) 
  (feet_count : totalFeet farm = 348) : 
  farm.hens = 66 := by
  sorry

end NUMINAMATH_CALUDE_farm_hens_count_l3377_337710


namespace NUMINAMATH_CALUDE_students_playing_both_football_and_cricket_l3377_337713

/-- The number of students playing both football and cricket -/
def students_playing_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Proof that 140 students play both football and cricket -/
theorem students_playing_both_football_and_cricket :
  students_playing_both 410 325 175 50 = 140 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_football_and_cricket_l3377_337713


namespace NUMINAMATH_CALUDE_distance_between_signs_l3377_337708

theorem distance_between_signs 
  (total_distance : ℕ) 
  (distance_to_first_sign : ℕ) 
  (distance_after_second_sign : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_to_first_sign - distance_after_second_sign = 375 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_signs_l3377_337708


namespace NUMINAMATH_CALUDE_monica_students_count_l3377_337790

/-- The number of students Monica sees each day -/
def monica_total_students : ℕ :=
  let first_class : ℕ := 20
  let second_third_classes : ℕ := 25 + 25
  let fourth_class : ℕ := first_class / 2
  let fifth_sixth_classes : ℕ := 28 + 28
  first_class + second_third_classes + fourth_class + fifth_sixth_classes

/-- Theorem stating the total number of students Monica sees each day -/
theorem monica_students_count : monica_total_students = 136 := by
  sorry

end NUMINAMATH_CALUDE_monica_students_count_l3377_337790


namespace NUMINAMATH_CALUDE_probability_two_female_contestants_l3377_337741

theorem probability_two_female_contestants (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 8 →
  female = 5 →
  male = 3 →
  (female.choose 2 : ℚ) / (total.choose 2) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_female_contestants_l3377_337741


namespace NUMINAMATH_CALUDE_sequence_difference_l3377_337787

theorem sequence_difference (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a n + a (n + 1) = 4 * n + 3) : 
  a 10 - a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3377_337787


namespace NUMINAMATH_CALUDE_max_product_of_three_numbers_l3377_337723

theorem max_product_of_three_numbers (n : ℕ) :
  let S := Finset.range (3 * n + 1) \ {0}
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a < b ∧ b < c ∧
    a + b + c = 3 * n ∧
    ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S →
      x < y → y < z →
      x + y + z = 3 * n →
      x * y * z ≤ a * b * c ∧
    a * b * c = n^3 - n :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_numbers_l3377_337723


namespace NUMINAMATH_CALUDE_max_area_triangle_l3377_337774

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def condition (t : Triangle) : Prop :=
  (Real.cos t.B / t.b) + (Real.cos t.C / t.c) = (2 * Real.sqrt 3 * Real.sin t.A) / (3 * Real.sin t.C)

/-- The theorem to be proved -/
theorem max_area_triangle (t : Triangle) (h1 : condition t) (h2 : t.B = Real.pi / 3) :
    (t.a * t.c * Real.sin t.B) / 2 ≤ 3 * Real.sqrt 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_area_triangle_l3377_337774


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3377_337795

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 50 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 50) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3377_337795


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3377_337706

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 10 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 100) ∧
  ((1 : ℚ) / 10 - (1 : ℚ) / 11 < (1 : ℚ) / 100) :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3377_337706


namespace NUMINAMATH_CALUDE_first_expression_value_l3377_337739

theorem first_expression_value (E a : ℝ) : 
  (E + (3 * a - 8)) / 2 = 84 → a = 32 → E = 80 := by
  sorry

end NUMINAMATH_CALUDE_first_expression_value_l3377_337739


namespace NUMINAMATH_CALUDE_hilt_garden_border_l3377_337703

/-- The number of rocks in Mrs. Hilt's completed garden border -/
def total_rocks : ℕ := 189

/-- The number of additional rocks Mrs. Hilt has yet to place -/
def remaining_rocks : ℕ := 64

/-- The number of rocks Mrs. Hilt has already placed -/
def placed_rocks : ℕ := total_rocks - remaining_rocks

theorem hilt_garden_border : placed_rocks = 125 := by
  sorry

end NUMINAMATH_CALUDE_hilt_garden_border_l3377_337703


namespace NUMINAMATH_CALUDE_extra_distance_for_early_arrival_l3377_337775

theorem extra_distance_for_early_arrival
  (S : ℝ) -- distance between A and B in kilometers
  (a : ℝ) -- original planned arrival time in hours
  (h : a > 2) -- condition that a > 2
  : (S / (a - 2) - S / a) = -- extra distance per hour needed for early arrival
    (S / (a - 2) - S / a) :=
by sorry

end NUMINAMATH_CALUDE_extra_distance_for_early_arrival_l3377_337775


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3377_337788

/-- Given vectors a and b, where a is parallel to b, prove that the magnitude of b is 3√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) (x : ℝ) :
  a = (2, -1) →
  b = (x, 3) →
  ∃ (k : ℝ), a = k • b →
  ‖b‖ = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3377_337788


namespace NUMINAMATH_CALUDE_mings_estimate_smaller_l3377_337704

theorem mings_estimate_smaller (x y δ : ℝ) (hx : x > y) (hy : y > 0) (hδ : δ > 0) :
  (x + δ) - (y + 2*δ) < x - y := by
  sorry

end NUMINAMATH_CALUDE_mings_estimate_smaller_l3377_337704


namespace NUMINAMATH_CALUDE_complex_division_result_l3377_337709

theorem complex_division_result : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l3377_337709


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l3377_337784

/-- Proves that for a rectangular plot with given conditions, the length is 20 metres more than the breadth -/
theorem rectangular_plot_length_difference (length width : ℝ) : 
  length = 60 ∧ 
  (2 * length + 2 * width) * 26.5 = 5300 →
  length - width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l3377_337784


namespace NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l3377_337748

/-- The ratio of lengths when a wire is cut to form a square and an octagon with equal areas -/
theorem wire_cut_square_octagon_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (a^2 / 16 = b^2 * (1 + Real.sqrt 2) / 32) → 
  (a / b = Real.sqrt ((1 + Real.sqrt 2) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l3377_337748


namespace NUMINAMATH_CALUDE_wrapping_paper_division_l3377_337716

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) 
  (h1 : total_used = 1/2)
  (h2 : num_presents = 5) :
  total_used / num_presents = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_division_l3377_337716


namespace NUMINAMATH_CALUDE_video_game_sales_earnings_l3377_337750

/-- The amount of money Zachary received from selling his video games -/
def zachary_earnings : ℕ := 40 * 5

/-- The amount of money Jason received from selling his video games -/
def jason_earnings : ℕ := zachary_earnings + (zachary_earnings * 30 / 100)

/-- The amount of money Ryan received from selling his video games -/
def ryan_earnings : ℕ := jason_earnings + 50

/-- The amount of money Emily received from selling her video games -/
def emily_earnings : ℕ := ryan_earnings - (ryan_earnings * 20 / 100)

/-- The amount of money Lily received from selling her video games -/
def lily_earnings : ℕ := emily_earnings + 70

/-- The total amount of money received by all five friends -/
def total_earnings : ℕ := zachary_earnings + jason_earnings + ryan_earnings + emily_earnings + lily_earnings

theorem video_game_sales_earnings : total_earnings = 1336 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_earnings_l3377_337750


namespace NUMINAMATH_CALUDE_remainder_problem_l3377_337770

theorem remainder_problem (k : ℕ+) (h : ∃ a : ℕ, 120 = a * k ^ 2 + 12) :
  ∃ b : ℕ, 144 = b * k + 0 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3377_337770


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l3377_337722

/-- The total number of girls -/
def total_girls : ℕ := 60

/-- The number of green-eyed redheads -/
def green_eyed_redheads : ℕ := 20

/-- The number of brunettes -/
def brunettes : ℕ := 35

/-- The number of brown-eyed girls -/
def brown_eyed : ℕ := 25

/-- Theorem: The number of brown-eyed brunettes is 20 -/
theorem brown_eyed_brunettes : 
  total_girls - (green_eyed_redheads + (brunettes - (brown_eyed - (total_girls - brunettes - green_eyed_redheads)))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l3377_337722


namespace NUMINAMATH_CALUDE_pencils_remaining_l3377_337789

theorem pencils_remaining (initial_pencils : ℕ) (removed_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : removed_pencils = 4) : 
  initial_pencils - removed_pencils = 83 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l3377_337789


namespace NUMINAMATH_CALUDE_profit_maximization_l3377_337752

/-- Represents the price reduction in yuan -/
def x : ℝ := 2.5

/-- Represents the initial selling price in yuan -/
def initial_price : ℝ := 60

/-- Represents the cost price in yuan -/
def cost_price : ℝ := 40

/-- Represents the initial weekly sales in items -/
def initial_sales : ℝ := 300

/-- Represents the increase in sales for each yuan of price reduction -/
def sales_increase_rate : ℝ := 20

/-- The profit function based on the price reduction x -/
def profit_function (x : ℝ) : ℝ :=
  (initial_price - x) * (initial_sales + sales_increase_rate * x) -
  cost_price * (initial_sales + sales_increase_rate * x)

/-- The maximum profit achieved -/
def max_profit : ℝ := 6125

theorem profit_maximization :
  profit_function x = max_profit ∧
  ∀ y, 0 ≤ y ∧ y < initial_price - cost_price →
    profit_function y ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l3377_337752


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3377_337793

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) :
  Real.sqrt (a - 2) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3377_337793


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_31_96_l3377_337783

noncomputable def f (x : ℝ) : ℝ := (x^5 - 1) / 3

theorem inverse_f_at_negative_31_96 : f⁻¹ (-31/96) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_31_96_l3377_337783


namespace NUMINAMATH_CALUDE_kim_morning_routine_time_l3377_337705

/-- Represents the types of employees in Kim's office. -/
inductive EmployeeType
  | Senior
  | Junior
  | Intern

/-- Represents whether an employee worked overtime or not. -/
inductive OvertimeStatus
  | Overtime
  | NoOvertime

/-- Calculates the total time for Kim's morning routine. -/
def morning_routine_time (
  senior_count junior_count intern_count : Nat
  ) (
  senior_overtime junior_overtime intern_overtime : Nat
  ) : Nat :=
  let coffee_time := 5
  let status_update_time :=
    3 * senior_count + 2 * junior_count + 1 * intern_count
  let payroll_update_time :=
    4 * senior_overtime + 2 * (senior_count - senior_overtime) +
    3 * junior_overtime + 1 * (junior_count - junior_overtime) +
    2 * intern_overtime + 1 -- 1 minute for 2 interns without overtime (30 seconds each)
  let task_allocation_time :=
    4 * senior_count + 3 * junior_count + 2 * intern_count
  let additional_tasks_time := 10 + 8 + 6 + 5

  coffee_time + status_update_time + payroll_update_time +
  task_allocation_time + additional_tasks_time

/-- Theorem stating that Kim's morning routine takes 101 minutes. -/
theorem kim_morning_routine_time :
  morning_routine_time 3 3 3 2 3 1 = 101 := by
  sorry

end NUMINAMATH_CALUDE_kim_morning_routine_time_l3377_337705


namespace NUMINAMATH_CALUDE_factorial_square_root_squared_l3377_337720

theorem factorial_square_root_squared : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_squared_l3377_337720


namespace NUMINAMATH_CALUDE_intersection_forms_hyperbola_l3377_337758

/-- The equation of the first line -/
def line1 (t x y : ℝ) : Prop := t * x - 2 * y - 3 * t = 0

/-- The equation of the second line -/
def line2 (t x y : ℝ) : Prop := x - 2 * t * y + 3 = 0

/-- The equation of a hyperbola -/
def is_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / (9/4) = 1

/-- Theorem stating that the intersection points form a hyperbola -/
theorem intersection_forms_hyperbola :
  ∀ t x y : ℝ, line1 t x y → line2 t x y → is_hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_intersection_forms_hyperbola_l3377_337758


namespace NUMINAMATH_CALUDE_carnival_game_earnings_l3377_337796

/-- The daily earnings of a carnival game -/
def daily_earnings (total_earnings : ℕ) (num_days : ℕ) : ℚ :=
  total_earnings / num_days

/-- Theorem stating that the daily earnings are $144 -/
theorem carnival_game_earnings : daily_earnings 3168 22 = 144 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_earnings_l3377_337796


namespace NUMINAMATH_CALUDE_river_depth_calculation_l3377_337711

theorem river_depth_calculation (depth_mid_may : ℝ) : 
  let depth_mid_june := depth_mid_may + 10
  let depth_june_20 := depth_mid_june - 5
  let depth_july_5 := depth_june_20 + 8
  let depth_mid_july := depth_july_5
  depth_mid_july = 45 → depth_mid_may = 32 := by sorry

end NUMINAMATH_CALUDE_river_depth_calculation_l3377_337711


namespace NUMINAMATH_CALUDE_product_sum_and_32_l3377_337753

theorem product_sum_and_32 : (12 + 25 + 52 + 21) * 32 = 3520 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_and_32_l3377_337753


namespace NUMINAMATH_CALUDE_q_value_at_minus_one_l3377_337737

-- Define the polynomial q(x)
def q (a b x : ℤ) : ℤ := x^2 + a*x + b

-- Define the two polynomials that q(x) divides
def p1 (x : ℤ) : ℤ := x^4 + 8*x^2 + 49
def p2 (x : ℤ) : ℤ := 2*x^4 + 5*x^2 + 18*x + 3

-- Theorem statement
theorem q_value_at_minus_one 
  (a b : ℤ) 
  (h1 : ∀ x, (p1 x) % (q a b x) = 0)
  (h2 : ∀ x, (p2 x) % (q a b x) = 0) :
  q a b (-1) = 66 := by
  sorry

end NUMINAMATH_CALUDE_q_value_at_minus_one_l3377_337737


namespace NUMINAMATH_CALUDE_chimney_bricks_count_l3377_337757

/-- The number of bricks in the chimney. -/
def chimney_bricks : ℕ := 288

/-- The time it takes Brenda to build the chimney alone (in hours). -/
def brenda_time : ℕ := 8

/-- The time it takes Brandon to build the chimney alone (in hours). -/
def brandon_time : ℕ := 12

/-- The reduction in combined output when working together (in bricks per hour). -/
def output_reduction : ℕ := 12

/-- The time it takes Brenda and Brandon to build the chimney together (in hours). -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the chimney is 288. -/
theorem chimney_bricks_count : 
  chimney_bricks = 288 ∧
  brenda_time = 8 ∧
  brandon_time = 12 ∧
  output_reduction = 12 ∧
  combined_time = 6 ∧
  (combined_time * ((chimney_bricks / brenda_time + chimney_bricks / brandon_time) - output_reduction) = chimney_bricks) :=
by sorry

end NUMINAMATH_CALUDE_chimney_bricks_count_l3377_337757


namespace NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l3377_337771

/-- Proves that if an item is sold at a 20% loss for 960 units, then its original price was 1200 units. -/
theorem original_price_from_loss_and_selling_price 
  (loss_percentage : ℝ) 
  (selling_price : ℝ) : 
  loss_percentage = 20 → 
  selling_price = 960 → 
  (1 - loss_percentage / 100) * (selling_price / (1 - loss_percentage / 100)) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l3377_337771


namespace NUMINAMATH_CALUDE_repeated_digit_sum_tower_exp_l3377_337735

-- Define the function for the tower of exponents
def tower_exp (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 6^(tower_exp n)

-- Define the repeated digit sum operation (conceptually)
def repeated_digit_sum (n : ℕ) : ℕ := n % 11

-- State the theorem
theorem repeated_digit_sum_tower_exp : 
  repeated_digit_sum (7^(tower_exp 5)) = 4 := by sorry

end NUMINAMATH_CALUDE_repeated_digit_sum_tower_exp_l3377_337735


namespace NUMINAMATH_CALUDE_circumcircle_equation_l3377_337736

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Define point P on the parabola
def P : ℝ × ℝ := (-4, -4)

-- Define the tangent line at P
def tangent_line (x y : ℝ) : Prop := y = 2*x + 4

-- Define point Q as the intersection of the tangent line and x-axis
def Q : ℝ × ℝ := (-2, 0)

-- Theorem statement
theorem circumcircle_equation :
  ∀ x y : ℝ,
  parabola (P.1) (P.2) →
  tangent_line (Q.1) (Q.2) →
  (x^2 + y^2 + 4*x + 5*y + 4 = 0) ↔ 
  ((x - (-2))^2 + (y - (-5/2))^2 = (5/2)^2) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l3377_337736


namespace NUMINAMATH_CALUDE_factorization_proof_l3377_337728

theorem factorization_proof (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3377_337728


namespace NUMINAMATH_CALUDE_gina_textbooks_l3377_337768

/-- Calculates the number of textbooks Gina needs to buy given her college expenses. -/
def calculate_textbooks (credits : ℕ) (credit_cost : ℕ) (facilities_fee : ℕ) (textbook_cost : ℕ) (total_spending : ℕ) : ℕ :=
  let credit_total := credits * credit_cost
  let non_textbook_cost := credit_total + facilities_fee
  let textbook_budget := total_spending - non_textbook_cost
  textbook_budget / textbook_cost

theorem gina_textbooks :
  calculate_textbooks 14 450 200 120 7100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gina_textbooks_l3377_337768


namespace NUMINAMATH_CALUDE_perfect_square_k_l3377_337799

theorem perfect_square_k (Z K : ℤ) 
  (h1 : 50 < Z ∧ Z < 5000)
  (h2 : K > 1)
  (h3 : Z = K * K^2) :
  ∃ (n : ℤ), K = n^2 ∧ 50 < K^3 ∧ K^3 ≤ 5000 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_k_l3377_337799


namespace NUMINAMATH_CALUDE_journey_distance_is_420_l3377_337794

/-- Represents the journey details -/
structure Journey where
  urban_speed : ℝ
  highway_speed : ℝ
  urban_time : ℝ
  highway_time : ℝ

/-- Calculates the total distance of the journey -/
def total_distance (j : Journey) : ℝ :=
  j.urban_speed * j.urban_time + j.highway_speed * j.highway_time

/-- Theorem stating that the journey distance is 420 km -/
theorem journey_distance_is_420 (j : Journey) 
  (h1 : j.urban_speed = 55)
  (h2 : j.highway_speed = 85)
  (h3 : j.urban_time = 3)
  (h4 : j.highway_time = 3) :
  total_distance j = 420 := by
  sorry

#eval total_distance { urban_speed := 55, highway_speed := 85, urban_time := 3, highway_time := 3 }

end NUMINAMATH_CALUDE_journey_distance_is_420_l3377_337794


namespace NUMINAMATH_CALUDE_number_equation_l3377_337725

theorem number_equation (x n : ℝ) : 
  x = 596.95 → 3639 + n - x = 3054 → n = 11.95 := by sorry

end NUMINAMATH_CALUDE_number_equation_l3377_337725


namespace NUMINAMATH_CALUDE_johns_allowance_l3377_337759

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (3/5 : ℚ) * A + (1/3 : ℚ) * (A - (3/5 : ℚ) * A) + (9/10 : ℚ) = A → A = (27/8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l3377_337759


namespace NUMINAMATH_CALUDE_milk_powder_cost_july_l3377_337718

/-- Proves the cost of milk powder in July given the conditions from the problem -/
theorem milk_powder_cost_july (june_cost : ℝ) 
  (h1 : june_cost > 0)
  (h2 : (4 * june_cost + 0.2 * june_cost) * 1.5 = 6.3) : 
  0.2 * june_cost = 0.2 := by
sorry

end NUMINAMATH_CALUDE_milk_powder_cost_july_l3377_337718


namespace NUMINAMATH_CALUDE_triangle_area_upper_bound_l3377_337755

/-- Given a triangle ABC with BC = 2 and AB · AC = 1, prove that its area is at most √2 -/
theorem triangle_area_upper_bound (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
  let triangle_area := Real.sqrt (((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))^2 / 4)
  Real.sqrt ((BC.1^2 + BC.2^2) / 4) = 1 →
  dot_product AB AC = 1 →
  triangle_area ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_upper_bound_l3377_337755


namespace NUMINAMATH_CALUDE_problem_solution_l3377_337781

def p (m : ℝ) : Prop := ∀ x, 2*x - 5 > 0 → x > m

def q (m : ℝ) : Prop := ∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ 
  ∀ x y, x^2/(m-1) + y^2/(2-m) = 1 ↔ (x/a)^2 - (y/b)^2 = 1

theorem problem_solution (m : ℝ) : 
  (p m ∧ q m → m < 1 ∨ (2 < m ∧ m ≤ 5/2)) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) → (1 ≤ m ∧ m ≤ 2) ∨ m > 5/2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3377_337781


namespace NUMINAMATH_CALUDE_gcd_360_150_l3377_337782

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_150_l3377_337782
