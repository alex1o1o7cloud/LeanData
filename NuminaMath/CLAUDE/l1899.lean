import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l1899_189913

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l1899_189913


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1899_189979

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 5 - x ∧ y ≥ 0) ↔ x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1899_189979


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1899_189916

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then 1 - Real.log x
  else if x > 1 then -1 + Real.log x
  else 0  -- This case is added to make the function total, but it's not used in our problem

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ab : f a = f b) :
  (∃ m : ℝ, m = 1 + 1 / Real.exp 2 ∧ ∀ x y : ℝ, 0 < x → 0 < y → f x = f y → 1 / x + 1 / y ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1899_189916


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2x_plus_1_range_of_t_when_f_geq_g_l1899_189941

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (t : ℝ) (x : ℝ) : ℝ := t * |x| - 2

-- Theorem for the first part of the problem
theorem solution_set_f_gt_2x_plus_1 :
  {x : ℝ | f x > 2 * x + 1} = {x : ℝ | x < 0} := by sorry

-- Theorem for the second part of the problem
theorem range_of_t_when_f_geq_g (t : ℝ) :
  (∀ x : ℝ, f x ≥ g t x) → t ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2x_plus_1_range_of_t_when_f_geq_g_l1899_189941


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1899_189912

theorem factor_difference_of_squares (x : ℝ) : 
  81 - 16 * (x - 1)^2 = (13 - 4*x) * (5 + 4*x) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1899_189912


namespace NUMINAMATH_CALUDE_solve_ice_cream_problem_l1899_189944

def ice_cream_problem (aaron_savings : ℚ) (carson_savings : ℚ) (dinner_bill_ratio : ℚ) 
  (ice_cream_cost_per_scoop : ℚ) (change_per_person : ℚ) : Prop :=
  let total_savings := aaron_savings + carson_savings
  let dinner_cost := dinner_bill_ratio * total_savings
  let remaining_money := total_savings - dinner_cost
  let ice_cream_total_cost := remaining_money - 2 * change_per_person
  let total_scoops := ice_cream_total_cost / ice_cream_cost_per_scoop
  (total_scoops / 2 : ℚ) = 6

theorem solve_ice_cream_problem :
  ice_cream_problem 40 40 (3/4) (3/2) 1 :=
by
  sorry

#check solve_ice_cream_problem

end NUMINAMATH_CALUDE_solve_ice_cream_problem_l1899_189944


namespace NUMINAMATH_CALUDE_triangle_problem_l1899_189991

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1899_189991


namespace NUMINAMATH_CALUDE_rounding_proof_l1899_189996

def base : ℚ := 1003 / 1000

def power : ℕ := 4

def exact_result : ℚ := base ^ power

def rounded_result : ℚ := 1012 / 1000

def decimal_places : ℕ := 3

theorem rounding_proof : 
  (round (exact_result * 10^decimal_places) / 10^decimal_places) = rounded_result := by
  sorry

end NUMINAMATH_CALUDE_rounding_proof_l1899_189996


namespace NUMINAMATH_CALUDE_special_function_value_l1899_189999

/-- A function satisfying specific conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) ≤ f x + 4) ∧
  (∀ x : ℝ, f (x + 2) ≥ f x + 2) ∧
  (f 3 = 4)

/-- Theorem stating the value of f(2007) for a special function f -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2007 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l1899_189999


namespace NUMINAMATH_CALUDE_vector_magnitude_range_l1899_189964

open Real
open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_range (a b : V) 
  (h1 : ‖b‖ = 2) 
  (h2 : ‖a‖ = 2 * ‖b - a‖) : 
  4/3 ≤ ‖a‖ ∧ ‖a‖ ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_l1899_189964


namespace NUMINAMATH_CALUDE_shooting_match_sequences_l1899_189981

/-- The number of permutations of a multiset with the given multiplicities -/
def multiset_permutations (n : ℕ) (multiplicities : List ℕ) : ℕ :=
  n.factorial / (multiplicities.map Nat.factorial).prod

/-- The number of different sequences for breaking targets in the shooting match -/
theorem shooting_match_sequences : 
  multiset_permutations 10 [3, 3, 2, 2] = 25200 := by
  sorry

end NUMINAMATH_CALUDE_shooting_match_sequences_l1899_189981


namespace NUMINAMATH_CALUDE_blue_pigment_percentage_l1899_189920

/-- Represents the composition of the brown paint mixture --/
structure BrownPaint where
  total_weight : ℝ
  blue_percentage : ℝ
  red_weight : ℝ

/-- Represents the composition of the dark blue paint --/
structure DarkBluePaint where
  blue_percentage : ℝ
  red_percentage : ℝ

/-- Represents the composition of the green paint --/
structure GreenPaint where
  blue_percentage : ℝ
  yellow_percentage : ℝ

/-- Theorem stating the percentage of blue pigment in dark blue and green paints --/
theorem blue_pigment_percentage
  (brown : BrownPaint)
  (dark_blue : DarkBluePaint)
  (green : GreenPaint)
  (h1 : brown.total_weight = 10)
  (h2 : brown.blue_percentage = 0.4)
  (h3 : brown.red_weight = 3)
  (h4 : dark_blue.red_percentage = 0.6)
  (h5 : green.yellow_percentage = 0.6)
  (h6 : dark_blue.blue_percentage = green.blue_percentage) :
  dark_blue.blue_percentage = 0.2 :=
sorry


end NUMINAMATH_CALUDE_blue_pigment_percentage_l1899_189920


namespace NUMINAMATH_CALUDE_apple_orange_probability_l1899_189942

theorem apple_orange_probability (n : ℕ) : 
  (n : ℚ) / (n + 3 : ℚ) = 2 / 3 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_probability_l1899_189942


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1899_189977

theorem sqrt_difference_equality : Real.sqrt (64 + 36) - Real.sqrt (81 - 64) = 10 - Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1899_189977


namespace NUMINAMATH_CALUDE_cost_price_satisfies_conditions_l1899_189949

/-- The cost price of a book satisfying the given conditions. -/
def cost_price : ℝ := 3000

/-- The selling price at 10% profit. -/
def selling_price_10_percent : ℝ := cost_price * 1.1

/-- The selling price at 15% profit. -/
def selling_price_15_percent : ℝ := cost_price * 1.15

/-- Theorem stating that the cost price satisfies the given conditions. -/
theorem cost_price_satisfies_conditions :
  (selling_price_15_percent - selling_price_10_percent = 150) ∧
  (selling_price_10_percent = cost_price * 1.1) ∧
  (selling_price_15_percent = cost_price * 1.15) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_satisfies_conditions_l1899_189949


namespace NUMINAMATH_CALUDE_budget_allocation_home_electronics_l1899_189915

theorem budget_allocation_home_electronics (total_degrees : ℝ) 
  (microphotonics_percent : ℝ) (food_additives_percent : ℝ) 
  (genetically_modified_microorganisms_percent : ℝ) (industrial_lubricants_percent : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  total_degrees = 360 ∧ 
  microphotonics_percent = 13 ∧ 
  food_additives_percent = 15 ∧ 
  genetically_modified_microorganisms_percent = 29 ∧ 
  industrial_lubricants_percent = 8 ∧ 
  basic_astrophysics_degrees = 39.6 →
  (100 - (microphotonics_percent + food_additives_percent + 
    genetically_modified_microorganisms_percent + industrial_lubricants_percent + 
    (basic_astrophysics_degrees / total_degrees * 100))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_home_electronics_l1899_189915


namespace NUMINAMATH_CALUDE_x_fourth_minus_six_x_l1899_189959

theorem x_fourth_minus_six_x (x : ℝ) : x = 3 → x^4 - 6*x = 63 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_six_x_l1899_189959


namespace NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l1899_189932

theorem division_of_mixed_number_by_fraction :
  (3 : ℚ) / 2 / ((5 : ℚ) / 6) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l1899_189932


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l1899_189990

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulationCost (length width height costPerSquareFoot : ℝ) : ℝ :=
  surfaceArea length width height * costPerSquareFoot

/-- Theorem: The cost to insulate a 4x5x2 feet tank at $20 per square foot is $1520 -/
theorem tank_insulation_cost :
  insulationCost 4 5 2 20 = 1520 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l1899_189990


namespace NUMINAMATH_CALUDE_lunch_bill_total_l1899_189902

-- Define the costs and discounts
def hotdog_cost : ℝ := 5.36
def salad_cost : ℝ := 5.10
def soda_original_cost : ℝ := 2.95
def chips_original_cost : ℝ := 1.89
def chips_discount : ℝ := 0.15
def soda_discount : ℝ := 0.10

-- Define the function to calculate the discounted price
def apply_discount (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price * (1 - discount)

-- Define the total cost function
def total_cost : ℝ :=
  hotdog_cost + salad_cost + 
  apply_discount soda_original_cost soda_discount +
  apply_discount chips_original_cost chips_discount

-- Theorem statement
theorem lunch_bill_total : total_cost = 14.7215 := by
  sorry

end NUMINAMATH_CALUDE_lunch_bill_total_l1899_189902


namespace NUMINAMATH_CALUDE_reflected_hyperbola_l1899_189943

/-- Given a hyperbola with equation xy = 1 reflected over the line y = 2x,
    the resulting hyperbola has the equation 12y² + 7xy - 12x² = 25 -/
theorem reflected_hyperbola (x y : ℝ) :
  (∃ x₀ y₀, x₀ * y₀ = 1 ∧ 
   ∃ x₁ y₁, y₁ = 2 * x₁ ∧
   ∃ x₂ y₂, (x₂ - x₀) = (y₁ - y₀) ∧ (y₂ - y₀) = -(x₁ - x₀) ∧
   x = x₂ ∧ y = y₂) →
  12 * y^2 + 7 * x * y - 12 * x^2 = 25 :=
by sorry


end NUMINAMATH_CALUDE_reflected_hyperbola_l1899_189943


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1899_189950

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1899_189950


namespace NUMINAMATH_CALUDE_even_function_sine_condition_l1899_189929

theorem even_function_sine_condition 
  (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x : ℝ, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (-x) + φ)) ↔ 
  ∃ k : ℤ, φ = k * π + π / 2 := by
sorry

end NUMINAMATH_CALUDE_even_function_sine_condition_l1899_189929


namespace NUMINAMATH_CALUDE_quadratic_root_and_m_l1899_189914

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 + 2*x + m = 0

-- Theorem statement
theorem quadratic_root_and_m :
  ∀ m : ℝ, quadratic_equation (-2) m → m = 0 ∧ quadratic_equation 0 m :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_and_m_l1899_189914


namespace NUMINAMATH_CALUDE_pandas_weekly_bamboo_consumption_l1899_189973

/-- The amount of bamboo eaten by pandas in a week -/
def bamboo_eaten_in_week (adult_daily : ℕ) (baby_daily : ℕ) : ℕ :=
  (adult_daily + baby_daily) * 7

/-- Theorem: Pandas eat 1316 pounds of bamboo in a week -/
theorem pandas_weekly_bamboo_consumption :
  bamboo_eaten_in_week 138 50 = 1316 := by
  sorry

end NUMINAMATH_CALUDE_pandas_weekly_bamboo_consumption_l1899_189973


namespace NUMINAMATH_CALUDE_tangent_slope_sin_pi_over_four_l1899_189982

theorem tangent_slope_sin_pi_over_four :
  let f : ℝ → ℝ := fun x ↦ Real.sin x
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_sin_pi_over_four_l1899_189982


namespace NUMINAMATH_CALUDE_no_solution_iff_m_zero_l1899_189972

theorem no_solution_iff_m_zero (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (2 - x) / (1 - x) ≠ (m + x) / (1 - x) + 1) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_zero_l1899_189972


namespace NUMINAMATH_CALUDE_max_value_f_in_interval_l1899_189956

/-- The function f(x) = x^3 - 3x^2 + 2 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

/-- Theorem: The maximum value of f(x) = x^3 - 3x^2 + 2 in the interval [-1, 1] is 2 -/
theorem max_value_f_in_interval :
  (∀ x : ℝ, x ≥ -1 ∧ x ≤ 1 → f x ≤ 2) ∧
  (∃ x : ℝ, x ≥ -1 ∧ x ≤ 1 ∧ f x = 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_f_in_interval_l1899_189956


namespace NUMINAMATH_CALUDE_intersection_A_B_l1899_189974

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x : ℝ | 2 - x > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1899_189974


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1899_189930

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.log x - x + 1 ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ Real.log x - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1899_189930


namespace NUMINAMATH_CALUDE_blinks_per_minute_l1899_189928

theorem blinks_per_minute (x : ℚ) 
  (h1 : x - (3/5 : ℚ) * x = 10) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_blinks_per_minute_l1899_189928


namespace NUMINAMATH_CALUDE_city_distance_l1899_189934

def is_valid_distance (S : ℕ+) : Prop :=
  ∀ x : ℕ, x < S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)

theorem city_distance : ∃ S : ℕ+, is_valid_distance S ∧ ∀ T : ℕ+, T < S → ¬is_valid_distance T :=
  sorry

end NUMINAMATH_CALUDE_city_distance_l1899_189934


namespace NUMINAMATH_CALUDE_fourteen_machines_four_minutes_l1899_189926

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let base_machines := 6
  let base_production := 270
  let production_per_machine_per_minute := base_production / base_machines
  machines * production_per_machine_per_minute * minutes

/-- Theorem stating that 14 machines produce 2520 bottles in 4 minutes -/
theorem fourteen_machines_four_minutes :
  bottles_produced 14 4 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_machines_four_minutes_l1899_189926


namespace NUMINAMATH_CALUDE_probability_of_valid_sequence_l1899_189909

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- The probability of a valid sequence of length 8 -/
def probability : ℚ := validSequences 8 / totalSequences 8

theorem probability_of_valid_sequence :
  probability = 55 / 256 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_sequence_l1899_189909


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l1899_189908

theorem binomial_expansion_problem (a b : ℝ) : 
  (∃ c d e : ℝ, (1 + a * x)^5 = 1 + 10*x + b*x^2 + c*x^3 + d*x^4 + a^5*x^5) → 
  a - b = -38 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l1899_189908


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l1899_189986

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l1899_189986


namespace NUMINAMATH_CALUDE_light_path_in_cube_l1899_189921

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light beam in the cube -/
structure LightBeam where
  start : Point3D
  reflectionPoint : Point3D

/-- The length of the light path in the cube -/
def lightPathLength (c : Cube) (lb : LightBeam) : ℝ :=
  sorry

theorem light_path_in_cube (c : Cube) (lb : LightBeam) :
  c.sideLength = 10 ∧
  lb.start = Point3D.mk 0 0 0 ∧
  lb.reflectionPoint = Point3D.mk 6 4 10 →
  lightPathLength c lb = 10 * Real.sqrt 152 :=
sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l1899_189921


namespace NUMINAMATH_CALUDE_vasya_numbers_l1899_189901

theorem vasya_numbers : ∃! (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l1899_189901


namespace NUMINAMATH_CALUDE_candy_distribution_l1899_189955

theorem candy_distribution (total_candies : ℕ) (num_bags : ℕ) (candies_per_bag : ℕ) :
  total_candies = 15 →
  num_bags = 5 →
  total_candies = num_bags * candies_per_bag →
  candies_per_bag = 3 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1899_189955


namespace NUMINAMATH_CALUDE_cosine_matrix_det_zero_l1899_189938

theorem cosine_matrix_det_zero : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos 1, Real.cos 2, Real.cos 3],
    ![Real.cos 4, Real.cos 5, Real.cos 6],
    ![Real.cos 7, Real.cos 8, Real.cos 9]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_matrix_det_zero_l1899_189938


namespace NUMINAMATH_CALUDE_geometric_progression_cubed_sum_l1899_189998

theorem geometric_progression_cubed_sum
  (b s : ℝ) (h1 : -1 < s) (h2 : s < 1) :
  let series := fun n => b^3 * s^(3*n)
  (∑' n, series n) = b^3 / (1 - s^3) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_cubed_sum_l1899_189998


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1899_189918

/-- Represents a seating arrangement for 3 people on 5 chairs -/
structure SeatingArrangement where
  seats : Fin 5 → Option (Fin 3)
  all_seated : ∀ p : Fin 3, ∃ s : Fin 5, seats s = some p
  no_sharing : ∀ s : Fin 5, ∀ p q : Fin 3, seats s = some p → seats s = some q → p = q
  ab_adjacent : ∃ s : Fin 5, (seats s = some 0 ∧ seats (s + 1) = some 1) ∨ (seats s = some 1 ∧ seats (s + 1) = some 0)
  not_all_adjacent : ¬∃ s : Fin 5, (seats s).isSome ∧ (seats (s + 1)).isSome ∧ (seats (s + 2)).isSome

/-- The number of valid seating arrangements -/
def num_seating_arrangements : ℕ := sorry

/-- Theorem stating that there are exactly 12 valid seating arrangements -/
theorem seating_arrangements_count : num_seating_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1899_189918


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1899_189936

/-- Proves that a boat's speed in still water is 42 km/hr given specific conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 3) 
  (h2 : downstream_distance = 33) 
  (h3 : downstream_time = 44 / 60) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 42 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1899_189936


namespace NUMINAMATH_CALUDE_brick_packing_theorem_l1899_189969

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular parallelepiped -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

theorem brick_packing_theorem (box : Dimensions) 
  (brick1 brick2 : Dimensions) 
  (h_box : box = ⟨10, 11, 14⟩) 
  (h_brick1 : brick1 = ⟨2, 5, 8⟩) 
  (h_brick2 : brick2 = ⟨2, 3, 7⟩) :
  ∃ (x y : ℕ), 
    x * volume brick1 + y * volume brick2 = volume box ∧ 
    x + y = 24 ∧ 
    ∀ (a b : ℕ), a * volume brick1 + b * volume brick2 = volume box → a + b ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_brick_packing_theorem_l1899_189969


namespace NUMINAMATH_CALUDE_minimum_jars_needed_spice_jar_problem_l1899_189945

theorem minimum_jars_needed 
  (medium_jar_capacity : ℕ) 
  (large_container_capacity : ℕ) 
  (potential_loss : ℕ) : ℕ :=
  let min_jars := (large_container_capacity + medium_jar_capacity - 1) / medium_jar_capacity
  min_jars + potential_loss

theorem spice_jar_problem : 
  minimum_jars_needed 50 825 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_minimum_jars_needed_spice_jar_problem_l1899_189945


namespace NUMINAMATH_CALUDE_suzy_jump_ropes_l1899_189976

theorem suzy_jump_ropes (yesterday : ℕ) (additional : ℕ) : 
  yesterday = 247 → additional = 131 → yesterday + (yesterday + additional) = 625 := by
  sorry

end NUMINAMATH_CALUDE_suzy_jump_ropes_l1899_189976


namespace NUMINAMATH_CALUDE_monotonicity_and_tangent_line_and_max_k_l1899_189911

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem monotonicity_and_tangent_line_and_max_k :
  -- Part 1: Monotonicity of f(x)
  (∀ a : ℝ, a ≤ 0 → StrictMono (f a)) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ x y : ℝ, x < y → y < Real.log a → f a y < f a x) ∧
    (∀ x y : ℝ, Real.log a < x → x < y → f a x < f a y)) ∧
  
  -- Part 2: Tangent line condition
  (∀ a : ℝ, (∃ x₀ : ℝ, f_deriv a x₀ = Real.exp 1 ∧ 
    f a x₀ = Real.exp x₀ - 2) → a = 0) ∧
  
  -- Part 3: Maximum value of k
  (∀ k : ℤ, (∀ x : ℝ, x > 0 → (x - ↑k) * (f_deriv 1 x) + x + 1 > 0) → 
    k ≤ 2) ∧
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (f_deriv 1 x) + x + 1 > 0)
  := by sorry

end NUMINAMATH_CALUDE_monotonicity_and_tangent_line_and_max_k_l1899_189911


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1899_189946

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ (0 ≤ m ∧ m < 8) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1899_189946


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1899_189958

theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I - 1) * z = 2 → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1899_189958


namespace NUMINAMATH_CALUDE_x_eq_two_sufficient_not_necessary_l1899_189960

def M (x : ℝ) : Set ℝ := {1, x}
def N : Set ℝ := {1, 2, 3}

theorem x_eq_two_sufficient_not_necessary :
  ∀ x : ℝ, 
  (x = 2 → M x ⊆ N) ∧ 
  ¬(M x ⊆ N → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_two_sufficient_not_necessary_l1899_189960


namespace NUMINAMATH_CALUDE_fibonacci_sum_convergence_l1899_189904

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_convergence :
  let S : ℝ := ∑' n, (fibonacci n : ℝ) / 5^n
  S = 5/19 := by sorry

end NUMINAMATH_CALUDE_fibonacci_sum_convergence_l1899_189904


namespace NUMINAMATH_CALUDE_system_solution_l1899_189948

theorem system_solution (x y : ℝ) : 
  x - 2*y = -5 → 3*x + 6*y = 7 → x + y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1899_189948


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1899_189957

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 3 = 0 ∧ 
   x₂^2 + a*x₂ + 3 = 0 ∧ 
   x₁^3 - 99/(2*x₂^2) = x₂^3 - 99/(2*x₁^2)) → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1899_189957


namespace NUMINAMATH_CALUDE_rectangle_cutting_l1899_189988

theorem rectangle_cutting (m : ℕ) (h : m > 12) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * y > m ∧ x * (y - 1) < m :=
by sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l1899_189988


namespace NUMINAMATH_CALUDE_rational_smallest_abs_value_and_monomial_degree_l1899_189931

-- Define the concept of absolute value for rational numbers
def abs_rat (q : ℚ) : ℚ := max q (-q)

-- Define the degree of a monomial
def monomial_degree (a b c : ℕ) : ℕ := a + b + c

theorem rational_smallest_abs_value_and_monomial_degree :
  (∀ q : ℚ, abs_rat q ≥ 0) ∧
  (∀ q : ℚ, abs_rat q = 0 ↔ q = 0) ∧
  (monomial_degree 2 1 0 = 3) :=
sorry

end NUMINAMATH_CALUDE_rational_smallest_abs_value_and_monomial_degree_l1899_189931


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l1899_189963

/-- The range of k for which the line y = kx - 1 intersects the right branch of
    the hyperbola x^2 - y^2 = 1 at two different points -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    x₁ > 0 ∧ x₂ > 0 ∧
    x₁^2 - y₁^2 = 1 ∧
    x₂^2 - y₂^2 = 1 ∧
    y₁ = k * x₁ - 1 ∧
    y₂ = k * x₂ - 1) ↔
  (1 < k ∧ k < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l1899_189963


namespace NUMINAMATH_CALUDE_unique_k_square_sum_l1899_189925

theorem unique_k_square_sum : ∃! (k : ℕ), k ≠ 1 ∧
  (∃ (n : ℕ), k = n^2 + (n+1)^2) ∧
  (∃ (m : ℕ), k^4 = m^2 + (m+1)^2) :=
by sorry

end NUMINAMATH_CALUDE_unique_k_square_sum_l1899_189925


namespace NUMINAMATH_CALUDE_dice_probability_l1899_189962

def red_die : Finset Nat := {4, 6}
def yellow_die : Finset Nat := {1, 2, 3, 4, 5, 6}

def total_outcomes : Finset (Nat × Nat) :=
  red_die.product yellow_die

def favorable_outcomes : Finset (Nat × Nat) :=
  total_outcomes.filter (fun p => p.1 * p.2 > 20)

theorem dice_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1899_189962


namespace NUMINAMATH_CALUDE_beef_weight_after_processing_l1899_189978

def initial_weight : ℝ := 1500
def weight_loss_percentage : ℝ := 50

theorem beef_weight_after_processing :
  let final_weight := initial_weight * (1 - weight_loss_percentage / 100)
  final_weight = 750 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_after_processing_l1899_189978


namespace NUMINAMATH_CALUDE_christian_yard_charge_l1899_189966

/-- Proves that Christian charged $5 for mowing each yard --/
theorem christian_yard_charge :
  let perfume_cost : ℚ := 50
  let christian_savings : ℚ := 5
  let sue_savings : ℚ := 7
  let christian_yards : ℕ := 4
  let sue_dogs : ℕ := 6
  let sue_dog_charge : ℚ := 2
  let additional_needed : ℚ := 6
  
  let total_needed := perfume_cost - additional_needed
  let initial_savings := christian_savings + sue_savings
  let chores_earnings := total_needed - initial_savings
  let sue_earnings := sue_dogs * sue_dog_charge
  let christian_earnings := chores_earnings - sue_earnings
  let christian_yard_charge := christian_earnings / christian_yards

  christian_yard_charge = 5 := by sorry

end NUMINAMATH_CALUDE_christian_yard_charge_l1899_189966


namespace NUMINAMATH_CALUDE_value_of_x_l1899_189970

theorem value_of_x : ∃ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) ∧ (x = 0) := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1899_189970


namespace NUMINAMATH_CALUDE_bouquet_stamens_l1899_189987

/-- Proves that the total number of stamens in a bouquet is 216 --/
theorem bouquet_stamens :
  ∀ (black_roses crimson_flowers : ℕ),
  (4 * black_roses + 8 * crimson_flowers) - (2 * black_roses + 3 * crimson_flowers) = 108 →
  4 * black_roses + 10 * crimson_flowers = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_bouquet_stamens_l1899_189987


namespace NUMINAMATH_CALUDE_ping_pong_balls_l1899_189939

theorem ping_pong_balls (y w : ℕ) : 
  y = 2 * (w - 10) →
  w - 10 = 5 * (y - 9) →
  y = 10 ∧ w = 15 := by
sorry

end NUMINAMATH_CALUDE_ping_pong_balls_l1899_189939


namespace NUMINAMATH_CALUDE_problem_solution_l1899_189927

theorem problem_solution (x : ℝ) 
  (h1 : 2 * Real.sin x * Real.tan x = 3)
  (h2 : -Real.pi < x ∧ x < 0) : 
  x = -Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1899_189927


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1899_189953

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 48) 
  (h5 : x*y + y*z + z*x = 30) : 
  x + y + z = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1899_189953


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1899_189917

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a-2)^2 - (b-2)^2 = 18) : 
  a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1899_189917


namespace NUMINAMATH_CALUDE_library_books_count_l1899_189997

theorem library_books_count (old_books : ℕ) 
  (h1 : old_books + 300 + 400 - 200 = 1000) : old_books = 500 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1899_189997


namespace NUMINAMATH_CALUDE_circle_increase_l1899_189937

/-- Theorem: When the radius of a circle is increased by 50%, 
    the circumference increases by 50% and the area increases by 125%. -/
theorem circle_increase (r : ℝ) (h : r > 0) : 
  let new_r := 1.5 * r
  let circ := 2 * Real.pi * r
  let new_circ := 2 * Real.pi * new_r
  let area := Real.pi * r^2
  let new_area := Real.pi * new_r^2
  (new_circ - circ) / circ = 0.5 ∧ (new_area - area) / area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_increase_l1899_189937


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l1899_189900

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l1899_189900


namespace NUMINAMATH_CALUDE_clayton_shells_proof_l1899_189905

/-- The number of shells collected by Jillian -/
def jillian_shells : ℕ := 29

/-- The number of shells collected by Savannah -/
def savannah_shells : ℕ := 17

/-- The number of friends who received shells -/
def num_friends : ℕ := 2

/-- The number of shells each friend received -/
def shells_per_friend : ℕ := 27

/-- The number of shells Clayton collected -/
def clayton_shells : ℕ := 8

theorem clayton_shells_proof :
  clayton_shells = 
    num_friends * shells_per_friend - (jillian_shells + savannah_shells) :=
by sorry

end NUMINAMATH_CALUDE_clayton_shells_proof_l1899_189905


namespace NUMINAMATH_CALUDE_floor_equation_solutions_range_l1899_189906

theorem floor_equation_solutions_range (a : ℝ) (n : ℕ) 
  (h1 : a > 1) 
  (h2 : n ≥ 2) 
  (h3 : ∃! (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, ⌊a * x⌋ = x) :
  1 + 1 / n ≤ a ∧ a < 1 + 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_range_l1899_189906


namespace NUMINAMATH_CALUDE_seal_initial_money_l1899_189968

/-- Represents the amount of coins Seal has at each stage --/
def seal_money (initial : ℕ) : ℕ → ℕ
| 0 => initial  -- Initial amount
| 1 => 2 * initial - 20  -- After first crossing
| 2 => 2 * (2 * initial - 20) - 40  -- After second crossing
| 3 => 2 * (2 * (2 * initial - 20) - 40) - 60  -- After third crossing
| _ => 0  -- We only care about the first three crossings

/-- The theorem stating that Seal must have started with 25 coins --/
theorem seal_initial_money : 
  ∃ (initial : ℕ), 
    initial = 25 ∧ 
    seal_money initial 0 > 0 ∧
    seal_money initial 1 > 0 ∧
    seal_money initial 2 > 0 ∧
    seal_money initial 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_seal_initial_money_l1899_189968


namespace NUMINAMATH_CALUDE_correct_calculation_l1899_189933

theorem correct_calculation (x y : ℝ) : -2*x*y + 3*y*x = x*y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1899_189933


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1899_189967

theorem inequality_system_solution_set : 
  {x : ℝ | 2 * x + 1 ≥ 3 ∧ 4 * x - 1 < 7} = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1899_189967


namespace NUMINAMATH_CALUDE_dance_ratio_l1899_189971

/-- Given the conditions of a dance, prove the ratio of boys to girls -/
theorem dance_ratio :
  ∀ (boys girls teachers : ℕ),
  girls = 60 →
  teachers = boys / 5 →
  boys + girls + teachers = 114 →
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ boys * b = girls * a ∧ a = 3 ∧ b = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_dance_ratio_l1899_189971


namespace NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l1899_189910

def yoongi_apples : ℕ := 4
def jungkook_apples : ℕ := 9
def yuna_apples : ℕ := 5

theorem yoongi_has_fewest_apples :
  yoongi_apples < jungkook_apples ∧ yoongi_apples < yuna_apples :=
sorry

end NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l1899_189910


namespace NUMINAMATH_CALUDE_lanas_tulips_l1899_189984

/-- The number of tulips Lana picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Lana picked -/
def total_flowers : ℕ := sorry

/-- The number of flowers Lana used -/
def used_flowers : ℕ := 70

/-- The number of roses Lana picked -/
def roses : ℕ := 37

/-- The extra flowers Lana picked -/
def extra_flowers : ℕ := 3

theorem lanas_tulips :
  (total_flowers = tulips + roses) →
  (total_flowers = used_flowers + extra_flowers) →
  tulips = 36 := by sorry

end NUMINAMATH_CALUDE_lanas_tulips_l1899_189984


namespace NUMINAMATH_CALUDE_mollys_bike_age_l1899_189919

/-- Molly's bike riding problem -/
theorem mollys_bike_age : 
  ∀ (miles_per_day : ℕ) (age_stopped : ℕ) (total_miles : ℕ) (days_per_year : ℕ),
  miles_per_day = 3 →
  age_stopped = 16 →
  total_miles = 3285 →
  days_per_year = 365 →
  age_stopped - (total_miles / miles_per_day / days_per_year) = 13 := by
sorry

end NUMINAMATH_CALUDE_mollys_bike_age_l1899_189919


namespace NUMINAMATH_CALUDE_larger_number_proof_l1899_189975

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1899_189975


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1899_189952

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.A * l2.A + l1.B * l2.B = 0

/-- The first line: 4x - (a+1)y + 9 = 0 -/
def line1 (a : ℝ) : Line :=
  { A := 4, B := -(a+1), C := 9 }

/-- The second line: (a^2-1)x - ay + 6 = 0 -/
def line2 (a : ℝ) : Line :=
  { A := a^2-1, B := -a, C := 6 }

/-- Statement: a = -1 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem perpendicular_condition :
  (∀ a : ℝ, a = -1 → are_perpendicular (line1 a) (line2 a)) ∧
  (∃ a : ℝ, a ≠ -1 ∧ are_perpendicular (line1 a) (line2 a)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1899_189952


namespace NUMINAMATH_CALUDE_train_length_l1899_189985

/-- Given a train that crosses a 150-meter platform in 27 seconds and a signal pole in 18 seconds,
    prove that the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : platform_length = 150)
    (h2 : platform_time = 27)
    (h3 : pole_time = 18) : 
  ∃ (train_length : ℝ), train_length = 300 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l1899_189985


namespace NUMINAMATH_CALUDE_right_triangle_from_equation_l1899_189935

theorem right_triangle_from_equation (a b c : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (h : (a - 6)^2 + Real.sqrt (b - 8) + |c - 10| = 0) : 
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_equation_l1899_189935


namespace NUMINAMATH_CALUDE_magic_box_solution_l1899_189995

-- Define the magic box function
def magicBox (a b : ℝ) : ℝ := a^2 + b - 1

-- State the theorem
theorem magic_box_solution :
  ∀ m : ℝ, magicBox m (-2*m) = 2 → m = 3 ∨ m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_magic_box_solution_l1899_189995


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1899_189923

theorem other_root_of_quadratic (k : ℝ) : 
  (1 : ℝ) ^ 2 + k * 1 - 2 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 + k * x - 2 = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1899_189923


namespace NUMINAMATH_CALUDE_product_of_digits_less_than_number_l1899_189951

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def digit_product (n : ℕ) : ℕ :=
  (digits n).prod

theorem product_of_digits_less_than_number (N : ℕ) (h : N > 9) :
  digit_product N < N :=
sorry

end NUMINAMATH_CALUDE_product_of_digits_less_than_number_l1899_189951


namespace NUMINAMATH_CALUDE_divisors_of_square_of_four_divisor_number_l1899_189924

/-- A natural number has exactly 4 divisors -/
def has_four_divisors (m : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 4

/-- The number of divisors of the square of a number with 4 divisors -/
theorem divisors_of_square_of_four_divisor_number (m : ℕ) :
  has_four_divisors m →
  (Finset.filter (· ∣ m^2) (Finset.range (m^2 + 1))).card = 7 ∨
  (Finset.filter (· ∣ m^2) (Finset.range (m^2 + 1))).card = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_divisors_of_square_of_four_divisor_number_l1899_189924


namespace NUMINAMATH_CALUDE_road_system_car_distribution_l1899_189903

theorem road_system_car_distribution :
  ∀ (total_cars : ℕ) (bc de bd ce cd : ℕ),
    total_cars = 36 →
    bc = de + 10 →
    cd = 2 →
    total_cars = bc + bd →
    bc = cd + ce →
    de = bd - cd →
    (bc = 24 ∧ bd = 12 ∧ de = 14 ∧ ce = 22) :=
by
  sorry

end NUMINAMATH_CALUDE_road_system_car_distribution_l1899_189903


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1899_189940

/-- The surface area of the circumscribed sphere of a rectangular parallelepiped
    with face diagonal lengths 2, √3, and √5 is 6π. -/
theorem circumscribed_sphere_surface_area 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 = 4) 
  (h2 : y^2 + z^2 = 3) 
  (h3 : z^2 + x^2 = 5) : 
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 6 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1899_189940


namespace NUMINAMATH_CALUDE_no_real_solution_l1899_189980

theorem no_real_solution : ¬∃ (a b c d : ℝ), 
  a^3 + c^3 = 2 ∧ 
  a^2*b + c^2*d = 0 ∧ 
  b^3 + d^3 = 1 ∧ 
  a*b^2 + c*d^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l1899_189980


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l1899_189961

/-- Given a positive real number a, prove that the difference in area between
    a rectangle with length (a-2) and width 7, and a rectangle with length a
    and width 5, is equal to 2a - 14. -/
theorem rectangle_area_difference (a : ℝ) (h : a > 0) :
  (a - 2) * 7 - a * 5 = 2 * a - 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l1899_189961


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_m_range_l1899_189983

/-- An ellipse centered at the origin with right focus at (1,0) and one vertex at (0,√3) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- Two points are symmetric about the line y = x + m -/
def SymmetricAboutLine (p q : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (t : ℝ), p.1 + q.1 = 2 * t ∧ p.2 + q.2 = 2 * (t + m)

theorem ellipse_symmetric_points_m_range :
  ∀ (m : ℝ),
    (∃ (p q : ℝ × ℝ), p ∈ Ellipse ∧ q ∈ Ellipse ∧ p ≠ q ∧ SymmetricAboutLine p q m) →
    -Real.sqrt 7 / 7 < m ∧ m < Real.sqrt 7 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_m_range_l1899_189983


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1899_189994

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1899_189994


namespace NUMINAMATH_CALUDE_festival_line_up_l1899_189947

/-- Represents the minimum number of Gennadys required for the festival line-up. -/
def min_gennadys (alexanders borises vasilies : ℕ) : ℕ :=
  (borises - 1) - (alexanders + vasilies)

/-- Theorem stating the minimum number of Gennadys required for the festival. -/
theorem festival_line_up (alexanders borises vasilies : ℕ) 
  (h1 : alexanders = 45)
  (h2 : borises = 122)
  (h3 : vasilies = 27) :
  min_gennadys alexanders borises vasilies = 49 := by
  sorry

#eval min_gennadys 45 122 27

end NUMINAMATH_CALUDE_festival_line_up_l1899_189947


namespace NUMINAMATH_CALUDE_integer_triangle_area_rational_l1899_189992

/-- A triangle with integer coordinates where two points form a line parallel to the x-axis -/
structure IntegerTriangle where
  x₁ : ℤ
  y₁ : ℤ
  x₂ : ℤ
  y₂ : ℤ
  x₃ : ℤ
  y₃ : ℤ
  parallel_to_x : y₁ = y₂

/-- The area of an IntegerTriangle is rational -/
theorem integer_triangle_area_rational (t : IntegerTriangle) : ∃ (q : ℚ), q = |((t.x₂ - t.x₁) * t.y₃) / 2| := by
  sorry

end NUMINAMATH_CALUDE_integer_triangle_area_rational_l1899_189992


namespace NUMINAMATH_CALUDE_bridge_length_l1899_189907

/-- The length of a bridge given train parameters --/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_length + (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 215 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1899_189907


namespace NUMINAMATH_CALUDE_min_value_expression_l1899_189922

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1899_189922


namespace NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l1899_189954

/-- Given a quadratic function f(x) = ax^2 + bx + c where a > b > c and a + b + c = 0,
    prove that f(x) is negative for all x in the open interval (0,1) -/
theorem quadratic_negative_on_unit_interval 
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x ∈ Set.Ioo 0 1, a * x^2 + b * x + c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l1899_189954


namespace NUMINAMATH_CALUDE_company_female_managers_l1899_189993

/-- Represents the number of female managers in a company -/
def female_managers (total_employees : ℕ) (female_employees : ℕ) (male_employees : ℕ) : ℕ :=
  (2 * female_employees) / 5

theorem company_female_managers :
  let total_employees := female_employees + male_employees
  let female_employees := 625
  let total_managers := (2 * total_employees) / 5
  let male_managers := (2 * male_employees) / 5
  female_managers total_employees female_employees male_employees = 250 :=
by
  sorry

#check company_female_managers

end NUMINAMATH_CALUDE_company_female_managers_l1899_189993


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1899_189965

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 11) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 184 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1899_189965


namespace NUMINAMATH_CALUDE_max_profit_month_and_value_l1899_189989

def f (x : ℕ) : ℝ := -3 * x^2 + 40 * x

def q (x : ℕ) : ℝ := 150 + 2 * x

def profit (x : ℕ) : ℝ := (185 - q x) * f x

theorem max_profit_month_and_value :
  ∃ (x : ℕ), 1 ≤ x ∧ x ≤ 12 ∧
  (∀ (y : ℕ), 1 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧
  x = 5 ∧ profit x = 3125 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_month_and_value_l1899_189989
