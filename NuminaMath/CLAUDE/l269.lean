import Mathlib

namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l269_26968

theorem smallest_satisfying_number : ∃ (n : ℕ), n = 1806 ∧ 
  (∀ (m : ℕ), m < n → 
    ∃ (p : ℕ), Prime p ∧ (m % (p - 1) = 0 → m % p ≠ 0)) ∧
  (∀ (p : ℕ), Prime p → (n % (p - 1) = 0 → n % p = 0)) := by
  sorry

#check smallest_satisfying_number

end NUMINAMATH_CALUDE_smallest_satisfying_number_l269_26968


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l269_26998

/-- The common factor of the polynomial 2m^2n + 6mn - 4m^3n is 2mn -/
theorem common_factor_of_polynomial (m n : ℤ) : 
  ∃ (k : ℤ), 2 * m^2 * n + 6 * m * n - 4 * m^3 * n = 2 * m * n * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l269_26998


namespace NUMINAMATH_CALUDE_complement_union_theorem_l269_26979

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l269_26979


namespace NUMINAMATH_CALUDE_thirdYearSelected_l269_26941

/-- Represents the number of students in each year -/
structure StudentPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Calculates the total number of students -/
def totalStudents (pop : StudentPopulation) : ℕ :=
  pop.firstYear + pop.secondYear + pop.thirdYear

/-- Calculates the number of students selected from a specific year -/
def selectedFromYear (pop : StudentPopulation) (year : ℕ) (sampleSize : ℕ) : ℕ :=
  (year * sampleSize) / totalStudents pop

/-- Theorem: The number of third-year students selected in the stratified sampling -/
theorem thirdYearSelected (pop : StudentPopulation) (sampleSize : ℕ) :
  pop.firstYear = 150 →
  pop.secondYear = 120 →
  pop.thirdYear = 180 →
  sampleSize = 50 →
  selectedFromYear pop pop.thirdYear sampleSize = 20 := by
  sorry


end NUMINAMATH_CALUDE_thirdYearSelected_l269_26941


namespace NUMINAMATH_CALUDE_max_dot_product_l269_26980

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

-- Define the moving point M
def M : Set (ℝ × ℝ) := {p | -2 ≤ p.1 ∧ p.1 ≤ 2 ∧ -2 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product :
  ∃ (max : ℝ), max = 4 ∧ ∀ m ∈ M, dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l269_26980


namespace NUMINAMATH_CALUDE_not_perfect_square_l269_26943

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), m^2 = 4*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l269_26943


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l269_26937

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 40)
  (sum3 : c + a = 45) :
  a + b + c = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l269_26937


namespace NUMINAMATH_CALUDE_min_marbles_for_ten_of_one_color_l269_26954

/-- Represents the number of marbles of each color in the container -/
structure MarbleContainer :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (white : ℕ)
  (black : ℕ)

/-- Defines the specific container from the problem -/
def problemContainer : MarbleContainer :=
  { red := 30
  , green := 25
  , yellow := 23
  , blue := 15
  , white := 10
  , black := 7 }

/-- 
  Theorem: The minimum number of marbles that must be drawn from the container
  without replacement to ensure that at least 10 marbles of a single color are drawn is 53.
-/
theorem min_marbles_for_ten_of_one_color (container : MarbleContainer := problemContainer) :
  (∃ (n : ℕ), n = 53 ∧
    (∀ (m : ℕ), m < n →
      ∃ (r g y b w bl : ℕ),
        r + g + y + b + w + bl = m ∧
        r ≤ container.red ∧
        g ≤ container.green ∧
        y ≤ container.yellow ∧
        b ≤ container.blue ∧
        w ≤ container.white ∧
        bl ≤ container.black ∧
        r < 10 ∧ g < 10 ∧ y < 10 ∧ b < 10 ∧ w < 10 ∧ bl < 10) ∧
    (∀ (r g y b w bl : ℕ),
      r + g + y + b + w + bl = n →
      r ≤ container.red →
      g ≤ container.green →
      y ≤ container.yellow →
      b ≤ container.blue →
      w ≤ container.white →
      bl ≤ container.black →
      (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10 ∨ bl ≥ 10))) :=
by sorry


end NUMINAMATH_CALUDE_min_marbles_for_ten_of_one_color_l269_26954


namespace NUMINAMATH_CALUDE_negation_of_existential_real_equation_l269_26919

theorem negation_of_existential_real_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_real_equation_l269_26919


namespace NUMINAMATH_CALUDE_trigonometric_identity_l269_26900

theorem trigonometric_identity : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / Real.cos (17 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l269_26900


namespace NUMINAMATH_CALUDE_integer_ratio_difference_l269_26962

theorem integer_ratio_difference (a b c : ℕ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_90 : a + b + c = 90)
  (ratio : 3 * a = 2 * b ∧ 5 * a = 2 * c) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |((c : ℝ) - (a : ℝ)) - 12.846| < ε :=
by sorry

end NUMINAMATH_CALUDE_integer_ratio_difference_l269_26962


namespace NUMINAMATH_CALUDE_center_is_midpoint_distance_between_foci_l269_26977

/-- The equation of an ellipse with foci at (6, -3) and (-4, 5) -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 6)^2 + (y + 3)^2) + Real.sqrt ((x + 4)^2 + (y - 5)^2) = 24

/-- The center of the ellipse -/
def center : ℝ × ℝ := (1, 1)

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (6, -3)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (-4, 5)

/-- The center is the midpoint of the foci -/
theorem center_is_midpoint : center = ((focus1.1 + focus2.1) / 2, (focus1.2 + focus2.2) / 2) := by sorry

/-- The distance between the foci of the ellipse is 2√41 -/
theorem distance_between_foci : 
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_center_is_midpoint_distance_between_foci_l269_26977


namespace NUMINAMATH_CALUDE_unique_input_for_542_l269_26918

def machine_operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 * n else 3 * n + 2

def iterate_machine (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => machine_operation (iterate_machine n k)

theorem unique_input_for_542 :
  ∃! n : ℕ, n > 0 ∧ iterate_machine n 5 = 542 :=
by
  -- The proof would go here
  sorry

#eval iterate_machine 112500 5  -- Should output 542

end NUMINAMATH_CALUDE_unique_input_for_542_l269_26918


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l269_26981

theorem unique_quadratic_solution (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → m = 0 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l269_26981


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l269_26993

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on other axles. -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck based on the number of axles. -/
def calculateToll (axles : ℕ) : ℚ :=
  1.5 + 1.5 * (axles - 2 : ℚ)

/-- Theorem stating that the toll for an 18-wheel truck with 2 wheels on the front axle
    and 4 wheels on each other axle is $6.00. -/
theorem eighteen_wheel_truck_toll :
  let axles := calculateAxles 18 2 4
  calculateToll axles = 6 := by
  sorry

#eval calculateAxles 18 2 4
#eval calculateToll (calculateAxles 18 2 4)

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l269_26993


namespace NUMINAMATH_CALUDE_existence_of_m_l269_26956

/-- The number of factors of 2 in m! -/
def n (m : ℕ) : ℕ := sorry

/-- Theorem stating the existence of m satisfying the given conditions -/
theorem existence_of_m : ∃ m : ℕ, m > 1990^1990 ∧ m = 3^1990 + n m := by sorry

end NUMINAMATH_CALUDE_existence_of_m_l269_26956


namespace NUMINAMATH_CALUDE_magnet_cost_is_three_l269_26916

/-- The cost of the magnet at the garage sale -/
def magnet_cost (stuffed_animal_cost : ℚ) : ℚ :=
  (2 * stuffed_animal_cost) / 4

/-- Theorem stating that the magnet cost $3 -/
theorem magnet_cost_is_three :
  magnet_cost 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_magnet_cost_is_three_l269_26916


namespace NUMINAMATH_CALUDE_solution_correctness_l269_26971

noncomputable def solution_set : Set ℂ :=
  {0, 15, (1 + Complex.I * Real.sqrt 7) / 2, (1 - Complex.I * Real.sqrt 7) / 2}

def original_equation (x : ℂ) : Prop :=
  (15 * x - x^2) / (x + 1) * (x + (15 - x) / (x + 1)) = 30

theorem solution_correctness :
  ∀ x : ℂ, x ∈ solution_set ↔ original_equation x :=
sorry

end NUMINAMATH_CALUDE_solution_correctness_l269_26971


namespace NUMINAMATH_CALUDE_chocolate_bars_original_count_l269_26952

/-- The number of chocolate bars remaining after eating a certain percentage each day for a given number of days -/
def remaining_bars (initial : ℕ) (eat_percentage : ℚ) (days : ℕ) : ℚ :=
  initial * (1 - eat_percentage) ^ days

/-- The theorem stating the original number of chocolate bars given the remaining bars after 4 days -/
theorem chocolate_bars_original_count :
  ∃ (initial : ℕ),
    remaining_bars initial (30 / 100) 4 = 16 ∧
    initial = 67 :=
sorry

end NUMINAMATH_CALUDE_chocolate_bars_original_count_l269_26952


namespace NUMINAMATH_CALUDE_final_temperature_of_mixed_gases_l269_26972

/-- The final temperature of mixed gases in thermally insulated vessels -/
theorem final_temperature_of_mixed_gases
  (V₁ V₂ : ℝ) (p₁ p₂ : ℝ) (T₁ T₂ : ℝ) (R : ℝ) :
  V₁ = 1 →
  V₂ = 2 →
  p₁ = 2 →
  p₂ = 3 →
  T₁ = 300 →
  T₂ = 400 →
  R > 0 →
  let n₁ := p₁ * V₁ / (R * T₁)
  let n₂ := p₂ * V₂ / (R * T₂)
  let T := (n₁ * T₁ + n₂ * T₂) / (n₁ + n₂)
  ∃ ε > 0, |T - 369| < ε :=
sorry

end NUMINAMATH_CALUDE_final_temperature_of_mixed_gases_l269_26972


namespace NUMINAMATH_CALUDE_function_minimum_implies_parameter_range_l269_26978

/-- Given a function f(x) with parameter a > 0, if its minimum value is ln²(a) + 3ln(a) + 2,
    then a ≥ e^(-3/2) -/
theorem function_minimum_implies_parameter_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = a^2 * Real.exp (-2*x) + a * (2*x + 1) * Real.exp (-x) + x^2 + x) ∧
    (∀ x, f x ≥ Real.log a ^ 2 + 3 * Real.log a + 2) ∧
    (∃ x₀, f x₀ = Real.log a ^ 2 + 3 * Real.log a + 2)) →
  a ≥ Real.exp (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_parameter_range_l269_26978


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l269_26967

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l269_26967


namespace NUMINAMATH_CALUDE_limit_one_minus_x_squared_over_sin_pi_x_l269_26903

/-- The limit of (1 - x^2) / sin(πx) as x approaches 1 is 2/π -/
theorem limit_one_minus_x_squared_over_sin_pi_x (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(1 - x^2) / Real.sin (π * x) - 2/π| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_one_minus_x_squared_over_sin_pi_x_l269_26903


namespace NUMINAMATH_CALUDE_power_2_2013_mod_11_l269_26984

theorem power_2_2013_mod_11 : 2^2013 % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_2_2013_mod_11_l269_26984


namespace NUMINAMATH_CALUDE_bisection_method_sign_l269_26989

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval (a, b)
variable (a b : ℝ)

-- Define the sequence of intervals
variable (seq : ℕ → ℝ × ℝ)

-- State the theorem
theorem bisection_method_sign (hcont : Continuous f) 
  (hunique : ∃! x, x ∈ Ioo a b ∧ f x = 0)
  (hseq : ∀ k, Ioo (seq k).1 (seq k).2 ⊆ Ioo (seq (k+1)).1 (seq (k+1)).2)
  (hzero : ∀ k, ∃ x, x ∈ Ioo (seq k).1 (seq k).2 ∧ f x = 0)
  (hinit : seq 0 = (a, b))
  (hsign : f a < 0 ∧ f b > 0) :
  ∀ k, f (seq k).1 < 0 :=
sorry

end NUMINAMATH_CALUDE_bisection_method_sign_l269_26989


namespace NUMINAMATH_CALUDE_xy_square_value_l269_26913

theorem xy_square_value (x y : ℝ) 
  (h1 : x * (x + y) = 22)
  (h2 : y * (x + y) = 78 - y) : 
  (x + y)^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_xy_square_value_l269_26913


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l269_26965

-- Define the given parameters
def train_length : ℝ := 140
def train_speed_kmh : ℝ := 45
def crossing_time : ℝ := 30

-- Define the theorem
theorem bridge_length_calculation :
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time
  let bridge_length : ℝ := total_distance - train_length
  bridge_length = 235 := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l269_26965


namespace NUMINAMATH_CALUDE_equality_from_conditions_l269_26902

theorem equality_from_conditions (a b x y : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_x : 0 < x) (positive_y : 0 < y)
  (sum_less_than_two : a + b + x + y < 2)
  (eq_one : a + b^2 = x + y^2)
  (eq_two : a^2 + b = x^2 + y) :
  a = x ∧ b = y := by sorry

end NUMINAMATH_CALUDE_equality_from_conditions_l269_26902


namespace NUMINAMATH_CALUDE_f_monotone_iff_f_greater_than_2x_l269_26991

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 + Real.log ((x / a) + 1)

-- Theorem 1: Monotonicity condition
theorem f_monotone_iff (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 0, Monotone (f a)) ↔ a ∈ Set.Iic (1 - Real.exp 1) ∪ Set.Ici 1 :=
sorry

-- Theorem 2: Inequality for specific a and x
theorem f_greater_than_2x (a : ℝ) (x : ℝ) (ha : a ∈ Set.Ioo 0 1) (hx : x > 0) :
  f a x > 2 * x :=
sorry

end NUMINAMATH_CALUDE_f_monotone_iff_f_greater_than_2x_l269_26991


namespace NUMINAMATH_CALUDE_fruit_salad_composition_l269_26929

theorem fruit_salad_composition (total : ℕ) (red_grapes : ℕ) (green_grapes : ℕ) (raspberries : ℕ) :
  total = 102 →
  red_grapes = 67 →
  raspberries = green_grapes - 5 →
  red_grapes = 3 * green_grapes + (red_grapes - 3 * green_grapes) →
  red_grapes - 3 * green_grapes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_composition_l269_26929


namespace NUMINAMATH_CALUDE_triangle_inradius_l269_26950

/-- The inradius of a triangle with side lengths 13, 84, and 85 is 6 -/
theorem triangle_inradius : ∀ (a b c r : ℝ),
  a = 13 ∧ b = 84 ∧ c = 85 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inradius_l269_26950


namespace NUMINAMATH_CALUDE_prob_no_increasing_pie_is_correct_l269_26953

/-- Represents the number of pies Alice has initially -/
def total_pies : ℕ := 6

/-- Represents the number of pies that increase in size -/
def increasing_pies : ℕ := 2

/-- Represents the number of pies that decrease in size -/
def decreasing_pies : ℕ := 4

/-- Represents the number of pies Alice gives to Mary -/
def pies_given : ℕ := 3

/-- Calculates the probability that one of the girls does not have a single size-increasing pie -/
def prob_no_increasing_pie : ℚ := 7/10

/-- Theorem stating that the probability of one girl having no increasing pie is 0.7 -/
theorem prob_no_increasing_pie_is_correct : 
  prob_no_increasing_pie = 7/10 :=
sorry

end NUMINAMATH_CALUDE_prob_no_increasing_pie_is_correct_l269_26953


namespace NUMINAMATH_CALUDE_triathlete_swimming_speed_l269_26904

/-- Calculates the swimming speed of a triathlete given the conditions of the problem -/
theorem triathlete_swimming_speed
  (distance : ℝ)
  (running_speed : ℝ)
  (average_rate : ℝ)
  (h1 : distance = 2)
  (h2 : running_speed = 10)
  (h3 : average_rate = 0.1111111111111111)
  : ∃ (swimming_speed : ℝ), swimming_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_triathlete_swimming_speed_l269_26904


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l269_26974

/-- The value of p for which the axis of the parabola y^2 = 2px intersects
    the circle (x+1)^2 + y^2 = 4 at two points with distance 2√3 -/
theorem parabola_circle_intersection (p : ℝ) : p > 0 →
  (∃ A B : ℝ × ℝ,
    (A.1 + 1)^2 + A.2^2 = 4 ∧
    (B.1 + 1)^2 + B.2^2 = 4 ∧
    A.2^2 = 2 * p * A.1 ∧
    B.2^2 = 2 * p * B.1 ∧
    A.1 = B.1 ∧
    (A.2 - B.2)^2 = 12) →
  p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l269_26974


namespace NUMINAMATH_CALUDE_money_sharing_l269_26969

theorem money_sharing (john jose binoy total : ℕ) : 
  john + jose + binoy = total →
  2 * jose = 4 * john →
  3 * binoy = 6 * john →
  john = 1440 →
  total = 8640 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l269_26969


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l269_26935

theorem sqrt_product_plus_one : 
  Real.sqrt (31 * 30 * 29 * 28 + 1) = 869 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l269_26935


namespace NUMINAMATH_CALUDE_discount_percentage_l269_26961

theorem discount_percentage (num_tickets : ℕ) (price_per_ticket : ℚ) (total_spent : ℚ) : 
  num_tickets = 24 →
  price_per_ticket = 7 →
  total_spent = 84 →
  (1 - total_spent / (num_tickets * price_per_ticket)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l269_26961


namespace NUMINAMATH_CALUDE_swimmers_return_simultaneously_l269_26990

/-- Represents a swimmer in the river scenario -/
structure Swimmer where
  speed : ℝ  -- Speed relative to water
  direction : Int  -- 1 for downstream, -1 for upstream

/-- Represents the river scenario -/
structure RiverScenario where
  current_speed : ℝ
  swim_time : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer

/-- Calculates the time taken for a swimmer to return to the raft -/
def return_time (scenario : RiverScenario) (swimmer : Swimmer) : ℝ :=
  2 * scenario.swim_time

theorem swimmers_return_simultaneously (scenario : RiverScenario) :
  return_time scenario scenario.swimmer1 = return_time scenario scenario.swimmer2 :=
sorry

end NUMINAMATH_CALUDE_swimmers_return_simultaneously_l269_26990


namespace NUMINAMATH_CALUDE_seulgi_kicks_to_win_l269_26997

theorem seulgi_kicks_to_win (hohyeon_first hohyeon_second hyunjeong_first hyunjeong_second seulgi_first : ℕ) 
  (h1 : hohyeon_first = 23)
  (h2 : hohyeon_second = 28)
  (h3 : hyunjeong_first = 32)
  (h4 : hyunjeong_second = 17)
  (h5 : seulgi_first = 27) :
  ∃ seulgi_second : ℕ, 
    seulgi_second ≥ 25 ∧ 
    seulgi_first + seulgi_second > hohyeon_first + hohyeon_second ∧ 
    seulgi_first + seulgi_second > hyunjeong_first + hyunjeong_second :=
by sorry

end NUMINAMATH_CALUDE_seulgi_kicks_to_win_l269_26997


namespace NUMINAMATH_CALUDE_box_negative_two_zero_negative_one_l269_26928

-- Define the box operation
def box (a b c : ℤ) : ℚ :=
  (a ^ b : ℚ) - if b = 0 ∧ c < 0 then 0 else (b ^ c : ℚ) + (c ^ a : ℚ)

-- State the theorem
theorem box_negative_two_zero_negative_one :
  box (-2) 0 (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_box_negative_two_zero_negative_one_l269_26928


namespace NUMINAMATH_CALUDE_algebraic_expression_range_l269_26921

theorem algebraic_expression_range (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_range_l269_26921


namespace NUMINAMATH_CALUDE_opposite_zero_l269_26945

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines an opposite -/
axiom opposite_def (x : ℝ) : x + opposite x = 0

/-- Theorem: The opposite of 0 is 0 -/
theorem opposite_zero : opposite 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_zero_l269_26945


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l269_26923

/-- A rectangle with length thrice its breadth and area 675 square meters has a perimeter of 120 meters. -/
theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let area := l * b
  area = 675 →
  2 * (l + b) = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l269_26923


namespace NUMINAMATH_CALUDE_S_is_two_rays_with_common_endpoint_l269_26932

/-- The set S of points (x, y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≥ 5) ∨
               (5 = y - 2 ∧ x + 3 ≥ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≥ x + 3)}

/-- Two rays with a common endpoint -/
def TwoRaysWithCommonEndpoint : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x = 2 ∧ y ≥ 7) ∨
               (y = 7 ∧ x ≥ 2)}

/-- Theorem stating that S is equivalent to two rays with a common endpoint -/
theorem S_is_two_rays_with_common_endpoint : S = TwoRaysWithCommonEndpoint := by
  sorry

end NUMINAMATH_CALUDE_S_is_two_rays_with_common_endpoint_l269_26932


namespace NUMINAMATH_CALUDE_cara_don_meeting_l269_26907

/-- Cara and Don walk towards each other's houses. -/
theorem cara_don_meeting 
  (distance_between_homes : ℝ) 
  (cara_speed : ℝ) 
  (don_speed : ℝ) 
  (don_start_delay : ℝ) 
  (h1 : distance_between_homes = 45) 
  (h2 : cara_speed = 6) 
  (h3 : don_speed = 5) 
  (h4 : don_start_delay = 2) : 
  ∃ x : ℝ, x = 30 ∧ 
  x + don_speed * (x / cara_speed - don_start_delay) = distance_between_homes :=
by
  sorry


end NUMINAMATH_CALUDE_cara_don_meeting_l269_26907


namespace NUMINAMATH_CALUDE_investment_rate_problem_l269_26964

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (r : ℝ) : 
  simple_interest 900 0.045 7 = simple_interest 900 (r / 100) 7 + 31.50 →
  r = 4 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l269_26964


namespace NUMINAMATH_CALUDE_cube_edge_probability_cube_edge_probability_proof_l269_26940

/-- The probability of randomly selecting two vertices that form an edge in a cube -/
theorem cube_edge_probability : ℚ :=
let num_vertices : ℕ := 8
let num_edges : ℕ := 12
let total_pairs : ℕ := num_vertices.choose 2
3 / 7

/-- Proof that the probability of randomly selecting two vertices that form an edge in a cube is 3/7 -/
theorem cube_edge_probability_proof :
  cube_edge_probability = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_probability_cube_edge_probability_proof_l269_26940


namespace NUMINAMATH_CALUDE_fishes_from_superior_is_44_l269_26966

/-- The number of fishes taken from Lake Superior -/
def fishes_from_superior (total : ℕ) (ontario_erie : ℕ) (huron_michigan : ℕ) : ℕ :=
  total - ontario_erie - huron_michigan

/-- Theorem: Given the conditions from the problem, prove that the number of fishes
    taken from Lake Superior is 44 -/
theorem fishes_from_superior_is_44 :
  fishes_from_superior 97 23 30 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fishes_from_superior_is_44_l269_26966


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l269_26906

theorem diophantine_equation_solutions :
  let S : Set (ℕ × ℕ × ℕ × ℕ) := {(x, y, z, w) | 2^x * 3^y - 5^z * 7^w = 1}
  S = {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l269_26906


namespace NUMINAMATH_CALUDE_functional_equation_zero_solution_l269_26986

theorem functional_equation_zero_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ t : ℝ, f t = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_solution_l269_26986


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_four_l269_26920

theorem no_real_sqrt_negative_four :
  ¬ ∃ (x : ℝ), x^2 = -4 := by
sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_four_l269_26920


namespace NUMINAMATH_CALUDE_min_value_floor_sum_l269_26987

theorem min_value_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (m : ℕ), m = 4 ∧
  (∀ (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0),
    ⌊(2*a + b) / c⌋ + ⌊(2*b + c) / a⌋ + ⌊(2*c + a) / b⌋ + ⌊(a + b + c) / (a + b)⌋ ≥ m) ∧
  (⌊(2*x + y) / z⌋ + ⌊(2*y + z) / x⌋ + ⌊(2*z + x) / y⌋ + ⌊(x + y + z) / (x + y)⌋ = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_floor_sum_l269_26987


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l269_26924

/-- A type representing points on a line -/
structure Point where
  x : ℝ

/-- The distance between two points on a line -/
def distance (p q : Point) : ℝ := |p.x - q.x|

/-- The sum of distances from a point to a list of points -/
def sum_distances (q : Point) (points : List Point) : ℝ :=
  points.foldl (fun sum p => sum + distance p q) 0

theorem minimize_sum_distances 
  (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ : Point)
  (h : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x) :
  ∃ (q : Point), 
    (∀ (r : Point), sum_distances q [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈] ≤ sum_distances r [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈]) ∧
    q.x = (p₄.x + p₅.x) / 2 := by
  sorry


end NUMINAMATH_CALUDE_minimize_sum_distances_l269_26924


namespace NUMINAMATH_CALUDE_special_function_characterization_l269_26955

/-- A function satisfying the given properties -/
def IsSpecialFunction (f : ℝ → ℝ) : Prop :=
  f 1 = 0 ∧ ∀ x y : ℝ, |f x - f y| = |x - y|

/-- The theorem stating that any function satisfying the given properties
    must be either x - 1 or 1 - x -/
theorem special_function_characterization (f : ℝ → ℝ) (hf : IsSpecialFunction f) :
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_characterization_l269_26955


namespace NUMINAMATH_CALUDE_field_width_calculation_l269_26910

/-- A rectangular football field with given dimensions and running conditions. -/
structure FootballField where
  length : ℝ
  width : ℝ
  laps : ℕ
  total_distance : ℝ

/-- The width of a football field given specific conditions. -/
def field_width (f : FootballField) : ℝ :=
  f.width

/-- Theorem stating the width of the field under given conditions. -/
theorem field_width_calculation (f : FootballField)
  (h1 : f.length = 100)
  (h2 : f.laps = 6)
  (h3 : f.total_distance = 1800)
  (h4 : f.total_distance = f.laps * (2 * f.length + 2 * f.width)) :
  field_width f = 50 := by
  sorry

#check field_width_calculation

end NUMINAMATH_CALUDE_field_width_calculation_l269_26910


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l269_26934

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = 2 ∧ x₁^2 - 8*x₁ + 12 = 0 ∧ x₂^2 - 8*x₂ + 12 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3 ∧ y₂ = -3 ∧ (y₁ - 3)^2 = 2*y₁*(y₁ - 3) ∧ (y₂ - 3)^2 = 2*y₂*(y₂ - 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l269_26934


namespace NUMINAMATH_CALUDE_simplify_and_square_l269_26958

theorem simplify_and_square : (8 * (15 / 9) * (-45 / 50))^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_square_l269_26958


namespace NUMINAMATH_CALUDE_laundry_time_ratio_l269_26927

/-- Proves that the ratio of time to wash towels to time to wash clothes is 2:1 --/
theorem laundry_time_ratio :
  ∀ (towel_time sheet_time clothes_time : ℕ),
    clothes_time = 30 →
    sheet_time = towel_time - 15 →
    towel_time + sheet_time + clothes_time = 135 →
    towel_time / clothes_time = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_laundry_time_ratio_l269_26927


namespace NUMINAMATH_CALUDE_even_function_interval_l269_26925

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the interval
def interval (m : ℝ) : Set ℝ := Set.Icc (2*m) (m+6)

-- State the theorem
theorem even_function_interval (m : ℝ) :
  (∀ x ∈ interval m, f x = f (-x)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_interval_l269_26925


namespace NUMINAMATH_CALUDE_two_b_values_for_two_integer_solutions_l269_26975

theorem two_b_values_for_two_integer_solutions : 
  ∃! (s : Finset ℤ), 
    (∀ b ∈ s, ∃! (t : Finset ℤ), (∀ x ∈ t, x^2 + b*x + 5 ≤ 0) ∧ t.card = 2) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_b_values_for_two_integer_solutions_l269_26975


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_six_squared_l269_26994

theorem gcd_factorial_eight_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_six_squared_l269_26994


namespace NUMINAMATH_CALUDE_power_function_increasing_l269_26901

/-- A function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing (m : ℝ) :
  (∀ x > 0, Monotone (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 :=
sorry

end NUMINAMATH_CALUDE_power_function_increasing_l269_26901


namespace NUMINAMATH_CALUDE_debate_club_girls_l269_26983

theorem debate_club_girls (total_members : ℕ) (present_members : ℕ) 
  (h_total : total_members = 22)
  (h_present : present_members = 14)
  (h_attendance : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧
    boys + (girls / 3) = present_members) :
  ∃ (girls : ℕ), girls = 12 ∧ 
    ∃ (boys : ℕ), boys + girls = total_members ∧
      boys + (girls / 3) = present_members := by
sorry

end NUMINAMATH_CALUDE_debate_club_girls_l269_26983


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l269_26970

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l269_26970


namespace NUMINAMATH_CALUDE_unique_number_l269_26963

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13*k

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  is_odd n ∧ 
  is_multiple_of_13 n ∧ 
  is_perfect_square (digit_product n) ∧
  n = 91 := by sorry

end NUMINAMATH_CALUDE_unique_number_l269_26963


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l269_26949

/-- A random variable following a normal distribution with mean 2 and variance 4 -/
def X : Real → Real := sorry

/-- The probability density function of X -/
def pdf_X : Real → Real := sorry

/-- The cumulative distribution function of X -/
def cdf_X : Real → Real := sorry

/-- The value of 'a' such that P(X < a) = 0.2 -/
def a : Real := sorry

/-- Theorem stating that if P(X < a) = 0.2, then P(X < 4-a) = 0.2 -/
theorem normal_distribution_symmetry :
  (cdf_X a = 0.2) → (cdf_X (4 - a) = 0.2) := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l269_26949


namespace NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l269_26985

theorem diophantine_equation_unique_solution :
  ∀ a b c : ℤ, 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l269_26985


namespace NUMINAMATH_CALUDE_erica_ride_duration_l269_26942

/-- The duration in minutes that Dave can ride the merry-go-round -/
def dave_duration : ℝ := 10

/-- The factor by which Chuck can ride longer than Dave -/
def chuck_factor : ℝ := 5

/-- The percentage longer that Erica can ride compared to Chuck -/
def erica_percentage : ℝ := 0.30

/-- The duration in minutes that Chuck can ride the merry-go-round -/
def chuck_duration : ℝ := dave_duration * chuck_factor

/-- The duration in minutes that Erica can ride the merry-go-round -/
def erica_duration : ℝ := chuck_duration * (1 + erica_percentage)

/-- Theorem stating that Erica can ride for 65 minutes -/
theorem erica_ride_duration : erica_duration = 65 := by sorry

end NUMINAMATH_CALUDE_erica_ride_duration_l269_26942


namespace NUMINAMATH_CALUDE_largest_number_in_set_l269_26926

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  -2 * a = max (-2 * a) (max (5 * a) (max (36 / a) (max (a ^ 3) 2))) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l269_26926


namespace NUMINAMATH_CALUDE_expression_equality_l269_26982

theorem expression_equality : 
  Real.sqrt 4 + |Real.sqrt 3 - 3| + 2 * Real.sin (π / 6) - (π - 2023)^0 = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l269_26982


namespace NUMINAMATH_CALUDE_total_footprints_pogo_and_grimzi_footprints_l269_26912

/-- Calculates the total number of footprints left by two creatures on their respective planets -/
theorem total_footprints (pogo_footprints_per_meter : ℕ) 
                         (grimzi_footprints_per_six_meters : ℕ) 
                         (distance : ℕ) : ℕ :=
  let pogo_total := pogo_footprints_per_meter * distance
  let grimzi_total := grimzi_footprints_per_six_meters * (distance / 6)
  pogo_total + grimzi_total

/-- Proves that the combined total number of footprints left by Pogo and Grimzi is 27,000 -/
theorem pogo_and_grimzi_footprints : 
  total_footprints 4 3 6000 = 27000 := by
  sorry

end NUMINAMATH_CALUDE_total_footprints_pogo_and_grimzi_footprints_l269_26912


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l269_26951

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 25 → area = 375 → area = (d1 * d2) / 2 → d2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l269_26951


namespace NUMINAMATH_CALUDE_union_of_sets_l269_26973

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l269_26973


namespace NUMINAMATH_CALUDE_square_difference_formula_l269_26976

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8/15)
  (h2 : x - y = 2/15) : 
  x^2 - y^2 = 16/225 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l269_26976


namespace NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l269_26914

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

theorem folded_paper_perimeter_ratio :
  let original := Rectangle.mk 6 8
  let folded := Rectangle.mk original.width (original.height / 2)
  let small := Rectangle.mk (folded.width / 2) folded.height
  let large := Rectangle.mk folded.width folded.height
  perimeter small / perimeter large = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l269_26914


namespace NUMINAMATH_CALUDE_billy_is_45_l269_26947

/-- Billy's age -/
def B : ℕ := sorry

/-- Joe's age -/
def J : ℕ := sorry

/-- Billy's age is three times Joe's age -/
axiom billy_age : B = 3 * J

/-- The sum of their ages is 60 -/
axiom total_age : B + J = 60

/-- Prove that Billy is 45 years old -/
theorem billy_is_45 : B = 45 := by sorry

end NUMINAMATH_CALUDE_billy_is_45_l269_26947


namespace NUMINAMATH_CALUDE_max_utilization_rate_square_plate_l269_26946

/-- Given a square steel plate with side length 4 and a rusted corner defined by AF = 2 and BF = 1,
    prove that the maximum utilization rate is 50%. -/
theorem max_utilization_rate_square_plate (side_length : ℝ) (af bf : ℝ) :
  side_length = 4 ∧ af = 2 ∧ bf = 1 →
  ∃ (rect_area : ℝ),
    rect_area ≤ side_length * side_length ∧
    rect_area = side_length * (side_length - af) ∧
    (rect_area / (side_length * side_length)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_utilization_rate_square_plate_l269_26946


namespace NUMINAMATH_CALUDE_triangle_properties_l269_26948

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) :
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = π →
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c →
    t.B = π / 3) ∧
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c ∧
   t.A = 5 * π / 12 ∧ t.b = 2 →
    t.c = 2 * Real.sqrt 6 / 3) ∧
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c ∧
   t.a + t.c = 4 →
    ∀ x : ℝ, x > 0 → t.b ≤ x → 2 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l269_26948


namespace NUMINAMATH_CALUDE_power_zero_of_three_minus_pi_l269_26917

theorem power_zero_of_three_minus_pi : (3 - Real.pi) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_of_three_minus_pi_l269_26917


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l269_26996

/-- Prove that for an ellipse with the given conditions, its eccentricity is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let P := (-c, b^2 / a)
  let A := (a, 0)
  let B := (0, b)
  let O := (0, 0)
  ellipse (-c) (b^2 / a) ∧ 
  (B.2 - A.2) / (B.1 - A.1) = (P.2 - O.2) / (P.1 - O.1) →
  c / a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l269_26996


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l269_26939

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ, b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ x : ℕ, x > 0 → 6 ∣ x → 15 ∣ x → b ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l269_26939


namespace NUMINAMATH_CALUDE_packaging_methods_different_boxes_l269_26992

theorem packaging_methods_different_boxes (n : ℕ) (m : ℕ) :
  n > 0 → m > 0 → (number_of_packaging_methods : ℕ) = m^n :=
by sorry

end NUMINAMATH_CALUDE_packaging_methods_different_boxes_l269_26992


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_in_range_l269_26930

theorem no_square_divisible_by_six_in_range : ¬∃ y : ℕ, 
  (∃ n : ℕ, y = n^2) ∧ 
  (y % 6 = 0) ∧ 
  (50 ≤ y) ∧ 
  (y ≤ 120) := by
  sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_in_range_l269_26930


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l269_26911

def N : ℕ := sorry  -- Definition of N as concatenation of integers from 34 to 76

theorem highest_power_of_three_dividing_N :
  ∃ k : ℕ, (3^k ∣ N) ∧ ¬(3^(k+1) ∣ N) ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l269_26911


namespace NUMINAMATH_CALUDE_price_per_bracelet_l269_26995

/-- Represents the problem of determining the price per bracelet --/
def bracelet_problem (total_cost : ℕ) (selling_period_weeks : ℕ) (avg_daily_sales : ℕ) : Prop :=
  let total_days : ℕ := selling_period_weeks * 7
  let total_bracelets : ℕ := total_days * avg_daily_sales
  total_bracelets = total_cost ∧ (total_cost : ℚ) / total_bracelets = 1

/-- Proves that the price per bracelet is $1 given the problem conditions --/
theorem price_per_bracelet :
  bracelet_problem 112 2 8 :=
by sorry

end NUMINAMATH_CALUDE_price_per_bracelet_l269_26995


namespace NUMINAMATH_CALUDE_problem_solution_l269_26909

theorem problem_solution (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 6) :
  (a^2 + b^2 = 37) ∧ (a^3*b - 2*a^2*b^2 + a*b^3 = 150) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l269_26909


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l269_26936

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l269_26936


namespace NUMINAMATH_CALUDE_inequality_system_solution_l269_26960

theorem inequality_system_solution (x a : ℝ) : 
  (1 - x < -1) ∧ (x - 1 > a) ∧ (∀ y, (1 - y < -1 ∧ y - 1 > a) ↔ y > 2) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l269_26960


namespace NUMINAMATH_CALUDE_vector_equations_true_l269_26922

-- Define a vector space over the real numbers
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define points A, B, and C
variable (A B C : V)

-- Theorem statement
theorem vector_equations_true :
  (a + b = b + a) ∧
  (-(-a) = a) ∧
  ((B - A) + (C - B) + (A - C) = 0) ∧
  (a + (-a) = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_equations_true_l269_26922


namespace NUMINAMATH_CALUDE_max_table_sum_l269_26999

def numbers : List ℕ := [2, 3, 5, 7, 11, 13]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset = numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem max_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
    table_sum top left ≤ 420 :=
  sorry

end NUMINAMATH_CALUDE_max_table_sum_l269_26999


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l269_26908

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (1 + 2 * Complex.I) * (2 - Complex.I)
  Complex.im z = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l269_26908


namespace NUMINAMATH_CALUDE_find_a_value_l269_26933

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the specific function for x < 0
def SpecificFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x < 0 → f x = x^2 + a*x

theorem find_a_value (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : OddFunction f)
  (h_spec : SpecificFunction f a)
  (h_f3 : f 3 = 6) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l269_26933


namespace NUMINAMATH_CALUDE_complex_square_sum_l269_26944

theorem complex_square_sum (a b : ℝ) : 
  (Complex.mk a b = (1 + Complex.I)^2) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l269_26944


namespace NUMINAMATH_CALUDE_juans_number_problem_l269_26957

theorem juans_number_problem (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 = 10) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_problem_l269_26957


namespace NUMINAMATH_CALUDE_parabola_translation_l269_26905

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-4) 0 0
  let translated := translate original 2 3
  y = -4 * x^2 → y = translated.a * (x + 2)^2 + translated.b * (x + 2) + translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l269_26905


namespace NUMINAMATH_CALUDE_taxi_fare_for_80_miles_l269_26988

/-- Represents the fare structure of a taxi company -/
structure TaxiFare where
  fixedFare : ℝ
  costPerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.fixedFare + tf.costPerMile * distance

theorem taxi_fare_for_80_miles :
  ∃ (tf : TaxiFare),
    tf.fixedFare = 15 ∧
    totalFare tf 60 = 135 ∧
    totalFare tf 80 = 175 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_for_80_miles_l269_26988


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l269_26931

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialRandomVariable (n : ℕ) (p : ℝ) where
  X : ℝ

/-- The variance of a binomial random variable -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- The variance of a linear transformation of a random variable -/
def linearTransformVariance (a : ℝ) (X : ℝ) : ℝ := a^2 * X

theorem variance_of_transformed_binomial :
  let n : ℕ := 10
  let p : ℝ := 0.8
  let X : BinomialRandomVariable n p := ⟨0⟩  -- The actual value doesn't matter for this theorem
  let var_X : ℝ := binomialVariance n p
  let var_2X_plus_1 : ℝ := linearTransformVariance 2 var_X
  var_2X_plus_1 = 6.4 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l269_26931


namespace NUMINAMATH_CALUDE_sector_angle_l269_26938

/-- 
Given a circular sector where:
- r is the radius of the sector
- α is the central angle of the sector in radians
- l is the arc length of the sector
- C is the circumference of the sector

Prove that if C = 4r, then α = 2.
-/
theorem sector_angle (r : ℝ) (α : ℝ) (l : ℝ) (C : ℝ) 
  (h1 : C = 4 * r)  -- Circumference is four times the radius
  (h2 : C = 2 * r + l)  -- Circumference formula for a sector
  (h3 : l = α * r)  -- Arc length formula
  : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l269_26938


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l269_26915

/-- Sarah's bowling score problem -/
theorem sarahs_bowling_score :
  ∀ (sarah greg : ℕ),
  sarah = greg + 60 →
  sarah + greg = 260 →
  sarah = 160 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l269_26915


namespace NUMINAMATH_CALUDE_fabric_problem_solution_l269_26959

/-- Represents the fabric and flag problem -/
structure FabricProblem where
  initial_fabric : Float
  square_side : Float
  square_count : Nat
  wide_rect_length : Float
  wide_rect_width : Float
  wide_rect_count : Nat
  tall_rect_length : Float
  tall_rect_width : Float
  tall_rect_count : Nat
  triangle_base : Float
  triangle_height : Float
  triangle_count : Nat
  hexagon_side : Float
  hexagon_apothem : Float
  hexagon_count : Nat

/-- Calculates the remaining fabric after making flags -/
def remaining_fabric (p : FabricProblem) : Float :=
  p.initial_fabric -
  (p.square_side * p.square_side * p.square_count.toFloat +
   p.wide_rect_length * p.wide_rect_width * p.wide_rect_count.toFloat +
   p.tall_rect_length * p.tall_rect_width * p.tall_rect_count.toFloat +
   (p.triangle_base * p.triangle_height / 2) * p.triangle_count.toFloat +
   (6 * p.hexagon_side * p.hexagon_apothem / 2) * p.hexagon_count.toFloat)

/-- The theorem stating the remaining fabric for the given problem -/
theorem fabric_problem_solution (p : FabricProblem) :
  p.initial_fabric = 1500 ∧
  p.square_side = 4 ∧
  p.square_count = 22 ∧
  p.wide_rect_length = 5 ∧
  p.wide_rect_width = 3 ∧
  p.wide_rect_count = 28 ∧
  p.tall_rect_length = 3 ∧
  p.tall_rect_width = 5 ∧
  p.tall_rect_count = 14 ∧
  p.triangle_base = 6 ∧
  p.triangle_height = 4 ∧
  p.triangle_count = 18 ∧
  p.hexagon_side = 3 ∧
  p.hexagon_apothem = 2.6 ∧
  p.hexagon_count = 24 →
  remaining_fabric p = -259.6 := by
  sorry

end NUMINAMATH_CALUDE_fabric_problem_solution_l269_26959
