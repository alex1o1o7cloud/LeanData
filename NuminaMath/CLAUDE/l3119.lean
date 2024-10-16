import Mathlib

namespace NUMINAMATH_CALUDE_audit_options_second_week_l3119_311946

def remaining_OR : ℕ := 13 - 2
def remaining_GTU : ℕ := 15 - 3

theorem audit_options_second_week : 
  (remaining_OR.choose 2) * (remaining_GTU.choose 3) = 12100 := by
  sorry

end NUMINAMATH_CALUDE_audit_options_second_week_l3119_311946


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3119_311994

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 ∧ b = 15 ∧ c^2 = a^2 + b^2 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3119_311994


namespace NUMINAMATH_CALUDE_smallest_cube_volume_for_pyramid_l3119_311956

/-- Represents the dimensions of a rectangular pyramid -/
structure PyramidDimensions where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ := side^3

/-- Theorem: The volume of the smallest cube-shaped box that can house a given rectangular pyramid uprightly -/
theorem smallest_cube_volume_for_pyramid (p : PyramidDimensions) 
  (h_height : p.height = 15)
  (h_length : p.baseLength = 8)
  (h_width : p.baseWidth = 12) :
  cubeVolume (max p.height (max p.baseLength p.baseWidth)) = 3375 := by
  sorry

#check smallest_cube_volume_for_pyramid

end NUMINAMATH_CALUDE_smallest_cube_volume_for_pyramid_l3119_311956


namespace NUMINAMATH_CALUDE_bryans_mineral_samples_l3119_311945

theorem bryans_mineral_samples (samples_per_shelf : ℕ) (total_shelves : ℕ) 
  (h1 : samples_per_shelf = 128) 
  (h2 : total_shelves = 13) : 
  samples_per_shelf * total_shelves = 1664 := by
sorry

end NUMINAMATH_CALUDE_bryans_mineral_samples_l3119_311945


namespace NUMINAMATH_CALUDE_find_P_l3119_311973

theorem find_P : ∃ P : ℕ+, (15 ^ 3 * 25 ^ 3 : ℕ) = 5 ^ 2 * P ^ 3 ∧ P = 375 := by
  sorry

end NUMINAMATH_CALUDE_find_P_l3119_311973


namespace NUMINAMATH_CALUDE_correct_calculation_l3119_311980

theorem correct_calculation (x : ℝ) (h : x * 3 - 45 = 159) : (x + 32) * 12 = 1200 := by
  sorry

#check correct_calculation

end NUMINAMATH_CALUDE_correct_calculation_l3119_311980


namespace NUMINAMATH_CALUDE_pool_wall_area_ratio_l3119_311958

theorem pool_wall_area_ratio :
  let pool_radius : ℝ := 20
  let wall_width : ℝ := 4
  let pool_area := π * pool_radius^2
  let total_area := π * (pool_radius + wall_width)^2
  let wall_area := total_area - pool_area
  wall_area / pool_area = 11 / 25 := by sorry

end NUMINAMATH_CALUDE_pool_wall_area_ratio_l3119_311958


namespace NUMINAMATH_CALUDE_unique_function_existence_l3119_311934

/-- Given positive real numbers a and b, and X being the set of non-negative real numbers,
    there exists a unique function f: X → X such that f(f(x)) = b(a + b)x - af(x) for all x ∈ X,
    and this function is f(x) = bx. -/
theorem unique_function_existence (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃! f : {x : ℝ | 0 ≤ x} → {x : ℝ | 0 ≤ x},
    (∀ x, f (f x) = b * (a + b) * x - a * f x) ∧
    (∀ x, f x = b * x) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l3119_311934


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3119_311959

/-- Given a geometric sequence {a_n} with first term a₁ = 1 and positive common ratio q,
    if S₄ - 5S₂ = 0, then S₅ = 31. -/
theorem geometric_sequence_sum (q : ℝ) (hq : q > 0) : 
  let a : ℕ → ℝ := λ n => q^(n-1)
  let S : ℕ → ℝ := λ n => (1 - q^n) / (1 - q)
  (S 4 - 5 * S 2 = 0) → S 5 = 31 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3119_311959


namespace NUMINAMATH_CALUDE_similar_triangles_leg_ratio_l3119_311966

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has
    corresponding legs y and 7, prove that y = 84/9 -/
theorem similar_triangles_leg_ratio (y : ℝ) : 
  (12 : ℝ) / y = 9 / 7 → y = 84 / 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_ratio_l3119_311966


namespace NUMINAMATH_CALUDE_smallest_voltage_l3119_311904

theorem smallest_voltage (a b c : ℕ) : 
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧
  (a + b + c : ℚ) / 3 = 4 ∧
  (a + b + c) % 5 = 0 →
  min a (min b c) = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_voltage_l3119_311904


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3119_311925

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 (a > 0) that passes through (-2, 0),
    its eccentricity is √7/2 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  (4 / a^2 - 0 / 3 = 1) →
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3119_311925


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3119_311984

theorem degenerate_ellipse_max_y_coordinate :
  ∀ x y : ℝ, (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3119_311984


namespace NUMINAMATH_CALUDE_simplify_linear_expression_l3119_311910

theorem simplify_linear_expression (y : ℝ) : 2*y + 3*y + 4*y = 9*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_linear_expression_l3119_311910


namespace NUMINAMATH_CALUDE_jakes_and_sister_weight_l3119_311962

/-- The combined weight of Jake and his sister given Jake's current weight and the condition about their weight ratio after Jake loses weight. -/
theorem jakes_and_sister_weight (jake_weight : ℕ) (weight_loss : ℕ) : 
  jake_weight = 93 →
  weight_loss = 15 →
  (jake_weight - weight_loss) = 2 * ((jake_weight - weight_loss) / 2) →
  jake_weight + ((jake_weight - weight_loss) / 2) = 132 := by
sorry

end NUMINAMATH_CALUDE_jakes_and_sister_weight_l3119_311962


namespace NUMINAMATH_CALUDE_largest_number_less_than_threshold_l3119_311920

def given_numbers : List ℚ := [4, 9/10, 6/5, 1/2, 13/10]
def threshold : ℚ := 111/100

theorem largest_number_less_than_threshold :
  (given_numbers.filter (· < threshold)).maximum? = some (9/10) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_less_than_threshold_l3119_311920


namespace NUMINAMATH_CALUDE_prime_power_modulo_l3119_311955

theorem prime_power_modulo (m : ℕ) (f : ℤ → ℤ) : m > 1 →
  (∃ x : ℤ, f x % m = 0) →
  (∃ y : ℤ, f y % m = 1) →
  (∃ z : ℤ, f z % m ≠ 0 ∧ f z % m ≠ 1) →
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, m = p ^ k) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_modulo_l3119_311955


namespace NUMINAMATH_CALUDE_minimum_gloves_needed_l3119_311965

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 63 → gloves_per_participant = 2 → participants * gloves_per_participant = 126 := by
  sorry

end NUMINAMATH_CALUDE_minimum_gloves_needed_l3119_311965


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l3119_311987

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l3119_311987


namespace NUMINAMATH_CALUDE_system_solutions_l3119_311938

def is_solution (x y z w : ℝ) : Prop :=
  x^2 + 2*y^2 + 2*z^2 + w^2 = 43 ∧
  y^2 + z^2 + w^2 = 29 ∧
  5*z^2 - 3*w^2 + 4*x*y + 12*y*z + 6*z*x = 95

theorem system_solutions :
  {(x, y, z, w) : ℝ × ℝ × ℝ × ℝ | is_solution x y z w} =
  {(1, 2, 3, 4), (1, 2, 3, -4), (-1, -2, -3, 4), (-1, -2, -3, -4)} :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3119_311938


namespace NUMINAMATH_CALUDE_system_equations_solution_system_inequalities_solution_l3119_311906

-- Part 1: System of equations
theorem system_equations_solution (x y : ℝ) : 
  (x = 4*y + 1 ∧ 2*x - 5*y = 8) → (x = 9 ∧ y = 2) := by sorry

-- Part 2: System of inequalities
theorem system_inequalities_solution (x : ℝ) :
  (4*x - 5 ≤ 3 ∧ (x - 1) / 3 < (2*x + 1) / 5) ↔ (-8 < x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_system_equations_solution_system_inequalities_solution_l3119_311906


namespace NUMINAMATH_CALUDE_gcf_of_36_and_12_l3119_311937

theorem gcf_of_36_and_12 :
  let n : ℕ := 36
  let m : ℕ := 12
  let lcm_nm : ℕ := 54
  lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_12_l3119_311937


namespace NUMINAMATH_CALUDE_nested_root_simplification_l3119_311923

theorem nested_root_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x^9)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l3119_311923


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3119_311943

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3119_311943


namespace NUMINAMATH_CALUDE_integer_solution_squared_sum_eq_product_l3119_311951

theorem integer_solution_squared_sum_eq_product (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_squared_sum_eq_product_l3119_311951


namespace NUMINAMATH_CALUDE_train_length_l3119_311989

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 18 → ∃ (length_m : ℝ), 
    (length_m ≥ 300.05 ∧ length_m ≤ 300.07) ∧ 
    length_m = speed_kmh * (1000 / 3600) * time_s :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3119_311989


namespace NUMINAMATH_CALUDE_mars_500_duration_notation_l3119_311971

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation :=
  sorry

theorem mars_500_duration_notation (duration : ℝ) (h : duration = 12480) :
  (toScientificNotation duration).coefficient = 1.248 ∧
  (toScientificNotation duration).exponent = 4 ∧
  (roundToSignificantFigures (toScientificNotation duration) 3).coefficient = 1.25 :=
  sorry

end NUMINAMATH_CALUDE_mars_500_duration_notation_l3119_311971


namespace NUMINAMATH_CALUDE_only_postcard_win_is_systematic_l3119_311991

-- Define the type for sampling methods
inductive SamplingMethod
| EmployeeRep
| MarketResearch
| LotteryDraw
| PostcardWin
| ExamAnalysis

-- Define what constitutes systematic sampling
def is_systematic_sampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.PostcardWin => True
  | _ => False

-- Theorem stating that only PostcardWin is systematic sampling
theorem only_postcard_win_is_systematic :
  ∀ (method : SamplingMethod),
    is_systematic_sampling method ↔ method = SamplingMethod.PostcardWin :=
by sorry

#check only_postcard_win_is_systematic

end NUMINAMATH_CALUDE_only_postcard_win_is_systematic_l3119_311991


namespace NUMINAMATH_CALUDE_carrot_bundle_price_is_two_dollars_l3119_311902

/-- Represents the farmer's harvest and sales data -/
structure FarmerData where
  potatoes : ℕ
  carrots : ℕ
  potatoesPerBundle : ℕ
  carrotsPerBundle : ℕ
  potatoBundlePrice : ℚ
  totalRevenue : ℚ

/-- Calculates the price of each carrot bundle -/
def carrotBundlePrice (data : FarmerData) : ℚ :=
  let potatoBundles := data.potatoes / data.potatoesPerBundle
  let potatoRevenue := potatoBundles * data.potatoBundlePrice
  let carrotRevenue := data.totalRevenue - potatoRevenue
  let carrotBundles := data.carrots / data.carrotsPerBundle
  carrotRevenue / carrotBundles

/-- Theorem stating that the carrot bundle price is $2.00 -/
theorem carrot_bundle_price_is_two_dollars 
  (data : FarmerData) 
  (h1 : data.potatoes = 250)
  (h2 : data.carrots = 320)
  (h3 : data.potatoesPerBundle = 25)
  (h4 : data.carrotsPerBundle = 20)
  (h5 : data.potatoBundlePrice = 19/10)
  (h6 : data.totalRevenue = 51) :
  carrotBundlePrice data = 2 := by
  sorry

#eval carrotBundlePrice {
  potatoes := 250,
  carrots := 320,
  potatoesPerBundle := 25,
  carrotsPerBundle := 20,
  potatoBundlePrice := 19/10,
  totalRevenue := 51
}

end NUMINAMATH_CALUDE_carrot_bundle_price_is_two_dollars_l3119_311902


namespace NUMINAMATH_CALUDE_pizza_size_increase_l3119_311913

theorem pizza_size_increase (r : ℝ) (h : r > 0) :
  let R := r * (1 + 0.5)
  (π * R^2) / (π * r^2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_pizza_size_increase_l3119_311913


namespace NUMINAMATH_CALUDE_deborah_finishes_first_l3119_311944

def lawn_problem (z r : ℝ) : Prop :=
  let jonathan_area := z
  let deborah_area := z / 3
  let ezekiel_area := z / 4
  let jonathan_rate := r
  let deborah_rate := r / 4
  let ezekiel_rate := r / 6
  let jonathan_time := jonathan_area / jonathan_rate
  let deborah_time := deborah_area / deborah_rate
  let ezekiel_time := ezekiel_area / ezekiel_rate
  (deborah_time < jonathan_time) ∧ (deborah_time < ezekiel_time)

theorem deborah_finishes_first (z r : ℝ) (hz : z > 0) (hr : r > 0) :
  lawn_problem z r :=
by
  sorry

end NUMINAMATH_CALUDE_deborah_finishes_first_l3119_311944


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l3119_311926

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 7*I) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), Complex.abs (z' + 2 + 4*I) = 2 → Complex.abs (w' - 6 - 7*I) = 4 
      → Complex.abs (z' - w') ≥ min_dist) ∧ 
    (∃ (z₀ w₀ : ℂ), Complex.abs (z₀ + 2 + 4*I) = 2 ∧ Complex.abs (w₀ - 6 - 7*I) = 4 
      ∧ Complex.abs (z₀ - w₀) = min_dist) ∧
    min_dist = Real.sqrt 185 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l3119_311926


namespace NUMINAMATH_CALUDE_square_diagonal_length_l3119_311901

/-- The length of the diagonal of a square with area 72 and perimeter 33.94112549695428 is 12 -/
theorem square_diagonal_length (area : ℝ) (perimeter : ℝ) (h_area : area = 72) (h_perimeter : perimeter = 33.94112549695428) :
  let side := (perimeter / 4 : ℝ)
  Real.sqrt (2 * side ^ 2) = 12 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l3119_311901


namespace NUMINAMATH_CALUDE_multiples_equality_l3119_311974

theorem multiples_equality (n : ℕ) : 
  let a : ℚ := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b : ℚ := 2 * n
  (a ^ 2 = b ^ 2) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_equality_l3119_311974


namespace NUMINAMATH_CALUDE_strategy_D_lowest_price_l3119_311979

/-- Represents a pricing strategy with an increase followed by a decrease -/
structure PricingStrategy where
  increase : ℝ
  decrease : ℝ

/-- Calculates the final price factor for a given pricing strategy -/
def finalPriceFactor (strategy : PricingStrategy) : ℝ :=
  (1 + strategy.increase) * (1 - strategy.decrease)

/-- The four pricing strategies -/
def strategyA : PricingStrategy := ⟨0.1, 0.1⟩
def strategyB : PricingStrategy := ⟨-0.1, -0.1⟩
def strategyC : PricingStrategy := ⟨0.2, 0.2⟩
def strategyD : PricingStrategy := ⟨0.3, 0.3⟩

theorem strategy_D_lowest_price :
  finalPriceFactor strategyD ≤ finalPriceFactor strategyA ∧
  finalPriceFactor strategyD ≤ finalPriceFactor strategyB ∧
  finalPriceFactor strategyD ≤ finalPriceFactor strategyC :=
sorry

end NUMINAMATH_CALUDE_strategy_D_lowest_price_l3119_311979


namespace NUMINAMATH_CALUDE_thirteenth_number_l3119_311931

theorem thirteenth_number (results : Vector ℝ 25) 
  (h1 : results.toList.sum / 25 = 18)
  (h2 : (results.take 12).toList.sum / 12 = 14)
  (h3 : (results.drop 13).toList.sum / 12 = 17) :
  results[12] = 78 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_number_l3119_311931


namespace NUMINAMATH_CALUDE_average_age_of_three_students_l3119_311993

theorem average_age_of_three_students
  (total_students : ℕ)
  (total_average : ℝ)
  (eleven_students : ℕ)
  (eleven_average : ℝ)
  (fifteenth_student_age : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average = 15)
  (h3 : eleven_students = 11)
  (h4 : eleven_average = 16)
  (h5 : fifteenth_student_age = 7)
  : (total_students * total_average - eleven_students * eleven_average - fifteenth_student_age) / (total_students - eleven_students - 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_three_students_l3119_311993


namespace NUMINAMATH_CALUDE_equation_solution_l3119_311986

theorem equation_solution : ∃ x : ℝ, 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3119_311986


namespace NUMINAMATH_CALUDE_value_of_expression_l3119_311964

theorem value_of_expression (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  (a^3 + 3*b^2) / 9 = 25 / 3 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_l3119_311964


namespace NUMINAMATH_CALUDE_man_speed_against_current_l3119_311954

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specified speeds, 
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l3119_311954


namespace NUMINAMATH_CALUDE_net_income_for_specific_case_l3119_311997

/-- Calculates the net income after tax for a tax resident --/
def net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * (1 - tax_rate)

/-- Theorem stating the net income after tax for a specific case --/
theorem net_income_for_specific_case :
  let gross_income : ℝ := 45000
  let tax_rate : ℝ := 0.13
  net_income_after_tax gross_income tax_rate = 39150 := by
  sorry

#eval net_income_after_tax 45000 0.13

end NUMINAMATH_CALUDE_net_income_for_specific_case_l3119_311997


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l3119_311922

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : Prop := x^2 - (2*k + 1)*x + k^2 + k = 0

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a right triangle with sides a, b, c
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

-- Main theorem
theorem quadratic_roots_and_triangle (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_eq k x ∧ quadratic_eq k y) ∧
  (∃ a b : ℝ, quadratic_eq k a ∧ quadratic_eq k b ∧ is_right_triangle a b 5 → k = 3 ∨ k = 12) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l3119_311922


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l3119_311967

theorem one_and_two_thirds_of_number_is_45 : ∃ x : ℚ, (5 / 3) * x = 45 ∧ x = 27 := by sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l3119_311967


namespace NUMINAMATH_CALUDE_dacids_physics_marks_l3119_311911

theorem dacids_physics_marks :
  let english_marks : ℕ := 73
  let math_marks : ℕ := 69
  let chemistry_marks : ℕ := 64
  let biology_marks : ℕ := 82
  let average_marks : ℕ := 76
  let num_subjects : ℕ := 5

  let total_marks : ℕ := average_marks * num_subjects
  let known_marks : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks : ℕ := total_marks - known_marks

  physics_marks = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_dacids_physics_marks_l3119_311911


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3119_311999

/-- An isosceles triangle with a circle inscribed in it -/
structure IsoscelesTriangleWithInscribedCircle where
  -- Base of the isosceles triangle
  base : ℝ
  -- Height of the isosceles triangle
  height : ℝ
  -- Radius of the inscribed circle
  radius : ℝ
  -- The circle touches the base and both equal sides of the triangle
  touches_sides : True

/-- Theorem stating that for an isosceles triangle with base 20 and height 24, 
    the radius of the inscribed circle is 20/3 -/
theorem inscribed_circle_radius 
  (triangle : IsoscelesTriangleWithInscribedCircle)
  (h_base : triangle.base = 20)
  (h_height : triangle.height = 24) :
  triangle.radius = 20 / 3 := by
  sorry

#check inscribed_circle_radius

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3119_311999


namespace NUMINAMATH_CALUDE_minimal_intercept_line_l3119_311981

-- Define a line by its intercepts
structure Line where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0

-- Define the condition that the line passes through (1, 4)
def passesThrough (l : Line) : Prop :=
  1 / l.a + 4 / l.b = 1

-- Define the sum of intercepts
def sumOfIntercepts (l : Line) : ℝ :=
  l.a + l.b

-- State the theorem
theorem minimal_intercept_line :
  ∃ (l : Line),
    passesThrough l ∧
    ∀ (l' : Line), passesThrough l' → sumOfIntercepts l ≤ sumOfIntercepts l' ∧
    2 * (1 : ℝ) + 4 - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_minimal_intercept_line_l3119_311981


namespace NUMINAMATH_CALUDE_square_side_length_l3119_311916

theorem square_side_length (s : ℝ) (h : s > 0) : s ^ 2 = 2 * (4 * s) → s = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3119_311916


namespace NUMINAMATH_CALUDE_whole_number_between_l3119_311950

theorem whole_number_between : 
  ∀ N : ℕ, (6 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7) → (N = 25 ∨ N = 26 ∨ N = 27) := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_l3119_311950


namespace NUMINAMATH_CALUDE_reappearance_is_lcm_l3119_311998

/-- The number of letters in the sequence -/
def num_letters : ℕ := 6

/-- The number of digits in the sequence -/
def num_digits : ℕ := 4

/-- The line number where the original sequence first reappears -/
def reappearance_line : ℕ := 12

/-- Theorem stating that the reappearance line is the LCM of the letter and digit cycle lengths -/
theorem reappearance_is_lcm : 
  reappearance_line = Nat.lcm num_letters num_digits := by sorry

end NUMINAMATH_CALUDE_reappearance_is_lcm_l3119_311998


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l3119_311990

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length : ℝ := 12

-- Define the condition for point M
def point_M_condition (a b c : ℝ) : Prop :=
  b^2 / a = 2 * (a + c)

-- Theorem statement
theorem hyperbola_a_value (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧
  c = focal_length / 2 ∧
  c^2 = a^2 + b^2 ∧
  point_M_condition a b c →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l3119_311990


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_4_l3119_311939

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property

/-- Theorem: For a geometric sequence satisfying given conditions, S_4 = 75 -/
theorem geometric_sequence_sum_4 (seq : GeometricSequence)
  (h1 : seq.a 3 - seq.a 1 = 15)
  (h2 : seq.a 2 - seq.a 1 = 5) :
  seq.S 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_4_l3119_311939


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l3119_311927

theorem max_digits_product_5_4 : 
  ∃ (a b : ℕ), 
    (10000 ≤ a ∧ a < 100000) ∧ 
    (1000 ≤ b ∧ b < 10000) ∧ 
    (∀ x y : ℕ, (10000 ≤ x ∧ x < 100000) → (1000 ≤ y ∧ y < 10000) → x * y ≤ a * b) ∧
    (Nat.digits 10 (a * b)).length = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l3119_311927


namespace NUMINAMATH_CALUDE_sculpture_area_is_62_l3119_311928

/-- Represents a cube with a given edge length -/
structure Cube where
  edge : ℝ

/-- Represents a layer of the sculpture -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_sides : ℕ

/-- Represents the entire sculpture -/
structure Sculpture where
  bottom : Layer
  middle : Layer
  top : Layer

/-- Calculates the exposed surface area of a layer -/
def layer_area (c : Cube) (l : Layer) : ℝ :=
  (l.exposed_top + l.exposed_sides) * c.edge ^ 2

/-- Calculates the total exposed surface area of the sculpture -/
def total_area (c : Cube) (s : Sculpture) : ℝ :=
  layer_area c s.bottom + layer_area c s.middle + layer_area c s.top

/-- The sculpture described in the problem -/
def problem_sculpture : Sculpture :=
  { bottom := { cubes := 12, exposed_top := 12, exposed_sides := 24 },
    middle := { cubes := 6, exposed_top := 6, exposed_sides := 10 },
    top := { cubes := 2, exposed_top := 2, exposed_sides := 8 } }

/-- The cube used in the sculpture -/
def unit_cube : Cube :=
  { edge := 1 }

theorem sculpture_area_is_62 :
  total_area unit_cube problem_sculpture = 62 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_area_is_62_l3119_311928


namespace NUMINAMATH_CALUDE_circle_transformation_l3119_311921

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

theorem circle_transformation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point := reflect_x initial_point
  let final_point := translate_right reflected_point 10
  final_point = (13, 4) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l3119_311921


namespace NUMINAMATH_CALUDE_quadratic_trinomials_common_point_l3119_311976

theorem quadratic_trinomials_common_point (a b c : ℝ) :
  ∃ x : ℝ, (a * x^2 - b * x + c) = (b * x^2 - c * x + a) ∧
            (a * x^2 - b * x + c) = (c * x^2 - a * x + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomials_common_point_l3119_311976


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l3119_311972

theorem smaller_cube_side_length (a : ℕ) : 
  a > 0 ∧ 
  6 % a = 0 ∧ 
  6 * a^2 * (216 / a^3) = 432 → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l3119_311972


namespace NUMINAMATH_CALUDE_modulo_17_residue_l3119_311909

theorem modulo_17_residue : (342 + 6 * 47 + 8 * 157 + 3^3 * 21) % 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulo_17_residue_l3119_311909


namespace NUMINAMATH_CALUDE_derivative_at_one_l3119_311905

/-- Given a function f(x) = 2k*ln(x) - x, where k is a constant, prove that f'(1) = 1 -/
theorem derivative_at_one (k : ℝ) : 
  let f := fun (x : ℝ) => 2 * k * Real.log x - x
  deriv f 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3119_311905


namespace NUMINAMATH_CALUDE_circumradius_of_specific_triangle_l3119_311968

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the circumcircle radius
def circumradius (t : Triangle) : ℝ := sorry

-- State the theorem
theorem circumradius_of_specific_triangle :
  ∀ t : Triangle,
  t.a = 2 →
  t.A = 2 * π / 3 →  -- 120° in radians
  circumradius t = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_circumradius_of_specific_triangle_l3119_311968


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3119_311988

theorem complex_equation_solution (a : ℝ) (i : ℂ) 
  (hi : i * i = -1) 
  (h : (1 + a * i) * i = 3 + i) : 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3119_311988


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l3119_311918

/-- Given an ellipse and a hyperbola with equations as specified,
    if they have the same foci, then m = ±1 -/
theorem ellipse_hyperbola_same_foci (m : ℝ) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → x^2 / m^2 - y^2 / 2 = 1 → 
    (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m^2 + 2)) →
  m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l3119_311918


namespace NUMINAMATH_CALUDE_number_of_divisors_of_90_l3119_311900

theorem number_of_divisors_of_90 : Nat.card {d : ℕ | d ∣ 90} = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_90_l3119_311900


namespace NUMINAMATH_CALUDE_roses_in_vase_l3119_311947

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 7

/-- The number of roses added to the vase -/
def added_roses : ℕ := 16

/-- The total number of roses after addition -/
def total_roses : ℕ := 23

theorem roses_in_vase : initial_roses + added_roses = total_roses := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3119_311947


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3119_311970

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3119_311970


namespace NUMINAMATH_CALUDE_expected_value_specific_coin_l3119_311952

/-- A three-sided coin with probabilities and winnings for each outcome -/
structure ThreeSidedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  win_heads : ℚ
  win_tails : ℚ
  win_edge : ℚ

/-- The expected value of winnings for a three-sided coin flip -/
def expectedValue (coin : ThreeSidedCoin) : ℚ :=
  coin.prob_heads * coin.win_heads +
  coin.prob_tails * coin.win_tails +
  coin.prob_edge * coin.win_edge

/-- Theorem stating the expected value of winnings for a specific three-sided coin -/
theorem expected_value_specific_coin :
  ∃ (coin : ThreeSidedCoin),
    coin.prob_heads = 1/4 ∧
    coin.prob_tails = 3/4 - 1/20 ∧
    coin.prob_edge = 1/20 ∧
    coin.win_heads = 4 ∧
    coin.win_tails = -3 ∧
    coin.win_edge = -1 ∧
    coin.prob_heads + coin.prob_tails + coin.prob_edge = 1 ∧
    expectedValue coin = -23/20 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_specific_coin_l3119_311952


namespace NUMINAMATH_CALUDE_julie_work_hours_julie_school_year_hours_l3119_311935

/-- Given Julie's work schedule and earnings, calculate her required weekly hours during the school year --/
theorem julie_work_hours (summer_weekly_hours : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℕ :=
  let hourly_rate := summer_earnings / (summer_weekly_hours * summer_weeks)
  let school_year_hours := school_year_target / hourly_rate
  let school_year_weekly_hours := school_year_hours / school_year_weeks
  school_year_weekly_hours

/-- Prove that Julie needs to work 15 hours per week during the school year --/
theorem julie_school_year_hours : 
  julie_work_hours 60 10 8000 50 10000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_julie_work_hours_julie_school_year_hours_l3119_311935


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_25_l3119_311907

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a FourDigitNumber is valid (each digit is less than 10) -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  n.1 < 10 ∧ n.2.1 < 10 ∧ n.2.2.1 < 10 ∧ n.2.2.2 < 10

/-- Calculates the sum of digits of a FourDigitNumber -/
def digitSum (n : FourDigitNumber) : Nat :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2

/-- Converts a FourDigitNumber to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  1000 * n.1 + 100 * n.2.1 + 10 * n.2.2.1 + n.2.2.2

/-- The main theorem stating that 9970 is the largest four-digit number with digit sum 25 -/
theorem largest_four_digit_sum_25 :
  ∀ n : FourDigitNumber,
    isValidFourDigitNumber n →
    digitSum n = 25 →
    toNumber n ≤ 9970 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_25_l3119_311907


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3119_311957

theorem complex_absolute_value (z : ℂ) : z = 10 + 3*I → Complex.abs (z^2 + 8*z + 85) = 4 * Real.sqrt 3922 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3119_311957


namespace NUMINAMATH_CALUDE_bazylev_inequality_l3119_311940

theorem bazylev_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^y + y^z + z^x > 1 := by
  sorry

end NUMINAMATH_CALUDE_bazylev_inequality_l3119_311940


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3119_311969

theorem last_two_digits_sum : (13^27 + 17^27) % 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3119_311969


namespace NUMINAMATH_CALUDE_simplify_expression_l3119_311996

theorem simplify_expression (x : ℝ) : 2 - (2 * (1 - (2 - (2 * (1 - x))))) = 4 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3119_311996


namespace NUMINAMATH_CALUDE_boat_journey_time_l3119_311919

/-- Calculates the total time for a round trip boat journey affected by a stream -/
theorem boat_journey_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 9) 
  (h2 : stream_speed = 6) 
  (h3 : distance = 300) : 
  (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_boat_journey_time_l3119_311919


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_real_l3119_311977

theorem inequality_solution_implies_a_real : 
  (∃ x : ℝ, x^2 - a*x + a ≤ 1) → a ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_real_l3119_311977


namespace NUMINAMATH_CALUDE_rectangle_circle_tangent_l3119_311924

theorem rectangle_circle_tangent (r : ℝ) (h1 : r = 3) : 
  let circle_area := π * r^2
  let rectangle_area := 3 * circle_area
  let short_side := 2 * r
  let long_side := rectangle_area / short_side
  long_side = 4.5 * π := by sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangent_l3119_311924


namespace NUMINAMATH_CALUDE_base_eight_representation_l3119_311961

theorem base_eight_representation : ∃ (a b : Nat), 
  a ≠ b ∧ 
  a < 8 ∧ 
  b < 8 ∧
  777 = a * 8^3 + b * 8^2 + b * 8^1 + a * 8^0 :=
by sorry

end NUMINAMATH_CALUDE_base_eight_representation_l3119_311961


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l3119_311941

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l3119_311941


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3119_311995

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3119_311995


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3119_311933

theorem sqrt_product_equality : (Real.sqrt 12 + 2) * (Real.sqrt 3 - 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3119_311933


namespace NUMINAMATH_CALUDE_muffin_selection_problem_l3119_311948

theorem muffin_selection_problem :
  let n : ℕ := 6  -- Total number of muffins to select
  let k : ℕ := 4  -- Number of types of muffins
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_muffin_selection_problem_l3119_311948


namespace NUMINAMATH_CALUDE_opposite_pairs_l3119_311978

theorem opposite_pairs : 
  (-((-2)^3) ≠ -|((-2)^3)|) ∧ 
  ((-2)^3 ≠ -(2^3)) ∧ 
  (-2^2 = -(((-2)^2))) ∧ 
  (-(-2) ≠ -|(-2)|) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l3119_311978


namespace NUMINAMATH_CALUDE_james_sales_l3119_311912

/-- Given James' sales over two days, prove the total number of items sold --/
theorem james_sales (day1_houses day2_houses : ℕ) (day2_sale_rate : ℚ) : 
  day1_houses = 20 →
  day2_houses = 2 * day1_houses →
  day2_sale_rate = 4/5 →
  (day1_houses + (day2_houses : ℚ) * day2_sale_rate) * 2 = 104 := by
sorry

end NUMINAMATH_CALUDE_james_sales_l3119_311912


namespace NUMINAMATH_CALUDE_at_most_two_out_of_three_l3119_311942

-- Define the probability of a single event
def p : ℚ := 3 / 5

-- Define the number of events
def n : ℕ := 3

-- Define the maximum number of events we want to occur
def k : ℕ := 2

-- Theorem statement
theorem at_most_two_out_of_three (p : ℚ) (n : ℕ) (k : ℕ) 
  (h1 : p = 3 / 5) 
  (h2 : n = 3) 
  (h3 : k = 2) : 
  1 - p^n = 98 / 125 := by
  sorry

#check at_most_two_out_of_three p n k

end NUMINAMATH_CALUDE_at_most_two_out_of_three_l3119_311942


namespace NUMINAMATH_CALUDE_least_2023_divisors_decomposition_l3119_311975

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The least positive integer with exactly 2023 distinct positive divisors -/
def least_2023_divisors : ℕ := sorry

/-- m is the factor of least_2023_divisors not divisible by 10 -/
def m : ℕ := sorry

/-- k is the highest power of 10 that divides least_2023_divisors -/
def k : ℕ := sorry

theorem least_2023_divisors_decomposition :
  least_2023_divisors = m * 10^k ∧
  ¬(10 ∣ m) ∧
  num_divisors least_2023_divisors = 2023 ∧
  m + k = 999846 := by sorry

end NUMINAMATH_CALUDE_least_2023_divisors_decomposition_l3119_311975


namespace NUMINAMATH_CALUDE_pythagorean_theorem_construct_incommensurable_segments_l3119_311992

-- Define a type for geometric constructions
def GeometricConstruction : Type := Unit

-- Define a function to represent the construction of a segment
def constructSegment (length : ℝ) : GeometricConstruction := sorry

-- Define the Pythagorean theorem
theorem pythagorean_theorem (a b c : ℝ) : 
  a^2 + b^2 = c^2 ↔ ∃ (triangle : GeometricConstruction), true := sorry

-- Theorem stating that √2, √3, and √5 can be geometrically constructed
theorem construct_incommensurable_segments : 
  ∃ (construct_sqrt2 construct_sqrt3 construct_sqrt5 : GeometricConstruction),
    (∃ (a : ℝ), a^2 = 2 ∧ constructSegment a = construct_sqrt2) ∧
    (∃ (b : ℝ), b^2 = 3 ∧ constructSegment b = construct_sqrt3) ∧
    (∃ (c : ℝ), c^2 = 5 ∧ constructSegment c = construct_sqrt5) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_construct_incommensurable_segments_l3119_311992


namespace NUMINAMATH_CALUDE_engine_system_theorems_l3119_311930

/-- Engine connecting rod and crank system -/
structure EngineSystem where
  a : ℝ  -- length of crank OA
  b : ℝ  -- length of connecting rod AP
  α : ℝ  -- angle AOP
  β : ℝ  -- angle APO
  h : 0 < a ∧ 0 < b  -- positive lengths

/-- Theorems about the engine connecting rod and crank system -/
theorem engine_system_theorems (sys : EngineSystem) :
  -- Part 1
  sys.a * Real.sin sys.α = sys.b * Real.sin sys.β ∧
  -- Part 2
  (∀ β', Real.sin β' ≤ sys.a / sys.b) ∧
  -- Part 3
  ∀ x, x = sys.a * (1 - Real.cos sys.α) + sys.b * (1 - Real.cos sys.β) := by
  sorry

end NUMINAMATH_CALUDE_engine_system_theorems_l3119_311930


namespace NUMINAMATH_CALUDE_smaller_circle_area_l3119_311963

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  r : ℝ  -- radius of smaller circle
  R : ℝ  -- radius of larger circle
  PA : ℝ  -- length of tangent segment PA
  AB : ℝ  -- length of tangent segment AB
  tangent : r > 0 ∧ R > 0 ∧ R > r  -- circles are externally tangent
  common_tangent : PA = AB  -- common tangent property
  length_condition : PA = 4  -- given length condition

/-- The area of the smaller circle in the TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : π * tc.r^2 = 2 * π := by
  sorry

#check smaller_circle_area

end NUMINAMATH_CALUDE_smaller_circle_area_l3119_311963


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3119_311985

/-- The perimeter of a semicircle with radius 4.8 cm -/
theorem semicircle_perimeter : ℝ → ℝ := fun π =>
  let r : ℝ := 4.8
  π * r + 2 * r

#check semicircle_perimeter

end NUMINAMATH_CALUDE_semicircle_perimeter_l3119_311985


namespace NUMINAMATH_CALUDE_interest_rate_frequency_relationship_l3119_311982

/-- The nominal annual interest rate -/
def nominal_rate : ℝ := 0.16

/-- The effective annual interest rate -/
def effective_rate : ℝ := 0.1664

/-- The frequency of interest payments per year -/
def frequency : ℕ := 2

/-- Theorem stating that the given frequency satisfies the relationship between nominal and effective rates -/
theorem interest_rate_frequency_relationship : 
  (1 + nominal_rate / frequency)^frequency - 1 = effective_rate := by sorry

end NUMINAMATH_CALUDE_interest_rate_frequency_relationship_l3119_311982


namespace NUMINAMATH_CALUDE_larger_number_proof_l3119_311953

theorem larger_number_proof (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A - B = 1660) (h4 : 0.075 * A = 0.125 * B) : A = 4150 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3119_311953


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l3119_311917

/-- The probability that both A and B hit the target -/
def prob_both_hit : ℝ := 0.6

/-- The probability that A hits the target -/
def prob_A_hit : ℝ := prob_both_hit

/-- The probability that B hits the target -/
def prob_B_hit : ℝ := prob_both_hit

/-- The probability that exactly one of A and B hits the target -/
def prob_exactly_one_hit : ℝ := prob_A_hit * (1 - prob_B_hit) + (1 - prob_A_hit) * prob_B_hit

theorem exactly_one_hit_probability :
  prob_exactly_one_hit = 0.48 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l3119_311917


namespace NUMINAMATH_CALUDE_aquarium_has_hundred_fish_l3119_311949

/-- Represents the number of fish in an aquarium with specific conditions. -/
structure Aquarium where
  totalFish : ℕ
  clownfish : ℕ
  blowfish : ℕ
  blowfishInOwnTank : ℕ
  clownfishInDisplayTank : ℕ

/-- The aquarium satisfies the given conditions. -/
def validAquarium (a : Aquarium) : Prop :=
  a.clownfish = a.blowfish ∧
  a.blowfishInOwnTank = 26 ∧
  a.clownfishInDisplayTank = 16 ∧
  a.totalFish = a.clownfish + a.blowfish ∧
  a.clownfishInDisplayTank = (2 / 3 : ℚ) * (a.blowfish - a.blowfishInOwnTank)

/-- The theorem stating that a valid aquarium has 100 fish in total. -/
theorem aquarium_has_hundred_fish (a : Aquarium) (h : validAquarium a) : a.totalFish = 100 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_has_hundred_fish_l3119_311949


namespace NUMINAMATH_CALUDE_max_piles_is_30_l3119_311908

/-- Represents a configuration of stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : (piles.sum = 660)
  valid_ratio : ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- The maximum number of piles that can be formed -/
def max_piles : Nat := 30

/-- Theorem stating that 30 is the maximum number of piles -/
theorem max_piles_is_30 :
  ∀ sp : StonePiles, sp.piles.length ≤ max_piles :=
by sorry

end NUMINAMATH_CALUDE_max_piles_is_30_l3119_311908


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l3119_311903

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 9 →
    rectangle_height = 12 →
    (rectangle_width ^ 2 + rectangle_height ^ 2).sqrt * π = circle_circumference →
    circle_circumference = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l3119_311903


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3119_311932

/-- The perimeter of a semicircle with radius r is πr + 2r -/
theorem semicircle_perimeter (r : ℝ) (h : r > 0) : 
  let P := r * Real.pi + 2 * r
  P = r * Real.pi + 2 * r :=
by sorry

#check semicircle_perimeter

end NUMINAMATH_CALUDE_semicircle_perimeter_l3119_311932


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l3119_311936

/-- A geometric sequence with a_1 = 1 and a_9 = 3 has a_5 = √3 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →
  a 9 = 3 →
  a 5 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l3119_311936


namespace NUMINAMATH_CALUDE_smallest_k_for_zero_diff_l3119_311929

def u (n : ℕ) : ℕ := n^3 + 2*n^2 + n

def diff (f : ℕ → ℕ) (n : ℕ) : ℕ := f (n + 1) - f n

def diff_k (f : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => f
  | k + 1 => diff (diff_k f k)

theorem smallest_k_for_zero_diff (n : ℕ) : 
  (∀ n, diff_k u 4 n = 0) ∧ 
  (∀ k < 4, ∃ n, diff_k u k n ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_zero_diff_l3119_311929


namespace NUMINAMATH_CALUDE_vector_operation_result_l3119_311983

theorem vector_operation_result :
  let v1 : Fin 3 → ℝ := ![(-3), 4, 2]
  let v2 : Fin 3 → ℝ := ![1, 6, (-3)]
  2 • v1 + v2 = ![-5, 14, 1] := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3119_311983


namespace NUMINAMATH_CALUDE_complex_equation_l3119_311914

theorem complex_equation : (2 * Complex.I) * (1 + Complex.I)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l3119_311914


namespace NUMINAMATH_CALUDE_angle_ABH_in_regular_octagon_l3119_311960

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem angle_ABH_in_regular_octagon (ABCDEFGH : RegularOctagon) :
  let vertices := ABCDEFGH.vertices
  angle_measure (vertices 0) (vertices 1) (vertices 7) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABH_in_regular_octagon_l3119_311960


namespace NUMINAMATH_CALUDE_inequality_proof_l3119_311915

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  2 * (x^2 + y^2) ≥ (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3119_311915
