import Mathlib

namespace NUMINAMATH_CALUDE_projection_onto_yOz_l957_95717

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the projection of a point onto the yOz plane -/
def projectToYOZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Theorem stating that the projection of (1, -2, 3) onto the yOz plane is (-1, -2, 3) -/
theorem projection_onto_yOz :
  let p := Point3D.mk 1 (-2) 3
  projectToYOZ p = Point3D.mk (-1) (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_yOz_l957_95717


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l957_95769

theorem line_slope_intercept_product (m b : ℝ) : m = 3/4 ∧ b = -2 → m * b < -1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l957_95769


namespace NUMINAMATH_CALUDE_function_lower_bound_l957_95710

theorem function_lower_bound (a b : ℝ) (h : a + b = 4) :
  ∀ x : ℝ, |x + a^2| + |x - b^2| ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l957_95710


namespace NUMINAMATH_CALUDE_marathon_distance_l957_95733

/-- Calculates the distance Tomas can run after a given number of months of training -/
def distance_after_months (initial_distance : ℕ) (months : ℕ) : ℕ :=
  initial_distance * 2^months

/-- The marathon problem -/
theorem marathon_distance (initial_distance : ℕ) (training_months : ℕ) 
  (h1 : initial_distance = 3) 
  (h2 : training_months = 5) : 
  distance_after_months initial_distance training_months = 48 := by
  sorry

#eval distance_after_months 3 5

end NUMINAMATH_CALUDE_marathon_distance_l957_95733


namespace NUMINAMATH_CALUDE_grunters_win_probability_l957_95773

theorem grunters_win_probability : 
  let n_games : ℕ := 6
  let p_first_half : ℚ := 3/4
  let p_second_half : ℚ := 4/5
  let n_first_half : ℕ := 3
  let n_second_half : ℕ := 3
  
  (n_first_half + n_second_half = n_games) →
  (p_first_half ^ n_first_half * p_second_half ^ n_second_half = 27/125) :=
by sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l957_95773


namespace NUMINAMATH_CALUDE_absolute_value_even_and_increasing_l957_95771

def f (x : ℝ) := abs x

theorem absolute_value_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_even_and_increasing_l957_95771


namespace NUMINAMATH_CALUDE_jar_price_proportion_l957_95715

/-- Given two cylindrical jars with diameters d₁ and d₂, heights h₁ and h₂, 
    and the price of the first jar p₁, if the price is proportional to the volume, 
    then the price of the second jar p₂ is equal to p₁ * (d₂/d₁)² * (h₂/h₁). -/
theorem jar_price_proportion (d₁ d₂ h₁ h₂ p₁ p₂ : ℝ) (h_d₁_pos : d₁ > 0) (h_d₂_pos : d₂ > 0) 
    (h_h₁_pos : h₁ > 0) (h_h₂_pos : h₂ > 0) (h_p₁_pos : p₁ > 0) :
  p₂ = p₁ * (d₂/d₁)^2 * (h₂/h₁) ↔ 
  p₂ / (π * (d₂/2)^2 * h₂) = p₁ / (π * (d₁/2)^2 * h₁) := by
sorry

end NUMINAMATH_CALUDE_jar_price_proportion_l957_95715


namespace NUMINAMATH_CALUDE_multiply_72517_and_9999_l957_95701

theorem multiply_72517_and_9999 : 72517 * 9999 = 725097483 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72517_and_9999_l957_95701


namespace NUMINAMATH_CALUDE_exponent_simplification_l957_95795

theorem exponent_simplification (a : ℝ) (h : a > 0) : 
  a^(1/2) * a^(2/3) / a^(1/6) = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l957_95795


namespace NUMINAMATH_CALUDE_temperature_matches_data_temperature_decreases_with_altitude_constant_temperature_change_rate_l957_95758

-- Define the relationship between altitude and temperature
def temperature (h : ℝ) : ℝ := 20 - 6 * h

-- Define the set of data points from the table
def data_points : List (ℝ × ℝ) := [
  (0, 20), (1, 14), (2, 8), (3, 2), (4, -4), (5, -10)
]

-- Theorem stating that the temperature function matches the data points
theorem temperature_matches_data : ∀ (point : ℝ × ℝ), 
  point ∈ data_points → temperature point.1 = point.2 := by
  sorry

-- Theorem stating that the temperature decreases as altitude increases
theorem temperature_decreases_with_altitude : 
  ∀ (h1 h2 : ℝ), h1 < h2 → temperature h1 > temperature h2 := by
  sorry

-- Theorem stating that the rate of temperature change is constant
theorem constant_temperature_change_rate : 
  ∀ (h1 h2 : ℝ), h1 ≠ h2 → (temperature h2 - temperature h1) / (h2 - h1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_temperature_matches_data_temperature_decreases_with_altitude_constant_temperature_change_rate_l957_95758


namespace NUMINAMATH_CALUDE_cottage_build_time_l957_95711

/-- Represents the time (in days) it takes to build a cottage given the number of builders -/
def build_time (num_builders : ℕ) : ℚ := sorry

theorem cottage_build_time :
  build_time 3 = 8 →
  build_time 6 = 4 :=
by sorry

end NUMINAMATH_CALUDE_cottage_build_time_l957_95711


namespace NUMINAMATH_CALUDE_order_of_powers_l957_95737

theorem order_of_powers : 3^15 < 2^30 ∧ 2^30 < 10^10 := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l957_95737


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l957_95724

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (large : Rectangle) (shaded : Rectangle) :
  large.width = 12 ∧
  large.height = 10 ∧
  shaded.width = 5 ∧
  shaded.height = 11 ∧
  area shaded = 55 →
  perimeter large + perimeter shaded - 2 * (shaded.width + shaded.height) = 48 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l957_95724


namespace NUMINAMATH_CALUDE_factorial_ratio_sum_l957_95734

theorem factorial_ratio_sum (p q : ℕ) : 
  p < 10 → q < 10 → p > 0 → q > 0 → (840 : ℕ) = p! / q! → p + q = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_sum_l957_95734


namespace NUMINAMATH_CALUDE_limit_implies_a_and_b_limit_implies_a_range_l957_95705

-- Problem 1
theorem limit_implies_a_and_b (a b : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |2*n^2 / (n+2) - n*a - b| < ε) →
  a = 2 ∧ b = 4 := by sorry

-- Problem 2
theorem limit_implies_a_range (a : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) →
  -4 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_limit_implies_a_and_b_limit_implies_a_range_l957_95705


namespace NUMINAMATH_CALUDE_cary_walk_distance_l957_95728

/-- The number of calories Cary burns per mile walked -/
def calories_per_mile : ℝ := 150

/-- The number of calories in the candy bar Cary eats -/
def candy_bar_calories : ℝ := 200

/-- Cary's net calorie deficit -/
def net_calorie_deficit : ℝ := 250

/-- The number of miles Cary walked round-trip -/
def miles_walked : ℝ := 3

theorem cary_walk_distance :
  miles_walked * calories_per_mile - candy_bar_calories = net_calorie_deficit :=
by sorry

end NUMINAMATH_CALUDE_cary_walk_distance_l957_95728


namespace NUMINAMATH_CALUDE_only_C_is_comprehensive_unique_comprehensive_survey_l957_95740

/-- Represents a survey option -/
inductive SurveyOption
| A  -- Survey of the environmental awareness of the people nationwide
| B  -- Survey of the quality of mooncakes in the market during the Mid-Autumn Festival
| C  -- Survey of the weight of 40 students in a class
| D  -- Survey of the safety and quality of a certain type of fireworks and firecrackers

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that only option C is suitable for a comprehensive survey -/
theorem only_C_is_comprehensive :
  ∀ s : SurveyOption, isComprehensive s ↔ s = SurveyOption.C :=
by sorry

/-- Corollary: There exists exactly one comprehensive survey option -/
theorem unique_comprehensive_survey :
  ∃! s : SurveyOption, isComprehensive s :=
by sorry

end NUMINAMATH_CALUDE_only_C_is_comprehensive_unique_comprehensive_survey_l957_95740


namespace NUMINAMATH_CALUDE_power_function_value_l957_95761

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 2 = Real.sqrt 2 / 2) :
  f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l957_95761


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l957_95766

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  (∀ p : ℕ, Nat.Prime p → p ≤ 31 → ¬(p ∣ n)) →
  (∀ p : ℕ, Nat.Prime p → p < Real.sqrt (n : ℝ) → ¬(p ∣ n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l957_95766


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l957_95747

theorem average_of_five_numbers 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : (x₁ + x₂) / 2 = 12) 
  (h₂ : (x₃ + x₄ + x₅) / 3 = 7) : 
  (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l957_95747


namespace NUMINAMATH_CALUDE_cos_half_alpha_l957_95799

theorem cos_half_alpha (α : Real) 
  (h1 : 25 * (Real.sin α)^2 + Real.sin α - 24 = 0)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_half_alpha_l957_95799


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l957_95700

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 5 * x^2 + b₁ * x + 10 * x + 15 = 0 → (b₁ + 10)^2 = 300) ∧
  (∀ x, 5 * x^2 + b₂ * x + 10 * x + 15 = 0 → (b₂ + 10)^2 = 300) ∧
  (∀ b, (∀ x, 5 * x^2 + b * x + 10 * x + 15 = 0 → (b + 10)^2 = 300) → b = b₁ ∨ b = b₂) →
  b₁ + b₂ = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l957_95700


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l957_95770

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 - 3*i) / (-2 + 5*i) = (-27 : ℚ) / 29 - (24 : ℚ) / 29 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l957_95770


namespace NUMINAMATH_CALUDE_kanul_raw_materials_expenditure_l957_95787

/-- The problem of calculating Kanul's expenditure on raw materials -/
theorem kanul_raw_materials_expenditure
  (total : ℝ)
  (machinery : ℝ)
  (cash_percentage : ℝ)
  (h1 : total = 7428.57)
  (h2 : machinery = 200)
  (h3 : cash_percentage = 0.30)
  (h4 : ∃ (raw_materials : ℝ), raw_materials + machinery + cash_percentage * total = total) :
  ∃ (raw_materials : ℝ), raw_materials = 5000 :=
sorry

end NUMINAMATH_CALUDE_kanul_raw_materials_expenditure_l957_95787


namespace NUMINAMATH_CALUDE_function_characterization_l957_95744

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property that the function must satisfy
def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2*y*f x + (f y)^2

-- State the theorem
theorem function_characterization :
  ∀ f : RealFunction, SatisfiesEquation f ↔ (∀ x : ℝ, f x = x ∨ f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l957_95744


namespace NUMINAMATH_CALUDE_sin_three_pi_over_four_l957_95731

theorem sin_three_pi_over_four : Real.sin (3 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_over_four_l957_95731


namespace NUMINAMATH_CALUDE_angle_bisector_equation_l957_95764

/-- Given three lines y = x, y = 3x, and y = -x intersecting at the origin,
    the angle bisector of the smallest acute angle that passes through (1, 1)
    has the equation y = (2 - √11/2)x -/
theorem angle_bisector_equation (x y : ℝ) :
  let line1 : ℝ → ℝ := λ t => t
  let line2 : ℝ → ℝ := λ t => 3 * t
  let line3 : ℝ → ℝ := λ t => -t
  let bisector : ℝ → ℝ := λ t => (2 - Real.sqrt 11 / 2) * t
  (∀ t, line1 t = t ∧ line2 t = 3 * t ∧ line3 t = -t) →
  (bisector 0 = 0) →
  (bisector 1 = 1) →
  (∀ t, bisector t = (2 - Real.sqrt 11 / 2) * t) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_equation_l957_95764


namespace NUMINAMATH_CALUDE_local_value_of_three_is_300_l957_95792

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧
             hundreds ≥ 0 ∧ hundreds ≤ 9 ∧
             tens ≥ 0 ∧ tens ≤ 9 ∧
             ones ≥ 0 ∧ ones ≤ 9

/-- Calculate the local value of a digit given its place value -/
def localValue (digit : Nat) (placeValue : Nat) : Nat :=
  digit * placeValue

/-- Theorem: In the number 2345, if the sum of local values of all digits is 2345,
    then the local value of the digit 3 is 300 -/
theorem local_value_of_three_is_300 (n : FourDigitNumber)
    (h1 : n.thousands = 2 ∧ n.hundreds = 3 ∧ n.tens = 4 ∧ n.ones = 5)
    (h2 : localValue n.thousands 1000 + localValue n.hundreds 100 +
          localValue n.tens 10 + localValue n.ones 1 = 2345) :
    localValue n.hundreds 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_local_value_of_three_is_300_l957_95792


namespace NUMINAMATH_CALUDE_negative_square_range_l957_95703

theorem negative_square_range (x : ℝ) (h : -1 < x ∧ x < 0) : -1 < -x^2 ∧ -x^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_range_l957_95703


namespace NUMINAMATH_CALUDE_a_on_diameter_bck_l957_95723

/-- Given a triangle ABC with vertices A(a,b), B(0,0), C(c,0), and K as the intersection
    of the bisector of the exterior angle at C and the interior angle at B,
    prove that A lies on the line passing through K and the circumcenter of triangle BCK. -/
theorem a_on_diameter_bck (a b c : ℝ) : 
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (c, 0)
  let u : ℝ := Real.sqrt (a^2 + b^2)
  let w : ℝ := Real.sqrt ((a - c)^2 + b^2)
  let K : ℝ × ℝ := (c * (a + u) / (c + u - w), b * c / (c + u - w))
  let O : ℝ × ℝ := (c / 2, c * (a - c + w) * (c + u + w) / (2 * b * (c + u - w)))
  (∃ t : ℝ, A = (1 - t) • K + t • O) := by
    sorry

end NUMINAMATH_CALUDE_a_on_diameter_bck_l957_95723


namespace NUMINAMATH_CALUDE_yoongi_age_yoongi_age_when_namjoon_is_six_l957_95759

theorem yoongi_age (namjoon_age : ℕ) (age_difference : ℕ) : ℕ :=
  namjoon_age - age_difference

theorem yoongi_age_when_namjoon_is_six :
  yoongi_age 6 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_age_yoongi_age_when_namjoon_is_six_l957_95759


namespace NUMINAMATH_CALUDE_expand_product_l957_95784

theorem expand_product (x y : ℝ) : (3*x - 2) * (2*x + 4*y + 1) = 6*x^2 + 12*x*y - x - 8*y - 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l957_95784


namespace NUMINAMATH_CALUDE_iphone_price_proof_l957_95782

/-- The original price of an iPhone X -/
def original_price : ℝ := sorry

/-- The discount rate for buying at least 2 smartphones -/
def discount_rate : ℝ := 0.05

/-- The number of people buying iPhones -/
def num_buyers : ℕ := 3

/-- The amount saved by buying together -/
def amount_saved : ℝ := 90

theorem iphone_price_proof :
  (original_price * num_buyers * discount_rate = amount_saved) →
  original_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_iphone_price_proof_l957_95782


namespace NUMINAMATH_CALUDE_trigonometric_values_l957_95779

theorem trigonometric_values (α : Real) 
  (h1 : α ∈ Set.Ioo (π/3) (π/2))
  (h2 : Real.cos (π/6 + α) * Real.cos (π/3 - α) = -1/4) : 
  Real.sin (2*α) = Real.sqrt 3 / 2 ∧ 
  Real.tan α - 1 / Real.tan α = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_values_l957_95779


namespace NUMINAMATH_CALUDE_income_comparison_l957_95741

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mart = 0.84 * juan) : 
  (mart - tim) / tim = 0.4 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l957_95741


namespace NUMINAMATH_CALUDE_circle_area_l957_95789

theorem circle_area (r : ℝ) (h : r = 5) : π * r^2 = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l957_95789


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l957_95785

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 6 where f(-3) = -12, prove that f(3) = 24 -/
theorem polynomial_symmetry (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 6)
  (h2 : f (-3) = -12) : 
  f 3 = 24 := by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l957_95785


namespace NUMINAMATH_CALUDE_blue_eyed_blonde_proportion_l957_95745

/-- Proves that if the proportion of blondes among blue-eyed people is greater
    than the proportion of blondes among all people, then the proportion of
    blue-eyed people among blondes is greater than the proportion of blue-eyed
    people among all people. -/
theorem blue_eyed_blonde_proportion
  (l : ℕ) -- total number of people
  (g : ℕ) -- number of blue-eyed people
  (b : ℕ) -- number of blond-haired people
  (a : ℕ) -- number of people who are both blue-eyed and blond-haired
  (hl : l > 0)
  (hg : g > 0)
  (hb : b > 0)
  (ha : a > 0)
  (h_subset : a ≤ g ∧ a ≤ b ∧ g ≤ l ∧ b ≤ l)
  (h_proportion : (a : ℚ) / g > (b : ℚ) / l) :
  (a : ℚ) / b > (g : ℚ) / l :=
sorry

end NUMINAMATH_CALUDE_blue_eyed_blonde_proportion_l957_95745


namespace NUMINAMATH_CALUDE_parking_cost_average_l957_95708

/-- Parking cost structure and calculation -/
theorem parking_cost_average (base_cost : ℝ) (base_hours : ℝ) (additional_cost : ℝ) (total_hours : ℝ) : 
  base_cost = 20 →
  base_hours = 2 →
  additional_cost = 1.75 →
  total_hours = 9 →
  (base_cost + (total_hours - base_hours) * additional_cost) / total_hours = 3.58 := by
sorry

end NUMINAMATH_CALUDE_parking_cost_average_l957_95708


namespace NUMINAMATH_CALUDE_complex_magnitude_constraint_l957_95753

theorem complex_magnitude_constraint (a : ℝ) :
  let z : ℂ := 1 + a * I
  (Complex.abs z < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_constraint_l957_95753


namespace NUMINAMATH_CALUDE_rich_walking_distance_l957_95702

def house_to_sidewalk : ℕ := 20
def sidewalk_to_road_end : ℕ := 200

def total_distance : ℕ :=
  let to_road_end := house_to_sidewalk + sidewalk_to_road_end
  let to_intersection := to_road_end + 2 * to_road_end
  let to_route_end := to_intersection + to_intersection / 2
  2 * to_route_end

theorem rich_walking_distance :
  total_distance = 1980 := by
  sorry

end NUMINAMATH_CALUDE_rich_walking_distance_l957_95702


namespace NUMINAMATH_CALUDE_special_ring_classification_l957_95798

universe u

/-- A ring satisfying the given property -/
class SpecialRing (A : Type u) extends Ring A where
  special_property : ∀ (x : A), x ≠ 0 → x^(2^n + 1) = 1
  n : ℕ
  n_pos : n ≥ 1

/-- The theorem stating that any SpecialRing is isomorphic to F₂ or F₄ -/
theorem special_ring_classification (A : Type u) [SpecialRing A] :
  (∃ (f : A ≃+* Fin 2), Function.Bijective f) ∨
  (∃ (g : A ≃+* Fin 4), Function.Bijective g) :=
sorry

end NUMINAMATH_CALUDE_special_ring_classification_l957_95798


namespace NUMINAMATH_CALUDE_probability_real_roots_l957_95796

/-- The probability that the equation x^2 - mx + 4 = 0 has real roots,
    given that m is uniformly distributed in the interval [0, 6]. -/
theorem probability_real_roots : ℝ := by
  sorry

#check probability_real_roots

end NUMINAMATH_CALUDE_probability_real_roots_l957_95796


namespace NUMINAMATH_CALUDE_prob_last_roll_is_15th_l957_95748

/-- The number of sides on the die -/
def n : ℕ := 20

/-- The total number of rolls -/
def total_rolls : ℕ := 15

/-- The number of non-repeating rolls -/
def non_repeating_rolls : ℕ := 13

/-- Probability of getting a specific sequence of rolls on a n-sided die,
    where the first 'non_repeating_rolls' are different from their predecessors,
    and the last roll is the same as its predecessor -/
def prob_sequence (n : ℕ) (total_rolls : ℕ) (non_repeating_rolls : ℕ) : ℚ :=
  (n - 1 : ℚ)^non_repeating_rolls / n^(total_rolls - 1)

theorem prob_last_roll_is_15th :
  prob_sequence n total_rolls non_repeating_rolls = 19^13 / 20^14 := by
  sorry

end NUMINAMATH_CALUDE_prob_last_roll_is_15th_l957_95748


namespace NUMINAMATH_CALUDE_no_relationship_between_running_and_age_probability_of_one_not_interested_l957_95712

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ := !![15, 20; 10, 15]

-- Define the total sample size
def n : ℕ := 60

-- Define the K² formula
def K_squared (a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 90% confidence level
def critical_value : ℚ := 2.706

-- Theorem for part 1
theorem no_relationship_between_running_and_age :
  K_squared 15 20 10 15 < critical_value :=
sorry

-- Theorem for part 2
theorem probability_of_one_not_interested :
  (Nat.choose 5 3 - Nat.choose 3 3) / Nat.choose 5 3 = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_no_relationship_between_running_and_age_probability_of_one_not_interested_l957_95712


namespace NUMINAMATH_CALUDE_base12_addition_l957_95746

/-- Converts a base 12 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal number to its base 12 representation -/
def toBase12 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 12) ((m % 12) :: acc)
  aux n []

/-- The sum of 1704₁₂ and 259₁₂ in base 12 is equal to 1961₁₂ -/
theorem base12_addition :
  toBase12 (toDecimal [1, 7, 0, 4] + toDecimal [2, 5, 9]) = [1, 9, 6, 1] :=
by sorry

end NUMINAMATH_CALUDE_base12_addition_l957_95746


namespace NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l957_95729

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_scalene_triangle (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def prime_perimeter_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧
  is_valid_triangle a b c ∧
  is_prime (a + b + c)

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ),
    prime_perimeter_triangle a b c ∧
    (a + b + c = 29) ∧
    (∀ (x y z : ℕ),
      prime_perimeter_triangle x y z →
      (x + y + z ≠ 23) →
      (x + y + z ≥ 29)) :=
sorry

end NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l957_95729


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l957_95718

theorem simplify_sqrt_difference : 
  Real.sqrt (12 + 8 * Real.sqrt 3) - Real.sqrt (12 - 8 * Real.sqrt 3) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l957_95718


namespace NUMINAMATH_CALUDE_joan_attended_games_l957_95763

theorem joan_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 864) 
  (h2 : missed_games = 469) : 
  total_games - missed_games = 395 := by
  sorry

end NUMINAMATH_CALUDE_joan_attended_games_l957_95763


namespace NUMINAMATH_CALUDE_equation_root_implies_a_value_l957_95706

theorem equation_root_implies_a_value (x a : ℝ) : 
  ((x - 2) / (x + 4) = a / (x + 4)) → (∃ x, (x - 2) / (x + 4) = a / (x + 4)) → a = -6 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_implies_a_value_l957_95706


namespace NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l957_95749

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 8) :
  Complex.abs (w^3 + z^3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l957_95749


namespace NUMINAMATH_CALUDE_tunnel_safety_condition_l957_95767

def height_limit : ℝ := 4.5

def can_pass_safely (h : ℝ) : Prop := h ≤ height_limit

theorem tunnel_safety_condition (h : ℝ) :
  can_pass_safely h ↔ h ≤ height_limit :=
sorry

end NUMINAMATH_CALUDE_tunnel_safety_condition_l957_95767


namespace NUMINAMATH_CALUDE_simplify_expression_l957_95743

theorem simplify_expression (x : ℝ) : (3*x)^5 + (4*x)*(x^4) = 247*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l957_95743


namespace NUMINAMATH_CALUDE_equal_interval_points_ratio_l957_95755

theorem equal_interval_points_ratio : 
  ∀ (s S : ℝ), 
  (∃ d : ℝ, s = 9 * d ∧ S = 99 * d) → 
  S / s = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_interval_points_ratio_l957_95755


namespace NUMINAMATH_CALUDE_compute_expression_l957_95777

theorem compute_expression : 
  18 * (140 / 2 + 30 / 4 + 12 / 20 + 2 / 3) = 1417.8 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l957_95777


namespace NUMINAMATH_CALUDE_time_to_earn_house_cost_l957_95793

/-- Represents the financial situation of a man buying a house -/
structure HouseBuying where
  /-- The cost of the house -/
  houseCost : ℝ
  /-- Annual household expenses -/
  annualExpenses : ℝ
  /-- Annual savings -/
  annualSavings : ℝ
  /-- The man spends the same on expenses in 8 years as on savings in 12 years -/
  expensesSavingsRelation : 8 * annualExpenses = 12 * annualSavings
  /-- It takes 24 years to buy the house with all earnings -/
  buyingTime : houseCost = 24 * (annualExpenses + annualSavings)

/-- Theorem stating the time needed to earn the house cost -/
theorem time_to_earn_house_cost (hb : HouseBuying) :
  hb.houseCost / hb.annualSavings = 60 := by
  sorry

end NUMINAMATH_CALUDE_time_to_earn_house_cost_l957_95793


namespace NUMINAMATH_CALUDE_average_speed_calculation_l957_95760

/-- Given a distance of 100 kilometers traveled in 1.25 hours,
    prove that the average speed is 80 kilometers per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 100 →
  time = 1.25 →
  speed = distance / time →
  speed = 80 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l957_95760


namespace NUMINAMATH_CALUDE_four_integer_pairs_satisfy_equation_l957_95775

theorem four_integer_pairs_satisfy_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 1) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_integer_pairs_satisfy_equation_l957_95775


namespace NUMINAMATH_CALUDE_solve_equation_l957_95780

theorem solve_equation (x : ℚ) : (4 / 7) * (1 / 5) * x = 12 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l957_95780


namespace NUMINAMATH_CALUDE_jamie_earnings_l957_95768

def total_earnings (hourly_rate : ℝ) (days_per_week : ℕ) (hours_per_day : ℕ) (weeks_worked : ℕ) : ℝ :=
  hourly_rate * (days_per_week * hours_per_day * weeks_worked)

theorem jamie_earnings : 
  let hourly_rate : ℝ := 10
  let days_per_week : ℕ := 2
  let hours_per_day : ℕ := 3
  let weeks_worked : ℕ := 6
  total_earnings hourly_rate days_per_week hours_per_day weeks_worked = 360 := by
  sorry

end NUMINAMATH_CALUDE_jamie_earnings_l957_95768


namespace NUMINAMATH_CALUDE_expected_value_is_five_l957_95713

/-- Represents the outcome of rolling a fair 8-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- The probability of each outcome of the die roll -/
def prob : DieRoll → ℚ
  | _ => 1/8

/-- The winnings for each outcome of the die roll -/
def winnings : DieRoll → ℚ
  | DieRoll.two => 4
  | DieRoll.four => 8
  | DieRoll.six => 12
  | DieRoll.eight => 16
  | _ => 0

/-- The expected value of the winnings -/
def expected_value : ℚ :=
  (prob DieRoll.two * winnings DieRoll.two) +
  (prob DieRoll.four * winnings DieRoll.four) +
  (prob DieRoll.six * winnings DieRoll.six) +
  (prob DieRoll.eight * winnings DieRoll.eight)

theorem expected_value_is_five :
  expected_value = 5 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_five_l957_95713


namespace NUMINAMATH_CALUDE_algorithm_design_principle_l957_95742

-- Define the characteristics of algorithms
def Algorithm : Type := Unit

-- Define the properties of algorithms
def is_reversible (a : Algorithm) : Prop := sorry
def can_run_endlessly (a : Algorithm) : Prop := sorry
def is_unique_for_task (a : Algorithm) : Prop := sorry
def should_be_simple_and_convenient (a : Algorithm) : Prop := sorry

-- Define the theorem
theorem algorithm_design_principle :
  ∀ (a : Algorithm),
    ¬(is_reversible a) ∧
    ¬(can_run_endlessly a) ∧
    ¬(is_unique_for_task a) ∧
    should_be_simple_and_convenient a :=
by
  sorry

end NUMINAMATH_CALUDE_algorithm_design_principle_l957_95742


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l957_95751

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_two : a + b + c + d = 2) :
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l957_95751


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l957_95704

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  isosceles : dist A B = dist B C

-- Define the angles
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem isosceles_triangle_angle (A B C O : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : angle A B C = 80) 
  (h3 : angle O A C = 10) 
  (h4 : angle O C A = 30) : 
  angle A O B = 70 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_l957_95704


namespace NUMINAMATH_CALUDE_sara_wrapping_paper_l957_95783

theorem sara_wrapping_paper (total_paper : ℚ) (num_presents : ℕ) (paper_per_present : ℚ) :
  total_paper = 1/2 →
  num_presents = 5 →
  total_paper = num_presents * paper_per_present →
  paper_per_present = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_sara_wrapping_paper_l957_95783


namespace NUMINAMATH_CALUDE_only_one_proposition_is_true_l957_95750

-- Define the basic types
def Solid : Type := Unit
def View : Type := Unit

-- Define the properties
def has_three_identical_views (s : Solid) : Prop := sorry
def is_cube (s : Solid) : Prop := sorry
def front_view_is_rectangle (s : Solid) : Prop := sorry
def top_view_is_rectangle (s : Solid) : Prop := sorry
def is_cuboid (s : Solid) : Prop := sorry
def all_views_are_rectangles (s : Solid) : Prop := sorry
def front_view_is_isosceles_trapezoid (s : Solid) : Prop := sorry
def side_view_is_isosceles_trapezoid (s : Solid) : Prop := sorry
def is_frustum (s : Solid) : Prop := sorry

-- Define the propositions
def proposition1 : Prop := ∀ s : Solid, has_three_identical_views s → is_cube s
def proposition2 : Prop := ∀ s : Solid, front_view_is_rectangle s ∧ top_view_is_rectangle s → is_cuboid s
def proposition3 : Prop := ∀ s : Solid, all_views_are_rectangles s → is_cuboid s
def proposition4 : Prop := ∀ s : Solid, front_view_is_isosceles_trapezoid s ∧ side_view_is_isosceles_trapezoid s → is_frustum s

-- Theorem statement
theorem only_one_proposition_is_true : 
  (¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4) ∧
  (proposition1 → (¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4)) ∧
  (proposition2 → (¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4)) ∧
  (proposition4 → (¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3)) :=
sorry

end NUMINAMATH_CALUDE_only_one_proposition_is_true_l957_95750


namespace NUMINAMATH_CALUDE_net_salary_average_is_correct_l957_95776

/-- Represents Sharon's salary structure and performance scenarios -/
structure SalaryStructure where
  initial_salary : ℝ
  exceptional_increase : ℝ
  good_increase : ℝ
  average_increase : ℝ
  exceptional_bonus : ℝ
  good_bonus : ℝ
  federal_tax : ℝ
  state_tax : ℝ
  healthcare_deduction : ℝ

/-- Calculates the net salary for average performance -/
def net_salary_average (s : SalaryStructure) : ℝ :=
  let increased_salary := s.initial_salary * (1 + s.average_increase)
  let tax_deduction := increased_salary * (s.federal_tax + s.state_tax)
  increased_salary - tax_deduction - s.healthcare_deduction

/-- Theorem stating that the net salary for average performance is 497.40 -/
theorem net_salary_average_is_correct (s : SalaryStructure) 
    (h1 : s.initial_salary = 560)
    (h2 : s.average_increase = 0.15)
    (h3 : s.federal_tax = 0.10)
    (h4 : s.state_tax = 0.05)
    (h5 : s.healthcare_deduction = 50) :
    net_salary_average s = 497.40 := by
  sorry

#eval net_salary_average { 
  initial_salary := 560,
  exceptional_increase := 0.25,
  good_increase := 0.20,
  average_increase := 0.15,
  exceptional_bonus := 0.05,
  good_bonus := 0.03,
  federal_tax := 0.10,
  state_tax := 0.05,
  healthcare_deduction := 50
}

end NUMINAMATH_CALUDE_net_salary_average_is_correct_l957_95776


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l957_95790

theorem smallest_digit_for_divisibility_by_nine :
  ∃ (d : ℕ), d < 10 ∧ 
  (∀ (x : ℕ), x < d → ¬(9 ∣ (438000 + x * 100 + 4))) ∧
  (9 ∣ (438000 + d * 100 + 4)) ∧
  d = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l957_95790


namespace NUMINAMATH_CALUDE_dara_jane_age_ratio_l957_95781

-- Define the given conditions
def minimum_employment_age : ℕ := 25
def jane_current_age : ℕ := 28
def years_until_dara_minimum_age : ℕ := 14
def years_in_future : ℕ := 6

-- Define Dara's current age
def dara_current_age : ℕ := minimum_employment_age - years_until_dara_minimum_age

-- Define Dara's and Jane's ages in 6 years
def dara_future_age : ℕ := dara_current_age + years_in_future
def jane_future_age : ℕ := jane_current_age + years_in_future

-- Theorem to prove
theorem dara_jane_age_ratio : 
  dara_future_age * 2 = jane_future_age := by sorry

end NUMINAMATH_CALUDE_dara_jane_age_ratio_l957_95781


namespace NUMINAMATH_CALUDE_michaels_pets_l957_95756

theorem michaels_pets (total_pets : ℕ) 
  (h1 : (total_pets : ℝ) * 0.25 = total_pets * 0.5 + 9) 
  (h2 : (total_pets : ℝ) * 0.25 + total_pets * 0.5 + 9 = total_pets) : 
  total_pets = 36 := by
  sorry

end NUMINAMATH_CALUDE_michaels_pets_l957_95756


namespace NUMINAMATH_CALUDE_west_movement_representation_l957_95725

/-- Represents the direction of movement on an east-west road -/
inductive Direction
| East
| West

/-- Represents a movement on the road with a direction and distance -/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its numerical representation -/
def movementToNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

/-- The theorem stating that moving west by 7m should be denoted as -7m -/
theorem west_movement_representation :
  let eastMovement := Movement.mk Direction.East 3
  let westMovement := Movement.mk Direction.West 7
  movementToNumber eastMovement = 3 →
  movementToNumber westMovement = -7 :=
by sorry

end NUMINAMATH_CALUDE_west_movement_representation_l957_95725


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l957_95757

theorem solution_satisfies_equation :
  ∃ (x y : ℝ), x ≥ 0 ∧ y > 0 ∧
  Real.sqrt (9 + x) + Real.sqrt (9 - x) + Real.sqrt y = 5 * Real.sqrt 3 ∧
  x = 0 ∧ y = 111 - 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l957_95757


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l957_95738

/-- A triangle with an inscribed circle where the area is numerically twice the perimeter -/
structure SpecialTriangle where
  -- The semiperimeter of the triangle
  s : ℝ
  -- The area of the triangle
  A : ℝ
  -- The perimeter of the triangle
  p : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The semiperimeter is positive
  s_pos : 0 < s
  -- The perimeter is twice the semiperimeter
  perim_eq : p = 2 * s
  -- The area is twice the perimeter
  area_eq : A = 2 * p
  -- The area formula using inradius
  area_formula : A = r * s

/-- The radius of the inscribed circle in a SpecialTriangle is 4 -/
theorem inscribed_circle_radius (t : SpecialTriangle) : t.r = 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l957_95738


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l957_95754

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁*(x-1)^4 + a₂*(x-1)^3 + a₃*(x-1)^2 + a₄*(x-1) + a₅ = x^4) →
  a₂ + a₃ + a₄ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l957_95754


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l957_95732

/-- Given n, s, and r where s = 3^n + 2 and r = 4^s - 2s, 
    prove that when n = 3, r = 4^29 - 58 -/
theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
    (h1 : s = 3^n + 2) 
    (h2 : r = 4^s - 2*s) : 
  n = 3 → r = 4^29 - 58 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l957_95732


namespace NUMINAMATH_CALUDE_fraction_equality_l957_95707

theorem fraction_equality (p q r u v w : ℝ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0 ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_sum_squares1 : p^2 + q^2 + r^2 = 49)
  (h_sum_squares2 : u^2 + v^2 + w^2 = 64)
  (h_dot_product : p*u + q*v + r*w = 56)
  (h_p_2q : p = 2*q) :
  (p + q + r) / (u + v + w) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l957_95707


namespace NUMINAMATH_CALUDE_five_student_committees_from_eight_l957_95730

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of 5-student committees from 8 students is 56 -/
theorem five_student_committees_from_eight (n k : ℕ) (hn : n = 8) (hk : k = 5) :
  binomial n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_from_eight_l957_95730


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l957_95778

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 24 * x + 35 = (d * x + e)^2 + f) → 
  d * e = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l957_95778


namespace NUMINAMATH_CALUDE_cubic_root_values_l957_95721

theorem cubic_root_values (m : ℝ) : 
  (-1 : ℂ)^3 - (m^2 - m + 7)*(-1 : ℂ) - (3*m^2 - 3*m - 6) = 0 ↔ m = -2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_values_l957_95721


namespace NUMINAMATH_CALUDE_problem_solution_l957_95774

def f (a : ℝ) : ℝ → ℝ := fun x ↦ |x - a|

theorem problem_solution :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) ∧
   (∀ m : ℝ, (∀ x : ℝ, f 3 (2*x) + f 3 (x+2) ≥ m) ↔ m ≤ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l957_95774


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l957_95726

/-- Calculates the percentage of chromium in a new alloy formed by combining two alloys -/
theorem chromium_percentage_in_new_alloy 
  (weight1 : ℝ) (percentage1 : ℝ) 
  (weight2 : ℝ) (percentage2 : ℝ) :
  weight1 = 10 →
  weight2 = 30 →
  percentage1 = 12 →
  percentage2 = 8 →
  (weight1 * percentage1 / 100 + weight2 * percentage2 / 100) / (weight1 + weight2) * 100 = 9 :=
by sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l957_95726


namespace NUMINAMATH_CALUDE_vector_relations_l957_95736

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![3, -4]

def is_collinear (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

def is_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 = 0

theorem vector_relations :
  (∃ k : ℝ, is_collinear (fun i => k * a i - b i) (fun i => a i + b i) ∧ k = -1) ∧
  (∃ k : ℝ, is_perpendicular (fun i => k * a i - b i) (fun i => a i + b i) ∧ k = 16) := by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l957_95736


namespace NUMINAMATH_CALUDE_inequality_proof_l957_95797

theorem inequality_proof (k m n : ℕ+) (h1 : 1 < k) (h2 : k ≤ m) (h3 : m < n) :
  (1 + m.val : ℝ)^2 > (1 + n.val : ℝ)^m.val := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l957_95797


namespace NUMINAMATH_CALUDE_mod_multiplication_equivalence_l957_95786

theorem mod_multiplication_equivalence : 98 * 202 ≡ 71 [ZMOD 75] := by sorry

end NUMINAMATH_CALUDE_mod_multiplication_equivalence_l957_95786


namespace NUMINAMATH_CALUDE_circumcircle_radius_obtuse_triangle_consecutive_sides_l957_95735

/-- The radius of the circumcircle of an obtuse triangle with consecutive integer sides --/
theorem circumcircle_radius_obtuse_triangle_consecutive_sides : 
  ∀ (a b c : ℕ) (R : ℝ),
    a + 1 = b → b + 1 = c →  -- Consecutive integer sides
    a < b ∧ b < c →          -- Ordered sides
    a^2 + b^2 < c^2 →        -- Obtuse triangle condition
    R = (8 * Real.sqrt 15) / 15 →  -- Radius of circumcircle
    2 * R * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c := by
  sorry


end NUMINAMATH_CALUDE_circumcircle_radius_obtuse_triangle_consecutive_sides_l957_95735


namespace NUMINAMATH_CALUDE_no_married_triple_possible_all_married_triples_possible_l957_95722

/-- Represents a person in the alien race -/
structure Person where
  gender : Fin 3
  likes : Fin 3 → Finset (Fin n)

/-- Represents a married triple -/
structure MarriedTriple where
  male : Fin n
  female : Fin n
  emale : Fin n

/-- The set of all persons in the colony -/
def Colony (n : ℕ) : Finset Person := sorry

/-- Predicate to check if a person likes another person -/
def likes (p1 p2 : Person) : Prop := sorry

/-- Predicate to check if a triple is a valid married triple -/
def isMarriedTriple (t : MarriedTriple) (c : Colony n) : Prop := sorry

theorem no_married_triple_possible 
  (n : ℕ) 
  (k : ℕ) 
  (h1 : Even n) 
  (h2 : k ≥ n / 2) : 
  ∃ (c : Colony n), ∀ (t : MarriedTriple), ¬isMarriedTriple t c := by sorry

theorem all_married_triples_possible 
  (n : ℕ) 
  (k : ℕ) 
  (h : k ≥ 3 * n / 4) : 
  ∃ (c : Colony n), ∃ (triples : Finset MarriedTriple), 
    (∀ t ∈ triples, isMarriedTriple t c) ∧ 
    (triples.card = n) := by sorry

end NUMINAMATH_CALUDE_no_married_triple_possible_all_married_triples_possible_l957_95722


namespace NUMINAMATH_CALUDE_square_roots_problem_l957_95791

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (3*x - 2)^2 = a) (h3 : (5*x + 6)^2 = a) : a = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l957_95791


namespace NUMINAMATH_CALUDE_edward_spending_l957_95772

theorem edward_spending (initial : ℝ) : 
  initial > 0 →
  let after_clothes := initial - 250
  let after_food := after_clothes - (0.35 * after_clothes)
  let after_electronics := after_food - (0.5 * after_food)
  after_electronics = 200 →
  initial = 1875 := by sorry

end NUMINAMATH_CALUDE_edward_spending_l957_95772


namespace NUMINAMATH_CALUDE_sum_in_base8_l957_95788

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

theorem sum_in_base8 : 
  base10ToBase8 (base8ToBase10 24 + base8ToBase10 157) = 203 := by sorry

end NUMINAMATH_CALUDE_sum_in_base8_l957_95788


namespace NUMINAMATH_CALUDE_reflection_line_sum_l957_95739

/-- Given a line y = mx + b, if the reflection of point (2,2) across this line is (10,6), then m + b = 14 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), (x, y) = (10, 6) ∧ 
    (x - 2, y - 2) = (2 * (m * (x + 2) / 2 + b - y) / (1 + m^2), 
                      2 * (m * (y + 2) / 2 - (x + 2) / 2 + b) / (1 + m^2))) →
  m + b = 14 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l957_95739


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l957_95727

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  p ^ 2 % 12 = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l957_95727


namespace NUMINAMATH_CALUDE_arrangement_theorem_l957_95709

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- Calculate the number of ways person A and B can stand next to each other -/
def adjacent_AB : ℕ := sorry

/-- Calculate the number of ways person A and B can stand not next to each other -/
def not_adjacent_AB : ℕ := sorry

/-- Calculate the number of ways person A, B, and C can stand so that no two of them are next to each other -/
def no_two_adjacent_ABC : ℕ := sorry

/-- Calculate the number of ways person A, B, and C can stand so that at most two of them are not next to each other -/
def at_most_two_not_adjacent_ABC : ℕ := sorry

theorem arrangement_theorem :
  adjacent_AB = 1440 ∧
  not_adjacent_AB = 3600 ∧
  no_two_adjacent_ABC = 1440 ∧
  at_most_two_not_adjacent_ABC = 4320 := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l957_95709


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l957_95720

/-- Given a line segment AB of length 3r with point C on AB such that AC = r and CB = 2r,
    and semi-circles constructed on AB, AC, and CB, prove that the ratio of the shaded area
    to the area of a circle with radius equal to the radius of the semi-circle on CB is 2:1. -/
theorem shaded_area_ratio (r : ℝ) (h : r > 0) : 
  let total_area := π * (3 * r)^2 / 2
  let small_semicircle_area := π * r^2 / 2
  let medium_semicircle_area := π * (2 * r)^2 / 2
  let shaded_area := total_area - (small_semicircle_area + medium_semicircle_area)
  let circle_area := π * r^2
  shaded_area / circle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l957_95720


namespace NUMINAMATH_CALUDE_straight_A_students_after_increase_l957_95714

/-- The number of straight-A students after new students join, given the initial conditions -/
theorem straight_A_students_after_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (percentage_increase : ℚ) : ℕ :=
by
  -- Assume the initial conditions
  have h1 : initial_students = 25 := by sorry
  have h2 : new_students = 7 := by sorry
  have h3 : percentage_increase = 1/10 := by sorry

  -- Define the total number of students after new students join
  let total_students : ℕ := initial_students + new_students

  -- Define the function to calculate the number of straight-A students
  let calc_straight_A (x : ℕ) (y : ℕ) : Prop :=
    (x : ℚ) / initial_students + percentage_increase = ((x + y) : ℚ) / total_students

  -- Prove that there are 16 straight-A students after the increase
  have h4 : ∃ (x y : ℕ), calc_straight_A x y ∧ x + y = 16 := by sorry

  -- Conclude the theorem
  exact 16

end NUMINAMATH_CALUDE_straight_A_students_after_increase_l957_95714


namespace NUMINAMATH_CALUDE_pumpkin_pie_pieces_l957_95752

/-- The number of pieces a pumpkin pie is cut into -/
def pumpkin_pieces : ℕ := sorry

/-- The number of pieces a custard pie is cut into -/
def custard_pieces : ℕ := 6

/-- The price of a pumpkin pie slice in dollars -/
def pumpkin_price : ℕ := 5

/-- The price of a custard pie slice in dollars -/
def custard_price : ℕ := 6

/-- The number of pumpkin pies sold -/
def pumpkin_pies_sold : ℕ := 4

/-- The number of custard pies sold -/
def custard_pies_sold : ℕ := 5

/-- The total revenue in dollars -/
def total_revenue : ℕ := 340

theorem pumpkin_pie_pieces : 
  pumpkin_pieces * pumpkin_price * pumpkin_pies_sold + 
  custard_pieces * custard_price * custard_pies_sold = total_revenue → 
  pumpkin_pieces = 8 := by sorry

end NUMINAMATH_CALUDE_pumpkin_pie_pieces_l957_95752


namespace NUMINAMATH_CALUDE_product_three_consecutive_integers_div_by_6_l957_95719

theorem product_three_consecutive_integers_div_by_6 (n : ℤ) :
  ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by sorry

end NUMINAMATH_CALUDE_product_three_consecutive_integers_div_by_6_l957_95719


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l957_95762

/-- The type of condition sought in the analysis method for proving inequalities -/
inductive ConditionType
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  /-- The type of condition sought by the method -/
  condition_type : ConditionType

/-- Theorem: The analysis method for proving inequalities primarily seeks sufficient conditions -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (method : AnalysisMethod), method.condition_type = ConditionType.Sufficient :=
by
  sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l957_95762


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l957_95794

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l957_95794


namespace NUMINAMATH_CALUDE_inequality_system_solution_l957_95765

theorem inequality_system_solution (x : ℝ) :
  (x - 1) * Real.log 2 + Real.log (2^(x + 1) + 1) < Real.log (7 * 2^x + 12) →
  Real.log (x + 2) / Real.log x > 2 →
  1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l957_95765


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l957_95716

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (1 : ℝ) / (-(1 + m)) = m / (-2 : ℝ)

/-- If the lines x + (1+m)y = 2-m and mx + 2y + 8 = 0 are parallel, then m = 1 -/
theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel_lines m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l957_95716
