import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_B_l2820_282065

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2820_282065


namespace NUMINAMATH_CALUDE_volleyballs_left_l2820_282081

theorem volleyballs_left (total : ℕ) (lent : ℕ) (left : ℕ) : 
  total = 9 → lent = 5 → left = total - lent → left = 4 := by sorry

end NUMINAMATH_CALUDE_volleyballs_left_l2820_282081


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_252_675_l2820_282057

theorem lcm_gcf_ratio_252_675 : 
  Nat.lcm 252 675 / Nat.gcd 252 675 = 2100 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_252_675_l2820_282057


namespace NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l2820_282045

/-- The diameter of a circular field, given the cost per meter of fencing and the total cost. -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- The diameter of the specific circular field is approximately 16 meters. -/
theorem specific_field_diameter :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |circular_field_diameter 3 150.79644737231007 - 16| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l2820_282045


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2820_282084

theorem simplify_radical_product (y z : ℝ) :
  Real.sqrt (50 * y) * Real.sqrt (18 * z) * Real.sqrt (32 * y) = 40 * y * Real.sqrt (2 * z) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2820_282084


namespace NUMINAMATH_CALUDE_math_chemistry_intersection_l2820_282027

/-- Represents the number of students in various groups and their intersections -/
structure StudentGroups where
  total : ℕ
  math : ℕ
  physics : ℕ
  chemistry : ℕ
  math_physics : ℕ
  physics_chemistry : ℕ
  math_chemistry : ℕ

/-- The given conditions for the student groups -/
def given_groups : StudentGroups :=
  { total := 36
  , math := 26
  , physics := 15
  , chemistry := 13
  , math_physics := 6
  , physics_chemistry := 4
  , math_chemistry := 8 }

/-- Theorem stating that the number of students in both math and chemistry is 8 -/
theorem math_chemistry_intersection (g : StudentGroups) (h : g = given_groups) :
  g.math_chemistry = 8 := by
  sorry

end NUMINAMATH_CALUDE_math_chemistry_intersection_l2820_282027


namespace NUMINAMATH_CALUDE_doubled_roots_ratio_l2820_282037

theorem doubled_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : ∃ x₁ x₂ : ℝ, (x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0) ∧ 
                    ((2*x₁)^2 + b*(2*x₁) + c = 0 ∧ (2*x₂)^2 + b*(2*x₂) + c = 0)) :
  a / c = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_doubled_roots_ratio_l2820_282037


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2820_282009

-- Define the markup percentage
def markup : ℝ := 0.15

-- Define the selling price
def selling_price : ℝ := 6400

-- Theorem statement
theorem cost_price_calculation :
  ∃ (cost_price : ℝ), cost_price * (1 + markup) = selling_price :=
by
  sorry


end NUMINAMATH_CALUDE_cost_price_calculation_l2820_282009


namespace NUMINAMATH_CALUDE_yoongi_initial_money_l2820_282066

/-- The amount of money Yoongi had initially -/
def initial_money : ℕ := 590

/-- The cost of the candy Yoongi bought -/
def candy_cost : ℕ := 250

/-- The amount of pocket money Yoongi received -/
def pocket_money : ℕ := 500

/-- The amount of money Yoongi had left after all transactions -/
def money_left : ℕ := 420

theorem yoongi_initial_money :
  ∃ (pencil_cost : ℕ),
    initial_money = candy_cost + pencil_cost + money_left ∧
    initial_money + pocket_money - candy_cost = 2 * money_left :=
by
  sorry


end NUMINAMATH_CALUDE_yoongi_initial_money_l2820_282066


namespace NUMINAMATH_CALUDE_negative_square_nonpositive_l2820_282008

theorem negative_square_nonpositive (a : ℚ) : -a^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_nonpositive_l2820_282008


namespace NUMINAMATH_CALUDE_inequality_proof_l2820_282094

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2820_282094


namespace NUMINAMATH_CALUDE_cucumber_price_l2820_282061

theorem cucumber_price (cucumber_price : ℝ) 
  (tomato_price_relation : cucumber_price * 0.8 = cucumber_price - cucumber_price * 0.2)
  (total_price : 2 * (cucumber_price * 0.8) + 3 * cucumber_price = 23) :
  cucumber_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_price_l2820_282061


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2820_282004

/-- The line equation √3x - y + m = 0 is tangent to the circle x² + y² - 2x - 2 = 0 
    if and only if m = √3 or m = -3√3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, (Real.sqrt 3 * x - y + m = 0) → 
   (x^2 + y^2 - 2*x - 2 = 0) → 
   (∀ ε > 0, ∃ x' y' : ℝ, 
     x' ≠ x ∧ y' ≠ y ∧ 
     (Real.sqrt 3 * x' - y' + m = 0) ∧ 
     (x'^2 + y'^2 - 2*x' - 2 ≠ 0) ∧
     ((x' - x)^2 + (y' - y)^2 < ε^2))) ↔ 
  (m = Real.sqrt 3 ∨ m = -3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2820_282004


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2820_282067

theorem quadratic_inequality_theorem (k : ℝ) : 
  (¬∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0) → 
  (k = 1 ∨ (1 < k ∧ k < 7)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2820_282067


namespace NUMINAMATH_CALUDE_angle_properties_l2820_282044

theorem angle_properties (α : ℝ) (h : Real.tan α = -4/3) :
  (2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 2) ∧
  ((2 * Real.sin (π - α) + Real.sin (π/2 - α) + Real.sin (4*π)) / 
   (Real.cos (3*π/2 - α) + Real.cos (-α)) = -5/7) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l2820_282044


namespace NUMINAMATH_CALUDE_exponent_division_l2820_282092

theorem exponent_division (a : ℝ) (m n : ℕ) (h : m > n) :
  a^m / a^n = a^(m - n) := by sorry

end NUMINAMATH_CALUDE_exponent_division_l2820_282092


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2820_282096

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (4 / 10 : ℚ) + (6 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (12 / 10 : ℚ) + (14 / 10 : ℚ) + (16 / 10 : ℚ) + (18 / 10 : ℚ) + (20 / 10 : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2820_282096


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2820_282024

/-- If f(x) = kx - ln x is monotonically increasing on (1, +∞), then k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) : 
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2820_282024


namespace NUMINAMATH_CALUDE_house_profit_percentage_l2820_282080

/-- Proves that given two houses sold at $10,000 each, with a 10% loss on the second house
    and a 17% net profit overall, the profit percentage on the first house is approximately 67.15%. -/
theorem house_profit_percentage (selling_price : ℝ) (loss_percentage : ℝ) (net_profit_percentage : ℝ) :
  selling_price = 10000 →
  loss_percentage = 0.10 →
  net_profit_percentage = 0.17 →
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 0.6715) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_house_profit_percentage_l2820_282080


namespace NUMINAMATH_CALUDE_simplify_expression_l2820_282043

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2*x + 1)) = 1 / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2820_282043


namespace NUMINAMATH_CALUDE_height_equals_base_l2820_282063

/-- An isosceles triangle with constant perimeter for inscribed rectangles -/
structure ConstantPerimeterTriangle where
  -- The base of the triangle
  base : ℝ
  -- The height of the triangle
  height : ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The perimeter of any inscribed rectangle is constant
  constantPerimeter : True

/-- Theorem: In a ConstantPerimeterTriangle, the height equals the base -/
theorem height_equals_base (t : ConstantPerimeterTriangle) : t.height = t.base := by
  sorry

end NUMINAMATH_CALUDE_height_equals_base_l2820_282063


namespace NUMINAMATH_CALUDE_cubic_symmetry_extrema_l2820_282079

/-- A cubic function that is symmetric about the origin -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (b - 3) * x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * (a - 1) * x + 144

/-- The discriminant of f' = 0 -/
def discriminant (a : ℝ) : ℝ := 4 * (a^2 - 434 * a + 1)

theorem cubic_symmetry_extrema (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- symmetry about the origin
  (∃ x_min, ∀ x, f a b x_min ≤ f a b x) ∧ 
  (∃ x_max, ∀ x, f a b x ≤ f a b x_max) := by
  sorry

end NUMINAMATH_CALUDE_cubic_symmetry_extrema_l2820_282079


namespace NUMINAMATH_CALUDE_sarah_pencils_count_l2820_282022

/-- The number of pencils Sarah buys on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah buys on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The number of pencils Sarah buys on Wednesday -/
def wednesday_pencils : ℕ := 3 * tuesday_pencils

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := monday_pencils + tuesday_pencils + wednesday_pencils

theorem sarah_pencils_count : total_pencils = 92 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pencils_count_l2820_282022


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2820_282076

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1) + 2
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2820_282076


namespace NUMINAMATH_CALUDE_base_representation_equivalence_l2820_282077

/-- Represents a positive integer in different bases -/
structure BaseRepresentation where
  base8 : ℕ  -- Representation in base 8
  base5 : ℕ  -- Representation in base 5
  base10 : ℕ -- Representation in base 10
  is_valid : base8 ≥ 10 ∧ base8 < 100 ∧ base5 ≥ 10 ∧ base5 < 100

/-- Converts a two-digit number in base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  8 * (n / 10) + (n % 10)

/-- Converts a two-digit number in base 5 to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ :=
  5 * (n % 10) + (n / 10)

/-- Theorem stating the equivalence of the representations -/
theorem base_representation_equivalence (n : BaseRepresentation) : 
  base8_to_base10 n.base8 = base5_to_base10 n.base5 ∧ 
  base8_to_base10 n.base8 = n.base10 ∧ 
  n.base10 = 39 := by
  sorry

#check base_representation_equivalence

end NUMINAMATH_CALUDE_base_representation_equivalence_l2820_282077


namespace NUMINAMATH_CALUDE_abc_product_l2820_282036

theorem abc_product (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 198)
  (h2 : b * (c + a) = 210)
  (h3 : c * (a + b) = 222) :
  a * b * c = 1069 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2820_282036


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2820_282074

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → Complex.abs (4 + n * Complex.I) = 4 * Real.sqrt 13 → n = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2820_282074


namespace NUMINAMATH_CALUDE_remainder_3_pow_23_mod_11_l2820_282033

theorem remainder_3_pow_23_mod_11 : 3^23 % 11 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_3_pow_23_mod_11_l2820_282033


namespace NUMINAMATH_CALUDE_odometer_sum_of_squares_l2820_282030

def is_valid_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10

def circular_shift (a b c : ℕ) : ℕ :=
  100 * c + 10 * a + b

def original_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem odometer_sum_of_squares (a b c : ℕ) :
  is_valid_number a b c →
  (circular_shift a b c - original_number a b c) % 60 = 0 →
  a^2 + b^2 + c^2 = 54 := by
sorry

end NUMINAMATH_CALUDE_odometer_sum_of_squares_l2820_282030


namespace NUMINAMATH_CALUDE_cassidy_grounding_period_l2820_282001

/-- Calculates the total grounding period for Cassidy based on her grades and volunteering. -/
def calculate_grounding_period (
  initial_grounding : ℕ
  ) (extra_days_per_grade : ℕ
  ) (grades_below_b : ℕ
  ) (extracurricular_below_b : ℕ
  ) (volunteering_reduction : ℕ
  ) : ℕ :=
  let subject_penalty := grades_below_b * extra_days_per_grade
  let extracurricular_penalty := extracurricular_below_b * (extra_days_per_grade / 2)
  let total_before_volunteering := initial_grounding + subject_penalty + extracurricular_penalty
  total_before_volunteering - volunteering_reduction

/-- Theorem stating that Cassidy's total grounding period is 27 days. -/
theorem cassidy_grounding_period :
  calculate_grounding_period 14 3 4 2 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounding_period_l2820_282001


namespace NUMINAMATH_CALUDE_min_sum_of_product_3960_l2820_282091

theorem min_sum_of_product_3960 (a b c : ℕ+) (h : a * b * c = 3960) :
  (∀ x y z : ℕ+, x * y * z = 3960 → a + b + c ≤ x + y + z) ∧ a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_3960_l2820_282091


namespace NUMINAMATH_CALUDE_min_value_of_x_l2820_282034

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 2 + (1/2) * Real.log x) : x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l2820_282034


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l2820_282042

/-- The function f(x) = (x+1)^2 - 3 -/
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

/-- The theorem stating that -1 is the smallest value of c for which f has an inverse on [c,∞) -/
theorem smallest_c_for_inverse :
  ∀ c : ℝ, (∀ x y, x ∈ Set.Ici c → y ∈ Set.Ici c → f x = f y → x = y) ↔ c ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l2820_282042


namespace NUMINAMATH_CALUDE_uniform_probability_diff_colors_l2820_282035

def shorts_colors := Fin 3
def jersey_colors := Fin 3

def total_combinations : ℕ := 9

def matching_combinations : ℕ := 2

theorem uniform_probability_diff_colors :
  (total_combinations - matching_combinations) / total_combinations = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_uniform_probability_diff_colors_l2820_282035


namespace NUMINAMATH_CALUDE_exponent_division_l2820_282075

theorem exponent_division (a : ℝ) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2820_282075


namespace NUMINAMATH_CALUDE_chocolates_distribution_l2820_282007

theorem chocolates_distribution (total_children boys girls : ℕ) 
  (boys_chocolates girls_chocolates : ℕ) :
  total_children = 120 →
  boys = 60 →
  girls = 60 →
  boys + girls = total_children →
  boys_chocolates = 2 →
  girls_chocolates = 3 →
  boys * boys_chocolates + girls * girls_chocolates = 300 :=
by sorry

end NUMINAMATH_CALUDE_chocolates_distribution_l2820_282007


namespace NUMINAMATH_CALUDE_product_327_8_and_7_8_l2820_282049

/-- Convert a number from base 8 to base 10 -/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to base 8 -/
def base10To8 (n : ℕ) : ℕ := sorry

/-- Multiply two numbers in base 8 -/
def multiplyBase8 (a b : ℕ) : ℕ :=
  base10To8 (base8To10 a * base8To10 b)

theorem product_327_8_and_7_8 :
  multiplyBase8 327 7 = 2741 := by sorry

end NUMINAMATH_CALUDE_product_327_8_and_7_8_l2820_282049


namespace NUMINAMATH_CALUDE_correct_young_sample_size_l2820_282032

/-- Represents the stratified sampling problem for a company's employees. -/
structure CompanySampling where
  total_employees : ℕ
  young_employees : ℕ
  sample_size : ℕ
  young_in_sample : ℕ

/-- Theorem stating the correct number of young employees in the sample. -/
theorem correct_young_sample_size (c : CompanySampling) 
    (h1 : c.total_employees = 200)
    (h2 : c.young_employees = 120)
    (h3 : c.sample_size = 25)
    (h4 : c.young_in_sample = c.young_employees * c.sample_size / c.total_employees) :
  c.young_in_sample = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_young_sample_size_l2820_282032


namespace NUMINAMATH_CALUDE_greatest_x_value_l2820_282014

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k, p.Prime ∧ k > 0 ∧ n = p ^ k

theorem greatest_x_value (x : ℕ) 
  (h1 : Nat.lcm x (Nat.lcm 15 21) = 105)
  (h2 : is_prime_power x) :
  x ≤ 7 ∧ (∀ y, y > x → is_prime_power y → Nat.lcm y (Nat.lcm 15 21) ≠ 105) :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2820_282014


namespace NUMINAMATH_CALUDE_company_profit_ratio_l2820_282078

/-- Represents the revenues of a company in a given year -/
structure Revenue where
  amount : ℝ

/-- Calculates the profit given a revenue and a profit percentage -/
def profit (revenue : Revenue) (percentage : ℝ) : ℝ := revenue.amount * percentage

/-- Company N's revenues over three years -/
structure CompanyN where
  revenue2008 : Revenue
  revenue2009 : Revenue
  revenue2010 : Revenue
  revenue2009_eq : revenue2009.amount = 0.8 * revenue2008.amount
  revenue2010_eq : revenue2010.amount = 1.3 * revenue2009.amount

/-- Company M's revenues over three years -/
structure CompanyM where
  revenue : Revenue

theorem company_profit_ratio (n : CompanyN) (m : CompanyM) :
  (profit n.revenue2008 0.08 + profit n.revenue2009 0.15 + profit n.revenue2010 0.10) /
  (profit m.revenue 0.12 + profit m.revenue 0.18 + profit m.revenue 0.14) =
  (0.304 * n.revenue2008.amount) / (0.44 * m.revenue.amount) := by
  sorry

end NUMINAMATH_CALUDE_company_profit_ratio_l2820_282078


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2820_282002

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x - y + b₁ = 0 ↔ m₂ * x - y + b₂ = 0) ↔ m₁ = m₂

/-- The value of a for which ax-2y+2=0 is parallel to x+(a-3)y+1=0 -/
theorem parallel_lines_a_value :
  ∃ a : ℝ, (∀ x y : ℝ, a * x - 2 * y + 2 = 0 ↔ x + (a - 3) * y + 1 = 0) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_a_value_l2820_282002


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_solutions_l2820_282053

theorem quadratic_equation_positive_solutions :
  ∃! (x : ℝ), x > 0 ∧ x^2 = -6*x + 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_solutions_l2820_282053


namespace NUMINAMATH_CALUDE_deductive_reasoning_is_general_to_specific_l2820_282060

/-- Represents a form of reasoning --/
inductive ReasoningForm
  | GeneralToSpecific
  | SpecificToGeneral
  | GeneralToGeneral
  | SpecificToSpecific

/-- Definition of deductive reasoning --/
def deductive_reasoning : ReasoningForm := ReasoningForm.GeneralToSpecific

/-- Theorem stating that deductive reasoning is from general to specific --/
theorem deductive_reasoning_is_general_to_specific :
  deductive_reasoning = ReasoningForm.GeneralToSpecific := by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_is_general_to_specific_l2820_282060


namespace NUMINAMATH_CALUDE_no_single_digit_quadratic_solution_l2820_282020

theorem no_single_digit_quadratic_solution :
  ¬∃ (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧
  ∃ (x : ℕ), x > 0 ∧ x^2 - (2*A)*x + (A^2 + 1) = 0 :=
sorry

end NUMINAMATH_CALUDE_no_single_digit_quadratic_solution_l2820_282020


namespace NUMINAMATH_CALUDE_outfit_count_l2820_282041

/-- The number of different outfits that can be made with shirts, pants, and hats of different colors. -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
                (pants : ℕ) 
                (red_hats green_hats blue_hats : ℕ) : ℕ :=
  (red_shirts * pants * (green_hats + blue_hats)) +
  (green_shirts * pants * (red_hats + blue_hats)) +
  (blue_shirts * pants * (red_hats + green_hats))

/-- Theorem stating the number of outfits under given conditions. -/
theorem outfit_count : 
  num_outfits 4 4 4 10 6 6 4 = 1280 :=
by sorry

end NUMINAMATH_CALUDE_outfit_count_l2820_282041


namespace NUMINAMATH_CALUDE_ratio_satisfies_condition_l2820_282013

/-- Represents the number of people in each profession -/
structure ProfessionCount where
  doctors : ℕ
  lawyers : ℕ
  engineers : ℕ

/-- The average age of the entire group -/
def groupAverage : ℝ := 45

/-- The average age of doctors -/
def doctorAverage : ℝ := 40

/-- The average age of lawyers -/
def lawyerAverage : ℝ := 55

/-- The average age of engineers -/
def engineerAverage : ℝ := 35

/-- Checks if the given profession count satisfies the average age conditions -/
def satisfiesAverageCondition (count : ProfessionCount) : Prop :=
  let totalPeople := count.doctors + count.lawyers + count.engineers
  let totalAge := doctorAverage * count.doctors + lawyerAverage * count.lawyers + engineerAverage * count.engineers
  totalAge / totalPeople = groupAverage

/-- The theorem stating that the ratio 2:2:1 satisfies the average age conditions -/
theorem ratio_satisfies_condition :
  ∃ (k : ℕ), k > 0 ∧ satisfiesAverageCondition { doctors := 2 * k, lawyers := 2 * k, engineers := k } :=
sorry

end NUMINAMATH_CALUDE_ratio_satisfies_condition_l2820_282013


namespace NUMINAMATH_CALUDE_min_detectors_for_specific_board_and_ship_l2820_282070

/-- Represents a grid board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship -/
structure Ship :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a detector placement strategy -/
def DetectorStrategy := ℕ → ℕ → Bool

/-- Checks if a detector strategy can determine the ship's position -/
def can_determine_position (b : Board) (s : Ship) (strategy : DetectorStrategy) : Prop :=
  sorry

/-- The minimum number of detectors required -/
def min_detectors (b : Board) (s : Ship) : ℕ :=
  sorry

theorem min_detectors_for_specific_board_and_ship :
  let b : Board := ⟨2015, 2015⟩
  let s : Ship := ⟨1500, 1500⟩
  min_detectors b s = 1030 :=
sorry

end NUMINAMATH_CALUDE_min_detectors_for_specific_board_and_ship_l2820_282070


namespace NUMINAMATH_CALUDE_factors_imply_value_l2820_282086

/-- The polynomial p(x) = 3x^3 - mx + n -/
def p (m n : ℝ) (x : ℝ) : ℝ := 3 * x^3 - m * x + n

theorem factors_imply_value (m n : ℝ) 
  (h1 : p m n 3 = 0)  -- x-3 is a factor
  (h2 : p m n (-4) = 0)  -- x+4 is a factor
  : |3*m - 2*n| = 33 := by
  sorry

end NUMINAMATH_CALUDE_factors_imply_value_l2820_282086


namespace NUMINAMATH_CALUDE_bakery_stop_difference_l2820_282059

/-- Represents the distances between locations in Kona's trip -/
structure TripDistances where
  apartment_to_bakery : ℕ
  bakery_to_grandma : ℕ
  grandma_to_apartment : ℕ

/-- Calculates the additional miles driven with a bakery stop -/
def additional_miles (d : TripDistances) : ℕ :=
  (d.apartment_to_bakery + d.bakery_to_grandma + d.grandma_to_apartment) -
  (2 * d.grandma_to_apartment)

/-- Theorem stating that the additional miles driven with a bakery stop is 6 -/
theorem bakery_stop_difference (d : TripDistances)
  (h1 : d.apartment_to_bakery = 9)
  (h2 : d.bakery_to_grandma = 24)
  (h3 : d.grandma_to_apartment = 27) :
  additional_miles d = 6 := by
  sorry


end NUMINAMATH_CALUDE_bakery_stop_difference_l2820_282059


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2820_282093

theorem complex_simplification_and_multiplication :
  ((-5 + 3 * Complex.I) - (2 - 7 * Complex.I)) * (1 + 2 * Complex.I) = -27 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2820_282093


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2820_282039

theorem fraction_to_decimal : (45 : ℚ) / 64 = 0.703125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2820_282039


namespace NUMINAMATH_CALUDE_walters_hourly_wage_l2820_282051

/-- Walter's work schedule and earnings allocation --/
structure WorkSchedule where
  days_per_week : ℕ
  hours_per_day : ℕ
  school_allocation_ratio : ℚ
  school_allocation_amount : ℚ

/-- Calculate Walter's hourly wage --/
def hourly_wage (w : WorkSchedule) : ℚ :=
  w.school_allocation_amount / w.school_allocation_ratio / (w.days_per_week * w.hours_per_day)

/-- Theorem: Walter's hourly wage is $5 --/
theorem walters_hourly_wage (w : WorkSchedule)
  (h1 : w.days_per_week = 5)
  (h2 : w.hours_per_day = 4)
  (h3 : w.school_allocation_ratio = 3/4)
  (h4 : w.school_allocation_amount = 75) :
  hourly_wage w = 5 := by
  sorry

end NUMINAMATH_CALUDE_walters_hourly_wage_l2820_282051


namespace NUMINAMATH_CALUDE_roots_relation_l2820_282006

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- State the theorem
theorem roots_relation (b c d : ℝ) : 
  (∀ x y : ℝ, x ≠ y → f x = 0 → f y = 0 → x ≠ y) →  -- f has distinct roots
  (∀ r : ℝ, f r = 0 → g (r^3) b c d = 0) →          -- roots of g are cubes of roots of f
  b = -8 ∧ c = -36 ∧ d = -64 := by
  sorry

end NUMINAMATH_CALUDE_roots_relation_l2820_282006


namespace NUMINAMATH_CALUDE_square_fitting_theorem_l2820_282098

theorem square_fitting_theorem :
  ∃ (N : ℕ), N > 0 ∧ (N : ℝ) * (1 / N) ≤ 1 ∧ 4 * N = 1992 := by
  sorry

end NUMINAMATH_CALUDE_square_fitting_theorem_l2820_282098


namespace NUMINAMATH_CALUDE_tangent_circles_constant_l2820_282046

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1_center : ℝ × ℝ) (c1_radius : ℝ) (c2_center : ℝ × ℝ) (c2_radius : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (c1_radius + c2_radius)^2

/-- The theorem stating the value of 'a' for which the given circles are tangent -/
theorem tangent_circles_constant (a : ℝ) : 
  are_tangent (0, 0) 1 (-4, a) 5 ↔ a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_constant_l2820_282046


namespace NUMINAMATH_CALUDE_field_trip_solution_l2820_282072

/-- Represents the field trip problem -/
structure FieldTrip where
  students : ℕ
  small_bus_seats : ℕ
  large_bus_seats : ℕ
  small_bus_cost : ℕ
  large_bus_cost : ℕ
  total_buses : ℕ

/-- The specific field trip instance from the problem -/
def school_trip : FieldTrip :=
  { students := 245
  , small_bus_seats := 35
  , large_bus_seats := 45
  , small_bus_cost := 320
  , large_bus_cost := 380
  , total_buses := 6 }

/-- Calculates the total cost of a rental plan -/
def rental_cost (ft : FieldTrip) (small_buses : ℕ) : ℕ :=
  small_buses * ft.small_bus_cost + (ft.total_buses - small_buses) * ft.large_bus_cost

/-- Theorem stating the correct number of students and optimal rental plan -/
theorem field_trip_solution (ft : FieldTrip) : 
  ft.students = 245 ∧ 
  (∀ n : ℕ, n ≤ ft.total_buses → rental_cost ft 2 ≤ rental_cost ft n) :=
by
  sorry

#check field_trip_solution school_trip

end NUMINAMATH_CALUDE_field_trip_solution_l2820_282072


namespace NUMINAMATH_CALUDE_downward_parabola_properties_l2820_282097

/-- A parabola that opens downwards and passes through specific points -/
structure DownwardParabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  a_nonzero : a ≠ 0
  a_negative : a < 0
  m_range : -2 < m ∧ m < -1
  point_a : a + b + c = 0
  point_b : a * m^2 + b * m + c = 0

/-- Theorem stating properties of the downward parabola -/
theorem downward_parabola_properties (p : DownwardParabola) :
  p.a * p.b * p.c > 0 ∧
  p.a - p.b + p.c > 0 ∧
  p.a * (p.m + 1) - p.b + p.c > 0 := by
  sorry

end NUMINAMATH_CALUDE_downward_parabola_properties_l2820_282097


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l2820_282052

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := Set V
def Plane (V : Type*) [NormedAddCommGroup V] := Set V

-- Define parallel and perpendicular relations
def parallel (l₁ l₂ : Line V) : Prop := sorry
def perpendicular (l : Line V) (p : Plane V) : Prop := sorry

-- Theorem statement
theorem line_parallel_perpendicular 
  (a b : Line V) (α : Plane V) 
  (h₁ : a ≠ b) 
  (h₂ : parallel a b) 
  (h₃ : perpendicular a α) : 
  perpendicular b α := 
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l2820_282052


namespace NUMINAMATH_CALUDE_nursery_school_students_nursery_school_students_is_50_l2820_282017

/-- The number of students in a nursery school satisfying specific age distribution conditions -/
theorem nursery_school_students : ℕ :=
  let S : ℕ := 50  -- Total number of students
  let four_and_older : ℕ := S / 10  -- Students 4 years old or older
  let younger_than_three : ℕ := 20  -- Students younger than 3 years old
  let not_between_three_and_four : ℕ := 25  -- Students not between 3 and 4 years old
  have h1 : four_and_older = S / 10 := by sorry
  have h2 : younger_than_three = 20 := by sorry
  have h3 : not_between_three_and_four = 25 := by sorry
  have h4 : four_and_older + younger_than_three = not_between_three_and_four := by sorry
  S

/-- Proof that the number of students in the nursery school is 50 -/
theorem nursery_school_students_is_50 : nursery_school_students = 50 := by sorry

end NUMINAMATH_CALUDE_nursery_school_students_nursery_school_students_is_50_l2820_282017


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l2820_282028

/-- The surface area of a cuboid given its length, width, and height. -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The height of a cuboid with surface area 2400 cm², length 15 cm, and width 10 cm is 42 cm. -/
theorem cuboid_height_calculation (sa l w h : ℝ) 
  (h_sa : sa = 2400)
  (h_l : l = 15)
  (h_w : w = 10)
  (h_surface_area : surfaceArea l w h = sa) : h = 42 := by
  sorry

#check cuboid_height_calculation

end NUMINAMATH_CALUDE_cuboid_height_calculation_l2820_282028


namespace NUMINAMATH_CALUDE_tank_capacity_l2820_282069

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 8 →
  final_fraction = 7/8 →
  ∃ (total_capacity : Rat),
    initial_fraction * total_capacity + added_amount = final_fraction * total_capacity ∧
    total_capacity = 64 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2820_282069


namespace NUMINAMATH_CALUDE_locus_is_ray_l2820_282015

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-2,0) and N(2,0) are fixed points -/
def locus_of_P (P : ℝ × ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  Real.sqrt ((P.1 + 2)^2 + P.2^2) - Real.sqrt ((P.1 - 2)^2 + P.2^2) = 4

/-- The ray starting from the midpoint of MN and extending to the right -/
def ray_from_midpoint (P : ℝ × ℝ) : Prop :=
  P.1 ≥ 0 ∧ P.2 = 0

theorem locus_is_ray :
  ∀ P, locus_of_P P ↔ ray_from_midpoint P :=
sorry

end NUMINAMATH_CALUDE_locus_is_ray_l2820_282015


namespace NUMINAMATH_CALUDE_divide_friends_among_teams_l2820_282000

theorem divide_friends_among_teams (n : ℕ) (k : ℕ) : 
  n = 8 ∧ k = 3 →
  (k^n : ℕ) - k * ((k-1)^n : ℕ) + (k * (k-1) * (k-2) * 1^n) / 2 = 5796 :=
by sorry

end NUMINAMATH_CALUDE_divide_friends_among_teams_l2820_282000


namespace NUMINAMATH_CALUDE_ladies_work_theorem_l2820_282011

/-- Given a group of ladies who can complete a piece of work in 12 days,
    prove that twice the number of ladies will complete half of the work in 3 days. -/
theorem ladies_work_theorem (L : ℕ) (time : ℝ) (work : ℝ) : 
  (L > 0) →  -- Ensure there's at least one lady
  (time = 12) →  -- The original time to complete the work
  (work > 0) →  -- Ensure there's some work to be done
  (2 * L) * (time / 4) = work / 2 := by
  sorry

end NUMINAMATH_CALUDE_ladies_work_theorem_l2820_282011


namespace NUMINAMATH_CALUDE_divisor_pairing_l2820_282055

theorem divisor_pairing (n : ℕ+) (h : ¬ ∃ m : ℕ, n = m ^ 2) :
  ∃ f : {d : ℕ // d ∣ n} → {d : ℕ // d ∣ n},
    ∀ d : {d : ℕ // d ∣ n}, 
      (f (f d) = d) ∧ 
      ((d.val ∣ (f d).val) ∨ ((f d).val ∣ d.val)) :=
sorry

end NUMINAMATH_CALUDE_divisor_pairing_l2820_282055


namespace NUMINAMATH_CALUDE_cloud9_diving_total_money_l2820_282016

/-- The total money taken by Cloud 9 Diving Company -/
theorem cloud9_diving_total_money (individual_bookings group_bookings returned : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : returned = 1600) :
  individual_bookings + group_bookings - returned = 26400 := by
  sorry

end NUMINAMATH_CALUDE_cloud9_diving_total_money_l2820_282016


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2820_282073

/-- A function that checks if a number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Bool :=
  n % 2 ≠ 0

/-- A function that checks if a number is a valid first digit (non-zero even number) -/
def isValidFirstDigit (d : ℕ) : Bool :=
  d ≠ 0 ∧ isEven d ∧ d < 10

/-- A function that checks if a number is a valid second digit (odd number) -/
def isValidSecondDigit (d : ℕ) : Bool :=
  isOdd d ∧ d < 10

/-- A function that checks if two digits add up to an even number -/
def sumIsEven (d1 d2 : ℕ) : Bool :=
  isEven (d1 + d2)

/-- A function that checks if all digits in a four-digit number are different -/
def allDigitsDifferent (n : ℕ) : Bool :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- The main theorem stating that the count of valid four-digit numbers is 480 -/
theorem count_valid_numbers : 
  (Finset.filter (fun n : ℕ => 
    n ≥ 1000 ∧ n < 10000 ∧
    isValidFirstDigit (n / 1000) ∧
    isValidSecondDigit ((n / 100) % 10) ∧
    sumIsEven ((n / 10) % 10) (n % 10) ∧
    allDigitsDifferent n
  ) (Finset.range 10000)).card = 480 := by
  sorry


end NUMINAMATH_CALUDE_count_valid_numbers_l2820_282073


namespace NUMINAMATH_CALUDE_work_completion_time_l2820_282005

/-- Represents the work rate of one person per hour -/
structure WorkRate where
  man : ℝ
  woman : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours_per_day : ℝ
  days : ℝ

def total_work (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  (scenario.men * rate.man + scenario.women * rate.woman) * scenario.hours_per_day * scenario.days

theorem work_completion_time 
  (rate : WorkRate)
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario) :
  scenario1.men = 1 →
  scenario1.women = 3 →
  scenario1.hours_per_day = 7 →
  scenario1.days = 5 →
  scenario2.men = 4 →
  scenario2.women = 4 →
  scenario2.hours_per_day = 3 →
  scenario3.men = 7 →
  scenario3.women = 0 →
  scenario3.hours_per_day = 4 →
  scenario3.days = 5.000000000000001 →
  total_work rate scenario1 = total_work rate scenario3 →
  scenario2.days = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2820_282005


namespace NUMINAMATH_CALUDE_ngon_triangle_partition_l2820_282082

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A structure representing an n-gon -/
structure Ngon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (is_convex : sorry)  -- Additional property to ensure the n-gon is convex

/-- 
Given an n-gon and three vertex indices, return the lengths of the three parts 
of the boundary divided by these vertices
-/
def boundary_parts (n : ℕ) (poly : Ngon n) (i j k : Fin n) : ℝ × ℝ × ℝ := sorry

/-- The main theorem -/
theorem ngon_triangle_partition (n : ℕ) (h : n ≥ 3 ∧ n ≠ 4) :
  ∀ (poly : Ngon n), ∃ (i j k : Fin n),
    let (a, b, c) := boundary_parts n poly i j k
    can_form_triangle a b c :=
  sorry

end NUMINAMATH_CALUDE_ngon_triangle_partition_l2820_282082


namespace NUMINAMATH_CALUDE_atBatsAgainstLeft_is_180_l2820_282050

/-- Represents the batting statistics of a baseball player -/
structure BattingStats where
  totalAtBats : ℕ
  totalHits : ℕ
  avgAgainstLeft : ℚ
  avgAgainstRight : ℚ

/-- Calculates the number of at-bats against left-handed pitchers -/
def atBatsAgainstLeft (stats : BattingStats) : ℕ :=
  sorry

/-- Theorem stating that the number of at-bats against left-handed pitchers is 180 -/
theorem atBatsAgainstLeft_is_180 (stats : BattingStats) 
  (h1 : stats.totalAtBats = 600)
  (h2 : stats.totalHits = 192)
  (h3 : stats.avgAgainstLeft = 1/4)
  (h4 : stats.avgAgainstRight = 7/20)
  (h5 : (stats.totalHits : ℚ) / stats.totalAtBats = 8/25) :
  atBatsAgainstLeft stats = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_atBatsAgainstLeft_is_180_l2820_282050


namespace NUMINAMATH_CALUDE_notebooks_distribution_l2820_282048

/-- 
Given a class where:
- The total number of notebooks distributed is 512
- Each child initially receives a number of notebooks equal to 1/8 of the total number of children
Prove that if the number of children is halved, each child would receive 16 notebooks.
-/
theorem notebooks_distribution (C : ℕ) (h1 : C > 0) : 
  (C * (C / 8) = 512) → ((512 / (C / 2)) = 16) :=
by sorry

end NUMINAMATH_CALUDE_notebooks_distribution_l2820_282048


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2820_282099

/-- The quadratic equation ax^2 - x + 1 = 0 has real roots if and only if a ≤ 1/4 and a ≠ 0 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - x + 1 = 0) ↔ (a ≤ 1/4 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2820_282099


namespace NUMINAMATH_CALUDE_mp_eq_nq_l2820_282040

/-- Two circles in a plane -/
structure TwoCircles where
  c1 : Set (ℝ × ℝ)
  c2 : Set (ℝ × ℝ)

/-- Points on the circles -/
structure CirclePoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_A_intersect : A ∈ tc.c1 ∩ tc.c2
  h_B_intersect : B ∈ tc.c1 ∩ tc.c2
  h_M_on_c1 : M ∈ tc.c1
  h_N_on_c2 : N ∈ tc.c2
  h_P_on_c1 : P ∈ tc.c1
  h_Q_on_c2 : Q ∈ tc.c2

/-- AM is tangent to c2 at A -/
def is_tangent_AM (tc : TwoCircles) (cp : CirclePoints tc) : Prop := sorry

/-- AN is tangent to c1 at A -/
def is_tangent_AN (tc : TwoCircles) (cp : CirclePoints tc) : Prop := sorry

/-- B, M, and P are collinear -/
def collinear_BMP (cp : CirclePoints tc) : Prop := sorry

/-- B, N, and Q are collinear -/
def collinear_BNQ (cp : CirclePoints tc) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem mp_eq_nq (tc : TwoCircles) (cp : CirclePoints tc)
    (h_AM_tangent : is_tangent_AM tc cp)
    (h_AN_tangent : is_tangent_AN tc cp)
    (h_BMP_collinear : collinear_BMP cp)
    (h_BNQ_collinear : collinear_BNQ cp) :
    distance cp.M cp.P = distance cp.N cp.Q := by sorry

end NUMINAMATH_CALUDE_mp_eq_nq_l2820_282040


namespace NUMINAMATH_CALUDE_max_sum_constrained_length_l2820_282087

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

/-- The theorem states that given the conditions, the maximum value of x + 3y is 49156 -/
theorem max_sum_constrained_length (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_length_sum : length x + length y ≤ 16) :
  x + 3 * y ≤ 49156 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constrained_length_l2820_282087


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2820_282062

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + m*y - 6 = 0 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2820_282062


namespace NUMINAMATH_CALUDE_fred_basketball_games_l2820_282012

/-- The number of basketball games Fred went to last year -/
def last_year_games : ℕ := 36

/-- The number of games less Fred went to this year compared to last year -/
def games_difference : ℕ := 11

/-- The number of basketball games Fred went to this year -/
def this_year_games : ℕ := last_year_games - games_difference

theorem fred_basketball_games : this_year_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_basketball_games_l2820_282012


namespace NUMINAMATH_CALUDE_m_range_when_only_one_proposition_true_l2820_282047

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15

theorem m_range_when_only_one_proposition_true :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m) →
  1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_m_range_when_only_one_proposition_true_l2820_282047


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2820_282064

-- Define the sets A and B
def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | x^2 - 2*x < 3 }

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2820_282064


namespace NUMINAMATH_CALUDE_detergent_amount_in_altered_solution_l2820_282083

/-- The ratio of bleach to detergent to water in a solution -/
structure SolutionRatio :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

/-- The amount of each component in a solution -/
structure SolutionAmount :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

def original_ratio : SolutionRatio :=
  { bleach := 2, detergent := 25, water := 100 }

def altered_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := r.detergent,
    water := 2 * r.water }

def water_amount : ℚ := 300

theorem detergent_amount_in_altered_solution :
  ∀ (r : SolutionRatio) (w : ℚ),
  r = original_ratio →
  w = water_amount →
  ∃ (a : SolutionAmount),
    a.water = w ∧
    a.detergent = 37.5 ∧
    a.bleach / a.detergent = (altered_ratio r).bleach / (altered_ratio r).detergent ∧
    a.detergent / a.water = (altered_ratio r).detergent / (altered_ratio r).water :=
by sorry

end NUMINAMATH_CALUDE_detergent_amount_in_altered_solution_l2820_282083


namespace NUMINAMATH_CALUDE_problem_1_l2820_282068

theorem problem_1 : 4 * Real.sin (π / 3) + (1 / 3)⁻¹ + |-2| - Real.sqrt 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2820_282068


namespace NUMINAMATH_CALUDE_expression_signs_l2820_282010

theorem expression_signs :
  (10 - 3 * Real.sqrt 11 > 0) ∧
  (3 * Real.sqrt 11 - 10 < 0) ∧
  (18 - 5 * Real.sqrt 13 < 0) ∧
  (51 - 10 * Real.sqrt 26 > 0) ∧
  (10 * Real.sqrt 26 - 51 < 0) := by
  sorry

end NUMINAMATH_CALUDE_expression_signs_l2820_282010


namespace NUMINAMATH_CALUDE_cafeteria_choices_theorem_l2820_282090

/-- Represents the number of ways to choose foods in a cafeteria -/
def cafeteriaChoices (n : ℕ) : ℕ :=
  n + 1

/-- Theorem stating that the number of ways to choose n foods in the cafeteria is n + 1 -/
theorem cafeteria_choices_theorem (n : ℕ) : 
  cafeteriaChoices n = n + 1 := by
  sorry

/-- Apples are taken in groups of 3 -/
def appleGroup : ℕ := 3

/-- Yogurts are taken in pairs -/
def yogurtPair : ℕ := 2

/-- Maximum number of bread pieces allowed -/
def maxBread : ℕ := 2

/-- Maximum number of cereal bowls allowed -/
def maxCereal : ℕ := 1

end NUMINAMATH_CALUDE_cafeteria_choices_theorem_l2820_282090


namespace NUMINAMATH_CALUDE_gum_distribution_l2820_282019

/-- Given the number of gum pieces for each person and the total number of people,
    calculate the number of gum pieces each person will receive after equal distribution. -/
def distribute_gum (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) (num_people : ℕ) : ℕ :=
  (john_gum + cole_gum + aubrey_gum) / num_people

/-- Theorem stating that when 54 pieces of gum, 45 pieces of gum, and 0 pieces of gum
    are combined and divided equally among 3 people, each person will receive 33 pieces of gum. -/
theorem gum_distribution :
  distribute_gum 54 45 0 3 = 33 := by
  sorry

#eval distribute_gum 54 45 0 3

end NUMINAMATH_CALUDE_gum_distribution_l2820_282019


namespace NUMINAMATH_CALUDE_train_speed_problem_l2820_282021

/-- Proves that given two trains of equal length 70 meters, where one train travels at 50 km/hr
    and passes the other train in 36 seconds, the speed of the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 70 →
  faster_speed = 50 →
  passing_time = 36 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    train_length * 2 = (faster_speed - slower_speed) * passing_time * (1000 / 3600) ∧
    slower_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2820_282021


namespace NUMINAMATH_CALUDE_arlene_hike_distance_l2820_282056

/-- Calculates the total distance hiked given the hiking time and average pace. -/
def total_distance (time : ℝ) (pace : ℝ) : ℝ := time * pace

/-- Proves that Arlene hiked 24 miles on Saturday. -/
theorem arlene_hike_distance :
  let time : ℝ := 6 -- hours
  let pace : ℝ := 4 -- miles per hour
  total_distance time pace = 24 := by
  sorry

end NUMINAMATH_CALUDE_arlene_hike_distance_l2820_282056


namespace NUMINAMATH_CALUDE_product_base5_digit_sum_l2820_282058

/-- Converts a base-5 number represented as a list of digits to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5, returning a list of digits --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec toDigits (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else toDigits (m / 5) ((m % 5) :: acc)
    toDigits n []

/-- Sums the digits of a number represented as a list --/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

theorem product_base5_digit_sum (n1 n2 : List Nat) :
  sumDigits (base10ToBase5 (base5ToBase10 n1 * base5ToBase10 n2)) = 8 :=
sorry

end NUMINAMATH_CALUDE_product_base5_digit_sum_l2820_282058


namespace NUMINAMATH_CALUDE_fraction_equality_l2820_282088

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 25)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2820_282088


namespace NUMINAMATH_CALUDE_pet_owners_problem_l2820_282095

theorem pet_owners_problem (total_pet_owners : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (cats_dogs_snakes : ℕ) (total_snakes : ℕ)
  (h1 : total_pet_owners = 79)
  (h2 : only_dogs = 15)
  (h3 : only_cats = 10)
  (h4 : cats_dogs_snakes = 3)
  (h5 : total_snakes = 49) :
  total_pet_owners - only_dogs - only_cats - cats_dogs_snakes - (total_snakes - cats_dogs_snakes) = 5 :=
by sorry

end NUMINAMATH_CALUDE_pet_owners_problem_l2820_282095


namespace NUMINAMATH_CALUDE_unattainable_value_l2820_282085

theorem unattainable_value (x : ℝ) (y : ℝ) (h : x ≠ -4/3) : 
  y = (1 - x) / (3 * x + 4) → y ≠ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_value_l2820_282085


namespace NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l2820_282003

/-- Represents Lloyd's work information --/
structure WorkInfo where
  regularHours : Float
  regularRate : Float
  totalHours : Float
  totalEarnings : Float

/-- Calculates the multiple of regular rate for excess hours --/
def excessRateMultiple (info : WorkInfo) : Float :=
  let regularEarnings := info.regularHours * info.regularRate
  let excessHours := info.totalHours - info.regularHours
  let excessEarnings := info.totalEarnings - regularEarnings
  let excessRate := excessEarnings / excessHours
  excessRate / info.regularRate

/-- Theorem stating that the multiple of regular rate for excess hours is 1.5 --/
theorem lloyd_excess_rate_multiple :
  let lloyd : WorkInfo := {
    regularHours := 7.5,
    regularRate := 3.5,
    totalHours := 10.5,
    totalEarnings := 42
  }
  excessRateMultiple lloyd = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l2820_282003


namespace NUMINAMATH_CALUDE_stamp_problem_solution_l2820_282026

/-- Represents the number of stamps of each type -/
structure StampCounts where
  twopenny : ℕ
  penny : ℕ
  twohalfpenny : ℕ

/-- Calculates the total value of stamps in pence -/
def total_value (s : StampCounts) : ℕ :=
  2 * s.twopenny + s.penny + (5 * s.twohalfpenny) / 2

/-- Checks if the number of penny stamps is six times the number of twopenny stamps -/
def penny_constraint (s : StampCounts) : Prop :=
  s.penny = 6 * s.twopenny

/-- The main theorem stating the unique solution to the stamp problem -/
theorem stamp_problem_solution :
  ∃! s : StampCounts,
    total_value s = 60 ∧
    penny_constraint s ∧
    s.twopenny = 5 ∧
    s.penny = 30 ∧
    s.twohalfpenny = 8 := by
  sorry

end NUMINAMATH_CALUDE_stamp_problem_solution_l2820_282026


namespace NUMINAMATH_CALUDE_no_two_digit_product_concatenation_l2820_282018

theorem no_two_digit_product_concatenation : ¬∃ (a b c d : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_product_concatenation_l2820_282018


namespace NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l2820_282038

/-- Represents a playing card color -/
inductive CardColor
| Red
| Black

/-- Represents a box where cards are placed -/
inductive Box
| A
| B
| C

/-- Represents the state of the card distribution -/
structure CardDistribution where
  cardsInA : ℕ
  redInB : ℕ
  blackInB : ℕ
  redInC : ℕ
  blackInC : ℕ

/-- The card distribution process -/
def distributeCards : CardDistribution → CardColor → CardDistribution
| d, CardColor.Red => { d with
    cardsInA := d.cardsInA + 1,
    redInB := d.redInB + 1 }
| d, CardColor.Black => { d with
    cardsInA := d.cardsInA + 1,
    blackInC := d.blackInC + 1 }

/-- The theorem stating that the number of red cards in B equals the number of black cards in C -/
theorem red_in_B_equals_black_in_C (finalDist : CardDistribution)
  (h : finalDist.cardsInA = 52) :
  finalDist.redInB = finalDist.blackInC := by
  sorry

end NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l2820_282038


namespace NUMINAMATH_CALUDE_certain_number_is_thirty_l2820_282089

theorem certain_number_is_thirty : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → ∃ (b : ℕ), k * n = b^2 → k ≥ 30) ∧
  (∃ (b : ℕ), 30 * n = b^2) ∧
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_thirty_l2820_282089


namespace NUMINAMATH_CALUDE_remainder_8347_div_9_l2820_282054

theorem remainder_8347_div_9 : 8347 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8347_div_9_l2820_282054


namespace NUMINAMATH_CALUDE_stationery_prices_l2820_282025

-- Define the variables
variable (x : ℝ) -- Price of one notebook
variable (y : ℝ) -- Price of one pen

-- Define the theorem
theorem stationery_prices : 
  (3 * x + 5 * y = 30) ∧ 
  (30 - (3 * x + 5 * y + 2 * y) = -0.4) ∧ 
  (30 - (3 * x + 5 * y + 2 * x) = -2) → 
  (x = 3.6 ∧ y = 2.8) :=
by sorry

end NUMINAMATH_CALUDE_stationery_prices_l2820_282025


namespace NUMINAMATH_CALUDE_parabola_equation_l2820_282029

/-- A parabola with vertex at the origin, coordinate axes as axes of symmetry, 
    and passing through (-2, 3) has equation x^2 = (4/3)y or y^2 = -(9/2)x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3)) →  -- vertex at origin and passes through (-2, 3)
  (∀ x y, p (x, y) ↔ p (-x, y)) →  -- symmetry about y-axis
  (∀ x y, p (x, y) ↔ p (x, -y)) →  -- symmetry about x-axis
  (∃ a b : ℝ, (∀ x y, p (x, y) ↔ x^2 = a*y) ∨ (∀ x y, p (x, y) ↔ y^2 = b*x)) →
  (∀ x y, p (x, y) ↔ x^2 = (4/3)*y ∨ y^2 = -(9/2)*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2820_282029


namespace NUMINAMATH_CALUDE_solve_age_problem_l2820_282023

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 12)

theorem solve_age_problem :
  ∀ a b : ℕ, age_problem a b → b = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_age_problem_l2820_282023


namespace NUMINAMATH_CALUDE_simplify_expression_l2820_282031

theorem simplify_expression (x y : ℝ) (h : x ≠ y) :
  (x - y)^3 / (x - y)^2 * (y - x) = -(x - y)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2820_282031


namespace NUMINAMATH_CALUDE_daniela_shopping_cost_l2820_282071

-- Define the original prices and discount rates
def shoe_price : ℝ := 50
def dress_price : ℝ := 100
def shoe_discount : ℝ := 0.4
def dress_discount : ℝ := 0.2

-- Define the number of items purchased
def num_shoes : ℕ := 2
def num_dresses : ℕ := 1

-- Calculate the discounted prices
def discounted_shoe_price : ℝ := shoe_price * (1 - shoe_discount)
def discounted_dress_price : ℝ := dress_price * (1 - dress_discount)

-- Calculate the total cost
def total_cost : ℝ := num_shoes * discounted_shoe_price + num_dresses * discounted_dress_price

-- Theorem statement
theorem daniela_shopping_cost : total_cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_daniela_shopping_cost_l2820_282071
