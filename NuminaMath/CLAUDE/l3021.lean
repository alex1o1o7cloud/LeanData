import Mathlib

namespace NUMINAMATH_CALUDE_at_least_one_equation_has_distinct_roots_l3021_302128

theorem at_least_one_equation_has_distinct_roots (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_distinct_roots_l3021_302128


namespace NUMINAMATH_CALUDE_xiaoming_estimate_larger_l3021_302161

/-- Rounds a number up to the nearest ten -/
def roundUp (n : ℤ) : ℤ := sorry

/-- Rounds a number down to the nearest ten -/
def roundDown (n : ℤ) : ℤ := sorry

theorem xiaoming_estimate_larger (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  roundUp x - roundDown y > x - y := by sorry

end NUMINAMATH_CALUDE_xiaoming_estimate_larger_l3021_302161


namespace NUMINAMATH_CALUDE_complex_multiplication_imaginary_zero_l3021_302144

theorem complex_multiplication_imaginary_zero (a : ℝ) :
  (Complex.I * (a + Complex.I) + (1 : ℂ) * (a + Complex.I)).im = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_imaginary_zero_l3021_302144


namespace NUMINAMATH_CALUDE_geometric_seq_arithmetic_property_l3021_302142

/-- Given a geometric sequence with common ratio q, prove that if the sums S_m, S_n, and S_l
    form an arithmetic sequence, then for any natural number k, the terms a_{m+k}, a_{n+k},
    and a_{l+k} also form an arithmetic sequence. -/
theorem geometric_seq_arithmetic_property
  (a : ℝ) (q : ℝ) (m n l k : ℕ) :
  let a_seq : ℕ → ℝ := λ i => a * q ^ (i - 1)
  let S : ℕ → ℝ := λ i => if q = 1 then a * i else a * (1 - q^i) / (1 - q)
  (2 * S n = S m + S l) →
  2 * a_seq (n + k) = a_seq (m + k) + a_seq (l + k) :=
by sorry

end NUMINAMATH_CALUDE_geometric_seq_arithmetic_property_l3021_302142


namespace NUMINAMATH_CALUDE_bowl_glass_pairing_l3021_302167

theorem bowl_glass_pairing (n : ℕ) (h : n = 5) : 
  (n : ℕ) * (n - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowl_glass_pairing_l3021_302167


namespace NUMINAMATH_CALUDE_subtraction_result_l3021_302125

theorem subtraction_result : 3.79 - 2.15 = 1.64 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l3021_302125


namespace NUMINAMATH_CALUDE_platform_length_l3021_302196

/-- The length of a platform given train speed and passing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) : 
  train_speed = 54 → platform_time = 16 → man_time = 10 → 
  (train_speed * 5 / 18) * (platform_time - man_time) = 90 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3021_302196


namespace NUMINAMATH_CALUDE_tan_symmetry_cos_squared_plus_sin_min_value_l3021_302189

-- Define the tangent function
noncomputable def tan (x : ℝ) := Real.tan x

-- Define the cosine function
noncomputable def cos (x : ℝ) := Real.cos x

-- Define the sine function
noncomputable def sin (x : ℝ) := Real.sin x

-- Proposition ①
theorem tan_symmetry (k : ℤ) :
  ∀ x : ℝ, tan (k * π / 2 + x) = -tan (k * π / 2 - x) :=
sorry

-- Proposition ④
theorem cos_squared_plus_sin_min_value :
  ∃ x : ℝ, ∀ y : ℝ, cos y ^ 2 + sin y ≥ cos x ^ 2 + sin x ∧ cos x ^ 2 + sin x = -1 :=
sorry

end NUMINAMATH_CALUDE_tan_symmetry_cos_squared_plus_sin_min_value_l3021_302189


namespace NUMINAMATH_CALUDE_mans_age_twice_students_l3021_302186

/-- Proves that it takes 2 years for a man's age to be twice his student's age -/
theorem mans_age_twice_students (student_age : ℕ) (age_difference : ℕ) : 
  student_age = 24 → age_difference = 26 → 
  ∃ (years : ℕ), (student_age + years) * 2 = (student_age + age_difference + years) ∧ years = 2 :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_students_l3021_302186


namespace NUMINAMATH_CALUDE_power_sum_2001_l3021_302120

theorem power_sum_2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) :
  x^2001 + y^2001 = 2^2001 ∨ x^2001 + y^2001 = -(2^2001) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_2001_l3021_302120


namespace NUMINAMATH_CALUDE_constant_function_theorem_l3021_302193

theorem constant_function_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f x * (deriv f x) = 0) → ∃ C, ∀ x, f x = C := by
  sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l3021_302193


namespace NUMINAMATH_CALUDE_M_mod_500_l3021_302199

/-- A sequence of positive integers whose binary representations have exactly 6 ones -/
def T : ℕ → ℕ :=
  sorry

/-- The 500th term in the sequence T -/
def M : ℕ :=
  T 500

theorem M_mod_500 : M % 500 = 198 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_500_l3021_302199


namespace NUMINAMATH_CALUDE_magnitude_squared_l3021_302158

theorem magnitude_squared (w : ℂ) (h : Complex.abs w = 11) : (2 * Complex.abs w)^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_squared_l3021_302158


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3021_302178

/-- Given a positive real number a and a function f(x) = ax^2 + 2ax + 1,
    if f(m) < 0 for some real m, then f(m+2) > 1 -/
theorem quadratic_function_property (a : ℝ) (m : ℝ) (h_a : a > 0) :
  let f := λ x : ℝ ↦ a * x^2 + 2 * a * x + 1
  f m < 0 → f (m + 2) > 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3021_302178


namespace NUMINAMATH_CALUDE_eulers_formula_simply_connected_l3021_302153

/-- A simply connected polyhedron -/
structure SimplyConnectedPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  is_simply_connected : Bool

/-- Euler's formula for simply connected polyhedra -/
theorem eulers_formula_simply_connected (p : SimplyConnectedPolyhedron) 
  (h : p.is_simply_connected = true) : 
  p.faces - p.edges + p.vertices = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_simply_connected_l3021_302153


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3021_302156

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3021_302156


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l3021_302184

theorem sequence_fifth_term (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l3021_302184


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l3021_302181

/-- The maximum number of parts that three planes can divide space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem: The maximum number of parts that three planes can divide space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l3021_302181


namespace NUMINAMATH_CALUDE_no_fixed_extreme_points_l3021_302183

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 3

/-- The derivative of f -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem: There do not exist real numbers a and b such that f has two distinct extreme points that are also fixed points -/
theorem no_fixed_extreme_points :
  ¬ ∃ (a b x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    f' a b x₁ = 0 ∧
    f' a b x₂ = 0 ∧
    f a b x₁ = x₁ ∧
    f a b x₂ = x₂ := by
  sorry


end NUMINAMATH_CALUDE_no_fixed_extreme_points_l3021_302183


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_l3021_302129

theorem reciprocal_of_negative_five :
  ∃ x : ℚ, x * (-5) = 1 ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_l3021_302129


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3021_302170

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 2 + a 6 = 3) →
  (a 6 + a 10 = 12) →
  (a 8 + a 12 = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3021_302170


namespace NUMINAMATH_CALUDE_max_third_side_length_l3021_302140

theorem max_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 10) :
  ∃ (x : ℕ), x ≤ 14 ∧
  ∀ (y : ℕ), (y : ℝ) < a + b ∧ (y : ℝ) > |a - b| → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l3021_302140


namespace NUMINAMATH_CALUDE_income_ratio_l3021_302118

/-- Represents the financial data of a person --/
structure PersonFinance where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup --/
def problem_setup (p1 p2 : PersonFinance) : Prop :=
  p1.income = 4000 ∧
  p1.savings = 1600 ∧
  p2.savings = 1600 ∧
  3 * p2.expenditure = 2 * p1.expenditure ∧
  p1.savings = p1.income - p1.expenditure ∧
  p2.savings = p2.income - p2.expenditure

/-- The theorem to be proved --/
theorem income_ratio (p1 p2 : PersonFinance) :
  problem_setup p1 p2 → 5 * p2.income = 4 * p1.income :=
by
  sorry


end NUMINAMATH_CALUDE_income_ratio_l3021_302118


namespace NUMINAMATH_CALUDE_quadratic_inequality_min_value_l3021_302165

/-- Given a quadratic inequality with an empty solution set and a condition on its coefficients,
    prove that a certain expression has a minimum value of 4. -/
theorem quadratic_inequality_min_value (a b c : ℝ) :
  (∀ x, (1/a) * x^2 + b*x + c ≥ 0) →
  a * b > 1 →
  ∀ T, T = 1/(2*(a*b - 1)) + (a*(b + 2*c))/(a*b - 1) →
  T ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_min_value_l3021_302165


namespace NUMINAMATH_CALUDE_f_bounds_l3021_302130

/-- The maximum number of elements from Example 1 -/
def f (n : ℕ) : ℕ := sorry

/-- Proof that f(n) satisfies the given inequality -/
theorem f_bounds (n : ℕ) (hn : n > 0) : 
  (1 / 6 : ℚ) * (n^2 - 4*n : ℚ) ≤ (f n : ℚ) ∧ (f n : ℚ) ≤ (1 / 6 : ℚ) * (n^2 - n : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_f_bounds_l3021_302130


namespace NUMINAMATH_CALUDE_roy_pens_count_l3021_302136

def blue_pens : ℕ := 5

def black_pens : ℕ := 3 * blue_pens

def red_pens : ℕ := 2 * black_pens - 4

def total_pens : ℕ := blue_pens + black_pens + red_pens

theorem roy_pens_count : total_pens = 46 := by
  sorry

end NUMINAMATH_CALUDE_roy_pens_count_l3021_302136


namespace NUMINAMATH_CALUDE_square_perimeters_l3021_302146

theorem square_perimeters (a b : ℝ) (h1 : a = 3 * b) 
  (h2 : a ^ 2 + b ^ 2 = 130) (h3 : a ^ 2 - b ^ 2 = 108) : 
  4 * a + 4 * b = 16 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_square_perimeters_l3021_302146


namespace NUMINAMATH_CALUDE_computer_price_increase_l3021_302135

theorem computer_price_increase (x : ℝ) (h : x + 0.3 * x = 351) : x + 351 = 621 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3021_302135


namespace NUMINAMATH_CALUDE_jean_buys_two_cards_per_grandchild_l3021_302115

/-- Represents the scenario of Jean's gift-giving to her grandchildren --/
structure GiftGiving where
  num_grandchildren : ℕ
  amount_per_card : ℕ
  total_amount : ℕ

/-- Calculates the number of cards bought for each grandchild --/
def cards_per_grandchild (g : GiftGiving) : ℕ :=
  (g.total_amount / g.amount_per_card) / g.num_grandchildren

/-- Theorem stating that Jean buys 2 cards for each grandchild --/
theorem jean_buys_two_cards_per_grandchild :
  ∀ (g : GiftGiving),
    g.num_grandchildren = 3 →
    g.amount_per_card = 80 →
    g.total_amount = 480 →
    cards_per_grandchild g = 2 := by
  sorry

end NUMINAMATH_CALUDE_jean_buys_two_cards_per_grandchild_l3021_302115


namespace NUMINAMATH_CALUDE_max_square_plots_l3021_302100

/-- Represents the dimensions of the field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing -/
def availableFence : ℕ := 2250

/-- Calculates the number of square plots given the side length -/
def numPlots (dimensions : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (dimensions.width / sideLength) * (dimensions.length / sideLength)

/-- Calculates the required fencing for a given configuration -/
def requiredFencing (dimensions : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (dimensions.width / sideLength - 1) * dimensions.length +
  (dimensions.length / sideLength - 1) * dimensions.width

/-- Checks if a given side length is valid for the field dimensions -/
def isValidSideLength (dimensions : FieldDimensions) (sideLength : ℕ) : Prop :=
  sideLength > 0 ∧
  dimensions.width % sideLength = 0 ∧
  dimensions.length % sideLength = 0

theorem max_square_plots (dimensions : FieldDimensions)
    (h1 : dimensions.width = 30)
    (h2 : dimensions.length = 45) :
    (∃ (sideLength : ℕ),
      isValidSideLength dimensions sideLength ∧
      requiredFencing dimensions sideLength ≤ availableFence ∧
      numPlots dimensions sideLength = 150 ∧
      (∀ (s : ℕ), isValidSideLength dimensions s →
        requiredFencing dimensions s ≤ availableFence →
        numPlots dimensions s ≤ 150)) :=
  sorry

end NUMINAMATH_CALUDE_max_square_plots_l3021_302100


namespace NUMINAMATH_CALUDE_rachel_painting_time_l3021_302191

def minutes_per_day_first_6 : ℕ := 100
def days_first_period : ℕ := 6
def minutes_per_day_next_2 : ℕ := 120
def days_second_period : ℕ := 2
def target_average : ℕ := 110
def total_days : ℕ := 10

theorem rachel_painting_time :
  (minutes_per_day_first_6 * days_first_period +
   minutes_per_day_next_2 * days_second_period +
   (target_average * total_days - 
    (minutes_per_day_first_6 * days_first_period +
     minutes_per_day_next_2 * days_second_period))) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_rachel_painting_time_l3021_302191


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3021_302180

theorem quadratic_factorization_sum (d e f : ℝ) : 
  (∀ x, (x + d) * (x + e) = x^2 + 11*x + 24) →
  (∀ x, (x + e) * (x - f) = x^2 + 9*x - 36) →
  d + e + f = 14 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3021_302180


namespace NUMINAMATH_CALUDE_expression_evaluation_l3021_302177

theorem expression_evaluation : 
  (((15^15 / 15^14)^3 * 8^3) / 4^6 : ℚ) = 1728000 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3021_302177


namespace NUMINAMATH_CALUDE_probability_three_defective_shipment_l3021_302109

/-- The probability of selecting three defective smartphones from a shipment -/
def probability_three_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total *
  ((defective - 1) : ℚ) / (total - 1) *
  ((defective - 2) : ℚ) / (total - 2)

/-- Theorem stating the probability of selecting three defective smartphones
    from a shipment of 400 smartphones, of which 150 are defective -/
theorem probability_three_defective_shipment :
  probability_three_defective 400 150 = 150 / 400 * 149 / 399 * 148 / 398 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_defective_shipment_l3021_302109


namespace NUMINAMATH_CALUDE_initial_amount_proof_l3021_302166

/-- Proves that given specific conditions, the initial amount of money is 30000 --/
theorem initial_amount_proof (rate : ℝ) (time : ℝ) (difference : ℝ) : 
  rate = 0.20 →
  time = 2 →
  difference = 723.0000000000146 →
  (fun P : ℝ => P * ((1 + rate / 2) ^ (2 * time) - (1 + rate) ^ time)) difference = 30000 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l3021_302166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3021_302175

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_8th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 23)
  (h_6th : a 6 = 47) :
  a 8 = 71 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3021_302175


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3021_302172

theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 250 →
  bridge_length = 150 →
  time = 41.142857142857146 →
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  let speed_kmh := speed_ms * 3.6
  ⌊speed_kmh⌋ = 35 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3021_302172


namespace NUMINAMATH_CALUDE_second_month_sale_is_6927_l3021_302111

/-- Calculates the sale amount for the second month given the sales of other months and the average sale --/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the second month is 6927 given the problem conditions --/
theorem second_month_sale_is_6927 :
  calculate_second_month_sale 6435 6855 7230 6562 6191 6700 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_is_6927_l3021_302111


namespace NUMINAMATH_CALUDE_square_area_proof_l3021_302114

theorem square_area_proof (s : ℝ) (h1 : s = 4) : 
  (s^2 + s) - (4 * s) = 4 → s^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l3021_302114


namespace NUMINAMATH_CALUDE_min_slide_time_l3021_302113

/-- A vertical circle fixed to a horizontal line -/
structure VerticalCircle where
  center : ℝ × ℝ
  radius : ℝ
  is_vertical : center.2 = radius

/-- A point outside and above the circle -/
structure OutsidePoint (C : VerticalCircle) where
  coords : ℝ × ℝ
  is_outside : (coords.1 - C.center.1)^2 + (coords.2 - C.center.2)^2 > C.radius^2
  is_above : coords.2 > C.center.2 + C.radius

/-- A point on the circle -/
def CirclePoint (C : VerticalCircle) := { p : ℝ × ℝ // (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2 }

/-- The time function for a particle to slide down from P to Q under gravity -/
noncomputable def slide_time (C : VerticalCircle) (P : OutsidePoint C) (Q : CirclePoint C) : ℝ := sorry

/-- The lowest point on the circle -/
def lowest_point (C : VerticalCircle) : CirclePoint C :=
  ⟨(C.center.1, C.center.2 - C.radius), sorry⟩

/-- Theorem: The point Q that minimizes the slide time is the lowest point on the circle -/
theorem min_slide_time (C : VerticalCircle) (P : OutsidePoint C) :
  ∀ Q : CirclePoint C, slide_time C P Q ≥ slide_time C P (lowest_point C) :=
sorry

end NUMINAMATH_CALUDE_min_slide_time_l3021_302113


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3021_302108

/-- A quadratic function with vertex (2, 0) passing through (0, -50) has a = -12.5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2, 0) = (2, a * 2^2 + b * 2 + c) →
  (0, -50) = (0, a * 0^2 + b * 0 + c) →
  a = -12.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3021_302108


namespace NUMINAMATH_CALUDE_bus_capacity_proof_l3021_302127

theorem bus_capacity_proof (C : ℚ) 
  (h1 : (3 / 4) * C + (4 / 5) * C = 310) : C = 200 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_proof_l3021_302127


namespace NUMINAMATH_CALUDE_sanya_washing_days_l3021_302176

/-- Represents the number of days needed to wash all towels -/
def days_needed (towels_per_wash : ℕ) (hours_per_day : ℕ) (total_towels : ℕ) : ℕ :=
  (total_towels + towels_per_wash * hours_per_day - 1) / (towels_per_wash * hours_per_day)

/-- Theorem stating that Sanya needs 7 days to wash all towels -/
theorem sanya_washing_days :
  days_needed 7 2 98 = 7 :=
by sorry

#eval days_needed 7 2 98

end NUMINAMATH_CALUDE_sanya_washing_days_l3021_302176


namespace NUMINAMATH_CALUDE_bridge_length_l3021_302171

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 245 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3021_302171


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3021_302104

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 8*y - x*y = 0) : 
  x*y ≥ 64 ∧ x + y ≥ 18 := by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3021_302104


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3021_302197

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3021_302197


namespace NUMINAMATH_CALUDE_jess_walk_distance_l3021_302149

/-- The number of blocks Jess must walk to complete her errands and arrive at work -/
def remaining_blocks (post_office store gallery library work already_walked : ℕ) : ℕ :=
  post_office + store + gallery + library + work - already_walked

/-- Theorem stating the number of blocks Jess must walk given the problem conditions -/
theorem jess_walk_distance :
  remaining_blocks 24 18 15 14 22 9 = 84 := by
  sorry

end NUMINAMATH_CALUDE_jess_walk_distance_l3021_302149


namespace NUMINAMATH_CALUDE_car_speed_l3021_302139

/-- Proves that given the conditions, the speed of the car is 50 miles per hour -/
theorem car_speed (gasoline_consumption : Real) (tank_capacity : Real) (travel_time : Real) (gasoline_used_fraction : Real) :
  gasoline_consumption = 1 / 30 →
  tank_capacity = 10 →
  travel_time = 5 →
  gasoline_used_fraction = 0.8333333333333334 →
  (tank_capacity * gasoline_used_fraction) / (travel_time * gasoline_consumption) = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3021_302139


namespace NUMINAMATH_CALUDE_impossible_to_turn_all_lamps_off_l3021_302179

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On
| Off

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the chessboard state -/
def ChessboardState := Position → LampState

/-- Represents the allowed operations on the chessboard -/
inductive Operation
| InvertRow (row : Fin 8)
| InvertColumn (col : Fin 8)
| InvertDiagonal (d : ℤ) -- d represents the diagonal offset

/-- The initial state of the chessboard -/
def initialState : ChessboardState :=
  fun pos => if pos.row = 0 && pos.col = 3 then LampState.Off else LampState.On

/-- Apply an operation to the chessboard state -/
def applyOperation (state : ChessboardState) (op : Operation) : ChessboardState :=
  sorry

/-- Check if all lamps are off -/
def allLampsOff (state : ChessboardState) : Prop :=
  ∀ pos, state pos = LampState.Off

/-- The main theorem to be proved -/
theorem impossible_to_turn_all_lamps_off :
  ¬∃ (ops : List Operation), allLampsOff (ops.foldl applyOperation initialState) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_turn_all_lamps_off_l3021_302179


namespace NUMINAMATH_CALUDE_kristy_work_hours_l3021_302173

/-- Proves that given the conditions of Kristy's salary structure and earnings,
    she worked 160 hours in the month. -/
theorem kristy_work_hours :
  let hourly_rate : ℝ := 7.5
  let commission_rate : ℝ := 0.16
  let total_sales : ℝ := 25000
  let insurance_amount : ℝ := 260
  let insurance_rate : ℝ := 0.05
  let commission : ℝ := commission_rate * total_sales
  let total_earnings : ℝ := insurance_amount / insurance_rate
  let hours_worked : ℝ := (total_earnings - commission) / hourly_rate
  hours_worked = 160 := by sorry

end NUMINAMATH_CALUDE_kristy_work_hours_l3021_302173


namespace NUMINAMATH_CALUDE_arithmetic_seq_problem_l3021_302105

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given conditions for the arithmetic sequence -/
structure ArithSeqConditions (a : ℕ → ℚ) : Prop :=
  (is_arith : ArithmeticSequence a)
  (prod_eq : a 5 * a 7 = 6)
  (sum_eq : a 2 + a 10 = 5)

/-- Theorem statement -/
theorem arithmetic_seq_problem (a : ℕ → ℚ) (h : ArithSeqConditions a) :
  (a 10 - a 6 = 2) ∨ (a 10 - a 6 = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_problem_l3021_302105


namespace NUMINAMATH_CALUDE_special_square_area_l3021_302164

/-- A square with two points on its sides satisfying certain distance conditions -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The distance of point E from vertex A on side AD
  AE : ℝ
  -- The distance of point F from vertex B on side BC
  BF : ℝ
  -- Conditions
  h1 : 0 < side
  h2 : 0 ≤ AE ∧ AE ≤ side
  h3 : 0 ≤ BF ∧ BF ≤ side
  h4 : Real.sqrt ((side - AE)^2 + BF^2) = 20
  h5 : Real.sqrt (AE^2 + (side - BF)^2) = 20
  h6 : Real.sqrt ((side - AE)^2 + (side - BF)^2) = 40

/-- The area of the SpecialSquare is 6400 -/
theorem special_square_area (s : SpecialSquare) : s.side^2 = 6400 := by
  sorry

#check special_square_area

end NUMINAMATH_CALUDE_special_square_area_l3021_302164


namespace NUMINAMATH_CALUDE_problem_statement_l3021_302152

theorem problem_statement (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3021_302152


namespace NUMINAMATH_CALUDE_dog_walking_distance_l3021_302112

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog1_daily_miles : ℝ) (days_per_week : ℕ) :
  total_weekly_miles = 70 ∧ 
  dog1_daily_miles = 2 ∧ 
  days_per_week = 7 →
  (total_weekly_miles - dog1_daily_miles * days_per_week) / days_per_week = 8 :=
by sorry

end NUMINAMATH_CALUDE_dog_walking_distance_l3021_302112


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l3021_302168

theorem inequality_system_solution_range (x m : ℝ) : 
  ((x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l3021_302168


namespace NUMINAMATH_CALUDE_area_triangle_ADG_l3021_302159

/-- Regular octagon with side length 3 -/
structure RegularOctagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Triangle ADG in the regular octagon -/
def TriangleADG (octagon : RegularOctagon) : Set (Fin 3 → ℝ × ℝ) :=
  sorry

/-- Area of a triangle -/
def triangleArea (triangle : Set (Fin 3 → ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: Area of triangle ADG in a regular octagon with side length 3 -/
theorem area_triangle_ADG (octagon : RegularOctagon) :
  triangleArea (TriangleADG octagon) = (27 - 9 * Real.sqrt 2 + 9 * Real.sqrt (2 - 2 * Real.sqrt 2)) / (2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_ADG_l3021_302159


namespace NUMINAMATH_CALUDE_seeds_not_grown_marge_garden_problem_l3021_302107

theorem seeds_not_grown (total_seeds : ℕ) (final_plants : ℕ) : ℕ :=
  let uneaten_plants := (final_plants * 3) / 2
  let grown_plants := (uneaten_plants * 3) / 2
  total_seeds - grown_plants

theorem marge_garden_problem : seeds_not_grown 23 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seeds_not_grown_marge_garden_problem_l3021_302107


namespace NUMINAMATH_CALUDE_family_gathering_l3021_302126

theorem family_gathering (P : ℕ) : 
  P / 2 = P - 10 → 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_family_gathering_l3021_302126


namespace NUMINAMATH_CALUDE_acid_solution_replacement_l3021_302182

/-- Proves that the fraction of original 50% acid solution replaced with 20% acid solution to obtain a 35% acid solution is 0.5 -/
theorem acid_solution_replacement (V : ℝ) (h : V > 0) :
  let original_concentration : ℝ := 0.5
  let replacement_concentration : ℝ := 0.2
  let final_concentration : ℝ := 0.35
  let x : ℝ := (original_concentration - final_concentration) / (original_concentration - replacement_concentration)
  x = 0.5 := by sorry

end NUMINAMATH_CALUDE_acid_solution_replacement_l3021_302182


namespace NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l3021_302110

/-- Represents the problem of Patty paying her siblings with cookies to do her chores. -/
structure CookieChoresProblem where
  money_available : ℕ  -- Amount of money Patty has in dollars
  pack_cost : ℕ  -- Cost of one pack of cookies in dollars
  cookies_per_pack : ℕ  -- Number of cookies in each pack
  cookies_per_chore : ℕ  -- Number of cookies given for each chore
  chores_per_week : ℕ  -- Number of chores each kid has per week

/-- Calculates the number of weeks Patty can avoid doing chores. -/
def weeks_without_chores (problem : CookieChoresProblem) : ℕ :=
  let packs_bought := problem.money_available / problem.pack_cost
  let total_cookies := packs_bought * problem.cookies_per_pack
  let cookies_per_week := problem.cookies_per_chore * problem.chores_per_week
  total_cookies / cookies_per_week

/-- Theorem stating that given the problem conditions, Patty can avoid doing chores for 10 weeks. -/
theorem patty_avoids_chores_for_ten_weeks (problem : CookieChoresProblem) 
  (h1 : problem.money_available = 15)
  (h2 : problem.pack_cost = 3)
  (h3 : problem.cookies_per_pack = 24)
  (h4 : problem.cookies_per_chore = 3)
  (h5 : problem.chores_per_week = 4) :
  weeks_without_chores problem = 10 := by
  sorry

#eval weeks_without_chores { 
  money_available := 15, 
  pack_cost := 3, 
  cookies_per_pack := 24, 
  cookies_per_chore := 3, 
  chores_per_week := 4 
}

end NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l3021_302110


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3021_302145

theorem fraction_evaluation : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3021_302145


namespace NUMINAMATH_CALUDE_vacation_cost_l3021_302198

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 5 = 50) → C = 375 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l3021_302198


namespace NUMINAMATH_CALUDE_bottle_cap_configurations_l3021_302195

theorem bottle_cap_configurations : ∃ (n m : ℕ), n ≠ m ∧ n > 0 ∧ m > 0 ∧ 3 ∣ n ∧ 4 ∣ n ∧ 3 ∣ m ∧ 4 ∣ m :=
by sorry

end NUMINAMATH_CALUDE_bottle_cap_configurations_l3021_302195


namespace NUMINAMATH_CALUDE_container_production_l3021_302122

/-- Container production problem -/
theorem container_production
  (december : ℕ)
  (h_dec_nov : december = (110 * november) / 100)
  (h_nov_oct : november = (105 * october) / 100)
  (h_oct_sep : october = (120 * september) / 100)
  (h_december : december = 11088) :
  november = 10080 ∧ october = 9600 ∧ september = 8000 := by
  sorry

end NUMINAMATH_CALUDE_container_production_l3021_302122


namespace NUMINAMATH_CALUDE_bonus_calculation_l3021_302192

/-- A quadratic function f(x) = kx^2 + b that satisfies certain conditions -/
def f (k b x : ℝ) : ℝ := k * x^2 + b

theorem bonus_calculation (k b : ℝ) (h1 : k > 0) 
  (h2 : f k b 10 = 0) (h3 : f k b 20 = 2) : f k b 200 = 266 := by
  sorry

end NUMINAMATH_CALUDE_bonus_calculation_l3021_302192


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l3021_302137

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 45 = 0 ∧
  (n + 1) % 60 = 0 ∧
  (n + 1) % 72 = 0 ∧
  (n + 1) % 81 = 0 ∧
  (n + 1) % 100 = 0 ∧
  (n + 1) % 120 = 0

theorem smallest_divisible_number :
  is_divisible_by_all 16199 ∧
  ∀ m : ℕ, m < 16199 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l3021_302137


namespace NUMINAMATH_CALUDE_binary_remainder_is_two_l3021_302132

/-- Given a binary number represented as a list of bits (least significant bit first),
    calculate the remainder when divided by 4. -/
def binary_remainder_mod_4 (bits : List Bool) : Nat :=
  match bits with
  | [] => 0
  | [b₀] => if b₀ then 1 else 0
  | b₀ :: b₁ :: _ => (if b₁ then 2 else 0) + (if b₀ then 1 else 0)

/-- The binary representation of 100101110010₂ (least significant bit first) -/
def binary_number : List Bool :=
  [false, true, false, false, true, true, true, false, true, false, false, true]

/-- Theorem stating that the remainder when 100101110010₂ is divided by 4 is 2 -/
theorem binary_remainder_is_two :
  binary_remainder_mod_4 binary_number = 2 := by
  sorry


end NUMINAMATH_CALUDE_binary_remainder_is_two_l3021_302132


namespace NUMINAMATH_CALUDE_zongzi_prices_l3021_302174

-- Define variables
variable (x : ℝ) -- Purchase price of egg yolk zongzi
variable (y : ℝ) -- Purchase price of red bean zongzi
variable (m : ℝ) -- Selling price of egg yolk zongzi

-- Define conditions
def first_purchase : Prop := 60 * x + 90 * y = 4800
def second_purchase : Prop := 40 * x + 80 * y = 3600
def initial_sales : Prop := m = 70 ∧ (70 - 50) * 20 = 400
def sales_change : Prop := ∀ p, (p - 50) * (20 + 5 * (70 - p)) = 220 → p = 52

-- Theorem statement
theorem zongzi_prices :
  first_purchase x y ∧ second_purchase x y ∧ initial_sales m ∧ sales_change →
  x = 50 ∧ y = 20 ∧ m = 52 := by
  sorry

end NUMINAMATH_CALUDE_zongzi_prices_l3021_302174


namespace NUMINAMATH_CALUDE_simplify_expression_l3021_302138

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3021_302138


namespace NUMINAMATH_CALUDE_cakes_served_yesterday_l3021_302154

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := total_cakes - (lunch_cakes + dinner_cakes)

theorem cakes_served_yesterday : yesterday_cakes = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_yesterday_l3021_302154


namespace NUMINAMATH_CALUDE_exponent_equivalence_l3021_302131

theorem exponent_equivalence (y : ℕ) (some_exponent : ℕ) 
  (h1 : 9^y = 3^some_exponent) (h2 : y = 8) : some_exponent = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equivalence_l3021_302131


namespace NUMINAMATH_CALUDE_safe_descent_possible_l3021_302117

/-- Represents the cliff and rope setup --/
structure CliffSetup where
  cliff_height : ℝ
  rope_length : ℝ
  branch_height : ℝ

/-- Defines a safe descent --/
def safe_descent (setup : CliffSetup) : Prop :=
  setup.cliff_height > setup.rope_length ∧
  setup.branch_height < setup.cliff_height ∧
  setup.branch_height > 0 ∧
  setup.rope_length ≥ setup.cliff_height - setup.branch_height + setup.branch_height / 2

/-- Theorem stating that a safe descent is possible given the specific measurements --/
theorem safe_descent_possible : 
  ∃ (setup : CliffSetup), 
    setup.cliff_height = 100 ∧ 
    setup.rope_length = 75 ∧ 
    setup.branch_height = 50 ∧ 
    safe_descent setup := by
  sorry


end NUMINAMATH_CALUDE_safe_descent_possible_l3021_302117


namespace NUMINAMATH_CALUDE_sticker_distribution_l3021_302157

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 8 identical stickers among 4 distinct sheets of paper -/
theorem sticker_distribution : distribute 8 4 = 165 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3021_302157


namespace NUMINAMATH_CALUDE_probability_5_odd_in_8_rolls_l3021_302133

def roll_die_8_times : ℕ := 8
def fair_die_sides : ℕ := 6
def odd_outcomes : ℕ := 3
def target_odd_rolls : ℕ := 5

theorem probability_5_odd_in_8_rolls : 
  (Nat.choose roll_die_8_times target_odd_rolls * (odd_outcomes ^ target_odd_rolls) * ((fair_die_sides - odd_outcomes) ^ (roll_die_8_times - target_odd_rolls))) / (fair_die_sides ^ roll_die_8_times) = 7 / 32 :=
sorry

end NUMINAMATH_CALUDE_probability_5_odd_in_8_rolls_l3021_302133


namespace NUMINAMATH_CALUDE_fifth_score_calculation_l3021_302141

theorem fifth_score_calculation (score1 score2 score3 score4 score5 : ℝ) :
  score1 = 85 ∧ score2 = 90 ∧ score3 = 87 ∧ score4 = 92 →
  (score1 + score2 + score3 + score4 + score5) / 5 = 89 →
  score5 = 91 := by
sorry

end NUMINAMATH_CALUDE_fifth_score_calculation_l3021_302141


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l3021_302160

theorem cost_price_per_meter (cloth_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) 
  (h1 : cloth_length = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 25) : 
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 80 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l3021_302160


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3021_302151

theorem geometric_progression_fourth_term 
  (a : ℝ → ℝ) -- Sequence of real numbers
  (h1 : a 1 = 2^(1/2)) -- First term
  (h2 : a 2 = 2^(1/3)) -- Second term
  (h3 : a 3 = 2^(1/6)) -- Third term
  (h_geom : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1) -- Geometric progression condition
  : a 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3021_302151


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l3021_302123

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_area_inequality (t : Triangle) :
  area t / (t.a * t.b + t.b * t.c + t.c * t.a) ≤ 1 / (4 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l3021_302123


namespace NUMINAMATH_CALUDE_inverse_proportionality_l3021_302106

/-- Given that α is inversely proportional to β, prove that if α = 4 when β = 9, 
    then α = -1/2 when β = -72 -/
theorem inverse_proportionality (α β : ℝ) (h : ∃ k, ∀ x y, x * y = k → α = x ∧ β = y) :
  (α = 4 ∧ β = 9) → (β = -72 → α = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l3021_302106


namespace NUMINAMATH_CALUDE_blueprint_to_actual_length_l3021_302163

/-- Given a blueprint scale and a length on the blueprint, calculates the actual length in meters. -/
def actual_length (scale : ℚ) (blueprint_length : ℚ) : ℚ :=
  blueprint_length * scale / 100

/-- Proves that for a blueprint scale of 1:50 and a line segment of 10 cm on the blueprint,
    the actual length is 5 m. -/
theorem blueprint_to_actual_length :
  let scale : ℚ := 50
  let blueprint_length : ℚ := 10
  actual_length scale blueprint_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_blueprint_to_actual_length_l3021_302163


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3021_302194

/-- Given a point A(a, 4) in the second quadrant and a vertical line m with x = 2,
    the point symmetric to A with respect to m has coordinates (4-a, 4). -/
theorem symmetric_point_coordinates (a : ℝ) (h1 : a < 0) :
  let A : ℝ × ℝ := (a, 4)
  let m : Set (ℝ × ℝ) := {p | p.1 = 2}
  let symmetric_point := (4 - a, 4)
  symmetric_point.1 = 4 - a ∧ symmetric_point.2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3021_302194


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l3021_302187

/-- Proves that the amount lent is 1000 given the specified conditions -/
theorem loan_amount_calculation (P : ℝ) : 
  (P * 0.115 * 3 - P * 0.10 * 3 = 45) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l3021_302187


namespace NUMINAMATH_CALUDE_composite_function_sum_l3021_302162

/-- Given a function f(x) = px + q where p and q are real numbers,
    if f(f(f(x))) = 8x + 21, then p + q = 5 -/
theorem composite_function_sum (p q : ℝ) :
  (∀ x, ∃ f : ℝ → ℝ, f x = p * x + q) →
  (∀ x, ∃ f : ℝ → ℝ, f (f (f x)) = 8 * x + 21) →
  p + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_sum_l3021_302162


namespace NUMINAMATH_CALUDE_sum_reciprocal_product_20_l3021_302147

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 6n - n^2,
    this function returns the sum of the first k terms of the sequence {1/(a_n * a_{n+1})} -/
def sum_reciprocal_product (k : ℕ) : ℚ :=
  let S : ℕ → ℚ := λ n => 6 * n - n^2
  let a : ℕ → ℚ := λ n => S n - S (n-1)
  let term : ℕ → ℚ := λ n => 1 / (a n * a (n+1))
  (Finset.range k).sum term

/-- The main theorem stating that the sum of the first 20 terms of the 
    sequence {1/(a_n * a_{n+1})} is equal to -4/35 -/
theorem sum_reciprocal_product_20 : sum_reciprocal_product 20 = -4/35 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_product_20_l3021_302147


namespace NUMINAMATH_CALUDE_lecture_scheduling_l3021_302119

theorem lecture_scheduling (n : ℕ) (h : n = 7) :
  let total_permutations := n.factorial
  let valid_orderings := total_permutations / 4
  valid_orderings = 1260 :=
by sorry

end NUMINAMATH_CALUDE_lecture_scheduling_l3021_302119


namespace NUMINAMATH_CALUDE_no_real_solutions_l3021_302190

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3*x + 8)^2 + 4 = -2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3021_302190


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3021_302148

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/12))^(1/4))^6 * (((a^16)^(1/4))^(1/12))^3 = a^3 := by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3021_302148


namespace NUMINAMATH_CALUDE_beeswax_number_l3021_302155

/-- Represents a mapping from characters to digits -/
def CodeMapping : Char → Nat
| 'A' => 1
| 'T' => 2
| 'Q' => 3
| 'B' => 4
| 'K' => 5
| 'X' => 6
| 'S' => 7
| 'W' => 8
| 'E' => 9
| 'P' => 0
| _ => 0

/-- Converts a string to a number using the code mapping -/
def stringToNumber (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + CodeMapping c) 0

/-- The subtraction equation given in the problem -/
axiom subtraction_equation :
  stringToNumber "EASEBSBSX" - stringToNumber "BPWWKSETQ" = stringToNumber "KPEPWEKKQ"

/-- The main theorem to prove -/
theorem beeswax_number : stringToNumber "BEESWAX" = 4997816 := by
  sorry


end NUMINAMATH_CALUDE_beeswax_number_l3021_302155


namespace NUMINAMATH_CALUDE_base3_102012_equals_302_l3021_302134

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_102012_equals_302 :
  base3_to_base10 [1, 0, 2, 0, 1, 2] = 302 := by
  sorry

end NUMINAMATH_CALUDE_base3_102012_equals_302_l3021_302134


namespace NUMINAMATH_CALUDE_odd_expressions_l3021_302143

theorem odd_expressions (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) : 
  Odd (5*p^2 + 2*q^2) ∧ Odd (p^2 + p*q + q^2) := by
  sorry

end NUMINAMATH_CALUDE_odd_expressions_l3021_302143


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3021_302185

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5)*x - k + 9 > 1) ↔ k > -1 ∧ k < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3021_302185


namespace NUMINAMATH_CALUDE_pizza_night_theorem_l3021_302116

/-- Pizza night problem -/
theorem pizza_night_theorem 
  (small_pizza_slices : Nat) 
  (medium_pizza_slices : Nat) 
  (large_pizza_slices : Nat)
  (phil_eaten : Nat)
  (andre_eaten : Nat)
  (phil_ratio : Nat)
  (andre_ratio : Nat)
  (h1 : small_pizza_slices = 8)
  (h2 : medium_pizza_slices = 10)
  (h3 : large_pizza_slices = 14)
  (h4 : phil_eaten = 10)
  (h5 : andre_eaten = 12)
  (h6 : phil_ratio = 3)
  (h7 : andre_ratio = 2) :
  let total_slices := small_pizza_slices + 2 * medium_pizza_slices + large_pizza_slices
  let eaten_slices := phil_eaten + andre_eaten
  let remaining_slices := total_slices - eaten_slices
  let total_ratio := phil_ratio + andre_ratio
  let phil_share := (phil_ratio * remaining_slices) / total_ratio
  let andre_share := (andre_ratio * remaining_slices) / total_ratio
  remaining_slices = 20 ∧ phil_share = 12 ∧ andre_share = 8 := by
  sorry

#check pizza_night_theorem

end NUMINAMATH_CALUDE_pizza_night_theorem_l3021_302116


namespace NUMINAMATH_CALUDE_thirteen_people_in_line_l3021_302188

/-- The number of people in line at an amusement park ride -/
def people_in_line (people_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  people_in_front + 1 + (position_from_back - 1)

/-- Theorem stating that there are 13 people in line given the conditions -/
theorem thirteen_people_in_line :
  people_in_line 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_people_in_line_l3021_302188


namespace NUMINAMATH_CALUDE_experiment_sequences_l3021_302103

/-- The number of procedures in the experiment -/
def total_procedures : ℕ := 6

/-- The number of ways to place procedure A (first or last) -/
def a_placements : ℕ := 2

/-- The number of distinct units to arrange (including BC as one unit) -/
def distinct_units : ℕ := 4

/-- The number of ways to arrange B and C within their unit -/
def bc_arrangements : ℕ := 2

/-- The total number of possible sequences for the experiment procedures -/
def total_sequences : ℕ := a_placements * (distinct_units.factorial) * bc_arrangements

theorem experiment_sequences :
  total_sequences = 96 :=
sorry

end NUMINAMATH_CALUDE_experiment_sequences_l3021_302103


namespace NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_n_between_sqrt_l3021_302101

-- Define the table as a function
def f (x : ℝ) : ℝ := x^2

-- Theorem 1
theorem sqrt_190_44 : Real.sqrt 190.44 = 13.8 ∨ Real.sqrt 190.44 = -13.8 := by sorry

-- Theorem 2
theorem sqrt_176_9_and_18769 :
  (abs (Real.sqrt 176.9 - 13.3) < 0.1) ∧ (Real.sqrt 18769 = 137) := by sorry

-- Theorem 3
theorem integer_n_between_sqrt :
  ∀ n : ℕ, (13.5 < Real.sqrt n) ∧ (Real.sqrt n < 13.6) → (n = 183 ∨ n = 184) := by sorry

end NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_n_between_sqrt_l3021_302101


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l3021_302124

theorem greatest_prime_factor_of_154 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 154 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 154 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l3021_302124


namespace NUMINAMATH_CALUDE_evaluate_expression_l3021_302150

theorem evaluate_expression : 8^3 + 4*(8^2) + 6*8 + 3 = 1000 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3021_302150


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3021_302169

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 2 = 4 →
  geometric_sequence (1 + a 3) (a 6) (4 + a 10) →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3021_302169


namespace NUMINAMATH_CALUDE_chef_potato_problem_chef_leftover_potatoes_l3021_302121

/-- A chef's potato problem -/
theorem chef_potato_problem (total_potatoes : ℕ) 
  (fries_needed : ℕ) (fries_per_potato : ℕ) 
  (cubes_needed : ℕ) (cubes_per_potato : ℕ) : ℕ :=
  let potatoes_for_fries := (fries_needed + fries_per_potato - 1) / fries_per_potato
  let potatoes_for_cubes := (cubes_needed + cubes_per_potato - 1) / cubes_per_potato
  let potatoes_used := potatoes_for_fries + potatoes_for_cubes
  total_potatoes - potatoes_used

/-- The chef will have 17 potatoes leftover -/
theorem chef_leftover_potatoes : 
  chef_potato_problem 30 200 25 50 10 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_problem_chef_leftover_potatoes_l3021_302121


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3021_302102

-- Define the time to fill 1/11 of the cistern
def time_for_one_eleventh : ℝ := 3

-- Define the function to calculate the time to fill the entire cistern
def time_for_full_cistern : ℝ := time_for_one_eleventh * 11

-- Theorem statement
theorem cistern_filling_time : time_for_full_cistern = 33 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3021_302102
