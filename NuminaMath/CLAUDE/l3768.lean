import Mathlib

namespace NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l3768_376856

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The GDP value in ten thousand yuan -/
def gdp : ℕ := 84300000

/-- The GDP expressed in scientific notation -/
def gdp_scientific : ScientificNotation where
  coefficient := 8.43
  exponent := 7
  is_valid := by sorry

/-- Theorem stating that the GDP value is correctly expressed in scientific notation -/
theorem gdp_scientific_notation_correct : 
  (gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent) = gdp := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l3768_376856


namespace NUMINAMATH_CALUDE_max_correct_guesses_proof_l3768_376893

/-- Represents the maximum number of guaranteed correct hat color guesses 
    for n wise men with k insane among them. -/
def max_correct_guesses (n k : ℕ) : ℕ := n - k - 1

/-- Theorem stating that the maximum number of guaranteed correct hat color guesses
    is equal to n - k - 1, where n is the total number of wise men and k is the
    number of insane wise men. -/
theorem max_correct_guesses_proof (n k : ℕ) (h1 : k < n) :
  max_correct_guesses n k = n - k - 1 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_guesses_proof_l3768_376893


namespace NUMINAMATH_CALUDE_radical_simplification_l3768_376881

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (20 * q^3) * Real.sqrt (12 * q^5) = 60 * q^4 * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l3768_376881


namespace NUMINAMATH_CALUDE_power_function_k_values_l3768_376870

/-- A function f(x) = ax^n is a power function if a ≠ 0 and n is a non-zero constant. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ n ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y = (k^2-k-5)x^2 is a power function, then k = 3 or k = -2 -/
theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^2) → k = 3 ∨ k = -2 := by
  sorry


end NUMINAMATH_CALUDE_power_function_k_values_l3768_376870


namespace NUMINAMATH_CALUDE_remainder_sum_l3768_376811

theorem remainder_sum (n : ℤ) : n % 18 = 11 → (n % 3 + n % 6 = 7) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3768_376811


namespace NUMINAMATH_CALUDE_alberts_earnings_increase_l3768_376886

theorem alberts_earnings_increase (E : ℝ) (P : ℝ) : 
  1.27 * E = 567 →
  E + P * E = 562.54 →
  P = 0.26 := by
sorry

end NUMINAMATH_CALUDE_alberts_earnings_increase_l3768_376886


namespace NUMINAMATH_CALUDE_circle_equation_l3768_376841

-- Define the circle C
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 4}

-- Define the tangent line
def TangentLine := {(x, y) : ℝ × ℝ | 3*x + 4*y + 4 = 0}

-- Theorem statement
theorem circle_equation (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ (p : ℝ × ℝ), p ∈ Circle a ∧ p ∈ TangentLine) : 
  Circle a = Circle 2 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3768_376841


namespace NUMINAMATH_CALUDE_equation_solution_l3768_376897

theorem equation_solution :
  ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3768_376897


namespace NUMINAMATH_CALUDE_number_of_boys_l3768_376888

def total_students : ℕ := 1150

def is_valid_distribution (boys : ℕ) : Prop :=
  let girls := (boys * 100) / total_students
  boys + girls = total_students

theorem number_of_boys : ∃ (boys : ℕ), boys = 1058 ∧ is_valid_distribution boys := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l3768_376888


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l3768_376817

theorem quadratic_sum_zero (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : a * x₂^2 + b * x₂ + c = 0)
  (s₁ : ℝ := x₁^2005 + x₂^2005)
  (s₂ : ℝ := x₁^2004 + x₂^2004)
  (s₃ : ℝ := x₁^2003 + x₂^2003) :
  a * s₁ + b * s₂ + c * s₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l3768_376817


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3768_376890

/-- The number of handshakes in a basketball game -/
def total_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let inter_team := team_size * team_size * (num_teams - 1) / 2
  let intra_team := num_teams * (team_size * (team_size - 1) / 2)
  let with_referees := num_teams * team_size * num_referees
  inter_team + intra_team + with_referees

/-- Theorem stating the total number of handshakes in the specific basketball game scenario -/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l3768_376890


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l3768_376836

/-- The absorption rate of fiber for koalas -/
def koala_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
def fiber_absorbed : ℝ := 12

/-- Theorem: If a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day, 
    then the total amount of fiber it ate that day was 40 ounces. -/
theorem koala_fiber_consumption :
  fiber_absorbed = koala_absorption_rate * 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l3768_376836


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3768_376887

/-- For a positive arithmetic sequence with a_3 = 2, the common difference d is in the range [0, 1). -/
theorem arithmetic_sequence_common_difference_range 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_positive : ∀ n, a n > 0)
  (h_a3 : a 3 = 2) :
  ∃ d, (∀ n, a (n + 1) = a n + d) ∧ 0 ≤ d ∧ d < 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3768_376887


namespace NUMINAMATH_CALUDE_slope_implies_y_coordinate_l3768_376803

/-- Given two points A and B in a coordinate plane, if the slope of the line through A and B is 1/3, then the y-coordinate of B is 12. -/
theorem slope_implies_y_coordinate
  (xA yA xB : ℝ)
  (h1 : xA = -3)
  (h2 : yA = 9)
  (h3 : xB = 6) :
  (yB - yA) / (xB - xA) = 1/3 → yB = 12 :=
by sorry

end NUMINAMATH_CALUDE_slope_implies_y_coordinate_l3768_376803


namespace NUMINAMATH_CALUDE_exam_attendance_l3768_376801

theorem exam_attendance (passed_percentage : ℝ) (failed_count : ℕ) : 
  passed_percentage = 35 → 
  failed_count = 351 → 
  (failed_count : ℝ) / (100 - passed_percentage) * 100 = 540 := by
sorry

end NUMINAMATH_CALUDE_exam_attendance_l3768_376801


namespace NUMINAMATH_CALUDE_jar_servings_calculation_l3768_376839

/-- Represents the contents and serving sizes of peanut butter and jelly in a jar -/
structure JarContents where
  pb_amount : ℚ  -- Amount of peanut butter in tablespoons
  jelly_amount : ℚ  -- Amount of jelly in tablespoons
  pb_serving : ℚ  -- Size of one peanut butter serving in tablespoons
  jelly_serving : ℚ  -- Size of one jelly serving in tablespoons

/-- Calculates the number of servings for peanut butter and jelly -/
def calculate_servings (jar : JarContents) : ℚ × ℚ :=
  (jar.pb_amount / jar.pb_serving, jar.jelly_amount / jar.jelly_serving)

/-- Theorem stating the correct number of servings for the given jar contents -/
theorem jar_servings_calculation (jar : JarContents)
  (h1 : jar.pb_amount = 35 + 2/3)
  (h2 : jar.jelly_amount = 18 + 1/2)
  (h3 : jar.pb_serving = 2 + 1/6)
  (h4 : jar.jelly_serving = 1) :
  calculate_servings jar = (16 + 18/39, 18 + 1/2) := by
  sorry

#eval calculate_servings {
  pb_amount := 35 + 2/3,
  jelly_amount := 18 + 1/2,
  pb_serving := 2 + 1/6,
  jelly_serving := 1
}

end NUMINAMATH_CALUDE_jar_servings_calculation_l3768_376839


namespace NUMINAMATH_CALUDE_vegetable_count_l3768_376896

/-- The total number of vegetables in the supermarket -/
def total_vegetables (cucumbers carrots tomatoes radishes : ℕ) : ℕ :=
  cucumbers + carrots + tomatoes + radishes

/-- Theorem stating the total number of vegetables given the conditions -/
theorem vegetable_count :
  ∀ (cucumbers carrots tomatoes radishes : ℕ),
    cucumbers = 58 →
    cucumbers = carrots + 24 →
    cucumbers = tomatoes - 49 →
    radishes = carrots →
    total_vegetables cucumbers carrots tomatoes radishes = 233 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_count_l3768_376896


namespace NUMINAMATH_CALUDE_units_digit_G_500_l3768_376837

/-- The function G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of G(500) is 0 -/
theorem units_digit_G_500 : unitsDigit (G 500) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_500_l3768_376837


namespace NUMINAMATH_CALUDE_population_in_scientific_notation_l3768_376861

/-- Represents the population in billions -/
def population_in_billions : ℝ := 1.412

/-- Converts billions to scientific notation -/
def billions_to_scientific (x : ℝ) : ℝ := x * 10^9

/-- Theorem stating that 1.412 billion people in scientific notation is 1.412 × 10^9 -/
theorem population_in_scientific_notation :
  billions_to_scientific population_in_billions = 1.412 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_population_in_scientific_notation_l3768_376861


namespace NUMINAMATH_CALUDE_gcd_binomial_integer_l3768_376878

theorem gcd_binomial_integer (m n : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) :
  ∃ k : ℤ, (Nat.gcd m n : ℚ) / n * (n.choose m : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_integer_l3768_376878


namespace NUMINAMATH_CALUDE_savings_difference_l3768_376847

/-- Represents the price of a book in dollars -/
def book_price : ℝ := 25

/-- Represents the discount percentage for Discount A -/
def discount_a_percentage : ℝ := 0.4

/-- Represents the fixed discount amount for Discount B in dollars -/
def discount_b_amount : ℝ := 5

/-- Calculates the total cost with Discount A -/
def total_cost_a : ℝ := book_price + (book_price * (1 - discount_a_percentage))

/-- Calculates the total cost with Discount B -/
def total_cost_b : ℝ := book_price + (book_price - discount_b_amount)

/-- Theorem stating the difference in savings between Discount A and Discount B -/
theorem savings_difference : total_cost_b - total_cost_a = 5 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l3768_376847


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3768_376892

theorem magnitude_of_complex_number (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = (1 + 2*i) / i →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3768_376892


namespace NUMINAMATH_CALUDE_brazil_nut_price_is_five_l3768_376833

/-- Represents the price of Brazil nuts per pound -/
def brazil_nut_price : ℝ := 5

/-- Represents the price of cashews per pound -/
def cashew_price : ℝ := 6.75

/-- Represents the total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 50

/-- Represents the selling price of the mixture per pound -/
def mixture_selling_price : ℝ := 5.70

/-- Represents the weight of cashews used in the mixture in pounds -/
def cashew_weight : ℝ := 20

/-- Theorem stating that the price of Brazil nuts is $5 per pound given the conditions -/
theorem brazil_nut_price_is_five :
  brazil_nut_price = 5 ∧
  cashew_price = 6.75 ∧
  total_mixture_weight = 50 ∧
  mixture_selling_price = 5.70 ∧
  cashew_weight = 20 →
  brazil_nut_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_brazil_nut_price_is_five_l3768_376833


namespace NUMINAMATH_CALUDE_soccer_team_average_goals_l3768_376876

/-- Calculates the average number of goals per game for a soccer team -/
def average_goals_per_game (slices_per_pizza : ℕ) (pizzas_bought : ℕ) (games_played : ℕ) : ℚ :=
  (slices_per_pizza * pizzas_bought : ℚ) / games_played

/-- Theorem: Given the conditions, the average number of goals per game is 9 -/
theorem soccer_team_average_goals :
  let slices_per_pizza : ℕ := 12
  let pizzas_bought : ℕ := 6
  let games_played : ℕ := 8
  average_goals_per_game slices_per_pizza pizzas_bought games_played = 9 := by
sorry

#eval average_goals_per_game 12 6 8

end NUMINAMATH_CALUDE_soccer_team_average_goals_l3768_376876


namespace NUMINAMATH_CALUDE_sum_removal_proof_l3768_376807

theorem sum_removal_proof : 
  let original_sum := (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13
  let removed_terms := 1 / 11 + 1 / 13
  original_sum - removed_terms = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_removal_proof_l3768_376807


namespace NUMINAMATH_CALUDE_sum_in_range_l3768_376820

theorem sum_in_range : ∃ (s : ℚ), 
  s = (1 + 3/8) + (4 + 1/3) + (6 + 2/21) ∧ 11 < s ∧ s < 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l3768_376820


namespace NUMINAMATH_CALUDE_sqrt_3_minus_1_over_2_less_than_half_l3768_376891

theorem sqrt_3_minus_1_over_2_less_than_half : (Real.sqrt 3 - 1) / 2 < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_1_over_2_less_than_half_l3768_376891


namespace NUMINAMATH_CALUDE_injective_function_inequality_l3768_376889

theorem injective_function_inequality (f : Set.Icc 0 1 → ℝ) 
  (h_inj : Function.Injective f) (h_sum : f 0 + f 1 = 1) :
  ∃ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ ∧ 2 * f x₁ < f x₂ + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_injective_function_inequality_l3768_376889


namespace NUMINAMATH_CALUDE_walking_speed_l3768_376832

/-- 
Given that:
- Jack's speed is (x^2 - 13x - 30) miles per hour
- Jill covers (x^2 - 6x - 91) miles in (x + 7) hours
- Jack and Jill walk at the same rate

Prove that their speed is 4 miles per hour
-/
theorem walking_speed (x : ℝ) 
  (h1 : x ≠ -7)  -- Assumption to avoid division by zero
  (h2 : x > 0)   -- Assumption for positive speed
  (h3 : (x^2 - 6*x - 91) / (x + 7) = x^2 - 13*x - 30) :  -- Jack and Jill walk at the same rate
  x^2 - 13*x - 30 = 4 := by sorry

end NUMINAMATH_CALUDE_walking_speed_l3768_376832


namespace NUMINAMATH_CALUDE_tricycle_count_l3768_376821

/-- Represents the number of tricycles in a group of children riding bicycles and tricycles -/
def num_tricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  total_wheels - 2 * total_children

/-- Theorem stating that given 7 children and 19 wheels, the number of tricycles is 5 -/
theorem tricycle_count : num_tricycles 7 19 = 5 := by
  sorry

#eval num_tricycles 7 19  -- Should output 5

end NUMINAMATH_CALUDE_tricycle_count_l3768_376821


namespace NUMINAMATH_CALUDE_spoons_to_knives_ratio_l3768_376846

/-- Given a silverware set where the number of spoons is three times the number of knives,
    and the number of knives is 6, prove that the ratio of spoons to knives is 3:1. -/
theorem spoons_to_knives_ratio (knives : ℕ) (spoons : ℕ) : 
  knives = 6 → spoons = 3 * knives → spoons / knives = 3 :=
by
  sorry

#check spoons_to_knives_ratio

end NUMINAMATH_CALUDE_spoons_to_knives_ratio_l3768_376846


namespace NUMINAMATH_CALUDE_intersection_and_angle_condition_l3768_376879

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y + 1 = 0
def l2 (x y : ℝ) : Prop := 5 * x - y - 1 = 0
def l3 (x y : ℝ) : Prop := 3 * x + 2 * y + 1 = 0

-- Define the result lines
def result1 (x y : ℝ) : Prop := x + 5 * y + 5 = 0
def result2 (x y : ℝ) : Prop := 5 * x - y - 1 = 0

-- Define the 45° angle condition
def angle_45_deg (m1 m2 : ℝ) : Prop := (m1 - m2) / (1 + m1 * m2) = 1 || (m1 - m2) / (1 + m1 * m2) = -1

-- Main theorem
theorem intersection_and_angle_condition :
  ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧
  (∃ (m : ℝ), (angle_45_deg m (-3/2)) ∧
    ((result1 x y ∧ m = -1/5) ∨ (result2 x y ∧ m = 5))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_angle_condition_l3768_376879


namespace NUMINAMATH_CALUDE_min_c_value_l3768_376885

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2021 ∧ p.2 = |p.1 - a| + |p.1 - b| + |p.1 - c|) :
  c ≥ 1011 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3768_376885


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l3768_376818

/-- The number of elements in the n-th row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_sum (n : ℕ) : ℕ :=
  (n * (pascal_row_elements 0 + pascal_row_elements (n - 1))) / 2

theorem pascal_triangle_30_rows_sum :
  pascal_triangle_sum 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l3768_376818


namespace NUMINAMATH_CALUDE_lucy_groceries_l3768_376805

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 2

/-- The number of packs of cake Lucy bought -/
def cake : ℕ := 12

/-- The total number of grocery packs Lucy bought -/
def total_groceries : ℕ := cookies + cake

theorem lucy_groceries : total_groceries = 14 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l3768_376805


namespace NUMINAMATH_CALUDE_not_juggling_sequence_l3768_376830

/-- Definition of the juggling sequence -/
def j : ℕ → ℕ
| 0 => 5
| 1 => 7
| 2 => 2
| n + 3 => j n

/-- Function f that calculates the time when a ball will be caught -/
def f (t : ℕ) : ℕ := t + j (t % 3)

/-- Theorem stating that 572 is not a juggling sequence -/
theorem not_juggling_sequence : ¬ (∀ n m : ℕ, n < 3 → m < 3 → n ≠ m → f n ≠ f m) := by
  sorry

end NUMINAMATH_CALUDE_not_juggling_sequence_l3768_376830


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3768_376860

def num_tan_chips : ℕ := 3
def num_pink_chips : ℕ := 2
def num_violet_chips : ℕ := 4
def total_chips : ℕ := num_tan_chips + num_pink_chips + num_violet_chips

theorem consecutive_color_draw_probability :
  (Nat.factorial num_tan_chips * Nat.factorial num_pink_chips * 
   Nat.factorial num_violet_chips * Nat.factorial 3) / 
  Nat.factorial total_chips = 1 / 210 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3768_376860


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l3768_376828

theorem quadratic_equation_real_root (p : ℝ) : 
  ((-p)^2 - 4 * (3*(p+2)) * (-(4*p+7))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l3768_376828


namespace NUMINAMATH_CALUDE_area_ratio_is_five_sevenths_l3768_376895

-- Define the points
variable (A B C D O P X Y : ℝ × ℝ)

-- Define the lengths
def AD : ℝ := 13
def AO : ℝ := 13
def OB : ℝ := 13
def BC : ℝ := 13
def AB : ℝ := 15
def DO : ℝ := 15
def OC : ℝ := 15

-- Define the conditions
axiom triangle_dao_isosceles : AO = AD
axiom triangle_aob_isosceles : AO = OB
axiom triangle_obc_isosceles : OB = BC
axiom p_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom op_perpendicular_ab : (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) = 0
axiom x_midpoint_ad : X = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
axiom y_midpoint_bc : Y = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the areas of trapezoids
def area_ABYX : ℝ := sorry
def area_XYCD : ℝ := sorry

-- State the theorem
theorem area_ratio_is_five_sevenths :
  area_ABYX / area_XYCD = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_area_ratio_is_five_sevenths_l3768_376895


namespace NUMINAMATH_CALUDE_range_of_a_l3768_376806

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x / (x - 1) < 1

def q (x a : ℝ) : Prop := (x + a) * (x - 3) > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ∈ Set.Iic (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3768_376806


namespace NUMINAMATH_CALUDE_midpoint_chain_l3768_376864

/-- Given a line segment AB with multiple midpoints, prove that AB = 64 when AG = 2 -/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 2) →        -- AG = 2
  (B - A = 64) :=      -- AB = 64
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l3768_376864


namespace NUMINAMATH_CALUDE_set_membership_l3768_376852

def M : Set ℤ := {a | ∃ b c : ℤ, a = b^2 - c^2}

theorem set_membership : (8 ∈ M) ∧ (9 ∈ M) ∧ (10 ∉ M) := by sorry

end NUMINAMATH_CALUDE_set_membership_l3768_376852


namespace NUMINAMATH_CALUDE_square_root_difference_l3768_376819

theorem square_root_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_l3768_376819


namespace NUMINAMATH_CALUDE_impossible_circle_arrangement_l3768_376813

theorem impossible_circle_arrangement : ¬ ∃ (a : Fin 7 → ℕ),
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 1) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 2) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 3) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 4) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 5) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 6) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 7) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_circle_arrangement_l3768_376813


namespace NUMINAMATH_CALUDE_problem_solution_l3768_376822

theorem problem_solution (a b : ℝ) 
  (h1 : a^3 - b^3 = 4)
  (h2 : a^2 + a*b + b^2 + a - b = 4) : 
  a - b = 2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3768_376822


namespace NUMINAMATH_CALUDE_three_five_two_takes_five_steps_l3768_376831

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool := sorry

/-- Performs one step of the process: reverse, add 3, then add to original -/
def step (n : ℕ) : ℕ := n + (reverseNum n + 3)

/-- Counts the number of steps to reach a palindrome -/
def stepsToBecomePalindrome (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem three_five_two_takes_five_steps :
  352 ≥ 100 ∧ 352 ≤ 400 ∧ 
  ¬isPalindrome 352 ∧
  stepsToBecomePalindrome 352 = 5 := by sorry

end NUMINAMATH_CALUDE_three_five_two_takes_five_steps_l3768_376831


namespace NUMINAMATH_CALUDE_intersection_A_B_l3768_376800

def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3768_376800


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3768_376844

theorem trigonometric_simplification (x : ℝ) : 
  Real.sin (x + π / 3) + 2 * Real.sin (x - π / 3) - Real.sqrt 3 * Real.cos (2 * π / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3768_376844


namespace NUMINAMATH_CALUDE_three_tetrominoes_with_symmetry_l3768_376829

-- Define the set of tetrominoes
inductive Tetromino
| I -- Line
| O -- Square
| T
| S
| Z

-- Define a function to check if a tetromino has reflectional symmetry
def has_reflectional_symmetry : Tetromino → Bool
| Tetromino.I => true
| Tetromino.O => true
| Tetromino.T => true
| Tetromino.S => false
| Tetromino.Z => false

-- Define the set of all tetrominoes
def all_tetrominoes : List Tetromino :=
  [Tetromino.I, Tetromino.O, Tetromino.T, Tetromino.S, Tetromino.Z]

-- Theorem: Exactly 3 tetrominoes have reflectional symmetry
theorem three_tetrominoes_with_symmetry :
  (all_tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_tetrominoes_with_symmetry_l3768_376829


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3768_376814

/-- The number of games played in a chess tournament -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 15 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 105. -/
theorem chess_tournament_games :
  numGames 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3768_376814


namespace NUMINAMATH_CALUDE_square_sum_given_square_sum_and_product_l3768_376872

theorem square_sum_given_square_sum_and_product
  (x y : ℝ) (h1 : (x + y)^2 = 25) (h2 : x * y = -6) :
  x^2 + y^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_square_sum_and_product_l3768_376872


namespace NUMINAMATH_CALUDE_journey_time_ratio_l3768_376862

/-- Proves that for a journey of 288 km, if the original time taken is 6 hours
    and the new speed is 32 kmph, then the ratio of the new time to the original time is 3:2. -/
theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 288 →
  original_time = 6 →
  new_speed = 32 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_journey_time_ratio_l3768_376862


namespace NUMINAMATH_CALUDE_largest_valid_number_l3768_376865

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧  -- all digits are different
  (∀ i j, i < j → (n / 10^i) % 10 ≤ (n / 10^j) % 10)  -- digits in ascending order

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 7089 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3768_376865


namespace NUMINAMATH_CALUDE_eight_beads_two_identical_arrangements_l3768_376869

/-- The number of unique arrangements of n distinct beads, including k identical beads, on a bracelet, considering rotational and reflectional symmetry -/
def uniqueBraceletArrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial / k.factorial) / (2 * n)

theorem eight_beads_two_identical_arrangements :
  uniqueBraceletArrangements 8 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_eight_beads_two_identical_arrangements_l3768_376869


namespace NUMINAMATH_CALUDE_lost_people_problem_l3768_376873

/-- Calculates the number of people in the second group of lost people --/
def second_group_size (initial_group : ℕ) (initial_days : ℕ) (days_passed : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_food := initial_group * initial_days
  let remaining_food := total_food - (initial_group * days_passed)
  let total_people := remaining_food / remaining_days
  total_people - initial_group

/-- Theorem stating that given the problem conditions, the second group has 3 people --/
theorem lost_people_problem :
  second_group_size 9 5 1 3 = 3 := by
  sorry


end NUMINAMATH_CALUDE_lost_people_problem_l3768_376873


namespace NUMINAMATH_CALUDE_election_votes_proof_l3768_376849

theorem election_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) (winner_percentage : ℚ) :
  total_votes = 7000 →
  invalid_percentage = 1/5 →
  winner_percentage = 11/20 →
  let valid_votes := total_votes - (invalid_percentage * total_votes).num
  let winner_votes := (winner_percentage * valid_votes).num
  valid_votes - winner_votes = 2520 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_proof_l3768_376849


namespace NUMINAMATH_CALUDE_female_officers_count_l3768_376871

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) (female_percent_of_total : ℚ) :
  total_on_duty = 150 →
  female_on_duty_percent = 25 / 100 →
  female_percent_of_total = 40 / 100 →
  ∃ (total_female : ℕ), total_female = 240 ∧ 
    (female_on_duty_percent * total_female : ℚ) = (female_percent_of_total * total_on_duty : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3768_376871


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3768_376858

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3768_376858


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3768_376880

/-- Given a bus with an average speed including stoppages and the time it stops per hour,
    calculate the average speed excluding stoppages. -/
theorem bus_speed_excluding_stoppages
  (speed_with_stops : ℝ)
  (stop_time : ℝ)
  (h1 : speed_with_stops = 20)
  (h2 : stop_time = 40) :
  speed_with_stops * (60 / (60 - stop_time)) = 60 :=
sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3768_376880


namespace NUMINAMATH_CALUDE_art_fair_sales_l3768_376812

theorem art_fair_sales (total_visitors : ℕ) (two_painting_buyers : ℕ) (one_painting_buyers : ℕ) (total_paintings_sold : ℕ) :
  total_visitors = 20 →
  two_painting_buyers = 4 →
  one_painting_buyers = 12 →
  total_paintings_sold = 36 →
  ∃ (four_painting_buyers : ℕ),
    four_painting_buyers * 4 + two_painting_buyers * 2 + one_painting_buyers = total_paintings_sold ∧
    four_painting_buyers + two_painting_buyers + one_painting_buyers ≤ total_visitors ∧
    four_painting_buyers = 4 :=
by sorry

end NUMINAMATH_CALUDE_art_fair_sales_l3768_376812


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l3768_376853

theorem volunteer_schedule_lcm : Nat.lcm 2 (Nat.lcm 5 (Nat.lcm 9 11)) = 990 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l3768_376853


namespace NUMINAMATH_CALUDE_total_savings_ten_sets_l3768_376882

-- Define the cost of 2 packs
def cost_two_packs : ℚ := 2.5

-- Define the cost of an individual pack
def cost_individual : ℚ := 1.3

-- Define the number of sets
def num_sets : ℕ := 10

-- Theorem statement
theorem total_savings_ten_sets : 
  let cost_per_pack := cost_two_packs / 2
  let savings_per_pack := cost_individual - cost_per_pack
  let total_packs := num_sets * 2
  savings_per_pack * total_packs = 1 := by sorry

end NUMINAMATH_CALUDE_total_savings_ten_sets_l3768_376882


namespace NUMINAMATH_CALUDE_tens_digit_of_9_to_1024_l3768_376855

theorem tens_digit_of_9_to_1024 : ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 9^1024 ≡ n [ZMOD 100] ∧ (n / 10) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_to_1024_l3768_376855


namespace NUMINAMATH_CALUDE_solve_for_y_l3768_376848

theorem solve_for_y (x y : ℝ) (hx : x = 51) (heq : x^3 * y^2 - 4 * x^2 * y^2 + 4 * x * y^2 = 100800) :
  y = 1/34 ∨ y = -1/34 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3768_376848


namespace NUMINAMATH_CALUDE_claire_orange_price_l3768_376874

-- Define the given quantities
def liam_oranges : ℕ := 40
def liam_price : ℚ := 2.5 / 2
def claire_oranges : ℕ := 30
def total_savings : ℚ := 86

-- Define Claire's price per orange
def claire_price : ℚ := (total_savings - (liam_oranges : ℚ) * liam_price) / (claire_oranges : ℚ)

-- Theorem statement
theorem claire_orange_price : claire_price = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_claire_orange_price_l3768_376874


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_810_l3768_376899

theorem sin_n_eq_cos_810 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (810 * π / 180) ↔ n = -180 ∨ n = 0 ∨ n = 180) :=
by sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_810_l3768_376899


namespace NUMINAMATH_CALUDE_smallest_twin_egg_number_l3768_376857

def is_twin_egg_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ a ≠ b ∧ n = 1000 * a + 100 * b + 10 * b + a

def F (m : ℕ) : ℚ :=
  let m' := (m % 100) * 100 + (m / 100)
  (m - m') / 11

theorem smallest_twin_egg_number :
  ∃ (m : ℕ),
    is_twin_egg_number m ∧
    ∃ (k : ℕ), F m / 54 = k^2 ∧
    ∀ (n : ℕ), is_twin_egg_number n → (∃ (l : ℕ), F n / 54 = l^2) → m ≤ n ∧
    m = 7117 :=
sorry

end NUMINAMATH_CALUDE_smallest_twin_egg_number_l3768_376857


namespace NUMINAMATH_CALUDE_completing_square_sum_l3768_376842

theorem completing_square_sum (a b : ℝ) : 
  (∀ x, x^2 + 6*x - 1 = 0 ↔ (x + a)^2 = b) → a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l3768_376842


namespace NUMINAMATH_CALUDE_product_of_monomials_l3768_376859

theorem product_of_monomials (x y : ℝ) : 2 * x * (-3 * x^2 * y^3) = -6 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_monomials_l3768_376859


namespace NUMINAMATH_CALUDE_original_profit_percentage_l3768_376827

theorem original_profit_percentage (cost_price selling_price : ℝ) :
  cost_price > 0 →
  selling_price > cost_price →
  (2 * selling_price - cost_price) / cost_price = 3.2 →
  (selling_price - cost_price) / cost_price = 1.1 := by
  sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l3768_376827


namespace NUMINAMATH_CALUDE_lucy_fish_count_l3768_376815

theorem lucy_fish_count (initial_fish : ℕ) (fish_to_buy : ℕ) (total_fish : ℕ) : 
  initial_fish = 212 → fish_to_buy = 68 → total_fish = initial_fish + fish_to_buy → total_fish = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l3768_376815


namespace NUMINAMATH_CALUDE_triangle_tangent_solution_l3768_376845

theorem triangle_tangent_solution (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z →
  4 * x * y * z = 4 * (x + y + z) →
  ∃ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ x = Real.tan A ∧ y = Real.tan B ∧ z = Real.tan C :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_solution_l3768_376845


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3768_376854

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := x - 3 ≥ 0

-- Theorem statement
theorem sqrt_meaningful_range (x : ℝ) :
  meaningful_sqrt x ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3768_376854


namespace NUMINAMATH_CALUDE_gym_equipment_cost_l3768_376834

/-- Calculates the total cost in dollars including sales tax for gym equipment purchase -/
def total_cost_with_tax (squat_rack_cost : ℝ) (barbell_fraction : ℝ) (weights_cost : ℝ) 
  (exchange_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let barbell_cost := squat_rack_cost * barbell_fraction
  let total_euro := squat_rack_cost + barbell_cost + weights_cost
  let total_dollar := total_euro * exchange_rate
  let tax := total_dollar * tax_rate
  total_dollar + tax

/-- Theorem stating the total cost of gym equipment including tax -/
theorem gym_equipment_cost : 
  total_cost_with_tax 2500 0.1 750 1.15 0.06 = 4266.50 := by
  sorry

end NUMINAMATH_CALUDE_gym_equipment_cost_l3768_376834


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l3768_376884

theorem existence_of_counterexample : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l3768_376884


namespace NUMINAMATH_CALUDE_mp3_song_count_l3768_376826

theorem mp3_song_count (x y : ℕ) : 
  (15 : ℕ) - x + y = 2 * 15 → y = x + 15 := by
sorry

end NUMINAMATH_CALUDE_mp3_song_count_l3768_376826


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3768_376863

theorem triangle_side_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c) :
  2 * b^2 = a^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3768_376863


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3768_376894

theorem arithmetic_equation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3768_376894


namespace NUMINAMATH_CALUDE_train_platform_length_l3768_376851

/-- The length of a train platform problem -/
theorem train_platform_length 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : train_length = 270)
  (h2 : platform1_length = 120)
  (h3 : time1 = 15)
  (h4 : time2 = 20) :
  ∃ (platform2_length : ℝ),
    platform2_length = 250 ∧ 
    (train_length + platform1_length) / time1 = 
    (train_length + platform2_length) / time2 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_length_l3768_376851


namespace NUMINAMATH_CALUDE_angle_cosine_relation_l3768_376868

theorem angle_cosine_relation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let r := Real.sqrt (x^2 + y^2 + z^2)
  x / r = 1/4 ∧ y / r = 1/8 → z / r = Real.sqrt 59 / 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_relation_l3768_376868


namespace NUMINAMATH_CALUDE_three_true_propositions_l3768_376867

theorem three_true_propositions
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h_a_order : a₁ < a₂ ∧ a₂ < a₃)
  (h_b_order : b₁ < b₂ ∧ b₂ < b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_sum_prod : a₁*a₂ + a₁*a₃ + a₂*a₃ = b₁*b₂ + b₁*b₃ + b₂*b₃)
  (h_a₁_b₁ : a₁ < b₁) :
  ∃! (count : ℕ), count = 3 ∧ count = (
    (if b₂ < a₂ then 1 else 0) +
    (if a₃ < b₃ then 1 else 0) +
    (if a₁*a₂*a₃ < b₁*b₂*b₃ then 1 else 0) +
    (if (1-a₁)*(1-a₂)*(1-a₃) > (1-b₁)*(1-b₂)*(1-b₃) then 1 else 0)
  ) :=
sorry

end NUMINAMATH_CALUDE_three_true_propositions_l3768_376867


namespace NUMINAMATH_CALUDE_cannot_determine_jake_peaches_l3768_376875

def steven_peaches : ℕ := 9
def steven_apples : ℕ := 8

structure Jake where
  peaches : ℕ
  apples : ℕ
  fewer_peaches : peaches < steven_peaches
  more_apples : apples = steven_apples + 3

theorem cannot_determine_jake_peaches : ∀ (jake : Jake), ∃ (jake' : Jake), jake.peaches ≠ jake'.peaches := by
  sorry

end NUMINAMATH_CALUDE_cannot_determine_jake_peaches_l3768_376875


namespace NUMINAMATH_CALUDE_bob_tv_width_is_90_l3768_376866

/-- The width of Bob's TV -/
def bob_tv_width : ℝ := 90

/-- The height of Bob's TV -/
def bob_tv_height : ℝ := 60

/-- The width of Bill's TV -/
def bill_tv_width : ℝ := 100

/-- The height of Bill's TV -/
def bill_tv_height : ℝ := 48

/-- Weight of TV per square inch in ounces -/
def tv_weight_per_sq_inch : ℝ := 4

/-- Ounces per pound -/
def oz_per_pound : ℝ := 16

/-- Weight difference between Bob's and Bill's TVs in pounds -/
def weight_difference : ℝ := 150

theorem bob_tv_width_is_90 :
  bob_tv_width = 90 :=
by
  sorry

#check bob_tv_width_is_90

end NUMINAMATH_CALUDE_bob_tv_width_is_90_l3768_376866


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l3768_376809

theorem square_minus_product_plus_square : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l3768_376809


namespace NUMINAMATH_CALUDE_bubble_sort_correct_l3768_376840

def bubbleSort (xs : List Int) : List Int :=
  let rec pass : List Int → List Int
    | [] => []
    | [x] => [x]
    | x :: y :: rest => if x <= y then x :: pass (y :: rest) else y :: pass (x :: rest)
  let rec sort (xs : List Int) (n : Nat) : List Int :=
    if n = 0 then xs else sort (pass xs) (n - 1)
  sort xs xs.length

theorem bubble_sort_correct (xs : List Int) :
  bubbleSort [8, 6, 3, 18, 21, 67, 54] = [3, 6, 8, 18, 21, 54, 67] := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_correct_l3768_376840


namespace NUMINAMATH_CALUDE_negation_and_range_of_a_l3768_376824

def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0

theorem negation_and_range_of_a :
  (∀ a : ℝ, ¬(proposition_p a) ↔ ∀ x : ℝ, x^2 + 2*a*x + a > 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1)) :=
sorry

end NUMINAMATH_CALUDE_negation_and_range_of_a_l3768_376824


namespace NUMINAMATH_CALUDE_g_at_negative_three_l3768_376825

theorem g_at_negative_three (g : ℝ → ℝ) :
  (∀ x, g x = 10 * x^3 - 7 * x^2 - 5 * x + 6) →
  g (-3) = -312 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_three_l3768_376825


namespace NUMINAMATH_CALUDE_encyclopedia_cost_l3768_376808

/-- Proves that the cost of encyclopedias is approximately $1002.86 given the specified conditions --/
theorem encyclopedia_cost (down_payment : ℝ) (monthly_payment : ℝ) (num_monthly_payments : ℕ)
  (final_payment : ℝ) (interest_rate : ℝ) :
  down_payment = 300 →
  monthly_payment = 57 →
  num_monthly_payments = 9 →
  final_payment = 21 →
  interest_rate = 18.666666666666668 / 100 →
  ∃ (cost : ℝ), abs (cost - 1002.86) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_encyclopedia_cost_l3768_376808


namespace NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_25_and_26_l3768_376810

theorem exists_integer_divisible_by_15_with_sqrt_between_25_and_26 :
  ∃ n : ℕ+, 15 ∣ n ∧ (25 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 26 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_25_and_26_l3768_376810


namespace NUMINAMATH_CALUDE_parabola_shift_l3768_376823

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 2 1
  shifted = Parabola.mk 2 (-8) 7 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3768_376823


namespace NUMINAMATH_CALUDE_binomial_15_13_l3768_376835

theorem binomial_15_13 : Nat.choose 15 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_13_l3768_376835


namespace NUMINAMATH_CALUDE_rectangular_field_shortcut_l3768_376883

theorem rectangular_field_shortcut (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x < y) :
  x + y - Real.sqrt (x^2 + y^2) = x →
  y / x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_shortcut_l3768_376883


namespace NUMINAMATH_CALUDE_negation_of_implication_l3768_376843

theorem negation_of_implication (x y : ℝ) :
  ¬(xy = 0 → x = 0 ∨ y = 0) ↔ (xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3768_376843


namespace NUMINAMATH_CALUDE_eight_people_twentyeight_handshakes_l3768_376804

/-- The number of handshakes in a function where every person shakes hands with every other person exactly once -/
def total_handshakes : ℕ := 28

/-- Calculates the number of handshakes given the number of people -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Proves that 8 people results in 28 handshakes -/
theorem eight_people_twentyeight_handshakes :
  ∃ (n : ℕ), n > 0 ∧ handshakes n = total_handshakes ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_people_twentyeight_handshakes_l3768_376804


namespace NUMINAMATH_CALUDE_ellipse_canonical_equation_l3768_376898

/-- Proves that an ellipse with given minor axis length and distance between foci has the specified canonical equation -/
theorem ellipse_canonical_equation 
  (minor_axis : ℝ) 
  (foci_distance : ℝ) 
  (h_minor : minor_axis = 6) 
  (h_foci : foci_distance = 8) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 25 + y^2 / 9 = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_canonical_equation_l3768_376898


namespace NUMINAMATH_CALUDE_circles_intersection_parallel_lines_l3768_376850

-- Define the basic geometric objects
variable (Circle1 Circle2 : Set (ℝ × ℝ))
variable (M K A B C D : ℝ × ℝ)

-- Define the conditions
axiom intersect_points : M ∈ Circle1 ∧ M ∈ Circle2 ∧ K ∈ Circle1 ∧ K ∈ Circle2
axiom line_AB : M.1 * B.2 - M.2 * B.1 = A.1 * B.2 - A.2 * B.1
axiom line_CD : K.1 * D.2 - K.2 * D.1 = C.1 * D.2 - C.2 * D.1
axiom A_in_Circle1 : A ∈ Circle1
axiom B_in_Circle2 : B ∈ Circle2
axiom C_in_Circle1 : C ∈ Circle1
axiom D_in_Circle2 : D ∈ Circle2

-- Define parallel lines
def parallel (p q r s : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.2 - s.2) = (p.2 - q.2) * (r.1 - s.1)

-- State the theorem
theorem circles_intersection_parallel_lines :
  parallel A C B D :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_parallel_lines_l3768_376850


namespace NUMINAMATH_CALUDE_product_sum_relation_l3768_376838

theorem product_sum_relation (a b N : ℤ) : 
  b = 7 → 
  b - a = 4 → 
  a * b = 2 * (a + b) + N → 
  N = 1 := by
sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3768_376838


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l3768_376802

/-- Represents the maximum distance a car can travel with one tire swap --/
def max_distance_with_swap (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  front_tire_life + (rear_tire_life - front_tire_life) / 2

/-- Theorem: Given specific tire lifespans, the maximum distance with one swap is 48,000 km --/
theorem max_distance_for_given_tires :
  max_distance_with_swap 42000 56000 = 48000 := by
  sorry

#eval max_distance_with_swap 42000 56000

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l3768_376802


namespace NUMINAMATH_CALUDE_cheezits_calorie_count_l3768_376816

/-- The number of calories in an ounce of Cheezits -/
def calories_per_ounce : ℕ := sorry

/-- The number of bags of Cheezits James ate -/
def bags_eaten : ℕ := 3

/-- The number of ounces per bag of Cheezits -/
def ounces_per_bag : ℕ := 2

/-- The number of minutes James ran -/
def minutes_run : ℕ := 40

/-- The number of calories James burned per minute of running -/
def calories_burned_per_minute : ℕ := 12

/-- The number of excess calories James had after eating and running -/
def excess_calories : ℕ := 420

theorem cheezits_calorie_count :
  calories_per_ounce = 150 ∧
  bags_eaten * ounces_per_bag * calories_per_ounce - minutes_run * calories_burned_per_minute = excess_calories :=
by sorry

end NUMINAMATH_CALUDE_cheezits_calorie_count_l3768_376816


namespace NUMINAMATH_CALUDE_noemi_initial_amount_l3768_376877

def initial_amount (roulette_loss blackjack_loss remaining : ℕ) : ℕ :=
  roulette_loss + blackjack_loss + remaining

theorem noemi_initial_amount :
  initial_amount 400 500 800 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_amount_l3768_376877
