import Mathlib

namespace NUMINAMATH_CALUDE_original_cost_equals_new_cost_l361_36178

/-- Proves that the original manufacturing cost was equal to the new manufacturing cost
    when the profit percentage remains constant at 50% of the selling price. -/
theorem original_cost_equals_new_cost
  (selling_price : ℝ)
  (new_cost : ℝ)
  (h_profit_percentage : selling_price / 2 = selling_price - new_cost)
  (h_new_cost : new_cost = 50)
  : selling_price - (selling_price / 2) = new_cost :=
by sorry

end NUMINAMATH_CALUDE_original_cost_equals_new_cost_l361_36178


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l361_36158

theorem systematic_sampling_proof (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 883) (h2 : sample_size = 80) :
  ∃ (sampling_interval : ℕ) (n : ℕ),
    sampling_interval = 11 ∧ 
    n = 3 ∧ 
    total_students = sample_size * sampling_interval + n :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l361_36158


namespace NUMINAMATH_CALUDE_weaver_productivity_l361_36174

/-- Given that 16 weavers can weave 64 mats in 16 days at a constant rate,
    prove that 4 weavers can weave 16 mats in 4 days at the same rate. -/
theorem weaver_productivity 
  (rate : ℝ) -- The constant rate of weaving (mats per weaver per day)
  (h1 : 16 * rate * 16 = 64) -- 16 weavers can weave 64 mats in 16 days
  : 4 * rate * 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_weaver_productivity_l361_36174


namespace NUMINAMATH_CALUDE_ratio_problem_l361_36148

theorem ratio_problem (second_part : ℝ) (ratio_percent : ℝ) : 
  second_part = 4 → ratio_percent = 125 → (ratio_percent / 100) * second_part = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l361_36148


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l361_36156

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (-3 + 3 * z) = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l361_36156


namespace NUMINAMATH_CALUDE_music_student_count_l361_36193

/-- Represents the number of students in different categories -/
structure StudentCounts where
  total : ℕ
  art : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking music -/
def musicStudents (counts : StudentCounts) : ℕ :=
  counts.total - counts.neither - (counts.art - counts.both)

/-- Theorem stating the number of students taking music -/
theorem music_student_count (counts : StudentCounts)
    (h_total : counts.total = 500)
    (h_art : counts.art = 10)
    (h_both : counts.both = 10)
    (h_neither : counts.neither = 470) :
    musicStudents counts = 30 := by
  sorry

#eval musicStudents { total := 500, art := 10, both := 10, neither := 470 }

end NUMINAMATH_CALUDE_music_student_count_l361_36193


namespace NUMINAMATH_CALUDE_book_reading_fraction_l361_36191

theorem book_reading_fraction (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 60 →
  pages_read = (total_pages - pages_read) + 20 →
  (pages_read : ℚ) / total_pages = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l361_36191


namespace NUMINAMATH_CALUDE_unique_ambiguous_sum_l361_36119

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 36

def sum_triple (a b c : ℕ) : ℕ := a + b + c

theorem unique_ambiguous_sum :
  ∃ (s : ℕ), 
    (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
      is_valid_triple a₁ b₁ c₁ ∧ 
      is_valid_triple a₂ b₂ c₂ ∧ 
      sum_triple a₁ b₁ c₁ = s ∧ 
      sum_triple a₂ b₂ c₂ = s ∧ 
      (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)) ∧
    (∀ (t : ℕ), 
      t ≠ s → 
      ∀ (x y z u v w : ℕ), 
        is_valid_triple x y z → 
        is_valid_triple u v w → 
        sum_triple x y z = t → 
        sum_triple u v w = t → 
        (x, y, z) = (u, v, w)) →
  s = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_ambiguous_sum_l361_36119


namespace NUMINAMATH_CALUDE_billy_bumper_car_rides_l361_36104

/-- Calculates the number of bumper car rides given the number of ferris wheel rides,
    the cost per ride, and the total number of tickets used. -/
def bumper_car_rides (ferris_wheel_rides : ℕ) (cost_per_ride : ℕ) (total_tickets : ℕ) : ℕ :=
  (total_tickets - ferris_wheel_rides * cost_per_ride) / cost_per_ride

theorem billy_bumper_car_rides :
  bumper_car_rides 7 5 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_bumper_car_rides_l361_36104


namespace NUMINAMATH_CALUDE_min_roads_theorem_l361_36150

/-- A graph representing cities and roads -/
structure CityGraph where
  num_cities : ℕ
  num_roads : ℕ
  is_connected : Bool

/-- Check if a given number of roads is sufficient for connectivity -/
def is_sufficient (g : CityGraph) : Prop :=
  g.is_connected = true

/-- The minimum number of roads needed for connectivity -/
def min_roads_for_connectivity (num_cities : ℕ) : ℕ :=
  191

/-- Theorem stating that 191 roads are sufficient and necessary for connectivity -/
theorem min_roads_theorem (g : CityGraph) :
  g.num_cities = 21 → 
  (g.num_roads ≥ 191 → is_sufficient g) ∧
  (is_sufficient g → g.num_roads ≥ 191) :=
sorry

#check min_roads_theorem

end NUMINAMATH_CALUDE_min_roads_theorem_l361_36150


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l361_36130

/-- Given a geometric sequence {a_n} with a₂ = 8 and a₅ = 64, prove that the common ratio q = 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 2 = 8 →                    -- Given condition
  a 5 = 64 →                   -- Given condition
  q = 2 := by                  -- Conclusion to prove
sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l361_36130


namespace NUMINAMATH_CALUDE_stone_122_is_9_l361_36197

/-- Represents the number of stones in the line -/
def n : ℕ := 17

/-- The target count we're looking for -/
def target : ℕ := 122

/-- Function to determine the original stone number given a count in the sequence -/
def originalStone (count : ℕ) : ℕ :=
  let modulo := count % (2 * (n - 1))
  if modulo ≤ n then
    modulo
  else
    2 * n - modulo

/-- Theorem stating that the stone counted as 122 is originally stone number 9 -/
theorem stone_122_is_9 : originalStone target = 9 := by
  sorry


end NUMINAMATH_CALUDE_stone_122_is_9_l361_36197


namespace NUMINAMATH_CALUDE_sugar_amount_l361_36116

/-- The number of cups of flour Mary still needs to add -/
def flour_needed : ℕ := 21

/-- The difference between the total cups of flour and sugar in the recipe -/
def flour_sugar_difference : ℕ := 8

/-- The number of cups of sugar the recipe calls for -/
def sugar_in_recipe : ℕ := flour_needed - flour_sugar_difference

theorem sugar_amount : sugar_in_recipe = 13 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_l361_36116


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l361_36184

theorem decimal_to_fraction : 
  (0.32 : ℚ) = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l361_36184


namespace NUMINAMATH_CALUDE_evaluate_expression_l361_36109

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -3) :
  x^2 * y^3 * z^2 = 1/48 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l361_36109


namespace NUMINAMATH_CALUDE_xyz_equals_ten_l361_36132

theorem xyz_equals_ten (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (sum_prod : x*y + x*z + y*z = 10)
  (sum : x + y + z = 6) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_ten_l361_36132


namespace NUMINAMATH_CALUDE_sqrt_sum_power_inequality_l361_36105

theorem sqrt_sum_power_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (Real.sqrt x + Real.sqrt y)^8 ≥ 64 * x * y * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_power_inequality_l361_36105


namespace NUMINAMATH_CALUDE_function_inequality_l361_36153

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, (x - 1) * deriv f x ≤ 0) : 
  f 0 + f 2 ≤ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l361_36153


namespace NUMINAMATH_CALUDE_rationalize_denominator_l361_36124

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l361_36124


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_is_29_l361_36135

/-- The polynomial p(x) = x^4 + 2x^2 + 5 -/
def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

/-- The remainder theorem: For a polynomial p(x) and a real number a,
    the remainder when p(x) is divided by (x - a) is equal to p(a) -/
theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a :=
sorry

theorem remainder_is_29 :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x + 29 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_is_29_l361_36135


namespace NUMINAMATH_CALUDE_correct_addition_l361_36167

theorem correct_addition (x : ℤ) (h : x + 21 = 52) : x + 40 = 71 := by
  sorry

end NUMINAMATH_CALUDE_correct_addition_l361_36167


namespace NUMINAMATH_CALUDE_division_problem_l361_36176

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l361_36176


namespace NUMINAMATH_CALUDE_hyperbola_sum_theorem_l361_36160

-- Define the hyperbola equation
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

-- Define the theorem
theorem hyperbola_sum_theorem (h k a b : ℝ) :
  -- Given conditions
  hyperbola_equation h k h k a b ∧
  (h = 3 ∧ k = -5) ∧
  (2 * a = 10) ∧
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ 2 * c = 14) →
  -- Conclusion
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_theorem_l361_36160


namespace NUMINAMATH_CALUDE_power_of_power_three_l361_36126

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l361_36126


namespace NUMINAMATH_CALUDE_prime_cube_minus_one_not_divisible_by_40_l361_36122

theorem prime_cube_minus_one_not_divisible_by_40 (p : ℕ) (hp : Prime p) (hp_ge_7 : p ≥ 7) :
  ¬(40 ∣ p^3 - 1) :=
sorry

end NUMINAMATH_CALUDE_prime_cube_minus_one_not_divisible_by_40_l361_36122


namespace NUMINAMATH_CALUDE_monday_sales_is_five_l361_36128

/-- Represents the number of crates of eggs sold on each day of the week --/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Defines the conditions for Gabrielle's egg sales --/
def validEggSales (sales : EggSales) : Prop :=
  sales.tuesday = 2 * sales.monday ∧
  sales.wednesday = sales.tuesday - 2 ∧
  sales.thursday = sales.tuesday / 2 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28

/-- Theorem stating that if the egg sales satisfy the given conditions,
    then the number of crates sold on Monday is 5 --/
theorem monday_sales_is_five (sales : EggSales) 
  (h : validEggSales sales) : sales.monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_monday_sales_is_five_l361_36128


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l361_36113

theorem inscribed_rectangle_area (square_area : ℝ) (ratio : ℝ) 
  (h_square_area : square_area = 18)
  (h_ratio : ratio = 2)
  (h_positive : square_area > 0) :
  let square_side := Real.sqrt square_area
  let rect_short_side := 2 * square_side / (ratio + 1 + Real.sqrt (ratio^2 + 1))
  let rect_long_side := ratio * rect_short_side
  rect_short_side * rect_long_side = 8 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_rectangle_area_l361_36113


namespace NUMINAMATH_CALUDE_remainder_problem_l361_36185

theorem remainder_problem (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6)
  (hsum : (f + y) % 15 = 7) :
  (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l361_36185


namespace NUMINAMATH_CALUDE_absolute_value_of_seven_minus_sqrt_53_l361_36140

theorem absolute_value_of_seven_minus_sqrt_53 :
  |7 - Real.sqrt 53| = Real.sqrt 53 - 7 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_seven_minus_sqrt_53_l361_36140


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l361_36180

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3*p - 4) * (6*q - 8) = 122 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l361_36180


namespace NUMINAMATH_CALUDE_fraction_equality_l361_36163

theorem fraction_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : 0 / b = b / c) (h2 : b / c = 1 / a) :
  (a + b - c) / (a - b + c) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l361_36163


namespace NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l361_36198

theorem r_fourth_plus_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l361_36198


namespace NUMINAMATH_CALUDE_min_value_on_line_l361_36120

theorem min_value_on_line (x y : ℝ) (h : x + 2 * y = 3) :
  ∃ (m : ℝ), m = 4 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a + 2 * b = 3 → 2^a + 4^b ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l361_36120


namespace NUMINAMATH_CALUDE_min_value_of_f_l361_36157

/-- The function f(x) = 2x³ - 6x² + m, where m is a constant -/
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

/-- Theorem: Given f(x) = 2x³ - 6x² + m, where m is a constant,
    and f(x) reaches a maximum value of 2 within the interval [-2, 2],
    the minimum value of f(x) within [-2, 2] is -6. -/
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 2) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -6 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, -6 ≤ f y m :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_f_l361_36157


namespace NUMINAMATH_CALUDE_distance_from_rate_and_time_l361_36141

/-- Proves that given a constant walking rate and time, the distance covered is equal to the product of rate and time. -/
theorem distance_from_rate_and_time 
  (rate : ℝ) 
  (time : ℝ) 
  (h_rate : rate = 4) 
  (h_time : time = 2) : 
  rate * time = 8 := by
  sorry

#check distance_from_rate_and_time

end NUMINAMATH_CALUDE_distance_from_rate_and_time_l361_36141


namespace NUMINAMATH_CALUDE_common_chord_equation_l361_36175

/-- The equation of the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l361_36175


namespace NUMINAMATH_CALUDE_gcf_lcm_360_210_l361_36117

theorem gcf_lcm_360_210 : 
  (Nat.gcd 360 210 = 30) ∧ (Nat.lcm 360 210 = 2520) := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_360_210_l361_36117


namespace NUMINAMATH_CALUDE_derivative_zero_sufficient_not_necessary_for_extremum_l361_36111

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the property of having an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ ∀ x, |x - x₀| < ε → f x ≥ f x₀

-- State the theorem
theorem derivative_zero_sufficient_not_necessary_for_extremum :
  (∃ x₀ : ℝ, deriv f x₀ = 0 → HasExtremumAt f x₀) ∧
  (∃ x₀ : ℝ, HasExtremumAt f x₀ ∧ deriv f x₀ ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_sufficient_not_necessary_for_extremum_l361_36111


namespace NUMINAMATH_CALUDE_line_increase_l361_36121

/-- Given an initial number of lines and an increased number of lines with a specific percentage increase, 
    prove that the increase in the number of lines is 110. -/
theorem line_increase (L : ℝ) : 
  let L' : ℝ := 240
  let percent_increase : ℝ := 84.61538461538461
  (L' - L) / L * 100 = percent_increase →
  L' - L = 110 := by
sorry

end NUMINAMATH_CALUDE_line_increase_l361_36121


namespace NUMINAMATH_CALUDE_sidney_kittens_l361_36138

/-- The number of kittens Sidney has -/
def num_kittens : ℕ := sorry

/-- The number of adult cats Sidney has -/
def num_adult_cats : ℕ := 3

/-- The number of cans Sidney already has -/
def initial_cans : ℕ := 7

/-- The number of additional cans Sidney needs to buy -/
def additional_cans : ℕ := 35

/-- The number of days Sidney needs to feed the cats -/
def num_days : ℕ := 7

/-- The amount of food (in cans) an adult cat eats per day -/
def adult_cat_food_per_day : ℚ := 1

/-- The amount of food (in cans) a kitten eats per day -/
def kitten_food_per_day : ℚ := 3/4

theorem sidney_kittens : 
  num_kittens = 4 ∧
  (num_kittens : ℚ) * kitten_food_per_day * num_days + 
  (num_adult_cats : ℚ) * adult_cat_food_per_day * num_days = 
  initial_cans + additional_cans :=
sorry

end NUMINAMATH_CALUDE_sidney_kittens_l361_36138


namespace NUMINAMATH_CALUDE_f_minimum_value_l361_36194

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x + x - Real.log x

-- State the theorem
theorem f_minimum_value :
  ∃ (min_value : ℝ), min_value = Real.exp 1 + 1 ∧
  ∀ (x : ℝ), x > 0 → f x ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l361_36194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l361_36192

def arithmetic_sequence : List ℕ := [71, 75, 79, 83, 87, 91]

theorem arithmetic_sequence_sum : 
  3 * (arithmetic_sequence.sum) = 1458 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l361_36192


namespace NUMINAMATH_CALUDE_workshop_transfer_l361_36186

theorem workshop_transfer (w : ℕ) (n : ℕ) (x : ℕ) : 
  w ≥ 63 →
  w ≤ 64 →
  31 * w + n * (n + 1) / 2 = 1994 →
  (n = 4 ∧ x = 4) ∨ (n = 2 ∧ x = 21) :=
sorry

end NUMINAMATH_CALUDE_workshop_transfer_l361_36186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l361_36159

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : isArithmeticSequence a) 
  (h_sum : a 3 + a 4 + a 6 + a 7 = 25) : 
  a 2 + a 8 = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l361_36159


namespace NUMINAMATH_CALUDE_product_and_sum_relations_l361_36179

/-- Given positive integers p, q, r satisfying the specified conditions, prove that p - r = -430 --/
theorem product_and_sum_relations (p q r : ℕ+) 
  (h_product : p * q * r = Nat.factorial 10)
  (h_sum1 : p * q + p + q = 2450)
  (h_sum2 : q * r + q + r = 1012)
  (h_sum3 : r * p + r + p = 2020) :
  (p : ℤ) - (r : ℤ) = -430 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_relations_l361_36179


namespace NUMINAMATH_CALUDE_smallest_w_l361_36139

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 14^2) → 
  w ≥ 1764 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l361_36139


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_23_l361_36134

theorem smallest_four_digit_congruent_to_one_mod_23 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≡ 1 [MOD 23] → 1013 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_23_l361_36134


namespace NUMINAMATH_CALUDE_tangent_line_min_slope_l361_36149

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 - x + 6

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 12*x - 1

-- Theorem statement
theorem tangent_line_min_slope :
  ∃ (x₀ y₀ : ℝ),
    f x₀ = y₀ ∧
    (∀ x : ℝ, f' x₀ ≤ f' x) ∧
    (13 * x₀ + y₀ - 14 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_min_slope_l361_36149


namespace NUMINAMATH_CALUDE_odometer_problem_l361_36112

theorem odometer_problem (a b c : ℕ) (n : ℕ+) :
  100 ≤ 100 * a + 10 * b + c →
  100 * a + 10 * b + c ≤ 999 →
  a ≥ 1 →
  a + b + c ≤ 7 →
  100 * c + 10 * b + a - (100 * a + 10 * b + c) = 55 * n →
  a^2 + b^2 + c^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_odometer_problem_l361_36112


namespace NUMINAMATH_CALUDE_bruce_purchase_cost_l361_36103

/-- The total cost of Bruce's purchase of grapes and mangoes -/
def total_cost (grape_quantity : ℕ) (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price

/-- Theorem stating the total cost of Bruce's purchase -/
theorem bruce_purchase_cost :
  total_cost 8 70 11 55 = 1165 := by
  sorry

#eval total_cost 8 70 11 55

end NUMINAMATH_CALUDE_bruce_purchase_cost_l361_36103


namespace NUMINAMATH_CALUDE_no_integer_solution_quadratic_prime_l361_36170

theorem no_integer_solution_quadratic_prime : 
  ¬ ∃ (x : ℤ), Nat.Prime (Int.natAbs (4 * x^2 - 39 * x + 35)) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_quadratic_prime_l361_36170


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l361_36189

def vector_a : ℝ × ℝ := (2, -3)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x^2 - 5*x)

theorem parallel_vectors_solution :
  ∀ x : ℝ, (∃ k : ℝ, vector_a = k • vector_b x) → x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l361_36189


namespace NUMINAMATH_CALUDE_complex_equation_sum_l361_36106

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + i) / i = 1 + b * i → a + b = 0 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l361_36106


namespace NUMINAMATH_CALUDE_abs_neg_five_equals_five_l361_36102

theorem abs_neg_five_equals_five : |(-5 : ℤ)| = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_five_equals_five_l361_36102


namespace NUMINAMATH_CALUDE_percentage_problem_l361_36133

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 3200 →
  0.1 * N = (P / 100) * 650 + 190 →
  P = 20 :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l361_36133


namespace NUMINAMATH_CALUDE_work_days_calculation_l361_36177

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the total earnings of all workers -/
def totalEarnings (days : WorkDays) (wages : DailyWages) : ℕ :=
  days.a * wages.a + days.b * wages.b + days.c * wages.c

/-- The main theorem stating the problem conditions and the result to be proved -/
theorem work_days_calculation (days : WorkDays) (wages : DailyWages) :
  days.a = 6 ∧
  days.c = 4 ∧
  wages.a * 4 = wages.b * 3 ∧
  wages.b * 5 = wages.c * 4 ∧
  wages.c = 125 ∧
  totalEarnings days wages = 1850 →
  days.b = 9 := by
  sorry


end NUMINAMATH_CALUDE_work_days_calculation_l361_36177


namespace NUMINAMATH_CALUDE_S_bounds_l361_36173

-- Define the function S
def S (x y z : ℝ) : ℝ := 2*x^2*y^2 + 2*x^2*z^2 + 2*y^2*z^2 - x^4 - y^4 - z^4

-- State the theorem
theorem S_bounds :
  ∀ x y z : ℝ,
  (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) →
  (5 ≤ x ∧ x ≤ 8) →
  (5 ≤ y ∧ y ≤ 8) →
  (5 ≤ z ∧ z ≤ 8) →
  1875 ≤ S x y z ∧ S x y z ≤ 31488 :=
by sorry

end NUMINAMATH_CALUDE_S_bounds_l361_36173


namespace NUMINAMATH_CALUDE_quadratic_propositions_l361_36196

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_propositions (a b c : ℝ) (ha : a ≠ 0) :
  -- Proposition 1
  (a + b + c = 0 → discriminant a b c ≥ 0) ∧
  -- Proposition 2
  (∃ x y : ℝ, x = -1 ∧ y = 2 ∧ quadratic a b c x ∧ quadratic a b c y → 2*a + c = 0) ∧
  -- Proposition 3
  ((∃ x y : ℝ, x ≠ y ∧ quadratic a 0 c x ∧ quadratic a 0 c y) →
   ∃ u v : ℝ, u ≠ v ∧ quadratic a b c u ∧ quadratic a b c v) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_propositions_l361_36196


namespace NUMINAMATH_CALUDE_calculate_expression_l361_36114

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l361_36114


namespace NUMINAMATH_CALUDE_coprime_power_sum_not_divisible_by_11_l361_36143

theorem coprime_power_sum_not_divisible_by_11 (a b : ℤ) (h : Int.gcd a b = 1) :
  ¬(11 ∣ (a^5 + 2*b^5)) ∧ ¬(11 ∣ (a^5 - 2*b^5)) := by
  sorry

end NUMINAMATH_CALUDE_coprime_power_sum_not_divisible_by_11_l361_36143


namespace NUMINAMATH_CALUDE_plant_supplier_money_left_l361_36127

/-- Represents the plant supplier's business --/
structure PlantSupplier where
  orchids : ℕ
  orchidPrice : ℕ
  moneyPlants : ℕ
  moneyPlantPrice : ℕ
  bonsai : ℕ
  bonsaiPrice : ℕ
  cacti : ℕ
  cactiPrice : ℕ
  airPlants : ℕ
  airPlantPrice : ℕ
  fullTimeWorkers : ℕ
  fullTimeWage : ℕ
  partTimeWorkers : ℕ
  partTimeWage : ℕ
  ceramicPotsCost : ℕ
  plasticPotsCost : ℕ
  fertilizersCost : ℕ
  toolsCost : ℕ
  utilityBill : ℕ
  tax : ℕ

/-- Calculates the total earnings of the plant supplier --/
def totalEarnings (s : PlantSupplier) : ℕ :=
  s.orchids * s.orchidPrice +
  s.moneyPlants * s.moneyPlantPrice +
  s.bonsai * s.bonsaiPrice +
  s.cacti * s.cactiPrice +
  s.airPlants * s.airPlantPrice

/-- Calculates the total expenses of the plant supplier --/
def totalExpenses (s : PlantSupplier) : ℕ :=
  s.fullTimeWorkers * s.fullTimeWage +
  s.partTimeWorkers * s.partTimeWage +
  s.ceramicPotsCost +
  s.plasticPotsCost +
  s.fertilizersCost +
  s.toolsCost +
  s.utilityBill +
  s.tax

/-- Calculates the money left from the plant supplier's earnings --/
def moneyLeft (s : PlantSupplier) : ℕ :=
  totalEarnings s - totalExpenses s

/-- Theorem stating that the money left is $3755 given the specified conditions --/
theorem plant_supplier_money_left :
  ∃ (s : PlantSupplier),
    s.orchids = 35 ∧ s.orchidPrice = 52 ∧
    s.moneyPlants = 30 ∧ s.moneyPlantPrice = 32 ∧
    s.bonsai = 20 ∧ s.bonsaiPrice = 77 ∧
    s.cacti = 25 ∧ s.cactiPrice = 22 ∧
    s.airPlants = 40 ∧ s.airPlantPrice = 15 ∧
    s.fullTimeWorkers = 3 ∧ s.fullTimeWage = 65 ∧
    s.partTimeWorkers = 2 ∧ s.partTimeWage = 45 ∧
    s.ceramicPotsCost = 280 ∧
    s.plasticPotsCost = 150 ∧
    s.fertilizersCost = 100 ∧
    s.toolsCost = 125 ∧
    s.utilityBill = 225 ∧
    s.tax = 550 ∧
    moneyLeft s = 3755 := by
  sorry

end NUMINAMATH_CALUDE_plant_supplier_money_left_l361_36127


namespace NUMINAMATH_CALUDE_distinct_values_mod_p_l361_36118

theorem distinct_values_mod_p (p : ℕ) (a b : Fin p) (hp : Nat.Prime p) (hab : a ≠ b) :
  let f : Fin p → ℕ := λ n => (Finset.range (p - 1)).sum (λ i => (i + 1) * n^(i + 1))
  ¬ (f a ≡ f b [MOD p]) := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_mod_p_l361_36118


namespace NUMINAMATH_CALUDE_fish_value_in_rice_fish_value_in_rice_mixed_l361_36131

-- Define the trading rates
def fish_to_bread_rate : ℚ := 3 / 5
def bread_to_rice_rate : ℕ := 7

-- Theorem statement
theorem fish_value_in_rice : 
  fish_to_bread_rate * bread_to_rice_rate = 21 / 5 := by
  sorry

-- Converting the result to a mixed number
theorem fish_value_in_rice_mixed : 
  ∃ (whole : ℕ) (frac : ℚ), 
    fish_to_bread_rate * bread_to_rice_rate = whole + frac ∧ 
    whole = 4 ∧ 
    frac = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fish_value_in_rice_fish_value_in_rice_mixed_l361_36131


namespace NUMINAMATH_CALUDE_berry_difference_l361_36123

theorem berry_difference (stacy_initial : ℕ) (steve_initial : ℕ) (taken : ℕ) : 
  stacy_initial = 32 → 
  steve_initial = 21 → 
  taken = 4 → 
  stacy_initial - (steve_initial + taken) = 7 := by
sorry

end NUMINAMATH_CALUDE_berry_difference_l361_36123


namespace NUMINAMATH_CALUDE_sin_30_plus_cos_60_l361_36168

theorem sin_30_plus_cos_60 : Real.sin (30 * π / 180) + Real.cos (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_plus_cos_60_l361_36168


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l361_36169

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ+, n ∣ (2^n.val - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l361_36169


namespace NUMINAMATH_CALUDE_linda_income_l361_36172

/-- Represents the tax structure and Linda's income --/
structure TaxInfo where
  p : ℝ  -- base tax rate in decimal form
  income : ℝ  -- Linda's annual income

/-- Calculates the total tax based on the given tax structure --/
def calculateTax (info : TaxInfo) : ℝ :=
  let baseTax := info.p * 35000
  let excessTax := (info.p + 0.03) * (info.income - 35000)
  baseTax + excessTax

/-- Theorem stating that Linda's income is $42000 given the tax conditions --/
theorem linda_income (info : TaxInfo) :
  (calculateTax info = (info.p + 0.005) * info.income) →
  info.income = 42000 := by
  sorry

#check linda_income

end NUMINAMATH_CALUDE_linda_income_l361_36172


namespace NUMINAMATH_CALUDE_house_sale_profit_percentage_l361_36188

/-- Calculates the profit percentage for a house sale --/
theorem house_sale_profit_percentage
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (commission_rate : ℝ)
  (h1 : purchase_price = 80000)
  (h2 : selling_price = 100000)
  (h3 : commission_rate = 0.05)
  : (selling_price - commission_rate * purchase_price - purchase_price) / purchase_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_profit_percentage_l361_36188


namespace NUMINAMATH_CALUDE_whole_number_between_36_and_40_l361_36155

theorem whole_number_between_36_and_40 (M : ℕ) : 
  (9 < (M : ℚ) / 4) ∧ ((M : ℚ) / 4 < 10) → M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_36_and_40_l361_36155


namespace NUMINAMATH_CALUDE_twelve_not_feasible_fourteen_feasible_l361_36107

/-- Represents the conditions for forming a convex equiangular hexagon from equilateral triangular tiles. -/
def IsValidHexagonConfiguration (n ℓ a b c : ℕ) : Prop :=
  n = ℓ^2 - a^2 - b^2 - c^2 ∧ 
  ℓ > a + b ∧ 
  ℓ > a + c ∧ 
  ℓ > b + c

/-- States that 12 is not a feasible number of tiles for forming a convex equiangular hexagon. -/
theorem twelve_not_feasible : ¬ ∃ (ℓ a b c : ℕ), IsValidHexagonConfiguration 12 ℓ a b c :=
sorry

/-- States that 14 is a feasible number of tiles for forming a convex equiangular hexagon. -/
theorem fourteen_feasible : ∃ (ℓ a b c : ℕ), IsValidHexagonConfiguration 14 ℓ a b c :=
sorry

end NUMINAMATH_CALUDE_twelve_not_feasible_fourteen_feasible_l361_36107


namespace NUMINAMATH_CALUDE_circle_construction_theorem_circle_line_construction_theorem_l361_36195

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define tangency between circles
def CircleTangent (c1 c2 : Circle) : Prop := sorry

-- Define tangency between a circle and a line
def CircleLineTangent (c : Circle) (l : Line) : Prop := sorry

-- Define a circle passing through a point
def CirclePassesThrough (c : Circle) (p : Point) : Prop := sorry

theorem circle_construction_theorem 
  (P : Point) 
  (S1 S2 : Circle) : 
  ∃ (C : Circle), 
    CirclePassesThrough C P ∧ 
    CircleTangent C S1 ∧ 
    CircleTangent C S2 := by sorry

theorem circle_line_construction_theorem 
  (P : Point) 
  (S : Circle) 
  (L : Line) : 
  ∃ (C : Circle), 
    CirclePassesThrough C P ∧ 
    CircleTangent C S ∧ 
    CircleLineTangent C L := by sorry

end NUMINAMATH_CALUDE_circle_construction_theorem_circle_line_construction_theorem_l361_36195


namespace NUMINAMATH_CALUDE_next_challenge_digits_estimate_l361_36129

/-- The number of decimal digits in RSA-640 -/
def rsa640_digits : ℕ := 193

/-- The prize amount for RSA-640 in dollars -/
def rsa640_prize : ℕ := 20000

/-- The prize amount for the next challenge in dollars -/
def next_challenge_prize : ℕ := 30000

/-- A reasonable upper bound for the number of digits in the next challenge -/
def reasonable_upper_bound : ℕ := 220

/-- Theorem stating that a reasonable estimate for the number of digits
    in the next challenge is greater than RSA-640's digits and at most 220 -/
theorem next_challenge_digits_estimate :
  ∃ (N : ℕ), N > rsa640_digits ∧ N ≤ reasonable_upper_bound ∧
  next_challenge_prize > rsa640_prize :=
sorry

end NUMINAMATH_CALUDE_next_challenge_digits_estimate_l361_36129


namespace NUMINAMATH_CALUDE_irreducible_fraction_l361_36108

theorem irreducible_fraction : (201920192019 : ℚ) / 191719171917 = 673 / 639 := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l361_36108


namespace NUMINAMATH_CALUDE_distance_for_specific_cube_l361_36165

/-- Represents a cube suspended above a plane -/
structure SuspendedCube where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ

/-- The distance from the closest vertex to the plane for a suspended cube -/
def distance_to_plane (cube : SuspendedCube) : ℝ :=
  sorry

/-- Theorem stating the distance for the given cube configuration -/
theorem distance_for_specific_cube :
  let cube : SuspendedCube :=
    { side_length := 8
      adjacent_heights := ![8, 10, 9] }
  distance_to_plane cube = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_specific_cube_l361_36165


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l361_36142

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- The vertices of the triangle -/
def A : ℝ × ℝ × ℝ := (2, 3, 4)
def B : ℝ × ℝ × ℝ := (6, 4, 2)
def C : ℝ × ℝ × ℝ := (4, 6, 6)

/-- Theorem: The orthocenter of triangle ABC is (10/7, 51/7, 12/7) -/
theorem orthocenter_of_triangle :
  orthocenter A B C = (10/7, 51/7, 12/7) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l361_36142


namespace NUMINAMATH_CALUDE_derivative_cos_squared_at_pi_eighth_l361_36144

/-- Given a function f(x) = cos²(2x), its derivative at π/8 is -2. -/
theorem derivative_cos_squared_at_pi_eighth (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos (2 * x) ^ 2) :
  deriv f (π / 8) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_squared_at_pi_eighth_l361_36144


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l361_36161

/-- A line is represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Convert an equation of the form ax + by = c to slope-intercept form -/
def to_slope_intercept (a b c : ℚ) : Line :=
  { slope := -a / b, intercept := c / b }

theorem parallel_line_through_point 
  (l1 : Line) (p : Point) :
  ∃ (l2 : Line), 
    parallel l1 l2 ∧ 
    l2.passes_through p ∧
    l2.slope = 1/2 ∧ 
    l2.intercept = -2 :=
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l361_36161


namespace NUMINAMATH_CALUDE_cost_of_300_pencils_l361_36137

/-- The cost of pencils in dollars -/
def cost_in_dollars (num_pencils : ℕ) (cost_per_pencil_cents : ℕ) (cents_per_dollar : ℕ) : ℚ :=
  (num_pencils * cost_per_pencil_cents : ℚ) / cents_per_dollar

/-- Theorem: The cost of 300 pencils is 7.5 dollars -/
theorem cost_of_300_pencils :
  cost_in_dollars 300 5 200 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_300_pencils_l361_36137


namespace NUMINAMATH_CALUDE_larger_number_proof_l361_36183

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 1670) (h2 : 0.075 * x = 0.125 * y) (h3 : x > 0) (h4 : y > 0) : max x y = 4175 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l361_36183


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l361_36147

/-- The equation (3x^3 - 15x^2) / (x^2 - 5x) = 2x - 6 has exactly one solution -/
theorem equation_has_one_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = 2 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l361_36147


namespace NUMINAMATH_CALUDE_distinct_digit_sums_count_l361_36145

/-- Calculate the digit sum of a natural number. -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- The set of all digit sums for numbers from 1 to 2021. -/
def digitSumSet : Finset ℕ :=
  Finset.image digitSum (Finset.range 2021)

/-- Theorem: The number of distinct digit sums for integers from 1 to 2021 is 28. -/
theorem distinct_digit_sums_count : digitSumSet.card = 28 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digit_sums_count_l361_36145


namespace NUMINAMATH_CALUDE_a_12_upper_bound_a_12_no_lower_bound_l361_36146

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The upper bound of a_12 in an arithmetic sequence satisfying given conditions -/
theorem a_12_upper_bound
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8 : a 8 ≥ 15)
  (h_a9 : a 9 ≤ 13) :
  a 12 ≤ 7 :=
sorry

/-- The non-existence of a lower bound for a_12 in an arithmetic sequence satisfying given conditions -/
theorem a_12_no_lower_bound
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8 : a 8 ≥ 15)
  (h_a9 : a 9 ≤ 13) :
  ∀ x : ℝ, ∃ y : ℝ, y < x ∧ ∃ (a' : ℕ → ℝ), arithmetic_sequence a' ∧ a' 8 ≥ 15 ∧ a' 9 ≤ 13 ∧ a' 12 = y :=
sorry

end NUMINAMATH_CALUDE_a_12_upper_bound_a_12_no_lower_bound_l361_36146


namespace NUMINAMATH_CALUDE_final_position_l361_36125

def move_on_number_line (start : ℤ) (right : ℤ) (left : ℤ) : ℤ :=
  start + right - left

theorem final_position :
  move_on_number_line (-2) 3 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_final_position_l361_36125


namespace NUMINAMATH_CALUDE_series_divergent_l361_36151

open Complex

/-- The series ∑_{n=1}^{∞} (e^(iπ/n))/n is divergent -/
theorem series_divergent : 
  ¬ Summable (fun n : ℕ => (exp (I * π / n : ℂ)) / n) :=
sorry

end NUMINAMATH_CALUDE_series_divergent_l361_36151


namespace NUMINAMATH_CALUDE_solution_set_f_gt_g_min_m_for_inequality_l361_36115

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - |x + 1|
def g (x : ℝ) : ℝ := -x

-- Theorem for the solution of f(x) > g(x)
theorem solution_set_f_gt_g :
  {x : ℝ | f x > g x} = {x : ℝ | -3 < x ∧ x < 1 ∨ x > 3} := by sorry

-- Theorem for the minimum value of m
theorem min_m_for_inequality (x : ℝ) :
  ∃ m : ℝ, (∀ x : ℝ, f x - 2*x ≤ 2*(g x) + m) ∧
  (∀ m' : ℝ, (∀ x : ℝ, f x - 2*x ≤ 2*(g x) + m') → m ≤ m') ∧
  m = 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_g_min_m_for_inequality_l361_36115


namespace NUMINAMATH_CALUDE_max_product_decomposition_l361_36136

theorem max_product_decomposition :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y = 100 → x * y ≤ 50 * 50 :=
by sorry

end NUMINAMATH_CALUDE_max_product_decomposition_l361_36136


namespace NUMINAMATH_CALUDE_complex_equality_proof_l361_36182

theorem complex_equality_proof (a : ℂ) : (1 + a * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I → a = 5 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_proof_l361_36182


namespace NUMINAMATH_CALUDE_quadratic_roots_and_integer_k_l361_36162

/-- Represents a quadratic equation of the form kx^2 + (k-2)x - 2 = 0 --/
def QuadraticEquation (k : ℝ) : ℝ → Prop :=
  fun x => k * x^2 + (k - 2) * x - 2 = 0

theorem quadratic_roots_and_integer_k :
  ∀ k : ℝ, k ≠ 0 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ QuadraticEquation k x₁ ∧ QuadraticEquation k x₂) ∧
    (∃ k' : ℤ, k' ∈ ({-2, -1, 1, 2} : Set ℤ) ∧
      ∃ x₁ x₂ : ℤ, QuadraticEquation (k' : ℝ) x₁ ∧ QuadraticEquation (k' : ℝ) x₂) :=
by sorry

#check quadratic_roots_and_integer_k

end NUMINAMATH_CALUDE_quadratic_roots_and_integer_k_l361_36162


namespace NUMINAMATH_CALUDE_min_sum_squares_consecutive_integers_l361_36181

theorem min_sum_squares_consecutive_integers (y : ℤ) : 
  (∃ x : ℤ, y^2 = (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + 
              (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2) →
  y^2 ≥ 121 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_consecutive_integers_l361_36181


namespace NUMINAMATH_CALUDE_apartment_number_l361_36154

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def swap_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem apartment_number : 
  ∃! n : ℕ, is_three_digit n ∧ is_perfect_cube n ∧ Nat.Prime (swap_digits n) ∧ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_apartment_number_l361_36154


namespace NUMINAMATH_CALUDE_tiles_needed_is_108_l361_36101

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of a tile in inches -/
def tile : Dimensions := ⟨4, 6⟩

/-- The dimensions of the floor in feet -/
def floor : Dimensions := ⟨3, 6⟩

/-- The number of tiles needed to cover the floor -/
def tiles_needed : ℕ :=
  (area ⟨feet_to_inches floor.length, feet_to_inches floor.width⟩) / (area tile)

theorem tiles_needed_is_108 : tiles_needed = 108 := by
  sorry

#eval tiles_needed

end NUMINAMATH_CALUDE_tiles_needed_is_108_l361_36101


namespace NUMINAMATH_CALUDE_collinear_probability_l361_36164

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots we are selecting -/
def selectedDots : ℕ := 5

/-- The number of possible collinear sets of 5 dots in the grid -/
def collinearSets : ℕ := 2 * gridSize + 2

/-- The probability of selecting 5 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinearSets : ℚ) / Nat.choose totalDots selectedDots = 2 / 8855 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_l361_36164


namespace NUMINAMATH_CALUDE_donovan_candles_count_l361_36100

/-- The number of candles Donovan brought in -/
def donovans_candles : ℕ := 20

/-- The number of candles in Kalani's bedroom -/
def bedroom_candles : ℕ := 20

/-- The number of candles in the living room -/
def living_room_candles : ℕ := bedroom_candles / 2

/-- The total number of candles in the house -/
def total_candles : ℕ := 50

theorem donovan_candles_count :
  donovans_candles = total_candles - bedroom_candles - living_room_candles :=
by sorry

end NUMINAMATH_CALUDE_donovan_candles_count_l361_36100


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l361_36187

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 64*y^2 + 16*x - 32 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y, eq x y ↔ ((x - c)^2 / a^2) - ((y - d)^2 / b^2) = 1) ∨
  (∀ x y, eq x y ↔ ((y - d)^2 / a^2) - ((x - c)^2 / b^2) = 1)

/-- Theorem: The given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l361_36187


namespace NUMINAMATH_CALUDE_transportation_charges_l361_36152

theorem transportation_charges 
  (purchase_price : ℕ) 
  (repair_cost : ℕ) 
  (profit_percentage : ℚ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 13000)
  (h2 : repair_cost = 5000)
  (h3 : profit_percentage = 1/2)
  (h4 : selling_price = 28500) :
  ∃ (transportation_charges : ℕ),
    selling_price = (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) ∧
    transportation_charges = 1000 :=
by sorry

end NUMINAMATH_CALUDE_transportation_charges_l361_36152


namespace NUMINAMATH_CALUDE_fraction_split_l361_36110

theorem fraction_split (n d a b : ℕ) (h1 : d = a * b) (h2 : Nat.gcd a b = 1) (h3 : n = 58) (h4 : d = 77) (h5 : a = 11) (h6 : b = 7) :
  ∃ (x y : ℤ), (n : ℚ) / d = (x : ℚ) / b + (y : ℚ) / a :=
sorry

end NUMINAMATH_CALUDE_fraction_split_l361_36110


namespace NUMINAMATH_CALUDE_geometry_propositions_l361_36199

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom h₁ : p₁
axiom h₂ : ¬p₂
axiom h₃ : ¬p₃
axiom h₄ : p₄

-- Theorem to prove
theorem geometry_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l361_36199


namespace NUMINAMATH_CALUDE_collectors_edition_dolls_combined_l361_36166

theorem collectors_edition_dolls_combined (dina_dolls : ℕ) (ivy_dolls : ℕ) (luna_dolls : ℕ) :
  dina_dolls = 60 →
  dina_dolls = 2 * ivy_dolls →
  ivy_dolls = luna_dolls + 10 →
  (2 : ℕ) * (ivy_dolls * 2) = 3 * ivy_dolls →
  2 * luna_dolls = luna_dolls →
  (2 : ℕ) * (ivy_dolls * 2) / 3 + luna_dolls / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_collectors_edition_dolls_combined_l361_36166


namespace NUMINAMATH_CALUDE_test_scores_order_l361_36190

-- Define the scores as natural numbers
variable (J E N L : ℕ)

-- Define the theorem
theorem test_scores_order :
  -- Conditions
  (E = J) →  -- Elina's score is the same as Jasper's
  (N ≤ J) →  -- Norah's score is not higher than Jasper's
  (L > J) →  -- Liam's score is higher than Jasper's
  -- Conclusion: The order of scores from lowest to highest is N, E, L
  (N ≤ E ∧ E < L) := by
sorry

end NUMINAMATH_CALUDE_test_scores_order_l361_36190


namespace NUMINAMATH_CALUDE_sum_geq_five_x_squared_l361_36171

theorem sum_geq_five_x_squared (x : ℝ) (hx : x > 0) :
  1 + x + x^2 + x^3 + x^4 ≥ 5 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_five_x_squared_l361_36171
