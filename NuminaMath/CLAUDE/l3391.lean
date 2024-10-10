import Mathlib

namespace total_clothing_cost_l3391_339103

def shirt_cost : ℚ := 13.04
def jacket_cost : ℚ := 12.27

theorem total_clothing_cost : shirt_cost + jacket_cost = 25.31 := by
  sorry

end total_clothing_cost_l3391_339103


namespace first_group_size_l3391_339197

/-- The number of persons in the first group -/
def P : ℕ := 42

/-- The number of days the first group works -/
def days_first : ℕ := 12

/-- The number of hours per day the first group works -/
def hours_first : ℕ := 5

/-- The number of persons in the second group -/
def persons_second : ℕ := 30

/-- The number of days the second group works -/
def days_second : ℕ := 14

/-- The number of hours per day the second group works -/
def hours_second : ℕ := 6

/-- Theorem stating that P is the correct number of persons in the first group -/
theorem first_group_size :
  P * days_first * hours_first = persons_second * days_second * hours_second :=
by sorry

end first_group_size_l3391_339197


namespace find_constant_a_l3391_339171

theorem find_constant_a (t k a : ℝ) :
  (∀ x : ℝ, x^2 + 10*x + t = (x + a)^2 + k) →
  a = 5 := by
  sorry

end find_constant_a_l3391_339171


namespace quadratic_inequality_constant_value_theorem_constant_function_inequality_l3391_339119

-- 1. Prove that for all real x, x^2 + 2x + 2 ≥ 1
theorem quadratic_inequality (x : ℝ) : x^2 + 2*x + 2 ≥ 1 := by sorry

-- 2. Prove that for a > 0 and c < 0, min(3|ax^2 + bx + c| + 2) = 2
theorem constant_value_theorem (a b c : ℝ) (ha : a > 0) (hc : c < 0) :
  ∀ x, 3 * |a * x^2 + b * x + c| + 2 ≥ 2 := by sorry

-- 3. Prove that for y = ax^2 + bx + c where b > a > 0 and y ≥ 0 for all real x,
--    if (a+b+c)/(a+b) > m for all a, b, c satisfying the conditions, then m ≤ 9/8
theorem constant_function_inequality (a b c : ℝ) (hab : b > a) (ha : a > 0) 
  (h_nonneg : ∀ x, a * x^2 + b * x + c ≥ 0) 
  (h_inequality : (a + b + c) / (a + b) > 0) :
  (a + b + c) / (a + b) ≤ 9/8 := by sorry

end quadratic_inequality_constant_value_theorem_constant_function_inequality_l3391_339119


namespace part1_part2_l3391_339129

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2*a^2

/-- Part 1: The range of a when f(x) > -9 always holds -/
theorem part1 (a : ℝ) : (∀ x, f a x > -9) → a ∈ Set.Ioo (-2) 2 := by sorry

/-- Part 2: Solving the inequality f(x) > 0 with respect to x -/
theorem part2 (a : ℝ) (x : ℝ) :
  (a > 0 → (f a x > 0 ↔ x ∈ Set.Iio (-a) ∪ Set.Ioi (2*a))) ∧
  (a = 0 → (f a x > 0 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 0)) ∧
  (a < 0 → (f a x > 0 ↔ x ∈ Set.Iio (2*a) ∪ Set.Ioi (-a))) := by sorry

end part1_part2_l3391_339129


namespace distance_one_fourth_way_l3391_339140

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  orbit.perigee + fraction * (orbit.apogee - orbit.perigee)

/-- Theorem: For the given elliptical orbit, the distance from the focus to a point
    one-fourth way from perigee to apogee is 6.75 AU -/
theorem distance_one_fourth_way (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3)
    (h2 : orbit.apogee = 15) :
    distanceFromFocus orbit (1/4) = 6.75 := by
  sorry

#check distance_one_fourth_way

end distance_one_fourth_way_l3391_339140


namespace realtor_earnings_problem_l3391_339107

/-- A realtor's earnings and house sales problem -/
theorem realtor_earnings_problem 
  (base_salary : ℕ) 
  (commission_rate : ℚ) 
  (num_houses : ℕ) 
  (total_earnings : ℕ) 
  (house_a_cost : ℕ) :
  base_salary = 3000 →
  commission_rate = 2 / 100 →
  num_houses = 3 →
  total_earnings = 8000 →
  house_a_cost = 60000 →
  ∃ (house_b_cost house_c_cost : ℕ),
    house_b_cost = 3 * house_a_cost ∧
    ∃ (subtracted_amount : ℕ),
      house_c_cost = 2 * house_a_cost - subtracted_amount ∧
      house_a_cost + house_b_cost + house_c_cost = 
        (total_earnings - base_salary) / commission_rate ∧
      subtracted_amount = 110000 :=
by sorry

end realtor_earnings_problem_l3391_339107


namespace solution_form_l3391_339163

theorem solution_form (a b c d : ℝ) 
  (h1 : a + b + c = d) 
  (h2 : 1/a + 1/b + 1/c = 1/d) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) := by
sorry

end solution_form_l3391_339163


namespace potato_distribution_l3391_339130

theorem potato_distribution (total : ℕ) (gina : ℕ) (left : ℕ) :
  total = 300 →
  gina = 69 →
  left = 47 →
  ∃ (tom : ℕ) (k : ℕ),
    tom = k * gina ∧
    total = gina + tom + (tom / 3) + left →
    tom / gina = 2 :=
by sorry

end potato_distribution_l3391_339130


namespace inequality_range_l3391_339198

theorem inequality_range (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) → 
  m ∈ Set.Icc (-6) 2 := by
sorry

end inequality_range_l3391_339198


namespace fermat_numbers_coprime_l3391_339111

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2^(2^(n-1)) + 1) (2^(2^(m-1)) + 1) = 1 := by
sorry

end fermat_numbers_coprime_l3391_339111


namespace initial_acidic_percentage_l3391_339153

/-- Proves that the initial percentage of acidic liquid is 40% given the conditions -/
theorem initial_acidic_percentage (initial_volume : ℝ) (final_concentration : ℝ) (water_removed : ℝ) : 
  initial_volume = 18 →
  final_concentration = 60 →
  water_removed = 6 →
  (initial_volume * (40 / 100) = (initial_volume - water_removed) * (final_concentration / 100)) :=
by sorry

end initial_acidic_percentage_l3391_339153


namespace difference_of_squares_special_case_l3391_339174

theorem difference_of_squares_special_case : (831 : ℤ) * 831 - 830 * 832 = 1 := by
  sorry

end difference_of_squares_special_case_l3391_339174


namespace y_value_l3391_339101

theorem y_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : y = 2 := by
  sorry

end y_value_l3391_339101


namespace common_tangent_implies_t_value_l3391_339170

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := t * Real.log x
def g (x : ℝ) : ℝ := x^2 - 1

theorem common_tangent_implies_t_value :
  ∀ t : ℝ,
  (f t 1 = g 1) →
  (deriv (f t) 1 = deriv g 1) →
  t = 2 :=
by
  sorry

end common_tangent_implies_t_value_l3391_339170


namespace remaining_drawings_l3391_339149

-- Define the given parameters
def total_markers : ℕ := 12
def drawings_per_marker : ℚ := 3/2
def drawings_already_made : ℕ := 8

-- State the theorem
theorem remaining_drawings : 
  ⌊(total_markers : ℚ) * drawings_per_marker⌋ - drawings_already_made = 10 := by
  sorry

end remaining_drawings_l3391_339149


namespace sin_zero_degrees_l3391_339102

theorem sin_zero_degrees : Real.sin (0 * π / 180) = 0 := by sorry

end sin_zero_degrees_l3391_339102


namespace price_difference_l3391_339132

/-- The original price of Liz's old car -/
def original_price : ℝ := 32500

/-- The selling price of Liz's old car as a percentage of the original price -/
def selling_percentage : ℝ := 0.80

/-- The additional amount Liz needs to buy the new car -/
def additional_amount : ℝ := 4000

/-- The price of the new car -/
def new_car_price : ℝ := 30000

/-- The theorem stating the difference between the original price of the old car and the price of the new car -/
theorem price_difference : original_price - new_car_price = 2500 := by
  sorry

end price_difference_l3391_339132


namespace ball_probability_l3391_339189

theorem ball_probability (n : ℕ) : 
  (2 : ℝ) / ((n : ℝ) + 2) = 0.4 → n = 3 := by
  sorry

end ball_probability_l3391_339189


namespace no_integer_root_quadratic_pair_l3391_339146

theorem no_integer_root_quadratic_pair :
  ¬ ∃ (a b c : ℤ),
    (∃ (x₁ x₂ : ℤ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) ∧
    (∃ (y₁ y₂ : ℤ), (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0 ∧ y₁ ≠ y₂) :=
by sorry

end no_integer_root_quadratic_pair_l3391_339146


namespace second_column_halving_matrix_l3391_339188

def halve_second_column (N : Matrix (Fin 2) (Fin 2) ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ i j, (N * M) i j = if j = 1 then (1/2 : ℝ) * M i j else M i j

theorem second_column_halving_matrix :
  ∃ N : Matrix (Fin 2) (Fin 2) ℝ, 
    (N 0 0 = 1 ∧ N 0 1 = 0 ∧ N 1 0 = 0 ∧ N 1 1 = 1/2) ∧
    ∀ M : Matrix (Fin 2) (Fin 2) ℝ, halve_second_column N M :=
by
  sorry

end second_column_halving_matrix_l3391_339188


namespace problem_1_problem_2_l3391_339152

-- Part 1
theorem problem_1 : 8 - (-4) / (2^2) * 3 = 11 := by sorry

-- Part 2
theorem problem_2 (x : ℝ) : 2 * x^2 + 3 * (2*x - x^2) = -x^2 + 6*x := by sorry

end problem_1_problem_2_l3391_339152


namespace total_highlighters_count_l3391_339143

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 4

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 2

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 5

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

/-- Theorem stating that the total number of highlighters is 11 -/
theorem total_highlighters_count : total_highlighters = 11 := by
  sorry

end total_highlighters_count_l3391_339143


namespace total_cost_of_materials_l3391_339124

/-- The total cost of materials for a construction company -/
theorem total_cost_of_materials
  (gravel_quantity : ℝ)
  (gravel_price : ℝ)
  (sand_quantity : ℝ)
  (sand_price : ℝ)
  (h1 : gravel_quantity = 5.91)
  (h2 : gravel_price = 30.50)
  (h3 : sand_quantity = 8.11)
  (h4 : sand_price = 40.50) :
  gravel_quantity * gravel_price + sand_quantity * sand_price = 508.71 := by
  sorry

end total_cost_of_materials_l3391_339124


namespace multiply_by_one_seventh_squared_l3391_339199

theorem multiply_by_one_seventh_squared (x : ℝ) : x * (1/7)^2 = 7^3 ↔ x = 16807 := by
  sorry

end multiply_by_one_seventh_squared_l3391_339199


namespace inequality_proof_l3391_339134

theorem inequality_proof (x y : ℝ) (a : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin a)^2 * y^(Real.cos a)^2 < x + y := by
  sorry

end inequality_proof_l3391_339134


namespace min_value_of_expression_l3391_339167

theorem min_value_of_expression (x y : ℝ) : (x^2*y - 2)^2 + (x^2 + y)^2 ≥ 4 := by
  sorry

end min_value_of_expression_l3391_339167


namespace min_value_of_max_absolute_min_value_achievable_l3391_339177

theorem min_value_of_max_absolute (a b : ℝ) : 
  max (max (|a + b|) (|a - b|)) (|1 - b|) ≥ (1/2 : ℝ) := by sorry

theorem min_value_achievable : 
  ∃ (a b : ℝ), max (max (|a + b|) (|a - b|)) (|1 - b|) = (1/2 : ℝ) := by sorry

end min_value_of_max_absolute_min_value_achievable_l3391_339177


namespace root_difference_quadratic_l3391_339191

theorem root_difference_quadratic (p : ℝ) : 
  let r := (2*p + Real.sqrt (9 : ℝ))
  let s := (2*p - Real.sqrt (9 : ℝ))
  r - s = 6 := by
sorry

end root_difference_quadratic_l3391_339191


namespace elixir_combinations_eq_18_l3391_339109

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of enchanted gems available. -/
def num_gems : ℕ := 6

/-- Represents the number of incompatible gem-herb pairs. -/
def num_incompatible : ℕ := 6

/-- Calculates the number of ways the sorcerer can prepare the elixir. -/
def num_elixir_combinations : ℕ := num_herbs * num_gems - num_incompatible

/-- Proves that the number of ways to prepare the elixir is 18. -/
theorem elixir_combinations_eq_18 : num_elixir_combinations = 18 := by
  sorry

end elixir_combinations_eq_18_l3391_339109


namespace jill_watching_time_l3391_339184

/-- The total time Jill spent watching shows -/
def total_time (first_show_duration : ℕ) (multiplier : ℕ) : ℕ :=
  first_show_duration + first_show_duration * multiplier

/-- Proof that Jill spent 150 minutes watching shows -/
theorem jill_watching_time : total_time 30 4 = 150 := by
  sorry

end jill_watching_time_l3391_339184


namespace stock_decrease_duration_l3391_339105

/-- The number of bicycles the stock decreases each month -/
def monthly_decrease : ℕ := 2

/-- The number of months from January 1 to September 1 -/
def months_jan_to_sep : ℕ := 8

/-- The total decrease in bicycles from January 1 to September 1 -/
def total_decrease : ℕ := 18

/-- The number of months the stock has been decreasing -/
def months_decreasing : ℕ := 1

theorem stock_decrease_duration :
  monthly_decrease * months_decreasing + monthly_decrease * months_jan_to_sep = total_decrease :=
by sorry

end stock_decrease_duration_l3391_339105


namespace intersection_sum_l3391_339141

/-- Given two lines y = mx + 3 and y = 4x + b intersecting at (8, 14),
    where m and b are constants, prove that b + m = -133/8 -/
theorem intersection_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 3 ↔ y = 4 * x + b) → 
  (14 : ℚ) = m * 8 + 3 → 
  (14 : ℚ) = 4 * 8 + b → 
  b + m = -133/8 := by sorry

end intersection_sum_l3391_339141


namespace impossible_time_reduction_l3391_339151

theorem impossible_time_reduction (initial_speed : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → time_reduction = 1 → ¬ ∃ (new_speed : ℝ), 
    new_speed > 0 ∧ (1 / new_speed) * 60 = (1 / initial_speed) * 60 - time_reduction :=
by
  sorry

end impossible_time_reduction_l3391_339151


namespace decreasing_function_on_positive_reals_l3391_339154

/-- The function f(x) = 9 - x² is decreasing on the interval (0, +∞) -/
theorem decreasing_function_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → (9 - x^2 : ℝ) > (9 - y^2 : ℝ) := by
  sorry

end decreasing_function_on_positive_reals_l3391_339154


namespace arithmetic_calculation_l3391_339193

theorem arithmetic_calculation : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end arithmetic_calculation_l3391_339193


namespace medical_team_selection_l3391_339172

theorem medical_team_selection (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 6)
  (h2 : female_doctors = 3)
  (h3 : team_size = 5) :
  (Nat.choose (male_doctors + female_doctors) team_size) - 
  (Nat.choose male_doctors team_size) = 120 := by
  sorry

end medical_team_selection_l3391_339172


namespace triangle_angle_theorem_l3391_339117

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The main theorem stating that if tan(A+B)(1-tan A tan B) = (√3 sin C) / (sin A cos B), then A = π/3. -/
theorem triangle_angle_theorem (t : Triangle) :
  Real.tan (t.A + t.B) * (1 - Real.tan t.A * Real.tan t.B) = (Real.sqrt 3 * Real.sin t.C) / (Real.sin t.A * Real.cos t.B) →
  t.A = π / 3 :=
by sorry

end triangle_angle_theorem_l3391_339117


namespace diophantine_equation_solution_l3391_339173

theorem diophantine_equation_solution :
  ∀ (x y : ℤ), 3 * x + 5 * y = 7 ↔ ∃ k : ℤ, x = 4 + 5 * k ∧ y = -1 - 3 * k :=
by sorry

end diophantine_equation_solution_l3391_339173


namespace geometric_sequence_general_term_l3391_339128

/-- A geometric sequence with a_3 = 2 and a_6 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n) ∧ 
  a 3 = 2 ∧ 
  a 6 = 16

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 2^(n - 2)

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  ∀ n : ℕ, a n = general_term n :=
by
  sorry

end geometric_sequence_general_term_l3391_339128


namespace segment_polynomial_l3391_339125

/-- Given a line segment AB with point T, prove that x^2 - 6√2x + 16 has roots equal to AT and TB lengths -/
theorem segment_polynomial (AB : ℝ) (T : ℝ) (h1 : 0 < T ∧ T < AB) 
  (h2 : AB - T = (1/2) * T) (h3 : T * (AB - T) = 16) :
  ∃ (AT TB : ℝ), AT = T ∧ TB = AB - T ∧ 
  (∀ x : ℝ, x^2 - 6 * Real.sqrt 2 * x + 16 = 0 ↔ (x = AT ∨ x = TB)) :=
sorry

end segment_polynomial_l3391_339125


namespace compare_negative_decimals_l3391_339186

theorem compare_negative_decimals : -0.5 > -0.75 := by
  sorry

end compare_negative_decimals_l3391_339186


namespace parabola_equation_part1_parabola_equation_part2_l3391_339123

-- Part 1
theorem parabola_equation_part1 (a b c : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (1, 10) = (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c) →
  (-1, -2) = (-1, a * (-1)^2 + b * (-1) + c) →
  (∀ x y : ℝ, y = -3 * (x - 1)^2 + 10) := by sorry

-- Part 2
theorem parabola_equation_part2 (a b c : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (0 = a * (-1)^2 + b * (-1) + c) →
  (0 = a * 3^2 + b * 3 + c) →
  (3 = c) →
  (∀ x y : ℝ, y = -x^2 + 2 * x + 3) := by sorry

end parabola_equation_part1_parabola_equation_part2_l3391_339123


namespace binomial_sum_l3391_339114

theorem binomial_sum : 
  let p := Nat.choose 20 6
  let q := Nat.choose 20 5
  p + q = 62016 := by
  sorry

end binomial_sum_l3391_339114


namespace hardly_arrangements_l3391_339120

/-- The number of letters in the word "hardly" -/
def word_length : Nat := 6

/-- The number of letters to be arranged (excluding the fixed 'd') -/
def letters_to_arrange : Nat := 5

/-- Factorial function -/
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem hardly_arrangements :
  factorial letters_to_arrange = 120 :=
by sorry

end hardly_arrangements_l3391_339120


namespace special_function_unique_l3391_339175

/-- A function satisfying the given conditions -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  g 1 = 2 ∧ ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x + g y)

/-- Theorem stating that any function satisfying the conditions must be g(x) = 2x -/
theorem special_function_unique (g : ℝ → ℝ) (h : SpecialFunction g) :
    ∀ x : ℝ, g x = 2 * x := by
  sorry

end special_function_unique_l3391_339175


namespace complementary_angles_adjustment_l3391_339159

-- Define the ratio of the two complementary angles
def angle_ratio : ℚ := 3 / 7

-- Define the increase percentage for the smaller angle
def small_angle_increase : ℚ := 1 / 5

-- Function to calculate the decrease percentage for the larger angle
def large_angle_decrease (ratio : ℚ) (increase : ℚ) : ℚ :=
  1 - (90 - 90 * ratio / (1 + ratio) * (1 + increase)) / (90 * ratio / (1 + ratio))

-- Theorem statement
theorem complementary_angles_adjustment :
  large_angle_decrease angle_ratio small_angle_increase = 43 / 500 := by
  sorry

#eval large_angle_decrease angle_ratio small_angle_increase

end complementary_angles_adjustment_l3391_339159


namespace blue_pill_cost_l3391_339185

def treatment_duration : ℕ := 21 -- 3 weeks * 7 days

def daily_blue_pills : ℕ := 2
def daily_orange_pills : ℕ := 1

def total_cost : ℕ := 966

theorem blue_pill_cost (orange_pill_cost : ℕ) 
  (h1 : orange_pill_cost + 2 = 16) 
  (h2 : (daily_blue_pills * (orange_pill_cost + 2) + daily_orange_pills * orange_pill_cost) * treatment_duration = total_cost) : 
  orange_pill_cost + 2 = 16 := by
  sorry

end blue_pill_cost_l3391_339185


namespace lapis_share_l3391_339100

def treasure_problem (total_treasure : ℚ) (fonzie_contribution : ℚ) (aunt_bee_contribution : ℚ) (lapis_contribution : ℚ) : Prop :=
  let total_contribution := fonzie_contribution + aunt_bee_contribution + lapis_contribution
  let lapis_fraction := lapis_contribution / total_contribution
  lapis_fraction * total_treasure = 337500

theorem lapis_share :
  treasure_problem 900000 7000 8000 9000 := by
  sorry

#check lapis_share

end lapis_share_l3391_339100


namespace baseball_team_grouping_l3391_339194

theorem baseball_team_grouping (new_players returning_players num_groups : ℕ) 
  (h1 : new_players = 4)
  (h2 : returning_players = 6)
  (h3 : num_groups = 2) :
  (new_players + returning_players) / num_groups = 5 := by
  sorry

end baseball_team_grouping_l3391_339194


namespace vertical_angles_are_congruent_l3391_339147

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def are_vertical_angles (α β : Angle) : Prop := sorry

/-- Two angles are congruent if they have the same measure. -/
def are_congruent (α β : Angle) : Prop := sorry

/-- If two angles are vertical angles, then these two angles are congruent. -/
theorem vertical_angles_are_congruent (α β : Angle) : 
  are_vertical_angles α β → are_congruent α β := by sorry

end vertical_angles_are_congruent_l3391_339147


namespace problem_solution_l3391_339196

theorem problem_solution (x : ℝ) (h : x - 1/x = 5) : x^2 - 1/x^2 = 35 := by
  sorry

end problem_solution_l3391_339196


namespace trigonometric_calculation_quadratic_equation_solution_l3391_339164

-- Problem 1
theorem trigonometric_calculation :
  3 * Real.tan (30 * π / 180) - Real.tan (45 * π / 180)^2 + 2 * Real.sin (60 * π / 180) = 2 * Real.sqrt 3 - 1 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ (3*x - 1)*(x + 2) - (11*x - 4)
  (∃ x : ℝ, f x = 0) ↔ (f ((3 + Real.sqrt 3) / 3) = 0 ∧ f ((3 - Real.sqrt 3) / 3) = 0) := by
  sorry

end trigonometric_calculation_quadratic_equation_solution_l3391_339164


namespace second_number_20th_row_l3391_339110

/-- The first number in the nth row of the sequence -/
def first_number (n : ℕ) : ℕ := (n + 1)^2 - 1

/-- The second number in the nth row of the sequence -/
def second_number (n : ℕ) : ℕ := first_number n - 1

/-- Theorem stating that the second number in the 20th row is 439 -/
theorem second_number_20th_row : second_number 20 = 439 := by sorry

end second_number_20th_row_l3391_339110


namespace largest_812_double_l3391_339155

/-- Converts a natural number to its base-8 representation as a list of digits --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-12 number --/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is an 8-12 double --/
def is812Double (n : ℕ) : Prop :=
  fromBase12 (toBase8 n) = 3 * n

theorem largest_812_double :
  ∀ n : ℕ, n > 3 → ¬(is812Double n) :=
sorry

end largest_812_double_l3391_339155


namespace leila_toy_donation_l3391_339121

theorem leila_toy_donation (leila_bags : ℕ) (mohamed_bags : ℕ) (mohamed_toys_per_bag : ℕ) (extra_toys : ℕ) :
  leila_bags = 2 →
  mohamed_bags = 3 →
  mohamed_toys_per_bag = 19 →
  extra_toys = 7 →
  mohamed_bags * mohamed_toys_per_bag = leila_bags * (mohamed_bags * mohamed_toys_per_bag - extra_toys) / leila_bags →
  (mohamed_bags * mohamed_toys_per_bag - extra_toys) / leila_bags = 25 :=
by sorry

end leila_toy_donation_l3391_339121


namespace tree_distance_l3391_339195

/-- Given 8 equally spaced trees along a road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first
    and last tree is 175 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let distance_between (i j : ℕ) := d * (j - i : ℝ) / 4
  distance_between 1 n = 175 := by
  sorry

end tree_distance_l3391_339195


namespace distance_difference_l3391_339118

theorem distance_difference (john_distance jill_distance jim_distance : ℝ) : 
  john_distance = 15 →
  jim_distance = 0.2 * jill_distance →
  jim_distance = 2 →
  john_distance - jill_distance = 5 :=
by
  sorry

end distance_difference_l3391_339118


namespace relay_station_problem_l3391_339104

theorem relay_station_problem (x : ℝ) (h : x > 3) : 
  (∃ (slow_speed fast_speed : ℝ),
    slow_speed > 0 ∧ 
    fast_speed > 0 ∧
    fast_speed = 2 * slow_speed ∧
    900 / (x + 1) = slow_speed ∧
    900 / (x - 3) = fast_speed) ↔ 
  2 * (900 / (x + 1)) = 900 / (x - 3) :=
by sorry

end relay_station_problem_l3391_339104


namespace invoice_problem_l3391_339156

theorem invoice_problem (x y : ℚ) : 
  (0.3 < x) ∧ (x < 0.4) ∧ 
  (7000 < y) ∧ (y < 8000) ∧ 
  (y * 100 - (y.floor * 100) = 65) ∧
  (237 * x = y) →
  (x = 0.31245 ∧ y = 7400.65) := by
sorry

end invoice_problem_l3391_339156


namespace only_one_true_proposition_l3391_339106

theorem only_one_true_proposition :
  (∃! n : Fin 4, 
    (n = 0 → (∀ a b : ℝ, a > b ↔ a^2 > b^2)) ∧
    (n = 1 → (∀ a b : ℝ, a > b ↔ a^3 > b^3)) ∧
    (n = 2 → (∀ a b : ℝ, a > b → |a| > |b|)) ∧
    (n = 3 → (∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a > b))) :=
by sorry

end only_one_true_proposition_l3391_339106


namespace closest_integer_to_expression_l3391_339144

theorem closest_integer_to_expression : ∃ n : ℤ, 
  n = round ((3/2 : ℚ) * (4/9 : ℚ) + (7/2 : ℚ)) ∧ n = 4 := by
  sorry

end closest_integer_to_expression_l3391_339144


namespace greatest_divisor_four_consecutive_integers_l3391_339158

theorem greatest_divisor_four_consecutive_integers (n : ℕ+) :
  ∃ (k : ℕ), k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ (m : ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3)) → m ≤ k :=
by sorry

end greatest_divisor_four_consecutive_integers_l3391_339158


namespace interest_years_satisfies_equation_l3391_339126

/-- The number of years that satisfies the compound and simple interest difference equation -/
def interest_years : ℕ := 2

/-- The principal amount in rupees -/
def principal : ℚ := 3600

/-- The annual interest rate as a decimal -/
def rate : ℚ := 1/10

/-- The difference between compound and simple interest in rupees -/
def interest_difference : ℚ := 36

/-- The equation that relates the number of years to the interest difference -/
def interest_equation (n : ℕ) : Prop :=
  (1 + rate) ^ n - 1 - rate * n = interest_difference / principal

theorem interest_years_satisfies_equation : 
  interest_equation interest_years :=
sorry

end interest_years_satisfies_equation_l3391_339126


namespace greater_number_problem_l3391_339190

theorem greater_number_problem (A B : ℕ+) : 
  (Nat.gcd A B = 11) → 
  (A * B = 363) → 
  (max A B = 33) := by
sorry

end greater_number_problem_l3391_339190


namespace product_equals_specific_number_l3391_339160

theorem product_equals_specific_number (A B : ℕ) :
  990 * 991 * 992 * 993 = 966428000000 + A * 10000000 + 910000 + B * 100 + 40 →
  A * 10 + B = 50 := by
  sorry

end product_equals_specific_number_l3391_339160


namespace smallest_prime_sum_of_three_primes_l3391_339133

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is the sum of three different primes
def isSumOfThreeDifferentPrimes (n : Nat) : Prop :=
  ∃ (p q r : Nat), isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p + q + r = n

-- State the theorem
theorem smallest_prime_sum_of_three_primes :
  isPrime 19 ∧ 
  isSumOfThreeDifferentPrimes 19 ∧ 
  ∀ n : Nat, n < 19 → ¬(isPrime n ∧ isSumOfThreeDifferentPrimes n) :=
by sorry

end smallest_prime_sum_of_three_primes_l3391_339133


namespace houses_in_block_is_five_l3391_339116

/-- The number of houses in a block -/
def houses_in_block : ℕ := 5

/-- The number of candies received from each house -/
def candies_per_house : ℕ := 7

/-- The total number of candies received from each block -/
def candies_per_block : ℕ := 35

/-- Theorem: The number of houses in a block is 5 -/
theorem houses_in_block_is_five :
  houses_in_block = candies_per_block / candies_per_house :=
by sorry

end houses_in_block_is_five_l3391_339116


namespace total_handshakes_l3391_339166

-- Define the number of people in each group
def group_a : Nat := 25  -- people who all know each other
def group_b : Nat := 10  -- people who know no one
def group_c : Nat := 5   -- people who only know each other

-- Define the total number of people
def total_people : Nat := group_a + group_b + group_c

-- Define the function to calculate handshakes between two groups
def handshakes_between (group1 : Nat) (group2 : Nat) : Nat := group1 * group2

-- Define the function to calculate handshakes within a group
def handshakes_within (group : Nat) : Nat := group * (group - 1) / 2

-- Theorem statement
theorem total_handshakes : 
  handshakes_between group_a group_b + 
  handshakes_between group_a group_c + 
  handshakes_between group_b group_c + 
  handshakes_within group_b = 470 := by
  sorry

end total_handshakes_l3391_339166


namespace f_derivative_at_one_l3391_339138

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem f_derivative_at_one :
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = x / (1 + x) →
  HasDerivAt f (-1/4) 1 :=
by sorry

end f_derivative_at_one_l3391_339138


namespace inequality_proof_l3391_339127

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a + b)^n - a^n - b^n ≥ (2^n - 2) / 2^(n - 2) * a * b * (a + b)^(n - 2) := by
  sorry

end inequality_proof_l3391_339127


namespace sin_15_cos_15_equals_quarter_l3391_339108

theorem sin_15_cos_15_equals_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/4 := by
  sorry

end sin_15_cos_15_equals_quarter_l3391_339108


namespace exists_non_adjacent_divisible_l3391_339169

/-- A circular arrangement of seven natural numbers -/
def CircularArrangement := Fin 7 → ℕ+

/-- Predicate to check if one number divides another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Two positions in the circular arrangement are adjacent -/
def adjacent (i j : Fin 7) : Prop := i = j + 1 ∨ j = i + 1 ∨ (i = 0 ∧ j = 6) ∨ (j = 0 ∧ i = 6)

/-- Two positions in the circular arrangement are non-adjacent -/
def non_adjacent (i j : Fin 7) : Prop := ¬(adjacent i j) ∧ i ≠ j

/-- The main theorem -/
theorem exists_non_adjacent_divisible (arr : CircularArrangement) 
  (h : ∀ i j : Fin 7, adjacent i j → (divides (arr i) (arr j) ∨ divides (arr j) (arr i))) :
  ∃ i j : Fin 7, non_adjacent i j ∧ (divides (arr i) (arr j) ∨ divides (arr j) (arr i)) := by
  sorry

end exists_non_adjacent_divisible_l3391_339169


namespace kabadi_kho_kho_players_l3391_339122

theorem kabadi_kho_kho_players (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ) 
  (h_total : total = 50)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 40) :
  total = kabadi + kho_kho_only - 0 :=
by sorry

end kabadi_kho_kho_players_l3391_339122


namespace intersection_of_odd_integers_and_open_interval_l3391_339139

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_odd_integers_and_open_interval :
  A ∩ B = {1, 3} := by sorry

end intersection_of_odd_integers_and_open_interval_l3391_339139


namespace greatest_4digit_base7_divisible_by_7_l3391_339145

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.reverse.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Checks if a number is divisible by 7 --/
def isDivisibleBy7 (n : Nat) : Bool :=
  n % 7 = 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Checks if a list represents a 4-digit base 7 number --/
def is4DigitBase7 (n : List Nat) : Bool :=
  n.length = 4 && n.all (· < 7) && n.head! ≠ 0

theorem greatest_4digit_base7_divisible_by_7 :
  let n := [6, 6, 6, 0]
  is4DigitBase7 n ∧
  isDivisibleBy7 (toDecimal n) ∧
  ∀ m, is4DigitBase7 m → isDivisibleBy7 (toDecimal m) → toDecimal m ≤ toDecimal n :=
by sorry

end greatest_4digit_base7_divisible_by_7_l3391_339145


namespace dog_journey_distance_l3391_339180

/-- 
Given a journey where:
- The total time is 2 hours
- The first half of the distance is traveled at 10 km/h
- The second half of the distance is traveled at 5 km/h
Prove that the total distance traveled is 40/3 km
-/
theorem dog_journey_distance : 
  ∀ (total_distance : ℝ),
  (total_distance / 20 + total_distance / 10 = 2) →
  total_distance = 40 / 3 := by
  sorry

end dog_journey_distance_l3391_339180


namespace sum_transformation_l3391_339142

theorem sum_transformation (xs : List ℝ) 
  (h1 : xs.sum = 40)
  (h2 : (xs.map (λ x => 1 - x)).sum = 20) :
  (xs.map (λ x => 1 + x)).sum = 100 := by
  sorry

end sum_transformation_l3391_339142


namespace hyperbola_foci_and_incenter_l3391_339182

/-- Definition of the hyperbola C -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-5, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (5, 0)

/-- Definition of a point being on the left branch of the hyperbola -/
def on_left_branch (x y : ℝ) : Prop :=
  hyperbola x y ∧ x < 0

/-- The center of the incircle of a triangle -/
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- Definition of incenter calculation

theorem hyperbola_foci_and_incenter :
  (∀ x y : ℝ, hyperbola x y → 
    (F₁ = (-5, 0) ∧ F₂ = (5, 0))) ∧
  (∀ x y : ℝ, on_left_branch x y →
    (incenter F₁ (x, y) F₂).1 = -3) :=
sorry

end hyperbola_foci_and_incenter_l3391_339182


namespace digits_of_2_15_times_5_6_l3391_339162

/-- The number of digits in 2^15 * 5^6 is 9 -/
theorem digits_of_2_15_times_5_6 : (Nat.digits 10 (2^15 * 5^6)).length = 9 := by
  sorry

end digits_of_2_15_times_5_6_l3391_339162


namespace profit_increase_l3391_339150

theorem profit_increase (march_profit : ℝ) (march_profit_pos : march_profit > 0) :
  let april_profit := march_profit * 1.35
  let may_profit := april_profit * 0.8
  let june_profit := may_profit * 1.5
  (june_profit - march_profit) / march_profit = 0.62 := by
  sorry

end profit_increase_l3391_339150


namespace calculate_second_discount_other_discount_percentage_l3391_339179

/-- Given an article with a list price and two successive discounts, 
    calculate the second discount percentage. -/
theorem calculate_second_discount 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (first_discount : ℝ) : ℝ :=
  let price_after_first_discount := list_price * (1 - first_discount / 100)
  let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount

/-- Prove that for an article with a list price of 70 units, 
    after applying two successive discounts, one of which is 10%, 
    resulting in a final price of 56.16 units, 
    the other discount percentage is approximately 10.857%. -/
theorem other_discount_percentage : 
  let result := calculate_second_discount 70 56.16 10
  ∃ ε > 0, abs (result - 10.857) < ε :=
sorry

end calculate_second_discount_other_discount_percentage_l3391_339179


namespace dinner_bill_proof_l3391_339112

/-- The number of friends in the group -/
def num_friends : ℕ := 10

/-- The additional amount each paying friend contributes to cover the non-paying friend -/
def extra_payment : ℚ := 4

/-- The total bill for the group dinner -/
def total_bill : ℚ := 360

theorem dinner_bill_proof :
  ∃ (individual_share : ℚ),
    (num_friends - 1 : ℚ) * (individual_share + extra_payment) = total_bill :=
by sorry

end dinner_bill_proof_l3391_339112


namespace fraction_evaluation_l3391_339168

theorem fraction_evaluation (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  2 / (a - b) = 1 := by sorry

end fraction_evaluation_l3391_339168


namespace b_33_mod_35_l3391_339161

-- Definition of b_n
def b (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem b_33_mod_35 : b 33 % 35 = 21 := by sorry

end b_33_mod_35_l3391_339161


namespace tiling_ways_2x12_l3391_339136

/-- The number of ways to tile a 2 × n rectangle with 1 × 2 dominoes -/
def tiling_ways : ℕ → ℕ
  | 0 => 0  -- Added for completeness
  | 1 => 1
  | 2 => 2
  | n+3 => tiling_ways (n+2) + tiling_ways (n+1)

/-- Theorem: The number of ways to tile a 2 × 12 rectangle with 1 × 2 dominoes is 233 -/
theorem tiling_ways_2x12 : tiling_ways 12 = 233 := by
  sorry


end tiling_ways_2x12_l3391_339136


namespace min_product_of_reciprocal_sum_l3391_339148

theorem min_product_of_reciprocal_sum (x y : ℕ+) : 
  (1 : ℚ) / x + 1 / (3 * y) = 1 / 9 → 
  ∃ (a b : ℕ+), (1 : ℚ) / a + 1 / (3 * b) = 1 / 9 ∧ a * b = 108 ∧ 
  ∀ (c d : ℕ+), (1 : ℚ) / c + 1 / (3 * d) = 1 / 9 → c * d ≥ 108 := by
sorry

end min_product_of_reciprocal_sum_l3391_339148


namespace applicants_age_standard_deviation_l3391_339187

/-- The standard deviation of applicants' ages given specific conditions -/
theorem applicants_age_standard_deviation 
  (average_age : ℝ)
  (max_different_ages : ℕ)
  (h_average : average_age = 30)
  (h_max_ages : max_different_ages = 15)
  (h_range : max_different_ages = 2 * standard_deviation)
  (standard_deviation : ℝ) :
  standard_deviation = 7.5 := by
  sorry

end applicants_age_standard_deviation_l3391_339187


namespace geometric_sequence_problem_l3391_339137

/-- Given a geometric sequence {aₙ} where aₙ ∈ ℝ, if a₃ and a₁₁ are the two roots of the equation 3x² - 25x + 27 = 0, then a₇ = 3. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_roots : 3 * (a 3)^2 - 25 * (a 3) + 27 = 0 ∧ 3 * (a 11)^2 - 25 * (a 11) + 27 = 0) :
  a 7 = 3 := by
  sorry

end geometric_sequence_problem_l3391_339137


namespace hypotenuse_length_l3391_339165

def right_triangle_hypotenuse (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 36 ∧  -- perimeter condition
  (1/2) * a * b = 24 ∧  -- area condition
  a^2 + b^2 = c^2  -- Pythagorean theorem

theorem hypotenuse_length :
  ∃ a b c : ℝ, right_triangle_hypotenuse a b c ∧ c = 50/3 :=
by
  sorry

end hypotenuse_length_l3391_339165


namespace N2O3_molecular_weight_N2O3_is_limiting_reactant_l3391_339183

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of N2O3
def N2O3_formula : Nat × Nat := (2, 3)

-- Define the number of moles for each reactant
def moles_N2O3 : ℝ := 3
def moles_SO2 : ℝ := 4

-- Define the function to calculate molecular weight
def molecular_weight (n_atoms_N n_atoms_O : Nat) : ℝ :=
  (n_atoms_N : ℝ) * atomic_weight_N + (n_atoms_O : ℝ) * atomic_weight_O

-- Define the function to determine the limiting reactant
def is_limiting_reactant (moles_A moles_B : ℝ) (coeff_A coeff_B : Nat) : Prop :=
  moles_A / (coeff_A : ℝ) < moles_B / (coeff_B : ℝ)

-- Theorem statements
theorem N2O3_molecular_weight :
  molecular_weight N2O3_formula.1 N2O3_formula.2 = 76.02 := by sorry

theorem N2O3_is_limiting_reactant :
  is_limiting_reactant moles_N2O3 moles_SO2 1 1 := by sorry

end N2O3_molecular_weight_N2O3_is_limiting_reactant_l3391_339183


namespace notebooks_given_to_paula_notebooks_given_to_paula_is_five_l3391_339131

theorem notebooks_given_to_paula (gerald_notebooks : ℕ) (jack_initial_extra : ℕ) 
  (given_to_mike : ℕ) (jack_remaining : ℕ) : ℕ :=
  let jack_initial := gerald_notebooks + jack_initial_extra
  let jack_after_paula := jack_remaining + given_to_mike
  let given_to_paula := jack_initial - jack_after_paula
  given_to_paula

theorem notebooks_given_to_paula_is_five :
  notebooks_given_to_paula 8 13 6 10 = 5 := by sorry

end notebooks_given_to_paula_notebooks_given_to_paula_is_five_l3391_339131


namespace expression_evaluation_l3391_339157

/-- Given x = y + z and y > z > 0, prove that ((x+y)^z + (x+z)^y) / (y^z + z^y) = 2^y + 2^z -/
theorem expression_evaluation (x y z : ℝ) 
  (h1 : x = y + z) 
  (h2 : y > z) 
  (h3 : z > 0) : 
  ((x + y)^z + (x + z)^y) / (y^z + z^y) = 2^y + 2^z :=
sorry

end expression_evaluation_l3391_339157


namespace linear_equation_negative_root_m_range_l3391_339115

theorem linear_equation_negative_root_m_range 
  (m : ℝ) 
  (h : ∃ x : ℝ, (3 * x - m + 1 = 2 * x - 1) ∧ (x < 0)) : 
  m < 2 := by
  sorry

end linear_equation_negative_root_m_range_l3391_339115


namespace ellipse_properties_and_max_area_l3391_339192

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  ecc : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_ecc : ecc = 2 * Real.sqrt 2 / 3
  h_vertex : b = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ

/-- Triangle formed by intersection points and vertex -/
def triangle_area (E : Ellipse) (l : IntersectingLine E) : ℝ := sorry

/-- Theorem stating the properties of the ellipse and maximum triangle area -/
theorem ellipse_properties_and_max_area (E : Ellipse) :
  E.a = 3 ∧
  (∃ (l : IntersectingLine E), ∀ (l' : IntersectingLine E),
    triangle_area E l ≥ triangle_area E l') ∧
  (∃ (l : IntersectingLine E), triangle_area E l = 27/8) :=
sorry

end ellipse_properties_and_max_area_l3391_339192


namespace min_value_of_expression_l3391_339178

theorem min_value_of_expression (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 6) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 := by
  sorry

end min_value_of_expression_l3391_339178


namespace right_triangle_existence_l3391_339181

theorem right_triangle_existence (α β : ℝ) :
  (∃ (x y z h : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧ h > 0 ∧
    x^2 + y^2 = z^2 ∧
    x * y = z * h ∧
    x - y = α ∧
    z - h = β) ↔
  β > α :=
by sorry

end right_triangle_existence_l3391_339181


namespace triangle_abc_properties_l3391_339113

-- Define the triangle ABC
theorem triangle_abc_properties (A B C : ℝ) (AB BC : ℝ) :
  AB = Real.sqrt 3 →
  BC = 2 →
  -- Part I
  (Real.cos B = -1/2 → Real.sin C = Real.sqrt 3 / 2) ∧
  -- Part II
  (∃ (lower upper : ℝ), lower = 0 ∧ upper = 2 * Real.pi / 3 ∧
    ∀ (x : ℝ), lower < C ∧ C ≤ upper) :=
by sorry

end triangle_abc_properties_l3391_339113


namespace blue_highlighters_count_l3391_339176

def total_highlighters : ℕ := 33
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15

theorem blue_highlighters_count :
  total_highlighters - (pink_highlighters + yellow_highlighters) = 8 :=
by sorry

end blue_highlighters_count_l3391_339176


namespace average_of_distinct_t_is_22_3_l3391_339135

/-- Given a polynomial x^2 - 6x + t with only positive integer roots,
    this function returns the average of all distinct possible values of t. -/
def averageOfDistinctT : ℚ :=
  22 / 3

/-- The polynomial x^2 - 6x + t has only positive integer roots. -/
axiom has_positive_integer_roots (t : ℤ) : 
  ∃ (r₁ r₂ : ℕ+), r₁.val * r₁.val - 6 * r₁.val + t = 0 ∧ 
                  r₂.val * r₂.val - 6 * r₂.val + t = 0

/-- The main theorem stating that the average of all distinct possible values of t
    for the polynomial x^2 - 6x + t with only positive integer roots is 22/3. -/
theorem average_of_distinct_t_is_22_3 :
  averageOfDistinctT = 22 / 3 :=
sorry

end average_of_distinct_t_is_22_3_l3391_339135
