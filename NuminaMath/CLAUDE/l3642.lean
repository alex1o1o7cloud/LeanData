import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3642_364229

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3642_364229


namespace NUMINAMATH_CALUDE_fraction_power_four_l3642_364218

theorem fraction_power_four : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_fraction_power_four_l3642_364218


namespace NUMINAMATH_CALUDE_similar_cuts_possible_equilateral_cuts_impossible_l3642_364230

-- Define a triangular prism
structure TriangularPrism :=
  (base : Set (ℝ × ℝ × ℝ))
  (height : ℝ)

-- Define a cut on the prism
structure Cut :=
  (shape : Set (ℝ × ℝ × ℝ))
  (is_triangular : Bool)

-- Define similarity between two cuts
def are_similar (c1 c2 : Cut) : Prop := sorry

-- Define equality between two cuts
def are_equal (c1 c2 : Cut) : Prop := sorry

-- Define if a cut touches the base
def touches_base (c : Cut) (p : TriangularPrism) : Prop := sorry

-- Define if two cuts touch each other
def cuts_touch (c1 c2 : Cut) : Prop := sorry

-- Theorem for part (a)
theorem similar_cuts_possible (p : TriangularPrism) :
  ∃ (c1 c2 : Cut),
    c1.is_triangular ∧
    c2.is_triangular ∧
    are_similar c1 c2 ∧
    ¬are_equal c1 c2 ∧
    ¬touches_base c1 p ∧
    ¬touches_base c2 p ∧
    ¬cuts_touch c1 c2 := by sorry

-- Define an equilateral triangular cut
def is_equilateral_triangle (c : Cut) (side_length : ℝ) : Prop := sorry

-- Theorem for part (b)
theorem equilateral_cuts_impossible (p : TriangularPrism) :
  ¬∃ (c1 c2 : Cut),
    is_equilateral_triangle c1 1 ∧
    is_equilateral_triangle c2 2 ∧
    ¬touches_base c1 p ∧
    ¬touches_base c2 p ∧
    ¬cuts_touch c1 c2 := by sorry

end NUMINAMATH_CALUDE_similar_cuts_possible_equilateral_cuts_impossible_l3642_364230


namespace NUMINAMATH_CALUDE_solution_mixture_problem_l3642_364228

theorem solution_mixture_problem (x : ℝ) :
  -- Solution 1 composition
  x + 80 = 100 →
  -- Solution 2 composition
  45 + 55 = 100 →
  -- Mixture composition (50% each solution)
  (x + 45) / 2 + (80 + 55) / 2 = 100 →
  -- Mixture contains 67.5% carbonated water
  (80 + 55) / 2 = 67.5 →
  -- Conclusion: Solution 1 is 20% lemonade
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_solution_mixture_problem_l3642_364228


namespace NUMINAMATH_CALUDE_integer_solutions_count_l3642_364224

theorem integer_solutions_count : ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |7*x - 4| ≤ 14) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l3642_364224


namespace NUMINAMATH_CALUDE_triangle_inequality_with_powers_l3642_364210

theorem triangle_inequality_with_powers (n : ℕ) (a b c : ℝ) 
  (hn : n > 1) 
  (hab : a > 0) (hbc : b > 0) (hca : c > 0)
  (hsum : a + b + c = 1)
  (htriangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + 2^(1/n : ℝ)/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_powers_l3642_364210


namespace NUMINAMATH_CALUDE_matt_completes_in_100_days_l3642_364258

/-- The rate at which Matt and Peter complete work together -/
def combined_rate : ℚ := 1 / 20

/-- The rate at which Peter completes work alone -/
def peter_rate : ℚ := 1 / 25

/-- The rate at which Matt completes work alone -/
def matt_rate : ℚ := combined_rate - peter_rate

/-- The number of days Matt takes to complete the work alone -/
def matt_days : ℚ := 1 / matt_rate

theorem matt_completes_in_100_days : matt_days = 100 := by
  sorry

end NUMINAMATH_CALUDE_matt_completes_in_100_days_l3642_364258


namespace NUMINAMATH_CALUDE_fraction_relation_l3642_364294

theorem fraction_relation (a b c : ℚ) 
  (h1 : a / b = 2) 
  (h2 : b / c = 4 / 3) : 
  c / a = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l3642_364294


namespace NUMINAMATH_CALUDE_set_operations_l3642_364247

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ C)) = {6, 8, 10}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3642_364247


namespace NUMINAMATH_CALUDE_last_two_digits_2005_pow_base_3_representation_l3642_364235

-- Define the expression
def big_exp : ℕ := 2003^2004 + 3

-- Define the function to calculate the last two digits in base 3
def last_two_digits_base_3 (n : ℕ) : ℕ := n % 9

-- Theorem statement
theorem last_two_digits_2005_pow : last_two_digits_base_3 (2005^big_exp) = 4 := by
  sorry

-- Convert to base 3
theorem base_3_representation : (last_two_digits_base_3 (2005^big_exp)).digits 3 = [1, 1] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_2005_pow_base_3_representation_l3642_364235


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3642_364220

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) := x^2 - 2*x

-- State the theorem
theorem f_monotone_decreasing :
  MonotoneOn f (Set.Iic 1) := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3642_364220


namespace NUMINAMATH_CALUDE_largest_coefficient_term_l3642_364225

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The general term in the binomial expansion -/
def binomialTerm (n k : ℕ) (a b : ℝ) : ℝ := 
  (binomial n k : ℝ) * (a ^ (n - k)) * (b ^ k)

/-- The coefficient of the k-th term in the expansion of (2+3x)^10 -/
def coefficientTerm (k : ℕ) : ℝ := 
  (binomial 10 k : ℝ) * (2 ^ (10 - k)) * (3 ^ k)

theorem largest_coefficient_term :
  ∃ (k : ℕ), k = 5 ∧ 
  ∀ (j : ℕ), j ≠ k → coefficientTerm k ≥ coefficientTerm j :=
sorry

end NUMINAMATH_CALUDE_largest_coefficient_term_l3642_364225


namespace NUMINAMATH_CALUDE_welders_left_correct_l3642_364277

/-- The number of welders who left for the other project after the first day -/
def welders_who_left : ℕ := 11

/-- The initial number of welders -/
def initial_welders : ℕ := 16

/-- The number of days to complete the order with all welders -/
def initial_days : ℕ := 8

/-- The additional days needed by remaining welders to complete the order -/
def additional_days : ℕ := 16

/-- The total amount of work to be done -/
def total_work : ℝ := initial_welders * initial_days

/-- The work done in the first day -/
def first_day_work : ℝ := initial_welders

/-- The remaining work after the first day -/
def remaining_work : ℝ := total_work - first_day_work

theorem welders_left_correct :
  (initial_welders - welders_who_left) * (initial_days + additional_days) = remaining_work :=
sorry

end NUMINAMATH_CALUDE_welders_left_correct_l3642_364277


namespace NUMINAMATH_CALUDE_count_convex_cyclic_quads_l3642_364297

/-- A convex cyclic quadrilateral with integer sides --/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  sum_eq_40 : a + b + c + d = 40
  convex : a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c
  ordered : a ≥ b ∧ b ≥ c ∧ c ≥ d
  has_odd_side : Odd a ∨ Odd b ∨ Odd c ∨ Odd d

/-- The count of valid quadrilaterals --/
def count_valid_quads : ℕ := sorry

theorem count_convex_cyclic_quads : count_valid_quads = 760 := by
  sorry

end NUMINAMATH_CALUDE_count_convex_cyclic_quads_l3642_364297


namespace NUMINAMATH_CALUDE_cyclist_distance_theorem_l3642_364272

/-- A cyclist travels in a straight line for two minutes. -/
def cyclist_travel (v1 v2 : ℝ) : ℝ := v1 * 60 + v2 * 60

/-- The theorem states that a cyclist traveling at 2 m/s for the first minute
    and 4 m/s for the second minute covers a total distance of 360 meters. -/
theorem cyclist_distance_theorem :
  cyclist_travel 2 4 = 360 := by sorry

end NUMINAMATH_CALUDE_cyclist_distance_theorem_l3642_364272


namespace NUMINAMATH_CALUDE_pets_problem_l3642_364295

theorem pets_problem (total_students : ℕ) 
  (students_with_dogs : ℕ) 
  (students_with_cats : ℕ) 
  (students_with_other_pets : ℕ) 
  (students_no_pets : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (only_other_pets : ℕ) 
  (dogs_and_cats : ℕ) 
  (dogs_and_other : ℕ) 
  (cats_and_other : ℕ) :
  total_students = 40 →
  students_with_dogs = 20 →
  students_with_cats = total_students / 4 →
  students_with_other_pets = 10 →
  students_no_pets = 5 →
  only_dogs = 15 →
  only_cats = 4 →
  only_other_pets = 5 →
  total_students = only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other + 
    students_no_pets + (students_with_dogs + students_with_cats + 
    students_with_other_pets - (only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other)) →
  students_with_dogs + students_with_cats + students_with_other_pets - 
    (only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other) = 0 :=
by sorry

end NUMINAMATH_CALUDE_pets_problem_l3642_364295


namespace NUMINAMATH_CALUDE_negative_exponent_two_l3642_364286

theorem negative_exponent_two : 2⁻¹ = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_negative_exponent_two_l3642_364286


namespace NUMINAMATH_CALUDE_choir_size_l3642_364278

theorem choir_size :
  ∀ X : ℕ,
  (X / 2 : ℚ) - (X / 6 : ℚ) = 10 →
  X = 30 :=
by
  sorry

#check choir_size

end NUMINAMATH_CALUDE_choir_size_l3642_364278


namespace NUMINAMATH_CALUDE_bob_sandwich_options_l3642_364248

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def has_turkey : Prop := True

/-- Represents whether roast beef is available. -/
def has_roast_beef : Prop := True

/-- Represents whether Swiss cheese is available. -/
def has_swiss_cheese : Prop := True

/-- Represents whether rye bread is available. -/
def has_rye_bread : Prop := True

/-- Represents the number of sandwiches with turkey and Swiss cheese. -/
def turkey_swiss_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and roast beef. -/
def rye_roast_beef_combos : ℕ := num_cheeses

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : 
  num_breads * num_meats * num_cheeses - turkey_swiss_combos - rye_roast_beef_combos = 199 :=
sorry

end NUMINAMATH_CALUDE_bob_sandwich_options_l3642_364248


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3642_364246

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Theorem statement
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 5 < 0) :
  a 1 * a 2 * a 3 * a 4 > 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3642_364246


namespace NUMINAMATH_CALUDE_N_prime_iff_k_eq_two_l3642_364243

def N (k : ℕ) : ℕ := (10^(2*k) - 1) / 99

theorem N_prime_iff_k_eq_two :
  ∀ k : ℕ, k > 0 → (Nat.Prime (N k) ↔ k = 2) := by sorry

end NUMINAMATH_CALUDE_N_prime_iff_k_eq_two_l3642_364243


namespace NUMINAMATH_CALUDE_tanika_cracker_sales_l3642_364215

theorem tanika_cracker_sales (saturday_sales : ℕ) : 
  saturday_sales = 60 → 
  (saturday_sales + (saturday_sales + saturday_sales / 2)) = 150 := by
sorry

end NUMINAMATH_CALUDE_tanika_cracker_sales_l3642_364215


namespace NUMINAMATH_CALUDE_not_perfect_cube_1967_l3642_364262

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem not_perfect_cube_1967 :
  ∀ (p : Fin 1967 → Fin 1967), Function.Bijective p →
    ¬ (∃ (k : ℕ), sum_of_first_n 1967 = k^3) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_cube_1967_l3642_364262


namespace NUMINAMATH_CALUDE_special_sequence_a11_l3642_364261

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ+ → ℤ) : Prop :=
  (∀ p q : ℕ+, a (p + q) = a p + a q) ∧ (a 2 = -6)

/-- The theorem statement -/
theorem special_sequence_a11 (a : ℕ+ → ℤ) (h : SpecialSequence a) : a 11 = -33 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a11_l3642_364261


namespace NUMINAMATH_CALUDE_number_of_divisors_of_90_l3642_364257

theorem number_of_divisors_of_90 : Nat.card {d : ℕ | d ∣ 90} = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_90_l3642_364257


namespace NUMINAMATH_CALUDE_solution_set_f_geq_5_range_of_a_l3642_364245

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Part I: Solution set of f(x) ≥ 5
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 :=
sorry

-- Part II: Range of a for which f(x) > a^2 - 2a holds for all x
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2*a) ↔ a ∈ Set.Ioo (-1) 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_5_range_of_a_l3642_364245


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l3642_364266

/-- 
Given a boy who reaches school 4 minutes early when walking at 9/8 of his usual rate,
prove that his usual time to reach the school is 36 minutes.
-/
theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) 
  (h2 : usual_time > 0)
  (h3 : usual_rate * usual_time = (9/8 * usual_rate) * (usual_time - 4)) : 
  usual_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l3642_364266


namespace NUMINAMATH_CALUDE_annie_purchase_problem_l3642_364226

/-- Annie's hamburger and milkshake purchase problem -/
theorem annie_purchase_problem (hamburger_price milkshake_price hamburger_count milkshake_count remaining_money : ℕ) 
  (h1 : hamburger_price = 4)
  (h2 : milkshake_price = 5)
  (h3 : hamburger_count = 8)
  (h4 : milkshake_count = 6)
  (h5 : remaining_money = 70) :
  hamburger_price * hamburger_count + milkshake_price * milkshake_count + remaining_money = 132 := by
  sorry

end NUMINAMATH_CALUDE_annie_purchase_problem_l3642_364226


namespace NUMINAMATH_CALUDE_bingbing_correct_qianqian_incorrect_l3642_364282

-- Define the basic parameters of the problem
def downstream_time : ℝ := 2
def upstream_time : ℝ := 2.5
def water_speed : ℝ := 3

-- Define Bingbing's equation
def bingbing_equation (x : ℝ) : Prop :=
  2 * (x + water_speed) = upstream_time * (x - water_speed)

-- Define Qianqian's equation
def qianqian_equation (x : ℝ) : Prop :=
  x / downstream_time - x / upstream_time = water_speed * downstream_time

-- Theorem stating that Bingbing's equation correctly models the problem
theorem bingbing_correct :
  ∃ (x : ℝ), bingbing_equation x ∧ x > 0 ∧ 
  (x * downstream_time = x * upstream_time) :=
sorry

-- Theorem stating that Qianqian's equation does not correctly model the problem
theorem qianqian_incorrect :
  ¬(∃ (x : ℝ), qianqian_equation x ∧ 
  (x * downstream_time = x * upstream_time ∨ x > 0)) :=
sorry

end NUMINAMATH_CALUDE_bingbing_correct_qianqian_incorrect_l3642_364282


namespace NUMINAMATH_CALUDE_spending_calculation_l3642_364289

theorem spending_calculation (initial_amount : ℚ) : 
  let remaining_after_clothes : ℚ := initial_amount * (2/3)
  let remaining_after_food : ℚ := remaining_after_clothes * (4/5)
  let final_amount : ℚ := remaining_after_food * (3/4)
  final_amount = 300 → initial_amount = 750 := by
sorry

end NUMINAMATH_CALUDE_spending_calculation_l3642_364289


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l3642_364298

/-- The function f(x) = (x+1)^2 - 3 -/
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

/-- The theorem stating that -1 is the smallest value of c for which f has an inverse on [c,∞) -/
theorem smallest_c_for_inverse :
  ∀ c : ℝ, (∀ x y, x ∈ Set.Ici c → y ∈ Set.Ici c → f x = f y → x = y) ↔ c ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l3642_364298


namespace NUMINAMATH_CALUDE_perpendicular_length_between_l3642_364232

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the relations and functions
variable (on_line : Point → Line → Prop)
variable (between : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Line → Prop)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem perpendicular_length_between
  (a b : Line)
  (A₁ A₂ A₃ B₁ B₂ B₃ : Point)
  (h1 : on_line A₁ a)
  (h2 : on_line A₂ a)
  (h3 : on_line A₃ a)
  (h4 : between A₁ A₂ A₃)
  (h5 : perpendicular A₁ B₁ b)
  (h6 : perpendicular A₂ B₂ b)
  (h7 : perpendicular A₃ B₃ b) :
  (length A₁ B₁ ≤ length A₂ B₂ ∧ length A₂ B₂ ≤ length A₃ B₃) ∨
  (length A₃ B₃ ≤ length A₂ B₂ ∧ length A₂ B₂ ≤ length A₁ B₁) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_length_between_l3642_364232


namespace NUMINAMATH_CALUDE_matrix_determinant_equality_l3642_364271

theorem matrix_determinant_equality 
  (A B : Matrix (Fin 4) (Fin 4) ℝ) 
  (h1 : A * B = B * A) 
  (h2 : Matrix.det (A^2 + A*B + B^2) = 0) : 
  Matrix.det (A + B) + 3 * Matrix.det (A - B) = 6 * Matrix.det A + 6 * Matrix.det B := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_equality_l3642_364271


namespace NUMINAMATH_CALUDE_rounding_estimate_greater_l3642_364260

theorem rounding_estimate_greater (x y z x' y' z' : ℤ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hx' : x' ≥ x) (hy' : y' ≤ y) (hz' : z' ≤ z) :
  2 * ((x' : ℚ) / y' - z') > 2 * ((x : ℚ) / y - z) :=
sorry

end NUMINAMATH_CALUDE_rounding_estimate_greater_l3642_364260


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3642_364267

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Complex.abs z = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3642_364267


namespace NUMINAMATH_CALUDE_apple_production_theorem_l3642_364249

/-- The apple production problem -/
theorem apple_production_theorem :
  let first_year : ℕ := 40
  let second_year : ℕ := 2 * first_year + 8
  let third_year : ℕ := (3 * second_year) / 4
  first_year + second_year + third_year = 194 := by
sorry

end NUMINAMATH_CALUDE_apple_production_theorem_l3642_364249


namespace NUMINAMATH_CALUDE_pattern_perimeter_is_24_l3642_364209

/-- A pattern formed by squares, triangles, and a hexagon -/
structure Pattern where
  num_squares : ℕ
  num_triangles : ℕ
  square_side_length : ℝ
  triangle_perimeter_contribution : ℕ
  square_perimeter_contribution : ℕ

/-- Calculate the perimeter of the pattern -/
def pattern_perimeter (p : Pattern) : ℝ :=
  (p.num_triangles * p.triangle_perimeter_contribution +
   p.num_squares * p.square_perimeter_contribution) * p.square_side_length

/-- The specific pattern described in the problem -/
def specific_pattern : Pattern := {
  num_squares := 6,
  num_triangles := 6,
  square_side_length := 2,
  triangle_perimeter_contribution := 2,
  square_perimeter_contribution := 2
}

theorem pattern_perimeter_is_24 :
  pattern_perimeter specific_pattern = 24 := by
  sorry

end NUMINAMATH_CALUDE_pattern_perimeter_is_24_l3642_364209


namespace NUMINAMATH_CALUDE_stereo_system_trade_in_john_stereo_trade_in_l3642_364269

theorem stereo_system_trade_in (old_cost : ℝ) (trade_in_percentage : ℝ) 
  (new_cost : ℝ) (discount_percentage : ℝ) : ℝ :=
  let trade_in_value := old_cost * trade_in_percentage
  let discounted_new_cost := new_cost * (1 - discount_percentage)
  discounted_new_cost - trade_in_value

theorem john_stereo_trade_in :
  stereo_system_trade_in 250 0.8 600 0.25 = 250 := by
  sorry

end NUMINAMATH_CALUDE_stereo_system_trade_in_john_stereo_trade_in_l3642_364269


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3642_364263

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > 1) ↔ (∃ x : ℕ, x^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3642_364263


namespace NUMINAMATH_CALUDE_symmetric_abs_function_l3642_364273

/-- A function f is symmetric about a point c if f(c+x) = f(c-x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_abs_function (m n : ℝ) :
  SymmetricAbout (fun x ↦ |x + m| + |n * x + 1|) 2 → m + n = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_abs_function_l3642_364273


namespace NUMINAMATH_CALUDE_butanoic_acid_molecular_weight_l3642_364284

/-- The molecular weight of one mole of Butanoic acid. -/
def molecular_weight_one_mole : ℝ := 88

/-- The number of moles given in the problem. -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given number of moles. -/
def total_molecular_weight : ℝ := 792

/-- Theorem stating that the molecular weight of one mole of Butanoic acid is 88 g/mol,
    given that the molecular weight of 9 moles is 792. -/
theorem butanoic_acid_molecular_weight :
  molecular_weight_one_mole = total_molecular_weight / num_moles :=
by sorry

end NUMINAMATH_CALUDE_butanoic_acid_molecular_weight_l3642_364284


namespace NUMINAMATH_CALUDE_special_school_total_students_l3642_364227

/-- Represents a school with blind and deaf students -/
structure School where
  blind_students : ℕ
  deaf_students : ℕ

/-- The total number of students in the school -/
def total_students (s : School) : ℕ :=
  s.blind_students + s.deaf_students

/-- A special school with a specific ratio of deaf to blind students and a given number of blind students -/
def special_school : School :=
  { blind_students := 45,
    deaf_students := 3 * 45 }

theorem special_school_total_students :
  total_students special_school = 180 := by
  sorry

end NUMINAMATH_CALUDE_special_school_total_students_l3642_364227


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_l3642_364241

def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
      hyperbola x₁' y₁' ∧
      hyperbola x₂' y₂' ∧
      (is_midpoint 1 1 x₁' y₁' x₂' y₂' ∨
       is_midpoint (-1) 2 x₁' y₁' x₂' y₂' ∨
       is_midpoint 1 3 x₁' y₁' x₂' y₂') :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_l3642_364241


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3642_364265

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_third : a 3 = 7)
  (h_sixth : a 6 = 16) :
  a 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3642_364265


namespace NUMINAMATH_CALUDE_equation_solution_l3642_364212

theorem equation_solution : 
  ∀ x : ℝ, x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3642_364212


namespace NUMINAMATH_CALUDE_complex_number_theorem_l3642_364201

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_theorem (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z + 2)^2 + 5)) : 
  z = Complex.I * 3 ∨ z = Complex.I * (-3) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l3642_364201


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_and_angle_l3642_364205

/-- Given an arc length of 4 cm corresponding to a central angle of 2 radians,
    the area of the sector formed by this central angle is 4 cm^2. -/
theorem sector_area_from_arc_length_and_angle (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = 2) :
  let r := s / θ
  (1 / 2) * r^2 * θ = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_and_angle_l3642_364205


namespace NUMINAMATH_CALUDE_solutions_to_quartic_equation_l3642_364208

theorem solutions_to_quartic_equation :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_to_quartic_equation_l3642_364208


namespace NUMINAMATH_CALUDE_cereal_box_theorem_l3642_364244

/-- The number of clusters of oats in each spoonful -/
def clusters_per_spoonful : ℕ := 4

/-- The number of spoonfuls of cereal in each bowl -/
def spoonfuls_per_bowl : ℕ := 25

/-- The number of clusters of oats in each box -/
def clusters_per_box : ℕ := 500

/-- The number of bowlfuls of cereal in each box -/
def bowls_per_box : ℕ := 5

theorem cereal_box_theorem : 
  clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = bowls_per_box := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_theorem_l3642_364244


namespace NUMINAMATH_CALUDE_better_fit_for_lower_rss_l3642_364280

/-- Represents a model with its residual sum of squares -/
structure Model where
  rss : ℝ

/-- Definition of a better fit model -/
def better_fit (m1 m2 : Model) : Prop := m1.rss < m2.rss

theorem better_fit_for_lower_rss (model1 model2 : Model) 
  (h1 : model1.rss = 152.6) 
  (h2 : model2.rss = 159.8) : 
  better_fit model1 model2 := by
  sorry

#check better_fit_for_lower_rss

end NUMINAMATH_CALUDE_better_fit_for_lower_rss_l3642_364280


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l3642_364222

theorem geometric_series_second_term 
  (r : ℚ) 
  (S : ℚ) 
  (h1 : r = -1/3) 
  (h2 : S = 25) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : 
  second_term = -100/9 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l3642_364222


namespace NUMINAMATH_CALUDE_product_value_l3642_364216

theorem product_value : 
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l3642_364216


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l3642_364285

/-- Given a mixture of 15 litres with 25% alcohol, prove that after removing 2 litres of alcohol
    and adding 3 litres of water, the final alcohol percentage is approximately 10.94%. -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (alcohol_removed : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 15)
  (h2 : initial_alcohol_percentage = 0.25)
  (h3 : alcohol_removed = 2)
  (h4 : water_added = 3) :
  let initial_alcohol := initial_volume * initial_alcohol_percentage
  let remaining_alcohol := initial_alcohol - alcohol_removed
  let final_volume := initial_volume - alcohol_removed + water_added
  let final_percentage := (remaining_alcohol / final_volume) * 100
  ∃ ε > 0, abs (final_percentage - 10.94) < ε :=
sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l3642_364285


namespace NUMINAMATH_CALUDE_limit_problem_l3642_364264

open Real
open Function

/-- The limit of the given function as x approaches 2 is -8ln(2)/5 -/
theorem limit_problem : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
  0 < |x - 2| ∧ |x - 2| < δ → 
    |(1 - 2^(4 - x^2)) / (2 * (sqrt (2*x) - sqrt (3*x^2 - 5*x + 2))) + 8*log 2 / 5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_problem_l3642_364264


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l3642_364206

/-- Given a geometric sequence with common ratio 3, prove that a_3 = 3 if S_3 + S_4 = 53/3 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 3 * a n) →  -- Geometric sequence with common ratio 3
  (∀ n, S n = (a 1 * (3^n - 1)) / 2) →  -- Sum formula for geometric sequence
  S 3 + S 4 = 53 / 3 →  -- Given condition
  a 3 = 3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_a3_l3642_364206


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_l3642_364239

theorem expand_difference_of_squares (a : ℝ) : (a + 2) * (2 - a) = 4 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_l3642_364239


namespace NUMINAMATH_CALUDE_sum_mod_seven_l3642_364207

theorem sum_mod_seven : (4123 + 4124 + 4125 + 4126 + 4127) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l3642_364207


namespace NUMINAMATH_CALUDE_unique_prime_six_digit_number_l3642_364234

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B A : Nat) : Nat := 3000000 + B * 10000 + 1200 + A

theorem unique_prime_six_digit_number :
  ∃! (B A : Nat), B < 10 ∧ A < 10 ∧ 
    is_prime (six_digit_number B A) ∧
    B + A = 9 := by sorry

end NUMINAMATH_CALUDE_unique_prime_six_digit_number_l3642_364234


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l3642_364219

/-- The set of points equidistant from a fixed point and a line forms a parabola -/
theorem trajectory_is_parabola (x y : ℝ) : 
  (∃ (C : ℝ × ℝ), C.1 = x ∧ C.2 = y ∧ 
    (C.1^2 + (C.2 - 3)^2)^(1/2) = |C.2 + 3|) →
  ∃ (a : ℝ), y = (1 / (4 * a)) * x^2 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l3642_364219


namespace NUMINAMATH_CALUDE_derivative_of_f_l3642_364231

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x + 9

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3642_364231


namespace NUMINAMATH_CALUDE_amount_c_l3642_364217

/-- Given four amounts a, b, c, and d satisfying certain conditions, prove that c equals 225. -/
theorem amount_c (a b c d : ℕ) : 
  a + b + c + d = 750 →
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  c = 225 := by
  sorry


end NUMINAMATH_CALUDE_amount_c_l3642_364217


namespace NUMINAMATH_CALUDE_pat_has_42_cookies_l3642_364204

-- Define the given conditions
def candy : ℕ := 63
def brownies : ℕ := 21
def family_members : ℕ := 7
def dessert_per_person : ℕ := 18

-- Define the total dessert needed
def total_dessert : ℕ := family_members * dessert_per_person

-- Define the number of cookies
def cookies : ℕ := total_dessert - (candy + brownies)

-- Theorem to prove
theorem pat_has_42_cookies : cookies = 42 := by
  sorry

end NUMINAMATH_CALUDE_pat_has_42_cookies_l3642_364204


namespace NUMINAMATH_CALUDE_students_playing_neither_l3642_364268

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 39 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l3642_364268


namespace NUMINAMATH_CALUDE_unique_a_for_system_solution_l3642_364254

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  2^(b*x) + (a+1)*b*y^2 = a^2 ∧ (a-1)*x^3 + y^3 = 1

-- State the theorem
theorem unique_a_for_system_solution :
  ∃! a : ℝ, ∀ b : ℝ, ∃ x y : ℝ, system a b x y ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_a_for_system_solution_l3642_364254


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3642_364296

theorem polynomial_division_theorem (x : ℝ) : 
  (x - 3) * (x^4 + 3*x^3 - 7*x^2 - 10*x - 39) + (-47) = 
  x^5 - 16*x^3 + 11*x^2 - 9*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3642_364296


namespace NUMINAMATH_CALUDE_max_points_on_ellipse_l3642_364270

/-- Represents an ellipse with semi-major axis a and focal distance c -/
structure Ellipse where
  a : ℝ
  c : ℝ

/-- Represents a sequence of points on an ellipse -/
structure PointSequence where
  n : ℕ
  d : ℝ

theorem max_points_on_ellipse (e : Ellipse) (seq : PointSequence) :
  e.a - e.c = 1 →
  e.a + e.c = 3 →
  seq.d > 1/100 →
  (∀ i : ℕ, i < seq.n → 1 + i * seq.d ≤ 3) →
  seq.n ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_max_points_on_ellipse_l3642_364270


namespace NUMINAMATH_CALUDE_find_n_l3642_364290

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3642_364290


namespace NUMINAMATH_CALUDE_rectangular_field_fencing_l3642_364293

theorem rectangular_field_fencing (area : ℝ) (fencing : ℝ) :
  area = 680 ∧ fencing = 74 →
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    area = length * width ∧
    fencing = 2 * width + length ∧
    length = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fencing_l3642_364293


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_frac_l3642_364288

theorem ceiling_neg_sqrt_frac : ⌈-Real.sqrt (36 / 9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_frac_l3642_364288


namespace NUMINAMATH_CALUDE_toy_factory_wage_calculation_l3642_364240

/-- Toy factory production and wage calculation -/
theorem toy_factory_wage_calculation 
  (planned_weekly_production : ℕ)
  (average_daily_production : ℕ)
  (deviations : List ℤ)
  (base_wage_per_toy : ℕ)
  (bonus_per_extra_toy : ℕ)
  (deduction_per_missing_toy : ℕ)
  (h1 : planned_weekly_production = 700)
  (h2 : average_daily_production = 100)
  (h3 : deviations = [5, -2, -4, 13, -6, 6, -3])
  (h4 : base_wage_per_toy = 20)
  (h5 : bonus_per_extra_toy = 5)
  (h6 : deduction_per_missing_toy = 4)
  : (planned_weekly_production + deviations.sum) * base_wage_per_toy + 
    (deviations.sum * (base_wage_per_toy + bonus_per_extra_toy)) = 14225 := by
  sorry

end NUMINAMATH_CALUDE_toy_factory_wage_calculation_l3642_364240


namespace NUMINAMATH_CALUDE_gcd_242_154_l3642_364274

theorem gcd_242_154 : Nat.gcd 242 154 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_242_154_l3642_364274


namespace NUMINAMATH_CALUDE_annual_production_after_five_years_l3642_364259

/-- Given an initial value, growth rate, and time span, calculate the final value after compound growth -/
def compound_growth (initial_value : ℝ) (growth_rate : ℝ) (time_span : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ time_span

/-- Theorem: The annual production after 5 years with a given growth rate -/
theorem annual_production_after_five_years 
  (a : ℝ) -- initial production in 2005
  (x : ℝ) -- annual growth rate
  : 
  compound_growth a x 5 = a * (1 + x)^5 := by
  sorry

end NUMINAMATH_CALUDE_annual_production_after_five_years_l3642_364259


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3642_364291

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^(2*x) * (1000 : ℝ)^x = (10 : ℝ)^15 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3642_364291


namespace NUMINAMATH_CALUDE_base_b_problem_l3642_364255

theorem base_b_problem : ∃! (b : ℕ), b > 1 ∧ (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 := by
  sorry

end NUMINAMATH_CALUDE_base_b_problem_l3642_364255


namespace NUMINAMATH_CALUDE_remainder_of_123456789012_mod_252_l3642_364250

theorem remainder_of_123456789012_mod_252 : 123456789012 % 252 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_123456789012_mod_252_l3642_364250


namespace NUMINAMATH_CALUDE_dacids_physics_marks_l3642_364252

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

end NUMINAMATH_CALUDE_dacids_physics_marks_l3642_364252


namespace NUMINAMATH_CALUDE_stan_pays_magician_l3642_364214

/-- The total amount Stan pays the magician -/
def total_payment (hourly_rate : ℕ) (hours_per_day : ℕ) (weeks : ℕ) : ℕ :=
  hourly_rate * hours_per_day * (weeks * 7)

/-- Proof that Stan pays the magician $2520 -/
theorem stan_pays_magician :
  total_payment 60 3 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_stan_pays_magician_l3642_364214


namespace NUMINAMATH_CALUDE_substance_mass_proof_l3642_364256

/-- The volume of 1 gram of the substance in cubic centimeters -/
def volume_per_gram : ℝ := 1.3333333333333335

/-- The number of cubic centimeters in 1 cubic meter -/
def cm3_per_m3 : ℝ := 1000000

/-- The number of grams in 1 kilogram -/
def grams_per_kg : ℝ := 1000

/-- The mass of 1 cubic meter of the substance in kilograms -/
def mass_per_m3 : ℝ := 750

theorem substance_mass_proof :
  mass_per_m3 = cm3_per_m3 / (grams_per_kg * volume_per_gram) := by
  sorry

end NUMINAMATH_CALUDE_substance_mass_proof_l3642_364256


namespace NUMINAMATH_CALUDE_problem_solution_l3642_364292

theorem problem_solution (a b : ℤ) 
  (h1 : 4010 * a + 4014 * b = 4020) 
  (h2 : 4012 * a + 4016 * b = 4024) : 
  a - b = 2002 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3642_364292


namespace NUMINAMATH_CALUDE_monotonic_h_implies_a_leq_neg_one_l3642_364236

/-- Given functions f and g, prove that if h is monotonically increasing on [1,4],
    then a ≤ -1 -/
theorem monotonic_h_implies_a_leq_neg_one (a : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ Real.log x
  let g : ℝ → ℝ := λ x ↦ (1/2) * a * x^2 + 2*x
  let h : ℝ → ℝ := λ x ↦ f x - g x
  (∀ x ∈ Set.Icc 1 4, Monotone h) →
  a ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_monotonic_h_implies_a_leq_neg_one_l3642_364236


namespace NUMINAMATH_CALUDE_factorial_equivalences_l3642_364213

/-- The number of arrangements of n objects taken k at a time -/
def A (n k : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

theorem factorial_equivalences (n : ℕ) : 
  (A n (n - 1) = factorial n) ∧ 
  ((1 / (n + 1 : ℚ)) * A (n + 1) (n + 1) = factorial n) := by sorry

end NUMINAMATH_CALUDE_factorial_equivalences_l3642_364213


namespace NUMINAMATH_CALUDE_nail_decoration_time_l3642_364238

theorem nail_decoration_time (total_time : ℕ) (num_coats : ℕ) (time_per_coat : ℕ) : 
  total_time = 120 →
  num_coats = 3 →
  total_time = num_coats * 2 * time_per_coat →
  time_per_coat = 20 := by
sorry

end NUMINAMATH_CALUDE_nail_decoration_time_l3642_364238


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l3642_364287

/-- 
Given a parallelogram where one angle exceeds the other by 40 degrees,
prove that the measure of the smaller angle is 70 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (smaller_angle larger_angle : ℝ),
  -- Conditions
  (smaller_angle > 0) →  -- Angle measure is positive
  (larger_angle > 0) →  -- Angle measure is positive
  (larger_angle = smaller_angle + 40) →  -- One angle exceeds the other by 40
  (smaller_angle + larger_angle = 180) →  -- Adjacent angles are supplementary
  -- Conclusion
  smaller_angle = 70 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l3642_364287


namespace NUMINAMATH_CALUDE_system_no_solution_l3642_364211

theorem system_no_solution (n : ℝ) : 
  (∃ (x y z : ℝ), nx + y = 1 ∧ ny + z = 1 ∧ x + nz = 1) ↔ n ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l3642_364211


namespace NUMINAMATH_CALUDE_shelly_thread_needed_l3642_364251

def thread_per_keychain : ℕ := 12
def friends_in_classes : ℕ := 6
def friends_in_clubs : ℕ := friends_in_classes / 2

theorem shelly_thread_needed : 
  (friends_in_classes + friends_in_clubs) * thread_per_keychain = 108 := by
  sorry

end NUMINAMATH_CALUDE_shelly_thread_needed_l3642_364251


namespace NUMINAMATH_CALUDE_paint_distribution_321_60_l3642_364233

/-- Given a paint mixture with a ratio of red:white:blue and a total number of cans,
    calculate the number of cans for each color. -/
def paint_distribution (red white blue total : ℕ) : ℕ × ℕ × ℕ :=
  let sum := red + white + blue
  let red_cans := total * red / sum
  let white_cans := total * white / sum
  let blue_cans := total * blue / sum
  (red_cans, white_cans, blue_cans)

/-- Prove that for a 3:2:1 ratio and 60 total cans, we get 30 red, 20 white, and 10 blue cans. -/
theorem paint_distribution_321_60 :
  paint_distribution 3 2 1 60 = (30, 20, 10) := by
  sorry

end NUMINAMATH_CALUDE_paint_distribution_321_60_l3642_364233


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l3642_364203

theorem sin_cos_equation_solution (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x)^4 - (Real.cos x)^4 = 1 / (Real.cos x) - 1 / (Real.sin x) ↔ 
  x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l3642_364203


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3642_364237

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3642_364237


namespace NUMINAMATH_CALUDE_hose_fill_time_proof_l3642_364221

/-- Represents the time (in hours) it takes for the hose to fill the pool -/
def hose_fill_time (pool_capacity : ℝ) (drain_time : ℝ) (time_elapsed : ℝ) (remaining_water : ℝ) : ℝ :=
  3

/-- Proves that the hose fill time is correct given the problem conditions -/
theorem hose_fill_time_proof (pool_capacity : ℝ) (drain_time : ℝ) (time_elapsed : ℝ) (remaining_water : ℝ)
  (h1 : pool_capacity = 120)
  (h2 : drain_time = 4)
  (h3 : time_elapsed = 3)
  (h4 : remaining_water = 90) :
  hose_fill_time pool_capacity drain_time time_elapsed remaining_water = 3 := by
  sorry

#eval hose_fill_time 120 4 3 90

end NUMINAMATH_CALUDE_hose_fill_time_proof_l3642_364221


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l3642_364242

def workday_hours : ℝ := 10
def first_meeting_minutes : ℝ := 30
def second_meeting_minutes : ℝ := 3 * first_meeting_minutes
def third_meeting_minutes : ℝ := 2 * second_meeting_minutes

def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
def workday_minutes : ℝ := workday_hours * 60

theorem meetings_percentage_of_workday :
  (total_meeting_minutes / workday_minutes) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l3642_364242


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3642_364281

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (1 + 1 / (a - 1)) / (a / (a^2 - 1)) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3642_364281


namespace NUMINAMATH_CALUDE_range_of_a_l3642_364223

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc 4 8 ∧ a ≠ 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3642_364223


namespace NUMINAMATH_CALUDE_tv_power_consumption_l3642_364279

/-- Given a TV that runs for 4 hours a day, with electricity costing 14 cents per kWh,
    and the TV costing 49 cents to run for a week, prove that the TV uses 125 watts of electricity per hour. -/
theorem tv_power_consumption (hours_per_day : ℝ) (cost_per_kwh : ℝ) (weekly_cost : ℝ) :
  hours_per_day = 4 →
  cost_per_kwh = 0.14 →
  weekly_cost = 0.49 →
  ∃ (watts : ℝ), watts = 125 ∧ 
    (weekly_cost / cost_per_kwh) / (hours_per_day * 7) * 1000 = watts :=
by sorry

end NUMINAMATH_CALUDE_tv_power_consumption_l3642_364279


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l3642_364202

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular m α) 
  (h3 : subset n α) : 
  perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l3642_364202


namespace NUMINAMATH_CALUDE_james_sales_l3642_364253

/-- Given James' sales over two days, prove the total number of items sold --/
theorem james_sales (day1_houses day2_houses : ℕ) (day2_sale_rate : ℚ) : 
  day1_houses = 20 →
  day2_houses = 2 * day1_houses →
  day2_sale_rate = 4/5 →
  (day1_houses + (day2_houses : ℚ) * day2_sale_rate) * 2 = 104 := by
sorry

end NUMINAMATH_CALUDE_james_sales_l3642_364253


namespace NUMINAMATH_CALUDE_mrs_hilt_apple_pies_mrs_hilt_apple_pies_proof_l3642_364276

theorem mrs_hilt_apple_pies : ℝ → Prop :=
  fun apple_pies =>
    let pecan_pies : ℝ := 16.0
    let total_pies : ℝ := pecan_pies + apple_pies
    let new_total : ℝ := 150.0
    (5.0 * total_pies = new_total) → apple_pies = 14.0

-- The proof is omitted
theorem mrs_hilt_apple_pies_proof : mrs_hilt_apple_pies 14.0 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apple_pies_mrs_hilt_apple_pies_proof_l3642_364276


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3642_364200

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 3*a-1, a^2+1}
  A ∩ B = {-3} → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3642_364200


namespace NUMINAMATH_CALUDE_composite_ratio_l3642_364299

def first_seven_composites : List Nat := [4, 6, 8, 9, 10, 12, 14]
def next_seven_composites : List Nat := [15, 16, 18, 20, 21, 22, 24]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_of_list first_seven_composites) / 
  (product_of_list next_seven_composites) = 1 / 176 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_l3642_364299


namespace NUMINAMATH_CALUDE_equation_equivalence_l3642_364283

theorem equation_equivalence (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3642_364283


namespace NUMINAMATH_CALUDE_pool_water_after_20_days_l3642_364275

/-- Calculates the remaining water in a swimming pool after a given number of days -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (leak_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - (evaporation_rate + leak_rate) * days

/-- Theorem stating the remaining water in the pool after 20 days -/
theorem pool_water_after_20_days :
  remaining_water 500 1.5 0.8 20 = 454 := by
  sorry

#eval remaining_water 500 1.5 0.8 20

end NUMINAMATH_CALUDE_pool_water_after_20_days_l3642_364275
