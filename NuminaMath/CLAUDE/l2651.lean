import Mathlib

namespace NUMINAMATH_CALUDE_cone_slant_height_l2651_265193

/-- 
Given a cone whose lateral surface unfolds to a semicircle and whose base radius is 1,
prove that its slant height is 2.
-/
theorem cone_slant_height (r : ℝ) (l : ℝ) : 
  r = 1 → -- radius of the base is 1
  2 * π * r = π * l → -- lateral surface unfolds to a semicircle
  l = 2 := by sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2651_265193


namespace NUMINAMATH_CALUDE_min_value_of_f_l2651_265171

/-- The quadratic function we want to minimize -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 10

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -2/3 ∧ ∀ (x y : ℝ), f x y ≥ min := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2651_265171


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l2651_265150

/-- The function f(x) = 3x + ax^3 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x + a * x^3

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 + 3 * a * x^2

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  f_derivative a 1 = 6 → a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l2651_265150


namespace NUMINAMATH_CALUDE_division_of_fractions_l2651_265172

theorem division_of_fractions : (3 : ℚ) / (6 / 11) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l2651_265172


namespace NUMINAMATH_CALUDE_least_common_denominator_l2651_265159

theorem least_common_denominator (a b c d e f g h : ℕ) 
  (ha : a = 2) (hb : b = 3) (hc : c = 4) (hd : d = 5) 
  (he : e = 6) (hf : f = 7) (hg : g = 9) (hh : h = 10) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e (Nat.lcm f (Nat.lcm g h)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l2651_265159


namespace NUMINAMATH_CALUDE_original_average_age_l2651_265127

/-- Proves that the original average age of a class is 40 years given the specified conditions. -/
theorem original_average_age (original_strength : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_strength = 17 →
  new_students = 17 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ), 
    original_average * original_strength + new_students * new_average_age = 
    (original_strength + new_students) * (original_average - average_decrease) ∧
    original_average = 40 :=
by sorry

end NUMINAMATH_CALUDE_original_average_age_l2651_265127


namespace NUMINAMATH_CALUDE_darcie_age_ratio_l2651_265139

theorem darcie_age_ratio (darcie_age mother_age father_age : ℕ) :
  darcie_age = 4 →
  mother_age = (4 * father_age) / 5 →
  father_age = 30 →
  darcie_age * 6 = mother_age :=
by
  sorry

end NUMINAMATH_CALUDE_darcie_age_ratio_l2651_265139


namespace NUMINAMATH_CALUDE_max_gcd_sum_1980_l2651_265177

theorem max_gcd_sum_1980 :
  ∃ (a b : ℕ+), a + b = 1980 ∧
  ∀ (c d : ℕ+), c + d = 1980 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 990 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1980_l2651_265177


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_x_l2651_265122

theorem factorization_of_x_squared_minus_x (x : ℝ) : x^2 - x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_x_l2651_265122


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2651_265108

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {2, 5, 6}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2651_265108


namespace NUMINAMATH_CALUDE_range_of_sum_l2651_265195

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) :
  ∃ (z : ℝ), z = x + y ∧ -2 ≤ z ∧ z ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_sum_l2651_265195


namespace NUMINAMATH_CALUDE_amber_guppies_problem_l2651_265134

/-- The number of guppies Amber initially bought -/
def initial_guppies : ℕ := 7

/-- The number of baby guppies Amber saw in the first sighting (3 dozen) -/
def first_sighting : ℕ := 36

/-- The total number of guppies Amber has after the second sighting -/
def total_guppies : ℕ := 52

/-- The number of additional baby guppies Amber saw two days after the first sighting -/
def additional_guppies : ℕ := total_guppies - (initial_guppies + first_sighting)

theorem amber_guppies_problem :
  additional_guppies = 9 := by sorry

end NUMINAMATH_CALUDE_amber_guppies_problem_l2651_265134


namespace NUMINAMATH_CALUDE_cubic_extremum_difference_l2651_265198

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_extremum_difference (a b c : ℝ) :
  f' a b 2 = 0 → f' a b 1 = -3 →
  ∃ (min_val : ℝ), ∀ (x : ℝ), f a b c x ≥ min_val ∧ 
  ∀ (M : ℝ), ∃ (y : ℝ), f a b c y > M :=
by sorry

end NUMINAMATH_CALUDE_cubic_extremum_difference_l2651_265198


namespace NUMINAMATH_CALUDE_sqrt_11_diamond_sqrt_11_l2651_265120

-- Define the ¤ operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_11_diamond_sqrt_11 : diamond (Real.sqrt 11) (Real.sqrt 11) = 44 := by sorry

end NUMINAMATH_CALUDE_sqrt_11_diamond_sqrt_11_l2651_265120


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l2651_265128

/-- Represents a line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (x y : ℝ) → a * x + b * y + c = 0

/-- Generates the set of lines given by y = k, y = x + 3k, and y = -x + 3k for k from -6 to 6 --/
def generateLines : Set Line := sorry

/-- Checks if three lines form an equilateral triangle of side 1 --/
def formEquilateralTriangle (l1 l2 l3 : Line) : Prop := sorry

/-- Counts the number of equilateral triangles formed by the intersection of lines --/
def countEquilateralTriangles (lines : Set Line) : ℕ := sorry

/-- The main theorem stating that the number of equilateral triangles is 444 --/
theorem equilateral_triangle_count :
  ∃ (lines : Set Line), 
    lines = generateLines ∧ 
    countEquilateralTriangles lines = 444 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l2651_265128


namespace NUMINAMATH_CALUDE_brendas_weight_multiple_l2651_265157

theorem brendas_weight_multiple (brenda_weight mel_weight : ℕ) (multiple : ℚ) : 
  brenda_weight = 220 →
  mel_weight = 70 →
  brenda_weight = mel_weight * multiple + 10 →
  multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_weight_multiple_l2651_265157


namespace NUMINAMATH_CALUDE_max_non_fiction_books_l2651_265168

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_non_fiction_books :
  ∀ (fiction non_fiction : ℕ) (p : ℕ),
    fiction + non_fiction = 100 →
    fiction = non_fiction + p →
    is_prime p →
    non_fiction ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_max_non_fiction_books_l2651_265168


namespace NUMINAMATH_CALUDE_salt_mixture_theorem_l2651_265125

/-- Represents the salt mixture problem -/
def SaltMixture (cheap_price cheap_weight expensive_price expensive_weight profit_percentage : ℚ) : Prop :=
  let total_cost : ℚ := cheap_price * cheap_weight + expensive_price * expensive_weight
  let total_weight : ℚ := cheap_weight + expensive_weight
  let profit : ℚ := total_cost * (profit_percentage / 100)
  let selling_price : ℚ := total_cost + profit
  let selling_price_per_pound : ℚ := selling_price / total_weight
  selling_price_per_pound = 48 / 100

/-- The salt mixture theorem -/
theorem salt_mixture_theorem : SaltMixture (38/100) 40 (50/100) 8 20 := by
  sorry

end NUMINAMATH_CALUDE_salt_mixture_theorem_l2651_265125


namespace NUMINAMATH_CALUDE_ellipse_equation_for_given_properties_l2651_265142

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 36 + y^2 / 32 = 1

/-- Theorem stating the equation of the ellipse with given properties -/
theorem ellipse_equation_for_given_properties (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.minor_axis_length = 8 * Real.sqrt 2)
  (h4 : e.eccentricity = 1/3) :
  ellipse_equation e = fun x y => x^2 / 36 + y^2 / 32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_for_given_properties_l2651_265142


namespace NUMINAMATH_CALUDE_problem_solution_l2651_265169

theorem problem_solution (x : ℝ) : 3 * x = (26 - x) + 10 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2651_265169


namespace NUMINAMATH_CALUDE_gcd_problems_l2651_265188

theorem gcd_problems :
  (Nat.gcd 72 168 = 24) ∧ (Nat.gcd 98 280 = 14) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l2651_265188


namespace NUMINAMATH_CALUDE_athlete_c_most_suitable_l2651_265129

/-- Represents an athlete with their mean jump distance and variance --/
structure Athlete where
  name : String
  mean : ℝ
  variance : ℝ

/-- Determines if one athlete is more suitable than another --/
def moreSuitable (a b : Athlete) : Prop :=
  (a.mean > b.mean) ∨ (a.mean = b.mean ∧ a.variance < b.variance)

/-- Determines if an athlete is the most suitable among a list of athletes --/
def mostSuitable (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, a ≠ b → moreSuitable a b

theorem athlete_c_most_suitable :
  let athletes := [
    Athlete.mk "A" 380 12.5,
    Athlete.mk "B" 360 13.5,
    Athlete.mk "C" 380 2.4,
    Athlete.mk "D" 350 2.7
  ]
  let c := Athlete.mk "C" 380 2.4
  mostSuitable c athletes := by
  sorry

end NUMINAMATH_CALUDE_athlete_c_most_suitable_l2651_265129


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l2651_265189

theorem power_tower_mod_1000 : 3^(3^(3^3)) ≡ 387 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l2651_265189


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_3994001_l2651_265126

theorem sqrt_product_plus_one_equals_3994001 :
  Real.sqrt (1997 * 1998 * 1999 * 2000 + 1) = 3994001 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_3994001_l2651_265126


namespace NUMINAMATH_CALUDE_point_sum_coordinates_l2651_265183

/-- Given that (3, 8) is on the graph of y = g(x), prove that the sum of the coordinates
    of the point on the graph of 5y = 4g(2x) + 6 is 9.1 -/
theorem point_sum_coordinates (g : ℝ → ℝ) (h : g 3 = 8) :
  ∃ x y : ℝ, 5 * y = 4 * g (2 * x) + 6 ∧ x + y = 9.1 := by
  sorry

end NUMINAMATH_CALUDE_point_sum_coordinates_l2651_265183


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2651_265137

/-- Prove the ratio of monkeys to camels at the zoo -/
theorem zoo_animal_ratio : 
  ∀ (zebras camels monkeys giraffes : ℕ),
    zebras = 12 →
    camels = zebras / 2 →
    ∃ k : ℕ, monkeys = k * camels →
    giraffes = 2 →
    monkeys = giraffes + 22 →
    monkeys / camels = 4 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2651_265137


namespace NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l2651_265117

/-- Proves that if an item is sold at a 20% loss for 600 currency units, 
    then its original price was 750 currency units. -/
theorem original_price_from_loss_and_selling_price 
  (selling_price : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : selling_price = 600) 
  (h2 : loss_percentage = 20) : 
  ∃ original_price : ℝ, 
    original_price = 750 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l2651_265117


namespace NUMINAMATH_CALUDE_largest_decimal_l2651_265114

theorem largest_decimal (a b c d e : ℚ) 
  (ha : a = 0.803) 
  (hb : b = 0.809) 
  (hc : c = 0.8039) 
  (hd : d = 0.8091) 
  (he : e = 0.8029) : 
  c = max a (max b (max c (max d e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2651_265114


namespace NUMINAMATH_CALUDE_penguin_sea_horse_difference_l2651_265174

/-- Given a ratio of sea horses to penguins and the number of sea horses,
    calculate the difference between the number of penguins and sea horses. -/
theorem penguin_sea_horse_difference 
  (ratio_sea_horses : ℕ) 
  (ratio_penguins : ℕ) 
  (num_sea_horses : ℕ) 
  (h1 : ratio_sea_horses = 5) 
  (h2 : ratio_penguins = 11) 
  (h3 : num_sea_horses = 70) :
  (ratio_penguins * (num_sea_horses / ratio_sea_horses)) - num_sea_horses = 84 :=
by
  sorry

#check penguin_sea_horse_difference

end NUMINAMATH_CALUDE_penguin_sea_horse_difference_l2651_265174


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2651_265167

theorem quadratic_rewrite_sum (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 171) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2651_265167


namespace NUMINAMATH_CALUDE_study_group_equation_system_l2651_265155

theorem study_group_equation_system (x y : ℤ) : 
  (5 * y + 3 = x ∧ 6 * y - 3 = x) → 
  (5 * y = x - 3 ∧ 6 * y = x + 3) := by
  sorry

end NUMINAMATH_CALUDE_study_group_equation_system_l2651_265155


namespace NUMINAMATH_CALUDE_radical_combination_l2651_265146

theorem radical_combination (x : ℝ) : (2 + x = 5 - 2*x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_radical_combination_l2651_265146


namespace NUMINAMATH_CALUDE_card_sorting_theorem_l2651_265100

/-- A function that represents the cost of sorting n cards -/
def sortingCost (n : ℕ) : ℕ := sorry

/-- The theorem states that 365 cards can be sorted within 2000 comparisons -/
theorem card_sorting_theorem :
  ∃ (f : ℕ → ℕ), 
    (∀ n ≤ 365, f n ≤ sortingCost n) ∧ 
    (f 365 ≤ 2000) := by
  sorry

/-- The cost of sorting 3 cards is 1 -/
axiom sort_three_cost : sortingCost 3 = 1

/-- The cost of sorting n+1 cards is at most k+1 if n ≤ 3^k -/
axiom sort_cost_bound (n k : ℕ) :
  n ≤ 3^k → sortingCost (n + 1) ≤ sortingCost n + k + 1

/-- There are 365 cards -/
def total_cards : ℕ := 365

/-- The maximum allowed cost is 2000 -/
def max_cost : ℕ := 2000

end NUMINAMATH_CALUDE_card_sorting_theorem_l2651_265100


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l2651_265131

/-- Given two lines intersecting at point P(2,5) with slopes -1 and -2 respectively,
    and points Q and R on the x-axis, prove that the area of triangle PQR is 6.25 -/
theorem area_of_triangle_PQR : ∃ (Q R : ℝ × ℝ),
  let P : ℝ × ℝ := (2, 5)
  let slope_PQ : ℝ := -1
  let slope_PR : ℝ := -2
  Q.2 = 0 ∧ R.2 = 0 ∧
  (Q.1 - P.1) / (Q.2 - P.2) = slope_PQ ∧
  (R.1 - P.1) / (R.2 - P.2) = slope_PR ∧
  (1/2 : ℝ) * |Q.1 - R.1| * P.2 = 6.25 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_l2651_265131


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2651_265145

/-- A right triangle with sides 5, 12, and 13 (hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  hypotenuse : c = 13
  right_angle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12

/-- First inscribed square with side length x -/
def first_square (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- Second inscribed square with side length y -/
def second_square (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a - y) / y = (t.b - y) / y

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (h1 : first_square t x) (h2 : second_square t y) : 
  x / y = 78 / 102 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2651_265145


namespace NUMINAMATH_CALUDE_divisor_property_l2651_265111

theorem divisor_property (k : ℕ) : 
  (15 ^ k) ∣ 759325 → 3 ^ k - 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l2651_265111


namespace NUMINAMATH_CALUDE_sum_product_bounds_l2651_265148

theorem sum_product_bounds (x y z : ℝ) (h : x + y + z = 3) :
  -3/2 ≤ x*y + x*z + y*z ∧ x*y + x*z + y*z ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l2651_265148


namespace NUMINAMATH_CALUDE_salary_spending_problem_l2651_265170

/-- The problem statement about salaries and spending --/
theorem salary_spending_problem 
  (total_salary : ℝ)
  (a_salary : ℝ)
  (a_spend_percent : ℝ)
  (ha_total : total_salary = 6000)
  (ha_salary : a_salary = 4500)
  (ha_spend : a_spend_percent = 0.95)
  (h_equal_savings : a_salary * (1 - a_spend_percent) = (total_salary - a_salary) - ((total_salary - a_salary) * (85 / 100))) :
  (((total_salary - a_salary) - ((total_salary - a_salary) * (1 - 85 / 100))) / (total_salary - a_salary)) * 100 = 85 := by
sorry


end NUMINAMATH_CALUDE_salary_spending_problem_l2651_265170


namespace NUMINAMATH_CALUDE_custom_op_value_l2651_265112

/-- Custom operation * for non-zero integers -/
def custom_op (a b : ℤ) : ℚ := 1 / a + 1 / b

theorem custom_op_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 32) :
  custom_op a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_value_l2651_265112


namespace NUMINAMATH_CALUDE_power_sum_and_division_l2651_265153

theorem power_sum_and_division (a b c : ℕ) :
  2^345 + 9^5 / 9^3 = 2^345 + 81 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_l2651_265153


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2651_265140

theorem complex_equation_solution (z : ℂ) :
  z / (z - Complex.I) = Complex.I → z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2651_265140


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2651_265165

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem binary_multiplication_division_equality :
  let a := [false, true, true, true, false, true] -- 101110₂
  let b := [false, false, true, false, true, false, true] -- 1010100₂
  let c := [false, false, true] -- 100₂
  let result_binary := [true, true, false, false, true, true, false, true, true, false, true, true] -- 101110110011₂
  let result_decimal : ℕ := 2995
  (binary_to_decimal a * binary_to_decimal b) / binary_to_decimal c = binary_to_decimal result_binary ∧
  binary_to_decimal result_binary = result_decimal ∧
  decimal_to_binary result_decimal = result_binary :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2651_265165


namespace NUMINAMATH_CALUDE_is_quadratic_equation_f_l2651_265152

-- Define a quadratic equation in one variable
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation (x-1)(x+2)=1
def f (x : ℝ) : ℝ := (x - 1) * (x + 2) - 1

-- Theorem statement
theorem is_quadratic_equation_f : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_f_l2651_265152


namespace NUMINAMATH_CALUDE_pet_store_dogs_l2651_265106

theorem pet_store_dogs (initial_dogs : ℕ) : 
  initial_dogs + 5 + 3 = 10 → initial_dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l2651_265106


namespace NUMINAMATH_CALUDE_system_solution_unique_l2651_265105

theorem system_solution_unique :
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x + 2 * y = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2651_265105


namespace NUMINAMATH_CALUDE_puppies_adoption_l2651_265186

theorem puppies_adoption (first_week : ℕ) : 
  first_week + (2/5 : ℚ) * first_week + 2 * ((2/5 : ℚ) * first_week) + (first_week + 10) = 74 → 
  first_week = 20 := by
sorry

end NUMINAMATH_CALUDE_puppies_adoption_l2651_265186


namespace NUMINAMATH_CALUDE_quentavious_gum_pieces_l2651_265182

/-- Calculates the number of gum pieces received in an exchange. -/
def gum_pieces_received (initial_nickels : ℕ) (gum_per_nickel : ℕ) (remaining_nickels : ℕ) : ℕ :=
  (initial_nickels - remaining_nickels) * gum_per_nickel

/-- Proves that Quentavious received 6 pieces of gum. -/
theorem quentavious_gum_pieces :
  gum_pieces_received 5 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quentavious_gum_pieces_l2651_265182


namespace NUMINAMATH_CALUDE_sticker_distribution_l2651_265119

/-- The number of ways to partition n identical objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem sticker_distribution : partition_count 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2651_265119


namespace NUMINAMATH_CALUDE_jane_donuts_problem_l2651_265176

theorem jane_donuts_problem :
  ∀ (d c : ℕ),
  d + c = 6 →
  90 * d + 60 * c = 450 →
  d = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_donuts_problem_l2651_265176


namespace NUMINAMATH_CALUDE_import_tax_percentage_l2651_265160

theorem import_tax_percentage 
  (total_value : ℝ) 
  (non_taxed_portion : ℝ) 
  (import_tax_amount : ℝ) 
  (h1 : total_value = 2580) 
  (h2 : non_taxed_portion = 1000) 
  (h3 : import_tax_amount = 110.60) : 
  (import_tax_amount / (total_value - non_taxed_portion)) * 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_percentage_l2651_265160


namespace NUMINAMATH_CALUDE_two_dice_outcomes_l2651_265141

/-- The number of possible outcomes for a single die. -/
def outcomes_per_die : ℕ := 6

/-- The total number of possible outcomes when throwing two identical dice simultaneously. -/
def total_outcomes : ℕ := outcomes_per_die * outcomes_per_die

/-- Theorem stating that the total number of possible outcomes when throwing two identical dice simultaneously is 36. -/
theorem two_dice_outcomes : total_outcomes = 36 := by
  sorry

end NUMINAMATH_CALUDE_two_dice_outcomes_l2651_265141


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2651_265166

/-- A rhombus with given perimeter and one diagonal -/
structure Rhombus where
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem: In a rhombus with perimeter 52 and one diagonal 10, the other diagonal is 24 -/
theorem rhombus_diagonal (r : Rhombus) (h1 : r.perimeter = 52) (h2 : r.diagonal2 = 10) :
  ∃ (diagonal1 : ℝ), diagonal1 = 24 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_diagonal_l2651_265166


namespace NUMINAMATH_CALUDE_B_power_2023_l2651_265143

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, -1]]

theorem B_power_2023 :
  B ^ 2023 = ![![ 0,  1,  0],
               ![-1,  0,  0],
               ![ 0,  0, -1]] := by sorry

end NUMINAMATH_CALUDE_B_power_2023_l2651_265143


namespace NUMINAMATH_CALUDE_smallest_whole_number_solution_l2651_265164

theorem smallest_whole_number_solution : 
  (∀ n : ℕ, n < 6 → (2 : ℚ) / 5 + (n : ℚ) / 9 ≤ 1) ∧ 
  ((2 : ℚ) / 5 + (6 : ℚ) / 9 > 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_solution_l2651_265164


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2651_265178

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_inequality
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : a 1 = b 1)
  (h1_pos : a 1 > 0)
  (h11 : a 11 = b 11)
  (h11_pos : a 11 > 0) :
  a 6 ≥ b 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2651_265178


namespace NUMINAMATH_CALUDE_max_points_tournament_l2651_265109

-- Define the number of teams
def num_teams : ℕ := 8

-- Define the number of top teams with equal points
def num_top_teams : ℕ := 4

-- Define the points for win, draw, and loss
def win_points : ℕ := 3
def draw_points : ℕ := 1
def loss_points : ℕ := 0

-- Define the function to calculate the total number of games
def total_games (n : ℕ) : ℕ := n.choose 2 * 2

-- Define the function to calculate the maximum points for top teams
def max_points_top_team (n : ℕ) (k : ℕ) : ℕ :=
  (k - 1) * 3 + (n - k) * 3 * 2

-- Theorem statement
theorem max_points_tournament :
  max_points_top_team num_teams num_top_teams = 33 :=
sorry

end NUMINAMATH_CALUDE_max_points_tournament_l2651_265109


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2651_265161

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2651_265161


namespace NUMINAMATH_CALUDE_f_properties_l2651_265175

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 6 + Real.cos x ^ 6

theorem f_properties :
  (∀ x, f x ∈ Set.Icc (1/4 : ℝ) 1) ∧
  (∀ ε > 0, ∃ p ∈ Set.Ioo 0 ε, ∀ x, f (x + p) = f x) ∧
  (∀ k : ℤ, ∀ x, f (k * Real.pi / 4 - x) = f (k * Real.pi / 4 + x)) ∧
  (∀ k : ℤ, f (Real.pi / 8 + k * Real.pi / 4) = 5/8) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2651_265175


namespace NUMINAMATH_CALUDE_factor_expression_l2651_265194

theorem factor_expression (x : ℝ) : 5*x*(x+2) + 9*(x+2) = (x+2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2651_265194


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l2651_265138

/-- The quadratic equation x^2 + 2x - 1 = 0 is equivalent to (x+1)^2 = 2 -/
theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 + 2*x - 1 = 0 ↔ (x + 1)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l2651_265138


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l2651_265147

theorem smallest_number_of_eggs : ∀ n : ℕ,
  (∃ c : ℕ, n = 12 * c - 3) →  -- Eggs are in containers of 12, with 3 containers having 11 eggs
  n > 200 →                   -- More than 200 eggs
  n ≥ 201                     -- The smallest possible number is at least 201
:= by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l2651_265147


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l2651_265162

/-- Given a boat rowing in a stream, prove that the ratio of the boat's speed in still water
    to the average speed of the stream is 3:1, under certain conditions. -/
theorem boat_stream_speed_ratio :
  ∀ (B S : ℝ), 
    B > 0 → -- The boat's speed in still water is positive
    S > 0 → -- The stream's average speed is positive
    (B - S) / (B + S) = 1 / 2 → -- It takes twice as long to row against the stream as with it
    B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l2651_265162


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l2651_265196

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 + 17*x - 60 = 0 ∧ ∀ y, y^2 + 17*y - 60 = 0 → x ≤ y →
  x = -20 :=
sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l2651_265196


namespace NUMINAMATH_CALUDE_distinct_ages_count_l2651_265132

def average_age : ℕ := 31
def standard_deviation : ℕ := 5

def lower_bound : ℕ := average_age - standard_deviation
def upper_bound : ℕ := average_age + standard_deviation

theorem distinct_ages_count : 
  (Finset.range (upper_bound - lower_bound + 1)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_ages_count_l2651_265132


namespace NUMINAMATH_CALUDE_ping_pong_theorem_l2651_265113

/-- Represents a ping-pong match result between two players -/
inductive MatchResult
| Win
| Lose

/-- Represents a ping-pong team -/
def Team := Fin 1000

/-- Represents the result of all matches between two teams -/
def MatchResults := Team → Team → MatchResult

theorem ping_pong_theorem (results : MatchResults) : 
  ∃ (winning_team : Bool) (subset : Finset Team),
    subset.card ≤ 10 ∧ 
    ∀ (player : Team), 
      ∃ (winner : Team), winner ∈ subset ∧ 
        (if winning_team then 
          results winner player = MatchResult.Win
        else
          results player winner = MatchResult.Lose) :=
sorry

end NUMINAMATH_CALUDE_ping_pong_theorem_l2651_265113


namespace NUMINAMATH_CALUDE_paige_mp3_songs_l2651_265163

/-- Calculates the final number of songs on an mp3 player after deleting and adding songs. -/
def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

/-- Theorem: The final number of songs on Paige's mp3 player is 10. -/
theorem paige_mp3_songs : final_song_count 11 9 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paige_mp3_songs_l2651_265163


namespace NUMINAMATH_CALUDE_food_product_range_l2651_265115

/-- Represents the net content of a food product -/
structure NetContent where
  nominal : ℝ
  tolerance : ℝ

/-- Represents a range of values -/
structure Range where
  lower : ℝ
  upper : ℝ

/-- Calculates the qualified net content range for a given net content -/
def qualifiedRange (nc : NetContent) : Range :=
  { lower := nc.nominal - nc.tolerance,
    upper := nc.nominal + nc.tolerance }

/-- Theorem: The qualified net content range for a product labeled "500g ± 5g" is 495g to 505g -/
theorem food_product_range :
  let nc : NetContent := { nominal := 500, tolerance := 5 }
  let range := qualifiedRange nc
  range.lower = 495 ∧ range.upper = 505 := by
  sorry

end NUMINAMATH_CALUDE_food_product_range_l2651_265115


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_is_7_or_8_l2651_265136

def isosceles_triangle_perimeter (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∧  -- positive side lengths
  (x = y ∨ x + y > x)  -- triangle inequality
  ∧ y = Real.sqrt (2 - x) + Real.sqrt (3 * x - 6) + 3

theorem isosceles_triangle_perimeter_is_7_or_8 :
  ∀ x y : ℝ, isosceles_triangle_perimeter x y →
  (x + y + (if x = y then x else y) = 7 ∨ x + y + (if x = y then x else y) = 8) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_is_7_or_8_l2651_265136


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2651_265191

theorem arithmetic_sequence_length : 
  ∀ (a₁ : ℝ) (d : ℝ) (aₙ : ℝ),
  a₁ = 2.5 → d = 5 → aₙ = 72.5 →
  ∃ (n : ℕ), n = 15 ∧ aₙ = a₁ + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2651_265191


namespace NUMINAMATH_CALUDE_max_value_of_g_l2651_265101

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 * x - 2 * x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (x_max : ℝ), x_max ∈ Set.Icc (-2) 2 ∧
  (∀ x ∈ Set.Icc (-2) 2, g x ≤ g x_max) ∧
  g x_max = 6 ∧ x_max = -2 := by
  sorry


end NUMINAMATH_CALUDE_max_value_of_g_l2651_265101


namespace NUMINAMATH_CALUDE_total_annual_income_percentage_l2651_265135

def initial_investment : ℝ := 2800
def initial_rate : ℝ := 0.05
def additional_investment : ℝ := 1400
def additional_rate : ℝ := 0.08

theorem total_annual_income_percentage :
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
sorry

end NUMINAMATH_CALUDE_total_annual_income_percentage_l2651_265135


namespace NUMINAMATH_CALUDE_crofton_orchestra_max_members_l2651_265102

theorem crofton_orchestra_max_members :
  ∀ n : ℕ,
  (25 * n < 1000) →
  (25 * n % 24 = 5) →
  (∀ m : ℕ, (25 * m < 1000) ∧ (25 * m % 24 = 5) → m ≤ n) →
  25 * n = 725 :=
by
  sorry

end NUMINAMATH_CALUDE_crofton_orchestra_max_members_l2651_265102


namespace NUMINAMATH_CALUDE_range_of_a_l2651_265110

open Real

theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x * log x) →
  (∀ x, g x = x^3 + a*x^2 - x + 2) →
  (∀ x > 0, 2 * f x ≤ (deriv g) x + 2) →
  a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2651_265110


namespace NUMINAMATH_CALUDE_root_in_interval_l2651_265158

-- Define the function f(x) = x^3 + 3x - 1
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem root_in_interval :
  (f 0 < 0) → (f 1 > 0) → ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2651_265158


namespace NUMINAMATH_CALUDE_alpha_para_beta_sufficient_not_necessary_l2651_265187

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraLine : Line → Plane → Prop)

-- State the theorem
theorem alpha_para_beta_sufficient_not_necessary 
  (l m : Line) (α β : Plane) 
  (h1 : perp l α) 
  (h2 : paraLine m β) : 
  (∃ (config : Type), 
    (∀ (α β : Plane), para α β → perpLine l m) ∧ 
    (∃ (α β : Plane), perpLine l m ∧ ¬ para α β)) :=
sorry

end NUMINAMATH_CALUDE_alpha_para_beta_sufficient_not_necessary_l2651_265187


namespace NUMINAMATH_CALUDE_perfect_square_prime_l2651_265118

theorem perfect_square_prime (p : ℕ) (n : ℕ) : 
  Nat.Prime p → (5^p + 4*p^4 = n^2) → p = 5 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_prime_l2651_265118


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2651_265124

theorem min_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := by
  sorry

theorem lower_bound_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2651_265124


namespace NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l2651_265130

theorem limit_sequence_equals_one_over_e :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((2*n - 1) / (2*n + 1))^(n + 1) - 1/Real.exp 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l2651_265130


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2651_265173

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2651_265173


namespace NUMINAMATH_CALUDE_bakers_cakes_l2651_265103

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes bought_cakes sold_cakes : ℕ) 
  (h1 : initial_cakes = 8)
  (h2 : bought_cakes = 139)
  (h3 : sold_cakes = 145) :
  sold_cakes - bought_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l2651_265103


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2651_265197

/-- Proves that given a journey of 448 km completed in 20 hours, where the first half is traveled at 21 km/hr, the speed for the second half must be 24 km/hr. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  first_half_speed = 21 →
  (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time →
  second_half_speed = 24 := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_journey_speed_calculation_l2651_265197


namespace NUMINAMATH_CALUDE_toy_production_difference_l2651_265180

/-- The difference in daily toy production between two machines -/
theorem toy_production_difference : 
  let machine_a_total : ℕ := 288
  let machine_a_days : ℕ := 12
  let machine_b_total : ℕ := 243
  let machine_b_days : ℕ := 9
  let machine_a_daily : ℚ := machine_a_total / machine_a_days
  let machine_b_daily : ℚ := machine_b_total / machine_b_days
  machine_b_daily - machine_a_daily = 3 := by
sorry

end NUMINAMATH_CALUDE_toy_production_difference_l2651_265180


namespace NUMINAMATH_CALUDE_problem_solution_l2651_265121

theorem problem_solution (x y : ℝ) 
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x * y + x + y = 5) :
  x^2 * y + x * y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2651_265121


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l2651_265185

theorem reciprocal_equals_self (x : ℝ) : (1 / x = x) → (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l2651_265185


namespace NUMINAMATH_CALUDE_function_bounded_by_square_l2651_265107

/-- A function satisfying the given inequality is bounded by x² -/
theorem function_bounded_by_square {f : ℝ → ℝ} (hf_nonneg : ∀ x ≥ 0, f x ≥ 0)
  (hf_bounded : ∃ M > 0, ∀ x ∈ Set.Icc 0 1, f x ≤ M)
  (h_ineq : ∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ x^2 * f (y/2) + y^2 * f (x/2)) :
  ∀ x ≥ 0, f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_bounded_by_square_l2651_265107


namespace NUMINAMATH_CALUDE_ball_volume_ratio_l2651_265123

theorem ball_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) (h₃ : r₃ = 3 * r₁) :
  (4 / 3 * π * r₃^3) = 3 * ((4 / 3 * π * r₁^3) + (4 / 3 * π * r₂^3)) :=
by sorry

end NUMINAMATH_CALUDE_ball_volume_ratio_l2651_265123


namespace NUMINAMATH_CALUDE_probability_red_or_white_is_five_sixths_l2651_265144

def total_marbles : ℕ := 30
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9

def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

def probability_red_or_white : ℚ :=
  (red_marbles + white_marbles : ℚ) / total_marbles

theorem probability_red_or_white_is_five_sixths :
  probability_red_or_white = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_is_five_sixths_l2651_265144


namespace NUMINAMATH_CALUDE_cow_daily_water_consumption_l2651_265151

/-- The number of cows on Mr. Reyansh's farm -/
def num_cows : ℕ := 40

/-- The ratio of sheep to cows on Mr. Reyansh's farm -/
def sheep_to_cow_ratio : ℕ := 10

/-- The ratio of water consumption of a sheep to a cow -/
def sheep_to_cow_water_ratio : ℚ := 1/4

/-- Total water usage for all animals in a week (in liters) -/
def total_weekly_water : ℕ := 78400

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem cow_daily_water_consumption :
  ∃ (cow_daily_water : ℚ),
    cow_daily_water * (num_cows : ℚ) * days_in_week +
    cow_daily_water * sheep_to_cow_water_ratio * (num_cows * sheep_to_cow_ratio : ℚ) * days_in_week =
    total_weekly_water ∧
    cow_daily_water = 80 := by
  sorry

end NUMINAMATH_CALUDE_cow_daily_water_consumption_l2651_265151


namespace NUMINAMATH_CALUDE_factor_x_pow_10_minus_1296_l2651_265154

theorem factor_x_pow_10_minus_1296 (x : ℝ) : x^10 - 1296 = (x^5 + 36) * (x^5 - 36) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_pow_10_minus_1296_l2651_265154


namespace NUMINAMATH_CALUDE_trig_simplification_l2651_265184

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2651_265184


namespace NUMINAMATH_CALUDE_fourth_section_area_l2651_265116

/-- Represents a regular hexagon divided into four sections by three line segments -/
structure DividedHexagon where
  total_area : ℝ
  section1_area : ℝ
  section2_area : ℝ
  section3_area : ℝ
  section4_area : ℝ
  is_regular : total_area = 6 * (section1_area + section2_area + section3_area + section4_area) / 6
  sum_of_parts : total_area = section1_area + section2_area + section3_area + section4_area

/-- The theorem stating that if three sections of a divided regular hexagon have areas 2, 3, and 4,
    then the fourth section has an area of 11 -/
theorem fourth_section_area (h : DividedHexagon) 
    (h2 : h.section1_area = 2) 
    (h3 : h.section2_area = 3) 
    (h4 : h.section3_area = 4) : 
    h.section4_area = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_section_area_l2651_265116


namespace NUMINAMATH_CALUDE_tristan_study_hours_l2651_265104

/-- Represents the number of hours Tristan studies each day of the week -/
structure StudyHours where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Theorem stating the number of hours Tristan studies from Wednesday to Friday -/
theorem tristan_study_hours (h : StudyHours) : 
  h.monday = 4 ∧ 
  h.tuesday = 2 * h.monday ∧ 
  h.wednesday = h.thursday ∧ 
  h.thursday = h.friday ∧ 
  h.monday + h.tuesday + h.wednesday + h.thursday + h.friday + h.saturday + h.sunday = 25 ∧ 
  h.saturday = h.sunday → 
  h.wednesday = 13/3 := by
sorry

#eval 13/3  -- To show the result is approximately 4.33

end NUMINAMATH_CALUDE_tristan_study_hours_l2651_265104


namespace NUMINAMATH_CALUDE_continuous_function_solution_l2651_265133

open Set
open Function
open Real

theorem continuous_function_solution {f : ℝ → ℝ} (hf : Continuous f) 
  (hdom : ∀ x, x ∈ Ioo (-1) 1 → f x ≠ 0) 
  (heq : ∀ x ∈ Ioo (-1) 1, (1 - x^2) * f ((2*x) / (1 + x^2)) = (1 + x^2)^2 * f x) :
  ∃ c : ℝ, ∀ x ∈ Ioo (-1) 1, f x = c / (1 - x^2) :=
sorry

end NUMINAMATH_CALUDE_continuous_function_solution_l2651_265133


namespace NUMINAMATH_CALUDE_average_math_chem_score_l2651_265149

theorem average_math_chem_score (math physics chem : ℕ) : 
  math + physics = 40 →
  chem = physics + 20 →
  (math + chem) / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_average_math_chem_score_l2651_265149


namespace NUMINAMATH_CALUDE_triangle_cosC_l2651_265181

theorem triangle_cosC (A B C : Real) (a b c : Real) : 
  -- Conditions
  (a = 2) →
  (b = 3) →
  (C = 2 * A) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Law of cosines
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Conclusion
  Real.cos C = 1/4 := by sorry

end NUMINAMATH_CALUDE_triangle_cosC_l2651_265181


namespace NUMINAMATH_CALUDE_rounded_number_bounds_l2651_265179

def rounded_number : ℕ := 180000

theorem rounded_number_bounds :
  ∃ (min max : ℕ),
    (min ≤ rounded_number ∧ rounded_number < min + 5000) ∧
    (max - 5000 < rounded_number ∧ rounded_number ≤ max) ∧
    min = 175000 ∧ max = 184999 :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_bounds_l2651_265179


namespace NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l2651_265192

theorem same_root_implies_a_equals_three (a : ℝ) :
  (∃ x : ℝ, 3 * x - 2 * a = 0 ∧ 2 * x + 3 * a - 13 = 0) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l2651_265192


namespace NUMINAMATH_CALUDE_range_of_m_l2651_265199

def p (m : ℝ) : Prop := ∀ x > 0, (1/2)^x + m - 1 < 0

def q (m : ℝ) : Prop := ∃ x > 0, m*x^2 + 4*x - 1 = 0

theorem range_of_m : ∀ m : ℝ, (p m ∧ q m) ↔ m ∈ Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2651_265199


namespace NUMINAMATH_CALUDE_amir_weight_l2651_265156

theorem amir_weight (bulat_weight ilnur_weight : ℝ) : 
  let amir_weight := ilnur_weight + 8
  let daniyar_weight := bulat_weight + 4
  -- The sum of the weights of the heaviest and lightest boys is 2 kg less than the sum of the weights of the other two boys
  (amir_weight + ilnur_weight = daniyar_weight + bulat_weight - 2) →
  -- All four boys together weigh 250 kg
  (amir_weight + ilnur_weight + daniyar_weight + bulat_weight = 250) →
  amir_weight = 67 := by
sorry

end NUMINAMATH_CALUDE_amir_weight_l2651_265156


namespace NUMINAMATH_CALUDE_crackers_eaten_equals_180_l2651_265190

/-- Calculates the total number of animal crackers eaten by Mrs. Gable's students -/
def total_crackers_eaten (total_students : ℕ) (students_not_eating : ℕ) (crackers_per_pack : ℕ) : ℕ :=
  (total_students - students_not_eating) * crackers_per_pack

/-- Proves that the total number of animal crackers eaten is 180 -/
theorem crackers_eaten_equals_180 :
  total_crackers_eaten 20 2 10 = 180 := by
  sorry

#eval total_crackers_eaten 20 2 10

end NUMINAMATH_CALUDE_crackers_eaten_equals_180_l2651_265190
