import Mathlib

namespace NUMINAMATH_CALUDE_fruit_eating_permutations_l3198_319843

theorem fruit_eating_permutations :
  let total_fruits : ℕ := 4 + 2 + 1
  let apple_count : ℕ := 4
  let orange_count : ℕ := 2
  let banana_count : ℕ := 1
  (Nat.factorial total_fruits) / 
  (Nat.factorial apple_count * Nat.factorial orange_count * Nat.factorial banana_count) = 105 := by
sorry

end NUMINAMATH_CALUDE_fruit_eating_permutations_l3198_319843


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_5x5x5_l3198_319854

/-- Represents a cube composed of smaller unit cubes --/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ
  painted_surface : Bool

/-- Calculates the number of unpainted cubes in a large cube --/
def count_unpainted_cubes (c : LargeCube) : ℕ :=
  if c.painted_surface then (c.side_length - 2)^3 else c.total_cubes

/-- Theorem stating that a 5x5x5 cube with painted surface has 27 unpainted cubes --/
theorem unpainted_cubes_in_5x5x5 :
  let c : LargeCube := { side_length := 5, total_cubes := 125, painted_surface := true }
  count_unpainted_cubes c = 27 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_5x5x5_l3198_319854


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3198_319861

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3198_319861


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3198_319838

theorem perfect_square_condition (x : ℤ) : 
  (∃ y : ℤ, x^2 + 19*x + 95 = y^2) ↔ (x = -14 ∨ x = -5) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3198_319838


namespace NUMINAMATH_CALUDE_karina_birth_year_l3198_319835

def current_year : ℕ := 2022
def brother_birth_year : ℕ := 1990

theorem karina_birth_year (karina_age brother_age : ℕ) 
  (h1 : karina_age = 2 * brother_age)
  (h2 : brother_age = current_year - brother_birth_year) :
  current_year - karina_age = 1958 := by
sorry

end NUMINAMATH_CALUDE_karina_birth_year_l3198_319835


namespace NUMINAMATH_CALUDE_single_digit_equation_l3198_319819

theorem single_digit_equation (a b : ℕ) : 
  (0 < a ∧ a < 10) →
  (0 < b ∧ b < 10) →
  82 * 10 * a + 7 + 6 * b = 190 →
  a + 2 * b = 6 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_single_digit_equation_l3198_319819


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3198_319872

theorem cube_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b*c)) + (b^3 / (c*a)) + (c^3 / (a*b)) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3198_319872


namespace NUMINAMATH_CALUDE_annuity_distribution_l3198_319865

/-- Represents the annuity problem with three people inheriting an annuity --/
theorem annuity_distribution (e : Real) (h_e : e = 1.04) :
  let x := Real.log ((e^25 - 1) / 3 + 1)
  let y := Real.log ((Real.exp x * (e^25 - 1)) / 3 + 1)
  let z := 25 - (x + y)
  (x + y + z = 25) ∧ 
  (Real.exp x - 1 = (e^25 - 1) / 3) ∧
  (Real.exp y - 1 = Real.exp x * (e^25 - 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_annuity_distribution_l3198_319865


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3198_319877

/-- Given a parabola y = ax^2 + bx + c with vertex at (p, p) and y-intercept at (0, -3p),
    where p ≠ 0, the coefficient b is equal to 8/p. -/
theorem parabola_coefficient (a b c p : ℝ) : 
  p ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p) → 
  a * 0^2 + b * 0 + c = -3 * p → 
  b = 8 / p := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3198_319877


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3198_319871

theorem quadratic_root_implies_a (a : ℝ) : (2^2 - 2 + a = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3198_319871


namespace NUMINAMATH_CALUDE_almond_croissant_price_l3198_319828

/-- The price of an almond croissant given Harrison's croissant buying habits -/
theorem almond_croissant_price :
  let regular_price : ℚ := 7/2  -- $3.50
  let weeks_in_year : ℕ := 52
  let total_spent : ℚ := 468
  let almond_price : ℚ := (total_spent - weeks_in_year * regular_price) / weeks_in_year
  almond_price = 11/2  -- $5.50
  := by sorry

end NUMINAMATH_CALUDE_almond_croissant_price_l3198_319828


namespace NUMINAMATH_CALUDE_biology_magnet_combinations_l3198_319811

def word : String := "BIOLOGY"

def num_vowels : Nat := 3
def num_consonants : Nat := 2
def num_Os : Nat := 2

def vowels : Finset Char := {'I', 'O'}
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

theorem biology_magnet_combinations : 
  (Finset.card (Finset.powerset vowels) * Finset.card (Finset.powerset consonants)) +
  (Finset.card (Finset.powerset {0, 1}) * Finset.card (Finset.powerset consonants)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_biology_magnet_combinations_l3198_319811


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3198_319879

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x

-- State the theorem
theorem quadratic_max_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f m x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f m x = 3) →
  m = -4 ∨ m = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3198_319879


namespace NUMINAMATH_CALUDE_slipper_discount_percentage_l3198_319829

/-- Calculates the discount percentage on slippers given the original price, 
    embroidery cost per shoe, shipping cost, and final discounted price. -/
theorem slipper_discount_percentage 
  (original_price : ℝ) 
  (embroidery_cost_per_shoe : ℝ) 
  (shipping_cost : ℝ) 
  (final_price : ℝ) : 
  original_price = 50 ∧ 
  embroidery_cost_per_shoe = 5.5 ∧ 
  shipping_cost = 10 ∧ 
  final_price = 66 →
  (original_price - (final_price - shipping_cost - 2 * embroidery_cost_per_shoe)) / original_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_slipper_discount_percentage_l3198_319829


namespace NUMINAMATH_CALUDE_lehmer_mean_properties_l3198_319891

noncomputable def A (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def G (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def L (p a b : ℝ) : ℝ := (a^p + b^p) / (a^(p-1) + b^(p-1))

theorem lehmer_mean_properties (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  L 0.5 a b ≤ A a b ∧
  L 0 a b ≥ G a b ∧
  L 2 a b ≥ L 1 a b ∧
  ∃ n, L (n + 1) a b > L n a b :=
sorry

end NUMINAMATH_CALUDE_lehmer_mean_properties_l3198_319891


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3198_319887

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_ratio : a / b = 36 / 49) :
  (4 * Real.sqrt a) / (4 * Real.sqrt b) = 6 / 7 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3198_319887


namespace NUMINAMATH_CALUDE_function_value_at_five_l3198_319857

open Real

theorem function_value_at_five
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > -3 / x)
  (h2 : ∀ x > 0, f (f x + 3 / x) = 2) :
  f 5 = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_five_l3198_319857


namespace NUMINAMATH_CALUDE_latch_caught_14_necklaces_l3198_319824

/-- The number of necklaces caught by Boudreaux -/
def boudreaux_necklaces : ℕ := 12

/-- The number of necklaces caught by Rhonda -/
def rhonda_necklaces : ℕ := boudreaux_necklaces / 2

/-- The number of necklaces caught by Latch -/
def latch_necklaces : ℕ := 3 * rhonda_necklaces - 4

/-- Theorem stating that Latch caught 14 necklaces -/
theorem latch_caught_14_necklaces : latch_necklaces = 14 := by
  sorry

end NUMINAMATH_CALUDE_latch_caught_14_necklaces_l3198_319824


namespace NUMINAMATH_CALUDE_chicken_egg_production_l3198_319873

theorem chicken_egg_production (num_chickens : ℕ) (total_eggs : ℕ) (num_days : ℕ) 
  (h1 : num_chickens = 4)
  (h2 : total_eggs = 36)
  (h3 : num_days = 3) :
  total_eggs / (num_chickens * num_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l3198_319873


namespace NUMINAMATH_CALUDE_parabola_tangent_through_origin_l3198_319882

theorem parabola_tangent_through_origin (c : ℝ) : 
  (∃ y : ℝ, (y = (-2)^2 - (-2) + c) ∧ 
   (0 = y + 5 * 2)) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_through_origin_l3198_319882


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3198_319867

-- Define the functions
def f (x a b : ℝ) : ℝ := -2 * abs (x - a) + b
def g (x c d : ℝ) : ℝ := 2 * abs (x - c) + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) : 
  (f 1 a b = g 1 c d) ∧ (f 7 a b = g 7 c d) → a + c = 8 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_sum_l3198_319867


namespace NUMINAMATH_CALUDE_total_pies_baked_l3198_319895

/-- The number of pies Eddie can bake in a day -/
def eddie_pies_per_day : ℕ := 3

/-- The number of pies Eddie's sister can bake in a day -/
def sister_pies_per_day : ℕ := 6

/-- The number of pies Eddie's mother can bake in a day -/
def mother_pies_per_day : ℕ := 8

/-- The number of days they will bake pies -/
def days_baking : ℕ := 7

/-- Theorem stating the total number of pies baked in 7 days -/
theorem total_pies_baked : 
  (eddie_pies_per_day * days_baking) + 
  (sister_pies_per_day * days_baking) + 
  (mother_pies_per_day * days_baking) = 119 := by
sorry

end NUMINAMATH_CALUDE_total_pies_baked_l3198_319895


namespace NUMINAMATH_CALUDE_initially_calculated_average_is_175_l3198_319893

/-- The initially calculated average height of a class, given:
  * The class has 20 students
  * One student's height was incorrectly recorded as 40 cm more than their actual height
  * The actual average height of the students is 173 cm
-/
def initiallyCalculatedAverage (numStudents : ℕ) (heightError : ℕ) (actualAverage : ℕ) : ℕ :=
  actualAverage + heightError / numStudents

/-- Theorem stating that the initially calculated average height is 175 cm -/
theorem initially_calculated_average_is_175 :
  initiallyCalculatedAverage 20 40 173 = 175 := by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_is_175_l3198_319893


namespace NUMINAMATH_CALUDE_function_range_theorem_l3198_319862

open Real

theorem function_range_theorem (f : ℝ → ℝ) (a b m : ℝ) :
  (∀ x, x > 0 → f x = 2 - 1/x) →
  a < b →
  (∀ x, x ∈ Set.Ioo a b ↔ f x ∈ Set.Ioo (m*a) (m*b)) →
  m ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_function_range_theorem_l3198_319862


namespace NUMINAMATH_CALUDE_box_height_proof_l3198_319842

/-- Proves that the height of boxes is 12 inches given the specified conditions. -/
theorem box_height_proof (box_length : ℝ) (box_width : ℝ) (total_volume : ℝ) 
  (cost_per_box : ℝ) (min_spend : ℝ) (h : ℝ) : 
  box_length = 20 → 
  box_width = 20 → 
  total_volume = 2160000 → 
  cost_per_box = 0.4 → 
  min_spend = 180 → 
  (total_volume / (box_length * box_width * h)) * cost_per_box = min_spend → 
  h = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_height_proof_l3198_319842


namespace NUMINAMATH_CALUDE_work_earnings_equation_l3198_319896

theorem work_earnings_equation (t : ℚ) : 
  (t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5 → t = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l3198_319896


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l3198_319852

theorem sum_of_fractions_equals_two_ninths :
  (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
  (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l3198_319852


namespace NUMINAMATH_CALUDE_factorial_difference_l3198_319864

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3198_319864


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l3198_319888

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 8

/-- The amount of money Gwen has left -/
def money_left : ℕ := 6

/-- The total amount of money Gwen received for her birthday -/
def total_money : ℕ := money_spent + money_left

theorem gwen_birthday_money : total_money = 14 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l3198_319888


namespace NUMINAMATH_CALUDE_rectangle_sides_when_perimeter_equals_area_l3198_319818

theorem rectangle_sides_when_perimeter_equals_area :
  ∀ w l : ℝ,
  w > 0 →
  l = 3 * w →
  2 * (w + l) = w * l →
  w = 8 / 3 ∧ l = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_sides_when_perimeter_equals_area_l3198_319818


namespace NUMINAMATH_CALUDE_work_completion_time_l3198_319805

theorem work_completion_time (a_half_time b_third_time : ℝ) 
  (ha : a_half_time = 70)
  (hb : b_third_time = 35) :
  let a_rate := 1 / (2 * a_half_time)
  let b_rate := 1 / (3 * b_third_time)
  1 / (a_rate + b_rate) = 60 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3198_319805


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l3198_319802

theorem least_multiple_of_25_greater_than_450 : 
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 450 → n ≥ 475 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l3198_319802


namespace NUMINAMATH_CALUDE_nested_sqrt_solution_l3198_319809

theorem nested_sqrt_solution :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_solution_l3198_319809


namespace NUMINAMATH_CALUDE_berry_multiple_l3198_319858

/-- Given the number of berries for Skylar, Steve, and Stacy, and their relationships,
    prove that the multiple of Steve's berries that Stacy has 2 more than is 3. -/
theorem berry_multiple (skylar_berries : ℕ) (steve_berries : ℕ) (stacy_berries : ℕ) 
    (h1 : skylar_berries = 20)
    (h2 : steve_berries = skylar_berries / 2)
    (h3 : stacy_berries = 32)
    (h4 : ∃ m : ℕ, stacy_berries = m * steve_berries + 2) :
  ∃ m : ℕ, m = 3 ∧ stacy_berries = m * steve_berries + 2 :=
by sorry

end NUMINAMATH_CALUDE_berry_multiple_l3198_319858


namespace NUMINAMATH_CALUDE_square_root_of_square_l3198_319841

theorem square_root_of_square (n : ℝ) (h : n = 36) : Real.sqrt (n ^ 2) = n := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l3198_319841


namespace NUMINAMATH_CALUDE_function_inequality_l3198_319844

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, (2 - x) / (deriv^[2] f x) ≤ 0)

-- State the theorem
theorem function_inequality : f 1 + f 3 > 2 * f 2 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l3198_319844


namespace NUMINAMATH_CALUDE_kindergarten_cats_count_l3198_319804

/-- Represents the number of children in each category in the kindergarten. -/
structure KindergartenPets where
  total : ℕ
  dogsOnly : ℕ
  bothPets : ℕ
  catsOnly : ℕ

/-- Calculates the total number of children with cats in the kindergarten. -/
def childrenWithCats (k : KindergartenPets) : ℕ :=
  k.catsOnly + k.bothPets

/-- Theorem stating the number of children with cats in the kindergarten. -/
theorem kindergarten_cats_count (k : KindergartenPets)
    (h1 : k.total = 30)
    (h2 : k.dogsOnly = 18)
    (h3 : k.bothPets = 6)
    (h4 : k.total = k.dogsOnly + k.catsOnly + k.bothPets) :
    childrenWithCats k = 12 := by
  sorry

#check kindergarten_cats_count

end NUMINAMATH_CALUDE_kindergarten_cats_count_l3198_319804


namespace NUMINAMATH_CALUDE_square_floor_with_57_black_tiles_has_841_total_tiles_l3198_319860

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor where
  side_length : ℕ
  is_valid : side_length > 0

/-- Calculates the number of black tiles on the diagonals of a square floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Calculates the total number of tiles on a square floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem stating that a square floor with 57 black tiles on its diagonals has 841 total tiles. -/
theorem square_floor_with_57_black_tiles_has_841_total_tiles :
  ∀ (floor : SquareFloor), black_tiles floor = 57 → total_tiles floor = 841 :=
by
  sorry

end NUMINAMATH_CALUDE_square_floor_with_57_black_tiles_has_841_total_tiles_l3198_319860


namespace NUMINAMATH_CALUDE_min_students_with_glasses_and_scarf_l3198_319816

theorem min_students_with_glasses_and_scarf (n : ℕ) 
  (h1 : n > 0)
  (h2 : ∃ k : ℕ, n * 3 = k * 7)
  (h3 : ∃ m : ℕ, n * 5 = m * 6)
  (h4 : ∀ p : ℕ, p > 0 → (∃ q : ℕ, p * 3 = q * 7) → (∃ r : ℕ, p * 5 = r * 6) → p ≥ n) :
  ∃ x : ℕ, x = 11 ∧ 
    x = n * 3 / 7 + n * 5 / 6 - n :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_glasses_and_scarf_l3198_319816


namespace NUMINAMATH_CALUDE_chess_team_shirt_numbers_l3198_319810

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 
  10 ≤ n ∧ n ≤ 99

theorem chess_team_shirt_numbers 
  (d e f : ℕ) 
  (h1 : isPrime d ∧ isPrime e ∧ isPrime f)
  (h2 : isTwoDigit d ∧ isTwoDigit e ∧ isTwoDigit f)
  (h3 : d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) :
  f = 19 := by
sorry

end NUMINAMATH_CALUDE_chess_team_shirt_numbers_l3198_319810


namespace NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_for_abs_sum_equality_l3198_319845

theorem xy_positive_sufficient_not_necessary_for_abs_sum_equality (x y : ℝ) :
  (∀ x y : ℝ, x * y > 0 → |x + y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x + y| = |x| + |y| ∧ ¬(x * y > 0)) :=
sorry

end NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_for_abs_sum_equality_l3198_319845


namespace NUMINAMATH_CALUDE_total_pay_calculation_l3198_319812

def first_job_pay : ℕ := 2125
def pay_difference : ℕ := 375

def second_job_pay : ℕ := first_job_pay - pay_difference

def total_pay : ℕ := first_job_pay + second_job_pay

theorem total_pay_calculation : total_pay = 3875 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_calculation_l3198_319812


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3198_319899

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3198_319899


namespace NUMINAMATH_CALUDE_factorization_equality_l3198_319813

theorem factorization_equality (a b : ℝ) : 2 * a^2 * b - 8 * b = 2 * b * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3198_319813


namespace NUMINAMATH_CALUDE_land_division_l3198_319847

theorem land_division (total_land : ℝ) (num_siblings : ℕ) (jose_share : ℝ) : 
  total_land = 20000 ∧ num_siblings = 4 → 
  jose_share = total_land / (num_siblings + 1) ∧ 
  jose_share = 4000 := by
sorry

end NUMINAMATH_CALUDE_land_division_l3198_319847


namespace NUMINAMATH_CALUDE_smallest_positive_integer_linear_combination_l3198_319889

theorem smallest_positive_integer_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (m n p : ℤ), ∃ (x : ℤ), 1234 * m + 56789 * n + 345 * p = x * k) ∧ 
  (∀ (l : ℕ), l > 0 → (∀ (m n p : ℤ), ∃ (x : ℤ), 1234 * m + 56789 * n + 345 * p = x * l) → l ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_linear_combination_l3198_319889


namespace NUMINAMATH_CALUDE_tobias_driveways_l3198_319868

/-- The number of driveways Tobias shoveled -/
def num_driveways : ℕ :=
  let shoe_cost : ℕ := 95
  let months_saved : ℕ := 3
  let monthly_allowance : ℕ := 5
  let lawn_mowing_fee : ℕ := 15
  let driveway_shoveling_fee : ℕ := 7
  let change_after_purchase : ℕ := 15
  let lawns_mowed : ℕ := 4
  let total_money : ℕ := shoe_cost + change_after_purchase
  let money_from_allowance : ℕ := months_saved * monthly_allowance
  let money_from_mowing : ℕ := lawns_mowed * lawn_mowing_fee
  let money_from_shoveling : ℕ := total_money - money_from_allowance - money_from_mowing
  money_from_shoveling / driveway_shoveling_fee

theorem tobias_driveways : num_driveways = 2 := by
  sorry

end NUMINAMATH_CALUDE_tobias_driveways_l3198_319868


namespace NUMINAMATH_CALUDE_difference_of_squares_l3198_319840

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3198_319840


namespace NUMINAMATH_CALUDE_brian_watching_time_l3198_319890

def cat_video_length : ℕ := 4

def dog_video_length (cat_length : ℕ) : ℕ := 2 * cat_length

def gorilla_video_length (cat_length dog_length : ℕ) : ℕ := 2 * (cat_length + dog_length)

def total_watching_time (cat_length dog_length gorilla_length : ℕ) : ℕ :=
  cat_length + dog_length + gorilla_length

theorem brian_watching_time :
  total_watching_time cat_video_length 
    (dog_video_length cat_video_length) 
    (gorilla_video_length cat_video_length (dog_video_length cat_video_length)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_brian_watching_time_l3198_319890


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3198_319814

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 1) * (x - 2) > 0} = {x : ℝ | x < -1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3198_319814


namespace NUMINAMATH_CALUDE_devin_age_l3198_319875

theorem devin_age (devin_age eden_age mom_age : ℕ) : 
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 →
  devin_age = 12 := by
sorry

end NUMINAMATH_CALUDE_devin_age_l3198_319875


namespace NUMINAMATH_CALUDE_circle_tangent_area_zero_l3198_319886

-- Define the circle struct
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line struct
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

def CircleInternallyTangent (c1 c2 : Circle) : Prop := sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem circle_tangent_area_zero 
  (P Q R : Circle)
  (l : Line)
  (P' Q' R' : ℝ × ℝ)
  (h1 : P.radius = 2)
  (h2 : Q.radius = 3)
  (h3 : R.radius = 4)
  (h4 : CircleTangentToLine P l)
  (h5 : CircleTangentToLine Q l)
  (h6 : CircleTangentToLine R l)
  (h7 : P'.1 = P.center.1 ∧ P'.2 = P.center.2 + P.radius)
  (h8 : Q'.1 = Q.center.1 ∧ Q'.2 = Q.center.2 + Q.radius)
  (h9 : R'.1 = R.center.1 ∧ R'.2 = R.center.2 + R.radius)
  (h10 : PointBetween P' Q' R')
  (h11 : CircleInternallyTangent Q P)
  (h12 : CircleInternallyTangent Q R) :
  TriangleArea P.center Q.center R.center = 0 := by sorry

end NUMINAMATH_CALUDE_circle_tangent_area_zero_l3198_319886


namespace NUMINAMATH_CALUDE_right_pyramid_height_l3198_319874

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  base_perimeter : ℝ
  /-- The distance from the apex to each vertex of the base in inches -/
  apex_to_vertex : ℝ

/-- The height of a right pyramid from its apex to the center of its base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

theorem right_pyramid_height (p : RightPyramid) 
  (h1 : p.base_perimeter = 40)
  (h2 : p.apex_to_vertex = 12) :
  pyramid_height p = Real.sqrt 94 := by
  sorry

end NUMINAMATH_CALUDE_right_pyramid_height_l3198_319874


namespace NUMINAMATH_CALUDE_compute_expression_l3198_319822

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3198_319822


namespace NUMINAMATH_CALUDE_gray_area_division_l3198_319808

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

-- Define a square within the rectangle
structure InternalSquare where
  side : ℝ
  x : ℝ  -- x-coordinate of the square's top-left corner
  y : ℝ  -- y-coordinate of the square's top-left corner
  side_pos : side > 0
  within_rectangle : (r : Rectangle) → x ≥ 0 ∧ y ≥ 0 ∧ x + side ≤ r.width ∧ y + side ≤ r.height

-- Define the theorem
theorem gray_area_division (r : Rectangle) (s : InternalSquare) :
  ∃ (line : ℝ → ℝ → Prop), 
    (∀ (x y : ℝ), (x ≥ 0 ∧ x ≤ r.width ∧ y ≥ 0 ∧ y ≤ r.height) →
      (¬(x ≥ s.x ∧ x ≤ s.x + s.side ∧ y ≥ s.y ∧ y ≤ s.y + s.side) →
        (line x y ∨ ¬line x y))) ∧
    (∃ (area1 area2 : ℝ), area1 = area2 ∧
      area1 + area2 = r.width * r.height - s.side * s.side) :=
by sorry

end NUMINAMATH_CALUDE_gray_area_division_l3198_319808


namespace NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l3198_319898

/-- The first two-digit multiple of 8 -/
def first_multiple : Nat := 16

/-- The last two-digit multiple of 8 -/
def last_multiple : Nat := 96

/-- The common difference between consecutive multiples of 8 -/
def common_difference : Nat := 8

/-- The number of two-digit multiples of 8 -/
def num_multiples : Nat := (last_multiple - first_multiple) / common_difference + 1

/-- The arithmetic mean of all positive two-digit multiples of 8 is 56 -/
theorem arithmetic_mean_two_digit_multiples_of_8 :
  (first_multiple + last_multiple) * num_multiples / (2 * num_multiples) = 56 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l3198_319898


namespace NUMINAMATH_CALUDE_min_c_value_l3198_319827

theorem min_c_value (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b → b ≤ c →
  b = a + 13 →
  ∀ c' : ℕ, c' > 0 ∧ 
    (∃ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧
      (a' + b' + c') / 3 = 20 ∧
      a' ≤ b' ∧ b' ≤ c' ∧
      b' = a' + 13) →
    c ≤ c' →
  c = 45 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l3198_319827


namespace NUMINAMATH_CALUDE_finite_crosses_in_circle_l3198_319806

/-- A cross formed by the diagonals of a square with side length 1 -/
def Cross : Type := Unit

/-- A circle with radius 100 -/
def Circle : Type := Unit

/-- The maximum number of non-overlapping crosses that can fit inside the circle -/
noncomputable def maxCrosses : ℕ := sorry

/-- The theorem stating that the number of non-overlapping crosses that can fit inside the circle is finite -/
theorem finite_crosses_in_circle : ∃ n : ℕ, maxCrosses ≤ n := by sorry

end NUMINAMATH_CALUDE_finite_crosses_in_circle_l3198_319806


namespace NUMINAMATH_CALUDE_simplify_expression_l3198_319885

theorem simplify_expression (a : ℝ) (h : a < (1/4)) : 4*(4*a - 1)^2 = (1 - 4*a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3198_319885


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3198_319859

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + 3 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3198_319859


namespace NUMINAMATH_CALUDE_mn_solutions_l3198_319825

theorem mn_solutions (m n : ℤ) : 
  m * n ≥ 0 → m^3 + n^3 + 99*m*n = 33^3 → 
  ((m + n = 33 ∧ m ≥ 0 ∧ n ≥ 0) ∨ (m = -33 ∧ n = -33)) := by
  sorry

end NUMINAMATH_CALUDE_mn_solutions_l3198_319825


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3198_319863

def M : Set ℝ := {x | x^2 ≥ 4}
def N : Set ℝ := {-3, 0, 1, 3, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {-3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3198_319863


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3198_319817

theorem unique_solution_exponential_equation :
  ∀ x : ℝ, (4 : ℝ)^((9 : ℝ)^x) = (9 : ℝ)^((4 : ℝ)^x) ↔ x = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3198_319817


namespace NUMINAMATH_CALUDE_k_range_proof_l3198_319869

theorem k_range_proof (k : ℝ) : 
  (∀ x, x > k → 3 / (x + 1) < 1) ∧ 
  (∃ x, 3 / (x + 1) < 1 ∧ x ≤ k) ↔ 
  k ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_k_range_proof_l3198_319869


namespace NUMINAMATH_CALUDE_max_im_part_is_sin_90_deg_l3198_319846

-- Define the polynomial
def p (z : ℂ) : ℂ := z^6 - z^4 + z^3 - z + 1

-- Define the set of roots
def roots : Set ℂ := {z : ℂ | p z = 0}

-- Define the imaginary part function
def imPart (z : ℂ) : ℝ := z.im

-- Define the theorem
theorem max_im_part_is_sin_90_deg :
  ∃ (z : ℂ), z ∈ roots ∧ 
  (∀ (w : ℂ), w ∈ roots → imPart w ≤ imPart z) ∧
  imPart z = Real.sin (π / 2) :=
sorry

end NUMINAMATH_CALUDE_max_im_part_is_sin_90_deg_l3198_319846


namespace NUMINAMATH_CALUDE_sunflower_height_comparison_l3198_319820

/-- Given that sunflowers from Packet A are 20% taller than those from Packet B,
    and sunflowers from Packet A are 192 inches tall,
    prove that sunflowers from Packet B are 160 inches tall. -/
theorem sunflower_height_comparison (height_A height_B : ℝ) : 
  height_A = height_B * 1.2 → height_A = 192 → height_B = 160 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_comparison_l3198_319820


namespace NUMINAMATH_CALUDE_max_cards_48_36_16_12_l3198_319849

/-- The maximum number of rectangular cards that can be cut from a rectangular cardboard --/
def max_cards (cardboard_length cardboard_width card_length card_width : ℕ) : ℕ :=
  max ((cardboard_length / card_length) * (cardboard_width / card_width))
      ((cardboard_length / card_width) * (cardboard_width / card_length))

/-- Theorem: The maximum number of 16 cm x 12 cm cards that can be cut from a 48 cm x 36 cm cardboard is 9 --/
theorem max_cards_48_36_16_12 :
  max_cards 48 36 16 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_cards_48_36_16_12_l3198_319849


namespace NUMINAMATH_CALUDE_tan_alpha_equals_sqrt_two_l3198_319856

theorem tan_alpha_equals_sqrt_two (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.sin α = Real.sqrt 6 / 3) : 
  Real.tan α = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_sqrt_two_l3198_319856


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3198_319832

-- Problem 1
theorem problem_1 : -2.4 + 3.5 - 4.6 + 3.5 = 0 := by sorry

-- Problem 2
theorem problem_2 : (-40) - (-28) - (-19) + (-24) = -17 := by sorry

-- Problem 3
theorem problem_3 : (-3 : ℚ) * (5/6 : ℚ) * (-4/5 : ℚ) * (-1/4 : ℚ) = -1/2 := by sorry

-- Problem 4
theorem problem_4 : (-5/7 : ℚ) * (-4/3 : ℚ) / (-15/7 : ℚ) = -4/9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3198_319832


namespace NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l3198_319815

theorem lcm_of_8_24_36_54 : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 54)) = 216 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l3198_319815


namespace NUMINAMATH_CALUDE_garden_roller_length_l3198_319851

theorem garden_roller_length :
  let diameter : ℝ := 1.4
  let area_covered : ℝ := 66
  let revolutions : ℝ := 5
  let π : ℝ := 22 / 7
  let radius : ℝ := diameter / 2
  let length : ℝ := (area_covered / revolutions) / (2 * π * radius)
  length = 2.1 := by sorry

end NUMINAMATH_CALUDE_garden_roller_length_l3198_319851


namespace NUMINAMATH_CALUDE_total_length_is_6000_feet_l3198_319884

/-- Represents a path on a scale drawing -/
structure ScalePath where
  length : ℝ  -- length of the path on the drawing in inches
  scale : ℝ   -- scale factor (feet represented by 1 inch)

/-- Calculates the actual length of a path in feet -/
def actualLength (path : ScalePath) : ℝ := path.length * path.scale

/-- Theorem: The total length represented by two paths on a scale drawing is 6000 feet -/
theorem total_length_is_6000_feet (path1 path2 : ScalePath)
  (h1 : path1.length = 6 ∧ path1.scale = 500)
  (h2 : path2.length = 3 ∧ path2.scale = 1000) :
  actualLength path1 + actualLength path2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_total_length_is_6000_feet_l3198_319884


namespace NUMINAMATH_CALUDE_UA_intersect_B_equals_two_three_l3198_319880

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3, 4}

def A : Set Int := {x ∈ U | x * (x^2 - 1) = 0}

def B : Set Int := {x ∈ U | x ≥ 0 ∧ x^2 ≤ 9}

theorem UA_intersect_B_equals_two_three : (U \ A) ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_UA_intersect_B_equals_two_three_l3198_319880


namespace NUMINAMATH_CALUDE_weight_of_b_l3198_319831

/-- Given three weights a, b, and c, prove that b = 70 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 60 →
  (a + b) / 2 = 70 →
  (b + c) / 2 = 50 →
  b = 70 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l3198_319831


namespace NUMINAMATH_CALUDE_tracy_dogs_food_consumption_l3198_319803

/-- Proves that Tracy's two dogs consume 4 pounds of food per day -/
theorem tracy_dogs_food_consumption :
  let num_dogs : ℕ := 2
  let cups_per_meal_per_dog : ℚ := 3/2
  let meals_per_day : ℕ := 3
  let cups_per_pound : ℚ := 9/4
  
  let total_cups_per_day : ℚ := num_dogs * cups_per_meal_per_dog * meals_per_day
  let total_pounds_per_day : ℚ := total_cups_per_day / cups_per_pound
  
  total_pounds_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_tracy_dogs_food_consumption_l3198_319803


namespace NUMINAMATH_CALUDE_frame_uncovered_area_l3198_319830

/-- The area of a rectangular frame not covered by a photo -/
theorem frame_uncovered_area (frame_length frame_width photo_length photo_width : ℝ)
  (h1 : frame_length = 40)
  (h2 : frame_width = 32)
  (h3 : photo_length = 32)
  (h4 : photo_width = 28) :
  frame_length * frame_width - photo_length * photo_width = 384 := by
  sorry

end NUMINAMATH_CALUDE_frame_uncovered_area_l3198_319830


namespace NUMINAMATH_CALUDE_sqrt_of_three_times_two_five_cubed_l3198_319850

theorem sqrt_of_three_times_two_five_cubed (x : ℝ) : 
  x = Real.sqrt (2 * (5^3) + 2 * (5^3) + 2 * (5^3)) → x = 5 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_three_times_two_five_cubed_l3198_319850


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3198_319801

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : Nat
  sample_size : Nat
  interval : Nat
  first_element : Nat

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.first_element + k * s.interval ∧ k < s.sample_size

theorem systematic_sample_theorem (sample : SystematicSample)
  (h_pop_size : sample.population_size = 56)
  (h_sample_size : sample.sample_size = 4)
  (h_contains_6 : sample.contains 6)
  (h_contains_34 : sample.contains 34)
  (h_contains_48 : sample.contains 48) :
  sample.contains 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3198_319801


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3198_319807

theorem quadratic_root_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 9*x + d = 0 ↔ x = (-9 + Real.sqrt d) / 2 ∨ x = (-9 - Real.sqrt d) / 2) →
  d = 16.2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3198_319807


namespace NUMINAMATH_CALUDE_temporary_employee_percentage_is_51_5_l3198_319833

/-- Represents the composition of workers in a factory -/
structure WorkerComposition where
  technicians : Real
  skilled_laborers : Real
  unskilled_laborers : Real
  permanent_technicians : Real
  permanent_skilled : Real
  permanent_unskilled : Real

/-- Calculates the percentage of temporary employees in the factory -/
def temporary_employee_percentage (wc : WorkerComposition) : Real :=
  100 - (wc.technicians * wc.permanent_technicians + 
         wc.skilled_laborers * wc.permanent_skilled + 
         wc.unskilled_laborers * wc.permanent_unskilled)

/-- Theorem stating that given the conditions, the percentage of temporary employees is 51.5% -/
theorem temporary_employee_percentage_is_51_5 (wc : WorkerComposition) 
  (h1 : wc.technicians = 40)
  (h2 : wc.skilled_laborers = 35)
  (h3 : wc.unskilled_laborers = 25)
  (h4 : wc.permanent_technicians = 60)
  (h5 : wc.permanent_skilled = 45)
  (h6 : wc.permanent_unskilled = 35) :
  temporary_employee_percentage wc = 51.5 := by
  sorry

end NUMINAMATH_CALUDE_temporary_employee_percentage_is_51_5_l3198_319833


namespace NUMINAMATH_CALUDE_ball_probability_l3198_319836

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_yellow : yellow = 10)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.8 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l3198_319836


namespace NUMINAMATH_CALUDE_rebus_unique_solution_l3198_319848

/-- Represents a four-digit number ABCD where A, B, C, D are distinct non-zero digits. -/
structure Rebus where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0
  h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- The rebus equation ABCA = 182 * CD holds. -/
def rebusEquation (r : Rebus) : Prop :=
  1000 * r.a + 100 * r.b + 10 * r.c + r.a = 182 * (10 * r.c + r.d)

/-- The unique solution to the rebus is 2916. -/
theorem rebus_unique_solution :
  ∃! r : Rebus, rebusEquation r ∧ r.a = 2 ∧ r.b = 9 ∧ r.c = 1 ∧ r.d = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_unique_solution_l3198_319848


namespace NUMINAMATH_CALUDE_consecutive_lucky_years_exist_l3198_319837

def is_lucky_year (n : ℕ) : Prop :=
  let a := n / 100
  let b := n % 100
  n % (a + b) = 0

theorem consecutive_lucky_years_exist : ∃ n : ℕ, is_lucky_year n ∧ is_lucky_year (n + 1) :=
sorry

end NUMINAMATH_CALUDE_consecutive_lucky_years_exist_l3198_319837


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l3198_319839

theorem like_terms_exponent_sum (x y : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), -x^6 * y^(2*m) = k * x^(n+2) * y^4) →
  n + m = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l3198_319839


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3198_319883

def repeating_decimal : ℚ := 2 + 35 / 99

theorem repeating_decimal_as_fraction :
  repeating_decimal = 233 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3198_319883


namespace NUMINAMATH_CALUDE_prime_divisibility_l3198_319834

theorem prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → q ∣ (3^p - 2^p) → p ∣ (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l3198_319834


namespace NUMINAMATH_CALUDE_P_range_l3198_319897

theorem P_range (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let P := (a^2 / (a^2 + b^2 + c^2)) + (b^2 / (b^2 + c^2 + d^2)) +
           (c^2 / (c^2 + d^2 + a^2)) + (d^2 / (d^2 + a^2 + b^2))
  1 < P ∧ P < 2 := by
  sorry

end NUMINAMATH_CALUDE_P_range_l3198_319897


namespace NUMINAMATH_CALUDE_max_schools_donation_l3198_319855

/-- Represents the donation constraints for The Khan Corporation --/
structure DonationConstraints where
  total_computers : Nat
  total_printers : Nat
  total_tablets : Nat
  total_projectors : Nat
  min_computers : Nat
  max_computers : Nat
  min_printers : Nat
  max_printers : Nat
  min_tablets : Nat
  max_tablets : Nat
  min_projectors : Nat
  max_projectors : Nat

/-- Checks if a given number of schools satisfies all donation constraints --/
def satisfiesConstraints (n : Nat) (c : DonationConstraints) : Prop :=
  n > 0 ∧
  c.total_computers % n = 0 ∧
  c.total_printers % n = 0 ∧
  c.total_tablets % n = 0 ∧
  c.total_projectors % n = 0 ∧
  c.min_computers ≤ c.total_computers / n ∧ c.total_computers / n ≤ c.max_computers ∧
  c.min_printers ≤ c.total_printers / n ∧ c.total_printers / n ≤ c.max_printers ∧
  c.min_tablets ≤ c.total_tablets / n ∧ c.total_tablets / n ≤ c.max_tablets ∧
  c.min_projectors ≤ c.total_projectors / n ∧ c.total_projectors / n ≤ c.max_projectors

/-- The main theorem stating that 4 is the maximum number of schools that can receive donations --/
theorem max_schools_donation (c : DonationConstraints) 
  (h_computers : c.total_computers = 48)
  (h_printers : c.total_printers = 32)
  (h_tablets : c.total_tablets = 60)
  (h_projectors : c.total_projectors = 20)
  (h_min_computers : c.min_computers = 4)
  (h_max_computers : c.max_computers = 8)
  (h_min_printers : c.min_printers = 2)
  (h_max_printers : c.max_printers = 4)
  (h_min_tablets : c.min_tablets = 3)
  (h_max_tablets : c.max_tablets = 7)
  (h_min_projectors : c.min_projectors = 1)
  (h_max_projectors : c.max_projectors = 3) :
  satisfiesConstraints 4 c ∧ ∀ n : Nat, n > 4 → ¬satisfiesConstraints n c :=
by sorry

end NUMINAMATH_CALUDE_max_schools_donation_l3198_319855


namespace NUMINAMATH_CALUDE_square_root_equation_l3198_319853

theorem square_root_equation (N : ℝ) : 
  Real.sqrt (0.05 * N) * Real.sqrt 5 = 0.25 → N = 0.25 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_l3198_319853


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l3198_319881

theorem contrapositive_real_roots (m : ℝ) : 
  (¬(∃ x : ℝ, x^2 + 2*x - 3*m = 0) → m ≤ 0) ↔ (m > 0 → ∃ x : ℝ, x^2 + 2*x - 3*m = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l3198_319881


namespace NUMINAMATH_CALUDE_pentagon_area_l3198_319878

/-- Given a grid with distance m between adjacent points, prove that for a quadrilateral ABCD with area 23, the area of pentagon EFGHI is 28. -/
theorem pentagon_area (m : ℝ) (area_ABCD : ℝ) : 
  m > 0 → area_ABCD = 23 → ∃ (area_EFGHI : ℝ), area_EFGHI = 28 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l3198_319878


namespace NUMINAMATH_CALUDE_marble_doubling_l3198_319894

theorem marble_doubling (k : ℕ) : (∀ n : ℕ, n < k → 5 * 2^n ≤ 200) ∧ 5 * 2^k > 200 ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_marble_doubling_l3198_319894


namespace NUMINAMATH_CALUDE_candy_sharing_l3198_319892

theorem candy_sharing (hugh tommy melany : ℕ) (h1 : hugh = 8) (h2 : tommy = 6) (h3 : melany = 7) :
  (hugh + tommy + melany) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_l3198_319892


namespace NUMINAMATH_CALUDE_total_snails_is_294_l3198_319826

/-- The total number of snails found by a family of ducks -/
def total_snails : ℕ :=
  let total_ducklings : ℕ := 8
  let first_group_size : ℕ := 3
  let second_group_size : ℕ := 3
  let first_group_snails_per_duckling : ℕ := 5
  let second_group_snails_per_duckling : ℕ := 9
  let first_group_total : ℕ := first_group_size * first_group_snails_per_duckling
  let second_group_total : ℕ := second_group_size * second_group_snails_per_duckling
  let first_two_groups_total : ℕ := first_group_total + second_group_total
  let mother_duck_snails : ℕ := 3 * first_two_groups_total
  let remaining_ducklings : ℕ := total_ducklings - first_group_size - second_group_size
  let remaining_group_total : ℕ := remaining_ducklings * (mother_duck_snails / 2)
  first_group_total + second_group_total + mother_duck_snails + remaining_group_total

theorem total_snails_is_294 : total_snails = 294 := by
  sorry

end NUMINAMATH_CALUDE_total_snails_is_294_l3198_319826


namespace NUMINAMATH_CALUDE_min_n_for_equation_property_l3198_319876

theorem min_n_for_equation_property : ∃ (n : ℕ), n = 835 ∧ 
  (∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 999) →
    ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T.card = m ∧ (∀ x ∈ T, x ≥ 1 ∧ x ≤ 999) ∧
    ¬(∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d)) :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_equation_property_l3198_319876


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3198_319823

theorem square_floor_tiles (n : ℕ) : 
  (2 * n - 1 = 37) → n^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3198_319823


namespace NUMINAMATH_CALUDE_third_number_proof_l3198_319866

/-- The smallest number greater than 57 that leaves the same remainder as 25 and 57 when divided by 16 -/
def third_number : ℕ := 73

/-- The common divisor -/
def common_divisor : ℕ := 16

theorem third_number_proof :
  (third_number % common_divisor = 25 % common_divisor) ∧
  (third_number % common_divisor = 57 % common_divisor) ∧
  (third_number > 57) ∧
  (∀ n : ℕ, n > 57 ∧ n < third_number →
    (n % common_divisor ≠ 25 % common_divisor ∨
     n % common_divisor ≠ 57 % common_divisor)) :=
by sorry

end NUMINAMATH_CALUDE_third_number_proof_l3198_319866


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2002_l3198_319870

theorem sum_of_last_two_digits_of_9_pow_2002 : ∃ (n : ℕ), 9^2002 = 100 * n + 81 :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2002_l3198_319870


namespace NUMINAMATH_CALUDE_coinciding_rest_days_main_theorem_l3198_319821

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 7

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 500

/-- Number of rest days for Chris in one cycle -/
def chris_rest_days : ℕ := 2

/-- Number of rest days for Dana in one cycle -/
def dana_rest_days : ℕ := 1

/-- The number of days both Chris and Dana have rest-days on the same day
    within the first 500 days of their schedules -/
theorem coinciding_rest_days : ℕ := by
  sorry

/-- The main theorem stating that the number of coinciding rest days is 28 -/
theorem main_theorem : coinciding_rest_days = 28 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_main_theorem_l3198_319821


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_l3198_319800

def total_runs : ℕ := 120
def boundaries : ℕ := 5
def sixes : ℕ := 5
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem percentage_runs_by_running (total_runs boundaries sixes runs_per_boundary runs_per_six : ℕ) 
  (h1 : total_runs = 120)
  (h2 : boundaries = 5)
  (h3 : sixes = 5)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6) :
  (total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100 = 
  (120 - (5 * 4 + 5 * 6)) / 120 * 100 := by sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_l3198_319800
