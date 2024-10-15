import Mathlib

namespace NUMINAMATH_CALUDE_coloring_perfect_square_difference_l2018_201896

/-- A coloring of integers using three colors -/
def Coloring := ℤ → Fin 3

/-- Theorem: For any coloring of integers using three colors, 
    there exist two distinct integers with the same color 
    whose difference is a perfect square -/
theorem coloring_perfect_square_difference (c : Coloring) : 
  ∃ (x y k : ℤ), x ≠ y ∧ c x = c y ∧ y - x = k^2 := by
  sorry

end NUMINAMATH_CALUDE_coloring_perfect_square_difference_l2018_201896


namespace NUMINAMATH_CALUDE_product_sum_relation_l2018_201819

theorem product_sum_relation (a b x : ℤ) : 
  b = 9 → b - a = 5 → a * b = 2 * (a + b) + x → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2018_201819


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l2018_201829

theorem smallest_x_abs_equation : 
  (∀ x : ℝ, |2*x + 5| = 21 → x ≥ -13) ∧ 
  (|2*(-13) + 5| = 21) := by
sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l2018_201829


namespace NUMINAMATH_CALUDE_initial_apples_count_l2018_201814

/-- The number of apples Rachel picked from the tree -/
def apples_picked : ℕ := 4

/-- The number of apples remaining on the tree -/
def apples_remaining : ℕ := 3

/-- The initial number of apples on the tree -/
def initial_apples : ℕ := apples_picked + apples_remaining

theorem initial_apples_count : initial_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2018_201814


namespace NUMINAMATH_CALUDE_total_seeds_planted_l2018_201821

theorem total_seeds_planted (num_flowerbeds : ℕ) (seeds_per_flowerbed : ℕ) 
  (h1 : num_flowerbeds = 8) 
  (h2 : seeds_per_flowerbed = 4) : 
  num_flowerbeds * seeds_per_flowerbed = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_planted_l2018_201821


namespace NUMINAMATH_CALUDE_server_data_requests_l2018_201843

/-- The number of data requests processed by a server in 24 hours -/
def data_requests_per_day (requests_per_minute : ℕ) : ℕ :=
  requests_per_minute * (24 * 60)

/-- Theorem stating that a server processing 15,000 data requests per minute
    will process 21,600,000 data requests in 24 hours -/
theorem server_data_requests :
  data_requests_per_day 15000 = 21600000 := by
  sorry

end NUMINAMATH_CALUDE_server_data_requests_l2018_201843


namespace NUMINAMATH_CALUDE_visible_sides_is_seventeen_l2018_201840

/-- Represents a polygon with a given number of sides. -/
structure Polygon where
  sides : Nat
  sides_positive : sides > 0

/-- The configuration of polygons in the problem. -/
def polygon_configuration : List Polygon :=
  [⟨4, by norm_num⟩, ⟨3, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩]

/-- Calculates the number of visible sides in the configuration. -/
def visible_sides (config : List Polygon) : Nat :=
  (config.map (·.sides)).sum - 2 * (config.length - 1)

/-- Theorem stating that the number of visible sides in the given configuration is 17. -/
theorem visible_sides_is_seventeen :
  visible_sides polygon_configuration = 17 := by
  sorry

#eval visible_sides polygon_configuration

end NUMINAMATH_CALUDE_visible_sides_is_seventeen_l2018_201840


namespace NUMINAMATH_CALUDE_initial_men_count_l2018_201808

/-- Given a group of men with provisions lasting 18 days, prove that the initial number of men is 1000 
    when 400 more men join and the provisions then last 12.86 days, assuming the total amount of provisions remains constant. -/
theorem initial_men_count (initial_days : ℝ) (final_days : ℝ) (additional_men : ℕ) :
  initial_days = 18 →
  final_days = 12.86 →
  additional_men = 400 →
  ∃ (initial_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    initial_men = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2018_201808


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l2018_201880

open Real

theorem inclination_angle_of_line (x y : ℝ) :
  let line_equation := x * tan (π / 3) + y + 2 = 0
  let inclination_angle := 2 * π / 3
  line_equation → ∃ α, α = inclination_angle ∧ tan α = -tan (π / 3) ∧ 0 ≤ α ∧ α < π :=
by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l2018_201880


namespace NUMINAMATH_CALUDE_expansion_theorem_l2018_201891

theorem expansion_theorem (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_theorem_l2018_201891


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2018_201827

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2018_201827


namespace NUMINAMATH_CALUDE_balls_after_2017_steps_l2018_201851

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- The sum of digits in the base-5 representation of 2017 equals 9 -/
theorem balls_after_2017_steps : sumDigits (toBase5 2017) = 9 := by
  sorry


end NUMINAMATH_CALUDE_balls_after_2017_steps_l2018_201851


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2018_201855

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2018_201855


namespace NUMINAMATH_CALUDE_red_mushrooms_with_spots_l2018_201800

/-- Represents the number of mushrooms gathered by Bill and Ted -/
structure MushroomGathering where
  red : ℕ
  brown : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the fraction of red mushrooms with white spots -/
def fraction_red_with_spots (g : MushroomGathering) (total_spotted : ℕ) : ℚ :=
  (total_spotted - g.brown - g.blue / 2) / g.red

/-- The main theorem stating the fraction of red mushrooms with white spots -/
theorem red_mushrooms_with_spots :
  let g := MushroomGathering.mk 12 6 14 6
  let total_spotted := 17
  fraction_red_with_spots g total_spotted = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_red_mushrooms_with_spots_l2018_201800


namespace NUMINAMATH_CALUDE_steven_erasers_count_l2018_201841

/-- The number of skittles Steven has -/
def skittles : ℕ := 4502

/-- The number of groups the items are organized into -/
def groups : ℕ := 154

/-- The number of items in each group -/
def items_per_group : ℕ := 57

/-- The total number of items (skittles and erasers) -/
def total_items : ℕ := groups * items_per_group

/-- The number of erasers Steven has -/
def erasers : ℕ := total_items - skittles

theorem steven_erasers_count : erasers = 4276 := by
  sorry

end NUMINAMATH_CALUDE_steven_erasers_count_l2018_201841


namespace NUMINAMATH_CALUDE_factor_sum_l2018_201865

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 50 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2018_201865


namespace NUMINAMATH_CALUDE_sqrt_15_has_two_roots_l2018_201806

-- Define √15 as a real number
noncomputable def sqrt15 : ℝ := Real.sqrt 15

-- State the theorem
theorem sqrt_15_has_two_roots :
  ∃ (x : ℝ), x ≠ sqrt15 ∧ x * x = 15 :=
by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end NUMINAMATH_CALUDE_sqrt_15_has_two_roots_l2018_201806


namespace NUMINAMATH_CALUDE_notebook_cost_l2018_201810

theorem notebook_cost (total_cost cover_cost notebook_cost : ℚ) : 
  total_cost = 3.5 →
  notebook_cost = cover_cost + 2 →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.75 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l2018_201810


namespace NUMINAMATH_CALUDE_max_product_permutation_l2018_201854

theorem max_product_permutation (a : Fin 1987 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : Set.range a = Finset.range 1988) : 
  (Finset.range 1988).sup (λ k => k * a k) ≥ 994^2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_permutation_l2018_201854


namespace NUMINAMATH_CALUDE_sum_of_coefficients_fifth_power_one_plus_sqrt_two_l2018_201801

theorem sum_of_coefficients_fifth_power_one_plus_sqrt_two (a b : ℚ) : 
  (1 + Real.sqrt 2) ^ 5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_fifth_power_one_plus_sqrt_two_l2018_201801


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2018_201897

theorem simplify_fraction_product : (90 : ℚ) / 150 * 35 / 21 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2018_201897


namespace NUMINAMATH_CALUDE_coefficient_sum_equals_negative_eight_l2018_201803

/-- Given a polynomial equation, prove that a specific linear combination of its coefficients equals -8 -/
theorem coefficient_sum_equals_negative_eight 
  (a : Fin 9 → ℝ) 
  (h : ∀ x : ℝ, x^5 * (x+3)^3 = (a 8)*(x+1)^8 + (a 7)*(x+1)^7 + (a 6)*(x+1)^6 + 
                               (a 5)*(x+1)^5 + (a 4)*(x+1)^4 + (a 3)*(x+1)^3 + 
                               (a 2)*(x+1)^2 + (a 1)*(x+1) + (a 0)) : 
  7*(a 7) + 5*(a 5) + 3*(a 3) + (a 1) = -8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_equals_negative_eight_l2018_201803


namespace NUMINAMATH_CALUDE_largest_and_smallest_A_l2018_201895

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_digit_to_front (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 10^7 + r

def satisfies_conditions (b : ℕ) : Prop :=
  b > 44444444 ∧ is_coprime b 12

theorem largest_and_smallest_A :
  ∃ (a_max a_min : ℕ),
    (∀ a b : ℕ, 
      a = move_last_digit_to_front b ∧ 
      satisfies_conditions b →
      a ≤ a_max ∧ a ≥ a_min) ∧
    a_max = 99999998 ∧
    a_min = 14444446 :=
sorry

end NUMINAMATH_CALUDE_largest_and_smallest_A_l2018_201895


namespace NUMINAMATH_CALUDE_only_one_student_passes_l2018_201874

theorem only_one_student_passes (prob_A prob_B prob_C : ℚ)
  (hA : prob_A = 4/5)
  (hB : prob_B = 3/5)
  (hC : prob_C = 7/10) :
  (prob_A * (1 - prob_B) * (1 - prob_C)) +
  ((1 - prob_A) * prob_B * (1 - prob_C)) +
  ((1 - prob_A) * (1 - prob_B) * prob_C) = 47/250 := by
  sorry

end NUMINAMATH_CALUDE_only_one_student_passes_l2018_201874


namespace NUMINAMATH_CALUDE_circle_area_circumference_difference_l2018_201863

theorem circle_area_circumference_difference (a b c : ℝ) (h1 : a = 24) (h2 : b = 70) (h3 : c = 74) 
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) : 
  let r := c / 2
  (π * r ^ 2) - (2 * π * r) = 1295 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_circumference_difference_l2018_201863


namespace NUMINAMATH_CALUDE_max_b_value_l2018_201852

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) 
  (h_order : 1 < c ∧ c < b ∧ b < a) (h_prime : Nat.Prime c) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ Nat.Prime c' ∧ b' = 12 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2018_201852


namespace NUMINAMATH_CALUDE_multiples_properties_l2018_201884

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ k : ℤ, b = 8 * k) : 
  (∃ k : ℤ, b = 4 * k) ∧ 
  (∃ k : ℤ, a - b = 4 * k) ∧ 
  (∃ k : ℤ, a + b = 2 * k) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l2018_201884


namespace NUMINAMATH_CALUDE_tower_blocks_sum_l2018_201883

/-- The total number of blocks in a tower after adding more blocks -/
def total_blocks (initial : Real) (added : Real) : Real :=
  initial + added

/-- Theorem: The total number of blocks is the sum of initial and added blocks -/
theorem tower_blocks_sum (initial : Real) (added : Real) :
  total_blocks initial added = initial + added := by
  sorry

end NUMINAMATH_CALUDE_tower_blocks_sum_l2018_201883


namespace NUMINAMATH_CALUDE_equality_condition_for_sum_squares_equation_l2018_201857

theorem equality_condition_for_sum_squares_equation (a b c : ℝ) :
  (a^2 + b^2 + c^2 = a*b + b*c + a*c) ↔ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equality_condition_for_sum_squares_equation_l2018_201857


namespace NUMINAMATH_CALUDE_sales_ratio_l2018_201879

/-- Proves that the ratio of sales on a tough week to sales on a good week is 1:2 -/
theorem sales_ratio (tough_week_sales : ℝ) (total_sales : ℝ) : 
  tough_week_sales = 800 →
  total_sales = 10400 →
  ∃ (good_week_sales : ℝ),
    5 * good_week_sales + 3 * tough_week_sales = total_sales ∧
    tough_week_sales / good_week_sales = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sales_ratio_l2018_201879


namespace NUMINAMATH_CALUDE_three_digit_equation_l2018_201876

/-- 
Given a three-digit number A7B where 7 is the tens digit, 
prove that A = 6 if A7B + 23 = 695
-/
theorem three_digit_equation (A B : ℕ) : 
  (A * 100 + 70 + B) + 23 = 695 → 
  0 ≤ A ∧ A ≤ 9 → 
  0 ≤ B ∧ B ≤ 9 → 
  A = 6 := by
sorry

end NUMINAMATH_CALUDE_three_digit_equation_l2018_201876


namespace NUMINAMATH_CALUDE_happy_dictionary_problem_l2018_201899

theorem happy_dictionary_problem (a b : ℤ) (c : ℚ) : 
  (∀ n : ℤ, n > 0 → a ≤ n) → 
  (∀ n : ℤ, n < 0 → n ≤ b) → 
  (∀ q : ℚ, q ≠ 0 → |c| ≤ |q|) → 
  a - b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_happy_dictionary_problem_l2018_201899


namespace NUMINAMATH_CALUDE_pink_crayons_l2018_201832

def crayon_box (total red blue green yellow pink purple : ℕ) : Prop :=
  total = 48 ∧
  red = 12 ∧
  blue = 8 ∧
  green = (3 * blue) / 4 ∧
  yellow = (15 * total) / 100 ∧
  pink = purple ∧
  total = red + blue + green + yellow + pink + purple

theorem pink_crayons (total red blue green yellow pink purple : ℕ) :
  crayon_box total red blue green yellow pink purple → pink = 8 := by
  sorry

end NUMINAMATH_CALUDE_pink_crayons_l2018_201832


namespace NUMINAMATH_CALUDE_angle_inequalities_l2018_201831

theorem angle_inequalities (α β : Real) (h1 : π / 2 < α) (h2 : α < β) (h3 : β < π) :
  (π < α + β ∧ α + β < 2 * π) ∧
  (-π / 2 < α - β ∧ α - β < 0) ∧
  (1 / 2 < α / β ∧ α / β < 1) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequalities_l2018_201831


namespace NUMINAMATH_CALUDE_basket_weight_l2018_201850

/-- Proves that the weight of an empty basket is 1.40 kg given specific conditions -/
theorem basket_weight (total_weight : Real) (remaining_weight : Real) 
  (h1 : total_weight = 11.48)
  (h2 : remaining_weight = 8.12) : 
  ∃ (basket_weight : Real) (apple_weight : Real),
    basket_weight = 1.40 ∧ 
    apple_weight > 0 ∧
    total_weight = basket_weight + 12 * apple_weight ∧
    remaining_weight = basket_weight + 8 * apple_weight :=
by
  sorry

end NUMINAMATH_CALUDE_basket_weight_l2018_201850


namespace NUMINAMATH_CALUDE_quadratic_root_sum_property_l2018_201882

theorem quadratic_root_sum_property (a b c : ℝ) (x₁ x₂ : ℝ) (p q r : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  (h4 : p = x₁ + x₂)
  (h5 : q = x₁^2 + x₂^2)
  (h6 : r = x₁^3 + x₂^3) :
  a * r + b * q + c * p = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_property_l2018_201882


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2018_201807

/-- The polynomial to be divided -/
def P (z : ℝ) : ℝ := 4*z^4 - 3*z^3 + 2*z^2 - 16*z + 9

/-- The divisor polynomial -/
def D (z : ℝ) : ℝ := 4*z + 6

/-- The theorem stating that the remainder of P(z) divided by D(z) is 173/12 -/
theorem polynomial_division_remainder :
  ∃ Q : ℝ → ℝ, ∀ z : ℝ, P z = D z * Q z + 173/12 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2018_201807


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2018_201892

theorem rectangular_prism_volume 
  (x y z : ℕ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : 4 * (x + y + z - 3) = 40)
  (h3 : 2 * (x * y + x * z + y * z - 2 * (x + y + z - 3)) = 66) :
  x * y * z = 150 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2018_201892


namespace NUMINAMATH_CALUDE_total_remaining_pictures_l2018_201873

structure ColoringBook where
  purchaseDay : Nat
  totalPictures : Nat
  coloredPerDay : Nat

def daysOfColoring (book : ColoringBook) : Nat :=
  6 - book.purchaseDay

def picturesColored (book : ColoringBook) : Nat :=
  book.coloredPerDay * daysOfColoring book

def picturesRemaining (book : ColoringBook) : Nat :=
  book.totalPictures - picturesColored book

def books : List ColoringBook := [
  ⟨1, 24, 4⟩,
  ⟨2, 37, 5⟩,
  ⟨3, 50, 6⟩,
  ⟨4, 33, 3⟩,
  ⟨5, 44, 7⟩
]

theorem total_remaining_pictures :
  (books.map picturesRemaining).sum = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_pictures_l2018_201873


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2018_201869

theorem quadratic_solution_property (k : ℝ) : 
  (∃ a b : ℝ, a ≠ b ∧ 
   3 * a^2 + 6 * a + k = 0 ∧ 
   3 * b^2 + 6 * b + k = 0 ∧
   |a - b| = (1/2) * (a^2 + b^2)) ↔ 
  (k = 0 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2018_201869


namespace NUMINAMATH_CALUDE_pheasants_and_rabbits_l2018_201877

theorem pheasants_and_rabbits (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 35)
  (h2 : total_legs = 94) :
  ∃ (pheasants rabbits : ℕ),
    pheasants + rabbits = total_heads ∧
    2 * pheasants + 4 * rabbits = total_legs ∧
    pheasants = 23 ∧
    rabbits = 12 := by
  sorry

end NUMINAMATH_CALUDE_pheasants_and_rabbits_l2018_201877


namespace NUMINAMATH_CALUDE_cow_count_l2018_201859

/-- Represents the number of cows in a farm -/
def num_cows : ℕ := 40

/-- Represents the number of bags of husk consumed by a group of cows in 40 days -/
def group_consumption : ℕ := 40

/-- Represents the number of days it takes one cow to consume one bag of husk -/
def days_per_bag : ℕ := 40

/-- Represents the number of days over which the consumption is measured -/
def total_days : ℕ := 40

theorem cow_count :
  num_cows = group_consumption * days_per_bag / total_days :=
by sorry

end NUMINAMATH_CALUDE_cow_count_l2018_201859


namespace NUMINAMATH_CALUDE_current_calculation_l2018_201835

/-- Given complex numbers V₁, V₂, Z, V, and I, prove that I = -1 + i -/
theorem current_calculation (V₁ V₂ Z V I : ℂ) 
  (h1 : V₁ = 2 + I)
  (h2 : V₂ = -1 + 4*I)
  (h3 : Z = 2 + 2*I)
  (h4 : V = V₁ + V₂)
  (h5 : I = V / Z) :
  I = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_current_calculation_l2018_201835


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2018_201856

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2018_201856


namespace NUMINAMATH_CALUDE_right_handed_players_count_l2018_201860

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) 
  (left_handed_percentage : ℚ) :
  total_players = 120 →
  throwers = 58 →
  left_handed_percentage = 40 / 100 →
  (total_players - throwers : ℚ) * left_handed_percentage = 24 →
  throwers + (total_players - throwers - 24) = 96 :=
by sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l2018_201860


namespace NUMINAMATH_CALUDE_eldest_child_age_l2018_201898

theorem eldest_child_age (y m e : ℕ) : 
  m = y + 3 →
  e = 3 * y →
  e = y + m + 2 →
  e = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_eldest_child_age_l2018_201898


namespace NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l2018_201894

/-- The distance between city A and city C given the travel times and speeds of Eddy and Freddy -/
theorem distance_A_to_C : ℝ :=
  let eddy_time : ℝ := 3
  let freddy_time : ℝ := 4
  let distance_A_to_B : ℝ := 570
  let speed_ratio : ℝ := 2.533333333333333

  let eddy_speed : ℝ := distance_A_to_B / eddy_time
  let freddy_speed : ℝ := eddy_speed / speed_ratio
  
  freddy_speed * freddy_time

/-- The distance between city A and city C is 300 km -/
theorem distance_A_to_C_is_300 : distance_A_to_C = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l2018_201894


namespace NUMINAMATH_CALUDE_courtyard_width_l2018_201812

/-- The width of a rectangular courtyard given its length and paving stone requirements. -/
theorem courtyard_width
  (length : ℝ)
  (num_stones : ℕ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (h1 : length = 50)
  (h2 : num_stones = 165)
  (h3 : stone_length = 5/2)
  (h4 : stone_width = 2)
  : (num_stones * stone_length * stone_width) / length = 33/2 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l2018_201812


namespace NUMINAMATH_CALUDE_vacation_homework_pages_l2018_201824

/-- Represents the number of days Garin divided her homework for -/
def days : ℕ := 24

/-- Represents the number of pages Garin can solve per day -/
def pages_per_day : ℕ := 19

/-- Calculates the total number of pages in Garin's vacation homework -/
def total_pages : ℕ := days * pages_per_day

/-- Proves that the total number of pages in Garin's vacation homework is 456 -/
theorem vacation_homework_pages : total_pages = 456 := by
  sorry

end NUMINAMATH_CALUDE_vacation_homework_pages_l2018_201824


namespace NUMINAMATH_CALUDE_robin_gum_count_l2018_201844

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 25

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 42

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 1050 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l2018_201844


namespace NUMINAMATH_CALUDE_buy_one_get_one_free_promotion_l2018_201825

/-- Calculates the total number of items received in a "buy one get one free" promotion --/
def itemsReceived (itemCost : ℕ) (totalPaid : ℕ) : ℕ :=
  2 * (totalPaid / itemCost)

/-- Theorem: Given a "buy one get one free" promotion where each item costs $3
    and a total payment of $15, the number of items received is 10 --/
theorem buy_one_get_one_free_promotion (itemCost : ℕ) (totalPaid : ℕ) 
    (h1 : itemCost = 3) (h2 : totalPaid = 15) : 
    itemsReceived itemCost totalPaid = 10 := by
  sorry

#eval itemsReceived 3 15  -- Should output 10

end NUMINAMATH_CALUDE_buy_one_get_one_free_promotion_l2018_201825


namespace NUMINAMATH_CALUDE_triangle_inequality_l2018_201834

theorem triangle_inequality (A B C : Real) (h : A + B + C = π) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤ 1 + (1 / 2) * (Real.cos ((A - B) / 4))^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2018_201834


namespace NUMINAMATH_CALUDE_function_inequality_l2018_201826

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative condition
variable (h : ∀ x, deriv f x > deriv g x)

-- Define the theorem
theorem function_inequality (a x b : ℝ) (h_order : a < x ∧ x < b) :
  f x + g a > g x + f a := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2018_201826


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l2018_201862

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_16 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 16 → n ≤ 952 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l2018_201862


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2018_201813

theorem complex_magnitude_product : Complex.abs (3 - 5 * Complex.I) * Complex.abs (3 + 5 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2018_201813


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2018_201870

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2018_201870


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2018_201842

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2018_201842


namespace NUMINAMATH_CALUDE_marys_tickets_l2018_201818

theorem marys_tickets (total_tickets : ℕ) (probability : ℚ) (marys_tickets : ℕ) : 
  total_tickets = 120 →
  probability = 1 / 15 →
  (marys_tickets : ℚ) / total_tickets = probability →
  marys_tickets = 8 := by
  sorry

end NUMINAMATH_CALUDE_marys_tickets_l2018_201818


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2018_201805

/-- Partnership profit sharing problem -/
theorem partnership_profit_share
  (x : ℝ)  -- A's investment amount
  (annual_gain : ℝ)  -- Total annual gain
  (h1 : annual_gain = 18900)  -- Given annual gain
  (h2 : x > 0)  -- Assumption that A's investment is positive
  : x * 12 / (x * 12 + 2 * x * 6 + 3 * x * 4) * annual_gain = 6300 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l2018_201805


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l2018_201888

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  max
    ((floor.length / tile.length) * (floor.width / tile.width))
    ((floor.length / tile.width) * (floor.width / tile.length))

/-- Theorem stating the maximum number of tiles that can be placed on the given floor -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 560 240
  let tile := Dimensions.mk 60 56
  maxTiles floor tile = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l2018_201888


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l2018_201872

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem largest_five_digit_with_product_120 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 120 → n ≤ 85311 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l2018_201872


namespace NUMINAMATH_CALUDE_cartesian_equation_circle_C_arc_length_ratio_circle_C_line_l_l2018_201845

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = 3 + (1/2) * t ∧ y = -3 + (Real.sqrt 3 / 2) * t

-- Theorem for the Cartesian equation of circle C
theorem cartesian_equation_circle_C :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, circle_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ 
  (x - 3)^2 + y^2 = 9 :=
sorry

-- Define a function to represent the ratio of arc lengths
def arc_length_ratio (r₁ r₂ : ℝ) : Prop := r₁ / r₂ = 1 / 2

-- Theorem for the ratio of arc lengths
theorem arc_length_ratio_circle_C_line_l :
  ∃ r₁ r₂ : ℝ, arc_length_ratio r₁ r₂ ∧ 
  (∀ x y : ℝ, (x - 3)^2 + y^2 = 9 → 
    (∃ t : ℝ, line_l t x y) → 
    (r₁ + r₂ = 2 * Real.pi * 3 ∧ r₁ ≤ r₂)) :=
sorry

end NUMINAMATH_CALUDE_cartesian_equation_circle_C_arc_length_ratio_circle_C_line_l_l2018_201845


namespace NUMINAMATH_CALUDE_lead_is_29_points_l2018_201837

/-- The lead in points between two teams -/
def lead (our_score green_score : ℕ) : ℕ :=
  our_score - green_score

/-- Theorem: Given the final scores, prove the lead is 29 points -/
theorem lead_is_29_points : lead 68 39 = 29 := by
  sorry

end NUMINAMATH_CALUDE_lead_is_29_points_l2018_201837


namespace NUMINAMATH_CALUDE_min_distance_sum_of_quadratic_roots_l2018_201871

theorem min_distance_sum_of_quadratic_roots : 
  ∃ (α β : ℝ), (α^2 - 6*α + 5 = 0) ∧ (β^2 - 6*β + 5 = 0) ∧
  (∀ x : ℝ, |x - α| + |x - β| ≥ 4) ∧
  (∃ x : ℝ, |x - α| + |x - β| = 4) := by
sorry

end NUMINAMATH_CALUDE_min_distance_sum_of_quadratic_roots_l2018_201871


namespace NUMINAMATH_CALUDE_luisa_trip_cost_l2018_201833

/-- Represents a leg of Luisa's trip -/
structure TripLeg where
  distance : Float
  fuelEfficiency : Float
  gasPrice : Float

/-- Calculates the cost of gas for a single leg of the trip -/
def gasCost (leg : TripLeg) : Float :=
  (leg.distance / leg.fuelEfficiency) * leg.gasPrice

/-- Luisa's trip legs -/
def luisaTrip : List TripLeg := [
  { distance := 10, fuelEfficiency := 15, gasPrice := 3.50 },
  { distance := 6,  fuelEfficiency := 12, gasPrice := 3.60 },
  { distance := 7,  fuelEfficiency := 14, gasPrice := 3.40 },
  { distance := 5,  fuelEfficiency := 10, gasPrice := 3.55 },
  { distance := 3,  fuelEfficiency := 13, gasPrice := 3.55 },
  { distance := 9,  fuelEfficiency := 15, gasPrice := 3.50 }
]

/-- Calculates the total cost of Luisa's trip -/
def totalTripCost : Float :=
  luisaTrip.map gasCost |> List.sum

/-- Proves that the total cost of Luisa's trip is approximately $10.53 -/
theorem luisa_trip_cost : 
  (totalTripCost - 10.53).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_luisa_trip_cost_l2018_201833


namespace NUMINAMATH_CALUDE_ac_value_l2018_201889

def letter_value (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

theorem ac_value : word_value "ac" = 8 := by
  sorry

end NUMINAMATH_CALUDE_ac_value_l2018_201889


namespace NUMINAMATH_CALUDE_factor_expression_l2018_201836

theorem factor_expression (x : ℝ) : 75*x + 45 = 15*(5*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2018_201836


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l2018_201864

/-- The total path length of a vertex of an equilateral triangle rotating inside a square --/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (rotation_angle : ℝ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : rotation_angle = 60 * π / 180) : 
  (4 : ℝ) * 3 * triangle_side * rotation_angle = 12 * π := by
  sorry

#check triangle_rotation_path_length

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l2018_201864


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2018_201890

def numbers : List ℝ := [1924, 2057, 2170, 2229, 2301, 2365]

theorem mean_of_remaining_numbers (subset : List ℝ) (h1 : subset ⊆ numbers) 
  (h2 : subset.length = 4) (h3 : (subset.sum / subset.length) = 2187.25) :
  let remaining := numbers.filter (fun x => x ∉ subset)
  (remaining.sum / remaining.length) = 2148.5 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2018_201890


namespace NUMINAMATH_CALUDE_product_42_sum_9_l2018_201839

theorem product_42_sum_9 (a b c : ℕ+) : 
  a * b * c = 42 → a + b = 9 → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_42_sum_9_l2018_201839


namespace NUMINAMATH_CALUDE_unique_triple_lcm_gcd_l2018_201887

theorem unique_triple_lcm_gcd : 
  ∃! (x y z : ℕ+), 
    Nat.lcm x y = 100 ∧ 
    Nat.lcm x z = 450 ∧ 
    Nat.lcm y z = 1100 ∧ 
    Nat.gcd (Nat.gcd x y) z = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_lcm_gcd_l2018_201887


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2018_201866

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ (z : ℂ), z = (a^2 + a - 2 : ℝ) + (a^2 - 3*a + 2 : ℝ)*I ∧ z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2018_201866


namespace NUMINAMATH_CALUDE_remainder_sum_l2018_201802

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 47) (hd : d % 45 = 14) : (c + d) % 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2018_201802


namespace NUMINAMATH_CALUDE_train_route_encoding_l2018_201815

def encode_letter (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def decode_digit (n : ℕ) : Char :=
  Char.ofNat (n + 'A'.toNat - 1)

def encode_city (s : String) : List ℕ :=
  s.toList.map encode_letter

theorem train_route_encoding :
  (encode_city "UFA" = [21, 6, 1]) ∧
  (encode_city "BAKU" = [2, 1, 11, 21]) →
  "21221-211221".splitOn "-" = ["21221", "211221"] →
  ∃ (departure arrival : String),
    departure = "UFA" ∧
    arrival = "BAKU" ∧
    encode_city departure = [21, 6, 1] ∧
    encode_city arrival = [2, 1, 11, 21] :=
by sorry

end NUMINAMATH_CALUDE_train_route_encoding_l2018_201815


namespace NUMINAMATH_CALUDE_probability_same_parity_l2018_201822

-- Define the type for function parity
inductive Parity
| Even
| Odd
| Neither

-- Define a function to represent the parity of each given function
def function_parity : Fin 4 → Parity
| 0 => Parity.Neither  -- y = x^3 + 3x^2
| 1 => Parity.Even     -- y = (e^x + e^-x) / 2
| 2 => Parity.Odd      -- y = log_2 ((3-x)/(3+x))
| 3 => Parity.Even     -- y = x sin x

-- Define a function to check if two functions have the same parity
def same_parity (f1 f2 : Fin 4) : Bool :=
  match function_parity f1, function_parity f2 with
  | Parity.Even, Parity.Even => true
  | Parity.Odd, Parity.Odd => true
  | _, _ => false

-- Theorem statement
theorem probability_same_parity :
  (Finset.filter (fun p => same_parity p.1 p.2) (Finset.univ : Finset (Fin 4 × Fin 4))).card /
  (Finset.univ : Finset (Fin 4 × Fin 4)).card = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_same_parity_l2018_201822


namespace NUMINAMATH_CALUDE_final_savings_is_105_l2018_201828

/-- Calculates the final savings amount after a series of bets and savings --/
def finalSavings (initialWinnings : ℝ) : ℝ :=
  let firstSavings := initialWinnings * 0.5
  let secondBetAmount := initialWinnings * 0.5
  let secondBetProfit := secondBetAmount * 0.6
  let secondBetTotal := secondBetAmount + secondBetProfit
  let secondSavings := secondBetTotal * 0.5
  let remainingAfterSecond := secondBetTotal
  let thirdBetAmount := remainingAfterSecond * 0.3
  let thirdBetProfit := thirdBetAmount * 0.25
  let thirdBetTotal := thirdBetAmount + thirdBetProfit
  let thirdSavings := thirdBetTotal * 0.5
  firstSavings + secondSavings + thirdSavings

/-- The theorem stating that the final savings amount is $105.00 --/
theorem final_savings_is_105 :
  finalSavings 100 = 105 := by
  sorry

end NUMINAMATH_CALUDE_final_savings_is_105_l2018_201828


namespace NUMINAMATH_CALUDE_absent_days_calculation_l2018_201853

/-- Calculates the number of days absent given the total days, daily wage, daily fine, and total earnings -/
def days_absent (total_days : ℕ) (daily_wage : ℕ) (daily_fine : ℕ) (total_earnings : ℕ) : ℕ :=
  total_days - (total_earnings + total_days * daily_fine) / (daily_wage + daily_fine)

theorem absent_days_calculation :
  days_absent 30 10 2 216 = 7 := by
  sorry

end NUMINAMATH_CALUDE_absent_days_calculation_l2018_201853


namespace NUMINAMATH_CALUDE_ribbon_left_l2018_201868

theorem ribbon_left (num_gifts : ℕ) (ribbon_per_gift : ℚ) (total_ribbon : ℚ) :
  num_gifts = 8 →
  ribbon_per_gift = 3/2 →
  total_ribbon = 15 →
  total_ribbon - (num_gifts : ℚ) * ribbon_per_gift = 3 := by
sorry

end NUMINAMATH_CALUDE_ribbon_left_l2018_201868


namespace NUMINAMATH_CALUDE_seventh_term_approx_l2018_201804

/-- Represents a geometric sequence with 10 terms -/
structure GeometricSequence where
  a₁ : ℝ
  r : ℝ
  len : ℕ
  h_len : len = 10
  h_a₁ : a₁ = 4
  h_a₄ : a₁ * r^3 = 64
  h_a₁₀ : a₁ * r^9 = 39304

/-- The 7th term of the geometric sequence -/
def seventh_term (seq : GeometricSequence) : ℝ :=
  seq.a₁ * seq.r^6

/-- Theorem stating that the 7th term is approximately 976 -/
theorem seventh_term_approx (seq : GeometricSequence) :
  ∃ ε > 0, |seventh_term seq - 976| < ε :=
sorry

end NUMINAMATH_CALUDE_seventh_term_approx_l2018_201804


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2018_201886

theorem inequality_solution_set (x : ℝ) : (2 * x - 1 ≤ 3) ↔ (x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2018_201886


namespace NUMINAMATH_CALUDE_cake_recipe_flour_amount_l2018_201811

/-- The total number of cups of flour in Mary's cake recipe -/
def total_flour : ℕ := 9

/-- The total number of cups of sugar in the recipe -/
def total_sugar : ℕ := 11

/-- The number of cups of flour already added -/
def flour_added : ℕ := 4

/-- The difference between remaining sugar and remaining flour to be added -/
def sugar_flour_diff : ℕ := 6

theorem cake_recipe_flour_amount :
  total_flour = 9 ∧
  total_sugar = 11 ∧
  flour_added = 4 ∧
  sugar_flour_diff = 6 →
  total_flour = 9 :=
by sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_amount_l2018_201811


namespace NUMINAMATH_CALUDE_divisible_by_91_l2018_201847

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) :=
sorry

end NUMINAMATH_CALUDE_divisible_by_91_l2018_201847


namespace NUMINAMATH_CALUDE_negation_forall_positive_converse_product_zero_symmetry_implies_even_symmetry_shifted_functions_l2018_201867

-- Statement 1
theorem negation_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x, P x) ↔ ∃ x, ¬(P x) :=
sorry

-- Statement 2
theorem converse_product_zero (a b : ℝ) :
  (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (a = 0 ∨ b = 0 → a * b = 0) :=
sorry

-- Statement 3
theorem symmetry_implies_even (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = f (x - 1)) → (∀ x, f x = f (-x)) :=
sorry

-- Statement 4
theorem symmetry_shifted_functions (f : ℝ → ℝ) :
  ∀ x, f (x + 1) = f (-(x - 1)) :=
sorry

end NUMINAMATH_CALUDE_negation_forall_positive_converse_product_zero_symmetry_implies_even_symmetry_shifted_functions_l2018_201867


namespace NUMINAMATH_CALUDE_distance_between_foci_rectangular_hyperbola_l2018_201893

/-- The distance between the foci of a rectangular hyperbola -/
theorem distance_between_foci_rectangular_hyperbola (c : ℝ) :
  let hyperbola := {(x, y) : ℝ × ℝ | x * y = c^2}
  let foci := {(c, c), (-c, -c)}
  (Set.ncard foci = 2) →
  ∀ (f₁ f₂ : ℝ × ℝ), f₁ ∈ foci → f₂ ∈ foci → f₁ ≠ f₂ →
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 2 * c :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_rectangular_hyperbola_l2018_201893


namespace NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l2018_201809

theorem three_a_in_S_implies_a_in_S (a : ℤ) : 
  (∃ x y : ℤ, 3 * a = x^2 + 2 * y^2) → 
  (∃ u v : ℤ, a = u^2 + 2 * v^2) := by
sorry

end NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l2018_201809


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l2018_201838

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64 / 9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l2018_201838


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2018_201823

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (a - Complex.I * (a + 2)) = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2018_201823


namespace NUMINAMATH_CALUDE_no_triangle_with_heights_1_2_3_l2018_201820

theorem no_triangle_with_heights_1_2_3 : 
  ¬ ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧  -- triangle inequality
    (1 : ℝ) = (2 * (a * b * c).sqrt) / (b * c) ∧  -- height 1
    (2 : ℝ) = (2 * (a * b * c).sqrt) / (a * c) ∧  -- height 2
    (3 : ℝ) = (2 * (a * b * c).sqrt) / (a * b) :=  -- height 3
by sorry


end NUMINAMATH_CALUDE_no_triangle_with_heights_1_2_3_l2018_201820


namespace NUMINAMATH_CALUDE_tobias_lawn_mowing_charge_tobias_lawn_mowing_charge_proof_l2018_201878

/-- Tobias' lawn mowing problem -/
theorem tobias_lawn_mowing_charge : ℕ → Prop :=
  fun x =>
    let shoe_cost : ℕ := 95
    let saving_months : ℕ := 3
    let monthly_allowance : ℕ := 5
    let shovel_charge : ℕ := 7
    let remaining_money : ℕ := 15
    let lawns_mowed : ℕ := 4
    let driveways_shoveled : ℕ := 5
    
    (saving_months * monthly_allowance + lawns_mowed * x + driveways_shoveled * shovel_charge
      = shoe_cost + remaining_money) →
    x = 15

/-- The proof of Tobias' lawn mowing charge -/
theorem tobias_lawn_mowing_charge_proof : tobias_lawn_mowing_charge 15 := by
  sorry

end NUMINAMATH_CALUDE_tobias_lawn_mowing_charge_tobias_lawn_mowing_charge_proof_l2018_201878


namespace NUMINAMATH_CALUDE_other_number_proof_l2018_201816

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 24)
  (h2 : Nat.gcd a b = 4)
  (h3 : a = 12) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2018_201816


namespace NUMINAMATH_CALUDE_total_weight_N2O3_l2018_201849

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of Dinitrogen trioxide
def N_atoms_in_N2O3 : ℕ := 2
def O_atoms_in_N2O3 : ℕ := 3

-- Define the total molecular weight
def total_molecular_weight : ℝ := 228

-- Define the molecular weight of a single molecule of N2O3
def molecular_weight_N2O3 : ℝ := 
  N_atoms_in_N2O3 * atomic_weight_N * O_atoms_in_N2O3 * atomic_weight_O

-- Theorem: The total molecular weight of some moles of N2O3 is 228 g
theorem total_weight_N2O3 : 
  ∃ (n : ℝ), n * molecular_weight_N2O3 = total_molecular_weight :=
sorry

end NUMINAMATH_CALUDE_total_weight_N2O3_l2018_201849


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l2018_201861

/-- The locus of intersection points of two parameterized lines forms a circle -/
theorem intersection_locus_is_circle :
  ∀ (u x y : ℝ), 
    (3 * u - 4 * y + 2 = 0) →
    (2 * x - 3 * u * y - 4 = 0) →
    ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l2018_201861


namespace NUMINAMATH_CALUDE_part1_part2_part3_l2018_201848

-- Define the operation
def matrixOp (a b c d : ℚ) : ℚ := a * d - c * b

-- Theorem 1
theorem part1 : matrixOp (-3) (-2) 4 5 = -7 := by sorry

-- Theorem 2
theorem part2 : matrixOp 2 (-2 * x) 3 (-5 * x) = 2 → x = -1/2 := by sorry

-- Theorem 3
theorem part3 (x : ℚ) : 
  matrixOp (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = matrixOp 6 (-1) (-n) x →
  m = -3/8 ∧ n = -7 := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l2018_201848


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2018_201881

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^2) 
  (h₂ : a₂ = 2^(3/2)) 
  (h₃ : a₃ = 2) 
  (h_gp : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2018_201881


namespace NUMINAMATH_CALUDE_binomial_150_1_l2018_201858

theorem binomial_150_1 : Nat.choose 150 1 = 150 := by sorry

end NUMINAMATH_CALUDE_binomial_150_1_l2018_201858


namespace NUMINAMATH_CALUDE_soccer_ball_donation_l2018_201830

theorem soccer_ball_donation (total_balls : ℕ) (balls_per_class : ℕ) 
  (elementary_classes_per_school : ℕ) (num_schools : ℕ) 
  (h1 : total_balls = 90) 
  (h2 : balls_per_class = 5)
  (h3 : elementary_classes_per_school = 4)
  (h4 : num_schools = 2) : 
  (total_balls / (balls_per_class * num_schools)) - elementary_classes_per_school = 5 := by
  sorry

#check soccer_ball_donation

end NUMINAMATH_CALUDE_soccer_ball_donation_l2018_201830


namespace NUMINAMATH_CALUDE_range_of_a_l2018_201885

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

def prop_p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, x ≤ y → f a x ≥ f a y

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 1 ≤ 0

theorem range_of_a : 
  {a : ℝ | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)} = 
  Set.Iic (-2) ∪ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2018_201885


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_attained_l2018_201875

theorem max_value_of_function (x : ℝ) : 
  (3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x))) ≤ 5 := by
  sorry

theorem max_value_attained (x : ℝ) : 
  ∃ x, 3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_attained_l2018_201875


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2018_201846

def total_players : ℕ := 16
def num_quadruplets : ℕ := 4
def num_starters : ℕ := 6
def num_quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose num_quadruplets num_quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets + num_quadruplets_in_lineup)
              (num_starters - num_quadruplets_in_lineup)) = 6006 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2018_201846


namespace NUMINAMATH_CALUDE_largest_x_value_l2018_201817

theorem largest_x_value (x : ℝ) :
  (x / 7 + 3 / (7 * x) = 2 / 3) →
  x ≤ (7 + Real.sqrt 22) / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_l2018_201817
