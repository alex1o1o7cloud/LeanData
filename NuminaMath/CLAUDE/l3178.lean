import Mathlib

namespace NUMINAMATH_CALUDE_album_photos_l3178_317883

theorem album_photos (n : ℕ) 
  (h1 : ∀ (album : ℕ), album > 0 → ∃ (page : ℕ), page > 0 ∧ page ≤ n)
  (h2 : ∀ (page : ℕ), page > 0 → page ≤ n → ∃ (photos : Fin 4), True)
  (h3 : ∃ (album : ℕ), album > 0 ∧ 81 ∈ Set.range (λ i => 4*(n*(album-1) + 5) - 3 + i) ∧ (∀ j, j ∈ Set.range (λ i => 4*(n*(album-1) + 5) - 3 + i) → j ≤ 4*n*album))
  (h4 : ∃ (album : ℕ), album > 0 ∧ 171 ∈ Set.range (λ i => 4*(n*(album-1) + 3) - 3 + i) ∧ (∀ j, j ∈ Set.range (λ i => 4*(n*(album-1) + 3) - 3 + i) → j ≤ 4*n*album))
  : n = 8 ∧ 4*n = 32 := by
  sorry

end NUMINAMATH_CALUDE_album_photos_l3178_317883


namespace NUMINAMATH_CALUDE_cos_neg_seventeen_pi_fourths_l3178_317808

theorem cos_neg_seventeen_pi_fourths : Real.cos (-17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_seventeen_pi_fourths_l3178_317808


namespace NUMINAMATH_CALUDE_willies_bananas_unchanged_l3178_317873

/-- Willie's banana count remains unchanged regardless of Charles' banana count changes -/
theorem willies_bananas_unchanged (willie_initial : ℕ) (charles_initial charles_lost : ℕ) :
  willie_initial = 48 → willie_initial = willie_initial :=
by
  sorry

end NUMINAMATH_CALUDE_willies_bananas_unchanged_l3178_317873


namespace NUMINAMATH_CALUDE_people_on_boats_l3178_317859

/-- Given 5 boats in a lake, each with 3 people, prove that the total number of people on boats is 15. -/
theorem people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) 
  (h1 : num_boats = 5) 
  (h2 : people_per_boat = 3) : 
  num_boats * people_per_boat = 15 := by
  sorry

end NUMINAMATH_CALUDE_people_on_boats_l3178_317859


namespace NUMINAMATH_CALUDE_triangle_properties_l3178_317871

/-- Given a, b, and c are side lengths of a triangle, prove the following properties --/
theorem triangle_properties (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) :
  (a + b - c > 0) ∧ 
  (a - b + c > 0) ∧ 
  (a - b - c < 0) ∧
  (|a + b - c| - |a - b + c| + |a - b - c| = -a + 3*b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3178_317871


namespace NUMINAMATH_CALUDE_school_sections_l3178_317888

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l3178_317888


namespace NUMINAMATH_CALUDE_locus_of_sine_zero_l3178_317830

theorem locus_of_sine_zero (x y : ℝ) : 
  Real.sin (x + y) = 0 ↔ ∃ k : ℤ, x + y = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_locus_of_sine_zero_l3178_317830


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3178_317852

open Set
open Function

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality 
  (hf_domain : ∀ x, x ∈ (Set.Ioi 0) → DifferentiableAt ℝ f x)
  (hf'_def : ∀ x, x ∈ (Set.Ioi 0) → HasDerivAt f (f' x) x)
  (hf'_condition : ∀ x, x ∈ (Set.Ioi 0) → x * f' x > f x) :
  {x : ℝ | (x - 1) * f (x + 1) > f (x^2 - 1)} = Ioo 1 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3178_317852


namespace NUMINAMATH_CALUDE_max_three_digit_sum_not_factor_l3178_317854

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_not_factor_of_product (n : ℕ) : Prop :=
  ¬(2 * Nat.factorial (n - 1)) % (n + 1) = 0

theorem max_three_digit_sum_not_factor :
  ∃ (n : ℕ), is_three_digit n ∧ sum_not_factor_of_product n ∧
  ∀ (m : ℕ), is_three_digit m → sum_not_factor_of_product m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_sum_not_factor_l3178_317854


namespace NUMINAMATH_CALUDE_largest_power_dividing_product_l3178_317874

-- Define pow function
def pow (n : ℕ) : ℕ := sorry

-- Define the product of pow(n) from 2 to 5300
def product : ℕ := sorry

-- State the theorem
theorem largest_power_dividing_product : 
  (∃ m : ℕ, (2010 ^ m : ℕ) ∣ product ∧ 
   ∀ k : ℕ, k > m → ¬((2010 ^ k : ℕ) ∣ product)) ∧ 
  (∃ m : ℕ, m = 77 ∧ (2010 ^ m : ℕ) ∣ product ∧ 
   ∀ k : ℕ, k > m → ¬((2010 ^ k : ℕ) ∣ product)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_dividing_product_l3178_317874


namespace NUMINAMATH_CALUDE_quadratic_bounded_values_l3178_317881

/-- A quadratic function f(x) = ax^2 + bx + c where a > 100 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_bounded_values (a b c : ℝ) (ha : a > 100) :
  ∃ (n : ℕ), n ≤ 2 ∧
  ∀ (S : Finset ℤ), (∀ x ∈ S, |QuadraticFunction a b c x| ≤ 50) →
  Finset.card S ≤ n :=
sorry

end NUMINAMATH_CALUDE_quadratic_bounded_values_l3178_317881


namespace NUMINAMATH_CALUDE_unique_prime_pair_l3178_317809

theorem unique_prime_pair : ∃! p : ℕ, Prime p ∧ Prime (p + 15) := by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l3178_317809


namespace NUMINAMATH_CALUDE_correct_calculation_l3178_317818

theorem correct_calculation (x : ℤ) (h : x - 48 = 52) : x + 48 = 148 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3178_317818


namespace NUMINAMATH_CALUDE_min_k_value_l3178_317806

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is the length of side AB
  AB : ℝ
  -- h is the height of the trapezoid
  h : ℝ
  -- E and F are midpoints of AD and AB respectively
  -- CD = 2AB (implied by the structure)

/-- The area difference between triangle CDG and quadrilateral AEGF -/
def areaDifference (t : Trapezoid) : ℝ :=
  2 * t.AB * t.h - t.AB * t.h

/-- The area of the trapezoid ABCD -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  3 * t.AB * t.h

/-- Main theorem: The minimum value of k is 8 -/
theorem min_k_value (t : Trapezoid) (k : ℕ+) 
    (h1 : areaDifference t = k / 24)
    (h2 : ∃ n : ℕ, trapezoidArea t = n) : 
  k ≥ 8 ∧ ∃ (t : Trapezoid) (k : ℕ+), k = 8 ∧ areaDifference t = k / 24 ∧ ∃ (n : ℕ), trapezoidArea t = n :=
by
  sorry

end NUMINAMATH_CALUDE_min_k_value_l3178_317806


namespace NUMINAMATH_CALUDE_mono_increasing_minus_decreasing_mono_decreasing_minus_increasing_l3178_317837

-- Define monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Theorem for proposition ②
theorem mono_increasing_minus_decreasing
  (f g : ℝ → ℝ) (hf : MonoIncreasing f) (hg : MonoDecreasing g) :
  MonoIncreasing (fun x ↦ f x - g x) :=
sorry

-- Theorem for proposition ③
theorem mono_decreasing_minus_increasing
  (f g : ℝ → ℝ) (hf : MonoDecreasing f) (hg : MonoIncreasing g) :
  MonoDecreasing (fun x ↦ f x - g x) :=
sorry

end NUMINAMATH_CALUDE_mono_increasing_minus_decreasing_mono_decreasing_minus_increasing_l3178_317837


namespace NUMINAMATH_CALUDE_total_wheels_is_25_l3178_317889

/-- The number of wheels in Zoe's garage --/
def total_wheels : ℕ :=
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 4
  let num_unicycles : ℕ := 7
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  let wheels_per_unicycle : ℕ := 1
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle

theorem total_wheels_is_25 : total_wheels = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_25_l3178_317889


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l3178_317810

theorem unique_solution_quadratic_equation :
  ∃! x : ℝ, (3012 + x)^2 = x^2 ∧ x = -1506 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l3178_317810


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3178_317805

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13_to_base10 (n : ℕ) : ℕ := sorry

theorem base_conversion_sum :
  let base8_num := 357
  let base13_num := 4 * 13^2 + 12 * 13 + 13
  (base8_to_base10 base8_num) + (base13_to_base10 base13_num) = 1084 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3178_317805


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3178_317822

def cost_price : ℝ := 180
def selling_price : ℝ := 207

theorem profit_percentage_calculation :
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3178_317822


namespace NUMINAMATH_CALUDE_virus_memory_growth_l3178_317815

theorem virus_memory_growth (initial_memory : ℕ) (growth_interval : ℕ) (final_memory : ℕ) :
  initial_memory = 2 →
  growth_interval = 3 →
  final_memory = 64 * 2^10 →
  (fun n => initial_memory * 2^n) (15 * growth_interval / growth_interval) = final_memory :=
by sorry

end NUMINAMATH_CALUDE_virus_memory_growth_l3178_317815


namespace NUMINAMATH_CALUDE_triangle_inequality_l3178_317819

/-- A triangle with heights and an internal point -/
structure Triangle where
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  l_a : ℝ
  l_b : ℝ
  l_c : ℝ
  h_a_pos : h_a > 0
  h_b_pos : h_b > 0
  h_c_pos : h_c > 0
  l_a_pos : l_a > 0
  l_b_pos : l_b > 0
  l_c_pos : l_c > 0

/-- The inequality holds for any triangle -/
theorem triangle_inequality (t : Triangle) :
  t.h_a / t.l_a + t.h_b / t.l_b + t.h_c / t.l_c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3178_317819


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_over_81_l3178_317829

theorem sqrt_of_sqrt_16_over_81 : Real.sqrt (Real.sqrt (16 / 81)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_over_81_l3178_317829


namespace NUMINAMATH_CALUDE_function_equality_l3178_317816

theorem function_equality (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_function_equality_l3178_317816


namespace NUMINAMATH_CALUDE_tangent_segment_length_l3178_317842

theorem tangent_segment_length (r : ℝ) (a b : ℝ) : 
  r = 15 ∧ a = 6 ∧ b = 3 →
  ∃ x : ℝ, x = 12 ∧
    r^2 = x^2 + ((x + r - a - b) / 2)^2 ∧
    x + r = a + b + x + r - a - b :=
by sorry

end NUMINAMATH_CALUDE_tangent_segment_length_l3178_317842


namespace NUMINAMATH_CALUDE_heather_oranges_l3178_317828

def oranges_problem (initial : ℕ) (russell_takes : ℕ) (samantha_takes : ℕ) : Prop :=
  initial - russell_takes - samantha_takes = 13

theorem heather_oranges :
  oranges_problem 60 35 12 := by
  sorry

end NUMINAMATH_CALUDE_heather_oranges_l3178_317828


namespace NUMINAMATH_CALUDE_windy_driving_time_l3178_317884

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  non_windy_speed : ℝ  -- Speed in non-windy conditions (miles per hour)
  windy_speed : ℝ      -- Speed in windy conditions (miles per hour)
  total_distance : ℝ   -- Total distance covered (miles)
  total_time : ℝ       -- Total time spent driving (minutes)

/-- Calculates the time spent driving in windy conditions -/
def time_in_windy_conditions (scenario : DrivingScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the time spent in windy conditions is 20 minutes -/
theorem windy_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.non_windy_speed = 40)
  (h2 : scenario.windy_speed = 25)
  (h3 : scenario.total_distance = 25)
  (h4 : scenario.total_time = 45) :
  time_in_windy_conditions scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_windy_driving_time_l3178_317884


namespace NUMINAMATH_CALUDE_evaluate_expression_l3178_317844

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3178_317844


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3178_317887

/-- The probability of selecting two non-defective pens from a box of pens -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 4) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  (total_pens - defective_pens - 1) / (total_pens - 1) = 14 / 33 := by
  sorry

#check probability_two_non_defective_pens

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3178_317887


namespace NUMINAMATH_CALUDE_courtyard_tile_cost_l3178_317802

/-- Calculate the total cost of tiles for a courtyard -/
theorem courtyard_tile_cost : 
  let courtyard_length : ℝ := 10
  let courtyard_width : ℝ := 25
  let tiles_per_sqft : ℝ := 4
  let green_tile_percentage : ℝ := 0.4
  let green_tile_cost : ℝ := 3
  let red_tile_cost : ℝ := 1.5

  let total_area : ℝ := courtyard_length * courtyard_width
  let total_tiles : ℝ := total_area * tiles_per_sqft
  let green_tiles : ℝ := green_tile_percentage * total_tiles
  let red_tiles : ℝ := total_tiles - green_tiles

  let total_cost : ℝ := green_tiles * green_tile_cost + red_tiles * red_tile_cost

  total_cost = 2100 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_tile_cost_l3178_317802


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3178_317851

/-- The number of positive single-digit integers A for which x^2 - (2A + 1)x + 3A = 0 has positive integer solutions is 1. -/
theorem unique_quadratic_solution : 
  ∃! (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 
  ∃ (x : ℕ), x > 0 ∧ x^2 - (2 * A + 1) * x + 3 * A = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3178_317851


namespace NUMINAMATH_CALUDE_no_high_grades_l3178_317836

/-- Represents the test scenario with given conditions -/
structure TestScenario where
  n : ℕ  -- number of students excluding Peter
  k : ℕ  -- number of problems solved by each student except Peter
  total_problems_solved : ℕ  -- total number of problems solved by all students

/-- The conditions of the test scenario -/
def valid_scenario (s : TestScenario) : Prop :=
  s.total_problems_solved = 25 ∧
  s.n * s.k + (s.k + 1) = s.total_problems_solved ∧
  s.k ≤ 5

/-- The theorem stating that no student received a grade of 4 or 5 -/
theorem no_high_grades (s : TestScenario) (h : valid_scenario s) : 
  s.k < 4 ∧ s.k + 1 < 5 := by
  sorry

#check no_high_grades

end NUMINAMATH_CALUDE_no_high_grades_l3178_317836


namespace NUMINAMATH_CALUDE_base5_1204_eq_179_l3178_317867

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 5^3 + d₂ * 5^2 + d₁ * 5^1 + d₀ * 5^0

/-- Proves that 1204₍₅₎ is equal to 179 in decimal --/
theorem base5_1204_eq_179 : base5ToDecimal 1 2 0 4 = 179 := by
  sorry

end NUMINAMATH_CALUDE_base5_1204_eq_179_l3178_317867


namespace NUMINAMATH_CALUDE_bhanu_petrol_expense_l3178_317835

theorem bhanu_petrol_expense (income : ℝ) (petrol_percent house_rent_percent : ℝ) 
  (house_rent : ℝ) : 
  petrol_percent = 0.3 →
  house_rent_percent = 0.1 →
  house_rent = 70 →
  house_rent_percent * (income - petrol_percent * income) = house_rent →
  petrol_percent * income = 300 :=
by sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expense_l3178_317835


namespace NUMINAMATH_CALUDE_tshirt_shop_profit_l3178_317876

theorem tshirt_shop_profit : 
  let profit_per_shirt : ℚ := 9
  let cost_per_shirt : ℚ := 4
  let num_shirts : ℕ := 245
  let discount_rate : ℚ := 1/5

  let original_price : ℚ := profit_per_shirt + cost_per_shirt
  let discounted_price : ℚ := original_price * (1 - discount_rate)
  let total_revenue : ℚ := (discounted_price * num_shirts : ℚ)
  let total_cost : ℚ := (cost_per_shirt * num_shirts : ℚ)
  let total_profit : ℚ := total_revenue - total_cost

  total_profit = 1568 := by sorry

end NUMINAMATH_CALUDE_tshirt_shop_profit_l3178_317876


namespace NUMINAMATH_CALUDE_power_equality_l3178_317864

theorem power_equality (m : ℕ) : 9^4 = 3^m → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3178_317864


namespace NUMINAMATH_CALUDE_no_single_liar_l3178_317860

-- Define the propositions
variable (J : Prop) -- Jean is lying
variable (P : Prop) -- Pierre is lying

-- Jean's statement: "When I am not lying, you are not lying either"
axiom jean_statement : ¬J → ¬P

-- Pierre's statement: "When I am lying, you are lying too"
axiom pierre_statement : P → J

-- Theorem: It's impossible for one to be lying and the other not
theorem no_single_liar : ¬(J ∧ ¬P) ∧ ¬(¬J ∧ P) := by
  sorry


end NUMINAMATH_CALUDE_no_single_liar_l3178_317860


namespace NUMINAMATH_CALUDE_currency_notes_count_l3178_317862

/-- Given a total amount of currency notes and specific conditions, 
    prove the total number of notes. -/
theorem currency_notes_count 
  (total_amount : ℕ) 
  (denomination_70 : ℕ) 
  (denomination_50 : ℕ) 
  (amount_in_50 : ℕ) 
  (h1 : total_amount = 5000)
  (h2 : denomination_70 = 70)
  (h3 : denomination_50 = 50)
  (h4 : amount_in_50 = 100)
  (h5 : ∃ (x y : ℕ), denomination_70 * x + denomination_50 * y = total_amount ∧ 
                     denomination_50 * (amount_in_50 / denomination_50) = amount_in_50) :
  ∃ (x y : ℕ), denomination_70 * x + denomination_50 * y = total_amount ∧ x + y = 72 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_count_l3178_317862


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3178_317801

/-- Given a cylinder with volume 72π cm³, prove that a cone with double the height 
    and the same radius as the cylinder has a volume of 48π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) → 
  (1/3 : ℝ) * π * r^2 * (2 * h) = 48 * π := by
sorry


end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3178_317801


namespace NUMINAMATH_CALUDE_housing_boom_proof_l3178_317897

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_proof : houses_built = 574 := by
  sorry

end NUMINAMATH_CALUDE_housing_boom_proof_l3178_317897


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3178_317891

noncomputable def f (a x : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, x > (1/2) ∨ x < -(1/2) → f a x = f a (-x)) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3178_317891


namespace NUMINAMATH_CALUDE_periodic_function_value_l3178_317800

theorem periodic_function_value (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 4) = f x) 
  (h2 : f 0.5 = 9) : 
  f 8.5 = 9 := by sorry

end NUMINAMATH_CALUDE_periodic_function_value_l3178_317800


namespace NUMINAMATH_CALUDE_square_of_difference_101_minus_2_l3178_317866

theorem square_of_difference_101_minus_2 :
  (101 - 2)^2 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_101_minus_2_l3178_317866


namespace NUMINAMATH_CALUDE_highlighter_profit_l3178_317845

/-- Calculates the profit from selling highlighter pens under specific conditions --/
theorem highlighter_profit : 
  let total_boxes : ℕ := 12
  let pens_per_box : ℕ := 30
  let cost_per_box : ℕ := 10
  let rearranged_boxes : ℕ := 5
  let pens_per_package : ℕ := 6
  let price_per_package : ℕ := 3
  let pens_per_group : ℕ := 3
  let price_per_group : ℕ := 2

  let total_cost : ℕ := total_boxes * cost_per_box
  let total_pens : ℕ := total_boxes * pens_per_box
  let packages : ℕ := rearranged_boxes * (pens_per_box / pens_per_package)
  let revenue_packages : ℕ := packages * price_per_package
  let remaining_pens : ℕ := total_pens - (rearranged_boxes * pens_per_box)
  let groups : ℕ := remaining_pens / pens_per_group
  let revenue_groups : ℕ := groups * price_per_group
  let total_revenue : ℕ := revenue_packages + revenue_groups
  let profit : ℕ := total_revenue - total_cost

  profit = 115 := by sorry

end NUMINAMATH_CALUDE_highlighter_profit_l3178_317845


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3178_317885

/-- Given vectors a and b in ℝ², where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3178_317885


namespace NUMINAMATH_CALUDE_prime_pairs_congruence_l3178_317831

theorem prime_pairs_congruence (p : Nat) : Prime p →
  (∃! n : Nat, n = (Finset.filter (fun pair : Nat × Nat =>
    0 ≤ pair.1 ∧ pair.1 ≤ p ∧
    0 ≤ pair.2 ∧ pair.2 ≤ p ∧
    (pair.2 ^ 2) % p = ((pair.1 ^ 3) - pair.1) % p)
    (Finset.product (Finset.range (p + 1)) (Finset.range (p + 1)))).card ∧ n = p) ↔
  (p = 2 ∨ p % 4 = 3) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_congruence_l3178_317831


namespace NUMINAMATH_CALUDE_triangle_area_relationship_uncertain_l3178_317899

/-- A triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

/-- Proposition: The relationship between areas of two triangles is uncertain -/
theorem triangle_area_relationship_uncertain 
  (ABC : Triangle) (A₁B₁C₁ : Triangle) 
  (h1 : ABC.a > A₁B₁C₁.a) 
  (h2 : ABC.b > A₁B₁C₁.b) 
  (h3 : ABC.c > A₁B₁C₁.c) :
  ¬ (∀ (ABC A₁B₁C₁ : Triangle), 
    ABC.a > A₁B₁C₁.a → ABC.b > A₁B₁C₁.b → ABC.c > A₁B₁C₁.c → 
    (ABC.area > A₁B₁C₁.area ∨ ABC.area < A₁B₁C₁.area ∨ ABC.area = A₁B₁C₁.area)) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_relationship_uncertain_l3178_317899


namespace NUMINAMATH_CALUDE_exists_valid_permutation_l3178_317803

/-- A permutation of numbers from 1 to 200 -/
def Permutation := Fin 200 → Fin 200

/-- Check if a permutation satisfies the adjacent difference condition -/
def ValidPermutation (p : Permutation) : Prop :=
  ∀ i : Fin 199, (p (i + 1) - p i).val = 3 ∨ (p (i + 1) - p i).val = 5 ∨
                 (p i - p (i + 1)).val = 3 ∨ (p i - p (i + 1)).val = 5

/-- Theorem stating the existence of a valid permutation -/
theorem exists_valid_permutation : ∃ p : Permutation, ValidPermutation p :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_permutation_l3178_317803


namespace NUMINAMATH_CALUDE_equation_solution_l3178_317879

theorem equation_solution : ∃ x : ℚ, (2 / 5 - 1 / 3 : ℚ) = 1 / x ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3178_317879


namespace NUMINAMATH_CALUDE_min_value_constraint_min_value_achieved_l3178_317868

theorem min_value_constraint (x y : ℝ) (h : 2 * x + 8 * y = 3) :
  x^2 + 4 * y^2 - 2 * x ≥ -19/20 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 2 * x + 8 * y = 3 ∧ x^2 + 4 * y^2 - 2 * x < -19/20 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_min_value_achieved_l3178_317868


namespace NUMINAMATH_CALUDE_children_born_in_current_marriage_l3178_317825

/-- Represents the number of children in a blended family scenario -/
structure BlendedFamily where
  x : ℕ  -- children from father's previous marriage
  y : ℕ  -- children from mother's previous marriage
  z : ℕ  -- children born in current marriage
  total_children : x + y + z = 12
  father_bio_children : x + z = 9
  mother_bio_children : y + z = 9

/-- Theorem stating that in this blended family scenario, 6 children were born in the current marriage -/
theorem children_born_in_current_marriage (family : BlendedFamily) : family.z = 6 := by
  sorry

#check children_born_in_current_marriage

end NUMINAMATH_CALUDE_children_born_in_current_marriage_l3178_317825


namespace NUMINAMATH_CALUDE_josh_recording_time_l3178_317861

/-- A device that records temperature data at regular intervals. -/
structure TemperatureRecorder where
  interval : ℕ  -- Recording interval in seconds
  instances : ℕ  -- Number of recorded instances

/-- Calculates the total recording time in hours for a TemperatureRecorder. -/
def totalRecordingTime (recorder : TemperatureRecorder) : ℚ :=
  (recorder.interval * recorder.instances : ℚ) / 3600

/-- Theorem: Josh's device recorded data for 1 hour. -/
theorem josh_recording_time :
  let device : TemperatureRecorder := { interval := 5, instances := 720 }
  totalRecordingTime device = 1 := by sorry

end NUMINAMATH_CALUDE_josh_recording_time_l3178_317861


namespace NUMINAMATH_CALUDE_log_35_28_in_terms_of_a_and_b_l3178_317869

theorem log_35_28_in_terms_of_a_and_b (a b : ℝ) 
  (h1 : Real.log 7 / Real.log 14 = a) 
  (h2 : Real.log 5 / Real.log 14 = b) : 
  Real.log 28 / Real.log 35 = (2 - a) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_log_35_28_in_terms_of_a_and_b_l3178_317869


namespace NUMINAMATH_CALUDE_number_of_possible_scores_l3178_317865

-- Define the scoring system
def problem_scores : List Nat := [1, 2, 3, 4]
def time_bonuses : List Nat := [1, 2, 3, 4]
def all_correct_bonus : Nat := 20

-- Function to calculate all possible scores
def calculate_scores : List Nat :=
  let base_scores := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let multiplied_scores := 
    List.join (base_scores.map (λ s => time_bonuses.map (λ t => s * t)))
  let all_correct_scores := 
    time_bonuses.map (λ t => 10 * t + all_correct_bonus)
  List.eraseDups (multiplied_scores ++ all_correct_scores)

-- Theorem statement
theorem number_of_possible_scores : 
  calculate_scores.length = 25 := by sorry

end NUMINAMATH_CALUDE_number_of_possible_scores_l3178_317865


namespace NUMINAMATH_CALUDE_log_difference_theorem_l3178_317855

noncomputable def logBase (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def satisfies_condition (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc 1 3, logBase a x ≤ logBase a 3) ∧
  (∀ x ∈ Set.Icc 1 3, logBase a x ≥ logBase a 1) ∧
  logBase a 3 - logBase a 1 = 2

theorem log_difference_theorem :
  {a : ℝ | satisfies_condition a} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
sorry

end NUMINAMATH_CALUDE_log_difference_theorem_l3178_317855


namespace NUMINAMATH_CALUDE_max_squares_covered_proof_l3178_317820

/-- The side length of a checkerboard square in inches -/
def checkerboard_square_side : ℝ := 1.25

/-- The side length of the square card in inches -/
def card_side : ℝ := 1.75

/-- The maximum number of checkerboard squares that can be covered by the card -/
def max_squares_covered : ℕ := 9

/-- Theorem stating the maximum number of squares that can be covered by the card -/
theorem max_squares_covered_proof :
  ∀ (card_placement : ℝ × ℝ → Bool),
  (∃ (covered_squares : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ covered_squares →
      ∃ (x y : ℝ), 0 ≤ x ∧ x < card_side ∧ 0 ≤ y ∧ y < card_side ∧
        card_placement (x + i * checkerboard_square_side, y + j * checkerboard_square_side)) ∧
    covered_squares.card ≤ max_squares_covered) ∧
  (∃ (optimal_placement : ℝ × ℝ → Bool) (optimal_covered_squares : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ optimal_covered_squares →
      ∃ (x y : ℝ), 0 ≤ x ∧ x < card_side ∧ 0 ≤ y ∧ y < card_side ∧
        optimal_placement (x + i * checkerboard_square_side, y + j * checkerboard_square_side)) ∧
    optimal_covered_squares.card = max_squares_covered) :=
by sorry

end NUMINAMATH_CALUDE_max_squares_covered_proof_l3178_317820


namespace NUMINAMATH_CALUDE_non_fiction_count_l3178_317814

/-- The number of fiction books -/
def fiction_books : ℕ := 5

/-- The number of ways to select 2 fiction and 2 non-fiction books -/
def selection_ways : ℕ := 150

/-- Combination function -/
def C (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of non-fiction books -/
def non_fiction_books : ℕ := 6

theorem non_fiction_count : 
  C fiction_books 2 * C non_fiction_books 2 = selection_ways := by sorry

end NUMINAMATH_CALUDE_non_fiction_count_l3178_317814


namespace NUMINAMATH_CALUDE_factorial_7_base_9_trailing_zeros_l3178_317892

-- Define 7!
def factorial_7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the base conversion function (simplified)
noncomputable def to_base_9 (n : ℕ) : List ℕ :=
  sorry  -- Actual implementation would go here

-- Define a function to count trailing zeros
def count_trailing_zeros (digits : List ℕ) : ℕ :=
  sorry  -- Actual implementation would go here

-- Theorem statement
theorem factorial_7_base_9_trailing_zeros :
  count_trailing_zeros (to_base_9 factorial_7) = 1 :=
sorry

end NUMINAMATH_CALUDE_factorial_7_base_9_trailing_zeros_l3178_317892


namespace NUMINAMATH_CALUDE_jane_sarah_age_sum_l3178_317863

theorem jane_sarah_age_sum : 
  ∀ (jane sarah : ℝ),
  jane = sarah + 5 →
  jane + 9 = 3 * (sarah - 3) →
  jane + sarah = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_sarah_age_sum_l3178_317863


namespace NUMINAMATH_CALUDE_a_properties_l3178_317878

/-- Sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1  -- a_1 = 1
  | 1 => 6  -- a_2 = 6
  | (n+2) => ((n+3) * (a (n+1) - 1)) / (n+2)

/-- Theorem stating the properties of sequence a_n -/
theorem a_properties :
  (∀ n : ℕ, a n = 2 * n^2 - n) ∧
  (∃ p q : ℚ, p ≠ 0 ∧ q ≠ 0 ∧
    (∃ d : ℚ, ∀ n : ℕ, a (n+1) / (p * (n+1) + q) - a n / (p * n + q) = d) ↔
    p + 2*q = 0) := by sorry

end NUMINAMATH_CALUDE_a_properties_l3178_317878


namespace NUMINAMATH_CALUDE_partnership_gain_l3178_317894

/-- Represents the annual gain of a partnership given investments and durations -/
def annual_gain (x : ℝ) : ℝ :=
  let a_investment := x * 12
  let b_investment := 2 * x * 6
  let c_investment := 3 * x * 4
  let total_investment := a_investment + b_investment + c_investment
  let a_share := 6400
  3 * a_share

/-- Theorem stating that the annual gain of the partnership is 19200 -/
theorem partnership_gain : annual_gain x = 19200 :=
sorry

end NUMINAMATH_CALUDE_partnership_gain_l3178_317894


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3178_317872

theorem cubic_equation_solution (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3178_317872


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l3178_317848

theorem sqrt_division_equality : Real.sqrt 10 / Real.sqrt 5 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l3178_317848


namespace NUMINAMATH_CALUDE_total_harvest_l3178_317886

/-- The number of sacks of oranges harvested per day -/
def daily_harvest : ℕ := 83

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- Theorem stating the total number of sacks harvested after 6 days -/
theorem total_harvest : daily_harvest * harvest_days = 498 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_l3178_317886


namespace NUMINAMATH_CALUDE_three_inequality_propositions_l3178_317840

theorem three_inequality_propositions (a b c d : ℝ) :
  (∃ (f g h : Prop),
    (f = (a * b > 0)) ∧
    (g = (c / a > d / b)) ∧
    (h = (b * c > a * d)) ∧
    ((f ∧ g → h) ∧ (f ∧ h → g) ∧ (g ∧ h → f)) ∧
    (∀ (p q r : Prop),
      ((p = f ∨ p = g ∨ p = h) ∧
       (q = f ∨ q = g ∨ q = h) ∧
       (r = f ∨ r = g ∨ r = h) ∧
       (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r) ∧
       (p ∧ q → r)) →
      ((p = f ∧ q = g ∧ r = h) ∨
       (p = f ∧ q = h ∧ r = g) ∨
       (p = g ∧ q = h ∧ r = f)))) :=
by sorry

end NUMINAMATH_CALUDE_three_inequality_propositions_l3178_317840


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3178_317875

/-- Two cars traveling from opposite ends of a highway meet after a certain time. -/
theorem cars_meeting_time
  (highway_length : ℝ)
  (car1_speed : ℝ)
  (car2_speed : ℝ)
  (h1 : highway_length = 175)
  (h2 : car1_speed = 25)
  (h3 : car2_speed = 45) :
  (highway_length / (car1_speed + car2_speed)) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_cars_meeting_time_l3178_317875


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3178_317895

theorem quadratic_root_problem (m : ℝ) :
  (1 : ℝ) ^ 2 - 4 * (1 : ℝ) + m + 1 = 0 →
  m = 2 ∧ ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 - 4 * x + m + 1 = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3178_317895


namespace NUMINAMATH_CALUDE_length_XX₁_l3178_317880

-- Define the triangles and circle
def triangle_DEF (DE DF : ℝ) : Prop := DE = 7 ∧ DF = 3
def inscribed_circle (F₁ : ℝ × ℝ) : Prop := sorry  -- Details of circle inscription

-- Define the second triangle XYZ
def triangle_XYZ (XY XZ : ℝ) (F₁E F₁D : ℝ) : Prop :=
  XY = F₁E ∧ XZ = F₁D

-- Define the angle bisector and point X₁
def angle_bisector (X₁ : ℝ × ℝ) : Prop := sorry  -- Details of angle bisector

-- Main theorem
theorem length_XX₁ (DE DF : ℝ) (F₁ : ℝ × ℝ) (XY XZ : ℝ) (X₁ : ℝ × ℝ) :
  triangle_DEF DE DF →
  inscribed_circle F₁ →
  triangle_XYZ XY XZ (Real.sqrt 10 - 2) (Real.sqrt 10 - 2) →
  angle_bisector X₁ →
  ∃ (XX₁ : ℝ), XX₁ = 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_length_XX₁_l3178_317880


namespace NUMINAMATH_CALUDE_outfits_count_l3178_317882

/-- The number of different outfits that can be made from a given number of shirts, ties, and shoes. -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (shoes : ℕ) : ℕ :=
  shirts * ties * shoes

/-- Theorem stating that the number of outfits is 192 given 8 shirts, 6 ties, and 4 pairs of shoes. -/
theorem outfits_count : number_of_outfits 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3178_317882


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l3178_317817

theorem sum_of_squares_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l3178_317817


namespace NUMINAMATH_CALUDE_ages_cube_sum_l3178_317847

theorem ages_cube_sum (r j m : ℕ) : 
  (5 * r + 2 * j = 3 * m) →
  (3 * m^2 + 2 * j^2 = 5 * r^2) →
  (Nat.gcd r j = 1 ∧ Nat.gcd j m = 1 ∧ Nat.gcd r m = 1) →
  r^3 + j^3 + m^3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ages_cube_sum_l3178_317847


namespace NUMINAMATH_CALUDE_divisors_sum_product_l3178_317857

theorem divisors_sum_product (n a b : ℕ) : 
  n ≥ 1 → 
  a > 0 → 
  b > 0 → 
  n % a = 0 → 
  n % b = 0 → 
  a + b + a * b = n → 
  a = b := by
sorry

end NUMINAMATH_CALUDE_divisors_sum_product_l3178_317857


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l3178_317821

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedronVolume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 12

/-- Theorem: The volume ratio of two regular tetrahedrons with edge lengths a and 2a is 1:8 -/
theorem tetrahedron_volume_ratio (a : ℝ) (h : a > 0) :
  tetrahedronVolume (2 * a) / tetrahedronVolume a = 8 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l3178_317821


namespace NUMINAMATH_CALUDE_integer_pair_condition_l3178_317841

theorem integer_pair_condition (m n : ℕ+) :
  (∃ k : ℤ, (3 * n.val ^ 2 : ℚ) / m.val = k) ∧
  (∃ l : ℕ, (n.val ^ 2 + m.val : ℕ) = l ^ 2) →
  ∃ a : ℕ+, n = a ∧ m = 3 * a ^ 2 := by
sorry

end NUMINAMATH_CALUDE_integer_pair_condition_l3178_317841


namespace NUMINAMATH_CALUDE_gcf_of_180_252_315_l3178_317858

theorem gcf_of_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_252_315_l3178_317858


namespace NUMINAMATH_CALUDE_nell_gave_jeff_cards_nell_gave_jeff_cards_proof_l3178_317898

/-- Given that Nell initially had 304 baseball cards and now has 276 cards left,
    prove that she gave 28 cards to Jeff. -/
theorem nell_gave_jeff_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards remaining_cards cards_given =>
    initial_cards = 304 →
    remaining_cards = 276 →
    cards_given = initial_cards - remaining_cards →
    cards_given = 28

/-- Proof of the theorem -/
theorem nell_gave_jeff_cards_proof : nell_gave_jeff_cards 304 276 28 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_jeff_cards_nell_gave_jeff_cards_proof_l3178_317898


namespace NUMINAMATH_CALUDE_product_fourth_minus_seven_l3178_317838

theorem product_fourth_minus_seven (a b c d : ℕ) (h₁ : a = 5) (h₂ : b = 9) (h₃ : c = 4) (h₄ : d = 7) :
  (a * b * c : ℚ) / 4 - d = 38 := by
  sorry

end NUMINAMATH_CALUDE_product_fourth_minus_seven_l3178_317838


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3178_317812

theorem sqrt_inequality (x : ℝ) (h : x ≥ -3) :
  Real.sqrt (x + 5) - Real.sqrt (x + 3) > Real.sqrt (x + 6) - Real.sqrt (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3178_317812


namespace NUMINAMATH_CALUDE_nut_weight_l3178_317823

/-- A proof that determines the weight of a nut attached to a scale -/
theorem nut_weight (wL wS : ℝ) (h1 : wL + 20 = 300) (h2 : wS + 20 = 200) (h3 : wL + wS + 20 = 480) : 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_nut_weight_l3178_317823


namespace NUMINAMATH_CALUDE_triangle_bd_length_l3178_317834

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point D on AB
def D (t : Triangle) : ℝ × ℝ := sorry

-- State the theorem
theorem triangle_bd_length (t : Triangle) :
  -- Conditions
  (dist t.A t.C = 7) →
  (dist t.B t.C = 7) →
  (dist t.A (D t) = 8) →
  (dist t.C (D t) = 3) →
  -- Conclusion
  (dist t.B (D t) = 5) := by
  sorry

where
  dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

end NUMINAMATH_CALUDE_triangle_bd_length_l3178_317834


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3178_317807

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 11) (h3 : θ = 150 * π / 180) :
  c^2 = a^2 + b^2 - 2*a*b*Real.cos θ → c = Real.sqrt (202 + 99 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3178_317807


namespace NUMINAMATH_CALUDE_puppies_per_cage_l3178_317804

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 18) 
  (h2 : sold_puppies = 3) 
  (h3 : num_cages = 3) 
  : (initial_puppies - sold_puppies) / num_cages = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l3178_317804


namespace NUMINAMATH_CALUDE_melon_count_l3178_317890

/-- Given the number of watermelons and apples, calculate the number of melons -/
theorem melon_count (watermelons apples : ℕ) (h1 : watermelons = 3) (h2 : apples = 7) :
  2 * (watermelons + apples) = 20 := by
  sorry

end NUMINAMATH_CALUDE_melon_count_l3178_317890


namespace NUMINAMATH_CALUDE_expensive_handcuffs_time_l3178_317870

/-- The time it takes to pick the lock on an expensive pair of handcuffs -/
def time_expensive : ℝ := 8

/-- The time it takes to pick the lock on a cheap pair of handcuffs -/
def time_cheap : ℝ := 6

/-- The number of friends to rescue -/
def num_friends : ℕ := 3

/-- The total time it takes to free all friends -/
def total_time : ℝ := 42

theorem expensive_handcuffs_time :
  time_expensive = (total_time - num_friends * time_cheap) / num_friends := by
  sorry

end NUMINAMATH_CALUDE_expensive_handcuffs_time_l3178_317870


namespace NUMINAMATH_CALUDE_book_pages_proof_l3178_317893

theorem book_pages_proof (x : ℝ) : 
  let day1_remaining := x - (x / 6 + 10)
  let day2_remaining := day1_remaining - (day1_remaining / 3 + 20)
  let day3_remaining := day2_remaining - (day2_remaining / 2 + 25)
  day3_remaining = 120 → x = 552 := by
sorry

end NUMINAMATH_CALUDE_book_pages_proof_l3178_317893


namespace NUMINAMATH_CALUDE_five_digit_twice_divisible_by_11_l3178_317832

theorem five_digit_twice_divisible_by_11 (a : ℕ) (h : 10000 ≤ a ∧ a < 100000) :
  ∃ k : ℕ, 100001 * a = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_five_digit_twice_divisible_by_11_l3178_317832


namespace NUMINAMATH_CALUDE_fund_raising_ratio_l3178_317826

def fund_raising (goal : ℕ) (ken_collection : ℕ) (excess : ℕ) : Prop :=
  ∃ (mary_collection scott_collection : ℕ),
    mary_collection = 5 * ken_collection ∧
    ∃ (k : ℕ), mary_collection = k * scott_collection ∧
    mary_collection + scott_collection + ken_collection = goal + excess ∧
    mary_collection / scott_collection = 3

theorem fund_raising_ratio :
  fund_raising 4000 600 600 :=
sorry

end NUMINAMATH_CALUDE_fund_raising_ratio_l3178_317826


namespace NUMINAMATH_CALUDE_border_mass_of_28_coin_triangle_l3178_317824

/-- Represents a triangular arrangement of coins -/
structure CoinTriangle where
  total_coins : ℕ
  border_coins : ℕ
  trio_mass : ℝ

/-- The mass of all border coins in a CoinTriangle -/
def border_mass (ct : CoinTriangle) : ℝ := sorry

/-- Theorem stating the mass of border coins in the specific arrangement -/
theorem border_mass_of_28_coin_triangle (ct : CoinTriangle) 
  (h1 : ct.total_coins = 28)
  (h2 : ct.border_coins = 18)
  (h3 : ct.trio_mass = 10) :
  border_mass ct = 60 := by sorry

end NUMINAMATH_CALUDE_border_mass_of_28_coin_triangle_l3178_317824


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l3178_317813

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The plane xOz in three-dimensional space -/
def PlaneXOZ : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Symmetry with respect to the plane xOz -/
def symmetricXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetric_point_xoz :
  let A : Point3D := ⟨-3, 2, -4⟩
  symmetricXOZ A = ⟨-3, -2, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l3178_317813


namespace NUMINAMATH_CALUDE_integer_list_mean_mode_l3178_317856

theorem integer_list_mean_mode (y : ℕ) : 
  y > 0 ∧ y ≤ 150 →
  let l := [45, 76, 123, y, y, y]
  (l.sum / l.length : ℚ) = 2 * y →
  y = 27 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_mode_l3178_317856


namespace NUMINAMATH_CALUDE_circle_center_coordinate_product_l3178_317850

/-- Given a circle with equation x^2 + y^2 = 6x + 10y - 14, 
    the product of its center coordinates is 15 -/
theorem circle_center_coordinate_product : 
  ∀ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 = 6*x + 10*y - 14 → (x - h)^2 + (y - k)^2 = 20) → 
  h * k = 15 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_product_l3178_317850


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_1260_l3178_317827

theorem sum_of_extreme_prime_factors_1260 : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ 1260 ∧ q ∣ 1260 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 1260 → p ≤ r ∧ r ≤ q) ∧
  p + q = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_1260_l3178_317827


namespace NUMINAMATH_CALUDE_cookies_distribution_l3178_317877

/-- Represents the number of cookies the oldest son gets after school -/
def oldest_son_cookies : ℕ := 4

/-- Represents the number of cookies the youngest son gets after school -/
def youngest_son_cookies : ℕ := 2

/-- Represents the total number of cookies in a box -/
def cookies_in_box : ℕ := 54

/-- Represents the number of days the box lasts -/
def days_box_lasts : ℕ := 9

theorem cookies_distribution :
  oldest_son_cookies * days_box_lasts + youngest_son_cookies * days_box_lasts = cookies_in_box :=
by sorry

end NUMINAMATH_CALUDE_cookies_distribution_l3178_317877


namespace NUMINAMATH_CALUDE_characterization_of_good_numbers_l3178_317833

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_good_numbers_l3178_317833


namespace NUMINAMATH_CALUDE_probability_of_event_B_l3178_317843

theorem probability_of_event_B 
  (P_A : ℝ) 
  (P_A_and_B : ℝ) 
  (P_A_or_B : ℝ) 
  (h1 : P_A = 0.4)
  (h2 : P_A_and_B = 0.25)
  (h3 : P_A_or_B = 0.6) :
  P_A + (P_A_or_B - P_A + P_A_and_B) - P_A_and_B = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_event_B_l3178_317843


namespace NUMINAMATH_CALUDE_max_roses_is_317_l3178_317853

/-- Represents the price of roses in cents to avoid floating-point issues -/
def individual_price : ℕ := 530
def dozen_price : ℕ := 3600
def two_dozen_price : ℕ := 5000
def budget : ℕ := 68000

/-- Calculates the maximum number of roses that can be purchased with the given budget -/
def max_roses : ℕ :=
  let two_dozen_sets := budget / two_dozen_price
  let remaining_budget := budget - two_dozen_sets * two_dozen_price
  let individual_roses := remaining_budget / individual_price
  two_dozen_sets * 24 + individual_roses

theorem max_roses_is_317 : max_roses = 317 := by sorry

end NUMINAMATH_CALUDE_max_roses_is_317_l3178_317853


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l3178_317846

theorem cube_root_of_negative_eight :
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l3178_317846


namespace NUMINAMATH_CALUDE_cliffs_rock_collection_l3178_317849

/-- The number of rocks in Cliff's collection -/
def total_rocks (igneous sedimentary metamorphic comet : ℕ) : ℕ :=
  igneous + sedimentary + metamorphic + comet

theorem cliffs_rock_collection :
  ∀ (igneous sedimentary metamorphic comet : ℕ),
    igneous = sedimentary / 2 →
    metamorphic = igneous / 3 →
    comet = 2 * metamorphic →
    igneous / 4 = 15 →
    comet / 2 = 20 →
    total_rocks igneous sedimentary metamorphic comet = 240 := by
  sorry

end NUMINAMATH_CALUDE_cliffs_rock_collection_l3178_317849


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l3178_317839

theorem scooter_gain_percent (initial_cost repair1 repair2 repair3 selling_price : ℚ) : 
  initial_cost = 800 →
  repair1 = 150 →
  repair2 = 75 →
  repair3 = 225 →
  selling_price = 1600 →
  let total_cost := initial_cost + repair1 + repair2 + repair3
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 28 := by
sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l3178_317839


namespace NUMINAMATH_CALUDE_cube_sum_gt_mixed_product_l3178_317811

theorem cube_sum_gt_mixed_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_gt_mixed_product_l3178_317811


namespace NUMINAMATH_CALUDE_inverse_f_at_135_l3178_317896

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^3 + 5

-- State the theorem
theorem inverse_f_at_135 :
  ∃ (y : ℝ), f y = 135 ∧ y = (26 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_135_l3178_317896
