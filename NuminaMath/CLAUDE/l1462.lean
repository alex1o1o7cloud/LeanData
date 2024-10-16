import Mathlib

namespace NUMINAMATH_CALUDE_keith_added_scissors_l1462_146267

/-- The number of scissors Keith added to the drawer -/
def scissors_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Keith added 22 scissors to the drawer -/
theorem keith_added_scissors : scissors_added 54 76 = 22 := by
  sorry

end NUMINAMATH_CALUDE_keith_added_scissors_l1462_146267


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1462_146265

/-- The total surface area of a rectangular solid -/
def total_surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 5 meters, width 4 meters, and depth 1 meter is 58 square meters -/
theorem rectangular_solid_surface_area :
  total_surface_area 5 4 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1462_146265


namespace NUMINAMATH_CALUDE_greatest_multiple_of_12_with_unique_digits_M_mod_1000_l1462_146289

/-- A function that checks if a natural number has all unique digits -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 12 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_12_with_unique_digits : 
  M % 12 = 0 ∧ 
  has_unique_digits M ∧ 
  ∀ k, k % 12 = 0 → has_unique_digits k → k ≤ M :=
sorry

theorem M_mod_1000 : M % 1000 = 320 := sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_12_with_unique_digits_M_mod_1000_l1462_146289


namespace NUMINAMATH_CALUDE_mrs_hilt_walking_distance_l1462_146270

/-- The total distance walked to and from a water fountain -/
def total_distance (distance_to_fountain : ℕ) (num_trips : ℕ) : ℕ :=
  2 * distance_to_fountain * num_trips

/-- Theorem: Mrs. Hilt walks 240 feet given the problem conditions -/
theorem mrs_hilt_walking_distance :
  total_distance 30 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_walking_distance_l1462_146270


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l1462_146252

/-- Given an initial price and a discount amount, the discounted price is the difference between the initial price and the discount. -/
theorem discounted_price_calculation (initial_price discount : ℝ) :
  initial_price = 475 →
  discount = 276 →
  initial_price - discount = 199 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l1462_146252


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l1462_146297

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l1462_146297


namespace NUMINAMATH_CALUDE_f_x₁_gt_f_x₂_l1462_146224

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f(x+1) is an even function
axiom f_even : ∀ x, f (x + 1) = f (-x + 1)

-- (x-1)f'(x) < 0
axiom f_decreasing : ∀ x, (x - 1) * f' x < 0

-- x₁ < x₂
variable (x₁ x₂ : ℝ)
axiom x₁_lt_x₂ : x₁ < x₂

-- x₁ + x₂ > 2
axiom sum_gt_two : x₁ + x₂ > 2

-- The theorem to prove
theorem f_x₁_gt_f_x₂ : f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_f_x₁_gt_f_x₂_l1462_146224


namespace NUMINAMATH_CALUDE_spider_human_leg_relationship_l1462_146237

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a human has -/
def human_legs : ℕ := sorry

/-- The relationship between spider legs and human legs -/
def leg_relationship : ℕ := sorry

theorem spider_human_leg_relationship :
  spider_legs = leg_relationship * human_legs :=
by sorry

end NUMINAMATH_CALUDE_spider_human_leg_relationship_l1462_146237


namespace NUMINAMATH_CALUDE_same_color_probability_l1462_146283

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 30

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 8

/-- Represents the number of orange sides on each die -/
def orangeSides : ℕ := 9

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 10

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 2

/-- Represents the number of sparkly sides on each die -/
def sparklySides : ℕ := 1

/-- Theorem stating that the probability of rolling the same color on both dice is 25/90 -/
theorem same_color_probability :
  (purpleSides^2 + orangeSides^2 + greenSides^2 + blueSides^2 + sparklySides^2) / totalSides^2 = 25 / 90 := by
  sorry


end NUMINAMATH_CALUDE_same_color_probability_l1462_146283


namespace NUMINAMATH_CALUDE_circle_radius_is_five_thirds_l1462_146256

/-- An isosceles triangle with a circle constructed on its base -/
structure IsoscelesTriangleWithCircle where
  /-- The base length of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the circle constructed on the base -/
  radius : ℝ

/-- The radius of the circle in an isosceles triangle with given base and height -/
def circleRadius (triangle : IsoscelesTriangleWithCircle) : ℝ :=
  triangle.radius

/-- Theorem: The radius of the circle is 5/3 given the specified conditions -/
theorem circle_radius_is_five_thirds (triangle : IsoscelesTriangleWithCircle)
    (h1 : triangle.base = 8)
    (h2 : triangle.height = 3) :
    circleRadius triangle = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_thirds_l1462_146256


namespace NUMINAMATH_CALUDE_power_equation_solutions_l1462_146278

theorem power_equation_solutions :
  ∀ x y : ℕ, 2^x = 3^y + 5 ↔ (x = 3 ∧ y = 1) ∨ (x = 5 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_power_equation_solutions_l1462_146278


namespace NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l1462_146205

/-- Given a quadratic function f(x) = x^2 + (1-k)x - k, if f has a root in the interval (2, 3), 
    then k is in the open interval (2, 3) -/
theorem root_in_interval_implies_k_range (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + (1-k)*x - k
  (∃ x ∈ Set.Ioo 2 3, f x = 0) → k ∈ Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l1462_146205


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1462_146258

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 - 2*m*x + 4 = 0 ∧ y^2 - 2*m*y + 4 = 0 ∧ x ≠ y ∧ x > 2 ∧ y < 2) → 
  m > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1462_146258


namespace NUMINAMATH_CALUDE_new_encoding_of_original_message_l1462_146299

/-- Represents the encoding of a character in the old system -/
def OldEncoding : Char → String
| 'A' => "011"
| 'B' => "011"
| 'C' => "0"
| _ => ""

/-- Represents the encoding of a character in the new system -/
def NewEncoding : Char → String
| 'A' => "21"
| 'B' => "122"
| 'C' => "1"
| _ => ""

/-- Decodes a string from the old encoding system -/
def decodeOld (s : String) : String := sorry

/-- Encodes a string using the new encoding system -/
def encodeNew (s : String) : String := sorry

/-- The original message in the old encoding -/
def originalMessage : String := "011011010011"

/-- Theorem stating that the new encoding of the original message is "211221121" -/
theorem new_encoding_of_original_message :
  encodeNew (decodeOld originalMessage) = "211221121" := by sorry

end NUMINAMATH_CALUDE_new_encoding_of_original_message_l1462_146299


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1462_146238

theorem functional_equation_solution (f : ℕ → ℝ) 
  (h : ∀ x y : ℕ, f (x + y) + f (x - y) = f (3 * x)) : 
  ∀ x : ℕ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1462_146238


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l1462_146223

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a7 (a : ℕ → ℚ) :
  GeometricSequence a →
  a 5 = 1/2 →
  4 * a 3 + a 7 = 2 →
  a 7 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l1462_146223


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1462_146218

-- Define the function f(x) = |x+2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≤ f y} = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1462_146218


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_three_digit_product_l1462_146206

theorem smallest_multiplier_for_three_digit_product : 
  (∀ k : ℕ, k < 4 → 27 * k < 100) ∧ (27 * 4 ≥ 100 ∧ 27 * 4 < 1000) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_three_digit_product_l1462_146206


namespace NUMINAMATH_CALUDE_impossibility_of_2023_linked_triangles_l1462_146250

-- Define the space and points
def Space := Type
def Point : Type := Unit

-- Define the colors of points
inductive Color
| Yellow
| Red

-- Define the properties of the space
structure SpaceProperties (s : Space) :=
  (total_points : Nat)
  (yellow_points : Nat)
  (red_points : Nat)
  (no_four_coplanar : Prop)
  (total_points_eq : total_points = yellow_points + red_points)

-- Define a triangle
structure Triangle (s : Space) :=
  (vertices : Fin 3 → Point)

-- Define the linking relation between triangles
def isLinked (s : Space) (yellow : Triangle s) (red : Triangle s) : Prop := sorry

-- Define the count of linked triangles
def linkedTrianglesCount (s : Space) (props : SpaceProperties s) : Nat := sorry

-- The main theorem
theorem impossibility_of_2023_linked_triangles (s : Space) 
  (props : SpaceProperties s) 
  (h1 : props.total_points = 43)
  (h2 : props.yellow_points = 3)
  (h3 : props.red_points = 40)
  (h4 : props.no_four_coplanar) :
  linkedTrianglesCount s props ≠ 2023 := by sorry

end NUMINAMATH_CALUDE_impossibility_of_2023_linked_triangles_l1462_146250


namespace NUMINAMATH_CALUDE_count_numbers_theorem_l1462_146202

/-- The count of positive integers less than 50000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  let one_digit := 45  -- 5 * 9
  let two_digits_without_zero := 1872  -- 36 * 52
  let two_digits_with_zero := 234  -- 9 * 26
  let three_digits_with_zero := 900  -- 36 * 25
  let three_digits_without_zero := 4452  -- 84 * 53
  one_digit + two_digits_without_zero + two_digits_with_zero + three_digits_with_zero + three_digits_without_zero

/-- The theorem stating that the count of positive integers less than 50000 
    with at most three different digits is 7503 -/
theorem count_numbers_theorem : count_numbers_with_at_most_three_digits = 7503 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_theorem_l1462_146202


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1462_146279

/-- A quadratic function f(x) = mx² - 2mx + 1 with m > 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

/-- Theorem stating the relationship between f(6), f(-1), and f(5/2) -/
theorem quadratic_inequality (m : ℝ) (h : m > 0) : 
  f m 6 > f m (-1) ∧ f m (-1) > f m (5/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1462_146279


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1462_146251

theorem maintenance_check_increase (original_time new_time : ℝ) 
  (h1 : original_time = 25)
  (h2 : new_time = 30) :
  (new_time - original_time) / original_time * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1462_146251


namespace NUMINAMATH_CALUDE_ab_minus_bc_plus_ac_equals_seven_l1462_146290

theorem ab_minus_bc_plus_ac_equals_seven 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 14) 
  (h2 : a = b + c) : 
  a*b - b*c + a*c = 7 := by
sorry

end NUMINAMATH_CALUDE_ab_minus_bc_plus_ac_equals_seven_l1462_146290


namespace NUMINAMATH_CALUDE_balloons_given_to_fred_l1462_146296

/-- Given that Tom initially had 30 balloons and now has 14 balloons,
    prove that he gave 16 balloons to Fred. -/
theorem balloons_given_to_fred 
  (initial_balloons : ℕ) 
  (remaining_balloons : ℕ) 
  (h1 : initial_balloons = 30) 
  (h2 : remaining_balloons = 14) : 
  initial_balloons - remaining_balloons = 16 :=
by sorry

end NUMINAMATH_CALUDE_balloons_given_to_fred_l1462_146296


namespace NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l1462_146215

theorem temperature_difference (B : ℝ) (N : ℝ) : 
  (∃ A : ℝ, A = B + N) → 
  (|((B + N) - 4) - (B + 5)| = 1) →
  (N = 10 ∨ N = 8) :=
by sorry

theorem product_of_N_values :
  ∃ N₁ N₂ : ℝ, 
    (∃ B : ℝ, (∃ A : ℝ, A = B + N₁) ∧ |((B + N₁) - 4) - (B + 5)| = 1) ∧
    (∃ B : ℝ, (∃ A : ℝ, A = B + N₂) ∧ |((B + N₂) - 4) - (B + 5)| = 1) ∧
    N₁ * N₂ = 80 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l1462_146215


namespace NUMINAMATH_CALUDE_boa_constrictors_count_l1462_146271

/-- The number of boa constrictors in the park -/
def num_boa : ℕ := sorry

/-- The number of pythons in the park -/
def num_python : ℕ := sorry

/-- The number of rattlesnakes in the park -/
def num_rattlesnake : ℕ := 40

/-- The total number of snakes in the park -/
def total_snakes : ℕ := 200

theorem boa_constrictors_count :
  (num_boa + num_python + num_rattlesnake = total_snakes) →
  (num_python = 3 * num_boa) →
  (num_boa = 40) :=
by sorry

end NUMINAMATH_CALUDE_boa_constrictors_count_l1462_146271


namespace NUMINAMATH_CALUDE_vector_magnitude_l1462_146209

theorem vector_magnitude (a b : ℝ × ℝ) :
  a = (Real.cos (10 * π / 180), Real.sin (10 * π / 180)) →
  b = (Real.cos (70 * π / 180), Real.sin (70 * π / 180)) →
  ‖a - 2 • b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1462_146209


namespace NUMINAMATH_CALUDE_g_properties_l1462_146212

-- Define the function g
def g : ℝ → ℝ := fun x ↦ -x

-- Theorem stating that g is an odd function and monotonically decreasing
theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ 
  (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry


end NUMINAMATH_CALUDE_g_properties_l1462_146212


namespace NUMINAMATH_CALUDE_total_amount_is_2150_6_l1462_146244

/-- Calculates the total amount paid for fruits with discounts and taxes -/
def total_amount_paid (
  grapes_kg : ℝ)   (grapes_price : ℝ)
  (mangoes_kg : ℝ)  (mangoes_price : ℝ)
  (apples_kg : ℝ)   (apples_price : ℝ)
  (strawberries_kg : ℝ) (strawberries_price : ℝ)
  (oranges_kg : ℝ)  (oranges_price : ℝ)
  (kiwis_kg : ℝ)    (kiwis_price : ℝ)
  (grapes_apples_discount : ℝ)
  (oranges_kiwis_discount : ℝ)
  (mangoes_strawberries_tax : ℝ) : ℝ :=
  let grapes_total := grapes_kg * grapes_price
  let mangoes_total := mangoes_kg * mangoes_price
  let apples_total := apples_kg * apples_price
  let strawberries_total := strawberries_kg * strawberries_price
  let oranges_total := oranges_kg * oranges_price
  let kiwis_total := kiwis_kg * kiwis_price
  
  let total_before_discounts_taxes := grapes_total + mangoes_total + apples_total + 
                                      strawberries_total + oranges_total + kiwis_total
  
  let grapes_apples_discount_amount := (grapes_total + apples_total) * grapes_apples_discount
  let oranges_kiwis_discount_amount := (oranges_total + kiwis_total) * oranges_kiwis_discount
  let mangoes_strawberries_tax_amount := (mangoes_total + strawberries_total) * mangoes_strawberries_tax
  
  total_before_discounts_taxes - grapes_apples_discount_amount - oranges_kiwis_discount_amount + mangoes_strawberries_tax_amount

/-- Theorem stating that the total amount paid for fruits is 2150.6 -/
theorem total_amount_is_2150_6 :
  total_amount_paid 8 70 9 45 5 30 3 100 10 40 6 60 0.1 0.05 0.12 = 2150.6 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_2150_6_l1462_146244


namespace NUMINAMATH_CALUDE_min_value_expression_l1462_146269

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 ∧ y > 1 ∧ x + y = 2 → 4/x + 1/(y-1) ≥ 4/a + 1/(b-1)) →
  4/a + 1/(b-1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1462_146269


namespace NUMINAMATH_CALUDE_alien_running_time_l1462_146245

/-- The time taken by an Alien to run a certain distance when chasing different animals -/
theorem alien_running_time 
  (speed_rabbit : ℝ) 
  (speed_frog : ℝ) 
  (time_difference : ℝ) 
  (h1 : speed_rabbit = 15) 
  (h2 : speed_frog = 10) 
  (h3 : time_difference = 0.5) :
  ∃ (distance : ℝ) (time_rabbit : ℝ),
    distance = speed_rabbit * time_rabbit ∧
    distance = speed_frog * (time_rabbit + time_difference) ∧
    time_rabbit + time_difference = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_alien_running_time_l1462_146245


namespace NUMINAMATH_CALUDE_evaluate_expression_l1462_146288

theorem evaluate_expression : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1462_146288


namespace NUMINAMATH_CALUDE_inequality_solution_l1462_146287

theorem inequality_solution (x : ℝ) : 
  |((3 * x - 2) / (x^2 - x - 2))| > 3 ↔ 
  (x > -1 ∧ x < -2/3) ∨ (x > 1/3 ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1462_146287


namespace NUMINAMATH_CALUDE_largest_nice_sequence_l1462_146210

/-- A sequence is nice if it satisfies the given conditions -/
def IsNice (a : ℕ → ℝ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ 
  a 0 + a 1 = -1 / n ∧ 
  ∀ k : ℕ, k ≥ 1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) + a (k+1)

/-- The largest N for which a nice sequence of length N+1 exists is equal to n -/
theorem largest_nice_sequence (n : ℕ) : 
  n ≥ 1 → 
  (∃ (N : ℕ) (a : ℕ → ℝ), IsNice a n ∧ N = n) ∧ 
  (∀ (M : ℕ) (a : ℕ → ℝ), M > n → ¬ IsNice a n) :=
sorry

end NUMINAMATH_CALUDE_largest_nice_sequence_l1462_146210


namespace NUMINAMATH_CALUDE_max_integer_solutions_l1462_146233

/-- A quadratic function f(x) = ax^2 + bx + c where a > 100 -/
def QuadraticFunction (a b c : ℝ) (h : a > 100) := fun (x : ℤ) => a * x^2 + b * x + c

/-- The maximum number of integer solutions for |f(x)| ≤ 50 is at most 2 -/
theorem max_integer_solutions (a b c : ℝ) (h : a > 100) :
  ∃ (n : ℕ), n ≤ 2 ∧ 
  ∀ (S : Finset ℤ), (∀ x ∈ S, |QuadraticFunction a b c h x| ≤ 50) → S.card ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_integer_solutions_l1462_146233


namespace NUMINAMATH_CALUDE_estimate_husk_amount_l1462_146276

/-- Estimate the amount of husk in a batch of rice given a sample -/
theorem estimate_husk_amount (total_rice : ℕ) (sample_size : ℕ) (husk_in_sample : ℕ) 
  (h1 : total_rice = 1520)
  (h2 : sample_size = 144)
  (h3 : husk_in_sample = 18) :
  (total_rice : ℚ) * (husk_in_sample : ℚ) / (sample_size : ℚ) = 190 := by
  sorry

#check estimate_husk_amount

end NUMINAMATH_CALUDE_estimate_husk_amount_l1462_146276


namespace NUMINAMATH_CALUDE_base_16_digits_for_5_digit_base_4_l1462_146242

theorem base_16_digits_for_5_digit_base_4 (n : ℕ) (h : 256 ≤ n ∧ n ≤ 1023) :
  (Nat.log 16 n).succ = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_16_digits_for_5_digit_base_4_l1462_146242


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l1462_146227

/-- A regular triangular prism inscribed in a sphere -/
structure RegularTriangularPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Length of AD -/
  AD : ℝ
  /-- Assertion that CD is a diameter of the sphere -/
  CD_is_diameter : True

/-- Volume of a regular triangular prism -/
def volume (prism : RegularTriangularPrism) : ℝ := sorry

/-- Theorem stating the volume of the specific prism -/
theorem volume_of_specific_prism :
  ∀ (prism : RegularTriangularPrism),
    prism.R = 3 ∧ prism.AD = 2 * Real.sqrt 6 →
    volume prism = 6 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l1462_146227


namespace NUMINAMATH_CALUDE_hash_seven_three_l1462_146255

-- Define the # operation
def hash (a b : ℤ) : ℚ := 2 * a + a / b + 3

-- Theorem statement
theorem hash_seven_three : hash 7 3 = 19 + 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hash_seven_three_l1462_146255


namespace NUMINAMATH_CALUDE_complex_norm_squared_l1462_146228

theorem complex_norm_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a (-b)
  Complex.normSq z = a^2 + b^2 := by sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l1462_146228


namespace NUMINAMATH_CALUDE_bird_triangle_theorem_l1462_146266

/-- A bird's position on a regular n-gon --/
structure BirdPosition (n : ℕ) where
  vertex : Fin n

/-- The type of a triangle --/
inductive TriangleType
  | Acute
  | Obtuse
  | RightAngled

/-- Determine the type of a triangle formed by three birds on a regular n-gon --/
def triangleType (n : ℕ) (a b c : BirdPosition n) : TriangleType := sorry

/-- A permutation of birds --/
def BirdPermutation (n : ℕ) := Fin n → Fin n

/-- The main theorem --/
theorem bird_triangle_theorem (n : ℕ) (h : n ≥ 3 ∧ n ≠ 5) :
  ∀ (perm : BirdPermutation n),
  ∃ (a b c : Fin n),
    triangleType n ⟨a⟩ ⟨b⟩ ⟨c⟩ = triangleType n ⟨perm a⟩ ⟨perm b⟩ ⟨perm c⟩ :=
sorry

end NUMINAMATH_CALUDE_bird_triangle_theorem_l1462_146266


namespace NUMINAMATH_CALUDE_largest_whole_number_times_eleven_less_than_150_l1462_146282

theorem largest_whole_number_times_eleven_less_than_150 :
  (∀ x : ℕ, 11 * x < 150 → x ≤ 13) ∧ (11 * 13 < 150) := by sorry

end NUMINAMATH_CALUDE_largest_whole_number_times_eleven_less_than_150_l1462_146282


namespace NUMINAMATH_CALUDE_tom_batteries_in_toys_l1462_146249

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys (total batteries_in_flashlights batteries_in_controllers : ℕ) : ℕ :=
  total - (batteries_in_flashlights + batteries_in_controllers)

/-- Theorem stating that Tom used 15 batteries in his toys -/
theorem tom_batteries_in_toys :
  batteries_in_toys 19 2 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tom_batteries_in_toys_l1462_146249


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l1462_146275

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_of_roots_specific :
  let f : ℝ → ℝ := λ x => 5*x^2 + 2*x - 15
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁^2 + x₂^2 = 154/25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l1462_146275


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1462_146211

/-- Given a rectangular plot where the length is thrice the breadth 
    and the area is 972 sq m, prove that the breadth is 18 meters. -/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 972 →
  breadth = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1462_146211


namespace NUMINAMATH_CALUDE_slope_of_solutions_l1462_146231

theorem slope_of_solutions (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : 3 / x₁ + 4 / y₁ = 0) (h₃ : 3 / x₂ + 4 / y₂ = 0) : 
  (y₂ - y₁) / (x₂ - x₁) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l1462_146231


namespace NUMINAMATH_CALUDE_principal_is_15000_l1462_146232

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest : ℚ) * 100 / (rate * time)

/-- Theorem: Given the specified conditions, the principal sum is 15000 -/
theorem principal_is_15000 :
  let simple_interest : ℕ := 2700
  let rate : ℚ := 6 / 100
  let time : ℕ := 3
  calculate_principal simple_interest rate time = 15000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_15000_l1462_146232


namespace NUMINAMATH_CALUDE_function_max_at_pi_third_l1462_146219

/-- The function f(x) reaches its maximum value at x₀ = π/3 --/
theorem function_max_at_pi_third (x : ℝ) : 
  let f := λ x : ℝ => (Real.sqrt 3 / 2) * Real.sin x + (1 / 2) * Real.cos x
  ∃ (x₀ : ℝ), x₀ = π/3 ∧ ∀ y, f y ≤ f x₀ :=
by sorry

end NUMINAMATH_CALUDE_function_max_at_pi_third_l1462_146219


namespace NUMINAMATH_CALUDE_final_x_value_l1462_146284

/-- Represents the state of the program at each iteration -/
structure ProgramState :=
  (x : ℕ)
  (S : ℕ)

/-- The initial state of the program -/
def initial_state : ProgramState :=
  { x := 3, S := 0 }

/-- Updates the program state for one iteration -/
def update_state (state : ProgramState) : ProgramState :=
  { x := state.x + 2, S := state.S + (state.x + 2) }

/-- Predicate to check if the loop should continue -/
def continue_loop (state : ProgramState) : Prop :=
  state.S < 10000

/-- The final state of the program after all iterations -/
noncomputable def final_state : ProgramState :=
  sorry  -- The actual computation of the final state

/-- Theorem stating that the final x value is 201 -/
theorem final_x_value :
  (final_state.x = 201) ∧ (final_state.S ≥ 10000) ∧ 
  (update_state final_state).S > 10000 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l1462_146284


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1462_146263

-- Define the repeating decimal 0.333...
def repeating_3 : ℚ := 1 / 3

-- Define the repeating decimal 0.0202...
def repeating_02 : ℚ := 2 / 99

-- Theorem statement
theorem sum_of_repeating_decimals : repeating_3 + repeating_02 = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1462_146263


namespace NUMINAMATH_CALUDE_min_value_product_l1462_146281

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 1 →
    (a + 1) * (b + 1) * (c + 1) ≤ (x + 1) * (y + 1) * (z + 1) ∧
    (a + 1) * (b + 1) * (c + 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1462_146281


namespace NUMINAMATH_CALUDE_bird_count_l1462_146201

theorem bird_count (total_heads : ℕ) (total_legs : ℕ) (bird_legs : ℕ) (mammal_legs : ℕ) (insect_legs : ℕ) :
  total_heads = 300 →
  total_legs = 1112 →
  bird_legs = 2 →
  mammal_legs = 4 →
  insect_legs = 6 →
  ∃ (birds mammals insects : ℕ),
    birds + mammals + insects = total_heads ∧
    birds * bird_legs + mammals * mammal_legs + insects * insect_legs = total_legs ∧
    birds = 122 :=
by sorry

end NUMINAMATH_CALUDE_bird_count_l1462_146201


namespace NUMINAMATH_CALUDE_min_words_for_passing_score_l1462_146285

/-- Represents the German vocabulary exam parameters and conditions -/
structure GermanExam where
  total_words : ℕ := 800
  correct_points : ℚ := 1
  incorrect_penalty : ℚ := 1/4
  target_score_percentage : ℚ := 90/100

/-- Calculates the exam score based on the number of words learned -/
def examScore (exam : GermanExam) (words_learned : ℕ) : ℚ :=
  (words_learned : ℚ) * exam.correct_points - 
  ((exam.total_words - words_learned) : ℚ) * exam.incorrect_penalty

/-- Theorem stating that learning at least 736 words ensures a score of at least 90% -/
theorem min_words_for_passing_score (exam : GermanExam) :
  ∀ words_learned : ℕ, words_learned ≥ 736 →
  examScore exam words_learned ≥ (exam.target_score_percentage * exam.total_words) :=
by sorry

#check min_words_for_passing_score

end NUMINAMATH_CALUDE_min_words_for_passing_score_l1462_146285


namespace NUMINAMATH_CALUDE_bacteria_growth_l1462_146222

theorem bacteria_growth (quadruple_time : ℕ) (total_time : ℕ) (final_count : ℕ) 
  (h1 : quadruple_time = 20)
  (h2 : total_time = 4 * 60)
  (h3 : final_count = 1048576)
  (h4 : (total_time / quadruple_time : ℚ) = 12) :
  ∃ (initial_count : ℚ), 
    initial_count * (4 ^ (total_time / quadruple_time)) = final_count ∧ 
    initial_count = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1462_146222


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1462_146225

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1462_146225


namespace NUMINAMATH_CALUDE_figurine_cost_l1462_146239

/-- The cost of a single figurine given Annie's purchase details -/
theorem figurine_cost (tv_count : ℕ) (tv_price : ℕ) (figurine_count : ℕ) (total_spent : ℕ) : 
  tv_count = 5 → 
  tv_price = 50 → 
  figurine_count = 10 → 
  total_spent = 260 → 
  (total_spent - tv_count * tv_price) / figurine_count = 1 :=
by
  sorry

#check figurine_cost

end NUMINAMATH_CALUDE_figurine_cost_l1462_146239


namespace NUMINAMATH_CALUDE_rectangle_width_l1462_146294

/-- Given a rectangular piece of metal with length 19 cm and perimeter 70 cm, 
    prove that its width is 16 cm. -/
theorem rectangle_width (length perimeter : ℝ) (h1 : length = 19) (h2 : perimeter = 70) :
  let width := (perimeter / 2) - length
  width = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l1462_146294


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l1462_146234

/-- The trajectory of point Q given point P on a circle -/
theorem trajectory_of_Q (m n : ℝ) : 
  m^2 + n^2 = 2 →   -- P is on the circle x^2 + y^2 = 2
  ∃ x y : ℝ,
    x = m + n ∧     -- x-coordinate of Q
    y = 2 * m * n ∧ -- y-coordinate of Q
    y = x^2 - 2 ∧   -- trajectory equation
    -2 ≤ x ∧ x ≤ 2  -- domain constraint
  := by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l1462_146234


namespace NUMINAMATH_CALUDE_game_cost_calculation_l1462_146207

theorem game_cost_calculation (total_earnings : ℕ) (blade_cost : ℕ) (num_games : ℕ) 
  (h1 : total_earnings = 101)
  (h2 : blade_cost = 47)
  (h3 : num_games = 9)
  (h4 : (total_earnings - blade_cost) % num_games = 0) :
  (total_earnings - blade_cost) / num_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_calculation_l1462_146207


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1462_146204

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1462_146204


namespace NUMINAMATH_CALUDE_flu_free_inhabitants_l1462_146291

theorem flu_free_inhabitants (total_population : ℕ) (flu_percentage : ℚ) : 
  total_population = 14000000 →
  flu_percentage = 15 / 10000 →
  (total_population : ℚ) - (flu_percentage * total_population) = 13979000 := by
  sorry

end NUMINAMATH_CALUDE_flu_free_inhabitants_l1462_146291


namespace NUMINAMATH_CALUDE_odd_function_property_l1462_146241

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1462_146241


namespace NUMINAMATH_CALUDE_volleyball_team_score_l1462_146259

/-- Volleyball team scoring problem -/
theorem volleyball_team_score (lizzie_score : ℕ) (team_total : ℕ) : 
  lizzie_score = 4 →
  team_total = 50 →
  17 = team_total - (lizzie_score + (lizzie_score + 3) + 2 * (lizzie_score + (lizzie_score + 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_score_l1462_146259


namespace NUMINAMATH_CALUDE_margaret_age_in_twelve_years_l1462_146280

/-- Given the ages of Brian, Christian, and Margaret, prove Margaret's age in 12 years -/
theorem margaret_age_in_twelve_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 := by
  sorry

end NUMINAMATH_CALUDE_margaret_age_in_twelve_years_l1462_146280


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1462_146217

theorem cubic_equation_solution (t s : ℝ) : t = 8 * s^3 ∧ t = 64 → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1462_146217


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1462_146226

theorem quadratic_minimum (x : ℝ) : 
  (∀ y : ℝ, x^2 - 6*x + 5 ≤ y^2 - 6*y + 5) ↔ x = 3 ∧ x^2 - 6*x + 5 = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1462_146226


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l1462_146203

theorem product_of_repeating_decimal_and_nine (q : ℚ) : 
  (∃ (n : ℕ), q * (100 : ℚ) - q = (45 : ℚ) + n * (100 : ℚ)) → q * 9 = 45 / 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l1462_146203


namespace NUMINAMATH_CALUDE_print_shop_charge_l1462_146260

/-- The charge per color copy at print shop X -/
def charge_x : ℝ := 1.20

/-- The charge per color copy at print shop Y -/
def charge_y : ℝ := 1.70

/-- The number of copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℝ := 20

theorem print_shop_charge : 
  charge_x * num_copies + additional_charge = charge_y * num_copies :=
by sorry

end NUMINAMATH_CALUDE_print_shop_charge_l1462_146260


namespace NUMINAMATH_CALUDE_greater_number_problem_l1462_146243

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 16) : max x y = 33 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1462_146243


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1462_146295

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes x ± 2y = 0 is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let asymptote (x y : ℝ) := x = 2 * y ∨ x = -2 * y
  asymptote x y ∧ x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1462_146295


namespace NUMINAMATH_CALUDE_min_sum_squares_y_coords_l1462_146298

/-- 
Given a line passing through (4, 0) and intersecting the parabola y^2 = 4x at two points,
the minimum value of the sum of the squares of the y-coordinates of these two points is 32.
-/
theorem min_sum_squares_y_coords : 
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
  y₁^2 = 4 * (m * y₁ + 4) →
  y₂^2 = 4 * (m * y₂ + 4) →
  y₁ ≠ y₂ →
  ∀ (z₁ z₂ : ℝ),
  z₁^2 = 4 * (m * z₁ + 4) →
  z₂^2 = 4 * (m * z₂ + 4) →
  z₁ ≠ z₂ →
  y₁^2 + y₂^2 ≤ z₁^2 + z₂^2 →
  y₁^2 + y₂^2 = 32 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_squares_y_coords_l1462_146298


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1462_146253

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_unique :
  ∀ a b c : ℝ,
  (f a b c (-1) = 0) →
  (∀ x : ℝ, x ≤ f a b c x) →
  (∀ x : ℝ, f a b c x ≤ (x^2 + 1) / 2) →
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1462_146253


namespace NUMINAMATH_CALUDE_parallel_condition_theorem_l1462_146216

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition_theorem 
  (a b : Line) (α : Plane) 
  (h_different : a ≠ b) 
  (h_contained : contained_in b α) :
  (∀ x y : Line, ∀ p : Plane, 
    contained_in y p → 
    parallel_lines x y → 
    parallel_line_plane x p) ∧
  (∃ x y : Line, ∃ p : Plane,
    contained_in y p ∧ 
    parallel_line_plane x p ∧ 
    ¬parallel_lines x y) →
  (parallel_line_plane a α → parallel_lines a b) ∧
  ¬(parallel_lines a b → parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_theorem_l1462_146216


namespace NUMINAMATH_CALUDE_same_solution_equations_l1462_146235

theorem same_solution_equations (x : ℝ) (d : ℝ) : 
  (3 * x + 8 = 5) ∧ (d * x - 15 = -7) → d = -8 := by
sorry

end NUMINAMATH_CALUDE_same_solution_equations_l1462_146235


namespace NUMINAMATH_CALUDE_modified_fibonacci_sum_l1462_146213

-- Define the modified Fibonacci sequence
def F : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => F (n + 1) + F n

-- Define the sum of the series
noncomputable def S : ℝ := ∑' n, (F n : ℝ) / 5^n

-- Theorem statement
theorem modified_fibonacci_sum : S = 10 / 19 := by sorry

end NUMINAMATH_CALUDE_modified_fibonacci_sum_l1462_146213


namespace NUMINAMATH_CALUDE_binomial_20_5_l1462_146286

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_5_l1462_146286


namespace NUMINAMATH_CALUDE_females_wearing_glasses_l1462_146247

theorem females_wearing_glasses (total_population : ℕ) (male_population : ℕ) (female_glasses_percentage : ℚ) :
  total_population = 5000 →
  male_population = 2000 →
  female_glasses_percentage = 30 / 100 →
  (total_population - male_population) * female_glasses_percentage = 900 := by
  sorry

end NUMINAMATH_CALUDE_females_wearing_glasses_l1462_146247


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1462_146268

/-- A trinomial x^2 + 2ax + 9 is a perfect square if and only if a = ±3 -/
theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 2*a*x + 9 = (x + b)^2) ↔ (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1462_146268


namespace NUMINAMATH_CALUDE_john_gave_money_to_two_friends_l1462_146293

/-- Calculates the number of friends John gave money to -/
def number_of_friends (initial_amount spent_on_sweets amount_per_friend final_amount : ℚ) : ℕ :=
  ((initial_amount - spent_on_sweets - final_amount) / amount_per_friend).num.toNat

/-- Proves that John gave money to 2 friends -/
theorem john_gave_money_to_two_friends :
  number_of_friends 7.10 1.05 1.00 4.05 = 2 := by
  sorry

#eval number_of_friends 7.10 1.05 1.00 4.05

end NUMINAMATH_CALUDE_john_gave_money_to_two_friends_l1462_146293


namespace NUMINAMATH_CALUDE_prob_at_least_two_heads_l1462_146272

-- Define the number of coins
def n : ℕ := 5

-- Define the probability of getting heads on a single coin toss
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of getting exactly k heads in n tosses
def prob_exactly (k : ℕ) : ℚ := (binomial n k : ℚ) * p^n

-- State the theorem
theorem prob_at_least_two_heads :
  1 - (prob_exactly 0 + prob_exactly 1) = 13/16 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_heads_l1462_146272


namespace NUMINAMATH_CALUDE_class_score_theorem_l1462_146248

def average_score : ℕ := 90

def is_valid_total_score (total : ℕ) : Prop :=
  1000 ≤ total ∧ total ≤ 9999 ∧ total % 10 = 0

def construct_number (A B : ℕ) : ℕ :=
  A * 1000 + 800 + 60 + B

theorem class_score_theorem (A B : ℕ) :
  A < 10 → B < 10 →
  is_valid_total_score (construct_number A B) →
  (construct_number A B) / (construct_number A B / average_score) = average_score →
  A = 4 ∧ B = 0 := by
sorry

end NUMINAMATH_CALUDE_class_score_theorem_l1462_146248


namespace NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l1462_146220

theorem sum_of_sqrt_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  ∃ (c : ℝ), c = 7 ∧ 
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) ≤ c) ∧
  (∀ (c' : ℝ), c' < c → 
    ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 
      Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) > c') :=
by sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l1462_146220


namespace NUMINAMATH_CALUDE_marble_bag_count_l1462_146208

theorem marble_bag_count :
  ∀ (total blue red white : ℕ),
    blue = 5 →
    red = 9 →
    total = blue + red + white →
    (red + white : ℚ) / total = 5 / 6 →
    total = 30 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_count_l1462_146208


namespace NUMINAMATH_CALUDE_maciek_purchases_cost_l1462_146274

/-- The cost of Maciek's purchases given the prices of pretzels and chips -/
def cost_of_purchases (pretzel_price : ℚ) (chip_price_ratio : ℚ) : ℚ :=
  let chip_price := pretzel_price * (1 + chip_price_ratio)
  2 * pretzel_price + 2 * chip_price

/-- Theorem stating that Maciek's purchases cost $22 -/
theorem maciek_purchases_cost :
  cost_of_purchases 4 (75 / 100) = 22 := by
  sorry

#eval cost_of_purchases 4 (75 / 100)

end NUMINAMATH_CALUDE_maciek_purchases_cost_l1462_146274


namespace NUMINAMATH_CALUDE_combined_transformation_correct_l1462_146246

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -2; 2, 0]

theorem combined_transformation_correct :
  combined_transformation = rotation_matrix_90_ccw * dilation_matrix 2 := by
  sorry

end NUMINAMATH_CALUDE_combined_transformation_correct_l1462_146246


namespace NUMINAMATH_CALUDE_project_time_ratio_l1462_146240

theorem project_time_ratio (kate mark pat : ℕ) : 
  kate + mark + pat = 144 →
  pat = 2 * kate →
  mark = kate + 80 →
  pat / mark = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_project_time_ratio_l1462_146240


namespace NUMINAMATH_CALUDE_father_ate_eight_brownies_l1462_146261

/-- The number of brownies Father ate -/
def fatherAte (initialBrownies : ℕ) (mooneyAte : ℕ) (additionalBrownies : ℕ) (finalBrownies : ℕ) : ℕ :=
  initialBrownies + additionalBrownies - mooneyAte - finalBrownies

/-- Proves that Father ate 8 brownies given the problem conditions -/
theorem father_ate_eight_brownies :
  fatherAte (2 * 12) 4 (2 * 12) 36 = 8 := by
  sorry

#eval fatherAte (2 * 12) 4 (2 * 12) 36

end NUMINAMATH_CALUDE_father_ate_eight_brownies_l1462_146261


namespace NUMINAMATH_CALUDE_infinite_product_of_a_l1462_146200

noncomputable def a : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 1 + (a n - 1)^3

theorem infinite_product_of_a : ∏' n, a n = 3/5 := by sorry

end NUMINAMATH_CALUDE_infinite_product_of_a_l1462_146200


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1462_146254

theorem license_plate_palindrome_probability :
  let prob_4digit_palindrome : ℚ := 1 / 100
  let prob_3letter_palindrome : ℚ := 1 / 26
  let prob_at_least_one_palindrome : ℚ := 
    prob_3letter_palindrome + prob_4digit_palindrome - (prob_3letter_palindrome * prob_4digit_palindrome)
  prob_at_least_one_palindrome = 5 / 104 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1462_146254


namespace NUMINAMATH_CALUDE_properties_of_negative_three_l1462_146273

theorem properties_of_negative_three :
  (- (-3) = 3) ∧
  (((-3)⁻¹ : ℚ) = -1/3) ∧
  (abs (-3) = 3) := by
sorry

end NUMINAMATH_CALUDE_properties_of_negative_three_l1462_146273


namespace NUMINAMATH_CALUDE_hcf_is_three_l1462_146262

-- Define the properties of our two numbers
def number_properties (a b : ℕ) : Prop :=
  ∃ (k : ℕ), a = 3 * k ∧ b = 4 * k ∧ Nat.lcm a b = 36

-- Theorem statement
theorem hcf_is_three {a b : ℕ} (h : number_properties a b) : Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hcf_is_three_l1462_146262


namespace NUMINAMATH_CALUDE_milk_fraction_after_pours_l1462_146229

/-- Represents the contents of a cup -/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups -/
structure TwoCapState where
  cup1 : CupContents
  cup2 : CupContents

/-- Initial state of the cups -/
def initial_state : TwoCapState :=
  { cup1 := { tea := 6, milk := 0 },
    cup2 := { tea := 0, milk := 6 } }

/-- Pour one-third of tea from cup1 to cup2 -/
def pour_tea (state : TwoCapState) : TwoCapState := sorry

/-- Pour half of the mixture from cup2 to cup1 -/
def pour_mixture (state : TwoCapState) : TwoCapState := sorry

/-- Calculate the fraction of milk in a cup -/
def milk_fraction (cup : CupContents) : ℚ := sorry

/-- The main theorem to prove -/
theorem milk_fraction_after_pours :
  let state1 := pour_tea initial_state
  let state2 := pour_mixture state1
  milk_fraction state2.cup1 = 3/8 := by sorry

end NUMINAMATH_CALUDE_milk_fraction_after_pours_l1462_146229


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l1462_146292

def original_price : ℝ := 250
def first_discount : ℝ := 0.60
def second_discount : ℝ := 0.25

theorem final_price_after_discounts :
  (original_price * (1 - first_discount) * (1 - second_discount)) = 75 := by
sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l1462_146292


namespace NUMINAMATH_CALUDE_opposite_sign_pairs_l1462_146257

theorem opposite_sign_pairs : 
  ¬((-2^3) * ((-2)^3) < 0) ∧
  ¬((|-4|) * (-(-4)) < 0) ∧
  ((-3^4) * ((-3)^4) < 0) ∧
  ¬((10^2) * (2^10) < 0) := by
sorry

end NUMINAMATH_CALUDE_opposite_sign_pairs_l1462_146257


namespace NUMINAMATH_CALUDE_sqrt_neg_six_squared_minus_one_l1462_146214

theorem sqrt_neg_six_squared_minus_one : Real.sqrt ((-6)^2) - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_six_squared_minus_one_l1462_146214


namespace NUMINAMATH_CALUDE_nonagon_diagonals_count_l1462_146264

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex polygon with 9 sides -/
def nonagon : ℕ := 9

/-- Theorem stating that the number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals_count : nonagon_diagonals = (nonagon * (nonagon - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_count_l1462_146264


namespace NUMINAMATH_CALUDE_maria_coin_count_l1462_146230

theorem maria_coin_count (num_stacks : ℕ) (coins_per_stack : ℕ) : 
  num_stacks = 5 → coins_per_stack = 3 → num_stacks * coins_per_stack = 15 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_count_l1462_146230


namespace NUMINAMATH_CALUDE_giraffe_difference_l1462_146277

/- Define the number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/- Define the number of snakes in Safari National Park -/
def safari_snakes : ℕ := safari_lions / 2

/- Define the number of giraffes in Safari National Park -/
def safari_giraffes : ℕ := safari_snakes - 10

/- Define the number of lions in Savanna National Park -/
def savanna_lions : ℕ := 2 * safari_lions

/- Define the number of snakes in Savanna National Park -/
def savanna_snakes : ℕ := 3 * safari_snakes

/- Define the total number of animals in Savanna National Park -/
def savanna_total : ℕ := 410

/- Theorem: The difference in the number of giraffes between Savanna and Safari National Parks is 20 -/
theorem giraffe_difference : 
  savanna_total - savanna_lions - savanna_snakes - safari_giraffes = 20 := by
  sorry

end NUMINAMATH_CALUDE_giraffe_difference_l1462_146277


namespace NUMINAMATH_CALUDE_solve_equation_l1462_146221

theorem solve_equation (a : ℚ) : a + a / 3 = 8 / 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1462_146221


namespace NUMINAMATH_CALUDE_binomial_seven_four_l1462_146236

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l1462_146236
