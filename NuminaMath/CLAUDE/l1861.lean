import Mathlib

namespace NUMINAMATH_CALUDE_m_range_l1861_186114

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : m ∈ Set.Ioo (-2 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1861_186114


namespace NUMINAMATH_CALUDE_diameter_is_65_l1861_186130

/-- Represents a circle with a diameter and a perpendicular chord --/
structure Circle where
  diameter : ℕ
  chord : ℕ
  is_two_digit : 10 ≤ diameter ∧ diameter < 100
  is_reversed : chord = (diameter % 10) * 10 + (diameter / 10)

/-- The distance from the center to the intersection of the chord and diameter --/
def center_to_intersection (c : Circle) : ℚ :=
  let r := c.diameter / 2
  let h := c.chord / 2
  ((r * r - h * h : ℚ) / (r * r)).sqrt

theorem diameter_is_65 (c : Circle) 
  (h_rational : ∃ (q : ℚ), center_to_intersection c = q) :
  c.diameter = 65 := by
  sorry

#check diameter_is_65

end NUMINAMATH_CALUDE_diameter_is_65_l1861_186130


namespace NUMINAMATH_CALUDE_labor_costs_l1861_186133

/-- Calculate the overall labor costs for one day given the salaries of different workers -/
theorem labor_costs (worker_salary : ℕ) : 
  worker_salary = 100 →
  (2 * worker_salary) + (2 * worker_salary) + (5/2 * worker_salary) = 650 := by
  sorry

#check labor_costs

end NUMINAMATH_CALUDE_labor_costs_l1861_186133


namespace NUMINAMATH_CALUDE_third_circle_radius_l1861_186190

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle is 5. -/
theorem third_circle_radius (P Q R : ℝ × ℝ) (r : ℝ) : 
  let d := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (d = 8) →  -- Distance between centers P and Q
  (∀ X : ℝ × ℝ, (X.1 - P.1)^2 + (X.2 - P.2)^2 = 3^2 → 
    (X.1 - Q.1)^2 + (X.2 - Q.2)^2 = 8^2) →  -- Circles are externally tangent
  (∀ Y : ℝ × ℝ, (Y.1 - P.1)^2 + (Y.2 - P.2)^2 = (3 + r)^2 ∧ 
    (Y.1 - Q.1)^2 + (Y.2 - Q.2)^2 = (5 - r)^2) →  -- Third circle is tangent to both circles
  (∃ Z : ℝ × ℝ, (Z.1 - R.1)^2 + (Z.2 - R.2)^2 = r^2 ∧ 
    ((Z.1 - P.1) * (Q.2 - P.2) = (Z.2 - P.2) * (Q.1 - P.1))) →  -- Third circle is tangent to common external tangent
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_third_circle_radius_l1861_186190


namespace NUMINAMATH_CALUDE_matrix_N_computation_l1861_186165

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![3, -2]) = ![4, 0])
  (h2 : N.mulVec (![(-4), 6]) = ![(-2), -2]) :
  N.mulVec (![7, 2]) = ![16, -4] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_computation_l1861_186165


namespace NUMINAMATH_CALUDE_bisecting_circle_relation_l1861_186158

/-- A circle that always bisects another circle -/
structure BisectingCircle where
  a : ℝ
  b : ℝ
  eq_bisecting : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1
  eq_bisected : ∀ (x y : ℝ), (x + 1)^2 + (y + 1)^2 = 4
  bisects : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1 → 
    ∃ (t : ℝ), (x + 1)^2 + (y + 1)^2 = 4 ∧ 
    ((1 - t) * x + t * (-1))^2 + ((1 - t) * y + t * (-1))^2 = 1

/-- The relationship between a and b in a bisecting circle -/
theorem bisecting_circle_relation (c : BisectingCircle) : 
  c.a^2 + 2*c.a + 2*c.b + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_circle_relation_l1861_186158


namespace NUMINAMATH_CALUDE_no_adjacent_birch_probability_l1861_186110

def num_maple : ℕ := 4
def num_oak : ℕ := 5
def num_birch : ℕ := 6
def num_pine : ℕ := 2
def total_trees : ℕ := num_maple + num_oak + num_birch + num_pine

def probability_no_adjacent_birch : ℚ := 21 / 283

theorem no_adjacent_birch_probability :
  let total_arrangements := (total_trees.choose num_birch : ℚ)
  let valid_arrangements := ((total_trees - num_birch + 1).choose num_birch : ℚ)
  valid_arrangements / total_arrangements = probability_no_adjacent_birch :=
by sorry

end NUMINAMATH_CALUDE_no_adjacent_birch_probability_l1861_186110


namespace NUMINAMATH_CALUDE_natural_number_equality_integer_absolute_equality_l1861_186155

-- Part a
theorem natural_number_equality (x y n : ℕ) 
  (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y := by sorry

-- Part b
theorem integer_absolute_equality (x y : ℤ) (n : ℕ) 
  (hx : x ≠ 0) (hy : y ≠ 0)
  (h : x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| := by sorry

end NUMINAMATH_CALUDE_natural_number_equality_integer_absolute_equality_l1861_186155


namespace NUMINAMATH_CALUDE_ratio_equality_l1861_186100

theorem ratio_equality (x y a b : ℝ) 
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / (3 * b - y) = 3) :
  a / b = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1861_186100


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l1861_186161

/-- Given a cube with space diagonal 5√3, prove its volume is 125 -/
theorem cube_volume_from_diagonal (s : ℝ) : 
  s * Real.sqrt 3 = 5 * Real.sqrt 3 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l1861_186161


namespace NUMINAMATH_CALUDE_church_attendance_l1861_186176

/-- Proves the number of female adults in a church given the number of children, male adults, and total people. -/
theorem church_attendance (children : ℕ) (male_adults : ℕ) (total_people : ℕ) 
  (h1 : children = 80)
  (h2 : male_adults = 60)
  (h3 : total_people = 200) :
  total_people - (children + male_adults) = 60 := by
  sorry

#check church_attendance

end NUMINAMATH_CALUDE_church_attendance_l1861_186176


namespace NUMINAMATH_CALUDE_grassland_area_ratio_l1861_186152

/-- Represents a grassland with two parts -/
structure Grassland where
  areaA : ℝ
  areaB : ℝ
  growthRate : ℝ
  cowEatingRate : ℝ

/-- The conditions of the problem -/
def problem_conditions (g : Grassland) : Prop :=
  g.areaA > 0 ∧ g.areaB > 0 ∧ g.areaA ≠ g.areaB ∧
  g.growthRate > 0 ∧ g.cowEatingRate > 0 ∧
  g.areaA * g.growthRate = 7 * g.cowEatingRate ∧
  g.areaB * g.growthRate = 4 * g.cowEatingRate ∧
  7 * g.growthRate = g.areaA * g.growthRate

/-- The theorem stating the ratio of areas -/
theorem grassland_area_ratio (g : Grassland) :
  problem_conditions g → g.areaA / g.areaB = 105 / 44 :=
by sorry

end NUMINAMATH_CALUDE_grassland_area_ratio_l1861_186152


namespace NUMINAMATH_CALUDE_three_primes_sum_l1861_186103

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def smallest_square_greater_than_15 : ℕ := 16

theorem three_primes_sum (p q r : ℕ) : 
  is_prime p → is_prime q → is_prime r →
  p + q + r = smallest_square_greater_than_15 →
  1 < p → p < q → q < r →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_three_primes_sum_l1861_186103


namespace NUMINAMATH_CALUDE_complex_product_real_condition_l1861_186136

theorem complex_product_real_condition (a b c d : ℝ) :
  (Complex.I * b + a) * (Complex.I * d + c) ∈ Set.range Complex.ofReal ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_condition_l1861_186136


namespace NUMINAMATH_CALUDE_number_with_one_third_equal_to_twelve_l1861_186148

theorem number_with_one_third_equal_to_twelve (x : ℝ) : (1 / 3 : ℝ) * x = 12 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_with_one_third_equal_to_twelve_l1861_186148


namespace NUMINAMATH_CALUDE_food_drive_problem_l1861_186112

/-- Food drive problem -/
theorem food_drive_problem (rachel_cans jaydon_cans mark_cans : ℕ) : 
  jaydon_cans = 2 * rachel_cans + 5 →
  mark_cans = 4 * jaydon_cans →
  rachel_cans + jaydon_cans + mark_cans = 135 →
  mark_cans = 100 := by
  sorry

#check food_drive_problem

end NUMINAMATH_CALUDE_food_drive_problem_l1861_186112


namespace NUMINAMATH_CALUDE_soy_milk_calculation_l1861_186149

/-- The amount of soy milk drunk by Mitch's family in a week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk drunk by Mitch's family in a week -/
def total_milk : ℝ := 0.6

/-- The amount of regular milk drunk by Mitch's family in a week -/
def regular_milk : ℝ := 0.5

/-- Theorem stating that the amount of soy milk is the difference between total milk and regular milk -/
theorem soy_milk_calculation : soy_milk = total_milk - regular_milk := by
  sorry

end NUMINAMATH_CALUDE_soy_milk_calculation_l1861_186149


namespace NUMINAMATH_CALUDE_stake_B_maximizes_grazing_area_l1861_186108

/-- Represents a stake on the edge of the pond -/
inductive Stake
| A
| B
| C
| D

/-- The side length of the square pond in meters -/
def pondSideLength : ℝ := 12

/-- The distance between adjacent stakes in meters -/
def stakesDistance : ℝ := 3

/-- The length of the rope in meters -/
def ropeLength : ℝ := 4

/-- Calculates the grazing area for a given stake -/
noncomputable def grazingArea (s : Stake) : ℝ :=
  match s with
  | Stake.A => 4.25 * Real.pi
  | Stake.B => 8 * Real.pi
  | Stake.C => 4.25 * Real.pi
  | Stake.D => 4.25 * Real.pi

/-- Theorem stating that stake B maximizes the grazing area -/
theorem stake_B_maximizes_grazing_area :
  ∀ s : Stake, grazingArea Stake.B ≥ grazingArea s :=
sorry


end NUMINAMATH_CALUDE_stake_B_maximizes_grazing_area_l1861_186108


namespace NUMINAMATH_CALUDE_alan_age_is_29_l1861_186115

/-- Represents the ages of Alan and Chris --/
structure Ages where
  alan : ℕ
  chris : ℕ

/-- The condition that the sum of their ages is 52 --/
def sum_of_ages (ages : Ages) : Prop :=
  ages.alan + ages.chris = 52

/-- The complex age relationship between Alan and Chris --/
def age_relationship (ages : Ages) : Prop :=
  ages.chris = ages.alan - (ages.alan - (ages.alan - (ages.alan / 3)))

/-- The theorem stating Alan's age is 29 given the conditions --/
theorem alan_age_is_29 (ages : Ages) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relationship ages) : 
  ages.alan = 29 := by
  sorry


end NUMINAMATH_CALUDE_alan_age_is_29_l1861_186115


namespace NUMINAMATH_CALUDE_trig_identity_quadratic_solution_l1861_186195

-- Part 1
theorem trig_identity : 
  Real.tan (π / 6) ^ 2 + 2 * Real.sin (π / 4) - 2 * Real.cos (π / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

-- Part 2
theorem quadratic_solution :
  let x₁ := (-2 + Real.sqrt 2) / 2
  let x₂ := (-2 - Real.sqrt 2) / 2
  2 * x₁ ^ 2 + 4 * x₁ + 1 = 0 ∧ 2 * x₂ ^ 2 + 4 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_quadratic_solution_l1861_186195


namespace NUMINAMATH_CALUDE_exp_convex_and_ln_concave_l1861_186172

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the natural logarithm function
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem exp_convex_and_ln_concave :
  (∀ x y : ℝ, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    g (t * x + (1 - t) * y) ≥ t * g x + (1 - t) * g y) :=
by sorry

end NUMINAMATH_CALUDE_exp_convex_and_ln_concave_l1861_186172


namespace NUMINAMATH_CALUDE_ascending_order_of_rationals_l1861_186127

theorem ascending_order_of_rationals (a b : ℚ) 
  (ha : a > 0) (hb : b < 0) (hab : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by sorry

end NUMINAMATH_CALUDE_ascending_order_of_rationals_l1861_186127


namespace NUMINAMATH_CALUDE_shopkeeper_bananas_l1861_186178

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) : 
  oranges = 600 →
  (oranges * 85 / 100 + bananas * 97 / 100 : ℚ) = (oranges + bananas) * 898 / 1000 →
  bananas = 400 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_bananas_l1861_186178


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2010_l1861_186139

theorem units_digit_of_7_to_2010 : (7^2010) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2010_l1861_186139


namespace NUMINAMATH_CALUDE_four_digit_permutations_2033_eq_18_l1861_186187

/-- The number of unique four-digit permutations of the digits in 2033 -/
def four_digit_permutations_2033 : ℕ := 18

/-- The set of digits in 2033 -/
def digits_2033 : Finset ℕ := {0, 2, 3}

/-- The function to count valid permutations -/
def count_valid_permutations (digits : Finset ℕ) : ℕ :=
  sorry

theorem four_digit_permutations_2033_eq_18 :
  count_valid_permutations digits_2033 = four_digit_permutations_2033 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_permutations_2033_eq_18_l1861_186187


namespace NUMINAMATH_CALUDE_replacement_solution_percentage_l1861_186154

theorem replacement_solution_percentage
  (original_percentage : ℝ)
  (replaced_portion : ℝ)
  (final_percentage : ℝ)
  (h1 : original_percentage = 85)
  (h2 : replaced_portion = 0.6923076923076923)
  (h3 : final_percentage = 40)
  (x : ℝ) :
  (original_percentage * (1 - replaced_portion) + x * replaced_portion = final_percentage) →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_replacement_solution_percentage_l1861_186154


namespace NUMINAMATH_CALUDE_provisions_duration_l1861_186124

/-- Given provisions for a certain number of boys and days, calculate how long the provisions will last with additional boys. -/
theorem provisions_duration (initial_boys : ℕ) (initial_days : ℕ) (additional_boys : ℕ) :
  let total_boys := initial_boys + additional_boys
  let new_days := (initial_boys * initial_days) / total_boys
  initial_boys = 1500 → initial_days = 25 → additional_boys = 350 →
  ⌊(new_days : ℚ)⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_provisions_duration_l1861_186124


namespace NUMINAMATH_CALUDE_integer_average_l1861_186185

theorem integer_average (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 68 →
  (a + b + c + d + e) / 5 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_integer_average_l1861_186185


namespace NUMINAMATH_CALUDE_train_length_l1861_186118

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 36 * (1000 / 3600) →
  time = 25.997920166386688 →
  bridge_length = 150 →
  speed * time - bridge_length = 109.97920166386688 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1861_186118


namespace NUMINAMATH_CALUDE_square_root_equation_l1861_186132

theorem square_root_equation (n : ℝ) : 
  Real.sqrt (10 + n) = 9 → n = 71 := by sorry

end NUMINAMATH_CALUDE_square_root_equation_l1861_186132


namespace NUMINAMATH_CALUDE_candy_necklace_problem_l1861_186113

/-- Candy necklace problem -/
theorem candy_necklace_problem (blocks : ℕ) (pieces_per_block : ℕ) (people : ℕ) 
  (h1 : blocks = 3) 
  (h2 : pieces_per_block = 30) 
  (h3 : people = 9) :
  (blocks * pieces_per_block) / people = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklace_problem_l1861_186113


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_longest_side_l1861_186160

-- Define the triangle
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop :=
  -- We don't implement the full definition, just declare it as a property
  sorry

def longestSide (t : Triangle) (side : ℝ) : Prop :=
  -- We don't implement the full definition, just declare it as a property
  sorry

def triangleArea (t : Triangle) : ℝ :=
  -- We don't implement the full calculation, just declare it as a function
  sorry

-- Main theorem
theorem isosceles_right_triangle_longest_side 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : longestSide t (dist t.X t.Y)) 
  (h3 : triangleArea t = 49) : 
  dist t.X t.Y = 14 :=
sorry

-- Note: dist is a function that calculates the distance between two points

end NUMINAMATH_CALUDE_isosceles_right_triangle_longest_side_l1861_186160


namespace NUMINAMATH_CALUDE_max_volume_right_triangle_rotation_l1861_186126

theorem max_volume_right_triangle_rotation (a b c : ℝ) : 
  a = 3 → b = 4 → c = 5 → a^2 + b^2 = c^2 →
  (max (1/3 * Real.pi * a^2 * b) (max (1/3 * Real.pi * b^2 * a) (1/3 * Real.pi * (2 * (1/2 * a * b) / c)^2 * c))) = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_max_volume_right_triangle_rotation_l1861_186126


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_seven_l1861_186105

theorem smallest_four_digit_mod_seven : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 7 = 6 → n ≥ 1000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_seven_l1861_186105


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_l1861_186191

/-- Given a class where there are 3 more girls than boys and the total number of students is 41,
    prove that the number of girls in the class is 22. -/
theorem number_of_girls_in_class (boys girls : ℕ) : 
  girls = boys + 3 → 
  boys + girls = 41 → 
  girls = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_l1861_186191


namespace NUMINAMATH_CALUDE_congruence_solution_l1861_186170

theorem congruence_solution (n : ℤ) : 13 * 26 ≡ 8 [ZMOD 47] := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1861_186170


namespace NUMINAMATH_CALUDE_right_triangle_area_l1861_186181

theorem right_triangle_area (leg1 leg2 : ℝ) (h1 : leg1 = 45) (h2 : leg2 = 48) :
  (1 / 2 : ℝ) * leg1 * leg2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1861_186181


namespace NUMINAMATH_CALUDE_perpendicular_lines_minimum_product_l1861_186116

theorem perpendicular_lines_minimum_product (b a : ℝ) : 
  b > 0 → 
  ((b^2 + 1) * (-b^2) = -1) →
  ab ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_minimum_product_l1861_186116


namespace NUMINAMATH_CALUDE_kitchen_length_l1861_186163

/-- Calculates the length of a rectangular kitchen given its dimensions and painting information. -/
theorem kitchen_length (width height : ℝ) (total_area_painted : ℝ) : 
  width = 16 ∧ 
  height = 10 ∧ 
  total_area_painted = 1680 → 
  ∃ length : ℝ, length = 12 ∧ 
    total_area_painted / 3 = 2 * (length * height + width * height) :=
by sorry

end NUMINAMATH_CALUDE_kitchen_length_l1861_186163


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_32_l1861_186147

/-- The area of the right triangle formed by the lines y = x, x = -8, and the x-axis -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let line1 : ℝ → ℝ → Prop := fun x y => y = x
    let line2 : ℝ → Prop := fun x => x = -8
    let x_axis : ℝ → Prop := fun y => y = 0
    let intersection_point : ℝ × ℝ := (-8, -8)
    let base : ℝ := 8
    let height : ℝ := 8
    (∀ x y, line1 x y → line2 x → (x, y) = intersection_point) ∧
    (∀ x, line2 x → x_axis 0) ∧
    (area = (1/2) * base * height) →
    area = 32

theorem triangle_area_is_32 : triangle_area 32 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_32_l1861_186147


namespace NUMINAMATH_CALUDE_m_value_theorem_l1861_186109

theorem m_value_theorem (m : ℕ) : 
  2^2000 - 3 * 2^1999 + 5 * 2^1998 - 2^1997 = m * 2^1997 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_m_value_theorem_l1861_186109


namespace NUMINAMATH_CALUDE_smallest_class_size_is_42_l1861_186196

/-- Represents the number of students in a physical education class. -/
def ClassSize (n : ℕ) : ℕ := 5 * n + 2

/-- The smallest class size satisfying the given conditions -/
def SmallestClassSize : ℕ := 42

theorem smallest_class_size_is_42 :
  (∀ m : ℕ, ClassSize m > 40 → m ≥ SmallestClassSize) ∧
  (ClassSize (SmallestClassSize - 1) ≤ 40) ∧
  (ClassSize SmallestClassSize > 40) :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_is_42_l1861_186196


namespace NUMINAMATH_CALUDE_store_buying_combinations_l1861_186177

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of item choices for each student --/
def num_choices : ℕ := 2

/-- The total number of possible buying combinations --/
def total_combinations : ℕ := num_choices ^ num_students

/-- The number of valid buying combinations --/
def valid_combinations : ℕ := total_combinations - 1

theorem store_buying_combinations :
  valid_combinations = 15 := by sorry

end NUMINAMATH_CALUDE_store_buying_combinations_l1861_186177


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l1861_186144

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ+), (2 * x.val)^2 + 2 * 41 * (2 * x.val) + 41^2 ≡ 0 [MOD 53] ∧
  ∀ (y : ℕ+), ((2 * y.val)^2 + 2 * 41 * (2 * y.val) + 41^2 ≡ 0 [MOD 53]) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l1861_186144


namespace NUMINAMATH_CALUDE_derivative_of_y_l1861_186121

-- Define the function y
def y (x a b c : ℝ) : ℝ := (x - a) * (x - b) * (x - c)

-- State the theorem
theorem derivative_of_y (x a b c : ℝ) :
  deriv (fun x => y x a b c) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l1861_186121


namespace NUMINAMATH_CALUDE_card_area_problem_l1861_186180

theorem card_area_problem (length width : ℝ) : 
  length = 5 ∧ width = 7 →
  (∃ (shortened_side : ℝ), (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
                           shortened_side * (if shortened_side = length - 2 then width else length) = 21) →
  (if length - 2 * width = 21 then (length * (width - 2) = 25) else ((length - 2) * width = 25)) :=
by sorry

end NUMINAMATH_CALUDE_card_area_problem_l1861_186180


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_4_is_acute_l1861_186186

theorem triangle_with_angle_ratio_2_3_4_is_acute (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  b = (3/2) * a →          -- ratio between second and first angle
  c = 2 * a →              -- ratio between third and first angle
  a < 90 ∧ b < 90 ∧ c < 90 -- all angles are less than 90 degrees (acute triangle)
  := by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_4_is_acute_l1861_186186


namespace NUMINAMATH_CALUDE_m_range_l1861_186102

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 2 * P.1

-- Define the distance ratio condition
def satisfies_distance_ratio (P : ℝ × ℝ) (m : ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = m^2 * ((P.1 - 1)^2 + P.2^2)

-- Main theorem
theorem m_range (P : ℝ × ℝ) (m : ℝ) 
  (h1 : on_parabola P) 
  (h2 : satisfies_distance_ratio P m) : 
  1 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1861_186102


namespace NUMINAMATH_CALUDE_world_grain_ratio_l1861_186173

def world_grain_supply : ℝ := 1800000
def world_grain_demand : ℝ := 2400000

theorem world_grain_ratio : 
  world_grain_supply / world_grain_demand = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_world_grain_ratio_l1861_186173


namespace NUMINAMATH_CALUDE_rosy_fish_count_l1861_186189

/-- The number of fish Lilly has -/
def lillys_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 18

/-- The number of fish Rosy has -/
def rosys_fish : ℕ := total_fish - lillys_fish

theorem rosy_fish_count : rosys_fish = 8 := by sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l1861_186189


namespace NUMINAMATH_CALUDE_intersection_points_l1861_186198

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*(x - y) - 18 = 0

theorem intersection_points :
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) ∧
  (circle1 3 3 ∧ circle2 3 3) ∧
  (circle1 (-3) 5 ∧ circle2 (-3) 5) ∧
  (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → (x = 3 ∧ y = 3) ∨ (x = -3 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l1861_186198


namespace NUMINAMATH_CALUDE_field_pond_area_ratio_l1861_186162

/-- Given a rectangular field and a square pond, prove the ratio of their areas -/
theorem field_pond_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 2 * field_width →
  field_length = 32 →
  pond_side = 8 →
  (pond_side^2) / (field_length * field_width) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_field_pond_area_ratio_l1861_186162


namespace NUMINAMATH_CALUDE_parallel_lines_k_equals_negative_one_l1861_186166

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.b ≠ 0

theorem parallel_lines_k_equals_negative_one :
  ∀ k : ℝ,
  let l1 : Line := ⟨k, -1, 1⟩
  let l2 : Line := ⟨1, -k, 1⟩
  parallel l1 l2 → k = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_equals_negative_one_l1861_186166


namespace NUMINAMATH_CALUDE_angies_age_equation_l1861_186197

theorem angies_age_equation (angie_age : ℕ) (result : ℕ) : 
  angie_age = 8 → result = 2 * angie_age + 4 → result = 20 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_equation_l1861_186197


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1861_186174

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 2 ∧ b = 3 ∧ c^2 = a^2 + b^2 → c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1861_186174


namespace NUMINAMATH_CALUDE_sqrt_of_four_is_plus_minus_two_l1861_186171

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four_is_plus_minus_two : sqrt 4 = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_is_plus_minus_two_l1861_186171


namespace NUMINAMATH_CALUDE_tan_neg_alpha_implies_expression_l1861_186199

theorem tan_neg_alpha_implies_expression (α : Real) 
  (h : Real.tan (-α) = 3) : 
  (Real.sin α)^2 - Real.sin (2 * α) = (-15/8) * Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_alpha_implies_expression_l1861_186199


namespace NUMINAMATH_CALUDE_stickers_at_end_of_week_l1861_186188

def initial_stickers : ℝ := 39.0
def given_away_stickers : ℝ := 22.0

theorem stickers_at_end_of_week : 
  initial_stickers - given_away_stickers = 17.0 := by
  sorry

end NUMINAMATH_CALUDE_stickers_at_end_of_week_l1861_186188


namespace NUMINAMATH_CALUDE_set_game_combinations_l1861_186143

theorem set_game_combinations (n : ℕ) (k : ℕ) (h1 : n = 81) (h2 : k = 3) :
  Nat.choose n k = 85320 := by
  sorry

end NUMINAMATH_CALUDE_set_game_combinations_l1861_186143


namespace NUMINAMATH_CALUDE_reading_time_difference_l1861_186134

/-- The difference in reading time between two people reading the same book -/
theorem reading_time_difference 
  (jonathan_speed : ℝ) 
  (alice_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : jonathan_speed = 150) 
  (h2 : alice_speed = 75) 
  (h3 : book_pages = 450) : 
  (book_pages / alice_speed - book_pages / jonathan_speed) * 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1861_186134


namespace NUMINAMATH_CALUDE_total_out_of_pocket_cost_l1861_186183

/-- Calculates the total out-of-pocket cost for medical treatment --/
theorem total_out_of_pocket_cost 
  (doctor_visit_cost : ℕ) 
  (cast_cost : ℕ) 
  (initial_insurance_coverage : ℚ) 
  (pt_sessions : ℕ) 
  (pt_cost_per_session : ℕ) 
  (pt_insurance_coverage : ℚ) : 
  doctor_visit_cost = 300 →
  cast_cost = 200 →
  initial_insurance_coverage = 60 / 100 →
  pt_sessions = 8 →
  pt_cost_per_session = 100 →
  pt_insurance_coverage = 40 / 100 →
  (1 - initial_insurance_coverage) * (doctor_visit_cost + cast_cost) +
  (1 - pt_insurance_coverage) * (pt_sessions * pt_cost_per_session) = 680 := by
sorry

end NUMINAMATH_CALUDE_total_out_of_pocket_cost_l1861_186183


namespace NUMINAMATH_CALUDE_logarithm_equations_solutions_l1861_186131

theorem logarithm_equations_solutions :
  (∀ x : ℝ, x^2 - 1 > 0 ∧ x^2 - 1 ≠ 1 ∧ x^3 + 6 > 0 ∧ x^3 + 6 = 4*x^2 - x →
    x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 - 4 > 0 ∧ x^3 + x > 0 ∧ x^3 + x ≠ 1 ∧ x^3 + x = 4*x^2 - 6 →
    x = 3) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equations_solutions_l1861_186131


namespace NUMINAMATH_CALUDE_inequality_proofs_l1861_186123

theorem inequality_proofs :
  (∀ (a b : ℝ), a > 0 → b > 0 → a^3 + b^3 ≥ a*b^2 + a^2*b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y > 2 → (1 + y) / x < 2 ∨ (1 + x) / y < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1861_186123


namespace NUMINAMATH_CALUDE_horner_v2_value_l1861_186104

/-- Horner's method for polynomial evaluation -/
def horner_step (x : ℝ) (a : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = x^4 + 2x^3 - 3x^2 + x + 5 -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - 3*x^2 + x + 5

theorem horner_v2_value :
  let x : ℝ := 2
  let v₁ : ℝ := horner_step x 2 1  -- Corresponds to x + 2
  let v₂ : ℝ := horner_step x (-3) v₁  -- Corresponds to v₁ * x - 3
  v₂ = 5 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l1861_186104


namespace NUMINAMATH_CALUDE_olympic_mascot_pricing_and_purchasing_l1861_186140

theorem olympic_mascot_pricing_and_purchasing
  (small_price large_price : ℝ)
  (h1 : large_price - 2 * small_price = 20)
  (h2 : 3 * small_price + 2 * large_price = 390)
  (budget : ℝ) (total_sets : ℕ)
  (h3 : budget = 1500)
  (h4 : total_sets = 20) :
  small_price = 50 ∧ 
  large_price = 120 ∧ 
  (∃ m : ℕ, m ≤ total_sets ∧ 
    m * large_price + (total_sets - m) * small_price ≤ budget ∧
    ∀ n : ℕ, n > m → n * large_price + (total_sets - n) * small_price > budget) ∧
  (7 : ℕ) * large_price + (total_sets - 7) * small_price ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_olympic_mascot_pricing_and_purchasing_l1861_186140


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1861_186156

/-- Given a reflection of point (2, -2) across line y = mx + b to point (8, 4), prove m + b = 5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The reflected point (x, y) satisfies the reflection property
    (x - 2)^2 + (y + 2)^2 = (8 - 2)^2 + (4 + 2)^2 ∧
    -- The midpoint of the original and reflected points lies on y = mx + b
    (1 : ℝ) = m * 5 + b ∧
    -- The line y = mx + b is perpendicular to the line connecting the original and reflected points
    m * ((8 - 2) / (4 + 2)) = -1) →
  m + b = 5 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1861_186156


namespace NUMINAMATH_CALUDE_planes_with_three_common_points_l1861_186101

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define what it means for a point to be on a plane
def on_plane (p : Point) (plane : Plane) : Prop :=
  let (x, y, z) := p
  plane x y z

-- Define what it means for two planes to intersect
def intersect (p1 p2 : Plane) : Prop :=
  ∃ (p : Point), on_plane p p1 ∧ on_plane p p2

-- Define what it means for two planes to coincide
def coincide (p1 p2 : Plane) : Prop :=
  ∀ (p : Point), on_plane p p1 ↔ on_plane p p2

-- Theorem statement
theorem planes_with_three_common_points 
  (p1 p2 : Plane) (a b c : Point)
  (h1 : on_plane a p1 ∧ on_plane a p2)
  (h2 : on_plane b p1 ∧ on_plane b p2)
  (h3 : on_plane c p1 ∧ on_plane c p2) :
  intersect p1 p2 ∨ coincide p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_with_three_common_points_l1861_186101


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1861_186119

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 18 = 0 → n ≥ 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1861_186119


namespace NUMINAMATH_CALUDE_parabola_focus_for_x_squared_l1861_186194

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- A parabola is symmetric about the y-axis if its equation has no x term -/
def isSymmetricAboutYAxis (p : Parabola) : Prop := p.b = 0

theorem parabola_focus_for_x_squared (p : Parabola) 
  (h1 : p.a = 1) 
  (h2 : p.b = 0) 
  (h3 : p.c = 0) 
  (h4 : isSymmetricAboutYAxis p) : 
  focus p = (0, 1/4) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_for_x_squared_l1861_186194


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l1861_186193

theorem power_mod_seventeen : 3^45 % 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l1861_186193


namespace NUMINAMATH_CALUDE_negation_equivalence_l1861_186137

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x > 2) ↔ (∀ x : ℝ, x^2 - 2*x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1861_186137


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1861_186164

theorem dennis_teaching_years 
  (total_years : ℕ) 
  (virginia_adrienne_diff : ℕ) 
  (dennis_virginia_diff : ℕ) 
  (h1 : total_years = 102)
  (h2 : virginia_adrienne_diff = 9)
  (h3 : dennis_virginia_diff = 9)
  : ∃ (adrienne virginia dennis : ℕ),
    adrienne + virginia + dennis = total_years ∧
    virginia = adrienne + virginia_adrienne_diff ∧
    dennis = virginia + dennis_virginia_diff ∧
    dennis = 43 :=
by sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1861_186164


namespace NUMINAMATH_CALUDE_first_player_wins_l1861_186184

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game --/
inductive Player
  | First
  | Second

/-- Represents the game state --/
structure GameState :=
  (grid : Grid)
  (currentPlayer : Player)
  (shadedCells : Set (ℕ × ℕ))

/-- Defines a valid move in the game --/
def ValidMove (state : GameState) (move : Set (ℕ × ℕ)) : Prop :=
  ∀ cell ∈ move,
    cell.1 ≤ state.grid.rows ∧
    cell.2 ≤ state.grid.cols ∧
    cell ∉ state.shadedCells

/-- Defines the winning condition --/
def IsWinningState (state : GameState) : Prop :=
  ∀ move : Set (ℕ × ℕ), ¬(ValidMove state move)

/-- Theorem: The first player has a winning strategy in a 19 × 94 grid game --/
theorem first_player_wins :
  ∃ (strategy : GameState → Set (ℕ × ℕ)),
    let initialState := GameState.mk (Grid.mk 19 94) Player.First ∅
    ∀ (game : ℕ → GameState),
      game 0 = initialState →
      (∀ n : ℕ,
        (game n).currentPlayer = Player.First →
        ValidMove (game n) (strategy (game n)) ∧
        (game (n + 1)).shadedCells = (game n).shadedCells ∪ (strategy (game n))) →
      (∀ n : ℕ,
        (game n).currentPlayer = Player.Second →
        ∃ move,
          ValidMove (game n) move ∧
          (game (n + 1)).shadedCells = (game n).shadedCells ∪ move) →
      ∃ m : ℕ, IsWinningState (game m) ∧ (game m).currentPlayer = Player.First :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1861_186184


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1861_186153

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧ 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1861_186153


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l1861_186120

theorem crushing_load_calculation (T H L : ℝ) : 
  T = 3 → H = 9 → L = (36 * T^3) / H^3 → L = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l1861_186120


namespace NUMINAMATH_CALUDE_square_side_irrational_l1861_186157

theorem square_side_irrational (area : ℝ) (h : area = 3) :
  ∃ (side : ℝ), side * side = area ∧ Irrational side := by
  sorry

end NUMINAMATH_CALUDE_square_side_irrational_l1861_186157


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1861_186138

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 2)) → x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1861_186138


namespace NUMINAMATH_CALUDE_rachel_bought_seven_chairs_l1861_186107

/-- Calculates the number of chairs Rachel bought given the number of tables,
    time spent per furniture piece, and total time spent. -/
def chairs_bought (num_tables : ℕ) (time_per_piece : ℕ) (total_time : ℕ) : ℕ :=
  (total_time - num_tables * time_per_piece) / time_per_piece

/-- Theorem stating that Rachel bought 7 chairs given the problem conditions. -/
theorem rachel_bought_seven_chairs :
  chairs_bought 3 4 40 = 7 := by
  sorry

end NUMINAMATH_CALUDE_rachel_bought_seven_chairs_l1861_186107


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l1861_186125

/-- Calculates the number of teachers at King Middle School given the specified conditions -/
theorem king_middle_school_teachers :
  let num_students : ℕ := 1500
  let classes_per_student : ℕ := 5
  let classes_per_teacher : ℕ := 5
  let students_per_class : ℕ := 25
  let total_class_instances : ℕ := num_students * classes_per_student
  let unique_classes : ℕ := total_class_instances / students_per_class
  let num_teachers : ℕ := unique_classes / classes_per_teacher
  num_teachers = 60 := by sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l1861_186125


namespace NUMINAMATH_CALUDE_distance_first_to_last_l1861_186179

-- Define the number of trees
def num_trees : ℕ := 8

-- Define the distance between first and fifth tree
def distance_1_to_5 : ℝ := 80

-- Theorem to prove
theorem distance_first_to_last :
  let distance_between_trees := distance_1_to_5 / 4
  let num_spaces := num_trees - 1
  distance_between_trees * num_spaces = 140 := by
sorry

end NUMINAMATH_CALUDE_distance_first_to_last_l1861_186179


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1861_186122

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 6 * y - 12 * x = b) (h3 : b ≠ 0) : a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1861_186122


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l1861_186129

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - (shoes_lost + 1) / 2

/-- Theorem stating that with 25 initial pairs and 9 shoes lost,
    the maximum number of complete pairs remaining is 20. -/
theorem max_pairs_after_loss : max_remaining_pairs 25 9 = 20 := by
  sorry

#eval max_remaining_pairs 25 9

end NUMINAMATH_CALUDE_max_pairs_after_loss_l1861_186129


namespace NUMINAMATH_CALUDE_triangle_inequality_l1861_186145

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) 
  (h_S : S = Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * 
    (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c))) :
  a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1861_186145


namespace NUMINAMATH_CALUDE_tmobile_additional_line_cost_l1861_186192

theorem tmobile_additional_line_cost 
  (tmobile_base : ℕ) 
  (mmobile_base : ℕ) 
  (mmobile_additional : ℕ) 
  (total_lines : ℕ) 
  (price_difference : ℕ) 
  (h1 : tmobile_base = 50)
  (h2 : mmobile_base = 45)
  (h3 : mmobile_additional = 14)
  (h4 : total_lines = 5)
  (h5 : price_difference = 11)
  (h6 : tmobile_base + (total_lines - 2) * x = 
        mmobile_base + (total_lines - 2) * mmobile_additional + price_difference) :
  x = 16 := by
  sorry

#check tmobile_additional_line_cost

end NUMINAMATH_CALUDE_tmobile_additional_line_cost_l1861_186192


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1861_186142

theorem isosceles_right_triangle (a b c : ℝ) :
  a = 2 * Real.sqrt 6 ∧ 
  b = 2 * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt 3 →
  (a^2 = b^2 + c^2) ∧ (b = c) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1861_186142


namespace NUMINAMATH_CALUDE_rudolph_stop_signs_l1861_186151

/-- Calculates the number of stop signs encountered on a car trip -/
def stop_signs_encountered (base_distance : ℕ) (additional_distance : ℕ) (signs_per_mile : ℕ) : ℕ :=
  (base_distance + additional_distance) * signs_per_mile

/-- Theorem: Rudolph encountered 14 stop signs on his trip -/
theorem rudolph_stop_signs :
  stop_signs_encountered 5 2 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_rudolph_stop_signs_l1861_186151


namespace NUMINAMATH_CALUDE_x_one_value_l1861_186146

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l1861_186146


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1861_186159

theorem rope_cutting_problem (rope1 rope2 rope3 rope4 : ℕ) 
  (h1 : rope1 = 30) (h2 : rope2 = 45) (h3 : rope3 = 60) (h4 : rope4 = 75) :
  Nat.gcd rope1 (Nat.gcd rope2 (Nat.gcd rope3 rope4)) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1861_186159


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l1861_186168

/-- The number of penalty kicks in a soccer challenge --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: In a soccer team with 26 players, including 4 goalies, 
    where each player kicks against each goalie once, 
    the total number of penalty kicks is 100. --/
theorem soccer_penalty_kicks : 
  penalty_kicks 26 4 = 100 := by
sorry

#eval penalty_kicks 26 4

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l1861_186168


namespace NUMINAMATH_CALUDE_probability_all_sides_of_decagon_l1861_186106

/-- A regular decagon --/
structure RegularDecagon where

/-- A triangle formed from three vertices of a regular decagon --/
structure DecagonTriangle where
  decagon : RegularDecagon
  vertex1 : Nat
  vertex2 : Nat
  vertex3 : Nat

/-- Predicate to check if three vertices are sequentially adjacent in a decagon --/
def are_sequential_adjacent (v1 v2 v3 : Nat) : Prop :=
  (v2 = (v1 + 1) % 10) ∧ (v3 = (v2 + 1) % 10)

/-- Predicate to check if a triangle's sides are all sides of the decagon --/
def all_sides_of_decagon (t : DecagonTriangle) : Prop :=
  are_sequential_adjacent t.vertex1 t.vertex2 t.vertex3

/-- The total number of possible triangles in a decagon --/
def total_triangles : Nat := 120

/-- The number of triangles with all sides being sides of the decagon --/
def favorable_triangles : Nat := 10

/-- The main theorem --/
theorem probability_all_sides_of_decagon :
  (favorable_triangles : ℚ) / total_triangles = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_sides_of_decagon_l1861_186106


namespace NUMINAMATH_CALUDE_car_speed_problem_l1861_186128

/-- 
Given a car that travels for two hours, with speed x km/h in the first hour
and 60 km/h in the second hour, if the average speed is 79 km/h, 
then the speed x in the first hour must be 98 km/h.
-/
theorem car_speed_problem (x : ℝ) : 
  (x + 60) / 2 = 79 → x = 98 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1861_186128


namespace NUMINAMATH_CALUDE_quadratic_solution_l1861_186182

/-- A quadratic function passing through (-3,0) and (4,0) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The equation we want to solve -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + c - (b - b * x)

theorem quadratic_solution (a b c : ℝ) (h1 : f a b c (-3) = 0) (h2 : f a b c 4 = 0) :
  (∀ x : ℝ, g a b c x = 0 ↔ x = -2 ∨ x = 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1861_186182


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1861_186167

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52) 
    (Finset.range 200)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1861_186167


namespace NUMINAMATH_CALUDE_root_in_interval_l1861_186117

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) : ℝ := x^3 - x - 3

-- State the theorem
theorem root_in_interval :
  Continuous f ∧ f 1 < 0 ∧ 0 < f 2 →
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1861_186117


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_result_l1861_186141

/-- Calculates the length of a bridge given train specifications --/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length

/-- The length of the bridge is approximately 299.95 meters --/
theorem bridge_length_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length_calculation 200 45 40 - 299.95| < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_result_l1861_186141


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1861_186150

theorem quadratic_equation_solution :
  let x₁ : ℝ := (-1 + Real.sqrt 5) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 5) / 2
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1861_186150


namespace NUMINAMATH_CALUDE_bookstore_problem_l1861_186175

/-- Represents the number of magazine types at each price point -/
structure MagazineTypes :=
  (twoYuan : ℕ)
  (oneYuan : ℕ)

/-- Represents the total budget and purchasing constraints -/
structure PurchaseConstraints :=
  (budget : ℕ)
  (maxPerType : ℕ)

/-- Calculates the number of different purchasing methods -/
def purchasingMethods (types : MagazineTypes) (constraints : PurchaseConstraints) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem bookstore_problem :
  let types := MagazineTypes.mk 8 3
  let constraints := PurchaseConstraints.mk 10 1
  purchasingMethods types constraints = 266 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_problem_l1861_186175


namespace NUMINAMATH_CALUDE_water_tank_full_time_l1861_186169

/-- Represents the state of a water tank system with three pipes -/
structure WaterTankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the net water change after one cycle -/
def net_change_per_cycle (system : WaterTankSystem) : ℤ :=
  system.pipeA_rate + system.pipeB_rate - system.pipeC_rate

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : WaterTankSystem) : ℕ :=
  (system.capacity / (net_change_per_cycle system).natAbs) * 3

/-- Theorem stating that the given water tank system will be full after 48 minutes -/
theorem water_tank_full_time (system : WaterTankSystem) 
  (h1 : system.capacity = 800)
  (h2 : system.pipeA_rate = 40)
  (h3 : system.pipeB_rate = 30)
  (h4 : system.pipeC_rate = 20) : 
  time_to_fill system = 48 := by
  sorry

#eval time_to_fill { capacity := 800, pipeA_rate := 40, pipeB_rate := 30, pipeC_rate := 20 }

end NUMINAMATH_CALUDE_water_tank_full_time_l1861_186169


namespace NUMINAMATH_CALUDE_emily_egg_collection_l1861_186111

theorem emily_egg_collection (num_baskets : ℕ) (eggs_per_basket : ℕ) 
  (h1 : num_baskets = 303) 
  (h2 : eggs_per_basket = 28) : 
  num_baskets * eggs_per_basket = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l1861_186111


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1861_186135

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (8 + t * Complex.I) = 15) ↔ t = Real.sqrt 161 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1861_186135
