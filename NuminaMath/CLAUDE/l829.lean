import Mathlib

namespace NUMINAMATH_CALUDE_shaded_areas_sum_l829_82957

theorem shaded_areas_sum (R : ℝ) (h1 : R > 0) (h2 : π * R^2 = 81 * π) : 
  (π * R^2) / 2 + (π * (R/2)^2) / 2 = 50.625 * π := by sorry

end NUMINAMATH_CALUDE_shaded_areas_sum_l829_82957


namespace NUMINAMATH_CALUDE_x_squared_plus_2xy_range_l829_82967

theorem x_squared_plus_2xy_range :
  ∀ x y : ℝ, x^2 + y^2 = 1 →
  (∃ (z : ℝ), z = x^2 + 2*x*y ∧ 1/2 - Real.sqrt 5 / 2 ≤ z ∧ z ≤ 1/2 + Real.sqrt 5 / 2) ∧
  (∃ (a b : ℝ), a = x^2 + 2*x*y ∧ b = x^2 + 2*x*y ∧ 
   a = 1/2 - Real.sqrt 5 / 2 ∧ b = 1/2 + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_2xy_range_l829_82967


namespace NUMINAMATH_CALUDE_part_one_part_two_l829_82936

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem part_one :
  ∀ x : ℝ, (1 < x ∧ x < 3) ∧ (2 < x ∧ x ≤ 3) ↔ (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, a > 0 →
  ((∀ x : ℝ, 2 < x ∧ x ≤ 3 → a < x ∧ x < 3*a) ∧
   (∃ x : ℝ, a < x ∧ x < 3*a ∧ ¬(2 < x ∧ x ≤ 3))) →
  (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l829_82936


namespace NUMINAMATH_CALUDE_tv_price_with_tax_l829_82913

/-- Calculates the final price of a TV including value-added tax -/
theorem tv_price_with_tax (original_price : ℝ) (tax_rate : ℝ) (final_price : ℝ) :
  original_price = 1700 →
  tax_rate = 0.15 →
  final_price = original_price * (1 + tax_rate) →
  final_price = 1955 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_with_tax_l829_82913


namespace NUMINAMATH_CALUDE_f_negative_when_x_greater_half_l829_82972

/-- The linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- When x > 1/2, f(x) < 0 -/
theorem f_negative_when_x_greater_half : ∀ x : ℝ, x > (1/2) → f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_when_x_greater_half_l829_82972


namespace NUMINAMATH_CALUDE_no_power_ending_222_l829_82998

theorem no_power_ending_222 :
  ¬ ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ ∃ (n : ℕ), x^y = 1000*n + 222 :=
sorry

end NUMINAMATH_CALUDE_no_power_ending_222_l829_82998


namespace NUMINAMATH_CALUDE_arrangement_counts_l829_82975

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 4

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

theorem arrangement_counts :
  (∃ (n₁ n₂ n₃ n₄ : ℕ),
    /- (1) Person A and Person B at the ends -/
    n₁ = 240 ∧
    /- (2) All male students grouped together -/
    n₂ = 720 ∧
    /- (3) No male students next to each other -/
    n₃ = 1440 ∧
    /- (4) Exactly one person between Person A and Person B -/
    n₄ = 1200 ∧
    /- The numbers represent valid arrangement counts -/
    n₁ > 0 ∧ n₂ > 0 ∧ n₃ > 0 ∧ n₄ > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l829_82975


namespace NUMINAMATH_CALUDE_officer_selection_count_correct_l829_82987

/-- The number of members in the club -/
def club_size : Nat := 12

/-- The number of officer positions to be filled -/
def officer_positions : Nat := 5

/-- The number of ways to choose officers from club members -/
def officer_selection_count : Nat := 95040

/-- Theorem stating that the number of ways to choose officers is correct -/
theorem officer_selection_count_correct :
  (club_size.factorial) / ((club_size - officer_positions).factorial) = officer_selection_count := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_correct_l829_82987


namespace NUMINAMATH_CALUDE_shortest_side_length_l829_82960

theorem shortest_side_length (a b c : ℝ) : 
  a + b + c = 15 ∧ a = 2 * c ∧ b = 2 * c → c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l829_82960


namespace NUMINAMATH_CALUDE_water_formed_ethanol_combustion_l829_82923

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation :=
  (reactants : List (String × ℚ))
  (products : List (String × ℚ))

/-- Represents the available moles of reactants -/
structure AvailableReactants :=
  (ethanol : ℚ)
  (oxygen : ℚ)

/-- The balanced chemical equation for ethanol combustion -/
def ethanolCombustion : ChemicalEquation :=
  { reactants := [("C2H5OH", 1), ("O2", 3)],
    products := [("CO2", 2), ("H2O", 3)] }

/-- Calculates the amount of H2O formed in the ethanol combustion reaction -/
def waterFormed (available : AvailableReactants) (equation : ChemicalEquation) : ℚ :=
  sorry

/-- Theorem stating that 2 moles of H2O are formed when 2 moles of ethanol react with 2 moles of oxygen -/
theorem water_formed_ethanol_combustion :
  waterFormed { ethanol := 2, oxygen := 2 } ethanolCombustion = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_ethanol_combustion_l829_82923


namespace NUMINAMATH_CALUDE_smallest_integer_result_l829_82996

def expression : List ℕ := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

def is_valid_bracketing (b : List (List ℕ)) : Prop :=
  b.join = expression ∧ ∀ l ∈ b, l.length > 0

def evaluate_bracketing (b : List (List ℕ)) : ℚ :=
  b.foldl (λ acc l => acc / l.foldl (λ x y => x / y) 1) 1

def is_integer_result (b : List (List ℕ)) : Prop :=
  ∃ n : ℤ, (evaluate_bracketing b).num = n * (evaluate_bracketing b).den

theorem smallest_integer_result :
  ∃ b : List (List ℕ),
    is_valid_bracketing b ∧
    is_integer_result b ∧
    evaluate_bracketing b = 7 ∧
    ∀ b' : List (List ℕ),
      is_valid_bracketing b' →
      is_integer_result b' →
      evaluate_bracketing b' ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_result_l829_82996


namespace NUMINAMATH_CALUDE_max_sum_of_exponents_l829_82906

theorem max_sum_of_exponents (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) :
  x + y ≤ -2 ∧ ∃ (a b : ℝ), (2 : ℝ)^a + (2 : ℝ)^b = 1 ∧ a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_exponents_l829_82906


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l829_82983

def volleyball_team_size : ℕ := 10
def lineup_size : ℕ := 5

theorem volleyball_lineup_combinations :
  (volleyball_team_size.factorial) / ((volleyball_team_size - lineup_size).factorial) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l829_82983


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l829_82997

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023) ↔
    ((x = 3 ∧ y = 3 ∧ z = 2) ∨
     (x = 3 ∧ y = 2 ∧ z = 3) ∨
     (x = 2 ∧ y = 3 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l829_82997


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l829_82961

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x| + 1

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l829_82961


namespace NUMINAMATH_CALUDE_angle_equation_solution_l829_82979

theorem angle_equation_solution (A : Real) :
  (1/2 * Real.sin (A/2) + Real.cos (A/2) = 1) → A = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_solution_l829_82979


namespace NUMINAMATH_CALUDE_lcm_prime_sum_l829_82952

theorem lcm_prime_sum (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x > y → Nat.lcm x y = 10 → 2 * x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_prime_sum_l829_82952


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l829_82905

theorem arithmetic_mean_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l829_82905


namespace NUMINAMATH_CALUDE_twentieth_base4_is_110_l829_82959

/-- Converts a decimal number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The 20th number in the base-4 system -/
def twentieth_base4 : List ℕ := toBase4 20

theorem twentieth_base4_is_110 : twentieth_base4 = [1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_twentieth_base4_is_110_l829_82959


namespace NUMINAMATH_CALUDE_gcf_of_40_and_56_l829_82966

theorem gcf_of_40_and_56 : Nat.gcd 40 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_40_and_56_l829_82966


namespace NUMINAMATH_CALUDE_security_code_combinations_l829_82993

theorem security_code_combinations : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_security_code_combinations_l829_82993


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l829_82954

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2*x - 1)
  f (1/2) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l829_82954


namespace NUMINAMATH_CALUDE_cafeteria_milk_stacks_l829_82976

/-- Given a total number of cartons and the number of cartons per stack, 
    calculate the maximum number of full stacks that can be made. -/
def maxFullStacks (totalCartons : ℕ) (cartonsPerStack : ℕ) : ℕ :=
  totalCartons / cartonsPerStack

theorem cafeteria_milk_stacks : maxFullStacks 799 6 = 133 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_milk_stacks_l829_82976


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l829_82931

theorem no_solution_absolute_value_equation :
  (∀ x : ℝ, (x - 3)^2 ≠ 0 → False) ∧
  (∀ x : ℝ, |2*x| + 4 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (3*x) - 1 ≠ 0 → False) ∧
  (∀ x : ℝ, x ≤ 0 → Real.sqrt (-3*x) - 3 ≠ 0 → False) ∧
  (∀ x : ℝ, |5*x| - 6 ≠ 0 → False) := by
  sorry

#check no_solution_absolute_value_equation

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l829_82931


namespace NUMINAMATH_CALUDE_rotation_of_A_about_B_l829_82904

-- Define the points
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)

-- Define the rotation function
def rotate180AboutPoint (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (cx, cy) := center
  (2 * cx - px, 2 * cy - py)

-- Theorem statement
theorem rotation_of_A_about_B :
  rotate180AboutPoint A B = (2, 7) := by sorry

end NUMINAMATH_CALUDE_rotation_of_A_about_B_l829_82904


namespace NUMINAMATH_CALUDE_least_divisible_by_3_4_5_6_8_divisible_by_3_4_5_6_8_120_least_number_120_l829_82907

theorem least_divisible_by_3_4_5_6_8 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) → n ≥ 120 :=
by
  sorry

theorem divisible_by_3_4_5_6_8_120 : (3 ∣ 120) ∧ (4 ∣ 120) ∧ (5 ∣ 120) ∧ (6 ∣ 120) ∧ (8 ∣ 120) :=
by
  sorry

theorem least_number_120 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) → n = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_3_4_5_6_8_divisible_by_3_4_5_6_8_120_least_number_120_l829_82907


namespace NUMINAMATH_CALUDE_largest_product_of_three_l829_82973

def S : Set ℤ := {-3, -2, 4, 5}

theorem largest_product_of_three (a b c : ℤ) :
  a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : ℤ, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ 30 ∧ (∃ p q r : ℤ, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p * q * r = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l829_82973


namespace NUMINAMATH_CALUDE_flour_measurement_l829_82981

theorem flour_measurement (flour_needed : ℚ) (cup_capacity : ℚ) : 
  flour_needed = 4 + 3 / 4 →
  cup_capacity = 1 / 2 →
  ⌈flour_needed / cup_capacity⌉ = 10 := by
  sorry

end NUMINAMATH_CALUDE_flour_measurement_l829_82981


namespace NUMINAMATH_CALUDE_fractional_equation_root_l829_82941

theorem fractional_equation_root (k : ℚ) : 
  (∃ x : ℚ, x ≠ 1 ∧ (2 * k) / (x - 1) - 3 / (1 - x) = 1) → k = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l829_82941


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_and_product_l829_82925

theorem quadratic_roots_sum_squares_and_product (u v : ℝ) : 
  u^2 - 5*u + 3 = 0 → v^2 - 5*v + 3 = 0 → u^2 + v^2 + u*v = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_and_product_l829_82925


namespace NUMINAMATH_CALUDE_unique_n_mod_10_l829_82946

theorem unique_n_mod_10 : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4000 [ZMOD 10] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_mod_10_l829_82946


namespace NUMINAMATH_CALUDE_jim_tree_planting_l829_82968

/-- The age at which Jim started planting a new row of trees every year -/
def start_age : ℕ := sorry

/-- The number of trees Jim started with -/
def initial_trees : ℕ := 2 * 4

/-- The number of trees in each new row -/
def trees_per_row : ℕ := 4

/-- Jim's age when he doubles his trees -/
def doubling_age : ℕ := 15

/-- The total number of trees after doubling -/
def total_trees : ℕ := 56

theorem jim_tree_planting :
  2 * (initial_trees + trees_per_row * (doubling_age - start_age)) = total_trees :=
sorry

end NUMINAMATH_CALUDE_jim_tree_planting_l829_82968


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l829_82920

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (40.9 * 1000000000) =
    ScientificNotation.mk 4.09 9 (by sorry) := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l829_82920


namespace NUMINAMATH_CALUDE_apple_distribution_theorem_l829_82932

def distribute_apples (total_apples : ℕ) (num_people : ℕ) (min_apples : ℕ) : ℕ :=
  Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)

theorem apple_distribution_theorem :
  distribute_apples 30 3 3 = 253 :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_theorem_l829_82932


namespace NUMINAMATH_CALUDE_curve_condition_iff_l829_82953

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A curve defined by a function f(x, y) = 0 -/
structure Curve where
  f : ℝ → ℝ → ℝ

/-- Predicate for a point being on a curve -/
def IsOnCurve (p : Point) (c : Curve) : Prop :=
  c.f p.x p.y = 0

/-- Theorem stating that f(x, y) = 0 is a necessary and sufficient condition
    for a point P(x, y) to be on the curve f(x, y) = 0 -/
theorem curve_condition_iff (c : Curve) (p : Point) :
  IsOnCurve p c ↔ c.f p.x p.y = 0 := by sorry

end NUMINAMATH_CALUDE_curve_condition_iff_l829_82953


namespace NUMINAMATH_CALUDE_empty_bottle_weight_l829_82910

/-- Given a full bottle of sesame oil weighing 3.4 kg and the same bottle weighing 2.98 kg
    after using 1/5 of the oil, the weight of the empty bottle is 1.3 kg. -/
theorem empty_bottle_weight (full_weight : ℝ) (partial_weight : ℝ) (empty_weight : ℝ) : 
  full_weight = 3.4 →
  partial_weight = 2.98 →
  full_weight = empty_weight + (5/4) * (partial_weight - empty_weight) →
  empty_weight = 1.3 := by
  sorry

#check empty_bottle_weight

end NUMINAMATH_CALUDE_empty_bottle_weight_l829_82910


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l829_82989

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the inequality function
def inequality_function (t : Triangle) : ℝ :=
  t.a^2 * t.b * (t.a - t.b) + t.b^2 * t.c * (t.b - t.c) + t.c^2 * t.a * (t.c - t.a)

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  inequality_function t ≥ 0 ∧
  (inequality_function t = 0 ↔ t.a = t.b ∧ t.b = t.c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l829_82989


namespace NUMINAMATH_CALUDE_smallest_class_size_exists_class_size_l829_82977

theorem smallest_class_size (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 8 = 5) → n ≥ 53 :=
by sorry

theorem exists_class_size : 
  ∃ n : ℕ, (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 8 = 5) ∧ n = 53 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_exists_class_size_l829_82977


namespace NUMINAMATH_CALUDE_ellipse_intersection_properties_l829_82943

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the upper vertex A
def A : ℝ × ℝ := (0, 1)

-- Define a line not passing through A
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the condition that the line intersects the ellipse at P and Q
def intersects (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    y₁ = line k m x₁ ∧ y₂ = line k m x₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - A.1) * (x₂ - A.1) + (y₁ - A.2) * (y₂ - A.2) = 0

-- Main theorem
theorem ellipse_intersection_properties :
  ∀ (k m : ℝ),
    intersects k m →
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
      y₁ = line k m x₁ ∧ y₂ = line k m x₂ →
      perpendicular x₁ y₁ x₂ y₂) →
    (m = -1/2) ∧
    (∃ (S : Set ℝ), S = {s | s ≥ 9/4} ∧
      ∀ (x₁ y₁ x₂ y₂ : ℝ),
        ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
        y₁ = line k m x₁ ∧ y₂ = line k m x₂ →
        ∃ (area : ℝ), area ∈ S ∧
          (∃ (Bx By : ℝ), area = 1/2 * |Bx - x₁| * |By - y₁|)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_properties_l829_82943


namespace NUMINAMATH_CALUDE_prism_surface_area_l829_82929

/-- A right rectangular prism with integer dimensions -/
structure RectPrism where
  l : ℕ
  w : ℕ
  h : ℕ
  l_ne_w : l ≠ w
  w_ne_h : w ≠ h
  h_ne_l : h ≠ l

/-- The processing fee calculation function -/
def processingFee (p : RectPrism) : ℚ :=
  0.3 * p.l + 0.4 * p.w + 0.5 * p.h

/-- The surface area calculation function -/
def surfaceArea (p : RectPrism) : ℕ :=
  2 * (p.l * p.w + p.l * p.h + p.w * p.h)

/-- The main theorem -/
theorem prism_surface_area (p : RectPrism) :
  (∃ (σ₁ σ₂ σ₃ σ₄ : Equiv.Perm (Fin 3)),
    3 * (σ₁.toFun 0 : ℕ) + 4 * (σ₁.toFun 1 : ℕ) + 5 * (σ₁.toFun 2 : ℕ) = 81 ∧
    3 * (σ₂.toFun 0 : ℕ) + 4 * (σ₂.toFun 1 : ℕ) + 5 * (σ₂.toFun 2 : ℕ) = 81 ∧
    3 * (σ₃.toFun 0 : ℕ) + 4 * (σ₃.toFun 1 : ℕ) + 5 * (σ₃.toFun 2 : ℕ) = 87 ∧
    3 * (σ₄.toFun 0 : ℕ) + 4 * (σ₄.toFun 1 : ℕ) + 5 * (σ₄.toFun 2 : ℕ) = 87) →
  surfaceArea p = 276 := by
  sorry


end NUMINAMATH_CALUDE_prism_surface_area_l829_82929


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l829_82927

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l829_82927


namespace NUMINAMATH_CALUDE_digit_property_l829_82980

theorem digit_property (z : Nat) :
  (z < 10) →
  (∀ k : Nat, k ≥ 1 → ∃ n : Nat, n ≥ 1 ∧ n^9 % (10^k) = z^k % (10^k)) ↔
  z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9 :=
by sorry

end NUMINAMATH_CALUDE_digit_property_l829_82980


namespace NUMINAMATH_CALUDE_stratified_sampling_grade10_l829_82940

theorem stratified_sampling_grade10 (total_students : ℕ) (grade10_students : ℕ) (sample_size : ℕ) :
  total_students = 700 →
  grade10_students = 300 →
  sample_size = 35 →
  (grade10_students * sample_size) / total_students = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_grade10_l829_82940


namespace NUMINAMATH_CALUDE_crate_dimensions_for_largest_tank_l829_82903

/-- Represents a rectangular crate with length, width, and height -/
structure RectangularCrate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank with radius and height -/
structure CylindricalTank where
  radius : ℝ
  height : ℝ

/-- The tank fits in the crate when standing upright -/
def tankFitsInCrate (tank : CylindricalTank) (crate : RectangularCrate) : Prop :=
  2 * tank.radius ≤ min crate.length crate.width ∧ tank.height ≤ crate.height

theorem crate_dimensions_for_largest_tank (crate : RectangularCrate) 
    (h : ∃ tank : CylindricalTank, tank.radius = 10 ∧ tankFitsInCrate tank crate) :
    crate.length ≥ 20 ∧ crate.width ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_crate_dimensions_for_largest_tank_l829_82903


namespace NUMINAMATH_CALUDE_paper_clip_cost_l829_82999

/-- The cost of Eldora's purchase -/
def eldora_cost : ℝ := 55.40

/-- The cost of Finn's purchase -/
def finn_cost : ℝ := 61.70

/-- The number of paper clip boxes Eldora bought -/
def eldora_clips : ℕ := 15

/-- The number of index card packages Eldora bought -/
def eldora_cards : ℕ := 7

/-- The number of paper clip boxes Finn bought -/
def finn_clips : ℕ := 12

/-- The number of index card packages Finn bought -/
def finn_cards : ℕ := 10

/-- The cost of one box of paper clips -/
noncomputable def clip_cost : ℝ := 1.835

theorem paper_clip_cost : 
  ∃ (card_cost : ℝ), 
    (eldora_clips : ℝ) * clip_cost + (eldora_cards : ℝ) * card_cost = eldora_cost ∧ 
    (finn_clips : ℝ) * clip_cost + (finn_cards : ℝ) * card_cost = finn_cost :=
by sorry

end NUMINAMATH_CALUDE_paper_clip_cost_l829_82999


namespace NUMINAMATH_CALUDE_fraction_equivalence_l829_82908

theorem fraction_equivalence (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∀ x : ℝ, x ≠ -c → x ≠ -3*c → (x + 2*b) / (x + 3*c) = (x + b) / (x + c)) ↔ b = 2*c :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l829_82908


namespace NUMINAMATH_CALUDE_inequality_proof_l829_82951

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l829_82951


namespace NUMINAMATH_CALUDE_equation_solution_l829_82939

def solution_set : Set (ℕ × ℕ) :=
  {(150, 150), (150, 145), (145, 135), (135, 120), (120, 100), (100, 75),
   (75, 45), (45, 10), (145, 150), (135, 145), (120, 135), (100, 120),
   (75, 100), (45, 75), (10, 45)}

def satisfies_equation (p : ℕ × ℕ) : Prop :=
  let (x, y) := p
  x^2 - 2*x*y + y^2 + 5*x + 5*y = 1500

theorem equation_solution :
  ∀ p : ℕ × ℕ, p ∈ solution_set ↔ satisfies_equation p :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l829_82939


namespace NUMINAMATH_CALUDE_ben_win_probability_l829_82963

theorem ben_win_probability (lose_prob : ℚ) (win_prob : ℚ) : 
  lose_prob = 5/8 → win_prob = 1 - lose_prob → win_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l829_82963


namespace NUMINAMATH_CALUDE_ivanov_family_problem_l829_82974

/-- The Ivanov family problem -/
theorem ivanov_family_problem (father mother daughter : ℕ) : 
  father + mother + daughter = 74 →  -- Current sum of ages
  father + mother + daughter - 30 = 47 →  -- Sum of ages 10 years ago
  mother - 26 = daughter →  -- Mother's age at daughter's birth
  mother = 33 := by
  sorry

end NUMINAMATH_CALUDE_ivanov_family_problem_l829_82974


namespace NUMINAMATH_CALUDE_power_sum_equality_l829_82916

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l829_82916


namespace NUMINAMATH_CALUDE_sqrt_decimal_movement_l829_82984

theorem sqrt_decimal_movement (a b : ℝ) (n : ℤ) (h : Real.sqrt a = b) :
  Real.sqrt (a * (10 : ℝ)^(2*n)) = b * (10 : ℝ)^n := by sorry

end NUMINAMATH_CALUDE_sqrt_decimal_movement_l829_82984


namespace NUMINAMATH_CALUDE_factor_polynomial_implies_specific_c_l829_82969

theorem factor_polynomial_implies_specific_c (c d : ℤ) :
  (∀ x : ℝ, (x^2 - x - 1) ∣ (c * x^18 + d * x^17 + x^2 + 1)) →
  c = -1597 :=
by sorry

end NUMINAMATH_CALUDE_factor_polynomial_implies_specific_c_l829_82969


namespace NUMINAMATH_CALUDE_sum_of_products_l829_82986

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 124) :
  x*y + y*z + x*z = 48 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l829_82986


namespace NUMINAMATH_CALUDE_sum_reciprocal_plus_one_bounds_l829_82985

theorem sum_reciprocal_plus_one_bounds (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < (1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z)) ∧ 
  (1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z)) < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_plus_one_bounds_l829_82985


namespace NUMINAMATH_CALUDE_sum_of_squares_orthogonal_matrix_l829_82901

theorem sum_of_squares_orthogonal_matrix (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A.transpose = A⁻¹) : 
  (A 0 0)^2 + (A 0 1)^2 + (A 0 2)^2 + 
  (A 1 0)^2 + (A 1 1)^2 + (A 1 2)^2 + 
  (A 2 0)^2 + (A 2 1)^2 + (A 2 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_orthogonal_matrix_l829_82901


namespace NUMINAMATH_CALUDE_quadratic_roots_sine_cosine_l829_82945

theorem quadratic_roots_sine_cosine (α : Real) (c : Real) :
  (∃ (x y : Real), x = Real.sin α ∧ y = Real.cos α ∧ 
   10 * x^2 - 7 * x - c = 0 ∧ 10 * y^2 - 7 * y - c = 0) →
  c = 2.55 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sine_cosine_l829_82945


namespace NUMINAMATH_CALUDE_triangle_inequalities_l829_82912

/-- Triangle inequalities -/
theorem triangle_inequalities (a b c P S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : P = a + b + c)
  (h_area : S = Real.sqrt ((P/2) * ((P/2) - a) * ((P/2) - b) * ((P/2) - c))) :
  (1/a + 1/b + 1/c ≥ 9/P) ∧
  (a^2 + b^2 + c^2 ≥ P^2/3) ∧
  (P^2 ≥ 12 * Real.sqrt 3 * S) ∧
  (a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) ∧
  (a^3 + b^3 + c^3 ≥ P^3/9) ∧
  (a^3 + b^3 + c^3 ≥ (4 * Real.sqrt 3 / 3) * S * P) ∧
  (a^4 + b^4 + c^4 ≥ 16 * S^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l829_82912


namespace NUMINAMATH_CALUDE_factorization_implies_sum_l829_82934

theorem factorization_implies_sum (C D : ℤ) :
  (∀ y : ℝ, 6 * y^2 - 31 * y + 35 = (C * y - 5) * (D * y - 7)) →
  C * D + C = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorization_implies_sum_l829_82934


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l829_82933

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + a*x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 := by
  sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l829_82933


namespace NUMINAMATH_CALUDE_P_subset_Q_l829_82918

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem P_subset_Q : P ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_P_subset_Q_l829_82918


namespace NUMINAMATH_CALUDE_third_month_sale_l829_82947

/-- Calculates the missing sale amount given the average sale and other known sales. -/
def calculate_missing_sale (average : ℕ) (num_months : ℕ) (known_sales : List ℕ) : ℕ :=
  average * num_months - known_sales.sum

/-- The problem statement -/
theorem third_month_sale (average : ℕ) (num_months : ℕ) (known_sales : List ℕ) :
  average = 5600 ∧ 
  num_months = 6 ∧ 
  known_sales = [5266, 5768, 5678, 6029, 4937] →
  calculate_missing_sale average num_months known_sales = 5922 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l829_82947


namespace NUMINAMATH_CALUDE_sum_of_second_progression_l829_82982

/-- Given two arithmetic progressions with specific conditions, prove that the sum of the terms of the second progression is 14. -/
theorem sum_of_second_progression (a₁ a₅ b₁ bₙ : ℚ) (N : ℕ) : 
  a₁ = 7 →
  a₅ = -5 →
  b₁ = 0 →
  bₙ = 7/2 →
  N > 1 →
  (∃ d D : ℚ, a₁ + 2*d = b₁ + 2*D ∧ a₅ = a₁ + 4*d ∧ bₙ = b₁ + (N-1)*D) →
  (N/2 : ℚ) * (b₁ + bₙ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_second_progression_l829_82982


namespace NUMINAMATH_CALUDE_xy_divided_by_three_l829_82978

theorem xy_divided_by_three (x y : ℚ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : x + 2 * y = 5) : 
  (x + y) / 3 = 1.222222222222222 := by
sorry

end NUMINAMATH_CALUDE_xy_divided_by_three_l829_82978


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l829_82922

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ a ≥ (1/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l829_82922


namespace NUMINAMATH_CALUDE_integers_between_sqrt3_and_sqrt13_two_and_three_between_sqrt3_and_sqrt13_only_two_and_three_between_sqrt3_and_sqrt13_l829_82924

theorem integers_between_sqrt3_and_sqrt13 :
  ∃ (n : ℤ), (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < Real.sqrt 13 :=
by
  sorry

theorem two_and_three_between_sqrt3_and_sqrt13 :
  (2 : ℝ) > Real.sqrt 3 ∧ (2 : ℝ) < Real.sqrt 13 ∧
  (3 : ℝ) > Real.sqrt 3 ∧ (3 : ℝ) < Real.sqrt 13 :=
by
  sorry

theorem only_two_and_three_between_sqrt3_and_sqrt13 :
  ∀ (n : ℤ), (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < Real.sqrt 13 → n = 2 ∨ n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_integers_between_sqrt3_and_sqrt13_two_and_three_between_sqrt3_and_sqrt13_only_two_and_three_between_sqrt3_and_sqrt13_l829_82924


namespace NUMINAMATH_CALUDE_prob_one_pascal_20_l829_82995

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle -/
def pascal_triangle_ones (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def prob_one_pascal (n : ℕ) : ℚ :=
  (pascal_triangle_ones n) / (pascal_triangle_elements n)

theorem prob_one_pascal_20 :
  prob_one_pascal 20 = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_pascal_20_l829_82995


namespace NUMINAMATH_CALUDE_matrix_commutation_result_l829_82942

/-- Given two 2x2 matrices A and B that commute, prove that (2a - 3d) / (4b - 3c) = -3 --/
theorem matrix_commutation_result (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  4 * b ≠ 3 * c →
  A * B = B * A →
  (2 * a - 3 * d) / (4 * b - 3 * c) = -3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_result_l829_82942


namespace NUMINAMATH_CALUDE_election_winner_votes_l829_82900

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 65 / 100 →
  vote_difference = 300 →
  winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference →
  winner_percentage * total_votes = 650 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l829_82900


namespace NUMINAMATH_CALUDE_cookie_sales_problem_l829_82921

/-- Represents the number of boxes of cookies sold -/
structure CookieSales where
  chocolate : ℕ
  plain : ℕ

/-- Represents the price of cookies in cents -/
def CookiePrice : ℕ × ℕ := (125, 75)

theorem cookie_sales_problem (sales : CookieSales) : 
  sales.chocolate + sales.plain = 1585 →
  125 * sales.chocolate + 75 * sales.plain = 158675 →
  sales.plain = 789 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_problem_l829_82921


namespace NUMINAMATH_CALUDE_equation_solutions_l829_82909

theorem equation_solutions : 
  {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ 2*x^2 + 5*x*y + 2*y^2 = 2006} = 
  {(28, 3), (3, 28)} :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l829_82909


namespace NUMINAMATH_CALUDE_faye_remaining_money_is_correct_l829_82949

/-- Calculates Faye's remaining money after her shopping spree -/
def faye_remaining_money (original_money : ℚ) : ℚ :=
  let father_gift := 3 * original_money
  let mother_gift := 2 * father_gift
  let grandfather_gift := 4 * original_money
  let total_money := original_money + father_gift + mother_gift + grandfather_gift
  let muffin_cost := 15 * 1.75
  let cookie_cost := 10 * 2.5
  let juice_cost := 2 * 4
  let candy_cost := 25 * 0.25
  let total_item_cost := muffin_cost + cookie_cost + juice_cost + candy_cost
  let tip := 0.15 * (muffin_cost + cookie_cost)
  let total_spent := total_item_cost + tip
  total_money - total_spent

theorem faye_remaining_money_is_correct : 
  faye_remaining_money 20 = 206.81 := by sorry

end NUMINAMATH_CALUDE_faye_remaining_money_is_correct_l829_82949


namespace NUMINAMATH_CALUDE_greg_initial_amount_l829_82988

/-- Represents the initial and final monetary states of Earl, Fred, and Greg -/
structure MonetaryState where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  earl_owes_fred : ℕ
  fred_owes_greg : ℕ
  greg_owes_earl : ℕ
  earl_final : ℕ
  fred_final : ℕ
  greg_final : ℕ

/-- The theorem states that given the initial conditions and debt payments,
    Greg's initial amount is 36 dollars -/
theorem greg_initial_amount (state : MonetaryState)
  (h1 : state.earl_initial = 90)
  (h2 : state.fred_initial = 48)
  (h3 : state.earl_owes_fred = 28)
  (h4 : state.fred_owes_greg = 32)
  (h5 : state.greg_owes_earl = 40)
  (h6 : state.earl_final + state.greg_final = 130)
  (h7 : state.earl_final = state.earl_initial - state.earl_owes_fred + state.greg_owes_earl)
  (h8 : state.fred_final = state.fred_initial + state.earl_owes_fred - state.fred_owes_greg)
  (h9 : state.greg_final = state.greg_initial + state.fred_owes_greg - state.greg_owes_earl) :
  state.greg_initial = 36 := by
  sorry


end NUMINAMATH_CALUDE_greg_initial_amount_l829_82988


namespace NUMINAMATH_CALUDE_curve_tangent_theorem_l829_82915

/-- A curve defined by y = x² + ax + b -/
def curve (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2*x + a

/-- The tangent line at x = 0 -/
def tangent_at_zero (a b : ℝ) : ℝ → ℝ := λ x ↦ 3*x - b + 1

theorem curve_tangent_theorem (a b : ℝ) :
  (∀ x, tangent_at_zero a b x = 3*x - (curve a b 0) + 1) →
  curve_derivative a 0 = 3 →
  a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_theorem_l829_82915


namespace NUMINAMATH_CALUDE_segment_and_polygon_inequalities_l829_82970

/-- Segment with projections a and b on perpendicular lines has length c -/
structure Segment where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Polygon with projections a and b on coordinate axes has perimeter P -/
structure Polygon where
  a : ℝ
  b : ℝ
  P : ℝ

/-- Theorem about segment length and polygon perimeter -/
theorem segment_and_polygon_inequalities 
  (s : Segment) (p : Polygon) : 
  s.c ≥ (s.a + s.b) / Real.sqrt 2 ∧ 
  p.P ≥ Real.sqrt 2 * (p.a + p.b) := by
  sorry


end NUMINAMATH_CALUDE_segment_and_polygon_inequalities_l829_82970


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l829_82956

/-- Given a line equation 3x + 5y + d = 0, proves that if the sum of x- and y-intercepts is 16, then d = -30 -/
theorem line_intercepts_sum (d : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l829_82956


namespace NUMINAMATH_CALUDE_determine_constant_b_l829_82994

theorem determine_constant_b (b c : ℝ) : 
  (∀ x, (3*x^2 - 4*x + 8/3)*(2*x^2 + b*x + c) = 6*x^4 - 17*x^3 + 21*x^2 - 16/3*x + 9/3) → 
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_determine_constant_b_l829_82994


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l829_82902

/-- The perimeter of a rectangle formed when a smaller square is cut from the corner of a larger square -/
theorem rectangle_perimeter (t s : ℝ) (h : t > s) : 2 * s + 2 * (t - s) = 2 * t := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l829_82902


namespace NUMINAMATH_CALUDE_bill_take_home_salary_l829_82919

def take_home_salary (gross_salary property_taxes sales_taxes income_tax_rate : ℝ) : ℝ :=
  gross_salary - (property_taxes + sales_taxes + income_tax_rate * gross_salary)

theorem bill_take_home_salary :
  take_home_salary 50000 2000 3000 0.1 = 40000 := by
  sorry

end NUMINAMATH_CALUDE_bill_take_home_salary_l829_82919


namespace NUMINAMATH_CALUDE_cube_root_increasing_l829_82955

/-- The cube root function is increasing on the real numbers. -/
theorem cube_root_increasing :
  ∀ x y : ℝ, x < y → x^(1/3) < y^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_increasing_l829_82955


namespace NUMINAMATH_CALUDE_g_is_odd_l829_82944

noncomputable def g (x : ℝ) : ℝ := Real.log (x^3 + Real.sqrt (1 + x^6))

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_g_is_odd_l829_82944


namespace NUMINAMATH_CALUDE_gcf_of_90_135_225_l829_82958

theorem gcf_of_90_135_225 : Nat.gcd 90 (Nat.gcd 135 225) = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_135_225_l829_82958


namespace NUMINAMATH_CALUDE_triangular_pizza_area_l829_82911

theorem triangular_pizza_area :
  ∀ (base height hypotenuse : ℝ),
  base = 9 →
  hypotenuse = 15 →
  base ^ 2 + height ^ 2 = hypotenuse ^ 2 →
  (base * height) / 2 = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_pizza_area_l829_82911


namespace NUMINAMATH_CALUDE_circle_tangent_to_lines_l829_82971

/-- The circle with center (1, 1) and radius √5 is tangent to both lines 2x - y + 4 = 0 and 2x - y - 6 = 0 -/
theorem circle_tangent_to_lines :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 5}
  let line1 := {(x, y) : ℝ × ℝ | 2*x - y + 4 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 2*x - y - 6 = 0}
  (∃ p ∈ circle ∩ line1, ∀ q ∈ circle, q ∉ line1 ∨ q = p) ∧
  (∃ p ∈ circle ∩ line2, ∀ q ∈ circle, q ∉ line2 ∨ q = p) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_lines_l829_82971


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l829_82928

theorem smallest_x_absolute_value (x : ℝ) : 
  (|x - 10| = 15) → (x ≥ -5 ∧ (∃ y : ℝ, |y - 10| = 15 ∧ y = -5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l829_82928


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l829_82935

theorem unique_quadratic_solution :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, q * x^2 - 8 * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l829_82935


namespace NUMINAMATH_CALUDE_equation_solutions_l829_82990

theorem equation_solutions :
  {x : ℝ | x * (2 * x + 1) = 2 * x + 1} = {-1/2, 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l829_82990


namespace NUMINAMATH_CALUDE_sum_of_products_l829_82948

-- Define the problem statement
theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 2)
  (eq2 : y^2 + y*z + z^2 = 5)
  (eq3 : z^2 + x*z + x^2 = 3) :
  x*y + y*z + x*z = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l829_82948


namespace NUMINAMATH_CALUDE_firefighter_water_delivery_time_l829_82930

/-- Proves that 5 firefighters can deliver 4000 gallons of water in 40 minutes -/
theorem firefighter_water_delivery_time :
  let water_needed : ℕ := 4000
  let firefighters : ℕ := 5
  let water_per_minute_per_hose : ℕ := 20
  let total_water_per_minute : ℕ := firefighters * water_per_minute_per_hose
  water_needed / total_water_per_minute = 40 := by
  sorry

end NUMINAMATH_CALUDE_firefighter_water_delivery_time_l829_82930


namespace NUMINAMATH_CALUDE_school_population_l829_82950

/-- Given a school with boys, girls, and teachers, prove the total population. -/
theorem school_population (b g t : ℕ) : 
  b = 4 * g ∧ g = 8 * t → b + g + t = (41 * b) / 32 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l829_82950


namespace NUMINAMATH_CALUDE_binary_101_to_decimal_l829_82926

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101_to_decimal :
  binary_to_decimal [true, false, true] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_to_decimal_l829_82926


namespace NUMINAMATH_CALUDE_representative_selection_count_l829_82992

def female_students : ℕ := 5
def male_students : ℕ := 7
def total_representatives : ℕ := 5
def max_female_representatives : ℕ := 2

theorem representative_selection_count :
  (Nat.choose male_students total_representatives) +
  (Nat.choose female_students 1 * Nat.choose male_students 4) +
  (Nat.choose female_students 2 * Nat.choose male_students 3) = 546 := by
  sorry

end NUMINAMATH_CALUDE_representative_selection_count_l829_82992


namespace NUMINAMATH_CALUDE_equation_solution_l829_82938

theorem equation_solution :
  ∃! x : ℝ, 5 * x + 4 = -6 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l829_82938


namespace NUMINAMATH_CALUDE_square_area_increase_l829_82991

/-- The increase in area of a square when its side length increases by 6 -/
theorem square_area_increase (a : ℝ) : 
  (a + 6)^2 - a^2 = 12*a + 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l829_82991


namespace NUMINAMATH_CALUDE_son_age_problem_l829_82962

theorem son_age_problem (son_age man_age : ℕ) : 
  man_age = son_age + 40 →
  man_age + 6 = 3 * (son_age + 6) →
  son_age = 14 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l829_82962


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l829_82917

theorem fraction_equality_solution : ∃! x : ℝ, (4 + x) / (6 + x) = (2 + x) / (3 + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l829_82917


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l829_82964

/-- Two lines in the form A₁x + B₁y + C₁ = 0 and A₂x + B₂y + C₂ = 0 are perpendicular -/
def are_perpendicular (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  A₁ * A₂ + B₁ * B₂ = 0

/-- The theorem stating the necessary and sufficient condition for two lines to be perpendicular -/
theorem perpendicular_lines_condition
  (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) :
  (∃ x y : ℝ, A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) →
  (are_perpendicular A₁ B₁ C₁ A₂ B₂ C₂ ↔ 
   ∀ x₁ y₁ x₂ y₂ : ℝ, 
   A₁ * x₁ + B₁ * y₁ + C₁ = 0 ∧ 
   A₁ * x₂ + B₁ * y₂ + C₁ = 0 ∧ 
   A₂ * x₁ + B₂ * y₁ + C₂ = 0 ∧ 
   A₂ * x₂ + B₂ * y₂ + C₂ = 0 →
   (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
   ((x₂ - x₁) * (y₂ - y₁) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l829_82964


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l829_82965

theorem polynomial_division_theorem (x : ℝ) : 
  8 * x^3 + 4 * x^2 - 6 * x - 9 = (x + 3) * (8 * x^2 - 20 * x + 54) - 171 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l829_82965


namespace NUMINAMATH_CALUDE_max_min_product_l829_82914

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 12) (hprod : x * y + y * z + z * x = 30) :
  ∃ (n : ℝ), n = min (x * y) (min (y * z) (z * x)) ∧ n ≤ 2 ∧
  ∀ (m : ℝ), m = min (x * y) (min (y * z) (z * x)) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l829_82914


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l829_82937

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid :=
  (longer_base : ℝ)
  (base_angle : ℝ)

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_area :
  ∀ t : IsoscelesTrapezoid,
    t.longer_base = 20 ∧
    t.base_angle = Real.arcsin 0.6 →
    area t = 72 :=
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l829_82937
