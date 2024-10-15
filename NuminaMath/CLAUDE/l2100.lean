import Mathlib

namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2100_210036

theorem ratio_x_to_y (x y : ℚ) (h : (12*x - 5*y) / (15*x - 4*y) = 4/7) : x/y = 19/24 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2100_210036


namespace NUMINAMATH_CALUDE_student_decrease_percentage_l2100_210087

theorem student_decrease_percentage
  (initial_students : ℝ)
  (initial_price : ℝ)
  (price_increase : ℝ)
  (consumption_decrease : ℝ)
  (h1 : price_increase = 0.20)
  (h2 : consumption_decrease = 0.074074074074074066)
  (h3 : initial_students > 0)
  (h4 : initial_price > 0) :
  let new_price := initial_price * (1 + price_increase)
  let new_consumption := 1 - consumption_decrease
  let new_students := initial_students * (1 - 0.10)
  initial_students * initial_price = new_students * new_price * new_consumption :=
by sorry

end NUMINAMATH_CALUDE_student_decrease_percentage_l2100_210087


namespace NUMINAMATH_CALUDE_dime_difference_l2100_210000

/-- Represents the content of a piggy bank --/
structure PiggyBank where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins in the piggy bank --/
def totalCoins (pb : PiggyBank) : ℕ :=
  pb.pennies + pb.nickels + pb.dimes

/-- Calculates the total value in cents of the coins in the piggy bank --/
def totalValue (pb : PiggyBank) : ℕ :=
  pb.pennies + 5 * pb.nickels + 10 * pb.dimes

/-- Checks if a piggy bank configuration is valid --/
def isValidPiggyBank (pb : PiggyBank) : Prop :=
  totalCoins pb = 150 ∧ totalValue pb = 500

/-- The set of all valid piggy bank configurations --/
def validPiggyBanks : Set PiggyBank :=
  {pb | isValidPiggyBank pb}

/-- The theorem to be proven --/
theorem dime_difference : 
  (⨆ (pb : PiggyBank) (h : pb ∈ validPiggyBanks), pb.dimes) -
  (⨅ (pb : PiggyBank) (h : pb ∈ validPiggyBanks), pb.dimes) = 39 := by
  sorry

end NUMINAMATH_CALUDE_dime_difference_l2100_210000


namespace NUMINAMATH_CALUDE_solution_sum_l2100_210080

theorem solution_sum (c d : ℝ) : 
  c^2 - 6*c + 15 = 27 →
  d^2 - 6*d + 15 = 27 →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l2100_210080


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2100_210099

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

theorem perpendicular_vectors (k : ℝ) : 
  let c := (a.1 + k * b.1, a.2 + k * b.2)
  (a.1 * c.1 + a.2 * c.2 = 0) → k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2100_210099


namespace NUMINAMATH_CALUDE_system_solution_unique_l2100_210027

/-- Proves that x = 2 and y = 1 is the unique solution to the given system of equations -/
theorem system_solution_unique :
  ∃! (x y : ℝ), (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2100_210027


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l2100_210034

/-- The coefficient of x^3 in the expansion of (1+2x^2)(1+x)^4 is 12 -/
theorem coefficient_x_cubed_expansion : ∃ (p : Polynomial ℝ),
  p = (1 + 2 * X^2) * (1 + X)^4 ∧ p.coeff 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l2100_210034


namespace NUMINAMATH_CALUDE_clothing_purchase_optimal_l2100_210009

/-- Represents the prices and quantities of clothing types A and B -/
structure ClothingPrices where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- The conditions and solution for the clothing purchase problem -/
def clothing_problem (p : ClothingPrices) : Prop :=
  -- Conditions
  p.price_a + 2 * p.price_b = 110 ∧
  2 * p.price_a + 3 * p.price_b = 190 ∧
  p.quantity_a + p.quantity_b = 100 ∧
  p.quantity_a ≥ (p.quantity_b : ℝ) / 3 ∧
  -- Solution
  p.price_a = 50 ∧
  p.price_b = 30 ∧
  p.quantity_a = 25 ∧
  p.quantity_b = 75

/-- The total cost of purchasing the clothing with the discount -/
def total_cost (p : ClothingPrices) : ℝ :=
  (p.price_a - 5) * p.quantity_a + p.price_b * p.quantity_b

/-- Theorem stating that the given solution minimizes the cost -/
theorem clothing_purchase_optimal (p : ClothingPrices) :
  clothing_problem p →
  total_cost p = 3375 ∧
  (∀ q : ClothingPrices, clothing_problem q → total_cost q ≥ total_cost p) :=
sorry

end NUMINAMATH_CALUDE_clothing_purchase_optimal_l2100_210009


namespace NUMINAMATH_CALUDE_train_passing_pole_l2100_210026

/-- Proves that a train of given length and speed takes a specific time to pass a pole -/
theorem train_passing_pole (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) : 
  train_length = 500 → 
  train_speed_kmh = 90 → 
  time = train_length / (train_speed_kmh * (1000 / 3600)) → 
  time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_l2100_210026


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2100_210041

structure GeometricSpace where
  Line : Type
  Plane : Type
  perpendicular : Line → Line → Prop
  parallel : Line → Plane → Prop
  perpendicular_plane : Line → Plane → Prop

variable (S : GeometricSpace)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem perpendicular_necessary_not_sufficient
  (l m : S.Line) (α : S.Plane)
  (h1 : l ≠ m)
  (h2 : S.perpendicular_plane m α) :
  necessary_not_sufficient (S.perpendicular l m) (S.parallel l α) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2100_210041


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2100_210002

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧
  (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2100_210002


namespace NUMINAMATH_CALUDE_seating_arrangements_l2100_210058

def total_people : ℕ := 10
def restricted_group : ℕ := 4

def arrangements_with_restriction (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial - (n - k + 1).factorial * k.factorial

theorem seating_arrangements :
  arrangements_with_restriction total_people restricted_group = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2100_210058


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2100_210043

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * Real.pi * r^2 = 36 * Real.pi →
  (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2100_210043


namespace NUMINAMATH_CALUDE_f_symmetric_about_origin_l2100_210096

def f (x : ℝ) : ℝ := x^3 + x

theorem f_symmetric_about_origin : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_symmetric_about_origin_l2100_210096


namespace NUMINAMATH_CALUDE_square_sum_ge_third_square_sum_l2100_210006

theorem square_sum_ge_third_square_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_third_square_sum_l2100_210006


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2100_210033

theorem cubic_expression_evaluation : 
  3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2100_210033


namespace NUMINAMATH_CALUDE_prime_factorization_property_l2100_210065

theorem prime_factorization_property (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ y : ℕ, y ≤ p / 2 ∧ ¬∃ (a b : ℕ), a > y ∧ b > y ∧ p * y + 1 = a * b :=
by sorry

end NUMINAMATH_CALUDE_prime_factorization_property_l2100_210065


namespace NUMINAMATH_CALUDE_total_teachers_count_l2100_210085

/-- Given a school with major and minor departments, calculate the total number of teachers -/
theorem total_teachers_count (total_departments : Nat) (major_departments : Nat) (minor_departments : Nat)
  (teachers_per_major : Nat) (teachers_per_minor : Nat)
  (h1 : total_departments = major_departments + minor_departments)
  (h2 : total_departments = 17)
  (h3 : major_departments = 9)
  (h4 : minor_departments = 8)
  (h5 : teachers_per_major = 45)
  (h6 : teachers_per_minor = 29) :
  major_departments * teachers_per_major + minor_departments * teachers_per_minor = 637 := by
  sorry

#check total_teachers_count

end NUMINAMATH_CALUDE_total_teachers_count_l2100_210085


namespace NUMINAMATH_CALUDE_quadratic_polynomial_theorem_l2100_210035

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x, q x ^ 3 - x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_theorem (q : QuadraticPolynomial ℝ) 
  (h : DivisibilityCondition q) : q 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_theorem_l2100_210035


namespace NUMINAMATH_CALUDE_expression_value_l2100_210052

theorem expression_value : (40 + 15)^2 - 15^2 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2100_210052


namespace NUMINAMATH_CALUDE_second_bouquet_carnations_proof_l2100_210063

/-- The number of carnations in the second bouquet -/
def second_bouquet_carnations : ℕ := 14

/-- The number of bouquets -/
def num_bouquets : ℕ := 3

/-- The number of carnations in the first bouquet -/
def first_bouquet_carnations : ℕ := 9

/-- The number of carnations in the third bouquet -/
def third_bouquet_carnations : ℕ := 13

/-- The average number of carnations per bouquet -/
def average_carnations : ℕ := 12

theorem second_bouquet_carnations_proof :
  (first_bouquet_carnations + second_bouquet_carnations + third_bouquet_carnations) / num_bouquets = average_carnations :=
by sorry

end NUMINAMATH_CALUDE_second_bouquet_carnations_proof_l2100_210063


namespace NUMINAMATH_CALUDE_constant_area_l2100_210020

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the point P on C₂
def P : ℝ × ℝ → Prop := λ p => C₂ p.1 p.2

-- Define the line OP
def OP (p : ℝ × ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | ∃ t : ℝ, q.1 = t * p.1 ∧ q.2 = t * p.2}

-- Define the points A and B
def A (p : ℝ × ℝ) : ℝ × ℝ := (2 * p.1, 2 * p.2)
def B : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define B explicitly

-- Define the tangent line l to C₂ at P
def l (p : ℝ × ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | p.1 * q.1 + 4 * p.2 * q.2 = 4}

-- Define the points C and D
def C : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define C explicitly
def D : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define D explicitly

-- Define the area of quadrilateral ACBD
def area_ACBD (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem constant_area (p : ℝ × ℝ) (hp : P p) :
  area_ACBD p = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_constant_area_l2100_210020


namespace NUMINAMATH_CALUDE_combination_sum_equality_l2100_210078

def combination (n m : ℕ) : ℚ :=
  if n ≥ m then
    (List.range m).foldl (λ acc i => acc * (n - i : ℚ) / (i + 1)) 1
  else 0

theorem combination_sum_equality : combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equality_l2100_210078


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2100_210076

theorem polynomial_multiplication_simplification :
  ∀ (x : ℝ),
  (3 * x - 2) * (5 * x^12 + 3 * x^11 - 4 * x^9 + x^8) =
  15 * x^13 - x^12 - 6 * x^11 - 12 * x^10 + 11 * x^9 - 2 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2100_210076


namespace NUMINAMATH_CALUDE_last_k_digits_power_l2100_210023

theorem last_k_digits_power (A B : ℤ) (k n : ℕ) (h : A ≡ B [ZMOD 10^k]) :
  A^n ≡ B^n [ZMOD 10^k] := by sorry

end NUMINAMATH_CALUDE_last_k_digits_power_l2100_210023


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2100_210086

/-- The surface area of a cylinder with lateral surface net as a rectangle with sides 6π and 4π -/
theorem cylinder_surface_area : 
  ∀ (r h : ℝ), 
  (2 * π * r = 6 * π) → 
  (h = 4 * π) → 
  (2 * π * r * h + 2 * π * r^2 = 24 * π^2 + 18 * π) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2100_210086


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_three_shared_l2100_210067

theorem greatest_common_divisor_of_three_shared (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
   d1 ∣ 120 ∧ d2 ∣ 120 ∧ d3 ∣ 120 ∧
   d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
   (∀ (x : ℕ), x ∣ 120 ∧ x ∣ n → x = d1 ∨ x = d2 ∨ x = d3)) →
  (∃ (d : ℕ), d ∣ 120 ∧ d ∣ n ∧ d = 9 ∧ 
   (∀ (x : ℕ), x ∣ 120 ∧ x ∣ n → x ≤ d)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_three_shared_l2100_210067


namespace NUMINAMATH_CALUDE_sidewalk_snow_volume_l2100_210062

theorem sidewalk_snow_volume (length width height : ℝ) 
  (h1 : length = 15)
  (h2 : width = 3)
  (h3 : height = 0.6) :
  length * width * height = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_sidewalk_snow_volume_l2100_210062


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2100_210097

theorem max_sum_of_factors (p q : ℕ+) (h : p * q = 100) : 
  ∃ (a b : ℕ+), a * b = 100 ∧ a + b ≤ p + q ∧ a + b = 101 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2100_210097


namespace NUMINAMATH_CALUDE_omega_sum_equality_l2100_210092

theorem omega_sum_equality (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 = 1 := by
sorry

end NUMINAMATH_CALUDE_omega_sum_equality_l2100_210092


namespace NUMINAMATH_CALUDE_odd_power_congruence_l2100_210011

theorem odd_power_congruence (a n : ℕ) (h_odd : Odd a) (h_pos : 0 < n) :
  (a ^ (2 ^ n)) ≡ 1 [MOD 2 ^ (n + 2)] := by
  sorry

end NUMINAMATH_CALUDE_odd_power_congruence_l2100_210011


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l2100_210018

theorem chicken_wings_distribution (num_friends : ℕ) (total_wings : ℕ) :
  num_friends = 9 →
  total_wings = 27 →
  ∃ (wings_per_person : ℕ), 
    wings_per_person * num_friends = total_wings ∧
    wings_per_person = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l2100_210018


namespace NUMINAMATH_CALUDE_bus_distribution_solution_l2100_210094

/-- Represents the problem of distributing passengers among buses --/
structure BusDistribution where
  k : ℕ  -- Original number of buses
  n : ℕ  -- Number of passengers per bus after redistribution
  max_capacity : ℕ  -- Maximum capacity of each bus

/-- The conditions of the bus distribution problem --/
def valid_distribution (bd : BusDistribution) : Prop :=
  bd.k ≥ 2 ∧
  bd.n ≤ bd.max_capacity ∧
  22 * bd.k + 1 = bd.n * (bd.k - 1)

/-- The theorem stating the solution to the bus distribution problem --/
theorem bus_distribution_solution :
  ∃ (bd : BusDistribution),
    bd.max_capacity = 32 ∧
    valid_distribution bd ∧
    bd.k = 24 ∧
    bd.n * (bd.k - 1) = 529 :=
sorry


end NUMINAMATH_CALUDE_bus_distribution_solution_l2100_210094


namespace NUMINAMATH_CALUDE_square_sum_value_l2100_210016

theorem square_sum_value (x y : ℝ) (h1 : x * y = 16) (h2 : x^2 + y^2 = 34) : 
  (x + y)^2 = 66 := by sorry

end NUMINAMATH_CALUDE_square_sum_value_l2100_210016


namespace NUMINAMATH_CALUDE_cave_door_weight_calculation_l2100_210082

/-- The weight already on the switch (in pounds) -/
def weight_on_switch : ℕ := 234

/-- The total weight needed to open the cave doors (in pounds) -/
def total_weight_needed : ℕ := 712

/-- The additional weight needed to open the cave doors (in pounds) -/
def additional_weight_needed : ℕ := total_weight_needed - weight_on_switch

theorem cave_door_weight_calculation :
  additional_weight_needed = 478 := by
  sorry

end NUMINAMATH_CALUDE_cave_door_weight_calculation_l2100_210082


namespace NUMINAMATH_CALUDE_percent_problem_l2100_210047

theorem percent_problem (x : ℝ) (h : 120 = 0.75 * x) : x = 160 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l2100_210047


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2100_210007

theorem sum_of_fractions_equals_one (a b c : ℝ) (h : a * b * c = 1) :
  1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2100_210007


namespace NUMINAMATH_CALUDE_sequence_decomposition_l2100_210091

theorem sequence_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ), (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n > 0, z n ≥ z (n - 1)) ∧
    (∀ n > 0, y n * (z n - z (n - 1)) = 0) ∧
    (z 0 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_decomposition_l2100_210091


namespace NUMINAMATH_CALUDE_wendys_pastries_l2100_210089

/-- Wendy's pastry problem -/
theorem wendys_pastries (cupcakes cookies sold : ℕ) 
  (h1 : cupcakes = 4)
  (h2 : cookies = 29)
  (h3 : sold = 9) :
  cupcakes + cookies - sold = 24 := by
  sorry

end NUMINAMATH_CALUDE_wendys_pastries_l2100_210089


namespace NUMINAMATH_CALUDE_x_equals_six_l2100_210064

def floor (y : ℤ) : ℤ :=
  if y % 2 = 0 then y / 2 + 1 else 2 * y + 1

theorem x_equals_six :
  ∃ x : ℤ, floor x * floor 3 = 28 ∧ x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_six_l2100_210064


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_4b_squared_l2100_210056

theorem min_value_a_squared_plus_4b_squared (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 2/a + 1/b = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 1 → a^2 + 4*b^2 ≤ x^2 + 4*y^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_4b_squared_l2100_210056


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2100_210021

/-- Given a polynomial function f(x) = px³ - qx² + rx - s, 
    if f(1) = 4, then 2p + q - 3r + 2s = -8 -/
theorem polynomial_value_theorem (p q r s : ℝ) : 
  let f := fun (x : ℝ) => p * x^3 - q * x^2 + r * x - s
  (f 1 = 4) → (2*p + q - 3*r + 2*s = -8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2100_210021


namespace NUMINAMATH_CALUDE_initial_customer_count_l2100_210025

/-- Represents the number of customers at different times -/
structure CustomerCount where
  initial : ℕ
  after_first_hour : ℕ
  after_second_hour : ℕ

/-- Calculates the number of customers after the first hour -/
def first_hour_change (c : CustomerCount) : ℕ := c.initial + 7 - 4

/-- Calculates the number of customers after the second hour -/
def second_hour_change (c : CustomerCount) : ℕ := c.after_first_hour + 3 - 9

/-- The main theorem stating the initial number of customers -/
theorem initial_customer_count : ∃ (c : CustomerCount), 
  c.initial = 15 ∧ 
  c.after_first_hour = first_hour_change c ∧
  c.after_second_hour = second_hour_change c ∧
  c.after_second_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_customer_count_l2100_210025


namespace NUMINAMATH_CALUDE_inspector_meter_count_l2100_210001

theorem inspector_meter_count : 
  ∀ (total_meters : ℕ) (defective_meters : ℕ) (rejection_rate : ℚ),
    rejection_rate = 1/10 →
    defective_meters = 15 →
    (rejection_rate * total_meters : ℚ) = defective_meters →
    total_meters = 150 := by
  sorry

end NUMINAMATH_CALUDE_inspector_meter_count_l2100_210001


namespace NUMINAMATH_CALUDE_expression_evaluation_l2100_210024

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (x + 3) * (x - 3) - x * (x - 2) = 2 * Real.sqrt 2 - 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2100_210024


namespace NUMINAMATH_CALUDE_nuts_left_l2100_210084

theorem nuts_left (total : ℕ) (eaten_fraction : ℚ) (h1 : total = 30) (h2 : eaten_fraction = 5/6) :
  total - (total * eaten_fraction).floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_nuts_left_l2100_210084


namespace NUMINAMATH_CALUDE_cube_difference_l2100_210053

theorem cube_difference (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x - y = 3) (h4 : x + y = 5) : x^3 - y^3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l2100_210053


namespace NUMINAMATH_CALUDE_height_is_four_l2100_210075

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- A right-angled triangle on a parabola -/
structure RightTriangleOnParabola where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  right_angle_at_C : (B.x - C.x) * (A.x - C.x) + (B.y - C.y) * (A.y - C.y) = 0
  hypotenuse_parallel_to_y : A.x = B.x

/-- The height from the hypotenuse of a right-angled triangle on a parabola -/
def height_from_hypotenuse (t : RightTriangleOnParabola) : ℝ :=
  |t.B.x - t.C.x|

/-- Theorem: The height from the hypotenuse is 4 -/
theorem height_is_four (t : RightTriangleOnParabola) : height_from_hypotenuse t = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_is_four_l2100_210075


namespace NUMINAMATH_CALUDE_remainder_of_s_1012_l2100_210042

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (x^1012 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^3 + x^2 + x + 1

-- Define s(x) as the polynomial remainder
noncomputable def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_1012 : |s 1012| % 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_s_1012_l2100_210042


namespace NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2100_210032

theorem solution_set_linear_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2100_210032


namespace NUMINAMATH_CALUDE_motorboat_problem_l2100_210057

/-- Represents the problem of calculating the time taken by a motorboat to reach an island in still water -/
theorem motorboat_problem (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) (island_distance : ℝ) :
  downstream_distance = 160 →
  downstream_time = 8 →
  upstream_time = 16 →
  island_distance = 100 →
  ∃ (boat_speed : ℝ) (current_speed : ℝ),
    boat_speed + current_speed = downstream_distance / downstream_time ∧
    boat_speed - current_speed = downstream_distance / upstream_time ∧
    island_distance / boat_speed = 20 / 3 :=
by sorry

end NUMINAMATH_CALUDE_motorboat_problem_l2100_210057


namespace NUMINAMATH_CALUDE_cubic_root_product_l2100_210098

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if the product of any two roots equals 3, then c = 3a -/
theorem cubic_root_product (a b c d : ℝ) (ha : a ≠ 0) :
  (∃ r s t : ℝ, r * s = 3 ∧ r * t = 3 ∧ s * t = 3 ∧
    a * r^3 + b * r^2 + c * r + d = 0 ∧
    a * s^3 + b * s^2 + c * s + d = 0 ∧
    a * t^3 + b * t^2 + c * t + d = 0) →
  c = 3 * a :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_product_l2100_210098


namespace NUMINAMATH_CALUDE_first_number_in_second_set_l2100_210010

theorem first_number_in_second_set (x : ℝ) : 
  (24 + 35 + 58) / 3 = ((x + 51 + 29) / 3) + 6 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_second_set_l2100_210010


namespace NUMINAMATH_CALUDE_number_of_spiders_l2100_210073

def total_legs : ℕ := 136
def num_ants : ℕ := 12
def spider_legs : ℕ := 8
def ant_legs : ℕ := 6

theorem number_of_spiders :
  ∃ (num_spiders : ℕ), 
    num_spiders * spider_legs + num_ants * ant_legs = total_legs ∧ 
    num_spiders = 8 :=
by sorry

end NUMINAMATH_CALUDE_number_of_spiders_l2100_210073


namespace NUMINAMATH_CALUDE_N_is_k_times_sum_of_digits_l2100_210055

/-- A number consisting of k nines -/
def N (k : ℕ) : ℕ := 10^k - 1

/-- The sum of digits of a number consisting of k nines -/
def sum_of_digits (k : ℕ) : ℕ := 9 * k

/-- Theorem stating that N(k) is k times greater than the sum of its digits for all natural k -/
theorem N_is_k_times_sum_of_digits (k : ℕ) :
  N k = k * (sum_of_digits k) :=
sorry

end NUMINAMATH_CALUDE_N_is_k_times_sum_of_digits_l2100_210055


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2100_210072

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2100_210072


namespace NUMINAMATH_CALUDE_mikes_painting_area_l2100_210074

/-- The area Mike needs to paint on the wall -/
def area_to_paint (wall_height wall_length window_height window_length painting_side : ℝ) : ℝ :=
  wall_height * wall_length - (window_height * window_length + painting_side * painting_side)

/-- Theorem stating the area Mike needs to paint -/
theorem mikes_painting_area :
  area_to_paint 10 15 3 5 2 = 131 := by
  sorry

end NUMINAMATH_CALUDE_mikes_painting_area_l2100_210074


namespace NUMINAMATH_CALUDE_symmetry_oyz_coordinates_l2100_210031

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The Oyz plane -/
def Oyz : Set Point3D :=
  {p : Point3D | p.x = 0}

/-- Symmetry with respect to the Oyz plane -/
def symmetricOyz (a b : Point3D) : Prop :=
  b.x = -a.x ∧ b.y = a.y ∧ b.z = a.z

theorem symmetry_oyz_coordinates :
  let a : Point3D := ⟨3, 4, 5⟩
  let b : Point3D := ⟨-3, 4, 5⟩
  symmetricOyz a b := by sorry

end NUMINAMATH_CALUDE_symmetry_oyz_coordinates_l2100_210031


namespace NUMINAMATH_CALUDE_ship_speed_in_still_water_l2100_210003

theorem ship_speed_in_still_water :
  let downstream_distance : ℝ := 81
  let upstream_distance : ℝ := 69
  let water_flow_speed : ℝ := 2
  let ship_speed : ℝ := 25
  (downstream_distance / (ship_speed + water_flow_speed) =
   upstream_distance / (ship_speed - water_flow_speed)) →
  ship_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_in_still_water_l2100_210003


namespace NUMINAMATH_CALUDE_petes_marbles_l2100_210013

theorem petes_marbles (total_initial : ℕ) (blue_percent : ℚ) (trade_ratio : ℕ) (kept_red : ℕ) :
  total_initial = 10 ∧
  blue_percent = 2/5 ∧
  trade_ratio = 2 ∧
  kept_red = 1 →
  (total_initial * blue_percent).floor +
  kept_red +
  trade_ratio * ((total_initial * (1 - blue_percent)).floor - kept_red) = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_marbles_l2100_210013


namespace NUMINAMATH_CALUDE_black_marble_probability_l2100_210049

theorem black_marble_probability (yellow blue green black : ℕ) 
  (h1 : yellow = 12)
  (h2 : blue = 10)
  (h3 : green = 5)
  (h4 : black = 1) :
  (black * 14000) / (yellow + blue + green + black) = 500 := by
  sorry

end NUMINAMATH_CALUDE_black_marble_probability_l2100_210049


namespace NUMINAMATH_CALUDE_max_value_2q_minus_r_l2100_210014

theorem max_value_2q_minus_r : 
  ∃ (q r : ℕ+), 1024 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1024 = 23 * q' + r' → 2 * q - r ≥ 2 * q' - r' ∧
  2 * q - r = 76 := by
sorry

end NUMINAMATH_CALUDE_max_value_2q_minus_r_l2100_210014


namespace NUMINAMATH_CALUDE_hockey_pad_cost_calculation_l2100_210004

def hockey_pad_cost (initial_amount : ℝ) (skate_fraction : ℝ) (remaining : ℝ) : ℝ :=
  initial_amount - initial_amount * skate_fraction - remaining

theorem hockey_pad_cost_calculation :
  hockey_pad_cost 150 (1/2) 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hockey_pad_cost_calculation_l2100_210004


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2100_210090

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (D : ℝ), D = Real.sqrt (12 / 17) ∧
  (∀ (x y : ℝ), x^2 + 2*y^2 + 3 ≥ D*(3*x + 4*y)) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 + 3 ≥ D'*(3*x + 4*y)) → D' ≤ D) :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2100_210090


namespace NUMINAMATH_CALUDE_total_fertilizer_needed_l2100_210005

def petunia_flats : ℕ := 4
def petunias_per_flat : ℕ := 8
def rose_flats : ℕ := 3
def roses_per_flat : ℕ := 6
def venus_flytraps : ℕ := 2
def fertilizer_per_petunia : ℕ := 8
def fertilizer_per_rose : ℕ := 3
def fertilizer_per_venus_flytrap : ℕ := 2

theorem total_fertilizer_needed : 
  petunia_flats * petunias_per_flat * fertilizer_per_petunia + 
  rose_flats * roses_per_flat * fertilizer_per_rose + 
  venus_flytraps * fertilizer_per_venus_flytrap = 314 := by
  sorry

end NUMINAMATH_CALUDE_total_fertilizer_needed_l2100_210005


namespace NUMINAMATH_CALUDE_equation_solution_l2100_210083

theorem equation_solution :
  let f (x : ℝ) := 4 / (Real.sqrt (x + 5) - 7) + 3 / (Real.sqrt (x + 5) - 2) +
                   6 / (Real.sqrt (x + 5) + 2) + 9 / (Real.sqrt (x + 5) + 7)
  {x : ℝ | f x = 0} = {-796/169, 383/22} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2100_210083


namespace NUMINAMATH_CALUDE_n_has_four_digits_l2100_210028

def n : ℕ := 9376

theorem n_has_four_digits :
  (∃ k : ℕ, n^2 % 10000 = n) →
  (∃ m : ℕ, 10^3 ≤ n ∧ n < 10^4) :=
by sorry

end NUMINAMATH_CALUDE_n_has_four_digits_l2100_210028


namespace NUMINAMATH_CALUDE_trash_bin_charge_is_10_l2100_210071

/-- Represents the garbage bill calculation -/
def garbage_bill (T : ℚ) : Prop :=
  let weeks : ℕ := 4
  let trash_bins : ℕ := 2
  let recycling_bins : ℕ := 1
  let recycling_charge : ℚ := 5
  let discount_rate : ℚ := 0.18
  let fine : ℚ := 20
  let final_bill : ℚ := 102

  let pre_discount := weeks * (trash_bins * T + recycling_bins * recycling_charge)
  let discount := discount_rate * pre_discount
  let post_discount := pre_discount - discount
  let total_bill := post_discount + fine

  total_bill = final_bill

/-- Theorem stating that the charge per trash bin is $10 -/
theorem trash_bin_charge_is_10 : garbage_bill 10 := by
  sorry

end NUMINAMATH_CALUDE_trash_bin_charge_is_10_l2100_210071


namespace NUMINAMATH_CALUDE_exists_n_with_constant_term_l2100_210019

/-- A function that checks if the expansion of (x - 1/x³)ⁿ contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 4 * r

/-- Theorem stating that there exists an n between 3 and 16 (inclusive) 
    such that the expansion of (x - 1/x³)ⁿ contains a constant term -/
theorem exists_n_with_constant_term : 
  ∃ n : ℕ, 3 ≤ n ∧ n ≤ 16 ∧ has_constant_term n :=
sorry

end NUMINAMATH_CALUDE_exists_n_with_constant_term_l2100_210019


namespace NUMINAMATH_CALUDE_fault_line_current_movement_l2100_210081

/-- The movement of a fault line over two years -/
structure FaultLineMovement where
  total : ℝ  -- Total movement over two years
  previous : ℝ  -- Movement in the previous year
  current : ℝ  -- Movement in the current year

/-- Theorem: Given the total movement and previous year's movement, 
    calculate the current year's movement -/
theorem fault_line_current_movement 
  (f : FaultLineMovement) 
  (h1 : f.total = 6.5) 
  (h2 : f.previous = 5.25) 
  (h3 : f.total = f.previous + f.current) : 
  f.current = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_current_movement_l2100_210081


namespace NUMINAMATH_CALUDE_gift_purchase_solution_l2100_210069

/-- Pricing function based on quantity --/
def price (q : ℕ) : ℚ :=
  if q ≤ 120 then 3.5
  else if q ≤ 300 then 3.2
  else 3

/-- Total cost for a given quantity --/
def total_cost (q : ℕ) : ℚ :=
  if q ≤ 120 then q * price q
  else if q ≤ 300 then 120 * 3.5 + (q - 120) * price q
  else 120 * 3.5 + 180 * 3.2 + (q - 300) * price q

/-- Theorem stating the correctness of the solution --/
theorem gift_purchase_solution :
  let xiaoli_units : ℕ := 290
  let xiaowang_units : ℕ := 110
  xiaoli_units + xiaowang_units = 400 ∧
  xiaoli_units > 280 ∧
  total_cost xiaoli_units + total_cost xiaowang_units = 1349 :=
by sorry

end NUMINAMATH_CALUDE_gift_purchase_solution_l2100_210069


namespace NUMINAMATH_CALUDE_max_xy_value_l2100_210039

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a/3 + b/4 = 1 → x*y ≥ a*b ∧ x*y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l2100_210039


namespace NUMINAMATH_CALUDE_value_of_4x2y2_l2100_210045

theorem value_of_4x2y2 (x y : ℤ) (h : y^2 + 4*x^2*y^2 = 40*x^2 + 817) : 
  4*x^2*y^2 = 3484 := by sorry

end NUMINAMATH_CALUDE_value_of_4x2y2_l2100_210045


namespace NUMINAMATH_CALUDE_exam_average_l2100_210061

theorem exam_average (total_candidates : ℕ) (passed_candidates : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_candidates = 120)
  (h2 : passed_candidates = 100)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := passed_avg * passed_candidates + failed_avg * failed_candidates
  total_marks / total_candidates = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l2100_210061


namespace NUMINAMATH_CALUDE_straight_row_not_tetrahedron_l2100_210066

/-- A pattern of squares that can be folded -/
structure FoldablePattern :=
  (squares : ℕ)
  (arrangement : String)

/-- Properties of a regular tetrahedron -/
structure RegularTetrahedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Definition of a straight row pattern -/
def straightRowPattern : FoldablePattern :=
  { squares := 4,
    arrangement := "straight row" }

/-- Definition of a regular tetrahedron -/
def regularTetrahedron : RegularTetrahedron :=
  { faces := 4,
    edges := 6,
    vertices := 4 }

/-- Function to check if a pattern can be folded into a regular tetrahedron -/
def canFoldToTetrahedron (pattern : FoldablePattern) : Prop :=
  ∃ (t : RegularTetrahedron), t = regularTetrahedron

/-- Theorem stating that a straight row pattern cannot be folded into a regular tetrahedron -/
theorem straight_row_not_tetrahedron :
  ¬(canFoldToTetrahedron straightRowPattern) :=
sorry

end NUMINAMATH_CALUDE_straight_row_not_tetrahedron_l2100_210066


namespace NUMINAMATH_CALUDE_absolute_value_sum_l2100_210060

theorem absolute_value_sum : -2 + |(-3)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l2100_210060


namespace NUMINAMATH_CALUDE_some_number_problem_l2100_210037

theorem some_number_problem (n : ℝ) :
  (∃ x₁ x₂ : ℝ, |x₁ - n| = 50 ∧ |x₂ - n| = 50 ∧ x₁ + x₂ = 50) →
  n = 25 :=
by sorry

end NUMINAMATH_CALUDE_some_number_problem_l2100_210037


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l2100_210077

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 8*x^2 + a*x - b = 0 ∧
    y^3 - 8*y^2 + a*y - b = 0 ∧
    z^3 - 8*z^2 + a*z - b = 0) →
  (∀ a' b' : ℝ, (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 8*x^2 + a'*x - b' = 0 ∧
    y^3 - 8*y^2 + a'*y - b' = 0 ∧
    z^3 - 8*z^2 + a'*z - b' = 0) →
  a + b ≤ a' + b') →
  a + b = 27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l2100_210077


namespace NUMINAMATH_CALUDE_race_distance_l2100_210093

/-- The race problem -/
theorem race_distance (speed_A speed_B : ℝ) (head_start win_margin total_distance : ℝ) :
  speed_A > 0 ∧ speed_B > 0 →
  speed_A / speed_B = 3 / 4 →
  head_start = 200 →
  win_margin = 100 →
  total_distance / speed_A = (total_distance - head_start - win_margin) / speed_B →
  total_distance = 900 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l2100_210093


namespace NUMINAMATH_CALUDE_pipe_A_rate_correct_l2100_210044

/-- Represents the rate at which pipe A fills the tank -/
def pipe_A_rate : ℝ := 40

/-- Represents the rate at which pipe B fills the tank -/
def pipe_B_rate : ℝ := 30

/-- Represents the rate at which pipe C drains the tank -/
def pipe_C_rate : ℝ := 20

/-- Represents the capacity of the tank -/
def tank_capacity : ℝ := 850

/-- Represents the time it takes to fill the tank -/
def fill_time : ℝ := 51

/-- Represents the duration of one cycle -/
def cycle_duration : ℝ := 3

/-- Theorem stating that pipe A's rate satisfies the given conditions -/
theorem pipe_A_rate_correct : 
  (fill_time / cycle_duration) * (pipe_A_rate + pipe_B_rate - pipe_C_rate) = tank_capacity :=
by sorry

end NUMINAMATH_CALUDE_pipe_A_rate_correct_l2100_210044


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2100_210054

theorem angle_sum_is_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2100_210054


namespace NUMINAMATH_CALUDE_expression_simplification_l2100_210038

/-- Proves that the given expression simplifies to 1 when a = 1 and b = -2 -/
theorem expression_simplification (a b : ℤ) (ha : a = 1) (hb : b = -2) :
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_expression_simplification_l2100_210038


namespace NUMINAMATH_CALUDE_ben_spending_correct_l2100_210046

/-- Calculates Ben's spending at the bookstore with given prices and discounts --/
def benSpending (notebookPrice magazinePrice penPrice bookPrice : ℚ)
                (notebookCount magazineCount penCount bookCount : ℕ)
                (penDiscount membershipDiscount membershipThreshold : ℚ) : ℚ :=
  let subtotal := notebookPrice * notebookCount +
                  magazinePrice * magazineCount +
                  penPrice * (1 - penDiscount) * penCount +
                  bookPrice * bookCount
  if subtotal ≥ membershipThreshold then
    subtotal - membershipDiscount
  else
    subtotal

/-- Theorem stating that Ben's spending matches the calculated amount --/
theorem ben_spending_correct :
  benSpending 2 6 1.5 12 4 3 5 2 0.25 10 50 = 45.625 := by sorry

end NUMINAMATH_CALUDE_ben_spending_correct_l2100_210046


namespace NUMINAMATH_CALUDE_place_value_ratio_l2100_210017

def number : ℚ := 86743.2951

def place_value_6 : ℚ := 10000
def place_value_5 : ℚ := 0.1

theorem place_value_ratio :
  place_value_6 / place_value_5 = 100000 := by
  sorry

#check place_value_ratio

end NUMINAMATH_CALUDE_place_value_ratio_l2100_210017


namespace NUMINAMATH_CALUDE_triangle_inequality_relationships_l2100_210068

/-- A triangle with perimeter, circumradius, and inradius -/
structure Triangle where
  perimeter : ℝ
  circumradius : ℝ
  inradius : ℝ
  perimeter_pos : 0 < perimeter
  circumradius_pos : 0 < circumradius
  inradius_pos : 0 < inradius

/-- Theorem stating that none of the given relationships hold universally for all triangles -/
theorem triangle_inequality_relationships (t : Triangle) : 
  ¬(∀ t : Triangle, t.perimeter > t.circumradius + t.inradius) ∧ 
  ¬(∀ t : Triangle, t.perimeter ≤ t.circumradius + t.inradius) ∧ 
  ¬(∀ t : Triangle, 1/6 < t.circumradius + t.inradius ∧ t.circumradius + t.inradius < 6*t.perimeter) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_relationships_l2100_210068


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2100_210059

theorem fractional_equation_solution : 
  ∃ (x : ℝ), (x ≠ 0 ∧ x ≠ 2) ∧ (5 / (x - 2) = 3 / x) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2100_210059


namespace NUMINAMATH_CALUDE_equation_solution_l2100_210022

theorem equation_solution : 
  ∃! x : ℚ, (2 * x) / (x + 3) + 1 = 7 / (2 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2100_210022


namespace NUMINAMATH_CALUDE_arrange_5_balls_4_boxes_l2100_210070

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def arrange_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to put 5 distinguishable balls into 4 distinguishable boxes -/
theorem arrange_5_balls_4_boxes : arrange_balls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_arrange_5_balls_4_boxes_l2100_210070


namespace NUMINAMATH_CALUDE_jake_weight_proof_l2100_210030

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 196

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 290 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 290) →
  jake_weight = 196 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l2100_210030


namespace NUMINAMATH_CALUDE_min_trucks_required_l2100_210029

/-- Represents the total weight of boxes in tons -/
def total_weight : ℝ := 10

/-- Represents the maximum weight of a single box in tons -/
def max_box_weight : ℝ := 1

/-- Represents the capacity of each truck in tons -/
def truck_capacity : ℝ := 3

/-- Calculates the minimum number of trucks required -/
def min_trucks : ℕ := 5

theorem min_trucks_required :
  ∀ (weights : List ℝ),
    weights.sum = total_weight →
    (∀ w ∈ weights, w ≤ max_box_weight) →
    (∀ n : ℕ, n < min_trucks → 
      ∃ partition : List (List ℝ),
        partition.length = n ∧
        partition.join.sum = total_weight ∧
        (∀ part ∈ partition, part.sum > truck_capacity)) →
    ∃ partition : List (List ℝ),
      partition.length = min_trucks ∧
      partition.join.sum = total_weight ∧
      (∀ part ∈ partition, part.sum ≤ truck_capacity) :=
by sorry

#check min_trucks_required

end NUMINAMATH_CALUDE_min_trucks_required_l2100_210029


namespace NUMINAMATH_CALUDE_ball_probabilities_l2100_210008

def total_balls : ℕ := 4
def red_balls : ℕ := 2

def prob_two_red : ℚ := 1 / 6
def prob_at_least_one_red : ℚ := 5 / 6

theorem ball_probabilities :
  (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1)) = prob_two_red ∧
  1 - ((total_balls - red_balls) * (total_balls - red_balls - 1)) / (total_balls * (total_balls - 1)) = prob_at_least_one_red :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2100_210008


namespace NUMINAMATH_CALUDE_pascal_triangle_prob_one_20_l2100_210051

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Set ℕ := sorry

/-- The number of elements in the first n rows of Pascal's Triangle -/
def numElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle -/
def numOnes (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probSelectOne (n : ℕ) : ℚ := (numOnes n : ℚ) / (numElements n : ℚ)

theorem pascal_triangle_prob_one_20 : 
  probSelectOne 20 = 39 / 210 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_prob_one_20_l2100_210051


namespace NUMINAMATH_CALUDE_profit_maximized_at_twelve_point_five_l2100_210088

/-- The profit function for the bookstore -/
def P (p : ℝ) : ℝ := 150 * p - 6 * p^2 - 200

/-- The theorem stating that the profit is maximized at p = 12.5 -/
theorem profit_maximized_at_twelve_point_five :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧ 
  (∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → P p ≥ P q) ∧
  p = 12.5 := by
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_twelve_point_five_l2100_210088


namespace NUMINAMATH_CALUDE_total_trucks_l2100_210048

/-- The number of trucks Namjoon and Taehyung have together -/
theorem total_trucks (namjoon_trucks taehyung_trucks : ℕ) 
  (h1 : namjoon_trucks = 3) 
  (h2 : taehyung_trucks = 2) : 
  namjoon_trucks + taehyung_trucks = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_trucks_l2100_210048


namespace NUMINAMATH_CALUDE_kittens_given_to_friends_l2100_210050

/-- Given that Joan initially had 8 kittens and now has 6 kittens,
    prove that she gave 2 kittens to her friends. -/
theorem kittens_given_to_friends : 
  ∀ (initial current given : ℕ), 
    initial = 8 → 
    current = 6 → 
    given = initial - current → 
    given = 2 := by
  sorry

end NUMINAMATH_CALUDE_kittens_given_to_friends_l2100_210050


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2100_210012

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 + y^3 + z^3 ≥ x^2 * Real.sqrt (y*z) + y^2 * Real.sqrt (z*x) + z^2 * Real.sqrt (x*y) :=
sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2100_210012


namespace NUMINAMATH_CALUDE_max_third_term_in_arithmetic_sequence_l2100_210095

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem max_third_term_in_arithmetic_sequence :
  ∀ a b c d : ℕ,
  a > 0 → b > 0 → c > 0 → d > 0 →
  is_arithmetic_sequence a b c d →
  a + b + c + d = 50 →
  c ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_third_term_in_arithmetic_sequence_l2100_210095


namespace NUMINAMATH_CALUDE_book_student_difference_l2100_210040

/-- Proves that in 5 classrooms, where each classroom has 18 students and each student has 3 books,
    the difference between the total number of books and the total number of students is 180. -/
theorem book_student_difference :
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 18
  let books_per_student : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_books : ℕ := total_students * books_per_student
  total_books - total_students = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_book_student_difference_l2100_210040


namespace NUMINAMATH_CALUDE_younger_brother_bricks_l2100_210079

theorem younger_brother_bricks (total_bricks : ℕ) (final_difference : ℕ) : 
  total_bricks = 26 ∧ final_difference = 2 → 
  ∃ (initial_younger : ℕ), 
    initial_younger = 16 ∧
    (total_bricks - initial_younger) + (initial_younger / 2) - 
    ((total_bricks - initial_younger + (initial_younger / 2)) / 2) + 5 = 
    initial_younger - (initial_younger / 2) + 
    ((total_bricks - initial_younger + (initial_younger / 2)) / 2) - 5 + final_difference :=
by
  sorry

#check younger_brother_bricks

end NUMINAMATH_CALUDE_younger_brother_bricks_l2100_210079


namespace NUMINAMATH_CALUDE_triangle_area_l2100_210015

theorem triangle_area (base height : Real) (h1 : base = 8.4) (h2 : height = 5.8) :
  (base * height) / 2 = 24.36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2100_210015
