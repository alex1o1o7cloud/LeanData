import Mathlib

namespace NUMINAMATH_CALUDE_corveus_sleep_lack_l1580_158002

/-- The number of hours Corveus lacks sleep in a week -/
def sleep_lack_per_week (actual_sleep : ℕ) (recommended_sleep : ℕ) (days_in_week : ℕ) : ℕ :=
  (recommended_sleep - actual_sleep) * days_in_week

/-- Theorem stating that Corveus lacks 14 hours of sleep in a week -/
theorem corveus_sleep_lack :
  sleep_lack_per_week 4 6 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_corveus_sleep_lack_l1580_158002


namespace NUMINAMATH_CALUDE_sock_pairs_problem_l1580_158009

theorem sock_pairs_problem (n : ℕ) : 
  (2 * n * (2 * n - 1)) / 2 = 6 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_problem_l1580_158009


namespace NUMINAMATH_CALUDE_zeros_of_f_range_of_m_l1580_158073

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

-- Theorem for part (I)
theorem zeros_of_f'_depend_on_a (a : ℝ) :
  ∃ n : ℕ, n ∈ ({1, 2} : Set ℕ) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-1) 3 ∧ x₂ ∈ Set.Icc (-1) 3 ∧ 
   f' a x₁ = 0 ∧ f' a x₂ = 0 ∧ 
   ∀ x ∈ Set.Icc (-1) 3, f' a x = 0 → x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for part (II)
theorem range_of_m (a : ℝ) (h : a ∈ Set.Icc (-3) 0) :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → 
    m - a * m^2 ≥ |f a x₁ - f a x₂|) → 
  m ∈ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_range_of_m_l1580_158073


namespace NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l1580_158080

theorem smallest_y_with_given_remainders : 
  ∃! y : ℕ, 
    y > 0 ∧
    y % 3 = 2 ∧ 
    y % 5 = 4 ∧ 
    y % 7 = 6 ∧
    ∀ z : ℕ, z > 0 ∧ z % 3 = 2 ∧ z % 5 = 4 ∧ z % 7 = 6 → y ≤ z :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l1580_158080


namespace NUMINAMATH_CALUDE_labrador_starting_weight_l1580_158099

/-- The starting weight of the labrador puppy -/
def L : ℝ := 40

/-- The starting weight of the dachshund puppy -/
def dachshund_weight : ℝ := 12

/-- The weight gain percentage for both dogs -/
def weight_gain_percentage : ℝ := 0.25

/-- The weight difference between the dogs at the end of the year -/
def weight_difference : ℝ := 35

/-- Theorem stating that the labrador puppy's starting weight satisfies the given conditions -/
theorem labrador_starting_weight :
  L * (1 + weight_gain_percentage) - dachshund_weight * (1 + weight_gain_percentage) = weight_difference := by
  sorry

end NUMINAMATH_CALUDE_labrador_starting_weight_l1580_158099


namespace NUMINAMATH_CALUDE_final_expression_l1580_158067

theorem final_expression (b : ℚ) : 
  (3 * b + 6 - 5 * b) / 3 = -2/3 * b + 2 := by sorry

end NUMINAMATH_CALUDE_final_expression_l1580_158067


namespace NUMINAMATH_CALUDE_complement_of_union_relative_to_U_l1580_158082

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union_relative_to_U :
  (U \ (A ∪ B)) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_relative_to_U_l1580_158082


namespace NUMINAMATH_CALUDE_brownie_division_l1580_158063

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of brownies -/
def pan : Dimensions := ⟨15, 25⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 5⟩

/-- Theorem stating that the pan can be divided into exactly 25 pieces -/
theorem brownie_division :
  (area pan) / (area piece) = 25 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l1580_158063


namespace NUMINAMATH_CALUDE_double_average_l1580_158052

theorem double_average (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) :
  n = 11 →
  initial_avg = 36 →
  new_avg = 2 * initial_avg →
  new_avg = 72 :=
by sorry

end NUMINAMATH_CALUDE_double_average_l1580_158052


namespace NUMINAMATH_CALUDE_base_conversion_proof_l1580_158015

/-- 
Given a positive integer n with the following properties:
1. Its base 9 representation is AB
2. Its base 7 representation is BA
3. A and B are single digits in their respective bases

This theorem proves that n = 31 in base 10.
-/
theorem base_conversion_proof (n : ℕ) (A B : ℕ) 
  (h1 : n = 9 * A + B)
  (h2 : n = 7 * B + A)
  (h3 : A < 9 ∧ B < 9)
  (h4 : A < 7 ∧ B < 7)
  (h5 : n > 0) :
  n = 31 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_proof_l1580_158015


namespace NUMINAMATH_CALUDE_dogs_with_neither_l1580_158093

/-- Given a kennel with dogs, prove the number of dogs wearing neither tags nor flea collars -/
theorem dogs_with_neither (total : ℕ) (with_tags : ℕ) (with_collars : ℕ) (with_both : ℕ) 
  (h1 : total = 80)
  (h2 : with_tags = 45)
  (h3 : with_collars = 40)
  (h4 : with_both = 6) :
  total - (with_tags + with_collars - with_both) = 1 := by
  sorry

#check dogs_with_neither

end NUMINAMATH_CALUDE_dogs_with_neither_l1580_158093


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l1580_158077

theorem ceiling_floor_calculation : 
  ⌈(18 : ℚ) / 5 * (-25 : ℚ) / 4⌉ - ⌊(18 : ℚ) / 5 * ⌊(-25 : ℚ) / 4⌋⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l1580_158077


namespace NUMINAMATH_CALUDE_black_length_is_two_l1580_158086

def pencil_length : ℝ := 6
def purple_length : ℝ := 3
def blue_length : ℝ := 1

theorem black_length_is_two :
  pencil_length - purple_length - blue_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_black_length_is_two_l1580_158086


namespace NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l1580_158068

/-- Represents a circular pizza with pepperoni toppings -/
structure PizzaWithPepperoni where
  pizza_diameter : ℝ
  pepperoni_count_across : ℕ
  total_pepperoni_count : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def fraction_covered (p : PizzaWithPepperoni) : ℚ :=
  sorry

/-- Theorem stating that for a pizza with given specifications, 
    the fraction covered by pepperoni is 4/9 -/
theorem pepperoni_coverage_fraction 
  (p : PizzaWithPepperoni) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_count_across = 9)
  (h3 : p.total_pepperoni_count = 36) : 
  fraction_covered p = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l1580_158068


namespace NUMINAMATH_CALUDE_sound_distance_at_zero_celsius_sound_distance_in_five_seconds_l1580_158038

/-- Represents the speed of sound in air at different temperatures -/
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- Default case for temperatures not in the table

/-- Calculates the distance traveled by sound in a given time at 0°C -/
def distance_traveled (time : Int) : Int :=
  (speed_of_sound 0) * time

theorem sound_distance_at_zero_celsius (time : Int) :
  distance_traveled time = speed_of_sound 0 * time :=
by sorry

theorem sound_distance_in_five_seconds :
  distance_traveled 5 = 1650 :=
by sorry

end NUMINAMATH_CALUDE_sound_distance_at_zero_celsius_sound_distance_in_five_seconds_l1580_158038


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1580_158035

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2*x - 1) ∧
  f 0 = 2 ∧
  (∀ x ∈ Set.Icc (-2) 2, 1 ≤ f x ∧ f x ≤ 10) ∧
  (∀ t, 
    let min_value := 
      if t ≥ 1 then t^2 - 2*t + 2
      else if 0 < t ∧ t < 1 then 1
      else t^2 + 2*t + 1
    ∀ x ∈ Set.Icc t (t + 1), f x ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1580_158035


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1580_158032

theorem sum_of_fractions_equals_one (x y z : ℝ) (h : x * y * z = 1) :
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1580_158032


namespace NUMINAMATH_CALUDE_carries_hourly_wage_l1580_158001

/-- Carrie's work and savings scenario --/
theorem carries_hourly_wage (hours_per_week : ℕ) (weeks : ℕ) (bike_cost : ℕ) (leftover : ℕ) :
  hours_per_week = 35 →
  weeks = 4 →
  bike_cost = 400 →
  leftover = 720 →
  ∃ (hourly_wage : ℚ), hourly_wage = 8 ∧ 
    (hourly_wage * (hours_per_week * weeks : ℚ) : ℚ) = (bike_cost + leftover : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_carries_hourly_wage_l1580_158001


namespace NUMINAMATH_CALUDE_opposite_of_sin_60_degrees_l1580_158091

theorem opposite_of_sin_60_degrees :
  -(Real.sin (π / 3)) = -(Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sin_60_degrees_l1580_158091


namespace NUMINAMATH_CALUDE_camp_food_consumption_l1580_158087

/-- Represents the amount of food eaten by dogs and puppies in a day -/
def total_food_eaten (num_puppies num_dogs : ℕ) 
                     (dog_meal_frequency puppy_meal_frequency : ℕ) 
                     (dog_meal_amount : ℚ) 
                     (dog_puppy_food_ratio : ℚ) : ℚ :=
  let dog_daily_food := dog_meal_amount * dog_meal_frequency
  let puppy_meal_amount := dog_meal_amount / dog_puppy_food_ratio
  let puppy_daily_food := puppy_meal_amount * puppy_meal_frequency
  (num_dogs : ℚ) * dog_daily_food + (num_puppies : ℚ) * puppy_daily_food

/-- Theorem stating the total food eaten by dogs and puppies in a day -/
theorem camp_food_consumption : 
  total_food_eaten 6 5 2 8 6 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_camp_food_consumption_l1580_158087


namespace NUMINAMATH_CALUDE_value_of_expression_l1580_158084

theorem value_of_expression (a b : ℝ) (h : a - 2*b = -2) : 4 - 2*a + 4*b = 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1580_158084


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_two_l1580_158021

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line: 2x + my - 2m + 4 = 0 -/
def line1 (m : ℝ) : Line :=
  { a := 2, b := m, c := -2*m + 4 }

/-- The second line: mx + 2y - m + 2 = 0 -/
def line2 (m : ℝ) : Line :=
  { a := m, b := 2, c := -m + 2 }

/-- Theorem stating that the lines are parallel if and only if m = -2 -/
theorem lines_parallel_iff_m_eq_neg_two :
  ∀ m : ℝ, parallel (line1 m) (line2 m) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_two_l1580_158021


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1580_158055

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (∀ m : ℕ, m ≥ n → (11 ∣ m ∧ ∀ k : ℕ, 3 ≤ k ∧ k ≤ 8 → m % k = 2) → m ≥ 3362) ∧
  (11 ∣ 3362) ∧
  (∀ k : ℕ, 3 ≤ k ∧ k ≤ 8 → 3362 % k = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1580_158055


namespace NUMINAMATH_CALUDE_and_implies_or_but_not_conversely_l1580_158095

-- Define propositions p and q
variable (p q : Prop)

-- State the theorem
theorem and_implies_or_but_not_conversely :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) :=
by
  sorry


end NUMINAMATH_CALUDE_and_implies_or_but_not_conversely_l1580_158095


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1580_158040

def M : Set ℝ := {-1, 1, 2}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1580_158040


namespace NUMINAMATH_CALUDE_tan_eq_sin_cos_unique_solution_l1580_158069

open Real

theorem tan_eq_sin_cos_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arccos 0.1 ∧ tan x = sin (cos x) := by
  sorry

end NUMINAMATH_CALUDE_tan_eq_sin_cos_unique_solution_l1580_158069


namespace NUMINAMATH_CALUDE_trapezoid_area_l1580_158007

/-- The area of a trapezoid with height x, one base 4x, and the other base (4x - 2x) is 3x² -/
theorem trapezoid_area (x : ℝ) : 
  let height := x
  let base1 := 4 * x
  let base2 := 4 * x - 2 * x
  (base1 + base2) / 2 * height = 3 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1580_158007


namespace NUMINAMATH_CALUDE_sqrt_25_l1580_158020

theorem sqrt_25 : Real.sqrt 25 = 5 ∨ Real.sqrt 25 = -5 := by sorry

end NUMINAMATH_CALUDE_sqrt_25_l1580_158020


namespace NUMINAMATH_CALUDE_mathilda_debt_l1580_158098

theorem mathilda_debt (initial_payment : ℝ) (remaining_percentage : ℝ) (original_debt : ℝ) : 
  initial_payment = 125 →
  remaining_percentage = 75 →
  initial_payment = (100 - remaining_percentage) / 100 * original_debt →
  original_debt = 500 := by
sorry

end NUMINAMATH_CALUDE_mathilda_debt_l1580_158098


namespace NUMINAMATH_CALUDE_inradius_formula_l1580_158048

theorem inradius_formula (β γ R : Real) (hβ : 0 < β) (hγ : 0 < γ) (hβγ : β + γ < π) (hR : R > 0) :
  ∃ (r : Real), r = 4 * R * Real.sin (β / 2) * Real.sin (γ / 2) * Real.cos ((β + γ) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inradius_formula_l1580_158048


namespace NUMINAMATH_CALUDE_max_distance_MN_l1580_158050

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

def C₂ (x y : ℝ) : Prop := ∃ φ : ℝ, x = 2 * Real.cos φ ∧ y = Real.sin φ

def C₃ (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Define the transformation
def transformation (x y : ℝ) : Prop := x^2 = 2*x ∧ y^2 = y

-- Define the tangent point condition
def is_tangent_point (M N : ℝ × ℝ) : Prop :=
  C₂ M.1 M.2 ∧ C₃ N.1 N.2 ∧
  ∃ t : ℝ, (N.1 - M.1)^2 + (N.2 - M.2)^2 = t^2 ∧
           ∀ P : ℝ × ℝ, C₃ P.1 P.2 → (P.1 - M.1)^2 + (P.2 - M.2)^2 ≥ t^2

-- Theorem statement
theorem max_distance_MN :
  ∀ M N : ℝ × ℝ, is_tangent_point M N →
  (N.1 - M.1)^2 + (N.2 - M.2)^2 ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_distance_MN_l1580_158050


namespace NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l1580_158045

/-- Given the equations for velocity and displacement, prove the formula for time. -/
theorem time_from_velocity_and_displacement
  (g V V₀ S S₀ a t : ℝ)
  (hV : V = g * (t - a) + V₀)
  (hS : S = (1/2) * g * (t - a)^2 + V₀ * (t - a) + S₀) :
  t = a + (V - V₀) / g :=
sorry

end NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l1580_158045


namespace NUMINAMATH_CALUDE_circle_relationship_l1580_158072

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Theorem about the relationship between two circles -/
theorem circle_relationship (R₁ R₂ d : ℝ) (c₁ c₂ : Circle) 
  (h₁ : c₁.radius = R₁)
  (h₂ : c₂.radius = R₂)
  (h₃ : R₁ ≠ R₂)
  (h₄ : ∃ x : ℝ, x^2 - 2*R₁*x + R₂^2 - d*(R₂ - R₁) = 0 ∧ 
        ∀ y : ℝ, y^2 - 2*R₁*y + R₂^2 - d*(R₂ - R₁) = 0 → y = x) :
  R₁ + R₂ = d ∧ (∀ p : ℝ × ℝ, ‖p‖ ≠ R₁ ∨ ‖p - (d, 0)‖ ≠ R₂) := by
sorry

end NUMINAMATH_CALUDE_circle_relationship_l1580_158072


namespace NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l1580_158088

theorem sum_of_angles_two_triangles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ) :
  angle1 + angle3 + angle5 = 180 →
  angle2 + angle4 + angle6 = 180 →
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l1580_158088


namespace NUMINAMATH_CALUDE_principal_amount_l1580_158083

/-- Given a principal amount P lent at simple interest rate r,
    prove that P = 710 given the conditions from the problem. -/
theorem principal_amount (P r : ℝ) : 
  (P + P * r * 3 = 920) →
  (P + P * r * 9 = 1340) →
  P = 710 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l1580_158083


namespace NUMINAMATH_CALUDE_tile_arrangement_count_l1580_158023

/-- The number of distinguishable arrangements of tiles -/
def tileArrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 1 purple, 3 green, and 2 yellow tiles is 420 -/
theorem tile_arrangement_count :
  tileArrangements 1 1 3 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangement_count_l1580_158023


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_asymptotes_l1580_158071

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 - y^2/25 = 1

-- Define the hyperbola
def hyperbola (K : ℝ) (x y : ℝ) : Prop := x^2/K + y^2/25 = 1

-- Define the asymptote condition
def same_asymptotes (K : ℝ) : Prop := ∀ (x y : ℝ), y = (5/4)*x ↔ y = (5/Real.sqrt K)*x

-- Theorem statement
theorem ellipse_hyperbola_asymptotes (K : ℝ) : 
  (∀ (x y : ℝ), ellipse x y ∧ hyperbola K x y) → same_asymptotes K → K = 16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_asymptotes_l1580_158071


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1580_158056

/-- Given a quadratic function f(x) = x^2 + 4x + c, prove that f(1) > c > f(-2) -/
theorem quadratic_inequality (c : ℝ) : let f : ℝ → ℝ := λ x ↦ x^2 + 4*x + c
  f 1 > c ∧ c > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1580_158056


namespace NUMINAMATH_CALUDE_integral_sqrt_rational_l1580_158061

open Real MeasureTheory

/-- The definite integral of 5√(x+24) / ((x+24)^2 * √x) from x = 1 to x = 8 is equal to 1/8 -/
theorem integral_sqrt_rational : 
  ∫ x in (1 : ℝ)..8, (5 * Real.sqrt (x + 24)) / ((x + 24)^2 * Real.sqrt x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_rational_l1580_158061


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l1580_158089

/-- A function f is decreasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of all real numbers x satisfying f(1/|x|) < f(1) for a decreasing function f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f (1 / |x|) < f 1}

theorem solution_set_is_open_interval (f : ℝ → ℝ) (h : DecreasingOn f) :
  SolutionSet f = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l1580_158089


namespace NUMINAMATH_CALUDE_previous_painting_price_l1580_158005

/-- Proves the price of a previous painting given the price of the most recent painting and the relationship between the two prices. -/
theorem previous_painting_price (recent_price : ℝ) (h1 : recent_price = 49000) 
  (h2 : recent_price = 3.5 * previous_price - 1000) : previous_price = 14285.71 := by
  sorry

end NUMINAMATH_CALUDE_previous_painting_price_l1580_158005


namespace NUMINAMATH_CALUDE_percentage_with_no_conditions_is_10_percent_l1580_158022

-- Define the total number of teachers
def total_teachers : ℕ := 150

-- Define the number of teachers with each condition
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 60
def diabetes : ℕ := 30

-- Define the number of teachers with combinations of conditions
def high_blood_pressure_and_heart_trouble : ℕ := 25
def heart_trouble_and_diabetes : ℕ := 10
def high_blood_pressure_and_diabetes : ℕ := 15
def all_three_conditions : ℕ := 5

-- Define the function to calculate the percentage
def percentage_with_no_conditions : ℚ :=
  let teachers_with_conditions := high_blood_pressure + heart_trouble + diabetes
    - high_blood_pressure_and_heart_trouble - heart_trouble_and_diabetes - high_blood_pressure_and_diabetes
    + all_three_conditions
  let teachers_with_no_conditions := total_teachers - teachers_with_conditions
  (teachers_with_no_conditions : ℚ) / (total_teachers : ℚ) * 100

-- Theorem statement
theorem percentage_with_no_conditions_is_10_percent :
  percentage_with_no_conditions = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_no_conditions_is_10_percent_l1580_158022


namespace NUMINAMATH_CALUDE_order_of_roots_l1580_158037

theorem order_of_roots (a b c : ℝ) 
  (ha : a = 4^(2/3)) 
  (hb : b = 3^(2/3)) 
  (hc : c = 25^(1/3)) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_roots_l1580_158037


namespace NUMINAMATH_CALUDE_G_equals_4F_l1580_158066

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (4*x + x^4)/(1 + 4*x^3)) / (1 - (4*x + x^4)/(1 + 4*x^3)))

theorem G_equals_4F (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1 ∧ 1 + 4*x^3 ≠ 0) : G x = 4 * F x := by
  sorry

end NUMINAMATH_CALUDE_G_equals_4F_l1580_158066


namespace NUMINAMATH_CALUDE_origin_symmetry_coordinates_l1580_158060

def point_symmetry (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem origin_symmetry_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let P1 : ℝ × ℝ := point_symmetry P.1 P.2
  P1 = (2, -3) := by sorry

end NUMINAMATH_CALUDE_origin_symmetry_coordinates_l1580_158060


namespace NUMINAMATH_CALUDE_not_always_parallel_to_intersection_l1580_158090

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_to_intersection
  (α β : Plane) (m n : Line) :
  ¬(∀ (α β : Plane) (m n : Line),
    line_parallel_plane m α ∧ intersect α β n → parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_not_always_parallel_to_intersection_l1580_158090


namespace NUMINAMATH_CALUDE_budget_allocation_home_electronics_l1580_158013

theorem budget_allocation_home_electronics (total_degrees : ℝ) 
  (microphotonics_percent : ℝ) (food_additives_percent : ℝ) 
  (genetically_modified_microorganisms_percent : ℝ) (industrial_lubricants_percent : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  total_degrees = 360 ∧ 
  microphotonics_percent = 13 ∧ 
  food_additives_percent = 15 ∧ 
  genetically_modified_microorganisms_percent = 29 ∧ 
  industrial_lubricants_percent = 8 ∧ 
  basic_astrophysics_degrees = 39.6 →
  (100 - (microphotonics_percent + food_additives_percent + 
    genetically_modified_microorganisms_percent + industrial_lubricants_percent + 
    (basic_astrophysics_degrees / total_degrees * 100))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_home_electronics_l1580_158013


namespace NUMINAMATH_CALUDE_tournament_committee_count_l1580_158065

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- The number of members chosen from the host team -/
def host_members : ℕ := 3

/-- The number of teams that select 2 members -/
def teams_select_two : ℕ := 3

/-- The number of teams that select 3 members (excluding the host) -/
def teams_select_three : ℕ := 1

/-- The total number of ways to form a tournament committee -/
def total_committees : ℕ := 229105500

theorem tournament_committee_count :
  (num_teams) *
  (Nat.choose team_size host_members) *
  (Nat.choose (num_teams - 1) teams_select_three) *
  (Nat.choose team_size host_members) *
  (Nat.choose team_size 2 ^ teams_select_two) = total_committees := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l1580_158065


namespace NUMINAMATH_CALUDE_sum_first_five_multiples_of_twelve_l1580_158000

theorem sum_first_five_multiples_of_twelve : 
  (Finset.range 5).sum (fun i => 12 * (i + 1)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_five_multiples_of_twelve_l1580_158000


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_plane_l1580_158070

/-- The line passing through the point (1, -1, 1) in the direction (1, 0, -1) -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (1 + t, -1, 1 - t)

/-- The plane with equation 3x - 2y - 4z - 8 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  3 * x - 2 * y - 4 * z - 8 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (-6, -1, 8)

theorem intersection_point_on_line_and_plane :
  (∃ t : ℝ, line t = intersection_point) ∧
  plane intersection_point ∧
  (∀ p : ℝ × ℝ × ℝ, (∃ t : ℝ, line t = p) → plane p → p = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_line_and_plane_l1580_158070


namespace NUMINAMATH_CALUDE_f_of_5_eq_110_l1580_158026

/-- The polynomial function f(x) = 3x^4 - 20x^3 + 38x^2 - 35x - 40 -/
def f (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 38 * x^2 - 35 * x - 40

/-- Theorem: f(5) = 110 -/
theorem f_of_5_eq_110 : f 5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_eq_110_l1580_158026


namespace NUMINAMATH_CALUDE_gold_distribution_theorem_l1580_158044

/-- The number of gold nuggets -/
def n : ℕ := 2020

/-- The sum of masses of all nuggets -/
def total_mass : ℕ := n * (n + 1) / 2

/-- The maximum difference in mass between the two chests -/
def max_diff : ℕ := n

/-- The guaranteed amount of gold in the heavier chest -/
def guaranteed_mass : ℕ := total_mass / 2 + max_diff / 2

theorem gold_distribution_theorem :
  ∃ (chest_mass : ℕ), chest_mass ≥ guaranteed_mass ∧ 
  chest_mass ≤ total_mass - (total_mass / 2 - max_diff / 2) :=
sorry

end NUMINAMATH_CALUDE_gold_distribution_theorem_l1580_158044


namespace NUMINAMATH_CALUDE_function_properties_l1580_158034

-- Define the function f from X to Y
variable {X Y : Type*}
variable (f : X → Y)

-- Theorem stating that none of the given statements are necessarily true for all functions
theorem function_properties :
  (∃ y : Y, ∀ x : X, f x ≠ y) ∧  -- Some elements in Y might not have a preimage in X
  (∃ x₁ x₂ : X, x₁ ≠ x₂ ∧ f x₁ = f x₂) ∧  -- Different elements in X can have the same image in Y
  (∃ y : Y, True)  -- Y is not empty
  :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1580_158034


namespace NUMINAMATH_CALUDE_half_job_days_l1580_158092

/-- 
Proves that the number of days to complete half a job is 6, 
given that it takes 6 more days to finish the entire job after completing half of it.
-/
theorem half_job_days : 
  ∀ (x : ℝ), (x + 6 = 2*x) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_half_job_days_l1580_158092


namespace NUMINAMATH_CALUDE_polynomial_identity_l1580_158094

theorem polynomial_identity (P : ℝ → ℝ) 
  (h1 : ∀ x, P (x^3) = (P x)^3) 
  (h2 : P 2 = 2) :
  ∀ x, P x = x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1580_158094


namespace NUMINAMATH_CALUDE_tims_age_l1580_158030

theorem tims_age (tim rommel jenny : ℕ) 
  (h1 : rommel = 3 * tim)
  (h2 : jenny = rommel + 2)
  (h3 : tim + 12 = jenny) :
  tim = 5 := by
sorry

end NUMINAMATH_CALUDE_tims_age_l1580_158030


namespace NUMINAMATH_CALUDE_count_divides_sum_product_l1580_158006

def divides_sum_product (n : ℕ+) : Prop :=
  (n.val * (n.val + 1) / 2) ∣ (10 * n.val)

theorem count_divides_sum_product :
  ∃ (S : Finset ℕ+), (∀ n, n ∈ S ↔ divides_sum_product n) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_divides_sum_product_l1580_158006


namespace NUMINAMATH_CALUDE_second_number_proof_l1580_158029

theorem second_number_proof (a b c : ℚ) : 
  a + b + c = 98 ∧ 
  a / b = 2 / 3 ∧ 
  b / c = 5 / 8 → 
  b = 30 :=
by sorry

end NUMINAMATH_CALUDE_second_number_proof_l1580_158029


namespace NUMINAMATH_CALUDE_smallest_number_l1580_158075

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def number_a : List Nat := [2, 0]
def number_b : List Nat := [3, 0]
def number_c : List Nat := [2, 3]
def number_d : List Nat := [3, 1]

theorem smallest_number :
  let a := base_to_decimal number_a 7
  let b := base_to_decimal number_b 5
  let c := base_to_decimal number_c 6
  let d := base_to_decimal number_d 4
  d < a ∧ d < b ∧ d < c := by sorry

end NUMINAMATH_CALUDE_smallest_number_l1580_158075


namespace NUMINAMATH_CALUDE_unique_natural_number_satisfying_conditions_l1580_158028

theorem unique_natural_number_satisfying_conditions :
  ∃! (x : ℕ), 
    (∃ (k : ℕ), 3 * x + 1 = k^2) ∧ 
    (∃ (t : ℕ), 6 * x - 2 = t^2) ∧ 
    Nat.Prime (6 * x^2 - 1) ∧
    x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_natural_number_satisfying_conditions_l1580_158028


namespace NUMINAMATH_CALUDE_cloth_cost_calculation_l1580_158027

/-- The total cost of cloth given the length and price per metre -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem: The total cost of 9.25 m of cloth at $45 per metre is $416.25 -/
theorem cloth_cost_calculation :
  total_cost 9.25 45 = 416.25 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_calculation_l1580_158027


namespace NUMINAMATH_CALUDE_complex_number_coordinate_l1580_158017

theorem complex_number_coordinate (i : ℂ) (h : i^2 = -1) :
  (i^2015) / (i - 2) = -1/5 + 2/5 * i := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinate_l1580_158017


namespace NUMINAMATH_CALUDE_range_of_a_for_f_with_two_zeros_l1580_158039

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- State the theorem
theorem range_of_a_for_f_with_two_zeros :
  (∃ a : ℝ, ∀ x : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0)) →
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) → 1 ≤ a ∧ a ≤ 5) ∧
  (∀ a : ℝ, 1 ≤ a ∧ a ≤ 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_f_with_two_zeros_l1580_158039


namespace NUMINAMATH_CALUDE_negation_of_exactly_one_intersection_negation_of_if_3_or_4_then_equation_l1580_158097

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the proposition for exactly one intersection point
def exactly_one_intersection (a b c : ℝ) : Prop :=
  ∃! x, f a b c x = 0

-- Define the proposition for no or at least two intersection points
def no_or_at_least_two_intersections (a b c : ℝ) : Prop :=
  (∀ x, f a b c x ≠ 0) ∨ (∃ x y, x ≠ y ∧ f a b c x = 0 ∧ f a b c y = 0)

-- Theorem for the negation of the first proposition
theorem negation_of_exactly_one_intersection (a b c : ℝ) :
  ¬(exactly_one_intersection a b c) ↔ no_or_at_least_two_intersections a b c :=
sorry

-- Define the proposition for the second statement
def if_3_or_4_then_equation : Prop :=
  (3^2 - 7*3 + 12 = 0) ∧ (4^2 - 7*4 + 12 = 0)

-- Theorem for the negation of the second proposition
theorem negation_of_if_3_or_4_then_equation :
  ¬if_3_or_4_then_equation ↔ (3^2 - 7*3 + 12 ≠ 0) ∨ (4^2 - 7*4 + 12 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_exactly_one_intersection_negation_of_if_3_or_4_then_equation_l1580_158097


namespace NUMINAMATH_CALUDE_work_completion_time_l1580_158058

-- Define the work completion times for Paul and Rose
def paul_time : ℝ := 80
def rose_time : ℝ := 120

-- Define the theorem
theorem work_completion_time : 
  let paul_rate := 1 / paul_time
  let rose_rate := 1 / rose_time
  let combined_rate := paul_rate + rose_rate
  (1 / combined_rate) = 48 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1580_158058


namespace NUMINAMATH_CALUDE_missing_village_population_l1580_158043

def village_populations : List ℕ := [803, 900, 1100, 945, 980, 1249]

theorem missing_village_population 
  (total_villages : ℕ) 
  (average_population : ℕ) 
  (known_populations : List ℕ) 
  (h1 : total_villages = 7)
  (h2 : average_population = 1000)
  (h3 : known_populations = village_populations)
  (h4 : known_populations.length = 6) :
  ∃ (missing_population : ℕ), 
    missing_population = total_villages * average_population - known_populations.sum ∧
    missing_population = 1023 :=
by sorry

end NUMINAMATH_CALUDE_missing_village_population_l1580_158043


namespace NUMINAMATH_CALUDE_no_solutions_to_absolute_value_equation_l1580_158041

theorem no_solutions_to_absolute_value_equation :
  ¬∃ y : ℝ, |y - 2| = |y - 1| + |y - 4| := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_absolute_value_equation_l1580_158041


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_divisors_eq_two_l1580_158079

def sum_of_divisors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x)

def sum_of_reciprocal_divisors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => 1 / x)

theorem sum_of_reciprocal_divisors_eq_two (n : ℕ) (h : sum_of_divisors n = 2 * n) :
  sum_of_reciprocal_divisors n = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_divisors_eq_two_l1580_158079


namespace NUMINAMATH_CALUDE_cameron_typing_difference_l1580_158096

theorem cameron_typing_difference (
  speed_before : ℕ) 
  (speed_after : ℕ) 
  (time : ℕ) 
  (h1 : speed_before = 10) 
  (h2 : speed_after = 8) 
  (h3 : time = 5) : 
  speed_before * time - speed_after * time = 10 :=
by sorry

end NUMINAMATH_CALUDE_cameron_typing_difference_l1580_158096


namespace NUMINAMATH_CALUDE_wire_ratio_l1580_158074

theorem wire_ratio (x y : ℝ) : 
  x > 0 → y > 0 → 
  (4 * (x / 4) = 5 * (y / 5)) → 
  x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_l1580_158074


namespace NUMINAMATH_CALUDE_perimeter_of_fourth_figure_l1580_158008

/-- Given four planar figures composed of identical triangles, prove that the perimeter of the fourth figure is 10 cm. -/
theorem perimeter_of_fourth_figure
  (p₁ : ℝ) (p₂ : ℝ) (p₃ : ℝ) (p₄ : ℝ)
  (h₁ : p₁ = 8)
  (h₂ : p₂ = 11.4)
  (h₃ : p₃ = 14.7)
  (h_relation : p₁ + p₂ + p₄ = 2 * p₃) :
  p₄ = 10 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_fourth_figure_l1580_158008


namespace NUMINAMATH_CALUDE_particular_number_problem_l1580_158081

theorem particular_number_problem : ∃! x : ℚ, ((x / 23) - 67) * 2 = 102 :=
  by sorry

end NUMINAMATH_CALUDE_particular_number_problem_l1580_158081


namespace NUMINAMATH_CALUDE_ball_returns_in_three_throws_l1580_158016

/-- The number of boys in the circle -/
def n : ℕ := 15

/-- The number of positions skipped in each throw (including the thrower) -/
def skip : ℕ := 5

/-- The sequence of positions the ball reaches -/
def ball_sequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | i + 1 => (ball_sequence start i + skip) % n

/-- The theorem stating that it takes 3 throws for the ball to return to the start -/
theorem ball_returns_in_three_throws (start : ℕ) (h : start > 0 ∧ start ≤ n) : 
  ball_sequence start 3 = start :=
sorry

end NUMINAMATH_CALUDE_ball_returns_in_three_throws_l1580_158016


namespace NUMINAMATH_CALUDE_polynomial_product_simplification_l1580_158010

theorem polynomial_product_simplification (x y : ℝ) :
  (3 * x^2 - 7 * y^3) * (9 * x^4 + 21 * x^2 * y^3 + 49 * y^6) = 27 * x^6 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_simplification_l1580_158010


namespace NUMINAMATH_CALUDE_gcd_228_1995_l1580_158053

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l1580_158053


namespace NUMINAMATH_CALUDE_equation_roots_l1580_158042

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (21 / (x^2 - 9)) - (3 / (x - 3)) - 2
  ∀ x : ℝ, f x = 0 ↔ x = -3 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l1580_158042


namespace NUMINAMATH_CALUDE_athena_spent_14_l1580_158059

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℝ) (sandwich_count : ℕ) (drink_price : ℝ) (drink_count : ℕ) : ℝ :=
  sandwich_price * sandwich_count + drink_price * drink_count

/-- Theorem stating that Athena spent $14 on snacks -/
theorem athena_spent_14 :
  total_spent 3 3 2.5 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_athena_spent_14_l1580_158059


namespace NUMINAMATH_CALUDE_cookies_per_person_l1580_158014

/-- The number of cookie batches Beth bakes in a week -/
def batches : ℕ := 8

/-- The number of dozens of cookies in each batch -/
def dozens_per_batch : ℕ := 5

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of people sharing the cookies -/
def people : ℕ := 30

/-- Theorem: If 8 batches of 5 dozen cookies are shared equally among 30 people,
    each person will receive 16 cookies -/
theorem cookies_per_person :
  (batches * dozens_per_batch * cookies_per_dozen) / people = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l1580_158014


namespace NUMINAMATH_CALUDE_area_of_region_l1580_158004

-- Define the region
def region (x y : ℝ) : Prop := 
  |x - 2*y^2| + x + 2*y^2 ≤ 8 - 4*y

-- Define symmetry about y-axis
def symmetric_about_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem area_of_region : 
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ region x y) ∧ 
    symmetric_about_y_axis S ∧
    MeasureTheory.volume S = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l1580_158004


namespace NUMINAMATH_CALUDE_orange_count_l1580_158062

/-- The number of oranges in a bin after some changes -/
def final_oranges (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Proof that given 40 initial oranges, removing 25 and adding 21 results in 36 oranges -/
theorem orange_count : final_oranges 40 25 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l1580_158062


namespace NUMINAMATH_CALUDE_angle_a_is_30_angle_b_is_150_polygon_sides_is_12_march_day_is_24_l1580_158024

-- Define the angle a from the geometric figure
def a : ℝ := 30

-- Define the angle b
def b : ℝ := 150

-- Define the number of sides n in the regular polygon
def n : ℕ := 12

-- Define k as the day of March
def k : ℕ := 24

-- Theorem 1: The angle a in the given geometric figure is 30°
theorem angle_a_is_30 : a = 30 := by sorry

-- Theorem 2: If sin(30° + 210°) = cos b° and 90° < b < 180°, then b = 150°
theorem angle_b_is_150 (h1 : Real.sin (30 + 210) = Real.cos b) (h2 : 90 < b ∧ b < 180) : b = 150 := by sorry

-- Theorem 3: If each interior angle of an n-sided regular polygon is 150°, then n = 12
theorem polygon_sides_is_12 (h : (n - 2) * 180 / n = 150) : n = 12 := by sorry

-- Theorem 4: If the nth day of March is Friday, the kth day is Wednesday, and 20 < k < 25, then k = 24
theorem march_day_is_24 (h1 : k % 7 = (n + 3) % 7) (h2 : 20 < k ∧ k < 25) : k = 24 := by sorry

end NUMINAMATH_CALUDE_angle_a_is_30_angle_b_is_150_polygon_sides_is_12_march_day_is_24_l1580_158024


namespace NUMINAMATH_CALUDE_computer_purchase_cost_l1580_158012

theorem computer_purchase_cost (computer_cost : ℕ) (base_video_card_cost : ℕ) 
  (h1 : computer_cost = 1500)
  (h2 : base_video_card_cost = 300) : 
  computer_cost + 
  (computer_cost / 5) + 
  (2 * base_video_card_cost - base_video_card_cost) = 2100 := by
  sorry

#check computer_purchase_cost

end NUMINAMATH_CALUDE_computer_purchase_cost_l1580_158012


namespace NUMINAMATH_CALUDE_translate_down_two_units_l1580_158036

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateVertically (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - amount }

theorem translate_down_two_units :
  let original := Line.mk (-2) 0
  let translated := translateVertically original 2
  translated = Line.mk (-2) (-2) := by sorry

end NUMINAMATH_CALUDE_translate_down_two_units_l1580_158036


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l1580_158076

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l1580_158076


namespace NUMINAMATH_CALUDE_factorize_expression1_factorize_expression2_l1580_158078

-- First expression
theorem factorize_expression1 (y : ℝ) :
  y + (y - 4) * (y - 1) = (y - 2)^2 := by sorry

-- Second expression
theorem factorize_expression2 (x y a b : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a - 2 * b) * (3 * a + 2 * b) := by sorry

end NUMINAMATH_CALUDE_factorize_expression1_factorize_expression2_l1580_158078


namespace NUMINAMATH_CALUDE_abc_product_l1580_158031

theorem abc_product (a b c : ℤ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a + b + c = 30 → 
  1 / a + 1 / b + 1 / c + 450 / (a * b * c) = 1 → 
  a * b * c = 1920 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1580_158031


namespace NUMINAMATH_CALUDE_parallelogram_properties_l1580_158003

def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

theorem parallelogram_properties :
  let D : ℂ := 4 + 3 * Complex.I
  let diagonal_BD : ℂ := D - B
  (A + C = B + D) ∧ 
  (Complex.abs diagonal_BD = 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l1580_158003


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l1580_158049

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 8y + 24
    and the point (-3, 4) is 10. -/
theorem distance_circle_center_to_point :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = 6*x - 8*y + 24
  let center : ℝ × ℝ := (3, -4)
  let point : ℝ × ℝ := (-3, 4)
  (∃ (x y : ℝ), circle_eq x y) →
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l1580_158049


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1580_158047

theorem smallest_fraction_between (p q : ℕ+) 
  (h1 : (3 : ℚ) / 5 < p / q)
  (h2 : p / q < (5 : ℚ) / 8)
  (h3 : ∀ (r s : ℕ+), (3 : ℚ) / 5 < r / s → r / s < (5 : ℚ) / 8 → s ≤ q) :
  q - p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1580_158047


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1580_158025

/-- The coefficient of the third term in the binomial expansion of (a + √x)^5 -/
def third_term_coefficient (a : ℝ) (x : ℝ) : ℝ := 10 * a^3 * x

/-- Theorem: If the coefficient of the third term in (a + √x)^5 is 80, then a = 2 -/
theorem binomial_expansion_coefficient (a : ℝ) (x : ℝ) :
  third_term_coefficient a x = 80 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1580_158025


namespace NUMINAMATH_CALUDE_people_joined_line_l1580_158018

theorem people_joined_line (initial : ℕ) (left : ℕ) (current : ℕ) : 
  initial ≥ left → 
  current = (initial - left) + (current - (initial - left)) :=
by sorry

end NUMINAMATH_CALUDE_people_joined_line_l1580_158018


namespace NUMINAMATH_CALUDE_movie_ticket_price_decrease_l1580_158064

theorem movie_ticket_price_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100)
  (h2 : new_price = 80) : 
  (old_price - new_price) / old_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_price_decrease_l1580_158064


namespace NUMINAMATH_CALUDE_kate_lives_on_15_dollars_per_month_kate_has_frugal_lifestyle_l1580_158085

/-- Represents a person living in New York --/
structure NYResident where
  name : String
  monthly_expenses : ℕ
  uses_dumpster_diving : Bool
  has_frugal_habits : Bool

/-- Represents Kate Hashimoto --/
def kate : NYResident :=
  { name := "Kate Hashimoto"
  , monthly_expenses := 15
  , uses_dumpster_diving := true
  , has_frugal_habits := true }

/-- Theorem stating that Kate can live on $15 a month in New York --/
theorem kate_lives_on_15_dollars_per_month :
  kate.monthly_expenses = 15 ∧ kate.uses_dumpster_diving ∧ kate.has_frugal_habits :=
by sorry

/-- Definition of a frugal lifestyle in New York --/
def is_frugal_lifestyle (r : NYResident) : Prop :=
  r.monthly_expenses ≤ 15 ∧ r.uses_dumpster_diving ∧ r.has_frugal_habits

/-- Theorem stating that Kate has a frugal lifestyle --/
theorem kate_has_frugal_lifestyle : is_frugal_lifestyle kate :=
by sorry

end NUMINAMATH_CALUDE_kate_lives_on_15_dollars_per_month_kate_has_frugal_lifestyle_l1580_158085


namespace NUMINAMATH_CALUDE_inequality_proof_l1580_158033

theorem inequality_proof (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : 1/x + 1/y + 1/z = 1) : 
  a^x + b^y + c^z ≥ (4*a*b*c*x*y*z) / ((x+y+z-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1580_158033


namespace NUMINAMATH_CALUDE_min_value_expression_l1580_158054

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y^2 * z = 64) :
  x^2 + 8*x*y + 8*y^2 + 4*z^2 ≥ 1536 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y^2 * z = 64 ∧ x^2 + 8*x*y + 8*y^2 + 4*z^2 = 1536 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1580_158054


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1580_158057

theorem arithmetic_expression_evaluation : 2 - (3 - 4) - (5 - 6 - 7) = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1580_158057


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1580_158011

theorem triangle_perimeter (a b x : ℝ) : 
  a = 7 → 
  b = 11 → 
  x^2 - 25 = 2*(x - 5)^2 → 
  x > 0 →
  a + b > x →
  x + b > a →
  x + a > b →
  (a + b + x = 23 ∨ a + b + x = 33) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1580_158011


namespace NUMINAMATH_CALUDE_crown_cost_l1580_158046

/-- Given a total payment of $22,000 for a crown including a 10% tip,
    prove that the original cost of the crown was $20,000. -/
theorem crown_cost (total_payment : ℝ) (tip_percentage : ℝ) (h1 : total_payment = 22000)
    (h2 : tip_percentage = 0.1) : 
  ∃ (original_cost : ℝ), 
    original_cost * (1 + tip_percentage) = total_payment ∧ 
    original_cost = 20000 := by
  sorry

end NUMINAMATH_CALUDE_crown_cost_l1580_158046


namespace NUMINAMATH_CALUDE_function_properties_l1580_158051

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (4-a)*x^2 - 15*x + a

-- Define the derivative of f(x) with respect to x
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(4-a)*x - 15

theorem function_properties :
  -- Part 1: When f(0) = -2, a = -2
  (∀ a : ℝ, f a 0 = -2 → a = -2) ∧
  
  -- Part 2: The minimum value of f(x) when a = -2 is -10
  (∃ x : ℝ, f (-2) x = -10 ∧ ∀ y : ℝ, f (-2) y ≥ -10) ∧
  
  -- Part 3: The maximum value of a for which f'(x) ≤ 0 on (-1, 1) is 10
  (∀ a : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f' a x ≤ 0) → a ≤ 10) ∧
  (∃ a : ℝ, a = 10 ∧ ∀ x : ℝ, -1 < x ∧ x < 1 → f' a x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1580_158051


namespace NUMINAMATH_CALUDE_lee_lawn_mowing_earnings_l1580_158019

/-- Lee's lawn mowing earnings problem -/
theorem lee_lawn_mowing_earnings :
  ∀ (charge_per_lawn : ℕ) (lawns_mowed : ℕ) (tip_amount : ℕ) (num_tippers : ℕ),
    charge_per_lawn = 33 →
    lawns_mowed = 16 →
    tip_amount = 10 →
    num_tippers = 3 →
    charge_per_lawn * lawns_mowed + tip_amount * num_tippers = 558 :=
by
  sorry


end NUMINAMATH_CALUDE_lee_lawn_mowing_earnings_l1580_158019
