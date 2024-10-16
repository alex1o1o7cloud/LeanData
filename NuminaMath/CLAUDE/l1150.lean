import Mathlib

namespace NUMINAMATH_CALUDE_restaurant_order_combinations_l1150_115042

def menu_size : ℕ := 15
def num_people : ℕ := 3

theorem restaurant_order_combinations :
  menu_size ^ num_people = 3375 := by sorry

end NUMINAMATH_CALUDE_restaurant_order_combinations_l1150_115042


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1150_115037

theorem water_tank_capacity : 
  ∀ (tank_capacity : ℝ),
  (0.75 * tank_capacity - 0.4 * tank_capacity = 36) →
  ⌈tank_capacity⌉ = 103 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1150_115037


namespace NUMINAMATH_CALUDE_line_param_correct_l1150_115052

/-- The line y = 2x - 6 parameterized as (x, y) = (r, 2) + t(3, k) -/
def line_param (r k t : ℝ) : ℝ × ℝ :=
  (r + 3 * t, 2 + k * t)

/-- The line equation y = 2x - 6 -/
def line_eq (x y : ℝ) : Prop :=
  y = 2 * x - 6

theorem line_param_correct (r k : ℝ) : 
  (∀ t, line_eq (line_param r k t).1 (line_param r k t).2) ↔ r = 4 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_param_correct_l1150_115052


namespace NUMINAMATH_CALUDE_division_problem_l1150_115044

theorem division_problem (n : ℕ) : n = 867 → n / 37 = 23 ∧ n % 37 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1150_115044


namespace NUMINAMATH_CALUDE_markus_family_ages_l1150_115024

theorem markus_family_ages :
  ∀ (grandson_age son_age markus_age : ℕ),
    son_age = 2 * grandson_age →
    markus_age = 2 * son_age →
    grandson_age + son_age + markus_age = 140 →
    grandson_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_markus_family_ages_l1150_115024


namespace NUMINAMATH_CALUDE_vector_decomposition_l1150_115000

def x : Fin 3 → ℝ := ![(-9 : ℝ), 5, 5]
def p : Fin 3 → ℝ := ![(4 : ℝ), 1, 1]
def q : Fin 3 → ℝ := ![(2 : ℝ), 0, -3]
def r : Fin 3 → ℝ := ![(-1 : ℝ), 2, 1]

theorem vector_decomposition :
  x = (-1 : ℝ) • p + (-1 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1150_115000


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l1150_115004

theorem quadratic_equation_two_distinct_roots (c d : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ + c) * (x₁ + d) - (2 * x₁ + c + d) = 0 ∧
  (x₂ + c) * (x₂ + d) - (2 * x₂ + c + d) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l1150_115004


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1150_115032

/-- The quadratic polynomial that satisfies given conditions -/
def q (x : ℝ) : ℝ := 2.1 * x^2 - 3.1 * x - 1.2

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions : q (-1) = 4 ∧ q 2 = 1 ∧ q 4 = 20 := by
  sorry

#eval q (-1)
#eval q 2
#eval q 4

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1150_115032


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l1150_115048

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) + a n

def increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem fibonacci_like_sequence (a : ℕ → ℕ) 
  (h1 : sequence_property a) 
  (h2 : increasing_sequence a)
  (h3 : a 7 = 120) : 
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l1150_115048


namespace NUMINAMATH_CALUDE_book_reading_ratio_l1150_115012

theorem book_reading_ratio (total_pages : ℕ) (total_days : ℕ) (speed1 speed2 : ℕ) 
  (h1 : total_pages = 500)
  (h2 : total_days = 75)
  (h3 : speed1 = 10)
  (h4 : speed2 = 5)
  (h5 : ∃ x : ℕ, speed1 * x + speed2 * (total_days - x) = total_pages) :
  ∃ x : ℕ, (speed1 * x : ℚ) / total_pages = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l1150_115012


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1150_115006

theorem quadratic_one_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 30 * x + 12 = 0) :
  ∃ x, a * x^2 + 30 * x + 12 = 0 ∧ x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1150_115006


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l1150_115030

theorem sin_cos_fourth_power_range :
  ∀ x : ℝ, (1/2 : ℝ) ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l1150_115030


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1150_115035

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.00000065 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 6.5 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1150_115035


namespace NUMINAMATH_CALUDE_inequality_proof_l1150_115096

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1150_115096


namespace NUMINAMATH_CALUDE_power_function_through_point_l1150_115051

/-- Given a power function f(x) = x^n that passes through (2, √2/2), prove f(4) = 1/2 -/
theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1150_115051


namespace NUMINAMATH_CALUDE_expression_value_l1150_115045

theorem expression_value : 3^(1^(2^8)) + ((3^1)^2)^8 = 43046724 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1150_115045


namespace NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l1150_115027

theorem largest_sum_is_five_sixths : 
  let sums := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/6]
  ∀ x ∈ sums, x ≤ 5/6 ∧ (5/6 : ℚ) ∈ sums :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_is_five_sixths_l1150_115027


namespace NUMINAMATH_CALUDE_milk_consumption_l1150_115069

/-- The amount of regular milk consumed in a week -/
def regular_milk : ℝ := 0.5

/-- The amount of soy milk consumed in a week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk consumed in a week -/
def total_milk : ℝ := regular_milk + soy_milk

theorem milk_consumption : total_milk = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_milk_consumption_l1150_115069


namespace NUMINAMATH_CALUDE_shed_area_calculation_l1150_115014

/-- The total inside surface area of a rectangular shed -/
def shedSurfaceArea (width length height : ℝ) : ℝ :=
  2 * (width * height + length * height + width * length)

theorem shed_area_calculation :
  let width : ℝ := 12
  let length : ℝ := 15
  let height : ℝ := 7
  shedSurfaceArea width length height = 738 := by
  sorry

end NUMINAMATH_CALUDE_shed_area_calculation_l1150_115014


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l1150_115081

/-- Represents the number of valid delivery sequences for n houses -/
def E : ℕ → ℕ
  | 0 => 0  -- No houses, no deliveries
  | 1 => 2  -- For one house, two options: deliver or not
  | 2 => 4  -- For two houses, all combinations are valid
  | 3 => 8  -- E_3 = E_2 + E_1 + 2
  | n + 4 => E (n + 3) + E (n + 2) + E (n + 1)

/-- The problem statement -/
theorem paperboy_delivery_ways : E 12 = 1854 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l1150_115081


namespace NUMINAMATH_CALUDE_not_perfect_square_l1150_115033

theorem not_perfect_square : ¬ ∃ (n : ℕ), n^2 = 425102348541 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1150_115033


namespace NUMINAMATH_CALUDE_circle_m_range_and_perpendicular_intersection_l1150_115017

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_m_range_and_perpendicular_intersection :
  -- Part 1: If the equation represents a circle, then m ∈ (-∞, 5)
  (∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5) ∧
  -- Part 2: If the circle intersects the line and OM ⟂ ON, then m = 8/5
  (∀ m : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ, 
      circle_equation x1 y1 m ∧ 
      circle_equation x2 y2 m ∧
      line_equation x1 y1 ∧ 
      line_equation x2 y2 ∧
      perpendicular x1 y1 x2 y2) → 
    m = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_circle_m_range_and_perpendicular_intersection_l1150_115017


namespace NUMINAMATH_CALUDE_disc_probability_l1150_115022

theorem disc_probability (p_A p_B p_C p_D p_E : ℝ) : 
  p_A = 1/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_disc_probability_l1150_115022


namespace NUMINAMATH_CALUDE_carrots_per_pound_l1150_115063

/-- Given the number of carrots in three beds and the total weight of the harvest,
    calculate the number of carrots that weigh one pound. -/
theorem carrots_per_pound 
  (bed1 bed2 bed3 : ℕ) 
  (total_weight : ℕ) 
  (h1 : bed1 = 55)
  (h2 : bed2 = 101)
  (h3 : bed3 = 78)
  (h4 : total_weight = 39) :
  (bed1 + bed2 + bed3) / total_weight = 6 := by
  sorry

#check carrots_per_pound

end NUMINAMATH_CALUDE_carrots_per_pound_l1150_115063


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l1150_115013

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - x

theorem f_monotonicity_and_range (a : ℝ) :
  (a ≥ 1/8 → ∀ x > 0, StrictMono (f a)) ∧
  (∀ x ≥ 1, f a x ≥ 0 ↔ a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l1150_115013


namespace NUMINAMATH_CALUDE_jason_remaining_cards_l1150_115059

def initial_cards : ℕ := 3
def cards_bought : ℕ := 2

theorem jason_remaining_cards : initial_cards - cards_bought = 1 := by
  sorry

end NUMINAMATH_CALUDE_jason_remaining_cards_l1150_115059


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1150_115088

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I : ℂ) * 2 + 1 * (a : ℂ) + (b : ℂ) = Complex.I * 2 →
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1150_115088


namespace NUMINAMATH_CALUDE_sector_central_angle_l1150_115015

/-- A sector with radius 1 cm and circumference 4 cm has a central angle of 2 radians. -/
theorem sector_central_angle (r : ℝ) (circ : ℝ) (h1 : r = 1) (h2 : circ = 4) :
  let arc_length := circ - 2 * r
  arc_length / r = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1150_115015


namespace NUMINAMATH_CALUDE_percentage_increase_l1150_115087

theorem percentage_increase (initial : ℝ) (final : ℝ) (percentage : ℝ) : 
  initial = 240 → final = 288 → percentage = 20 →
  (final - initial) / initial * 100 = percentage := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1150_115087


namespace NUMINAMATH_CALUDE_min_c_value_l1150_115060

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2*x + y = 2019 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1010 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l1150_115060


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l1150_115016

theorem positive_integer_solutions_of_equation : 
  {(x, y) : ℕ × ℕ | x + y + x * y = 2008} = 
  {(6, 286), (286, 6), (40, 48), (48, 40)} := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l1150_115016


namespace NUMINAMATH_CALUDE_triangle_side_value_l1150_115084

noncomputable section

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions for a valid triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : triangle A B C a b c)
  (h_angle : A = 2 * C)
  (h_side_c : c = 2)
  (h_side_a : a^2 = 4*b - 4) :
  a = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l1150_115084


namespace NUMINAMATH_CALUDE_f_monotonicity_and_maximum_l1150_115058

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2

theorem f_monotonicity_and_maximum (k : ℝ) :
  (k = 1 →
    (∀ x y, x < y ∧ y < 0 → f k x < f k y) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < Real.log 2 → f k x > f k y) ∧
    (∀ x y, Real.log 2 < x ∧ x < y → f k x < f k y)) ∧
  (1/2 < k ∧ k ≤ 1 →
    ∀ x, x ∈ Set.Icc 0 k → f k x ≤ (k - 1) * Real.exp k - k^3) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_maximum_l1150_115058


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l1150_115074

theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3944) : 
  total_cost / (4 * Real.sqrt area) = 58 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l1150_115074


namespace NUMINAMATH_CALUDE_roots_when_p_is_8_p_value_when_root_is_3_plus_4i_l1150_115007

-- Define the complex quadratic equation
def complex_quadratic (p : ℝ) (x : ℂ) : ℂ := x^2 - p*x + 25

-- Part 1: Prove that when p = 8, the roots are 4 + 3i and 4 - 3i
theorem roots_when_p_is_8 :
  let p : ℝ := 8
  let x₁ : ℂ := 4 + 3*I
  let x₂ : ℂ := 4 - 3*I
  complex_quadratic p x₁ = 0 ∧ complex_quadratic p x₂ = 0 :=
sorry

-- Part 2: Prove that when one root is 3 + 4i, p = 6
theorem p_value_when_root_is_3_plus_4i :
  let x₁ : ℂ := 3 + 4*I
  ∃ p : ℝ, complex_quadratic p x₁ = 0 ∧ p = 6 :=
sorry

end NUMINAMATH_CALUDE_roots_when_p_is_8_p_value_when_root_is_3_plus_4i_l1150_115007


namespace NUMINAMATH_CALUDE_sin_product_equality_l1150_115019

theorem sin_product_equality : 
  Real.sin (π / 14) * Real.sin (3 * π / 14) * Real.sin (5 * π / 14) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l1150_115019


namespace NUMINAMATH_CALUDE_dance_relationship_l1150_115049

/-- The number of girls that the nth boy dances with -/
def girls_danced (n : ℕ) : ℕ := n + 7

/-- The relationship between the number of boys (b) and girls (g) at a school dance -/
theorem dance_relationship (b g : ℕ) : 
  (∀ n : ℕ, n ≤ b → girls_danced n ≤ g) → 
  girls_danced b = g → 
  b = g - 7 :=
by sorry

end NUMINAMATH_CALUDE_dance_relationship_l1150_115049


namespace NUMINAMATH_CALUDE_sector_arc_length_l1150_115079

/-- Given a circular sector with a central angle of 40° and a radius of 18,
    the arc length is equal to 4π. -/
theorem sector_arc_length (θ : ℝ) (r : ℝ) (h1 : θ = 40) (h2 : r = 18) :
  (θ / 360) * (2 * π * r) = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1150_115079


namespace NUMINAMATH_CALUDE_hotel_profit_equation_correct_l1150_115073

/-- Represents a hotel's pricing and occupancy model -/
structure Hotel where
  baseRooms : ℕ
  basePrice : ℝ
  costPerRoom : ℝ
  vacancyRate : ℝ
  priceIncrease : ℝ
  desiredProfit : ℝ

/-- The profit equation for the hotel -/
def profitEquation (h : Hotel) : Prop :=
  (h.basePrice + h.priceIncrease - h.costPerRoom) * 
  (h.baseRooms - h.priceIncrease / h.vacancyRate) = h.desiredProfit

/-- Theorem stating that the given equation correctly represents the hotel's profit scenario -/
theorem hotel_profit_equation_correct (h : Hotel) 
  (hRooms : h.baseRooms = 50)
  (hBasePrice : h.basePrice = 180)
  (hCost : h.costPerRoom = 20)
  (hVacancy : h.vacancyRate = 10)
  (hProfit : h.desiredProfit = 10890) :
  profitEquation h := by sorry

end NUMINAMATH_CALUDE_hotel_profit_equation_correct_l1150_115073


namespace NUMINAMATH_CALUDE_intersection_distance_l1150_115047

theorem intersection_distance : ∃ (p1 p2 : ℝ × ℝ),
  (p1.1^2 + p1.2 = 12 ∧ p1.1 + p1.2 = 8) ∧
  (p2.1^2 + p2.2 = 12 ∧ p2.1 + p2.2 = 8) ∧
  p1 ≠ p2 ∧
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l1150_115047


namespace NUMINAMATH_CALUDE_P_homogeneous_P_symmetry_P_normalization_P_unique_l1150_115078

/-- A binary polynomial that satisfies the given conditions -/
def P (n : ℕ+) (x y : ℝ) : ℝ := x^(n : ℕ) - y^(n : ℕ)

/-- P is homogeneous of degree n -/
theorem P_homogeneous (n : ℕ+) (t x y : ℝ) :
  P n (t * x) (t * y) = t^(n : ℕ) * P n x y := by sorry

/-- P satisfies the symmetry condition -/
theorem P_symmetry (n : ℕ+) (a b c : ℝ) :
  P n (a + b) c + P n (b + c) a + P n (c + a) b = 0 := by sorry

/-- P satisfies the normalization condition -/
theorem P_normalization (n : ℕ+) :
  P n 1 0 = 1 := by sorry

/-- P is the unique polynomial satisfying all conditions -/
theorem P_unique (n : ℕ+) (Q : ℝ → ℝ → ℝ) 
  (h_homogeneous : ∀ t x y, Q (t * x) (t * y) = t^(n : ℕ) * Q x y)
  (h_symmetry : ∀ a b c, Q (a + b) c + Q (b + c) a + Q (c + a) b = 0)
  (h_normalization : Q 1 0 = 1) :
  ∀ x y, Q x y = P n x y := by sorry

end NUMINAMATH_CALUDE_P_homogeneous_P_symmetry_P_normalization_P_unique_l1150_115078


namespace NUMINAMATH_CALUDE_bottles_theorem_l1150_115057

/-- The number of ways to take out 24 bottles, where each time either 3 or 4 bottles are taken -/
def ways_to_take_bottles : ℕ :=
  -- Number of ways to take out 4 bottles 6 times
  1 +
  -- Number of ways to take out 3 bottles 8 times
  1 +
  -- Number of ways to take out 3 bottles 4 times and 4 bottles 3 times
  (Nat.choose 7 3)

/-- Theorem stating that the number of ways to take out the bottles is 37 -/
theorem bottles_theorem : ways_to_take_bottles = 37 := by
  sorry

#eval ways_to_take_bottles

end NUMINAMATH_CALUDE_bottles_theorem_l1150_115057


namespace NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l1150_115011

/-- Given a triangle ABC with centroid G, if GA^2 + GB^2 + GC^2 = 88, 
    then AB^2 + AC^2 + BC^2 = 396 -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  ((G.1 - A.1)^2 + (G.2 - A.2)^2 + 
   (G.1 - B.1)^2 + (G.2 - B.2)^2 + 
   (G.1 - C.1)^2 + (G.2 - C.2)^2 = 88) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 + 
   (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 396) := by
sorry

end NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l1150_115011


namespace NUMINAMATH_CALUDE_ngo_wage_problem_l1150_115009

/-- Calculates the initial daily average wage of illiterate employees in an NGO -/
def initial_illiterate_wage (num_illiterate : ℕ) (num_literate : ℕ) (new_illiterate_wage : ℕ) (overall_decrease : ℕ) : ℕ :=
  let total_employees := num_illiterate + num_literate
  let total_wage_decrease := total_employees * overall_decrease
  (total_wage_decrease + num_illiterate * new_illiterate_wage) / num_illiterate

theorem ngo_wage_problem :
  initial_illiterate_wage 20 10 10 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ngo_wage_problem_l1150_115009


namespace NUMINAMATH_CALUDE_f_neg_one_value_l1150_115038

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x^2 + x

theorem f_neg_one_value (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f_nonneg f) : f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_value_l1150_115038


namespace NUMINAMATH_CALUDE_room_length_calculation_l1150_115062

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 10 → width = 2 → area = length * width → length = 5 := by
sorry

end NUMINAMATH_CALUDE_room_length_calculation_l1150_115062


namespace NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l1150_115034

/-- Calculates the total assembly time for furniture --/
def total_assembly_time (chairs tables bookshelves : ℕ) 
  (chair_time table_time bookshelf_time : ℕ) : ℕ :=
  chairs * chair_time + tables * table_time + bookshelves * bookshelf_time

/-- Proves that the total assembly time for Rachel's furniture is 244 minutes --/
theorem rachel_furniture_assembly_time :
  total_assembly_time 20 8 5 6 8 12 = 244 := by
  sorry

end NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l1150_115034


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1150_115029

theorem imaginary_part_of_z (z : ℂ) (h : z * (Complex.I + 1) + Complex.I = 1 + 3 * Complex.I) : 
  z.im = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1150_115029


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l1150_115050

theorem max_sum_squared_distances (a b c d : Fin 4 → ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  (‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2) ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l1150_115050


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l1150_115046

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l1150_115046


namespace NUMINAMATH_CALUDE_additive_function_is_linear_l1150_115068

theorem additive_function_is_linear (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
sorry

end NUMINAMATH_CALUDE_additive_function_is_linear_l1150_115068


namespace NUMINAMATH_CALUDE_f_g_3_equals_95_l1150_115036

def f (x : ℝ) : ℝ := 4 * x - 5

def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem f_g_3_equals_95 : f (g 3) = 95 := by
  sorry

end NUMINAMATH_CALUDE_f_g_3_equals_95_l1150_115036


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1150_115041

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 2 ∧ (427398 - x) % 13 = 0 ∧ ∀ (y : ℕ), y < x → (427398 - y) % 13 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1150_115041


namespace NUMINAMATH_CALUDE_video_game_pricing_l1150_115071

theorem video_game_pricing (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) :
  total_games = 15 →
  non_working_games = 9 →
  total_earnings = 30 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_pricing_l1150_115071


namespace NUMINAMATH_CALUDE_bike_cost_calculation_l1150_115083

/-- The cost of Trey's new bike -/
def bike_cost : ℕ := 112

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of bracelets Trey needs to sell each day -/
def bracelets_per_day : ℕ := 8

/-- The price of each bracelet in dollars -/
def price_per_bracelet : ℕ := 1

/-- Theorem stating that the bike cost is equal to the product of days, bracelets per day, and price per bracelet -/
theorem bike_cost_calculation : 
  bike_cost = days_in_two_weeks * bracelets_per_day * price_per_bracelet := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_calculation_l1150_115083


namespace NUMINAMATH_CALUDE_pizza_party_group_size_l1150_115097

/-- Given a group of people consisting of children and adults, where the number of children
    is twice the number of adults and there are 80 children, prove that the total number
    of people in the group is 120. -/
theorem pizza_party_group_size :
  ∀ (num_children num_adults : ℕ),
    num_children = 80 →
    num_children = 2 * num_adults →
    num_children + num_adults = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_party_group_size_l1150_115097


namespace NUMINAMATH_CALUDE_function_properties_l1150_115091

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

theorem function_properties (a : ℝ) :
  (∀ x > 0, f a x = x / 4 + a / x - Real.log x - 3 / 2) →
  (f' a 1 = -2) →
  ∃ (x_min : ℝ),
    (a = 5 / 4) ∧
    (x_min = 5) ∧
    (∀ x ∈ Set.Ioo 0 x_min, (f' (5/4)) x < 0) ∧
    (∀ x ∈ Set.Ioi x_min, (f' (5/4)) x > 0) ∧
    (∀ x > 0, f (5/4) x ≥ f (5/4) x_min) ∧
    (f (5/4) x_min = -Real.log 5) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l1150_115091


namespace NUMINAMATH_CALUDE_condition_for_f_sum_positive_l1150_115080

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem condition_for_f_sum_positive :
  ∀ (a b : ℝ), (a + b > 0 ↔ f a + f b > 0) := by sorry

end NUMINAMATH_CALUDE_condition_for_f_sum_positive_l1150_115080


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l1150_115072

/-- Proves that given a bowl with 10 ounces of water, if 4% of the original amount
    evaporates over 50 days, then the amount of water that evaporates each day is 0.008 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percent : ℝ) :
  initial_water = 10 →
  days = 50 →
  evaporation_percent = 4 →
  (evaporation_percent / 100 * initial_water) / days = 0.008 := by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l1150_115072


namespace NUMINAMATH_CALUDE_journey_mpg_is_28_l1150_115092

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer final_odometer : ℕ) 
                (initial_fill first_refill second_refill : ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := initial_fill + first_refill + second_refill
  (total_distance : ℚ) / total_gas

/-- Theorem stating that the average MPG for the given journey is 28 -/
theorem journey_mpg_is_28 :
  let initial_odometer := 56100
  let final_odometer := 57500
  let initial_fill := 10
  let first_refill := 15
  let second_refill := 25
  average_mpg initial_odometer final_odometer initial_fill first_refill second_refill = 28 := by
  sorry

#eval average_mpg 56100 57500 10 15 25

end NUMINAMATH_CALUDE_journey_mpg_is_28_l1150_115092


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l1150_115064

/-- The sum of exterior angles of a polygon --/
def sum_exterior_angles : ℝ := 360

/-- The sum of exterior angles calculated by Robert --/
def roberts_sum : ℝ := 345

/-- Theorem: The measure of the forgotten exterior angle is 15° --/
theorem forgotten_angle_measure :
  sum_exterior_angles - roberts_sum = 15 :=
by sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l1150_115064


namespace NUMINAMATH_CALUDE_fastest_reaction_rate_C_l1150_115021

-- Define the reaction rates
def rate_A : ℝ := 0.15
def rate_B : ℝ := 0.6
def rate_C : ℝ := 0.5
def rate_D : ℝ := 0.4

-- Define the stoichiometric coefficients
def coeff_A : ℝ := 1
def coeff_B : ℝ := 3
def coeff_C : ℝ := 2
def coeff_D : ℝ := 2

-- Theorem: The reaction rate of C is the fastest
theorem fastest_reaction_rate_C :
  rate_C / coeff_C > rate_A / coeff_A ∧
  rate_C / coeff_C > rate_B / coeff_B ∧
  rate_C / coeff_C > rate_D / coeff_D :=
by sorry

end NUMINAMATH_CALUDE_fastest_reaction_rate_C_l1150_115021


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_and_polygon_vertices_l1150_115008

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals_and_polygon_vertices : 
  (num_diagonals 12 = 54) ∧ 
  (∃ n : ℕ, num_diagonals n = 135 ∧ n = 18) := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_and_polygon_vertices_l1150_115008


namespace NUMINAMATH_CALUDE_ladder_slip_l1150_115099

theorem ladder_slip (initial_length initial_distance slip_down slide_out : ℝ) 
  (h_length : initial_length = 30)
  (h_distance : initial_distance = 9)
  (h_slip : slip_down = 5)
  (h_slide : slide_out = 3) :
  let final_distance := initial_distance + slide_out
  final_distance = 12 := by sorry

end NUMINAMATH_CALUDE_ladder_slip_l1150_115099


namespace NUMINAMATH_CALUDE_ball_hit_time_l1150_115065

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_time : ∃ t : ℝ, t > 0 ∧ -6 * t^2 - 10 * t + 56 = 0 ∧ t = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ball_hit_time_l1150_115065


namespace NUMINAMATH_CALUDE_sphere_stack_ratio_l1150_115031

theorem sphere_stack_ratio (n : ℕ) (sphere_volume_ratio : ℚ) 
  (h1 : n = 5)
  (h2 : sphere_volume_ratio = 2/3) : 
  (n : ℚ) * (1 - sphere_volume_ratio) / (n * sphere_volume_ratio) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_stack_ratio_l1150_115031


namespace NUMINAMATH_CALUDE_white_square_area_l1150_115040

-- Define the cube's edge length
def cube_edge : ℝ := 15

-- Define the total area of blue paint
def blue_paint_area : ℝ := 500

-- Define the number of faces of a cube
def cube_faces : ℕ := 6

-- Theorem statement
theorem white_square_area :
  let face_area := cube_edge ^ 2
  let blue_area_per_face := blue_paint_area / cube_faces
  let white_area_per_face := face_area - blue_area_per_face
  white_area_per_face = 425 / 3 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_l1150_115040


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1150_115055

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1150_115055


namespace NUMINAMATH_CALUDE_tuesday_toys_bought_l1150_115082

/-- The number of dog toys Daisy had on Monday -/
def monday_toys : ℕ := 5

/-- The number of dog toys Daisy had left on Tuesday after losing some -/
def tuesday_remaining : ℕ := 3

/-- The number of dog toys Daisy's owner bought on Wednesday -/
def wednesday_new : ℕ := 5

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_if_found : ℕ := 13

/-- The number of dog toys Daisy's owner bought on Tuesday -/
def tuesday_new : ℕ := total_if_found - tuesday_remaining - wednesday_new

theorem tuesday_toys_bought :
  tuesday_new = 5 :=
by sorry

end NUMINAMATH_CALUDE_tuesday_toys_bought_l1150_115082


namespace NUMINAMATH_CALUDE_sphere_to_cube_volume_ratio_l1150_115076

/-- The ratio of the volume of a sphere with diameter 12 inches to the volume of a cube with edge length 6 inches is 4π/3. -/
theorem sphere_to_cube_volume_ratio : 
  let sphere_diameter : ℝ := 12
  let cube_edge : ℝ := 6
  let sphere_volume := (4 / 3) * Real.pi * (sphere_diameter / 2) ^ 3
  let cube_volume := cube_edge ^ 3
  sphere_volume / cube_volume = (4 * Real.pi) / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_to_cube_volume_ratio_l1150_115076


namespace NUMINAMATH_CALUDE_number_difference_l1150_115089

theorem number_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1150_115089


namespace NUMINAMATH_CALUDE_market_prices_l1150_115090

/-- The cost of one pound of rice in dollars -/
def rice_cost : ℝ := 0.33

/-- The number of eggs that cost the same as one pound of rice -/
def eggs_per_rice : ℕ := 1

/-- The number of eggs that cost the same as half a liter of kerosene -/
def eggs_per_half_liter : ℕ := 8

/-- The cost of one liter of kerosene in cents -/
def kerosene_cost : ℕ := 528

theorem market_prices :
  (rice_cost = rice_cost / eggs_per_rice) ∧
  (kerosene_cost = 2 * eggs_per_half_liter * rice_cost * 100) := by
sorry

end NUMINAMATH_CALUDE_market_prices_l1150_115090


namespace NUMINAMATH_CALUDE_b_formula_T_formula_l1150_115003

/-- An arithmetic sequence with first term 1 and common difference 1 -/
def arithmetic_sequence (n : ℕ) : ℕ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sequence b_n defined as 1/S_n -/
def b (n : ℕ) : ℚ := 1 / (S n)

/-- Sum of the first n terms of the sequence b_n -/
def T (n : ℕ) : ℚ := sorry

theorem b_formula (n : ℕ) : b n = 2 / (n * (n + 1)) :=
  sorry

theorem T_formula (n : ℕ) : T n = 2 * n / (n + 1) :=
  sorry

end NUMINAMATH_CALUDE_b_formula_T_formula_l1150_115003


namespace NUMINAMATH_CALUDE_textbook_profit_l1150_115001

/-- The profit of a textbook sale -/
def profit (cost_price selling_price : ℝ) : ℝ :=
  selling_price - cost_price

/-- Theorem: The profit of a textbook is $11 given that its cost price is $44 and its selling price is $55 -/
theorem textbook_profit :
  let cost_price : ℝ := 44
  let selling_price : ℝ := 55
  profit cost_price selling_price = 11 := by
  sorry

end NUMINAMATH_CALUDE_textbook_profit_l1150_115001


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_l1150_115094

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the inscribed circle
structure InscribedCircle (t : EquilateralTriangle) where
  center : ℝ × ℝ  -- Representing the center point P
  radius : ℝ
  radius_pos : radius > 0
  touches_sides : True  -- Assumption that the circle touches all sides

-- Define the perpendicular distances
def perpendicular_distances (t : EquilateralTriangle) (c : InscribedCircle t) : ℝ × ℝ × ℝ :=
  (c.radius, c.radius, c.radius)

-- Theorem statement
theorem sum_of_perpendiculars (t : EquilateralTriangle) (c : InscribedCircle t) :
  let (d1, d2, d3) := perpendicular_distances t c
  d1 + d2 + d3 = (Real.sqrt 3 * t.side_length) / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_l1150_115094


namespace NUMINAMATH_CALUDE_right_triangle_cotangent_l1150_115056

theorem right_triangle_cotangent (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 12) :
  a / b = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cotangent_l1150_115056


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_4_increasing_l1150_115067

open Real

theorem sin_2x_minus_pi_4_increasing (k : ℤ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → 
  x₁ ∈ Set.Ioo (- π/8 + k*π) (3*π/8 + k*π) → 
  x₂ ∈ Set.Ioo (- π/8 + k*π) (3*π/8 + k*π) → 
  sin (2*x₁ - π/4) < sin (2*x₂ - π/4) := by
sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_4_increasing_l1150_115067


namespace NUMINAMATH_CALUDE_f_value_l1150_115025

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem f_value (α : Real) 
  (h1 : Real.cos (α + 3 * Real.pi / 2) = 1 / 5)
  (h2 : 0 < α - Real.pi / 2 ∧ α - Real.pi / 2 < Real.pi / 2) : 
  f α = 2 * Real.sqrt 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_f_value_l1150_115025


namespace NUMINAMATH_CALUDE_R_zero_value_l1150_115039

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^2 - 3*x - 7

-- Define the properties for Q and R
def is_valid_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b, ∀ x, f x = x^2 + a*x + b

-- Define the condition that P + Q, P + R, and Q + R each have a common root
def have_common_roots (P Q R : ℝ → ℝ) : Prop :=
  ∃ p q r, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (P p + Q p = 0 ∧ P p + R p = 0) ∧
    (P q + Q q = 0 ∧ Q q + R q = 0) ∧
    (P r + R r = 0 ∧ Q r + R r = 0)

-- Main theorem
theorem R_zero_value (Q R : ℝ → ℝ) 
  (hQ : is_valid_polynomial Q)
  (hR : is_valid_polynomial R)
  (hQR : have_common_roots P Q R)
  (hQ0 : Q 0 = 2) :
  R 0 = 52 / 19 :=
sorry

end NUMINAMATH_CALUDE_R_zero_value_l1150_115039


namespace NUMINAMATH_CALUDE_b_share_is_600_l1150_115086

/-- Given a partnership where A invests 3 times as much as B, and B invests two-thirds of what C invests,
    this function calculates B's share of the profit when the total profit is 3300 Rs. -/
def calculate_B_share (total_profit : ℚ) : ℚ :=
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 2/3
  let c_ratio : ℚ := 1
  let total_ratio : ℚ := a_ratio + b_ratio + c_ratio
  (b_ratio / total_ratio) * total_profit

/-- Theorem stating that B's share of the profit is 600 Rs -/
theorem b_share_is_600 :
  calculate_B_share 3300 = 600 := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_600_l1150_115086


namespace NUMINAMATH_CALUDE_square_difference_equals_eight_xy_l1150_115077

theorem square_difference_equals_eight_xy (x y A : ℝ) :
  (x + 2*y)^2 = (x - 2*y)^2 + A → A = 8*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_eight_xy_l1150_115077


namespace NUMINAMATH_CALUDE_circle_reconstruction_uniqueness_l1150_115093

-- Define the types for lines and circles
def Line : Type := ℝ × ℝ → Prop
def Circle : Type := ℝ × ℝ → Prop

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the property of two lines being perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the property of a line being tangent to a circle
def tangent_to (l : Line) (c : Circle) : Prop := sorry

-- Define the distance between two parallel lines
def distance_between_parallel_lines (l1 l2 : Line) : ℝ := sorry

-- Main theorem
theorem circle_reconstruction_uniqueness 
  (e1 e2 f1 f2 : Line) 
  (h_parallel_e : parallel e1 e2) 
  (h_parallel_f : parallel f1 f2) 
  (h_not_perp_e1f1 : ¬ perpendicular e1 f1) 
  (h_not_perp_e2f2 : ¬ perpendicular e2 f2) :
  (∃! (k1 k2 : Circle), 
    tangent_to e1 k1 ∧ tangent_to e2 k2 ∧ 
    tangent_to f1 k1 ∧ tangent_to f2 k2 ∧ 
    (∃ (e f : Line), tangent_to e k1 ∧ tangent_to e k2 ∧ 
                     tangent_to f k1 ∧ tangent_to f k2)) ↔ 
  distance_between_parallel_lines e1 e2 ≠ distance_between_parallel_lines f1 f2 :=
sorry

end NUMINAMATH_CALUDE_circle_reconstruction_uniqueness_l1150_115093


namespace NUMINAMATH_CALUDE_ab_value_l1150_115095

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36) :
  a * b = -15 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1150_115095


namespace NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l1150_115005

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rectangle in a plane --/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment --/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The number of intersection points between a circle and a rectangle --/
def intersectionPointsCircleRectangle (c : Circle) (r : Rectangle) : ℕ :=
  (intersectionPointsCircleLine c (r.corners 0) (r.corners 1)) +
  (intersectionPointsCircleLine c (r.corners 1) (r.corners 2)) +
  (intersectionPointsCircleLine c (r.corners 2) (r.corners 3)) +
  (intersectionPointsCircleLine c (r.corners 3) (r.corners 0))

/-- Theorem: The maximum number of intersection points between a circle and a rectangle is 8 --/
theorem max_intersection_points_circle_rectangle :
  ∀ c : Circle, ∀ r : Rectangle, intersectionPointsCircleRectangle c r ≤ 8 ∧
  ∃ c : Circle, ∃ r : Rectangle, intersectionPointsCircleRectangle c r = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l1150_115005


namespace NUMINAMATH_CALUDE_fraction_simplification_l1150_115026

theorem fraction_simplification : (25 : ℚ) / 24 * 18 / 35 * 56 / 45 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1150_115026


namespace NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_equals_22542_l1150_115020

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qin_jiushao (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  let v0 := 1
  let v1 := x * v0 + 2
  let v2 := x * v1 + 0
  let v3 := x * v2 + 4
  let v4 := x * v3 + 5
  let v5 := x * v4 + 6
  x * v5 + 12

/-- The polynomial f(x) = x^6 + 2x^5 + 4x^3 + 5x^2 + 6x + 12 -/
def f (x : ℝ) : ℝ := x^6 + 2*x^5 + 4*x^3 + 5*x^2 + 6*x + 12

/-- Theorem: Qin Jiushao's algorithm correctly evaluates f(3) -/
theorem qin_jiushao_correct : qin_jiushao f 3 = 22542 := by
  sorry

/-- Theorem: f(3) equals 22542 -/
theorem f_3_equals_22542 : f 3 = 22542 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_equals_22542_l1150_115020


namespace NUMINAMATH_CALUDE_possible_average_82_l1150_115070

def test_scores : List Nat := [71, 77, 80, 87]

theorem possible_average_82 (last_score : Nat) 
  (h1 : last_score ≥ 0)
  (h2 : last_score ≤ 100) :
  ∃ (avg : Rat), 
    avg = (test_scores.sum + last_score) / 5 ∧ 
    avg = 82 := by
  sorry

end NUMINAMATH_CALUDE_possible_average_82_l1150_115070


namespace NUMINAMATH_CALUDE_correct_regression_equation_l1150_115054

/-- Represents the selling price of a product in yuan/piece -/
def selling_price : ℝ → Prop :=
  λ x => x > 0

/-- Represents the sales volume of a product in pieces -/
def sales_volume : ℝ → Prop :=
  λ y => y > 0

/-- Represents a negative correlation between sales volume and selling price -/
def negative_correlation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The regression equation for sales volume based on selling price -/
def regression_equation (x : ℝ) : ℝ :=
  -10 * x + 200

theorem correct_regression_equation :
  (∀ x, selling_price x → sales_volume (regression_equation x)) ∧
  negative_correlation regression_equation :=
sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l1150_115054


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1150_115043

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3}
def N : Set ℕ := {1, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1150_115043


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1150_115098

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1150_115098


namespace NUMINAMATH_CALUDE_square_division_square_coverage_l1150_115010

/-- A square is a shape with four equal sides and four right angles -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- A larger square can be divided into four equal smaller squares -/
theorem square_division (large : Square) : 
  ∃ (small : Square), 
    4 * small.side^2 = large.side^2 ∧ 
    small.side > 0 :=
sorry

/-- Four smaller squares can completely cover a larger square without gaps or overlaps -/
theorem square_coverage (large : Square) 
  (h : ∃ (small : Square), 4 * small.side^2 = large.side^2 ∧ small.side > 0) : 
  ∃ (small : Square), 
    4 * small.side^2 = large.side^2 ∧ 
    small.side > 0 ∧
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ large.side ∧ 0 ≤ y ∧ y ≤ large.side → 
      ∃ (i j : Fin 2), 
        i * small.side ≤ x ∧ x < (i + 1) * small.side ∧
        j * small.side ≤ y ∧ y < (j + 1) * small.side) :=
sorry

end NUMINAMATH_CALUDE_square_division_square_coverage_l1150_115010


namespace NUMINAMATH_CALUDE_probability_all_cocaptains_l1150_115053

def team1_size : ℕ := 6
def team2_size : ℕ := 9
def team3_size : ℕ := 10
def cocaptains_per_team : ℕ := 3
def num_teams : ℕ := 3
def selected_members : ℕ := 3

theorem probability_all_cocaptains :
  (1 / num_teams) * (
    1 / (team1_size.choose selected_members) +
    1 / (team2_size.choose selected_members) +
    1 / (team3_size.choose selected_members)
  ) = 53 / 2520 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_cocaptains_l1150_115053


namespace NUMINAMATH_CALUDE_max_fleas_on_chessboard_l1150_115028

/-- Represents a 10x10 chessboard -/
def Chessboard := Fin 10 × Fin 10

/-- Represents the four possible directions a flea can move -/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a flea's position and direction -/
structure Flea where
  position : Chessboard
  direction : Direction

/-- Represents the state of the board at a given time -/
def BoardState := List Flea

/-- Simulates the movement of fleas for one hour (60 minutes) -/
def simulateMovement (initial : BoardState) : List BoardState := sorry

/-- Checks if two fleas occupy the same square -/
def noCollision (state : BoardState) : Prop := sorry

/-- Checks if the simulation is valid (no collisions for 60 minutes) -/
def validSimulation (states : List BoardState) : Prop := sorry

/-- The main theorem: The maximum number of fleas on a 10x10 chessboard is 40 -/
theorem max_fleas_on_chessboard :
  ∀ (initial : BoardState),
    validSimulation (simulateMovement initial) →
    initial.length ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_max_fleas_on_chessboard_l1150_115028


namespace NUMINAMATH_CALUDE_marks_tanks_l1150_115075

/-- The number of tanks Mark has for pregnant fish -/
def num_tanks : ℕ := sorry

/-- The number of pregnant fish in each tank -/
def fish_per_tank : ℕ := 4

/-- The number of young fish each pregnant fish gives birth to -/
def young_per_fish : ℕ := 20

/-- The total number of young fish Mark has -/
def total_young : ℕ := 240

/-- Theorem stating that the number of tanks Mark has is 3 -/
theorem marks_tanks : num_tanks = 3 := by sorry

end NUMINAMATH_CALUDE_marks_tanks_l1150_115075


namespace NUMINAMATH_CALUDE_product_of_numbers_l1150_115018

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1150_115018


namespace NUMINAMATH_CALUDE_remainder_3005_div_98_l1150_115023

theorem remainder_3005_div_98 : 3005 % 98 = 65 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3005_div_98_l1150_115023


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1150_115002

theorem quadratic_inequality (x : ℝ) : x^2 + x - 12 > 0 ↔ x > 3 ∨ x < -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1150_115002


namespace NUMINAMATH_CALUDE_stratified_sampling_example_l1150_115061

/-- Given a total number of positions, male doctors, and female doctors,
    calculate the number of male doctors to be selected through stratified sampling. -/
def stratified_sampling (total_positions : ℕ) (male_doctors : ℕ) (female_doctors : ℕ) : ℕ :=
  (total_positions * male_doctors) / (male_doctors + female_doctors)

theorem stratified_sampling_example :
  stratified_sampling 15 120 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_example_l1150_115061


namespace NUMINAMATH_CALUDE_meeting_participants_count_l1150_115085

theorem meeting_participants_count : 
  ∀ (F M : ℕ),
  F = 330 →
  (F / 2 : ℚ) = 165 →
  (F + M) / 3 = F / 2 + M / 4 →
  F + M = 990 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_count_l1150_115085


namespace NUMINAMATH_CALUDE_problem_figure_area_l1150_115066

/-- A figure composed of square segments -/
structure SegmentedFigure where
  /-- The number of segments along one side of the square -/
  segments_per_side : ℕ
  /-- The length of each segment in cm -/
  segment_length : ℝ

/-- The area of a SegmentedFigure in cm² -/
def area (figure : SegmentedFigure) : ℝ :=
  (figure.segments_per_side * figure.segment_length) ^ 2

/-- The specific figure from the problem -/
def problem_figure : SegmentedFigure :=
  { segments_per_side := 3
  , segment_length := 3 }

theorem problem_figure_area :
  area problem_figure = 81 := by sorry

end NUMINAMATH_CALUDE_problem_figure_area_l1150_115066
