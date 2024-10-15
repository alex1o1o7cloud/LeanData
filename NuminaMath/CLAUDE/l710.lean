import Mathlib

namespace NUMINAMATH_CALUDE_car_material_cost_is_100_l710_71017

/-- Represents the factory's production and sales data -/
structure FactoryData where
  car_production : Nat
  car_price : Nat
  motorcycle_production : Nat
  motorcycle_price : Nat
  motorcycle_material_cost : Nat
  profit_difference : Nat

/-- Calculates the cost of materials for car production -/
def calculate_car_material_cost (data : FactoryData) : Nat :=
  data.motorcycle_production * data.motorcycle_price - 
  data.motorcycle_material_cost - 
  (data.car_production * data.car_price - 
  data.profit_difference)

/-- Theorem stating that the cost of materials for car production is $100 -/
theorem car_material_cost_is_100 (data : FactoryData) 
  (h1 : data.car_production = 4)
  (h2 : data.car_price = 50)
  (h3 : data.motorcycle_production = 8)
  (h4 : data.motorcycle_price = 50)
  (h5 : data.motorcycle_material_cost = 250)
  (h6 : data.profit_difference = 50) :
  calculate_car_material_cost data = 100 := by
  sorry

end NUMINAMATH_CALUDE_car_material_cost_is_100_l710_71017


namespace NUMINAMATH_CALUDE_ascending_order_of_a_l710_71048

theorem ascending_order_of_a (a : ℝ) (h : a^2 - a < 0) :
  -a < -a^2 ∧ -a^2 < a^2 ∧ a^2 < a :=
by sorry

end NUMINAMATH_CALUDE_ascending_order_of_a_l710_71048


namespace NUMINAMATH_CALUDE_number_thought_of_l710_71070

theorem number_thought_of (x : ℝ) : (x / 5 + 8 = 61) → x = 265 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l710_71070


namespace NUMINAMATH_CALUDE_circle_center_theorem_l710_71086

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def circle_tangent_to_parabola (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  y = parabola x ∧
  (x - cx)^2 + (y - cy)^2 = c.radius^2 ∧
  (2 * x * (x - cx) + 2 * (y - cy))^2 = 4 * ((x - cx)^2 + (y - cy)^2)

-- Theorem statement
theorem circle_center_theorem :
  ∃ (c : Circle),
    circle_passes_through c (0, 1) ∧
    circle_tangent_to_parabola c (2, 4) ∧
    c.center = (-16/5, 53/10) :=
sorry

end NUMINAMATH_CALUDE_circle_center_theorem_l710_71086


namespace NUMINAMATH_CALUDE_average_pages_per_day_l710_71060

theorem average_pages_per_day 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (remaining_days : ℕ) 
  (h1 : total_pages = 212) 
  (h2 : pages_read = 97) 
  (h3 : remaining_days = 5) :
  (total_pages - pages_read) / remaining_days = 23 := by
sorry

end NUMINAMATH_CALUDE_average_pages_per_day_l710_71060


namespace NUMINAMATH_CALUDE_curve_C_equation_sum_of_slopes_constant_l710_71015

noncomputable section

def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def Curve_C := {N : ℝ × ℝ | N.1^2 / 6 + N.2^2 / 3 = 1}

def Point_on_circle (P : ℝ × ℝ) := P.1^2 + P.2^2 = 6

def Point_N (P N : ℝ × ℝ) := 
  ∃ (M : ℝ × ℝ), M.2 = 0 ∧ (P.1 - M.1)^2 + (P.2 - M.2)^2 = 2 * ((N.1 - M.1)^2 + (N.2 - M.2)^2)

def Line_through_B (k : ℝ) := {P : ℝ × ℝ | P.2 = k * (P.1 - 3)}

def Slope (A B : ℝ × ℝ) := (B.2 - A.2) / (B.1 - A.1)

theorem curve_C_equation :
  ∀ N : ℝ × ℝ, (∃ P : ℝ × ℝ, Point_on_circle P ∧ Point_N P N) → N ∈ Curve_C := by sorry

theorem sum_of_slopes_constant :
  ∀ k : ℝ, ∀ D E : ℝ × ℝ,
    D ∈ Curve_C ∧ E ∈ Curve_C ∧ D ∈ Line_through_B k ∧ E ∈ Line_through_B k ∧ D ≠ E →
    Slope (2, 1) D + Slope (2, 1) E = -2 := by sorry

end NUMINAMATH_CALUDE_curve_C_equation_sum_of_slopes_constant_l710_71015


namespace NUMINAMATH_CALUDE_apple_difference_l710_71028

theorem apple_difference (ben_apples phillip_apples tom_apples : ℕ) : 
  ben_apples > phillip_apples →
  tom_apples = (3 * ben_apples) / 8 →
  phillip_apples = 40 →
  tom_apples = 18 →
  ben_apples - phillip_apples = 8 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l710_71028


namespace NUMINAMATH_CALUDE_absolute_difference_of_integers_l710_71053

theorem absolute_difference_of_integers (x y : ℤ) 
  (h1 : x ≠ y)
  (h2 : (x + y) / 2 = 15)
  (h3 : Real.sqrt (x * y) + 6 = 15) : 
  |x - y| = 24 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_integers_l710_71053


namespace NUMINAMATH_CALUDE_f_composition_value_l710_71089

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else Real.cos x

theorem f_composition_value : f (f (-Real.pi/3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l710_71089


namespace NUMINAMATH_CALUDE_circular_garden_radius_l710_71007

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1/3) * π * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l710_71007


namespace NUMINAMATH_CALUDE_root_condition_for_k_l710_71075

/-- The function f(x) = kx - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 3

/-- A function has a root in an interval if its values at the endpoints have different signs -/
def has_root_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a * f b ≤ 0

theorem root_condition_for_k (k : ℝ) :
  (k ≥ 3 → has_root_in_interval (f k) (-1) 1) ∧
  (∃ k', k' < 3 ∧ has_root_in_interval (f k') (-1) 1) :=
sorry

end NUMINAMATH_CALUDE_root_condition_for_k_l710_71075


namespace NUMINAMATH_CALUDE_gcd_4536_8721_l710_71034

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4536_8721_l710_71034


namespace NUMINAMATH_CALUDE_intersection_equal_angles_not_always_perpendicular_l710_71095

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Plane → Plane → Line)
variable (angle_with : Line → Plane → ℝ)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem intersection_equal_angles_not_always_perpendicular
  (m n : Line) (α β : Plane) :
  ¬(∀ (m n : Line) (α β : Plane),
    (intersect α β = m) →
    (angle_with n α = angle_with n β) →
    (perpendicular m n)) :=
sorry

end NUMINAMATH_CALUDE_intersection_equal_angles_not_always_perpendicular_l710_71095


namespace NUMINAMATH_CALUDE_AB_squared_is_8_l710_71044

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 4 * x + 2

/-- Point A on the parabola -/
def A : ℝ × ℝ := sorry

/-- Point B on the parabola -/
def B : ℝ × ℝ := sorry

/-- The origin is the midpoint of AB -/
axiom origin_is_midpoint : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- A and B are on the parabola -/
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

/-- The square of the length of AB -/
def AB_squared : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- Theorem: The square of the length of AB is 8 -/
theorem AB_squared_is_8 : AB_squared = 8 := sorry

end NUMINAMATH_CALUDE_AB_squared_is_8_l710_71044


namespace NUMINAMATH_CALUDE_conference_handshakes_l710_71040

/-- Represents a group of people at a conference -/
structure ConferenceGroup where
  total : ℕ
  group_a : ℕ
  group_b : ℕ
  h_total : total = group_a + group_b

/-- Calculates the number of handshakes in a conference group -/
def count_handshakes (g : ConferenceGroup) : ℕ :=
  (g.group_b * (g.group_b - 1)) / 2

/-- Theorem stating the number of handshakes in the specific conference scenario -/
theorem conference_handshakes :
  ∀ g : ConferenceGroup,
    g.total = 30 →
    g.group_a = 25 →
    g.group_b = 5 →
    count_handshakes g = 10 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_l710_71040


namespace NUMINAMATH_CALUDE_difference_of_101st_terms_l710_71005

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem difference_of_101st_terms : 
  let X := arithmetic_sequence 40 12
  let Y := arithmetic_sequence 40 (-8)
  |X 101 - Y 101| = 2000 := by
sorry

end NUMINAMATH_CALUDE_difference_of_101st_terms_l710_71005


namespace NUMINAMATH_CALUDE_quadratic_equation_q_value_l710_71078

theorem quadratic_equation_q_value : ∀ (p q : ℝ),
  (∃ x : ℝ, 3 * x^2 + p * x + q = 0 ∧ x = -3) →
  (∃ x₁ x₂ : ℝ, 3 * x₁^2 + p * x₁ + q = 0 ∧ 3 * x₂^2 + p * x₂ + q = 0 ∧ x₁ + x₂ = -2) →
  q = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_q_value_l710_71078


namespace NUMINAMATH_CALUDE_paper_tape_overlap_l710_71090

/-- Given 12 sheets of paper tape, each 18 cm long, glued to form a round loop
    with a perimeter of 210 cm and overlapped by the same length,
    the length of each overlapped part is 5 mm. -/
theorem paper_tape_overlap (num_sheets : ℕ) (sheet_length : ℝ) (perimeter : ℝ) :
  num_sheets = 12 →
  sheet_length = 18 →
  perimeter = 210 →
  (num_sheets * sheet_length - perimeter) / num_sheets * 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_paper_tape_overlap_l710_71090


namespace NUMINAMATH_CALUDE_initial_amount_simple_interest_l710_71057

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Total amount after applying simple interest --/
def total_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem: Initial amount in a simple interest scenario --/
theorem initial_amount_simple_interest :
  ∃ (principal : ℝ),
    total_amount principal 0.10 5 = 1125 ∧
    principal = 750 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_simple_interest_l710_71057


namespace NUMINAMATH_CALUDE_age_squares_sum_l710_71000

theorem age_squares_sum (T J A : ℕ) 
  (sum_TJ : T + J = 23)
  (sum_JA : J + A = 24)
  (sum_TA : T + A = 25) :
  T^2 + J^2 + A^2 = 434 := by
sorry

end NUMINAMATH_CALUDE_age_squares_sum_l710_71000


namespace NUMINAMATH_CALUDE_solve_abc_values_l710_71032

theorem solve_abc_values (A B : Set ℝ) (a b c : ℝ) :
  A = {x : ℝ | x^2 - a*x - 2 = 0} →
  B = {x : ℝ | x^3 + b*x + c = 0} →
  -2 ∈ A ∩ B →
  A ∩ B = A →
  a = -1 ∧ b = -3 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_abc_values_l710_71032


namespace NUMINAMATH_CALUDE_sequence_growth_l710_71099

theorem sequence_growth (a : ℕ → ℤ) (h1 : a 1 > a 0) (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) :
  a 100 > 299 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l710_71099


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_area_l710_71097

/-- The area of a right isosceles triangle with perimeter 3p -/
theorem right_isosceles_triangle_area (p : ℝ) :
  let a : ℝ := p * (3 / (2 + Real.sqrt 2))
  let b : ℝ := a
  let c : ℝ := Real.sqrt 2 * a
  let perimeter : ℝ := a + b + c
  let area : ℝ := (1 / 2) * a * b
  (perimeter = 3 * p) → (area = (9 * p^2 * (3 - 2 * Real.sqrt 2)) / 4) :=
by sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_area_l710_71097


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_1800_l710_71018

def sum_of_distinct_prime_divisors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_divisors_1800 :
  sum_of_distinct_prime_divisors 1800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_divisors_1800_l710_71018


namespace NUMINAMATH_CALUDE_f_property_l710_71026

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1)^2 - a * Real.log x

theorem f_property (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    ∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → y₁ > 0 → y₂ > 0 →
      (f a (y₁ + 1) - f a (y₂ + 1)) / (y₁ - y₂) > 1) →
  a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_f_property_l710_71026


namespace NUMINAMATH_CALUDE_minimum_transportation_cost_l710_71056

/-- Represents the capacity of a truck -/
structure TruckCapacity where
  tents : ℕ
  food : ℕ

/-- Represents a truck arrangement -/
structure TruckArrangement where
  typeA : ℕ
  typeB : ℕ

/-- Calculate the total items an arrangement can carry -/
def totalCapacity (c : TruckCapacity × TruckCapacity) (a : TruckArrangement) : ℕ × ℕ :=
  (a.typeA * c.1.tents + a.typeB * c.2.tents, a.typeA * c.1.food + a.typeB * c.2.food)

/-- Calculate the cost of an arrangement -/
def arrangementCost (costs : ℕ × ℕ) (a : TruckArrangement) : ℕ :=
  a.typeA * costs.1 + a.typeB * costs.2

theorem minimum_transportation_cost :
  let totalItems : ℕ := 320
  let tentsDiff : ℕ := 80
  let totalTrucks : ℕ := 8
  let typeACapacity : TruckCapacity := ⟨40, 10⟩
  let typeBCapacity : TruckCapacity := ⟨20, 20⟩
  let costs : ℕ × ℕ := (4000, 3600)
  let tents : ℕ := (totalItems + tentsDiff) / 2
  let food : ℕ := (totalItems - tentsDiff) / 2
  ∃ (a : TruckArrangement),
    a.typeA + a.typeB = totalTrucks ∧
    totalCapacity (typeACapacity, typeBCapacity) a = (tents, food) ∧
    ∀ (b : TruckArrangement),
      b.typeA + b.typeB = totalTrucks →
      totalCapacity (typeACapacity, typeBCapacity) b = (tents, food) →
      arrangementCost costs a ≤ arrangementCost costs b ∧
      arrangementCost costs a = 29600 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_transportation_cost_l710_71056


namespace NUMINAMATH_CALUDE_max_group_size_problem_l710_71074

/-- The maximum number of people in a group for two classes with given total students and leftovers -/
def max_group_size (class1_total : ℕ) (class2_total : ℕ) (class1_leftover : ℕ) (class2_leftover : ℕ) : ℕ :=
  Nat.gcd (class1_total - class1_leftover) (class2_total - class2_leftover)

/-- Theorem stating that the maximum group size for the given problem is 16 -/
theorem max_group_size_problem : max_group_size 69 86 5 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_group_size_problem_l710_71074


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_four_times_exterior_l710_71062

theorem polygon_sides_when_interior_four_times_exterior : 
  ∀ n : ℕ, n > 2 →
  (n - 2) * 180 = 4 * 360 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_four_times_exterior_l710_71062


namespace NUMINAMATH_CALUDE_perfect_square_from_48_numbers_l710_71068

theorem perfect_square_from_48_numbers (S : Finset ℕ) 
  (h1 : S.card = 48)
  (h2 : (S.prod id).factors.toFinset.card = 10) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  ∃ (m : ℕ), a * b * c * d = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_from_48_numbers_l710_71068


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l710_71006

/-- Given a quadratic function y = x^2 + 2mx + 2 with symmetry axis x = 2, prove that m = -2 -/
theorem quadratic_symmetry_axis (m : ℝ) : 
  (∀ x, x^2 + 2*m*x + 2 = (x-2)^2 + (2^2 + 2*m*2 + 2)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l710_71006


namespace NUMINAMATH_CALUDE_aftershave_dilution_l710_71066

/-- Proves that adding 10 ounces of water to a 12-ounce bottle of 60% alcohol solution, 
    then removing 4 ounces of the mixture, results in a 40% alcohol solution. -/
theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (removed_amount : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧ 
  water_added = 10 ∧ 
  removed_amount = 4 ∧ 
  final_concentration = 0.4 →
  let initial_alcohol := initial_volume * initial_concentration
  let total_volume := initial_volume + water_added
  let final_volume := total_volume - removed_amount
  initial_alcohol / final_volume = final_concentration :=
by
  sorry

#check aftershave_dilution

end NUMINAMATH_CALUDE_aftershave_dilution_l710_71066


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l710_71042

theorem units_digit_of_expression : ∃ n : ℕ, (13 + Real.sqrt 196)^21 + (13 - Real.sqrt 196)^21 = 10 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l710_71042


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l710_71093

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℝ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 :=
by sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l710_71093


namespace NUMINAMATH_CALUDE_maggots_eaten_second_feeding_correct_l710_71096

/-- Given the total number of maggots served, the number of maggots laid out and eaten in the first feeding,
    and the number laid out in the second feeding, calculate the number of maggots eaten in the second feeding. -/
def maggots_eaten_second_feeding (total_served : ℕ) (first_feeding_laid_out : ℕ) (first_feeding_eaten : ℕ) (second_feeding_laid_out : ℕ) : ℕ :=
  total_served - first_feeding_eaten - second_feeding_laid_out

/-- Theorem stating that the number of maggots eaten in the second feeding is correct -/
theorem maggots_eaten_second_feeding_correct
  (total_served : ℕ)
  (first_feeding_laid_out : ℕ)
  (first_feeding_eaten : ℕ)
  (second_feeding_laid_out : ℕ)
  (h1 : total_served = 20)
  (h2 : first_feeding_laid_out = 10)
  (h3 : first_feeding_eaten = 1)
  (h4 : second_feeding_laid_out = 10) :
  maggots_eaten_second_feeding total_served first_feeding_laid_out first_feeding_eaten second_feeding_laid_out = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_maggots_eaten_second_feeding_correct_l710_71096


namespace NUMINAMATH_CALUDE_number_equation_solution_l710_71055

theorem number_equation_solution : 
  ∃ n : ℚ, (3/4 : ℚ) * n - (8/5 : ℚ) * n + 63 = 12 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l710_71055


namespace NUMINAMATH_CALUDE_mikes_work_days_l710_71083

/-- Given that Mike worked 3 hours each day for a total of 15 hours,
    prove that he worked for 5 days. -/
theorem mikes_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 → total_hours = 15 → days * hours_per_day = total_hours → days = 5 := by
  sorry

end NUMINAMATH_CALUDE_mikes_work_days_l710_71083


namespace NUMINAMATH_CALUDE_charles_reading_time_l710_71022

-- Define the parameters of the problem
def total_pages : ℕ := 96
def pages_per_day : ℕ := 8

-- Define the function to calculate the number of days
def days_to_finish (total : ℕ) (per_day : ℕ) : ℕ := total / per_day

-- Theorem statement
theorem charles_reading_time : days_to_finish total_pages pages_per_day = 12 := by
  sorry

end NUMINAMATH_CALUDE_charles_reading_time_l710_71022


namespace NUMINAMATH_CALUDE_f_range_at_1_3_l710_71076

def f (a b x y : ℝ) : ℝ := a * (x^3 + 3*x) + b * (y^2 + 2*y + 1)

theorem f_range_at_1_3 (a b : ℝ) (h1 : 1 ≤ f a b 1 2) (h2 : f a b 1 2 ≤ 2) 
  (h3 : 2 ≤ f a b 3 4) (h4 : f a b 3 4 ≤ 5) : 
  (3/2 : ℝ) ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_f_range_at_1_3_l710_71076


namespace NUMINAMATH_CALUDE_units_digit_of_S_is_3_l710_71020

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def S : ℕ := (List.range 12).map (λ i => factorial (i + 1)) |>.sum

theorem units_digit_of_S_is_3 : S % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_S_is_3_l710_71020


namespace NUMINAMATH_CALUDE_discount_percentage_l710_71045

theorem discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  (M - SP) / M * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l710_71045


namespace NUMINAMATH_CALUDE_power_three_mod_five_l710_71092

theorem power_three_mod_five : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_five_l710_71092


namespace NUMINAMATH_CALUDE_proposition_and_variations_l710_71047

theorem proposition_and_variations (x : ℝ) :
  ((x = 3 ∨ x = 7) → (x - 3) * (x - 7) = 0) ∧
  ((x - 3) * (x - 7) = 0 → (x = 3 ∨ x = 7)) ∧
  ((x ≠ 3 ∧ x ≠ 7) → (x - 3) * (x - 7) ≠ 0) ∧
  ((x - 3) * (x - 7) ≠ 0 → (x ≠ 3 ∧ x ≠ 7)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variations_l710_71047


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l710_71013

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 2*a + 1 = 0) : 
  4*a - 2*a^2 + 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l710_71013


namespace NUMINAMATH_CALUDE_sequence_convergence_bound_l710_71067

def x : ℕ → ℚ
  | 0 => 6
  | n + 1 => (x n ^ 2 + 6 * x n + 7) / (x n + 7)

theorem sequence_convergence_bound :
  ∃ m : ℕ, m ∈ Set.Icc 151 300 ∧
    x m ≤ 4 + 1 / (2^25) ∧
    ∀ k : ℕ, k < m → x k > 4 + 1 / (2^25) :=
by sorry

end NUMINAMATH_CALUDE_sequence_convergence_bound_l710_71067


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_l710_71023

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] where
  F₁ : α
  F₂ : α

/-- A point P in the plane satisfying the locus condition -/
structure LocusPoint (α : Type*) [NormedAddCommGroup α] (FP : FixedPoints α) where
  P : α
  k : ℝ
  h_positive : k > 0
  h_less : k < ‖FP.F₁ - FP.F₂‖
  h_condition : ‖P - FP.F₁‖ - ‖P - FP.F₂‖ = k

/-- Definition of a hyperbola -/
def IsHyperbola (α : Type*) [NormedAddCommGroup α] (S : Set α) (FP : FixedPoints α) :=
  ∃ k : ℝ, k > 0 ∧ k < ‖FP.F₁ - FP.F₂‖ ∧
    S = {P | ‖P - FP.F₁‖ - ‖P - FP.F₂‖ = k ∨ ‖P - FP.F₂‖ - ‖P - FP.F₁‖ = k}

/-- The main theorem: The locus of points satisfying the given condition forms a hyperbola -/
theorem locus_is_hyperbola {α : Type*} [NormedAddCommGroup α] (FP : FixedPoints α) :
  IsHyperbola α {P | ∃ LP : LocusPoint α FP, LP.P = P} FP :=
sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_l710_71023


namespace NUMINAMATH_CALUDE_product_of_terms_l710_71084

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1) ^ 2 - 8 * (a 1) + 1 = 0 →
  (a 13) ^ 2 - 8 * (a 13) + 1 = 0 →
  a 5 * a 7 * a 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_terms_l710_71084


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l710_71019

-- Define the sets A and B
def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l710_71019


namespace NUMINAMATH_CALUDE_octal_minus_quinary_in_decimal_l710_71014

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def octal_54321 : List Nat := [1, 2, 3, 4, 5]
def quinary_4321 : List Nat := [1, 2, 3, 4]

theorem octal_minus_quinary_in_decimal : 
  base_to_decimal octal_54321 8 - base_to_decimal quinary_4321 5 = 22151 := by
  sorry

end NUMINAMATH_CALUDE_octal_minus_quinary_in_decimal_l710_71014


namespace NUMINAMATH_CALUDE_second_day_to_full_distance_ratio_l710_71049

/-- Represents a three-day hike with given distances --/
structure ThreeDayHike where
  fullDistance : ℕ
  firstDayDistance : ℕ
  thirdDayDistance : ℕ

/-- Calculates the second day distance --/
def secondDayDistance (hike : ThreeDayHike) : ℕ :=
  hike.fullDistance - (hike.firstDayDistance + hike.thirdDayDistance)

/-- Theorem: The ratio of the second day distance to the full hike distance is 1:2 --/
theorem second_day_to_full_distance_ratio 
  (hike : ThreeDayHike) 
  (h1 : hike.fullDistance = 50) 
  (h2 : hike.firstDayDistance = 10) 
  (h3 : hike.thirdDayDistance = 15) : 
  (secondDayDistance hike) * 2 = hike.fullDistance := by
  sorry

end NUMINAMATH_CALUDE_second_day_to_full_distance_ratio_l710_71049


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l710_71087

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l710_71087


namespace NUMINAMATH_CALUDE_total_money_from_tshirts_l710_71046

/-- The amount of money made from each t-shirt -/
def money_per_tshirt : ℕ := 215

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 20

/-- The total money made from selling t-shirts -/
def total_money : ℕ := money_per_tshirt * tshirts_sold

theorem total_money_from_tshirts : total_money = 4300 := by
  sorry

end NUMINAMATH_CALUDE_total_money_from_tshirts_l710_71046


namespace NUMINAMATH_CALUDE_combination_sum_equals_c_11_3_l710_71036

theorem combination_sum_equals_c_11_3 :
  (Finset.range 9).sum (fun k => Nat.choose (k + 2) 2) = Nat.choose 11 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_c_11_3_l710_71036


namespace NUMINAMATH_CALUDE_chocolate_count_l710_71039

theorem chocolate_count : ∀ x : ℚ,
  let day1_remaining := (3 / 5 : ℚ) * x - 3
  let day2_remaining := (3 / 4 : ℚ) * day1_remaining - 5
  day2_remaining = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l710_71039


namespace NUMINAMATH_CALUDE_average_correction_l710_71008

def correct_average (num_students : ℕ) (initial_average : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) : ℚ :=
  (num_students * initial_average - (wrong_mark - correct_mark)) / num_students

theorem average_correction (num_students : ℕ) (initial_average : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ)
  (h1 : num_students = 30)
  (h2 : initial_average = 60)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 15) :
  correct_average num_students initial_average wrong_mark correct_mark = 57.5 := by
sorry

end NUMINAMATH_CALUDE_average_correction_l710_71008


namespace NUMINAMATH_CALUDE_stream_speed_l710_71001

/-- Proves that given a man rowing 84 km downstream and 60 km upstream, each taking 4 hours, the speed of the stream is 3 km/h. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 84)
  (h2 : upstream_distance = 60)
  (h3 : time = 4) :
  let boat_speed := (downstream_distance + upstream_distance) / (2 * time)
  let stream_speed := (downstream_distance - upstream_distance) / (4 * time)
  stream_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l710_71001


namespace NUMINAMATH_CALUDE_nail_trimming_sounds_l710_71098

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The number of customers -/
def num_customers : ℕ := 6

/-- The total number of nail trimming sounds -/
def total_sounds : ℕ := nails_per_customer * num_customers

theorem nail_trimming_sounds : total_sounds = 120 := by
  sorry

end NUMINAMATH_CALUDE_nail_trimming_sounds_l710_71098


namespace NUMINAMATH_CALUDE_sin_cos_sum_l710_71077

theorem sin_cos_sum (θ : Real) (h : Real.sin θ * Real.cos θ = 1/8) :
  Real.sin θ + Real.cos θ = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_l710_71077


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l710_71082

theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) :
  ∃ w : ℝ, w > 0 ∧ w^2 = 39 ∧ d^2 = l^2 + w^2 + h^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l710_71082


namespace NUMINAMATH_CALUDE_max_reflections_is_largest_l710_71054

/-- Represents the angle between lines AD and CD in degrees -/
def angle_CDA : ℝ := 5

/-- Represents the maximum allowed path length -/
def max_path_length : ℝ := 100

/-- Calculates the total angle after n reflections -/
def total_angle (n : ℕ) : ℝ := n * angle_CDA

/-- Represents the condition that the total angle must not exceed 90 degrees -/
def angle_condition (n : ℕ) : Prop := total_angle n ≤ 90

/-- Represents an approximation of the path length after n reflections -/
def approx_path_length (n : ℕ) : ℝ := 2 * n * 5

/-- Represents the condition that the path length must not exceed the maximum allowed length -/
def path_length_condition (n : ℕ) : Prop := approx_path_length n ≤ max_path_length

/-- Represents the maximum number of reflections that satisfies all conditions -/
def max_reflections : ℕ := 10

/-- Theorem stating that max_reflections is the largest value that satisfies all conditions -/
theorem max_reflections_is_largest :
  (angle_condition max_reflections) ∧
  (path_length_condition max_reflections) ∧
  (∀ m : ℕ, m > max_reflections → ¬(angle_condition m ∧ path_length_condition m)) :=
sorry

end NUMINAMATH_CALUDE_max_reflections_is_largest_l710_71054


namespace NUMINAMATH_CALUDE_equation_solutions_l710_71069

-- Define the equation
def equation (r p : ℤ) : Prop := r^2 - r*(p + 6) + p^2 + 5*p + 6 = 0

-- Define the set of solution pairs
def solution_set : Set (ℤ × ℤ) := {(3,1), (4,1), (0,-2), (4,-2), (0,-3), (3,-3)}

-- Theorem statement
theorem equation_solutions :
  ∀ (r p : ℤ), equation r p ↔ (r, p) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l710_71069


namespace NUMINAMATH_CALUDE_cook_selection_theorem_l710_71085

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 cooks from a group of 10 people,
    where one specific person must always be selected. -/
def cookSelectionWays : ℕ := choose 9 1

theorem cook_selection_theorem :
  cookSelectionWays = 9 := by sorry

end NUMINAMATH_CALUDE_cook_selection_theorem_l710_71085


namespace NUMINAMATH_CALUDE_pairing_probability_l710_71012

theorem pairing_probability (n : ℕ) (h : n = 28) :
  (1 : ℚ) / (n - 1) = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_pairing_probability_l710_71012


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l710_71016

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) : 
  (|x₁ - 20| + |x₁ + 20| = 2020) ∧ 
  (|x₂ - 20| + |x₂ + 20| = 2020) ∧ 
  (∀ x : ℝ, |x - 20| + |x + 20| = 2020 → x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l710_71016


namespace NUMINAMATH_CALUDE_sum_equals_difference_l710_71010

theorem sum_equals_difference (N : ℤ) : 
  995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_difference_l710_71010


namespace NUMINAMATH_CALUDE_coffee_cost_is_18_l710_71061

/-- Represents the coffee consumption and cost parameters --/
structure CoffeeParams where
  cups_per_day : ℕ
  oz_per_cup : ℚ
  bag_cost : ℚ
  oz_per_bag : ℚ
  milk_gal_per_week : ℚ
  milk_cost_per_gal : ℚ

/-- Calculates the weekly cost of coffee given the parameters --/
def weekly_coffee_cost (params : CoffeeParams) : ℚ :=
  let beans_oz_per_week := params.cups_per_day * params.oz_per_cup * 7
  let bags_per_week := beans_oz_per_week / params.oz_per_bag
  let bean_cost := bags_per_week * params.bag_cost
  let milk_cost := params.milk_gal_per_week * params.milk_cost_per_gal
  bean_cost + milk_cost

/-- Theorem stating that the weekly coffee cost is $18 --/
theorem coffee_cost_is_18 :
  ∃ (params : CoffeeParams),
    params.cups_per_day = 2 ∧
    params.oz_per_cup = 3/2 ∧
    params.bag_cost = 8 ∧
    params.oz_per_bag = 21/2 ∧
    params.milk_gal_per_week = 1/2 ∧
    params.milk_cost_per_gal = 4 ∧
    weekly_coffee_cost params = 18 :=
  sorry

end NUMINAMATH_CALUDE_coffee_cost_is_18_l710_71061


namespace NUMINAMATH_CALUDE_train_carriages_l710_71072

theorem train_carriages (num_trains : ℕ) (rows_per_carriage : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  num_trains = 4 → rows_per_carriage = 3 → wheels_per_row = 5 → total_wheels = 240 →
  (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_train_carriages_l710_71072


namespace NUMINAMATH_CALUDE_balloon_altitude_l710_71031

/-- Calculates the altitude of a balloon given temperature conditions -/
theorem balloon_altitude 
  (temp_decrease_rate : ℝ) -- Temperature decrease rate per 1000 meters
  (ground_temp : ℝ)        -- Ground temperature in °C
  (balloon_temp : ℝ)       -- Balloon temperature in °C
  (h : temp_decrease_rate = 6)
  (i : ground_temp = 5)
  (j : balloon_temp = -2) :
  (ground_temp - balloon_temp) / temp_decrease_rate = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_balloon_altitude_l710_71031


namespace NUMINAMATH_CALUDE_complex_equation_solution_l710_71058

theorem complex_equation_solution (z : ℂ) :
  z / (1 - 2 * Complex.I) = Complex.I → z = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l710_71058


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_intersection_and_double_angle_line_l710_71063

-- Define the two original lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 9 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (3, 3)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + 3*y - 1 = 0

theorem intersection_and_parallel_line :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → (2*x + 3*y - 15 = 0) := by sorry

theorem intersection_and_double_angle_line :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → (4*x - 3*y - 3 = 0) := by sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_intersection_and_double_angle_line_l710_71063


namespace NUMINAMATH_CALUDE_hockey_league_games_l710_71009

/-- The number of teams in the hockey league -/
def num_teams : ℕ := 19

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_pair

theorem hockey_league_games :
  total_games = 1710 :=
sorry

end NUMINAMATH_CALUDE_hockey_league_games_l710_71009


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l710_71011

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l710_71011


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l710_71052

/-- Given a class of students where half the number of girls equals one-fifth of the total number of students, prove that the ratio of boys to girls is 3:2. -/
theorem boys_to_girls_ratio (S : ℕ) (G : ℕ) (h : 2 * G = S) :
  (S - G) / G = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l710_71052


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l710_71081

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 14 →
  ratio = 2 / 5 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l710_71081


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l710_71027

/-- The greatest power of 4 that divides an even positive integer -/
def h (x : ℕ+) : ℕ :=
  sorry

/-- Sum of h(4k) from k = 1 to 2^(n-1) -/
def T (n : ℕ+) : ℕ :=
  sorry

/-- Predicate for perfect squares -/
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem greatest_n_for_perfect_square_T :
  ∀ n : ℕ+, n < 500 → is_perfect_square (T n) → n ≤ 143 ∧
  is_perfect_square (T 143) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l710_71027


namespace NUMINAMATH_CALUDE_employee_count_l710_71094

theorem employee_count : 
  ∀ (E : ℕ) (M : ℝ),
    M = 0.99 * (E : ℝ) →
    (M - 99.99999999999991) / (E : ℝ) = 0.98 →
    E = 10000 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l710_71094


namespace NUMINAMATH_CALUDE_calculation_proof_l710_71065

theorem calculation_proof : (30 / (7 + 2 - 3)) * 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l710_71065


namespace NUMINAMATH_CALUDE_compare_expressions_min_value_expression_l710_71041

variable (m n : ℝ)

/-- Part 1: Compare m² + n and mn + m when m > n > 1 -/
theorem compare_expressions (hm : m > 0) (hn : n > 0) (hmn : m > n) (hn1 : n > 1) :
  m^2 + n > m*n + m := by sorry

/-- Part 2: Find the minimum value of 2/m + 1/n when m + 2n = 1 -/
theorem min_value_expression (hm : m > 0) (hn : n > 0) (hmn : m > n) (hsum : m + 2*n = 1) :
  ∃ (min_val : ℝ), min_val = 8 ∧ ∀ x, x = 2/m + 1/n → x ≥ min_val := by sorry

end NUMINAMATH_CALUDE_compare_expressions_min_value_expression_l710_71041


namespace NUMINAMATH_CALUDE_tax_discount_commute_l710_71059

theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : tax_rate < 1) (h4 : discount_rate < 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_l710_71059


namespace NUMINAMATH_CALUDE_profit_calculation_l710_71080

/-- Given a profit divided between two parties X and Y in the ratio 1/2 : 1/3,
    where the difference between their shares is 140, prove that the total profit is 700. -/
theorem profit_calculation (profit_x profit_y : ℚ) :
  profit_x / profit_y = 1/2 / (1/3 : ℚ) →
  profit_x - profit_y = 140 →
  profit_x + profit_y = 700 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l710_71080


namespace NUMINAMATH_CALUDE_jimmy_and_irene_payment_l710_71003

/-- The amount paid by Jimmy and Irene for their clothing purchases with a senior citizen discount --/
def amountPaid (jimmyShorts : ℕ) (jimmyShortPrice : ℚ) (ireneShirts : ℕ) (ireneShirtPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  let totalCost := jimmyShorts * jimmyShortPrice + ireneShirts * ireneShirtPrice
  let discountAmount := totalCost * (discountPercentage / 100)
  totalCost - discountAmount

/-- Theorem stating that Jimmy and Irene pay $117 for their purchases --/
theorem jimmy_and_irene_payment :
  amountPaid 3 15 5 17 10 = 117 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_and_irene_payment_l710_71003


namespace NUMINAMATH_CALUDE_cosine_adjacent_extrema_distance_l710_71071

/-- The distance between adjacent highest and lowest points on the graph of y = cos(x+1) is √(π² + 4) -/
theorem cosine_adjacent_extrema_distance : 
  let f : ℝ → ℝ := λ x => Real.cos (x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ - x₁ = π ∧
    f x₁ = 1 ∧ f x₂ = -1 ∧
    Real.sqrt (π^2 + 4) = Real.sqrt ((x₂ - x₁)^2 + (f x₁ - f x₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_cosine_adjacent_extrema_distance_l710_71071


namespace NUMINAMATH_CALUDE_body_speeds_correct_l710_71030

/-- The distance between points A and B in meters -/
def distance : ℝ := 270

/-- The time (in seconds) after which the second body starts moving -/
def delay : ℝ := 11

/-- The time (in seconds) of the first meeting after the second body starts moving -/
def first_meeting : ℝ := 10

/-- The time (in seconds) of the second meeting after the second body starts moving -/
def second_meeting : ℝ := 40

/-- The speed of the first body in meters per second -/
def v1 : ℝ := 16

/-- The speed of the second body in meters per second -/
def v2 : ℝ := 9.6

theorem body_speeds_correct : 
  (delay + first_meeting) * v1 + first_meeting * v2 = distance ∧
  (delay + second_meeting) * v1 - second_meeting * v2 = distance ∧
  v1 > v2 ∧ v2 > 0 := by sorry

end NUMINAMATH_CALUDE_body_speeds_correct_l710_71030


namespace NUMINAMATH_CALUDE_seventh_grade_rooms_l710_71024

/-- The number of rooms on the first floor where seventh-grade boys live -/
def num_rooms : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := sorry

theorem seventh_grade_rooms :
  (6 * (num_rooms - 1) = total_students) ∧
  (5 * num_rooms + 4 = total_students) →
  num_rooms = 10 := by
sorry

end NUMINAMATH_CALUDE_seventh_grade_rooms_l710_71024


namespace NUMINAMATH_CALUDE_representation_theorem_l710_71025

theorem representation_theorem (a b : ℕ+) :
  (∃ (S : Finset ℕ), ∀ (n : ℕ), ∃ (x y : ℕ) (s : ℕ), s ∈ S ∧ n = x^(a:ℕ) + y^(b:ℕ) + s) ↔
  (a = 1 ∨ b = 1) :=
sorry

end NUMINAMATH_CALUDE_representation_theorem_l710_71025


namespace NUMINAMATH_CALUDE_max_one_truthful_dwarf_l710_71033

/-- Represents the height claim of a dwarf -/
structure HeightClaim where
  position : Nat
  claimed_height : Nat

/-- The problem setup for the seven dwarfs -/
def dwarfs_problem : List HeightClaim :=
  [
    ⟨1, 60⟩,
    ⟨2, 61⟩,
    ⟨3, 62⟩,
    ⟨4, 63⟩,
    ⟨5, 64⟩,
    ⟨6, 65⟩,
    ⟨7, 66⟩
  ]

/-- A function to count the maximum number of truthful dwarfs -/
def max_truthful_dwarfs (claims : List HeightClaim) : Nat :=
  sorry

/-- The theorem stating that the maximum number of truthful dwarfs is 1 -/
theorem max_one_truthful_dwarf :
  max_truthful_dwarfs dwarfs_problem = 1 :=
sorry

end NUMINAMATH_CALUDE_max_one_truthful_dwarf_l710_71033


namespace NUMINAMATH_CALUDE_factorization_theorem_l710_71029

theorem factorization_theorem (x y a : ℝ) : 2*x*(a-2) - y*(2-a) = (a-2)*(2*x+y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l710_71029


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l710_71073

/-- Represents the profit function for a product with given pricing and sales conditions -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - (x - 10) * 10)

/-- The selling price that maximizes profit -/
def optimal_price : ℝ := 14

theorem optimal_price_maximizes_profit :
  ∀ (x : ℝ), profit_function x ≤ profit_function optimal_price :=
sorry

#check optimal_price_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l710_71073


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l710_71050

/-- The equation of a line perpendicular to x-2y=3 and passing through (1,2) is y=-2x+4 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (∃ (m b : ℝ), y = m*x + b ∧ 
                 (1, 2) ∈ {(x, y) | y = m*x + b} ∧
                 m * (1/2) = -1) →
  y = -2*x + 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l710_71050


namespace NUMINAMATH_CALUDE_find_y_value_l710_71088

theorem find_y_value (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^10) (h2 : x = 12) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l710_71088


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l710_71037

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 + 1
  (x / (x^2 - 1)) / (1 - 1 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l710_71037


namespace NUMINAMATH_CALUDE_sqrt_inequality_l710_71035

theorem sqrt_inequality (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l710_71035


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l710_71043

theorem complex_fraction_equality : Complex.I * 5 / (1 - Complex.I) = -5/2 + Complex.I * (5/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l710_71043


namespace NUMINAMATH_CALUDE_coffee_price_increase_l710_71091

/-- Given the conditions of the tea and coffee pricing problem, prove that the price of coffee increased by 100% from June to July. -/
theorem coffee_price_increase (june_price : ℝ) (july_mixture_price : ℝ) (july_tea_price : ℝ) :
  -- In June, the price of green tea and coffee were the same
  -- In July, the price of green tea dropped by 70%
  -- In July, a mixture of equal quantities of green tea and coffee costs $3.45 for 3 lbs
  -- In July, a pound of green tea costs $0.3
  june_price > 0 ∧
  july_mixture_price = 3.45 ∧
  july_tea_price = 0.3 ∧
  july_tea_price = june_price * 0.3 →
  -- The price of coffee increased by 100%
  (((july_mixture_price - 3 * july_tea_price / 2) * 2 / 3 - june_price) / june_price) * 100 = 100 :=
by sorry

end NUMINAMATH_CALUDE_coffee_price_increase_l710_71091


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l710_71021

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l710_71021


namespace NUMINAMATH_CALUDE_factorization_proof_l710_71004

theorem factorization_proof (m n : ℝ) : m^2 - m*n = m*(m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l710_71004


namespace NUMINAMATH_CALUDE_mark_second_play_time_l710_71064

/-- Calculates the time Mark played in the second part of a soccer game. -/
def second_play_time (total_time initial_play sideline : ℕ) : ℕ :=
  total_time - initial_play - sideline

/-- Theorem: Mark played 35 minutes in the second part of the game. -/
theorem mark_second_play_time :
  let total_time : ℕ := 90
  let initial_play : ℕ := 20
  let sideline : ℕ := 35
  second_play_time total_time initial_play sideline = 35 := by
  sorry

end NUMINAMATH_CALUDE_mark_second_play_time_l710_71064


namespace NUMINAMATH_CALUDE_f_sum_equals_e_minus_one_l710_71079

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem f_sum_equals_e_minus_one 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : has_period_two f)
  (h_interval : ∀ x ∈ Set.Icc 0 1, f x = Real.exp x - 1) :
  f 2018 + f (-2019) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_e_minus_one_l710_71079


namespace NUMINAMATH_CALUDE_marcus_final_cards_l710_71038

def marcus_initial_cards : ℕ := 2100
def carter_initial_cards : ℕ := 3040
def carter_gift_cards : ℕ := 750
def carter_gift_percentage : ℚ := 125 / 1000

theorem marcus_final_cards : 
  marcus_initial_cards + carter_gift_cards + 
  (carter_initial_cards * carter_gift_percentage).floor = 3230 :=
by sorry

end NUMINAMATH_CALUDE_marcus_final_cards_l710_71038


namespace NUMINAMATH_CALUDE_set_equality_l710_71002

-- Define the sets A, B, and C as subsets of ℝ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}
def C : Set ℝ := {x | 3 < x ∧ x ≤ 4}

-- State the theorem
theorem set_equality : C = (Set.univ \ A) ∩ B := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l710_71002


namespace NUMINAMATH_CALUDE_cookie_radius_l710_71051

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 - 8 = 2*x + 4*y) →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l710_71051
