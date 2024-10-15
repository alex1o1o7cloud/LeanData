import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l4048_404854

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (a + 2*b + 2/(a + 1)) * (b + 2*a + 2/(b + 1)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4048_404854


namespace NUMINAMATH_CALUDE_available_seats_l4048_404808

theorem available_seats (total_seats : ℕ) (taken_fraction : ℚ) (broken_fraction : ℚ) : 
  total_seats = 500 →
  taken_fraction = 2 / 5 →
  broken_fraction = 1 / 10 →
  (total_seats : ℚ) - (taken_fraction * total_seats + broken_fraction * total_seats) = 250 := by
  sorry

end NUMINAMATH_CALUDE_available_seats_l4048_404808


namespace NUMINAMATH_CALUDE_triangle_problem_l4048_404829

open Real

theorem triangle_problem (A B C : ℝ) (h : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A + B = 3 * C ∧
  2 * sin (A - C) = sin B →
  sin A = 3 * sqrt 10 / 10 ∧
  (∀ AB : ℝ, AB = 5 → h = 6 ∧ h * AB / 2 = sin C * AB * sin A / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l4048_404829


namespace NUMINAMATH_CALUDE_special_sale_savings_l4048_404828

/-- Given a special sale where 25 tickets can be purchased for the price of 21.5 tickets,
    prove that buying 50 tickets at this rate results in a 14% savings compared to the original price. -/
theorem special_sale_savings : ∀ (P : ℝ), P > 0 →
  let sale_price : ℝ := 21.5 * P / 25
  let original_price_50 : ℝ := 50 * P
  let sale_price_50 : ℝ := 50 * sale_price
  let savings : ℝ := original_price_50 - sale_price_50
  let savings_percentage : ℝ := savings / original_price_50 * 100
  savings_percentage = 14 := by
  sorry

end NUMINAMATH_CALUDE_special_sale_savings_l4048_404828


namespace NUMINAMATH_CALUDE_midpoint_sum_l4048_404820

/-- Given that C = (4, 3) is the midpoint of line segment AB, where A = (2, 7) and B = (x, y), prove that x + y = 5. -/
theorem midpoint_sum (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (7 + y) / 2 → 
  x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_l4048_404820


namespace NUMINAMATH_CALUDE_digit_divisible_by_9_l4048_404876

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

theorem digit_divisible_by_9 :
  is_divisible_by_9 5274 ∧ 
  ∀ B : ℕ, B ≤ 9 → B ≠ 4 → ¬(is_divisible_by_9 (5270 + B)) :=
by sorry

end NUMINAMATH_CALUDE_digit_divisible_by_9_l4048_404876


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l4048_404869

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  (a^3 * b^3) / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  (a^3 * c^3) / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  (b^3 * c^3) / ((b^3 - a^2 * c) * (c^3 - a^2 * b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l4048_404869


namespace NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l4048_404853

theorem numbers_less_than_reciprocals : ∃ (S : Set ℝ), 
  S = {-1/2, -3, 3, 1/2, 0} ∧ 
  (∀ x ∈ S, x ≠ 0 → (x < 1/x ↔ (x = -3 ∨ x = 1/2))) := by
  sorry

end NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l4048_404853


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l4048_404831

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l4048_404831


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l4048_404840

noncomputable def f' (f'1 : ℝ) : ℝ → ℝ := fun x ↦ 2 * f'1 / x - 1

theorem f_derivative_at_one :
  ∃ f'1 : ℝ, (f' f'1) 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l4048_404840


namespace NUMINAMATH_CALUDE_total_pools_count_l4048_404857

/-- The number of stores operated by Pat's Pool Supply -/
def pool_supply_stores : ℕ := 4

/-- The number of stores operated by Pat's Ark & Athletic Wear -/
def ark_athletic_stores : ℕ := 6

/-- The ratio of swimming pools between Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def pool_ratio : ℕ := 3

/-- The number of pools in one Pat's Ark & Athletic Wear store -/
def pools_per_ark_athletic : ℕ := 200

/-- The total number of swimming pools across all Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def total_pools : ℕ := pool_supply_stores * pool_ratio * pools_per_ark_athletic + ark_athletic_stores * pools_per_ark_athletic

theorem total_pools_count : total_pools = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_pools_count_l4048_404857


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l4048_404833

theorem fraction_equality_solution : ∃! x : ℚ, (1 + x) / (5 + x) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l4048_404833


namespace NUMINAMATH_CALUDE_secret_spreading_l4048_404851

/-- 
Theorem: Secret Spreading
Given:
- On day 0 (Monday), one person knows a secret.
- Each day, every person who knows the secret tells two new people.
- The number of people who know the secret on day n is 2^(n+1) - 1.

Prove: It takes 9 days for 1023 people to know the secret.
-/
theorem secret_spreading (n : ℕ) : 
  (2^(n+1) - 1 = 1023) → n = 9 := by
  sorry

#check secret_spreading

end NUMINAMATH_CALUDE_secret_spreading_l4048_404851


namespace NUMINAMATH_CALUDE_willey_farm_land_allocation_l4048_404865

/-- The Willey Farm Collective land allocation problem -/
theorem willey_farm_land_allocation :
  let corn_cost : ℝ := 42
  let wheat_cost : ℝ := 35
  let total_capital : ℝ := 165200
  let wheat_acres : ℝ := 3400
  let corn_acres : ℝ := (total_capital - wheat_cost * wheat_acres) / corn_cost
  corn_acres + wheat_acres = 4500 := by
  sorry

end NUMINAMATH_CALUDE_willey_farm_land_allocation_l4048_404865


namespace NUMINAMATH_CALUDE_ball_height_properties_l4048_404826

/-- The height of a ball as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- Theorem stating the maximum height and height at t = 1 -/
theorem ball_height_properties :
  (∀ t, h t ≤ 40) ∧ (h 1 = 40) := by
  sorry

end NUMINAMATH_CALUDE_ball_height_properties_l4048_404826


namespace NUMINAMATH_CALUDE_opposite_of_three_l4048_404849

/-- The opposite of a real number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of 3 is -3. -/
theorem opposite_of_three : opposite 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l4048_404849


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l4048_404896

/-- Given a rectangle with actual length L and width W, where one side is measured
    as 1.05L and the other as 0.96W, the error percent in the calculated area is 0.8%. -/
theorem rectangle_area_error_percent (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let actual_area := L * W
  let measured_area := (1.05 * L) * (0.96 * W)
  let error := measured_area - actual_area
  let error_percent := (error / actual_area) * 100
  error_percent = 0.8 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l4048_404896


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l4048_404815

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (digits : List Bool) : ℕ :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true]  -- 101101₂
  let b := [true, true, false, true]               -- 1101₂
  let product := [true, false, false, false, true, false, false, false, true, true, true]  -- 10001000111₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l4048_404815


namespace NUMINAMATH_CALUDE_circumradius_inscribed_radius_inequality_l4048_404823

/-- A triangle with its circumscribed and inscribed circles -/
structure Triangle where
  -- Radius of the circumscribed circle
  R : ℝ
  -- Radius of the inscribed circle
  r : ℝ
  -- Predicate indicating if the triangle is equilateral
  is_equilateral : Prop

/-- The radius of the circumscribed circle is at least twice the radius of the inscribed circle,
    with equality if and only if the triangle is equilateral -/
theorem circumradius_inscribed_radius_inequality (t : Triangle) :
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ t.is_equilateral) := by
  sorry

end NUMINAMATH_CALUDE_circumradius_inscribed_radius_inequality_l4048_404823


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4048_404891

theorem algebraic_expression_value (a b : ℝ) :
  2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18 →
  9 * b - 6 * a + 2 = 32 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4048_404891


namespace NUMINAMATH_CALUDE_no_rational_roots_odd_coeff_l4048_404868

theorem no_rational_roots_odd_coeff (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Int.gcd p q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_rational_roots_odd_coeff_l4048_404868


namespace NUMINAMATH_CALUDE_present_worth_calculation_l4048_404848

/-- Calculates the present worth given the banker's gain, interest rate, and time period -/
def present_worth (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  bankers_gain / (interest_rate * time)

/-- Theorem stating that under given conditions, the present worth is 120 -/
theorem present_worth_calculation :
  let bankers_gain : ℚ := 24
  let interest_rate : ℚ := 1/10  -- 10% as a rational number
  let time : ℚ := 2
  present_worth bankers_gain interest_rate time = 120 := by
sorry

#eval present_worth 24 (1/10) 2

end NUMINAMATH_CALUDE_present_worth_calculation_l4048_404848


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l4048_404835

theorem matching_shoes_probability (n : ℕ) (h : n = 12) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 23 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l4048_404835


namespace NUMINAMATH_CALUDE_f_50_solutions_l4048_404843

-- Define f_0
def f_0 (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

-- Define f_n recursively
def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => f_0 x
  | n + 1 => |f n x| - 1

-- Theorem statement
theorem f_50_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f 50 x = 0) ∧ 
                    (∀ x ∉ S, f 50 x ≠ 0) ∧ 
                    Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_f_50_solutions_l4048_404843


namespace NUMINAMATH_CALUDE_class_average_weight_l4048_404894

/-- Given two sections A and B in a class, with their respective number of students and average weights,
    prove that the average weight of the whole class is as calculated. -/
theorem class_average_weight 
  (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_A : ℝ) (avg_weight_B : ℝ) :
  students_A = 40 →
  students_B = 20 →
  avg_weight_A = 50 →
  avg_weight_B = 40 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 46.67 :=
by sorry

end NUMINAMATH_CALUDE_class_average_weight_l4048_404894


namespace NUMINAMATH_CALUDE_max_value_of_f_l4048_404830

def f (x : ℝ) := x^4 - 4*x + 3

theorem max_value_of_f : 
  ∃ (m : ℝ), m = 72 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4048_404830


namespace NUMINAMATH_CALUDE_problem_solving_probability_l4048_404890

theorem problem_solving_probability :
  let p_xavier : ℚ := 1/4
  let p_yvonne : ℚ := 2/3
  let p_zelda : ℚ := 5/8
  let p_william : ℚ := 7/10
  (p_xavier * p_yvonne * p_william * (1 - p_zelda) : ℚ) = 7/160 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l4048_404890


namespace NUMINAMATH_CALUDE_square_with_specific_digits_l4048_404844

theorem square_with_specific_digits (S : ℕ) : 
  (∃ (E : ℕ), S^2 = 10 * (10 * (10^100 * E + 2*E) + 2) + 5) →
  (S^2 % 10 = 5 ∧ S = (10^101 + 5) / 3) :=
by sorry

end NUMINAMATH_CALUDE_square_with_specific_digits_l4048_404844


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l4048_404847

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l4048_404847


namespace NUMINAMATH_CALUDE_min_value_a_l4048_404855

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 3 - x / Real.exp x

theorem min_value_a :
  (∃ (a : ℝ), ∀ (x : ℝ), x ≥ -2 → f x ≤ a) ∧ 
  (∀ (b : ℝ), (∃ (x : ℝ), x ≥ -2 ∧ f x > b) → b < 1 - 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l4048_404855


namespace NUMINAMATH_CALUDE_circle_center_transformation_l4048_404838

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Translates a point up by a given amount -/
def translate_up (p : ℝ × ℝ) (amount : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + amount)

/-- The final position of the center of circle S after transformations -/
def final_position (initial : ℝ × ℝ) : ℝ × ℝ :=
  translate_up (reflect_x (reflect_y initial)) 5

theorem circle_center_transformation :
  final_position (3, -4) = (-3, 9) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l4048_404838


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_225_l4048_404856

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_225 :
  (∀ m : ℕ, m < 111111100 → ¬(225 ∣ m ∧ is_binary_number m)) ∧
  (225 ∣ 111111100 ∧ is_binary_number 111111100) :=
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_225_l4048_404856


namespace NUMINAMATH_CALUDE_major_premise_is_false_l4048_404893

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in_plane : Line → Plane → Prop)

-- State the theorem
theorem major_premise_is_false :
  ¬(∀ (l : Line) (p : Plane),
    parallel_line_plane l p →
    ∀ (m : Line), contained_in_plane m p → parallel_lines l m) :=
by sorry

end NUMINAMATH_CALUDE_major_premise_is_false_l4048_404893


namespace NUMINAMATH_CALUDE_distance_OQ_l4048_404887

-- Define the geometric setup
structure GeometricSetup where
  R : ℝ  -- Radius of larger circle
  r : ℝ  -- Radius of smaller circle
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C

-- Define the theorem
theorem distance_OQ (setup : GeometricSetup) : 
  ∃ (OQ : ℝ), OQ = Real.sqrt (setup.R^2 - 2*setup.r*setup.R) :=
sorry

end NUMINAMATH_CALUDE_distance_OQ_l4048_404887


namespace NUMINAMATH_CALUDE_linear_function_composition_l4048_404864

/-- Given two functions f and g, where f is a linear function with real coefficients a and b,
    and g is defined as g(x) = 3x - 4, prove that a + b = 11/3 if g(f(x)) = 4x + 3 for all x. -/
theorem linear_function_composition (a b : ℝ) :
  (∀ x, (3 * ((a * x + b) : ℝ) - 4) = 4 * x + 3) →
  a + b = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l4048_404864


namespace NUMINAMATH_CALUDE_monomial_replacement_l4048_404879

theorem monomial_replacement (x : ℝ) : 
  let expression := (x^4 - 3)^2 + (x^3 + 3*x)^2
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    expression = a * x^n₁ + b * x^n₂ + c * x^n₃ + d * x^n₄ ∧
    n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > n₄ ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_monomial_replacement_l4048_404879


namespace NUMINAMATH_CALUDE_marco_juice_mixture_l4048_404874

/-- Calculates the remaining mixture after giving some away -/
def remaining_mixture (apple_juice orange_juice given_away : ℚ) : ℚ :=
  apple_juice + orange_juice - given_away

/-- Proves that the remaining mixture is 13/4 gallons -/
theorem marco_juice_mixture :
  let apple_juice : ℚ := 4
  let orange_juice : ℚ := 7/4
  let given_away : ℚ := 5/2
  remaining_mixture apple_juice orange_juice given_away = 13/4 := by
sorry

end NUMINAMATH_CALUDE_marco_juice_mixture_l4048_404874


namespace NUMINAMATH_CALUDE_vegetables_sold_mass_l4048_404839

/-- Proves that given 15 kg of carrots, 13 kg of zucchini, and 8 kg of broccoli,
    if a merchant sells half of the total vegetables, the mass of vegetables sold is 18 kg. -/
theorem vegetables_sold_mass 
  (carrots : ℝ) 
  (zucchini : ℝ) 
  (broccoli : ℝ) 
  (h1 : carrots = 15)
  (h2 : zucchini = 13)
  (h3 : broccoli = 8) :
  (carrots + zucchini + broccoli) / 2 = 18 := by
  sorry

#check vegetables_sold_mass

end NUMINAMATH_CALUDE_vegetables_sold_mass_l4048_404839


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l4048_404866

theorem quadratic_equation_sum (a b : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 25 = 0 ↔ (x + a)^2 = b) → a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l4048_404866


namespace NUMINAMATH_CALUDE_stream_speed_l4048_404819

/-- Proves that the speed of the stream is 8 kmph given the conditions -/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 24 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 8 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l4048_404819


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_one_l4048_404800

theorem negation_of_forall_geq_one :
  (¬ (∀ x : ℝ, x ≥ 1)) ↔ (∃ x : ℝ, x < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_one_l4048_404800


namespace NUMINAMATH_CALUDE_distance_between_cities_l4048_404871

/-- The distance between two cities given specific travel conditions -/
theorem distance_between_cities (cara_speed dan_min_speed : ℝ) 
  (dan_delay : ℝ) (h1 : cara_speed = 30) (h2 : dan_min_speed = 36) 
  (h3 : dan_delay = 1) : 
  ∃ D : ℝ, D = 180 ∧ D / dan_min_speed = D / cara_speed - dan_delay := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l4048_404871


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l4048_404850

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 145

/-- The number of cakes left after selling -/
def cakes_left : ℕ := 72

/-- The total number of cakes made by the baker -/
def total_cakes : ℕ := cakes_sold + cakes_left

theorem baker_cakes_theorem : total_cakes = 217 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l4048_404850


namespace NUMINAMATH_CALUDE_license_plate_difference_l4048_404882

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible license plates in Alphazia -/
def alphazia_plates : ℕ := num_letters^4 * num_digits^3

/-- The number of possible license plates in Betaland -/
def betaland_plates : ℕ := num_letters^5 * num_digits^2

/-- The difference in the number of possible license plates between Alphazia and Betaland -/
def plate_difference : ℤ := alphazia_plates - betaland_plates

theorem license_plate_difference :
  plate_difference = -731161600 := by sorry

end NUMINAMATH_CALUDE_license_plate_difference_l4048_404882


namespace NUMINAMATH_CALUDE_digit_sum_problem_l4048_404898

theorem digit_sum_problem (x y z w : ℕ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x < 10 ∧ y < 10 ∧ z < 10 ∧ w < 10 →
  100 * x + 10 * y + w + 100 * z + 10 * w + x = 1000 →
  x + y + z + w = 18 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l4048_404898


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l4048_404827

theorem three_digit_multiples_of_seven : 
  (Finset.filter (fun k => 100 ≤ 7 * k ∧ 7 * k ≤ 999) (Finset.range 1000)).card = 128 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l4048_404827


namespace NUMINAMATH_CALUDE_roots_squared_relation_l4048_404880

-- Define the polynomials h(x) and p(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem roots_squared_relation (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → p a b c (x^2) = 0) →
  a = -1 ∧ b = -2 ∧ c = 16 :=
by sorry

end NUMINAMATH_CALUDE_roots_squared_relation_l4048_404880


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l4048_404816

theorem quadratic_expression_equality (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l4048_404816


namespace NUMINAMATH_CALUDE_infinitely_many_primes_not_in_S_a_l4048_404803

-- Define the set S_a
def S_a (a : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ b : ℕ, Odd b ∧ p ∣ (2^(2^a))^b - 1}

-- State the theorem
theorem infinitely_many_primes_not_in_S_a :
  ∀ a : ℕ, a > 0 → Set.Infinite {p : ℕ | Nat.Prime p ∧ p ∉ S_a a} :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_not_in_S_a_l4048_404803


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l4048_404824

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3^(n-2) + k, prove that k = -1/9 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n : ℕ, S n = 3^(n - 2) + k) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n - 1)) →
  (∀ n : ℕ, n ≥ 2 → a n / a (n - 1) = a (n + 1) / a n) →
  k = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l4048_404824


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l4048_404895

theorem toy_store_revenue_ratio :
  ∀ (N D J : ℝ),
  J = (1/3) * N →
  D = 3.75 * ((N + J) / 2) →
  N / D = 2/5 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l4048_404895


namespace NUMINAMATH_CALUDE_four_digit_numbers_proof_l4048_404885

theorem four_digit_numbers_proof (A B : ℕ) : 
  (1000 ≤ A) ∧ (A < 10000) ∧ 
  (1000 ≤ B) ∧ (B < 10000) ∧ 
  (Real.log A / Real.log 10 = 3 + Real.log 4 / Real.log 10) ∧
  (B.div 1000 + B % 10 = 10) ∧
  (B = A / 2 - 21) →
  (A = 4000 ∧ B = 1979) := by
sorry

end NUMINAMATH_CALUDE_four_digit_numbers_proof_l4048_404885


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l4048_404899

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of our circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

theorem circle_passes_through_points :
  ∃ (c : Circle),
    (∀ (x y : ℝ), circle_equation x y ↔ c.contains (x, y)) ∧
    c.contains (0, 0) ∧
    c.contains (4, 0) ∧
    c.contains (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l4048_404899


namespace NUMINAMATH_CALUDE_shaded_area_problem_l4048_404817

theorem shaded_area_problem (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 4 →
  triangle_base = 4 →
  triangle_height = 3 →
  square_side * square_side - (1 / 2 * triangle_base * triangle_height) = 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l4048_404817


namespace NUMINAMATH_CALUDE_professors_seating_arrangements_l4048_404821

/-- Represents the seating arrangement problem with professors and students. -/
structure SeatingArrangement where
  totalChairs : Nat
  numStudents : Nat
  numProfessors : Nat
  professorsBetweenStudents : Bool

/-- Calculates the number of ways professors can choose their chairs. -/
def waysToChooseChairs (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Theorem stating that the number of ways to choose chairs is 24 for the given problem. -/
theorem professors_seating_arrangements
  (arrangement : SeatingArrangement)
  (h1 : arrangement.totalChairs = 11)
  (h2 : arrangement.numStudents = 7)
  (h3 : arrangement.numProfessors = 4)
  (h4 : arrangement.professorsBetweenStudents = true) :
  waysToChooseChairs arrangement = 24 := by
  sorry

end NUMINAMATH_CALUDE_professors_seating_arrangements_l4048_404821


namespace NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l4048_404845

/-- Represents the savings of a person over time -/
structure Savings where
  initial : ℕ  -- Initial savings
  monthly : ℕ  -- Monthly savings rate
  months : ℕ   -- Number of months passed

/-- Calculates the total savings after a given number of months -/
def totalSavings (s : Savings) : ℕ :=
  s.initial + s.monthly * s.months

/-- Xiaoxia's savings parameters -/
def xiaoxia : Savings :=
  { initial := 52, monthly := 15, months := 0 }

/-- Xiaoming's savings parameters -/
def xiaoming : Savings :=
  { initial := 70, monthly := 12, months := 0 }

/-- Theorem stating when Xiaoxia's savings exceed Xiaoming's -/
theorem xiaoxia_exceeds_xiaoming (n : ℕ) :
  totalSavings { xiaoxia with months := n } > totalSavings { xiaoming with months := n } ↔
  52 + 15 * n > 70 + 12 * n :=
sorry

end NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l4048_404845


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l4048_404802

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l4048_404802


namespace NUMINAMATH_CALUDE_apple_sale_total_l4048_404812

theorem apple_sale_total (red_apples : ℕ) (ratio_red : ℕ) (ratio_green : ℕ) : 
  red_apples = 32 → 
  ratio_red = 8 → 
  ratio_green = 3 → 
  red_apples + (red_apples * ratio_green / ratio_red) = 44 := by
sorry

end NUMINAMATH_CALUDE_apple_sale_total_l4048_404812


namespace NUMINAMATH_CALUDE_wire_division_l4048_404873

/-- Given a wire that can be divided into two parts of 120 cm each with 2.4 cm left over,
    prove that when divided into three equal parts, each part is 80.8 cm long. -/
theorem wire_division (wire_length : ℝ) (h1 : wire_length = 2 * 120 + 2.4) :
  wire_length / 3 = 80.8 := by
sorry

end NUMINAMATH_CALUDE_wire_division_l4048_404873


namespace NUMINAMATH_CALUDE_pony_lesson_cost_l4048_404872

/-- The cost per lesson for Andrea's pony, given the following conditions:
  * Monthly pasture rent is $500
  * Daily food cost is $10
  * There are two lessons per week
  * Total annual expenditure on the pony is $15890
-/
theorem pony_lesson_cost : 
  let monthly_pasture_rent : ℕ := 500
  let daily_food_cost : ℕ := 10
  let lessons_per_week : ℕ := 2
  let total_annual_cost : ℕ := 15890
  let annual_pasture_cost : ℕ := monthly_pasture_rent * 12
  let annual_food_cost : ℕ := daily_food_cost * 365
  let annual_lessons : ℕ := lessons_per_week * 52
  let lesson_cost : ℕ := (total_annual_cost - (annual_pasture_cost + annual_food_cost)) / annual_lessons
  lesson_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_pony_lesson_cost_l4048_404872


namespace NUMINAMATH_CALUDE_min_value_when_a_2_a_values_for_max_3_l4048_404852

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- Part 1: Minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (min : ℝ), min = -1 ∧ ∀ x ∈ Set.Icc 0 3, f 2 x ≥ min :=
sorry

-- Part 2: Values of a for maximum 3 in [0, 1]
theorem a_values_for_max_3 :
  (∃ (max : ℝ), max = 3 ∧ ∀ x ∈ Set.Icc 0 1, f a x ≤ max) →
  (a = -2 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_2_a_values_for_max_3_l4048_404852


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l4048_404877

theorem fraction_equals_zero (x : ℝ) : x = 5 → (x - 5) / (6 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l4048_404877


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4048_404811

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), 
    (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → 
      (5 * x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2)) ∧
    C = 32/9 ∧ D = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4048_404811


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l4048_404875

theorem gcd_of_specific_numbers : 
  let m : ℕ := 3333333
  let n : ℕ := 99999999
  Nat.gcd m n = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l4048_404875


namespace NUMINAMATH_CALUDE_scout_sale_profit_l4048_404884

/-- Represents the scout troop's candy bar sale scenario -/
structure CandyBarSale where
  total_bars : ℕ
  purchase_price : ℚ
  sold_bars : ℕ
  selling_price : ℚ

/-- Calculates the profit for the candy bar sale -/
def calculate_profit (sale : CandyBarSale) : ℚ :=
  sale.selling_price * sale.sold_bars - sale.purchase_price * sale.total_bars

/-- The specific candy bar sale scenario from the problem -/
def scout_sale : CandyBarSale :=
  { total_bars := 2000
  , purchase_price := 3 / 4
  , sold_bars := 1950
  , selling_price := 2 / 3 }

/-- Theorem stating that the profit for the scout troop's candy bar sale is -200 -/
theorem scout_sale_profit :
  calculate_profit scout_sale = -200 := by
  sorry


end NUMINAMATH_CALUDE_scout_sale_profit_l4048_404884


namespace NUMINAMATH_CALUDE_three_workers_completion_time_l4048_404809

/-- The time taken for three workers to complete a task together, given their individual completion times -/
theorem three_workers_completion_time 
  (x_time y_time z_time : ℝ) 
  (hx : x_time = 30) 
  (hy : y_time = 45) 
  (hz : z_time = 60) : 
  (1 / x_time + 1 / y_time + 1 / z_time)⁻¹ = 180 / 13 := by
  sorry

#check three_workers_completion_time

end NUMINAMATH_CALUDE_three_workers_completion_time_l4048_404809


namespace NUMINAMATH_CALUDE_final_employee_count_l4048_404805

/-- Represents the workforce of Company X throughout the year --/
structure CompanyWorkforce where
  initial_total : ℕ
  initial_female : ℕ
  second_quarter_total : ℕ
  second_quarter_female : ℕ
  third_quarter_total : ℕ
  third_quarter_female : ℕ
  final_total : ℕ
  final_female : ℕ

/-- Theorem stating the final number of employees given the workforce changes --/
theorem final_employee_count (w : CompanyWorkforce) : w.final_total = 700 :=
  by
  have h1 : w.initial_female = (60 : ℚ) / 100 * w.initial_total := by sorry
  have h2 : w.second_quarter_total = w.initial_total + 30 := by sorry
  have h3 : w.second_quarter_female = w.initial_female := by sorry
  have h4 : w.second_quarter_female = (57 : ℚ) / 100 * w.second_quarter_total := by sorry
  have h5 : w.third_quarter_total = w.second_quarter_total + 50 := by sorry
  have h6 : w.third_quarter_female = w.second_quarter_female + 50 := by sorry
  have h7 : w.third_quarter_female = (62 : ℚ) / 100 * w.third_quarter_total := by sorry
  have h8 : w.final_total = w.third_quarter_total + 50 := by sorry
  have h9 : w.final_female = w.third_quarter_female + 10 := by sorry
  have h10 : w.final_female = (58 : ℚ) / 100 * w.final_total := by sorry
  sorry


end NUMINAMATH_CALUDE_final_employee_count_l4048_404805


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l4048_404841

/-- The probability of A and B drawing in Chinese chess -/
theorem chinese_chess_draw_probability 
  (p_a_not_lose : ℝ) 
  (p_b_not_lose : ℝ) 
  (h1 : p_a_not_lose = 0.8) 
  (h2 : p_b_not_lose = 0.7) : 
  ∃ (p_draw : ℝ), p_draw = 0.5 ∧ p_a_not_lose = (1 - p_b_not_lose) + p_draw :=
sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l4048_404841


namespace NUMINAMATH_CALUDE_container_volume_ratio_l4048_404836

theorem container_volume_ratio : 
  ∀ (A B : ℝ), A > 0 → B > 0 → 
  (4/5 * A = 2/3 * B) → 
  (A / B = 5/6) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l4048_404836


namespace NUMINAMATH_CALUDE_right_triangle_properties_l4048_404859

/-- A right triangle with hypotenuse 13 and one leg 5 -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : hypotenuse^2 = leg1^2 + leg2^2
  hypotenuse_is_13 : hypotenuse = 13
  leg1_is_5 : leg1 = 5

/-- Properties of the specific right triangle -/
theorem right_triangle_properties (t : RightTriangle) :
  t.leg2 = 12 ∧
  (1/2 : ℝ) * t.leg1 * t.leg2 = 30 ∧
  t.leg1 + t.leg2 + t.hypotenuse = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l4048_404859


namespace NUMINAMATH_CALUDE_alans_current_rate_prove_alans_current_rate_l4048_404846

/-- Alan's attempt to beat Kevin's hot wings eating record -/
theorem alans_current_rate (kevin_wings : ℕ) (kevin_time : ℕ) (alan_additional : ℕ) : ℕ :=
  let kevin_rate := kevin_wings / kevin_time
  let alan_target_rate := kevin_rate + 1
  alan_target_rate - alan_additional

/-- Proof of Alan's current rate of eating hot wings -/
theorem prove_alans_current_rate :
  alans_current_rate 64 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_alans_current_rate_prove_alans_current_rate_l4048_404846


namespace NUMINAMATH_CALUDE_prob_one_six_max_l4048_404842

/-- The probability of rolling exactly one six when rolling n dice -/
def prob_one_six (n : ℕ) : ℚ :=
  (n : ℚ) * (5 ^ (n - 1) : ℚ) / (6 ^ n : ℚ)

/-- The statement that the probability of rolling exactly one six is maximized for 5 or 6 dice -/
theorem prob_one_six_max :
  (∀ k : ℕ, prob_one_six k ≤ prob_one_six 5) ∧
  (prob_one_six 5 = prob_one_six 6) ∧
  (∀ k : ℕ, k > 6 → prob_one_six k < prob_one_six 6) :=
sorry

end NUMINAMATH_CALUDE_prob_one_six_max_l4048_404842


namespace NUMINAMATH_CALUDE_largest_angle_in_tangent_circles_triangle_l4048_404863

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent --/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the x-axis --/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  let (_, y) := c.center
  y = c.radius

/-- Theorem about the largest angle in a triangle formed by centers of three mutually tangent circles --/
theorem largest_angle_in_tangent_circles_triangle (A B C : Circle) :
  are_externally_tangent A B ∧ 
  are_externally_tangent B C ∧ 
  are_externally_tangent C A ∧
  is_tangent_to_x_axis A ∧
  is_tangent_to_x_axis B ∧
  is_tangent_to_x_axis C →
  ∃ γ : ℝ, π/2 < γ ∧ γ ≤ 2 * Real.arcsin (4/5) ∧ 
  γ = max (Real.arccos ((A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 - A.radius^2 - C.radius^2) / (2 * A.radius * C.radius))
          (max (Real.arccos ((B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 - B.radius^2 - C.radius^2) / (2 * B.radius * C.radius))
               (Real.arccos ((A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 - A.radius^2 - B.radius^2) / (2 * A.radius * B.radius))) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_tangent_circles_triangle_l4048_404863


namespace NUMINAMATH_CALUDE_robbie_win_probability_l4048_404862

/-- A special six-sided die where rolling number x is x times as likely as rolling a 1 -/
structure SpecialDie :=
  (prob_one : ℝ)
  (sum_to_one : prob_one * (1 + 2 + 3 + 4 + 5 + 6) = 1)

/-- The game where two players roll the special die three times each -/
def Game (d : SpecialDie) :=
  { score : ℕ × ℕ // score.1 ≤ 18 ∧ score.2 ≤ 18 }

/-- The probability of rolling a specific number on the special die -/
def prob_roll (d : SpecialDie) (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 6 then n * d.prob_one else 0

/-- The probability of Robbie winning given the current game state -/
def prob_robbie_win (d : SpecialDie) (g : Game d) : ℝ :=
  sorry

theorem robbie_win_probability (d : SpecialDie) (g : Game d) 
  (h1 : g.val.1 = 8) (h2 : g.val.2 = 10) : 
  prob_robbie_win d g = 55 / 441 :=
sorry

end NUMINAMATH_CALUDE_robbie_win_probability_l4048_404862


namespace NUMINAMATH_CALUDE_actors_in_one_hour_show_l4048_404878

/-- Calculates the number of actors in a show given the show duration, performance time per set, and number of actors per set. -/
def actors_in_show (show_duration : ℕ) (performance_time : ℕ) (actors_per_set : ℕ) : ℕ :=
  (show_duration / performance_time) * actors_per_set

/-- Proves that given the specified conditions, the number of actors in a 1-hour show is 20. -/
theorem actors_in_one_hour_show :
  actors_in_show 60 15 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_actors_in_one_hour_show_l4048_404878


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l4048_404810

/-- The average speed of a round trip, given the outbound speed and the fact that the return journey takes twice as long -/
theorem round_trip_average_speed (outbound_speed : ℝ) 
  (h1 : outbound_speed = 54) 
  (h2 : return_time = 2 * outbound_time) : 
  average_speed = 36 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l4048_404810


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l4048_404867

def divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem divisibility_equivalence :
  (∀ n : ℕ, divisible n 6 → divisible n 3) ↔
  (∀ n : ℕ, ¬(divisible n 3) → ¬(divisible n 6)) ∧
  (∀ n : ℕ, ¬(divisible n 6) ∨ divisible n 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l4048_404867


namespace NUMINAMATH_CALUDE_sum_of_numbers_l4048_404861

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l4048_404861


namespace NUMINAMATH_CALUDE_binomial_coefficient_and_increase_l4048_404870

variable (n : ℕ)

theorem binomial_coefficient_and_increase :
  (Nat.choose n 2 = n * (n - 1) / 2) ∧
  (Nat.choose (n + 1) 2 - Nat.choose n 2 = n) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_and_increase_l4048_404870


namespace NUMINAMATH_CALUDE_circle_equation_l4048_404858

/-- Given a real number a, prove that the equation a²x² + (a+2)y² + 4x + 8y + 5a = 0
    represents a circle with center (-2, -4) and radius 5 if and only if a = -1 -/
theorem circle_equation (a : ℝ) :
  (∃ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0) ∧
  (∀ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0 ↔
    (x + 2)^2 + (y + 4)^2 = 25) ↔
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l4048_404858


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4048_404801

theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = 4 / 5) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 41 / 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4048_404801


namespace NUMINAMATH_CALUDE_cycle_original_price_l4048_404804

/-- The original price of a cycle sold at a loss -/
def original_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem: The original price of a cycle is 1750, given a selling price of 1610 and a loss of 8% -/
theorem cycle_original_price : 
  original_price 1610 8 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l4048_404804


namespace NUMINAMATH_CALUDE_consecutive_product_divisibility_l4048_404825

theorem consecutive_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 8 * m) →
  ¬ (∀ m : ℤ, n = 64 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_divisibility_l4048_404825


namespace NUMINAMATH_CALUDE_waiter_customers_proof_l4048_404897

/-- Calculates the number of remaining customers for a waiter given the initial number of tables,
    number of tables that left, and number of customers per table. -/
def remaining_customers (initial_tables : ℝ) (tables_left : ℝ) (customers_per_table : ℝ) : ℝ :=
  (initial_tables - tables_left) * customers_per_table

/-- Proves that the number of remaining customers for a waiter with 44.0 initial tables,
    12.0 tables that left, and 8.0 customers per table is 256.0. -/
theorem waiter_customers_proof :
  remaining_customers 44.0 12.0 8.0 = 256.0 := by
  sorry

#eval remaining_customers 44.0 12.0 8.0

end NUMINAMATH_CALUDE_waiter_customers_proof_l4048_404897


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l4048_404832

theorem quadratic_one_solution (c : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + c = 0) ↔ c = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l4048_404832


namespace NUMINAMATH_CALUDE_four_color_theorem_l4048_404837

/-- A type representing the four colors used for edge coloring -/
inductive EdgeColor
| one
| two
| three
| four

/-- A graph with edges colored using four colors -/
structure ColoredGraph (α : Type*) where
  edges : α → α → Option EdgeColor
  edge_coloring_property : ∀ (a b c : α), 
    edges a b ≠ none → edges b c ≠ none → 
    ∀ (d : α), edges c d ≠ none → 
    edges a b ≠ edges c d

/-- A type representing the four colors used for vertex coloring -/
inductive VertexColor
| one
| two
| three
| four

/-- A proper vertex coloring of a graph -/
def ProperVertexColoring (G : ColoredGraph α) (f : α → VertexColor) :=
  ∀ (a b : α), G.edges a b ≠ none → f a ≠ f b

theorem four_color_theorem (α : Type*) (G : ColoredGraph α) :
  ∃ (f : α → VertexColor), ProperVertexColoring G f :=
sorry

end NUMINAMATH_CALUDE_four_color_theorem_l4048_404837


namespace NUMINAMATH_CALUDE_value_of_x_l4048_404806

theorem value_of_x (x y z : ℝ) 
  (h1 : x = y / 4)
  (h2 : y = z / 3)
  (h3 : z = 90) :
  x = 7.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l4048_404806


namespace NUMINAMATH_CALUDE_chemistry_question_ratio_l4048_404807

theorem chemistry_question_ratio 
  (total_multiple_choice : ℕ) 
  (total_problem_solving : ℕ) 
  (problem_solving_fraction_written : ℚ) 
  (remaining_questions : ℕ) : 
  total_multiple_choice = 35 →
  total_problem_solving = 15 →
  problem_solving_fraction_written = 1/3 →
  remaining_questions = 31 →
  (total_multiple_choice - remaining_questions + 
   (total_problem_solving - ⌊total_problem_solving * problem_solving_fraction_written⌋)) / 
   total_multiple_choice = 9/35 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_question_ratio_l4048_404807


namespace NUMINAMATH_CALUDE_magnitude_of_z_l4048_404886

theorem magnitude_of_z (z : ℂ) (h : z * Complex.I = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l4048_404886


namespace NUMINAMATH_CALUDE_haleys_stickers_haleys_stickers_specific_l4048_404892

/-- Haley's sticker distribution problem -/
theorem haleys_stickers (num_friends : ℕ) (stickers_per_friend : ℕ) : 
  num_friends * stickers_per_friend = num_friends * stickers_per_friend := by
  sorry

/-- The specific case of Haley's problem -/
theorem haleys_stickers_specific : 9 * 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_haleys_stickers_haleys_stickers_specific_l4048_404892


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4048_404813

theorem functional_equation_solution (f : ℝ → ℝ) (h_continuous : Continuous f) 
  (h_equation : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ c : ℝ, ∀ x : ℝ, f x = Real.exp (c * x)) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4048_404813


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_ratio_l4048_404889

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem arithmetic_to_geometric_ratio 
  (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d ∧ 
  d ≠ 0 ∧
  (∀ n : ℕ, a n ≠ 0) ∧
  ((is_geometric_sequence (λ n => a n) ∧ is_geometric_sequence (λ n => a (n + 1))) ∨
   (is_geometric_sequence (λ n => a n) ∧ is_geometric_sequence (λ n => a (n + 2))) ∨
   (is_geometric_sequence (λ n => a (n + 1)) ∧ is_geometric_sequence (λ n => a (n + 2)))) →
  a 0 / d = 1 ∨ a 0 / d = -4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_ratio_l4048_404889


namespace NUMINAMATH_CALUDE_lee_cookies_with_five_cups_l4048_404822

/-- Given that Lee can make 24 cookies with 3 cups of flour,
    this function calculates how many cookies he can make with any number of cups. -/
def cookies_per_cups (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies_with_five_cups :
  cookies_per_cups 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_with_five_cups_l4048_404822


namespace NUMINAMATH_CALUDE_school_children_count_l4048_404860

theorem school_children_count :
  ∀ (total_children : ℕ) (total_bananas : ℕ),
    total_bananas = 2 * total_children →
    total_bananas = 4 * (total_children - 390) →
    total_children = 780 := by
  sorry

end NUMINAMATH_CALUDE_school_children_count_l4048_404860


namespace NUMINAMATH_CALUDE_child_admission_is_five_l4048_404881

/-- Calculates the admission price for children given the following conditions:
  * Adult admission is $8
  * Total amount paid is $201
  * Total number of tickets is 33
  * Number of children's tickets is 21
-/
def childAdmissionPrice (adultPrice totalPaid totalTickets childTickets : ℕ) : ℕ :=
  (totalPaid - adultPrice * (totalTickets - childTickets)) / childTickets

/-- Proves that the admission price for children is $5 under the given conditions -/
theorem child_admission_is_five :
  childAdmissionPrice 8 201 33 21 = 5 := by
  sorry

end NUMINAMATH_CALUDE_child_admission_is_five_l4048_404881


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_in_open_unit_interval_l4048_404814

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - 3) - x + 2 * a

theorem two_zeros_iff_a_in_open_unit_interval (a : ℝ) :
  (a > 0) →
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f a z1 = 0 ∧ f a z2 = 0 ∧ ∀ z, f a z = 0 → z = z1 ∨ z = z2) ↔
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_in_open_unit_interval_l4048_404814


namespace NUMINAMATH_CALUDE_equation_solution_l4048_404888

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x - 3 = 5 ∧ x = 112 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4048_404888


namespace NUMINAMATH_CALUDE_concert_problem_l4048_404834

/-- Represents the number of songs sung by each friend -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  laura : ℕ

/-- Conditions for the concert problem -/
def ConcertConditions (sc : SongCount) : Prop :=
  sc.hanna = 9 ∧
  sc.mary = 3 ∧
  sc.alina > sc.mary ∧ sc.alina < sc.hanna ∧
  sc.tina > sc.mary ∧ sc.tina < sc.hanna ∧
  sc.laura > sc.mary ∧ sc.laura < sc.hanna

/-- The total number of songs performed -/
def TotalSongs (sc : SongCount) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna + sc.laura) / 4

/-- Theorem stating that under the given conditions, the total number of songs is 9 -/
theorem concert_problem (sc : SongCount) :
  ConcertConditions sc → TotalSongs sc = 9 := by
  sorry


end NUMINAMATH_CALUDE_concert_problem_l4048_404834


namespace NUMINAMATH_CALUDE_quadratic_triple_root_l4048_404818

/-- For a quadratic equation ax^2 + bx + c = 0, one root is triple the other 
    if and only if 3b^2 = 16ac -/
theorem quadratic_triple_root (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) ↔ 
  3 * b^2 = 16 * a * c :=
sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_l4048_404818


namespace NUMINAMATH_CALUDE_modulus_of_z_l4048_404883

theorem modulus_of_z : Complex.abs ((1 - Complex.I) * Complex.I) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4048_404883
