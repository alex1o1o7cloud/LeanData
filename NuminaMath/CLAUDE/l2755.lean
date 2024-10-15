import Mathlib

namespace NUMINAMATH_CALUDE_dinner_cakes_count_l2755_275591

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := total_cakes - lunch_cakes - yesterday_cakes

theorem dinner_cakes_count : dinner_cakes = 6 := by sorry

end NUMINAMATH_CALUDE_dinner_cakes_count_l2755_275591


namespace NUMINAMATH_CALUDE_seventeen_factorial_minus_fifteen_factorial_prime_divisors_l2755_275513

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of prime divisors of a natural number -/
def numPrimeDivisors (n : ℕ) : ℕ := (Nat.factors n).length

/-- The main theorem -/
theorem seventeen_factorial_minus_fifteen_factorial_prime_divisors :
  numPrimeDivisors (factorial 17 - factorial 15) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_factorial_minus_fifteen_factorial_prime_divisors_l2755_275513


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2755_275534

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2755_275534


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_l2755_275581

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials :
  units_digit (sum_factorials 99) = units_digit (sum_factorials 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_l2755_275581


namespace NUMINAMATH_CALUDE_emily_elephant_four_hops_l2755_275524

/-- The distance covered in a single hop, given the remaining distance to the target -/
def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

/-- The remaining distance to the target after a hop -/
def remaining_after_hop (remaining : ℚ) : ℚ := remaining - hop_distance remaining

/-- The total distance covered after n hops -/
def total_distance (n : ℕ) : ℚ :=
  let rec aux (k : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if k = 0 then acc
    else aux (k - 1) (remaining_after_hop remaining) (acc + hop_distance remaining)
  aux n 1 0

theorem emily_elephant_four_hops :
  total_distance 4 = 175 / 256 := by sorry

end NUMINAMATH_CALUDE_emily_elephant_four_hops_l2755_275524


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l2755_275551

/-- Given Andrew's purchase of grapes and mangoes, prove the rate per kg for mangoes -/
theorem mango_rate_calculation (grapes_kg : ℕ) (grapes_rate : ℕ) (mangoes_kg : ℕ) (total_paid : ℕ) :
  grapes_kg = 11 →
  grapes_rate = 98 →
  mangoes_kg = 7 →
  total_paid = 1428 →
  (total_paid - grapes_kg * grapes_rate) / mangoes_kg = 50 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l2755_275551


namespace NUMINAMATH_CALUDE_y_value_proof_l2755_275598

theorem y_value_proof (y : ℝ) (h : 9 / y^2 = 3 * y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2755_275598


namespace NUMINAMATH_CALUDE_boot_purchase_theorem_l2755_275559

def boot_purchase_problem (initial_amount hand_sanitizer_discount toilet_paper_cost : ℚ) : ℚ :=
  let hand_sanitizer_cost : ℚ := 6
  let large_ham_cost : ℚ := 2 * toilet_paper_cost
  let cheese_cost : ℚ := hand_sanitizer_cost / 2
  let total_spent : ℚ := toilet_paper_cost + hand_sanitizer_cost + large_ham_cost + cheese_cost
  let remaining : ℚ := initial_amount - total_spent
  let savings : ℚ := remaining * (1/5)
  let spendable : ℚ := remaining - savings
  let per_twin : ℚ := spendable / 2
  let boot_cost : ℚ := per_twin * 4
  let total_boot_cost : ℚ := boot_cost * 2
  (total_boot_cost - spendable) / 2

theorem boot_purchase_theorem :
  boot_purchase_problem 100 (1/4) 12 = 66 := by sorry

end NUMINAMATH_CALUDE_boot_purchase_theorem_l2755_275559


namespace NUMINAMATH_CALUDE_product_of_logarithms_l2755_275545

theorem product_of_logarithms (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  (Real.log d / Real.log c = 2) → (d - c = 630) → (c + d = 1260) := by
  sorry

end NUMINAMATH_CALUDE_product_of_logarithms_l2755_275545


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2755_275512

theorem power_of_three_mod_five : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2755_275512


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2755_275533

theorem cone_lateral_surface_area 
  (r : Real) 
  (l : Real) 
  (h_r : r = Real.sqrt 2) 
  (h_l : l = 3 * Real.sqrt 2) : 
  r * l * Real.pi = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2755_275533


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2755_275590

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x ≠ 0 ∧ y ≠ 0) ∧ (x^2 / y - y^2 / x = 3 * (2 + 1 / (x * y))) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2755_275590


namespace NUMINAMATH_CALUDE_square_sum_constant_l2755_275553

theorem square_sum_constant (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_constant_l2755_275553


namespace NUMINAMATH_CALUDE_carly_lollipop_ratio_l2755_275589

/-- Given a total number of lollipops and the number of grape lollipops,
    calculate the ratio of cherry lollipops to the total number of lollipops. -/
def lollipop_ratio (total : ℕ) (grape : ℕ) : ℚ :=
  let other_flavors := grape * 3
  let cherry := total - other_flavors
  (cherry : ℚ) / total

/-- Theorem stating that given the conditions in the problem,
    the ratio of cherry lollipops to the total is 1/2. -/
theorem carly_lollipop_ratio :
  lollipop_ratio 42 7 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carly_lollipop_ratio_l2755_275589


namespace NUMINAMATH_CALUDE_odd_function_property_l2755_275548

def f (x : ℝ) (g : ℝ → ℝ) : ℝ := g x - 8

theorem odd_function_property (g : ℝ → ℝ) (m : ℝ) :
  (∀ x, g (-x) = -g x) →
  f (-m) g = 10 →
  f m g = -26 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l2755_275548


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l2755_275566

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 4*a

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x, |f a x| ≤ 2) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l2755_275566


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l2755_275563

variable {V : Type*} [AddCommGroup V]

theorem vector_expression_simplification
  (CE AC DE AD : V) :
  CE + AC - DE - AD = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l2755_275563


namespace NUMINAMATH_CALUDE_A_intersect_B_l2755_275584

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x < 3}

theorem A_intersect_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2755_275584


namespace NUMINAMATH_CALUDE_zoe_winter_clothing_boxes_l2755_275542

theorem zoe_winter_clothing_boxes :
  let items_per_box := 4 + 6  -- 4 scarves and 6 mittens per box
  let total_items := 80       -- total pieces of winter clothing
  total_items / items_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_zoe_winter_clothing_boxes_l2755_275542


namespace NUMINAMATH_CALUDE_little_twelve_games_l2755_275580

/-- Represents a basketball conference with divisions and teams. -/
structure BasketballConference where
  num_divisions : ℕ
  teams_per_division : ℕ
  intra_division_games : ℕ
  inter_division_games : ℕ

/-- Calculates the total number of scheduled games in the conference. -/
def total_games (conf : BasketballConference) : ℕ :=
  let intra_games := conf.num_divisions * (conf.teams_per_division.choose 2) * conf.intra_division_games
  let inter_games := (conf.num_divisions * (conf.num_divisions - 1) / 2) * (conf.teams_per_division ^ 2) * conf.inter_division_games
  intra_games + inter_games

/-- Theorem stating that the Little Twelve Basketball Conference has 102 scheduled games. -/
theorem little_twelve_games :
  let conf : BasketballConference := {
    num_divisions := 3,
    teams_per_division := 4,
    intra_division_games := 3,
    inter_division_games := 2
  }
  total_games conf = 102 := by
  sorry


end NUMINAMATH_CALUDE_little_twelve_games_l2755_275580


namespace NUMINAMATH_CALUDE_modulus_of_z_l2755_275555

-- Define the complex number z
def z : ℂ := Complex.I * (1 - Complex.I)

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2755_275555


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2755_275575

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + (1/4 : ℝ) ≤ 0) ↔ (0 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2755_275575


namespace NUMINAMATH_CALUDE_number_of_boys_l2755_275508

theorem number_of_boys (total_pupils : ℕ) (number_of_girls : ℕ) 
  (h1 : total_pupils = 485) 
  (h2 : number_of_girls = 232) : 
  total_pupils - number_of_girls = 253 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l2755_275508


namespace NUMINAMATH_CALUDE_square_area_ratio_l2755_275567

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / ((s * Real.sqrt 5)^2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2755_275567


namespace NUMINAMATH_CALUDE_min_value_a1a3_l2755_275540

theorem min_value_a1a3 (a₁ a₂ a₃ : ℝ) (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (h_a₂ : a₂ = 6) 
  (h_arithmetic : ∃ d : ℝ, 1 / (a₃ + 3) - 1 / (a₂ + 2) = 1 / (a₂ + 2) - 1 / (a₁ + 1)) :
  a₁ * a₃ ≥ 16 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a1a3_l2755_275540


namespace NUMINAMATH_CALUDE_g_minimum_value_l2755_275503

noncomputable def g (x : ℝ) : ℝ := x + x / (x^2 + 2) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_g_minimum_value_l2755_275503


namespace NUMINAMATH_CALUDE_tuesday_is_valid_start_day_l2755_275516

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDays d m)

def isValidRedemptionSchedule (startDay : DayOfWeek) : Prop :=
  ∀ i : Fin 7, advanceDays startDay (i.val * 12) ≠ DayOfWeek.Saturday

theorem tuesday_is_valid_start_day :
  isValidRedemptionSchedule DayOfWeek.Tuesday ∧
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Tuesday → ¬ isValidRedemptionSchedule d :=
sorry

end NUMINAMATH_CALUDE_tuesday_is_valid_start_day_l2755_275516


namespace NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l2755_275546

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem cubic_inches_in_cubic_foot : 
  1 * (inches_per_foot ^ 3) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l2755_275546


namespace NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l2755_275515

theorem sum_of_squares_with_given_means (a b : ℝ) :
  (a + b) / 2 = 8 → Real.sqrt (a * b) = 2 * Real.sqrt 5 → a^2 + b^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l2755_275515


namespace NUMINAMATH_CALUDE_cube_surface_area_l2755_275576

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1331 →
  volume = side ^ 3 →
  surface_area = 6 * side ^ 2 →
  surface_area = 726 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2755_275576


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2755_275535

theorem deal_or_no_deal_probability (total_boxes : Nat) (high_value_boxes : Nat) 
  (h1 : total_boxes = 26)
  (h2 : high_value_boxes = 7) :
  total_boxes - (high_value_boxes + high_value_boxes) = 12 := by
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2755_275535


namespace NUMINAMATH_CALUDE_square_of_98_l2755_275523

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_square_of_98_l2755_275523


namespace NUMINAMATH_CALUDE_power_of_product_l2755_275541

theorem power_of_product (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2755_275541


namespace NUMINAMATH_CALUDE_cost_of_three_rides_is_171_l2755_275506

/-- The cost of tickets for three rides at a fair -/
def cost_of_rides (ferris_wheel_tickets : ℕ) (roller_coaster_tickets : ℕ) (bumper_cars_tickets : ℕ) (cost_per_ticket : ℕ) : ℕ :=
  (ferris_wheel_tickets + roller_coaster_tickets + bumper_cars_tickets) * cost_per_ticket

/-- Theorem stating that the cost of the three rides is $171 -/
theorem cost_of_three_rides_is_171 :
  cost_of_rides 6 8 5 9 = 171 := by
  sorry

#eval cost_of_rides 6 8 5 9

end NUMINAMATH_CALUDE_cost_of_three_rides_is_171_l2755_275506


namespace NUMINAMATH_CALUDE_product_pricing_l2755_275568

theorem product_pricing (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 0.9 * a > b) : 
  0.9 * a - b = 0.2 * b := by
sorry

end NUMINAMATH_CALUDE_product_pricing_l2755_275568


namespace NUMINAMATH_CALUDE_unique_solution_implies_equal_or_opposite_l2755_275517

theorem unique_solution_implies_equal_or_opposite (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x : ℝ, a * (x - a)^2 + b * (x - b)^2 = 0) → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_equal_or_opposite_l2755_275517


namespace NUMINAMATH_CALUDE_sam_average_letters_per_day_l2755_275574

/-- Given that Sam wrote 7 letters on Tuesday and 3 letters on Wednesday,
    prove that the average number of letters he wrote per day is 5. -/
theorem sam_average_letters_per_day :
  let tuesday_letters : ℕ := 7
  let wednesday_letters : ℕ := 3
  let total_days : ℕ := 2
  let total_letters : ℕ := tuesday_letters + wednesday_letters
  let average_letters : ℚ := total_letters / total_days
  average_letters = 5 := by
sorry

end NUMINAMATH_CALUDE_sam_average_letters_per_day_l2755_275574


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l2755_275557

theorem car_fuel_efficiency (H : ℝ) : 
  (H > 0) →
  (4 / H + 4 / 20 = 8 / H * 1.3499999999999999) →
  H = 34 := by
sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l2755_275557


namespace NUMINAMATH_CALUDE_second_triangle_side_length_l2755_275573

/-- Given a sequence of equilateral triangles where each triangle is formed by joining
    the midpoints of the sides of the previous triangle, if the first triangle has sides
    of 80 cm and the sum of all triangle perimeters is 480 cm, then the side length of
    the second triangle is 40 cm. -/
theorem second_triangle_side_length
  (first_triangle_side : ℝ)
  (total_perimeter : ℝ)
  (h1 : first_triangle_side = 80)
  (h2 : total_perimeter = 480)
  (h3 : total_perimeter = (3 * first_triangle_side) / (1 - 1/2)) :
  first_triangle_side / 2 = 40 :=
sorry

end NUMINAMATH_CALUDE_second_triangle_side_length_l2755_275573


namespace NUMINAMATH_CALUDE_average_pen_price_l2755_275570

/-- Represents the types of pens --/
inductive PenType
  | A
  | B
  | C
  | D

/-- Given data about pen sales --/
def pen_data : List (PenType × Nat × Nat) :=
  [(PenType.A, 5, 5), (PenType.B, 3, 8), (PenType.C, 2, 27), (PenType.D, 1, 10)]

/-- Total number of pens sold --/
def total_pens : Nat := 50

/-- Theorem stating that the average unit price of pens sold is 2.26元 --/
theorem average_pen_price :
  let total_revenue := (pen_data.map (fun (_, price, quantity) => price * quantity)).sum
  let average_price := (total_revenue : ℚ) / total_pens
  average_price = 226 / 100 := by
  sorry

#check average_pen_price

end NUMINAMATH_CALUDE_average_pen_price_l2755_275570


namespace NUMINAMATH_CALUDE_certain_number_proof_l2755_275560

theorem certain_number_proof (N : ℕ) (h1 : N < 81) 
  (h2 : ∀ k : ℕ, k ∈ Finset.range 15 → N + k + 1 < 81) 
  (h3 : N + 16 ≥ 81) : N = 65 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2755_275560


namespace NUMINAMATH_CALUDE_negation_of_exists_exponential_l2755_275579

theorem negation_of_exists_exponential (x : ℝ) :
  (¬ ∃ x₀ : ℝ, (2 : ℝ) ^ x₀ ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ) ^ x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_exponential_l2755_275579


namespace NUMINAMATH_CALUDE_cosine_transformation_symmetry_l2755_275561

open Real

theorem cosine_transformation_symmetry (ω : ℝ) :
  ω > 0 →
  (∀ x, ∃ y, cos (ω * (x - π / 12)) = y) →
  (∀ x, cos (ω * ((π / 4 + (π / 4 - x)) - π / 12)) = cos (ω * (x - π / 12))) →
  ω ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_cosine_transformation_symmetry_l2755_275561


namespace NUMINAMATH_CALUDE_least_sum_m_n_l2755_275544

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 330 = 1) ∧ 
  (∃ (k : ℕ), m.val^m.val = k * n.val^n.val) ∧ 
  (∀ (j : ℕ), m.val ≠ j * n.val) ∧
  (m.val + n.val = 247) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m'.val + n'.val) 330 = 1) → 
    (∃ (k : ℕ), m'.val^m'.val = k * n'.val^n'.val) → 
    (∀ (j : ℕ), m'.val ≠ j * n'.val) → 
    (m'.val + n'.val ≥ 247)) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l2755_275544


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2755_275525

/-- The foci coordinates of the hyperbola x^2/4 - y^2 = 1 are (±√5, 0) -/
theorem hyperbola_foci_coordinates :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 4 - y^2 = 1}
  ∃ (c : ℝ), c^2 = 5 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ hyperbola → 
      ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2755_275525


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l2755_275509

/-- A square with side length 4 containing two right triangles whose hypotenuses
    are opposite sides of the square. -/
structure SquareWithTriangles where
  side_length : ℝ
  hypotenuse_length : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  h_side_length : side_length = 4
  h_hypotenuse : hypotenuse_length = side_length
  h_right_triangle : rectangle_width ^ 2 + rectangle_height ^ 2 = hypotenuse_length ^ 2
  h_rectangle_dim : rectangle_width + rectangle_height = side_length

/-- The area of the rectangle formed by the intersection of the triangles is 8. -/
theorem rectangle_area_is_eight (s : SquareWithTriangles) : 
  s.rectangle_width * s.rectangle_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l2755_275509


namespace NUMINAMATH_CALUDE_hex_20F_to_decimal_l2755_275538

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a HexDigit to its decimal value --/
def hexToDecimal (d : HexDigit) : ℕ :=
  match d with
  | HexDigit.D0 => 0 | HexDigit.D1 => 1 | HexDigit.D2 => 2 | HexDigit.D3 => 3
  | HexDigit.D4 => 4 | HexDigit.D5 => 5 | HexDigit.D6 => 6 | HexDigit.D7 => 7
  | HexDigit.D8 => 8 | HexDigit.D9 => 9 | HexDigit.A => 10 | HexDigit.B => 11
  | HexDigit.C => 12 | HexDigit.D => 13 | HexDigit.E => 14 | HexDigit.F => 15

/-- Converts a list of HexDigits to its decimal value --/
def hexListToDecimal (digits : List HexDigit) : ℤ :=
  digits.enum.foldl (fun acc (i, d) => acc + (hexToDecimal d : ℤ) * 16^(digits.length - 1 - i)) 0

/-- The hexadecimal number -20F --/
def hex20F : List HexDigit := [HexDigit.D2, HexDigit.D0, HexDigit.F]

theorem hex_20F_to_decimal :
  -hexListToDecimal hex20F = -527 := by sorry

end NUMINAMATH_CALUDE_hex_20F_to_decimal_l2755_275538


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2755_275520

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem ninth_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_third : a 3 = 5/11) 
  (h_fifteenth : a 15 = 7/8) : 
  a 9 = 117/176 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2755_275520


namespace NUMINAMATH_CALUDE_check_mistake_l2755_275502

theorem check_mistake (x y : ℕ) : 
  (100 * y + x) - (100 * x + y) = 1368 → y = x + 14 := by
  sorry

end NUMINAMATH_CALUDE_check_mistake_l2755_275502


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2755_275532

theorem sqrt_inequality (a : ℝ) (h : a > 6) :
  Real.sqrt (a - 3) - Real.sqrt (a - 4) < Real.sqrt (a - 5) - Real.sqrt (a - 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2755_275532


namespace NUMINAMATH_CALUDE_train_length_l2755_275594

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 42 →
  crossing_time = 60 →
  bridge_length = 200 →
  ∃ (train_length : ℝ), abs (train_length - 500.2) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l2755_275594


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l2755_275550

/-- The number of positive divisors of a positive integer -/
def numDivisors (n : ℕ+) : ℕ := sorry

/-- Returns true if n is the smallest positive integer with exactly k positive divisors -/
def isSmallestWithDivisors (n k : ℕ+) : Prop :=
  numDivisors n = k ∧ ∀ m : ℕ+, m < n → numDivisors m ≠ k

theorem smallest_with_18_divisors :
  isSmallestWithDivisors 288 18 := by sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l2755_275550


namespace NUMINAMATH_CALUDE_lcm_gcd_product_40_100_l2755_275562

theorem lcm_gcd_product_40_100 : Nat.lcm 40 100 * Nat.gcd 40 100 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_40_100_l2755_275562


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l2755_275514

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℤ, x^2 + a*x + 1 = b ∧ x ≠ y) → 
  b ≠ 1 → 
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ m * n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l2755_275514


namespace NUMINAMATH_CALUDE_smallest_symmetric_set_l2755_275595

-- Define a point in the xy-plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the set T
def T : Set Point := sorry

-- Define symmetry conditions
def symmetricAboutOrigin (p : Point) : Prop :=
  Point.mk (-p.x) (-p.y) ∈ T

def symmetricAboutXAxis (p : Point) : Prop :=
  Point.mk p.x (-p.y) ∈ T

def symmetricAboutYAxis (p : Point) : Prop :=
  Point.mk (-p.x) p.y ∈ T

def symmetricAboutNegativeDiagonal (p : Point) : Prop :=
  Point.mk (-p.y) (-p.x) ∈ T

-- State the theorem
theorem smallest_symmetric_set :
  (∀ p ∈ T, symmetricAboutOrigin p ∧ 
            symmetricAboutXAxis p ∧ 
            symmetricAboutYAxis p ∧ 
            symmetricAboutNegativeDiagonal p) →
  Point.mk 1 4 ∈ T →
  (∃ (s : Finset Point), s.card = 8 ∧ ↑s = T) ∧
  ¬∃ (s : Finset Point), s.card < 8 ∧ ↑s = T :=
by sorry

end NUMINAMATH_CALUDE_smallest_symmetric_set_l2755_275595


namespace NUMINAMATH_CALUDE_factor_expression_l2755_275588

theorem factor_expression (b : ℝ) : 52 * b^2 + 208 * b = 52 * b * (b + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2755_275588


namespace NUMINAMATH_CALUDE_equation_solution_l2755_275586

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3 ↔ x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2755_275586


namespace NUMINAMATH_CALUDE_is_point_of_tangency_l2755_275583

/-- The point of tangency between two circles -/
def point_of_tangency : ℝ × ℝ := (2.5, 5)

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 10*y + 17 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 49 = 0

/-- Theorem stating that point_of_tangency is the point of tangency between the two circles -/
theorem is_point_of_tangency :
  let (x, y) := point_of_tangency
  circle1 x y ∧ circle2 x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(circle1 x' y' ∧ circle2 x' y') :=
by sorry

end NUMINAMATH_CALUDE_is_point_of_tangency_l2755_275583


namespace NUMINAMATH_CALUDE_two_digit_number_with_divisibility_properties_l2755_275537

theorem two_digit_number_with_divisibility_properties : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n + 3) % 3 = 0 ∧ 
  (n + 7) % 7 = 0 ∧ 
  (n - 4) % 4 = 0 ∧
  n = 84 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_with_divisibility_properties_l2755_275537


namespace NUMINAMATH_CALUDE_three_heads_in_four_tosses_l2755_275526

/-- The probability of getting exactly k successes in n trials with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 0.5 of landing heads -/
def fairCoinProbability : ℝ := 0.5

theorem three_heads_in_four_tosses :
  binomialProbability 4 3 fairCoinProbability = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_four_tosses_l2755_275526


namespace NUMINAMATH_CALUDE_n_even_factors_l2755_275504

def n : ℕ := 2^3 * 3^2 * 5^1 * 7^3

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem n_even_factors :
  num_even_factors n = 72 := by sorry

end NUMINAMATH_CALUDE_n_even_factors_l2755_275504


namespace NUMINAMATH_CALUDE_abs_difference_symmetry_l2755_275593

theorem abs_difference_symmetry (a b : ℚ) : |a - b| = |b - a| := by sorry

end NUMINAMATH_CALUDE_abs_difference_symmetry_l2755_275593


namespace NUMINAMATH_CALUDE_rain_probability_both_days_l2755_275577

theorem rain_probability_both_days (prob_monday : ℝ) (prob_tuesday : ℝ) 
  (h1 : prob_monday = 0.4)
  (h2 : prob_tuesday = 0.3)
  (h3 : 0 ≤ prob_monday ∧ prob_monday ≤ 1)
  (h4 : 0 ≤ prob_tuesday ∧ prob_tuesday ≤ 1) :
  prob_monday * prob_tuesday = 0.12 :=
by
  sorry

end NUMINAMATH_CALUDE_rain_probability_both_days_l2755_275577


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2755_275521

/-- Calculate the cost of plastering a tank's walls and bottom -/
theorem tank_plastering_cost 
  (length width depth : ℝ)
  (cost_per_sqm_paise : ℝ)
  (h_length : length = 25)
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost : cost_per_sqm_paise = 75) :
  let surface_area := 2 * (length * depth + width * depth) + length * width
  let cost_rupees := surface_area * (cost_per_sqm_paise / 100)
  cost_rupees = 558 := by
sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l2755_275521


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2755_275519

/-- A quadratic function f(x) = x^2 + bx + c with real constants b and c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_roots_properties (b c x₁ x₂ : ℝ) 
  (hroot₁ : f b c x₁ = x₁)
  (hroot₂ : f b c x₂ = x₂)
  (hx₁_pos : x₁ > 0)
  (hx₂_x₁ : x₂ - x₁ > 1) :
  (b^2 > 2*(b + 2*c)) ∧ 
  (∀ t : ℝ, 0 < t → t < x₁ → f b c t > x₁) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2755_275519


namespace NUMINAMATH_CALUDE_probability_perfect_square_three_digit_l2755_275547

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A perfect square is a natural number that is the square of an integer. -/
def PerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The count of three-digit numbers that are perfect squares. -/
def CountPerfectSquareThreeDigit : ℕ := 22

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being a perfect square is 11/450. -/
theorem probability_perfect_square_three_digit :
  (CountPerfectSquareThreeDigit : ℚ) / (TotalThreeDigitNumbers : ℚ) = 11 / 450 := by
  sorry

end NUMINAMATH_CALUDE_probability_perfect_square_three_digit_l2755_275547


namespace NUMINAMATH_CALUDE_weeks_to_save_shirt_l2755_275569

/-- 
Given:
- shirt_cost: The cost of the shirt in dollars
- initial_savings: The amount Macey has already saved in dollars
- weekly_savings: The amount Macey saves per week in dollars

Prove that the number of weeks needed to save the remaining amount is 3.
-/
theorem weeks_to_save_shirt (shirt_cost initial_savings weekly_savings : ℚ) 
  (h1 : shirt_cost = 3)
  (h2 : initial_savings = 3/2)
  (h3 : weekly_savings = 1/2) :
  (shirt_cost - initial_savings) / weekly_savings = 3 := by
sorry

end NUMINAMATH_CALUDE_weeks_to_save_shirt_l2755_275569


namespace NUMINAMATH_CALUDE_tangent_line_m_values_l2755_275597

/-- The equation of a line that may be tangent to a circle -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (m : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation x y m ∧ circle_equation x y

/-- Theorem stating the possible values of m when the line is tangent to the circle -/
theorem tangent_line_m_values :
  ∀ m : ℝ, is_tangent m → m = -3 ∨ m = -13 := by sorry

end NUMINAMATH_CALUDE_tangent_line_m_values_l2755_275597


namespace NUMINAMATH_CALUDE_equivalent_representations_l2755_275572

theorem equivalent_representations (n : ℕ+) :
  (∃ (x y : ℕ+), n = 3 * x^2 + y^2) ↔ (∃ (u v : ℕ+), n = u^2 + u * v + v^2) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_representations_l2755_275572


namespace NUMINAMATH_CALUDE_random_variable_distribution_invariance_l2755_275510

-- Define a type for random variables
variable (Ω : Type) [MeasurableSpace Ω]
def RandomVariable (α : Type) [MeasurableSpace α] := Ω → α

-- Define a type for distribution functions
def DistributionFunction (α : Type) [MeasurableSpace α] := α → ℝ

-- State the theorem
theorem random_variable_distribution_invariance
  (ξ : RandomVariable Ω ℝ)
  (h_non_degenerate : ∀ (c : ℝ), ¬(∀ (ω : Ω), ξ ω = c))
  (a : ℝ)
  (b : ℝ)
  (h_a_pos : a > 0)
  (h_distribution_equal : ∀ (F : DistributionFunction ℝ),
    (∀ (x : ℝ), F x = F ((x - b) / a))) :
  a = 1 ∧ b = 0 :=
sorry

end NUMINAMATH_CALUDE_random_variable_distribution_invariance_l2755_275510


namespace NUMINAMATH_CALUDE_parabola_c_value_l2755_275543

/-- A parabola passing through two specific points has a unique c-value -/
theorem parabola_c_value (b : ℝ) :
  ∃! c : ℝ, (2^2 + 2*b + c = 20) ∧ ((-2)^2 + (-2)*b + c = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2755_275543


namespace NUMINAMATH_CALUDE_cows_for_96_days_l2755_275599

/-- Represents the number of cows that can eat all the grass in a given number of days -/
structure GrazingScenario where
  cows : ℕ
  days : ℕ

/-- Represents the meadow with growing grass -/
structure Meadow where
  scenario1 : GrazingScenario
  scenario2 : GrazingScenario
  growth_rate : ℚ

/-- Calculate the number of cows that can eat all the grass in 96 days -/
def calculate_cows (m : Meadow) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem cows_for_96_days (m : Meadow) : 
  m.scenario1 = ⟨70, 24⟩ → 
  m.scenario2 = ⟨30, 60⟩ → 
  calculate_cows m = 20 := by
  sorry

end NUMINAMATH_CALUDE_cows_for_96_days_l2755_275599


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2755_275530

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a - 2 * i) * i^2013 = b - i →
  a^2 + b^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2755_275530


namespace NUMINAMATH_CALUDE_probability_one_instrument_l2755_275578

theorem probability_one_instrument (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 1/5 →
  two_or_more = 128 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 1/25 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_instrument_l2755_275578


namespace NUMINAMATH_CALUDE_calculate_gladys_speed_l2755_275528

def team_size : Nat := 5

def rudy_speed : Nat := 64
def joyce_speed : Nat := 76
def lisa_speed : Nat := 80
def mike_speed : Nat := 89

def team_average : Nat := 80

def gladys_speed : Nat := 91

theorem calculate_gladys_speed :
  team_size * team_average - (rudy_speed + joyce_speed + lisa_speed + mike_speed) = gladys_speed := by
  sorry

end NUMINAMATH_CALUDE_calculate_gladys_speed_l2755_275528


namespace NUMINAMATH_CALUDE_analytical_method_seeks_sufficient_conditions_l2755_275527

/-- The analytical method for proving inequalities -/
structure AnalyticalMethod where
  /-- The method proceeds from effect to cause -/
  effect_to_cause : Bool

/-- A condition in the context of proving inequalities -/
inductive Condition
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- The reasoning process sought by the analytical method -/
def reasoning_process (method : AnalyticalMethod) : Condition :=
  Condition.Sufficient

/-- Theorem stating that the analytical method seeks sufficient conditions -/
theorem analytical_method_seeks_sufficient_conditions (method : AnalyticalMethod) :
  reasoning_process method = Condition.Sufficient := by
  sorry

end NUMINAMATH_CALUDE_analytical_method_seeks_sufficient_conditions_l2755_275527


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l2755_275522

/-- The outer perimeter of a square fence with evenly spaced posts -/
theorem square_fence_perimeter
  (num_posts : ℕ)
  (post_width_inches : ℕ)
  (gap_between_posts_feet : ℕ)
  (h1 : num_posts = 16)
  (h2 : post_width_inches = 6)
  (h3 : gap_between_posts_feet = 4) :
  (4 * (↑num_posts / 4 * (↑post_width_inches / 12 + ↑gap_between_posts_feet) - ↑gap_between_posts_feet)) = 56 :=
by sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l2755_275522


namespace NUMINAMATH_CALUDE_divide_multiply_result_l2755_275511

theorem divide_multiply_result : (3 / 4) * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_result_l2755_275511


namespace NUMINAMATH_CALUDE_john_paintball_cost_l2755_275507

/-- John's monthly expenditure on paintballs -/
def monthly_paintball_cost (plays_per_month : ℕ) (boxes_per_play : ℕ) (cost_per_box : ℕ) : ℕ :=
  plays_per_month * boxes_per_play * cost_per_box

/-- Theorem: John spends $225 a month on paintballs -/
theorem john_paintball_cost :
  monthly_paintball_cost 3 3 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_paintball_cost_l2755_275507


namespace NUMINAMATH_CALUDE_current_population_calculation_l2755_275500

def initial_population : ℕ := 4399
def bombardment_percentage : ℚ := 1/10
def fear_percentage : ℚ := 1/5

theorem current_population_calculation :
  let remaining_after_bombardment := initial_population - ⌊initial_population * bombardment_percentage⌋
  let current_population := remaining_after_bombardment - ⌊remaining_after_bombardment * fear_percentage⌋
  current_population = 3167 := by sorry

end NUMINAMATH_CALUDE_current_population_calculation_l2755_275500


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2755_275558

/-- Represents a parabola in 2D space -/
structure Parabola where
  equation : ℝ → ℝ

/-- Two parabolas are symmetric about the origin -/
def symmetric_about_origin (p1 p2 : Parabola) : Prop :=
  ∀ x y : ℝ, p1.equation x = y ↔ p2.equation (-x) = -y

theorem parabola_symmetry (C1 C2 : Parabola) 
  (h1 : C1.equation = fun x ↦ (x - 2)^2 + 3)
  (h2 : symmetric_about_origin C1 C2) :
  C2.equation = fun x ↦ -(x + 2)^2 - 3 := by
  sorry


end NUMINAMATH_CALUDE_parabola_symmetry_l2755_275558


namespace NUMINAMATH_CALUDE_min_three_digit_quotient_l2755_275536

def three_digit_quotient (a b : ℕ) : ℚ :=
  (100 * a + 10 * b + 1) / (a + b + 1)

theorem min_three_digit_quotient :
  ∀ a b : ℕ, 2 ≤ a → a ≤ 9 → 2 ≤ b → b ≤ 9 → a ≠ b →
  three_digit_quotient a b ≥ 24.25 ∧
  ∃ a₀ b₀ : ℕ, 2 ≤ a₀ ∧ a₀ ≤ 9 ∧ 2 ≤ b₀ ∧ b₀ ≤ 9 ∧ a₀ ≠ b₀ ∧
  three_digit_quotient a₀ b₀ = 24.25 :=
sorry

end NUMINAMATH_CALUDE_min_three_digit_quotient_l2755_275536


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l2755_275549

/-- A trapezoid with integer side lengths and a right angle -/
structure RightTrapezoid where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AB_parallel_CD : AB = CD
  right_angle_BCD : BC^2 + CD^2 = BD^2

/-- The diagonal length of the specific trapezoid -/
def diagonal_length (t : RightTrapezoid) : ℕ := 20

/-- Theorem: The diagonal length of the specific trapezoid is 20 -/
theorem specific_trapezoid_diagonal : 
  ∀ (t : RightTrapezoid), 
  t.AB = 7 → t.BC = 19 → t.CD = 7 → t.DA = 11 → 
  diagonal_length t = 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l2755_275549


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2755_275582

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2755_275582


namespace NUMINAMATH_CALUDE_number_problem_l2755_275564

theorem number_problem (x : ℚ) : (3 / 4) * x = x - 19 → x = 76 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2755_275564


namespace NUMINAMATH_CALUDE_max_value_theorem_l2755_275529

theorem max_value_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 6 + 5 * y * z ≤ Real.sqrt 6 * (2 * Real.sqrt (375/481)) + 5 * (2 * Real.sqrt (106/481)) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2755_275529


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l2755_275587

def has_common_factor_greater_than_one (a b : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k ∣ a ∧ k ∣ b

theorem smallest_n_with_common_factor : 
  (∀ n : ℕ, n > 0 ∧ n < 14 → ¬(has_common_factor_greater_than_one (8*n + 3) (10*n - 4))) ∧
  (has_common_factor_greater_than_one (8*14 + 3) (10*14 - 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l2755_275587


namespace NUMINAMATH_CALUDE_negate_neg_sum_l2755_275556

theorem negate_neg_sum (a b : ℝ) : -(-a - b) = a + b := by
  sorry

end NUMINAMATH_CALUDE_negate_neg_sum_l2755_275556


namespace NUMINAMATH_CALUDE_seokjin_position_l2755_275596

/-- Given the positions of Jungkook, Yoojeong, and Seokjin on the stairs,
    prove that Seokjin is 3 steps higher than Jungkook. -/
theorem seokjin_position (jungkook yoojeong seokjin : ℕ) 
  (h1 : jungkook = 19)
  (h2 : yoojeong = jungkook + 8)
  (h3 : seokjin = yoojeong - 5) :
  seokjin - jungkook = 3 := by
sorry

end NUMINAMATH_CALUDE_seokjin_position_l2755_275596


namespace NUMINAMATH_CALUDE_product_of_powers_equals_thousand_l2755_275592

theorem product_of_powers_equals_thousand :
  (10 ^ 0.25) * (10 ^ 0.25) * (10 ^ 0.5) * (10 ^ 0.5) * (10 ^ 0.75) * (10 ^ 0.75) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_thousand_l2755_275592


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2755_275554

theorem rectangle_area_diagonal (length width diagonal : ℝ) (h_ratio : length / width = 5 / 2) 
  (h_diagonal : diagonal^2 = length^2 + width^2) :
  length * width = (10 / 29) * diagonal^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2755_275554


namespace NUMINAMATH_CALUDE_museum_ticket_problem_l2755_275571

/-- Represents the cost calculation for museum tickets with discounts -/
structure TicketCost where
  basePrice : ℕ
  option1Discount : ℚ
  option2Discount : ℚ
  freeTickets : ℕ

/-- Calculates the cost for Option 1 -/
def option1Cost (tc : TicketCost) (students : ℕ) : ℚ :=
  tc.basePrice * (1 - tc.option1Discount) * students

/-- Calculates the cost for Option 2 -/
def option2Cost (tc : TicketCost) (students : ℕ) : ℚ :=
  tc.basePrice * (1 - tc.option2Discount) * (students - tc.freeTickets)

theorem museum_ticket_problem (tc : TicketCost)
    (h1 : tc.basePrice = 30)
    (h2 : tc.option1Discount = 0.3)
    (h3 : tc.option2Discount = 0.2)
    (h4 : tc.freeTickets = 5) :
  (option1Cost tc 45 < option2Cost tc 45) ∧
  (∃ x : ℕ, x = 40 ∧ option1Cost tc x = option2Cost tc x) := by
  sorry


end NUMINAMATH_CALUDE_museum_ticket_problem_l2755_275571


namespace NUMINAMATH_CALUDE_scientists_sum_equals_total_germany_japan_us_ratio_l2755_275565

/-- The total number of scientists in the research project. -/
def total_scientists : ℕ := 150

/-- The number of scientists from Germany. -/
def germany_scientists : ℕ := 27

/-- The number of scientists from other European countries. -/
def other_europe_scientists : ℕ := 33

/-- The number of scientists from Japan. -/
def japan_scientists : ℕ := 18

/-- The number of scientists from China. -/
def china_scientists : ℕ := 15

/-- The number of scientists from other Asian countries. -/
def other_asia_scientists : ℕ := 12

/-- The number of scientists from Canada. -/
def canada_scientists : ℕ := 23

/-- The number of scientists from the United States. -/
def us_scientists : ℕ := 12

/-- The number of scientists from South America. -/
def south_america_scientists : ℕ := 8

/-- The number of scientists from Australia. -/
def australia_scientists : ℕ := 3

/-- Theorem stating that the sum of scientists from all countries equals the total number of scientists. -/
theorem scientists_sum_equals_total :
  germany_scientists + other_europe_scientists + japan_scientists + china_scientists +
  other_asia_scientists + canada_scientists + us_scientists + south_america_scientists +
  australia_scientists = total_scientists :=
by sorry

/-- Theorem stating the ratio of scientists from Germany, Japan, and the United States. -/
theorem germany_japan_us_ratio :
  ∃ (k : ℕ), k ≠ 0 ∧ germany_scientists = 9 * k ∧ japan_scientists = 6 * k ∧ us_scientists = 4 * k :=
by sorry

end NUMINAMATH_CALUDE_scientists_sum_equals_total_germany_japan_us_ratio_l2755_275565


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equals_B_iff_l2755_275505

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1: When m = 1, A ∩ B = {x | 3 ≤ x < 4}
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2: A ∪ B = B if and only if m ≥ 3 or m ≤ -3
theorem union_equals_B_iff (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equals_B_iff_l2755_275505


namespace NUMINAMATH_CALUDE_train_crossing_time_l2755_275539

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 2 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2755_275539


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2755_275518

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2755_275518


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2755_275531

/-- Given a geometric series {a_n} with positive terms, if a_3 = 18 and S_3 = 26, then q = 3 -/
theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_a3 : a 3 = 18)
  (h_S3 : S 3 = 26) :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2755_275531


namespace NUMINAMATH_CALUDE_combined_weight_proof_l2755_275501

def combined_weight (mary_weight jamison_weight john_weight peter_weight : ℝ) : ℝ :=
  mary_weight + jamison_weight + john_weight + peter_weight

theorem combined_weight_proof (mary_weight : ℝ) 
  (h1 : mary_weight = 160)
  (h2 : ∃ jamison_weight : ℝ, jamison_weight = mary_weight + 20)
  (h3 : ∃ john_weight : ℝ, john_weight = mary_weight * 1.25)
  (h4 : ∃ peter_weight : ℝ, peter_weight = john_weight * 1.15) :
  ∃ total_weight : ℝ, combined_weight mary_weight 
    (mary_weight + 20) (mary_weight * 1.25) (mary_weight * 1.25 * 1.15) = 770 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_weight_proof_l2755_275501


namespace NUMINAMATH_CALUDE_scientific_notation_of_238_billion_l2755_275552

/-- A billion is defined as 10^9 -/
def billion : ℕ := 10^9

/-- The problem statement -/
theorem scientific_notation_of_238_billion :
  (238 : ℝ) * billion = 2.38 * (10 : ℝ)^10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_238_billion_l2755_275552


namespace NUMINAMATH_CALUDE_particular_propositions_count_l2755_275585

/-- A proposition is particular if it contains quantifiers like "some", "exists", or "some of". -/
def is_particular_proposition (p : Prop) : Prop := sorry

/-- The first proposition: Some triangles are isosceles triangles. -/
def prop1 : Prop := sorry

/-- The second proposition: There exists an integer x such that x^2 - 2x - 3 = 0. -/
def prop2 : Prop := sorry

/-- The third proposition: There exists a triangle whose sum of interior angles is 170°. -/
def prop3 : Prop := sorry

/-- The fourth proposition: Rectangles are parallelograms. -/
def prop4 : Prop := sorry

/-- The list of all given propositions. -/
def propositions : List Prop := [prop1, prop2, prop3, prop4]

/-- Count the number of particular propositions in a list. -/
def count_particular_propositions (props : List Prop) : Nat := sorry

theorem particular_propositions_count :
  count_particular_propositions propositions = 3 := by sorry

end NUMINAMATH_CALUDE_particular_propositions_count_l2755_275585
