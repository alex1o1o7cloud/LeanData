import Mathlib

namespace NUMINAMATH_CALUDE_count_solutions_for_a_main_result_l1182_118200

theorem count_solutions_for_a (max_a : Nat) : Nat :=
  let count_pairs (a : Nat) : Nat :=
    (Finset.filter (fun p : Nat × Nat =>
      let m := p.1
      let n := p.2
      n * (1 - m) + a * (1 + m) = 0 ∧ 
      m > 0 ∧ n > 0
    ) (Finset.product (Finset.range (max_a + 1)) (Finset.range (max_a + 1)))).card

  (Finset.filter (fun a : Nat =>
    a > 0 ∧ a ≤ max_a ∧ count_pairs a = 6
  ) (Finset.range (max_a + 1))).card

theorem main_result : count_solutions_for_a 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_for_a_main_result_l1182_118200


namespace NUMINAMATH_CALUDE_ice_cream_sales_ratio_l1182_118240

/-- Ice cream sales problem -/
theorem ice_cream_sales_ratio (tuesday_sales wednesday_sales : ℕ) : 
  tuesday_sales = 12000 →
  wednesday_sales = 36000 - tuesday_sales →
  (wednesday_sales : ℚ) / tuesday_sales = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_ratio_l1182_118240


namespace NUMINAMATH_CALUDE_equation_solution_l1182_118259

theorem equation_solution :
  ∃ y : ℚ, y ≠ -2 ∧ (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ∧ y = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1182_118259


namespace NUMINAMATH_CALUDE_sock_ratio_is_one_to_two_l1182_118253

/-- Represents the order of socks --/
structure SockOrder where
  green : ℕ
  red : ℕ
  green_price : ℝ
  red_price : ℝ

/-- The original order --/
def original_order : SockOrder := {
  green := 6,
  red := 0,  -- We don't know this value yet
  green_price := 0,  -- We don't know this value yet
  red_price := 0  -- We don't know this value yet
}

/-- The swapped order --/
def swapped_order (o : SockOrder) : SockOrder := {
  green := o.red,
  red := o.green,
  green_price := o.green_price,
  red_price := o.red_price
}

/-- The cost of an order --/
def cost (o : SockOrder) : ℝ :=
  o.green * o.green_price + o.red * o.red_price

/-- The theorem to prove --/
theorem sock_ratio_is_one_to_two :
  ∃ (o : SockOrder),
    o.green = 6 ∧
    o.green_price = 3 * o.red_price ∧
    cost (swapped_order o) = 1.4 * cost o ∧
    o.green / o.red = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_sock_ratio_is_one_to_two_l1182_118253


namespace NUMINAMATH_CALUDE_greatest_valid_number_divisible_by_11_l1182_118282

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    A < 9 ∧
    n = A * 10000 + B * 1000 + C * 100 + B * 10 + A

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem greatest_valid_number_divisible_by_11 :
  ∀ n : ℕ, is_valid_number n → is_divisible_by_11 n → n ≤ 87978 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_number_divisible_by_11_l1182_118282


namespace NUMINAMATH_CALUDE_run_6000_ends_at_S_S_associated_with_D_or_A_l1182_118280

/-- Represents the quarters of the circular track -/
inductive Quarter
| A
| B
| C
| D

/-- Represents a point on the circular track -/
structure Point where
  quarter : Quarter
  distance : ℝ
  h_distance_bound : 0 ≤ distance ∧ distance < 15

/-- The circular track -/
structure Track where
  circumference : ℝ
  h_circumference : circumference = 60

/-- Runner's position after running a given distance -/
def run_position (track : Track) (start : Point) (distance : ℝ) : Point :=
  sorry

/-- Theorem stating that running 6000 feet from point S ends at point S -/
theorem run_6000_ends_at_S (track : Track) (S : Point) 
    (h_S : S.quarter = Quarter.D ∧ S.distance = 0) : 
    run_position track S 6000 = S :=
  sorry

/-- Theorem stating that point S is associated with quarter D or A -/
theorem S_associated_with_D_or_A (S : Point) 
    (h_S : S.quarter = Quarter.D ∧ S.distance = 0) : 
    S.quarter = Quarter.D ∨ S.quarter = Quarter.A :=
  sorry

end NUMINAMATH_CALUDE_run_6000_ends_at_S_S_associated_with_D_or_A_l1182_118280


namespace NUMINAMATH_CALUDE_equal_numbers_l1182_118228

theorem equal_numbers (a : Fin 100 → ℝ)
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
by sorry


end NUMINAMATH_CALUDE_equal_numbers_l1182_118228


namespace NUMINAMATH_CALUDE_power_sum_inequality_l1182_118251

theorem power_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l1182_118251


namespace NUMINAMATH_CALUDE_wonderful_coloring_bounds_l1182_118271

/-- A wonderful coloring of a regular polygon is a coloring where no triangle
    formed by its vertices has exactly two colors among its sides. -/
def WonderfulColoring (n : ℕ) (m : ℕ) : Prop := sorry

/-- N is the largest positive integer for which there exists a wonderful coloring
    of a regular N-gon with M colors. -/
def LargestWonderfulN (m : ℕ) : ℕ := sorry

theorem wonderful_coloring_bounds (m : ℕ) (h : m ≥ 3) :
  let n := LargestWonderfulN m
  (n ≤ (m - 1)^2) ∧
  (Nat.Prime (m - 1) → n = (m - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_wonderful_coloring_bounds_l1182_118271


namespace NUMINAMATH_CALUDE_max_apples_capacity_l1182_118298

theorem max_apples_capacity (num_boxes : ℕ) (trays_per_box : ℕ) (extra_trays : ℕ) (apples_per_tray : ℕ) : 
  num_boxes = 6 → trays_per_box = 12 → extra_trays = 4 → apples_per_tray = 8 →
  (num_boxes * trays_per_box + extra_trays) * apples_per_tray = 608 := by
  sorry

end NUMINAMATH_CALUDE_max_apples_capacity_l1182_118298


namespace NUMINAMATH_CALUDE_crayons_per_day_l1182_118263

theorem crayons_per_day (total_crayons : ℕ) (crayons_per_box : ℕ) 
  (h1 : total_crayons = 321)
  (h2 : crayons_per_box = 7) : 
  (total_crayons / crayons_per_box : ℕ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_day_l1182_118263


namespace NUMINAMATH_CALUDE_product_of_digits_is_64_l1182_118250

/-- Represents a number in different bases -/
structure NumberInBases where
  base10 : ℕ
  b : ℕ
  base_b : ℕ
  base_b_plus_2 : ℕ

/-- Calculates the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the product of digits of N is 64 -/
theorem product_of_digits_is_64 (N : NumberInBases) 
  (h1 : N.base_b = 503)
  (h2 : N.base_b_plus_2 = 305)
  (h3 : N.b > 0) : 
  productOfDigits N.base10 = 64 := by sorry

end NUMINAMATH_CALUDE_product_of_digits_is_64_l1182_118250


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1182_118264

/-- A circle with center O and radius 3 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A line m containing point P -/
structure Line :=
  (m : Set (ℝ × ℝ))
  (P : ℝ × ℝ)
  (h_P_on_m : P ∈ m)

/-- The distance between two points in ℝ² -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

/-- Defines what it means for a line to be tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (Q : ℝ × ℝ), Q ∈ l.m ∧ distance Q c.O = c.radius ∧
  ∀ (R : ℝ × ℝ), R ∈ l.m → R ≠ Q → distance R c.O > c.radius

theorem line_tangent_to_circle (c : Circle) (l : Line) :
  distance l.P c.O = c.radius →
  is_tangent l c :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1182_118264


namespace NUMINAMATH_CALUDE_equal_angle_vector_l1182_118224

theorem equal_angle_vector (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (4, 2) → 
  c ≠ (0, 0) → 
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) = 
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  ∃ (k : ℝ), k ≠ 0 ∧ c = (k, k) := by
sorry

end NUMINAMATH_CALUDE_equal_angle_vector_l1182_118224


namespace NUMINAMATH_CALUDE_smallest_m_for_square_inequality_l1182_118273

theorem smallest_m_for_square_inequality : ∃ (m : ℕ+), 
  (m = 16144325) ∧ 
  (∀ (n : ℕ+), n ≥ m → ∃ (l : ℕ+), (n : ℝ) < (l : ℝ)^2 ∧ (l : ℝ)^2 < (1 + 1/2009) * (n : ℝ)) ∧
  (∀ (m' : ℕ+), m' < m → ∃ (n : ℕ+), n ≥ m' ∧ ∀ (l : ℕ+), ((n : ℝ) ≥ (l : ℝ)^2 ∨ (l : ℝ)^2 ≥ (1 + 1/2009) * (n : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_square_inequality_l1182_118273


namespace NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l1182_118225

/-- The cost per trip to an amusement park given the following conditions:
  * Two season passes are purchased
  * Each pass costs 100 (in some currency unit)
  * One person uses their pass 35 times
  * Another person uses their pass 15 times
-/
theorem amusement_park_cost_per_trip 
  (pass_cost : ℝ) 
  (num_passes : ℕ) 
  (trips_person1 : ℕ) 
  (trips_person2 : ℕ) 
  (h1 : pass_cost = 100) 
  (h2 : num_passes = 2) 
  (h3 : trips_person1 = 35) 
  (h4 : trips_person2 = 15) : 
  (num_passes * pass_cost) / (trips_person1 + trips_person2 : ℝ) = 4 := by
  sorry

#check amusement_park_cost_per_trip

end NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l1182_118225


namespace NUMINAMATH_CALUDE_parallelogram_inequality_l1182_118261

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Parallelogram P -/
structure Parallelogram (n : ℕ) (t : ℝ) where
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (0, t)
  v3 : ℝ × ℝ := (t * fib (2 * n + 1), t * fib (2 * n))
  v4 : ℝ × ℝ := (t * fib (2 * n + 1), t * fib (2 * n) + t)

/-- Number of integer points inside P -/
def L (n : ℕ) (t : ℝ) : ℕ := sorry

/-- Area of P -/
def M (n : ℕ) (t : ℝ) : ℝ := t^2 * fib (2 * n + 1)

/-- Main theorem -/
theorem parallelogram_inequality (n : ℕ) (t : ℝ) (hn : n > 1) (ht : t ≥ 1) :
  |Real.sqrt (L n t) - Real.sqrt (M n t)| ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_parallelogram_inequality_l1182_118261


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1182_118246

theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1182_118246


namespace NUMINAMATH_CALUDE_sin_sq_plus_cos_sq_eq_one_s_sq_plus_c_sq_eq_one_l1182_118293

/-- Given an angle θ, prove that sin²θ + cos²θ = 1 -/
theorem sin_sq_plus_cos_sq_eq_one (θ : Real) : (Real.sin θ)^2 + (Real.cos θ)^2 = 1 := by
  sorry

/-- Given s = sin θ and c = cos θ for some angle θ, prove that s² + c² = 1 -/
theorem s_sq_plus_c_sq_eq_one (s c : Real) (h : ∃ θ : Real, s = Real.sin θ ∧ c = Real.cos θ) : s^2 + c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_sq_plus_cos_sq_eq_one_s_sq_plus_c_sq_eq_one_l1182_118293


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l1182_118221

theorem max_subjects_per_teacher 
  (total_subjects : Nat) 
  (min_teachers : Nat) 
  (h1 : total_subjects = 18) 
  (h2 : min_teachers = 6) : 
  Nat.ceil (total_subjects / min_teachers) = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l1182_118221


namespace NUMINAMATH_CALUDE_rose_orchid_difference_l1182_118234

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 5

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 3

/-- The number of roses finally in the vase -/
def final_roses : ℕ := 12

/-- The number of orchids finally in the vase -/
def final_orchids : ℕ := 2

/-- The difference between the final number of roses and orchids in the vase -/
theorem rose_orchid_difference : final_roses - final_orchids = 10 := by
  sorry

end NUMINAMATH_CALUDE_rose_orchid_difference_l1182_118234


namespace NUMINAMATH_CALUDE_students_just_passed_l1182_118256

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first_div : first_div_percent = 30 / 100)
  (h_second_div : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l1182_118256


namespace NUMINAMATH_CALUDE_company_production_theorem_l1182_118274

/-- Represents the production schedule of a company making parts --/
structure ProductionSchedule where
  initialRate : ℕ            -- Initial production rate (parts per day)
  initialDays : ℕ            -- Number of days at initial rate
  increasedRate : ℕ          -- Increased production rate (parts per day)
  extraParts : ℕ             -- Extra parts produced beyond the plan

/-- Calculates the total number of parts produced given a production schedule --/
def totalPartsProduced (schedule : ProductionSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific production schedule, 675 parts are produced --/
theorem company_production_theorem :
  let schedule := ProductionSchedule.mk 25 3 30 100
  totalPartsProduced schedule = 675 :=
by sorry

end NUMINAMATH_CALUDE_company_production_theorem_l1182_118274


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1182_118211

theorem sqrt_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 3 ∧
  (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) = 2 * Real.sqrt 3 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1182_118211


namespace NUMINAMATH_CALUDE_horner_rule_v₄_l1182_118283

-- Define the polynomial coefficients
def a₀ : ℤ := 12
def a₁ : ℤ := 35
def a₂ : ℤ := -8
def a₃ : ℤ := 6
def a₄ : ℤ := 5
def a₅ : ℤ := 3

-- Define x
def x : ℤ := -2

-- Define Horner's Rule steps
def v₀ : ℤ := a₅
def v₁ : ℤ := v₀ * x + a₄
def v₂ : ℤ := v₁ * x + a₃
def v₃ : ℤ := v₂ * x + a₂
def v₄ : ℤ := v₃ * x + a₁

-- Theorem statement
theorem horner_rule_v₄ : v₄ = 83 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_v₄_l1182_118283


namespace NUMINAMATH_CALUDE_min_height_rectangular_container_l1182_118296

theorem min_height_rectangular_container (h : ℝ) (y : ℝ) :
  h = 2 * y →                -- height is twice the side length
  y > 0 →                    -- side length is positive
  10 * y^2 ≥ 150 →           -- surface area is at least 150
  h ≥ 2 * Real.sqrt 15 :=    -- minimum height is 2√15
sorry

end NUMINAMATH_CALUDE_min_height_rectangular_container_l1182_118296


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1182_118208

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ - m = 0 ∧ x₂^2 - 4*x₂ - m = 0) ↔ m > -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1182_118208


namespace NUMINAMATH_CALUDE_tan_double_angle_l1182_118254

theorem tan_double_angle (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) : 
  Real.tan (2 * α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1182_118254


namespace NUMINAMATH_CALUDE_stormi_lawn_mowing_charge_l1182_118270

/-- Proves that Stormi charges $13 for mowing each lawn given the problem conditions -/
theorem stormi_lawn_mowing_charge : 
  (car_wash_count : ℕ) →
  (car_wash_price : ℚ) →
  (lawn_count : ℕ) →
  (bicycle_price : ℚ) →
  (additional_money_needed : ℚ) →
  car_wash_count = 3 →
  car_wash_price = 10 →
  lawn_count = 2 →
  bicycle_price = 80 →
  additional_money_needed = 24 →
  (bicycle_price - additional_money_needed - car_wash_count * car_wash_price) / lawn_count = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_stormi_lawn_mowing_charge_l1182_118270


namespace NUMINAMATH_CALUDE_congruence_problem_l1182_118223

theorem congruence_problem (x : ℤ) : 
  (5 * x + 8) % 14 = 3 → (3 * x + 10) % 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1182_118223


namespace NUMINAMATH_CALUDE_x_squared_plus_4x_plus_5_range_l1182_118205

theorem x_squared_plus_4x_plus_5_range :
  ∀ x : ℝ, x^2 - 7*x + 12 < 0 →
  ∃ y ∈ Set.Ioo 26 37, y = x^2 + 4*x + 5 ∧
  ∀ z, z = x^2 + 4*x + 5 → z ∈ Set.Ioo 26 37 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_4x_plus_5_range_l1182_118205


namespace NUMINAMATH_CALUDE_factorial_fraction_is_integer_l1182_118247

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ∃ k : ℤ, (↑((2 * m).factorial * (2 * n).factorial) : ℚ) / 
    (↑(m.factorial * n.factorial * (m + n).factorial) : ℚ) = ↑k :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_is_integer_l1182_118247


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l1182_118230

/-- The decimal representation of 0.0808... -/
def repeating_decimal_08 : ℚ := 8 / 99

/-- The decimal representation of 0.3636... -/
def repeating_decimal_36 : ℚ := 36 / 99

/-- The product of 0.0808... and 0.3636... is equal to 288/9801 -/
theorem product_of_repeating_decimals : 
  repeating_decimal_08 * repeating_decimal_36 = 288 / 9801 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l1182_118230


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l1182_118231

theorem fraction_to_zero_power (a b : ℤ) (h : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l1182_118231


namespace NUMINAMATH_CALUDE_division_problem_l1182_118285

theorem division_problem : (144 : ℚ) / ((12 : ℚ) / 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1182_118285


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l1182_118266

/-- Given two people moving in opposite directions, this theorem proves the speed of one person
    given the speed of the other and their final distance after a certain time. -/
theorem opposite_direction_speed
  (time : ℝ)
  (speed_person2 : ℝ)
  (final_distance : ℝ)
  (h1 : time > 0)
  (h2 : speed_person2 > 0)
  (h3 : final_distance > 0)
  (h4 : final_distance = (speed_person1 + speed_person2) * time)
  (h5 : time = 4)
  (h6 : speed_person2 = 3)
  (h7 : final_distance = 36) :
  speed_person1 = 6 :=
sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l1182_118266


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l1182_118276

theorem lucky_larry_problem (p q r s t : ℤ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8) :
  p - (q - (r - (s - t))) = p - q - r + s - t → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l1182_118276


namespace NUMINAMATH_CALUDE_S_2023_eq_half_l1182_118239

def S : ℕ → ℚ
  | 0 => 1 / 2
  | n + 1 => if n % 2 = 0 then 1 / S n else -S n - 1

theorem S_2023_eq_half : S 2022 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_S_2023_eq_half_l1182_118239


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l1182_118222

/-- A dishonest dealer's profit percentage when using underweight measurements --/
theorem dishonest_dealer_profit_percentage 
  (actual_weight : ℝ) 
  (claimed_weight : ℝ) 
  (actual_weight_positive : 0 < actual_weight)
  (claimed_weight_greater : actual_weight < claimed_weight) : 
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  (1000 - 920) / 920 * 100 := by
sorry

#eval (1000 - 920) / 920 * 100  -- To show the approximate result

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l1182_118222


namespace NUMINAMATH_CALUDE_min_white_fraction_4x4x4_cube_l1182_118237

/-- Represents a cube composed of smaller unit cubes -/
structure CompositeCube where
  edge_length : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- The minimum fraction of white surface area for a given composite cube -/
def min_white_fraction (c : CompositeCube) : ℚ :=
  sorry

theorem min_white_fraction_4x4x4_cube :
  let c : CompositeCube := ⟨4, 50, 14⟩
  min_white_fraction c = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_min_white_fraction_4x4x4_cube_l1182_118237


namespace NUMINAMATH_CALUDE_bowen_total_spent_l1182_118272

/-- The price of a pencil in dollars -/
def pencil_price : ℚ := 25/100

/-- The price of a pen in dollars -/
def pen_price : ℚ := 15/100

/-- The number of pens Bowen buys -/
def num_pens : ℕ := 40

/-- The number of pencils Bowen buys -/
def num_pencils : ℕ := num_pens + (2 * num_pens) / 5

/-- The total amount Bowen spends in dollars -/
def total_spent : ℚ := num_pens * pen_price + num_pencils * pencil_price

theorem bowen_total_spent : total_spent = 20 := by sorry

end NUMINAMATH_CALUDE_bowen_total_spent_l1182_118272


namespace NUMINAMATH_CALUDE_perpendicular_tangents_locus_l1182_118295

/-- The locus of points where mutually perpendicular tangents to x² + y² = 32 intersect -/
theorem perpendicular_tangents_locus (x₀ y₀ : ℝ) : 
  (∃ t₁ t₂ : ℝ → ℝ, 
    (∀ x y, x^2 + y^2 = 32 → (t₁ x = y ∨ t₂ x = y) → (x - x₀) * (y - y₀) = 0) ∧ 
    (∀ x, (t₁ x - y₀) * (t₂ x - y₀) = -1)) →
  x₀^2 + y₀^2 = 64 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_locus_l1182_118295


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l1182_118217

theorem angle_sum_in_circle (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + 2 * x + x = 360) → x = 360 / 17 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l1182_118217


namespace NUMINAMATH_CALUDE_extreme_value_implies_f_2_l1182_118242

/-- A function f with an extreme value at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_implies_f_2 (a b : ℝ) :
  (f' a b 1 = 0) →  -- f has an extreme value at x = 1
  (f a b 1 = 10) →  -- The extreme value is 10
  (f a b 2 = 11 ∨ f a b 2 = 18) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_f_2_l1182_118242


namespace NUMINAMATH_CALUDE_cross_section_perimeter_bound_l1182_118206

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  edge_positive : 0 < a

/-- A triangular cross-section through one vertex of a regular tetrahedron -/
structure TriangularCrossSection (t : RegularTetrahedron) where
  perimeter : ℝ

/-- The perimeter of any triangular cross-section through one vertex of a regular tetrahedron
    is greater than twice the edge length -/
theorem cross_section_perimeter_bound (t : RegularTetrahedron) 
  (s : TriangularCrossSection t) : s.perimeter > 2 * t.a := by
  sorry

end NUMINAMATH_CALUDE_cross_section_perimeter_bound_l1182_118206


namespace NUMINAMATH_CALUDE_larger_root_of_quadratic_l1182_118226

theorem larger_root_of_quadratic (x : ℝ) : 
  x^2 + 17*x - 72 = 0 → x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_larger_root_of_quadratic_l1182_118226


namespace NUMINAMATH_CALUDE_max_value_of_sum_l1182_118260

noncomputable def f (x : ℝ) : ℝ := 3^(x-1) + x - 1

def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem max_value_of_sum (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = 3^(x-1) + x - 1) →
  is_inverse f f_inv →
  (∃ y, ∀ x ∈ Set.Icc 0 1, f x + f_inv x ≤ y) ∧
  (∃ x ∈ Set.Icc 0 1, f x + f_inv x = 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l1182_118260


namespace NUMINAMATH_CALUDE_store_earnings_is_400_l1182_118220

/-- Calculates the total earnings of a clothing store selling shirts and jeans -/
def store_earnings (num_shirts : ℕ) (num_jeans : ℕ) (shirt_price : ℕ) : ℕ :=
  let jeans_price := 2 * shirt_price
  num_shirts * shirt_price + num_jeans * jeans_price

/-- Theorem: The clothing store will earn $400 if all shirts and jeans are sold -/
theorem store_earnings_is_400 :
  store_earnings 20 10 10 = 400 := by
sorry

end NUMINAMATH_CALUDE_store_earnings_is_400_l1182_118220


namespace NUMINAMATH_CALUDE_power_of_product_squared_l1182_118252

theorem power_of_product_squared (a : ℝ) : (3 * a^2)^2 = 9 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l1182_118252


namespace NUMINAMATH_CALUDE_subtracted_amount_for_ratio_change_l1182_118267

theorem subtracted_amount_for_ratio_change : ∃ (a : ℝ),
  (72 : ℝ) / 192 = 3 / 8 ∧
  (72 - a) / (192 - a) = 4 / 9 ∧
  a = 24 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_for_ratio_change_l1182_118267


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1182_118216

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  a > 0 → b > 0 → a > b → a + b = 5 * (a - b) → a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1182_118216


namespace NUMINAMATH_CALUDE_jemma_grasshopper_count_l1182_118209

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found on the grass -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshopper_count_l1182_118209


namespace NUMINAMATH_CALUDE_second_even_integer_is_78_l1182_118275

/-- Given three consecutive even integers where the sum of the first and third is 156,
    prove that the second integer is 78. -/
theorem second_even_integer_is_78 :
  ∀ (a b c : ℤ),
  (b = a + 2) →  -- b is the next consecutive even integer after a
  (c = b + 2) →  -- c is the next consecutive even integer after b
  (a % 2 = 0) →  -- a is even
  (a + c = 156) →  -- sum of first and third is 156
  b = 78 := by
sorry

end NUMINAMATH_CALUDE_second_even_integer_is_78_l1182_118275


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l1182_118286

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l1182_118286


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1182_118281

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) * (1 - Complex.I) = 2) :
  z.im = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1182_118281


namespace NUMINAMATH_CALUDE_percentage_difference_l1182_118241

theorem percentage_difference (x y : ℝ) (h : x = 6 * y) :
  (x - y) / x * 100 = 83.33333333333333 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1182_118241


namespace NUMINAMATH_CALUDE_quadratic_properties_l1182_118287

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) 
  (h2 : quadratic_function a b c (-1) = 0) 
  (h3 : -b / (2 * a) = 1) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, quadratic_function a b c m ≤ -4 * a) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → quadratic_function a b c x1 = -1 → 
    quadratic_function a b c x2 = -1 → x1 < -1 ∧ x2 > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1182_118287


namespace NUMINAMATH_CALUDE_complex_equality_l1182_118202

theorem complex_equality (u v : ℂ) 
  (h1 : 3 * Complex.abs (u + 1) * Complex.abs (v + 1) ≥ Complex.abs (u * v + 5 * u + 5 * v + 1))
  (h2 : Complex.abs (u + v) = Complex.abs (u * v + 1)) :
  u = 1 ∨ v = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_equality_l1182_118202


namespace NUMINAMATH_CALUDE_line_intersects_segment_iff_a_gt_two_l1182_118227

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on the positive side of a line -/
def positiveSide (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

/-- Check if a point is on the negative side of a line -/
def negativeSide (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

/-- Check if two points are on opposite sides of a line -/
def oppositeSides (l : Line) (p1 p2 : Point) : Prop :=
  (positiveSide l p1 ∧ negativeSide l p2) ∨ (negativeSide l p1 ∧ positiveSide l p2)

/-- The main theorem -/
theorem line_intersects_segment_iff_a_gt_two (a : ℝ) :
  let A : Point := ⟨1, a⟩
  let B : Point := ⟨2, 4⟩
  let l : Line := ⟨1, -1, 1⟩
  oppositeSides l A B ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_segment_iff_a_gt_two_l1182_118227


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1182_118289

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 + 2 * I ^ 6) = (2 - 3 * I) / I → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1182_118289


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1182_118210

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = (t * a.1, t * a.2)

/-- The vectors a and b as defined in the problem -/
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b (k : ℝ) : ℝ × ℝ := (k, 4)

/-- k=-2 is sufficient for collinearity -/
theorem sufficient_condition (k : ℝ) : 
  k = -2 → collinear (a k) (b k) :=
sorry

/-- k=-2 is not necessary for collinearity -/
theorem not_necessary_condition : 
  ∃ k : ℝ, k ≠ -2 ∧ collinear (a k) (b k) :=
sorry

/-- The main theorem stating that k=-2 is sufficient but not necessary -/
theorem sufficient_but_not_necessary : 
  (∀ k : ℝ, k = -2 → collinear (a k) (b k)) ∧
  (∃ k : ℝ, k ≠ -2 ∧ collinear (a k) (b k)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1182_118210


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1182_118203

theorem pie_eating_contest :
  let student1 : ℚ := 8 / 9
  let student2 : ℚ := 5 / 6
  let student3 : ℚ := 2 / 3
  student1 + student2 + student3 = 43 / 18 :=
by sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1182_118203


namespace NUMINAMATH_CALUDE_percentage_difference_l1182_118232

theorem percentage_difference (A B C x : ℝ) : 
  C > B → B > A → A > 0 → C = A + 2*B → A = B * (1 - x/100) → 
  x = 100 * ((B - A) / B) := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1182_118232


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1182_118215

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1182_118215


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l1182_118269

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

structure Community where
  totalFamilies : Nat
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

structure School where
  femaleAthletes : Nat
  selectionSize : Nat

def optimalSamplingMethod (c : Community) (s : School) : 
  (SamplingMethod × SamplingMethod) :=
  sorry

theorem optimal_sampling_methods 
  (c : Community) 
  (s : School) 
  (h1 : c.totalFamilies = 500)
  (h2 : c.highIncomeFamilies = 125)
  (h3 : c.middleIncomeFamilies = 280)
  (h4 : c.lowIncomeFamilies = 95)
  (h5 : c.sampleSize = 100)
  (h6 : s.femaleAthletes = 12)
  (h7 : s.selectionSize = 3) :
  optimalSamplingMethod c s = (SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
  sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l1182_118269


namespace NUMINAMATH_CALUDE_sock_pair_count_l1182_118219

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: There are 47 ways to choose a pair of socks of different colors
    from 5 white socks, 4 brown socks, and 3 blue socks -/
theorem sock_pair_count :
  differentColorPairs 5 4 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l1182_118219


namespace NUMINAMATH_CALUDE_possible_m_values_l1182_118294

def A (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}
def B : Set ℝ := {2, 3}

theorem possible_m_values :
  ∀ m : ℝ, (A m) ⊆ B → (m = 0 ∨ m = 1/2 ∨ m = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_possible_m_values_l1182_118294


namespace NUMINAMATH_CALUDE_max_value_of_function_l1182_118233

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (y : ℝ), y = x^2 * (1 - 2*x) ∧ y ≤ 1/27 ∧ ∃ (x0 : ℝ), 0 < x0 ∧ x0 < 1/2 ∧ x0^2 * (1 - 2*x0) = 1/27 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1182_118233


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1182_118279

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Length of the smaller segment of the lateral side -/
  smaller_segment : ℝ
  /-- Length of the larger segment of the lateral side -/
  larger_segment : ℝ

/-- Calculate the area of the isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 156 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoidWithInscribedCircle := ⟨4, 9⟩
  area t = 156 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1182_118279


namespace NUMINAMATH_CALUDE_fractional_equation_solutions_l1182_118290

/-- The fractional equation in terms of x and m -/
def fractional_equation (x m : ℝ) : Prop :=
  3 * x / (x - 1) = m / (x - 1) + 2

theorem fractional_equation_solutions :
  (∃! x : ℝ, fractional_equation x 4) ∧
  (∀ x : ℝ, ¬fractional_equation x 3) ∧
  (∀ m : ℝ, m ≠ 3 → ∃ x : ℝ, fractional_equation x m) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_solutions_l1182_118290


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l1182_118212

/-- A linear function of the form y = kx + k passes through the point (-1, 0) for any non-zero k. -/
theorem linear_function_passes_through_point (k : ℝ) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ k * x + k
  f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l1182_118212


namespace NUMINAMATH_CALUDE_ellipse_min_area_l1182_118284

/-- An ellipse containing two specific circles has a minimum area -/
theorem ellipse_min_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  π * a * b ≥ (3 * Real.sqrt 3 / 2) * π := by
  sorry

#check ellipse_min_area

end NUMINAMATH_CALUDE_ellipse_min_area_l1182_118284


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l1182_118243

theorem line_ellipse_intersection
  (m n : ℝ)
  (h1 : m^2 + n^2 < 3)
  (h2 : 0 < m^2 + n^2) :
  ∀ (a b : ℝ), ∃! (x y : ℝ),
    x^2 / 7 + y^2 / 3 = 1 ∧
    y = a*x + b ∧
    a*m + b = n :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l1182_118243


namespace NUMINAMATH_CALUDE_subset_implies_lower_bound_l1182_118255

/-- Given sets A = [1, 4) and B = (-∞, a), if A ⊂ B, then a ≥ 4 -/
theorem subset_implies_lower_bound (a : ℝ) :
  let A := { x : ℝ | 1 ≤ x ∧ x < 4 }
  let B := { x : ℝ | x < a }
  A ⊆ B → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_lower_bound_l1182_118255


namespace NUMINAMATH_CALUDE_mean_median_difference_l1182_118297

theorem mean_median_difference (x : ℕ) (h : x > 0) : 
  let sequence := [x, x + 2, x + 4, x + 7, x + 37]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1182_118297


namespace NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_three_l1182_118238

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_iff_x_eq_neg_three :
  ∀ x : ℝ, perpendicular (x, -3) (2, -2) ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_three_l1182_118238


namespace NUMINAMATH_CALUDE_point_d_from_c_l1182_118248

/-- Given two points C and D in the Cartesian coordinate system, prove that D is obtained from C by moving 3 units downwards -/
theorem point_d_from_c (C D : ℝ × ℝ) : 
  C = (1, 2) → D = (1, -1) → 
  (C.2 - D.2 = 3) ∧ (D.2 < C.2) := by
  sorry

end NUMINAMATH_CALUDE_point_d_from_c_l1182_118248


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l1182_118244

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2

-- Define midpoint
def Midpoint (M X Y : ℝ × ℝ) : Prop :=
  M.1 = (X.1 + Y.1) / 2 ∧ M.2 = (X.2 + Y.2) / 2

-- Define the theorem
theorem shaded_area_ratio 
  (A B C D E F G H : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint D A B) 
  (h3 : Midpoint E B C) 
  (h4 : Midpoint F C A) 
  (h5 : Midpoint G D F) 
  (h6 : Midpoint H F E) :
  let shaded_area := 
    -- Area of triangle DEF + Area of three trapezoids
    (Real.sqrt 3 / 16 + 9 * Real.sqrt 3 / 32) * s^2
  let non_shaded_area := 
    -- Total area of triangle ABC - Shaded area
    (Real.sqrt 3 / 4 - 11 * Real.sqrt 3 / 32) * s^2
  shaded_area / non_shaded_area = 11 / 21 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l1182_118244


namespace NUMINAMATH_CALUDE_quadratic_trinomial_constant_l1182_118214

/-- Given that x^{|m|}+(m-2)x-10 is a quadratic trinomial where m is a constant, prove that m = -2 -/
theorem quadratic_trinomial_constant (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, x^(|m|) + (m-2)*x - 10 = a*x^2 + b*x + c ∧ a ≠ 0 ∧ b ≠ 0) → 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_constant_l1182_118214


namespace NUMINAMATH_CALUDE_min_crossing_time_is_21_l1182_118207

/-- Represents a person with their crossing time -/
structure Person where
  name : String
  time : ℕ

/-- Represents the tunnel crossing problem -/
structure TunnelProblem where
  people : List Person
  flashlight : ℕ := 1
  capacity : ℕ := 2

def minCrossingTime (problem : TunnelProblem) : ℕ :=
  sorry

/-- The specific problem instance -/
def ourProblem : TunnelProblem :=
  { people := [
      { name := "A", time := 3 },
      { name := "B", time := 4 },
      { name := "C", time := 5 },
      { name := "D", time := 6 }
    ]
  }

theorem min_crossing_time_is_21 :
  minCrossingTime ourProblem = 21 :=
by sorry

end NUMINAMATH_CALUDE_min_crossing_time_is_21_l1182_118207


namespace NUMINAMATH_CALUDE_sequence_not_contains_010101_l1182_118235

/-- Represents a sequence where each term after the sixth is the last digit of the sum of the previous six terms -/
def Sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (Sequence n + Sequence (n + 1) + Sequence (n + 2) + Sequence (n + 3) + Sequence (n + 4) + Sequence (n + 5)) % 10

/-- The weighted sum function used in the proof -/
def S (a b c d e f : ℕ) : ℕ := 2*a + 4*b + 6*c + 8*d + 10*e + 12*f

theorem sequence_not_contains_010101 :
  ∀ n : ℕ, ¬(Sequence n = 0 ∧ Sequence (n + 1) = 1 ∧ Sequence (n + 2) = 0 ∧
            Sequence (n + 3) = 1 ∧ Sequence (n + 4) = 0 ∧ Sequence (n + 5) = 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_contains_010101_l1182_118235


namespace NUMINAMATH_CALUDE_arc_length_30_degrees_l1182_118262

/-- The length of an arc in a circle with radius 3 and central angle 30° is π/2 -/
theorem arc_length_30_degrees (r : ℝ) (θ : ℝ) (L : ℝ) : 
  r = 3 → θ = 30 * π / 180 → L = r * θ → L = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_30_degrees_l1182_118262


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l1182_118249

theorem arithmetic_expression_equals_24 : ∃ (f : List ℝ → ℝ), 
  (f [5, 7, 8, 8] = 24) ∧ 
  (∀ x y z w, f [x, y, z, w] = 
    ((x + y) / z) * w ∨ 
    f [x, y, z, w] = (x - y) * z + w) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l1182_118249


namespace NUMINAMATH_CALUDE_clown_balloons_l1182_118257

/-- The number of balloons a clown has after blowing up an initial set and then an additional set -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the clown has 60 balloons after blowing up 47 and then 13 more -/
theorem clown_balloons : total_balloons 47 13 = 60 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l1182_118257


namespace NUMINAMATH_CALUDE_ten_people_leaders_and_committee_l1182_118258

/-- The number of ways to choose a president, vice-president, and committee from a group --/
def choose_leaders_and_committee (n : ℕ) : ℕ :=
  n * (n - 1) * Nat.choose (n - 2) 3

/-- The theorem stating the number of ways to choose leaders and committee from 10 people --/
theorem ten_people_leaders_and_committee :
  choose_leaders_and_committee 10 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_leaders_and_committee_l1182_118258


namespace NUMINAMATH_CALUDE_value_of_c_l1182_118236

theorem value_of_c (a b c d : ℝ) : 
  12 = 0.04 * (a + d) →
  4 = 0.12 * (b - d) →
  c = (b - d) / (a + d) →
  c = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l1182_118236


namespace NUMINAMATH_CALUDE_trapezoid_angle_bisector_area_ratio_l1182_118229

/-- The area ratio of the quadrilateral formed by angle bisector intersections to the trapezoid --/
def area_ratio (a b c d : ℝ) : Set ℝ :=
  {x | x = 1/45 ∨ x = 7/40}

/-- Theorem stating the area ratio for a trapezoid with given side lengths --/
theorem trapezoid_angle_bisector_area_ratio :
  ∀ (a b c d : ℝ),
  a = 5 ∧ b = 15 ∧ c = 15 ∧ d = 20 →
  ∃ (k : ℝ), k ∈ area_ratio a b c d :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_angle_bisector_area_ratio_l1182_118229


namespace NUMINAMATH_CALUDE_ratio_transformation_l1182_118245

theorem ratio_transformation (a b c d x : ℚ) : 
  a = 4 ∧ b = 15 ∧ c = 3 ∧ d = 4 ∧ x = 29 →
  (a + x) / (b + x) = c / d := by
sorry

end NUMINAMATH_CALUDE_ratio_transformation_l1182_118245


namespace NUMINAMATH_CALUDE_food_price_increase_l1182_118201

theorem food_price_increase 
  (initial_students : ℝ) 
  (initial_food_price : ℝ) 
  (initial_food_consumption : ℝ) 
  (h_students_positive : initial_students > 0) 
  (h_price_positive : initial_food_price > 0) 
  (h_consumption_positive : initial_food_consumption > 0) :
  let new_students := 0.9 * initial_students
  let new_food_consumption := 0.9259259259259259 * initial_food_consumption
  let new_food_price := x * initial_food_price
  x = 1.2 ↔ 
    new_students * new_food_consumption * new_food_price = 
    initial_students * initial_food_consumption * initial_food_price := by
sorry

end NUMINAMATH_CALUDE_food_price_increase_l1182_118201


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1182_118268

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 61) 
  (h2 : throwers = 37) 
  (h3 : throwers ≤ total_players) 
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + (2 * (total_players - throwers) / 3)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1182_118268


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1182_118218

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1182_118218


namespace NUMINAMATH_CALUDE_rock_song_requests_l1182_118299

/-- Represents the number of song requests for each genre --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rock song requests given the conditions --/
theorem rock_song_requests (req : SongRequests) : req.rock = 5 :=
  by
  have h1 : req.total = 30 := by sorry
  have h2 : req.electropop = req.total / 2 := by sorry
  have h3 : req.dance = req.electropop / 3 := by sorry
  have h4 : req.oldies = req.rock - 3 := by sorry
  have h5 : req.dj_choice = req.oldies / 2 := by sorry
  have h6 : req.rap = 2 := by sorry
  have h7 : req.total = req.electropop + req.dance + req.rap + req.rock + req.oldies + req.dj_choice := by sorry
  sorry

end NUMINAMATH_CALUDE_rock_song_requests_l1182_118299


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1182_118278

/-- The value of 'a' for a parabola y^2 = ax (a > 0) with a point P(3/2, y₀) on the parabola,
    where the distance from P to the focus is 2. -/
theorem parabola_focus_distance (a : ℝ) (y₀ : ℝ) : 
  a > 0 ∧ y₀^2 = a * (3/2) ∧ ((3/2 - (-a/4))^2 + y₀^2)^(1/2) = 2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1182_118278


namespace NUMINAMATH_CALUDE_inequality_proof_l1182_118291

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_3 : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1182_118291


namespace NUMINAMATH_CALUDE_rectangle_problem_l1182_118213

theorem rectangle_problem (A B C D E F G H I : ℕ) : 
  (A * B = D * E) →  -- Areas of ABCD and DEFG are equal
  (A * B = C * H) →  -- Areas of ABCD and CEIH are equal
  (B = 43) →         -- BC = 43
  (D > E) →          -- Assume DG > DE
  (D = 1892) →       -- DG = 1892
  True               -- Conclusion (to be proved)
  := by sorry

end NUMINAMATH_CALUDE_rectangle_problem_l1182_118213


namespace NUMINAMATH_CALUDE_expression_evaluation_l1182_118277

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1182_118277


namespace NUMINAMATH_CALUDE_test_point_value_l1182_118204

theorem test_point_value
  (total_points : ℕ)
  (total_questions : ℕ)
  (two_point_questions : ℕ)
  (other_type_questions : ℕ)
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : other_type_questions = 10)
  (h4 : two_point_questions + other_type_questions = total_questions)
  (h5 : 2 * two_point_questions + other_type_questions * (total_points - 2 * two_point_questions) / other_type_questions = total_points) :
  (total_points - 2 * two_point_questions) / other_type_questions = 4 :=
by sorry

end NUMINAMATH_CALUDE_test_point_value_l1182_118204


namespace NUMINAMATH_CALUDE_peanut_butter_probability_l1182_118265

def jenny_peanut_butter : ℕ := 40
def jenny_chocolate_chip : ℕ := 50
def marcus_peanut_butter : ℕ := 30
def marcus_lemon : ℕ := 20

def total_cookies : ℕ := jenny_peanut_butter + jenny_chocolate_chip + marcus_peanut_butter + marcus_lemon
def peanut_butter_cookies : ℕ := jenny_peanut_butter + marcus_peanut_butter

theorem peanut_butter_probability :
  (peanut_butter_cookies : ℚ) / (total_cookies : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_probability_l1182_118265


namespace NUMINAMATH_CALUDE_nine_five_dollar_bills_equal_45_dollars_l1182_118292

/-- The total value in dollars when a person has a certain number of five-dollar bills -/
def total_value (num_bills : ℕ) : ℕ := 5 * num_bills

/-- Theorem: If a person has 9 five-dollar bills, they have a total of 45 dollars -/
theorem nine_five_dollar_bills_equal_45_dollars :
  total_value 9 = 45 := by sorry

end NUMINAMATH_CALUDE_nine_five_dollar_bills_equal_45_dollars_l1182_118292


namespace NUMINAMATH_CALUDE_cosine_sine_sum_zero_l1182_118288

theorem cosine_sine_sum_zero (x : ℝ) 
  (h : Real.cos (π / 6 - x) = -Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_zero_l1182_118288
