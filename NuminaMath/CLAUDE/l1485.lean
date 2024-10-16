import Mathlib

namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l1485_148598

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The set of distinct positive factors of a positive integer -/
def factors (n : ℕ+) : Finset ℕ := sorry

theorem smallest_with_eight_factors : 
  (∀ m : ℕ+, m < 24 → num_factors m ≠ 8) ∧ 
  num_factors 24 = 8 ∧
  factors 24 = {1, 2, 3, 4, 6, 8, 12, 24} := by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l1485_148598


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1485_148559

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_positive : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1485_148559


namespace NUMINAMATH_CALUDE_solve_for_q_l1485_148572

theorem solve_for_q (n m q : ℚ) 
  (h1 : 7/9 = n/108)
  (h2 : 7/9 = (m+n)/126)
  (h3 : 7/9 = (q-m)/162) : 
  q = 140 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l1485_148572


namespace NUMINAMATH_CALUDE_smith_family_buffet_cost_l1485_148511

/-- Calculates the total cost of a family buffet given the pricing structure and family composition. -/
def familyBuffetCost (adultPrice childPrice : ℚ) (seniorDiscount : ℚ) 
  (numAdults numChildren numSeniors : ℕ) : ℚ :=
  (numAdults : ℚ) * adultPrice + 
  (numChildren : ℚ) * childPrice + 
  (numSeniors : ℚ) * (adultPrice * (1 - seniorDiscount))

/-- Theorem stating that Mr. Smith's family buffet cost is $162 -/
theorem smith_family_buffet_cost : 
  familyBuffetCost 30 15 (1/10) 3 3 1 = 162 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_buffet_cost_l1485_148511


namespace NUMINAMATH_CALUDE_exists_growth_rate_unique_growth_rate_l1485_148543

/-- Represents the average annual growth rate of Fujian's regional GDP from 2020 to 2022 -/
def average_annual_growth_rate (x : ℝ) : Prop :=
  43903.89 * (1 + x)^2 = 53109.85

/-- The initial GDP of Fujian in 2020 (in billion yuan) -/
def initial_gdp : ℝ := 43903.89

/-- The GDP of Fujian in 2022 (in billion yuan) -/
def final_gdp : ℝ := 53109.85

/-- Theorem stating that there exists an average annual growth rate satisfying the equation -/
theorem exists_growth_rate : ∃ x : ℝ, average_annual_growth_rate x :=
  sorry

/-- Theorem stating that the average annual growth rate is unique -/
theorem unique_growth_rate : ∀ x y : ℝ, average_annual_growth_rate x → average_annual_growth_rate y → x = y :=
  sorry

end NUMINAMATH_CALUDE_exists_growth_rate_unique_growth_rate_l1485_148543


namespace NUMINAMATH_CALUDE_probability_of_red_in_C_l1485_148503

-- Define the initial configuration of balls in each box
def box_A : ℕ × ℕ := (2, 1)  -- (red, yellow)
def box_B : ℕ × ℕ := (1, 2)  -- (red, yellow)
def box_C : ℕ × ℕ := (1, 1)  -- (red, yellow)

-- Define the process of transferring balls
def transfer_process : (ℕ × ℕ) → (ℕ × ℕ) → (ℕ × ℕ) → ℚ := sorry

-- Theorem statement
theorem probability_of_red_in_C :
  transfer_process box_A box_B box_C = 17/36 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_in_C_l1485_148503


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1485_148590

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1485_148590


namespace NUMINAMATH_CALUDE_points_symmetric_wrt_origin_l1485_148505

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Point A has coordinates (3, 4) -/
def A : ℝ × ℝ := (3, 4)

/-- Point B has coordinates (-3, -4) -/
def B : ℝ × ℝ := (-3, -4)

theorem points_symmetric_wrt_origin : symmetric_wrt_origin A B := by
  sorry

end NUMINAMATH_CALUDE_points_symmetric_wrt_origin_l1485_148505


namespace NUMINAMATH_CALUDE_largest_three_digit_base_7_is_342_l1485_148571

/-- The largest decimal number represented by a three-digit base-7 number -/
def largest_three_digit_base_7 : ℕ := 342

/-- The base of the number system -/
def base : ℕ := 7

/-- The number of digits -/
def num_digits : ℕ := 3

/-- Theorem: The largest decimal number represented by a three-digit base-7 number is 342 -/
theorem largest_three_digit_base_7_is_342 :
  largest_three_digit_base_7 = (base ^ num_digits - 1) := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_base_7_is_342_l1485_148571


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1485_148594

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → m ≥ 27720) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l1485_148594


namespace NUMINAMATH_CALUDE_quadratic_points_order_l1485_148562

/-- Given a quadratic function f(x) = x² - 4x - m, prove that the y-coordinates
    of the points (-1, y₃), (3, y₂), and (2, y₁) on this function satisfy y₃ > y₂ > y₁ -/
theorem quadratic_points_order (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - m
  let y₁ : ℝ := f 2
  let y₂ : ℝ := f 3
  let y₃ : ℝ := f (-1)
  y₃ > y₂ ∧ y₂ > y₁ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_points_order_l1485_148562


namespace NUMINAMATH_CALUDE_gcd_228_2010_l1485_148568

theorem gcd_228_2010 : Nat.gcd 228 2010 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_2010_l1485_148568


namespace NUMINAMATH_CALUDE_r_exceeds_s_by_two_l1485_148533

theorem r_exceeds_s_by_two (x y r s : ℝ) : 
  3 * x + 2 * y = 16 →
  5 * x + 3 * y = 26 →
  r = x →
  s = y →
  r - s = 2 := by
sorry

end NUMINAMATH_CALUDE_r_exceeds_s_by_two_l1485_148533


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l1485_148584

/-- The equation of the curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

/-- The equation of the curve in Cartesian coordinates -/
def cartesian_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) - y = 1

/-- The definition of a hyperbola in Cartesian coordinates -/
def is_hyperbola (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ x y, f (x, y) = a * x^2 + b * y^2 + c * x * y + d * x + e * y

theorem curve_is_hyperbola :
  ∃ f : ℝ × ℝ → ℝ, (∀ x y, f (x, y) = 0 ↔ cartesian_equation x y) ∧ is_hyperbola f :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l1485_148584


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l1485_148589

theorem sqrt_18_minus_sqrt_8_equals_sqrt_2 : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l1485_148589


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1485_148563

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1485_148563


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l1485_148566

/-- Given the total sales and average price of ping pong rackets, prove the number of pairs sold. -/
theorem ping_pong_rackets_sold (total_sales : ℝ) (avg_price : ℝ) (h1 : total_sales = 686) (h2 : avg_price = 9.8) :
  total_sales / avg_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l1485_148566


namespace NUMINAMATH_CALUDE_packet_weight_l1485_148521

-- Define constants
def pounds_per_ton : ℚ := 2200
def ounces_per_pound : ℚ := 16
def bag_capacity_tons : ℚ := 13
def num_packets : ℚ := 1760

-- Define the theorem
theorem packet_weight :
  let total_weight := bag_capacity_tons * pounds_per_ton
  let weight_per_packet := total_weight / num_packets
  weight_per_packet = 16.25 := by
sorry

end NUMINAMATH_CALUDE_packet_weight_l1485_148521


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1485_148579

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 9sin²B = 4sin²A and cosC = 1/4, then c/a = √(10)/3 -/
theorem triangle_side_ratio (a b c A B C : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 9 * (sin B)^2 = 4 * (sin A)^2) (h5 : cos C = 1/4) :
  c / a = sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1485_148579


namespace NUMINAMATH_CALUDE_lindas_additional_dimes_l1485_148520

/-- The number of additional dimes Linda's mother gives her -/
def additional_dimes : ℕ := 2

/-- The initial number of dimes Linda has -/
def initial_dimes : ℕ := 2

/-- The initial number of quarters Linda has -/
def initial_quarters : ℕ := 6

/-- The initial number of nickels Linda has -/
def initial_nickels : ℕ := 5

/-- The number of additional quarters Linda's mother gives her -/
def additional_quarters : ℕ := 10

/-- The total number of coins Linda has after receiving additional coins -/
def total_coins : ℕ := 35

theorem lindas_additional_dimes :
  initial_dimes + initial_quarters + initial_nickels +
  additional_dimes + additional_quarters + 2 * initial_nickels = total_coins :=
by sorry

end NUMINAMATH_CALUDE_lindas_additional_dimes_l1485_148520


namespace NUMINAMATH_CALUDE_square_identities_l1485_148510

theorem square_identities (a b c : ℝ) : 
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ 
  ((a - b)^2 = a^2 - 2*a*b + b^2) ∧ 
  ((a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) := by
  sorry

end NUMINAMATH_CALUDE_square_identities_l1485_148510


namespace NUMINAMATH_CALUDE_fly_path_on_cone_l1485_148539

/-- A right circular cone -/
structure Cone where
  base_radius : ℝ
  height : ℝ

/-- A point on the surface of a cone -/
structure ConePoint where
  distance_from_vertex : ℝ

/-- The shortest distance between two points on the surface of a cone -/
def shortest_distance (c : Cone) (p1 p2 : ConePoint) : ℝ := sorry

/-- The theorem statement -/
theorem fly_path_on_cone :
  let c : Cone := { base_radius := 600, height := 200 * Real.sqrt 7 }
  let p1 : ConePoint := { distance_from_vertex := 125 }
  let p2 : ConePoint := { distance_from_vertex := 375 * Real.sqrt 2 }
  shortest_distance c p1 p2 = 625 := by sorry

end NUMINAMATH_CALUDE_fly_path_on_cone_l1485_148539


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1485_148504

/-- Given three nonconstant geometric sequences with different common ratios,
    if a certain condition holds, then the sum of their common ratios is 1 + 2√2 -/
theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ c₂ c₃ m n o : ℝ) 
  (hm : m ≠ 1) (hn : n ≠ 1) (ho : o ≠ 1)  -- nonconstant sequences
  (hm_ne_n : m ≠ n) (hm_ne_o : m ≠ o) (hn_ne_o : n ≠ o)  -- different ratios
  (ha₂ : a₂ = k * m) (ha₃ : a₃ = k * m^2)  -- first sequence
  (hb₂ : b₂ = k * n) (hb₃ : b₃ = k * n^2)  -- second sequence
  (hc₂ : c₂ = k * o) (hc₃ : c₃ = k * o^2)  -- third sequence
  (heq : a₃ - b₃ + c₃ = 2 * (a₂ - b₂ + c₂))  -- given condition
  : m + n + o = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1485_148504


namespace NUMINAMATH_CALUDE_peters_horses_food_l1485_148534

/-- The amount of food needed for Peter's horses over 5 days -/
def food_needed (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) 
                (grain_per_meal : ℕ) (grain_meals_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_horses * (oats_per_meal * oats_meals_per_day + grain_per_meal * grain_meals_per_day) * num_days

/-- Theorem stating the total amount of food needed for Peter's horses -/
theorem peters_horses_food : 
  food_needed 6 5 3 4 2 5 = 690 := by
  sorry

end NUMINAMATH_CALUDE_peters_horses_food_l1485_148534


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l1485_148592

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 4*p^3 + 7*p^2 - 3*p + 2 = 0) →
  (q^4 - 4*q^3 + 7*q^2 - 3*q + 2 = 0) →
  (r^4 - 4*r^3 + 7*r^2 - 3*r + 2 = 0) →
  (s^4 - 4*s^3 + 7*s^2 - 3*s + 2 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 7/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l1485_148592


namespace NUMINAMATH_CALUDE_min_value_expression_l1485_148547

theorem min_value_expression (a b c d : ℝ) (h1 : b > c) (h2 : c > d) (h3 : d > a) (h4 : b ≠ 0) :
  (a + b)^2 + (b - c)^2 + (c - d)^2 + (d - a)^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1485_148547


namespace NUMINAMATH_CALUDE_spoke_forms_surface_l1485_148502

/-- Represents a spoke in a bicycle wheel -/
structure Spoke :=
  (length : ℝ)
  (angle : ℝ)

/-- Represents a rotating bicycle wheel -/
structure RotatingWheel :=
  (radius : ℝ)
  (angular_velocity : ℝ)
  (spokes : List Spoke)

/-- Represents the surface formed by rotating spokes -/
def SurfaceFormedBySpokes (wheel : RotatingWheel) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating that a rotating spoke forms a surface -/
theorem spoke_forms_surface (wheel : RotatingWheel) (s : Spoke) 
  (h : s ∈ wheel.spokes) : 
  ∃ (surface : Set (ℝ × ℝ × ℝ)), 
    surface = SurfaceFormedBySpokes wheel ∧ 
    (∀ t : ℝ, ∃ p : ℝ × ℝ × ℝ, p ∈ surface) :=
sorry

end NUMINAMATH_CALUDE_spoke_forms_surface_l1485_148502


namespace NUMINAMATH_CALUDE_tangent_line_and_triangle_area_l1485_148518

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Theorem statement
theorem tangent_line_and_triangle_area :
  let P : ℝ × ℝ := (1, -1)
  -- Condition: P is on the graph of f
  (f P.1 = P.2) →
  -- Claim 1: Equation of the tangent line
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ x - y - 2 = 0) ∧
  -- Claim 2: Area of the triangle
  (∃ A : ℝ, A = 2 ∧
    ∀ x₁ y₁ x₂ y₂,
      (x₁ - y₁ - 2 = 0 ∧ y₁ = 0) →
      (x₂ - y₂ - 2 = 0 ∧ x₂ = 0) →
      A = (1/2) * x₁ * (-y₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_triangle_area_l1485_148518


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1485_148528

theorem more_girls_than_boys :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / (girls : ℚ) = 3 / 5 →
  boys + girls = 16 →
  girls - boys = 4 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1485_148528


namespace NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l1485_148527

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The initial DateTime (midnight on February 1, 2022) -/
def initialDateTime : DateTime :=
  { year := 2022, month := 2, day := 1, hour := 0, minute := 0 }

/-- The final DateTime after adding 1553 minutes -/
def finalDateTime : DateTime :=
  addMinutes initialDateTime 1553

/-- Theorem stating that 1553 minutes after midnight on February 1, 2022 is February 2 at 1:53 AM -/
theorem minutes_after_midnight_theorem :
  finalDateTime = { year := 2022, month := 2, day := 2, hour := 1, minute := 53 } :=
  sorry

end NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l1485_148527


namespace NUMINAMATH_CALUDE_curve_L_properties_l1485_148551

/-- Definition of the curve L -/
def L (p : ℕ) (x y : ℤ) : Prop := 4 * y^2 = (x - p) * p

/-- A prime number is odd if it's not equal to 2 -/
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p ≠ 2

theorem curve_L_properties (p : ℕ) (hp : is_odd_prime p) :
  (∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ ∀ (x y : ℤ), (x, y) ∈ S → y ≠ 0 ∧ L p x y) ∧
  (∀ (x y : ℤ), L p x y → ¬ ∃ (d : ℤ), d^2 = x^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_curve_L_properties_l1485_148551


namespace NUMINAMATH_CALUDE_average_sale_l1485_148570

def sales : List ℕ := [5420, 5660, 6200, 6350, 6500]
def projected_sale : ℕ := 6470

theorem average_sale :
  (sales.sum + projected_sale) / (sales.length + 1) = 6100 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_l1485_148570


namespace NUMINAMATH_CALUDE_parallel_implies_m_half_perpendicular_implies_m_seven_fourths_l1485_148576

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define vector operations
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def AB : ℝ × ℝ := vector_sub OB OA
def BC (m : ℝ) : ℝ × ℝ := vector_sub (OC m) OB
def AC (m : ℝ) : ℝ × ℝ := vector_sub (OC m) OA

-- Define parallel and perpendicular conditions
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorems
theorem parallel_implies_m_half :
  parallel AB (BC (1/2)) :=
sorry

theorem perpendicular_implies_m_seven_fourths :
  perpendicular AB (AC (7/4)) :=
sorry

end NUMINAMATH_CALUDE_parallel_implies_m_half_perpendicular_implies_m_seven_fourths_l1485_148576


namespace NUMINAMATH_CALUDE_smallest_n_factor_smallest_n_is_75_l1485_148574

theorem smallest_n_factor (n : ℕ+) : 
  (5^2 ∣ n * (2^5) * (6^2) * (7^3)) ∧ 
  (3^3 ∣ n * (2^5) * (6^2) * (7^3)) →
  n ≥ 75 :=
by sorry

theorem smallest_n_is_75 : 
  ∃ (n : ℕ+), n = 75 ∧ 
  (5^2 ∣ n * (2^5) * (6^2) * (7^3)) ∧ 
  (3^3 ∣ n * (2^5) * (6^2) * (7^3)) ∧
  ∀ (m : ℕ+), m < 75 → 
    ¬((5^2 ∣ m * (2^5) * (6^2) * (7^3)) ∧ 
      (3^3 ∣ m * (2^5) * (6^2) * (7^3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_factor_smallest_n_is_75_l1485_148574


namespace NUMINAMATH_CALUDE_circle_radius_l1485_148523

theorem circle_radius (x y : ℝ) (h : x + y = 100 * Real.pi) :
  let r := Real.sqrt 101 - 1
  x = Real.pi * r^2 ∧ y = 2 * Real.pi * r := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1485_148523


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1485_148542

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 30 →
  red = 9 →
  purple = 3 →
  prob = 88/100 →
  prob = (white + green + (total - white - green - red - purple)) / total →
  total - white - green - red - purple = 8 :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1485_148542


namespace NUMINAMATH_CALUDE_loading_dock_problem_l1485_148508

/-- Proves that given the conditions of the loading dock problem, 
    the fraction of boxes loaded by each night crew worker 
    compared to each day crew worker is 5/14 -/
theorem loading_dock_problem 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (h1 : night_crew = (4 : ℚ) / 5 * day_crew) 
  (h2 : (5 : ℚ) / 7 = day_crew_boxes / total_boxes) 
  (day_crew_boxes : ℚ) 
  (night_crew_boxes : ℚ) 
  (total_boxes : ℚ) 
  (h3 : total_boxes = day_crew_boxes + night_crew_boxes) 
  (h4 : total_boxes ≠ 0) 
  (h5 : day_crew ≠ 0) 
  (h6 : night_crew ≠ 0) :
  (night_crew_boxes / night_crew) / (day_crew_boxes / day_crew) = (5 : ℚ) / 14 := by
  sorry

end NUMINAMATH_CALUDE_loading_dock_problem_l1485_148508


namespace NUMINAMATH_CALUDE_f_2017_neg_two_eq_three_fifths_l1485_148519

def f (x : ℚ) : ℚ := (1 + x) / (1 - 3*x)

def f_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => f ∘ f_n n

theorem f_2017_neg_two_eq_three_fifths :
  f_n 2017 (-2) = 3/5 := by sorry

end NUMINAMATH_CALUDE_f_2017_neg_two_eq_three_fifths_l1485_148519


namespace NUMINAMATH_CALUDE_problem_solution_l1485_148552

theorem problem_solution (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2003 + b^2004 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1485_148552


namespace NUMINAMATH_CALUDE_fraction_comparison_l1485_148597

theorem fraction_comparison (x : ℝ) (hx : x > 0) :
  ∀ n : ℕ, (x^n + 1) / (x^(n+1) + 1) > (x^(n+1) + 1) / (x^(n+2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1485_148597


namespace NUMINAMATH_CALUDE_disk_arrangement_sum_l1485_148517

theorem disk_arrangement_sum (n : ℕ) (r : ℝ) :
  n = 8 →
  r > 0 →
  r = 2 - Real.sqrt 2 →
  ∃ (a b c : ℕ), 
    c = 2 ∧
    n * (π * r^2) = π * (a - b * Real.sqrt c) ∧
    a + b + c = 82 :=
sorry

end NUMINAMATH_CALUDE_disk_arrangement_sum_l1485_148517


namespace NUMINAMATH_CALUDE_count_propositions_with_connectives_l1485_148537

-- Define a proposition type
inductive Proposition
| feb14_2010 : Proposition
| multiple_10_5 : Proposition
| trapezoid_rectangle : Proposition

-- Define a function to check if a proposition uses a logical connective
def uses_logical_connective (p : Proposition) : Bool :=
  match p with
  | Proposition.feb14_2010 => true  -- Uses "and"
  | Proposition.multiple_10_5 => false
  | Proposition.trapezoid_rectangle => true  -- Uses "not"

-- Define the list of propositions
def propositions : List Proposition :=
  [Proposition.feb14_2010, Proposition.multiple_10_5, Proposition.trapezoid_rectangle]

-- Theorem statement
theorem count_propositions_with_connectives :
  (propositions.filter uses_logical_connective).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_propositions_with_connectives_l1485_148537


namespace NUMINAMATH_CALUDE_farm_cows_count_l1485_148509

/-- The total number of cows on the farm -/
def total_cows : ℕ := 140

/-- The percentage of cows with a red spot -/
def red_spot_percentage : ℚ := 40 / 100

/-- The percentage of cows without a red spot that have a blue spot -/
def blue_spot_percentage : ℚ := 25 / 100

/-- The number of cows with no spot -/
def no_spot_cows : ℕ := 63

theorem farm_cows_count :
  (total_cows : ℚ) * (1 - red_spot_percentage) * (1 - blue_spot_percentage) = no_spot_cows :=
sorry

end NUMINAMATH_CALUDE_farm_cows_count_l1485_148509


namespace NUMINAMATH_CALUDE_projection_correct_l1485_148558

/-- Given vectors u and t, prove that the projection of u onto t is correct. -/
theorem projection_correct (u t : ℝ × ℝ) : 
  u = (4, -3) → t = (-6, 8) → 
  let proj := ((u.1 * t.1 + u.2 * t.2) / (t.1 * t.1 + t.2 * t.2)) • t
  proj.1 = 288 / 100 ∧ proj.2 = -384 / 100 := by
  sorry

end NUMINAMATH_CALUDE_projection_correct_l1485_148558


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1485_148535

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def real_axis_length (a : ℝ) : ℝ := 2 * a

def imaginary_axis_length (b : ℝ) : ℝ := 2 * b

def asymptote_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : real_axis_length a = 2 * Real.sqrt 2)
  (h4 : imaginary_axis_length b = 2) :
  ∀ (x y : ℝ), asymptote_equation a b x y ↔ y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1485_148535


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1485_148599

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (x - 3) / x ≥ 0 ↔ x < 0 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1485_148599


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_m_l1485_148573

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

-- Theorem statement
theorem m_intersect_n_equals_m : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_m_l1485_148573


namespace NUMINAMATH_CALUDE_periodic_function_extension_l1485_148501

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_smallest_period : ∀ p, 0 < p → p < 2 → ¬ is_periodic f p)
  (h_def : ∀ x, 0 ≤ x → x < 2 → f x = x^3 - x) :
  ∀ x, -2 ≤ x → x < 0 → f x = x^3 + 6*x^2 + 11*x + 6 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_extension_l1485_148501


namespace NUMINAMATH_CALUDE_five_people_handshakes_l1485_148595

/-- The number of handshakes when n people meet, where each pair shakes hands exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: When 5 people meet, they shake hands a total of 10 times -/
theorem five_people_handshakes : handshakes 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_people_handshakes_l1485_148595


namespace NUMINAMATH_CALUDE_some_number_value_l1485_148588

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1485_148588


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l1485_148564

/-- The probability mass function for a Binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- Theorem: For a random variable ξ following Binomial distribution B(3, 1/3), P(ξ=2) = 2/9 -/
theorem binomial_probability_two_successes :
  binomial_pmf 3 (1/3 : ℝ) 2 = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l1485_148564


namespace NUMINAMATH_CALUDE_track_length_is_50_l1485_148514

/-- Calculates the length of a running track given weekly distance, days per week, and loops per day -/
def track_length (weekly_distance : ℕ) (days_per_week : ℕ) (loops_per_day : ℕ) : ℕ :=
  weekly_distance / (days_per_week * loops_per_day)

/-- Proves that given the specified conditions, the track length is 50 meters -/
theorem track_length_is_50 : 
  track_length 3500 7 10 = 50 := by
  sorry

#eval track_length 3500 7 10

end NUMINAMATH_CALUDE_track_length_is_50_l1485_148514


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l1485_148565

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are not collinear -/
theorem vectors_not_collinear (a b : Fin 3 → ℝ) 
  (ha : a = ![1, 4, -2])
  (hb : b = ![1, 1, -1])
  (c₁ : Fin 3 → ℝ)
  (hc₁ : c₁ = a + b)
  (c₂ : Fin 3 → ℝ)
  (hc₂ : c₂ = 4 • a + 2 • b) :
  ¬ ∃ (k : ℝ), c₁ = k • c₂ := by
sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l1485_148565


namespace NUMINAMATH_CALUDE_integer_division_property_l1485_148583

theorem integer_division_property (n : ℕ) : 
  100 ≤ n ∧ n ≤ 1997 →
  (∃ k : ℕ, (2^n + 2 : ℕ) = k * n) ↔ n ∈ ({66, 198, 398, 798} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_integer_division_property_l1485_148583


namespace NUMINAMATH_CALUDE_hotel_towels_l1485_148544

theorem hotel_towels (rooms : ℕ) (people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : rooms = 20)
  (h2 : people_per_room = 5)
  (h3 : towels_per_person = 3) :
  rooms * people_per_room * towels_per_person = 300 :=
by sorry

end NUMINAMATH_CALUDE_hotel_towels_l1485_148544


namespace NUMINAMATH_CALUDE_village_population_equality_l1485_148522

/-- The number of years it takes for the populations of two villages to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (x_decrease + y_increase)

theorem village_population_equality :
  years_to_equal_population 68000 1200 42000 800 = 13 := by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l1485_148522


namespace NUMINAMATH_CALUDE_stating_gray_cube_count_gray_cube_count_3x3x3_gray_cube_count_5x5x5_l1485_148561

/-- 
Represents the number of 1x1x1 cubes with a specific number of gray faces 
in an nxnxn cube where all outer faces are painted gray.
-/
def grayCubes (n : ℕ) : ℕ × ℕ :=
  (6 * (n - 2)^2, (n - 2)^3)

/-- 
Theorem stating the correct number of 1x1x1 cubes with exactly one gray face 
and with no gray faces in an nxnxn cube where all outer faces are painted gray.
-/
theorem gray_cube_count (n : ℕ) (h : n ≥ 3) : 
  grayCubes n = (6 * (n - 2)^2, (n - 2)^3) := by
  sorry

/-- 
Corollary for the specific case of a 3x3x3 cube, giving the number of cubes 
with exactly one gray face and exactly two gray faces.
-/
theorem gray_cube_count_3x3x3 : 
  (grayCubes 3).1 = 6 ∧ 12 = 12 := by
  sorry

/-- 
Corollary for the specific case of a 5x5x5 cube, giving the number of cubes 
with exactly one gray face and with no gray faces.
-/
theorem gray_cube_count_5x5x5 : 
  (grayCubes 5).1 = 54 ∧ (grayCubes 5).2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_stating_gray_cube_count_gray_cube_count_3x3x3_gray_cube_count_5x5x5_l1485_148561


namespace NUMINAMATH_CALUDE_smallest_variance_l1485_148593

def minimumVariance (n : ℕ) (s : Finset ℝ) : Prop :=
  n ≥ 2 ∧
  s.card = n ∧
  (0 : ℝ) ∈ s ∧
  (1 : ℝ) ∈ s ∧
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 1) →
  ∀ ε > 0, ∃ (v : ℝ), v ≥ 1 / (2 * n) ∧
    v = (s.sum (λ x => (x - s.sum (λ y => y) / n) ^ 2)) / n

theorem smallest_variance (n : ℕ) (s : Finset ℝ) (h : minimumVariance n s) :
  ∃ (v : ℝ), v = 1 / (2 * n) ∧
    v = (s.sum (λ x => (x - s.sum (λ y => y) / n) ^ 2)) / n :=
sorry

end NUMINAMATH_CALUDE_smallest_variance_l1485_148593


namespace NUMINAMATH_CALUDE_cos_difference_x1_x2_l1485_148538

theorem cos_difference_x1_x2 (x₁ x₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < π)
  (h4 : Real.sin (2 * x₁ - π / 3) = 4 / 5)
  (h5 : Real.sin (2 * x₂ - π / 3) = 4 / 5) :
  Real.cos (x₁ - x₂) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_x1_x2_l1485_148538


namespace NUMINAMATH_CALUDE_compute_fraction_power_l1485_148529

theorem compute_fraction_power : 8 * (2 / 3)^4 = 128 / 81 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l1485_148529


namespace NUMINAMATH_CALUDE_money_redistribution_l1485_148581

theorem money_redistribution (younger_money : ℝ) :
  let elder_money := 1.25 * younger_money
  let total_money := younger_money + elder_money
  let equal_share := total_money / 2
  let transfer_amount := equal_share - younger_money
  (transfer_amount / elder_money) = 0.1 := by
sorry

end NUMINAMATH_CALUDE_money_redistribution_l1485_148581


namespace NUMINAMATH_CALUDE_octagon_arc_length_l1485_148500

/-- The length of the arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (s : ℝ) (h : s = 4) :
  let R := s / (2 * Real.sin (π / 8))
  let C := 2 * π * R
  C / 8 = (Real.sqrt 2 * π) / 2 := by sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l1485_148500


namespace NUMINAMATH_CALUDE_carpet_square_size_l1485_148556

theorem carpet_square_size (floor_length : ℝ) (floor_width : ℝ) 
  (total_cost : ℝ) (square_cost : ℝ) :
  floor_length = 24 →
  floor_width = 64 →
  total_cost = 576 →
  square_cost = 24 →
  ∃ (square_side : ℝ),
    square_side = 8 ∧
    (floor_length * floor_width) / (square_side * square_side) * square_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_carpet_square_size_l1485_148556


namespace NUMINAMATH_CALUDE_rope_ratio_proof_l1485_148560

theorem rope_ratio_proof (total_length shorter_length longer_length : ℝ) 
  (h1 : total_length = 60)
  (h2 : shorter_length = 20)
  (h3 : longer_length = total_length - shorter_length) :
  longer_length / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_rope_ratio_proof_l1485_148560


namespace NUMINAMATH_CALUDE_triangular_gcd_bound_l1485_148554

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- Theorem: The GCD of 6T_n and n-1 is at most 3, and this bound is achievable -/
theorem triangular_gcd_bound (n : ℕ+) : 
  ∃ (m : ℕ+), Nat.gcd (6 * T m) (m - 1) = 3 ∧ 
  ∀ (k : ℕ+), Nat.gcd (6 * T k) (k - 1) ≤ 3 := by
  sorry

#check triangular_gcd_bound

end NUMINAMATH_CALUDE_triangular_gcd_bound_l1485_148554


namespace NUMINAMATH_CALUDE_f_at_2_equals_neg_26_l1485_148596

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_at_2_equals_neg_26 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_f_at_2_equals_neg_26_l1485_148596


namespace NUMINAMATH_CALUDE_factorization_identity_l1485_148530

theorem factorization_identity (a b : ℝ) : a^2 + a*b = a*(a + b) := by sorry

end NUMINAMATH_CALUDE_factorization_identity_l1485_148530


namespace NUMINAMATH_CALUDE_log_cube_exp_inequality_l1485_148512

theorem log_cube_exp_inequality (x : ℝ) (h : 0 < x ∧ x < 1) :
  Real.log x / Real.log 3 < x^3 ∧ x^3 < 3^x := by
  sorry

end NUMINAMATH_CALUDE_log_cube_exp_inequality_l1485_148512


namespace NUMINAMATH_CALUDE_retirement_sum_is_70_l1485_148586

/-- Represents the retirement policy of a company -/
structure RetirementPolicy where
  hireYear : Nat
  hireAge : Nat
  retirementYear : Nat
  retirementSum : Nat

/-- Theorem: The required total of age and years of employment for retirement is 70 -/
theorem retirement_sum_is_70 (policy : RetirementPolicy) 
  (h1 : policy.hireYear = 1987)
  (h2 : policy.hireAge = 32)
  (h3 : policy.retirementYear = 2006) :
  policy.retirementSum = 70 := by
  sorry

#check retirement_sum_is_70

end NUMINAMATH_CALUDE_retirement_sum_is_70_l1485_148586


namespace NUMINAMATH_CALUDE_student_marks_l1485_148557

theorem student_marks (M P C : ℤ) 
  (h1 : C = P + 20) 
  (h2 : (M + C) / 2 = 45) : 
  M + P = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_l1485_148557


namespace NUMINAMATH_CALUDE_train_commute_additional_time_l1485_148525

/-- Proves that the additional time for train commute is 10.5 minutes -/
theorem train_commute_additional_time 
  (distance_to_work : Real) 
  (walking_speed : Real) 
  (train_speed : Real) 
  (walking_time_difference : Real) :
  distance_to_work = 1.5 →
  walking_speed = 3 →
  train_speed = 20 →
  walking_time_difference = 15 →
  ∃ x : Real,
    60 * distance_to_work / walking_speed = 
    60 * distance_to_work / train_speed + x + walking_time_difference ∧
    x = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_train_commute_additional_time_l1485_148525


namespace NUMINAMATH_CALUDE_intersection_when_m_is_2_subset_iff_m_leq_1_l1485_148550

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (3 - 2*x) ∧ x ∈ Set.Icc (-13/2) (3/2)}
def B (m : ℝ) : Set ℝ := Set.Icc (1 - m) (m + 1)

-- Statement 1: When m = 2, A ∩ B = [0, 3]
theorem intersection_when_m_is_2 : A ∩ B 2 = Set.Icc 0 3 := by sorry

-- Statement 2: B ⊆ A if and only if m ≤ 1
theorem subset_iff_m_leq_1 : ∀ m, B m ⊆ A ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_2_subset_iff_m_leq_1_l1485_148550


namespace NUMINAMATH_CALUDE_turtle_conservation_l1485_148545

theorem turtle_conservation (G H L : ℕ) : 
  G = 800 → H = 2 * G → L = 3 * G → G + H + L = 4800 := by
  sorry

end NUMINAMATH_CALUDE_turtle_conservation_l1485_148545


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l1485_148553

def matrix_det (x y : ℝ) : ℝ := x * y - 6

theorem det_2x2_matrix (x y : ℝ) :
  Matrix.det ![![x, 2], ![3, y]] = matrix_det x y := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l1485_148553


namespace NUMINAMATH_CALUDE_percentage_problem_l1485_148585

theorem percentage_problem : 
  let percentage : ℝ := 12
  let total : ℝ := 160
  let given_percentage : ℝ := 38
  let given_total : ℝ := 80
  let difference : ℝ := 11.2
  (given_percentage / 100) * given_total - (percentage / 100) * total = difference
  := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1485_148585


namespace NUMINAMATH_CALUDE_consecutive_digits_divisible_by_11_l1485_148546

/-- Given four consecutive digits x, x+1, x+2, x+3, the number formed by
    interchanging the first two digits of (1000x + 100(x+1) + 10(x+2) + (x+3))
    is divisible by 11 for any integer x. -/
theorem consecutive_digits_divisible_by_11 (x : ℤ) :
  ∃ k : ℤ, (1000 * (x + 1) + 100 * x + 10 * (x + 2) + (x + 3)) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_divisible_by_11_l1485_148546


namespace NUMINAMATH_CALUDE_sum_of_powers_zero_l1485_148582

theorem sum_of_powers_zero : -(-1)^2006 - (-1)^2007 - 1^2008 - (-1)^2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_zero_l1485_148582


namespace NUMINAMATH_CALUDE_union_intersection_equality_union_subset_l1485_148548

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Theorem for part (1)
theorem union_intersection_equality (a : ℝ) :
  A ∪ B a = A ∩ B a → a = 1 := by sorry

-- Theorem for part (2)
theorem union_subset (a : ℝ) :
  A ∪ B a = A → a ≤ -1 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_union_intersection_equality_union_subset_l1485_148548


namespace NUMINAMATH_CALUDE_student_grade_average_l1485_148516

theorem student_grade_average (grade1 grade2 : ℝ) 
  (h1 : grade1 = 70)
  (h2 : grade2 = 80) : 
  ∃ (grade3 : ℝ), (grade1 + grade2 + grade3) / 3 = grade3 ∧ grade3 = 75 := by
sorry

end NUMINAMATH_CALUDE_student_grade_average_l1485_148516


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l1485_148580

/-- Given two curves y = x^2 - 1 and y = 1 + x^3 with perpendicular tangents at x = x₀, 
    prove that x₀ = - ∛(36) / 6 -/
theorem perpendicular_tangents_intersection (x₀ : ℝ) : 
  (∀ x, (2 * x) * (3 * x^2) = -1 → x = x₀) →
  x₀ = - (36 : ℝ)^(1/3) / 6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l1485_148580


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1485_148524

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 16 :=          -- The shorter leg is 16 units long
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1485_148524


namespace NUMINAMATH_CALUDE_sqrt_product_equals_24_l1485_148577

theorem sqrt_product_equals_24 (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (6 * x) * Real.sqrt (24 * x) = 24) : 
  x = Real.sqrt (3 / 22) := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_24_l1485_148577


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1485_148532

/-- The repeating decimal 0.̅5̅6̅ is equal to the fraction 56/99 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ = 0.56) ∧ x = 56/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1485_148532


namespace NUMINAMATH_CALUDE_diagonals_perpendicular_l1485_148578

/-- Given four points A, B, C, and D in a 2D plane, prove that the diagonals of the quadrilateral ABCD are perpendicular. -/
theorem diagonals_perpendicular (A B C D : ℝ × ℝ) 
  (hA : A = (-2, 3))
  (hB : B = (2, 6))
  (hC : C = (6, -1))
  (hD : D = (-3, -4)) : 
  (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0 := by
  sorry

#check diagonals_perpendicular

end NUMINAMATH_CALUDE_diagonals_perpendicular_l1485_148578


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l1485_148536

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 41 / 3

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 1

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of medication for the treatment period -/
def total_cost : ℚ := 819

theorem green_pill_cost_proof :
  (green_pill_cost + 2 * pink_pill_cost) * treatment_days = total_cost :=
sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l1485_148536


namespace NUMINAMATH_CALUDE_largest_t_value_l1485_148549

theorem largest_t_value : ∃ (t_max : ℝ), 
  (∀ t : ℝ, (15 * t^2 - 40 * t + 18) / (4 * t - 3) + 3 * t = 4 * t + 2 → t ≤ t_max) ∧
  ((15 * t_max^2 - 40 * t_max + 18) / (4 * t_max - 3) + 3 * t_max = 4 * t_max + 2) ∧
  t_max = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_l1485_148549


namespace NUMINAMATH_CALUDE_infinite_set_A_l1485_148591

/-- Given a function f: ℝ → ℝ satisfying the inequality f²(x) ≤ 2x² f(x/2) for all x,
    and a non-empty set A = {a ∈ ℝ | f(a) > a²}, prove that A is infinite. -/
theorem infinite_set_A (f : ℝ → ℝ) 
    (h1 : ∀ x : ℝ, f x ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
    (A : Set ℝ)
    (h2 : A = {a : ℝ | f a > a ^ 2})
    (h3 : Set.Nonempty A) :
  Set.Infinite A :=
sorry

end NUMINAMATH_CALUDE_infinite_set_A_l1485_148591


namespace NUMINAMATH_CALUDE_product_of_specific_primes_l1485_148575

theorem product_of_specific_primes : 
  let largest_one_digit_prime := 7
  let smallest_two_digit_prime1 := 11
  let smallest_two_digit_prime2 := 13
  largest_one_digit_prime * smallest_two_digit_prime1 * smallest_two_digit_prime2 = 1001 := by
sorry

end NUMINAMATH_CALUDE_product_of_specific_primes_l1485_148575


namespace NUMINAMATH_CALUDE_similar_triangles_not_necessarily_equal_sides_l1485_148587

-- Define a structure for triangles
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ ∧
    t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c ∧
    t1.a / t2.a = k

-- Define a property for equal corresponding sides
def equal_sides (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem similar_triangles_not_necessarily_equal_sides :
  ¬ (∀ t1 t2 : Triangle, similar t1 t2 → equal_sides t1 t2) :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_not_necessarily_equal_sides_l1485_148587


namespace NUMINAMATH_CALUDE_certain_number_operations_l1485_148526

theorem certain_number_operations (x : ℝ) : 
  ∃ (p q : ℕ), p < q ∧ ((x + 20) * 2 / 2 - 2 = x + 18) ∧ (x + 18 = (p : ℝ) / q * 88) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_operations_l1485_148526


namespace NUMINAMATH_CALUDE_unique_solution_l1485_148541

/-- Represents the number of children in each class -/
structure ClassSizes where
  judo : ℕ
  agriculture : ℕ
  math : ℕ

/-- Checks if the given class sizes satisfy all conditions -/
def satisfiesConditions (sizes : ClassSizes) : Prop :=
  sizes.judo + sizes.agriculture + sizes.math = 32 ∧
  sizes.judo > 0 ∧ sizes.agriculture > 0 ∧ sizes.math > 0 ∧
  sizes.judo / 2 + sizes.agriculture / 4 + sizes.math / 8 = 6

/-- The theorem stating that the unique solution satisfying all conditions is (4, 4, 24) -/
theorem unique_solution : 
  ∃! sizes : ClassSizes, satisfiesConditions sizes ∧ 
  sizes.judo = 4 ∧ sizes.agriculture = 4 ∧ sizes.math = 24 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1485_148541


namespace NUMINAMATH_CALUDE_unique_award_implies_all_defeated_l1485_148531

def Tournament (α : Type) := α → α → Prop

structure Award (α : Type) (t : Tournament α) (winner : α) : Prop :=
  (defeated_or_indirect : ∀ b : α, b ≠ winner → t winner b ∨ ∃ c, t winner c ∧ t c b)

theorem unique_award_implies_all_defeated 
  {α : Type} (t : Tournament α) (winner : α) :
  (∀ a b : α, a ≠ b → t a b ∨ t b a) →
  (∀ x : α, Award α t x ↔ x = winner) →
  (∀ b : α, b ≠ winner → t winner b) :=
sorry

end NUMINAMATH_CALUDE_unique_award_implies_all_defeated_l1485_148531


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1485_148513

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1485_148513


namespace NUMINAMATH_CALUDE_lukes_trips_l1485_148540

/-- Luke's tray-carrying problem -/
theorem lukes_trips (trays_per_trip : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ)
  (h1 : trays_per_trip = 4)
  (h2 : trays_table1 = 20)
  (h3 : trays_table2 = 16) :
  (trays_table1 + trays_table2) / trays_per_trip = 9 :=
by sorry

end NUMINAMATH_CALUDE_lukes_trips_l1485_148540


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1485_148555

def is_monic_cubic (q : ℝ → ℂ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_theorem (q : ℝ → ℂ) 
  (h_monic : is_monic_cubic q)
  (h_root : q (2 - 3*I) = 0)
  (h_const : q 0 = -72) :
  ∀ x, q x = x^3 - (100/13)*x^2 + (236/13)*x - 936/13 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1485_148555


namespace NUMINAMATH_CALUDE_symmetric_points_y_axis_l1485_148507

theorem symmetric_points_y_axis (m n : ℝ) : 
  (m - 1 = -2 ∧ 4 = n + 2) → n^m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_y_axis_l1485_148507


namespace NUMINAMATH_CALUDE_n_cube_minus_n_l1485_148567

theorem n_cube_minus_n (n : ℕ) (h : ∃ k : ℕ, 33 * 20 * n = k) : n^3 - n = 388944 := by
  sorry

end NUMINAMATH_CALUDE_n_cube_minus_n_l1485_148567


namespace NUMINAMATH_CALUDE_max_product_a2_a6_l1485_148569

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the maximum value of a₂ * a₆ in an arithmetic sequence where a₄ = 2 -/
theorem max_product_a2_a6 (a : ℕ → ℝ) (h : ArithmeticSequence a) (h4 : a 4 = 2) :
  (∀ b c : ℝ, a 2 = b ∧ a 6 = c → b * c ≤ 4) ∧ (∃ b c : ℝ, a 2 = b ∧ a 6 = c ∧ b * c = 4) :=
sorry

end NUMINAMATH_CALUDE_max_product_a2_a6_l1485_148569


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l1485_148515

-- Define the percentages of spending
def clothing_percent : ℝ := 0.50
def food_percent : ℝ := 0.20
def other_percent : ℝ := 0.30

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0
def total_tax_rate : ℝ := 0.044

-- Define the unknown tax rate on other items
def other_tax_rate : ℝ := sorry

theorem shopping_tax_calculation :
  let total_spent := 100  -- Assume total spent is 100 for simplicity
  let clothing_tax := clothing_percent * total_spent * clothing_tax_rate
  let other_tax := other_percent * total_spent * other_tax_rate
  clothing_tax + other_tax = total_tax_rate * total_spent →
  other_tax_rate = 0.08 := by sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l1485_148515


namespace NUMINAMATH_CALUDE_smallest_natural_divisible_l1485_148506

theorem smallest_natural_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 4 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 6 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 10 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 12 * k)) →
  (∃ k1 k2 k3 k4 : ℕ, n + 1 = 4 * k1 ∧ n + 1 = 6 * k2 ∧ n + 1 = 10 * k3 ∧ n + 1 = 12 * k4) →
  n = 59 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_divisible_l1485_148506
