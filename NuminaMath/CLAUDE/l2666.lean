import Mathlib

namespace NUMINAMATH_CALUDE_decimal_to_percentage_l2666_266619

theorem decimal_to_percentage (d : ℝ) (h : d = 0.05) : d * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l2666_266619


namespace NUMINAMATH_CALUDE_complement_of_A_l2666_266628

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Theorem statement
theorem complement_of_A : 
  (U \ A) = {x : ℝ | x < -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2666_266628


namespace NUMINAMATH_CALUDE_complex_modulus_of_z_l2666_266693

theorem complex_modulus_of_z (z : ℂ) : z = 1 - (1 / Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_z_l2666_266693


namespace NUMINAMATH_CALUDE_max_value_expression_l2666_266694

theorem max_value_expression (a b : ℝ) (h : a^2 + b^2 = 3 + a*b) :
  (∃ x y : ℝ, x^2 + y^2 = 3 + x*y ∧ (2*x - 3*y)^2 + (x + 2*y)*(x - 2*y) ≤ 22) ∧
  (∃ x y : ℝ, x^2 + y^2 = 3 + x*y ∧ (2*x - 3*y)^2 + (x + 2*y)*(x - 2*y) = 22) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2666_266694


namespace NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l2666_266698

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l2666_266698


namespace NUMINAMATH_CALUDE_hcf_of_210_and_517_l2666_266692

theorem hcf_of_210_and_517 (lcm_value : ℕ) (a b : ℕ) (h_lcm : Nat.lcm a b = lcm_value) 
  (h_a : a = 210) (h_b : b = 517) (h_lcm_value : lcm_value = 2310) : Nat.gcd a b = 47 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_210_and_517_l2666_266692


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2666_266664

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = Real.sqrt (1 - b^2 / a^2)

/-- A point on the ellipse -/
structure EllipsePoint (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The upper vertex of the ellipse -/
def upperVertex (E : Ellipse) : EllipsePoint E where
  x := 0
  y := E.b
  h_on_ellipse := by sorry

/-- A focus of the ellipse -/
structure Focus (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_major_axis : y = 0
  h_distance_from_center : x^2 = E.a^2 * E.e^2

/-- A line perpendicular to the line connecting a focus and the upper vertex -/
structure PerpendicularLine (E : Ellipse) (F : Focus E) where
  slope : ℝ
  h_perpendicular : slope * (F.x / E.b) = -1

/-- The intersection points of the perpendicular line with the ellipse -/
structure IntersectionPoints (E : Ellipse) (F : Focus E) (L : PerpendicularLine E F) where
  D : EllipsePoint E
  E : EllipsePoint E
  h_on_line_D : D.y = L.slope * (D.x - F.x)
  h_on_line_E : E.y = L.slope * (E.x - F.x)
  h_distance : (D.x - E.x)^2 + (D.y - E.y)^2 = 36

/-- The main theorem -/
theorem ellipse_triangle_perimeter
  (E : Ellipse)
  (h_e : E.e = 1/2)
  (F₁ F₂ : Focus E)
  (L : PerpendicularLine E F₁)
  (I : IntersectionPoints E F₁ L) :
  let A := upperVertex E
  let D := I.D
  let E := I.E
  (Real.sqrt ((A.x - D.x)^2 + (A.y - D.y)^2) +
   Real.sqrt ((A.x - E.x)^2 + (A.y - E.y)^2) +
   Real.sqrt ((D.x - E.x)^2 + (D.y - E.y)^2)) = 13 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2666_266664


namespace NUMINAMATH_CALUDE_dot_product_MN_MO_l2666_266681

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define a line l (we don't need to specify its equation, just that it exists)
def line_l : Set (ℝ × ℝ) := sorry

-- Define points M and N as the intersection of line l and circle O
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Define point O as the center of the circle
def O : ℝ × ℝ := (0, 0)

-- State that M and N are on the circle
axiom M_on_circle : M ∈ circle_O
axiom N_on_circle : N ∈ circle_O

-- State that M and N are on the line l
axiom M_on_line : M ∈ line_l
axiom N_on_line : N ∈ line_l

-- Define the distance between M and N
axiom MN_distance : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4

-- Define vectors MN and MO
def vec_MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
def vec_MO : ℝ × ℝ := (O.1 - M.1, O.2 - M.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem to prove
theorem dot_product_MN_MO : dot_product vec_MN vec_MO = 8 := by sorry

end NUMINAMATH_CALUDE_dot_product_MN_MO_l2666_266681


namespace NUMINAMATH_CALUDE_total_fruits_picked_l2666_266682

theorem total_fruits_picked (sara_pears tim_pears lily_apples max_oranges : ℕ)
  (h1 : sara_pears = 6)
  (h2 : tim_pears = 5)
  (h3 : lily_apples = 4)
  (h4 : max_oranges = 3) :
  sara_pears + tim_pears + lily_apples + max_oranges = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_picked_l2666_266682


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l2666_266674

theorem sqrt_sum_simplification : 
  Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l2666_266674


namespace NUMINAMATH_CALUDE_f_not_in_second_quadrant_l2666_266620

/-- A linear function f(x) = 2x - 1 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of f(x) = 2x - 1 does not pass through the second quadrant -/
theorem f_not_in_second_quadrant :
  ∀ x y : ℝ, f x = y → ¬(second_quadrant x y) :=
by sorry

end NUMINAMATH_CALUDE_f_not_in_second_quadrant_l2666_266620


namespace NUMINAMATH_CALUDE_probability_all_girls_chosen_l2666_266610

def total_members : ℕ := 15
def num_boys : ℕ := 8
def num_girls : ℕ := 7
def num_chosen : ℕ := 3

theorem probability_all_girls_chosen :
  (Nat.choose num_girls num_chosen : ℚ) / (Nat.choose total_members num_chosen) = 1 / 13 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_girls_chosen_l2666_266610


namespace NUMINAMATH_CALUDE_identical_asymptotes_hyperbolas_l2666_266647

theorem identical_asymptotes_hyperbolas (M : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) → M = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_identical_asymptotes_hyperbolas_l2666_266647


namespace NUMINAMATH_CALUDE_pancake_price_l2666_266661

/-- Janina's pancake stand problem -/
theorem pancake_price (daily_rent : ℝ) (daily_supplies : ℝ) (pancakes_to_cover_expenses : ℕ) :
  daily_rent = 30 ∧ daily_supplies = 12 ∧ pancakes_to_cover_expenses = 21 →
  (daily_rent + daily_supplies) / pancakes_to_cover_expenses = 2 :=
by sorry

end NUMINAMATH_CALUDE_pancake_price_l2666_266661


namespace NUMINAMATH_CALUDE_five_hour_pay_calculation_l2666_266609

/-- Represents the hourly pay rate in dollars -/
def hourly_rate (three_hour_pay six_hour_pay : ℚ) : ℚ :=
  three_hour_pay / 3

/-- Calculates the pay for a given number of hours -/
def calculate_pay (rate : ℚ) (hours : ℚ) : ℚ :=
  rate * hours

theorem five_hour_pay_calculation 
  (three_hour_pay six_hour_pay : ℚ) 
  (h1 : three_hour_pay = 24.75)
  (h2 : six_hour_pay = 49.50)
  (h3 : hourly_rate three_hour_pay six_hour_pay = hourly_rate three_hour_pay six_hour_pay) :
  calculate_pay (hourly_rate three_hour_pay six_hour_pay) 5 = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_five_hour_pay_calculation_l2666_266609


namespace NUMINAMATH_CALUDE_down_payment_ratio_l2666_266657

theorem down_payment_ratio (total_cost balance_due daily_payment : ℚ) 
  (h1 : total_cost = 120)
  (h2 : balance_due = 60)
  (h3 : daily_payment = 6)
  (h4 : balance_due = daily_payment * 10) :
  (total_cost - balance_due) / total_cost = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_down_payment_ratio_l2666_266657


namespace NUMINAMATH_CALUDE_koi_fish_multiple_l2666_266679

theorem koi_fish_multiple (num_koi : ℕ) (target : ℕ) : 
  num_koi = 39 → target = 64 → 
  ∃ m : ℕ, m * num_koi > target ∧ 
           ∀ k : ℕ, k * num_koi > target → k ≥ m ∧
           m * num_koi = 78 := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_multiple_l2666_266679


namespace NUMINAMATH_CALUDE_divisor_sum_of_2_3_power_l2666_266600

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Sum of geometric series -/
def geometric_sum (a r : ℕ) (n : ℕ) : ℕ := sorry

theorem divisor_sum_of_2_3_power (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 540 → i + j = 5 := by sorry

end NUMINAMATH_CALUDE_divisor_sum_of_2_3_power_l2666_266600


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l2666_266699

theorem solution_set_abs_inequality (x : ℝ) :
  (Set.Icc 1 3 : Set ℝ) = {x | |2 - x| ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l2666_266699


namespace NUMINAMATH_CALUDE_entrance_fee_is_five_l2666_266695

/-- The entrance fee per person for a concert, given the following conditions:
  * Tickets cost $50.00 each
  * There's a 15% processing fee for tickets
  * There's a $10.00 parking fee
  * The total cost for two people is $135.00
-/
def entrance_fee : ℝ := by
  sorry

theorem entrance_fee_is_five : entrance_fee = 5 := by
  sorry

end NUMINAMATH_CALUDE_entrance_fee_is_five_l2666_266695


namespace NUMINAMATH_CALUDE_vasya_numbers_l2666_266690

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) (h3 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l2666_266690


namespace NUMINAMATH_CALUDE_min_value_theorem_l2666_266635

theorem min_value_theorem (α₁ α₂ : ℝ) 
  (h : (2 + Real.sin α₁)⁻¹ + (2 + Real.sin (2 * α₂))⁻¹ = 2) : 
  ∃ (k₁ k₂ : ℤ), ∀ (α₁' α₂' : ℝ), 
    (2 + Real.sin α₁')⁻¹ + (2 + Real.sin (2 * α₂'))⁻¹ = 2 →
    |10 * Real.pi - α₁' - α₂'| ≥ |10 * Real.pi - ((-π/2 : ℝ) + 2 * ↑k₁ * π) - ((-π/4 : ℝ) + ↑k₂ * π)| ∧
    |10 * Real.pi - ((-π/2 : ℝ) + 2 * ↑k₁ * π) - ((-π/4 : ℝ) + ↑k₂ * π)| = π/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2666_266635


namespace NUMINAMATH_CALUDE_store_desktop_sales_l2666_266677

/-- Given a ratio of laptops to desktops and an expected number of laptop sales,
    calculate the expected number of desktop sales. -/
def expected_desktop_sales (laptop_ratio : ℕ) (desktop_ratio : ℕ) (expected_laptops : ℕ) : ℕ :=
  (expected_laptops * desktop_ratio) / laptop_ratio

/-- Proof that given the specific ratio and expected laptop sales,
    the expected desktop sales is 24. -/
theorem store_desktop_sales : expected_desktop_sales 5 3 40 = 24 := by
  sorry

#eval expected_desktop_sales 5 3 40

end NUMINAMATH_CALUDE_store_desktop_sales_l2666_266677


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2666_266653

noncomputable section

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the line l passing through the origin
def line_through_origin (m n : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, x = m * t ∧ y = n * t}

-- Define the condition that M and N are on the hyperbola and the line
def points_on_hyperbola_and_line (a b m n : ℝ) (M N : ℝ × ℝ) : Prop :=
  hyperbola a b M.1 M.2 ∧ hyperbola a b N.1 N.2 ∧
  M ∈ line_through_origin m n ∧ N ∈ line_through_origin m n

-- Define the perpendicularity condition
def perpendicular_vectors (F M N : ℝ × ℝ) : Prop :=
  (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0

-- Define the area condition
def triangle_area (F M N : ℝ × ℝ) (a b : ℝ) : Prop :=
  abs ((M.1 - F.1) * (N.2 - F.2) - (N.1 - F.1) * (M.2 - F.2)) / 2 = a * b

-- Main theorem
theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (F : ℝ × ℝ)
  (hF : F = right_focus a b c)
  (M N : ℝ × ℝ)
  (h_points : ∃ m n : ℝ, points_on_hyperbola_and_line a b m n M N)
  (h_perp : perpendicular_vectors F M N)
  (h_area : triangle_area F M N a b) :
  c^2 / a^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2666_266653


namespace NUMINAMATH_CALUDE_sections_after_five_lines_l2666_266684

/-- The number of sections in a rectangle after drawing n line segments,
    where each line increases the number of sections by its sequence order. -/
def sections (n : ℕ) : ℕ :=
  1 + (List.range n).sum

/-- Theorem: After drawing 5 line segments in a rectangle that initially has 1 section,
    where each new line segment increases the number of sections by its sequence order,
    the final number of sections is 16. -/
theorem sections_after_five_lines :
  sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sections_after_five_lines_l2666_266684


namespace NUMINAMATH_CALUDE_school_average_age_l2666_266630

/-- Given a school with the following properties:
  * Total number of students is 600
  * Average age of boys is 12 years
  * Average age of girls is 11 years
  * Number of girls is 150
  Prove that the average age of the school is 11.75 years -/
theorem school_average_age 
  (total_students : ℕ) 
  (boys_avg_age girls_avg_age : ℚ)
  (num_girls : ℕ) :
  total_students = 600 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 150 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_school_average_age_l2666_266630


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2666_266606

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^3 + 2*x^2 = (x^2 + 3*x + 2) * q + (x + 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2666_266606


namespace NUMINAMATH_CALUDE_track_length_track_length_is_350_l2666_266654

/-- The length of a circular track given specific running conditions -/
theorem track_length : ℝ → ℝ → ℝ → Prop :=
  λ first_meet second_meet track_length =>
    -- Brenda and Sally start at diametrically opposite points
    -- They first meet after Brenda has run 'first_meet' meters
    -- They next meet after Sally has run 'second_meet' meters past their first meeting point
    -- 'track_length' is the length of the circular track
    first_meet = 150 ∧
    second_meet = 200 ∧
    track_length = 350 ∧
    -- The total distance run by both runners is twice the track length
    2 * track_length = 2 * first_meet + second_meet

theorem track_length_is_350 : ∃ (l : ℝ), track_length 150 200 l :=
  sorry

end NUMINAMATH_CALUDE_track_length_track_length_is_350_l2666_266654


namespace NUMINAMATH_CALUDE_garden_breadth_l2666_266618

theorem garden_breadth (perimeter length : ℕ) (h1 : perimeter = 1200) (h2 : length = 360) :
  let breadth := (perimeter / 2) - length
  breadth = 240 :=
by sorry

end NUMINAMATH_CALUDE_garden_breadth_l2666_266618


namespace NUMINAMATH_CALUDE_find_k_l2666_266660

theorem find_k : ∃ k : ℕ, (1/2)^16 * (1/81)^k = 1/(18^16) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2666_266660


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l2666_266632

def a : Fin 2 → ℝ := ![2, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem collinear_vectors_m_value :
  ∃ (m : ℝ), ∃ (k : ℝ),
    (k ≠ 0) ∧
    (∀ i : Fin 2, k * (m * a i + 4 * b i) = (a i - 2 * b i)) →
    m = -2 :=
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l2666_266632


namespace NUMINAMATH_CALUDE_ant_path_circle_containment_l2666_266623

/-- A closed path in a plane -/
structure ClosedPath where
  path : Set (ℝ × ℝ)
  is_closed : path.Nonempty ∧ ∃ p, p ∈ path ∧ p ∈ frontier path
  length : ℝ

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem statement -/
theorem ant_path_circle_containment (γ : ClosedPath) (h : γ.length = 1) :
  ∃ (c : Circle), c.radius = 1/4 ∧ γ.path ⊆ {p : ℝ × ℝ | dist p c.center ≤ c.radius } :=
sorry

end NUMINAMATH_CALUDE_ant_path_circle_containment_l2666_266623


namespace NUMINAMATH_CALUDE_maria_travel_fraction_l2666_266637

theorem maria_travel_fraction (total_distance : ℝ) (remaining_distance : ℝ) 
  (first_stop_fraction : ℝ) :
  total_distance = 480 →
  remaining_distance = 180 →
  remaining_distance = (1 - first_stop_fraction) * total_distance * (3/4) →
  first_stop_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_maria_travel_fraction_l2666_266637


namespace NUMINAMATH_CALUDE_two_not_units_digit_of_square_l2666_266608

def units_digit (n : ℕ) : ℕ := n % 10

theorem two_not_units_digit_of_square : ∀ n : ℕ, units_digit (n^2) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_not_units_digit_of_square_l2666_266608


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2666_266659

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2666_266659


namespace NUMINAMATH_CALUDE_april_price_index_april_price_increase_l2666_266670

/-- Represents the price index for a given month -/
structure PriceIndex where
  month : Nat
  value : Real

/-- Calculates the price index for a given month based on the initial index and monthly decrease rate -/
def calculate_price_index (initial_index : Real) (monthly_decrease : Real) (month : Nat) : Real :=
  initial_index - (month - 1) * monthly_decrease

/-- Theorem stating that the price index in April is 1.12 given the conditions -/
theorem april_price_index 
  (january_index : PriceIndex)
  (monthly_decrease : Real)
  (h1 : january_index.month = 1)
  (h2 : january_index.value = 1.15)
  (h3 : monthly_decrease = 0.01)
  : ∃ (april_index : PriceIndex), 
    april_index.month = 4 ∧ 
    april_index.value = calculate_price_index january_index.value monthly_decrease 4 ∧
    april_index.value = 1.12 :=
sorry

/-- Theorem stating that the price in April has increased by 12% compared to the same month last year -/
theorem april_price_increase 
  (april_index : PriceIndex)
  (h : april_index.value = 1.12)
  : (april_index.value - 1) * 100 = 12 :=
sorry

end NUMINAMATH_CALUDE_april_price_index_april_price_increase_l2666_266670


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainders_l2666_266685

theorem unique_divisor_with_remainders :
  ∃! b : ℕ, b > 1 ∧ 826 % b = 7 ∧ 4373 % b = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainders_l2666_266685


namespace NUMINAMATH_CALUDE_approximate_solution_exists_l2666_266686

def f (x : ℝ) := 2 * x^3 + 3 * x - 3

theorem approximate_solution_exists :
  (f 0.625 < 0) →
  (f 0.75 > 0) →
  (f 0.6875 < 0) →
  ∃ x : ℝ, x ∈ Set.Icc 0.6 0.8 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_approximate_solution_exists_l2666_266686


namespace NUMINAMATH_CALUDE_max_value_of_a_l2666_266641

-- Define the condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 2| + |x - a| ≥ a

-- State the theorem
theorem max_value_of_a :
  ∃ a_max : ℝ, a_max = 1 ∧
  inequality_holds a_max ∧
  ∀ a : ℝ, inequality_holds a → a ≤ a_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2666_266641


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l2666_266649

theorem trapezoid_median_length :
  let large_side : ℝ := 4
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side^2
  let small_area : ℝ := large_area / 3
  let small_side : ℝ := Real.sqrt ((4 * small_area) / Real.sqrt 3)
  let median : ℝ := (large_side + small_side) / 2
  median = (2 * (Real.sqrt 3 + 1)) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_length_l2666_266649


namespace NUMINAMATH_CALUDE_inequality_proof_l2666_266666

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2666_266666


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l2666_266624

theorem congruence_solutions_count : 
  ∃! (s : Finset Nat), 
    (∀ x ∈ s, x > 0 ∧ x < 150 ∧ (x + 17) % 45 = 80 % 45) ∧ 
    (∀ x, x > 0 ∧ x < 150 ∧ (x + 17) % 45 = 80 % 45 → x ∈ s) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l2666_266624


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2666_266636

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 2000 < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < 2005) ∧
    (∀ n : ℕ, 2000 < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < 2005 → n ∈ S) ∧
    Finset.card S = 5 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2666_266636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2666_266612

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequenceWithSum where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic sequence property
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2  -- Sum formula

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequenceWithSum) 
  (h1 : seq.a 8 - seq.a 5 = 9)
  (h2 : seq.S 8 - seq.S 5 = 66) :
  seq.a 33 = 100 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2666_266612


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2666_266607

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2666_266607


namespace NUMINAMATH_CALUDE_gracies_height_l2666_266631

/-- Proves that Gracie's height is 56 inches given the relationships between Gracie, Grayson, and Griffin's heights. -/
theorem gracies_height (griffin_height grayson_height gracie_height : ℕ) : 
  griffin_height = 61 →
  grayson_height = griffin_height + 2 →
  gracie_height = grayson_height - 7 →
  gracie_height = 56 :=
by sorry

end NUMINAMATH_CALUDE_gracies_height_l2666_266631


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2666_266669

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (2 + i) / (1 + i)^2
  (z.im : ℝ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2666_266669


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2666_266616

def is_between (x a b : ℕ) : Prop := a < x ∧ x < b

def is_single_digit (x : ℕ) : Prop := x < 10

theorem unique_number_satisfying_conditions :
  ∃! x : ℕ, is_between x 5 9 ∧ is_single_digit x ∧ x > 7 :=
sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2666_266616


namespace NUMINAMATH_CALUDE_smallest_b_value_l2666_266697

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^4 + b.val^4) / (a.val + b.val)) (a.val * b.val) = 16) :
  b.val ≥ 4 ∧ ∃ (a₀ b₀ : ℕ+), b₀.val = 4 ∧ a₀.val - b₀.val = 8 ∧ 
    Nat.gcd ((a₀.val^4 + b₀.val^4) / (a₀.val + b₀.val)) (a₀.val * b₀.val) = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2666_266697


namespace NUMINAMATH_CALUDE_monomial_count_is_four_l2666_266622

/-- A monomial is an algebraic expression consisting of one term. It can be a constant, a variable, or a product of constants and variables raised to whole number powers. -/
def is_monomial (expr : String) : Bool := sorry

/-- The list of algebraic expressions given in the problem -/
def expressions : List String := ["-2/3*a^3*b", "xy/2", "-4", "-2/a", "0", "x-y"]

/-- Count the number of monomials in a list of expressions -/
def count_monomials (exprs : List String) : Nat :=
  exprs.filter is_monomial |>.length

/-- The theorem to be proved -/
theorem monomial_count_is_four : count_monomials expressions = 4 := by sorry

end NUMINAMATH_CALUDE_monomial_count_is_four_l2666_266622


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l2666_266656

/-- For a normal distribution with mean 14.5 and standard deviation 1.7,
    the value that is exactly 2 standard deviations less than the mean is 11.1. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (h1 : μ = 14.5) (h2 : σ = 1.7) :
  μ - 2 * σ = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l2666_266656


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2666_266675

/-- Given two lines l₁ and l₂, prove that if they are perpendicular, then m = 1/2 -/
theorem perpendicular_lines_m_value (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | m * x + y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x + (m - 1) * y + 2 = 0}
  (∀ (p₁ p₂ q₁ q₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₁ → q₁ ∈ l₂ → q₂ ∈ l₂ → 
    p₁ ≠ p₂ → q₁ ≠ q₂ → (p₁.1 - p₂.1) * (q₁.1 - q₂.1) + (p₁.2 - p₂.2) * (q₁.2 - q₂.2) = 0) →
  m = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2666_266675


namespace NUMINAMATH_CALUDE_expression_evaluation_l2666_266617

/-- Proves that the given expression evaluates to the specified value -/
theorem expression_evaluation (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  3500 - (1000 / (20.50 + x * 10)) / (y^2 - 2*z) = 3496.6996699669967 := by
  sorry

#eval (3500 - (1000 / (20.50 + 3 * 10)) / (4^2 - 2*5) : Float)

end NUMINAMATH_CALUDE_expression_evaluation_l2666_266617


namespace NUMINAMATH_CALUDE_min_sum_product_l2666_266613

theorem min_sum_product (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ) : 
  (({a1, a2, a3, b1, b2, b3, c1, c2, c3, d} : Finset ℕ) = Finset.range 10) →
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 + d ≥ 609 ∧
  ∃ (p1 p2 p3 q1 q2 q3 r1 r2 r3 s : ℕ),
    ({p1, p2, p3, q1, q2, q3, r1, r2, r3, s} : Finset ℕ) = Finset.range 10 ∧
    p1 * p2 * p3 + q1 * q2 * q3 + r1 * r2 * r3 + s = 609 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_product_l2666_266613


namespace NUMINAMATH_CALUDE_largest_y_floor_div_l2666_266691

theorem largest_y_floor_div : 
  ∀ y : ℝ, (⌊y⌋ : ℝ) / y = 8 / 9 → y ≤ 63 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_y_floor_div_l2666_266691


namespace NUMINAMATH_CALUDE_tape_length_calculation_l2666_266643

/-- The total length of overlapping tape sheets -/
def total_tape_length (n : ℕ) (sheet_length : ℝ) (overlap : ℝ) : ℝ :=
  sheet_length + (n - 1 : ℝ) * (sheet_length - overlap)

/-- Theorem: The total length of 15 sheets of tape, each 20 cm long and overlapping by 5 cm, is 230 cm -/
theorem tape_length_calculation :
  total_tape_length 15 20 5 = 230 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_calculation_l2666_266643


namespace NUMINAMATH_CALUDE_absolute_difference_of_mn_l2666_266667

theorem absolute_difference_of_mn (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_mn_l2666_266667


namespace NUMINAMATH_CALUDE_candy_box_solution_l2666_266651

/-- Represents the number of candies of each type in a box -/
structure CandyBox where
  chocolate : ℕ
  hard : ℕ
  jelly : ℕ

/-- Conditions for the candy box problem -/
def CandyBoxConditions (box : CandyBox) : Prop :=
  (box.chocolate + box.hard + box.jelly = 110) ∧
  (box.chocolate + box.hard = 100) ∧
  (box.hard + box.jelly = box.chocolate + box.jelly + 20)

/-- Theorem stating the solution to the candy box problem -/
theorem candy_box_solution :
  ∃ (box : CandyBox), CandyBoxConditions box ∧ 
    box.chocolate = 40 ∧ box.hard = 60 ∧ box.jelly = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_solution_l2666_266651


namespace NUMINAMATH_CALUDE_smallest_among_four_l2666_266604

theorem smallest_among_four (a b c d : ℚ) (h1 : a = -2) (h2 : b = -1) (h3 : c = 0) (h4 : d = 1) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_four_l2666_266604


namespace NUMINAMATH_CALUDE_roots_equation_sum_l2666_266634

theorem roots_equation_sum (a b : ℝ) : 
  (a^2 + a - 2022 = 0) → 
  (b^2 + b - 2022 = 0) → 
  (a ≠ b) →
  (a^2 + 2*a + b = 2021) := by
sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l2666_266634


namespace NUMINAMATH_CALUDE_position_2025_l2666_266642

/-- Represents the possible positions of the square -/
inductive SquarePosition
  | ABCD
  | CDAB
  | BADC
  | DCBA

/-- Applies the transformation pattern to a given position -/
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

/-- Returns the position after n transformations -/
def nthPosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.CDAB
  | 2 => SquarePosition.BADC
  | _ => SquarePosition.DCBA

theorem position_2025 : nthPosition 2025 = SquarePosition.ABCD := by
  sorry


end NUMINAMATH_CALUDE_position_2025_l2666_266642


namespace NUMINAMATH_CALUDE_binomial_12_3_l2666_266650

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l2666_266650


namespace NUMINAMATH_CALUDE_solve_equation_l2666_266678

theorem solve_equation (x : ℝ) :
  let y := 1 / (4 * x^2 + 2 * x + 1)
  y = 1 → x = 0 ∨ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2666_266678


namespace NUMINAMATH_CALUDE_circle_angle_equality_l2666_266680

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the angle between two vectors
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_angle_equality (Γ : Circle) (O A B M N : ℝ × ℝ) 
  (hO : O = Γ.center)
  (hA : PointOnCircle Γ A)
  (hB : PointOnCircle Γ B)
  (hM : PointOnCircle Γ M)
  (hN : PointOnCircle Γ N) :
  angle (M.1 - A.1, M.2 - A.2) (M.1 - B.1, M.2 - B.2) = 
  (angle (O.1 - A.1, O.2 - A.2) (O.1 - B.1, O.2 - B.2)) / 2 ∧
  angle (N.1 - A.1, N.2 - A.2) (N.1 - B.1, N.2 - B.2) = 
  (angle (O.1 - A.1, O.2 - A.2) (O.1 - B.1, O.2 - B.2)) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_angle_equality_l2666_266680


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2666_266658

/-- Given a quadratic function f(x) = x^2 + 2px + r, 
    if the minimum value of f(x) is 1, then r = p^2 + 1 -/
theorem quadratic_minimum (p r : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + 2*p*x + r) ∧ 
   (∃ (m : ℝ), ∀ x, f x ≥ m ∧ (∃ y, f y = m)) ∧
   (∃ x, f x = 1)) →
  r = p^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2666_266658


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l2666_266696

theorem cos_thirty_degrees : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l2666_266696


namespace NUMINAMATH_CALUDE_cube_root_sixteen_to_sixth_l2666_266673

theorem cube_root_sixteen_to_sixth (x : ℝ) : x = (16 ^ (1/3 : ℝ)) → x^6 = 256 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sixteen_to_sixth_l2666_266673


namespace NUMINAMATH_CALUDE_card_area_after_shortening_l2666_266672

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_after_shortening (initial : Rectangle) :
  initial.length = 6 ∧ initial.width = 8 →
  ∃ (shortened : Rectangle), 
    (shortened.length = initial.length ∧ shortened.width = initial.width - 2 ∧ 
     area shortened = 36) →
    area { length := initial.length - 2, width := initial.width } = 32 := by
  sorry

end NUMINAMATH_CALUDE_card_area_after_shortening_l2666_266672


namespace NUMINAMATH_CALUDE_sugar_amount_l2666_266663

/-- The total amount of sugar the store owner started with, given the conditions. -/
theorem sugar_amount (num_packs : ℕ) (pack_weight : ℕ) (remaining_sugar : ℕ) 
  (h1 : num_packs = 12)
  (h2 : pack_weight = 250)
  (h3 : remaining_sugar = 20) :
  num_packs * pack_weight + remaining_sugar = 3020 :=
by sorry

end NUMINAMATH_CALUDE_sugar_amount_l2666_266663


namespace NUMINAMATH_CALUDE_problem_statement_l2666_266652

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^2 + 16/((x - 3)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2666_266652


namespace NUMINAMATH_CALUDE_middle_number_of_seven_consecutive_l2666_266683

def is_middle_of_seven_consecutive (n : ℕ) : Prop :=
  ∃ (a : ℕ), a + (a + 1) + (a + 2) + n + (n + 1) + (n + 2) + (n + 3) = 63

theorem middle_number_of_seven_consecutive :
  ∃ (n : ℕ), is_middle_of_seven_consecutive n ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_of_seven_consecutive_l2666_266683


namespace NUMINAMATH_CALUDE_rotation_volume_sum_l2666_266629

/-- Given a square ABCD with side length a and a point M at distance b from its center,
    the sum of volumes of solids obtained by rotating triangles ABM, BCM, CDM, and DAM
    around lines AB, BC, CD, and DA respectively is equal to 3a^3/8 -/
theorem rotation_volume_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let square := {A : ℝ × ℝ | ∃ (x y : ℝ), x ∈ [0, a] ∧ y ∈ [0, a] ∧ A = (x, y)}
  let center := (a/2, a/2)
  let M : ℝ × ℝ := sorry -- A point at distance b from the center
  let volume_sum := sorry -- Sum of volumes of rotated triangles
  volume_sum = 3 * a^3 / 8 := by
sorry

end NUMINAMATH_CALUDE_rotation_volume_sum_l2666_266629


namespace NUMINAMATH_CALUDE_green_blue_difference_l2666_266648

/-- Represents the number of tiles in a hexagonal figure -/
structure HexFigure where
  blue : Nat
  green : Nat

/-- Calculates the number of tiles needed for a double border -/
def doubleBorderTiles : Nat := 2 * 18

/-- The initial hexagonal figure -/
def initialFigure : HexFigure := { blue := 13, green := 6 }

/-- Creates a new figure with twice as many tiles -/
def doubleFigure (f : HexFigure) : HexFigure :=
  { blue := 2 * f.blue, green := 2 * f.green }

/-- Adds a double border of green tiles to a figure -/
def addGreenBorder (f : HexFigure) : HexFigure :=
  { blue := f.blue, green := f.green + doubleBorderTiles }

/-- Calculates the total tiles for two figures -/
def totalTiles (f1 f2 : HexFigure) : HexFigure :=
  { blue := f1.blue + f2.blue, green := f1.green + f2.green }

theorem green_blue_difference :
  let secondFigure := addGreenBorder (doubleFigure initialFigure)
  let totalFigure := totalTiles initialFigure secondFigure
  totalFigure.green - totalFigure.blue = 15 := by sorry

end NUMINAMATH_CALUDE_green_blue_difference_l2666_266648


namespace NUMINAMATH_CALUDE_stratified_sampling_best_l2666_266601

/-- Represents different sampling methods -/
inductive SamplingMethod
| Lottery
| RandomNumberTable
| Systematic
| Stratified

/-- Represents product quality classes -/
inductive ProductClass
| FirstClass
| SecondClass
| Defective

/-- Represents a collection of products with their quantities -/
structure ProductCollection :=
  (total : ℕ)
  (firstClass : ℕ)
  (secondClass : ℕ)
  (defective : ℕ)

/-- Determines the most appropriate sampling method for quality analysis -/
def bestSamplingMethod (products : ProductCollection) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the best method for the given conditions -/
theorem stratified_sampling_best :
  let products : ProductCollection := {
    total := 40,
    firstClass := 10,
    secondClass := 25,
    defective := 5
  }
  let sampleSize := 8
  bestSamplingMethod products sampleSize = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_l2666_266601


namespace NUMINAMATH_CALUDE_chef_almond_weight_l2666_266639

/-- The weight of pecans bought by the chef in kilograms. -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms. -/
def total_nut_weight : ℝ := 0.52

/-- The weight of almonds bought by the chef in kilograms. -/
def almond_weight : ℝ := total_nut_weight - pecan_weight

theorem chef_almond_weight :
  almond_weight = 0.14 := by sorry

end NUMINAMATH_CALUDE_chef_almond_weight_l2666_266639


namespace NUMINAMATH_CALUDE_sector_central_angle_l2666_266605

/-- Proves that a circular sector with radius 4 cm and area 4 cm² has a central angle of 1/4 radians -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (θ : ℝ) : 
  r = 4 → area = 4 → area = 1/2 * r^2 * θ → θ = 1/4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2666_266605


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2666_266633

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2666_266633


namespace NUMINAMATH_CALUDE_chessboard_game_outcomes_l2666_266687

/-- Represents the outcome of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents the starting position of the piece -/
inductive StartPosition
  | Corner
  | AdjacentToCorner

/-- Defines the game on an n × n chessboard -/
def chessboardGame (n : ℕ) (startPos : StartPosition) : GameOutcome :=
  match n, startPos with
  | n, StartPosition.Corner =>
      if n % 2 = 0 then
        GameOutcome.FirstPlayerWins
      else
        GameOutcome.SecondPlayerWins
  | _, StartPosition.AdjacentToCorner => GameOutcome.FirstPlayerWins

/-- Theorem stating the game outcomes -/
theorem chessboard_game_outcomes :
  (∀ n : ℕ, n > 1 →
    (n % 2 = 0 → chessboardGame n StartPosition.Corner = GameOutcome.FirstPlayerWins) ∧
    (n % 2 = 1 → chessboardGame n StartPosition.Corner = GameOutcome.SecondPlayerWins)) ∧
  (∀ n : ℕ, n > 1 → chessboardGame n StartPosition.AdjacentToCorner = GameOutcome.FirstPlayerWins) :=
sorry

end NUMINAMATH_CALUDE_chessboard_game_outcomes_l2666_266687


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2666_266621

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((a + 3 * Complex.I) / (1 - Complex.I)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2666_266621


namespace NUMINAMATH_CALUDE_binary_representation_of_70_has_7_digits_l2666_266671

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_representation_of_70_has_7_digits :
  (decimal_to_binary 70).length = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_70_has_7_digits_l2666_266671


namespace NUMINAMATH_CALUDE_fourth_term_is_one_l2666_266655

-- Define the geometric progression
def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_is_one :
  let a₁ := (3 : ℝ) ^ (1/3)
  let a₂ := (3 : ℝ) ^ (1/4)
  let a₃ := (3 : ℝ) ^ (1/12)
  let r := a₂ / a₁
  geometric_progression a₁ r 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_is_one_l2666_266655


namespace NUMINAMATH_CALUDE_election_votes_l2666_266668

theorem election_votes (total_votes : ℕ) : 
  (0.7 * (0.85 * total_votes : ℝ) = 333200) → 
  total_votes = 560000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2666_266668


namespace NUMINAMATH_CALUDE_cube_of_negative_two_times_t_l2666_266640

theorem cube_of_negative_two_times_t (t : ℝ) : (-2 * t)^3 = -8 * t^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_two_times_t_l2666_266640


namespace NUMINAMATH_CALUDE_product_increase_theorem_l2666_266688

theorem product_increase_theorem :
  ∃ (a b c d e f g : ℕ),
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) * (f - 3) * (g - 3) =
    13 * (a * b * c * d * e * f * g) :=
by sorry

end NUMINAMATH_CALUDE_product_increase_theorem_l2666_266688


namespace NUMINAMATH_CALUDE_tan_product_greater_than_one_l2666_266644

/-- In an acute triangle ABC, where a, b, c are sides opposite to angles A, B, C respectively,
    and a² = b² + bc, the product of tan A and tan B is always greater than 1. -/
theorem tan_product_greater_than_one (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  a^2 = b^2 + b*c →
  Real.tan A * Real.tan B > 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_greater_than_one_l2666_266644


namespace NUMINAMATH_CALUDE_complex_number_problem_l2666_266603

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition1 : Prop := (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I
def condition2 : Prop := z₂.im = 2
def condition3 : Prop := (z₁ * z₂).im = 0

-- State the theorem
theorem complex_number_problem :
  condition1 z₁ → condition2 z₂ → condition3 z₁ z₂ → z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2666_266603


namespace NUMINAMATH_CALUDE_apollonian_circle_m_range_l2666_266676

theorem apollonian_circle_m_range :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (2, 0)
  let C (m : ℝ) := {P : ℝ × ℝ | (P.1 - 2)^2 + (P.2 - m)^2 = 1/4}
  ∀ m > 0, (∃ P ∈ C m, dist P A = 2 * dist P B) →
    m ∈ Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2) :=
by sorry


end NUMINAMATH_CALUDE_apollonian_circle_m_range_l2666_266676


namespace NUMINAMATH_CALUDE_quiz_winning_probability_l2666_266625

def num_questions : ℕ := 4
def num_choices : ℕ := 3
def min_correct : ℕ := 3

def probability_correct : ℚ := 1 / num_choices

/-- The probability of winning the quiz. -/
def probability_winning : ℚ :=
  (num_questions.choose min_correct) * (probability_correct ^ min_correct) * ((1 - probability_correct) ^ (num_questions - min_correct)) +
  (num_questions.choose (min_correct + 1)) * (probability_correct ^ (min_correct + 1)) * ((1 - probability_correct) ^ (num_questions - (min_correct + 1)))

theorem quiz_winning_probability :
  probability_winning = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quiz_winning_probability_l2666_266625


namespace NUMINAMATH_CALUDE_negation_of_existence_l2666_266645

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2666_266645


namespace NUMINAMATH_CALUDE_pears_left_l2666_266662

theorem pears_left (keith_pears : ℕ) (total_pears : ℕ) : 
  keith_pears = 62 →
  total_pears = 186 →
  total_pears = keith_pears + 2 * keith_pears →
  140 = total_pears - (total_pears / 4) := by
  sorry

#check pears_left

end NUMINAMATH_CALUDE_pears_left_l2666_266662


namespace NUMINAMATH_CALUDE_complex_inequality_l2666_266638

theorem complex_inequality (z₁ z₂ z₃ z₄ : ℂ) :
  Complex.abs (z₁ - z₃)^2 + Complex.abs (z₂ - z₄)^2 ≤
  Complex.abs (z₁ - z₂)^2 + Complex.abs (z₂ - z₃)^2 +
  Complex.abs (z₃ - z₄)^2 + Complex.abs (z₄ - z₁)^2 ∧
  (Complex.abs (z₁ - z₃)^2 + Complex.abs (z₂ - z₄)^2 =
   Complex.abs (z₁ - z₂)^2 + Complex.abs (z₂ - z₃)^2 +
   Complex.abs (z₃ - z₄)^2 + Complex.abs (z₄ - z₁)^2 ↔
   z₁ + z₃ = z₂ + z₄) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l2666_266638


namespace NUMINAMATH_CALUDE_skittles_per_friend_l2666_266626

/-- Represents the number of Skittles Joshua has -/
def total_skittles : ℕ := 40

/-- Represents the number of friends Joshua shares the Skittles with -/
def number_of_friends : ℕ := 5

/-- Theorem stating that each friend receives 8 Skittles when Joshua shares his Skittles equally -/
theorem skittles_per_friend :
  total_skittles / number_of_friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_skittles_per_friend_l2666_266626


namespace NUMINAMATH_CALUDE_complex_power_twelve_l2666_266646

/-- If z = 2 cos(π/8) * (sin(3π/4) + i*cos(3π/4) + i), then z^12 = -64i. -/
theorem complex_power_twelve (z : ℂ) : 
  z = 2 * Real.cos (π/8) * (Real.sin (3*π/4) + Complex.I * Real.cos (3*π/4) + Complex.I) → 
  z^12 = -64 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_twelve_l2666_266646


namespace NUMINAMATH_CALUDE_kristin_laps_theorem_l2666_266614

/-- Kristin's running speed relative to Sarith's -/
def kristin_speed_ratio : ℚ := 3

/-- Ratio of adult field size to children's field size -/
def field_size_ratio : ℚ := 2

/-- Number of times Sarith went around the children's field -/
def sarith_laps : ℕ := 8

/-- Number of times Kristin went around the adult field -/
def kristin_laps : ℕ := 12

theorem kristin_laps_theorem (speed_ratio : ℚ) (field_ratio : ℚ) (sarith_runs : ℕ) :
  speed_ratio = kristin_speed_ratio →
  field_ratio = field_size_ratio →
  sarith_runs = sarith_laps →
  ↑kristin_laps = ↑sarith_runs * (speed_ratio / field_ratio) := by
  sorry

end NUMINAMATH_CALUDE_kristin_laps_theorem_l2666_266614


namespace NUMINAMATH_CALUDE_apple_production_total_l2666_266615

/-- The number of apples produced by a tree over three years -/
def appleProduction : ℕ → ℕ
| 1 => 40
| 2 => 2 * appleProduction 1 + 8
| 3 => appleProduction 2 - (appleProduction 2 / 4)
| _ => 0

/-- The total number of apples produced over three years -/
def totalApples : ℕ := appleProduction 1 + appleProduction 2 + appleProduction 3

theorem apple_production_total : totalApples = 194 := by
  sorry

end NUMINAMATH_CALUDE_apple_production_total_l2666_266615


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2666_266627

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle ACB is 2π/3
  angle_acb : Real.cos (2 * Real.pi / 3) = (c - a)^2 + (c - b)^2 - c^2 / (2 * (c - a) * (c - b))
  -- Sides form arithmetic sequence with common difference 2
  arithmetic_sequence : b - a = 2 ∧ c - b = 2
  -- Area of circumcircle is π
  circumcircle_area : π = π * (c / (2 * Real.sin (2 * Real.pi / 3)))^2

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.c = 7 ∧
  (∃ (θ : ℝ), 0 < θ ∧ θ < Real.pi / 3 ∧
    2 * Real.sin θ + 2 * Real.sin (Real.pi / 3 - θ) + Real.sqrt 3 ≤ 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2666_266627


namespace NUMINAMATH_CALUDE_rectangular_field_ratio_l2666_266689

theorem rectangular_field_ratio (perimeter width : ℝ) :
  perimeter = 432 →
  width = 90 →
  let length := (perimeter - 2 * width) / 2
  (length / width) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_ratio_l2666_266689


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2666_266665

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0) ↔ (∃ x₀ : ℝ, 2 * x₀^2 - x₀ + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2666_266665


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2666_266602

theorem absolute_value_inequality (x : ℝ) : 
  |x^2 - 3*x| > 4 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2666_266602


namespace NUMINAMATH_CALUDE_krtecek_return_distance_l2666_266611

/-- Represents a direction in 2D space -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Calculates the net displacement in centimeters for a list of movements -/
def netDisplacement (movements : List Movement) : ℝ × ℝ :=
  sorry

/-- Calculates the distance to the starting point given a net displacement -/
def distanceToStart (displacement : ℝ × ℝ) : ℝ :=
  sorry

/-- The list of Krteček's movements -/
def krtecekMovements : List Movement := [
  ⟨500, Direction.North⟩,
  ⟨230, Direction.West⟩,
  ⟨150, Direction.South⟩,
  ⟨370, Direction.West⟩,
  ⟨620, Direction.South⟩,
  ⟨53, Direction.East⟩,
  ⟨270, Direction.North⟩
]

theorem krtecek_return_distance :
  distanceToStart (netDisplacement krtecekMovements) = 547 := by
  sorry

end NUMINAMATH_CALUDE_krtecek_return_distance_l2666_266611
