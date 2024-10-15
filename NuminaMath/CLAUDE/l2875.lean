import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2875_287514

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2875_287514


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2875_287566

/-- The maximum ratio of the radius of the third sphere to the radius of the first sphere
    in a specific geometric configuration. -/
theorem sphere_radius_ratio (r x : ℝ) (h1 : r > 0) (h2 : x > 0) : 
  let R := 3 * r
  let t := x / r
  let α := π / 3
  let cone_height := R / 2
  let slant_height := R
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 3 → 
    2 * Real.cos θ ≤ (3 - 2*t) / Real.sqrt (t^2 + 2*t)) →
  (3 * t^2 - 14 * t + 9 = 0) →
  t ≤ 3 / 2 →
  t = (7 - Real.sqrt 22) / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2875_287566


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2875_287530

/-- Given three lines that intersect at one point, prove the value of a -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, a * p.1 + 2 * p.2 + 8 = 0 ∧ 
                  4 * p.1 + 3 * p.2 = 10 ∧ 
                  2 * p.1 - p.2 = 10) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2875_287530


namespace NUMINAMATH_CALUDE_chocolate_sales_theorem_l2875_287519

/-- Represents the chocolate sales problem -/
structure ChocolateSales where
  total_customers : ℕ
  price_A : ℕ
  price_B : ℕ
  max_B_ratio : ℚ
  price_increase_step : ℕ
  A_decrease_rate : ℕ
  B_decrease_rate : ℕ

/-- The main theorem for the chocolate sales problem -/
theorem chocolate_sales_theorem (cs : ChocolateSales)
  (h_total : cs.total_customers = 480)
  (h_price_A : cs.price_A = 90)
  (h_price_B : cs.price_B = 50)
  (h_max_B_ratio : cs.max_B_ratio = 3/5)
  (h_price_increase_step : cs.price_increase_step = 3)
  (h_A_decrease_rate : cs.A_decrease_rate = 5)
  (h_B_decrease_rate : cs.B_decrease_rate = 3) :
  ∃ (min_A : ℕ) (women_day_price_A : ℕ),
    min_A = 300 ∧
    women_day_price_A = 150 ∧
    min_A + (cs.total_customers - min_A) ≤ cs.total_customers ∧
    (cs.total_customers - min_A) ≤ cs.max_B_ratio * min_A ∧
    (min_A - (women_day_price_A - cs.price_A) / cs.price_increase_step * cs.A_decrease_rate) *
      women_day_price_A +
    ((cs.total_customers - min_A) - (women_day_price_A - cs.price_A) / cs.price_increase_step * cs.B_decrease_rate) *
      cs.price_B =
    min_A * cs.price_A + (cs.total_customers - min_A) * cs.price_B :=
by sorry


end NUMINAMATH_CALUDE_chocolate_sales_theorem_l2875_287519


namespace NUMINAMATH_CALUDE_opera_house_rows_l2875_287577

/-- Represents an opera house with a certain number of rows -/
structure OperaHouse where
  rows : ℕ

/-- Represents a show at the opera house -/
structure Show where
  earnings : ℕ
  occupancyRate : ℚ

/-- Calculates the total number of seats in the opera house -/
def totalSeats (oh : OperaHouse) : ℕ := oh.rows * 10

/-- Calculates the number of tickets sold for a show -/
def ticketsSold (s : Show) : ℕ := s.earnings / 10

/-- Theorem: Given the conditions, the opera house has 150 rows -/
theorem opera_house_rows (oh : OperaHouse) (s : Show) :
  totalSeats oh = ticketsSold s / s.occupancyRate →
  s.earnings = 12000 →
  s.occupancyRate = 4/5 →
  oh.rows = 150 := by
  sorry


end NUMINAMATH_CALUDE_opera_house_rows_l2875_287577


namespace NUMINAMATH_CALUDE_water_left_in_bucket_l2875_287508

/-- Converts milliliters to liters -/
def ml_to_l (ml : ℚ) : ℚ := ml / 1000

/-- Calculates the remaining water in a bucket after some is removed -/
def remaining_water (initial : ℚ) (removed_ml : ℚ) (removed_l : ℚ) : ℚ :=
  initial - (ml_to_l removed_ml + removed_l)

theorem water_left_in_bucket : 
  remaining_water 30 150 1.65 = 28.20 := by sorry

end NUMINAMATH_CALUDE_water_left_in_bucket_l2875_287508


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l2875_287534

theorem bucket_capacity_problem (capacity : ℝ) : 
  (24 * capacity = 36 * 9) → capacity = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_problem_l2875_287534


namespace NUMINAMATH_CALUDE_second_divisor_l2875_287575

theorem second_divisor (k : ℕ) (h1 : k > 0) (h2 : k < 42) 
  (h3 : k % 5 = 2) (h4 : k % 7 = 3) 
  (d : ℕ) (h5 : d > 0) (h6 : k % d = 5) : d = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_l2875_287575


namespace NUMINAMATH_CALUDE_fraction_simplification_l2875_287520

theorem fraction_simplification :
  ((3^1005)^2 - (3^1003)^2) / ((3^1004)^2 - (3^1002)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2875_287520


namespace NUMINAMATH_CALUDE_xy_difference_squared_l2875_287550

theorem xy_difference_squared (x y : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x - y = -2) : 
  x^2 * y - x * y^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_xy_difference_squared_l2875_287550


namespace NUMINAMATH_CALUDE_fifth_month_sale_l2875_287513

def sale_month1 : ℕ := 6535
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month6 : ℕ := 4891
def average_sale : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sale_month5 : ℕ),
    sale_month5 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month6) ∧
    sale_month5 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l2875_287513


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2875_287562

/-- Given that N(5, -1) is the midpoint of segment CD and C has coordinates (11, 10),
    prove that the sum of the coordinates of point D is -13. -/
theorem midpoint_coordinate_sum (N C D : ℝ × ℝ) : 
  N = (5, -1) →
  C = (11, 10) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2875_287562


namespace NUMINAMATH_CALUDE_percent_of_a_l2875_287595

theorem percent_of_a (a b c : ℝ) (h1 : c = 0.1 * b) (h2 : b = 2.5 * a) : c = 0.25 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l2875_287595


namespace NUMINAMATH_CALUDE_milburg_population_l2875_287517

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The total population of Milburg -/
def total_population : ℕ := grown_ups + children

theorem milburg_population : total_population = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l2875_287517


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2875_287597

/-- Given a quadratic equation x^2 + bx + c = 0 whose roots are each three more than
    the roots of 2x^2 - 4x - 8 = 0, prove that c = 11 -/
theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ p q : ℝ, (2 * p^2 - 4 * p - 8 = 0) ∧ 
              (2 * q^2 - 4 * q - 8 = 0) ∧ 
              ((p + 3)^2 + b * (p + 3) + c = 0) ∧ 
              ((q + 3)^2 + b * (q + 3) + c = 0)) →
  c = 11 := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_relation_l2875_287597


namespace NUMINAMATH_CALUDE_minimize_distance_sum_l2875_287549

/-- Given points P and Q in the xy-plane, and R on the line segment PQ, 
    prove that R(2, -1/9) minimizes the sum of distances PR + RQ -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) : 
  P = (-3, -4) → 
  Q = (6, 3) → 
  R.1 = 2 → 
  R.2 = -1/9 → 
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • P + t • Q) →
  ∀ (S : ℝ × ℝ), (∃ (u : ℝ), 0 ≤ u ∧ u ≤ 1 ∧ S = (1 - u) • P + u • Q) →
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≤ 
    Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) + Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_minimize_distance_sum_l2875_287549


namespace NUMINAMATH_CALUDE_problem_statement_l2875_287576

theorem problem_statement (x y z : ℝ) 
  (h1 : x * z / (x + y) + y * x / (y + z) + z * y / (z + x) = 2)
  (h2 : z * y / (x + y) + x * z / (y + z) + y * x / (z + x) = 3) :
  y / (x + y) + z / (y + z) + x / (z + x) = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2875_287576


namespace NUMINAMATH_CALUDE_negative_three_times_two_l2875_287561

theorem negative_three_times_two : (-3 : ℤ) * 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_two_l2875_287561


namespace NUMINAMATH_CALUDE_weight_ratio_john_to_mary_l2875_287515

/-- Proves that the ratio of John's weight to Mary's weight is 5:4 given the specified conditions -/
theorem weight_ratio_john_to_mary :
  ∀ (john_weight mary_weight jamison_weight : ℕ),
    mary_weight = 160 →
    mary_weight + 20 = jamison_weight →
    john_weight + mary_weight + jamison_weight = 540 →
    (john_weight : ℚ) / mary_weight = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_john_to_mary_l2875_287515


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l2875_287557

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 11.2)
  (h2 : current_speed = 3.4) : 
  speed_against_current + 2 * current_speed = 18 := by
  sorry

#check mans_speed_with_current

end NUMINAMATH_CALUDE_mans_speed_with_current_l2875_287557


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2875_287572

theorem angle_measure_proof (x : ℝ) : x + (4 * x + 5) = 90 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2875_287572


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2875_287587

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (3 * x - 4)) = 4 → x = 173 / 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2875_287587


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2875_287525

/-- A rectangle with length thrice its breadth and area 108 square meters has a perimeter of 48 meters -/
theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 108) : 2 * (3 * b + b) = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2875_287525


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_two_l2875_287598

theorem fraction_meaningful_iff_not_neg_two (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_two_l2875_287598


namespace NUMINAMATH_CALUDE_starting_to_running_current_ratio_l2875_287558

/-- Proves the ratio of starting current to running current for machinery units -/
theorem starting_to_running_current_ratio
  (num_units : ℕ)
  (running_current : ℝ)
  (min_transformer_load : ℝ)
  (h1 : num_units = 3)
  (h2 : running_current = 40)
  (h3 : min_transformer_load = 240)
  : min_transformer_load / (num_units * running_current) = 2 := by
  sorry

#check starting_to_running_current_ratio

end NUMINAMATH_CALUDE_starting_to_running_current_ratio_l2875_287558


namespace NUMINAMATH_CALUDE_distinct_triangles_in_cube_l2875_287551

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles that can be formed by connecting three different vertices of a cube -/
def distinct_triangles : ℕ := Nat.choose cube_vertices triangle_vertices

theorem distinct_triangles_in_cube :
  distinct_triangles = 56 :=
sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_cube_l2875_287551


namespace NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l2875_287592

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 14*x^3 + 57*x^2 - 132*x + 36

-- Theorem statement
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 14*x^3 + 57*x^2 - 132*x + 36) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3 + √5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 4 - √7 is a root
  p (4 - Real.sqrt 7) = 0 :=
sorry

end NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l2875_287592


namespace NUMINAMATH_CALUDE_other_communities_count_l2875_287538

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 153 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l2875_287538


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_5_not_p_sufficient_not_necessary_implies_a_leq_2_l2875_287582

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem for part (1)
theorem intersection_empty_implies_a_geq_5 (a : ℝ) :
  (a > 0) → (A ∩ B a = ∅) → a ≥ 5 := by sorry

-- Theorem for part (2)
theorem not_p_sufficient_not_necessary_implies_a_leq_2 (a : ℝ) :
  (a > 0) → (∀ x, ¬(p x) → q a x) → (∃ x, q a x ∧ p x) → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_5_not_p_sufficient_not_necessary_implies_a_leq_2_l2875_287582


namespace NUMINAMATH_CALUDE_polygon_perimeter_sum_tan_greater_than_x_l2875_287545

theorem polygon_perimeter_sum (R : ℝ) (h : R > 0) :
  let n : ℕ := 1985
  let θ : ℝ := 2 * Real.pi / n
  let inner_side := 2 * R * Real.sin (θ / 2)
  let outer_side := 2 * R * Real.tan (θ / 2)
  let inner_perimeter := n * inner_side
  let outer_perimeter := n * outer_side
  inner_perimeter + outer_perimeter ≥ 4 * Real.pi * R :=
by
  sorry

theorem tan_greater_than_x (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  Real.tan x ≥ x :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_perimeter_sum_tan_greater_than_x_l2875_287545


namespace NUMINAMATH_CALUDE_intersection_A_B_l2875_287556

-- Define set A
def A : Set ℝ := {x | (x - 2) * (2 * x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | x < 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1/2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2875_287556


namespace NUMINAMATH_CALUDE_total_value_is_20_31_l2875_287586

/-- Represents the value of coins in U.S. Dollars -/
def total_value : ℝ :=
  let us_quarter_value : ℝ := 0.25
  let us_nickel_value : ℝ := 0.05
  let canadian_dime_value : ℝ := 0.10
  let euro_cent_value : ℝ := 0.01
  let british_pence_value : ℝ := 0.01
  let cad_to_usd : ℝ := 0.8
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.4
  let us_quarters : ℝ := 4 * 10 * us_quarter_value
  let us_nickels : ℝ := 9 * 10 * us_nickel_value
  let canadian_dimes : ℝ := 6 * 10 * canadian_dime_value * cad_to_usd
  let euro_cents : ℝ := 5 * 10 * euro_cent_value * eur_to_usd
  let british_pence : ℝ := 3 * 10 * british_pence_value * gbp_to_usd
  us_quarters + us_nickels + canadian_dimes + euro_cents + british_pence

/-- Theorem stating that the total value of Rocco's coins is $20.31 -/
theorem total_value_is_20_31 : total_value = 20.31 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_20_31_l2875_287586


namespace NUMINAMATH_CALUDE_james_age_l2875_287510

theorem james_age (dan_age james_age : ℕ) : 
  (dan_age : ℚ) / james_age = 6 / 5 →
  dan_age + 4 = 28 →
  james_age = 20 := by
sorry

end NUMINAMATH_CALUDE_james_age_l2875_287510


namespace NUMINAMATH_CALUDE_average_MTWT_is_48_l2875_287569

/-- The average temperature for some days -/
def average_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def average_TWTF : ℝ := 46

/-- The temperature on Monday -/
def temp_Monday : ℝ := 43

/-- The temperature on Friday -/
def temp_Friday : ℝ := 35

/-- The number of days in the TWTF group -/
def num_days_TWTF : ℕ := 4

/-- The number of days in the MTWT group -/
def num_days_MTWT : ℕ := 4

/-- Theorem: The average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem average_MTWT_is_48 : 
  (temp_Monday + (average_TWTF * num_days_TWTF - temp_Friday)) / num_days_MTWT = average_some_days :=
by sorry

end NUMINAMATH_CALUDE_average_MTWT_is_48_l2875_287569


namespace NUMINAMATH_CALUDE_max_sides_cube_cross_section_l2875_287555

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where

/-- A plane is a flat, two-dimensional surface that extends infinitely far. -/
structure Plane where

/-- A cross-section is the intersection of a plane with a three-dimensional object. -/
def CrossSection (c : Cube) (p : Plane) : Set (ℝ × ℝ × ℝ) := sorry

/-- The number of sides in a polygon. -/
def NumberOfSides (polygon : Set (ℝ × ℝ × ℝ)) : ℕ := sorry

/-- The maximum number of sides in any cross-section of a cube is 6. -/
theorem max_sides_cube_cross_section (c : Cube) : 
  ∀ p : Plane, NumberOfSides (CrossSection c p) ≤ 6 ∧ 
  ∃ p : Plane, NumberOfSides (CrossSection c p) = 6 :=
sorry

end NUMINAMATH_CALUDE_max_sides_cube_cross_section_l2875_287555


namespace NUMINAMATH_CALUDE_sum_exponents_of_sqrt_largest_perfect_square_12_factorial_l2875_287521

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largest_perfect_square_divisor (n : ℕ) : ℕ := sorry

def prime_factor_exponents (n : ℕ) : List ℕ := sorry

theorem sum_exponents_of_sqrt_largest_perfect_square_12_factorial : 
  (prime_factor_exponents (largest_perfect_square_divisor (factorial 12)).sqrt).sum = 8 := by sorry

end NUMINAMATH_CALUDE_sum_exponents_of_sqrt_largest_perfect_square_12_factorial_l2875_287521


namespace NUMINAMATH_CALUDE_total_miles_traveled_l2875_287599

theorem total_miles_traveled (initial_reading additional_distance : Real) 
  (h1 : initial_reading = 212.3)
  (h2 : additional_distance = 372.0) : 
  initial_reading + additional_distance = 584.3 := by
sorry

end NUMINAMATH_CALUDE_total_miles_traveled_l2875_287599


namespace NUMINAMATH_CALUDE_allocation_methods_l2875_287568

def doctors : ℕ := 2
def nurses : ℕ := 4
def schools : ℕ := 2
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

theorem allocation_methods :
  (Nat.choose doctors doctors_per_school) * (Nat.choose nurses nurses_per_school) = 12 := by
  sorry

end NUMINAMATH_CALUDE_allocation_methods_l2875_287568


namespace NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l2875_287570

theorem consecutive_product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l2875_287570


namespace NUMINAMATH_CALUDE_min_z_shapes_cover_min_z_shapes_necessary_l2875_287529

/-- Represents a cell on the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a Z shape on the table -/
structure ZShape where
  base : Cell
  rotation : Nat

/-- The size of the table -/
def tableSize : Nat := 8

/-- Checks if a cell is within the table bounds -/
def isValidCell (c : Cell) : Prop :=
  c.row ≥ 1 ∧ c.row ≤ tableSize ∧ c.col ≥ 1 ∧ c.col ≤ tableSize

/-- Checks if a Z shape covers a given cell -/
def coversCell (z : ZShape) (c : Cell) : Prop :=
  match z.rotation % 4 with
  | 0 => c = z.base ∨ c = ⟨z.base.row, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 2⟩
  | 1 => c = z.base ∨ c = ⟨z.base.row + 1, z.base.col⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 2⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 2⟩
  | 2 => c = z.base ∨ c = ⟨z.base.row, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col + 2⟩
  | _ => c = z.base ∨ c = ⟨z.base.row - 1, z.base.col⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col - 1⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col - 2⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col - 2⟩

/-- The main theorem stating that 12 Z shapes are sufficient to cover the table -/
theorem min_z_shapes_cover (shapes : List ZShape) : 
  (∀ c : Cell, isValidCell c → ∃ z ∈ shapes, coversCell z c) → 
  shapes.length ≥ 12 :=
sorry

/-- The main theorem stating that 12 Z shapes are necessary to cover the table -/
theorem min_z_shapes_necessary : 
  ∃ shapes : List ZShape, shapes.length = 12 ∧ 
  ∀ c : Cell, isValidCell c → ∃ z ∈ shapes, coversCell z c :=
sorry

end NUMINAMATH_CALUDE_min_z_shapes_cover_min_z_shapes_necessary_l2875_287529


namespace NUMINAMATH_CALUDE_vector_parallelism_l2875_287516

theorem vector_parallelism (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ k : ℝ, k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l2875_287516


namespace NUMINAMATH_CALUDE_gcd_13924_32451_l2875_287542

theorem gcd_13924_32451 : Nat.gcd 13924 32451 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_13924_32451_l2875_287542


namespace NUMINAMATH_CALUDE_jasper_kite_raising_time_l2875_287539

/-- Given Omar's kite-raising rate and Jasper's relative speed, prove Jasper's time to raise his kite -/
theorem jasper_kite_raising_time
  (omar_height : ℝ)
  (omar_time : ℝ)
  (jasper_speed_ratio : ℝ)
  (jasper_height : ℝ)
  (h1 : omar_height = 240)
  (h2 : omar_time = 12)
  (h3 : jasper_speed_ratio = 3)
  (h4 : jasper_height = 600) :
  (jasper_height / (jasper_speed_ratio * (omar_height / omar_time))) = 10 :=
by sorry

end NUMINAMATH_CALUDE_jasper_kite_raising_time_l2875_287539


namespace NUMINAMATH_CALUDE_fifth_day_income_correct_l2875_287552

/-- Calculates the cab driver's income on the fifth day given the income for the first four days and the average income for five days. -/
def fifth_day_income (day1 day2 day3 day4 avg : ℚ) : ℚ :=
  5 * avg - (day1 + day2 + day3 + day4)

/-- Proves that the calculated fifth day income is correct given the income for the first four days and the average income for five days. -/
theorem fifth_day_income_correct (day1 day2 day3 day4 avg : ℚ) :
  let fifth_day := fifth_day_income day1 day2 day3 day4 avg
  (day1 + day2 + day3 + day4 + fifth_day) / 5 = avg :=
by sorry

#eval fifth_day_income 400 250 650 400 440

end NUMINAMATH_CALUDE_fifth_day_income_correct_l2875_287552


namespace NUMINAMATH_CALUDE_problem_statement_l2875_287574

theorem problem_statement (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^4374 - 1/x^4374 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2875_287574


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l2875_287504

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of cars brought by guests -/
def guest_cars : ℕ := 10

/-- The number of cars belonging to Dylan's parents -/
def parent_cars : ℕ := 2

/-- The total number of cars in the parking lot -/
def total_cars : ℕ := guest_cars + parent_cars

/-- Theorem stating the total number of car wheels in the parking lot -/
theorem total_wheels_in_parking_lot : 
  (total_cars * wheels_per_car) = 48 := by
sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l2875_287504


namespace NUMINAMATH_CALUDE_spherical_coords_reflection_l2875_287560

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    prove that the point (x, y, -z) has spherical coordinates (ρ, θ, π - φ) -/
theorem spherical_coords_reflection (x y z ρ θ φ : Real) 
  (h1 : ρ > 0) 
  (h2 : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h3 : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h4 : x = ρ * Real.sin φ * Real.cos θ)
  (h5 : y = ρ * Real.sin φ * Real.sin θ)
  (h6 : z = ρ * Real.cos φ)
  (h7 : ρ = 4)
  (h8 : θ = Real.pi / 4)
  (h9 : φ = Real.pi / 6) :
  ∃ (ρ' θ' φ' : Real),
    ρ' = ρ ∧
    θ' = θ ∧
    φ' = Real.pi - φ ∧
    x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    -z = ρ' * Real.cos φ' ∧
    ρ' > 0 ∧
    0 ≤ θ' ∧ θ' < 2 * Real.pi ∧
    0 ≤ φ' ∧ φ' ≤ Real.pi :=
by sorry

end NUMINAMATH_CALUDE_spherical_coords_reflection_l2875_287560


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2875_287518

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2875_287518


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2875_287567

theorem sum_remainder_mod_seven : 
  (51730 % 7 + 51731 % 7 + 51732 % 7 + 51733 % 7 + 51734 % 7 + 51735 % 7) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2875_287567


namespace NUMINAMATH_CALUDE_four_digit_sum_mod_1000_l2875_287563

def four_digit_sum : ℕ := sorry

theorem four_digit_sum_mod_1000 : four_digit_sum % 1000 = 320 := by sorry

end NUMINAMATH_CALUDE_four_digit_sum_mod_1000_l2875_287563


namespace NUMINAMATH_CALUDE_equivalence_of_inequalities_l2875_287531

theorem equivalence_of_inequalities (a : ℝ) : a - 1 > 0 ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_inequalities_l2875_287531


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2875_287528

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

theorem intersection_complement_equality : B ∩ (U \ A) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2875_287528


namespace NUMINAMATH_CALUDE_product_expansion_l2875_287564

theorem product_expansion (x : ℝ) : (x + 2) * (x^2 + 3*x + 4) = x^3 + 5*x^2 + 10*x + 8 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2875_287564


namespace NUMINAMATH_CALUDE_andrew_mango_purchase_l2875_287523

-- Define the given constants
def grape_quantity : ℕ := 6
def grape_price : ℕ := 74
def mango_price : ℕ := 59
def total_paid : ℕ := 975

-- Define the function to calculate the mango quantity
def mango_quantity : ℕ := (total_paid - grape_quantity * grape_price) / mango_price

-- Theorem statement
theorem andrew_mango_purchase :
  mango_quantity = 9 := by
  sorry

end NUMINAMATH_CALUDE_andrew_mango_purchase_l2875_287523


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2875_287593

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2875_287593


namespace NUMINAMATH_CALUDE_average_salary_raj_roshan_l2875_287533

theorem average_salary_raj_roshan (raj_salary roshan_salary : ℕ) : 
  (raj_salary + roshan_salary + 7000) / 3 = 5000 →
  (raj_salary + roshan_salary) / 2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_raj_roshan_l2875_287533


namespace NUMINAMATH_CALUDE_hcd_8100_270_minus_8_l2875_287591

theorem hcd_8100_270_minus_8 : Nat.gcd 8100 270 - 8 = 262 := by
  sorry

end NUMINAMATH_CALUDE_hcd_8100_270_minus_8_l2875_287591


namespace NUMINAMATH_CALUDE_fibonacci_sum_cube_square_l2875_287581

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define a predicate for Fibonacci numbers
def isFibonacci (n : ℕ) : Prop := ∃ k, fib k = n

-- Define the theorem
theorem fibonacci_sum_cube_square :
  ∀ a b : ℕ,
  isFibonacci a ∧ 49 < a ∧ a < 61 ∧
  isFibonacci b ∧ 59 < b ∧ b < 71 →
  a^3 + b^2 = 170096 :=
sorry

end NUMINAMATH_CALUDE_fibonacci_sum_cube_square_l2875_287581


namespace NUMINAMATH_CALUDE_rhombus_area_from_diagonals_l2875_287589

/-- The area of a rhombus given its diagonals -/
theorem rhombus_area_from_diagonals (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  (1 / 2 : ℝ) * d1 * d2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_diagonals_l2875_287589


namespace NUMINAMATH_CALUDE_horner_method_correct_l2875_287578

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 4x^2 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x - 4

theorem horner_method_correct :
  f 3 = horner [2, 3, 4, 5, -4] 3 ∧ horner [2, 3, 4, 5, -4] 3 = 290 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_correct_l2875_287578


namespace NUMINAMATH_CALUDE_quadrilateral_existence_l2875_287548

theorem quadrilateral_existence : ∃ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
  d = 2 * a ∧
  a + b + c + d = 2 ∧
  a + b + c > d ∧
  a + b + d > c ∧
  a + c + d > b ∧
  b + c + d > a := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_existence_l2875_287548


namespace NUMINAMATH_CALUDE_paths_count_l2875_287532

/-- The number of paths between two points given the number of right and down steps -/
def numPaths (right down : ℕ) : ℕ := sorry

/-- The total number of paths from A to D via B and C -/
def totalPaths : ℕ :=
  let pathsAB := numPaths 2 2  -- B is 2 right and 2 down from A
  let pathsBC := numPaths 1 3  -- C is 1 right and 3 down from B
  let pathsCD := numPaths 3 1  -- D is 3 right and 1 down from C
  pathsAB * pathsBC * pathsCD

theorem paths_count : totalPaths = 96 := by sorry

end NUMINAMATH_CALUDE_paths_count_l2875_287532


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2875_287507

theorem y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2875_287507


namespace NUMINAMATH_CALUDE_business_income_calculation_l2875_287522

theorem business_income_calculation 
  (spending income : ℕ) 
  (spending_income_ratio : spending * 9 = income * 5) 
  (profit : ℕ) 
  (profit_equation : profit = income - spending) 
  (profit_value : profit = 48000) : income = 108000 := by
sorry

end NUMINAMATH_CALUDE_business_income_calculation_l2875_287522


namespace NUMINAMATH_CALUDE_chairs_remaining_l2875_287584

def classroom_chairs (total red yellow blue green orange : ℕ) : Prop :=
  total = 62 ∧
  red = 4 ∧
  yellow = 2 * red ∧
  blue = 3 * yellow ∧
  green = blue / 2 ∧
  orange = green + 2 ∧
  total = red + yellow + blue + green + orange

def lisa_borrows (total borrowed : ℕ) : Prop :=
  borrowed = total / 10

def carla_borrows (remaining borrowed : ℕ) : Prop :=
  borrowed = remaining / 5

theorem chairs_remaining 
  (total red yellow blue green orange : ℕ)
  (lisa_borrowed carla_borrowed : ℕ)
  (h1 : classroom_chairs total red yellow blue green orange)
  (h2 : lisa_borrows total lisa_borrowed)
  (h3 : carla_borrows (total - lisa_borrowed) carla_borrowed) :
  total - lisa_borrowed - carla_borrowed = 45 :=
sorry

end NUMINAMATH_CALUDE_chairs_remaining_l2875_287584


namespace NUMINAMATH_CALUDE_cubic_two_intersections_l2875_287503

/-- A cubic function that intersects the x-axis at exactly two points -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem cubic_two_intersections :
  ∃! a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ∧
  a = -4/27 :=
sorry

end NUMINAMATH_CALUDE_cubic_two_intersections_l2875_287503


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2875_287594

def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (p 1 = 0) ∧ (p 2 = 0) ∧ (p 4 = 0) ∧
  (∀ x : ℝ, p x = 0 → x = 1 ∨ x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2875_287594


namespace NUMINAMATH_CALUDE_difference_of_x_and_y_l2875_287553

theorem difference_of_x_and_y (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_x_and_y_l2875_287553


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l2875_287590

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 1 ∧ (4 / a₀) + (9 / b₀) = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l2875_287590


namespace NUMINAMATH_CALUDE_equation_solutions_l2875_287579

theorem equation_solutions :
  (∀ x : ℝ, 25 * x^2 = 81 ↔ x = 9/5 ∨ x = -9/5) ∧
  (∀ x : ℝ, (x - 2)^2 = 25 ↔ x = 7 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2875_287579


namespace NUMINAMATH_CALUDE_committee_vote_change_l2875_287571

theorem committee_vote_change (total : ℕ) (a b a' b' : ℕ) : 
  total = 300 →
  a + b = total →
  b > a →
  a' + b' = total →
  a' - b' = 3 * (b - a) →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by sorry

end NUMINAMATH_CALUDE_committee_vote_change_l2875_287571


namespace NUMINAMATH_CALUDE_boss_contribution_l2875_287506

def gift_cost : ℝ := 100
def employee_contribution : ℝ := 11
def num_employees : ℕ := 5

theorem boss_contribution :
  ∃ (boss_amount : ℝ),
    boss_amount = 15 ∧
    ∃ (todd_amount : ℝ),
      todd_amount = 2 * boss_amount ∧
      boss_amount + todd_amount + (num_employees : ℝ) * employee_contribution = gift_cost :=
by sorry

end NUMINAMATH_CALUDE_boss_contribution_l2875_287506


namespace NUMINAMATH_CALUDE_contractor_average_wage_l2875_287541

def average_wage (male_count female_count child_count : ℕ)
                 (male_wage female_wage child_wage : ℚ) : ℚ :=
  let total_workers := male_count + female_count + child_count
  let total_wage := male_count * male_wage + female_count * female_wage + child_count * child_wage
  total_wage / total_workers

theorem contractor_average_wage :
  average_wage 20 15 5 25 20 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_contractor_average_wage_l2875_287541


namespace NUMINAMATH_CALUDE_min_value_theorem_l2875_287526

/-- Given real numbers a, b, c, d satisfying the given conditions, 
    the minimum value of (a - c)^2 + (b - d)^2 is 1/10 -/
theorem min_value_theorem (a b c d : ℝ) 
    (h1 : (2 * a^2 - Real.log a) / b = 1) 
    (h2 : (3 * c - 2) / d = 1) : 
  ∃ (x y : ℝ), ∀ (a' b' c' d' : ℝ), 
    (2 * a'^2 - Real.log a') / b' = 1 → 
    (3 * c' - 2) / d' = 1 → 
    (a' - c')^2 + (b' - d')^2 ≥ (1 : ℝ) / 10 ∧
    (x - y)^2 + ((2 * x^2 - Real.log x) - (3 * y - 2))^2 = (1 : ℝ) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2875_287526


namespace NUMINAMATH_CALUDE_sports_competition_results_l2875_287583

-- Define the probabilities of School A winning each event
def p1 : ℝ := 0.5
def p2 : ℝ := 0.4
def p3 : ℝ := 0.8

-- Define the score for winning an event
def win_score : ℕ := 10

-- Define the probability of School A winning the championship
def prob_A_wins : ℝ := p1 * p2 * p3 + p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3

-- Define the distribution of School B's total score
def dist_B : List (ℝ × ℝ) := [
  (0, (1 - p1) * (1 - p2) * (1 - p3)),
  (10, p1 * (1 - p2) * (1 - p3) + (1 - p1) * p2 * (1 - p3) + (1 - p1) * (1 - p2) * p3),
  (20, p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3),
  (30, p1 * p2 * p3)
]

-- Define the expectation of School B's total score
def exp_B : ℝ := (dist_B.map (λ p => p.1 * p.2)).sum

-- Theorem statement
theorem sports_competition_results :
  prob_A_wins = 0.6 ∧ exp_B = 13 := by sorry

end NUMINAMATH_CALUDE_sports_competition_results_l2875_287583


namespace NUMINAMATH_CALUDE_celine_book_days_l2875_287535

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The daily charge for borrowing a book (in dollars) -/
def daily_charge : ℚ := 1/2

/-- The total amount Celine paid (in dollars) -/
def total_paid : ℚ := 41

/-- The number of books Celine borrowed -/
def num_books : ℕ := 3

theorem celine_book_days :
  ∃ (x : ℕ), 
    daily_charge * x + daily_charge * (num_books - 1) * days_in_may = total_paid ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_celine_book_days_l2875_287535


namespace NUMINAMATH_CALUDE_transformed_quadratic_roots_l2875_287559

theorem transformed_quadratic_roots 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * r^2 + b * r + c = 0) 
  (h3 : a * s^2 + b * s + c = 0) : 
  (a * r + b)^2 - b * (a * r + b) + a * c = 0 ∧ 
  (a * s + b)^2 - b * (a * s + b) + a * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_transformed_quadratic_roots_l2875_287559


namespace NUMINAMATH_CALUDE_mary_ray_difference_l2875_287580

/-- The number of chickens taken by each person -/
structure ChickenDistribution where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken distribution problem -/
def valid_distribution (d : ChickenDistribution) : Prop :=
  d.john = d.mary + 5 ∧
  d.ray < d.mary ∧
  d.ray = 10 ∧
  d.john = d.ray + 11

/-- The theorem stating the difference between Mary's and Ray's chickens -/
theorem mary_ray_difference (d : ChickenDistribution) 
  (h : valid_distribution d) : d.mary - d.ray = 6 := by
  sorry

#check mary_ray_difference

end NUMINAMATH_CALUDE_mary_ray_difference_l2875_287580


namespace NUMINAMATH_CALUDE_exists_uncolored_diameter_l2875_287501

/-- Represents a circle with some arcs colored black -/
structure BlackArcCircle where
  /-- The total circumference of the circle -/
  circumference : ℝ
  /-- The total length of black arcs -/
  blackArcLength : ℝ
  /-- Assumption that the black arc length is less than half the circumference -/
  blackArcLengthLessThanHalf : blackArcLength < circumference / 2

/-- A point on the circle -/
structure CirclePoint where
  /-- The angle of the point relative to a fixed reference point -/
  angle : ℝ

/-- Represents a diameter of the circle -/
structure Diameter where
  /-- One endpoint of the diameter -/
  point1 : CirclePoint
  /-- The other endpoint of the diameter -/
  point2 : CirclePoint
  /-- Assumption that the points are opposite each other on the circle -/
  oppositePoints : point2.angle = point1.angle + π

/-- Function to determine if a point is on a black arc -/
def isOnBlackArc (c : BlackArcCircle) (p : CirclePoint) : Prop := sorry

/-- Theorem stating that there exists a diameter with both ends uncolored -/
theorem exists_uncolored_diameter (c : BlackArcCircle) : 
  ∃ d : Diameter, ¬isOnBlackArc c d.point1 ∧ ¬isOnBlackArc c d.point2 := by sorry

end NUMINAMATH_CALUDE_exists_uncolored_diameter_l2875_287501


namespace NUMINAMATH_CALUDE_divided_triangle_angles_l2875_287540

/-- A triangle that can be divided into several smaller triangles -/
structure DividedTriangle where
  -- The number of triangles the original triangle is divided into
  num_divisions : ℕ
  -- Assertion that there are at least two divisions
  h_at_least_two : num_divisions ≥ 2

/-- Represents the properties of the divided triangles -/
structure DivisionProperties (T : DividedTriangle) where
  -- The number of equilateral triangles in the division
  num_equilateral : ℕ
  -- The number of isosceles (non-equilateral) triangles in the division
  num_isosceles : ℕ
  -- Assertion that there is exactly one isosceles triangle
  h_one_isosceles : num_isosceles = 1
  -- Assertion that all other triangles are equilateral
  h_rest_equilateral : num_equilateral + num_isosceles = T.num_divisions

/-- The theorem stating the angles of the original triangle -/
theorem divided_triangle_angles (T : DividedTriangle) (P : DivisionProperties T) :
  ∃ (a b c : ℝ), a = 30 ∧ b = 60 ∧ c = 90 ∧ a + b + c = 180 :=
sorry

end NUMINAMATH_CALUDE_divided_triangle_angles_l2875_287540


namespace NUMINAMATH_CALUDE_total_students_l2875_287511

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 300) :
  boys + girls = 780 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l2875_287511


namespace NUMINAMATH_CALUDE_fraction_value_l2875_287588

theorem fraction_value : (2222 - 2123)^2 / 121 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2875_287588


namespace NUMINAMATH_CALUDE_probability_point_not_in_inner_square_l2875_287502

/-- The probability that a random point in a larger square is not in a smaller square inside it. -/
theorem probability_point_not_in_inner_square
  (area_A : ℝ) (perimeter_B : ℝ)
  (h_area_A : area_A = 65)
  (h_perimeter_B : perimeter_B = 16)
  (h_positive_A : area_A > 0)
  (h_positive_B : perimeter_B > 0) :
  let side_B := perimeter_B / 4
  let area_B := side_B ^ 2
  (area_A - area_B) / area_A = (65 - 16) / 65 := by
  sorry


end NUMINAMATH_CALUDE_probability_point_not_in_inner_square_l2875_287502


namespace NUMINAMATH_CALUDE_intersection_sum_l2875_287565

theorem intersection_sum (c d : ℝ) :
  (∀ x y : ℝ, x = (1/3) * y + c ↔ y = (1/3) * x + d) →
  3 = (1/3) * 0 + c →
  0 = (1/3) * 3 + d →
  c + d = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2875_287565


namespace NUMINAMATH_CALUDE_savings_to_earnings_ratio_l2875_287544

/-- Proves that the ratio of combined savings to total earnings is 1:2 --/
theorem savings_to_earnings_ratio
  (kimmie_earnings : ℚ)
  (zahra_earnings : ℚ)
  (combined_savings : ℚ)
  (h1 : kimmie_earnings = 450)
  (h2 : zahra_earnings = kimmie_earnings - kimmie_earnings / 3)
  (h3 : combined_savings = 375) :
  combined_savings / (kimmie_earnings + zahra_earnings) = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_savings_to_earnings_ratio_l2875_287544


namespace NUMINAMATH_CALUDE_smallest_n_for_milly_victory_l2875_287546

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ a b c d : ℕ, a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n →
    (coloring a = coloring b ∧ coloring b = coloring c ∧ coloring c = coloring d) →
    a + b + c ≠ d

theorem smallest_n_for_milly_victory : 
  (∀ n < 11, ∃ coloring : ℕ → Bool, is_valid_coloring n coloring) ∧
  (∀ coloring : ℕ → Bool, ¬ is_valid_coloring 11 coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_milly_victory_l2875_287546


namespace NUMINAMATH_CALUDE_largest_colorable_3subsets_correct_l2875_287509

/-- The largest number of 3-subsets that can be chosen from a set of n elements
    such that there always exists a 2-coloring with no monochromatic chosen 3-subset -/
def largest_colorable_3subsets (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 4
  else if n = 5 ∨ n = 6 then 9
  else if n ≥ 7 then 6
  else 0

/-- The theorem stating the correct values for the largest number of colorable 3-subsets -/
theorem largest_colorable_3subsets_correct (n : ℕ) (h : n ≥ 3) :
  largest_colorable_3subsets n =
    if n = 3 then 1
    else if n = 4 then 4
    else if n = 5 ∨ n = 6 then 9
    else 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_colorable_3subsets_correct_l2875_287509


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l2875_287500

/-- Sum of arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- Sum of even integers from 2 to 120 -/
def a : ℕ := arithmeticSum 2 120 60

/-- Sum of odd integers from 1 to 119 -/
def b : ℕ := arithmeticSum 1 119 60

/-- The difference between the sum of even integers from 2 to 120 and
    the sum of odd integers from 1 to 119 is 60 -/
theorem even_odd_sum_difference : a - b = 60 := by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l2875_287500


namespace NUMINAMATH_CALUDE_sqrt_36_minus_k_squared_minus_6_equals_zero_l2875_287573

theorem sqrt_36_minus_k_squared_minus_6_equals_zero (k : ℝ) :
  Real.sqrt (36 - k^2) - 6 = 0 ↔ k = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_36_minus_k_squared_minus_6_equals_zero_l2875_287573


namespace NUMINAMATH_CALUDE_not_right_triangle_A_right_triangle_B_right_triangle_C_right_triangle_D_main_result_l2875_287512

/-- A function to check if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that √3, 2, √5 cannot form a right triangle --/
theorem not_right_triangle_A : ¬ isRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) := by
  sorry

/-- Theorem stating that 3, 4, 5 can form a right triangle --/
theorem right_triangle_B : isRightTriangle 3 4 5 := by
  sorry

/-- Theorem stating that 0.6, 0.8, 1 can form a right triangle --/
theorem right_triangle_C : isRightTriangle 0.6 0.8 1 := by
  sorry

/-- Theorem stating that 130, 120, 50 can form a right triangle --/
theorem right_triangle_D : isRightTriangle 130 120 50 := by
  sorry

/-- Main theorem combining all the above results --/
theorem main_result : 
  ¬ isRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) ∧
  isRightTriangle 3 4 5 ∧
  isRightTriangle 0.6 0.8 1 ∧
  isRightTriangle 130 120 50 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_A_right_triangle_B_right_triangle_C_right_triangle_D_main_result_l2875_287512


namespace NUMINAMATH_CALUDE_equal_area_perimeter_rectangle_dimensions_l2875_287596

/-- A rectangle with integer side lengths where the area equals the perimeter. -/
structure EqualAreaPerimeterRectangle where
  width : ℕ
  length : ℕ
  area_eq_perimeter : width * length = 2 * (width + length)

/-- The possible dimensions of a rectangle with integer side lengths where the area equals the perimeter. -/
def valid_dimensions : Set (ℕ × ℕ) :=
  {(4, 4), (3, 6), (6, 3)}

/-- Theorem stating that the only valid dimensions for a rectangle with integer side lengths
    where the area equals the perimeter are 4x4, 3x6, or 6x3. -/
theorem equal_area_perimeter_rectangle_dimensions (r : EqualAreaPerimeterRectangle) :
  (r.width, r.length) ∈ valid_dimensions := by
  sorry

#check equal_area_perimeter_rectangle_dimensions

end NUMINAMATH_CALUDE_equal_area_perimeter_rectangle_dimensions_l2875_287596


namespace NUMINAMATH_CALUDE_total_money_is_75_l2875_287543

/-- Represents the money distribution and orange selling scenario -/
structure MoneyDistribution where
  x : ℝ  -- The common factor in the money distribution
  cara_money : ℝ := 4 * x
  janet_money : ℝ := 5 * x
  jerry_money : ℝ := 6 * x
  total_money : ℝ := cara_money + janet_money + jerry_money
  combined_money : ℝ := cara_money + janet_money
  selling_price_ratio : ℝ := 0.8
  loss : ℝ := combined_money - (selling_price_ratio * combined_money)

/-- Theorem stating the total amount of money given the conditions -/
theorem total_money_is_75 (d : MoneyDistribution) 
  (h_loss : d.loss = 9) : d.total_money = 75 := by
  sorry


end NUMINAMATH_CALUDE_total_money_is_75_l2875_287543


namespace NUMINAMATH_CALUDE_sam_investment_result_l2875_287524

/-- Calculates the final amount of an investment given initial conditions and interest rates --/
def calculate_investment (initial_investment : ℝ) (first_rate : ℝ) (first_years : ℕ) 
  (multiplier : ℝ) (second_rate : ℝ) : ℝ :=
  let first_phase := initial_investment * (1 + first_rate) ^ first_years
  let second_phase := first_phase * multiplier
  let final_amount := second_phase * (1 + second_rate)
  final_amount

/-- Theorem stating the final amount of Sam's investment --/
theorem sam_investment_result : 
  calculate_investment 10000 0.20 3 3 0.15 = 59616 := by
  sorry

#eval calculate_investment 10000 0.20 3 3 0.15

end NUMINAMATH_CALUDE_sam_investment_result_l2875_287524


namespace NUMINAMATH_CALUDE_randys_trip_length_l2875_287537

theorem randys_trip_length :
  ∀ (total : ℚ),
  (total / 3 : ℚ) + 20 + (total / 5 : ℚ) = total →
  total = 300 / 7 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2875_287537


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l2875_287547

theorem max_value_x_sqrt_1_minus_4x_squared (x : ℝ) :
  0 < x → x < 1/2 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l2875_287547


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_count_l2875_287505

/-- The number of ways to choose 4 points from 10 distinct points on a circle's circumference to form convex quadrilaterals -/
def convex_quadrilaterals_from_circle_points (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Theorem stating that the number of convex quadrilaterals formed from 10 points on a circle is 210 -/
theorem convex_quadrilaterals_count :
  convex_quadrilaterals_from_circle_points 10 4 = 210 := by
  sorry

#eval convex_quadrilaterals_from_circle_points 10 4

end NUMINAMATH_CALUDE_convex_quadrilaterals_count_l2875_287505


namespace NUMINAMATH_CALUDE_x_intercept_is_four_l2875_287585

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 4 -/
theorem x_intercept_is_four :
  let l : Line := { x₁ := 10, y₁ := 3, x₂ := -10, y₂ := -7 }
  x_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_four_l2875_287585


namespace NUMINAMATH_CALUDE_factor_expression_l2875_287527

theorem factor_expression (x : ℝ) : 75 * x^11 + 135 * x^22 = 15 * x^11 * (5 + 9 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2875_287527


namespace NUMINAMATH_CALUDE_thomas_blocks_count_l2875_287536

/-- The number of wooden blocks Thomas used in total -/
def total_blocks (stack1 stack2 stack3 stack4 stack5 : ℕ) : ℕ :=
  stack1 + stack2 + stack3 + stack4 + stack5

/-- Theorem stating the total number of blocks Thomas used -/
theorem thomas_blocks_count :
  ∃ (stack1 stack2 stack3 stack4 stack5 : ℕ),
    stack1 = 7 ∧
    stack2 = stack1 + 3 ∧
    stack3 = stack2 - 6 ∧
    stack4 = stack3 + 10 ∧
    stack5 = 2 * stack2 ∧
    total_blocks stack1 stack2 stack3 stack4 stack5 = 55 :=
by
  sorry


end NUMINAMATH_CALUDE_thomas_blocks_count_l2875_287536


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l2875_287554

theorem tan_alpha_minus_pi_over_four (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan β = 1/3) :
  Real.tan (α - π/4) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l2875_287554
