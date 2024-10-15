import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_subset_sequence_l2317_231735

theorem smallest_n_for_subset_sequence (X : Finset ℕ) (h : X.card = 100) :
  let n := 2 * Nat.choose 100 50 + 2 * Nat.choose 100 49 + 1
  ∀ (A : Fin n → Finset ℕ), (∀ i, A i ⊆ X) →
    (∃ i j k, i < j ∧ j < k ∧ (A i ⊆ A j ∧ A j ⊆ A k ∨ A i ⊇ A j ∧ A j ⊇ A k)) ∧
  ∀ m < n, ∃ (B : Fin m → Finset ℕ), (∀ i, B i ⊆ X) ∧
    ¬(∃ i j k, i < j ∧ j < k ∧ (B i ⊆ B j ∧ B j ⊆ B k ∨ B i ⊇ B j ∧ B j ⊇ B k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_subset_sequence_l2317_231735


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l2317_231715

theorem square_root_of_sixteen (x : ℝ) : (x + 3) ^ 2 = 16 → x = 1 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l2317_231715


namespace NUMINAMATH_CALUDE_min_transportation_fee_l2317_231729

/-- Represents the transportation problem with given parameters -/
structure TransportProblem where
  total_goods : ℕ
  large_truck_capacity : ℕ
  large_truck_cost : ℕ
  small_truck_capacity : ℕ
  small_truck_cost : ℕ

/-- Calculates the transportation cost for a given number of large and small trucks -/
def transportation_cost (p : TransportProblem) (large_trucks : ℕ) (small_trucks : ℕ) : ℕ :=
  large_trucks * p.large_truck_cost + small_trucks * p.small_truck_cost

/-- Checks if a combination of trucks can transport all goods -/
def can_transport_all (p : TransportProblem) (large_trucks : ℕ) (small_trucks : ℕ) : Prop :=
  large_trucks * p.large_truck_capacity + small_trucks * p.small_truck_capacity ≥ p.total_goods

/-- Theorem stating that the minimum transportation fee is 1800 yuan -/
theorem min_transportation_fee (p : TransportProblem) 
    (h1 : p.total_goods = 20)
    (h2 : p.large_truck_capacity = 7)
    (h3 : p.large_truck_cost = 600)
    (h4 : p.small_truck_capacity = 4)
    (h5 : p.small_truck_cost = 400) :
    (∀ large_trucks small_trucks, can_transport_all p large_trucks small_trucks →
      transportation_cost p large_trucks small_trucks ≥ 1800) ∧
    (∃ large_trucks small_trucks, can_transport_all p large_trucks small_trucks ∧
      transportation_cost p large_trucks small_trucks = 1800) :=
  sorry


end NUMINAMATH_CALUDE_min_transportation_fee_l2317_231729


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l2317_231751

theorem right_triangle_squares_area (a b : ℝ) (ha : a = 3) (hb : b = 9) :
  let c := Real.sqrt (a^2 + b^2)
  a^2 + b^2 + c^2 + (1/2 * a * b) = 193.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l2317_231751


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2317_231759

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2317_231759


namespace NUMINAMATH_CALUDE_investment_proportion_l2317_231714

/-- Given two investors X and Y, where X invested 5000 and their profit is divided in the ratio 2:6,
    prove that Y's investment is 15000. -/
theorem investment_proportion (x_investment y_investment : ℕ) (profit_ratio_x profit_ratio_y : ℕ) :
  x_investment = 5000 →
  profit_ratio_x = 2 →
  profit_ratio_y = 6 →
  y_investment = 15000 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_proportion_l2317_231714


namespace NUMINAMATH_CALUDE_sum_of_first_five_primes_l2317_231710

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem sum_of_first_five_primes :
  first_five_primes.sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_five_primes_l2317_231710


namespace NUMINAMATH_CALUDE_novel_to_history_ratio_l2317_231720

theorem novel_to_history_ratio :
  let science_pages : ℕ := 600
  let history_pages : ℕ := 300
  let novel_pages : ℕ := science_pages / 4
  novel_pages.gcd history_pages = novel_pages →
  (novel_pages / novel_pages.gcd history_pages) = 1 ∧
  (history_pages / novel_pages.gcd history_pages) = 2 :=
by sorry

end NUMINAMATH_CALUDE_novel_to_history_ratio_l2317_231720


namespace NUMINAMATH_CALUDE_remainder_problem_l2317_231727

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 19) → N % 13 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2317_231727


namespace NUMINAMATH_CALUDE_monday_rain_inches_l2317_231753

/-- Proves that the number of inches of rain collected on Monday is 4 -/
theorem monday_rain_inches (
  gallons_per_inch : ℝ)
  (tuesday_rain : ℝ)
  (price_per_gallon : ℝ)
  (total_revenue : ℝ)
  (h1 : gallons_per_inch = 15)
  (h2 : tuesday_rain = 3)
  (h3 : price_per_gallon = 1.2)
  (h4 : total_revenue = 126)
  : ∃ (monday_rain : ℝ), monday_rain = 4 ∧
    gallons_per_inch * (monday_rain + tuesday_rain) * price_per_gallon = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_monday_rain_inches_l2317_231753


namespace NUMINAMATH_CALUDE_arrange_40521_eq_96_l2317_231732

/-- The number of ways to arrange the digits of 40,521 to form a 5-digit number -/
def arrange_40521 : ℕ :=
  let digits : List ℕ := [4, 0, 5, 2, 1]
  let n : ℕ := digits.length
  let non_zero_digits : ℕ := (digits.filter (· ≠ 0)).length
  (n - 1) * Nat.factorial (n - 1)

theorem arrange_40521_eq_96 : arrange_40521 = 96 := by
  sorry

end NUMINAMATH_CALUDE_arrange_40521_eq_96_l2317_231732


namespace NUMINAMATH_CALUDE_five_people_six_chairs_l2317_231700

/-- The number of ways to arrange n people in m chairs in a row -/
def arrangePeopleInChairs (n : ℕ) (m : ℕ) : ℕ :=
  (m - n + 1).factorial

theorem five_people_six_chairs :
  arrangePeopleInChairs 5 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_five_people_six_chairs_l2317_231700


namespace NUMINAMATH_CALUDE_alley_width_l2317_231758

/-- Given a ladder of length a in an alley, making angles of 60° and 45° with the ground on opposite walls, 
    the width of the alley w is equal to (√3 * a) / 2. -/
theorem alley_width (a : ℝ) (w : ℝ) (h : ℝ) (k : ℝ) : 
  a > 0 → 
  k = a * (1 / 2) → 
  h = a * (Real.sqrt 2 / 2) → 
  w ^ 2 = h ^ 2 + k ^ 2 → 
  w = (Real.sqrt 3 * a) / 2 := by
sorry

end NUMINAMATH_CALUDE_alley_width_l2317_231758


namespace NUMINAMATH_CALUDE_D_300_l2317_231763

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 300 -/
def primeFactor300 : List ℕ+ := [2, 2, 3, 5, 5]

/-- Theorem: The number of ways to write 300 as a product of integers greater than 1, where the order matters, is 35. -/
theorem D_300 : D 300 = 35 := by sorry

end NUMINAMATH_CALUDE_D_300_l2317_231763


namespace NUMINAMATH_CALUDE_cross_out_all_stars_star_remains_uncrossed_l2317_231776

/-- Represents a 2n × 2n table with stars -/
structure StarTable (n : ℕ) where
  stars : Finset (Fin (2*n) × Fin (2*n))

/-- Theorem for part (a) -/
theorem cross_out_all_stars (n : ℕ) (table : StarTable n) 
  (h : table.stars.card = 3*n) :
  ∃ (rows columns : Finset (Fin (2*n))),
    rows.card = n ∧ 
    columns.card = n ∧
    (∀ star ∈ table.stars, star.1 ∈ rows ∨ star.2 ∈ columns) :=
sorry

/-- Theorem for part (b) -/
theorem star_remains_uncrossed (n : ℕ) (table : StarTable n)
  (h : table.stars.card = 3*n + 1) :
  ∀ (rows columns : Finset (Fin (2*n))),
    rows.card = n →
    columns.card = n →
    ∃ star ∈ table.stars, star.1 ∉ rows ∧ star.2 ∉ columns :=
sorry

end NUMINAMATH_CALUDE_cross_out_all_stars_star_remains_uncrossed_l2317_231776


namespace NUMINAMATH_CALUDE_intersection_A_B_l2317_231750

def A : Set ℕ := {x | 0 < x ∧ x < 6}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_A_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2317_231750


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2317_231712

/-- The parabola y^2 = 2px where p > 0 -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The moving line y = kx + b where k ≠ 0 and b ≠ 0 -/
def movingLine (k b x y : ℝ) : Prop := y = k*x + b ∧ k ≠ 0 ∧ b ≠ 0

/-- The slopes of OA and OB multiply to √3 -/
def slopeProduct (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = Real.sqrt 3

/-- The theorem stating that the line always passes through a fixed point -/
theorem fixed_point_theorem (p k b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ k₁ k₂,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    movingLine k b x₁ y₁ ∧ movingLine k b x₂ y₂ ∧
    slopeProduct k₁ k₂) →
  movingLine k b (-2 * Real.sqrt 3 * p / 3) 0 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2317_231712


namespace NUMINAMATH_CALUDE_festival_attendance_l2317_231788

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900)
  (h3 : ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧
    (3 * girls) / 4 + (boys / 3) = festival_attendees) :
  ∃ (girls : ℕ), (3 * girls) / 4 = 720 := by
sorry

end NUMINAMATH_CALUDE_festival_attendance_l2317_231788


namespace NUMINAMATH_CALUDE_min_value_fraction_l2317_231742

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (a + 2) * (b + 2) / (16 * a * b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2317_231742


namespace NUMINAMATH_CALUDE_sarka_age_l2317_231762

/-- Represents the ages of three sisters and their mother -/
structure FamilyAges where
  sarka : ℕ
  liba : ℕ
  eliska : ℕ
  mother : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.liba = ages.sarka + 3 ∧
  ages.eliska = ages.sarka + 8 ∧
  ages.mother = ages.sarka + 29 ∧
  (ages.sarka + ages.liba + ages.eliska + ages.mother) / 4 = 21

/-- The theorem stating Šárka's age -/
theorem sarka_age :
  ∃ (ages : FamilyAges), problem_conditions ages ∧ ages.sarka = 11 := by
  sorry

end NUMINAMATH_CALUDE_sarka_age_l2317_231762


namespace NUMINAMATH_CALUDE_factor_condition_l2317_231765

theorem factor_condition (t : ℚ) : 
  (∃ k : ℚ, (X - t) * k = 3 * X^2 + 10 * X - 8) ↔ (t = 2/3 ∨ t = -4) :=
by sorry

end NUMINAMATH_CALUDE_factor_condition_l2317_231765


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l2317_231798

theorem unique_solution_cubic_system :
  ∃! (x y z : ℝ), x^3 = 2*y - 1 ∧ y^3 = 2*z - 1 ∧ z^3 = 2*x - 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l2317_231798


namespace NUMINAMATH_CALUDE_nancy_pears_l2317_231782

/-- Given that Alyssa picked 42 pears and the total number of pears picked was 59,
    prove that Nancy picked 17 pears. -/
theorem nancy_pears (alyssa_pears total_pears : ℕ) 
  (h1 : alyssa_pears = 42)
  (h2 : total_pears = 59) :
  total_pears - alyssa_pears = 17 := by
  sorry

end NUMINAMATH_CALUDE_nancy_pears_l2317_231782


namespace NUMINAMATH_CALUDE_joshua_friends_count_l2317_231749

def total_skittles : ℕ := 40
def skittles_per_friend : ℕ := 8

theorem joshua_friends_count : 
  total_skittles / skittles_per_friend = 5 := by sorry

end NUMINAMATH_CALUDE_joshua_friends_count_l2317_231749


namespace NUMINAMATH_CALUDE_estimate_pi_l2317_231784

theorem estimate_pi (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let estimate := 4 * (m / n + 1 / 2)
  estimate = 78 / 25 := by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_l2317_231784


namespace NUMINAMATH_CALUDE_henry_skittles_l2317_231702

/-- The number of Skittles Bridget has initially -/
def bridget_initial : ℕ := 4

/-- The total number of Skittles Bridget has after receiving Henry's Skittles -/
def bridget_final : ℕ := 8

/-- The number of Skittles Henry has -/
def henry : ℕ := bridget_final - bridget_initial

theorem henry_skittles : henry = 4 := by
  sorry

end NUMINAMATH_CALUDE_henry_skittles_l2317_231702


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2317_231790

/-- Given a line segment from (-3, 9) to (4, 10) parameterized by x = at + b and y = ct + d,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (-3, 9), prove that a^2 + b^2 + c^2 + d^2 = 140 -/
theorem line_segment_param_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  b = -3 →
  d = 9 →
  a + b = 4 →
  c + d = 10 →
  a^2 + b^2 + c^2 + d^2 = 140 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2317_231790


namespace NUMINAMATH_CALUDE_inequality_solution_l2317_231719

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - a*x - 6*a^2 > 0

-- Define the solution set
def solution_set (x₁ x₂ : ℝ) : Set ℝ := {x | x < x₁ ∨ x > x₂}

theorem inequality_solution (a : ℝ) (x₁ x₂ : ℝ) :
  a < 0 →
  (∀ x, inequality x a ↔ x ∈ solution_set x₁ x₂) →
  x₂ - x₁ = 5 * Real.sqrt 2 →
  a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2317_231719


namespace NUMINAMATH_CALUDE_equilateral_triangle_tiling_l2317_231739

theorem equilateral_triangle_tiling (m : ℕ) : 
  (∃ (t₁ t₂ : ℕ), 
    m = t₁ + t₂ ∧ 
    t₁ - t₂ = 5 ∧ 
    t₁ ≥ 5 ∧ 
    3 * t₁ + t₂ + 2 * (25 - t₁ - t₂) = 55) ↔ 
  (m % 2 = 1 ∧ m ≥ 5 ∧ m ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_tiling_l2317_231739


namespace NUMINAMATH_CALUDE_game_result_l2317_231792

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 2 = 0 then 2
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2]
def betty_rolls : List ℕ := [6, 3, 3, 2]

def total_points (rolls : List ℕ) : ℕ :=
  (rolls.map f).sum

theorem game_result : 
  (total_points allie_rolls) * (total_points betty_rolls) = 32 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2317_231792


namespace NUMINAMATH_CALUDE_abs_neg_2023_l2317_231722

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l2317_231722


namespace NUMINAMATH_CALUDE_integral_one_plus_cos_over_pi_half_interval_l2317_231707

theorem integral_one_plus_cos_over_pi_half_interval :
  ∫ x in (-π/2)..(π/2), (1 + Real.cos x) = π + 2 := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_cos_over_pi_half_interval_l2317_231707


namespace NUMINAMATH_CALUDE_a_4_equals_8_l2317_231733

def a (n : ℕ) : ℤ := (-1)^n * (2 * n)

theorem a_4_equals_8 : a 4 = 8 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l2317_231733


namespace NUMINAMATH_CALUDE_min_sales_to_break_even_l2317_231793

-- Define the given conditions
def current_salary : ℕ := 90000
def new_base_salary : ℕ := 45000
def commission_rate : ℚ := 15 / 100
def sale_value : ℕ := 1500

-- Define the function to calculate the total earnings in the new job
def new_job_earnings (num_sales : ℕ) : ℚ :=
  new_base_salary + (num_sales * sale_value * commission_rate)

-- Theorem statement
theorem min_sales_to_break_even :
  ∃ n : ℕ, (∀ m : ℕ, m < n → new_job_earnings m < current_salary) ∧
           new_job_earnings n ≥ current_salary ∧
           n = 200 := by
  sorry


end NUMINAMATH_CALUDE_min_sales_to_break_even_l2317_231793


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l2317_231786

theorem rational_inequality_solution (x : ℝ) :
  (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6)) > 0 ↔ 
  x < 2 ∨ (4 < x ∧ x < 5) ∨ 6 < x :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l2317_231786


namespace NUMINAMATH_CALUDE_power_of_power_l2317_231767

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2317_231767


namespace NUMINAMATH_CALUDE_popsicle_stick_cost_l2317_231797

/-- Represents the cost of popsicle supplies in dollars -/
structure PopsicleSupplies where
  total_budget : ℚ
  mold_cost : ℚ
  juice_cost_per_bottle : ℚ
  popsicles_per_bottle : ℕ
  total_sticks : ℕ
  remaining_sticks : ℕ

/-- Calculates the cost of the pack of popsicle sticks -/
def stick_pack_cost (supplies : PopsicleSupplies) : ℚ :=
  supplies.total_budget - supplies.mold_cost - 
  (supplies.juice_cost_per_bottle * ((supplies.total_sticks - supplies.remaining_sticks) / supplies.popsicles_per_bottle))

/-- Theorem stating that the cost of the pack of popsicle sticks is $1 -/
theorem popsicle_stick_cost (supplies : PopsicleSupplies) 
  (h1 : supplies.total_budget = 10)
  (h2 : supplies.mold_cost = 3)
  (h3 : supplies.juice_cost_per_bottle = 2)
  (h4 : supplies.popsicles_per_bottle = 20)
  (h5 : supplies.total_sticks = 100)
  (h6 : supplies.remaining_sticks = 40) :
  stick_pack_cost supplies = 1 := by
  sorry

#eval stick_pack_cost { 
  total_budget := 10, 
  mold_cost := 3, 
  juice_cost_per_bottle := 2, 
  popsicles_per_bottle := 20, 
  total_sticks := 100, 
  remaining_sticks := 40 
}

end NUMINAMATH_CALUDE_popsicle_stick_cost_l2317_231797


namespace NUMINAMATH_CALUDE_pure_imaginary_m_value_l2317_231745

/-- A complex number z is defined as z = (m^2+m-2) + (m^2+4m-5)i -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)

/-- A complex number is pure imaginary if its real part is zero and imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_m_value :
  ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_value_l2317_231745


namespace NUMINAMATH_CALUDE_jim_grove_other_row_l2317_231755

/-- The number of lemons produced by a normal lemon tree per year -/
def normal_lemon_production : ℕ := 60

/-- The percentage increase in lemon production for Jim's engineered trees -/
def engineered_production_increase : ℚ := 50 / 100

/-- The number of trees in one row of Jim's grove -/
def trees_in_one_row : ℕ := 50

/-- The total number of lemons produced by Jim's grove in 5 years -/
def total_lemons_produced : ℕ := 675000

/-- The number of years of lemon production -/
def years_of_production : ℕ := 5

/-- The number of trees in the other row of Jim's grove -/
def trees_in_other_row : ℕ := 1450

theorem jim_grove_other_row :
  trees_in_other_row = 
    (total_lemons_produced / (normal_lemon_production * (1 + engineered_production_increase) * years_of_production)).floor - trees_in_one_row :=
by sorry

end NUMINAMATH_CALUDE_jim_grove_other_row_l2317_231755


namespace NUMINAMATH_CALUDE_ace_spade_probability_l2317_231760

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size_eq : cards.card = 52)

/-- Represents the event of drawing an Ace first and a spade second -/
def ace_spade_event (deck : Deck) : Finset (Nat × Nat × Nat × Nat) :=
  sorry

/-- The probability of the ace_spade_event -/
def ace_spade_prob (deck : Deck) : ℚ :=
  (ace_spade_event deck).card / deck.cards.card / (deck.cards.card - 1)

theorem ace_spade_probability (deck : Deck) : 
  ace_spade_prob deck = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_ace_spade_probability_l2317_231760


namespace NUMINAMATH_CALUDE_geese_survival_l2317_231756

/-- Given the following conditions:
  1. 500 goose eggs were laid
  2. 2/3 of eggs hatched
  3. 3/4 of hatched geese survived the first month
  4. 2/5 of geese that survived the first month survived the first year
Prove that 100 geese survived the first year -/
theorem geese_survival (total_eggs : ℕ) (hatch_rate first_month_rate first_year_rate : ℚ) :
  total_eggs = 500 →
  hatch_rate = 2/3 →
  first_month_rate = 3/4 →
  first_year_rate = 2/5 →
  (total_eggs : ℚ) * hatch_rate * first_month_rate * first_year_rate = 100 := by
  sorry

#eval (500 : ℚ) * (2/3) * (3/4) * (2/5)

end NUMINAMATH_CALUDE_geese_survival_l2317_231756


namespace NUMINAMATH_CALUDE_line_slope_product_l2317_231777

/-- Given two lines L₁ and L₂ with equations y = mx and y = nx respectively,
    where L₁ makes twice as large of an angle with the horizontal as L₂,
    L₁ has 3 times the slope of L₂, and L₁ is not horizontal,
    then mn = 1. -/
theorem line_slope_product (m n : ℝ) (hm : m ≠ 0) :
  (∃ θ : ℝ, m = Real.tan (2 * θ) ∧ n = Real.tan θ) →
  m = 3 * n →
  m * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_product_l2317_231777


namespace NUMINAMATH_CALUDE_three_times_x_not_much_different_from_two_l2317_231721

theorem three_times_x_not_much_different_from_two :
  ∃ (x : ℝ), 3 * x - 2 ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_three_times_x_not_much_different_from_two_l2317_231721


namespace NUMINAMATH_CALUDE_tower_of_hanoi_correct_l2317_231730

/-- Minimum number of moves required to solve the Tower of Hanoi problem with n discs -/
def tower_of_hanoi (n : ℕ) : ℕ :=
  2^n - 1

/-- Theorem: The minimum number of moves for the Tower of Hanoi problem with n discs is 2^n - 1 -/
theorem tower_of_hanoi_correct (n : ℕ) : tower_of_hanoi n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_tower_of_hanoi_correct_l2317_231730


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2317_231746

/-- Represents the production plan and profit calculation for a company producing two types of crafts. -/
structure ProductionPlan where
  /-- Cost of material A in yuan -/
  cost_A : ℕ
  /-- Cost of material B in yuan -/
  cost_B : ℕ
  /-- Number of units of craft X produced -/
  units_X : ℕ
  /-- Number of units of craft Y produced -/
  units_Y : ℕ
  /-- Condition: Cost of B is 40 yuan more than A -/
  cost_diff : cost_B = cost_A + 40
  /-- Condition: 2 units of A and 3 units of B cost 420 yuan -/
  total_cost : 2 * cost_A + 3 * cost_B = 420
  /-- Condition: Total number of crafts is 560 -/
  total_units : units_X + units_Y = 560
  /-- Condition: X should not exceed 180 units -/
  max_X : units_X ≤ 180

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  360 * plan.units_X + 450 * plan.units_Y -
  (plan.cost_A * (plan.units_X + 3 * plan.units_Y) +
   plan.cost_B * (2 * plan.units_X + 2 * plan.units_Y))

/-- Theorem stating the maximum profit and optimal production plan -/
theorem max_profit_theorem (plan : ProductionPlan) : 
  plan.cost_A = 60 ∧ plan.cost_B = 100 ∧ plan.units_X = 180 ∧ plan.units_Y = 380 →
  profit plan = 44600 ∧ ∀ other_plan : ProductionPlan, profit other_plan ≤ profit plan := by
  sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l2317_231746


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2317_231799

/-- Given a parabola y = 2x^2, the distance from its focus to its directrix is 1/2 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y = 2 * x^2 →
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ focus_y = 1/4) ∧
    directrix_y = -1/4 ∧
    focus_y - directrix_y = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2317_231799


namespace NUMINAMATH_CALUDE_tangent_line_correct_l2317_231770

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The point through which the tangent line passes -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the proposed tangent line -/
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_correct :
  (f point.1 = point.2) ∧ 
  (∀ x y : ℝ, tangent_line x y → y - point.2 = f' point.1 * (x - point.1)) ∧
  (∀ x : ℝ, x ≠ point.1 → f x ≠ (f' point.1) * (x - point.1) + point.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_correct_l2317_231770


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2317_231737

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 1 = 0 ∧ m^2 * x - 2 * y - 1 = 0 → 
    (1 : ℝ) * (-m^2 : ℝ) = -1) → 
  m = 0 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2317_231737


namespace NUMINAMATH_CALUDE_limit_exists_and_equals_20_21_l2317_231795

/-- Sum of exponents of 71 and 97 in the prime factorization of n -/
def s (n : ℕ+) : ℕ :=
  sorry

/-- Function f(n) = (-1)^(s(n)) -/
def f (n : ℕ+) : ℤ :=
  (-1) ^ (s n)

/-- Sum of f(x) from x = 1 to n -/
def S (n : ℕ+) : ℤ :=
  (Finset.range n).sum (fun x => f ⟨x + 1, Nat.succ_pos x⟩)

/-- The main theorem: limit of S(n)/n exists and equals 20/21 -/
theorem limit_exists_and_equals_20_21 :
    ∃ (L : ℚ), L = 20 / 21 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N,
      |((S n : ℚ) / n) - L| < ε :=
  sorry

end NUMINAMATH_CALUDE_limit_exists_and_equals_20_21_l2317_231795


namespace NUMINAMATH_CALUDE_points_cover_rectangles_l2317_231741

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  left : ℝ
  bottom : ℝ
  width : ℝ
  height : ℝ

/-- The unit square -/
def unitSquare : Rectangle := { left := 0, bottom := 0, width := 1, height := 1 }

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  r.left ≤ p.x ∧ p.x ≤ r.left + r.width ∧
  r.bottom ≤ p.y ∧ p.y ≤ r.bottom + r.height

/-- Check if a rectangle is inside another rectangle -/
def isContained (inner outer : Rectangle) : Prop :=
  outer.left ≤ inner.left ∧ inner.left + inner.width ≤ outer.left + outer.width ∧
  outer.bottom ≤ inner.bottom ∧ inner.bottom + inner.height ≤ outer.bottom + outer.height

/-- The main theorem -/
theorem points_cover_rectangles : ∃ (points : Finset Point),
  points.card ≤ 1600 ∧
  ∀ (r : Rectangle),
    isContained r unitSquare →
    r.width * r.height = 0.005 →
    ∃ (p : Point), p ∈ points ∧ isInside p r := by
  sorry

end NUMINAMATH_CALUDE_points_cover_rectangles_l2317_231741


namespace NUMINAMATH_CALUDE_graduating_class_size_l2317_231764

theorem graduating_class_size 
  (geometry : ℕ) 
  (biology : ℕ) 
  (overlap_diff : ℕ) 
  (h1 : geometry = 144) 
  (h2 : biology = 119) 
  (h3 : overlap_diff = 88) :
  geometry + biology - min geometry biology = 263 :=
by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l2317_231764


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_8_l2317_231703

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * a 1 + (n * (n - 1) : ℤ) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_sum_8 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 1 = -40 →
  a 6 + a 10 = -10 →
  sum_of_arithmetic_sequence a 8 = -180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_8_l2317_231703


namespace NUMINAMATH_CALUDE_relationship_abc_l2317_231725

theorem relationship_abc :
  let a : ℝ := (0.9 : ℝ) ^ (0.3 : ℝ)
  let b : ℝ := (1.2 : ℝ) ^ (0.3 : ℝ)
  let c : ℝ := (0.5 : ℝ) ^ (-0.3 : ℝ)
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2317_231725


namespace NUMINAMATH_CALUDE_f_derivative_l2317_231740

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - (x + 1)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  (deriv f) x = 4 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l2317_231740


namespace NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l2317_231717

/-- Given a plane P (z = 0), points A on P and O not on P, prove that the locus of points H,
    where H is the foot of the perpendicular from O to any line in P through A,
    forms a circle with the given equation. -/
theorem locus_of_perpendicular_foot (a b d e f : ℝ) :
  let P : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  let A : ℝ × ℝ × ℝ := (a, b, 0)
  let O : ℝ × ℝ × ℝ := (d, e, f)
  let H := {h : ℝ × ℝ × ℝ | ∃ (u v : ℝ),
    h = ((a * (u^2 + v^2) + (d*u + e*v - a*u - b*v)*u) / (u^2 + v^2),
         (b * (u^2 + v^2) + (d*u + e*v - a*u - b*v)*v) / (u^2 + v^2),
         0)}
  ∀ (x y : ℝ), (x, y, 0) ∈ H ↔ x^2 + y^2 - (a+d)*x - (b+e)*y + a*d + b*e = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l2317_231717


namespace NUMINAMATH_CALUDE_sum_equation_l2317_231772

theorem sum_equation : 27474 + 3699 + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_l2317_231772


namespace NUMINAMATH_CALUDE_middle_term_coefficient_l2317_231705

/-- Given a binomial expansion (x^2 - 2/x)^n where the 5th term is constant,
    prove that the coefficient of the middle term is -160 -/
theorem middle_term_coefficient
  (x : ℝ) (n : ℕ)
  (h_constant : ∃ k : ℝ, (n.choose 4) * (x^2)^(n-4) * (-2/x)^4 = k) :
  ∃ m : ℕ, m = (n+1)/2 ∧ (n.choose (m-1)) * (x^2)^(m-1) * (-2/x)^(n-m+1) = -160 * x^(2*m-n-1) :=
sorry

end NUMINAMATH_CALUDE_middle_term_coefficient_l2317_231705


namespace NUMINAMATH_CALUDE_towel_bleaching_l2317_231781

theorem towel_bleaching (original_length original_breadth : ℝ) 
  (h_positive : original_length > 0 ∧ original_breadth > 0) :
  let new_length := 0.7 * original_length
  let new_area := 0.42 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.6 * original_breadth ∧
    new_length * new_breadth = new_area :=
by sorry

end NUMINAMATH_CALUDE_towel_bleaching_l2317_231781


namespace NUMINAMATH_CALUDE_courtyard_length_l2317_231754

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) : 
  width = 18 → 
  brick_length = 0.2 → 
  brick_width = 0.1 → 
  num_bricks = 22500 → 
  (width * (num_bricks * brick_length * brick_width / width)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l2317_231754


namespace NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l2317_231769

/-- Given four consecutive digits p, q, r, s, the sum of pqrs and srqp is divisible by 1111 -/
theorem consecutive_digits_sum_divisibility (p : ℕ) (h1 : p < 7) :
  ∃ (k : ℕ), 1000 * p + 100 * (p + 1) + 10 * (p + 2) + (p + 3) +
             1000 * (p + 3) + 100 * (p + 2) + 10 * (p + 1) + p = 1111 * k := by
  sorry

#check consecutive_digits_sum_divisibility

end NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l2317_231769


namespace NUMINAMATH_CALUDE_husk_consumption_rate_l2317_231747

/-- Given that 20 cows eat 20 bags of husk in 20 days, prove that 1 cow will eat 1 bag of husk in 20 days -/
theorem husk_consumption_rate (cows bags days : ℕ) (h1 : cows = 20) (h2 : bags = 20) (h3 : days = 20) :
  (1 : ℚ) / cows * bags * days = 20 := by
  sorry

end NUMINAMATH_CALUDE_husk_consumption_rate_l2317_231747


namespace NUMINAMATH_CALUDE_bells_toll_together_l2317_231748

theorem bells_toll_together (bell_intervals : List ℕ := [13, 17, 21, 26, 34, 39]) : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 17) 21) 26) 34) 39 = 9272 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l2317_231748


namespace NUMINAMATH_CALUDE_race_speed_calculation_l2317_231757

/-- Given two teams racing on a 300-mile course, where one team finishes 3 hours earlier
    and has an average speed 5 mph greater than the other, prove that the slower team's
    average speed is 20 mph. -/
theorem race_speed_calculation (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 300 ∧ time_diff = 3 ∧ speed_diff = 5 →
  ∃ (speed_e : ℝ) (time_e : ℝ),
    speed_e > 0 ∧
    time_e > 0 ∧
    distance = speed_e * time_e ∧
    distance = (speed_e + speed_diff) * (time_e - time_diff) ∧
    speed_e = 20 :=
by sorry

end NUMINAMATH_CALUDE_race_speed_calculation_l2317_231757


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2317_231738

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2317_231738


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2317_231768

theorem quadratic_factorization (a b : ℤ) : 
  (∀ x : ℝ, 12 * x^2 - 38 * x - 40 = (4 * x + a) * (3 * x + b)) → 
  a - 3 * b = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2317_231768


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l2317_231794

theorem floor_sqrt_eight_count :
  (Finset.filter (fun x : ℕ => ⌊Real.sqrt x⌋ = 8) (Finset.range 81)).card = 17 :=
sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l2317_231794


namespace NUMINAMATH_CALUDE_triangle_inequality_l2317_231787

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2317_231787


namespace NUMINAMATH_CALUDE_equation_solution_l2317_231723

theorem equation_solution (m : ℕ+) : 
  (∃ x : ℕ+, x ≠ 8 ∧ (m * x : ℚ) / (x - 8 : ℚ) = ((4 * m + x) : ℚ) / (x - 8 : ℚ)) ↔ 
  m = 3 ∨ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2317_231723


namespace NUMINAMATH_CALUDE_average_of_multiples_10_to_100_l2317_231743

def multiples_of_10 : List ℕ := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem average_of_multiples_10_to_100 : 
  average multiples_of_10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_10_to_100_l2317_231743


namespace NUMINAMATH_CALUDE_certain_number_calculation_l2317_231731

theorem certain_number_calculation (x y : ℝ) :
  x = 77.7 ∧ x = y + 0.11 * y → y = 77.7 / 1.11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l2317_231731


namespace NUMINAMATH_CALUDE_factor_polynomial_l2317_231718

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 50 * x^10 = 25 * x^7 * (3 - 2 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2317_231718


namespace NUMINAMATH_CALUDE_gcf_360_150_l2317_231713

theorem gcf_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_360_150_l2317_231713


namespace NUMINAMATH_CALUDE_eggs_per_group_l2317_231783

theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) : 
  total_eggs = 8 → num_groups = 4 → eggs_per_group = total_eggs / num_groups → eggs_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l2317_231783


namespace NUMINAMATH_CALUDE_probability_sum_25_is_7_200_l2317_231711

-- Define the structure of a die
structure Die :=
  (faces : Finset ℕ)
  (blank_face : Bool)
  (fair : Bool)

-- Define the two dice
def die1 : Die :=
  { faces := Finset.range 20 \ {20},
    blank_face := true,
    fair := true }

def die2 : Die :=
  { faces := (Finset.range 21 \ {0, 8}),
    blank_face := true,
    fair := true }

-- Define the function to calculate the probability
def probability_sum_25 (d1 d2 : Die) : ℚ :=
  let total_outcomes := 20 * 20
  let valid_combinations := 14
  valid_combinations / total_outcomes

-- State the theorem
theorem probability_sum_25_is_7_200 :
  probability_sum_25 die1 die2 = 7 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_25_is_7_200_l2317_231711


namespace NUMINAMATH_CALUDE_prime_q_value_l2317_231709

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_q_value (p : ℕ) (hp : is_prime p) :
  let q := 13 * p + 2
  is_prime q ∧ (q - 1) % 3 = 0 → q = 67 := by
  sorry

end NUMINAMATH_CALUDE_prime_q_value_l2317_231709


namespace NUMINAMATH_CALUDE_mac_running_rate_l2317_231728

/-- The running rate of Apple in miles per hour -/
def apple_rate : ℝ := 3

/-- The race distance in miles -/
def race_distance : ℝ := 24

/-- The time difference between Apple and Mac in minutes -/
def time_difference : ℝ := 120

/-- Mac's running rate in miles per hour -/
def mac_rate : ℝ := 4

/-- Theorem stating that given the conditions, Mac's running rate is 4 miles per hour -/
theorem mac_running_rate : 
  let apple_time := race_distance / apple_rate * 60  -- Apple's time in minutes
  let mac_time := apple_time - time_difference       -- Mac's time in minutes
  mac_rate = race_distance / (mac_time / 60) := by
sorry

end NUMINAMATH_CALUDE_mac_running_rate_l2317_231728


namespace NUMINAMATH_CALUDE_min_pool_cost_l2317_231775

/-- Represents the cost of constructing a rectangular pool -/
def pool_cost (l w h : ℝ) : ℝ :=
  120 * l * w + 80 * 2 * h * (l + w)

/-- The minimum cost of constructing a pool with given specifications -/
theorem min_pool_cost :
  ∀ l w : ℝ,
  l > 0 ∧ w > 0 →
  l * w * 2 = 8 →
  pool_cost l w 2 ≥ 1760 :=
sorry

end NUMINAMATH_CALUDE_min_pool_cost_l2317_231775


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2317_231778

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) -- Sequence of integers indexed by natural numbers
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_a1 : a 1 = -1) -- First term condition
  (h_a4 : a 4 = 8) -- Fourth term condition
  : ∃ d : ℤ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2317_231778


namespace NUMINAMATH_CALUDE_a_2016_value_l2317_231785

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem a_2016_value : a 2016 = 3024 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_a_2016_value_l2317_231785


namespace NUMINAMATH_CALUDE_car_departure_time_l2317_231744

/-- 
Given two cars A and B that start simultaneously from locations A and B respectively:
- They will meet at some point
- If Car A departs earlier, they will meet 30 minutes earlier
- Car A travels at 60 kilometers per hour
- Car B travels at 40 kilometers per hour

Prove that Car A needs to depart 50 minutes earlier for them to meet 30 minutes earlier.
-/
theorem car_departure_time (speed_A speed_B : ℝ) (meeting_time_diff : ℝ) :
  speed_A = 60 →
  speed_B = 40 →
  meeting_time_diff = 30 →
  ∃ (departure_time : ℝ), 
    departure_time = 50 ∧
    speed_A * (departure_time / 60) = speed_A * (meeting_time_diff / 60) + speed_B * (meeting_time_diff / 60) :=
by sorry

end NUMINAMATH_CALUDE_car_departure_time_l2317_231744


namespace NUMINAMATH_CALUDE_triangle_symmetric_negative_three_four_l2317_231706

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a * b - a - b + 1

-- Theorem statement
theorem triangle_symmetric_negative_three_four : triangle (-3) 4 = triangle 4 (-3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_symmetric_negative_three_four_l2317_231706


namespace NUMINAMATH_CALUDE_cyclic_difference_sum_lower_bound_l2317_231704

theorem cyclic_difference_sum_lower_bound 
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
                a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
                a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
                a₆ ≠ a₇) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + 
  (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 := by
  sorry


end NUMINAMATH_CALUDE_cyclic_difference_sum_lower_bound_l2317_231704


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l2317_231796

theorem twentieth_fisherman_catch (total_fishermen : Nat) (total_fish : Nat) 
  (each_fish : Nat) (n : Nat) (h1 : total_fishermen = 20) 
  (h2 : total_fish = 10000) (h3 : each_fish = 400) (h4 : n = 19) : 
  total_fish - n * each_fish = 2400 := by
  sorry

#check twentieth_fisherman_catch

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l2317_231796


namespace NUMINAMATH_CALUDE_f_minus_one_equals_eight_l2317_231774

def f (x : ℝ) (c : ℝ) := x^2 + c

theorem f_minus_one_equals_eight (c : ℝ) (h : f 1 c = 8) : f (-1) c = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_one_equals_eight_l2317_231774


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l2317_231771

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l2317_231771


namespace NUMINAMATH_CALUDE_tuesday_income_l2317_231779

/-- Calculates Lauren's income from her social media channel --/
def laurens_income (commercial_rate : ℚ) (subscription_rate : ℚ) (commercials_viewed : ℕ) (new_subscribers : ℕ) : ℚ :=
  commercial_rate * commercials_viewed + subscription_rate * new_subscribers

/-- Proves that Lauren's income on Tuesday is $77.00 --/
theorem tuesday_income : 
  laurens_income (1/2) 1 100 27 = 77 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_income_l2317_231779


namespace NUMINAMATH_CALUDE_limit_example_l2317_231752

/-- The limit of (2x^2 - 5x + 2) / (x - 1/2) as x approaches 1/2 is -3 -/
theorem limit_example (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, x ≠ 1/2 → |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_example_l2317_231752


namespace NUMINAMATH_CALUDE_total_consumption_30_days_l2317_231734

/-- Represents the daily food consumption for each dog -/
structure DogConsumption where
  a : Float
  b : Float
  c : Float
  d : Float
  e : Float

/-- Calculates the total daily consumption for all dogs -/
def totalDailyConsumption (dc : DogConsumption) : Float :=
  dc.a + dc.b + dc.c + dc.d + dc.e

/-- Represents the food consumption for each dog on Sundays -/
structure SundayConsumption where
  a : Float
  b : Float
  c : Float
  d : Float
  e : Float

/-- Calculates the total consumption for all dogs on a Sunday -/
def totalSundayConsumption (sc : SundayConsumption) : Float :=
  sc.a + sc.b + sc.c + sc.d + sc.e

/-- Theorem: Total dog food consumption over 30 days is 60 scoops -/
theorem total_consumption_30_days 
  (dc : DogConsumption)
  (sc : SundayConsumption)
  (h1 : dc.a = 0.125)
  (h2 : dc.b = 0.25)
  (h3 : dc.c = 0.375)
  (h4 : dc.d = 0.5)
  (h5 : dc.e = 0.75)
  (h6 : sc.a = dc.a)
  (h7 : sc.b = dc.b)
  (h8 : sc.c = dc.c + 0.1)
  (h9 : sc.d = dc.d)
  (h10 : sc.e = dc.e - 0.1)
  (h11 : totalDailyConsumption dc = totalSundayConsumption sc) :
  30 * totalDailyConsumption dc = 60 := by
  sorry


end NUMINAMATH_CALUDE_total_consumption_30_days_l2317_231734


namespace NUMINAMATH_CALUDE_area_ratio_S₂_to_S₁_l2317_231726

-- Define the sets S₁ and S₂
def S₁ : Set (ℝ × ℝ) := {p | Real.log (1 + p.1^2 + p.2^2) ≤ 1 + Real.log (p.1 + p.2)}
def S₂ : Set (ℝ × ℝ) := {p | Real.log (2 + p.1^2 + p.2^2) ≤ 2 + Real.log (p.1 + p.2)}

-- Define the areas of S₁ and S₂
noncomputable def area_S₁ : ℝ := Real.pi * 49
noncomputable def area_S₂ : ℝ := Real.pi * 4998

-- Theorem statement
theorem area_ratio_S₂_to_S₁ : area_S₂ / area_S₁ = 102 := by sorry

end NUMINAMATH_CALUDE_area_ratio_S₂_to_S₁_l2317_231726


namespace NUMINAMATH_CALUDE_square_roots_problem_l2317_231791

theorem square_roots_problem (x : ℝ) (h : x > x - 6) :
  (x ^ 2 = (x - 6) ^ 2) → x ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2317_231791


namespace NUMINAMATH_CALUDE_smallest_integer_for_inequality_l2317_231766

theorem smallest_integer_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y z : ℝ, (a*x^2 + b*y^2 + c*z^2)^2 ≤ 3*(a^2*x^4 + b^2*y^4 + c^2*z^4)) ∧
  (∀ n : ℕ, n < 3 → ∃ x y z : ℝ, (a*x^2 + b*y^2 + c*z^2)^2 > n*(a^2*x^4 + b^2*y^4 + c^2*z^4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_for_inequality_l2317_231766


namespace NUMINAMATH_CALUDE_log_xyz_equals_one_l2317_231736

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : log (x^2 * y^2 * z) = 2) 
  (h2 : log (x * y * z^3) = 2) : 
  log (x * y * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_xyz_equals_one_l2317_231736


namespace NUMINAMATH_CALUDE_rook_placement_count_l2317_231789

def chessboard_size : ℕ := 8
def num_rooks : ℕ := 6

theorem rook_placement_count :
  (Nat.choose chessboard_size num_rooks) * (Nat.factorial chessboard_size / Nat.factorial (chessboard_size - num_rooks)) = 564480 :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_count_l2317_231789


namespace NUMINAMATH_CALUDE_second_brand_growth_rate_l2317_231761

/-- Proves that the growth rate of the second brand of computers is approximately 0.7 million households per year -/
theorem second_brand_growth_rate (initial_first : ℝ) (growth_first : ℝ) (initial_second : ℝ) (time_to_equal : ℝ)
  (h1 : initial_first = 4.9)
  (h2 : growth_first = 0.275)
  (h3 : initial_second = 2.5)
  (h4 : time_to_equal = 5.647)
  (h5 : initial_first + growth_first * time_to_equal = initial_second + growth_second * time_to_equal) :
  ∃ growth_second : ℝ, abs (growth_second - 0.7) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_second_brand_growth_rate_l2317_231761


namespace NUMINAMATH_CALUDE_strongest_correlation_l2317_231773

-- Define the type for a pair of observations
structure Observation where
  n : ℕ
  r : ℝ

-- Define the four given observations
def obs1 : Observation := ⟨10, 0.9533⟩
def obs2 : Observation := ⟨15, 0.3012⟩
def obs3 : Observation := ⟨17, 0.9991⟩
def obs4 : Observation := ⟨3, 0.9950⟩

-- Define a function to check if an observation indicates strong linear correlation
def isStrongCorrelation (obs : Observation) : Prop :=
  abs obs.r > 0.95

-- Theorem stating that obs1 and obs3 have the strongest linear correlation
theorem strongest_correlation :
  isStrongCorrelation obs1 ∧ isStrongCorrelation obs3 ∧
  ¬isStrongCorrelation obs2 ∧ ¬isStrongCorrelation obs4 :=
sorry

end NUMINAMATH_CALUDE_strongest_correlation_l2317_231773


namespace NUMINAMATH_CALUDE_sum_of_arguments_fifth_roots_l2317_231701

/-- The sum of arguments of the fifth roots of 81(1+i) is 765 degrees -/
theorem sum_of_arguments_fifth_roots (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : z₁^5 = 81 * (1 + Complex.I))
  (h₂ : z₂^5 = 81 * (1 + Complex.I))
  (h₃ : z₃^5 = 81 * (1 + Complex.I))
  (h₄ : z₄^5 = 81 * (1 + Complex.I))
  (h₅ : z₅^5 = 81 * (1 + Complex.I))
  (hr₁ : Complex.abs z₁ > 0)
  (hr₂ : Complex.abs z₂ > 0)
  (hr₃ : Complex.abs z₃ > 0)
  (hr₄ : Complex.abs z₄ > 0)
  (hr₅ : Complex.abs z₅ > 0)
  (hθ₁ : 0 ≤ Complex.arg z₁ ∧ Complex.arg z₁ < 2 * Real.pi)
  (hθ₂ : 0 ≤ Complex.arg z₂ ∧ Complex.arg z₂ < 2 * Real.pi)
  (hθ₃ : 0 ≤ Complex.arg z₃ ∧ Complex.arg z₃ < 2 * Real.pi)
  (hθ₄ : 0 ≤ Complex.arg z₄ ∧ Complex.arg z₄ < 2 * Real.pi)
  (hθ₅ : 0 ≤ Complex.arg z₅ ∧ Complex.arg z₅ < 2 * Real.pi) :
  Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + Complex.arg z₄ + Complex.arg z₅ = 
    (765 * Real.pi) / 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arguments_fifth_roots_l2317_231701


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l2317_231724

/-- A function f(x) = ax² + x + 1 is monotonically increasing in the interval [-2, +∞) if and only if 0 ≤ a ≤ 1/4 -/
theorem monotonic_quadratic_function (a : ℝ) :
  (∀ x : ℝ, x ≥ -2 → Monotone (fun x => a * x^2 + x + 1)) ↔ 0 ≤ a ∧ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l2317_231724


namespace NUMINAMATH_CALUDE_find_number_l2317_231780

theorem find_number (N : ℚ) : (4 / 5 * N) + 18 = N / (4 / 5) → N = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2317_231780


namespace NUMINAMATH_CALUDE_exists_circle_with_n_lattice_points_l2317_231716

/-- A point with integer coordinates in the plane -/
def LatticePoint := ℤ × ℤ

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of lattice points on the circumference of a circle -/
def latticePointsOnCircle (c : Circle) : ℕ :=
  sorry

/-- For every natural number n, there exists a circle with exactly n lattice points on its circumference -/
theorem exists_circle_with_n_lattice_points (n : ℕ) :
  ∃ c : Circle, latticePointsOnCircle c = n :=
sorry

end NUMINAMATH_CALUDE_exists_circle_with_n_lattice_points_l2317_231716


namespace NUMINAMATH_CALUDE_unit_digit_of_product_l2317_231708

def numbers : List Nat := [6245, 7083, 9137, 4631, 5278, 3974]

theorem unit_digit_of_product (nums : List Nat := numbers) :
  (nums.foldl (· * ·) 1) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_l2317_231708
