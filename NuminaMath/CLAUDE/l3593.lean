import Mathlib

namespace NUMINAMATH_CALUDE_sine_equation_equality_l3593_359331

theorem sine_equation_equality (α β γ τ : ℝ) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0) 
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) : 
  α = γ ∨ α = τ := by
  sorry

end NUMINAMATH_CALUDE_sine_equation_equality_l3593_359331


namespace NUMINAMATH_CALUDE_average_listening_time_approx_33_l3593_359300

/-- Represents the distribution of audience members and their listening times --/
structure AudienceDistribution where
  total_audience : ℕ
  talk_duration : ℕ
  full_listeners_percent : ℚ
  sleepers_percent : ℚ
  half_listeners_percent : ℚ
  quarter_listeners_percent : ℚ

/-- Calculates the average listening time for the audience --/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  let full_listeners := (dist.full_listeners_percent * dist.total_audience) * dist.talk_duration
  let sleepers := 0
  let half_listeners := (dist.half_listeners_percent * dist.total_audience) * (dist.talk_duration / 2)
  let quarter_listeners := (dist.quarter_listeners_percent * dist.total_audience) * (dist.talk_duration / 4)
  (full_listeners + sleepers + half_listeners + quarter_listeners) / dist.total_audience

/-- The given audience distribution --/
def lecture_distribution : AudienceDistribution :=
  { total_audience := 200
  , talk_duration := 90
  , full_listeners_percent := 15 / 100
  , sleepers_percent := 15 / 100
  , half_listeners_percent := (1 / 4) * (70 / 100)
  , quarter_listeners_percent := (3 / 4) * (70 / 100)
  }

/-- Theorem stating that the average listening time is approximately 33 minutes --/
theorem average_listening_time_approx_33 :
  ∃ ε > 0, |average_listening_time lecture_distribution - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_average_listening_time_approx_33_l3593_359300


namespace NUMINAMATH_CALUDE_tangent_line_p_values_l3593_359397

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8 = 0

/-- The equation of the tangent line -/
def tangent_line (p x : ℝ) : Prop := x = -p/2

/-- The line is tangent to the circle -/
def is_tangent (p : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ tangent_line p x

/-- Theorem: If the line x = -p/2 is tangent to the circle x^2 + y^2 + 6x + 8 = 0, then p = 4 or p = 8 -/
theorem tangent_line_p_values (p : ℝ) : is_tangent p → p = 4 ∨ p = 8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_p_values_l3593_359397


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3593_359349

open Real

theorem trig_equation_solution (x : ℝ) : 
  (sin (x + 15 * π / 180) + sin (x + 45 * π / 180) + sin (x + 75 * π / 180) = 
   sin (15 * π / 180) + sin (45 * π / 180) + sin (75 * π / 180)) ↔ 
  (∃ k : ℤ, x = k * 2 * π ∨ x = π / 2 + k * 2 * π) :=
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3593_359349


namespace NUMINAMATH_CALUDE_sum_of_ages_is_41_l3593_359361

/-- The sum of Henry and Jill's present ages -/
def sumOfAges (henryAge : ℕ) (jillAge : ℕ) : ℕ :=
  henryAge + jillAge

/-- Theorem stating that the sum of Henry and Jill's present ages is 41 -/
theorem sum_of_ages_is_41 (henryAge : ℕ) (jillAge : ℕ) 
  (h1 : henryAge = 25) 
  (h2 : jillAge = 16) : 
  sumOfAges henryAge jillAge = 41 := by
  sorry

#check sum_of_ages_is_41

end NUMINAMATH_CALUDE_sum_of_ages_is_41_l3593_359361


namespace NUMINAMATH_CALUDE_inequality_proof_l3593_359345

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 2 * c > a + b) : 
  c - Real.sqrt (c^2 - a*b) < a ∧ a < c + Real.sqrt (c^2 - a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3593_359345


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l3593_359341

def total_pay : ℝ := 570
def x_pay_ratio : ℝ := 1.2

theorem employee_pay_calculation (x y : ℝ) 
  (h1 : x + y = total_pay) 
  (h2 : x = x_pay_ratio * y) : 
  y = 259.09 := by sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l3593_359341


namespace NUMINAMATH_CALUDE_at_least_three_identical_digits_l3593_359308

theorem at_least_three_identical_digits (p n : ℕ) : 
  Prime p → p > 3 → (10^19 ≤ p^n) ∧ (p^n < 10^20) → 
  ∃ (d : ℕ) (i j k : ℕ), i < j ∧ j < k ∧ k < 20 ∧ 
  (p^n / 10^i) % 10 = d ∧ (p^n / 10^j) % 10 = d ∧ (p^n / 10^k) % 10 = d :=
by sorry

end NUMINAMATH_CALUDE_at_least_three_identical_digits_l3593_359308


namespace NUMINAMATH_CALUDE_value_of_T_l3593_359326

theorem value_of_T : ∃ T : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * T = (1/4 : ℝ) * (1/5 : ℝ) * 120 ∧ T = 108 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l3593_359326


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l3593_359391

/-- The cost of tickets for a group attending a circus -/
def ticket_cost (adult_price : ℚ) (child_price : ℚ) (num_adults : ℕ) (num_children : ℕ) : ℚ :=
  (adult_price * num_adults) + (child_price * num_children)

/-- Theorem: The total cost of tickets for 2 adults at $44.00 each and 5 children at $28.00 each is $228.00 -/
theorem circus_ticket_cost :
  ticket_cost 44 28 2 5 = 228 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l3593_359391


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3593_359317

theorem unique_solution_for_equation : 
  ∃! (p n : ℕ), 
    n > 0 ∧ 
    Nat.Prime p ∧ 
    17^n * 2^(n^2) - p = (2^(n^2 + 3) + 2^(n^2) - 1) * n^2 ∧ 
    p = 17 ∧ 
    n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3593_359317


namespace NUMINAMATH_CALUDE_paper_mill_inspection_theorem_l3593_359322

/-- Represents the number of paper mills -/
def num_mills : ℕ := 5

/-- Probability of passing initial inspection -/
def prob_pass_initial : ℚ := 1/2

/-- Probability of passing after rectification -/
def prob_pass_rectification : ℚ := 4/5

/-- Probability of exactly two mills needing rectification -/
def prob_two_rectified : ℚ := 5/16

/-- Probability of at least one mill being shut down -/
def prob_at_least_one_shutdown : ℚ := 1 - (9/10)^5

/-- Average number of mills needing rectification -/
def avg_mills_rectified : ℚ := 5/2

theorem paper_mill_inspection_theorem :
  (prob_two_rectified = Nat.choose num_mills 2 * (1 - prob_pass_initial)^2 * prob_pass_initial^3) ∧
  (prob_at_least_one_shutdown = 1 - (1 - (1 - prob_pass_initial) * (1 - prob_pass_rectification))^num_mills) ∧
  (avg_mills_rectified = num_mills * (1 - prob_pass_initial)) :=
by sorry

end NUMINAMATH_CALUDE_paper_mill_inspection_theorem_l3593_359322


namespace NUMINAMATH_CALUDE_kaylin_is_33_l3593_359369

def freyja_age : ℕ := 10

def eli_age (freyja_age : ℕ) : ℕ := freyja_age + 9

def sarah_age (eli_age : ℕ) : ℕ := 2 * eli_age

def kaylin_age (sarah_age : ℕ) : ℕ := sarah_age - 5

theorem kaylin_is_33 : 
  kaylin_age (sarah_age (eli_age freyja_age)) = 33 := by
sorry

end NUMINAMATH_CALUDE_kaylin_is_33_l3593_359369


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3593_359347

/-- Given that k, -1, and b form an arithmetic sequence,
    prove that the line y = kx + b passes through the point (1, -2) -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  (-1 = (k + b) / 2) →
  ∀ x y : ℝ, y = k * x + b → (x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3593_359347


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3593_359324

theorem square_ratio_side_length_sum (area_ratio : ℚ) : 
  area_ratio = 50 / 98 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b / c : ℝ) = Real.sqrt (area_ratio) ∧ 
    a = 5 ∧ b = 1 ∧ c = 7 ∧
    a + b + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3593_359324


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l3593_359342

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

/-- The monomial -1/4 * π * x^2 * y^3 has degree 5 -/
theorem degree_of_specific_monomial :
  degree_of_monomial 2 3 = 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l3593_359342


namespace NUMINAMATH_CALUDE_angle_expression_value_l3593_359330

theorem angle_expression_value (m : ℝ) (α : ℝ) : 
  m < 0 → 
  let M : ℝ × ℝ := (3 * m, -m)
  (∃ r : ℝ, r > 0 ∧ M.1 = r * Real.cos α ∧ M.2 = r * Real.sin α) →
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l3593_359330


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3593_359334

theorem cube_root_simplification : 
  (54880000 : ℝ)^(1/3) = 140 * 20^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3593_359334


namespace NUMINAMATH_CALUDE_power_bank_sales_theorem_l3593_359352

/-- Represents the sales scenario of mobile power banks -/
structure PowerBankSales where
  m : ℝ  -- Wholesale price per power bank
  n : ℝ  -- Markup per power bank
  total_count : ℕ := 100  -- Total number of power banks
  full_price_sold : ℕ := 60  -- Number of power banks sold at full price
  discount_rate : ℝ := 0.2  -- Discount rate for remaining power banks

/-- Calculates the total selling price of all power banks -/
def total_selling_price (s : PowerBankSales) : ℝ :=
  s.total_count * (s.m + s.n)

/-- Calculates the actual total revenue -/
def actual_revenue (s : PowerBankSales) : ℝ :=
  s.full_price_sold * (s.m + s.n) + 
  (s.total_count - s.full_price_sold) * (1 - s.discount_rate) * (s.m + s.n)

/-- Calculates the additional profit without discount -/
def additional_profit (s : PowerBankSales) : ℝ :=
  s.total_count * s.n - (actual_revenue s - s.total_count * s.m)

theorem power_bank_sales_theorem (s : PowerBankSales) :
  total_selling_price s = 100 * (s.m + s.n) ∧
  actual_revenue s = 92 * (s.m + s.n) ∧
  additional_profit s = 8 * (s.m + s.n) := by
  sorry

#check power_bank_sales_theorem

end NUMINAMATH_CALUDE_power_bank_sales_theorem_l3593_359352


namespace NUMINAMATH_CALUDE_max_square_plots_for_given_field_l3593_359374

/-- Represents the dimensions of a rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing and field dimensions -/
structure FencingProblem where
  field : FieldDimensions
  internalFencing : ℕ

/-- Calculates the maximum number of square plots given a fencing problem -/
def maxSquarePlots (problem : FencingProblem) : ℕ :=
  sorry

/-- The main theorem stating the solution to the specific problem -/
theorem max_square_plots_for_given_field :
  let problem : FencingProblem := {
    field := { width := 40, length := 60 },
    internalFencing := 2400
  }
  maxSquarePlots problem = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_square_plots_for_given_field_l3593_359374


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3593_359351

/-- A cylinder with a square axial cross-section of area 5 has a lateral surface area of 5π. -/
theorem cylinder_lateral_surface_area (h : ℝ) (r : ℝ) : 
  h * h = 5 → 2 * r = h → 2 * π * r * h = 5 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3593_359351


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3593_359332

theorem circle_equation_proof (x y : ℝ) :
  let circle_eq := (x - 5/3)^2 + y^2 = 25/9
  let line_eq := 3*x + y - 5 = 0
  let origin := (0, 0)
  let point := (3, -1)
  (∃ (center : ℝ × ℝ), 
    (center.1 - 5/3)^2 + center.2^2 = 25/9 ∧ 
    3*center.1 + center.2 - 5 = 0) ∧
  ((0 - 5/3)^2 + 0^2 = 25/9) ∧
  ((3 - 5/3)^2 + (-1)^2 = 25/9) →
  circle_eq
:= by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3593_359332


namespace NUMINAMATH_CALUDE_ratio_equality_l3593_359366

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 5 ∧ y / 5 = z / 7) :
  (x - y + z) / (x + y - z) = 5 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l3593_359366


namespace NUMINAMATH_CALUDE_post_office_packages_l3593_359329

theorem post_office_packages (letters_per_day : ℕ) (total_mail : ℕ) (months : ℕ) (days_per_month : ℕ) :
  letters_per_day = 60 →
  total_mail = 14400 →
  months = 6 →
  days_per_month = 30 →
  (total_mail - letters_per_day * months * days_per_month) / (months * days_per_month) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_post_office_packages_l3593_359329


namespace NUMINAMATH_CALUDE_purchase_total_cost_l3593_359312

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℚ := 2.44

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 2

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 0.87

/-- The number of sodas purchased -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 8.36

theorem purchase_total_cost : 
  (num_sandwiches : ℚ) * sandwich_cost + (num_sodas : ℚ) * soda_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_purchase_total_cost_l3593_359312


namespace NUMINAMATH_CALUDE_oyster_ratio_proof_l3593_359335

/-- Proves that the ratio of oysters on the second day to the first day is 1:2 -/
theorem oyster_ratio_proof (oysters_day1 crabs_day1 total_count : ℕ) 
  (h1 : oysters_day1 = 50)
  (h2 : crabs_day1 = 72)
  (h3 : total_count = 195)
  (h4 : ∃ (oysters_day2 crabs_day2 : ℕ), 
    crabs_day2 = 2 * crabs_day1 / 3 ∧ 
    oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 = total_count) :
  ∃ (oysters_day2 : ℕ), oysters_day2 * 2 = oysters_day1 :=
sorry

end NUMINAMATH_CALUDE_oyster_ratio_proof_l3593_359335


namespace NUMINAMATH_CALUDE_circle_center_l3593_359301

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_center (x y : ℝ) :
  CircleEquation 3 0 3 x y → (3, 0) = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3593_359301


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3593_359393

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the intersecting line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop := (2*k + 4) / (k^2) = 2

-- Main theorem
theorem parabola_intersection_length :
  ∀ (k : ℝ),
  parabola 2 4 →
  k > -1 →
  k ≠ 0 →
  midpoint_condition k →
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line k x₁ y₁ ∧ line k x₂ y₂ ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 60 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3593_359393


namespace NUMINAMATH_CALUDE_concert_revenue_is_930_l3593_359367

/-- Calculates the total revenue for a concert given the number of tickets sold and their prices. -/
def concert_revenue (student_tickets : ℕ) (non_student_tickets : ℕ) (student_price : ℕ) (non_student_price : ℕ) : ℕ :=
  student_tickets * student_price + non_student_tickets * non_student_price

/-- Proves that the total revenue for the concert is $930 given the specified conditions. -/
theorem concert_revenue_is_930 :
  let total_tickets : ℕ := 150
  let student_price : ℕ := 5
  let non_student_price : ℕ := 8
  let student_tickets : ℕ := 90
  let non_student_tickets : ℕ := 60
  concert_revenue student_tickets non_student_tickets student_price non_student_price = 930 :=
by
  sorry

#eval concert_revenue 90 60 5 8

end NUMINAMATH_CALUDE_concert_revenue_is_930_l3593_359367


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l3593_359340

theorem largest_perfect_square_factor_of_1800 : 
  ∃ (n : ℕ), n * n = 3600 ∧ 
  3600 ∣ 1800 ∧
  ∀ (m : ℕ), m * m ∣ 1800 → m * m ≤ 3600 :=
sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l3593_359340


namespace NUMINAMATH_CALUDE_total_lockers_is_399_l3593_359370

/-- Represents the position of Minyoung's locker in the classroom -/
structure LockerPosition where
  front : ℕ
  back : ℕ
  left : ℕ
  right : ℕ

/-- Calculates the total number of lockers in the classroom based on Minyoung's locker position -/
def total_lockers (pos : LockerPosition) : ℕ :=
  (pos.front + pos.back - 1) * (pos.left + pos.right - 1)

/-- Theorem stating that the total number of lockers is 399 given Minyoung's locker position -/
theorem total_lockers_is_399 (pos : LockerPosition) 
  (h_front : pos.front = 8)
  (h_back : pos.back = 14)
  (h_left : pos.left = 7)
  (h_right : pos.right = 13) : 
  total_lockers pos = 399 := by
  sorry

#eval total_lockers ⟨8, 14, 7, 13⟩

end NUMINAMATH_CALUDE_total_lockers_is_399_l3593_359370


namespace NUMINAMATH_CALUDE_unit_digit_of_2_power_2024_l3593_359354

theorem unit_digit_of_2_power_2024 (unit_digit : ℕ → ℕ) (h : ∀ n : ℕ, unit_digit (2^n) = unit_digit (2^(n % 4))) : unit_digit (2^2024) = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_2_power_2024_l3593_359354


namespace NUMINAMATH_CALUDE_translation_theorem_l3593_359359

/-- The original function f(x) = 2x^2 - 2x -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

/-- The transformed function g(x) = 2x^2 - 10x - 9 -/
def g (x : ℝ) : ℝ := 2 * x^2 - 10 * x - 9

/-- Theorem stating that g is the result of translating f 2 units right and 3 units down -/
theorem translation_theorem : ∀ x : ℝ, g x = f (x - 2) - 3 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l3593_359359


namespace NUMINAMATH_CALUDE_proportionality_coefficient_l3593_359309

/-- Given variables x y z : ℝ and a constant k : ℕ+, prove that k = 3 -/
theorem proportionality_coefficient (x y z : ℝ) (k : ℕ+) : 
  (z - y = k * x) →  -- The difference of z and y is proportional to x
  (x - z = k * y) →  -- The difference of x and z is proportional to y
  (∃ (x' y' z' : ℝ), z' = (5/3) * (x' - y')) →  -- A certain value of z is 5/3 times the difference of x and y
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_proportionality_coefficient_l3593_359309


namespace NUMINAMATH_CALUDE_c_months_correct_l3593_359375

/-- The number of months c put his oxen for grazing -/
def c_months : ℝ :=
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let total_rent := 210
  let c_share := 53.99999999999999
  3

/-- Theorem stating that c_months is correct given the problem conditions -/
theorem c_months_correct :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let total_rent := 210
  let c_share := 53.99999999999999
  let total_ox_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months
  c_share = (c_oxen * c_months / total_ox_months) * total_rent :=
by sorry

#eval c_months

end NUMINAMATH_CALUDE_c_months_correct_l3593_359375


namespace NUMINAMATH_CALUDE_eight_to_twelve_sum_equals_two_to_x_l3593_359323

theorem eight_to_twelve_sum_equals_two_to_x : ∃ x : ℕ, 
  8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 = 2^x ∧ x = 39 := by
  sorry

end NUMINAMATH_CALUDE_eight_to_twelve_sum_equals_two_to_x_l3593_359323


namespace NUMINAMATH_CALUDE_robert_ate_ten_chocolates_l3593_359311

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The number of additional chocolates Robert ate compared to Nickel -/
def robert_additional_chocolates : ℕ := 5

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := nickel_chocolates + robert_additional_chocolates

theorem robert_ate_ten_chocolates : robert_chocolates = 10 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_ten_chocolates_l3593_359311


namespace NUMINAMATH_CALUDE_only_prime_of_form_l3593_359399

theorem only_prime_of_form (p : ℕ) : 
  (∃ x : ℤ, p = 4 * x^4 + 1) ∧ Nat.Prime p ↔ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_only_prime_of_form_l3593_359399


namespace NUMINAMATH_CALUDE_f_max_value_l3593_359338

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + x + 1)

theorem f_max_value (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1/2 → ∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ Real.exp (-1/a)) ∧
  (a > 1/2 → ∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ Real.exp (-2) * (4*a - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3593_359338


namespace NUMINAMATH_CALUDE_area_of_DEFGHT_l3593_359368

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a square -/
structure Square :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Function to calculate the area of a shape formed by DEFGHT -/
def areaOfDEFGHT (ABC : Triangle) (ABDE : Square) (CAFG : Square) (BCHT : Triangle) : ℝ :=
  sorry

/-- Theorem stating the area of shape DEFGHT -/
theorem area_of_DEFGHT :
  ∀ (ABC : Triangle) (ABDE : Square) (CAFG : Square) (BCHT : Triangle),
  (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 = 4 ∧  -- Side length of ABC is 2
  (ABC.B.x - ABC.C.x)^2 + (ABC.B.y - ABC.C.y)^2 = 4 ∧
  (ABC.C.x - ABC.A.x)^2 + (ABC.C.y - ABC.A.y)^2 = 4 ∧
  (ABDE.A = ABC.A ∧ ABDE.B = ABC.B) ∧  -- ABDE is a square outside ABC
  (CAFG.C = ABC.A ∧ CAFG.A = ABC.C) ∧  -- CAFG is a square outside ABC
  (BCHT.B = ABC.B ∧ BCHT.C = ABC.C) ∧  -- BCHT is an equilateral triangle outside ABC
  (BCHT.A.x - BCHT.B.x)^2 + (BCHT.A.y - BCHT.B.y)^2 = 4 ∧  -- BCHT is equilateral with side length 2
  (BCHT.B.x - BCHT.C.x)^2 + (BCHT.B.y - BCHT.C.y)^2 = 4 ∧
  (BCHT.C.x - BCHT.A.x)^2 + (BCHT.C.y - BCHT.A.y)^2 = 4 →
  areaOfDEFGHT ABC ABDE CAFG BCHT = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_DEFGHT_l3593_359368


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l3593_359390

/-- The amount of flour in cups that Mary has already added to the recipe. -/
def flour_already_added : ℕ := 2

/-- The amount of flour in cups that Mary needs to add to the recipe. -/
def flour_to_be_added : ℕ := 5

/-- The total amount of flour in cups that the recipe calls for. -/
def total_flour : ℕ := flour_already_added + flour_to_be_added

theorem recipe_flour_amount :
  total_flour = 7 :=
by sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l3593_359390


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l3593_359389

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l3593_359389


namespace NUMINAMATH_CALUDE_port_vessels_count_l3593_359395

theorem port_vessels_count :
  let cruise_ships : ℕ := 4
  let cargo_ships : ℕ := 2 * cruise_ships
  let sailboats : ℕ := cargo_ships + 6
  let fishing_boats : ℕ := sailboats / 7
  let total_vessels : ℕ := cruise_ships + cargo_ships + sailboats + fishing_boats
  total_vessels = 28 :=
by sorry

end NUMINAMATH_CALUDE_port_vessels_count_l3593_359395


namespace NUMINAMATH_CALUDE_min_score_for_maria_l3593_359377

def min_score_for_advanced_class (scores : List ℚ) (required_average : ℚ) : ℚ :=
  let total_terms := 5
  let current_sum := scores.sum
  max ((required_average * total_terms) - current_sum) 0

theorem min_score_for_maria : 
  min_score_for_advanced_class [84/100, 80/100, 82/100, 83/100] (85/100) = 96/100 := by
  sorry

end NUMINAMATH_CALUDE_min_score_for_maria_l3593_359377


namespace NUMINAMATH_CALUDE_triangle_problem_l3593_359306

/-- Given a triangle ABC with sides AB, BC, and AC, prove the length of BC and the value of tan 2B. -/
theorem triangle_problem (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let dot_product := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  AB = 6 ∧ AC = 3 * Real.sqrt 2 ∧ dot_product = -18 →
  BC = 3 * Real.sqrt 10 ∧ Real.tan (2 * Real.arctan ((B.2 - A.2) / (B.1 - A.1))) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3593_359306


namespace NUMINAMATH_CALUDE_min_distance_to_point_l3593_359372

theorem min_distance_to_point (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) :
  ∃ (min : ℝ), min = 7 / 10 ∧ ∀ (x' y' : ℝ), 6 * x' + 8 * y' - 1 = 0 →
    Real.sqrt (x' ^ 2 + y' ^ 2 - 2 * y' + 1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l3593_359372


namespace NUMINAMATH_CALUDE_expression_value_l3593_359336

theorem expression_value (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  (3 * x - 4 * y) / z = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3593_359336


namespace NUMINAMATH_CALUDE_pathway_area_is_196_l3593_359325

/-- Represents the farm layout --/
structure FarmLayout where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  pathway_width : Nat

/-- Calculates the total area of pathways in the farm --/
def pathway_area (farm : FarmLayout) : Nat :=
  let total_width := farm.columns * farm.bed_width + (farm.columns + 1) * farm.pathway_width
  let total_height := farm.rows * farm.bed_height + (farm.rows + 1) * farm.pathway_width
  let total_area := total_width * total_height
  let beds_area := farm.rows * farm.columns * farm.bed_width * farm.bed_height
  total_area - beds_area

/-- Theorem stating that the pathway area for the given farm layout is 196 square feet --/
theorem pathway_area_is_196 (farm : FarmLayout) 
    (h1 : farm.rows = 4)
    (h2 : farm.columns = 3)
    (h3 : farm.bed_width = 4)
    (h4 : farm.bed_height = 3)
    (h5 : farm.pathway_width = 2) : 
  pathway_area farm = 196 := by
  sorry

end NUMINAMATH_CALUDE_pathway_area_is_196_l3593_359325


namespace NUMINAMATH_CALUDE_particle_paths_l3593_359319

theorem particle_paths (n k : ℕ) : 
  (n = 5 ∧ k = 3) → (Nat.choose n ((n + k) / 2) = 5) ∧
  (n = 20 ∧ k = 16) → (Nat.choose n ((n + k) / 2) = 190) :=
by sorry

end NUMINAMATH_CALUDE_particle_paths_l3593_359319


namespace NUMINAMATH_CALUDE_product_equals_power_of_three_l3593_359387

theorem product_equals_power_of_three : 25 * 15 * 9 * 5.4 * 3.24 = 3^10 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_power_of_three_l3593_359387


namespace NUMINAMATH_CALUDE_existence_of_divisibility_l3593_359378

/-- The largest proper divisor of a positive integer -/
def largest_proper_divisor (n : ℕ) : ℕ := sorry

/-- The sequence u_n as defined in the problem -/
def u : ℕ → ℕ
  | 0 => sorry  -- This value is not specified in the problem
  | 1 => sorry  -- We only know u_1 > 0, but not its exact value
  | (n + 2) => u (n + 1) + largest_proper_divisor (u (n + 1))

theorem existence_of_divisibility :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (3^2019 : ℕ) ∣ u n :=
sorry

end NUMINAMATH_CALUDE_existence_of_divisibility_l3593_359378


namespace NUMINAMATH_CALUDE_fraction_simplification_l3593_359362

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3593_359362


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3593_359343

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 64 → a / b = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3593_359343


namespace NUMINAMATH_CALUDE_hot_chocolate_max_servings_l3593_359318

/-- Represents the recipe for hot chocolate -/
structure Recipe where
  chocolate : ℚ
  sugar : ℚ
  water : ℚ
  milk : ℚ
  servings : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  chocolate : ℚ
  sugar : ℚ
  milk : ℚ

/-- Calculates the maximum number of servings possible given a recipe and available ingredients -/
def max_servings (recipe : Recipe) (ingredients : Ingredients) : ℚ :=
  min (ingredients.chocolate / recipe.chocolate * recipe.servings)
      (min (ingredients.sugar / recipe.sugar * recipe.servings)
           (ingredients.milk / recipe.milk * recipe.servings))

theorem hot_chocolate_max_servings :
  let recipe : Recipe := {
    chocolate := 3,
    sugar := 1/3,
    water := 3/2,
    milk := 5,
    servings := 6
  }
  let ingredients : Ingredients := {
    chocolate := 8,
    sugar := 3,
    milk := 12
  }
  max_servings recipe ingredients = 16 := by sorry

end NUMINAMATH_CALUDE_hot_chocolate_max_servings_l3593_359318


namespace NUMINAMATH_CALUDE_limit_of_rational_function_l3593_359386

theorem limit_of_rational_function (f : ℝ → ℝ) (h : ∀ x ≠ 1, f x = (x^4 - 1) / (2*x^4 - x^2 - 1)) :
  ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |f x - 2/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_rational_function_l3593_359386


namespace NUMINAMATH_CALUDE_sheep_buying_equation_l3593_359396

/-- Represents the price of the sheep -/
def sheep_price (x : ℤ) : ℤ := 5 * x + 45

/-- Represents the total contribution when each person gives 7 coins -/
def contribution_7 (x : ℤ) : ℤ := 7 * x

theorem sheep_buying_equation (x : ℤ) : 
  sheep_price x = contribution_7 x - 3 := by sorry

end NUMINAMATH_CALUDE_sheep_buying_equation_l3593_359396


namespace NUMINAMATH_CALUDE_sixth_element_row_20_l3593_359307

/-- Pascal's triangle function: computes the k-th element of the n-th row -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- The sixth element in Row 20 of Pascal's triangle is 15504 -/
theorem sixth_element_row_20 : pascal 20 5 = 15504 := by
  sorry

end NUMINAMATH_CALUDE_sixth_element_row_20_l3593_359307


namespace NUMINAMATH_CALUDE_sum_of_digits_l3593_359346

theorem sum_of_digits (a b c d : ℕ) : 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  100 * a + 10 * b + c + 100 * d + 10 * c + a = 1100 →
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3593_359346


namespace NUMINAMATH_CALUDE_train_speed_l3593_359328

theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3593_359328


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3593_359320

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x - 5

-- State the theorem
theorem monotonic_increasing_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3593_359320


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l3593_359353

theorem min_value_of_sum_of_fractions (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) (hy : -1 < y ∧ y < 0) (hz : -1 < z ∧ z < 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (w : ℝ), w = 1/((1-x)*(1-y)*(1-z)) + 1/((1+x)*(1+y)*(1+z)) → m ≤ w :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l3593_359353


namespace NUMINAMATH_CALUDE_sum_in_base8_l3593_359384

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

theorem sum_in_base8 : 
  base10ToBase8 (base8ToBase10 24 + base8ToBase10 157) = 203 := by sorry

end NUMINAMATH_CALUDE_sum_in_base8_l3593_359384


namespace NUMINAMATH_CALUDE_ellipse_segment_area_l3593_359304

/-- The area of the segment of an ellipse above an inscribed isosceles triangle -/
theorem ellipse_segment_area (a b c : ℝ) (h_a : a = 3) (h_b : b = 2) (h_c : c = 1) :
  let y := (4 * Real.sqrt 2) / 3
  let ellipse_area := π * a * b
  let triangle_area := c * (b - y)
  let segment_area := ellipse_area - triangle_area
  segment_area = 6 * π - 2 + (4 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_segment_area_l3593_359304


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_angle_l3593_359382

/-- A quadrilateral with specific side lengths and angles has diagonals that intersect at a 60° angle. -/
theorem quadrilateral_diagonal_angle (a b c : ℝ) (angle_ab angle_bc : ℝ) :
  a = 4 * Real.sqrt 3 →
  b = 9 →
  c = Real.sqrt 3 →
  angle_ab = π / 6 →  -- 30° in radians
  angle_bc = π / 2 →  -- 90° in radians
  ∃ (angle_diagonals : ℝ), angle_diagonals = π / 3 :=  -- 60° in radians
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_angle_l3593_359382


namespace NUMINAMATH_CALUDE_jean_stuffies_fraction_l3593_359337

theorem jean_stuffies_fraction (total : ℕ) (kept_fraction : ℚ) (janet_received : ℕ) :
  total = 60 →
  kept_fraction = 1/3 →
  janet_received = 10 →
  (janet_received : ℚ) / (total - total * kept_fraction) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_jean_stuffies_fraction_l3593_359337


namespace NUMINAMATH_CALUDE_special_polygon_is_heptagon_l3593_359327

/-- A polygon where all diagonals passing through one vertex divide it into 5 triangles -/
structure SpecialPolygon where
  /-- The number of triangles formed by diagonals passing through one vertex -/
  num_triangles : ℕ
  /-- The number of triangles is exactly 5 -/
  h_triangles : num_triangles = 5

/-- The number of sides in a SpecialPolygon -/
def SpecialPolygon.num_sides (p : SpecialPolygon) : ℕ :=
  p.num_triangles + 2

theorem special_polygon_is_heptagon (p : SpecialPolygon) : p.num_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_is_heptagon_l3593_359327


namespace NUMINAMATH_CALUDE_expression_evaluation_l3593_359388

theorem expression_evaluation : 
  |-2| + (1/3)⁻¹ - Real.sqrt 9 + (Real.sin (45 * π / 180) - 1)^0 - (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3593_359388


namespace NUMINAMATH_CALUDE_cake_piece_volume_and_icing_area_sum_l3593_359302

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the volume of a triangular prism -/
def triangularPrismVolume (base : ℝ) (height : ℝ) : ℝ := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ := sorry

/-- Main theorem: The sum of the volume and icing area of the cake piece is 19.8 -/
theorem cake_piece_volume_and_icing_area_sum :
  let a : ℝ := 3  -- edge length of the cube
  let p : Point3D := ⟨0, 0, 0⟩
  let q : Point3D := ⟨a, 0, 0⟩
  let r : Point3D := ⟨0, a, 0⟩
  let m : Point3D := ⟨a/3, a, 0⟩
  let triangleQMR_area : ℝ := triangleArea q m r
  let volume : ℝ := triangularPrismVolume triangleQMR_area a
  let icingArea : ℝ := triangleQMR_area + rectangleArea a a
  volume + icingArea = 19.8 := by sorry

end NUMINAMATH_CALUDE_cake_piece_volume_and_icing_area_sum_l3593_359302


namespace NUMINAMATH_CALUDE_math_books_count_l3593_359357

/-- Given a shelf of books with the following properties:
  * There are 100 books in total
  * 32 of them are history books
  * 25 of them are geography books
  * The rest are math books
  This theorem proves that there are 43 math books. -/
theorem math_books_count (total : ℕ) (history : ℕ) (geography : ℕ) (math : ℕ) 
  (h_total : total = 100)
  (h_history : history = 32)
  (h_geography : geography = 25)
  (h_sum : total = history + geography + math) :
  math = 43 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l3593_359357


namespace NUMINAMATH_CALUDE_max_value_3m_4n_l3593_359380

/-- The sum of the first m positive even numbers -/
def sumEven (m : ℕ) : ℕ := m * (m + 1)

/-- The sum of the first n positive odd numbers -/
def sumOdd (n : ℕ) : ℕ := n^2

/-- The constraint that the sum of m distinct positive even numbers 
    and n distinct positive odd numbers is 1987 -/
def constraint (m n : ℕ) : Prop := sumEven m + sumOdd n = 1987

/-- The theorem stating that the maximum value of 3m + 4n is 221 
    given the constraint -/
theorem max_value_3m_4n : 
  ∀ m n : ℕ, constraint m n → 3 * m + 4 * n ≤ 221 :=
sorry

end NUMINAMATH_CALUDE_max_value_3m_4n_l3593_359380


namespace NUMINAMATH_CALUDE_camel_moves_divisible_by_three_l3593_359383

/-- Represents the color of a square --/
inductive SquareColor
| Black
| White

/-- Represents a camel's movement --/
def CamelMove := ℕ → SquareColor

/-- A camel's movement pattern that alternates between black and white squares --/
def alternatingPattern : CamelMove :=
  fun n => match n % 3 with
    | 0 => SquareColor.Black
    | 1 => SquareColor.White
    | _ => SquareColor.Black

/-- Theorem: If a camel makes n moves in an alternating pattern and returns to its starting position, then n is divisible by 3 --/
theorem camel_moves_divisible_by_three (n : ℕ) 
  (h1 : alternatingPattern n = alternatingPattern 0) : 
  3 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_camel_moves_divisible_by_three_l3593_359383


namespace NUMINAMATH_CALUDE_factorial_square_root_theorem_l3593_359371

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_square_root_theorem :
  (((factorial 5 * factorial 4).sqrt) ^ 2 : ℕ) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_theorem_l3593_359371


namespace NUMINAMATH_CALUDE_number_of_lineups_l3593_359350

/-- Represents the number of players in the team -/
def total_players : ℕ := 15

/-- Represents the number of players in the starting lineup -/
def lineup_size : ℕ := 4

/-- Represents the number of players that must be in the starting lineup -/
def fixed_players : ℕ := 3

/-- Calculates the number of possible starting lineups -/
def possible_lineups : ℕ := Nat.choose (total_players - fixed_players) (lineup_size - fixed_players)

/-- Theorem stating that the number of possible starting lineups is 12 -/
theorem number_of_lineups : possible_lineups = 12 := by sorry

end NUMINAMATH_CALUDE_number_of_lineups_l3593_359350


namespace NUMINAMATH_CALUDE_reduction_after_four_trials_l3593_359394

/-- The reduction factor for the 0.618 method -/
def golden_ratio_inverse : ℝ := 0.618

/-- The number of trials -/
def num_trials : ℕ := 4

/-- The reduction factor after n trials using the 0.618 method -/
def reduction_factor (n : ℕ) : ℝ := golden_ratio_inverse ^ (n - 1)

/-- Theorem stating the reduction factor after 4 trials -/
theorem reduction_after_four_trials :
  reduction_factor num_trials = golden_ratio_inverse ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_reduction_after_four_trials_l3593_359394


namespace NUMINAMATH_CALUDE_vector_square_difference_l3593_359364

theorem vector_square_difference (a b : ℝ × ℝ) (h1 : a + b = (-3, 6)) (h2 : a - b = (-3, 2)) :
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by sorry

end NUMINAMATH_CALUDE_vector_square_difference_l3593_359364


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_is_false_l3593_359305

/-- Definition of corresponding angles -/
def corresponding_angles (α β : Angle) : Prop := sorry

/-- Theorem stating that the proposition "If two angles are corresponding angles, then these two angles are equal" is false in general -/
theorem corresponding_angles_equal_is_false :
  ¬ (∀ α β : Angle, corresponding_angles α β → α = β) :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_is_false_l3593_359305


namespace NUMINAMATH_CALUDE_set_operations_l3593_359315

def I : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 5, 6, 7}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7}) ∧
  (A ∩ (I \ B) = {1, 2, 4}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l3593_359315


namespace NUMINAMATH_CALUDE_equation_solution_l3593_359321

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3593_359321


namespace NUMINAMATH_CALUDE_average_price_per_book_l3593_359314

theorem average_price_per_book (books_shop1 : ℕ) (price_shop1 : ℕ) 
  (books_shop2 : ℕ) (price_shop2 : ℕ) 
  (h1 : books_shop1 = 65) (h2 : price_shop1 = 1380) 
  (h3 : books_shop2 = 55) (h4 : price_shop2 = 900) : 
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l3593_359314


namespace NUMINAMATH_CALUDE_six_steps_position_l3593_359303

/-- Given a number line where 8 equal steps span from 0 to 32, 
    the position after 6 steps from 0 is 24. -/
theorem six_steps_position (step_size : ℝ) (h1 : step_size > 0) 
  (h2 : 8 * step_size = 32) : 6 * step_size = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_steps_position_l3593_359303


namespace NUMINAMATH_CALUDE_dachshund_starting_weight_l3593_359339

theorem dachshund_starting_weight :
  ∀ (labrador_start dachshund_start : ℝ),
    labrador_start = 40 →
    (labrador_start * 1.25 - dachshund_start * 1.25 = 35) →
    dachshund_start = 12 := by
  sorry

end NUMINAMATH_CALUDE_dachshund_starting_weight_l3593_359339


namespace NUMINAMATH_CALUDE_tan_sqrt_three_iff_periodic_l3593_359360

theorem tan_sqrt_three_iff_periodic (x : ℝ) : 
  Real.tan x = Real.sqrt 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt_three_iff_periodic_l3593_359360


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l3593_359333

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  2.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    wheels on the front axle, and wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem (totalWheels frontAxleWheels otherAxleWheels : ℕ)
    (h1 : totalWheels = 18)
    (h2 : frontAxleWheels = 2)
    (h3 : otherAxleWheels = 4) :
  toll (calculateAxles totalWheels frontAxleWheels otherAxleWheels) = 4 :=
by
  sorry

#eval toll (calculateAxles 18 2 4)

end NUMINAMATH_CALUDE_truck_toll_theorem_l3593_359333


namespace NUMINAMATH_CALUDE_sequence_properties_l3593_359356

def sequence_a (n : ℕ) : ℝ := 2^n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sequence_b (n : ℕ) : ℝ := (n + 1 : ℝ) * sequence_a n

def sum_T (n : ℕ) : ℝ := n * 2^(n + 1)

theorem sequence_properties (n : ℕ) :
  (∀ k, sum_S k = 2 * sequence_a k - 2) →
  (sequence_a n = 2^n ∧
   sum_T n = n * 2^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3593_359356


namespace NUMINAMATH_CALUDE_library_seating_l3593_359373

theorem library_seating (x : ℕ) : 
  (∃ (y : ℕ), x + y = 16) →  -- Total number of chairs and stools is 16
  (4 * x + 3 * (16 - x) = 60) -- Equation representing the situation
  :=
by
  sorry

end NUMINAMATH_CALUDE_library_seating_l3593_359373


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3593_359376

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 24 * x + 35 = (d * x + e)^2 + f) → 
  d * e = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3593_359376


namespace NUMINAMATH_CALUDE_paint_left_calculation_paint_problem_solution_l3593_359355

/-- Given the total amount of paint needed and the amount of paint to buy,
    calculate the amount of paint left from the previous project. -/
theorem paint_left_calculation (total_paint : ℕ) (paint_to_buy : ℕ) :
  total_paint ≥ paint_to_buy →
  total_paint - paint_to_buy = total_paint - paint_to_buy :=
by
  sorry

/-- The specific problem instance -/
def paint_problem : ℕ × ℕ := (333, 176)

/-- The solution to the specific problem instance -/
theorem paint_problem_solution :
  let (total_paint, paint_to_buy) := paint_problem
  total_paint - paint_to_buy = 157 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_left_calculation_paint_problem_solution_l3593_359355


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3593_359363

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  c = 65 →                 -- Hypotenuse length
  a < b →                  -- a is the shorter leg
  a = 39 :=                -- Shorter leg length
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3593_359363


namespace NUMINAMATH_CALUDE_circle_area_l3593_359385

theorem circle_area (r : ℝ) (h : r = 5) : π * r^2 = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l3593_359385


namespace NUMINAMATH_CALUDE_second_scenario_pipes_l3593_359348

/-- The number of hours it takes to fill the tank with 3 pipes -/
def three_pipe_time : ℕ := 8

/-- The number of hours it takes to fill the tank with an unknown number of pipes -/
def unknown_pipe_time : ℕ := 12

/-- The capacity of each pipe (amount of water filled per hour) -/
def pipe_capacity : ℝ := 1

/-- The total capacity needed to fill the tank -/
def tank_capacity : ℝ := 3 * pipe_capacity * three_pipe_time

/-- The number of pipes in the second scenario -/
def num_pipes : ℕ := 2

theorem second_scenario_pipes :
  num_pipes * pipe_capacity * unknown_pipe_time = tank_capacity := by sorry

end NUMINAMATH_CALUDE_second_scenario_pipes_l3593_359348


namespace NUMINAMATH_CALUDE_tire_price_calculation_l3593_359365

theorem tire_price_calculation (num_tires sale_price total_savings : ℕ) 
  (h1 : num_tires = 4)
  (h2 : sale_price = 75)
  (h3 : total_savings = 36)
  : sale_price + total_savings / num_tires = 84 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_calculation_l3593_359365


namespace NUMINAMATH_CALUDE_no_positive_solutions_iff_p_in_range_l3593_359344

/-- The set A of real solutions to the quadratic equation x^2 + (p + 2)x + 1 = 0 -/
def A (p : ℝ) : Set ℝ :=
  {x : ℝ | x^2 + (p + 2)*x + 1 = 0}

/-- The theorem stating the equivalence between A having no positive real solutions
    and p belonging to the specified range -/
theorem no_positive_solutions_iff_p_in_range (p : ℝ) :
  (A p ∩ Set.Ici 0 = ∅) ↔ p ∈ Set.Ioo (-4) 0 ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_solutions_iff_p_in_range_l3593_359344


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_49_l3593_359398

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_49 :
  units_digit (sum_factorials 49) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_49_l3593_359398


namespace NUMINAMATH_CALUDE_right_triangle_height_l3593_359392

theorem right_triangle_height (a b h : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  a = 1 → b = 4 → h^2 = a * b → h = 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_height_l3593_359392


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3593_359310

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 4020) : 
  (∃ (a' b' c' d' e' f' : ℕ+), 
    a' + b' + c' + d' + e' + f' = 4020 ∧ 
    max (a' + b') (max (b' + c') (max (c' + d') (max (d' + e') (e' + f')))) = 805) ∧
  (∀ (a'' b'' c'' d'' e'' f'' : ℕ+),
    a'' + b'' + c'' + d'' + e'' + f'' = 4020 →
    max (a'' + b'') (max (b'' + c'') (max (c'' + d'') (max (d'' + e'') (e'' + f'')))) ≥ 805) :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3593_359310


namespace NUMINAMATH_CALUDE_harriet_siblings_product_l3593_359358

/-- Represents a family with a specific structure -/
structure Family where
  harry_sisters : Nat
  harry_brothers : Nat

/-- Calculates the number of Harriet's sisters (excluding herself) -/
def harriet_sisters (f : Family) : Nat :=
  f.harry_sisters - 1

/-- Calculates the number of Harriet's brothers -/
def harriet_brothers (f : Family) : Nat :=
  f.harry_brothers

/-- Theorem stating that the product of Harriet's siblings is 9 -/
theorem harriet_siblings_product (f : Family) 
  (h1 : f.harry_sisters = 4) 
  (h2 : f.harry_brothers = 3) : 
  (harriet_sisters f) * (harriet_brothers f) = 9 := by
  sorry


end NUMINAMATH_CALUDE_harriet_siblings_product_l3593_359358


namespace NUMINAMATH_CALUDE_min_value_theorem_l3593_359381

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  4 / x + 9 / y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 4 / x₀ + 9 / y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3593_359381


namespace NUMINAMATH_CALUDE_lady_walking_distance_l3593_359379

theorem lady_walking_distance (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 12) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_lady_walking_distance_l3593_359379


namespace NUMINAMATH_CALUDE_simplify_expression_l3593_359316

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3593_359316


namespace NUMINAMATH_CALUDE_female_officers_count_l3593_359313

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_duty_percentage : ℚ) :
  total_on_duty = 360 →
  female_on_duty_ratio = 1/2 →
  female_duty_percentage = 3/5 →
  (↑total_on_duty * female_on_duty_ratio / female_duty_percentage : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3593_359313
