import Mathlib

namespace NUMINAMATH_CALUDE_overlapping_segments_length_l2953_295367

/-- Given a set of overlapping segments with the following properties:
  - The total length of all segments is 98 cm
  - The actual distance from one end to the other is 83 cm
  - There are 6 overlapping regions of equal length
  Prove that the length of each overlapping region is 2.5 cm -/
theorem overlapping_segments_length
  (total_length : ℝ)
  (actual_distance : ℝ)
  (num_overlaps : ℕ)
  (h1 : total_length = 98)
  (h2 : actual_distance = 83)
  (h3 : num_overlaps = 6) :
  (total_length - actual_distance) / num_overlaps = 2.5 :=
sorry

end NUMINAMATH_CALUDE_overlapping_segments_length_l2953_295367


namespace NUMINAMATH_CALUDE_agnes_current_age_l2953_295383

/-- Agnes's current age -/
def agnes_age : ℕ := 25

/-- Jane's current age -/
def jane_age : ℕ := 6

/-- Years into the future when Agnes will be twice Jane's age -/
def years_future : ℕ := 13

theorem agnes_current_age :
  agnes_age = 25 ∧
  jane_age = 6 ∧
  agnes_age + years_future = 2 * (jane_age + years_future) :=
by sorry

end NUMINAMATH_CALUDE_agnes_current_age_l2953_295383


namespace NUMINAMATH_CALUDE_triangle_third_height_bound_l2953_295338

theorem triangle_third_height_bound (a b c : ℝ) (ha hb : ℝ) (h : ℝ) : 
  ha = 12 → hb = 20 → 
  a * ha = b * hb → 
  c * h = a * ha → 
  c > a - b → 
  h < 30 := by sorry

end NUMINAMATH_CALUDE_triangle_third_height_bound_l2953_295338


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2953_295384

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 7 + 47 / 99 ∧ 
  n + d = 839 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2953_295384


namespace NUMINAMATH_CALUDE_oil_drop_probability_l2953_295398

theorem oil_drop_probability (r : Real) (s : Real) (h1 : r = 1) (h2 : s = 0.5) : 
  (s^2) / (π * r^2) = 1 / (4 * π) :=
sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l2953_295398


namespace NUMINAMATH_CALUDE_set_operation_proof_l2953_295311

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, 1, 2}

theorem set_operation_proof :
  A ∪ (U \ B) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_proof_l2953_295311


namespace NUMINAMATH_CALUDE_length_OP_greater_than_radius_l2953_295345

-- Define a circle with radius 5
def circle_radius : ℝ := 5

-- Define a point P outside the circle
def point_outside_circle (P : ℝ × ℝ) : Prop :=
  let O := (0, 0)  -- Assume the circle center is at the origin
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) > circle_radius

-- Theorem statement
theorem length_OP_greater_than_radius (P : ℝ × ℝ) 
  (h : point_outside_circle P) : 
  Real.sqrt ((P.1)^2 + (P.2)^2) > circle_radius :=
sorry

end NUMINAMATH_CALUDE_length_OP_greater_than_radius_l2953_295345


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l2953_295307

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = -2 + 3*I) → k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l2953_295307


namespace NUMINAMATH_CALUDE_remainder_problem_l2953_295392

theorem remainder_problem (x : Int) : 
  x % 14 = 11 → x % 84 = 81 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2953_295392


namespace NUMINAMATH_CALUDE_sum_of_consecutive_even_numbers_l2953_295388

theorem sum_of_consecutive_even_numbers : 
  let a : ℕ := 80
  let b : ℕ := 82
  let c : ℕ := 84
  (a + 2 = b) ∧ (b + 2 = c) → a + b + c = 246 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_even_numbers_l2953_295388


namespace NUMINAMATH_CALUDE_total_amount_paid_l2953_295328

def ticket_cost : ℕ := 44
def num_people : ℕ := 3
def service_fee : ℕ := 18

theorem total_amount_paid : 
  ticket_cost * num_people + service_fee = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l2953_295328


namespace NUMINAMATH_CALUDE_quadratic_sum_l2953_295355

/-- Given a quadratic polynomial 20x^2 + 160x + 800, when expressed in the form a(x+b)^2 + c,
    the sum a + b + c equals 504. -/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧
  (a + b + c = 504) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2953_295355


namespace NUMINAMATH_CALUDE_ratio_approximation_l2953_295368

/-- The set of numbers from 1 to 10^13 in powers of 10 -/
def powerSet : Set ℕ := {n | ∃ k : ℕ, k ≤ 13 ∧ n = 10^k}

/-- The largest element in the set -/
def largestElement : ℕ := 10^13

/-- The sum of all elements in the set except the largest -/
def sumOfOthers : ℕ := (largestElement - 1) / 9

/-- The ratio of the largest element to the sum of others -/
def ratio : ℚ := largestElement / sumOfOthers

theorem ratio_approximation : ∃ ε > 0, abs (ratio - 9) < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_approximation_l2953_295368


namespace NUMINAMATH_CALUDE_range_of_fraction_l2953_295370

theorem range_of_fraction (x y : ℝ) (h : x + Real.sqrt (1 - y^2) = 0) :
  ∃ (a b : ℝ), a = -Real.sqrt 3 / 3 ∧ b = Real.sqrt 3 / 3 ∧
  ∀ (z : ℝ), (∃ (x' y' : ℝ), x' + Real.sqrt (1 - y'^2) = 0 ∧ z = y' / (x' - 2)) →
  a ≤ z ∧ z ≤ b :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2953_295370


namespace NUMINAMATH_CALUDE_triangle_third_angle_l2953_295386

theorem triangle_third_angle (a b : ℝ) (ha : a = 115) (hb : b = 30) : 180 - a - b = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l2953_295386


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2953_295306

def B : Set ℕ := {n | ∃ x : ℕ, x > 0 ∧ n = 4*x + 2}

theorem gcd_of_B_is_two : 
  ∃ (d : ℕ), d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2953_295306


namespace NUMINAMATH_CALUDE_white_pairs_count_l2953_295344

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles when folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_blue : ℕ

/-- The main theorem stating the number of coinciding white pairs -/
theorem white_pairs_count (half_count : TriangleCount) (coinciding : CoincidingPairs) : 
  half_count.red = 4 ∧ 
  half_count.blue = 4 ∧ 
  half_count.white = 6 ∧
  coinciding.red_red = 3 ∧
  coinciding.blue_blue = 2 ∧
  coinciding.red_blue = 1 →
  (half_count.white : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_count_l2953_295344


namespace NUMINAMATH_CALUDE_system_solution_equation_solution_l2953_295375

-- Part 1: System of equations
theorem system_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (x + y = -12) ∧ (x = -3) ∧ (y = -9) := by
  sorry

-- Part 2: Single equation
theorem equation_solution :
  ∃! x : ℝ, (2 / (1 - x) + 1 = x / (1 + x)) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_equation_solution_l2953_295375


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l2953_295393

/-- Proves that the number of apples handed out to students is 42 -/
theorem apples_handed_out_to_students (initial_apples : ℕ) (pies : ℕ) (apples_per_pie : ℕ) 
  (h1 : initial_apples = 96)
  (h2 : pies = 9)
  (h3 : apples_per_pie = 6) : 
  initial_apples - pies * apples_per_pie = 42 := by
  sorry

#check apples_handed_out_to_students

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l2953_295393


namespace NUMINAMATH_CALUDE_merchant_gross_profit_l2953_295333

/-- The merchant's gross profit on a jacket sale --/
theorem merchant_gross_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  purchase_price = 42 ∧ 
  markup_percent = 0.3 ∧ 
  discount_percent = 0.2 → 
  let selling_price := purchase_price / (1 - markup_percent)
  let discounted_price := selling_price * (1 - discount_percent)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 6 := by
sorry


end NUMINAMATH_CALUDE_merchant_gross_profit_l2953_295333


namespace NUMINAMATH_CALUDE_francisFamily_violins_l2953_295376

theorem francisFamily_violins :
  let ukuleles : ℕ := 2
  let guitars : ℕ := 4
  let ukuleleStrings : ℕ := 4
  let guitarStrings : ℕ := 6
  let violinStrings : ℕ := 4
  let totalStrings : ℕ := 40
  
  ∃ violins : ℕ,
    violins * violinStrings + ukuleles * ukuleleStrings + guitars * guitarStrings = totalStrings ∧
    violins = 2 :=
by sorry

end NUMINAMATH_CALUDE_francisFamily_violins_l2953_295376


namespace NUMINAMATH_CALUDE_additional_charge_correct_l2953_295382

/-- The charge for each additional 1/5 of a mile in a taxi ride -/
def additional_charge : ℝ := 0.40

/-- The initial charge for the first 1/5 of a mile -/
def initial_charge : ℝ := 2.50

/-- The total distance of the ride in miles -/
def total_distance : ℝ := 8

/-- The total charge for the ride -/
def total_charge : ℝ := 18.10

/-- Theorem stating that the additional charge is correct given the conditions -/
theorem additional_charge_correct :
  initial_charge + (total_distance - 1/5) / (1/5) * additional_charge = total_charge := by
  sorry

end NUMINAMATH_CALUDE_additional_charge_correct_l2953_295382


namespace NUMINAMATH_CALUDE_calculate_principal_l2953_295380

/-- Given simple interest, rate, and time, calculate the principal sum -/
theorem calculate_principal (simple_interest rate time : ℝ) : 
  simple_interest = 16065 * rate * time / 100 →
  rate = 5 →
  time = 5 →
  simple_interest = 4016.25 := by
  sorry

#check calculate_principal

end NUMINAMATH_CALUDE_calculate_principal_l2953_295380


namespace NUMINAMATH_CALUDE_james_writing_speed_l2953_295340

/-- James writes some pages an hour. -/
def pages_per_hour : ℝ := sorry

/-- James writes 5 pages a day to 2 different people. -/
def pages_per_day : ℝ := 5 * 2

/-- James spends 7 hours a week writing. -/
def hours_per_week : ℝ := 7

/-- The number of days in a week. -/
def days_per_week : ℝ := 7

theorem james_writing_speed :
  pages_per_hour = 10 :=
sorry

end NUMINAMATH_CALUDE_james_writing_speed_l2953_295340


namespace NUMINAMATH_CALUDE_a_profit_share_is_3750_l2953_295343

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_profit_share (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  (investment_a / (investment_a + investment_b + investment_c)) * total_profit

/-- Theorem: Given the investments and total profit, A's share of the profit is 3750 -/
theorem a_profit_share_is_3750 
  (investment_a : ℚ) 
  (investment_b : ℚ) 
  (investment_c : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 12500) :
  calculate_profit_share investment_a investment_b investment_c total_profit = 3750 := by
  sorry

#eval calculate_profit_share 6300 4200 10500 12500

end NUMINAMATH_CALUDE_a_profit_share_is_3750_l2953_295343


namespace NUMINAMATH_CALUDE_cats_owners_percentage_l2953_295313

/-- The percentage of students who own cats, given 75 out of 450 students own cats. -/
def percentage_cats_owners : ℚ :=
  75 / 450 * 100

/-- Theorem: The percentage of students who own cats is 16.6% (recurring). -/
theorem cats_owners_percentage :
  percentage_cats_owners = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cats_owners_percentage_l2953_295313


namespace NUMINAMATH_CALUDE_exists_large_configuration_l2953_295387

/-- A configuration in the plane is a finite set of points where each point
    has at least k other points at a distance of exactly 1 unit. -/
def IsConfiguration (S : Set (ℝ × ℝ)) (k : ℕ) : Prop :=
  S.Finite ∧ 
  ∀ P ∈ S, ∃ T ⊆ S, T.ncard ≥ k ∧ ∀ Q ∈ T, Q ≠ P ∧ dist P Q = 1

/-- There exists a configuration of 3^1000 points where each point
    has at least 2000 other points at a distance of 1 unit. -/
theorem exists_large_configuration :
  ∃ S : Set (ℝ × ℝ), IsConfiguration S 2000 ∧ S.ncard = 3^1000 := by
  sorry


end NUMINAMATH_CALUDE_exists_large_configuration_l2953_295387


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2953_295360

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem max_rectangle_area (length width : ℕ) :
  length + width = 25 →
  isPrime length →
  isComposite width →
  ∀ l w : ℕ, l + w = 25 → isPrime l → isComposite w → 
    rectangle_area length width ≥ rectangle_area l w →
  rectangle_area length width = 156 :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2953_295360


namespace NUMINAMATH_CALUDE_sum_of_qp_is_zero_l2953_295312

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x

def evaluation_points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_is_zero :
  (evaluation_points.map (λ x => q (p x))).sum = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_qp_is_zero_l2953_295312


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l2953_295358

/-- Two vectors are collinear in opposite directions if one is a negative scalar multiple of the other -/
def collinear_opposite (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, -4)
  let b : ℝ × ℝ := (-1, m + 3)
  collinear_opposite a b → m = 1 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l2953_295358


namespace NUMINAMATH_CALUDE_refrigerator_profit_theorem_l2953_295342

/-- Represents the financial details of a refrigerator sale --/
structure RefrigeratorSale where
  costPrice : ℝ
  markedPrice : ℝ
  discountPercentage : ℝ

/-- Calculates the profit from a refrigerator sale --/
def calculateProfit (sale : RefrigeratorSale) : ℝ :=
  sale.markedPrice * (1 - sale.discountPercentage) - sale.costPrice

/-- Theorem stating the profit for a specific refrigerator sale scenario --/
theorem refrigerator_profit_theorem (sale : RefrigeratorSale) 
  (h1 : sale.costPrice = 2000)
  (h2 : sale.markedPrice = 2750)
  (h3 : sale.discountPercentage = 0.15) :
  calculateProfit sale = 337.5 := by
  sorry

#eval calculateProfit { costPrice := 2000, markedPrice := 2750, discountPercentage := 0.15 }

end NUMINAMATH_CALUDE_refrigerator_profit_theorem_l2953_295342


namespace NUMINAMATH_CALUDE_first_part_value_l2953_295302

theorem first_part_value (x y : ℝ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_part_value_l2953_295302


namespace NUMINAMATH_CALUDE_pen_cost_l2953_295365

theorem pen_cost (x y : ℕ) (h1 : 5 * x + 4 * y = 345) (h2 : 3 * x + 6 * y = 285) : x = 52 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l2953_295365


namespace NUMINAMATH_CALUDE_g_equality_l2953_295319

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -2*x^5 + 3*x^4 - 11*x^3 + x^2 + 5*x - 5

-- State the theorem
theorem g_equality (x : ℝ) : 2*x^5 + 4*x^3 - 5*x + 3 + g x = 3*x^4 - 7*x^3 + x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_g_equality_l2953_295319


namespace NUMINAMATH_CALUDE_billys_weekend_activities_l2953_295339

/-- Billy's weekend activities theorem -/
theorem billys_weekend_activities :
  -- Define the given conditions
  let free_time_per_day : ℕ := 8
  let weekend_days : ℕ := 2
  let pages_per_hour : ℕ := 60
  let pages_per_book : ℕ := 80
  let books_read : ℕ := 3

  -- Calculate total free time
  let total_free_time : ℕ := free_time_per_day * weekend_days

  -- Calculate total pages read
  let total_pages_read : ℕ := pages_per_book * books_read

  -- Calculate time spent reading
  let reading_time : ℕ := total_pages_read / pages_per_hour

  -- Calculate time spent playing video games
  let gaming_time : ℕ := total_free_time - reading_time

  -- Calculate percentage of time spent playing video games
  let gaming_percentage : ℚ := (gaming_time : ℚ) / (total_free_time : ℚ) * 100

  -- Prove that Billy spends 75% of his time playing video games
  gaming_percentage = 75 := by sorry

end NUMINAMATH_CALUDE_billys_weekend_activities_l2953_295339


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l2953_295350

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_go_if_rain : ℝ := 0.3
def prob_go_if_sunny : ℝ := 0.9
def prob_remember : ℝ := 0.9

-- Define the theorem
theorem sheila_picnic_probability :
  prob_rain * prob_go_if_rain * prob_remember +
  prob_sunny * prob_go_if_sunny * prob_remember = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_sheila_picnic_probability_l2953_295350


namespace NUMINAMATH_CALUDE_herd_division_l2953_295309

theorem herd_division (total : ℕ) 
  (h1 : (1 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + (1 : ℚ) / 9 * total + 12 = total) : 
  total = 54 := by
  sorry

end NUMINAMATH_CALUDE_herd_division_l2953_295309


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l2953_295348

/-- Represents the exchange of US dollars to Canadian dollars and subsequent spending -/
def exchange_and_spend (d : ℕ) : Prop :=
  (8 * d) / 5 - 75 = d

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem representing the problem -/
theorem isabella_currency_exchange :
  ∃ d : ℕ, exchange_and_spend d ∧ d = 125 ∧ sum_of_digits d = 8 :=
sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l2953_295348


namespace NUMINAMATH_CALUDE_min_sum_of_product_1716_l2953_295391

theorem min_sum_of_product_1716 (a b c : ℕ+) (h : a * b * c = 1716) :
  ∃ (x y z : ℕ+), x * y * z = 1716 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 31 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1716_l2953_295391


namespace NUMINAMATH_CALUDE_min_odd_integers_l2953_295316

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 28)
  (sum2 : a + b + c + d = 46)
  (sum3 : a + b + c + d + e + f = 66) :
  ∃ (a' b' c' d' e' f' : ℤ), 
    (a' + b' = 28) ∧ 
    (a' + b' + c' + d' = 46) ∧ 
    (a' + b' + c' + d' + e' + f' = 66) ∧
    (Even a') ∧ (Even b') ∧ (Even c') ∧ (Even d') ∧ (Even e') ∧ (Even f') :=
by
  sorry

end NUMINAMATH_CALUDE_min_odd_integers_l2953_295316


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2953_295371

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h : i * i = -1) :
  i / z = 1 + i → z = (1 + i) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2953_295371


namespace NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l2953_295304

/-- Parabola defined by the equation 8y = (x-3)^2 -/
def parabola (x y : ℝ) : Prop := 8 * y = (x - 3)^2

/-- Point M -/
def M : ℝ × ℝ := (0, -2)

/-- Tangent line equation -/
def is_tangent_line (m b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ 
    (∀ x y, y = m * x + b ↔ (x = x₀ ∧ y = y₀ ∨ (y - y₀) = (x - x₀) * ((x₀ - 3) / 4)))

/-- Theorem stating the properties of the tangent lines and the triangle -/
theorem parabola_tangents_and_triangle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Tangent lines equations
    is_tangent_line (-2) (-2) ∧
    is_tangent_line (1/2) (-2) ∧
    -- Points A and B are on the parabola
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    -- A and B are on the tangent lines
    y₁ = -2 * x₁ - 2 ∧
    y₂ = 1/2 * x₂ - 2 ∧
    -- Tangent lines are perpendicular
    (-2) * (1/2) = -1 ∧
    -- Area of triangle ABM
    abs ((x₁ - 0) * (y₂ - (-2)) - (x₂ - 0) * (y₁ - (-2))) / 2 = 125/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l2953_295304


namespace NUMINAMATH_CALUDE_roots_product_theorem_l2953_295325

-- Define the polynomial f(x) = x⁶ + x³ + 1
def f (x : ℂ) : ℂ := x^6 + x^3 + 1

-- Define the function g(x) = x² + 1
def g (x : ℂ) : ℂ := x^2 + 1

-- State the theorem
theorem roots_product_theorem : 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ), 
    (∀ x, f x = (x - x₁) * (x - x₂) * (x - x₃) * (x - x₄) * (x - x₅) * (x - x₆)) →
    g x₁ * g x₂ * g x₃ * g x₄ * g x₅ * g x₆ = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_theorem_l2953_295325


namespace NUMINAMATH_CALUDE_sine_of_angle_through_point_l2953_295321

theorem sine_of_angle_through_point (α : Real) :
  let P : Real × Real := (Real.cos (3 * Real.pi / 4), Real.sin (3 * Real.pi / 4))
  (∃ k : Real, k > 0 ∧ (k * Real.cos α = P.1) ∧ (k * Real.sin α = P.2)) →
  Real.sin α = Real.sqrt 2 / 2 ∨ Real.sin α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_through_point_l2953_295321


namespace NUMINAMATH_CALUDE_sandra_theorem_l2953_295323

def sandra_problem (savings : ℚ) (mother_gift : ℚ) (father_gift_multiplier : ℚ)
  (candy_cost : ℚ) (jelly_bean_cost : ℚ) (candy_count : ℕ) (jelly_bean_count : ℕ) : Prop :=
  let total_money := savings + mother_gift + (father_gift_multiplier * mother_gift)
  let total_cost := (candy_cost * candy_count) + (jelly_bean_cost * jelly_bean_count)
  let remaining_money := total_money - total_cost
  remaining_money = 11

theorem sandra_theorem :
  sandra_problem 10 4 2 (1/2) (1/5) 14 20 := by
  sorry

end NUMINAMATH_CALUDE_sandra_theorem_l2953_295323


namespace NUMINAMATH_CALUDE_hurdle_distance_l2953_295301

theorem hurdle_distance (total_distance : ℕ) (num_hurdles : ℕ) (start_distance : ℕ) (end_distance : ℕ) 
  (h1 : total_distance = 600)
  (h2 : num_hurdles = 12)
  (h3 : start_distance = 50)
  (h4 : end_distance = 55) :
  ∃ d : ℕ, d = 45 ∧ total_distance = start_distance + (num_hurdles - 1) * d + end_distance :=
by sorry

end NUMINAMATH_CALUDE_hurdle_distance_l2953_295301


namespace NUMINAMATH_CALUDE_minimal_divisible_number_l2953_295364

theorem minimal_divisible_number : ∃! n : ℕ,
  2007000 ≤ n ∧ n < 2008000 ∧
  n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧
  (∀ m : ℕ, 2007000 ≤ m ∧ m < n → (m % 3 ≠ 0 ∨ m % 5 ≠ 0 ∨ m % 7 ≠ 0)) ∧
  n = 2007075 :=
sorry

end NUMINAMATH_CALUDE_minimal_divisible_number_l2953_295364


namespace NUMINAMATH_CALUDE_cards_distribution_l2953_295396

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 48) (h2 : num_people = 7) :
  let cards_per_person := total_cards / num_people
  let remaining_cards := total_cards % num_people
  num_people - remaining_cards = 1 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2953_295396


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_six_l2953_295385

theorem sin_thirteen_pi_six : Real.sin (13 * π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_six_l2953_295385


namespace NUMINAMATH_CALUDE_circulation_year_in_range_l2953_295389

/-- Represents the circulation data for magazine P --/
structure CirculationData where
  avg_1962_1970 : ℝ  -- Average yearly circulation for 1962-1970
  circ_1961 : ℝ      -- Circulation in 1961
  circ_certain_year : ℝ -- Circulation in the certain year

/-- The conditions given in the problem --/
def problem_conditions (data : CirculationData) : Prop :=
  data.circ_certain_year = 4 * data.avg_1962_1970 ∧
  data.circ_certain_year / (data.circ_1961 + 9 * data.avg_1962_1970) = 2 / 7

/-- The theorem statement --/
theorem circulation_year_in_range (data : CirculationData) 
  (h : problem_conditions data) : 
  ∃ (year : ℕ), 1962 ≤ year ∧ year ≤ 1970 ∧ 
  (∀ (y : ℕ), y = year ↔ data.circ_certain_year = 4 * data.avg_1962_1970) ∧
  (¬ ∃! (y : ℕ), 1962 ≤ y ∧ y ≤ 1970 ∧ data.circ_certain_year = 4 * data.avg_1962_1970) :=
sorry

end NUMINAMATH_CALUDE_circulation_year_in_range_l2953_295389


namespace NUMINAMATH_CALUDE_max_square_vertices_on_ellipse_l2953_295349

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  a : ℝ  -- Semi-major axis length

/-- A square centered at a given point -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- The number of vertices of a square that lie on an ellipse -/
def vertices_on_ellipse (E : Ellipse) (S : Square) : ℕ :=
  sorry

theorem max_square_vertices_on_ellipse (E : Ellipse) (S : Square) :
  S.center = E.F₁ → (∃ n : ℕ, vertices_on_ellipse E S = n ∧ ∀ m : ℕ, vertices_on_ellipse E S ≤ m) → vertices_on_ellipse E S = 1 :=
sorry

end NUMINAMATH_CALUDE_max_square_vertices_on_ellipse_l2953_295349


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l2953_295394

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l2953_295394


namespace NUMINAMATH_CALUDE_area_of_awesome_points_l2953_295327

/-- A right triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 3 ∧ b = 4 ∧ c = 5

/-- A point is awesome if it's the center of a parallelogram with vertices on the triangle's boundary -/
def is_awesome (T : RightTriangle) (P : ℝ × ℝ) : Prop := sorry

/-- The set of awesome points -/
def awesome_points (T : RightTriangle) : Set (ℝ × ℝ) :=
  {P | is_awesome T P}

/-- The area of a set of points in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem: The area of awesome points in the 3-4-5 right triangle is 3/2 -/
theorem area_of_awesome_points (T : RightTriangle) :
  area (awesome_points T) = 3/2 := by sorry

end NUMINAMATH_CALUDE_area_of_awesome_points_l2953_295327


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l2953_295374

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_tangent_line (a b : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = -3 ∧ b = 4) ∧
  (∃ k m : ℝ, k * 0 + m = f (-3) 4 0 ∧ k = f' (-3) 4 0 ∧ k = 12 ∧ m = 8) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l2953_295374


namespace NUMINAMATH_CALUDE_sequence_conclusions_l2953_295308

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1)^2 - a (n + 1) = a n)

theorem sequence_conclusions (a : ℕ → ℝ) (h : sequence_property a) :
  (∀ n ≥ 2, a n > 1) ∧
  ((0 < a 1 ∧ a 1 < 2) → (∀ n, a n < a (n + 1))) ∧
  (a 1 > 2 → ∀ n ≥ 2, 2 < a n ∧ a n < a 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_conclusions_l2953_295308


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2953_295359

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x + 1, -2)
  let b : ℝ × ℝ := (-2*x, 3)
  parallel a b → x = 3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2953_295359


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l2953_295351

theorem odd_prime_square_root (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  let k := (p + 1)^2 / 4
  ∃ n : ℕ, n > 0 ∧ n^2 = k - p * k := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l2953_295351


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l2953_295324

-- Define the redistribution function
def redistribute (a j t : ℚ) : ℚ × ℚ × ℚ :=
  let (a1, j1, t1) := (a - (j + t), 2*j, 2*t)
  let (a2, j2, t2) := (2*a1, j1 - (a1 + t1), 2*t1)
  (2*a2, 2*j2, t2 - (a2 + j2))

-- Theorem statement
theorem total_money_after_redistribution :
  ∀ a j : ℚ,
  let (a_final, j_final, t_final) := redistribute a j 24
  t_final = 24 →
  a_final + j_final + t_final = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_money_after_redistribution_l2953_295324


namespace NUMINAMATH_CALUDE_roses_remaining_is_nine_l2953_295315

/-- Represents the number of roses in a dozen -/
def dozen : ℕ := 12

/-- Calculates the number of unwilted roses remaining after a series of events -/
def remaining_roses (initial_dozens : ℕ) (traded_dozens : ℕ) : ℕ :=
  let initial_roses := initial_dozens * dozen
  let after_trade := initial_roses + traded_dozens * dozen
  let after_first_wilt := after_trade / 2
  after_first_wilt / 2

/-- Proves that given the initial conditions and subsequent events, 
    the number of unwilted roses remaining is 9 -/
theorem roses_remaining_is_nine :
  remaining_roses 2 1 = 9 := by
  sorry

#eval remaining_roses 2 1

end NUMINAMATH_CALUDE_roses_remaining_is_nine_l2953_295315


namespace NUMINAMATH_CALUDE_sales_composition_l2953_295310

/-- The percentage of sales that are not pens, pencils, or erasers -/
def other_sales_percentage (pen_sales pencil_sales eraser_sales : ℝ) : ℝ :=
  100 - (pen_sales + pencil_sales + eraser_sales)

/-- Theorem stating that the percentage of sales not consisting of pens, pencils, or erasers is 25% -/
theorem sales_composition 
  (pen_sales : ℝ) 
  (pencil_sales : ℝ) 
  (eraser_sales : ℝ) 
  (h1 : pen_sales = 25)
  (h2 : pencil_sales = 30)
  (h3 : eraser_sales = 20) :
  other_sales_percentage pen_sales pencil_sales eraser_sales = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_composition_l2953_295310


namespace NUMINAMATH_CALUDE_classroom_ratio_l2953_295395

theorem classroom_ratio (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 30 → boys = 20 → girls = total_students - boys → 
  (girls : ℚ) / (boys : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_classroom_ratio_l2953_295395


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_indeterminate_l2953_295363

theorem maggie_bouncy_balls_indeterminate 
  (yellow_packs : ℝ) 
  (green_packs_given : ℝ) 
  (balls_per_pack : ℝ) 
  (total_kept : ℕ) 
  (h1 : yellow_packs = 8.0)
  (h2 : green_packs_given = 4.0)
  (h3 : balls_per_pack = 10.0)
  (h4 : total_kept = 80)
  (h5 : yellow_packs * balls_per_pack = total_kept) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (yellow_packs * balls_per_pack - green_packs_given * balls_per_pack + x * balls_per_pack = total_kept) ∧
    (yellow_packs * balls_per_pack - green_packs_given * balls_per_pack + y * balls_per_pack = total_kept) :=
by sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_indeterminate_l2953_295363


namespace NUMINAMATH_CALUDE_jack_barbecue_sauce_l2953_295335

/-- The amount of vinegar used in Jack's barbecue sauce recipe -/
def vinegar_amount : ℚ → Prop :=
  fun v =>
    let ketchup : ℚ := 3
    let honey : ℚ := 1
    let burger_sauce : ℚ := 1/4
    let sandwich_sauce : ℚ := 1/6
    let num_burgers : ℚ := 8
    let num_sandwiches : ℚ := 18
    let total_sauce : ℚ := num_burgers * burger_sauce + num_sandwiches * sandwich_sauce
    ketchup + v + honey = total_sauce

theorem jack_barbecue_sauce :
  vinegar_amount 1 := by sorry

end NUMINAMATH_CALUDE_jack_barbecue_sauce_l2953_295335


namespace NUMINAMATH_CALUDE_fair_haired_men_nonmanagerial_percentage_l2953_295373

/-- Represents the hair color distribution in the company -/
structure HairColorDistribution where
  fair : ℝ
  dark : ℝ
  red : ℝ
  ratio_fair_dark_red : fair / dark = 4 / 9 ∧ fair / red = 4 / 7

/-- Represents the gender distribution in the company -/
structure GenderDistribution where
  women : ℝ
  men : ℝ
  ratio_women_men : women / men = 3 / 5

/-- Represents the position distribution in the company -/
structure PositionDistribution where
  managerial : ℝ
  nonmanagerial : ℝ
  ratio_managerial_nonmanagerial : managerial / nonmanagerial = 1 / 4

/-- Represents the distribution of fair-haired employees -/
structure FairHairedDistribution where
  women_percentage : ℝ
  women_percentage_is_40 : women_percentage = 0.4
  women_managerial_percentage : ℝ
  women_managerial_percentage_is_60 : women_managerial_percentage = 0.6
  men_nonmanagerial_percentage : ℝ
  men_nonmanagerial_percentage_is_70 : men_nonmanagerial_percentage = 0.7

/-- Theorem: The percentage of fair-haired men in non-managerial positions is 42% -/
theorem fair_haired_men_nonmanagerial_percentage
  (hair : HairColorDistribution)
  (gender : GenderDistribution)
  (position : PositionDistribution)
  (fair_haired : FairHairedDistribution) :
  (1 - fair_haired.women_percentage) * fair_haired.men_nonmanagerial_percentage = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_fair_haired_men_nonmanagerial_percentage_l2953_295373


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2953_295334

theorem cost_price_percentage (selling_price cost_price : ℝ) :
  selling_price > 0 →
  cost_price > 0 →
  selling_price - cost_price = (1 / 3) * cost_price →
  (cost_price / selling_price) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l2953_295334


namespace NUMINAMATH_CALUDE_remainder_zero_prime_l2953_295320

theorem remainder_zero_prime (N : ℕ) (h_odd : Odd N) :
  (∀ i j, 2 ≤ i ∧ i < j ∧ j ≤ 1000 → N % i ≠ N % j) →
  (∃ k, 2 ≤ k ∧ k ≤ 1000 ∧ N % k = 0) →
  ∃ p, Prime p ∧ 500 < p ∧ p < 1000 ∧ N % p = 0 :=
sorry

end NUMINAMATH_CALUDE_remainder_zero_prime_l2953_295320


namespace NUMINAMATH_CALUDE_horner_rule_v4_l2953_295361

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_rule_v4 :
  horner_v4 2 = 80 :=
by sorry

#eval horner_v4 2
#eval horner_polynomial 2

end NUMINAMATH_CALUDE_horner_rule_v4_l2953_295361


namespace NUMINAMATH_CALUDE_a_10_value_l2953_295318

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 = 2 → a 3 = 4 → a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l2953_295318


namespace NUMINAMATH_CALUDE_factorization_proof_l2953_295303

theorem factorization_proof (x y : ℝ) : 2 * x^3 - 18 * x * y^2 = 2 * x * (x + 3 * y) * (x - 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2953_295303


namespace NUMINAMATH_CALUDE_semicircle_problem_l2953_295326

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 18 → N = 19 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_problem_l2953_295326


namespace NUMINAMATH_CALUDE_line_intercepts_l2953_295390

/-- Given a line with equation x/4 - y/3 = 1, prove that its x-intercept is 4 and y-intercept is -3 -/
theorem line_intercepts (x y : ℝ) :
  x/4 - y/3 = 1 → (x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l2953_295390


namespace NUMINAMATH_CALUDE_cyclist_speed_l2953_295362

theorem cyclist_speed (distance : ℝ) (time_difference : ℝ) : 
  distance = 96 →
  time_difference = 16 →
  ∃ (speed : ℝ), 
    speed > 0 ∧
    distance / (speed - 4) = distance / (1.5 * speed) + time_difference ∧
    speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l2953_295362


namespace NUMINAMATH_CALUDE_soccer_goal_ratio_l2953_295356

/-- Prove the ratio of goals scored by The Spiders to The Kickers in the first period -/
theorem soccer_goal_ratio :
  let kickers_first := 2
  let kickers_second := 2 * kickers_first
  let spiders_second := 2 * kickers_second
  let total_goals := 15
  let spiders_first := total_goals - (kickers_first + kickers_second + spiders_second)
  (spiders_first : ℚ) / kickers_first = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goal_ratio_l2953_295356


namespace NUMINAMATH_CALUDE_painter_work_days_l2953_295332

/-- Represents the number of work-days required for a given number of painters to complete a job -/
def work_days (painters : ℕ) (days : ℚ) : Prop :=
  painters * days = 6 * 2

theorem painter_work_days :
  work_days 6 2 → work_days 4 3 := by
  sorry

end NUMINAMATH_CALUDE_painter_work_days_l2953_295332


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2953_295341

theorem quadratic_inequality_range (x : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc 2 4 ∧ a * x^2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Ioi (-1) ∪ Set.Iio (3/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2953_295341


namespace NUMINAMATH_CALUDE_infinitely_many_rational_pairs_l2953_295330

theorem infinitely_many_rational_pairs :
  ∃ f : ℚ → ℚ × ℚ,
    (∀ k : ℚ, k > 1 → 
      let (x, y) := f k
      (x > 0 ∧ y > 0) ∧
      (x ≠ y) ∧
      (∃ r : ℚ, r^2 = x^2 + y^3) ∧
      (∃ s : ℚ, s^2 = x^3 + y^2)) ∧
    (∀ k₁ k₂ : ℚ, k₁ > 1 → k₂ > k₁ → f k₁ ≠ f k₂) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_rational_pairs_l2953_295330


namespace NUMINAMATH_CALUDE_positive_difference_problem_l2953_295379

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0))

theorem positive_difference_problem (x : ℕ) 
  (h1 : (45 + x) / 2 = 50) 
  (h2 : is_prime x) : 
  Int.natAbs (x - 45) = 8 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_problem_l2953_295379


namespace NUMINAMATH_CALUDE_classroom_books_l2953_295329

/-- The total number of books in a classroom -/
def total_books (num_children : ℕ) (books_per_child : ℕ) (teacher_books : ℕ) : ℕ :=
  num_children * books_per_child + teacher_books

/-- Theorem: The total number of books in the classroom is 202 -/
theorem classroom_books : 
  total_books 15 12 22 = 202 := by
  sorry

end NUMINAMATH_CALUDE_classroom_books_l2953_295329


namespace NUMINAMATH_CALUDE_problem_solution_l2953_295346

/-- The condition p: x^2 - 5ax + 4a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0

/-- The condition q: 3 < x ≤ 4 -/
def q (x : ℝ) : Prop := 3 < x ∧ x ≤ 4

theorem problem_solution (a : ℝ) (h : a > 0) :
  (a = 1 → ∀ x, p x a ∧ q x ↔ 3 < x ∧ x < 4) ∧
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) ↔ 1 < a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2953_295346


namespace NUMINAMATH_CALUDE_min_attempts_correct_l2953_295300

/-- Represents the minimum number of attempts to make a lamp work given a set of batteries. -/
def min_attempts (total : ℕ) (good : ℕ) (bad : ℕ) : ℕ :=
  if total = 2 * good - 1 then good else good - 1

theorem min_attempts_correct (n : ℕ) (h : n > 2) :
  (min_attempts (2 * n + 1) (n + 1) n = n + 1) ∧
  (min_attempts (2 * n) n n = n) :=
by sorry

#check min_attempts_correct

end NUMINAMATH_CALUDE_min_attempts_correct_l2953_295300


namespace NUMINAMATH_CALUDE_expression_evaluation_l2953_295314

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2953_295314


namespace NUMINAMATH_CALUDE_pastry_sets_problem_l2953_295347

theorem pastry_sets_problem (N : ℕ) 
  (h1 : ∃ (x y : ℕ), x + y = N ∧ 3*x + 5*y = 25)
  (h2 : ∃ (a b : ℕ), a + b = N ∧ 3*a + 5*b = 35) : 
  N = 7 := by
sorry

end NUMINAMATH_CALUDE_pastry_sets_problem_l2953_295347


namespace NUMINAMATH_CALUDE_power_equality_l2953_295372

theorem power_equality (m n : ℤ) (P Q : ℝ) (h1 : P = 2^m) (h2 : Q = 3^n) :
  P^(2*n) * Q^m = 12^(m*n) := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2953_295372


namespace NUMINAMATH_CALUDE_tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3_l2953_295336

theorem tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3 (θ : Real) 
  (h : Real.tan θ = Real.sqrt 3) : 
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3_l2953_295336


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2953_295354

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((246835 + y) % 169 = 0 ∧ (246835 + y) % 289 = 0)) ∧ 
  ((246835 + x) % 169 = 0 ∧ (246835 + x) % 289 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2953_295354


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2953_295377

/-- The equation of a hyperbola sharing foci with an ellipse and passing through a point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    (x - 2)^2 / a^2 - (y - 1)^2 / b^2 = 1) →
  x^2 / 2 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2953_295377


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2953_295378

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.5875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2953_295378


namespace NUMINAMATH_CALUDE_playground_length_l2953_295305

theorem playground_length (garden_width garden_perimeter playground_width : ℝ) 
  (hw : garden_width = 24)
  (hp : garden_perimeter = 64)
  (pw : playground_width = 12)
  (area_eq : garden_width * ((garden_perimeter / 2) - garden_width) = playground_width * (garden_width * ((garden_perimeter / 2) - garden_width) / playground_width)) :
  (garden_width * ((garden_perimeter / 2) - garden_width) / playground_width) = 16 := by
sorry

end NUMINAMATH_CALUDE_playground_length_l2953_295305


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2953_295322

theorem compound_interest_problem (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8820)
  (h2 : P * (1 + r)^3 = 9261) : 
  P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2953_295322


namespace NUMINAMATH_CALUDE_shanna_garden_theorem_l2953_295399

/-- Calculates the number of vegetables per plant given the initial number of plants,
    the number of plants that died, and the total number of vegetables harvested. -/
def vegetables_per_plant (tomato_plants eggplant_plants pepper_plants : ℕ)
                         (dead_tomato_plants dead_pepper_plants : ℕ)
                         (total_vegetables : ℕ) : ℕ :=
  let surviving_plants := (tomato_plants - dead_tomato_plants) +
                          eggplant_plants +
                          (pepper_plants - dead_pepper_plants)
  total_vegetables / surviving_plants

/-- Theorem stating that given Shanna's garden conditions, each remaining plant gave 7 vegetables. -/
theorem shanna_garden_theorem :
  vegetables_per_plant 6 2 4 3 1 56 = 7 :=
by sorry

end NUMINAMATH_CALUDE_shanna_garden_theorem_l2953_295399


namespace NUMINAMATH_CALUDE_sphere_volume_condition_l2953_295337

theorem sphere_volume_condition (R V : ℝ) : 
  (V = (4 / 3) * π * R^3) → (R > Real.sqrt 10 → V > 36 * π) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_condition_l2953_295337


namespace NUMINAMATH_CALUDE_solve_for_y_l2953_295366

theorem solve_for_y (x y : ℝ) 
  (h1 : x = 151) 
  (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3423000) : 
  y = 3423000 / 3375001 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l2953_295366


namespace NUMINAMATH_CALUDE_prob_same_group_is_one_third_l2953_295352

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The set of all possible outcomes when two students choose interest groups -/
def total_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_groups) (Finset.range num_groups)

/-- The set of outcomes where both students choose the same group -/
def same_group_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 = p.2) total_outcomes

/-- The probability of two students choosing the same interest group -/
def prob_same_group : ℚ :=
  (same_group_outcomes.card : ℚ) / (total_outcomes.card : ℚ)

theorem prob_same_group_is_one_third :
  prob_same_group = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_same_group_is_one_third_l2953_295352


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_2_l2953_295397

theorem least_prime_factor_of_5_4_minus_5_2 :
  Nat.minFac (5^4 - 5^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_2_l2953_295397


namespace NUMINAMATH_CALUDE_ball_attendees_l2953_295369

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l2953_295369


namespace NUMINAMATH_CALUDE_officer_selection_ways_l2953_295331

def club_members : ℕ := 12
def officer_positions : ℕ := 5

theorem officer_selection_ways :
  (club_members.factorial) / ((club_members - officer_positions).factorial) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_ways_l2953_295331


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l2953_295317

theorem no_solution_quadratic_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l2953_295317


namespace NUMINAMATH_CALUDE_min_integral_abs_exp_minus_a_l2953_295381

theorem min_integral_abs_exp_minus_a :
  let f (a : ℝ) := ∫ x in (0 : ℝ)..1, |Real.exp (-x) - a|
  ∃ m : ℝ, (∀ a : ℝ, f a ≥ m) ∧ (∃ a : ℝ, f a = m) ∧ m = 1 - 2 * Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_min_integral_abs_exp_minus_a_l2953_295381


namespace NUMINAMATH_CALUDE_expression_simplification_l2953_295357

theorem expression_simplification (a b : ℤ) (h : b = a + 1) (ha : a = 2015) :
  (a^4 - 3*a^3*b + 3*a^2*b^2 - b^4 + a) / (a*b) = -(a-1)^2 / a^3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2953_295357


namespace NUMINAMATH_CALUDE_f_symmetry_l2953_295353

/-- Given a function f(x) = x^5 + ax^3 + bx, if f(-2) = 10, then f(2) = -10 -/
theorem f_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 + a*x^3 + b*x
  f (-2) = 10 → f 2 = -10 := by sorry

end NUMINAMATH_CALUDE_f_symmetry_l2953_295353
