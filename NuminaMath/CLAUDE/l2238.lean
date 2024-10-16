import Mathlib

namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_five_digit_palindrome_product_result_171_l2238_223806

/-- A function to check if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function to check if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_not_five_digit_palindrome_product :
  ∀ n : ℕ, isThreeDigitPalindrome n → n < 171 → isFiveDigitPalindrome (n * 111) :=
by sorry

/-- The result theorem -/
theorem result_171 :
  isThreeDigitPalindrome 171 ∧ ¬ isFiveDigitPalindrome (171 * 111) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_five_digit_palindrome_product_result_171_l2238_223806


namespace NUMINAMATH_CALUDE_expression_evaluation_l2238_223843

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 72 / 35 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2238_223843


namespace NUMINAMATH_CALUDE_square_land_side_length_l2238_223853

theorem square_land_side_length (area : ℝ) (is_square : Bool) : 
  area = 400 ∧ is_square = true → ∃ (side : ℝ), side * side = area ∧ side = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l2238_223853


namespace NUMINAMATH_CALUDE_sequence_properties_l2238_223827

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

axiom a_def (n : ℕ) : n ≠ 0 → sequence_a n = 2 * sum_S n - 1

def sequence_b (n : ℕ) : ℝ := (2 * n + 1) * sequence_a n

def sum_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n ≠ 0 → sequence_a n = (-1)^(n-1)) ∧
  (∀ n : ℕ, sum_T n = 1 - (n + 1) * (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2238_223827


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2238_223862

theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = 4 / 5) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 41 / 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2238_223862


namespace NUMINAMATH_CALUDE_minimum_dimes_needed_l2238_223892

def shoe_cost : ℚ := 45.50
def five_dollar_bills : ℕ := 4
def one_dollar_coins : ℕ := 10
def dime_value : ℚ := 0.10

theorem minimum_dimes_needed (n : ℕ) : 
  (five_dollar_bills * 5 + one_dollar_coins * 1 + n * dime_value ≥ shoe_cost) →
  n ≥ 155 := by
  sorry

end NUMINAMATH_CALUDE_minimum_dimes_needed_l2238_223892


namespace NUMINAMATH_CALUDE_waterpark_total_cost_calculation_l2238_223826

def waterpark_total_cost (adult_price child_price teen_price : ℚ)
                         (num_adults num_children num_teens : ℕ)
                         (activity_discount coupon_discount : ℚ)
                         (soda_price : ℚ) (num_sodas : ℕ) : ℚ :=
  let base_cost := adult_price * num_adults + child_price * num_children + teen_price * num_teens
  let discounted_cost := base_cost * (1 - activity_discount) * (1 - coupon_discount)
  let soda_cost := soda_price * num_sodas
  discounted_cost + soda_cost

theorem waterpark_total_cost_calculation :
  waterpark_total_cost 30 15 20 4 2 4 (1/10) (1/20) 5 5 = 221.65 := by
  sorry

end NUMINAMATH_CALUDE_waterpark_total_cost_calculation_l2238_223826


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2238_223888

theorem nested_fraction_equality : 
  1 + 4 / (5 + 6 / 7) = 69 / 41 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2238_223888


namespace NUMINAMATH_CALUDE_min_adventurers_l2238_223882

structure AdventurerGroup where
  R : Finset Nat  -- Adventurers with rubies
  E : Finset Nat  -- Adventurers with emeralds
  S : Finset Nat  -- Adventurers with sapphires
  D : Finset Nat  -- Adventurers with diamonds

def AdventurerGroup.valid (g : AdventurerGroup) : Prop :=
  (g.R.card = 13) ∧
  (g.E.card = 9) ∧
  (g.S.card = 15) ∧
  (g.D.card = 6) ∧
  (∀ x ∈ g.S, (x ∈ g.E ∨ x ∈ g.D) ∧ ¬(x ∈ g.E ∧ x ∈ g.D)) ∧
  (∀ x ∈ g.E, (x ∈ g.R ∨ x ∈ g.S) ∧ ¬(x ∈ g.R ∧ x ∈ g.S))

theorem min_adventurers (g : AdventurerGroup) (h : g.valid) :
  (g.R ∪ g.E ∪ g.S ∪ g.D).card = 22 :=
sorry

end NUMINAMATH_CALUDE_min_adventurers_l2238_223882


namespace NUMINAMATH_CALUDE_arcade_play_time_l2238_223832

def weekly_pay : ℕ := 100
def arcade_budget : ℕ := weekly_pay / 2
def food_cost : ℕ := 10
def token_budget : ℕ := arcade_budget - food_cost
def play_cost : ℕ := 8
def total_play_time : ℕ := 300

theorem arcade_play_time : 
  (token_budget / play_cost) * (total_play_time / (token_budget / play_cost)) = total_play_time :=
by sorry

end NUMINAMATH_CALUDE_arcade_play_time_l2238_223832


namespace NUMINAMATH_CALUDE_number_added_at_end_l2238_223857

theorem number_added_at_end (x : ℝ) : (26.3 * 12 * 20) / 3 + x = 2229 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_added_at_end_l2238_223857


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2238_223840

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ 3/2 ∧ k ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2238_223840


namespace NUMINAMATH_CALUDE_first_group_size_l2238_223898

/-- The number of men in the first group -/
def M : ℕ := 42

/-- The number of days the first group takes to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 27

/-- The number of days the second group takes to complete the work -/
def days_second_group : ℕ := 28

/-- The work done by a group is inversely proportional to the number of days they take -/
axiom work_inverse_proportion (men days : ℕ) : men * days = men_second_group * days_second_group

theorem first_group_size : M = 42 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l2238_223898


namespace NUMINAMATH_CALUDE_gondor_laptop_repair_fee_l2238_223803

/-- The amount Gondor earns from repairing a phone -/
def phone_repair_fee : ℝ := 10

/-- The number of phones Gondor repaired on Monday -/
def monday_phones : ℕ := 3

/-- The number of phones Gondor repaired on Tuesday -/
def tuesday_phones : ℕ := 5

/-- The number of laptops Gondor repaired on Wednesday -/
def wednesday_laptops : ℕ := 2

/-- The number of laptops Gondor repaired on Thursday -/
def thursday_laptops : ℕ := 4

/-- The total amount Gondor earned -/
def total_earnings : ℝ := 200

/-- The amount Gondor earns from repairing a laptop -/
def laptop_repair_fee : ℝ := 20

theorem gondor_laptop_repair_fee :
  laptop_repair_fee = 20 ∧
  (monday_phones + tuesday_phones) * phone_repair_fee +
  (wednesday_laptops + thursday_laptops) * laptop_repair_fee = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_gondor_laptop_repair_fee_l2238_223803


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l2238_223851

theorem circus_ticket_cost (total_cost : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℕ) :
  total_cost = 308 →
  num_tickets = 7 →
  cost_per_ticket * num_tickets = total_cost →
  cost_per_ticket = 44 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l2238_223851


namespace NUMINAMATH_CALUDE_fraction_equality_l2238_223823

theorem fraction_equality (a b : ℝ) (h : a ≠ b) :
  (a^2 - b^2) / (a - b)^2 = (a + b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2238_223823


namespace NUMINAMATH_CALUDE_computer_profit_percentage_l2238_223834

-- Define the computer's cost
variable (cost : ℝ)

-- Define the two selling prices
def selling_price_1 : ℝ := 2240
def selling_price_2 : ℝ := 2400

-- Define the profit percentages
def profit_percentage_1 : ℝ := 0.4  -- 40%
def profit_percentage_2 : ℝ := 0.5  -- 50%

-- Theorem statement
theorem computer_profit_percentage :
  (selling_price_2 - cost = profit_percentage_2 * cost) →
  (selling_price_1 - cost = profit_percentage_1 * cost) :=
by sorry

end NUMINAMATH_CALUDE_computer_profit_percentage_l2238_223834


namespace NUMINAMATH_CALUDE_marks_weekly_reading_time_l2238_223830

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Mark's current daily reading time in hours -/
def daily_reading_time : ℕ := 3

/-- Mark's planned weekly increase in reading time in hours -/
def weekly_increase : ℕ := 6

/-- Theorem: Mark's total weekly reading time after the increase will be 27 hours -/
theorem marks_weekly_reading_time :
  daily_reading_time * days_in_week + weekly_increase = 27 := by
  sorry

end NUMINAMATH_CALUDE_marks_weekly_reading_time_l2238_223830


namespace NUMINAMATH_CALUDE_amy_small_gardens_l2238_223824

def small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem amy_small_gardens :
  small_gardens 101 47 6 = 9 :=
by sorry

end NUMINAMATH_CALUDE_amy_small_gardens_l2238_223824


namespace NUMINAMATH_CALUDE_inequality_proof_l2238_223872

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2238_223872


namespace NUMINAMATH_CALUDE_natalia_crates_l2238_223801

theorem natalia_crates (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) (crate_capacity : ℕ) :
  novels = 145 →
  comics = 271 →
  documentaries = 419 →
  albums = 209 →
  crate_capacity = 9 →
  (novels + comics + documentaries + albums + crate_capacity - 1) / crate_capacity = 116 :=
by sorry

end NUMINAMATH_CALUDE_natalia_crates_l2238_223801


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l2238_223811

-- Define the circles
def circle_C1 (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y + 2)^2 = 9
def circle_C2 (m : ℝ) (x y : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 4

-- Define the condition for internal tangency
def internally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C1 m x y ∧ circle_C2 m x y

-- Theorem statement
theorem circles_internally_tangent :
  ∀ m : ℝ, internally_tangent m ↔ m = -2 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l2238_223811


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2238_223883

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2*a*x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2238_223883


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l2238_223819

theorem subtraction_of_decimals : 3.05 - 5.678 = -2.628 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l2238_223819


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2238_223849

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) : 
  i * i = -1 → 
  (a + i) / i = b + i → 
  a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l2238_223849


namespace NUMINAMATH_CALUDE_range_of_m_correct_l2238_223876

/-- The range of m satisfying the given conditions -/
def range_of_m : Set ℝ :=
  {m | m ≥ 3 ∨ (1 < m ∧ m ≤ 2)}

/-- Condition p: x^2 + mx + 1 = 0 has two distinct negative roots -/
def condition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
    x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Condition q: 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def condition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m_correct :
  ∀ m : ℝ, (condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m) ↔ m ∈ range_of_m :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_correct_l2238_223876


namespace NUMINAMATH_CALUDE_no_natural_function_satisfies_equation_l2238_223825

theorem no_natural_function_satisfies_equation :
  ¬ ∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_function_satisfies_equation_l2238_223825


namespace NUMINAMATH_CALUDE_circle_equation_polar_l2238_223828

/-- The equation of a circle in polar coordinates with center at (√2, π) passing through the pole -/
theorem circle_equation_polar (ρ θ : ℝ) : 
  (ρ = -2 * Real.sqrt 2 * Real.cos θ) ↔ 
  (∃ (x y : ℝ), 
    -- Convert polar to Cartesian coordinates
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
    -- Circle equation in Cartesian coordinates
    ((x + Real.sqrt 2)^2 + y^2 = 2) ∧
    -- Circle passes through the pole (origin in Cartesian)
    (∃ (θ₀ : ℝ), ρ * Real.cos θ₀ = 0 ∧ ρ * Real.sin θ₀ = 0)) := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_polar_l2238_223828


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l2238_223817

theorem sum_of_three_consecutive_even_numbers (m : ℤ) : 
  m % 2 = 0 → (m + (m + 2) + (m + 4)) = 3 * m + 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_even_numbers_l2238_223817


namespace NUMINAMATH_CALUDE_manager_chef_wage_difference_l2238_223854

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- Conditions for wages at Joe's Steakhouse -/
def validSteakhouseWages (w : SteakhouseWages) : Prop :=
  w.manager = 6.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.20

/-- Theorem stating the wage difference between manager and chef -/
theorem manager_chef_wage_difference (w : SteakhouseWages) 
  (h : validSteakhouseWages w) : w.manager - w.chef = 2.60 := by
  sorry

end NUMINAMATH_CALUDE_manager_chef_wage_difference_l2238_223854


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2238_223847

theorem division_remainder_problem :
  let dividend : ℕ := 13698
  let divisor : ℚ := 153.75280898876406
  let quotient : ℕ := 89
  let remainder := dividend - (divisor * quotient).floor
  remainder = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2238_223847


namespace NUMINAMATH_CALUDE_exponent_monotonicity_l2238_223871

theorem exponent_monotonicity (a x₁ x₂ : ℝ) :
  (a > 1 ∧ x₁ > x₂ → a^x₁ > a^x₂) ∧
  (0 < a ∧ a < 1 ∧ x₁ > x₂ → a^x₁ < a^x₂) := by
  sorry

end NUMINAMATH_CALUDE_exponent_monotonicity_l2238_223871


namespace NUMINAMATH_CALUDE_symmetry_implies_ratio_l2238_223881

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℚ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if points A(m+4, -1) and B(1, n-3) are symmetric with respect to the origin,
    then m/n = -5/4 -/
theorem symmetry_implies_ratio (m n : ℚ) :
  symmetric_wrt_origin (m + 4) (-1) 1 (n - 3) →
  m / n = -5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_ratio_l2238_223881


namespace NUMINAMATH_CALUDE_perimeter_of_parallelogram_PSTU_l2238_223844

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  pq_eq_pr : dist P Q = dist P R
  pq_eq_15 : dist P Q = 15
  qr_eq_14 : dist Q R = 14

-- Define points S, T, U on the sides of the triangle
def S (P Q : ℝ × ℝ) : ℝ × ℝ := sorry
def T (Q R : ℝ × ℝ) : ℝ × ℝ := sorry
def U (P R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the parallelism conditions
def parallel (A B C D : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem perimeter_of_parallelogram_PSTU (P Q R : ℝ × ℝ) 
  (h : Triangle P Q R) 
  (h_st_parallel : parallel (S P Q) (T Q R) P R)
  (h_tu_parallel : parallel (T Q R) (U P R) P Q) : 
  dist P (S P Q) + dist (S P Q) (T Q R) + dist (T Q R) (U P R) + dist (U P R) P = 30 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_parallelogram_PSTU_l2238_223844


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2238_223812

theorem complex_product_theorem (y : ℂ) (h : y = Complex.exp (4 * Real.pi * Complex.I / 9)) :
  (3 * y^2 + y^4) * (3 * y^4 + y^8) * (3 * y^6 + y^12) * 
  (3 * y^8 + y^16) * (3 * y^10 + y^20) * (3 * y^12 + y^24) = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2238_223812


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l2238_223833

theorem airplane_seats_theorem :
  ∀ (total_seats : ℕ),
  (30 : ℕ) +                            -- First Class seats
  (total_seats * 20 / 100 : ℕ) +         -- Business Class seats (20% of total)
  (15 : ℕ) +                            -- Premium Economy Class seats
  (total_seats - (30 + (total_seats * 20 / 100) + 15) : ℕ) -- Economy Class seats
  = total_seats →
  total_seats = 288 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l2238_223833


namespace NUMINAMATH_CALUDE_alternating_sum_fraction_equals_two_l2238_223802

theorem alternating_sum_fraction_equals_two :
  (15 - 14 + 13 - 12 + 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_fraction_equals_two_l2238_223802


namespace NUMINAMATH_CALUDE_marks_total_votes_l2238_223875

/-- Calculates the total votes Mark received in an election given specific conditions --/
theorem marks_total_votes (first_area_voters : ℕ) 
  (first_area_undecided_percent : ℚ)
  (first_area_mark_percent : ℚ)
  (remaining_area_mark_multiplier : ℕ)
  (remaining_area_undecided_percent : ℚ)
  (remaining_area_population_increase : ℚ) :
  first_area_voters = 100000 →
  first_area_undecided_percent = 5 / 100 →
  first_area_mark_percent = 70 / 100 →
  remaining_area_mark_multiplier = 2 →
  remaining_area_undecided_percent = 7 / 100 →
  remaining_area_population_increase = 20 / 100 →
  ∃ (total_votes : ℕ), total_votes = 199500 ∧
    total_votes = 
      (first_area_voters * (1 - first_area_undecided_percent) * first_area_mark_percent).floor +
      (remaining_area_mark_multiplier * 
        (first_area_voters * (1 - first_area_undecided_percent) * first_area_mark_percent).floor) :=
by
  sorry


end NUMINAMATH_CALUDE_marks_total_votes_l2238_223875


namespace NUMINAMATH_CALUDE_comparison_and_inequality_l2238_223877

theorem comparison_and_inequality (x y m : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : m > 0) : 
  y / x < (y + m) / (x + m) ∧ Real.sqrt (x * y) * (2 - Real.sqrt (x * y)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_comparison_and_inequality_l2238_223877


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_one_l2238_223861

theorem negation_of_forall_geq_one :
  (¬ (∀ x : ℝ, x ≥ 1)) ↔ (∃ x : ℝ, x < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_one_l2238_223861


namespace NUMINAMATH_CALUDE_donation_problem_l2238_223891

/-- Calculates the total number of articles of clothing donated given the initial number of items set aside by Adam and the number of friends donating. -/
def total_donated_clothing (adam_pants : ℕ) (adam_jumpers : ℕ) (adam_pajama_sets : ℕ) (adam_tshirts : ℕ) (num_friends : ℕ) : ℕ := 
  let adam_initial := adam_pants + adam_jumpers + (2 * adam_pajama_sets) + adam_tshirts
  let friends_donation := num_friends * adam_initial
  let adam_final := adam_initial / 2
  adam_final + friends_donation

/-- Theorem stating that the total number of articles of clothing donated is 126, given the specific conditions of the problem. -/
theorem donation_problem : total_donated_clothing 4 4 4 20 3 = 126 := by
  sorry

end NUMINAMATH_CALUDE_donation_problem_l2238_223891


namespace NUMINAMATH_CALUDE_rearrangeable_natural_segments_l2238_223808

theorem rearrangeable_natural_segments (A B : Fin 1961 → ℕ) : 
  ∃ (σ τ : Equiv.Perm (Fin 1961)) (m : ℕ),
    ∀ (i : Fin 1961), A (σ i) + B (τ i) = m + i.val :=
sorry

end NUMINAMATH_CALUDE_rearrangeable_natural_segments_l2238_223808


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l2238_223884

theorem cloth_coloring_problem (men1 men2 days1 days2 length2 : ℕ) 
  (h1 : men1 = 4)
  (h2 : men2 = 6)
  (h3 : days1 = 2)
  (h4 : days2 = 1)
  (h5 : length2 = 36)
  (h6 : men1 * days2 * length2 = men2 * days1 * length1) :
  length1 = 48 :=
sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l2238_223884


namespace NUMINAMATH_CALUDE_john_jane_difference_l2238_223821

-- Define the street width
def street_width : ℕ := 25

-- Define the block side length
def block_side : ℕ := 500

-- Define Jane's path length (same as block side)
def jane_path : ℕ := block_side

-- Define John's path length (block side + 2 * street width)
def john_path : ℕ := block_side + 2 * street_width

-- Theorem statement
theorem john_jane_difference : 
  4 * john_path - 4 * jane_path = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_jane_difference_l2238_223821


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l2238_223842

theorem sum_of_reciprocals_lower_bound (a₁ a₂ a₃ : ℝ) 
  (pos₁ : a₁ > 0) (pos₂ : a₂ > 0) (pos₃ : a₃ > 0) 
  (sum_eq_one : a₁ + a₂ + a₃ = 1) : 
  1 / a₁ + 1 / a₂ + 1 / a₃ ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l2238_223842


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l2238_223815

theorem yan_distance_ratio :
  ∀ (a b v : ℝ),
  a > 0 → b > 0 → v > 0 →
  (b / v = a / v + (a + b) / (7 * v)) →
  (a / b = 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_yan_distance_ratio_l2238_223815


namespace NUMINAMATH_CALUDE_four_digit_product_sum_l2238_223870

theorem four_digit_product_sum (A B : ℕ) : 
  1000 ≤ A ∧ A < 10000 ∧ 1000 ≤ B ∧ B < 10000 →
  A * B = 16^5 + 2^10 →
  A + B = 2049 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_product_sum_l2238_223870


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2238_223885

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2238_223885


namespace NUMINAMATH_CALUDE_flight_portion_cost_l2238_223887

theorem flight_portion_cost (total_cost ground_cost flight_additional_cost : ℕ) :
  total_cost = 1275 →
  flight_additional_cost = 625 →
  ground_cost = 325 →
  ground_cost + flight_additional_cost = 950 := by
  sorry

end NUMINAMATH_CALUDE_flight_portion_cost_l2238_223887


namespace NUMINAMATH_CALUDE_max_triangle_area_l2238_223837

/-- The maximum area of a triangle ABC with side AB = 13 and BC:AC ratio of 60:61 is 3634 -/
theorem max_triangle_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 13 ∧ BC / AC = 60 / 61 → area ≤ 3634 :=
by sorry


end NUMINAMATH_CALUDE_max_triangle_area_l2238_223837


namespace NUMINAMATH_CALUDE_largest_certain_divisor_of_visible_product_l2238_223895

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem largest_certain_divisor_of_visible_product :
  ∀ (visible : Finset ℕ), visible ⊆ die_numbers → visible.card = 7 →
  ∃ (k : ℕ), (visible.prod id = 192 * k) := by
  sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_of_visible_product_l2238_223895


namespace NUMINAMATH_CALUDE_intersection_point_l2238_223893

-- Define the line
def line (x y : ℝ) : Prop := 5 * y - 2 * x = 10

-- Define a point on the x-axis
def on_x_axis (x y : ℝ) : Prop := y = 0

-- Theorem: The point (-5, 0) is on the line and the x-axis
theorem intersection_point : 
  line (-5) 0 ∧ on_x_axis (-5) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2238_223893


namespace NUMINAMATH_CALUDE_edward_final_lives_l2238_223869

/-- Calculates the final number of lives for Edward in a game --/
def edwardsLives (initialLives lostLives bonusLives penaltyLives : ℕ) : ℕ :=
  initialLives - lostLives + bonusLives - penaltyLives

/-- Theorem stating that Edward's final number of lives is 20 --/
theorem edward_final_lives :
  edwardsLives 30 12 5 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_lives_l2238_223869


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2238_223865

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h : a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) ≤ 1) : 
  1 / (b + c + 1) + 1 / (c + a + 1) + 1 / (a + b + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2238_223865


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l2238_223867

/-- Convert a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + (if c = '1' then 1 else 0)) 0

/-- Convert a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
  let rec aux (m : ℕ) : String :=
    if m = 0 then "" else aux (m / 2) ++ (if m % 2 = 1 then "1" else "0")
  aux n

theorem binary_multiplication_division :
  let a := binary_to_nat "11100"
  let b := binary_to_nat "11010"
  let c := binary_to_nat "100"
  nat_to_binary ((a * b) / c) = "10100110" := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l2238_223867


namespace NUMINAMATH_CALUDE_expression_simplification_l2238_223890

theorem expression_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2238_223890


namespace NUMINAMATH_CALUDE_sum_of_odd_decreasing_function_is_negative_l2238_223879

-- Define a structure for our function properties
structure OddDecreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x y, x < y → f x > f y

-- Main theorem
theorem sum_of_odd_decreasing_function_is_negative
  (f : ℝ → ℝ)
  (h_f : OddDecreasingFunction f)
  (α β γ : ℝ)
  (h_αβ : α + β > 0)
  (h_βγ : β + γ > 0)
  (h_γα : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_decreasing_function_is_negative_l2238_223879


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l2238_223807

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def prob_even : ℚ := 1/2
def prob_odd : ℚ := 1/2

theorem equal_even_odd_probability :
  (num_dice.choose (num_dice / 2)) * (prob_even ^ num_dice) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l2238_223807


namespace NUMINAMATH_CALUDE_sum_after_removal_l2238_223845

def original_series : List ℚ := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]

def removed_terms : List ℚ := [1/8, 1/10]

theorem sum_after_removal :
  (original_series.filter (λ x => x ∉ removed_terms)).sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_removal_l2238_223845


namespace NUMINAMATH_CALUDE_rachel_piggy_bank_l2238_223846

/-- The amount of money Rachel originally had in her piggy bank -/
def original_amount : ℕ := 5

/-- The amount of money Rachel now has in her piggy bank -/
def current_amount : ℕ := 3

/-- The amount of money Rachel took from her piggy bank -/
def amount_taken : ℕ := original_amount - current_amount

theorem rachel_piggy_bank :
  amount_taken = 2 :=
sorry

end NUMINAMATH_CALUDE_rachel_piggy_bank_l2238_223846


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2238_223859

theorem trigonometric_identity :
  Real.sin (20 * π / 180) * Real.sin (80 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2238_223859


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2238_223818

/-- Given a geometric sequence {aₙ} satisfying a₂ + a₄ = 20 and a₃ + a₅ = 40, prove a₅ + a₇ = 160 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_sum1 : a 2 + a 4 = 20) (h_sum2 : a 3 + a 5 = 40) : 
  a 5 + a 7 = 160 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2238_223818


namespace NUMINAMATH_CALUDE_unique_six_digit_reverse_when_multiplied_by_nine_l2238_223899

/-- A function that returns the digits of a natural number in reverse order -/
def reverseDigits (n : ℕ) : List ℕ :=
  sorry

/-- A function that checks if a number is a six-digit number -/
def isSixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

/-- The main theorem stating that 109989 is the only six-digit number
    that, when multiplied by 9, has its digits arranged in reverse order -/
theorem unique_six_digit_reverse_when_multiplied_by_nine :
  ∀ n : ℕ, isSixDigitNumber n →
    (reverseDigits n = reverseDigits (9 * n) → n = 109989) ∧
    (n = 109989 → reverseDigits n = reverseDigits (9 * n)) :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_reverse_when_multiplied_by_nine_l2238_223899


namespace NUMINAMATH_CALUDE_sequence_convergence_comparison_l2238_223863

theorem sequence_convergence_comparison
  (k : ℝ) (h_k : 0 < k ∧ k < 1/2)
  (a₀ b₀ : ℝ) (h_a₀ : 0 < a₀ ∧ a₀ < 1) (h_b₀ : 0 < b₀ ∧ b₀ < 1)
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_a : ∀ n, a (n + 1) = (a n + 1) / 2)
  (h_b : ∀ n, b (n + 1) = (b n) ^ k) :
  ∃ N, ∀ n ≥ N, a n < b n :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_comparison_l2238_223863


namespace NUMINAMATH_CALUDE_medal_award_ways_l2238_223886

-- Define the total number of sprinters
def total_sprinters : ℕ := 10

-- Define the number of Spanish sprinters
def spanish_sprinters : ℕ := 4

-- Define the number of medals
def medals : ℕ := 3

-- Function to calculate the number of ways to award medals
def award_medals : ℕ := sorry

-- Theorem statement
theorem medal_award_ways :
  award_medals = 696 :=
sorry

end NUMINAMATH_CALUDE_medal_award_ways_l2238_223886


namespace NUMINAMATH_CALUDE_student_ticket_cost_l2238_223874

/-- Proves that the cost of each student ticket is 2 dollars given the conditions of the ticket sales -/
theorem student_ticket_cost (total_tickets : ℕ) (total_revenue : ℕ) 
  (nonstudent_price : ℕ) (student_tickets : ℕ) :
  total_tickets = 821 →
  total_revenue = 1933 →
  nonstudent_price = 3 →
  student_tickets = 530 →
  ∃ (student_price : ℕ),
    student_price * student_tickets + 
    nonstudent_price * (total_tickets - student_tickets) = total_revenue ∧
    student_price = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_student_ticket_cost_l2238_223874


namespace NUMINAMATH_CALUDE_yarn_theorem_l2238_223841

def yarn_problem (B1 : ℝ) : Prop :=
  let B2 := 2 * B1
  let B3 := 3 * B1
  let B4 := 2 * B2
  let B5 := B3 + B4
  B3 = 27 ∧ B2 = 18

theorem yarn_theorem : ∃ B1 : ℝ, yarn_problem B1 := by
  sorry

end NUMINAMATH_CALUDE_yarn_theorem_l2238_223841


namespace NUMINAMATH_CALUDE_inequality_proof_l2238_223855

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb1 : 1 > b) (hb2 : b > -1) :
  a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2238_223855


namespace NUMINAMATH_CALUDE_clare_remaining_money_l2238_223838

-- Define the initial amount and item costs
def initial_amount : ℚ := 47
def bread_cost : ℚ := 2
def milk_cost : ℚ := 2
def cereal_cost : ℚ := 3
def apple_cost : ℚ := 4

-- Define the quantities of each item
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def cereal_quantity : ℕ := 3
def apple_quantity : ℕ := 1

-- Define the discount and tax rates
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining money
def calculate_remaining_money : ℚ :=
  let total_cost := bread_cost * bread_quantity + milk_cost * milk_quantity + 
                    cereal_cost * cereal_quantity + apple_cost * apple_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  let final_cost := discounted_cost + tax_amount
  initial_amount - final_cost

-- Theorem statement
theorem clare_remaining_money :
  calculate_remaining_money = 23.37 := by sorry

end NUMINAMATH_CALUDE_clare_remaining_money_l2238_223838


namespace NUMINAMATH_CALUDE_good_student_count_l2238_223836

/-- Represents a student in the class -/
inductive Student
| Good
| Troublemaker

/-- The total number of students in the class -/
def totalStudents : Nat := 25

/-- The number of students making the first claim -/
def firstClaimCount : Nat := 5

/-- The number of students making the second claim -/
def secondClaimCount : Nat := 20

/-- Represents the statements made by students -/
structure Statements where
  firstClaim : Bool  -- True if the statement is true
  secondClaim : Bool -- True if the statement is true

/-- Checks if the first claim is consistent with the given number of good students -/
def checkFirstClaim (goodCount : Nat) : Bool :=
  totalStudents - goodCount > (totalStudents - 1) / 2

/-- Checks if the second claim is consistent with the given number of good students -/
def checkSecondClaim (goodCount : Nat) : Bool :=
  totalStudents - goodCount = 3 * (goodCount - 1)

/-- Checks if the given number of good students is consistent with all statements -/
def isConsistent (goodCount : Nat) (statements : Statements) : Bool :=
  (statements.firstClaim = checkFirstClaim goodCount) &&
  (statements.secondClaim = checkSecondClaim goodCount)

/-- Theorem: The number of good students is either 5 or 7 -/
theorem good_student_count :
  ∃ (statements : Statements),
    (isConsistent 5 statements ∨ isConsistent 7 statements) ∧
    ∀ (n : Nat), n ≠ 5 ∧ n ≠ 7 → ¬ isConsistent n statements :=
by sorry

end NUMINAMATH_CALUDE_good_student_count_l2238_223836


namespace NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_l2238_223831

/-- Represents the outcome of throwing a coin -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the outcome of throwing 3 coins simultaneously -/
def ThreeCoinsOutcome := (CoinOutcome × CoinOutcome × CoinOutcome)

/-- Counts the number of heads in a ThreeCoinsOutcome -/
def countHeads : ThreeCoinsOutcome → Nat
  | (CoinOutcome.Heads, CoinOutcome.Heads, CoinOutcome.Heads) => 3
  | (CoinOutcome.Heads, CoinOutcome.Heads, CoinOutcome.Tails) => 2
  | (CoinOutcome.Heads, CoinOutcome.Tails, CoinOutcome.Heads) => 2
  | (CoinOutcome.Tails, CoinOutcome.Heads, CoinOutcome.Heads) => 2
  | (CoinOutcome.Heads, CoinOutcome.Tails, CoinOutcome.Tails) => 1
  | (CoinOutcome.Tails, CoinOutcome.Heads, CoinOutcome.Tails) => 1
  | (CoinOutcome.Tails, CoinOutcome.Tails, CoinOutcome.Heads) => 1
  | (CoinOutcome.Tails, CoinOutcome.Tails, CoinOutcome.Tails) => 0

/-- Event: At most one head facing up -/
def atMostOneHead (outcome : ThreeCoinsOutcome) : Prop :=
  countHeads outcome ≤ 1

/-- Event: At least two heads facing up -/
def atLeastTwoHeads (outcome : ThreeCoinsOutcome) : Prop :=
  countHeads outcome ≥ 2

/-- Theorem: The events "at most one head facing up" and "at least two heads facing up" 
    are mutually exclusive when throwing 3 coins simultaneously -/
theorem atMostOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : ThreeCoinsOutcome), ¬(atMostOneHead outcome ∧ atLeastTwoHeads outcome) :=
by sorry

end NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_l2238_223831


namespace NUMINAMATH_CALUDE_sum_last_two_digits_fibonacci_factorial_l2238_223860

def fibonacci_factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | 5 => 8
  | 6 => 13
  | 7 => 21
  | 8 => 34
  | 9 => 55
  | 10 => 89
  | 11 => 144
  | 12 => 233
  | _ => 0

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def last_two_digits (n : ℕ) : ℕ := n % 100

def modified_term (n : ℕ) : ℕ := last_two_digits (factorial (fibonacci_factorial n) + 2)

def sum_last_two_digits (n : ℕ) : ℕ := 
  (List.range n).map modified_term |> List.sum

theorem sum_last_two_digits_fibonacci_factorial : sum_last_two_digits 13 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_last_two_digits_fibonacci_factorial_l2238_223860


namespace NUMINAMATH_CALUDE_sheet_area_difference_l2238_223835

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem sheet_area_difference : 
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 9
  let sheet2_length : ℝ := 4.5
  let sheet2_width : ℝ := 11
  let combined_area (l w : ℝ) := 2 * l * w
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 99 := by
  sorry


end NUMINAMATH_CALUDE_sheet_area_difference_l2238_223835


namespace NUMINAMATH_CALUDE_wire_length_problem_l2238_223889

theorem wire_length_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 14 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 49 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_problem_l2238_223889


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l2238_223868

theorem quadratic_expression_equality : ∃ (a b c : ℝ), 
  (∀ x, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ 
  (10 * a - b - 4 * c = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l2238_223868


namespace NUMINAMATH_CALUDE_function_and_composition_proof_l2238_223880

def f (x b c : ℝ) : ℝ := x^2 - b*x + c

theorem function_and_composition_proof 
  (b c : ℝ) 
  (h1 : f 1 b c = 0) 
  (h2 : f 2 b c = -3) :
  (∀ x, f x b c = x^2 - 6*x + 5) ∧ 
  (∀ x, x > -1 → f (1 / Real.sqrt (x + 1)) b c = 1 / (x + 1) - 6 / Real.sqrt (x + 1) + 5) :=
by sorry

end NUMINAMATH_CALUDE_function_and_composition_proof_l2238_223880


namespace NUMINAMATH_CALUDE_small_circle_area_l2238_223896

theorem small_circle_area (large_circle_area : ℝ) (num_small_circles : ℕ) :
  large_circle_area = 120 →
  num_small_circles = 6 →
  ∃ small_circle_area : ℝ,
    small_circle_area = large_circle_area / (3 * num_small_circles) ∧
    small_circle_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_area_l2238_223896


namespace NUMINAMATH_CALUDE_student_groups_l2238_223810

theorem student_groups (group_size : ℕ) (left_early : ℕ) (remaining : ℕ) : 
  group_size = 8 → left_early = 2 → remaining = 22 → 
  (remaining + left_early) / group_size = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_student_groups_l2238_223810


namespace NUMINAMATH_CALUDE_distance_to_origin_l2238_223864

theorem distance_to_origin (a : ℝ) : |a - 0| = 5 → (3 - a = -2 ∨ 3 - a = 8) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2238_223864


namespace NUMINAMATH_CALUDE_car_rental_hours_per_day_l2238_223858

/-- Proves that given the rental conditions, the number of hours rented per day is 8 --/
theorem car_rental_hours_per_day 
  (hourly_rate : ℝ)
  (days_per_week : ℕ)
  (weekly_income : ℝ)
  (h : hourly_rate = 20)
  (d : days_per_week = 4)
  (w : weekly_income = 640) :
  (weekly_income / (hourly_rate * days_per_week : ℝ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_rental_hours_per_day_l2238_223858


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_squares_l2238_223856

/-- Given a line segment connecting (1, -3) and (4, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that a^2 + b^2 + c^2 + d^2 = 163 -/
theorem line_segment_parameter_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 4 ∧ c + d = 9) →
  a^2 + b^2 + c^2 + d^2 = 163 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_squares_l2238_223856


namespace NUMINAMATH_CALUDE_class_average_l2238_223850

theorem class_average (total_students : Nat) (perfect_score_students : Nat) (zero_score_students : Nat) (class_average : ℚ) : 
  total_students = 20 →
  perfect_score_students = 2 →
  zero_score_students = 3 →
  class_average = 40 →
  let remaining_students := total_students - perfect_score_students - zero_score_students
  let total_score := total_students * class_average
  let perfect_score_total := perfect_score_students * 100
  let remaining_score := total_score - perfect_score_total
  remaining_score / remaining_students = 40 := by
sorry

end NUMINAMATH_CALUDE_class_average_l2238_223850


namespace NUMINAMATH_CALUDE_product_real_implies_ratio_l2238_223873

def complex (a b : ℝ) : ℂ := a + b * Complex.I

theorem product_real_implies_ratio (a b : ℝ) (hb : b ≠ 0) :
  let z₁ : ℂ := 2 + 3 * Complex.I
  let z₂ : ℂ := complex a b
  (z₁ * z₂).im = 0 → a / b = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_real_implies_ratio_l2238_223873


namespace NUMINAMATH_CALUDE_expansion_equals_fourth_power_l2238_223805

theorem expansion_equals_fourth_power (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_fourth_power_l2238_223805


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l2238_223894

def f (x : ℝ) := 3*x - x^3

theorem monotonic_increasing_interval_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, StrictMonoOn f (Set.Ioo (-1 : ℝ) 1) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l2238_223894


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_equivalent_to_range_l2238_223852

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | 1 < x ∧ x < 2} := by sorry

theorem subset_condition_equivalent_to_range :
  ∀ a : ℝ, A a ⊆ A a ∩ B ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_equivalent_to_range_l2238_223852


namespace NUMINAMATH_CALUDE_roof_length_width_difference_roof_area_is_720_length_is_5_times_width_l2238_223822

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- The roof of an apartment building -/
def apartmentRoof : RoofDimensions where
  width := (720 / 5).sqrt
  length := 5 * (720 / 5).sqrt

theorem roof_length_width_difference : 
  apartmentRoof.length - apartmentRoof.width = 48 := by
  sorry

/-- The area of the roof -/
def roofArea (roof : RoofDimensions) : ℝ :=
  roof.length * roof.width

theorem roof_area_is_720 : roofArea apartmentRoof = 720 := by
  sorry

theorem length_is_5_times_width : 
  apartmentRoof.length = 5 * apartmentRoof.width := by
  sorry

end NUMINAMATH_CALUDE_roof_length_width_difference_roof_area_is_720_length_is_5_times_width_l2238_223822


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2238_223809

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y > 0 ∧ Nat.lcm x (Nat.lcm 15 21) = 210) → x ≤ 70 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2238_223809


namespace NUMINAMATH_CALUDE_eulers_formula_l2238_223897

/-- Euler's formula -/
theorem eulers_formula (a b : ℝ) :
  Complex.exp (a + Complex.I * b) = Complex.exp a * (Complex.cos b + Complex.I * Complex.sin b) := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2238_223897


namespace NUMINAMATH_CALUDE_treys_chores_l2238_223814

theorem treys_chores (task_duration : ℕ) (total_time : ℕ) (shower_tasks : ℕ) (dinner_tasks : ℕ) :
  task_duration = 10 →
  total_time = 120 →
  shower_tasks = 1 →
  dinner_tasks = 4 →
  (total_time / task_duration) - shower_tasks - dinner_tasks = 7 :=
by sorry

end NUMINAMATH_CALUDE_treys_chores_l2238_223814


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2238_223839

/-- Given the compound interest for the second and third years, calculate the interest rate. -/
theorem interest_rate_calculation (CI2 CI3 : ℝ) (h1 : CI2 = 1200) (h2 : CI3 = 1272) :
  ∃ (r : ℝ), r = 0.06 ∧ CI3 - CI2 = CI2 * r :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2238_223839


namespace NUMINAMATH_CALUDE_classroom_fraction_l2238_223800

theorem classroom_fraction (total : ℕ) (absent_fraction : ℚ) (canteen : ℕ) : 
  total = 40 → 
  absent_fraction = 1 / 10 → 
  canteen = 9 → 
  (total - (absent_fraction * total).num - canteen : ℚ) / (total - (absent_fraction * total).num) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_classroom_fraction_l2238_223800


namespace NUMINAMATH_CALUDE_euclidean_algorithm_fibonacci_bound_l2238_223848

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the Euclidean algorithm
def euclidean_algorithm (m₀ m₁ : ℕ) : ℕ → Prop
  | 0 => m₁ = 0
  | k + 1 => ∃ q r, m₀ = q * m₁ + r ∧ r < m₁ ∧ euclidean_algorithm m₁ r k

-- Theorem statement
theorem euclidean_algorithm_fibonacci_bound {m₀ m₁ k : ℕ} 
  (h : euclidean_algorithm m₀ m₁ k) : 
  m₁ ≥ fib (k + 1) ∧ m₀ ≥ fib (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_euclidean_algorithm_fibonacci_bound_l2238_223848


namespace NUMINAMATH_CALUDE_school_girls_count_l2238_223813

theorem school_girls_count (total_students sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : ∃ (girls_in_sample : ℕ), 
    girls_in_sample + (girls_in_sample + 10) = sample_size) :
  ∃ (girls_in_school : ℕ), 
    girls_in_school = (950 : ℕ) ∧ 
    (girls_in_school : ℚ) / total_students = 
      ((sample_size / 2 - 5) : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_school_girls_count_l2238_223813


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_y_eq_neg_four_l2238_223866

/-- Two vectors in ℝ² -/
def a : Fin 2 → ℝ := ![1, 2]
def b (y : ℝ) : Fin 2 → ℝ := ![-2, y]

/-- Parallel vectors in ℝ² have proportional coordinates -/
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ i, v i = k * u i

/-- If a and b are parallel plane vectors, then y = -4 -/
theorem parallel_vectors_implies_y_eq_neg_four :
  parallel a (b y) → y = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_y_eq_neg_four_l2238_223866


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2238_223816

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the perimeter is 7 + √19 under the following conditions:
    1) a² - c² + 3b = 0
    2) The area of the triangle is 5√3/2
    3) Angle A = 60° -/
theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  a^2 - c^2 + 3*b = 0 → 
  S = (5 * Real.sqrt 3) / 2 →
  A = π / 3 →
  a + b + c = 7 + Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2238_223816


namespace NUMINAMATH_CALUDE_function_properties_l2238_223829

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / x

theorem function_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ m n : ℝ, m > 0 ∧ n > 0 ∧ m < n ∧ f a m = 2*m ∧ f a n = 2*n → a > 2 * Real.sqrt 2) ∧
  ((∀ x : ℝ, x ∈ Set.Icc (1/3) (1/2) → x^2 * |f a x| ≤ 1) → a ∈ Set.Icc (-2) 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2238_223829


namespace NUMINAMATH_CALUDE_jack_apples_proof_l2238_223804

def initial_apples : ℕ := 150
def jill_percentage : ℚ := 30 / 100
def june_percentage : ℚ := 20 / 100
def gift_apples : ℕ := 2

def remaining_apples : ℕ := 82

theorem jack_apples_proof :
  let after_jill := initial_apples - (initial_apples * jill_percentage).floor
  let after_june := after_jill - (after_jill * june_percentage).floor
  after_june - gift_apples = remaining_apples :=
by sorry

end NUMINAMATH_CALUDE_jack_apples_proof_l2238_223804


namespace NUMINAMATH_CALUDE_h_value_l2238_223820

/-- The value of h for which the given conditions are satisfied -/
def h : ℝ := 32

/-- The y-coordinate of the first graph -/
def graph1 (x : ℝ) : ℝ := 4 * (x - h)^2 + 4032 - 4 * h^2

/-- The y-coordinate of the second graph -/
def graph2 (x : ℝ) : ℝ := 5 * (x - h)^2 + 5040 - 5 * h^2

theorem h_value :
  (graph1 0 = 4032) ∧
  (graph2 0 = 5040) ∧
  (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ graph1 x1 = 0 ∧ graph1 x2 = 0) ∧
  (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ graph2 x1 = 0 ∧ graph2 x2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_h_value_l2238_223820


namespace NUMINAMATH_CALUDE_product_expansion_sum_l2238_223878

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (2 * x^2 - 4 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -24 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l2238_223878
