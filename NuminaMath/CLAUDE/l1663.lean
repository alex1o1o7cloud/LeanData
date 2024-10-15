import Mathlib

namespace NUMINAMATH_CALUDE_binary_1011_equals_11_l1663_166352

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1011_equals_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_equals_11_l1663_166352


namespace NUMINAMATH_CALUDE_storm_average_rainfall_l1663_166344

theorem storm_average_rainfall 
  (duration : ℝ) 
  (first_30min : ℝ) 
  (next_30min : ℝ) 
  (last_hour : ℝ) :
  duration = 2 →
  first_30min = 5 →
  next_30min = first_30min / 2 →
  last_hour = 1 / 2 →
  (first_30min + next_30min + last_hour) / duration = 4 := by
sorry

end NUMINAMATH_CALUDE_storm_average_rainfall_l1663_166344


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1663_166376

theorem polynomial_divisibility (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  (x - y)^5 + (y - z)^5 + (z - x)^5 = 
  -5 * (x - y) * (y - z) * (z - x) * ((x - y)^2 + (x - y) * (y - z) + (y - z)^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1663_166376


namespace NUMINAMATH_CALUDE_lars_daily_bread_production_l1663_166388

-- Define the baking rates and working hours
def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

-- Define the function to calculate total breads
def total_breads : ℕ :=
  (loaves_per_hour * hours_per_day) + 
  (baguettes_per_two_hours * (hours_per_day / 2))

-- Theorem statement
theorem lars_daily_bread_production :
  total_breads = 150 := by
  sorry

end NUMINAMATH_CALUDE_lars_daily_bread_production_l1663_166388


namespace NUMINAMATH_CALUDE_alicia_local_taxes_l1663_166332

theorem alicia_local_taxes (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.02 → hourly_wage * tax_rate * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_alicia_local_taxes_l1663_166332


namespace NUMINAMATH_CALUDE_tesseract_hypervolume_l1663_166392

/-- Given a tesseract with face volumes 72, 75, 48, and 50 cubic units,
    its hyper-volume is 3600 hyper-cubic units. -/
theorem tesseract_hypervolume (a b c d : ℝ) 
    (h1 : a * b * c = 72)
    (h2 : b * c * d = 75)
    (h3 : c * d * a = 48)
    (h4 : d * a * b = 50) : 
    a * b * c * d = 3600 := by
  sorry

#check tesseract_hypervolume

end NUMINAMATH_CALUDE_tesseract_hypervolume_l1663_166392


namespace NUMINAMATH_CALUDE_domain_of_f_composed_with_exp2_l1663_166351

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_f_composed_with_exp2 :
  (∀ x, f x ≠ 0 → 1 < x ∧ x < 2) →
  (∀ x, f (2^x) ≠ 0 → 0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_composed_with_exp2_l1663_166351


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1663_166384

/-- The imaginary part of (1-i)^2 / (1+i) is -1 -/
theorem imaginary_part_of_complex_fraction : Complex.im ((1 - Complex.I)^2 / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1663_166384


namespace NUMINAMATH_CALUDE_pool_capacity_percentage_l1663_166377

/-- Calculates the current capacity percentage of a pool given its dimensions and draining parameters -/
theorem pool_capacity_percentage
  (width : ℝ) (length : ℝ) (depth : ℝ)
  (drain_rate : ℝ) (drain_time : ℝ)
  (h_width : width = 60)
  (h_length : length = 100)
  (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60)
  (h_drain_time : drain_time = 800) :
  (drain_rate * drain_time) / (width * length * depth) * 100 = 8 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_percentage_l1663_166377


namespace NUMINAMATH_CALUDE_henrys_brothers_ages_sum_l1663_166302

theorem henrys_brothers_ages_sum :
  ∀ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a < 10 ∧ b < 10 ∧ c < 10 →
    a > 0 ∧ b > 0 ∧ c > 0 →
    a = 2 * b →
    c * c = b →
    a + b + c = 14 :=
by sorry

end NUMINAMATH_CALUDE_henrys_brothers_ages_sum_l1663_166302


namespace NUMINAMATH_CALUDE_tree_height_equation_l1663_166397

/-- Represents the height of a tree over time -/
def tree_height (initial_height growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_height + growth_rate * months

/-- Theorem stating the relationship between tree height and time -/
theorem tree_height_equation (h x : ℝ) :
  h = tree_height 80 2 x ↔ h = 80 + 2 * x :=
by sorry

end NUMINAMATH_CALUDE_tree_height_equation_l1663_166397


namespace NUMINAMATH_CALUDE_eleven_play_both_l1663_166322

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def playsBoth (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem: In the given sports club, 11 members play both badminton and tennis -/
theorem eleven_play_both (club : SportsClub)
  (h_total : club.total = 27)
  (h_badminton : club.badminton = 17)
  (h_tennis : club.tennis = 19)
  (h_neither : club.neither = 2) :
  playsBoth club = 11 := by
  sorry

#eval playsBoth { total := 27, badminton := 17, tennis := 19, neither := 2 }

end NUMINAMATH_CALUDE_eleven_play_both_l1663_166322


namespace NUMINAMATH_CALUDE_average_marks_second_class_l1663_166304

/-- Theorem: Average marks of second class --/
theorem average_marks_second_class 
  (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg_total : ℝ) :
  n₁ = 55 →
  n₂ = 48 →
  avg₁ = 60 →
  avg_total = 59.067961165048544 →
  let avg₂ := ((n₁ + n₂ : ℝ) * avg_total - n₁ * avg₁) / n₂
  ∃ ε > 0, |avg₂ - 57.92| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_marks_second_class_l1663_166304


namespace NUMINAMATH_CALUDE_complex_norm_problem_l1663_166355

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 1) :
  Complex.abs z = Real.sqrt (225 / 7) :=
sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l1663_166355


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1663_166329

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1663_166329


namespace NUMINAMATH_CALUDE_solve_for_a_l1663_166313

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = 2 → a * x - 2 = 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1663_166313


namespace NUMINAMATH_CALUDE_cookie_problem_l1663_166326

theorem cookie_problem (tom mike millie lucy frank : ℕ) : 
  tom = 16 →
  lucy * lucy = tom →
  millie = 2 * lucy →
  mike = 3 * millie →
  frank = mike / 2 - 3 →
  frank = 9 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l1663_166326


namespace NUMINAMATH_CALUDE_john_total_cost_l1663_166359

def base_cost : ℝ := 25
def text_cost_per_message : ℝ := 0.1
def extra_minute_cost : ℝ := 0.15
def included_hours : ℝ := 20
def john_messages : ℕ := 150
def john_hours : ℝ := 22

def calculate_total_cost : ℝ :=
  base_cost +
  (↑john_messages * text_cost_per_message) +
  ((john_hours - included_hours) * 60 * extra_minute_cost)

theorem john_total_cost :
  calculate_total_cost = 58 :=
sorry

end NUMINAMATH_CALUDE_john_total_cost_l1663_166359


namespace NUMINAMATH_CALUDE_point_three_units_away_l1663_166380

theorem point_three_units_away (A : ℝ) (h : A = 2) :
  ∀ B : ℝ, abs (B - A) = 3 → (B = -1 ∨ B = 5) :=
by sorry

end NUMINAMATH_CALUDE_point_three_units_away_l1663_166380


namespace NUMINAMATH_CALUDE_number_of_coverings_number_of_coverings_eq_coverings_order_invariant_l1663_166342

/-- The number of coverings of a finite set -/
theorem number_of_coverings (n : ℕ) : ℕ := 
  2^(2^n - 1)

/-- The number of coverings of a finite set X with n elements is 2^(2^n - 1) -/
theorem number_of_coverings_eq (X : Finset ℕ) (h : X.card = n) :
  (Finset.powerset X).card = number_of_coverings n := by
  sorry

/-- The order of covering sets does not affect the total number of coverings -/
theorem coverings_order_invariant (X : Finset ℕ) (C₁ C₂ : Finset (Finset ℕ)) 
  (h₁ : ∀ x ∈ X, ∃ S ∈ C₁, x ∈ S) (h₂ : ∀ x ∈ X, ∃ S ∈ C₂, x ∈ S) :
  C₁.card = C₂.card := by
  sorry

end NUMINAMATH_CALUDE_number_of_coverings_number_of_coverings_eq_coverings_order_invariant_l1663_166342


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l1663_166370

theorem valentines_day_theorem (x y : ℕ) : 
  x * y = x + y + 28 → x * y = 60 :=
by sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l1663_166370


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1663_166340

theorem polar_to_cartesian_circle (x y ρ : ℝ) :
  ρ = 2 ↔ x^2 + y^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1663_166340


namespace NUMINAMATH_CALUDE_credit_card_more_profitable_min_days_for_credit_card_profitability_l1663_166357

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 20000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.01

/-- Represents the annual interest rate on the debit card -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of days in a month (assumed) -/
def days_in_month : ℕ := 30

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 360

/-- Theorem stating the minimum number of days for credit card to be more profitable -/
theorem credit_card_more_profitable (N : ℕ) : 
  (N : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
  credit_cashback_rate * purchase_amount > 
  debit_cashback_rate * purchase_amount → N ≥ 31 := by
  sorry

/-- Theorem stating that 31 days is the minimum for credit card to be more profitable -/
theorem min_days_for_credit_card_profitability : 
  ∃ (N : ℕ), N = 31 ∧ 
  (∀ (M : ℕ), M < N → 
    (M : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
    credit_cashback_rate * purchase_amount ≤ 
    debit_cashback_rate * purchase_amount) ∧
  ((N : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
   credit_cashback_rate * purchase_amount > 
   debit_cashback_rate * purchase_amount) := by
  sorry

end NUMINAMATH_CALUDE_credit_card_more_profitable_min_days_for_credit_card_profitability_l1663_166357


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l1663_166395

/-- The hyperbola with equation x^2/9 - y^2/16 = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) - (p.2^2 / 16) = 1}

/-- The right focus of the hyperbola -/
def F : ℝ × ℝ := (5, 0)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- A point on the hyperbola where a line perpendicular to an asymptote intersects it -/
def P : ℝ × ℝ := sorry

/-- The area of triangle OPF -/
def area_OPF : ℝ := sorry

theorem hyperbola_triangle_area :
  area_OPF = 6 := by sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l1663_166395


namespace NUMINAMATH_CALUDE_soldier_height_arrangement_l1663_166399

theorem soldier_height_arrangement (n : ℕ) (a b : Fin n → ℝ) :
  (∀ i : Fin n, a i ≤ b i) →
  (∀ i j : Fin n, i < j → a i ≥ a j) →
  (∀ i j : Fin n, i < j → b i ≥ b j) →
  ∀ i : Fin n, a i ≤ b i :=
by sorry

end NUMINAMATH_CALUDE_soldier_height_arrangement_l1663_166399


namespace NUMINAMATH_CALUDE_inequality_proof_l1663_166345

theorem inequality_proof (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1663_166345


namespace NUMINAMATH_CALUDE_fraction_inequality_l1663_166360

theorem fraction_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : m > 0) :
  (a + m) / (b + m) > a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1663_166360


namespace NUMINAMATH_CALUDE_ann_keeps_36_cookies_l1663_166312

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of oatmeal raisin cookies Ann bakes -/
def oatmeal_baked : ℕ := 3 * dozen

/-- The number of sugar cookies Ann bakes -/
def sugar_baked : ℕ := 2 * dozen

/-- The number of chocolate chip cookies Ann bakes -/
def chocolate_baked : ℕ := 4 * dozen

/-- The number of oatmeal raisin cookies Ann gives away -/
def oatmeal_given : ℕ := 2 * dozen

/-- The number of sugar cookies Ann gives away -/
def sugar_given : ℕ := (3 * dozen) / 2

/-- The number of chocolate chip cookies Ann gives away -/
def chocolate_given : ℕ := (5 * dozen) / 2

/-- The total number of cookies Ann keeps -/
def total_kept : ℕ := (oatmeal_baked - oatmeal_given) + (sugar_baked - sugar_given) + (chocolate_baked - chocolate_given)

theorem ann_keeps_36_cookies : total_kept = 36 := by
  sorry

end NUMINAMATH_CALUDE_ann_keeps_36_cookies_l1663_166312


namespace NUMINAMATH_CALUDE_price_increase_to_equality_l1663_166335

theorem price_increase_to_equality (price_B : ℝ) (price_A : ℝ) 
    (h1 : price_A = price_B * 0.8) : 
  (price_B - price_A) / price_A * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_to_equality_l1663_166335


namespace NUMINAMATH_CALUDE_fraction_puzzle_solvable_l1663_166318

def is_valid_fraction (a b : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ a ≠ b

def are_distinct (a b c d e f g h i : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem fraction_puzzle_solvable : 
  ∃ (a b c d e f g h i : ℕ),
    is_valid_fraction a b ∧ 
    is_valid_fraction c d ∧ 
    is_valid_fraction e f ∧ 
    is_valid_fraction g h ∧
    are_distinct a b c d e f g h i ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f + (g : ℚ) / h = i := by
  sorry

end NUMINAMATH_CALUDE_fraction_puzzle_solvable_l1663_166318


namespace NUMINAMATH_CALUDE_z_in_terms_of_a_b_s_l1663_166365

theorem z_in_terms_of_a_b_s 
  (z a b s : ℝ) 
  (hz : z ≠ 0) 
  (heq : z = a^3 * b^2 + 6*z*s - 9*s^2) :
  z = (a^3 * b^2 - 9*s^2) / (1 - 6*s) :=
by sorry

end NUMINAMATH_CALUDE_z_in_terms_of_a_b_s_l1663_166365


namespace NUMINAMATH_CALUDE_braiding_time_for_dance_team_l1663_166301

/-- Calculates the time in minutes to braid dancers' hair -/
def braidingTime (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℕ :=
  let total_braids := num_dancers * braids_per_dancer
  let total_seconds := total_braids * seconds_per_braid
  total_seconds / 60

theorem braiding_time_for_dance_team :
  braidingTime 8 5 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_braiding_time_for_dance_team_l1663_166301


namespace NUMINAMATH_CALUDE_minimum_driving_age_l1663_166375

/-- The minimum driving age problem -/
theorem minimum_driving_age 
  (kayla_age : ℕ) 
  (kimiko_age : ℕ) 
  (min_driving_age : ℕ) 
  (h1 : kayla_age * 2 = kimiko_age) 
  (h2 : kimiko_age = 26) 
  (h3 : min_driving_age = kayla_age + 5) : 
  min_driving_age = 18 := by
sorry

end NUMINAMATH_CALUDE_minimum_driving_age_l1663_166375


namespace NUMINAMATH_CALUDE_xy_value_l1663_166338

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(2*y) = 729) :
  x * y = -63/25 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l1663_166338


namespace NUMINAMATH_CALUDE_cheese_cost_is_50_l1663_166308

/-- The cost of a sandwich in cents -/
def sandwich_cost : ℕ := 90

/-- The cost of a slice of bread in cents -/
def bread_cost : ℕ := 15

/-- The cost of a slice of ham in cents -/
def ham_cost : ℕ := 25

/-- The cost of a slice of cheese in cents -/
def cheese_cost : ℕ := sandwich_cost - bread_cost - ham_cost

theorem cheese_cost_is_50 : cheese_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_cheese_cost_is_50_l1663_166308


namespace NUMINAMATH_CALUDE_task_pages_l1663_166354

/-- Represents the number of pages in the printing task -/
def P : ℕ := 480

/-- Represents the rate of Printer A in pages per minute -/
def rate_A : ℚ := P / 60

/-- Represents the rate of Printer B in pages per minute -/
def rate_B : ℚ := rate_A + 4

/-- Theorem stating that the number of pages in the task is 480 -/
theorem task_pages : P = 480 := by
  have h1 : rate_A + rate_B = P / 40 := by sorry
  have h2 : rate_A = P / 60 := by sorry
  have h3 : rate_B = rate_A + 4 := by sorry
  sorry

#check task_pages

end NUMINAMATH_CALUDE_task_pages_l1663_166354


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l1663_166310

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def all_digits_different (n : ℕ) : Prop := sorry

/-- A function that checks if two natural numbers have all different digits between them -/
def all_digits_different_between (a b : ℕ) : Prop := sorry

theorem smallest_sum_of_digits_of_sum :
  ∀ a b : ℕ,
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    a ≠ b ∧
    all_digits_different a ∧
    all_digits_different b ∧
    all_digits_different_between a b ∧
    1000 ≤ a + b ∧ a + b < 10000 →
    ∃ (s : ℕ), s = a + b ∧ sum_of_digits s = 1 ∧
    ∀ (t : ℕ), t = a + b → sum_of_digits s ≤ sum_of_digits t :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l1663_166310


namespace NUMINAMATH_CALUDE_fraction_sum_and_reciprocal_sum_integer_fraction_sum_and_reciprocal_sum_integer_distinct_numerators_l1663_166390

theorem fraction_sum_and_reciprocal_sum_integer :
  ∃ (a b c d e f : ℕ), 
    (0 < a ∧ a < b) ∧ 
    (0 < c ∧ c < d) ∧ 
    (0 < e ∧ e < f) ∧
    (Nat.gcd a b = 1) ∧
    (Nat.gcd c d = 1) ∧
    (Nat.gcd e f = 1) ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 ∧
    ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

theorem fraction_sum_and_reciprocal_sum_integer_distinct_numerators :
  ∃ (a b c d e f : ℕ), 
    (0 < a ∧ a < b) ∧ 
    (0 < c ∧ c < d) ∧ 
    (0 < e ∧ e < f) ∧
    (Nat.gcd a b = 1) ∧
    (Nat.gcd c d = 1) ∧
    (Nat.gcd e f = 1) ∧
    a ≠ c ∧ a ≠ e ∧ c ≠ e ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 ∧
    ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_and_reciprocal_sum_integer_fraction_sum_and_reciprocal_sum_integer_distinct_numerators_l1663_166390


namespace NUMINAMATH_CALUDE_michael_needs_additional_money_l1663_166378

def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_cost_gbp : ℝ := 30
def photo_album_cost_eur : ℝ := 25
def gbp_to_usd : ℝ := 1.4
def eur_to_usd : ℝ := 1.2

theorem michael_needs_additional_money :
  let perfume_cost_usd := perfume_cost_gbp * gbp_to_usd
  let photo_album_cost_usd := photo_album_cost_eur * eur_to_usd
  let total_cost := cake_cost + bouquet_cost + balloons_cost + perfume_cost_usd + photo_album_cost_usd
  total_cost - michael_money = 83 := by sorry

end NUMINAMATH_CALUDE_michael_needs_additional_money_l1663_166378


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l1663_166303

def b : ℕ := 8

-- S_n is the sum of reciprocals of non-zero digits of integers from 1 to b^n
def S (n : ℕ) : ℚ :=
  -- We don't implement the actual sum here, just define its signature
  sorry

-- Predicate to check if a number is an integer
def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

-- Main theorem
theorem smallest_n_for_integer_S :
  ∀ n : ℕ, n > 0 → is_integer (S n) → n ≥ 105 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l1663_166303


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1663_166346

/-- A convex decagon is a polygon with 10 sides -/
def ConvexDecagon : Type := Unit

/-- Number of sides in a convex decagon -/
def numSides : ℕ := 10

/-- Number of right angles in the given decagon -/
def numRightAngles : ℕ := 3

/-- The number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem decagon_diagonals (d : ConvexDecagon) : 
  numDiagonals numSides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1663_166346


namespace NUMINAMATH_CALUDE_problem_solution_l1663_166347

/-- The number of ways to distribute n distinct objects to k recipients -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinct objects to k recipients,
    where 2 specific objects must be given to the same recipient -/
def distributeWithPair (n k : ℕ) : ℕ := k * (k^(n - 2))

theorem problem_solution :
  distributeWithPair 8 10 = 10^7 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1663_166347


namespace NUMINAMATH_CALUDE_germination_probability_convergence_l1663_166386

/-- Represents the experimental data for rice seed germination --/
structure GerminationData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations
  h : m ≤ n

/-- The list of experimental data --/
def experimentalData : List GerminationData := [
  ⟨50, 47, sorry⟩,
  ⟨100, 89, sorry⟩,
  ⟨200, 188, sorry⟩,
  ⟨500, 461, sorry⟩,
  ⟨1000, 892, sorry⟩,
  ⟨2000, 1826, sorry⟩,
  ⟨3000, 2733, sorry⟩
]

/-- The germination frequency for a given experiment --/
def germinationFrequency (data : GerminationData) : ℚ :=
  data.m / data.n

/-- The estimated probability of germination --/
def estimatedProbability : ℚ := 91 / 100

/-- Theorem stating that the germination frequency approaches the estimated probability as sample size increases --/
theorem germination_probability_convergence :
  ∀ ε > 0, ∃ N, ∀ data ∈ experimentalData, data.n ≥ N →
    |germinationFrequency data - estimatedProbability| < ε :=
sorry

end NUMINAMATH_CALUDE_germination_probability_convergence_l1663_166386


namespace NUMINAMATH_CALUDE_appetizers_per_guest_is_six_l1663_166363

def number_of_guests : ℕ := 30

def prepared_appetizers : ℕ := 3 * 12 + 2 * 12 + 2 * 12

def additional_appetizers : ℕ := 8 * 12

def total_appetizers : ℕ := prepared_appetizers + additional_appetizers

def appetizers_per_guest : ℚ := total_appetizers / number_of_guests

theorem appetizers_per_guest_is_six :
  appetizers_per_guest = 6 := by
  sorry

end NUMINAMATH_CALUDE_appetizers_per_guest_is_six_l1663_166363


namespace NUMINAMATH_CALUDE_milk_conversion_theorem_l1663_166368

/-- Represents the conversion between milliliters and fluid ounces -/
structure MilkConversion where
  packets : Nat
  ml_per_packet : Nat
  total_ounces : Nat

/-- Calculates the number of milliliters in one fluid ounce -/
def ml_per_ounce (conv : MilkConversion) : Rat :=
  (conv.packets * conv.ml_per_packet) / conv.total_ounces

/-- Theorem stating that under the given conditions, one fluid ounce equals 30 ml -/
theorem milk_conversion_theorem (conv : MilkConversion) 
  (h1 : conv.packets = 150)
  (h2 : conv.ml_per_packet = 250)
  (h3 : conv.total_ounces = 1250) : 
  ml_per_ounce conv = 30 := by
  sorry

end NUMINAMATH_CALUDE_milk_conversion_theorem_l1663_166368


namespace NUMINAMATH_CALUDE_triangle_side_values_l1663_166339

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_side_values :
  ∀ x : ℕ+, 
    (triangle_exists 8 11 (x.val ^ 2)) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l1663_166339


namespace NUMINAMATH_CALUDE_inverse_multiplication_l1663_166373

theorem inverse_multiplication (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_multiplication_l1663_166373


namespace NUMINAMATH_CALUDE_atomic_weight_Br_l1663_166334

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := 137.33

/-- The molecular weight of the compound -/
def molecular_weight : ℝ := 297

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def num_Br : ℕ := 2

/-- Theorem: The atomic weight of Bromine (Br) is 79.835 -/
theorem atomic_weight_Br :
  let x := (molecular_weight - num_Ba * atomic_weight_Ba) / num_Br
  x = 79.835 := by sorry

end NUMINAMATH_CALUDE_atomic_weight_Br_l1663_166334


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1663_166356

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : a = b * Real.sin A + Real.sqrt 3 * a * Real.cos B)
  (h2 : b = 4) (h3 : (1/2) * a * c = 4) :
  B = Real.pi / 2 ∧ a + b + c = 4 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1663_166356


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1663_166348

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x + 1| :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1663_166348


namespace NUMINAMATH_CALUDE_problem_statement_l1663_166364

theorem problem_statement (x : ℝ) (h : x = 4) : 5 * x + 3 - x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1663_166364


namespace NUMINAMATH_CALUDE_acute_angles_equation_solution_l1663_166319

theorem acute_angles_equation_solution (A B : Real) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  Real.sin A * Real.cos B + Real.sqrt (2 * Real.sin A) * Real.sin B = (3 * Real.sin A + 1) / Real.sqrt 5 →
  A = π/6 ∧ B = π/2 - Real.arcsin (Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_equation_solution_l1663_166319


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l1663_166331

theorem hippopotamus_crayons (initial_crayons final_crayons : ℕ) 
  (h1 : initial_crayons = 87) 
  (h2 : final_crayons = 80) : 
  initial_crayons - final_crayons = 7 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_crayons_l1663_166331


namespace NUMINAMATH_CALUDE_oil_production_per_capita_correct_l1663_166366

/-- Oil production per capita for a region -/
structure OilProductionPerCapita where
  region : String
  value : Float

/-- Given oil production per capita data -/
def given_data : List OilProductionPerCapita := [
  ⟨"West", 55.084⟩,
  ⟨"Non-West", 214.59⟩,
  ⟨"Russia", 1038.33⟩
]

/-- Theorem: The oil production per capita for West, Non-West, and Russia are as given -/
theorem oil_production_per_capita_correct :
  ∀ region value, OilProductionPerCapita.mk region value ∈ given_data →
  (region = "West" → value = 55.084) ∧
  (region = "Non-West" → value = 214.59) ∧
  (region = "Russia" → value = 1038.33) :=
by sorry

end NUMINAMATH_CALUDE_oil_production_per_capita_correct_l1663_166366


namespace NUMINAMATH_CALUDE_min_area_MAB_l1663_166321

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define point F
def F : ℝ × ℝ := (0, 1)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the area of triangle MAB
def area_MAB (k : ℝ) : ℝ := 4*(1 + k^2)^(3/2)

-- State the theorem
theorem min_area_MAB :
  ∃ (min_area : ℝ), min_area = 4 ∧
  ∀ (k : ℝ), area_MAB k ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_area_MAB_l1663_166321


namespace NUMINAMATH_CALUDE_painting_time_equation_l1663_166358

/-- The time it takes Doug to paint the room alone, in hours -/
def doug_time : ℝ := 5

/-- The time it takes Dave to paint the room alone, in hours -/
def dave_time : ℝ := 7

/-- The number of one-hour breaks taken -/
def breaks : ℝ := 2

/-- The total time it takes Doug and Dave to paint the room together, including breaks -/
noncomputable def total_time : ℝ := sorry

/-- Theorem stating that the equation (1/5 + 1/7)(t - 2) = 1 is satisfied by the total time -/
theorem painting_time_equation : 
  (1 / doug_time + 1 / dave_time) * (total_time - breaks) = 1 := by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l1663_166358


namespace NUMINAMATH_CALUDE_termite_ridden_homes_l1663_166393

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden_homes : ℝ) 
  (h1 : termite_ridden_homes > 0)
  (h2 : (4 : ℝ) / 7 * termite_ridden_homes = termite_ridden_homes - (1 : ℝ) / 7 * total_homes) :
  termite_ridden_homes = (1 : ℝ) / 3 * total_homes := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_homes_l1663_166393


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_three_l1663_166349

/-- The function f(x) = (x^3 + x^2 + 1) / (x - 3) has a vertical asymptote at x = 3 -/
theorem vertical_asymptote_at_three (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^3 + x^2 + 1) / (x - 3)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ → δ < ε → |f (3 + δ)| > (1 / δ) ∧ |f (3 - δ)| > (1 / δ) :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_three_l1663_166349


namespace NUMINAMATH_CALUDE_probability_sum_five_l1663_166309

/-- The probability of the sum of four standard dice rolls equaling 5 -/
def prob_sum_five : ℚ := 1 / 324

/-- The number of faces on a standard die -/
def standard_die_faces : ℕ := 6

/-- The minimum value on a standard die -/
def min_die_value : ℕ := 1

/-- The maximum value on a standard die -/
def max_die_value : ℕ := 6

/-- A function representing a valid die roll -/
def valid_roll (n : ℕ) : Prop := min_die_value ≤ n ∧ n ≤ max_die_value

/-- The sum we're looking for -/
def target_sum : ℕ := 5

/-- The number of dice rolled -/
def num_dice : ℕ := 4

theorem probability_sum_five :
  ∀ (a b c d : ℕ), valid_roll a → valid_roll b → valid_roll c → valid_roll d →
  (a + b + c + d = target_sum) →
  (prob_sum_five = (↑(Nat.choose num_dice 1) / ↑(standard_die_faces ^ num_dice) : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l1663_166309


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1663_166361

theorem arithmetic_equality : 142 + 29 - 32 + 25 = 164 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1663_166361


namespace NUMINAMATH_CALUDE_age_difference_proof_l1663_166328

/-- The age difference between Mandy and Sarah --/
def age_difference : ℕ := by sorry

theorem age_difference_proof (mandy_age tom_age julia_age max_age sarah_age : ℕ) 
  (h1 : mandy_age = 3)
  (h2 : tom_age = 4 * mandy_age)
  (h3 : julia_age = tom_age - 5)
  (h4 : max_age = 2 * julia_age)
  (h5 : sarah_age = 3 * max_age - 1) :
  sarah_age - mandy_age = age_difference := by sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1663_166328


namespace NUMINAMATH_CALUDE_fraction_power_product_l1663_166323

theorem fraction_power_product : (2 / 3 : ℚ)^4 * (1 / 5 : ℚ)^2 = 16 / 2025 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1663_166323


namespace NUMINAMATH_CALUDE_total_pears_picked_l1663_166396

theorem total_pears_picked (alyssa_pears nancy_pears : ℕ) 
  (h1 : alyssa_pears = 42) 
  (h2 : nancy_pears = 17) : 
  alyssa_pears + nancy_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l1663_166396


namespace NUMINAMATH_CALUDE_vlecks_for_45_degrees_l1663_166382

/-- The number of vlecks in a full circle on Venus. -/
def full_circle_vlecks : ℕ := 600

/-- The number of degrees in a full circle on Earth. -/
def full_circle_degrees : ℕ := 360

/-- Converts an angle in degrees to vlecks. -/
def degrees_to_vlecks (degrees : ℚ) : ℚ :=
  (degrees / full_circle_degrees) * full_circle_vlecks

/-- Theorem: 45 degrees corresponds to 75 vlecks on Venus. -/
theorem vlecks_for_45_degrees : degrees_to_vlecks 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_vlecks_for_45_degrees_l1663_166382


namespace NUMINAMATH_CALUDE_legs_on_ground_l1663_166372

theorem legs_on_ground (num_horses : ℕ) (num_men : ℕ) (num_riding : ℕ) : 
  num_horses = 8 →
  num_men = num_horses →
  num_riding = num_men / 2 →
  (4 * num_horses + 2 * (num_men - num_riding)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_legs_on_ground_l1663_166372


namespace NUMINAMATH_CALUDE_intersection_M_N_l1663_166398

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | Real.log x / Real.log (1/2) > -1}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1663_166398


namespace NUMINAMATH_CALUDE_total_harvest_kg_l1663_166311

def apple_sections : ℕ := 8
def apple_yield_per_section : ℕ := 450

def orange_sections : ℕ := 10
def orange_crates_per_section : ℕ := 60
def orange_kg_per_crate : ℕ := 8

def peach_sections : ℕ := 3
def peach_sacks_per_section : ℕ := 55
def peach_kg_per_sack : ℕ := 12

def cherry_fields : ℕ := 5
def cherry_baskets_per_field : ℕ := 50
def cherry_kg_per_basket : ℚ := 3.5

theorem total_harvest_kg : 
  apple_sections * apple_yield_per_section + 
  orange_sections * orange_crates_per_section * orange_kg_per_crate + 
  peach_sections * peach_sacks_per_section * peach_kg_per_sack + 
  cherry_fields * cherry_baskets_per_field * cherry_kg_per_basket = 11255 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_kg_l1663_166311


namespace NUMINAMATH_CALUDE_polygon_area_is_300_l1663_166307

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polygon described in the problem -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨10, 10⟩, ⟨10, 20⟩, ⟨10, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨0, 10⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ :=
  sorry

/-- Theorem: The area of the given polygon is 300 square units -/
theorem polygon_area_is_300 : polygonArea polygon = 300 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_300_l1663_166307


namespace NUMINAMATH_CALUDE_jeff_sunday_morning_laps_l1663_166324

/-- The number of laps Jeff swam on Sunday morning before the break -/
def sunday_morning_laps (total_laps required_laps saturday_laps remaining_laps : ℕ) : ℕ :=
  total_laps - saturday_laps - remaining_laps

theorem jeff_sunday_morning_laps :
  sunday_morning_laps 98 27 56 = 15 := by
  sorry

end NUMINAMATH_CALUDE_jeff_sunday_morning_laps_l1663_166324


namespace NUMINAMATH_CALUDE_paco_initial_sweet_cookies_l1663_166383

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := sorry

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The number of sweet cookies Paco had left -/
def remaining_sweet_cookies : ℕ := 7

/-- Theorem: Paco had 22 sweet cookies initially -/
theorem paco_initial_sweet_cookies :
  initial_sweet_cookies = eaten_sweet_cookies + remaining_sweet_cookies ∧
  initial_sweet_cookies = 22 :=
by sorry

end NUMINAMATH_CALUDE_paco_initial_sweet_cookies_l1663_166383


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_l1663_166327

/-- The capacity of a gunny bag filled with wheat packets -/
theorem gunny_bag_capacity
  (pounds_per_ton : ℕ)
  (ounces_per_pound : ℕ)
  (num_packets : ℕ)
  (packet_weight_pounds : ℕ)
  (packet_weight_ounces : ℕ)
  (h1 : pounds_per_ton = 2200)
  (h2 : ounces_per_pound = 16)
  (h3 : num_packets = 1760)
  (h4 : packet_weight_pounds = 16)
  (h5 : packet_weight_ounces = 4) :
  (num_packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound : ℚ)) / pounds_per_ton = 13 := by
  sorry


end NUMINAMATH_CALUDE_gunny_bag_capacity_l1663_166327


namespace NUMINAMATH_CALUDE_fiftieth_islander_statement_l1663_166320

/-- Represents the type of islander: Knight (always tells the truth) or Liar (always lies) -/
inductive IslanderType
| Knight
| Liar

/-- Represents what an islander says about their neighbor -/
inductive Statement
| Knight
| Liar

/-- A function that determines what an islander at a given position says about their right neighbor -/
def whatTheySay (position : Nat) : Statement :=
  if position % 2 = 1 then Statement.Knight else Statement.Liar

/-- The main theorem to prove -/
theorem fiftieth_islander_statement :
  ∀ (islanders : Fin 50 → IslanderType),
  (∀ (i : Fin 50), 
    (islanders i = IslanderType.Knight → whatTheySay i.val = Statement.Knight → islanders (i + 1) = IslanderType.Knight) ∧
    (islanders i = IslanderType.Liar → whatTheySay i.val = Statement.Knight → islanders (i + 1) = IslanderType.Liar) ∧
    (islanders i = IslanderType.Knight → whatTheySay i.val = Statement.Liar → islanders (i + 1) = IslanderType.Liar) ∧
    (islanders i = IslanderType.Liar → whatTheySay i.val = Statement.Liar → islanders (i + 1) = IslanderType.Knight)) →
  whatTheySay 50 = Statement.Knight :=
sorry

end NUMINAMATH_CALUDE_fiftieth_islander_statement_l1663_166320


namespace NUMINAMATH_CALUDE_field_trip_adults_l1663_166314

/-- The number of adults going on a field trip --/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 7 → num_students = 33 → num_vans = 6 → 
  (num_vans * van_capacity) - num_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1663_166314


namespace NUMINAMATH_CALUDE_equal_prob_when_four_prob_when_six_l1663_166315

-- Define the set of paper slips
def slips : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the probability of winning for Xiao Ming and Xiao Ying given Xiao Ming's draw
def win_prob (xiao_ming_draw : ℕ) : ℚ × ℚ :=
  let remaining_slips := slips.erase xiao_ming_draw
  let xiao_ming_wins := (remaining_slips.filter (· < xiao_ming_draw)).card
  let xiao_ying_wins := (remaining_slips.filter (· > xiao_ming_draw)).card
  (xiao_ming_wins / remaining_slips.card, xiao_ying_wins / remaining_slips.card)

-- Theorem 1: When Xiao Ming draws 4, both have equal probability of winning
theorem equal_prob_when_four : win_prob 4 = (1/2, 1/2) := by sorry

-- Theorem 2: When Xiao Ming draws 6, probabilities are 5/6 and 1/6
theorem prob_when_six : win_prob 6 = (5/6, 1/6) := by sorry

end NUMINAMATH_CALUDE_equal_prob_when_four_prob_when_six_l1663_166315


namespace NUMINAMATH_CALUDE_kangaroo_equality_days_l1663_166350

/-- The number of days it takes for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos bert_kangaroos bert_daily_rate : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_daily_rate

/-- Theorem stating that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem kangaroo_equality_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

#eval days_to_equal_kangaroos 100 20 2

end NUMINAMATH_CALUDE_kangaroo_equality_days_l1663_166350


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_squares_multiple_of_360_l1663_166385

theorem smallest_k_for_sum_squares_multiple_of_360 :
  ∃ k : ℕ+, (k.val * (k.val + 1) * (2 * k.val + 1)) % 2160 = 0 ∧
  ∀ m : ℕ+, m < k → (m.val * (m.val + 1) * (2 * m.val + 1)) % 2160 ≠ 0 ∧
  k = 175 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_squares_multiple_of_360_l1663_166385


namespace NUMINAMATH_CALUDE_algebraic_identities_l1663_166367

theorem algebraic_identities (x y : ℝ) : 
  ((2*x - 3*y)^2 = 4*x^2 - 12*x*y + 9*y^2) ∧ 
  ((x + y)*(x + y)*(x^2 + y^2) = x^4 + 2*x^2*y^2 + y^4 + 2*x^3*y + 2*x*y^3) := by
sorry

end NUMINAMATH_CALUDE_algebraic_identities_l1663_166367


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1663_166371

/-- Proves that the ratio of Tom's current age to the number of years ago when his age was three times the sum of his children's ages is 5.5 -/
theorem toms_age_ratio :
  ∀ (T N : ℝ),
  (∃ (a b c d : ℝ), T = a + b + c + d) →  -- T is the sum of four children's ages
  (T - N = 3 * (T - 4 * N)) →              -- N years ago condition
  T / N = 5.5 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1663_166371


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l1663_166317

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 7391 ≡ 167 [ZMOD 12] ∧
  ∀ y : ℕ+, y.val + 7391 ≡ 167 [ZMOD 12] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l1663_166317


namespace NUMINAMATH_CALUDE_similar_triangles_leg_ratio_l1663_166300

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs x and 7,
    prove that x = 84/9 -/
theorem similar_triangles_leg_ratio (x : ℝ) : 
  (12 : ℝ) / x = 9 / 7 → x = 84 / 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_ratio_l1663_166300


namespace NUMINAMATH_CALUDE_polygon_arrangement_sides_l1663_166337

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents the arrangement of polygons as described in the problem. -/
structure PolygonArrangement where
  pentagon : RegularPolygon
  triangle : RegularPolygon
  heptagon : RegularPolygon
  nonagon : RegularPolygon
  dodecagon : RegularPolygon
  pentagon_sides : pentagon.sides = 5
  triangle_sides : triangle.sides = 3
  heptagon_sides : heptagon.sides = 7
  nonagon_sides : nonagon.sides = 9
  dodecagon_sides : dodecagon.sides = 12

/-- The number of exposed sides in the polygon arrangement. -/
def exposed_sides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.pentagon.sides + arrangement.triangle.sides + arrangement.heptagon.sides +
  arrangement.nonagon.sides + arrangement.dodecagon.sides - 7

theorem polygon_arrangement_sides (arrangement : PolygonArrangement) :
  exposed_sides arrangement = 28 := by
  sorry

end NUMINAMATH_CALUDE_polygon_arrangement_sides_l1663_166337


namespace NUMINAMATH_CALUDE_trajectory_max_value_l1663_166343

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x + 1)^2 + (4/3) * y^2 = 4

/-- The distance ratio condition -/
def distance_ratio (x y : ℝ) : Prop :=
  (x^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

theorem trajectory_max_value :
  ∀ x y : ℝ, 
    distance_ratio x y → 
    trajectory x y → 
    2 * x^2 + y^2 ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_trajectory_max_value_l1663_166343


namespace NUMINAMATH_CALUDE_pawsitive_training_center_dogs_l1663_166336

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  fetch : ℕ
  roll_over : ℕ
  sit_stay : ℕ
  sit_fetch : ℕ
  sit_roll : ℕ
  stay_fetch : ℕ
  stay_roll : ℕ
  fetch_roll : ℕ
  sit_stay_fetch : ℕ
  sit_stay_roll : ℕ
  sit_fetch_roll : ℕ
  stay_fetch_roll : ℕ
  all_four : ℕ
  none : ℕ

/-- Calculates the total number of dogs at the Pawsitive Training Center -/
def total_dogs (d : DogTricks) : ℕ := sorry

/-- Theorem stating that given the conditions, the total number of dogs is 135 -/
theorem pawsitive_training_center_dogs :
  let d : DogTricks := {
    sit := 60, stay := 35, fetch := 45, roll_over := 40,
    sit_stay := 20, sit_fetch := 15, sit_roll := 10,
    stay_fetch := 5, stay_roll := 8, fetch_roll := 6,
    sit_stay_fetch := 4, sit_stay_roll := 3,
    sit_fetch_roll := 2, stay_fetch_roll := 1,
    all_four := 2, none := 12
  }
  total_dogs d = 135 := by sorry

end NUMINAMATH_CALUDE_pawsitive_training_center_dogs_l1663_166336


namespace NUMINAMATH_CALUDE_condition1_condition2_max_type_A_dictionaries_l1663_166353

/-- The price of dictionary A -/
def price_A : ℝ := 70

/-- The price of dictionary B -/
def price_B : ℝ := 50

/-- The total number of dictionaries to be purchased -/
def total_dictionaries : ℕ := 300

/-- The maximum total cost -/
def max_cost : ℝ := 16000

/-- Verification of the first condition -/
theorem condition1 : price_A + 2 * price_B = 170 := by sorry

/-- Verification of the second condition -/
theorem condition2 : 2 * price_A + 3 * price_B = 290 := by sorry

/-- The main theorem proving the maximum number of type A dictionaries -/
theorem max_type_A_dictionaries : 
  ∀ m : ℕ, m ≤ total_dictionaries ∧ 
    m * price_A + (total_dictionaries - m) * price_B ≤ max_cost → 
    m ≤ 50 := by sorry

end NUMINAMATH_CALUDE_condition1_condition2_max_type_A_dictionaries_l1663_166353


namespace NUMINAMATH_CALUDE_four_integers_sum_l1663_166379

theorem four_integers_sum (a b c d : ℤ) :
  a + b + c = 6 ∧
  a + b + d = 7 ∧
  a + c + d = 8 ∧
  b + c + d = 9 →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_four_integers_sum_l1663_166379


namespace NUMINAMATH_CALUDE_kims_test_probability_l1663_166341

theorem kims_test_probability (p_english : ℝ) (p_history : ℝ) 
  (h_english : p_english = 5/9)
  (h_history : p_history = 1/3)
  (h_independent : True) -- We don't need to explicitly define independence in this statement
  : (1 - p_english) * p_history = 4/27 := by
  sorry

end NUMINAMATH_CALUDE_kims_test_probability_l1663_166341


namespace NUMINAMATH_CALUDE_davids_age_twice_daughters_l1663_166306

/-- 
Given:
- David is currently 40 years old
- David's daughter is currently 12 years old

Prove that 16 years will pass before David's age is twice his daughter's age
-/
theorem davids_age_twice_daughters (david_age : ℕ) (daughter_age : ℕ) :
  david_age = 40 →
  daughter_age = 12 →
  ∃ (years : ℕ), david_age + years = 2 * (daughter_age + years) ∧ years = 16 :=
by sorry

end NUMINAMATH_CALUDE_davids_age_twice_daughters_l1663_166306


namespace NUMINAMATH_CALUDE_inequality_bound_l1663_166325

theorem inequality_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) + 
  Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_bound_l1663_166325


namespace NUMINAMATH_CALUDE_peanuts_in_box_l1663_166381

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 4

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 8

/-- The total number of peanuts in the box after Mary adds more -/
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_in_box : total_peanuts = 12 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l1663_166381


namespace NUMINAMATH_CALUDE_sum_of_roots_l1663_166305

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 18*c^2 + 27*c - 100 = 0)
  (hd : 9*d^3 - 81*d^2 - 324*d + 3969 = 0) : 
  c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1663_166305


namespace NUMINAMATH_CALUDE_no_valid_tiling_l1663_166369

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  rows : ℕ
  cols : ℕ

/-- Represents a domino with given dimensions -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Represents a tiling configuration -/
structure Tiling where
  rect : Rectangle
  dominos : List Domino
  count : ℕ

def is_valid_tiling (t : Tiling) : Prop :=
  t.rect.rows = 11 ∧
  t.rect.cols = 12 ∧
  t.count = 19 ∧
  ∀ d ∈ t.dominos, (d.length = 6 ∧ d.width = 1) ∨ (d.length = 7 ∧ d.width = 1) ∨
                   (d.length = 1 ∧ d.width = 6) ∨ (d.length = 1 ∧ d.width = 7)

theorem no_valid_tiling :
  ¬ ∃ t : Tiling, is_valid_tiling t := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tiling_l1663_166369


namespace NUMINAMATH_CALUDE_equal_probability_for_all_l1663_166330

/-- Represents the sampling method used in the TV show -/
structure SamplingMethod where
  total_population : ℕ
  sample_size : ℕ
  removed_first : ℕ
  
/-- The probability of being selected for each individual in the population -/
def selection_probability (sm : SamplingMethod) : ℚ :=
  sm.sample_size / sm.total_population

/-- The specific sampling method used in the TV show -/
def tv_show_sampling : SamplingMethod := {
  total_population := 2014
  sample_size := 50
  removed_first := 14
}

theorem equal_probability_for_all (sm : SamplingMethod) :
  selection_probability sm = 25 / 1007 :=
sorry

#check equal_probability_for_all tv_show_sampling

end NUMINAMATH_CALUDE_equal_probability_for_all_l1663_166330


namespace NUMINAMATH_CALUDE_earloop_probability_is_0_12_l1663_166362

/-- Represents a mask factory with two types of products -/
structure MaskFactory where
  regularProportion : ℝ
  surgicalProportion : ℝ
  regularEarloopProportion : ℝ
  surgicalEarloopProportion : ℝ

/-- The probability of selecting a mask with ear loops from the factory -/
def earloopProbability (factory : MaskFactory) : ℝ :=
  factory.regularProportion * factory.regularEarloopProportion +
  factory.surgicalProportion * factory.surgicalEarloopProportion

/-- Theorem stating the probability of selecting a mask with ear loops -/
theorem earloop_probability_is_0_12 (factory : MaskFactory)
  (h1 : factory.regularProportion = 0.8)
  (h2 : factory.surgicalProportion = 0.2)
  (h3 : factory.regularEarloopProportion = 0.1)
  (h4 : factory.surgicalEarloopProportion = 0.2) :
  earloopProbability factory = 0.12 := by
  sorry


end NUMINAMATH_CALUDE_earloop_probability_is_0_12_l1663_166362


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_real_l1663_166394

theorem arithmetic_geometric_mean_ratio_real (A B : ℂ) :
  (∃ r : ℝ, (A + B) / 2 = r * (A * B)^(1/2 : ℂ)) →
  (∃ r : ℝ, A = r * B) ∨ Complex.abs A = Complex.abs B :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_real_l1663_166394


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1663_166374

theorem negative_cube_squared (a b : ℝ) : (-a^3 * b)^2 = a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1663_166374


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1663_166387

/-- Determines if the equation x²/(k-4) - y²/(k+4) = 1 represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop := (k - 4) * (k + 4) > 0

theorem sufficient_but_not_necessary :
  (∀ k : ℝ, k ≤ -5 → is_hyperbola k) ∧
  (∃ k : ℝ, k > -5 ∧ is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1663_166387


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_197_l1663_166333

theorem first_nonzero_digit_after_decimal_1_197 : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (1000 : ℚ) / 197 = (5 : ℚ) + (d : ℚ) / (10 : ℚ) ^ (n + 1) + (1 : ℚ) / (10 : ℚ) ^ (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_197_l1663_166333


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1663_166391

theorem smallest_multiple_of_6_and_15 :
  ∃ b : ℕ, b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ c : ℕ, c > 0 → 6 ∣ c → 15 ∣ c → b ≤ c :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1663_166391


namespace NUMINAMATH_CALUDE_system_solution_l1663_166316

theorem system_solution : 
  ∃ (j k : ℚ), (7 * j - 35 * k = -3) ∧ (3 * j - 2 * k = 5) ∧ (j = 547/273) ∧ (k = 44/91) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1663_166316


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l1663_166389

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x y, l ≤ x → x < y → y ≤ r → f x < f y

-- State the theorem
theorem a_equals_one_sufficient_not_necessary :
  (∀ x y, 2 ≤ x → x < y → is_increasing_on (f 1) 2 y) ∧
  ¬(∀ a : ℝ, (∀ x y, 2 ≤ x → x < y → is_increasing_on (f a) 2 y) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l1663_166389
