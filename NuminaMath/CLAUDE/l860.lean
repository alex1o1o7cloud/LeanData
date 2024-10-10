import Mathlib

namespace gina_netflix_minutes_l860_86069

/-- The number of shows Gina's sister watches per week -/
def sister_shows : ℕ := 24

/-- The length of each show in minutes -/
def show_length : ℕ := 50

/-- The ratio of shows Gina chooses compared to her sister -/
def gina_ratio : ℕ := 3

/-- The total number of shows watched by both Gina and her sister -/
def total_shows : ℕ := sister_shows * (gina_ratio + 1)

/-- The number of shows Gina chooses -/
def gina_shows : ℕ := total_shows * gina_ratio / (gina_ratio + 1)

theorem gina_netflix_minutes : gina_shows * show_length = 900 :=
by sorry

end gina_netflix_minutes_l860_86069


namespace fountain_distance_is_30_l860_86095

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℕ :=
  total_distance / num_trips

/-- Theorem stating that the distance to the water fountain is 30 feet -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end fountain_distance_is_30_l860_86095


namespace radish_patch_size_proof_l860_86064

/-- The size of a pea patch in square feet -/
def pea_patch_size : ℝ := 30

/-- The size of a radish patch in square feet -/
def radish_patch_size : ℝ := 15

theorem radish_patch_size_proof :
  (pea_patch_size = 2 * radish_patch_size) ∧
  (pea_patch_size / 6 = 5) →
  radish_patch_size = 15 := by
  sorry

end radish_patch_size_proof_l860_86064


namespace not_prime_m_plus_n_minus_one_l860_86050

theorem not_prime_m_plus_n_minus_one (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2)
  (h3 : (m + n - 1) ∣ (m^2 + n^2 - 1)) : ¬ Nat.Prime (m + n - 1) := by
  sorry

end not_prime_m_plus_n_minus_one_l860_86050


namespace daps_to_dips_l860_86086

/-- The number of daps equivalent to one dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dops equivalent to one dip -/
def dops_per_dip : ℚ := 3 / 11

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 66

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_to_dips : daps_per_dop * dops_per_dip⁻¹ * target_dips = 45 / 2 :=
by sorry

end daps_to_dips_l860_86086


namespace insufficient_info_for_unique_height_l860_86043

/-- Represents the relationship between height and shadow length -/
noncomputable def height_shadow_relation (a b : ℝ) (s : ℝ) : ℝ :=
  a * Real.sqrt s + b * s

theorem insufficient_info_for_unique_height :
  ∀ (a₁ b₁ a₂ b₂ : ℝ),
  (height_shadow_relation a₁ b₁ 40.25 = 17.5) →
  (height_shadow_relation a₂ b₂ 40.25 = 17.5) →
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂) →
  ∃ (h₁ h₂ : ℝ), 
    h₁ ≠ h₂ ∧ 
    height_shadow_relation a₁ b₁ 28.75 = h₁ ∧
    height_shadow_relation a₂ b₂ 28.75 = h₂ :=
by sorry

end insufficient_info_for_unique_height_l860_86043


namespace salary_percent_increase_l860_86011

theorem salary_percent_increase 
  (original_salary new_salary increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : increase = 25000)
  (h3 : original_salary = new_salary - increase) :
  (increase / original_salary) * 100 = (25000 / (90000 - 25000)) * 100 := by
sorry

end salary_percent_increase_l860_86011


namespace cubic_real_root_l860_86021

theorem cubic_real_root (c d : ℝ) (h : c ≠ 0) :
  (∃ z : ℂ, c * z^3 + 5 * z^2 + d * z - 104 = 0 ∧ z = -3 - 4*I) →
  (∃ x : ℝ, c * x^3 + 5 * x^2 + d * x - 104 = 0 ∧ x = 1) :=
by sorry

end cubic_real_root_l860_86021


namespace smallest_positive_multiple_of_45_l860_86044

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end smallest_positive_multiple_of_45_l860_86044


namespace f_properties_l860_86022

noncomputable def f (x : ℝ) : ℝ := (1 / (2^x - 1) + 1/2) * x^3

theorem f_properties :
  (∀ x, x ≠ 0 → f x ≠ 0) ∧
  (∀ x, x ≠ 0 → f (-x) = f x) :=
by sorry

end f_properties_l860_86022


namespace total_eggs_january_l860_86030

/-- Represents a hen with a specific egg-laying frequency -/
structure Hen where
  frequency : ℕ  -- Number of days between each egg

/-- Calculates the number of eggs laid by a hen in a given number of days -/
def eggsLaid (h : Hen) (days : ℕ) : ℕ :=
  (days + h.frequency - 1) / h.frequency

/-- The three hens owned by Xiao Ming's family -/
def hens : List Hen := [
  { frequency := 1 },  -- First hen lays an egg every day
  { frequency := 2 },  -- Second hen lays an egg every two days
  { frequency := 3 }   -- Third hen lays an egg every three days
]

/-- The total number of eggs laid by all hens in January -/
def totalEggsInJanuary : ℕ :=
  (hens.map (eggsLaid · 31)).sum

theorem total_eggs_january : totalEggsInJanuary = 56 := by
  sorry

#eval totalEggsInJanuary  -- This should output 56

end total_eggs_january_l860_86030


namespace error_clock_correct_time_l860_86072

/-- Represents a 12-hour digital clock with a display error -/
structure ErrorClock where
  /-- The number of hours in the clock cycle -/
  total_hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The number of hours affected by the display error -/
  incorrect_hours : Nat
  /-- The number of minutes per hour affected by the display error -/
  incorrect_minutes : Nat

/-- The fraction of the day when the ErrorClock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : Rat :=
  ((clock.total_hours - clock.incorrect_hours) * (clock.minutes_per_hour - clock.incorrect_minutes)) / 
  (clock.total_hours * clock.minutes_per_hour)

/-- The specific ErrorClock instance for the problem -/
def problem_clock : ErrorClock :=
  { total_hours := 12
  , minutes_per_hour := 60
  , incorrect_hours := 4
  , incorrect_minutes := 15 }

theorem error_clock_correct_time :
  correct_time_fraction problem_clock = 1/2 := by
  sorry

end error_clock_correct_time_l860_86072


namespace largest_integer_less_than_x_l860_86071

theorem largest_integer_less_than_x (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x ∧ x < 18)
  (h3 : x < 13)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∃ (y : ℤ), x > y ∧ ∀ (z : ℤ), x > z → z ≤ y ∧ y = 9 :=
by sorry

end largest_integer_less_than_x_l860_86071


namespace division_problem_l860_86076

theorem division_problem : (88 : ℚ) / ((4 : ℚ) / 2) = 44 := by
  sorry

end division_problem_l860_86076


namespace nested_sqrt_solution_l860_86073

/-- The positive solution to the nested square root equation -/
theorem nested_sqrt_solution : 
  ∃! (x : ℝ), x > 0 ∧ 
  (∃ (z : ℝ), z > 0 ∧ z = Real.sqrt (x + z)) ∧
  (∃ (y : ℝ), y > 0 ∧ y = Real.sqrt (x * y)) ∧
  x = 2 := by
sorry

end nested_sqrt_solution_l860_86073


namespace station_length_l860_86059

/-- The length of a station given a train passing through it -/
theorem station_length (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 36 →
  passing_time = 45 →
  (train_speed_kmh * 1000 / 3600) * passing_time - train_length = 200 :=
by sorry

end station_length_l860_86059


namespace small_s_conference_teams_l860_86068

-- Define the number of games in the tournament
def num_games : ℕ := 36

-- Define the function to calculate the number of games for n teams
def games_for_teams (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem small_s_conference_teams :
  ∃ (n : ℕ), n > 0 ∧ games_for_teams n = num_games ∧ n = 9 := by
  sorry

end small_s_conference_teams_l860_86068


namespace complex_quotient_real_l860_86005

theorem complex_quotient_real (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (r : ℝ), z₁ / z₂ = r) → a = -3/2 := by sorry

end complex_quotient_real_l860_86005


namespace lcm_gcf_ratio_160_420_l860_86098

theorem lcm_gcf_ratio_160_420 : 
  (Nat.lcm 160 420) / (Nat.gcd 160 420 - 2) = 187 := by sorry

end lcm_gcf_ratio_160_420_l860_86098


namespace expression_zero_iff_x_eq_three_l860_86045

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) →
  ((x^2 - 6*x + 9) / (4*x - 8) = 0 ↔ x = 3) :=
by sorry

end expression_zero_iff_x_eq_three_l860_86045


namespace gcd_problem_l860_86031

theorem gcd_problem : ∃ b : ℕ+, Nat.gcd (20 * b) (18 * 24) = 2 := by
  sorry

end gcd_problem_l860_86031


namespace harry_hours_formula_l860_86035

/-- Represents the payment structure and hours worked for Harry and James -/
structure PaymentSystem where
  x : ℝ  -- Base hourly rate
  S : ℝ  -- Number of hours James is paid at regular rate
  H : ℝ  -- Number of hours Harry worked

/-- Calculates Harry's pay for the week -/
def harry_pay (p : PaymentSystem) : ℝ :=
  18 * p.x + 1.5 * p.x * (p.H - 18)

/-- Calculates James' pay for the week -/
def james_pay (p : PaymentSystem) : ℝ :=
  p.S * p.x + 2 * p.x * (41 - p.S)

/-- Theorem stating the relationship between Harry's hours worked and James' regular hours -/
theorem harry_hours_formula (p : PaymentSystem) :
  harry_pay p = james_pay p →
  p.H = (91 - 3 * p.S) / 1.5 := by
  sorry

end harry_hours_formula_l860_86035


namespace determine_phi_l860_86065

-- Define the functions and constants
noncomputable def ω : ℝ := 2
noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (ω * x + φ)
noncomputable def g (x : ℝ) := Real.cos (ω * x)

-- State the theorem
theorem determine_phi :
  (ω > 0) →
  (∀ φ, |φ| < π / 2 →
    (∀ x, f x φ = f (x + π) φ) →
    (∀ x, f (x - 2*π/3) φ = g x)) →
  ∃ φ, φ = -π / 6 :=
by sorry

end determine_phi_l860_86065


namespace set_union_complement_and_subset_necessary_not_sufficient_condition_l860_86081

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def B (a : ℝ) : Set ℝ := {x | -a - 1 < x ∧ x < -a + 1}

theorem set_union_complement_and_subset (a : ℝ) :
  a = 3 → (Set.univ \ A) ∪ B a = {x | x < -2 ∨ x ≥ 1} :=
sorry

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x, x ∈ B a → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B a) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end set_union_complement_and_subset_necessary_not_sufficient_condition_l860_86081


namespace factorial_800_trailing_zeros_l860_86040

/-- Counts the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The number of trailing zeros in 800! is 199 -/
theorem factorial_800_trailing_zeros : trailingZeros 800 = 199 := by
  sorry

end factorial_800_trailing_zeros_l860_86040


namespace sum_of_polynomials_l860_86058

-- Define the polynomials
def f (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 4 * x^2 + 6 * x + 3
def j (x : ℝ) : ℝ := 3 * x^2 - x + 2

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x + j x = -x^2 + 11 * x - 9 := by
  sorry

end sum_of_polynomials_l860_86058


namespace households_with_bike_only_l860_86054

theorem households_with_bike_only 
  (total : ℕ) 
  (without_car_or_bike : ℕ) 
  (with_both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 18)
  (h4 : with_car = 44) :
  total - without_car_or_bike - (with_car - with_both) - with_both = 35 := by
  sorry

end households_with_bike_only_l860_86054


namespace equation_solution_l860_86079

theorem equation_solution : ∃ (z₁ z₂ : ℂ), 
  z₁ = (-1 + Complex.I * Real.sqrt 21) / 2 ∧
  z₂ = (-1 - Complex.I * Real.sqrt 21) / 2 ∧
  ∀ x : ℂ, (4 * x^2 + 3 * x + 1) / (x - 2) = 2 * x + 5 ↔ x = z₁ ∨ x = z₂ := by
sorry

end equation_solution_l860_86079


namespace three_numbers_sum_l860_86097

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →  -- Ascending order
  y = 7 →  -- Median is 7
  (x + y + z) / 3 = x + 12 →  -- Mean is 12 more than least
  (x + y + z) / 3 = z - 18 →  -- Mean is 18 less than greatest
  x + y + z = 39 := by
sorry

end three_numbers_sum_l860_86097


namespace quadratic_roots_sum_and_product_l860_86041

theorem quadratic_roots_sum_and_product (m n : ℝ) : 
  (m^2 - 4*m = 12) → (n^2 - 4*n = 12) → m + n + m*n = -8 := by
  sorry

end quadratic_roots_sum_and_product_l860_86041


namespace locus_equals_thales_circles_l860_86015

/-- A triangle in a plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in the plane -/
def Point : Type := ℝ × ℝ

/-- The angle subtended by a side of the triangle from a point -/
noncomputable def subtended_angle (t : Triangle) (p : Point) (side : Fin 3) : ℝ :=
  sorry

/-- The sum of angles subtended by the three sides of the triangle from a point -/
noncomputable def sum_of_subtended_angles (t : Triangle) (p : Point) : ℝ :=
  (subtended_angle t p 0) + (subtended_angle t p 1) + (subtended_angle t p 2)

/-- The Thales' circle for a side of the triangle -/
def thales_circle (t : Triangle) (side : Fin 3) : Set Point :=
  sorry

/-- The set of points on all Thales' circles, excluding the triangle's vertices -/
def thales_circles_points (t : Triangle) : Set Point :=
  (thales_circle t 0 ∪ thales_circle t 1 ∪ thales_circle t 2) \ {t.A, t.B, t.C}

/-- The theorem stating the equivalence of the locus and the Thales' circles points -/
theorem locus_equals_thales_circles (t : Triangle) :
  {p : Point | sum_of_subtended_angles t p = π} = thales_circles_points t :=
  sorry

end locus_equals_thales_circles_l860_86015


namespace arithmetic_square_root_of_nine_l860_86078

theorem arithmetic_square_root_of_nine (x : ℝ) :
  (x ≥ 0 ∧ x ^ 2 = 9) → x = 3 := by
  sorry

end arithmetic_square_root_of_nine_l860_86078


namespace pipe_fill_time_l860_86016

/-- The time it takes for the second pipe to empty the tank -/
def empty_time : ℝ := 24

/-- The time after which the second pipe is closed when both pipes are open -/
def close_time : ℝ := 48

/-- The total time it takes to fill the tank -/
def total_fill_time : ℝ := 30

/-- The time it takes for the first pipe to fill the tank -/
def fill_time : ℝ := 22

theorem pipe_fill_time : 
  (close_time * (1 / fill_time - 1 / empty_time) + (total_fill_time - close_time) * (1 / fill_time) = 1) →
  fill_time = 22 := by sorry

end pipe_fill_time_l860_86016


namespace num_expressions_correct_l860_86052

/-- The number of algebraically different expressions obtained by placing parentheses in a₁ / a₂ / ... / aₙ -/
def num_expressions (n : ℕ) : ℕ :=
  if n ≥ 2 then 2^(n-2) else 0

/-- Theorem stating that for n ≥ 2, the number of algebraically different expressions
    obtained by placing parentheses in a₁ / a₂ / ... / aₙ is equal to 2^(n-2) -/
theorem num_expressions_correct (n : ℕ) (h : n ≥ 2) :
  num_expressions n = 2^(n-2) := by
  sorry

end num_expressions_correct_l860_86052


namespace square_area_to_perimeter_ratio_l860_86051

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₁^2 / s₂^2 = 16 / 49 → (4 * s₁) / (4 * s₂) = 4 / 7 := by
sorry

end square_area_to_perimeter_ratio_l860_86051


namespace profit_rate_equal_with_without_discount_l860_86087

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the profit rate with discount
def profit_rate_with_discount : ℝ := 0.235

-- Theorem statement
theorem profit_rate_equal_with_without_discount :
  profit_rate_with_discount = (1 + profit_rate_with_discount) / (1 - discount_rate) - 1 :=
by sorry

end profit_rate_equal_with_without_discount_l860_86087


namespace simplify_fraction_division_l860_86093

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
sorry

end simplify_fraction_division_l860_86093


namespace quadratic_inequalities_l860_86003

theorem quadratic_inequalities (a : ℝ) :
  ((∀ x : ℝ, x^2 + a*x + 3 ≥ a) ↔ a ∈ Set.Icc (-6) 2) ∧
  ((∃ x : ℝ, x < 1 ∧ x^2 + a*x + 3 ≤ a) ↔ a ∈ Set.Ici 2) :=
sorry

end quadratic_inequalities_l860_86003


namespace worker_earnings_l860_86096

theorem worker_earnings
  (regular_rate : ℝ)
  (total_surveys : ℕ)
  (cellphone_rate_increase : ℝ)
  (cellphone_surveys : ℕ)
  (h1 : regular_rate = 30)
  (h2 : total_surveys = 100)
  (h3 : cellphone_rate_increase = 0.2)
  (h4 : cellphone_surveys = 50) :
  let cellphone_rate := regular_rate * (1 + cellphone_rate_increase)
  let regular_surveys := total_surveys - cellphone_surveys
  let total_earnings := regular_rate * regular_surveys + cellphone_rate * cellphone_surveys
  total_earnings = 3300 :=
by sorry

end worker_earnings_l860_86096


namespace smallest_n_value_l860_86038

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3000 ∧ Even a

def factorial_product_divisibility (a b c n : ℕ) (m : ℤ) : Prop :=
  ∃ (k : ℤ), (a.factorial * b.factorial * c.factorial : ℤ) = m * 10^n ∧ ¬(10 ∣ m)

theorem smallest_n_value (a b c : ℕ) (m : ℤ) :
  is_valid_triple a b c →
  (∃ n : ℕ, factorial_product_divisibility a b c n m) →
  ∃ n : ℕ, factorial_product_divisibility a b c n m ∧
    ∀ k : ℕ, factorial_product_divisibility a b c k m → n ≤ k ∧ n = 496 :=
by sorry

end smallest_n_value_l860_86038


namespace adult_tickets_sold_l860_86019

def adult_ticket_price : ℕ := 5
def child_ticket_price : ℕ := 2
def total_tickets : ℕ := 85
def total_amount : ℕ := 275

theorem adult_tickets_sold (a c : ℕ) : 
  a + c = total_tickets → 
  a * adult_ticket_price + c * child_ticket_price = total_amount → 
  a = 35 := by
sorry

end adult_tickets_sold_l860_86019


namespace triangle_problem_l860_86000

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : Real.tan (π/4 - abc.C) = Real.sqrt 3 - 2)
  (h2 : abc.c = Real.sqrt 7)
  (h3 : abc.a + abc.b = 5) :
  abc.C = π/3 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C = 3 * Real.sqrt 3 / 2) :=
by sorry

end triangle_problem_l860_86000


namespace male_average_is_100_l860_86083

/-- Represents the average number of tickets sold by a group of members -/
structure GroupAverage where
  count : ℕ  -- Number of members in the group
  average : ℝ  -- Average number of tickets sold by the group

/-- Represents the charitable association -/
structure Association where
  male : GroupAverage
  female : GroupAverage
  nonBinary : GroupAverage

/-- The ratio of male to female to non-binary members is 2:3:5 -/
def memberRatio (a : Association) : Prop :=
  a.male.count = 2 * a.female.count / 3 ∧
  a.nonBinary.count = 5 * a.female.count / 3

/-- The average number of tickets sold by all members is 66 -/
def totalAverage (a : Association) : Prop :=
  (a.male.count * a.male.average + a.female.count * a.female.average + a.nonBinary.count * a.nonBinary.average) /
  (a.male.count + a.female.count + a.nonBinary.count) = 66

/-- Main theorem: Given the conditions, prove that the average number of tickets sold by male members is 100 -/
theorem male_average_is_100 (a : Association)
  (h_ratio : memberRatio a)
  (h_total_avg : totalAverage a)
  (h_female_avg : a.female.average = 70)
  (h_nonbinary_avg : a.nonBinary.average = 50) :
  a.male.average = 100 := by
  sorry

end male_average_is_100_l860_86083


namespace oldest_harper_child_age_l860_86010

/-- The age of the oldest Harper child given the ages of the other three and the average age of all four. -/
theorem oldest_harper_child_age 
  (average_age : ℝ) 
  (younger_child1 : ℕ) 
  (younger_child2 : ℕ) 
  (younger_child3 : ℕ) 
  (h1 : average_age = 9) 
  (h2 : younger_child1 = 6) 
  (h3 : younger_child2 = 8) 
  (h4 : younger_child3 = 10) : 
  ∃ (oldest_child : ℕ), 
    (younger_child1 + younger_child2 + younger_child3 + oldest_child) / 4 = average_age ∧ 
    oldest_child = 12 := by
  sorry

end oldest_harper_child_age_l860_86010


namespace quadratic_equations_solutions_l860_86062

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ x₁^2 + 4*x₁ - 5 = 0 ∧ x₂^2 + 4*x₂ - 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -1 ∧ 3*y₁^2 + 2*y₁ = 1 ∧ 3*y₂^2 + 2*y₂ = 1) :=
by sorry

end quadratic_equations_solutions_l860_86062


namespace a_investment_value_l860_86075

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- Theorem stating that given the conditions of the partnership,
    a's investment is $45,000 --/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 63000)
  (hc : p.c_investment = 72000)
  (hp : p.total_profit = 60000)
  (hcs : p.c_profit_share = 24000) :
  p.a_investment = 45000 :=
sorry

end a_investment_value_l860_86075


namespace coopers_age_l860_86017

theorem coopers_age (cooper_age dante_age maria_age : ℕ) : 
  cooper_age + dante_age + maria_age = 31 →
  dante_age = 2 * cooper_age →
  maria_age = dante_age + 1 →
  cooper_age = 6 := by
sorry

end coopers_age_l860_86017


namespace circumscribed_odd_equal_sides_is_regular_l860_86077

/-- A polygon with an odd number of sides -/
structure OddPolygon where
  n : ℕ
  vertices : Fin (2 * n + 1) → ℝ × ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A polygon is circumscribed around a circle if all its sides are tangent to the circle -/
def isCircumscribed (p : OddPolygon) (c : Circle) : Prop := sorry

/-- All sides of a polygon have equal length -/
def hasEqualSides (p : OddPolygon) : Prop := sorry

/-- A polygon is regular if all its sides have equal length and all its angles are equal -/
def isRegular (p : OddPolygon) : Prop := sorry

/-- Main theorem: A circumscribed polygon with an odd number of sides and all sides of equal length is regular -/
theorem circumscribed_odd_equal_sides_is_regular 
  (p : OddPolygon) (c : Circle) 
  (h1 : isCircumscribed p c) 
  (h2 : hasEqualSides p) : 
  isRegular p := by sorry

end circumscribed_odd_equal_sides_is_regular_l860_86077


namespace f_decreasing_on_interval_f_one_over_e_gt_f_one_half_l860_86039

noncomputable def f (x : ℝ) : ℝ := -x / Real.exp x + Real.log 2

theorem f_decreasing_on_interval :
  ∀ x : ℝ, x < 1 → ∀ y : ℝ, x < y → f y < f x :=
sorry

theorem f_one_over_e_gt_f_one_half : f (1 / Real.exp 1) > f (1 / 2) :=
sorry

end f_decreasing_on_interval_f_one_over_e_gt_f_one_half_l860_86039


namespace odd_even_properties_l860_86012

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_properties (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g) :
  (∀ x, (|f x| + g x) = (|f (-x)| + g (-x))) ∧
  (∀ x, f x * |g x| = -(f (-x) * |g (-x)|)) :=
sorry

end odd_even_properties_l860_86012


namespace milena_grandfather_age_difference_l860_86023

/-- Calculates the age difference between a child and their grandfather given the child's age,
    the ratio of grandmother's age to child's age, and the age difference between grandparents. -/
def age_difference_child_grandfather (child_age : ℕ) (grandmother_ratio : ℕ) (grandparents_diff : ℕ) : ℕ :=
  (child_age * grandmother_ratio + grandparents_diff) - child_age

/-- The age difference between Milena and her grandfather is 58 years. -/
theorem milena_grandfather_age_difference :
  age_difference_child_grandfather 7 9 2 = 58 := by
  sorry

end milena_grandfather_age_difference_l860_86023


namespace max_value_on_circle_l860_86014

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 20*x + 24*y + 26 → (5*x + 3*y ≤ 73) ∧ ∃ x y, x^2 + y^2 = 20*x + 24*y + 26 ∧ 5*x + 3*y = 73 := by
  sorry

end max_value_on_circle_l860_86014


namespace tom_age_proof_l860_86020

theorem tom_age_proof (tom_age tim_age : ℕ) : 
  (tom_age + tim_age = 21) →
  (tom_age + 3 = 2 * (tim_age + 3)) →
  tom_age = 15 := by
sorry

end tom_age_proof_l860_86020


namespace mary_eggs_l860_86013

/-- Given that Mary starts with 27 eggs and finds 4 more, prove that she ends up with 31 eggs. -/
theorem mary_eggs :
  let initial_eggs : ℕ := 27
  let found_eggs : ℕ := 4
  let final_eggs : ℕ := initial_eggs + found_eggs
  final_eggs = 31 := by
sorry

end mary_eggs_l860_86013


namespace initial_bananas_per_child_l860_86004

/-- Proves that the initial number of bananas per child is 2 --/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : 
  total_children = 780 →
  absent_children = 390 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ), 
    (total_children - absent_children) * (initial_bananas + extra_bananas) = total_children * initial_bananas ∧
    initial_bananas = 2 := by
  sorry

end initial_bananas_per_child_l860_86004


namespace hot_dog_packs_l860_86070

theorem hot_dog_packs (n : ℕ) : 
  (∃ m : ℕ, m < n ∧ 12 * m ≡ 6 [MOD 8]) → 
  12 * n ≡ 6 [MOD 8] → 
  (∀ k : ℕ, k < n → k ≠ n → 12 * k ≡ 6 [MOD 8] → 
    (∃ l : ℕ, l < k ∧ 12 * l ≡ 6 [MOD 8])) → 
  n = 4 := by
sorry

end hot_dog_packs_l860_86070


namespace quadratic_real_equal_roots_l860_86008

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (m - 1) * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (m - 1) * y + 2 * y + 12 = 0 → y = x) ↔ 
  (m = -10 ∨ m = 14) := by
sorry

end quadratic_real_equal_roots_l860_86008


namespace square_difference_equality_l860_86037

theorem square_difference_equality (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_eq : a - b = 4) : 
  a^2 - b^2 = 40 := by
sorry

end square_difference_equality_l860_86037


namespace sphere_volume_ratio_l860_86046

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : 4 * Real.pi * r₁^2 = (2/3) * (4 * Real.pi * r₂^2)) :
  (4/3) * Real.pi * r₁^3 = (2 * Real.sqrt 6 / 9) * ((4/3) * Real.pi * r₂^3) := by
  sorry

end sphere_volume_ratio_l860_86046


namespace no_valid_integers_l860_86099

theorem no_valid_integers : ¬∃ (n : ℤ), ∃ (y : ℤ), 
  (n^2 - 21*n + 110 = y^2) ∧ (∃ (k : ℤ), n = 4*k) :=
by sorry

end no_valid_integers_l860_86099


namespace count_bijections_on_three_element_set_l860_86032

def S : Finset ℕ := {1, 2, 3}

theorem count_bijections_on_three_element_set :
  Fintype.card { f : S → S | Function.Bijective f } = 6 := by
  sorry

end count_bijections_on_three_element_set_l860_86032


namespace problem_statement_l860_86090

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 := by
  sorry

end problem_statement_l860_86090


namespace johns_income_l860_86057

theorem johns_income (john_tax_rate ingrid_tax_rate combined_tax_rate : ℚ)
  (ingrid_income : ℕ) :
  john_tax_rate = 30 / 100 →
  ingrid_tax_rate = 40 / 100 →
  combined_tax_rate = 35625 / 100000 →
  ingrid_income = 72000 →
  ∃ john_income : ℕ,
    john_income = 56000 ∧
    (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) /
      (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end johns_income_l860_86057


namespace line_equation_problem_l860_86042

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 16

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line of symmetry
def symmetry_line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a line
def line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop := ∃ x y : ℝ, unit_circle x y ∧ line m b x y

-- State the theorem
theorem line_equation_problem (M N : ℝ × ℝ) (k : ℝ) :
  (∃ k, symmetry_line k M.1 M.2 ∧ symmetry_line k N.1 N.2) →
  circle_C M.1 M.2 →
  circle_C N.1 N.2 →
  (∃ m b, is_tangent m b ∧ line m b M.1 M.2 ∧ line m b N.1 N.2) →
  ∃ m b, m = 1 ∧ b = 2 ∧ ∀ x y, line m b x y ↔ y = x + 2 :=
sorry

end line_equation_problem_l860_86042


namespace pure_imaginary_solutions_l860_86018

-- Define the polynomial
def p (x : ℂ) : ℂ := x^4 + 2*x^3 + 6*x^2 + 34*x + 49

-- State the theorem
theorem pure_imaginary_solutions :
  p (Complex.I * Real.sqrt 17) = 0 ∧ p (-Complex.I * Real.sqrt 17) = 0 :=
by sorry

end pure_imaginary_solutions_l860_86018


namespace john_sleep_week_total_l860_86026

/-- The amount of sleep John got during a week with varying sleep patterns. -/
def johnSleepWeek (recommendedSleep : ℝ) : ℝ :=
  let mondayTuesday := 2 * 3
  let wednesday := 0.8 * recommendedSleep
  let thursdayFriday := 2 * (0.5 * recommendedSleep)
  let saturday := 0.7 * recommendedSleep + 2
  let sunday := 0.4 * recommendedSleep
  mondayTuesday + wednesday + thursdayFriday + saturday + sunday

/-- Theorem stating that John's total sleep for the week is 31.2 hours. -/
theorem john_sleep_week_total : johnSleepWeek 8 = 31.2 := by
  sorry

#eval johnSleepWeek 8

end john_sleep_week_total_l860_86026


namespace min_value_fraction_l860_86067

theorem min_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

end min_value_fraction_l860_86067


namespace prob_AC_less_than_8_l860_86056

/-- The probability that AC < 8 cm given the conditions of the problem -/
def probability_AC_less_than_8 : ℝ := 0.46

/-- The length of AB in cm -/
def AB : ℝ := 10

/-- The length of BC in cm -/
def BC : ℝ := 6

/-- The angle ABC in radians -/
def angle_ABC : Set ℝ := Set.Ioo 0 (Real.pi / 2)

/-- The theorem stating the probability of AC < 8 cm -/
theorem prob_AC_less_than_8 :
  ∃ (p : ℝ → Bool), p = λ β => ‖(0, -AB) - (BC * Real.cos β, BC * Real.sin β)‖ < 8 ∧
  ∫ β in angle_ABC, (if p β then 1 else 0) / Real.pi * 2 = probability_AC_less_than_8 :=
sorry

end prob_AC_less_than_8_l860_86056


namespace two_distinct_prime_factors_iff_n_zero_l860_86007

def base_6_to_decimal (base_6_num : List Nat) : Nat :=
  base_6_num.enum.foldr (λ (i, digit) acc => acc + digit * (6 ^ i)) 0

def append_fives (n : Nat) : List Nat :=
  [1, 2, 0, 0] ++ List.replicate (10 * n + 2) 5

def result_number (n : Nat) : Nat :=
  base_6_to_decimal (append_fives n)

def has_exactly_two_distinct_prime_factors (x : Nat) : Prop :=
  ∃ p q : Nat, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  ∃ a b : Nat, x = p^a * q^b ∧ 
  ∀ r : Nat, Nat.Prime r → r ∣ x → (r = p ∨ r = q)

theorem two_distinct_prime_factors_iff_n_zero (n : Nat) :
  has_exactly_two_distinct_prime_factors (result_number n) ↔ n = 0 := by
  sorry

#check two_distinct_prime_factors_iff_n_zero

end two_distinct_prime_factors_iff_n_zero_l860_86007


namespace teacher_selection_theorem_l860_86074

/-- The number of male teachers -/
def num_male_teachers : ℕ := 4

/-- The number of female teachers -/
def num_female_teachers : ℕ := 3

/-- The total number of teachers to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select teachers with both genders represented -/
def num_ways_to_select : ℕ := 30

theorem teacher_selection_theorem :
  (num_ways_to_select = (Nat.choose num_male_teachers 2 * Nat.choose num_female_teachers 1) +
                        (Nat.choose num_male_teachers 1 * Nat.choose num_female_teachers 2)) ∧
  (num_ways_to_select = Nat.choose (num_male_teachers + num_female_teachers) num_selected -
                        Nat.choose num_male_teachers num_selected -
                        Nat.choose num_female_teachers num_selected) := by
  sorry

end teacher_selection_theorem_l860_86074


namespace train_length_calculation_l860_86025

-- Define the walking speed in meters per second
def walking_speed : ℝ := 1

-- Define the time taken for the train to pass Xiao Ming
def time_ming : ℝ := 22

-- Define the time taken for the train to pass Xiao Hong
def time_hong : ℝ := 24

-- Define the train's speed (to be solved)
def train_speed : ℝ := 23

-- Define the train's length (to be proved)
def train_length : ℝ := 528

-- Theorem statement
theorem train_length_calculation :
  train_length = time_ming * (train_speed + walking_speed) ∧
  train_length = time_hong * (train_speed - walking_speed) := by
  sorry

#check train_length_calculation

end train_length_calculation_l860_86025


namespace geometric_progression_first_term_l860_86088

/-- A geometric progression with second term 5 and third term 1 has first term 25. -/
theorem geometric_progression_first_term (a : ℝ) (q : ℝ) : 
  a * q = 5 ∧ a * q^2 = 1 → a = 25 := by
  sorry

end geometric_progression_first_term_l860_86088


namespace sheet_area_calculation_l860_86080

/-- Represents a rectangular sheet of paper. -/
structure Sheet where
  length : ℝ
  width : ℝ

/-- Represents the perimeters of the three rectangles after folding. -/
structure Perimeters where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ

/-- Calculates the perimeters of the three rectangles after folding. -/
def calculatePerimeters (s : Sheet) : Perimeters :=
  { p1 := 2 * s.length,
    p2 := 2 * s.width,
    p3 := 2 * (s.length - s.width) }

/-- The main theorem stating the conditions and the result to be proved. -/
theorem sheet_area_calculation (s : Sheet) :
  let p := calculatePerimeters s
  p.p1 = p.p2 + 20 ∧ p.p2 = p.p3 + 16 →
  s.length * s.width = 504 := by
  sorry


end sheet_area_calculation_l860_86080


namespace fish_tagging_problem_l860_86033

/-- The number of fish initially tagged in a pond -/
def initially_tagged (total_fish : ℕ) (catch_size : ℕ) (tagged_in_catch : ℕ) : ℕ :=
  (tagged_in_catch * total_fish) / catch_size

theorem fish_tagging_problem (total_fish : ℕ) (catch_size : ℕ) (tagged_in_catch : ℕ)
  (h1 : total_fish = 1500)
  (h2 : catch_size = 50)
  (h3 : tagged_in_catch = 2) :
  initially_tagged total_fish catch_size tagged_in_catch = 60 := by
  sorry

end fish_tagging_problem_l860_86033


namespace hen_count_l860_86091

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 50)
  (h2 : total_feet = 144)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) :
  ∃ (hens : ℕ) (cows : ℕ),
    hens + cows = total_animals ∧
    hens * hen_feet + cows * cow_feet = total_feet ∧
    hens = 28 :=
by sorry

end hen_count_l860_86091


namespace line_plane_perpendicularity_l860_86085

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β) 
  (h3 : parallel m n) (h4 : perpendicular m α) : 
  perpendicular n α :=
sorry

end line_plane_perpendicularity_l860_86085


namespace sufficient_but_not_necessary_l860_86027

theorem sufficient_but_not_necessary (x : ℝ) :
  (((1 : ℝ) / x < 1) → (x > 1)) ∧ ¬((x > 1) → ((1 : ℝ) / x < 1)) :=
by sorry

end sufficient_but_not_necessary_l860_86027


namespace right_triangle_division_area_ratio_l860_86009

/-- Given a right triangle divided by a point on its hypotenuse and lines parallel to its legs,
    forming a square and two smaller right triangles, this theorem proves the relationship
    between the areas of the smaller triangles and the square. -/
theorem right_triangle_division_area_ratio
  (square_side : ℝ)
  (m : ℝ)
  (h_square_side : square_side = 2)
  (h_small_triangle_area : ∃ (small_triangle_area : ℝ), small_triangle_area = m * square_side^2)
  : ∃ (other_triangle_area : ℝ), other_triangle_area / square_side^2 = 1 / (4 * m) :=
sorry

end right_triangle_division_area_ratio_l860_86009


namespace stadium_entrance_exit_plans_stadium_plans_eq_35_l860_86066

/-- The number of possible entrance and exit plans for a student at a school stadium. -/
theorem stadium_entrance_exit_plans : ℕ :=
  let south_gates : ℕ := 4
  let north_gates : ℕ := 3
  let west_gates : ℕ := 2
  let entrance_options : ℕ := south_gates + north_gates
  let exit_options : ℕ := west_gates + north_gates
  entrance_options * exit_options

/-- Proof that the number of possible entrance and exit plans is 35. -/
theorem stadium_plans_eq_35 : stadium_entrance_exit_plans = 35 := by
  sorry

end stadium_entrance_exit_plans_stadium_plans_eq_35_l860_86066


namespace smallest_binary_multiple_of_48_squared_l860_86048

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

def target_number : ℕ := 11111111100000000

theorem smallest_binary_multiple_of_48_squared :
  (target_number % (48^2) = 0) ∧
  is_binary_number target_number ∧
  ∀ m : ℕ, m < target_number →
    ¬(m % (48^2) = 0 ∧ is_binary_number m) :=
by sorry

#eval target_number % (48^2)  -- Should output 0
#eval target_number.digits 10  -- Should output [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

end smallest_binary_multiple_of_48_squared_l860_86048


namespace diamond_three_four_l860_86036

/-- Definition of the diamond operation -/
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

/-- Theorem stating that 3 ◊ 4 = 36 -/
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end diamond_three_four_l860_86036


namespace range_of_m_l860_86060

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + m ≤ 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-m)^x < (3-m)^y

-- Define the compound statements
def p_or_q (m : ℝ) : Prop := p m ∨ q m

def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- State the theorem
theorem range_of_m : 
  ∀ m : ℝ, (p_or_q m ∧ ¬(p_and_q m)) → (1 < m ∧ m < 2) :=
sorry

end range_of_m_l860_86060


namespace book_pages_theorem_l860_86006

/-- Calculate the number of digits used to number pages in a book -/
def digits_used (num_pages : ℕ) : ℕ :=
  let single_digit := min num_pages 9
  let double_digit := min (num_pages - 9) 90
  let triple_digit := max (num_pages - 99) 0
  single_digit + 2 * double_digit + 3 * triple_digit

theorem book_pages_theorem :
  ∃ (num_pages : ℕ), digits_used num_pages = 636 ∧ num_pages = 248 :=
by sorry

end book_pages_theorem_l860_86006


namespace rectangle_not_cuttable_from_square_cannot_cut_rectangle_from_square_l860_86002

/-- Proves that a rectangle with area 30 and length-to-width ratio 2:1 cannot be cut from a square with area 36 -/
theorem rectangle_not_cuttable_from_square : 
  ∀ (rect_length rect_width square_side : ℝ),
  rect_length > 0 → rect_width > 0 → square_side > 0 →
  rect_length * rect_width = 30 →
  rect_length = 2 * rect_width →
  square_side * square_side = 36 →
  rect_length > square_side :=
by sorry

/-- Concludes that the rectangular piece cannot be cut from the square piece -/
theorem cannot_cut_rectangle_from_square : 
  ∃ (rect_length rect_width square_side : ℝ),
  rect_length > 0 ∧ rect_width > 0 ∧ square_side > 0 ∧
  rect_length * rect_width = 30 ∧
  rect_length = 2 * rect_width ∧
  square_side * square_side = 36 ∧
  rect_length > square_side :=
by sorry

end rectangle_not_cuttable_from_square_cannot_cut_rectangle_from_square_l860_86002


namespace intersection_A_B_intersection_A_complement_BC_l860_86094

/-- The universal set U is ℝ -/
def U : Set ℝ := Set.univ

/-- Set A: { x | y = ln(x² - 9) } -/
def A : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 9)}

/-- Set B: { x | (x - 7)/(x + 1) > 0 } -/
def B : Set ℝ := {x | (x - 7) / (x + 1) > 0}

/-- Set C: { x | |x - 2| < 4 } -/
def C : Set ℝ := {x | |x - 2| < 4}

/-- Theorem 1: A ∩ B = { x | x < -3 or x > 7 } -/
theorem intersection_A_B : A ∩ B = {x | x < -3 ∨ x > 7} := by sorry

/-- Theorem 2: A ∩ (U \ (B ∩ C)) = { x | x < -3 or x > 3 } -/
theorem intersection_A_complement_BC : A ∩ (U \ (B ∩ C)) = {x | x < -3 ∨ x > 3} := by sorry

end intersection_A_B_intersection_A_complement_BC_l860_86094


namespace function_range_theorem_l860_86047

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f : ℝ → ℝ is decreasing on [0, +∞) if f(x) ≥ f(y) for all 0 ≤ x ≤ y -/
def IsDecreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y

theorem function_range_theorem (f : ℝ → ℝ) 
  (h_even : IsEven f) (h_decreasing : IsDecreasingOnNonnegatives f) :
  {x : ℝ | f (Real.log x) > f 1} = Set.Ioo (Real.exp (-1)) (Real.exp 1) := by
  sorry

end function_range_theorem_l860_86047


namespace arithmetic_sequence_sum_divisibility_l860_86028

theorem arithmetic_sequence_sum_divisibility :
  ∀ (x c : ℕ+),
  ∃ (d : ℕ+),
  (d = 15) ∧
  (d ∣ (15 * x + 105 * c)) ∧
  (∀ (k : ℕ+), k > d → ¬(∀ (y z : ℕ+), k ∣ (15 * y + 105 * z))) :=
by sorry

end arithmetic_sequence_sum_divisibility_l860_86028


namespace quadratic_root_in_unit_interval_l860_86053

theorem quadratic_root_in_unit_interval 
  (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end quadratic_root_in_unit_interval_l860_86053


namespace mushroom_collection_l860_86029

theorem mushroom_collection (N : ℕ) : 
  (100 ≤ N ∧ N < 1000) →  -- N is a three-digit number
  (N / 100 + (N / 10) % 10 + N % 10 = 14) →  -- sum of digits is 14
  (N % 50 = 0) →  -- divisible by 50
  (N % 25 = 0) →  -- 8% of N is an integer (since 8% = 2/25)
  (N % 50 = 0) →  -- 14% of N is an integer (since 14% = 7/50)
  N = 950 := by
sorry

end mushroom_collection_l860_86029


namespace monotone_increasing_condition_l860_86082

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * log x - x^2

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), MonotoneOn (f a) (Set.Icc 1 (exp 1))) ↔ a ≥ exp 1 :=
by sorry

end monotone_increasing_condition_l860_86082


namespace marbles_problem_l860_86024

/-- Represents the number of marbles left in a box after removing some marbles. -/
def marblesLeft (total white : ℕ) : ℕ :=
  let red := (total - white) / 2
  let blue := (total - white) / 2
  let removed := 2 * (white - blue)
  total - removed

/-- Theorem stating that given the conditions of the problem, 40 marbles are left. -/
theorem marbles_problem : marblesLeft 50 20 = 40 := by
  sorry

end marbles_problem_l860_86024


namespace ball_probabilities_solution_l860_86092

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Black
  | Yellow
  | Green

/-- Represents the probabilities of drawing balls of different colors -/
structure BallProbabilities where
  red : ℚ
  black : ℚ
  yellow : ℚ
  green : ℚ

/-- The conditions of the problem -/
def problem_conditions (p : BallProbabilities) : Prop :=
  p.red + p.black + p.yellow + p.green = 1 ∧
  p.red = 1/3 ∧
  p.black + p.yellow = 5/12 ∧
  p.yellow + p.green = 5/12

/-- The theorem stating the solution -/
theorem ball_probabilities_solution :
  ∃ (p : BallProbabilities), problem_conditions p ∧ 
    p.black = 1/4 ∧ p.yellow = 1/6 ∧ p.green = 1/4 := by
  sorry

end ball_probabilities_solution_l860_86092


namespace sqrt_45_minus_sqrt_20_equals_sqrt_5_l860_86084

theorem sqrt_45_minus_sqrt_20_equals_sqrt_5 : 
  Real.sqrt 45 - Real.sqrt 20 = Real.sqrt 5 := by
  sorry

end sqrt_45_minus_sqrt_20_equals_sqrt_5_l860_86084


namespace power_difference_l860_86049

theorem power_difference (a : ℕ) (h : 5^a = 3125) : 5^(a-3) = 25 := by
  sorry

end power_difference_l860_86049


namespace value_of_a_l860_86063

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 16 - 6 * a) : a = 8 / 5 := by
  sorry

end value_of_a_l860_86063


namespace percentage_of_b_grades_l860_86055

def grading_scale : List (String × Nat × Nat) :=
  [("A", 93, 100), ("B", 87, 92), ("C", 78, 86), ("D", 70, 77), ("F", 0, 69)]

def grades : List Nat :=
  [88, 66, 92, 83, 90, 99, 74, 78, 85, 72, 95, 86, 79, 68, 81, 64, 87, 91, 76, 89]

def is_grade_b (grade : Nat) : Bool :=
  87 ≤ grade ∧ grade ≤ 92

def count_grade_b (grades : List Nat) : Nat :=
  grades.filter is_grade_b |>.length

theorem percentage_of_b_grades :
  (count_grade_b grades : Rat) / grades.length * 100 = 25 := by
  sorry

end percentage_of_b_grades_l860_86055


namespace cryptarithm_solution_exists_and_unique_l860_86089

/-- Represents a cryptarithm solution -/
structure CryptarithmSolution where
  A : Nat
  H : Nat
  J : Nat
  O : Nat
  K : Nat
  E : Nat

/-- Checks if all digits in the solution are unique -/
def uniqueDigits (sol : CryptarithmSolution) : Prop :=
  sol.A ≠ sol.H ∧ sol.A ≠ sol.J ∧ sol.A ≠ sol.O ∧ sol.A ≠ sol.K ∧ sol.A ≠ sol.E ∧
  sol.H ≠ sol.J ∧ sol.H ≠ sol.O ∧ sol.H ≠ sol.K ∧ sol.H ≠ sol.E ∧
  sol.J ≠ sol.O ∧ sol.J ≠ sol.K ∧ sol.J ≠ sol.E ∧
  sol.O ≠ sol.K ∧ sol.O ≠ sol.E ∧
  sol.K ≠ sol.E

/-- Checks if the solution satisfies the cryptarithm equation -/
def satisfiesCryptarithm (sol : CryptarithmSolution) : Prop :=
  (100001 * sol.A + 11010 * sol.H) / (10 * sol.H + sol.A) = 
  1000 * sol.J + 100 * sol.O + 10 * sol.K + sol.E

/-- The main theorem stating that there exists a unique solution to the cryptarithm -/
theorem cryptarithm_solution_exists_and_unique :
  ∃! sol : CryptarithmSolution,
    uniqueDigits sol ∧
    satisfiesCryptarithm sol ∧
    sol.A = 3 ∧ sol.H = 7 ∧ sol.J = 5 ∧ sol.O = 1 ∧ sol.K = 6 ∧ sol.E = 9 :=
sorry

end cryptarithm_solution_exists_and_unique_l860_86089


namespace difference_of_integers_l860_86034

/-- Given positive integers a and b satisfying 2a - 9b + 18ab = 2018, prove that b - a = 223 -/
theorem difference_of_integers (a b : ℕ+) (h : 2 * a - 9 * b + 18 * a * b = 2018) : 
  b - a = 223 := by
  sorry

end difference_of_integers_l860_86034


namespace probability_two_girls_l860_86061

theorem probability_two_girls (p : ℝ) (h1 : p = 1 / 2) : p * p = 1 / 4 := by
  sorry

end probability_two_girls_l860_86061


namespace max_candy_types_l860_86001

/-- A type representing a student --/
def Student : Type := ℕ

/-- A type representing a candy type --/
def CandyType : Type := ℕ

/-- The total number of students --/
def total_students : ℕ := 1000

/-- A function representing whether a student received a certain candy type --/
def received (s : Student) (c : CandyType) : Prop := sorry

/-- The condition that for any 11 types of candy, each student received at least one of those types --/
def condition_eleven (N : ℕ) : Prop :=
  ∀ (s : Student) (cs : Finset CandyType),
    cs.card = 11 → (∃ c ∈ cs, received s c)

/-- The condition that for any two types of candy, there exists a student who received exactly one of those types --/
def condition_two (N : ℕ) : Prop :=
  ∀ (c1 c2 : CandyType),
    c1 ≠ c2 → (∃ s : Student, (received s c1 ∧ ¬received s c2) ∨ (¬received s c1 ∧ received s c2))

/-- The main theorem stating that the maximum possible value of N is 5501 --/
theorem max_candy_types :
  ∃ N : ℕ,
    (∀ N' : ℕ, condition_eleven N' ∧ condition_two N' → N' ≤ N) ∧
    condition_eleven N ∧ condition_two N ∧
    N = 5501 := by sorry

end max_candy_types_l860_86001
