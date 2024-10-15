import Mathlib

namespace NUMINAMATH_CALUDE_job_completion_time_l707_70746

/-- The time taken for two workers to complete a job together, given their relative efficiencies and the time taken by one worker. -/
theorem job_completion_time 
  (p_efficiency : ℝ) 
  (q_efficiency : ℝ) 
  (p_time : ℝ) 
  (h1 : p_efficiency = q_efficiency + 0.6 * q_efficiency) 
  (h2 : p_time = 26) :
  (p_efficiency * q_efficiency * p_time) / (p_efficiency * q_efficiency + p_efficiency * p_efficiency) = 1690 / 91 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l707_70746


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l707_70723

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℚ := 5
  let b : ℚ := 2
  (Nat.choose n (n / 2)) * (a ^ (n / 2)) * (b ^ (n / 2)) = 700000 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l707_70723


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l707_70729

theorem decimal_to_percentage (x : ℚ) : x = 2.08 → (x * 100 : ℚ) = 208 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l707_70729


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l707_70748

/-- Given a line segment from (1, 3) to (-7, y) with length 12 and y > 0, prove y = 3 + 4√5 -/
theorem line_segment_endpoint (y : ℝ) (h1 : y > 0) 
  (h2 : Real.sqrt (((-7) - 1)^2 + (y - 3)^2) = 12) : y = 3 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l707_70748


namespace NUMINAMATH_CALUDE_bird_speed_theorem_l707_70722

theorem bird_speed_theorem (d t : ℝ) 
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  d / t = 48 := by
  sorry

end NUMINAMATH_CALUDE_bird_speed_theorem_l707_70722


namespace NUMINAMATH_CALUDE_triangle_area_l707_70710

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  (a * b : ℝ) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l707_70710


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l707_70738

theorem necessary_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 2 * (1 - m) * x + 3 > 0) →
  (m > 0 ∧ ∃ m' > 0, ¬(∀ x : ℝ, (m' - 1) * x^2 + 2 * (1 - m') * x + 3 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l707_70738


namespace NUMINAMATH_CALUDE_shell_addition_problem_l707_70700

/-- Calculates the final addition of shells given the initial amount, additions, and removals. -/
def final_addition (initial : ℕ) (first_addition : ℕ) (removal : ℕ) (total : ℕ) : ℕ :=
  total - (initial + first_addition - removal)

/-- Proves that the final addition of shells is 16 pounds given the problem conditions. -/
theorem shell_addition_problem :
  final_addition 5 9 2 28 = 16 := by
  sorry

end NUMINAMATH_CALUDE_shell_addition_problem_l707_70700


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l707_70795

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- The distance between two five-digit numbers -/
def distance (a b : FiveDigitNumber) : Nat :=
  sorry

/-- A permutation of all five-digit numbers -/
def Permutation := Equiv.Perm FiveDigitNumber

/-- The sum of distances between consecutive numbers in a permutation -/
def sumOfDistances (p : Permutation) : Nat :=
  sorry

/-- The minimum possible sum of distances between consecutive five-digit numbers -/
theorem min_sum_of_distances :
  ∃ (p : Permutation), sumOfDistances p = 101105 ∧
  ∀ (q : Permutation), sumOfDistances q ≥ 101105 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l707_70795


namespace NUMINAMATH_CALUDE_group_size_calculation_l707_70715

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 6.2 →
  old_weight = 76 →
  new_weight = 119.4 →
  (new_weight - old_weight) / average_increase = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l707_70715


namespace NUMINAMATH_CALUDE_train_speed_l707_70780

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 250) (h2 : crossing_time = 4) :
  train_length / crossing_time = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l707_70780


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l707_70739

/-- Given a circle with equation x^2 - 8x + y^2 + 16y = -100, 
    prove that the sum of the x-coordinate of the center, 
    the y-coordinate of the center, and the radius is -4 + 2√5 -/
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), 
    (∀ (x y : ℝ), x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = -4 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l707_70739


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l707_70765

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (p : ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_roots : a 1 + a 5 = p ∧ a 1 * a 5 = 4)
  (h_p_neg : p < 0) :
  a 3 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l707_70765


namespace NUMINAMATH_CALUDE_largest_power_of_five_in_sum_of_factorials_l707_70788

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 98 + factorial 99 + factorial 100

theorem largest_power_of_five_in_sum_of_factorials :
  (∃ k : ℕ, sum_of_factorials = 5^26 * k ∧ ¬∃ m : ℕ, sum_of_factorials = 5^27 * m) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_five_in_sum_of_factorials_l707_70788


namespace NUMINAMATH_CALUDE_zainab_hourly_wage_l707_70761

/-- Zainab's work schedule and earnings -/
structure WorkSchedule where
  daysPerWeek : ℕ
  hoursPerDay : ℕ
  totalWeeks : ℕ
  totalEarnings : ℕ

/-- Calculate hourly wage given a work schedule -/
def hourlyWage (schedule : WorkSchedule) : ℚ :=
  schedule.totalEarnings / (schedule.daysPerWeek * schedule.hoursPerDay * schedule.totalWeeks)

/-- Zainab's specific work schedule -/
def zainabSchedule : WorkSchedule :=
  { daysPerWeek := 3
  , hoursPerDay := 4
  , totalWeeks := 4
  , totalEarnings := 96 }

/-- Theorem: Zainab's hourly wage is $2 -/
theorem zainab_hourly_wage :
  hourlyWage zainabSchedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_zainab_hourly_wage_l707_70761


namespace NUMINAMATH_CALUDE_solution_difference_l707_70702

theorem solution_difference (m : ℚ) : 
  (∃ x₁ x₂ : ℚ, 5*m + 3*x₁ = 1 + x₁ ∧ 2*x₂ + m = 3*m ∧ x₁ = x₂ + 2) ↔ m = -3/7 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l707_70702


namespace NUMINAMATH_CALUDE_andy_solves_two_problems_l707_70745

/-- Returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Returns true if n has an odd digit sum, false otherwise -/
def hasOddDigitSum (n : ℕ) : Prop := sorry

/-- The set of numbers we're considering -/
def problemSet : Set ℕ := {n : ℕ | 78 ≤ n ∧ n ≤ 125}

/-- The count of prime numbers with odd digit sums in our problem set -/
def countPrimesWithOddDigitSum : ℕ := sorry

theorem andy_solves_two_problems : countPrimesWithOddDigitSum = 2 := by sorry

end NUMINAMATH_CALUDE_andy_solves_two_problems_l707_70745


namespace NUMINAMATH_CALUDE_rental_cost_equality_l707_70760

/-- The daily rate for Sunshine Car Rentals in dollars -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate for Sunshine Car Rentals in dollars -/
def sunshine_mile_rate : ℝ := 0.18

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.16

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 48

theorem rental_cost_equality :
  sunshine_daily_rate + sunshine_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l707_70760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l707_70711

/-- The number of terms in the arithmetic sequence 2.5, 6.5, 10.5, ..., 54.5, 58.5 -/
def sequence_length : ℕ := 15

/-- The first term of the sequence -/
def a₁ : ℚ := 2.5

/-- The last term of the sequence -/
def aₙ : ℚ := 58.5

/-- The common difference of the sequence -/
def d : ℚ := 4

theorem arithmetic_sequence_length :
  sequence_length = (aₙ - a₁) / d + 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l707_70711


namespace NUMINAMATH_CALUDE_music_club_members_not_playing_l707_70759

theorem music_club_members_not_playing (total_members guitar_players piano_players both_players : ℕ) 
  (h1 : total_members = 80)
  (h2 : guitar_players = 45)
  (h3 : piano_players = 30)
  (h4 : both_players = 18) :
  total_members - (guitar_players + piano_players - both_players) = 23 := by
  sorry

end NUMINAMATH_CALUDE_music_club_members_not_playing_l707_70759


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l707_70782

/-- Given a geometric sequence of positive integers with first term 3 and sixth term 729,
    prove that the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →                             -- First term is 3
  a 6 = 729 →                           -- Sixth term is 729
  (∀ n, a n > 0) →                      -- All terms are positive
  a 7 = 2187 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l707_70782


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l707_70727

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ 
  (∃ x : ℝ, x ≥ 3 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l707_70727


namespace NUMINAMATH_CALUDE_disc_purchase_problem_l707_70791

theorem disc_purchase_problem (price_a price_b total_spent : ℚ) (num_b : ℕ) :
  price_a = 21/2 ∧ 
  price_b = 17/2 ∧ 
  total_spent = 93 ∧ 
  num_b = 6 →
  ∃ (num_a : ℕ), num_a + num_b = 10 ∧ 
    num_a * price_a + num_b * price_b = total_spent :=
by sorry

end NUMINAMATH_CALUDE_disc_purchase_problem_l707_70791


namespace NUMINAMATH_CALUDE_triangle_property_l707_70735

theorem triangle_property (A B C a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) ∧
  -- Sides are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Part 1
  (a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c → A = π / 6) ∧
  -- Part 2
  (a = 1 ∧ b * c = 2 - Real.sqrt 3 → b + c = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l707_70735


namespace NUMINAMATH_CALUDE_equal_perimeter_triangles_l707_70741

theorem equal_perimeter_triangles (a b c x y : ℝ) : 
  a = 7 → b = 12 → c = 9 → x = 2 → y = 7 → x + y = c →
  (a + x + (b - a)) = (b + y + (b - a)) := by sorry

end NUMINAMATH_CALUDE_equal_perimeter_triangles_l707_70741


namespace NUMINAMATH_CALUDE_time_to_write_michaels_name_l707_70732

/-- The number of letters in Michael's name -/
def name_length : ℕ := 7

/-- The number of rearrangements Michael can write per minute -/
def rearrangements_per_minute : ℕ := 10

/-- Calculate the total number of rearrangements for a name with distinct letters -/
def total_rearrangements (n : ℕ) : ℕ := Nat.factorial n

/-- Calculate the time in hours to write all rearrangements -/
def time_to_write_all (name_len : ℕ) (rearr_per_min : ℕ) : ℚ :=
  (total_rearrangements name_len : ℚ) / (rearr_per_min : ℚ) / 60

/-- Theorem: It takes 8.4 hours to write all rearrangements of Michael's name -/
theorem time_to_write_michaels_name :
  time_to_write_all name_length rearrangements_per_minute = 84 / 10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_write_michaels_name_l707_70732


namespace NUMINAMATH_CALUDE_trig_identity_l707_70755

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.cos (π/3 - x))^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l707_70755


namespace NUMINAMATH_CALUDE_square_of_85_l707_70716

theorem square_of_85 : (85 : ℕ) ^ 2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l707_70716


namespace NUMINAMATH_CALUDE_spherical_sector_central_angle_l707_70744

theorem spherical_sector_central_angle (R : ℝ) (α : ℝ) :
  R > 0 →
  (∃ r m : ℝ, R * π * r = 2 * R * π * m ∧ 
              R^2 = r^2 + (R - m)^2 ∧ 
              0 < m ∧ m < R) →
  α = 2 * Real.arccos (3/5) :=
sorry

end NUMINAMATH_CALUDE_spherical_sector_central_angle_l707_70744


namespace NUMINAMATH_CALUDE_principal_calculation_l707_70712

/-- The principal amount in dollars -/
def principal : ℝ := sorry

/-- The compounded amount after 7 years -/
def compounded_amount : ℝ := sorry

/-- The difference between compounded amount and principal -/
def difference : ℝ := 5000

/-- The interest rate for the first 2 years -/
def rate1 : ℝ := 0.03

/-- The interest rate for the next 3 years -/
def rate2 : ℝ := 0.04

/-- The interest rate for the last 2 years -/
def rate3 : ℝ := 0.05

/-- The number of years for each interest rate period -/
def years1 : ℕ := 2
def years2 : ℕ := 3
def years3 : ℕ := 2

theorem principal_calculation :
  principal * (1 + rate1) ^ years1 * (1 + rate2) ^ years2 * (1 + rate3) ^ years3 = principal + difference :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l707_70712


namespace NUMINAMATH_CALUDE_katherine_bottle_caps_l707_70789

def initial_bottle_caps : ℕ := 34
def eaten_bottle_caps : ℕ := 8
def remaining_bottle_caps : ℕ := 26

theorem katherine_bottle_caps :
  initial_bottle_caps = eaten_bottle_caps + remaining_bottle_caps :=
by sorry

end NUMINAMATH_CALUDE_katherine_bottle_caps_l707_70789


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l707_70764

theorem pure_imaginary_product (x : ℝ) : 
  (x^4 + 6*x^3 + 7*x^2 - 14*x - 12 = 0) ↔ (x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l707_70764


namespace NUMINAMATH_CALUDE_f_at_2_l707_70793

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

-- Theorem statement
theorem f_at_2 : f 2 = 259 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l707_70793


namespace NUMINAMATH_CALUDE_last_three_digits_of_power_l707_70772

theorem last_three_digits_of_power (N : ℕ) : 
  N = 2002^2001 → 2003^N ≡ 241 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_power_l707_70772


namespace NUMINAMATH_CALUDE_total_seedlings_sold_l707_70753

/-- Represents the number of seedlings sold for each type -/
structure Seedlings where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Theorem stating the total number of seedlings sold given the conditions -/
theorem total_seedlings_sold (s : Seedlings) : 
  (s.A : ℚ) / s.B = 1 / 2 →
  (s.B : ℚ) / s.C = 3 / 4 →
  3 * s.A + 2 * s.B + s.C = 29000 →
  s.A + s.B + s.C = 17000 := by
  sorry

#check total_seedlings_sold

end NUMINAMATH_CALUDE_total_seedlings_sold_l707_70753


namespace NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l707_70736

/-- Given a, b, c form a geometric sequence, prove that ax^2 + bx + c has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) 
  (h_geometric : b^2 = a*c) 
  (h_positive : a*c > 0) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l707_70736


namespace NUMINAMATH_CALUDE_business_profit_l707_70705

def total_subscription : ℕ := 50000
def a_more_than_b : ℕ := 4000
def b_more_than_c : ℕ := 5000
def a_profit : ℕ := 29400

theorem business_profit :
  ∃ (c_subscription : ℕ),
    let b_subscription := c_subscription + b_more_than_c
    let a_subscription := b_subscription + a_more_than_b
    a_subscription + b_subscription + c_subscription = total_subscription →
    (a_profit * total_subscription) / a_subscription = 70000 :=
sorry

end NUMINAMATH_CALUDE_business_profit_l707_70705


namespace NUMINAMATH_CALUDE_corn_acreage_l707_70709

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l707_70709


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l707_70737

theorem cos_2alpha_value (α : Real) (h : Real.tan (α - π/4) = -1/3) :
  Real.cos (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l707_70737


namespace NUMINAMATH_CALUDE_perfect_score_correct_l707_70757

/-- The perfect score for a single game, given that 3 perfect games result in 63 points. -/
def perfect_score : ℕ := 21

/-- The total score for three perfect games. -/
def three_game_score : ℕ := 63

/-- Theorem stating that the perfect score for a single game is correct. -/
theorem perfect_score_correct : perfect_score * 3 = three_game_score := by
  sorry

end NUMINAMATH_CALUDE_perfect_score_correct_l707_70757


namespace NUMINAMATH_CALUDE_certain_percentage_problem_l707_70784

theorem certain_percentage_problem (x : ℝ) : x = 12 → (x / 100) * 24.2 = 0.1 * 14.2 + 1.484 := by
  sorry

end NUMINAMATH_CALUDE_certain_percentage_problem_l707_70784


namespace NUMINAMATH_CALUDE_cookie_theorem_l707_70734

def cookie_problem (initial_total initial_chocolate initial_sugar initial_oatmeal : ℕ)
  (morning_chocolate morning_sugar : ℕ)
  (lunch_chocolate lunch_sugar lunch_oatmeal : ℕ)
  (afternoon_chocolate afternoon_sugar afternoon_oatmeal : ℕ)
  (damage_percent : ℚ) : Prop :=
  let total_chocolate_sold := morning_chocolate + lunch_chocolate + afternoon_chocolate
  let total_sugar_sold := morning_sugar + lunch_sugar + afternoon_sugar
  let total_oatmeal_sold := lunch_oatmeal + afternoon_oatmeal
  let remaining_chocolate := max (initial_chocolate - total_chocolate_sold) 0
  let remaining_sugar := initial_sugar - total_sugar_sold
  let remaining_oatmeal := initial_oatmeal - total_oatmeal_sold
  let total_remaining := remaining_chocolate + remaining_sugar + remaining_oatmeal
  let damaged := ⌊(damage_percent * total_remaining : ℚ)⌋
  total_remaining - damaged = 18

theorem cookie_theorem :
  cookie_problem 120 60 40 20 24 12 33 20 4 10 4 2 (1/20) := by sorry

end NUMINAMATH_CALUDE_cookie_theorem_l707_70734


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l707_70786

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  ¬(∃ x : ℝ, a * x^2 + x + 1 ≥ 0 → a ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l707_70786


namespace NUMINAMATH_CALUDE_pi_is_irrational_one_third_is_rational_sqrt_16_is_rational_finite_decimal_is_rational_pi_only_irrational_option_l707_70778

theorem pi_is_irrational :
  ∀ (a b : ℚ), (a : ℝ) ≠ π ∧ (b : ℝ) ≠ π → Irrational π := by
  sorry

theorem one_third_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ (1 : ℝ) / 3 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem sqrt_16_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 16 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem finite_decimal_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 3.1415926 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem pi_only_irrational_option : Irrational π := by
  sorry

end NUMINAMATH_CALUDE_pi_is_irrational_one_third_is_rational_sqrt_16_is_rational_finite_decimal_is_rational_pi_only_irrational_option_l707_70778


namespace NUMINAMATH_CALUDE_competition_scores_l707_70797

theorem competition_scores (n k : ℕ) : n ≥ 2 ∧ k ≥ 1 →
  (k * n * (n + 1) = 52 * n * (n - 1)) ↔ 
  ((n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13)) := by
sorry

end NUMINAMATH_CALUDE_competition_scores_l707_70797


namespace NUMINAMATH_CALUDE_rectangle_and_parallelogram_area_l707_70762

-- Define the shapes and their properties
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side ^ 2

structure Circle where
  radius : ℝ

structure Rectangle where
  length : ℝ
  breadth : ℝ
  area : ℝ
  area_eq : area = length * breadth

structure Parallelogram where
  base : ℝ
  height : ℝ
  diagonal : ℝ
  area : ℝ
  area_eq : area = base * height

-- Define the problem
def problem (s : Square) (c : Circle) (r : Rectangle) (p : Parallelogram) : Prop :=
  s.area = 3600 ∧
  s.side = c.radius ∧
  r.length = 2/5 * c.radius ∧
  r.breadth = 10 ∧
  r.breadth = 1/2 * p.diagonal ∧
  p.base = 20 * Real.sqrt 3 ∧
  p.height = r.breadth

-- Theorem to prove
theorem rectangle_and_parallelogram_area 
  (s : Square) (c : Circle) (r : Rectangle) (p : Parallelogram) 
  (h : problem s c r p) : 
  r.area = 240 ∧ p.area = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_and_parallelogram_area_l707_70762


namespace NUMINAMATH_CALUDE_safe_opening_l707_70776

theorem safe_opening (a b c n m k : ℕ) :
  ∃ (x y z : ℕ), ∃ (w : ℕ),
    (a^n * b^m * c^k = w^3) ∨
    (a^n * b^y * c^z = w^3) ∨
    (a^x * b^m * c^z = w^3) ∨
    (a^x * b^y * c^k = w^3) ∨
    (x^n * b^m * c^z = w^3) ∨
    (x^n * b^y * c^k = w^3) ∨
    (x^y * b^m * c^k = w^3) ∨
    (a^x * y^m * c^z = w^3) ∨
    (a^x * y^z * c^k = w^3) ∨
    (x^n * y^m * c^z = w^3) ∨
    (x^n * y^z * c^k = w^3) ∨
    (x^y * z^m * c^k = w^3) ∨
    (a^x * y^m * z^k = w^3) ∨
    (x^n * y^m * z^k = w^3) ∨
    (x^y * b^m * z^k = w^3) ∨
    (x^y * z^m * c^k = w^3) :=
by sorry


end NUMINAMATH_CALUDE_safe_opening_l707_70776


namespace NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l707_70754

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem three_percent_to_decimal : (3 : ℚ) / 100 = 0.03 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l707_70754


namespace NUMINAMATH_CALUDE_complex_equation_solution_l707_70774

theorem complex_equation_solution (m : ℝ) : 
  let z₁ : ℂ := m^2 - 3*m + m^2*Complex.I
  let z₂ : ℂ := 4 + (5*m + 6)*Complex.I
  z₁ - z₂ = 0 → m = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l707_70774


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_count_gem_stone_necklaces_count_proof_l707_70728

theorem gem_stone_necklaces_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_earnings cost_per_necklace bead_necklaces gem_stone_necklaces =>
    total_earnings = cost_per_necklace * (bead_necklaces + gem_stone_necklaces) →
    cost_per_necklace = 6 →
    bead_necklaces = 3 →
    total_earnings = 36 →
    gem_stone_necklaces = 3

-- Proof
theorem gem_stone_necklaces_count_proof :
  gem_stone_necklaces_count 36 6 3 3 := by
  sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_count_gem_stone_necklaces_count_proof_l707_70728


namespace NUMINAMATH_CALUDE_dennis_initial_money_dennis_initial_money_proof_l707_70706

/-- Proves that Dennis's initial amount of money equals $50, given the conditions of his purchase and change received. -/
theorem dennis_initial_money : ℕ → Prop :=
  fun initial : ℕ =>
    let shirt_cost : ℕ := 27
    let change_bills : ℕ := 2 * 10
    let change_coins : ℕ := 3
    let total_change : ℕ := change_bills + change_coins
    initial = shirt_cost + total_change ∧ initial = 50

/-- The theorem holds for the specific case where Dennis's initial money is 50. -/
theorem dennis_initial_money_proof : dennis_initial_money 50 := by
  sorry

#check dennis_initial_money
#check dennis_initial_money_proof

end NUMINAMATH_CALUDE_dennis_initial_money_dennis_initial_money_proof_l707_70706


namespace NUMINAMATH_CALUDE_descending_order_inequality_l707_70717

theorem descending_order_inequality (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  x * y > x * y^2 ∧ x * y^2 > x := by
  sorry

end NUMINAMATH_CALUDE_descending_order_inequality_l707_70717


namespace NUMINAMATH_CALUDE_division_theorem_l707_70726

/-- The dividend polynomial -/
def p (x : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + 5 * x - 8

/-- The divisor polynomial -/
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 14 * x - 14

/-- Theorem stating that r is the remainder when p is divided by d -/
theorem division_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_division_theorem_l707_70726


namespace NUMINAMATH_CALUDE_soda_distribution_l707_70747

theorem soda_distribution (total_sodas : ℕ) (sisters : ℕ) : 
  total_sodas = 12 →
  sisters = 2 →
  let brothers := 2 * sisters
  let total_siblings := sisters + brothers
  total_sodas / total_siblings = 2 := by
  sorry

end NUMINAMATH_CALUDE_soda_distribution_l707_70747


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_200_l707_70719

-- Define what a Mersenne number is
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

-- Define what a Mersenne prime is
def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, Nat.Prime n ∧ p = mersenne_number n

-- Theorem statement
theorem largest_mersenne_prime_under_200 :
  (∀ p : ℕ, is_mersenne_prime p ∧ p < 200 → p ≤ 127) ∧
  is_mersenne_prime 127 := by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_200_l707_70719


namespace NUMINAMATH_CALUDE_division_of_four_by_negative_two_l707_70792

theorem division_of_four_by_negative_two : 4 / (-2 : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_division_of_four_by_negative_two_l707_70792


namespace NUMINAMATH_CALUDE_probability_alternating_colors_is_correct_l707_70775

def total_balls : ℕ := 12
def white_balls : ℕ := 6
def black_balls : ℕ := 6

def alternating_sequence : List Bool := [true, false, true, false, true, false, true, false, true, false, true, false]

def probability_alternating_colors : ℚ :=
  1 / (total_balls.choose white_balls)

theorem probability_alternating_colors_is_correct :
  probability_alternating_colors = 1 / 924 :=
by sorry

end NUMINAMATH_CALUDE_probability_alternating_colors_is_correct_l707_70775


namespace NUMINAMATH_CALUDE_classrooms_needed_l707_70743

/-- Given a school with 390 students and classrooms that hold 30 students each,
    prove that 13 classrooms are needed. -/
theorem classrooms_needed (total_students : Nat) (students_per_classroom : Nat) :
  total_students = 390 →
  students_per_classroom = 30 →
  (total_students + students_per_classroom - 1) / students_per_classroom = 13 := by
  sorry

end NUMINAMATH_CALUDE_classrooms_needed_l707_70743


namespace NUMINAMATH_CALUDE_math_problem_l707_70707

theorem math_problem :
  (8 * 40 = 320) ∧
  (5 * (1 / 6) = 5 / 6) ∧
  (6 * 500 = 3000) ∧
  (∃ n : ℕ, 3000 = n * 1000) := by
sorry

end NUMINAMATH_CALUDE_math_problem_l707_70707


namespace NUMINAMATH_CALUDE_sum_largest_smallest_special_digits_l707_70773

def largest_three_digit (hundreds : Nat) (ones : Nat) : Nat :=
  hundreds * 100 + 90 + ones

def smallest_three_digit (hundreds : Nat) (ones : Nat) : Nat :=
  hundreds * 100 + ones

theorem sum_largest_smallest_special_digits :
  largest_three_digit 2 7 + smallest_three_digit 2 7 = 504 := by
  sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_special_digits_l707_70773


namespace NUMINAMATH_CALUDE_multiply_powers_l707_70713

theorem multiply_powers (x y : ℝ) : 3 * x^2 * (-2 * x * y^3) = -6 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_l707_70713


namespace NUMINAMATH_CALUDE_x_intercepts_count_l707_70740

theorem x_intercepts_count (x : ℝ) : 
  ∃! x, (x - 5) * (x^2 + x + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l707_70740


namespace NUMINAMATH_CALUDE_inheritance_problem_l707_70730

theorem inheritance_problem (x : ℝ) : 
  (100 + (1/10) * (x - 100) = 200 + (1/10) * (x - (100 + (1/10) * (x - 100)) - 200)) →
  x = 8100 := by
sorry

end NUMINAMATH_CALUDE_inheritance_problem_l707_70730


namespace NUMINAMATH_CALUDE_solve_candy_problem_l707_70779

def candy_problem (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ) : Prop :=
  let mary_initial := mary_multiplier * megan_candy
  let mary_total := mary_initial + mary_additional
  megan_candy = 5 ∧ mary_multiplier = 3 ∧ mary_additional = 10 → mary_total = 25

theorem solve_candy_problem :
  candy_problem 5 3 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l707_70779


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l707_70708

theorem gcd_of_three_numbers : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l707_70708


namespace NUMINAMATH_CALUDE_diamond_equal_forms_intersecting_lines_l707_70751

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ⋄ y = y ⋄ x -/
def diamond_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The set of points on the lines y = x and y = -x -/
def intersecting_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 ∨ p.2 = -p.1}

theorem diamond_equal_forms_intersecting_lines :
  diamond_equal_set = intersecting_lines :=
sorry

end NUMINAMATH_CALUDE_diamond_equal_forms_intersecting_lines_l707_70751


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l707_70763

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 16 = 0) → m = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l707_70763


namespace NUMINAMATH_CALUDE_ratio_evaluation_l707_70703

theorem ratio_evaluation : (2^2023 * 3^2025) / 6^2024 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l707_70703


namespace NUMINAMATH_CALUDE_money_ratio_problem_l707_70766

theorem money_ratio_problem (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  ram = 588 →
  krishan = 3468 →
  (gopal : ℚ) / krishan = 100 / 243 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l707_70766


namespace NUMINAMATH_CALUDE_correct_num_technicians_l707_70769

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the average salary of all workers in Rupees -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in Rupees -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technician workers in Rupees -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians :
  num_technicians = 7 ∧
  num_technicians ≤ total_workers ∧
  num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest =
    total_workers * avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_correct_num_technicians_l707_70769


namespace NUMINAMATH_CALUDE_length_AC_is_12_l707_70724

/-- Two circles in a plane with given properties -/
structure TwoCircles where
  A : ℝ × ℝ  -- Center of larger circle
  B : ℝ × ℝ  -- Center of smaller circle
  C : ℝ × ℝ  -- Point on line segment AB
  rA : ℝ     -- Radius of larger circle
  rB : ℝ     -- Radius of smaller circle

/-- The theorem to be proved -/
theorem length_AC_is_12 (circles : TwoCircles)
  (h1 : circles.rA = 12)
  (h2 : circles.rB = 7)
  (h3 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ circles.C = (1 - t) • circles.A + t • circles.B)
  (h4 : ‖circles.C - circles.B‖ = circles.rB) :
  ‖circles.A - circles.C‖ = 12 :=
sorry

end NUMINAMATH_CALUDE_length_AC_is_12_l707_70724


namespace NUMINAMATH_CALUDE_simple_interest_problem_l707_70720

/-- Given a sum put at simple interest for 10 years, if increasing the interest
    rate by 5% results in Rs. 600 more interest, then the sum is Rs. 1200. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 600 → P = 1200 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l707_70720


namespace NUMINAMATH_CALUDE_intersecting_circles_radius_l707_70767

/-- Two circles on a plane where each passes through the center of the other -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ  -- Center of first circle
  O₂ : ℝ × ℝ  -- Center of second circle
  A : ℝ × ℝ   -- First intersection point
  B : ℝ × ℝ   -- Second intersection point
  radius : ℝ  -- Common radius of both circles
  passes_through_center : dist O₁ O₂ = radius
  on_circle : dist O₁ A = radius ∧ dist O₂ A = radius ∧ dist O₁ B = radius ∧ dist O₂ B = radius

/-- The area of a quadrilateral given its four vertices -/
def quadrilateralArea (p₁ p₂ p₃ p₄ : ℝ × ℝ) : ℝ := sorry

theorem intersecting_circles_radius 
  (circles : IntersectingCircles) 
  (area_condition : quadrilateralArea circles.O₁ circles.A circles.O₂ circles.B = 2 * Real.sqrt 3) :
  circles.radius = 2 := by sorry

end NUMINAMATH_CALUDE_intersecting_circles_radius_l707_70767


namespace NUMINAMATH_CALUDE_sod_coverage_theorem_l707_70790

/-- The number of square sod pieces needed to cover two rectangular areas -/
def sod_squares_needed (length1 width1 length2 width2 sod_size : ℕ) : ℕ :=
  ((length1 * width1 + length2 * width2) : ℕ) / (sod_size * sod_size)

/-- Theorem stating that 1500 squares of 2x2-foot sod are needed to cover two areas of 30x40 feet and 60x80 feet -/
theorem sod_coverage_theorem :
  sod_squares_needed 30 40 60 80 2 = 1500 :=
by sorry

end NUMINAMATH_CALUDE_sod_coverage_theorem_l707_70790


namespace NUMINAMATH_CALUDE_andras_bela_numbers_l707_70758

theorem andras_bela_numbers :
  ∀ (a b : ℕ+),
  (a = b + 1992 ∨ b = a + 1992) →
  a > 1992 →
  b > 3984 →
  a ≤ 5976 →
  (a + 1 > 5976) →
  (a = 5976 ∧ b = 7968) :=
by sorry

end NUMINAMATH_CALUDE_andras_bela_numbers_l707_70758


namespace NUMINAMATH_CALUDE_highest_throw_l707_70701

def christine_first : ℕ := 20

def janice_first (christine_first : ℕ) : ℕ := christine_first - 4

def christine_second (christine_first : ℕ) : ℕ := christine_first + 10

def janice_second (janice_first : ℕ) : ℕ := janice_first * 2

def christine_third (christine_second : ℕ) : ℕ := christine_second + 4

def janice_third (christine_first : ℕ) : ℕ := christine_first + 17

theorem highest_throw :
  let c1 := christine_first
  let j1 := janice_first c1
  let c2 := christine_second c1
  let j2 := janice_second j1
  let c3 := christine_third c2
  let j3 := janice_third c1
  max c1 (max j1 (max c2 (max j2 (max c3 j3)))) = 37 := by sorry

end NUMINAMATH_CALUDE_highest_throw_l707_70701


namespace NUMINAMATH_CALUDE_brady_dwayne_earnings_difference_l707_70798

/-- Given that Dwayne makes $1,500 in a year and Brady and Dwayne's combined earnings are $3,450 in a year,
    prove that Brady makes $450 more than Dwayne in a year. -/
theorem brady_dwayne_earnings_difference :
  let dwayne_earnings : ℕ := 1500
  let combined_earnings : ℕ := 3450
  let brady_earnings : ℕ := combined_earnings - dwayne_earnings
  brady_earnings - dwayne_earnings = 450 := by
sorry

end NUMINAMATH_CALUDE_brady_dwayne_earnings_difference_l707_70798


namespace NUMINAMATH_CALUDE_correct_smaller_type_pages_l707_70787

/-- Represents the number of pages in smaller type -/
def smaller_type_pages : ℕ := 17

/-- Represents the number of pages in larger type -/
def larger_type_pages : ℕ := 21 - smaller_type_pages

/-- The total number of words in the article -/
def total_words : ℕ := 48000

/-- The number of words per page in larger type -/
def words_per_page_large : ℕ := 1800

/-- The number of words per page in smaller type -/
def words_per_page_small : ℕ := 2400

/-- The total number of pages -/
def total_pages : ℕ := 21

theorem correct_smaller_type_pages : 
  smaller_type_pages = 17 ∧ 
  larger_type_pages + smaller_type_pages = total_pages ∧
  words_per_page_large * larger_type_pages + words_per_page_small * smaller_type_pages = total_words :=
by sorry

end NUMINAMATH_CALUDE_correct_smaller_type_pages_l707_70787


namespace NUMINAMATH_CALUDE_circle_chord_intersection_l707_70733

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) : 
  r = 5 →
  chord_length = 8 →
  ∃ (ak kb : ℝ),
    ak + kb = 2 * r ∧
    ak * kb = r^2 - (chord_length / 2)^2 ∧
    ak = 1.25 ∧
    kb = 8.75 := by
sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_l707_70733


namespace NUMINAMATH_CALUDE_no_solution_l707_70725

/-- ab is a two-digit number -/
def ab : ℕ := sorry

/-- ba is a two-digit number, which is the reverse of ab -/
def ba : ℕ := sorry

/-- ab and ba are distinct -/
axiom ab_ne_ba : ab ≠ ba

/-- There is no real number x that satisfies the equation (ab)^x - 2 = (ba)^x - 7 -/
theorem no_solution : ¬∃ x : ℝ, (ab : ℝ) ^ x - 2 = (ba : ℝ) ^ x - 7 := by sorry

end NUMINAMATH_CALUDE_no_solution_l707_70725


namespace NUMINAMATH_CALUDE_quadratic_polynomials_sum_nonnegative_l707_70785

theorem quadratic_polynomials_sum_nonnegative 
  (b c p q m₁ m₂ k₁ k₂ : ℝ) 
  (hf : ∀ x, x^2 + b*x + c = (x - m₁) * (x - m₂))
  (hg : ∀ x, x^2 + p*x + q = (x - k₁) * (x - k₂)) :
  (k₁^2 + b*k₁ + c) + (k₂^2 + b*k₂ + c) + 
  (m₁^2 + p*m₁ + q) + (m₂^2 + p*m₂ + q) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_sum_nonnegative_l707_70785


namespace NUMINAMATH_CALUDE_only_negative_number_l707_70749

def is_negative (x : ℝ) : Prop := x < 0

theorem only_negative_number (a b c d : ℝ) 
  (ha : a = -2) (hb : b = 0) (hc : c = 1) (hd : d = 3) : 
  is_negative a ∧ ¬is_negative b ∧ ¬is_negative c ∧ ¬is_negative d :=
sorry

end NUMINAMATH_CALUDE_only_negative_number_l707_70749


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l707_70794

theorem greatest_value_quadratic_inequality :
  ∃ (a_max : ℝ), a_max = 8 ∧
  (∀ a : ℝ, a^2 - 12*a + 32 ≤ 0 → a ≤ a_max) ∧
  (a_max^2 - 12*a_max + 32 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l707_70794


namespace NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l707_70777

/-- Proves that the fraction of left-handed non-throwers is 1/3 given the conditions of the cricket team -/
theorem cricket_team_left_handed_fraction 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (right_handed : ℕ) 
  (h1 : total_players = 61) 
  (h2 : throwers = 37) 
  (h3 : right_handed = 53) 
  (h4 : throwers ≤ right_handed) : 
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l707_70777


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l707_70799

/-- Given a function f(x) = x^3 - ax^2 - x + a, where a is a real number,
    if f(x) has an extreme value at x = -1, then a = -1 -/
theorem extreme_value_implies_a_equals_negative_one (a : ℝ) :
  let f := λ x : ℝ => x^3 - a*x^2 - x + a
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f (-1) ≥ f x) ∨
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f (-1) ≤ f x) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l707_70799


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l707_70796

theorem quadratic_equation_solution :
  ∀ y : ℂ, 4 + 3 * y^2 = 0.7 * y - 40 ↔ y = (0.1167 : ℝ) + (3.8273 : ℝ) * I ∨ y = (0.1167 : ℝ) - (3.8273 : ℝ) * I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l707_70796


namespace NUMINAMATH_CALUDE_quadratic_transformation_l707_70752

theorem quadratic_transformation (x : ℝ) : x^2 - 8*x - 9 = (x - 4)^2 - 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l707_70752


namespace NUMINAMATH_CALUDE_recipe_people_l707_70731

/-- The number of people the original recipe is intended for -/
def P : ℕ := sorry

/-- The number of eggs required for the original recipe -/
def original_eggs : ℕ := 2

/-- The number of people Tyler wants to make the cake for -/
def tyler_people : ℕ := 8

/-- The number of eggs Tyler needs for his cake -/
def tyler_eggs : ℕ := 4

theorem recipe_people : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_recipe_people_l707_70731


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l707_70756

theorem unique_solution_square_equation :
  ∀ x y : ℕ+, x^2 = y^2 + 7*y + 6 ↔ x = 6 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l707_70756


namespace NUMINAMATH_CALUDE_sqrt_non_square_irrational_l707_70768

theorem sqrt_non_square_irrational (a : ℤ) 
  (h : ∀ n : ℤ, n^2 ≠ a) : 
  Irrational (Real.sqrt (a : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_non_square_irrational_l707_70768


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_triangular_prism_l707_70704

/-- Given a triangular prism with perimeter 12, prove that its maximum lateral surface area is 6 -/
theorem max_lateral_surface_area_triangular_prism :
  ∀ x y : ℝ, x > 0 → y > 0 → 6 * x + 3 * y = 12 →
  3 * x * y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_triangular_prism_l707_70704


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l707_70750

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 540) →
  total_land = 6000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l707_70750


namespace NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l707_70770

/-- The number of symmetry axes for a line segment -/
def line_segment_symmetry_axes : ℕ := 2

/-- The number of symmetry axes for an angle -/
def angle_symmetry_axes : ℕ := 1

/-- The minimum number of symmetry axes for an isosceles triangle -/
def isosceles_triangle_min_symmetry_axes : ℕ := 1

/-- The maximum number of symmetry axes for an isosceles triangle -/
def isosceles_triangle_max_symmetry_axes : ℕ := 3

/-- The number of symmetry axes for a square -/
def square_symmetry_axes : ℕ := 4

/-- Theorem stating that a square has the most symmetry axes among the given shapes -/
theorem square_has_most_symmetry_axes :
  square_symmetry_axes > line_segment_symmetry_axes ∧
  square_symmetry_axes > angle_symmetry_axes ∧
  square_symmetry_axes > isosceles_triangle_min_symmetry_axes ∧
  square_symmetry_axes > isosceles_triangle_max_symmetry_axes :=
sorry

end NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l707_70770


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l707_70714

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 440 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l707_70714


namespace NUMINAMATH_CALUDE_function_coefficient_sum_l707_70721

theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x - 2) = 2 * x^2 - 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_coefficient_sum_l707_70721


namespace NUMINAMATH_CALUDE_quartic_roots_equivalence_l707_70771

theorem quartic_roots_equivalence (x : ℂ) :
  (3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0) ↔
  (∃ y : ℂ, (y = x + 1/x) ∧
    ((y = (-1 + Real.sqrt 43) / 3) ∨ (y = (-1 - Real.sqrt 43) / 3))) := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_equivalence_l707_70771


namespace NUMINAMATH_CALUDE_trig_equation_solution_l707_70783

theorem trig_equation_solution (x : ℝ) : 
  2 * Real.cos x + Real.cos (3 * x) + Real.cos (5 * x) = 0 →
  ∃ (n : ℤ), x = n * (π / 4) ∧ n % 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l707_70783


namespace NUMINAMATH_CALUDE_asymptotic_lines_of_hyperbola_l707_70742

/-- The asymptotic lines of the hyperbola x²/9 - y² = 1 are y = ±x/3 -/
theorem asymptotic_lines_of_hyperbola :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/9 - y^2 = 1}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = x/3 ∨ y = -x/3}
  (∀ (p : ℝ × ℝ), p ∈ asymptotic_lines ↔ 
    (∃ (ε : ℝ → ℝ), (∀ t, t ≠ 0 → |ε t| < |t|) ∧
      ∀ t : ℝ, t ≠ 0 → (t*p.1 + ε t, t*p.2 + ε t) ∈ hyperbola)) :=
by sorry

end NUMINAMATH_CALUDE_asymptotic_lines_of_hyperbola_l707_70742


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l707_70781

/-- Calculates the total charge for a taxi trip -/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * charge_per_increment

/-- Proves that the total charge for a 3.6-mile trip is $4.95 -/
theorem taxi_charge_proof :
  let initial_fee : ℚ := 225/100
  let charge_per_increment : ℚ := 3/10
  let increment_distance : ℚ := 2/5
  let trip_distance : ℚ := 36/10
  calculate_taxi_charge initial_fee charge_per_increment increment_distance trip_distance = 495/100 := by
  sorry

#eval calculate_taxi_charge (225/100) (3/10) (2/5) (36/10)

end NUMINAMATH_CALUDE_taxi_charge_proof_l707_70781


namespace NUMINAMATH_CALUDE_apple_picking_theorem_l707_70718

/-- Represents the number of apples of each color in the bin -/
structure AppleBin :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (orange : ℕ)

/-- The minimum number of apples needed to guarantee a specific count of one color -/
def minApplesToGuarantee (bin : AppleBin) (targetCount : ℕ) : ℕ :=
  sorry

theorem apple_picking_theorem (bin : AppleBin) (h : bin = ⟨32, 24, 22, 15, 14⟩) :
  minApplesToGuarantee bin 18 = 81 :=
sorry

end NUMINAMATH_CALUDE_apple_picking_theorem_l707_70718
