import Mathlib

namespace NUMINAMATH_CALUDE_opposite_of_one_third_l58_5831

theorem opposite_of_one_third : 
  (opposite : ℚ → ℚ) (1/3) = -(1/3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_third_l58_5831


namespace NUMINAMATH_CALUDE_earth_sun_distance_in_scientific_notation_l58_5883

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3 * (10 ^ 8)

/-- The time it takes for sunlight to reach Earth in seconds -/
def time_to_earth : ℝ := 5 * (10 ^ 2)

/-- The distance between Earth and Sun in meters -/
def earth_sun_distance : ℝ := speed_of_light * time_to_earth

theorem earth_sun_distance_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), a ≥ 1 ∧ a < 10 ∧ earth_sun_distance = a * (10 ^ n) ∧ a = 1.5 ∧ n = 11 :=
sorry

end NUMINAMATH_CALUDE_earth_sun_distance_in_scientific_notation_l58_5883


namespace NUMINAMATH_CALUDE_inequality_proof_l58_5875

theorem inequality_proof (x : ℝ) (hx : x > 0) : 1 + x^2018 ≥ (2*x)^2017 / (1 + x)^2016 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l58_5875


namespace NUMINAMATH_CALUDE_group_size_problem_l58_5847

theorem group_size_problem (total_paise : ℕ) (h1 : total_paise = 4624) : ∃ n : ℕ, n * n = total_paise ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l58_5847


namespace NUMINAMATH_CALUDE_chef_michel_pies_sold_l58_5814

theorem chef_michel_pies_sold 
  (shepherds_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherds_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (h1 : shepherds_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherds_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  shepherds_pie_customers / shepherds_pie_pieces + 
  chicken_pot_pie_customers / chicken_pot_pie_pieces = 29 :=
by sorry

end NUMINAMATH_CALUDE_chef_michel_pies_sold_l58_5814


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l58_5802

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioc 0 1 → |a * x^3 - Real.log x| ≥ 1) → a ≥ Real.exp 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l58_5802


namespace NUMINAMATH_CALUDE_tangent_line_equations_l58_5898

/-- The function f(x) = x³ + 2 -/
def f (x : ℝ) : ℝ := x^3 + 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_equations (x : ℝ) :
  /- Part 1: Tangent line equation at x = 1 -/
  (∀ y : ℝ, (y - f 1) = f' 1 * (x - 1) ↔ 3 * x - y = 0) ∧
  /- Part 2: Tangent line equation passing through (0, 4) -/
  (∃ t : ℝ, t^3 + 2 = f t ∧
            4 - (t^3 + 2) = f' t * (0 - t) ∧
            (∀ y : ℝ, (y - f t) = f' t * (x - t) ↔ 3 * x - y + 4 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l58_5898


namespace NUMINAMATH_CALUDE_towels_used_is_285_l58_5867

/-- Calculates the total number of towels used in a gym over 4 hours -/
def totalTowelsUsed (firstHourGuests : ℕ) : ℕ :=
  let secondHourGuests := firstHourGuests + (firstHourGuests * 20 / 100)
  let thirdHourGuests := secondHourGuests + (secondHourGuests * 25 / 100)
  let fourthHourGuests := thirdHourGuests + (thirdHourGuests / 3)
  firstHourGuests + secondHourGuests + thirdHourGuests + fourthHourGuests

/-- Theorem stating that the total number of towels used is 285 -/
theorem towels_used_is_285 :
  totalTowelsUsed 50 = 285 := by
  sorry

#eval totalTowelsUsed 50

end NUMINAMATH_CALUDE_towels_used_is_285_l58_5867


namespace NUMINAMATH_CALUDE_inequality_equivalence_l58_5885

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 6 ∪ {6} := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l58_5885


namespace NUMINAMATH_CALUDE_no_odd_cube_ending_668_l58_5855

theorem no_odd_cube_ending_668 : ¬∃ (n : ℕ), 
  Odd n ∧ n > 0 ∧ n^3 % 1000 = 668 := by
  sorry

end NUMINAMATH_CALUDE_no_odd_cube_ending_668_l58_5855


namespace NUMINAMATH_CALUDE_remainder_twelve_thousand_one_hundred_eleven_div_three_l58_5880

theorem remainder_twelve_thousand_one_hundred_eleven_div_three : 
  12111 % 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_twelve_thousand_one_hundred_eleven_div_three_l58_5880


namespace NUMINAMATH_CALUDE_smallest_positive_t_l58_5842

theorem smallest_positive_t (x₁ x₂ x₃ x₄ x₅ t : ℝ) : 
  (x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) →
  (x₁ + x₂ + x₃ + x₄ + x₅ > 0) →
  (x₁ + x₃ = 2 * t * x₂) →
  (x₂ + x₄ = 2 * t * x₃) →
  (x₃ + x₅ = 2 * t * x₄) →
  t > 0 →
  t ≥ 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_t_l58_5842


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l58_5830

theorem sum_of_three_numbers : 6 + 8 + 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l58_5830


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l58_5825

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l58_5825


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_root_less_than_two_iff_l58_5800

/-- The quadratic equation x^2 - (k+4)x + 4k = 0 -/
def quadratic (k x : ℝ) : ℝ := x^2 - (k+4)*x + 4*k

theorem quadratic_two_real_roots (k : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 :=
sorry

theorem root_less_than_two_iff (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0 ∧ x < 2) ↔ k < 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_root_less_than_two_iff_l58_5800


namespace NUMINAMATH_CALUDE_value_of_a_l58_5803

theorem value_of_a (x a : ℝ) (hx : x ≠ 1) : 
  (8 * a) / (1 - x^32) = 2 / (1 - x) + 2 / (1 + x) + 4 / (1 + x^2) + 
                         8 / (1 + x^4) + 16 / (1 + x^8) + 32 / (1 + x^16) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l58_5803


namespace NUMINAMATH_CALUDE_same_solutions_imply_coefficients_l58_5818

-- Define the absolute value equation
def abs_equation (x : ℝ) : Prop := |x - 3| = 4

-- Define the quadratic equation
def quadratic_equation (x b c : ℝ) : Prop := x^2 + b*x + c = 0

-- Theorem statement
theorem same_solutions_imply_coefficients :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ abs_equation x₁ ∧ abs_equation x₂) →
  (∀ x : ℝ, abs_equation x ↔ ∃ b c : ℝ, quadratic_equation x b c) →
  ∃! (b c : ℝ), ∀ x : ℝ, abs_equation x ↔ quadratic_equation x b c :=
by sorry

end NUMINAMATH_CALUDE_same_solutions_imply_coefficients_l58_5818


namespace NUMINAMATH_CALUDE_proposition_q_undetermined_l58_5886

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) ∧ ¬(q ∧ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_q_undetermined_l58_5886


namespace NUMINAMATH_CALUDE_fourth_guard_theorem_l58_5844

/-- Represents a rectangular perimeter with guards at each corner -/
structure GuardedRectangle where
  perimeter : ℝ
  three_guard_distance : ℝ

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (rect : GuardedRectangle) : ℝ :=
  rect.perimeter - rect.three_guard_distance

/-- Theorem stating that for a rectangle with perimeter 1000 meters,
    if three guards run 850 meters, the fourth guard runs 150 meters -/
theorem fourth_guard_theorem (rect : GuardedRectangle)
  (h1 : rect.perimeter = 1000)
  (h2 : rect.three_guard_distance = 850) :
  fourth_guard_distance rect = 150 := by
  sorry

end NUMINAMATH_CALUDE_fourth_guard_theorem_l58_5844


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l58_5892

theorem fraction_sum_equality (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_sum : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 
  1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l58_5892


namespace NUMINAMATH_CALUDE_smallest_benches_proof_l58_5821

/-- The number of adults that can sit on one bench -/
def adults_per_bench : ℕ := 7

/-- The number of children that can sit on one bench -/
def children_per_bench : ℕ := 11

/-- A function that returns true if the given number of benches can seat an equal number of adults and children -/
def can_seat_equally (n : ℕ) : Prop :=
  ∃ (people : ℕ), people > 0 ∧ 
    n * adults_per_bench = people ∧
    n * children_per_bench = people

/-- The smallest number of benches that can seat an equal number of adults and children -/
def smallest_n : ℕ := 18

theorem smallest_benches_proof :
  (∀ m : ℕ, m > 0 → m < smallest_n → ¬(can_seat_equally m)) ∧
  can_seat_equally smallest_n :=
sorry

end NUMINAMATH_CALUDE_smallest_benches_proof_l58_5821


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l58_5815

theorem quadratic_equation_m_value : 
  ∃! m : ℝ, m^2 + 1 = 2 ∧ m - 1 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l58_5815


namespace NUMINAMATH_CALUDE_relay_race_proof_l58_5851

-- Define the total race distance
def total_distance : ℕ := 2004

-- Define the maximum time allowed (one week in hours)
def max_time : ℕ := 168

-- Define the properties of the race
theorem relay_race_proof :
  ∃ (stage_length : ℕ) (num_stages : ℕ),
    stage_length > 0 ∧
    num_stages > 0 ∧
    num_stages ≤ max_time ∧
    stage_length * num_stages = total_distance ∧
    num_stages = 167 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_proof_l58_5851


namespace NUMINAMATH_CALUDE_solve_equation_l58_5805

theorem solve_equation (x y z a b c : ℤ) 
  (hx : x = -2272)
  (hy : y = 10^3 + 10^2 * c + 10 * b + a)
  (hz : z = 1)
  (heq : a * x + b * y + c * z = 1)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hc_pos : c > 0)
  (hab : a < b)
  (hbc : b < c) :
  y = 1987 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l58_5805


namespace NUMINAMATH_CALUDE_only_13_remains_prime_l58_5882

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

def remains_prime_when_reversed (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem only_13_remains_prime : 
  (remains_prime_when_reversed 13) ∧ 
  (¬remains_prime_when_reversed 29) ∧ 
  (¬remains_prime_when_reversed 53) ∧ 
  (¬remains_prime_when_reversed 23) ∧ 
  (¬remains_prime_when_reversed 41) :=
sorry

end NUMINAMATH_CALUDE_only_13_remains_prime_l58_5882


namespace NUMINAMATH_CALUDE_vector_magnitude_l58_5884

def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (-1, 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_magnitude (t : ℝ) :
  parallel (a t) b →
  ‖(a t - b)‖ = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l58_5884


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l58_5812

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : x ≠ y
  /-- The perimeter is 20 -/
  perimeter : x + x + y = 20

/-- Properties of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  (t.y = -2 * t.x + 20) ∧ (5 < t.x ∧ t.x < 10) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l58_5812


namespace NUMINAMATH_CALUDE_area_BCD_equals_135_l58_5819

-- Define the triangle ABC
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the triangle BCD
def triangle_BCD : Set (ℝ × ℝ) := sorry

-- Define the area function
def area : Set (ℝ × ℝ) → ℝ := sorry

-- Define the length function
def length : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Define point A
def A : ℝ × ℝ := sorry

-- Define point C
def C : ℝ × ℝ := sorry

-- Define point D
def D : ℝ × ℝ := sorry

-- Theorem statement
theorem area_BCD_equals_135 :
  area triangle_ABC = 36 →
  length A C = 8 →
  length C D = 30 →
  area triangle_BCD = 135 := by
  sorry

end NUMINAMATH_CALUDE_area_BCD_equals_135_l58_5819


namespace NUMINAMATH_CALUDE_stratified_sampling_management_l58_5806

theorem stratified_sampling_management (total_employees : ℕ) (management_personnel : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : management_personnel = 32)
  (h3 : sample_size = 20) :
  (management_personnel : ℚ) * (sample_size : ℚ) / (total_employees : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_management_l58_5806


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l58_5829

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the complement of B in the universal set (real numbers)
def C_U_B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ C_U_B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l58_5829


namespace NUMINAMATH_CALUDE_friends_decks_count_l58_5857

/-- The number of decks Victor's friend bought -/
def friends_decks : ℕ := 2

/-- The cost of each deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Victor bought -/
def victors_decks : ℕ := 6

/-- The total amount spent by both Victor and his friend in dollars -/
def total_spent : ℕ := 64

theorem friends_decks_count : 
  deck_cost * (victors_decks + friends_decks) = total_spent := by sorry

end NUMINAMATH_CALUDE_friends_decks_count_l58_5857


namespace NUMINAMATH_CALUDE_binomial_max_prob_l58_5809

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The value of k that maximizes the probability mass function for B(200, 1/2) -/
theorem binomial_max_prob (ζ : ℕ → ℝ) (h : ∀ k, ζ k = binomial_pmf 200 (1/2) k) :
  ∃ k : ℕ, k = 100 ∧ ∀ j : ℕ, ζ k ≥ ζ j :=
sorry

end NUMINAMATH_CALUDE_binomial_max_prob_l58_5809


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l58_5841

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  Complex.im z = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l58_5841


namespace NUMINAMATH_CALUDE_clock_angle_at_2pm_l58_5890

/-- The number of hours on a standard clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a complete rotation -/
def full_rotation : ℕ := 360

/-- The number of degrees the hour hand moves per hour -/
def hour_hand_degrees_per_hour : ℚ := full_rotation / clock_hours

/-- The position of the hour hand at 2:00 -/
def hour_hand_position_at_2 : ℚ := 2 * hour_hand_degrees_per_hour

/-- The position of the minute hand at 2:00 -/
def minute_hand_position_at_2 : ℚ := 0

/-- The smaller angle between the hour hand and minute hand at 2:00 -/
def smaller_angle_at_2 : ℚ := hour_hand_position_at_2 - minute_hand_position_at_2

theorem clock_angle_at_2pm :
  smaller_angle_at_2 = 60 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_2pm_l58_5890


namespace NUMINAMATH_CALUDE_f_minus_g_at_one_l58_5840

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h x = -h (-x)

-- State the theorem
theorem f_minus_g_at_one
  (h1 : is_even f)
  (h2 : is_odd g)
  (h3 : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := by sorry

end NUMINAMATH_CALUDE_f_minus_g_at_one_l58_5840


namespace NUMINAMATH_CALUDE_janet_freelance_income_difference_l58_5853

/-- Calculates how much more Janet would make per month as a freelancer compared to her current job -/
theorem janet_freelance_income_difference :
  let hours_per_week : ℕ := 40
  let weeks_per_month : ℕ := 4
  let current_hourly_rate : ℚ := 30
  let freelance_hourly_rate : ℚ := 40
  let extra_fica_per_week : ℚ := 25
  let healthcare_premium_per_month : ℚ := 400

  let current_monthly_income := (hours_per_week * weeks_per_month : ℚ) * current_hourly_rate
  let freelance_monthly_income := (hours_per_week * weeks_per_month : ℚ) * freelance_hourly_rate
  let extra_costs_per_month := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month

  freelance_monthly_income - extra_costs_per_month - current_monthly_income = 1100 :=
by sorry

end NUMINAMATH_CALUDE_janet_freelance_income_difference_l58_5853


namespace NUMINAMATH_CALUDE_probability_red_then_blue_probability_red_then_blue_proof_l58_5810

/-- The probability of drawing a red marble first and a blue marble second from a bag containing 
    4 red marbles and 6 blue marbles, when drawing two marbles sequentially without replacement. -/
theorem probability_red_then_blue (red : ℕ) (blue : ℕ) 
    (h_red : red = 4) (h_blue : blue = 6) : ℚ :=
  4 / 15

/-- Proof of the theorem -/
theorem probability_red_then_blue_proof (red : ℕ) (blue : ℕ) 
    (h_red : red = 4) (h_blue : blue = 6) : 
    probability_red_then_blue red blue h_red h_blue = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_blue_probability_red_then_blue_proof_l58_5810


namespace NUMINAMATH_CALUDE_ariel_current_age_l58_5893

/-- Represents a person with birth year and fencing information -/
structure Person where
  birth_year : Nat
  fencing_start_year : Nat
  years_fencing : Nat

/-- Calculate the current year based on fencing information -/
def current_year (p : Person) : Nat :=
  p.fencing_start_year + p.years_fencing

/-- Calculate the age of a person in a given year -/
def age_in_year (p : Person) (year : Nat) : Nat :=
  year - p.birth_year

/-- Ariel's information -/
def ariel : Person :=
  { birth_year := 1992
  , fencing_start_year := 2006
  , years_fencing := 16 }

/-- Theorem: Ariel's current age is 30 years old -/
theorem ariel_current_age :
  age_in_year ariel (current_year ariel) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ariel_current_age_l58_5893


namespace NUMINAMATH_CALUDE_happy_formations_correct_l58_5813

def happy_formations (n : ℕ) : ℕ :=
  if n % 3 = 1 then 0
  else if n % 3 = 0 then
    (Nat.choose (n-1) (2*n/3) - Nat.choose (n-1) ((2*n+6)/3))^3 +
    (Nat.choose (n-1) ((2*n-3)/3) - Nat.choose (n-1) ((2*n+3)/3))^3
  else
    (Nat.choose (n-1) ((2*n-1)/3) - Nat.choose (n-1) ((2*n+1)/3))^3 +
    (Nat.choose (n-1) ((2*n-4)/3) - Nat.choose (n-1) ((2*n-2)/3))^3

theorem happy_formations_correct (n : ℕ) :
  happy_formations n =
    if n % 3 = 1 then 0
    else if n % 3 = 0 then
      (Nat.choose (n-1) (2*n/3) - Nat.choose (n-1) ((2*n+6)/3))^3 +
      (Nat.choose (n-1) ((2*n-3)/3) - Nat.choose (n-1) ((2*n+3)/3))^3
    else
      (Nat.choose (n-1) ((2*n-1)/3) - Nat.choose (n-1) ((2*n+1)/3))^3 +
      (Nat.choose (n-1) ((2*n-4)/3) - Nat.choose (n-1) ((2*n-2)/3))^3 :=
by sorry

end NUMINAMATH_CALUDE_happy_formations_correct_l58_5813


namespace NUMINAMATH_CALUDE_boot_shoe_price_difference_l58_5827

-- Define the price of shoes and boots as real numbers
variable (S B : ℝ)

-- Monday's sales equation
axiom monday_sales : 22 * S + 16 * B = 460

-- Tuesday's sales equation
axiom tuesday_sales : 8 * S + 32 * B = 560

-- Theorem stating the price difference between boots and shoes
theorem boot_shoe_price_difference : B - S = 5 := by sorry

end NUMINAMATH_CALUDE_boot_shoe_price_difference_l58_5827


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l58_5871

theorem perfect_cube_units_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n : ℕ, n ∈ S ↔ ∃ m : ℕ, n = m^3 % 10) ∧ 
    Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l58_5871


namespace NUMINAMATH_CALUDE_gcd_102_238_l58_5837

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l58_5837


namespace NUMINAMATH_CALUDE_california_texas_plates_equal_l58_5824

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_digits^3 * num_letters^3

/-- Theorem stating that California and Texas can issue the same number of license plates -/
theorem california_texas_plates_equal : california_plates = texas_plates := by
  sorry

end NUMINAMATH_CALUDE_california_texas_plates_equal_l58_5824


namespace NUMINAMATH_CALUDE_brenda_friends_count_l58_5820

/-- Prove that Brenda has 9 friends given the pizza ordering scenario -/
theorem brenda_friends_count :
  let slices_per_person : ℕ := 2
  let slices_per_pizza : ℕ := 4
  let pizzas_ordered : ℕ := 5
  let total_slices : ℕ := slices_per_pizza * pizzas_ordered
  let total_people : ℕ := total_slices / slices_per_person
  let brenda_friends : ℕ := total_people - 1
  brenda_friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_brenda_friends_count_l58_5820


namespace NUMINAMATH_CALUDE_fraction_equality_l58_5858

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 4 * y) = 3) : 
  (2 * x + 4 * y) / (4 * x - 2 * y) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l58_5858


namespace NUMINAMATH_CALUDE_rain_probability_l58_5834

theorem rain_probability (umbrellas : ℕ) (take_umbrella_prob : ℝ) :
  umbrellas = 2 →
  take_umbrella_prob = 0.2 →
  ∃ (rain_prob : ℝ),
    rain_prob + (rain_prob / (rain_prob + 1)) - (rain_prob^2 / (rain_prob + 1)) = take_umbrella_prob ∧
    rain_prob = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l58_5834


namespace NUMINAMATH_CALUDE_equation_not_linear_l58_5807

/-- A linear equation in two variables contains exactly two variables and the highest degree of terms involving these variables is 1. -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

/-- The equation xy = 3 -/
def equation (x y : ℝ) : ℝ := x * y - 3

theorem equation_not_linear : ¬ is_linear_equation_in_two_variables equation := by
  sorry

end NUMINAMATH_CALUDE_equation_not_linear_l58_5807


namespace NUMINAMATH_CALUDE_survey_problem_l58_5877

theorem survey_problem (A B : ℕ) : 
  (A * 20 = B * 50) →                   -- 20% of A equals 50% of B (students who read both books)
  (A - A * 20 / 100) - (B - B * 50 / 100) = 150 →  -- Difference between those who read only A and only B
  (A - A * 20 / 100) + (B - B * 50 / 100) + (A * 20 / 100) = 300 :=  -- Total number of students
by sorry

end NUMINAMATH_CALUDE_survey_problem_l58_5877


namespace NUMINAMATH_CALUDE_baseball_league_games_l58_5839

/-- The number of games played in a baseball league --/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with every other team,
    the total number of games played is 180. --/
theorem baseball_league_games :
  total_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_l58_5839


namespace NUMINAMATH_CALUDE_log_sum_equals_three_main_theorem_l58_5868

theorem log_sum_equals_three : Real.log 8 + 3 * Real.log 5 = 3 * Real.log 10 := by
  sorry

theorem main_theorem : Real.log 8 + 3 * Real.log 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_main_theorem_l58_5868


namespace NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equation_l58_5836

theorem no_distinct_complex_numbers_satisfying_equation :
  ∀ (a b c d : ℂ), 
    (a^3 - b*c*d = b^3 - a*c*d) ∧ 
    (b^3 - a*c*d = c^3 - a*b*d) ∧ 
    (c^3 - a*b*d = d^3 - a*b*c) →
    (a = b) ∨ (a = c) ∨ (a = d) ∨ (b = c) ∨ (b = d) ∨ (c = d) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equation_l58_5836


namespace NUMINAMATH_CALUDE_gcd_sequence_limit_l58_5861

theorem gcd_sequence_limit (n : ℕ) : 
  ∃ N : ℕ, ∀ m : ℕ, m ≥ N → 
    Nat.gcd (100 + 2 * m^2) (100 + 2 * (m + 1)^2) = 1 := by
  sorry

#check gcd_sequence_limit

end NUMINAMATH_CALUDE_gcd_sequence_limit_l58_5861


namespace NUMINAMATH_CALUDE_right_triangle_third_side_square_l58_5846

theorem right_triangle_third_side_square (a b c : ℝ) : 
  a = 6 → b = 8 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c^2 = 28 ∨ c^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_square_l58_5846


namespace NUMINAMATH_CALUDE_accounting_course_count_l58_5869

/-- Represents the number of employees who took an accounting course -/
def accounting_course : ℕ := sorry

/-- Represents the number of employees who took a finance course -/
def finance_course : ℕ := 14

/-- Represents the number of employees who took a marketing course -/
def marketing_course : ℕ := 15

/-- Represents the number of employees who took exactly two courses -/
def two_courses : ℕ := 10

/-- Represents the number of employees who took all three courses -/
def all_courses : ℕ := 1

/-- Represents the number of employees who took none of the courses -/
def no_courses : ℕ := 11

/-- The total number of employees -/
def total_employees : ℕ := 50

theorem accounting_course_count : accounting_course = 19 := by
  sorry

end NUMINAMATH_CALUDE_accounting_course_count_l58_5869


namespace NUMINAMATH_CALUDE_job_completion_time_l58_5852

theorem job_completion_time (job : ℝ) (days_A : ℝ) (efficiency_C : ℝ) : 
  job > 0 → days_A > 0 → efficiency_C > 0 →
  (job / days_A) * efficiency_C * 16 = job :=
by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l58_5852


namespace NUMINAMATH_CALUDE_lesser_fraction_l58_5823

theorem lesser_fraction (x y : ℝ) (sum_eq : x + y = 7/8) (prod_eq : x * y = 1/12) :
  min x y = (7 - Real.sqrt 17) / 16 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l58_5823


namespace NUMINAMATH_CALUDE_amara_clothes_thrown_away_l58_5816

/-- The number of clothes Amara threw away -/
def clothes_thrown_away (initial_count donated_first donated_second remaining_count : ℕ) : ℕ :=
  initial_count - donated_first - donated_second - remaining_count

/-- Proof that Amara threw away 15 pieces of clothing -/
theorem amara_clothes_thrown_away :
  clothes_thrown_away 100 5 (3 * 5) 65 = 15 := by
  sorry

end NUMINAMATH_CALUDE_amara_clothes_thrown_away_l58_5816


namespace NUMINAMATH_CALUDE_line_properties_l58_5881

/-- Represents a line in the form ax + 3y + 1 = 0 -/
structure Line where
  a : ℝ

/-- Checks if the intercepts of the line on the coordinate axes are equal -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.a = 3

/-- Checks if the line l is parallel to the line x + (a-2)y + a = 0 -/
def is_parallel_to_given_line (l : Line) : Prop :=
  l.a * (l.a - 2) - 3 = 0 ∧ l.a^2 - 1 ≠ 0

theorem line_properties (l : Line) :
  (has_equal_intercepts l ↔ l.a = 3) ∧
  (is_parallel_to_given_line l ↔ l.a = 3) := by sorry

end NUMINAMATH_CALUDE_line_properties_l58_5881


namespace NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l58_5887

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  (a / (4 - a)) + (1 / (a - 1)) ≥ 2 :=
sorry

theorem existence_of_minimum (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  ∃ a, (a / (4 - a)) + (1 / (a - 1)) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l58_5887


namespace NUMINAMATH_CALUDE_floor_equation_solution_l58_5859

theorem floor_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 90) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l58_5859


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l58_5811

theorem smallest_addition_for_divisibility : 
  ∃! x : ℕ, x < 169 ∧ (2714 + x) % 169 = 0 ∧ ∀ y : ℕ, y < x → (2714 + y) % 169 ≠ 0 :=
by
  use 119
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l58_5811


namespace NUMINAMATH_CALUDE_twelveSidedFigureArea_l58_5899

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices --/
structure Polygon where
  vertices : List Point

/-- The area of a polygon --/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- Our specific 12-sided figure --/
def twelveSidedFigure : Polygon := {
  vertices := [
    { x := 2, y := 1 }, { x := 3, y := 2 }, { x := 3, y := 3 }, { x := 5, y := 3 },
    { x := 6, y := 4 }, { x := 5, y := 5 }, { x := 4, y := 5 }, { x := 3, y := 6 },
    { x := 2, y := 5 }, { x := 2, y := 4 }, { x := 1, y := 3 }, { x := 2, y := 2 }
  ]
}

theorem twelveSidedFigureArea : area twelveSidedFigure = 12 := by sorry

end NUMINAMATH_CALUDE_twelveSidedFigureArea_l58_5899


namespace NUMINAMATH_CALUDE_max_truthful_students_2015_l58_5848

/-- The maximum number of truthful students in the described arrangement --/
def max_truthful_students (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that for n = 2015, the maximum number of truthful students is 2031120 --/
theorem max_truthful_students_2015 :
  max_truthful_students 2015 = 2031120 := by
  sorry

#eval max_truthful_students 2015

end NUMINAMATH_CALUDE_max_truthful_students_2015_l58_5848


namespace NUMINAMATH_CALUDE_exponent_multiplication_l58_5888

theorem exponent_multiplication (a : ℝ) : a * a^2 * a^3 = a^6 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l58_5888


namespace NUMINAMATH_CALUDE_number_equality_l58_5864

theorem number_equality (x : ℝ) : (0.4 * x = 0.25 * 80) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l58_5864


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l58_5865

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1) ↔ (∃ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l58_5865


namespace NUMINAMATH_CALUDE_sum_cos_dihedral_angles_eq_one_l58_5894

/-- A trihedral angle is a three-dimensional figure formed by three planes intersecting at a point. -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  plane_angles : Fin 3 → ℝ
  /-- The sum of the plane angles is 180° (π radians) -/
  sum_plane_angles : (plane_angles 0) + (plane_angles 1) + (plane_angles 2) = π

/-- The dihedral angles of a trihedral angle -/
def dihedral_angles (t : TrihedralAngle) : Fin 3 → ℝ := sorry

/-- Theorem: For a trihedral angle with plane angles summing to 180°, 
    the sum of the cosines of its dihedral angles is equal to 1 -/
theorem sum_cos_dihedral_angles_eq_one (t : TrihedralAngle) : 
  (Real.cos (dihedral_angles t 0)) + (Real.cos (dihedral_angles t 1)) + (Real.cos (dihedral_angles t 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_cos_dihedral_angles_eq_one_l58_5894


namespace NUMINAMATH_CALUDE_parabola_directrix_l58_5891

/-- Represents a parabola with equation x² = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => y = -1

theorem parabola_directrix (p : Parabola) : 
  directrix p = fun y => y = -1 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l58_5891


namespace NUMINAMATH_CALUDE_polynomial_roots_l58_5804

theorem polynomial_roots (r : ℝ) : 
  r^2 = r + 1 → r^5 = 5*r + 3 ∧ ∀ b c : ℤ, (∀ s : ℝ, s^2 = s + 1 → s^5 = b*s + c) → b = 5 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l58_5804


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l58_5862

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l58_5862


namespace NUMINAMATH_CALUDE_ball_box_arrangements_l58_5872

/-- The number of different arrangements of 4 balls in 4 boxes -/
def arrangements (n : ℕ) : ℕ := sorry

/-- The number of arrangements where exactly one box contains 2 balls -/
def one_box_two_balls : ℕ := arrangements 1

/-- The number of arrangements where exactly two boxes are left empty -/
def two_boxes_empty : ℕ := arrangements 2

theorem ball_box_arrangements :
  (one_box_two_balls = 144) ∧ (two_boxes_empty = 84) := by sorry

end NUMINAMATH_CALUDE_ball_box_arrangements_l58_5872


namespace NUMINAMATH_CALUDE_gideon_marbles_fraction_l58_5863

/-- The fraction of marbles Gideon gave to his sister -/
def fraction_given : ℚ := 3/4

theorem gideon_marbles_fraction :
  ∀ (f : ℚ),
  (100 : ℚ) = 100 →  -- Gideon has 100 marbles
  (45 : ℚ) = 45 →    -- Gideon is currently 45 years old
  2 * ((1 - f) * 100) = (45 + 5 : ℚ) →  -- After giving fraction f and doubling, he gets his age in 5 years
  f = fraction_given :=
by sorry

end NUMINAMATH_CALUDE_gideon_marbles_fraction_l58_5863


namespace NUMINAMATH_CALUDE_min_mn_value_l58_5856

theorem min_mn_value (m : ℝ) (n : ℝ) (h_m : m > 0) 
  (h_ineq : ∀ x : ℝ, x > -m → x + m ≤ Real.exp ((2 * x / m) + n)) :
  ∃ (min_mn : ℝ), min_mn = -2 / Real.exp 2 ∧ 
    ∀ (m' n' : ℝ), (∀ x : ℝ, x > -m' → x + m' ≤ Real.exp ((2 * x / m') + n')) → 
      m' * n' ≥ min_mn :=
sorry

end NUMINAMATH_CALUDE_min_mn_value_l58_5856


namespace NUMINAMATH_CALUDE_min_value_quadratic_plus_constant_l58_5801

theorem min_value_quadratic_plus_constant :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 ≥ 2018) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_plus_constant_l58_5801


namespace NUMINAMATH_CALUDE_spilled_bag_candies_l58_5874

theorem spilled_bag_candies (bags : ℕ) (average : ℕ) (known_bags : List ℕ) : 
  bags = 8 → 
  average = 22 → 
  known_bags = [12, 14, 18, 22, 24, 26, 29] → 
  (List.sum known_bags + (bags - known_bags.length) * average - List.sum known_bags) = 31 := by
  sorry

end NUMINAMATH_CALUDE_spilled_bag_candies_l58_5874


namespace NUMINAMATH_CALUDE_factorization_equality_l58_5873

theorem factorization_equality (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l58_5873


namespace NUMINAMATH_CALUDE_phillips_money_l58_5895

/-- The amount of money Phillip's mother gave him -/
def total_money : ℕ := sorry

/-- The amount Phillip spent on oranges -/
def oranges_cost : ℕ := 14

/-- The amount Phillip spent on apples -/
def apples_cost : ℕ := 25

/-- The amount Phillip spent on candy -/
def candy_cost : ℕ := 6

/-- The amount Phillip has left -/
def money_left : ℕ := 50

/-- Theorem stating that the total money given by Phillip's mother
    is equal to the sum of his expenses plus the amount left -/
theorem phillips_money :
  total_money = oranges_cost + apples_cost + candy_cost + money_left :=
sorry

end NUMINAMATH_CALUDE_phillips_money_l58_5895


namespace NUMINAMATH_CALUDE_inequality_minimum_l58_5850

theorem inequality_minimum (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_minimum_l58_5850


namespace NUMINAMATH_CALUDE_intercepts_sum_l58_5828

theorem intercepts_sum (x₀ y₀ : ℕ) : 
  x₀ < 42 → y₀ < 42 → 
  (5 * x₀) % 42 = 40 → 
  (3 * y₀) % 42 = 2 → 
  x₀ + y₀ = 36 := by
sorry

end NUMINAMATH_CALUDE_intercepts_sum_l58_5828


namespace NUMINAMATH_CALUDE_correct_sqrt_calculation_l58_5826

theorem correct_sqrt_calculation :
  (∃ (x y z : ℝ), x = Real.sqrt 2 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 6 ∧ x * y = z) ∧
  (∀ (x y z : ℝ), x = Real.sqrt 2 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 5 → x + y ≠ z) ∧
  (∀ (x y : ℝ), x = Real.sqrt 3 ∧ y = Real.sqrt 2 → x - y ≠ 1) ∧
  (∀ (x y : ℝ), x = Real.sqrt 4 ∧ y = Real.sqrt 2 → x / y ≠ 2) :=
by sorry


end NUMINAMATH_CALUDE_correct_sqrt_calculation_l58_5826


namespace NUMINAMATH_CALUDE_exam_marks_calculation_l58_5817

theorem exam_marks_calculation (T : ℕ) : 
  (T * 20 / 100 + 40 = 160) → 
  (T * 30 / 100 - 160 = 20) := by
  sorry

end NUMINAMATH_CALUDE_exam_marks_calculation_l58_5817


namespace NUMINAMATH_CALUDE_triangle_side_length_l58_5897

theorem triangle_side_length (AB BC AC : ℝ) : 
  AB = 6 → BC = 4 → 2 < AC ∧ AC < 10 → AC = 5 → True :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l58_5897


namespace NUMINAMATH_CALUDE_weightlifter_total_lift_l58_5808

/-- The weight a weightlifter can lift in one hand -/
def weight_per_hand : ℕ := 7

/-- The number of hands a weightlifter has -/
def number_of_hands : ℕ := 2

/-- The total weight a weightlifter can lift at once -/
def total_weight : ℕ := weight_per_hand * number_of_hands

/-- Theorem: The total weight a weightlifter can lift at once is 14 pounds -/
theorem weightlifter_total_lift : total_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_weightlifter_total_lift_l58_5808


namespace NUMINAMATH_CALUDE_camp_skills_l58_5822

theorem camp_skills (total : ℕ) (cant_sing cant_dance cant_perform : ℕ) :
  total = 100 ∧
  cant_sing = 42 ∧
  cant_dance = 65 ∧
  cant_perform = 29 →
  ∃ (only_sing only_dance only_perform sing_dance sing_perform dance_perform : ℕ),
    only_sing + only_dance + only_perform + sing_dance + sing_perform + dance_perform = total ∧
    only_dance + only_perform + dance_perform = cant_sing ∧
    only_sing + only_perform + sing_perform = cant_dance ∧
    only_sing + only_dance + sing_dance = cant_perform ∧
    sing_dance + sing_perform + dance_perform = 64 :=
by sorry

end NUMINAMATH_CALUDE_camp_skills_l58_5822


namespace NUMINAMATH_CALUDE_no_negative_roots_l58_5849

/-- Given f(x) = a^x + (x-2)/(x+1) where a > 1, prove that f(x) ≠ 0 for all x < 0 -/
theorem no_negative_roots (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, x < 0 → a^x + (x - 2) / (x + 1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l58_5849


namespace NUMINAMATH_CALUDE_max_circumference_error_l58_5870

def actual_radius : ℝ := 15
def max_error_rate : ℝ := 0.25

theorem max_circumference_error :
  let min_measured_radius := actual_radius * (1 - max_error_rate)
  let max_measured_radius := actual_radius * (1 + max_error_rate)
  let actual_circumference := 2 * Real.pi * actual_radius
  let min_computed_circumference := 2 * Real.pi * min_measured_radius
  let max_computed_circumference := 2 * Real.pi * max_measured_radius
  let min_error := (actual_circumference - min_computed_circumference) / actual_circumference
  let max_error := (max_computed_circumference - actual_circumference) / actual_circumference
  max min_error max_error = max_error_rate :=
by sorry

end NUMINAMATH_CALUDE_max_circumference_error_l58_5870


namespace NUMINAMATH_CALUDE_problem_statement_l58_5845

theorem problem_statement (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 90 →
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 4)
  y = 39 ∨ y = 6 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l58_5845


namespace NUMINAMATH_CALUDE_inequality_system_subset_circle_l58_5896

theorem inequality_system_subset_circle (m : ℝ) :
  m > 0 →
  (∀ x y : ℝ, x - 2*y + 5 ≥ 0 ∧ 3 - x ≥ 0 ∧ x + y ≥ 0 → x^2 + y^2 ≤ m^2) →
  m ≥ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_subset_circle_l58_5896


namespace NUMINAMATH_CALUDE_charity_draw_winnings_calculation_l58_5838

/-- Calculates the charity draw winnings given initial amount, expenses, lottery winnings, and final amount -/
def charity_draw_winnings (initial_amount expenses lottery_winnings final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount - expenses + lottery_winnings)

/-- Theorem stating that given the specific values from the problem, the charity draw winnings must be 19 -/
theorem charity_draw_winnings_calculation :
  charity_draw_winnings 10 (4 + 1 + 1) 65 94 = 19 := by
  sorry

end NUMINAMATH_CALUDE_charity_draw_winnings_calculation_l58_5838


namespace NUMINAMATH_CALUDE_proposition_analysis_l58_5854

theorem proposition_analysis :
  let converse := ∀ a b c : ℝ, a * c^2 > b * c^2 → a > b
  let negation := ∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2
  let contrapositive := ∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a ≤ b
  (converse ∧ negation ∧ ¬contrapositive) ∨
  (converse ∧ ¬negation ∧ contrapositive) ∨
  (¬converse ∧ negation ∧ contrapositive) :=
by sorry

#check proposition_analysis

end NUMINAMATH_CALUDE_proposition_analysis_l58_5854


namespace NUMINAMATH_CALUDE_cubic_function_sum_l58_5832

-- Define the function f
def f (a b c x : ℤ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_sum (a b c : ℤ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧   -- a, b, c are non-zero
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧   -- a, b, c are distinct
  f a b c a = a^3 ∧         -- f(a) = a^3
  f a b c b = b^3           -- f(b) = b^3
  → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_sum_l58_5832


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l58_5876

/-- The parabolas y = (x - 1)^2 and x - 3 = (y + 2)^2 intersect at four points that lie on a circle with radius squared equal to 1/2 -/
theorem intersection_points_on_circle : ∃ (c : ℝ × ℝ) (r : ℝ),
  (∀ (p : ℝ × ℝ), (p.2 = (p.1 - 1)^2 ∧ p.1 - 3 = (p.2 + 2)^2) →
    ((p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2)) ∧
  r^2 = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l58_5876


namespace NUMINAMATH_CALUDE_product_without_x2_x3_implies_p_plus_q_eq_neg_four_l58_5833

theorem product_without_x2_x3_implies_p_plus_q_eq_neg_four (p q : ℝ) :
  (∀ x : ℝ, (x^2 + p) * (x^2 - q*x + 4) = x^4 + (-p*q)*x + 4*p) →
  p + q = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_without_x2_x3_implies_p_plus_q_eq_neg_four_l58_5833


namespace NUMINAMATH_CALUDE_ratio_abc_l58_5843

theorem ratio_abc (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2*b + 3*c)^2) :
  ∃ k : ℝ, a = k ∧ b = 2*k ∧ c = 3*k :=
sorry

end NUMINAMATH_CALUDE_ratio_abc_l58_5843


namespace NUMINAMATH_CALUDE_power_division_equality_l58_5860

theorem power_division_equality : (2 ^ 24) / (8 ^ 3) = 32768 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l58_5860


namespace NUMINAMATH_CALUDE_inverse_A_times_B_l58_5878

theorem inverse_A_times_B : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 5]
  A⁻¹ * B = !![1/2, 1; -1/2, 5] := by sorry

end NUMINAMATH_CALUDE_inverse_A_times_B_l58_5878


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l58_5879

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l58_5879


namespace NUMINAMATH_CALUDE_triangle_property_l58_5889

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((Real.sin A + Real.sin B) * (a - b) = c * (Real.sin C - Real.sin B)) →
  (D.1 * (B / (B + C)) + D.2 * (C / (B + C)) = 2) →
  (A / 2 = Real.arctan ((D.2 - D.1) / 2)) →
  (A = π / 3 ∧ 4 * Real.sqrt 3 / 3 ≤ (1 / 2) * a * b * Real.sin C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l58_5889


namespace NUMINAMATH_CALUDE_sues_family_travel_l58_5866

/-- Given a constant speed and travel time, calculates the distance traveled -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Sue's family traveled 300 miles to the campground -/
theorem sues_family_travel : distance_traveled 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sues_family_travel_l58_5866


namespace NUMINAMATH_CALUDE_lcm_lower_bound_l58_5835

theorem lcm_lower_bound (a : Fin 10 → ℕ) (h_order : ∀ i j, i < j → a i < a j) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) ≥ 10 * a 0 := by
  sorry

end NUMINAMATH_CALUDE_lcm_lower_bound_l58_5835
