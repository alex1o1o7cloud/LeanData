import Mathlib

namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_12_l3737_373743

theorem sin_alpha_plus_pi_12 (α : ℝ) 
  (h1 : α ∈ Set.Ioo (-π/3) 0)
  (h2 : Real.cos (α + π/6) - Real.sin α = 4*Real.sqrt 3/5) :
  Real.sin (α + π/12) = -Real.sqrt 2/10 := by sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_12_l3737_373743


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l3737_373745

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l3737_373745


namespace NUMINAMATH_CALUDE_janes_tulip_bulbs_l3737_373740

theorem janes_tulip_bulbs :
  ∀ (T : ℕ),
    (T + T / 2 + 30 + 90 = 150) →
    T = 20 := by
  sorry

end NUMINAMATH_CALUDE_janes_tulip_bulbs_l3737_373740


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_one_l3737_373770

def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1

theorem monotonic_decreasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_one_l3737_373770


namespace NUMINAMATH_CALUDE_john_haircut_tip_percentage_l3737_373746

/-- Represents the growth rate of John's hair in inches per month -/
def hair_growth_rate : ℝ := 1.5

/-- Represents the length of John's hair in inches when he gets a haircut -/
def hair_length_at_cut : ℝ := 9

/-- Represents the length of John's hair in inches after a haircut -/
def hair_length_after_cut : ℝ := 6

/-- Represents the cost of a single haircut in dollars -/
def haircut_cost : ℝ := 45

/-- Represents the total amount John spends on haircuts in a year in dollars -/
def annual_haircut_spend : ℝ := 324

/-- Theorem stating that the percentage of the tip John gives for a haircut is 20% -/
theorem john_haircut_tip_percentage :
  let hair_growth_between_cuts := hair_length_at_cut - hair_length_after_cut
  let months_between_cuts := hair_growth_between_cuts / hair_growth_rate
  let haircuts_per_year := 12 / months_between_cuts
  let total_cost_per_haircut := annual_haircut_spend / haircuts_per_year
  let tip_amount := total_cost_per_haircut - haircut_cost
  let tip_percentage := (tip_amount / haircut_cost) * 100
  tip_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_john_haircut_tip_percentage_l3737_373746


namespace NUMINAMATH_CALUDE_smallest_square_l3737_373751

theorem smallest_square (a b : ℕ+) 
  (h1 : ∃ r : ℕ+, (15 * a + 16 * b : ℕ) = r ^ 2)
  (h2 : ∃ s : ℕ+, (16 * a - 15 * b : ℕ) = s ^ 2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 ^ 2 ∧
  ∃ (a₀ b₀ : ℕ+), (15 * a₀ + 16 * b₀ : ℕ) = 481 ^ 2 ∧ (16 * a₀ - 15 * b₀ : ℕ) = 481 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_l3737_373751


namespace NUMINAMATH_CALUDE_max_value_of_a_l3737_373706

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≤ 3/2 ∧ ∃ a₀ : ℝ, a₀ ≤ 3/2 ∧ ∀ x : ℝ, determinant (x - 1) (a₀ - 2) (a₀ + 1) x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3737_373706


namespace NUMINAMATH_CALUDE_x_equals_one_l3737_373767

theorem x_equals_one (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_l3737_373767


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l3737_373713

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - 2*y = 5 ∧ c*x + 3*y = 2) ↔ -3/2 < c ∧ c < 2/5 :=
sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l3737_373713


namespace NUMINAMATH_CALUDE_tv_show_total_watch_time_l3737_373765

theorem tv_show_total_watch_time :
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let extra_episodes_in_final_season : ℕ := 4
  let hours_per_episode : ℚ := 1/2

  let total_episodes : ℕ := 
    regular_seasons * episodes_per_regular_season + 
    (episodes_per_regular_season + extra_episodes_in_final_season)

  let total_watch_time : ℚ := total_episodes * hours_per_episode

  total_watch_time = 112 := by sorry

end NUMINAMATH_CALUDE_tv_show_total_watch_time_l3737_373765


namespace NUMINAMATH_CALUDE_odd_selections_from_eleven_l3737_373785

theorem odd_selections_from_eleven (n : ℕ) (h : n = 11) :
  (Finset.range n).sum (fun k => if k % 2 = 1 then Nat.choose n k else 0) = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_odd_selections_from_eleven_l3737_373785


namespace NUMINAMATH_CALUDE_estimate_student_population_l3737_373761

theorem estimate_student_population (n : ℕ) 
  (h1 : n > 0) 
  (h2 : 80 ≤ n) 
  (h3 : 100 ≤ n) : 
  (80 : ℝ) / n * 100 = 20 → n = 400 := by
  sorry

end NUMINAMATH_CALUDE_estimate_student_population_l3737_373761


namespace NUMINAMATH_CALUDE_inequality_theorem_l3737_373733

theorem inequality_theorem (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3737_373733


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l3737_373700

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ), c > 0 ∧ 
  (∀ (d : ℕ), d > 0 → (∃ (x y : ℕ), Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = (x * Real.sqrt 6 + y * Real.sqrt 8) / d) → c ≤ d) ∧
  Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = (a * Real.sqrt 6 + b * Real.sqrt 8) / c ∧
  a + b + c = 280 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l3737_373700


namespace NUMINAMATH_CALUDE_half_sqrt_is_one_l3737_373722

theorem half_sqrt_is_one (x : ℝ) : (1/2 : ℝ) * Real.sqrt x = 1 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_half_sqrt_is_one_l3737_373722


namespace NUMINAMATH_CALUDE_factorial_a_ratio_l3737_373786

/-- Definition of n_a! for positive n and a -/
def factorial_a (n a : ℕ) : ℕ :=
  (List.range ((n / a) + 1)).foldl (fun acc k => acc * (n - k * a)) n

/-- Theorem stating that 96_4! / 48_3! = 2^8 -/
theorem factorial_a_ratio : (factorial_a 96 4) / (factorial_a 48 3) = 2^8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_a_ratio_l3737_373786


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l3737_373797

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1050 → 
  (∀ p : ℕ, Prime p ∧ p ≤ 31 → ¬(p ∣ n)) → 
  Prime n ∨ n = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l3737_373797


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l3737_373707

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (t - 5, -2)
  let Q : ℝ × ℝ := (-3, t + 4)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let midpoint_to_endpoint_sq := ((midpoint.1 - P.1)^2 + (midpoint.2 - P.2)^2)
  midpoint_to_endpoint_sq = t^2 / 3 →
  t = -12 - 2 * Real.sqrt 21 ∨ t = -12 + 2 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l3737_373707


namespace NUMINAMATH_CALUDE_mika_stickers_l3737_373776

/-- The number of stickers Mika has left after various transactions --/
def stickers_left (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika is left with 2 stickers --/
theorem mika_stickers :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l3737_373776


namespace NUMINAMATH_CALUDE_product_xyz_w_l3737_373719

theorem product_xyz_w (x y z w : ℚ) 
  (eq1 : 3 * x + 4 * y = 60)
  (eq2 : 6 * x - 4 * y = 12)
  (eq3 : 2 * x - 3 * z = 38)
  (eq4 : x + y + z = w) :
  x * y * z * w = -5104 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_w_l3737_373719


namespace NUMINAMATH_CALUDE_candy_distribution_l3737_373783

theorem candy_distribution (total_candy : ℕ) (family_members : ℕ) 
  (h1 : total_candy = 45) (h2 : family_members = 5) : 
  total_candy % family_members = 0 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3737_373783


namespace NUMINAMATH_CALUDE_imo_2007_hktst_1_problem_6_l3737_373799

theorem imo_2007_hktst_1_problem_6 :
  ∀ x y : ℕ+, 
    (∃ k : ℕ+, x = 11 * k^2 ∧ y = 11 * k) ↔ 
    ∃ n : ℤ, (x.val^2 * y.val + x.val + y.val : ℤ) = n * (x.val * y.val^2 + y.val + 11) := by
  sorry

end NUMINAMATH_CALUDE_imo_2007_hktst_1_problem_6_l3737_373799


namespace NUMINAMATH_CALUDE_inequality_proof_l3737_373782

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) + 
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) + 
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3737_373782


namespace NUMINAMATH_CALUDE_company_fund_problem_l3737_373787

theorem company_fund_problem (n : ℕ) : 
  (∀ (initial_fund : ℕ),
    initial_fund = 60 * n - 10 ∧ 
    initial_fund = 50 * n + 110) →
  60 * n - 10 = 710 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l3737_373787


namespace NUMINAMATH_CALUDE_simplify_fraction_l3737_373711

theorem simplify_fraction (a : ℚ) (h : a = -2) : 18 * a^5 / (27 * a^3) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3737_373711


namespace NUMINAMATH_CALUDE_units_digit_of_3968_pow_805_l3737_373798

theorem units_digit_of_3968_pow_805 : (3968^805) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3968_pow_805_l3737_373798


namespace NUMINAMATH_CALUDE_base9_3671_equals_base10_2737_l3737_373795

def base9_to_base10 (n : Nat) : Nat :=
  (n / 1000) * (9^3) + ((n / 100) % 10) * (9^2) + ((n / 10) % 10) * 9 + (n % 10)

theorem base9_3671_equals_base10_2737 :
  base9_to_base10 3671 = 2737 := by
  sorry

end NUMINAMATH_CALUDE_base9_3671_equals_base10_2737_l3737_373795


namespace NUMINAMATH_CALUDE_slices_per_pizza_l3737_373796

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 17) 
  (h2 : total_slices = 68) : 
  total_slices / total_pizzas = 4 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_l3737_373796


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3737_373718

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3737_373718


namespace NUMINAMATH_CALUDE_half_coverage_days_l3737_373747

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 58

/-- Represents the growth factor of the lily pad patch per day -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days to cover half the lake is one less than the number of days to cover the full lake -/
theorem half_coverage_days : 
  ∃ (half_days : ℕ), half_days = full_coverage_days - 1 ∧ 
  (daily_growth_factor ^ half_days) * 2 = daily_growth_factor ^ full_coverage_days :=
sorry

end NUMINAMATH_CALUDE_half_coverage_days_l3737_373747


namespace NUMINAMATH_CALUDE_paint_remaining_l3737_373725

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 →
  let remaining_after_day1 := initial_paint - (3/8 * initial_paint)
  let remaining_after_day2 := remaining_after_day1 - (1/4 * remaining_after_day1)
  remaining_after_day2 = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l3737_373725


namespace NUMINAMATH_CALUDE_mijeong_box_volume_l3737_373701

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of Mijeong's cuboid box -/
theorem mijeong_box_volume :
  cuboid_volume 14 13 = 182 := by
  sorry

end NUMINAMATH_CALUDE_mijeong_box_volume_l3737_373701


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3737_373739

/-- Given a geometric sequence where the fourth term is 32 and the fifth term is 64, prove that the first term is 4. -/
theorem geometric_sequence_first_term (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ 32 = c * r ∧ 64 = 32 * r) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3737_373739


namespace NUMINAMATH_CALUDE_system_solution_l3737_373788

theorem system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^y = z) (eq2 : y^z = x) (eq3 : z^x = y) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3737_373788


namespace NUMINAMATH_CALUDE_speed_ratio_walking_l3737_373710

/-- Theorem: Ratio of speeds when two people walk towards each other and in the same direction -/
theorem speed_ratio_walking (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b > a) : ∃ (v₁ v₂ : ℝ),
  v₁ > 0 ∧ v₂ > 0 ∧ 
  (∃ (S : ℝ), S > 0 ∧ S = a * (v₁ + v₂) ∧ S = b * (v₁ - v₂)) ∧
  v₂ / v₁ = (a + b) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_walking_l3737_373710


namespace NUMINAMATH_CALUDE_product_abcd_l3737_373715

theorem product_abcd (a b c d : ℚ) : 
  (2 * a + 3 * b + 5 * c + 7 * d = 42) →
  (4 * (d + c) = b) →
  (2 * b + 2 * c = a) →
  (c - 2 = d) →
  (a * b * c * d = -26880 / 729) := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l3737_373715


namespace NUMINAMATH_CALUDE_fifth_valid_number_is_443_l3737_373728

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is valid (less than or equal to 600) --/
def isValidNumber (n : Nat) : Bool :=
  n ≤ 600

/-- Finds the nth valid number in a list --/
def findNthValidNumber (numbers : List Nat) (n : Nat) : Option Nat :=
  let validNumbers := numbers.filter isValidNumber
  validNumbers.get? (n - 1)

/-- The given random number table (partial) --/
def givenTable : RandomNumberTable :=
  [[84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76, 63, 1, 63],
   [78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 7, 44, 39, 52, 38, 79, 33, 21, 12, 34, 29, 78],
   [64, 56, 7, 82, 52, 42, 7, 44, 38, 15, 51, 0, 13, 42, 99, 66, 2, 79, 54]]

/-- The main theorem --/
theorem fifth_valid_number_is_443 :
  let numbers := (givenTable.get! 1).drop 7 ++ (givenTable.get! 2) ++ (givenTable.get! 3)
  findNthValidNumber numbers 5 = some 443 := by
  sorry

end NUMINAMATH_CALUDE_fifth_valid_number_is_443_l3737_373728


namespace NUMINAMATH_CALUDE_photographer_application_choices_l3737_373735

theorem photographer_application_choices :
  let n : ℕ := 5  -- Total number of pre-selected photos
  let k₁ : ℕ := 3 -- First option for number of photos to include
  let k₂ : ℕ := 4 -- Second option for number of photos to include
  (Nat.choose n k₁) + (Nat.choose n k₂) = 15 := by
  sorry

end NUMINAMATH_CALUDE_photographer_application_choices_l3737_373735


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3737_373752

theorem quadratic_equation_solution (x : ℝ) : 
  (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3737_373752


namespace NUMINAMATH_CALUDE_odd_function_a_value_l3737_373703

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem odd_function_a_value :
  ∃ a : ℝ, IsOdd (fun x ↦ lg ((2 / (1 - x)) + a)) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l3737_373703


namespace NUMINAMATH_CALUDE_carrots_picked_next_day_l3737_373731

theorem carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : 
  initial_carrots = 48 → thrown_out = 11 → total_carrots = 52 →
  total_carrots - (initial_carrots - thrown_out) = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrots_picked_next_day_l3737_373731


namespace NUMINAMATH_CALUDE_difference_proof_l3737_373773

/-- Given a total number of students and the number of first graders,
    calculate the difference between second graders and first graders. -/
def difference_between_grades (total : ℕ) (first_graders : ℕ) : ℕ :=
  (total - first_graders) - first_graders

theorem difference_proof :
  difference_between_grades 95 32 = 31 :=
by sorry

end NUMINAMATH_CALUDE_difference_proof_l3737_373773


namespace NUMINAMATH_CALUDE_first_question_percentage_l3737_373791

theorem first_question_percentage
  (second_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : second_correct = 0.3)
  (h2 : neither_correct = 0.2)
  (h3 : both_correct = 0.25) :
  ∃ (first_correct : Real),
    first_correct = 0.75 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l3737_373791


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3737_373734

/-- The interval between segments in systematic sampling --/
def systematic_sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: Given a population of 2000 and a sample size of 40, 
    the interval between segments in systematic sampling is 50 --/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 2000 40 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3737_373734


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3737_373768

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x - 1
  ∀ x : ℝ, f x = 0 ↔ x = -1/3 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3737_373768


namespace NUMINAMATH_CALUDE_clock_correction_theorem_l3737_373712

/-- The number of days between March 1st at noon and March 10th at 6 P.M. -/
def days_passed : ℚ := 9 + 6/24

/-- The rate at which the clock loses time, in minutes per day -/
def loss_rate : ℚ := 15

/-- The function to calculate the positive correction in minutes -/
def correction (d : ℚ) (r : ℚ) : ℚ := d * r

/-- Theorem stating that the positive correction needed is 138.75 minutes -/
theorem clock_correction_theorem :
  correction days_passed loss_rate = 138.75 := by sorry

end NUMINAMATH_CALUDE_clock_correction_theorem_l3737_373712


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l3737_373720

theorem lcm_ratio_sum (a b : ℕ+) : 
  Nat.lcm a b = 420 → a.val * 7 = b.val * 4 → a + b = 165 := by sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l3737_373720


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_l3737_373774

theorem mashed_potatoes_count : ∀ (bacon_count mashed_count : ℕ),
  bacon_count = 489 →
  bacon_count = mashed_count + 10 →
  mashed_count = 479 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_l3737_373774


namespace NUMINAMATH_CALUDE_three_odd_factors_is_nine_l3737_373750

theorem three_odd_factors_is_nine :
  ∃! n : ℕ, n > 1 ∧ (∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
    Odd a ∧ Odd b ∧ Odd c ∧
    {d : ℕ | d > 1 ∧ d ∣ n ∧ Odd d} = {a, b, c}) :=
by
  sorry

end NUMINAMATH_CALUDE_three_odd_factors_is_nine_l3737_373750


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l3737_373737

theorem reciprocal_of_negative_2022 : ((-2022)⁻¹ : ℚ) = -1 / 2022 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l3737_373737


namespace NUMINAMATH_CALUDE_ascending_six_digit_numbers_count_l3737_373772

/-- The number of six-digit natural numbers with digits in ascending order -/
def ascending_six_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem ascending_six_digit_numbers_count : ascending_six_digit_numbers = 84 := by
  sorry

end NUMINAMATH_CALUDE_ascending_six_digit_numbers_count_l3737_373772


namespace NUMINAMATH_CALUDE_exists_subset_with_common_gcd_l3737_373738

/-- A function that checks if a number is the product of at most 1987 prime factors -/
def is_valid_element (n : ℕ) : Prop := ∃ (factors : List ℕ), n = factors.prod ∧ factors.all Nat.Prime ∧ factors.length ≤ 1987

/-- The set A of integers, each being a product of at most 1987 prime factors -/
def A : Set ℕ := {n | is_valid_element n}

/-- The theorem to be proved -/
theorem exists_subset_with_common_gcd (h : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Infinite B ∧ B ⊆ A ∧ b > 0 ∧
  ∀ (x y : ℕ), x ∈ B → y ∈ B → Nat.gcd x y = b :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_common_gcd_l3737_373738


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3737_373781

/-- Given two vectors a and b, if (a + xb) is parallel to (a - b), then x = -1 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 4))
  (h2 : b = (2, 1))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + x * b.1, a.2 + x * b.2) = k • (a.1 - b.1, a.2 - b.2)) :
  x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3737_373781


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l3737_373794

/-- Calculates the number of female students in a stratified sample -/
def femaleInSample (totalPopulation malePopulation sampleSize : ℕ) : ℕ :=
  let femalePopulation := totalPopulation - malePopulation
  (femalePopulation * sampleSize) / totalPopulation

theorem stratified_sample_female_count :
  femaleInSample 900 500 45 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_count_l3737_373794


namespace NUMINAMATH_CALUDE_complementary_angles_difference_theorem_l3737_373757

def complementary_angles_difference (a b : ℝ) : Prop :=
  a + b = 90 ∧ a / b = 5 / 3 → |a - b| = 22.5

theorem complementary_angles_difference_theorem :
  ∀ a b : ℝ, complementary_angles_difference a b :=
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_theorem_l3737_373757


namespace NUMINAMATH_CALUDE_apple_sales_loss_percentage_l3737_373705

/-- Represents the shopkeeper's apple sales scenario -/
structure AppleSales where
  total_apples : ℝ
  sale_percentages : Fin 4 → ℝ
  profit_percentages : Fin 4 → ℝ
  unsold_percentage : ℝ
  storage_cost : ℝ
  packaging_cost : ℝ
  transportation_cost : ℝ

/-- Calculates the effective loss percentage for the given apple sales scenario -/
def effective_loss_percentage (sales : AppleSales) : ℝ :=
  sorry

/-- The given apple sales scenario -/
def given_scenario : AppleSales :=
  { total_apples := 150,
    sale_percentages := ![0.30, 0.25, 0.15, 0.10],
    profit_percentages := ![0.20, 0.30, 0.40, 0.35],
    unsold_percentage := 0.20,
    storage_cost := 15,
    packaging_cost := 10,
    transportation_cost := 25 }

/-- Theorem stating that the effective loss percentage for the given scenario is approximately 32.83% -/
theorem apple_sales_loss_percentage :
  abs (effective_loss_percentage given_scenario - 32.83) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_apple_sales_loss_percentage_l3737_373705


namespace NUMINAMATH_CALUDE_largest_number_l3737_373778

theorem largest_number (a b c d : ℝ) 
  (h : a + 5 = b^2 - 1 ∧ a + 5 = c^2 + 3 ∧ a + 5 = d - 4) : 
  d > a ∧ d > b ∧ d > c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3737_373778


namespace NUMINAMATH_CALUDE_number_equals_two_thirds_a_l3737_373792

/-- Given a and n are real numbers satisfying certain conditions, 
    prove that n equals 2a/3 -/
theorem number_equals_two_thirds_a (a n : ℝ) 
  (h1 : 2 * a = 3 * n) 
  (h2 : a * n ≠ 0) 
  (h3 : (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_two_thirds_a_l3737_373792


namespace NUMINAMATH_CALUDE_intersecting_line_equations_l3737_373775

/-- A line passing through a point and intersecting a circle --/
structure IntersectingLine where
  -- The point through which the line passes
  point : ℝ × ℝ
  -- The center of the circle
  center : ℝ × ℝ
  -- The radius of the circle
  radius : ℝ
  -- The length of the chord formed by the intersection
  chord_length : ℝ

/-- The equations of the line given the conditions --/
def line_equations (l : IntersectingLine) : Set (ℝ → ℝ → Prop) :=
  { (λ x y => x = -4),
    (λ x y => 4*x + 3*y + 25 = 0) }

/-- Theorem stating that the given conditions result in the specified line equations --/
theorem intersecting_line_equations 
  (l : IntersectingLine)
  (h1 : l.point = (-4, -3))
  (h2 : l.center = (-1, -2))
  (h3 : l.radius = 5)
  (h4 : l.chord_length = 8) :
  ∃ (eq : ℝ → ℝ → Prop), eq ∈ line_equations l ∧ 
    ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | eq p.1 p.2} → 
      ((x + 1)^2 + (y + 2)^2 = 25 ∨ (x, y) = l.point) :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_equations_l3737_373775


namespace NUMINAMATH_CALUDE_probability_at_least_one_male_l3737_373771

/-- The probability of selecting at least one male out of 3 contestants from a group of 8 finalists (5 females and 3 males) is 23/28. -/
theorem probability_at_least_one_male (total : ℕ) (females : ℕ) (males : ℕ) (selected : ℕ) :
  total = 8 →
  females = 5 →
  males = 3 →
  selected = 3 →
  (Nat.choose total selected - Nat.choose females selected : ℚ) / Nat.choose total selected = 23 / 28 := by
  sorry

#eval (Nat.choose 8 3 - Nat.choose 5 3 : ℚ) / Nat.choose 8 3 == 23 / 28

end NUMINAMATH_CALUDE_probability_at_least_one_male_l3737_373771


namespace NUMINAMATH_CALUDE_range_for_two_roots_roots_for_negative_integer_k_l3737_373736

/-- The quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ :=
  x^2 + (2*k + 1)*x + k^2 - 1

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  (2*k + 1)^2 - 4*(k^2 - 1)

/-- Theorem stating the range of k for which the equation has two distinct real roots -/
theorem range_for_two_roots :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) ↔ k > -5/4 :=
sorry

/-- Theorem stating the roots when k is a negative integer satisfying the range condition -/
theorem roots_for_negative_integer_k :
  ∀ k : ℤ, k < 0 → k > -5/4 → quadratic (↑k) 0 = 0 ∧ quadratic (↑k) 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_range_for_two_roots_roots_for_negative_integer_k_l3737_373736


namespace NUMINAMATH_CALUDE_rabbit_beaver_time_difference_rabbit_beaver_time_difference_holds_l3737_373764

/-- The time difference between a rabbit digging a hole and a beaver building a dam -/
theorem rabbit_beaver_time_difference : ℝ → Prop :=
  fun time_difference =>
    ∀ (rabbit_count rabbit_time hole_count : ℝ)
      (beaver_count beaver_time dam_count : ℝ),
    rabbit_count > 0 →
    rabbit_time > 0 →
    hole_count > 0 →
    beaver_count > 0 →
    beaver_time > 0 →
    dam_count > 0 →
    rabbit_count * rabbit_time * 60 / hole_count = 100 →
    beaver_count * beaver_time / dam_count = 90 →
    rabbit_count = 3 →
    rabbit_time = 5 →
    hole_count = 9 →
    beaver_count = 5 →
    beaver_time = 36 / 60 →
    dam_count = 2 →
    time_difference = 10

theorem rabbit_beaver_time_difference_holds : rabbit_beaver_time_difference 10 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_beaver_time_difference_rabbit_beaver_time_difference_holds_l3737_373764


namespace NUMINAMATH_CALUDE_rabbit_population_estimate_l3737_373716

/-- Calculates the approximate number of rabbits in a forest using the capture-recapture method. -/
def estimate_rabbit_population (initial_tagged : ℕ) (recaptured : ℕ) (tagged_in_recapture : ℕ) : ℕ :=
  (initial_tagged * recaptured) / tagged_in_recapture

/-- The approximate number of rabbits in the forest is 50. -/
theorem rabbit_population_estimate :
  let initial_tagged : ℕ := 10
  let recaptured : ℕ := 10
  let tagged_in_recapture : ℕ := 2
  estimate_rabbit_population initial_tagged recaptured tagged_in_recapture = 50 := by
  sorry

#eval estimate_rabbit_population 10 10 2

end NUMINAMATH_CALUDE_rabbit_population_estimate_l3737_373716


namespace NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l3737_373769

/-- A line in a 2D coordinate system --/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Calculate the x-coordinate for a given y-coordinate on a line --/
def xCoordAtY (line : Line) (y : ℚ) : ℚ :=
  (y - line.intercept) / line.slope

/-- Create a line from two points --/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  intercept := y1 - (y2 - y1) / (x2 - x1) * x1

theorem x_coordinate_difference_at_y_20 :
  let l := lineFromPoints 0 5 3 0
  let m := lineFromPoints 0 4 6 0
  let x_l := xCoordAtY l 20
  let x_m := xCoordAtY m 20
  |x_l - x_m| = 15 := by sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l3737_373769


namespace NUMINAMATH_CALUDE_sentences_at_start_l3737_373760

-- Define the typing rate
def typing_rate : ℕ := 6

-- Define the typing durations
def first_session : ℕ := 20
def second_session : ℕ := 15
def third_session : ℕ := 18

-- Define the number of erased sentences
def erased_sentences : ℕ := 40

-- Define the total number of sentences at the end of the day
def total_sentences : ℕ := 536

-- Theorem to prove
theorem sentences_at_start : 
  total_sentences - (typing_rate * (first_session + second_session + third_session) - erased_sentences) = 258 :=
by sorry

end NUMINAMATH_CALUDE_sentences_at_start_l3737_373760


namespace NUMINAMATH_CALUDE_three_diamonds_balance_two_circles_l3737_373766

/-- Represents the balance of symbols in the problem -/
structure Balance where
  triangle : ℕ  -- Δ
  diamond : ℕ   -- ◊
  circle : ℕ    -- •

/-- First balance equation: 4Δ + 2◊ = 12• -/
def balance_equation1 (b : Balance) : Prop :=
  4 * b.triangle + 2 * b.diamond = 12 * b.circle

/-- Second balance equation: Δ = ◊ + 2• -/
def balance_equation2 (b : Balance) : Prop :=
  b.triangle = b.diamond + 2 * b.circle

/-- Theorem stating that 3◊ balances 2• -/
theorem three_diamonds_balance_two_circles (b : Balance) 
  (h1 : balance_equation1 b) (h2 : balance_equation2 b) : 
  3 * b.diamond = 2 * b.circle :=
sorry

end NUMINAMATH_CALUDE_three_diamonds_balance_two_circles_l3737_373766


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3737_373724

theorem max_value_trig_expression :
  ∀ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) * 
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 
  (9 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3737_373724


namespace NUMINAMATH_CALUDE_snowboard_final_price_l3737_373729

/-- Calculates the final price of an item after applying two discounts and a sales tax. -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  priceAfterDiscount2 * (1 + salesTax)

/-- Theorem stating that the final price of a $200 snowboard after 40% and 20% discounts
    and 5% sales tax is $100.80. -/
theorem snowboard_final_price :
  finalPrice 200 0.4 0.2 0.05 = 100.80 := by
  sorry

end NUMINAMATH_CALUDE_snowboard_final_price_l3737_373729


namespace NUMINAMATH_CALUDE_eight_to_one_l3737_373790

theorem eight_to_one : (8/8)^(8/8) * 8/8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eight_to_one_l3737_373790


namespace NUMINAMATH_CALUDE_odd_function_has_zero_l3737_373784

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_has_zero (f : ℝ → ℝ) (h : OddFunction f) : 
  ∃ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_has_zero_l3737_373784


namespace NUMINAMATH_CALUDE_candy_division_l3737_373744

theorem candy_division (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 42 →
  num_bags = 2 →
  candy_per_bag * num_bags = total_candy →
  candy_per_bag = 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l3737_373744


namespace NUMINAMATH_CALUDE_min_m_intersection_nonempty_l3737_373753

def set_B (m : ℝ) : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 - m = 0}

theorem min_m_intersection_nonempty (A : Set (ℝ × ℝ)) (h : ∃ m : ℝ, (A ∩ set_B m).Nonempty) :
  ∃ m_min : ℝ, m_min = 0 ∧ (A ∩ set_B m_min).Nonempty ∧ ∀ m : ℝ, (A ∩ set_B m).Nonempty → m ≥ m_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_m_intersection_nonempty_l3737_373753


namespace NUMINAMATH_CALUDE_bill_and_caroline_ages_l3737_373748

/-- Given that Bill is 17 years old and 1 year less than twice as old as his sister Caroline,
    prove that the sum of their ages is 26. -/
theorem bill_and_caroline_ages : ∀ (caroline_age : ℕ),
  17 = 2 * caroline_age - 1 →
  17 + caroline_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_bill_and_caroline_ages_l3737_373748


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3737_373756

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 - 2 * Complex.I^3) / (1 + Complex.I)
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3737_373756


namespace NUMINAMATH_CALUDE_cosine_sum_zero_l3737_373749

theorem cosine_sum_zero (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos (y + 2 * Real.pi / 3) + Real.cos (z + 4 * Real.pi / 3) = 0)
  (h2 : Real.sin x + Real.sin (y + 2 * Real.pi / 3) + Real.sin (z + 4 * Real.pi / 3) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_zero_l3737_373749


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l3737_373732

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l3737_373732


namespace NUMINAMATH_CALUDE_min_value_expression_l3737_373708

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3737_373708


namespace NUMINAMATH_CALUDE_part_one_solution_set_part_two_m_value_l3737_373759

-- Define the function f
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

-- Part I
theorem part_one_solution_set :
  {x : ℝ | f x (-1) (-1) ≥ x} = {x : ℝ | x ≤ -2 ∨ 0 ≤ x ∧ x ≤ 2} :=
sorry

-- Part II
theorem part_two_m_value :
  ∀ (a m : ℝ), 0 < m → m < 1 → (∀ x, f x a m ≥ 2) → (a ≤ -3 ∨ a ≥ 3) → m = 1/3 :=
sorry

end NUMINAMATH_CALUDE_part_one_solution_set_part_two_m_value_l3737_373759


namespace NUMINAMATH_CALUDE_mapping_result_l3737_373704

-- Define the set A (and B) as pairs of real numbers
def A : Type := ℝ × ℝ

-- Define the mapping f
def f (p : A) : A :=
  let (x, y) := p
  (x - y, x + y)

-- Theorem statement
theorem mapping_result : f (-1, 2) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_mapping_result_l3737_373704


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_three_l3737_373730

def A : Set ℝ := {x | 3 + 2*x - x^2 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem intersection_nonempty_implies_a_less_than_three (a : ℝ) :
  (A ∩ B a).Nonempty → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_three_l3737_373730


namespace NUMINAMATH_CALUDE_unique_base_conversion_l3737_373762

def base_conversion (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem unique_base_conversion : ∃! x : Nat,
  x < 1000 ∧
  x ≥ 100 ∧
  let digits := [x / 100, (x / 10) % 10, x % 10]
  base_conversion digits 20 = 2 * base_conversion digits 13 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l3737_373762


namespace NUMINAMATH_CALUDE_interest_rate_beyond_five_years_l3737_373741

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_beyond_five_years 
  (principal : ℝ)
  (rate_first_two_years : ℝ)
  (rate_next_three_years : ℝ)
  (total_interest : ℝ)
  (h1 : principal = 12000)
  (h2 : rate_first_two_years = 0.06)
  (h3 : rate_next_three_years = 0.09)
  (h4 : total_interest = 11400)
  : ∃ (rate_beyond_five_years : ℝ),
    rate_beyond_five_years = 0.14 ∧
    total_interest = 
      simple_interest principal rate_first_two_years 2 +
      simple_interest principal rate_next_three_years 3 +
      simple_interest principal rate_beyond_five_years 4 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_beyond_five_years_l3737_373741


namespace NUMINAMATH_CALUDE_group_collection_l3737_373717

/-- Calculates the total collection in rupees for a group contribution -/
def total_collection (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Theorem stating that for a group of 93 members, the total collection is 86.49 rupees -/
theorem group_collection :
  total_collection 93 = 86.49 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_l3737_373717


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l3737_373714

/-- 
Given a natural number n, prove that if the coefficients of the first three terms 
in the expansion of (x/2 + 1)^n form an arithmetic sequence, then n = 8.
-/
theorem binomial_expansion_arithmetic_sequence (n : ℕ) : 
  (∃ d : ℚ, 1 = (n.choose 0) ∧ 
             (n.choose 1) / 2 = 1 + d ∧ 
             (n.choose 2) / 4 = 1 + 2*d) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l3737_373714


namespace NUMINAMATH_CALUDE_pet_store_problem_l3737_373709

def puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : ℕ :=
  initial_puppies - (puppies_per_cage * cages_used)

theorem pet_store_problem :
  puppies_sold 45 2 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_problem_l3737_373709


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l3737_373780

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The GDP value of Anning City in the first quarter of 2023 -/
def gdp_value : ℕ := 17580000000

/-- The scientific notation representation of the GDP value -/
def gdp_scientific : ScientificNotation :=
  { coefficient := 1.758
    exponent := 10
    is_valid := by sorry }

/-- Theorem stating that the GDP value is correctly represented in scientific notation -/
theorem gdp_scientific_notation_correct :
  (gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent) = gdp_value := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l3737_373780


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3737_373721

/-- Given two concentric circles where a 60-degree arc on the smaller circle has the same length as a 48-degree arc on the larger circle, the ratio of the area of the smaller circle to the area of the larger circle is 16/25. -/
theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * π * r₁) = 48 / 360 * (2 * π * r₂)) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3737_373721


namespace NUMINAMATH_CALUDE_nadia_hannah_walk_l3737_373779

/-- The total distance walked by Nadia and Hannah -/
def total_distance (nadia_distance : ℝ) (hannah_distance : ℝ) : ℝ :=
  nadia_distance + hannah_distance

/-- Theorem: Given Nadia walked 18 km and twice as far as Hannah, their total distance is 27 km -/
theorem nadia_hannah_walk :
  let nadia_distance : ℝ := 18
  let hannah_distance : ℝ := nadia_distance / 2
  total_distance nadia_distance hannah_distance = 27 := by
sorry

end NUMINAMATH_CALUDE_nadia_hannah_walk_l3737_373779


namespace NUMINAMATH_CALUDE_females_attending_correct_l3737_373742

/-- The number of females attending the meeting -/
def females_attending : ℕ := 50

/-- The total population of Nantucket -/
def total_population : ℕ := 300

/-- The number of people attending the meeting -/
def meeting_attendance : ℕ := total_population / 2

/-- The number of males attending the meeting -/
def males_attending : ℕ := 2 * females_attending

theorem females_attending_correct :
  females_attending = 50 ∧
  meeting_attendance = total_population / 2 ∧
  total_population = 300 ∧
  males_attending = 2 * females_attending ∧
  meeting_attendance = females_attending + males_attending :=
by sorry

end NUMINAMATH_CALUDE_females_attending_correct_l3737_373742


namespace NUMINAMATH_CALUDE_election_winner_margin_l3737_373727

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) :
  (winner_votes : ℝ) = 0.56 * total_votes →
  winner_votes = 1344 →
  winner_votes - (total_votes - winner_votes) = 288 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_margin_l3737_373727


namespace NUMINAMATH_CALUDE_initial_charge_is_3_5_l3737_373726

/-- A taxi company's pricing model -/
structure TaxiCompany where
  initialCharge : ℝ  -- Initial charge for the first 1/5 mile
  additionalCharge : ℝ  -- Charge for each additional 1/5 mile
  totalCharge : ℝ  -- Total charge for a specific ride
  rideLength : ℝ  -- Length of the ride in miles

/-- The initial charge for the first 1/5 mile is $3.5 -/
theorem initial_charge_is_3_5 (t : TaxiCompany) 
    (h1 : t.additionalCharge = 0.4)
    (h2 : t.totalCharge = 19.1)
    (h3 : t.rideLength = 8) : 
    t.initialCharge = 3.5 := by
  sorry

#check initial_charge_is_3_5

end NUMINAMATH_CALUDE_initial_charge_is_3_5_l3737_373726


namespace NUMINAMATH_CALUDE_correct_bill_writing_l3737_373789

/-- Represents the monthly electricity bill in yuan -/
def monthly_bill : ℚ := 71.08

/-- The correct way to write the monthly electricity bill -/
def correct_writing : String := "71.08"

/-- Theorem stating that the correct way to write the monthly electricity bill is "71.08" -/
theorem correct_bill_writing : 
  toString monthly_bill = correct_writing := by sorry

end NUMINAMATH_CALUDE_correct_bill_writing_l3737_373789


namespace NUMINAMATH_CALUDE_monthly_expenses_calculation_l3737_373763

-- Define the monthly deposit
def monthly_deposit : ℕ := 5000

-- Define the annual savings
def annual_savings : ℕ := 4800

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem to prove
theorem monthly_expenses_calculation :
  (monthly_deposit * months_in_year - annual_savings) / months_in_year = 4600 :=
by sorry

end NUMINAMATH_CALUDE_monthly_expenses_calculation_l3737_373763


namespace NUMINAMATH_CALUDE_perfect_square_2n_plus_256_l3737_373758

theorem perfect_square_2n_plus_256 (n : ℕ) :
  (∃ (k : ℕ+), 2^n + 256 = k^2) → n = 11 := by sorry

end NUMINAMATH_CALUDE_perfect_square_2n_plus_256_l3737_373758


namespace NUMINAMATH_CALUDE_william_land_percentage_l3737_373793

-- Define the tax amounts
def total_village_tax : ℝ := 3840
def william_tax : ℝ := 480

-- Define the theorem
theorem william_land_percentage :
  william_tax / total_village_tax * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_william_land_percentage_l3737_373793


namespace NUMINAMATH_CALUDE_hyperbola_point_outside_circle_l3737_373723

theorem hyperbola_point_outside_circle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (focus : c > 0)
  (eccentricity : c / a = 2)
  (x₁ x₂ : ℝ)
  (roots : a * x₁^2 + b * x₁ - c = 0 ∧ a * x₂^2 + b * x₂ - c = 0) :
  x₁^2 + x₂^2 > 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_outside_circle_l3737_373723


namespace NUMINAMATH_CALUDE_elijah_card_count_l3737_373755

/-- The number of cards in a standard deck of playing cards -/
def cards_per_deck : ℕ := 52

/-- The number of decks Elijah has -/
def number_of_decks : ℕ := 6

/-- The total number of cards Elijah has -/
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem elijah_card_count : total_cards = 312 := by
  sorry

end NUMINAMATH_CALUDE_elijah_card_count_l3737_373755


namespace NUMINAMATH_CALUDE_pauls_vertical_distance_l3737_373777

/-- The total vertical distance traveled by Paul in a week -/
def total_vertical_distance (floor : ℕ) (trips_per_day : ℕ) (days : ℕ) (story_height : ℕ) : ℕ :=
  floor * story_height * trips_per_day * 2 * days

/-- Theorem stating the total vertical distance Paul travels in a week -/
theorem pauls_vertical_distance :
  total_vertical_distance 5 3 7 10 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_pauls_vertical_distance_l3737_373777


namespace NUMINAMATH_CALUDE_steves_commute_l3737_373754

/-- The distance from Steve's house to work -/
def distance : ℝ := by sorry

/-- Steve's speed on the way to work -/
def speed_to_work : ℝ := by sorry

/-- Steve's speed on the way back from work -/
def speed_from_work : ℝ := 14

/-- The total time Steve spends on the roads -/
def total_time : ℝ := 6

theorem steves_commute :
  (speed_from_work = 2 * speed_to_work) →
  (distance / speed_to_work + distance / speed_from_work = total_time) →
  distance = 28 := by sorry

end NUMINAMATH_CALUDE_steves_commute_l3737_373754


namespace NUMINAMATH_CALUDE_sin_sum_equality_l3737_373702

theorem sin_sum_equality : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l3737_373702
