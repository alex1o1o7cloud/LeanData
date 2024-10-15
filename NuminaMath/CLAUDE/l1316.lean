import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_uniqueness_l1316_131665

/-- A linear function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f x < f y -/
def IsIncreasingLinear (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ (∀ x y, x < y → f x < f y)

/-- The main theorem -/
theorem linear_function_uniqueness (f : ℝ → ℝ) 
  (h_increasing : IsIncreasingLinear f)
  (h_composition : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_uniqueness_l1316_131665


namespace NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l1316_131604

theorem cos_alpha_plus_20_eq_neg_alpha (α : ℝ) (h : Real.sin (α - 70 * Real.pi / 180) = α) :
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l1316_131604


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1316_131674

theorem least_positive_integer_with_given_remainders : ∃! N : ℕ,
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (∀ M : ℕ, M < N →
    ¬((M % 11 = 10) ∧
      (M % 12 = 11) ∧
      (M % 13 = 12) ∧
      (M % 14 = 13))) ∧
  N = 12011 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1316_131674


namespace NUMINAMATH_CALUDE_peter_class_size_l1316_131600

/-- The number of students in Peter's class -/
def students_in_class : ℕ := 11

/-- The number of hands in the class, excluding Peter's -/
def hands_without_peter : ℕ := 20

/-- The number of hands each student has -/
def hands_per_student : ℕ := 2

/-- Theorem: The number of students in Peter's class is 11 -/
theorem peter_class_size :
  students_in_class = hands_without_peter / hands_per_student + 1 :=
by sorry

end NUMINAMATH_CALUDE_peter_class_size_l1316_131600


namespace NUMINAMATH_CALUDE_current_monthly_production_l1316_131650

/-- Represents the car manufacturing company's production data -/
structure CarProduction where
  targetAnnual : ℕ
  monthlyIncrease : ℕ
  currentMonthly : ℕ

/-- Theorem stating that the current monthly production is 100 cars -/
theorem current_monthly_production (cp : CarProduction) 
  (h1 : cp.targetAnnual = 1800)
  (h2 : cp.monthlyIncrease = 50)
  (h3 : cp.currentMonthly * 12 + cp.monthlyIncrease * 12 = cp.targetAnnual) :
  cp.currentMonthly = 100 := by
  sorry

#check current_monthly_production

end NUMINAMATH_CALUDE_current_monthly_production_l1316_131650


namespace NUMINAMATH_CALUDE_kitten_growth_l1316_131610

/-- The length of a kitten after doubling twice -/
def kittenLength (initialLength : ℕ) (doublings : ℕ) : ℕ :=
  initialLength * (2 ^ doublings)

/-- Theorem: A kitten with initial length 4 inches that doubles twice will be 16 inches long -/
theorem kitten_growth : kittenLength 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l1316_131610


namespace NUMINAMATH_CALUDE_line_parameterization_l1316_131605

/-- Given a line y = 5x - 7 parameterized by [x, y] = [p, 3] + t[3, q], 
    prove that p = 2 and q = 15 -/
theorem line_parameterization (x y p q t : ℝ) : 
  (y = 5*x - 7) ∧ 
  (∃ t, x = p + 3*t ∧ y = 3 + q*t) →
  p = 2 ∧ q = 15 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1316_131605


namespace NUMINAMATH_CALUDE_probability_penny_dime_heads_l1316_131673

-- Define the coin flip experiment
def coin_flip_experiment : ℕ := 5

-- Define the number of coins we're interested in (penny and dime)
def target_coins : ℕ := 2

-- Define the probability of a single coin coming up heads
def prob_heads : ℚ := 1/2

-- Theorem statement
theorem probability_penny_dime_heads :
  (prob_heads ^ target_coins) * (2 ^ (coin_flip_experiment - target_coins)) / (2 ^ coin_flip_experiment) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_probability_penny_dime_heads_l1316_131673


namespace NUMINAMATH_CALUDE_line_through_three_points_l1316_131689

/-- Given a line containing points (0, 5), (7, k), and (25, 2), prove that k = 104/25 -/
theorem line_through_three_points (k : ℝ) : 
  (∃ m b : ℝ, (0 = m * 0 + b ∧ 5 = m * 0 + b) ∧ 
              (k = m * 7 + b) ∧ 
              (2 = m * 25 + b)) → 
  k = 104 / 25 := by
sorry

end NUMINAMATH_CALUDE_line_through_three_points_l1316_131689


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1316_131648

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ, 
  x > 0 ∧ 
  x % 5 = 4 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ y : ℕ, y > 0 ∧ y % 5 = 4 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1316_131648


namespace NUMINAMATH_CALUDE_fun_run_no_shows_fun_run_no_shows_solution_l1316_131687

/-- Fun Run Attendance Problem -/
theorem fun_run_no_shows (signed_up_last_year : ℕ) (runners_this_year : ℕ) : ℕ :=
  let runners_last_year := runners_this_year / 2
  signed_up_last_year - runners_last_year

/-- The number of people who did not show up to run last year is 40 -/
theorem fun_run_no_shows_solution : fun_run_no_shows 200 320 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fun_run_no_shows_fun_run_no_shows_solution_l1316_131687


namespace NUMINAMATH_CALUDE_cone_volume_l1316_131641

def slant_height : ℝ := 5
def base_radius : ℝ := 3

theorem cone_volume : 
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * height = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1316_131641


namespace NUMINAMATH_CALUDE_zane_bought_two_shirts_l1316_131684

/-- Calculates the number of polo shirts bought given the discount percentage, regular price, and total amount paid. -/
def polo_shirts_bought (discount_percent : ℚ) (regular_price : ℚ) (total_paid : ℚ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  total_paid / discounted_price

/-- Proves that Zane bought 2 polo shirts given the specified conditions. -/
theorem zane_bought_two_shirts : 
  polo_shirts_bought (40/100) 50 60 = 2 := by
  sorry

#eval polo_shirts_bought (40/100) 50 60

end NUMINAMATH_CALUDE_zane_bought_two_shirts_l1316_131684


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1316_131688

-- Part (1)
theorem inequality_one (x : ℝ) :
  (2 < |2*x - 5| ∧ |2*x - 5| ≤ 7) ↔ ((-1 ≤ x ∧ x < 3/2) ∨ (7/2 < x ∧ x ≤ 6)) :=
sorry

-- Part (2)
theorem inequality_two (x : ℝ) :
  (1 / (x - 1) > x + 1) ↔ (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1316_131688


namespace NUMINAMATH_CALUDE_cube_with_hole_volume_is_384_l1316_131630

/-- The volume of a cube with a square hole cut through its center. -/
def cube_with_hole_volume (cube_side : ℝ) (hole_side : ℝ) : ℝ :=
  cube_side ^ 3 - hole_side ^ 2 * cube_side

/-- Theorem stating that a cube with side length 8 cm and a square hole
    with side length 4 cm cut through its center has a volume of 384 cm³. -/
theorem cube_with_hole_volume_is_384 :
  cube_with_hole_volume 8 4 = 384 := by
  sorry

#eval cube_with_hole_volume 8 4

end NUMINAMATH_CALUDE_cube_with_hole_volume_is_384_l1316_131630


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1316_131679

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) = 2) : 
  Complex.im z = -4/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1316_131679


namespace NUMINAMATH_CALUDE_dan_placed_13_scissors_l1316_131628

/-- The number of scissors Dan placed in the drawer -/
def scissors_placed (initial_count final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Proof that Dan placed 13 scissors in the drawer -/
theorem dan_placed_13_scissors (initial_count final_count : ℕ) 
  (h1 : initial_count = 39)
  (h2 : final_count = 52) : 
  scissors_placed initial_count final_count = 13 := by
  sorry

#eval scissors_placed 39 52  -- Should output 13

end NUMINAMATH_CALUDE_dan_placed_13_scissors_l1316_131628


namespace NUMINAMATH_CALUDE_probability_of_green_ball_l1316_131655

theorem probability_of_green_ball (total_balls green_balls : ℕ) 
  (h1 : total_balls = 10)
  (h2 : green_balls = 4) : 
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_green_ball_l1316_131655


namespace NUMINAMATH_CALUDE_range_of_a_l1316_131671

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 2^x - a = 1/(x-1)) → 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1316_131671


namespace NUMINAMATH_CALUDE_age_system_properties_l1316_131657

/-- Represents the ages and aging rates of four people -/
structure AgeSystem where
  a : ℝ  -- Age of person A
  b : ℝ  -- Age of person B
  c : ℝ  -- Age of person C
  d : ℝ  -- Age of person D
  x : ℝ  -- Age difference between A and C
  y : ℝ  -- Number of years passed
  rA : ℝ  -- Aging rate of A relative to C
  rB : ℝ  -- Aging rate of B relative to C
  rD : ℝ  -- Aging rate of D relative to C

/-- The age system satisfies the given conditions -/
def satisfiesConditions (s : AgeSystem) : Prop :=
  s.a + s.b = 13 + (s.b + s.c) ∧
  s.c = s.a - s.x ∧
  s.a + s.d = 2 * (s.b + s.c) ∧
  (s.a + s.rA * s.y) + (s.b + s.rB * s.y) = 25 + (s.b + s.rB * s.y) + (s.c + s.y)

/-- Theorem stating the properties of the age system -/
theorem age_system_properties (s : AgeSystem) 
  (h : satisfiesConditions s) : 
  s.x = 13 ∧ s.d = 2 * s.b + s.a - 26 ∧ s.rA * s.y = 12 + s.y := by
  sorry


end NUMINAMATH_CALUDE_age_system_properties_l1316_131657


namespace NUMINAMATH_CALUDE_union_M_N_l1316_131622

def M : Set ℕ := {1, 2}

def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l1316_131622


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l1316_131647

theorem smallest_divisible_number (n : ℕ) : 
  n = 719 + 288721 → 
  (∀ m : ℕ, 719 < m ∧ m < n → ¬(618 ∣ m ∧ 3648 ∣ m ∧ 60 ∣ m)) ∧ 
  (618 ∣ n ∧ 3648 ∣ n ∧ 60 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l1316_131647


namespace NUMINAMATH_CALUDE_last_k_digits_power_l1316_131633

theorem last_k_digits_power (k n : ℕ) (A B : ℤ) 
  (h : A ≡ B [ZMOD 10^k]) : 
  A^n ≡ B^n [ZMOD 10^k] := by sorry

end NUMINAMATH_CALUDE_last_k_digits_power_l1316_131633


namespace NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l1316_131646

/-- Represents the number of ways N people can shake hands with exactly two others each -/
def handshake_arrangements (N : ℕ) : ℕ :=
  sorry

/-- The number of ways 9 people can shake hands with exactly two others each -/
def N : ℕ := handshake_arrangements 9

/-- Theorem stating that the number of handshake arrangements for 9 people is congruent to 152 modulo 1000 -/
theorem handshake_arrangements_mod_1000 : N ≡ 152 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l1316_131646


namespace NUMINAMATH_CALUDE_largest_prime_satisfying_inequality_l1316_131664

theorem largest_prime_satisfying_inequality :
  ∃ (m : ℕ), m.Prime ∧ m^2 - 11*m + 28 < 0 ∧
  ∀ (n : ℕ), n.Prime → n^2 - 11*n + 28 < 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_satisfying_inequality_l1316_131664


namespace NUMINAMATH_CALUDE_probability_rgb_draw_specific_l1316_131609

/-- The probability of drawing a red shoe first, a green shoe second, and a blue shoe third
    from a closet containing red, green, and blue shoes. -/
def probability_rgb_draw (red green blue : ℕ) : ℚ :=
  (red : ℚ) / (red + green + blue) *
  (green : ℚ) / (red + green + blue - 1) *
  (blue : ℚ) / (red + green + blue - 2)

/-- Theorem stating that the probability of drawing a red shoe first, a green shoe second,
    and a blue shoe third from a closet containing 5 red shoes, 4 green shoes, and 3 blue shoes
    is equal to 1/22. -/
theorem probability_rgb_draw_specific : probability_rgb_draw 5 4 3 = 1 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_rgb_draw_specific_l1316_131609


namespace NUMINAMATH_CALUDE_women_count_at_gathering_l1316_131616

/-- Represents a social gathering where men and women dance. -/
structure SocialGathering where
  men : ℕ
  women : ℕ
  manDances : ℕ
  womanDances : ℕ

/-- The number of women at the gathering is correct if the total number of dances
    from men's perspective equals the total number of dances from women's perspective. -/
def isCorrectWomenCount (g : SocialGathering) : Prop :=
  g.men * g.manDances = g.women * g.womanDances

/-- Theorem stating that in a gathering with 15 men, where each man dances with 4 women
    and each woman dances with 3 men, there are 20 women. -/
theorem women_count_at_gathering :
  ∀ g : SocialGathering,
    g.men = 15 →
    g.manDances = 4 →
    g.womanDances = 3 →
    isCorrectWomenCount g →
    g.women = 20 := by
  sorry

end NUMINAMATH_CALUDE_women_count_at_gathering_l1316_131616


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1316_131601

/-- An arithmetic sequence with the given properties has the general term a_n = n -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : 
  d ≠ 0 ∧  -- non-zero common difference
  (∀ n, a (n + 1) = a n + d) ∧  -- arithmetic sequence property
  a 1 = 1 ∧  -- first term is 1
  (a 3)^2 = a 1 * a 9  -- geometric sequence property for a_1, a_3, a_9
  →
  ∀ n, a n = n := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1316_131601


namespace NUMINAMATH_CALUDE_tank_length_proof_l1316_131658

/-- Proves that a tank with given dimensions and plastering cost has a specific length -/
theorem tank_length_proof (width depth L : ℝ) (plastering_rate : ℝ) (total_cost : ℝ) : 
  width = 12 →
  depth = 6 →
  plastering_rate = 75 / 100 →
  total_cost = 558 →
  (2 * depth * L + 2 * depth * width + width * L) * plastering_rate = total_cost →
  L = 25 := by
sorry


end NUMINAMATH_CALUDE_tank_length_proof_l1316_131658


namespace NUMINAMATH_CALUDE_sin_cos_sum_10_20_l1316_131614

theorem sin_cos_sum_10_20 : 
  Real.sin (10 * π / 180) * Real.cos (20 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_10_20_l1316_131614


namespace NUMINAMATH_CALUDE_vasyas_capital_decreases_l1316_131640

/-- Represents the change in Vasya's capital after a series of trading days -/
def vasyas_capital_change (num_unsuccessful_days : ℕ) : ℝ :=
  (1.1^2 * 0.8)^num_unsuccessful_days

/-- Theorem stating that Vasya's capital decreases -/
theorem vasyas_capital_decreases (num_unsuccessful_days : ℕ) :
  vasyas_capital_change num_unsuccessful_days < 1 := by
  sorry

#check vasyas_capital_decreases

end NUMINAMATH_CALUDE_vasyas_capital_decreases_l1316_131640


namespace NUMINAMATH_CALUDE_jerry_lawsuit_compensation_l1316_131698

def annual_salary : ℕ := 50000
def years_lost : ℕ := 30
def medical_bills : ℕ := 200000
def punitive_multiplier : ℕ := 3
def award_percentage : ℚ := 80 / 100

theorem jerry_lawsuit_compensation :
  let total_salary := annual_salary * years_lost
  let direct_damages := total_salary + medical_bills
  let punitive_damages := direct_damages * punitive_multiplier
  let total_asked := direct_damages + punitive_damages
  let awarded_amount := (total_asked : ℚ) * award_percentage
  awarded_amount = 5440000 := by sorry

end NUMINAMATH_CALUDE_jerry_lawsuit_compensation_l1316_131698


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l1316_131672

theorem sum_of_five_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 5 * n + 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l1316_131672


namespace NUMINAMATH_CALUDE_range_of_a_l1316_131685

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 2 3, x^2 + 5 > a*x) = False → a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1316_131685


namespace NUMINAMATH_CALUDE_box_volume_increase_l1316_131683

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4320)
  (surface_area : 2 * (l * w + w * h + l * h) = 1704)
  (edge_sum : 4 * (l + w + h) = 208) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1316_131683


namespace NUMINAMATH_CALUDE_weaker_correlation_as_r_approaches_zero_l1316_131660

/-- The correlation coefficient type -/
def CorrelationCoefficient := { r : ℝ // -1 < r ∧ r < 1 }

/-- A measure of correlation strength -/
def correlationStrength (r : CorrelationCoefficient) : ℝ := |r.val|

/-- Theorem: As the absolute value of the correlation coefficient approaches 0, 
    the correlation between two variables becomes weaker -/
theorem weaker_correlation_as_r_approaches_zero 
  (r : CorrelationCoefficient) : 
  ∀ ε > 0, ∃ δ > 0, ∀ r' : CorrelationCoefficient, 
    correlationStrength r' < δ → correlationStrength r' < ε :=
sorry

end NUMINAMATH_CALUDE_weaker_correlation_as_r_approaches_zero_l1316_131660


namespace NUMINAMATH_CALUDE_retirement_ratio_is_one_to_one_l1316_131603

def monthly_income : ℚ := 2500
def rent : ℚ := 700
def car_payment : ℚ := 300
def groceries : ℚ := 50
def remaining_after_retirement : ℚ := 650

def total_expenses : ℚ := rent + car_payment + (car_payment / 2) + groceries

def money_after_expenses : ℚ := monthly_income - total_expenses

def retirement_contribution : ℚ := money_after_expenses - remaining_after_retirement

theorem retirement_ratio_is_one_to_one :
  retirement_contribution = remaining_after_retirement :=
by sorry

end NUMINAMATH_CALUDE_retirement_ratio_is_one_to_one_l1316_131603


namespace NUMINAMATH_CALUDE_number_problem_l1316_131668

theorem number_problem (x : ℝ) : 35 - 3 * x = 8 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1316_131668


namespace NUMINAMATH_CALUDE_adam_first_year_students_l1316_131693

/-- The number of students Adam teaches per year after the first year -/
def students_per_year : ℕ := 50

/-- The total number of years Adam teaches -/
def total_years : ℕ := 10

/-- The total number of students Adam teaches in 10 years -/
def total_students : ℕ := 490

/-- The number of students Adam taught in the first year -/
def first_year_students : ℕ := total_students - (students_per_year * (total_years - 1))

theorem adam_first_year_students :
  first_year_students = 40 := by sorry

end NUMINAMATH_CALUDE_adam_first_year_students_l1316_131693


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1316_131656

theorem inequality_solution_set : 
  {x : ℝ | 2 * (x^2 - x) < 4} = {x : ℝ | -1 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1316_131656


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l1316_131613

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ :=
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice * prob_one * prob_not_one ^ (num_dice - 1)) +
  2 * (num_dice * (num_dice - 1) / 2 * prob_one ^ 2 * prob_not_one) +
  3 * (prob_one ^ num_dice)

theorem expected_ones_is_half : expected_ones = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l1316_131613


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_base_6_l1316_131638

def base_6_to_10 (n : ℕ) : ℕ := n

def base_10_to_6 (n : ℕ) : ℕ := n

def arithmetic_sum (a₁ aₙ n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_base_6 (a₁ aₙ d : ℕ) 
  (h₁ : a₁ = base_6_to_10 5)
  (h₂ : aₙ = base_6_to_10 31)
  (h₃ : d = 2)
  (h₄ : aₙ = a₁ + (n - 1) * d) :
  base_10_to_6 (arithmetic_sum a₁ aₙ ((aₙ - a₁) / d + 1)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_base_6_l1316_131638


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l1316_131677

theorem product_of_two_digit_numbers (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a ≤ 99) 
  (h2 : 10 ≤ b ∧ b ≤ 99) 
  (h3 : a * b = 4500) 
  (h4 : a ≤ b) : 
  a = 50 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l1316_131677


namespace NUMINAMATH_CALUDE_gambler_winning_percentage_l1316_131667

/-- Calculates the final winning percentage of a gambler --/
theorem gambler_winning_percentage
  (initial_games : ℕ)
  (initial_win_rate : ℚ)
  (additional_games : ℕ)
  (new_win_rate : ℚ)
  (h1 : initial_games = 30)
  (h2 : initial_win_rate = 2/5)
  (h3 : additional_games = 30)
  (h4 : new_win_rate = 4/5) :
  let total_games := initial_games + additional_games
  let total_wins := initial_games * initial_win_rate + additional_games * new_win_rate
  total_wins / total_games = 3/5 := by
sorry

#eval (2/5 : ℚ)  -- To verify that 2/5 is indeed 0.4
#eval (4/5 : ℚ)  -- To verify that 4/5 is indeed 0.8
#eval (3/5 : ℚ)  -- To verify that 3/5 is indeed 0.6

end NUMINAMATH_CALUDE_gambler_winning_percentage_l1316_131667


namespace NUMINAMATH_CALUDE_min_additional_squares_for_axisymmetry_l1316_131682

/-- Represents a rectangle with shaded squares -/
structure ShadedRectangle where
  width : ℕ
  height : ℕ
  shadedSquares : Finset (ℕ × ℕ)

/-- Checks if a ShadedRectangle is axisymmetric with two lines of symmetry -/
def isAxisymmetric (rect : ShadedRectangle) : Prop :=
  ∀ (x y : ℕ), x < rect.width ∧ y < rect.height →
    ((x, y) ∈ rect.shadedSquares ↔ (rect.width - 1 - x, y) ∈ rect.shadedSquares) ∧
    ((x, y) ∈ rect.shadedSquares ↔ (x, rect.height - 1 - y) ∈ rect.shadedSquares)

/-- The theorem to be proved -/
theorem min_additional_squares_for_axisymmetry 
  (rect : ShadedRectangle) 
  (h : rect.shadedSquares.card = 3) : 
  ∃ (additionalSquares : Finset (ℕ × ℕ)),
    additionalSquares.card = 6 ∧
    isAxisymmetric ⟨rect.width, rect.height, rect.shadedSquares ∪ additionalSquares⟩ ∧
    ∀ (smallerSet : Finset (ℕ × ℕ)), 
      smallerSet.card < 6 → 
      ¬isAxisymmetric ⟨rect.width, rect.height, rect.shadedSquares ∪ smallerSet⟩ :=
sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_axisymmetry_l1316_131682


namespace NUMINAMATH_CALUDE_range_of_m_l1316_131619

-- Define the conditions
def p (x : ℝ) : Prop := (x + 2) / (10 - x) ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, q x m → p x) →  -- p is a necessary condition for q
  (m < 0) →             -- Given condition
  m ≥ -3 ∧ m < 0        -- Conclusion: range of m is [-3, 0)
  := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1316_131619


namespace NUMINAMATH_CALUDE_roots_sum_of_reciprocal_squares_l1316_131651

theorem roots_sum_of_reciprocal_squares (r s : ℂ) : 
  (3 * r^2 - 2 * r + 4 = 0) → 
  (3 * s^2 - 2 * s + 4 = 0) → 
  (1 / r^2 + 1 / s^2 = -5 / 4) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_reciprocal_squares_l1316_131651


namespace NUMINAMATH_CALUDE_fraction_problem_l1316_131632

theorem fraction_problem : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5100 = 765.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1316_131632


namespace NUMINAMATH_CALUDE_coin_packing_l1316_131617

theorem coin_packing (n : ℕ) (r R : ℝ) (hn : n > 0) (hr : r > 0) (hR : R > r) :
  (1 / 2 : ℝ) * (R / r - 1) ≤ Real.sqrt n ∧ Real.sqrt n ≤ R / r :=
by sorry

end NUMINAMATH_CALUDE_coin_packing_l1316_131617


namespace NUMINAMATH_CALUDE_functional_equation_problem_l1316_131654

/-- The functional equation problem -/
theorem functional_equation_problem (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f (y * f x - 1) = x^2 * f y - f x) ↔ (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l1316_131654


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l1316_131663

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The number of divisors of n -/
def numDivisors (n : ℕ) : ℕ := (Finset.filter (λ d => n % d = 0) (Finset.range (n + 1))).card

/-- The number of odd divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := (Finset.filter (λ d => n % d = 0 ∧ d % 2 ≠ 0) (Finset.range (n + 1))).card

/-- The probability of choosing an odd divisor of n -/
def probOddDivisor (n : ℕ) : ℚ := numOddDivisors n / numDivisors n

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l1316_131663


namespace NUMINAMATH_CALUDE_unique_monotonic_involutive_function_l1316_131636

-- Define the properties of the function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def Involutive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = x

-- Theorem statement
theorem unique_monotonic_involutive_function :
  ∀ f : ℝ → ℝ, Monotonic f → Involutive f → ∀ x : ℝ, f x = x :=
by
  sorry


end NUMINAMATH_CALUDE_unique_monotonic_involutive_function_l1316_131636


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1316_131678

-- Define the conditions
def condition_A (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

def condition_B (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

-- Theorem statement
theorem not_necessary_not_sufficient :
  (∃ θ a, condition_A θ a ∧ ¬condition_B θ a) ∧
  (∃ θ a, condition_B θ a ∧ ¬condition_A θ a) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1316_131678


namespace NUMINAMATH_CALUDE_picture_area_l1316_131680

/-- Given a sheet of paper with specified dimensions and margin, calculate the area of the picture --/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

#check picture_area

end NUMINAMATH_CALUDE_picture_area_l1316_131680


namespace NUMINAMATH_CALUDE_walter_fall_distance_l1316_131625

/-- The distance Walter fell before passing David -/
def distance_fallen (d : ℝ) : ℝ := 2 * d

theorem walter_fall_distance (d : ℝ) (h_positive : d > 0) :
  distance_fallen d = 2 * d :=
by sorry

end NUMINAMATH_CALUDE_walter_fall_distance_l1316_131625


namespace NUMINAMATH_CALUDE_michael_small_balls_l1316_131695

/-- Represents the number of rubber bands in a pack --/
def total_rubber_bands : ℕ := 5000

/-- Represents the number of rubber bands needed for a small ball --/
def small_ball_rubber_bands : ℕ := 50

/-- Represents the number of rubber bands needed for a large ball --/
def large_ball_rubber_bands : ℕ := 300

/-- Represents the number of large balls that can be made with remaining rubber bands --/
def remaining_large_balls : ℕ := 13

/-- Calculates the number of small balls Michael made --/
def small_balls_made : ℕ :=
  (total_rubber_bands - remaining_large_balls * large_ball_rubber_bands) / small_ball_rubber_bands

theorem michael_small_balls :
  small_balls_made = 22 :=
sorry

end NUMINAMATH_CALUDE_michael_small_balls_l1316_131695


namespace NUMINAMATH_CALUDE_rectangle_area_irrational_l1316_131653

-- Define a rectangle with rational length and irrational width
structure Rectangle where
  length : ℚ
  width : ℝ
  width_irrational : Irrational width

-- Define the area of the rectangle
def area (rect : Rectangle) : ℝ := (rect.length : ℝ) * rect.width

-- Theorem statement
theorem rectangle_area_irrational (rect : Rectangle) : Irrational (area rect) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_irrational_l1316_131653


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_16_l1316_131699

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter_16 (outer : Rectangle) (inner : Rectangle) (shaded_area : ℝ) :
  outer.width = 12 ∧ 
  outer.height = 10 ∧ 
  inner.width = 5 ∧ 
  inner.height = 3 ∧
  shaded_area = 120 →
  perimeter inner = 16 := by
sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_16_l1316_131699


namespace NUMINAMATH_CALUDE_complex_power_difference_l1316_131612

theorem complex_power_difference (x : ℂ) (h : x - 1/x = 2*I) : x^8 - 1/x^8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1316_131612


namespace NUMINAMATH_CALUDE_fourth_quarter_profits_l1316_131626

/-- Proves that given the annual profits, first quarter profits, and third quarter profits,
    the fourth quarter profits are equal to the difference between the annual profits
    and the sum of the first and third quarter profits. -/
theorem fourth_quarter_profits
  (annual_profits : ℕ)
  (first_quarter_profits : ℕ)
  (third_quarter_profits : ℕ)
  (h1 : annual_profits = 8000)
  (h2 : first_quarter_profits = 1500)
  (h3 : third_quarter_profits = 3000) :
  annual_profits - (first_quarter_profits + third_quarter_profits) = 3500 :=
by sorry

end NUMINAMATH_CALUDE_fourth_quarter_profits_l1316_131626


namespace NUMINAMATH_CALUDE_simplify_fraction_l1316_131692

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) :
  1 - 1 / (1 + a / (1 - a)) = a := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1316_131692


namespace NUMINAMATH_CALUDE_forest_tree_density_l1316_131666

/-- Calculates the tree density in a rectangular forest given the logging parameters --/
theorem forest_tree_density
  (forest_length : ℕ)
  (forest_width : ℕ)
  (loggers : ℕ)
  (months : ℕ)
  (days_per_month : ℕ)
  (trees_per_logger_per_day : ℕ)
  (h1 : forest_length = 4)
  (h2 : forest_width = 6)
  (h3 : loggers = 8)
  (h4 : months = 10)
  (h5 : days_per_month = 30)
  (h6 : trees_per_logger_per_day = 6) :
  (loggers * months * days_per_month * trees_per_logger_per_day) / (forest_length * forest_width) = 600 := by
  sorry

#check forest_tree_density

end NUMINAMATH_CALUDE_forest_tree_density_l1316_131666


namespace NUMINAMATH_CALUDE_train_speed_problem_l1316_131649

theorem train_speed_problem (v : ℝ) : 
  v > 0 →  -- The speed of the first train is positive
  (∃ t : ℝ, t > 0 ∧  -- There exists a positive time t when the trains meet
    v * t = 25 * t + 60 ∧  -- One train travels 60 km more than the other
    v * t + 25 * t = 540) →  -- Total distance traveled equals the distance between stations
  v = 31.25 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1316_131649


namespace NUMINAMATH_CALUDE_complex_division_result_l1316_131643

theorem complex_division_result : Complex.I / (1 - Complex.I) = -1/2 + Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1316_131643


namespace NUMINAMATH_CALUDE_solution_characterization_l1316_131607

theorem solution_characterization (x y : ℤ) :
  x^2 - y^4 = 2009 ↔ (x = 45 ∧ y = 2) ∨ (x = 45 ∧ y = -2) ∨ (x = -45 ∧ y = 2) ∨ (x = -45 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1316_131607


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1316_131691

theorem complex_fraction_simplification :
  (5 + 3*Complex.I) / (2 + 3*Complex.I) = 19/13 - 9/13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1316_131691


namespace NUMINAMATH_CALUDE_total_distance_rowed_total_distance_is_15_19_l1316_131644

/-- Calculates the total distance traveled by a man rowing upstream and downstream in a river -/
theorem total_distance_rowed (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (total_time * upstream_speed * downstream_speed) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

/-- Proves that the total distance traveled is approximately 15.19 km -/
theorem total_distance_is_15_19 :
  ∃ ε > 0, |total_distance_rowed 8 1.8 2 - 15.19| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_rowed_total_distance_is_15_19_l1316_131644


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_variables_l1316_131669

theorem arithmetic_geometric_inequality_two_variables 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ 
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_variables_l1316_131669


namespace NUMINAMATH_CALUDE_max_acute_angles_eq_three_l1316_131608

/-- A convex polygon with n sides, where n ≥ 3 --/
structure ConvexPolygon where
  n : ℕ
  n_ge_three : n ≥ 3

/-- The maximum number of acute angles in a convex polygon --/
def max_acute_angles (p : ConvexPolygon) : ℕ := 3

/-- Theorem: The maximum number of acute angles in a convex polygon is 3 --/
theorem max_acute_angles_eq_three (p : ConvexPolygon) :
  max_acute_angles p = 3 := by sorry

end NUMINAMATH_CALUDE_max_acute_angles_eq_three_l1316_131608


namespace NUMINAMATH_CALUDE_sehnenviereck_ungleichung_infinitely_many_equality_cases_l1316_131631

theorem sehnenviereck_ungleichung (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (ha1 : a < 1) (hb1 : b < 1) (hc1 : c < 1) (hd1 : d < 1)
  (sum : a + b + c + d = 2) :
  Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ (a * c + b * d) / 2 :=
sorry

theorem infinitely_many_equality_cases :
  ∃ S : Set (ℝ × ℝ × ℝ × ℝ), Cardinal.mk S = Cardinal.mk ℝ ∧
  ∀ (a b c d : ℝ), (a, b, c, d) ∈ S →
    0 < a ∧ a < 1 ∧
    0 < b ∧ b < 1 ∧
    0 < c ∧ c < 1 ∧
    0 < d ∧ d < 1 ∧
    a + b + c + d = 2 ∧
    Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = (a * c + b * d) / 2 :=
sorry

end NUMINAMATH_CALUDE_sehnenviereck_ungleichung_infinitely_many_equality_cases_l1316_131631


namespace NUMINAMATH_CALUDE_function_value_at_one_l1316_131624

/-- Given a function f where f(x-3) = 2x^2 - 3x + 1, prove that f(1) = 21 -/
theorem function_value_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f (x - 3) = 2 * x^2 - 3 * x + 1) : 
  f 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_one_l1316_131624


namespace NUMINAMATH_CALUDE_logarithm_expression_evaluation_l1316_131627

theorem logarithm_expression_evaluation :
  Real.log 5 / Real.log 10 + Real.log 2 / Real.log 10 + (3/5)^0 + Real.log (Real.exp (1/2)) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_evaluation_l1316_131627


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1316_131634

theorem linear_function_quadrants (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x + b ∧ x > 0 ∧ y > 0) ∧  -- First quadrant
  (∃ x y : ℝ, y = a * x + b ∧ x < 0 ∧ y > 0) ∧  -- Second quadrant
  (∃ x y : ℝ, y = a * x + b ∧ x > 0 ∧ y < 0) →  -- Fourth quadrant
  ¬(∃ x y : ℝ, y = b * x - a ∧ x > 0 ∧ y < 0)   -- Not in fourth quadrant
:= by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1316_131634


namespace NUMINAMATH_CALUDE_reema_loan_interest_l1316_131681

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal rate time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem reema_loan_interest :
  let principal : ℕ := 1200
  let rate : ℕ := 6
  let time : ℕ := rate
  simple_interest principal rate time = 432 := by
  sorry

end NUMINAMATH_CALUDE_reema_loan_interest_l1316_131681


namespace NUMINAMATH_CALUDE_grace_has_30_pastries_l1316_131602

/-- The number of pastries each person has -/
structure Pastries where
  frank : ℕ
  calvin : ℕ
  phoebe : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def pastry_conditions (p : Pastries) : Prop :=
  p.calvin = p.frank + 8 ∧
  p.phoebe = p.frank + 8 ∧
  p.grace = p.calvin + 5 ∧
  p.frank + p.calvin + p.phoebe + p.grace = 97

/-- The theorem stating that Grace has 30 pastries -/
theorem grace_has_30_pastries (p : Pastries) 
  (h : pastry_conditions p) : p.grace = 30 := by
  sorry


end NUMINAMATH_CALUDE_grace_has_30_pastries_l1316_131602


namespace NUMINAMATH_CALUDE_nth_equation_l1316_131618

theorem nth_equation (n : ℕ+) : (10 * n + 5)^2 = n * (n + 1) * 100 + 5^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l1316_131618


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1316_131675

theorem trigonometric_problem (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α + Real.cos α = 1/5) :
  (Real.sin α - Real.cos α = 7/5) ∧
  (Real.sin (2 * α + π/3) = -12/25 - 7 * Real.sqrt 3 / 50) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1316_131675


namespace NUMINAMATH_CALUDE_min_pool_cost_is_5400_l1316_131697

/-- Represents the specifications of a rectangular pool -/
structure PoolSpecs where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the minimum cost of constructing a rectangular pool given its specifications -/
def minPoolCost (specs : PoolSpecs) : ℝ :=
  sorry

/-- Theorem stating that the minimum cost of constructing the specified pool is 5400 yuan -/
theorem min_pool_cost_is_5400 :
  let specs : PoolSpecs := {
    volume := 18,
    depth := 2,
    bottomCost := 200,
    wallCost := 150
  }
  minPoolCost specs = 5400 :=
by sorry

end NUMINAMATH_CALUDE_min_pool_cost_is_5400_l1316_131697


namespace NUMINAMATH_CALUDE_jeans_savings_l1316_131615

/-- Calculates the total amount saved on a purchase with multiple discounts and taxes -/
def calculateSavings (originalPrice : ℝ) (saleDiscount : ℝ) (couponDiscount : ℝ) 
                     (creditCardDiscount : ℝ) (rebateDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterSale := originalPrice * (1 - saleDiscount)
  let priceAfterCoupon := priceAfterSale - couponDiscount
  let priceAfterCreditCard := priceAfterCoupon * (1 - creditCardDiscount)
  let priceBeforeRebate := priceAfterCreditCard
  let taxAmount := priceBeforeRebate * salesTax
  let finalPrice := (priceBeforeRebate - priceBeforeRebate * rebateDiscount) + taxAmount
  originalPrice - finalPrice

theorem jeans_savings :
  calculateSavings 125 0.20 10 0.10 0.05 0.08 = 41.57 := by
  sorry

end NUMINAMATH_CALUDE_jeans_savings_l1316_131615


namespace NUMINAMATH_CALUDE_solution_in_interval_l1316_131637

-- Define the function f(x) = x^3 + x - 4
def f (x : ℝ) : ℝ := x^3 + x - 4

-- State the theorem
theorem solution_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 (3/2) ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1316_131637


namespace NUMINAMATH_CALUDE_y_share_is_27_l1316_131670

/-- Given a sum divided among x, y, and z, where y gets 45 paisa and z gets 50 paisa for each rupee x gets, 
    and the total amount is Rs. 117, prove that y's share is Rs. 27. -/
theorem y_share_is_27 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 117) 
  (h2 : y_share = 0.45 * x_share) 
  (h3 : z_share = 0.50 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  y_share = 27 := by
sorry


end NUMINAMATH_CALUDE_y_share_is_27_l1316_131670


namespace NUMINAMATH_CALUDE_no_periodic_sum_with_periods_one_and_pi_l1316_131611

/-- A function is periodic if it takes at least two different values and there exists a positive real number p such that f(x + p) = f(x) for all x. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- p is a period of f if f(x + p) = f(x) for all x. -/
def IsPeriodOf (p : ℝ) (f : ℝ → ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem no_periodic_sum_with_periods_one_and_pi :
  ¬ ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ IsPeriodic h ∧
    IsPeriodOf 1 g ∧ IsPeriodOf Real.pi h ∧
    IsPeriodic (g + h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sum_with_periods_one_and_pi_l1316_131611


namespace NUMINAMATH_CALUDE_probability_two_absent_one_present_l1316_131661

theorem probability_two_absent_one_present :
  let p_absent : ℚ := 1 / 30
  let p_present : ℚ := 1 - p_absent
  let p_two_absent_one_present : ℚ := 
    3 * p_absent * p_absent * p_present
  p_two_absent_one_present = 29 / 9000 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_absent_one_present_l1316_131661


namespace NUMINAMATH_CALUDE_cube_sum_l1316_131606

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The sum of faces, edges, and vertices in a cube is 26 -/
theorem cube_sum : cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l1316_131606


namespace NUMINAMATH_CALUDE_f_properties_l1316_131645

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- Main theorem
theorem f_properties (a : ℝ) (h : a > 1) :
  -- 1. Explicit formula for f
  (∀ x, f a x = (a / (a^2 - 1)) * (a^x - a^(-x))) ∧
  -- 2. f is odd and increasing
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x y, x < y → f a x < f a y) ∧
  -- 3. Range of m
  (∀ m, (∀ x ∈ Set.Ioo (-1) 1, f a (1 - m) + f a (1 - m^2) < 0) →
    1 < m ∧ m < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1316_131645


namespace NUMINAMATH_CALUDE_vector_position_at_negative_two_l1316_131629

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  position : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfying the problem conditions -/
def given_line : ParametricLine :=
  { position := sorry }

theorem vector_position_at_negative_two :
  let l := given_line
  (l.position 1 = (2, 0, -3)) →
  (l.position 2 = (7, -2, 1)) →
  (l.position 4 = (17, -6, 9)) →
  l.position (-2) = (-1, 3, -9) := by
  sorry

end NUMINAMATH_CALUDE_vector_position_at_negative_two_l1316_131629


namespace NUMINAMATH_CALUDE_star_three_four_l1316_131690

-- Define the * operation
def star (a b : ℝ) : ℝ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_three_four : star 3 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l1316_131690


namespace NUMINAMATH_CALUDE_system_solution_l1316_131623

theorem system_solution (x y : ℝ) (h1 : 3 * x + 2 * y = 2) (h2 : 2 * x + 3 * y = 8) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1316_131623


namespace NUMINAMATH_CALUDE_justine_coloring_ratio_l1316_131694

/-- Given a total number of sheets, number of binders, and sheets used by Justine,
    prove that the ratio of sheets Justine colored to total sheets in her binder is 1:2 -/
theorem justine_coloring_ratio 
  (total_sheets : ℕ) 
  (num_binders : ℕ) 
  (sheets_used : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : num_binders = 5)
  (h3 : sheets_used = 245)
  : (sheets_used : ℚ) / (total_sheets / num_binders) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_justine_coloring_ratio_l1316_131694


namespace NUMINAMATH_CALUDE_poems_sally_can_recite_l1316_131662

theorem poems_sally_can_recite (initial_poems : ℕ) (forgotten_poems : ℕ) : 
  initial_poems = 8 → forgotten_poems = 5 → initial_poems - forgotten_poems = 3 := by
sorry

end NUMINAMATH_CALUDE_poems_sally_can_recite_l1316_131662


namespace NUMINAMATH_CALUDE_train_length_l1316_131639

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 65 * 1000 / 3600) 
    (h2 : t = 15.506451791548985) (h3 : bridge_length = 150) : 
    v * t - bridge_length = 130 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1316_131639


namespace NUMINAMATH_CALUDE_greatest_triangle_perimeter_l1316_131652

theorem greatest_triangle_perimeter : ∃ (a b c : ℕ),
  (a > 0 ∧ b > 0 ∧ c > 0) ∧  -- positive integer side lengths
  (b = 4 * a) ∧              -- one side is four times as long as a second side
  (c = 20) ∧                 -- the third side has length 20
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧  -- triangle inequality
  (∀ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) → 
    (y = 4 * x) → (z = 20) → 
    (x + y > z ∧ y + z > x ∧ z + x > y) →
    (x + y + z ≤ a + b + c)) ∧
  (a + b + c = 50) :=
by sorry

end NUMINAMATH_CALUDE_greatest_triangle_perimeter_l1316_131652


namespace NUMINAMATH_CALUDE_orthic_triangle_smallest_perimeter_l1316_131676

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if a triangle is inscribed in another triangle -/
def isInscribed (inner outer : Triangle) : Prop := 
  isOnSegment inner.A outer.B outer.C ∧
  isOnSegment inner.B outer.A outer.C ∧
  isOnSegment inner.C outer.A outer.B

/-- Constructs the orthic triangle of a given triangle -/
def orthicTriangle (t : Triangle) : Triangle := sorry

/-- The main theorem: the orthic triangle has the smallest perimeter among all inscribed triangles -/
theorem orthic_triangle_smallest_perimeter (ABC : Triangle) 
  (h_acute : isAcuteAngled ABC) :
  let PQR := orthicTriangle ABC
  ∀ XYZ : Triangle, isInscribed XYZ ABC → perimeter PQR ≤ perimeter XYZ := by
  sorry

end NUMINAMATH_CALUDE_orthic_triangle_smallest_perimeter_l1316_131676


namespace NUMINAMATH_CALUDE_inverse_proposition_true_l1316_131696

theorem inverse_proposition_true : 
  (∃ x y : ℝ, x ≤ y ∧ x ≤ |y|) ∧ 
  ¬(∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) ∧ 
  ¬(∃ x : ℝ, x = 1 ∧ x^2 + x - 2 ≠ 0) ∧ 
  ¬(∀ x : ℝ, x ≤ 1 → x^2 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_true_l1316_131696


namespace NUMINAMATH_CALUDE_meat_price_proof_l1316_131635

/-- The price of meat per ounce in cents -/
def meat_price : ℝ := 6

theorem meat_price_proof :
  (∃ (paid_16 paid_8 : ℝ),
    16 * meat_price = paid_16 - 30 ∧
    8 * meat_price = paid_8 + 18) :=
by sorry

end NUMINAMATH_CALUDE_meat_price_proof_l1316_131635


namespace NUMINAMATH_CALUDE_correct_calculation_l1316_131620

theorem correct_calculation (x : ℝ) (h : x ≠ 0) : (x^2 + x) / x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1316_131620


namespace NUMINAMATH_CALUDE_eight_person_round_robin_matches_l1316_131642

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: An 8-person round-robin tennis tournament has 28 matches -/
theorem eight_person_round_robin_matches :
  roundRobinMatches 8 = 28 := by
  sorry

#eval roundRobinMatches 8  -- This should output 28

end NUMINAMATH_CALUDE_eight_person_round_robin_matches_l1316_131642


namespace NUMINAMATH_CALUDE_trigonometric_identity_proof_l1316_131686

theorem trigonometric_identity_proof :
  Real.cos (13 * π / 180) * Real.sin (58 * π / 180) - 
  Real.sin (13 * π / 180) * Real.sin (32 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_proof_l1316_131686


namespace NUMINAMATH_CALUDE_right_triangle_30_hypotenuse_l1316_131659

/-- A right triangle with one angle of 30 degrees -/
structure RightTriangle30 where
  /-- The length of side XZ -/
  xz : ℝ
  /-- XZ is positive -/
  xz_pos : 0 < xz

/-- The length of the hypotenuse in a right triangle with a 30-degree angle -/
def hypotenuse (t : RightTriangle30) : ℝ := 2 * t.xz

/-- Theorem: In a right triangle XYZ with right angle at X, if angle YZX = 30° and XZ = 15, then XY = 30 -/
theorem right_triangle_30_hypotenuse :
  ∀ t : RightTriangle30, t.xz = 15 → hypotenuse t = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_30_hypotenuse_l1316_131659


namespace NUMINAMATH_CALUDE_friendly_numbers_solution_l1316_131621

/-- Two rational numbers are friendly if their sum is 66 -/
def friendly (m n : ℚ) : Prop := m + n = 66

/-- Given that 7x and -18 are friendly numbers, prove that x = 12 -/
theorem friendly_numbers_solution (x : ℚ) (h : friendly (7 * x) (-18)) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_friendly_numbers_solution_l1316_131621
