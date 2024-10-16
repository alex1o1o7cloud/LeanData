import Mathlib

namespace NUMINAMATH_CALUDE_jane_albert_same_committee_l877_87732

/-- The number of second-year MBAs -/
def total_mbas : ℕ := 9

/-- The number of committees to be formed -/
def num_committees : ℕ := 3

/-- The number of members in each committee -/
def committee_size : ℕ := 4

/-- The probability that Jane and Albert are on the same committee -/
def probability_same_committee : ℚ := 1 / 6

theorem jane_albert_same_committee :
  let total_ways := (total_mbas.choose committee_size) * ((total_mbas - committee_size).choose committee_size)
  let ways_together := ((total_mbas - 2).choose (committee_size - 2)) * ((total_mbas - committee_size).choose committee_size)
  (ways_together : ℚ) / total_ways = probability_same_committee :=
sorry

end NUMINAMATH_CALUDE_jane_albert_same_committee_l877_87732


namespace NUMINAMATH_CALUDE_system_solution_l877_87797

theorem system_solution (n k m : ℕ+) 
  (eq1 : n + k = (Nat.gcd n k)^2)
  (eq2 : k + m = (Nat.gcd k m)^2) :
  n = 2 ∧ k = 2 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l877_87797


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l877_87774

theorem max_value_on_ellipse :
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 9 = 1) → (2*x - y ≤ 5) ∧ ∃ x₀ y₀ : ℝ, (x₀^2 / 4 + y₀^2 / 9 = 1) ∧ (2*x₀ - y₀ = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l877_87774


namespace NUMINAMATH_CALUDE_no_valid_numbers_l877_87736

def digits : List Nat := [2, 3, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n % 15 = 0) ∧
  (∀ d : Nat, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d)) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem no_valid_numbers : ¬∃ n : Nat, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l877_87736


namespace NUMINAMATH_CALUDE_difference_of_squares_l877_87785

theorem difference_of_squares (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l877_87785


namespace NUMINAMATH_CALUDE_minimum_value_implies_ratio_l877_87754

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem minimum_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≥ 10) ∧  -- f(x) has a minimum value of 10
  (f a b 1 = 10) ∧  -- The minimum occurs at x = 1
  (f_derivative a b 1 = 0)  -- The derivative is zero at x = 1
  → b / a = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_ratio_l877_87754


namespace NUMINAMATH_CALUDE_julie_work_hours_l877_87765

/-- Calculates the number of hours Julie needs to work per week during the school year -/
def school_year_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℚ) 
                      (school_year_weeks : ℕ) (school_year_earnings : ℚ) : ℚ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let weekly_earnings := school_year_earnings / school_year_weeks
  weekly_earnings / hourly_wage

theorem julie_work_hours :
  school_year_hours 10 60 7500 50 7500 = 12 := by
  sorry

end NUMINAMATH_CALUDE_julie_work_hours_l877_87765


namespace NUMINAMATH_CALUDE_mark_election_votes_l877_87712

theorem mark_election_votes (first_area_voters : ℕ) (first_area_percentage : ℚ) :
  first_area_voters = 100000 →
  first_area_percentage = 70 / 100 →
  (first_area_voters * first_area_percentage).floor +
  2 * (first_area_voters * first_area_percentage).floor = 210000 :=
by sorry

end NUMINAMATH_CALUDE_mark_election_votes_l877_87712


namespace NUMINAMATH_CALUDE_pizza_theorem_l877_87727

def pizza_eaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else 1/3 + (1 - 1/3) * (1 - (1/2)^(n-1))

theorem pizza_theorem : pizza_eaten 4 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l877_87727


namespace NUMINAMATH_CALUDE_value_of_m_l877_87717

theorem value_of_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m
  let g : ℝ → ℝ := λ x => x^2 - 2*x + 2*m
  3 * f 3 = g 3 → m = 12 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l877_87717


namespace NUMINAMATH_CALUDE_problem_solution_l877_87743

theorem problem_solution (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + 2 / b = d) (h2 : b + 2 / c = d) (h3 : c + 2 / a = d) :
  d = Real.sqrt 2 ∨ d = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l877_87743


namespace NUMINAMATH_CALUDE_sqrt_360_simplification_l877_87701

theorem sqrt_360_simplification : Real.sqrt 360 = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360_simplification_l877_87701


namespace NUMINAMATH_CALUDE_point_on_line_l877_87729

theorem point_on_line (n : ℕ) (P : ℕ → ℤ × ℤ) : 
  (P 0 = (0, 1)) →
  (∀ k : ℕ, k ≥ 1 → k ≤ n → (P k).1 - (P (k-1)).1 = 1) →
  (∀ k : ℕ, k ≥ 1 → k ≤ n → (P k).2 - (P (k-1)).2 = 2) →
  (P n).1 = n →
  (P n).2 = 2*n + 1 →
  (P n).2 = 3*(P n).1 - 8 →
  n = 9 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l877_87729


namespace NUMINAMATH_CALUDE_special_line_equation_l877_87706

/-- A line passing through a point (x₀, y₀) with x-intercept twice the y-intercept --/
structure SpecialLine where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ  -- x-intercept
  b : ℝ  -- y-intercept
  h1 : a = 2 * b
  h2 : y₀ - (-b) = (x₀ - a) * (-b / a)

/-- The equation of the special line passing through (3, -1) --/
theorem special_line_equation :
  ∃ (l : SpecialLine), l.x₀ = 3 ∧ l.y₀ = -1 ∧
  ∀ (x y : ℝ), (x + 2*y = 1) ↔ (y - l.y₀ = (x - l.x₀) * (-1/2)) :=
by sorry

end NUMINAMATH_CALUDE_special_line_equation_l877_87706


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l877_87704

/-- The total area of the global ocean in million square kilometers -/
def ocean_area : ℝ := 36200

/-- The conversion factor from million to scientific notation -/
def million_to_scientific : ℝ := 10^6

theorem ocean_area_scientific_notation :
  ocean_area * million_to_scientific = 3.62 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l877_87704


namespace NUMINAMATH_CALUDE_shopper_receives_115_l877_87728

/-- Represents the amount of money each person has -/
structure MoneyDistribution where
  isabella : ℕ
  sam : ℕ
  giselle : ℕ

/-- Calculates the amount each shopper receives when the total is divided equally -/
def amountPerShopper (md : MoneyDistribution) : ℕ :=
  (md.isabella + md.sam + md.giselle) / 3

/-- Theorem stating the amount each shopper receives under the given conditions -/
theorem shopper_receives_115 (md : MoneyDistribution) 
  (h1 : md.isabella = md.sam + 45)
  (h2 : md.isabella = md.giselle + 15)
  (h3 : md.giselle = 120) :
  amountPerShopper md = 115 := by
  sorry

#eval amountPerShopper { isabella := 135, sam := 90, giselle := 120 }

end NUMINAMATH_CALUDE_shopper_receives_115_l877_87728


namespace NUMINAMATH_CALUDE_blue_lights_l877_87723

/-- The number of blue lights on a Christmas tree -/
theorem blue_lights (total : ℕ) (red : ℕ) (yellow : ℕ) 
  (h1 : total = 95)
  (h2 : red = 26)
  (h3 : yellow = 37) :
  total - (red + yellow) = 32 := by
  sorry

end NUMINAMATH_CALUDE_blue_lights_l877_87723


namespace NUMINAMATH_CALUDE_greatest_number_l877_87748

theorem greatest_number : 
  8^85 > 5^100 ∧ 8^85 > 6^91 ∧ 8^85 > 7^90 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l877_87748


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2537_l877_87746

theorem smallest_prime_factor_of_2537 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2537 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2537 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2537_l877_87746


namespace NUMINAMATH_CALUDE_nth_prime_power_bound_l877_87700

/-- p_nth n returns the n-th prime number -/
def p_nth : ℕ → ℕ := sorry

/-- Theorem stating that for any positive integers n and k, 
    n is less than the k-th power of the 2k-th prime number -/
theorem nth_prime_power_bound (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n < (p_nth (2 * k)) ^ k := by sorry

end NUMINAMATH_CALUDE_nth_prime_power_bound_l877_87700


namespace NUMINAMATH_CALUDE_root_in_interval_l877_87777

def f (x : ℝ) := 3*x + x - 3

theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l877_87777


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l877_87795

theorem factorization_cubic_minus_linear (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l877_87795


namespace NUMINAMATH_CALUDE_betty_bead_ratio_l877_87747

/-- Given that Betty has 30 red beads and 20 blue beads, prove that the ratio of red beads to blue beads is 3:2 -/
theorem betty_bead_ratio :
  let red_beads : ℕ := 30
  let blue_beads : ℕ := 20
  (red_beads : ℚ) / blue_beads = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_betty_bead_ratio_l877_87747


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l877_87731

theorem fractional_equation_solution :
  ∃ x : ℝ, x ≠ 3 ∧ (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l877_87731


namespace NUMINAMATH_CALUDE_sin_negative_150_degrees_l877_87780

theorem sin_negative_150_degrees :
  Real.sin (-(150 * π / 180)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_150_degrees_l877_87780


namespace NUMINAMATH_CALUDE_factor_z6_minus_64_l877_87779

theorem factor_z6_minus_64 (z : ℂ) : 
  z^6 - 64 = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := by
  sorry

#check factor_z6_minus_64

end NUMINAMATH_CALUDE_factor_z6_minus_64_l877_87779


namespace NUMINAMATH_CALUDE_problem_statement_l877_87744

theorem problem_statement (x y z : ℝ) (hx : x = 2) (hy : y = -3) (hz : z = 1) :
  x^2 + y^2 + z^2 + 2*x*y - z^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l877_87744


namespace NUMINAMATH_CALUDE_angle_halving_l877_87730

/-- An angle is in the third quadrant if it's between π and 3π/2 (modulo 2π) -/
def is_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

/-- An angle is in the second or fourth quadrant if it's between π/2 and 3π/4 or between 3π/2 and 7π/4 (modulo 2π) -/
def is_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + 3 * Real.pi / 4) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < k * Real.pi + 7 * Real.pi / 4)

theorem angle_halving (α : Real) :
  is_third_quadrant α → is_second_or_fourth_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_halving_l877_87730


namespace NUMINAMATH_CALUDE_siblings_age_sum_l877_87739

theorem siblings_age_sum (a b c : ℕ+) : 
  a < b ∧ b = c ∧ a * b * c = 144 → a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l877_87739


namespace NUMINAMATH_CALUDE_quadratic_expression_as_square_plus_constant_l877_87756

theorem quadratic_expression_as_square_plus_constant :
  ∃ k : ℤ, ∀ y : ℝ, y^2 + 14*y + 60 = (y + 7)^2 + k :=
by
  -- Proof goes here
  sorry

#eval (60 : ℤ) - (7 : ℤ)^2  -- This should evaluate to 11

end NUMINAMATH_CALUDE_quadratic_expression_as_square_plus_constant_l877_87756


namespace NUMINAMATH_CALUDE_correct_answer_l877_87720

theorem correct_answer : ∃ x : ℤ, (x + 3 = 45) ∧ (x - 3 = 39) := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l877_87720


namespace NUMINAMATH_CALUDE_shopping_money_l877_87784

theorem shopping_money (initial_amount : ℝ) (remaining_amount : ℝ) : 
  remaining_amount = 140 →
  remaining_amount = initial_amount * (1 - 0.3) →
  initial_amount = 200 := by
sorry

end NUMINAMATH_CALUDE_shopping_money_l877_87784


namespace NUMINAMATH_CALUDE_complex_multiplication_l877_87702

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l877_87702


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l877_87787

theorem cubic_equation_solutions :
  ∀ m n : ℤ, (n^3 + m^3 + 231 = n^2 * m^2 + n * m) ↔ ((m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l877_87787


namespace NUMINAMATH_CALUDE_complex_number_problem_l877_87733

theorem complex_number_problem (z : ℂ) (m n : ℝ) :
  (z.re > 0) →
  (Complex.abs z = 2 * Real.sqrt 5) →
  ((1 + 2 * Complex.I) * z).re = 0 →
  (z ^ 2 + m * z + n = 0) →
  (z = 4 + 2 * Complex.I ∧ m = -8 ∧ n = 20) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l877_87733


namespace NUMINAMATH_CALUDE_water_level_change_time_correct_l877_87724

noncomputable def water_level_change_time (S H h s V g : ℝ) : ℝ :=
  let a := S / (0.6 * s * Real.sqrt (2 * g))
  let b := V / (0.6 * s * Real.sqrt (2 * g))
  2 * a * (Real.sqrt H - Real.sqrt (H - h) + b * Real.log (abs ((Real.sqrt H - b) / (Real.sqrt (H - h) - b))))

theorem water_level_change_time_correct
  (S H h s V g : ℝ)
  (h_S : S > 0)
  (h_H : H > 0)
  (h_h : 0 < h ∧ h < H)
  (h_s : s > 0)
  (h_V : V ≥ 0)
  (h_g : g > 0) :
  ∃ T : ℝ, T = water_level_change_time S H h s V g ∧ T > 0 :=
sorry

end NUMINAMATH_CALUDE_water_level_change_time_correct_l877_87724


namespace NUMINAMATH_CALUDE_wages_payment_l877_87762

/-- Given a sum of money that can pay A's wages for 20 days and B's wages for 30 days,
    prove that it can pay both A and B's wages for 12 days. -/
theorem wages_payment (S A B : ℝ) (hA : S = 20 * A) (hB : S = 30 * B) :
  S = 12 * (A + B) := by
  sorry

end NUMINAMATH_CALUDE_wages_payment_l877_87762


namespace NUMINAMATH_CALUDE_two_primes_equal_l877_87768

theorem two_primes_equal (a b c : ℕ) 
  (hp : Nat.Prime (b^c + a))
  (hq : Nat.Prime (a^b + c))
  (hr : Nat.Prime (c^a + b)) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    ((x = b^c + a ∧ y = a^b + c) ∨
     (x = b^c + a ∧ y = c^a + b) ∨
     (x = a^b + c ∧ y = c^a + b)) ∧
    x = y :=
sorry

end NUMINAMATH_CALUDE_two_primes_equal_l877_87768


namespace NUMINAMATH_CALUDE_marked_squares_theorem_l877_87794

/-- A type representing a table with marked squares -/
def MarkedTable (n : ℕ) := Fin n → Fin n → Bool

/-- A function that checks if a square is on or above the main diagonal -/
def isAboveDiagonal {n : ℕ} (i j : Fin n) : Bool :=
  i.val ≤ j.val

/-- A function that counts the number of marked squares in a table -/
def countMarkedSquares {n : ℕ} (table : MarkedTable n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if table i j then 1 else 0))

/-- A predicate that checks if a table can be rearranged to satisfy the condition -/
def canRearrange {n : ℕ} (table : MarkedTable n) : Prop :=
  ∃ (rowPerm colPerm : Equiv.Perm (Fin n)),
    ∀ i j, table i j → isAboveDiagonal (rowPerm i) (colPerm j)

theorem marked_squares_theorem (n : ℕ) (h : n > 1) :
  ∀ (table : MarkedTable n),
    canRearrange table ↔ countMarkedSquares table ≤ n + 1 :=
by sorry

end NUMINAMATH_CALUDE_marked_squares_theorem_l877_87794


namespace NUMINAMATH_CALUDE_frog_climb_days_l877_87781

/-- The number of days required for a frog to climb out of a well -/
def days_to_climb (well_depth : ℕ) (climb_distance : ℕ) (slide_distance : ℕ) : ℕ :=
  (well_depth + climb_distance - slide_distance - 1) / (climb_distance - slide_distance) + 1

/-- Theorem: A frog in a 50-meter well, climbing 5 meters up and sliding 2 meters down daily, 
    takes at least 16 days to reach the top -/
theorem frog_climb_days :
  days_to_climb 50 5 2 ≥ 16 := by
  sorry

#eval days_to_climb 50 5 2

end NUMINAMATH_CALUDE_frog_climb_days_l877_87781


namespace NUMINAMATH_CALUDE_andrews_appetizers_l877_87791

/-- The number of hotdogs on sticks Andrew brought -/
def hotdogs : ℕ := 30

/-- The number of bite-sized cheese pops Andrew brought -/
def cheese_pops : ℕ := 20

/-- The number of chicken nuggets Andrew brought -/
def chicken_nuggets : ℕ := 40

/-- The total number of appetizer portions Andrew brought -/
def total_appetizers : ℕ := hotdogs + cheese_pops + chicken_nuggets

theorem andrews_appetizers :
  total_appetizers = 90 :=
by sorry

end NUMINAMATH_CALUDE_andrews_appetizers_l877_87791


namespace NUMINAMATH_CALUDE_quarter_count_l877_87734

/-- Given a sum of $3.35 consisting of quarters and dimes, with a total of 23 coins, 
    prove that the number of quarters is 7. -/
theorem quarter_count (total_value : ℚ) (total_coins : ℕ) (quarter_value dime_value : ℚ) 
  (h1 : total_value = 335/100)
  (h2 : total_coins = 23)
  (h3 : quarter_value = 25/100)
  (h4 : dime_value = 1/10)
  : ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧ 
    quarters * quarter_value + dimes * dime_value = total_value ∧
    quarters = 7 :=
by sorry

end NUMINAMATH_CALUDE_quarter_count_l877_87734


namespace NUMINAMATH_CALUDE_tangent_forms_345_triangle_l877_87771

/-- An isosceles triangle with leg 10 cm and base 12 cm -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ
  leg_positive : 0 < leg
  base_positive : 0 < base
  isosceles : leg = 10
  base_length : base = 12

/-- The inscribed circle of the triangle -/
def inscribed_circle (t : IsoscelesTriangle) : ℝ := sorry

/-- Tangent line to the inscribed circle parallel to the height of the triangle -/
def tangent_line (t : IsoscelesTriangle) (c : ℝ) : ℝ → ℝ := sorry

/-- Right triangle formed by the tangent line -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse_positive : 0 < hypotenuse
  leg1_positive : 0 < leg1
  leg2_positive : 0 < leg2
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- The theorem to be proved -/
theorem tangent_forms_345_triangle (t : IsoscelesTriangle) (c : ℝ) :
  ∃ (rt : RightTriangle), rt.leg1 = 3 ∧ rt.leg2 = 4 ∧ rt.hypotenuse = 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_forms_345_triangle_l877_87771


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l877_87789

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 17) (h₃ : a₃ = 31) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 409 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l877_87789


namespace NUMINAMATH_CALUDE_tangent_angle_inclination_l877_87783

/-- The angle of inclination of the tangent to y = (1/3)x³ - 2 at (1, -5/3) is 45° --/
theorem tangent_angle_inclination (f : ℝ → ℝ) (x : ℝ) :
  f x = (1/3) * x^3 - 2 →
  (deriv f) x = x^2 →
  x = 1 →
  f x = -5/3 →
  Real.arctan ((deriv f) x) = π/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_inclination_l877_87783


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l877_87715

def line_equation (x y : ℝ) : Prop :=
  2 * (x - 3) + (-1) * (y - (-4)) = 6

theorem line_equation_equivalence :
  ∀ x y : ℝ, line_equation x y ↔ y = 2 * x - 16 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l877_87715


namespace NUMINAMATH_CALUDE_abs_2x_minus_7_not_positive_l877_87761

theorem abs_2x_minus_7_not_positive (x : ℚ) : ¬(0 < |2*x - 7|) ↔ x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_minus_7_not_positive_l877_87761


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l877_87764

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l877_87764


namespace NUMINAMATH_CALUDE_train_length_calculation_l877_87738

/-- Proves that a train with given speed passing a platform of known length in a specific time has a certain length. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (passing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  platform_length = 390 →
  passing_time = 60 →
  train_speed * passing_time - platform_length = 360 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l877_87738


namespace NUMINAMATH_CALUDE_donnas_earnings_proof_l877_87708

/-- Calculates Donna's total earnings over 7 days based on her work schedule --/
def donnas_weekly_earnings (dog_walking_rate : ℚ) (dog_walking_hours : ℚ) 
  (card_shop_rate : ℚ) (card_shop_hours : ℚ) (card_shop_days : ℕ)
  (babysitting_rate : ℚ) (babysitting_hours : ℚ) : ℚ :=
  (dog_walking_rate * dog_walking_hours * 7) + 
  (card_shop_rate * card_shop_hours * card_shop_days) + 
  (babysitting_rate * babysitting_hours)

theorem donnas_earnings_proof : 
  donnas_weekly_earnings 10 2 12.5 2 5 10 4 = 305 := by
  sorry

end NUMINAMATH_CALUDE_donnas_earnings_proof_l877_87708


namespace NUMINAMATH_CALUDE_boxes_per_case_l877_87773

/-- Given that Shirley sold 10 boxes of trefoils and needs to deliver 5 cases of boxes,
    prove that there are 2 boxes in each case. -/
theorem boxes_per_case (total_boxes : ℕ) (num_cases : ℕ) 
    (h1 : total_boxes = 10) (h2 : num_cases = 5) :
  total_boxes / num_cases = 2 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_case_l877_87773


namespace NUMINAMATH_CALUDE_root_sum_squared_plus_three_times_root_l877_87775

theorem root_sum_squared_plus_three_times_root : ∀ (α β : ℝ), 
  (α^2 + 2*α - 2025 = 0) → 
  (β^2 + 2*β - 2025 = 0) → 
  (α^2 + 3*α + β = 2023) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squared_plus_three_times_root_l877_87775


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l877_87707

theorem smallest_number_with_given_remainders : ∃ (x : ℕ), 
  x > 0 ∧
  x % 6 = 2 ∧ 
  x % 5 = 3 ∧ 
  x % 7 = 4 ∧
  ∀ (y : ℕ), y > 0 → y % 6 = 2 → y % 5 = 3 → y % 7 = 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l877_87707


namespace NUMINAMATH_CALUDE_factorization_example_l877_87763

/-- Represents a factorization from left to right -/
def is_factorization (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) (h : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, f a b = g a b * h a b

theorem factorization_example :
  is_factorization (λ a b => a^2*b + a*b^3) (λ a b => a*b) (λ a b => a + b^2) ∧
  ¬is_factorization (λ x _ => x^2 - 1) (λ x _ => x) (λ x _ => x - 1) ∧
  ¬is_factorization (λ x y => x^2 + 2*y + 1) (λ x y => x) (λ x y => x + 2*y) ∧
  ¬is_factorization (λ x y => x*(x+y)) (λ x _ => x^2) (λ _ y => y) :=
by sorry

end NUMINAMATH_CALUDE_factorization_example_l877_87763


namespace NUMINAMATH_CALUDE_circle_area_ratio_l877_87742

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (30 : ℝ) / 360 * (2 * Real.pi * r₁) = (24 : ℝ) / 360 * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l877_87742


namespace NUMINAMATH_CALUDE_eiffel_tower_height_is_324_l877_87757

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := 830

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters -/
def height_difference : ℝ := 506

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := burj_khalifa_height - height_difference

/-- Proves that the height of the Eiffel Tower is 324 meters -/
theorem eiffel_tower_height_is_324 : eiffel_tower_height = 324 := by
  sorry

end NUMINAMATH_CALUDE_eiffel_tower_height_is_324_l877_87757


namespace NUMINAMATH_CALUDE_expression_value_l877_87745

theorem expression_value (m n a b x : ℝ) : 
  (m = -n) → 
  (a * b = 1) → 
  (abs x = 3) → 
  (x^3 - (1 + m + n - a*b) * x^2010 + (m + n) * x^2007 + (-a*b)^2009 = 26 ∨
   x^3 - (1 + m + n - a*b) * x^2010 + (m + n) * x^2007 + (-a*b)^2009 = -28) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l877_87745


namespace NUMINAMATH_CALUDE_center_is_eight_l877_87737

-- Define the type for our 3x3 grid
def Grid := Fin 3 → Fin 3 → Nat

-- Define what it means for two positions to share an edge
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

-- Define the property of consecutive numbers sharing an edge
def consecutiveShareEdge (g : Grid) : Prop :=
  ∀ (i j : Fin 3 × Fin 3), 
    g i.1 i.2 + 1 = g j.1 j.2 → sharesEdge i j

-- Define the sum of corner numbers
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Define the theorem
theorem center_is_eight (g : Grid) 
  (all_numbers : ∀ n : Fin 9, ∃ (i j : Fin 3), g i j = n.val + 1)
  (consec_edge : consecutiveShareEdge g)
  (corner_sum_20 : cornerSum g = 20) :
  g 1 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_center_is_eight_l877_87737


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l877_87741

/-- Given a quadratic function f(x) = ax² + bx + b²/(2a) where a > 0,
    prove that the graph of f has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), a * x^2 + b * x + b^2 / (2 * a) ≥ a * x_min^2 + b * x_min + b^2 / (2 * a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l877_87741


namespace NUMINAMATH_CALUDE_shane_semester_distance_l877_87722

/-- Calculates the total distance traveled for round trips during a semester -/
def total_semester_distance (daily_one_way_distance : ℕ) (semester_days : ℕ) : ℕ :=
  2 * daily_one_way_distance * semester_days

/-- Proves that the total distance traveled during the semester is 1600 miles -/
theorem shane_semester_distance :
  total_semester_distance 10 80 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_shane_semester_distance_l877_87722


namespace NUMINAMATH_CALUDE_ratio_of_40_to_8_l877_87718

theorem ratio_of_40_to_8 (certain_number : ℚ) (h : certain_number = 40) : 
  certain_number / 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_40_to_8_l877_87718


namespace NUMINAMATH_CALUDE_sequence_inequality_l877_87711

theorem sequence_inequality (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, a n > 0) 
  (h2 : k > 0) (h3 : ∀ n, a (n + 1) ≤ (a n)^k * (1 - a n)) :
  ∀ n ≥ 2, (1 / a n) ≥ ((k + 1 : ℝ)^(k + 1) / k^k) + (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l877_87711


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l877_87740

/-- The probability of selecting 2 red balls from a bag containing 3 red, 2 blue, and 4 green balls -/
theorem probability_two_red_balls (red blue green : ℕ) 
  (h_red : red = 3) 
  (h_blue : blue = 2) 
  (h_green : green = 4) : 
  (red.choose 2 : ℚ) / ((red + blue + green).choose 2) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l877_87740


namespace NUMINAMATH_CALUDE_count_lambs_l877_87792

def farmer_cunningham_lambs : Nat → Nat → Prop
  | white_lambs, black_lambs =>
    ∀ (total_lambs : Nat),
      (white_lambs = 193) →
      (black_lambs = 5855) →
      (total_lambs = white_lambs + black_lambs) →
      (total_lambs = 6048)

theorem count_lambs :
  farmer_cunningham_lambs 193 5855 := by
  sorry

end NUMINAMATH_CALUDE_count_lambs_l877_87792


namespace NUMINAMATH_CALUDE_age_difference_l877_87735

/-- Given that the sum of X and Y is 12 years greater than the sum of Y and Z,
    prove that Z is 12 years younger than X. -/
theorem age_difference (X Y Z : ℕ) (h : X + Y = Y + Z + 12) : X - Z = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l877_87735


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_constant_term_binomial_expansion_proof_l877_87749

/-- The constant term in the binomial expansion of (2x - 1/√x)^6 is 60 -/
theorem constant_term_binomial_expansion : ℕ :=
  let n : ℕ := 6
  let a : ℝ → ℝ := λ x ↦ 2 * x
  let b : ℝ → ℝ := λ x ↦ -1 / Real.sqrt x
  let expansion : ℝ → ℝ := λ x ↦ (a x + b x) ^ n
  let constant_term : ℕ := 60
  constant_term

/-- Proof of the theorem -/
theorem constant_term_binomial_expansion_proof : 
  constant_term_binomial_expansion = 60 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_constant_term_binomial_expansion_proof_l877_87749


namespace NUMINAMATH_CALUDE_triangle_side_length_l877_87705

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  B = π / 3 ∧ 
  b = 6 ∧
  Real.sin A - 2 * Real.sin C = 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  a = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l877_87705


namespace NUMINAMATH_CALUDE_addSecondsCorrect_l877_87778

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define a function to add seconds to a time
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

-- Define the initial time
def initialTime : Time :=
  { hours := 7, minutes := 45, seconds := 0 }

-- Define the number of seconds to add
def secondsToAdd : Nat := 9999

-- Theorem to prove
theorem addSecondsCorrect : 
  addSeconds initialTime secondsToAdd = { hours := 10, minutes := 31, seconds := 39 } :=
sorry

end NUMINAMATH_CALUDE_addSecondsCorrect_l877_87778


namespace NUMINAMATH_CALUDE_volume_right_prism_isosceles_base_l877_87790

/-- Volume of a right prism with isosceles triangular base -/
theorem volume_right_prism_isosceles_base 
  (a : ℝ) (α : ℝ) (S : ℝ) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_S : S > 0) : 
  ∃ V : ℝ, V = (a * S / 2) * Real.sin (α / 2) * Real.tan ((π - α) / 4) ∧ 
  V = (Real.sin α * a^2 / 2) * (S / (2 * a * (1 + Real.sin (α / 2)))) :=
sorry

end NUMINAMATH_CALUDE_volume_right_prism_isosceles_base_l877_87790


namespace NUMINAMATH_CALUDE_max_elephants_l877_87769

/-- The number of union members --/
def unionMembers : ℕ := 28

/-- The number of non-members --/
def nonMembers : ℕ := 37

/-- The total number of attendees --/
def totalAttendees : ℕ := unionMembers + nonMembers

/-- A function to check if a distribution is valid --/
def isValidDistribution (elephants : ℕ) : Prop :=
  ∃ (unionElephants nonUnionElephants : ℕ),
    elephants = unionElephants + nonUnionElephants ∧
    unionElephants % unionMembers = 0 ∧
    nonUnionElephants % nonMembers = 0 ∧
    unionElephants / unionMembers ≥ 1 ∧
    nonUnionElephants / nonMembers ≥ 1

/-- The theorem stating the maximum number of elephants --/
theorem max_elephants :
  ∃! (maxElephants : ℕ),
    isValidDistribution maxElephants ∧
    ∀ (n : ℕ), n > maxElephants → ¬isValidDistribution n :=
by
  sorry

end NUMINAMATH_CALUDE_max_elephants_l877_87769


namespace NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l877_87726

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) (h_isosceles : a = c) 
  (h_sides : a = 5 ∧ c = 5) (h_base : b = 4) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (13125/1764) * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l877_87726


namespace NUMINAMATH_CALUDE_hyperbola_condition_l877_87710

/-- The equation represents a hyperbola with foci on the x-axis -/
def is_hyperbola_x_axis (k : ℝ) : Prop :=
  ∃ (x y : ℝ → ℝ), ∀ t : ℝ, (x t)^2 / (k + 3) + (y t)^2 / (k + 2) = 1

theorem hyperbola_condition (k : ℝ) :
  is_hyperbola_x_axis k ↔ -3 < k ∧ k < -2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l877_87710


namespace NUMINAMATH_CALUDE_natalies_height_l877_87772

/-- Prove that Natalie's height is 176 cm given the conditions -/
theorem natalies_height (h_natalie : ℝ) (h_harpreet : ℝ) (h_jiayin : ℝ) 
  (h_same_height : h_natalie = h_harpreet)
  (h_jiayin_height : h_jiayin = 161)
  (h_average : (h_natalie + h_harpreet + h_jiayin) / 3 = 171) :
  h_natalie = 176 := by
  sorry

end NUMINAMATH_CALUDE_natalies_height_l877_87772


namespace NUMINAMATH_CALUDE_no_square_possible_l877_87719

/-- Represents the lengths of sticks available -/
def stick_lengths : List ℕ := [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

/-- The total length of all sticks -/
def total_length : ℕ := stick_lengths.sum

/-- Predicate to check if a square can be formed -/
def can_form_square (lengths : List ℕ) : Prop :=
  ∃ (side_length : ℕ), side_length > 0 ∧ 
  4 * side_length = lengths.sum ∧
  ∃ (subset : List ℕ), subset.sum = side_length ∧ subset.toFinset ⊆ lengths.toFinset

theorem no_square_possible : ¬(can_form_square stick_lengths) := by
  sorry

end NUMINAMATH_CALUDE_no_square_possible_l877_87719


namespace NUMINAMATH_CALUDE_points_per_enemy_l877_87721

/-- 
Given a video game level with the following conditions:
- There are 11 enemies in total
- Defeating all but 3 enemies results in 72 points
This theorem proves that the number of points earned for defeating one enemy is 9.
-/
theorem points_per_enemy (total_enemies : ℕ) (remaining_enemies : ℕ) (total_points : ℕ) :
  total_enemies = 11 →
  remaining_enemies = 3 →
  total_points = 72 →
  (total_points / (total_enemies - remaining_enemies) : ℚ) = 9 := by
  sorry

#check points_per_enemy

end NUMINAMATH_CALUDE_points_per_enemy_l877_87721


namespace NUMINAMATH_CALUDE_completing_square_l877_87752

theorem completing_square (x : ℝ) : 
  (x^2 - 4*x - 3 = 0) ↔ ((x - 2)^2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l877_87752


namespace NUMINAMATH_CALUDE_product_complex_polar_form_l877_87750

/-- The product of two complex numbers in polar form results in another complex number in polar form -/
theorem product_complex_polar_form 
  (z₁ : ℂ) (z₂ : ℂ) (r₁ θ₁ r₂ θ₂ : ℝ) :
  z₁ = r₁ * Complex.exp (θ₁ * Complex.I) →
  z₂ = r₂ * Complex.exp (θ₂ * Complex.I) →
  r₁ = 4 →
  r₂ = 5 →
  θ₁ = 45 * π / 180 →
  θ₂ = 72 * π / 180 →
  ∃ (r θ : ℝ), 
    z₁ * z₂ = r * Complex.exp (θ * Complex.I) ∧
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * π ∧
    r = 20 ∧
    θ = 297 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_product_complex_polar_form_l877_87750


namespace NUMINAMATH_CALUDE_integer_solutions_eq1_integer_solutions_eq2_l877_87767

-- Equation 1
theorem integer_solutions_eq1 :
  ∀ x y : ℤ, 11 * x + 5 * y = 7 ↔ ∃ t : ℤ, x = 2 - 5 * t ∧ y = -3 + 11 * t :=
sorry

-- Equation 2
theorem integer_solutions_eq2 :
  ∀ x y : ℤ, 4 * x + y = 3 * x * y ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_eq1_integer_solutions_eq2_l877_87767


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l877_87716

-- Define the relationship between x and y
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y * x^2 = k

-- State the theorem
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : inverse_relation x₁ y₁)
  (h₂ : x₁ = 3)
  (h₃ : y₁ = 2)
  (h₄ : y₂ = 18)
  (h₅ : inverse_relation x₂ y₂) :
  x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l877_87716


namespace NUMINAMATH_CALUDE_unique_number_with_property_l877_87793

/-- Calculate the total number of digits needed to write all integers from 1 to n -/
def totalDigits (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

/-- The property that the number, when doubled, equals the total number of digits -/
def hasProperty (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ 2 * x = totalDigits x

theorem unique_number_with_property :
  ∃! x : ℕ, hasProperty x ∧ x = 108 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_property_l877_87793


namespace NUMINAMATH_CALUDE_problem_solution_l877_87725

theorem problem_solution : 
  |1 - Real.sqrt (4/3)| + (Real.sqrt 3 - 1/2)^0 = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l877_87725


namespace NUMINAMATH_CALUDE_encyclopedia_chapters_l877_87709

theorem encyclopedia_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h1 : total_pages = 3962) (h2 : pages_per_chapter = 566) :
  total_pages / pages_per_chapter = 7 := by
sorry

end NUMINAMATH_CALUDE_encyclopedia_chapters_l877_87709


namespace NUMINAMATH_CALUDE_fraction_is_one_ninth_l877_87776

/-- Represents a taxi trip with given parameters -/
structure TaxiTrip where
  initialFee : ℚ
  additionalChargePerFraction : ℚ
  totalDistance : ℚ
  totalCharge : ℚ

/-- Calculates the fraction of a mile for which the additional charge applies -/
def fractionOfMile (trip : TaxiTrip) : ℚ :=
  let additionalCharge := trip.totalCharge - trip.initialFee
  let numberOfFractions := additionalCharge / trip.additionalChargePerFraction
  trip.totalDistance / numberOfFractions

/-- Theorem stating that for the given trip parameters, the fraction of a mile
    for which the additional charge applies is 1/9 -/
theorem fraction_is_one_ninth :
  let trip := TaxiTrip.mk 2.25 0.15 3.6 3.60
  fractionOfMile trip = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_one_ninth_l877_87776


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l877_87713

/-- The function f(x) = 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x + m

/-- m is not in the open interval (-3, -1) -/
def not_in_interval (m : ℝ) : Prop := m ≤ -3 ∨ m ≥ -1

/-- f has no zero in the interval [0, 1] -/
def no_zero_in_interval (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, f m x ≠ 0

theorem necessary_not_sufficient :
  (∀ m : ℝ, no_zero_in_interval m → not_in_interval m) ∧
  (∃ m : ℝ, not_in_interval m ∧ ¬(no_zero_in_interval m)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l877_87713


namespace NUMINAMATH_CALUDE_product_plus_one_composite_l877_87786

theorem product_plus_one_composite : 
  ∃ (a b : ℤ), b > 1 ∧ 2014 * 2015 * 2016 * 2017 + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_composite_l877_87786


namespace NUMINAMATH_CALUDE_sum_of_numbers_l877_87796

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) 
  (h4 : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l877_87796


namespace NUMINAMATH_CALUDE_flight_cost_A_to_C_via_B_l877_87760

/-- Represents the cost of a flight with a given distance and number of stops -/
def flight_cost (distance : ℝ) (stops : ℕ) : ℝ :=
  120 + 0.15 * distance + 50 * stops

/-- The cities A, B, and C form a right-angled triangle -/
axiom right_triangle : ∃ (AB BC AC : ℝ), AB^2 + BC^2 = AC^2

/-- The distance between A and C is 2000 km -/
axiom AC_distance : ∃ AC : ℝ, AC = 2000

/-- The distance between A and B is 4000 km -/
axiom AB_distance : ∃ AB : ℝ, AB = 4000

/-- Theorem: The cost to fly from A to C with one stop at B is $1289.62 -/
theorem flight_cost_A_to_C_via_B : 
  ∃ (AB BC AC : ℝ), 
    AB^2 + BC^2 = AC^2 ∧ 
    AC = 2000 ∧ 
    AB = 4000 ∧ 
    flight_cost (AB + BC) 1 = 1289.62 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_A_to_C_via_B_l877_87760


namespace NUMINAMATH_CALUDE_prob_rain_all_days_l877_87714

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.2

theorem prob_rain_all_days :
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_all_days_l877_87714


namespace NUMINAMATH_CALUDE_theater_ticket_price_l877_87703

/-- Calculates the ticket price for a theater performance --/
theorem theater_ticket_price
  (capacity : ℕ)
  (fill_rate : ℚ)
  (num_performances : ℕ)
  (total_earnings : ℕ)
  (h1 : capacity = 400)
  (h2 : fill_rate = 4/5)
  (h3 : num_performances = 3)
  (h4 : total_earnings = 28800) :
  (total_earnings : ℚ) / ((capacity : ℚ) * fill_rate * num_performances) = 30 :=
by
  sorry

#check theater_ticket_price

end NUMINAMATH_CALUDE_theater_ticket_price_l877_87703


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l877_87766

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (2 + a) * x + 2 * a > 0}
  (a < 2 → S = {x : ℝ | x < a ∨ x > 2}) ∧
  (a = 2 → S = {x : ℝ | x ≠ 2}) ∧
  (a > 2 → S = {x : ℝ | x > a ∨ x < 2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l877_87766


namespace NUMINAMATH_CALUDE_ball_probability_comparison_l877_87751

theorem ball_probability_comparison :
  let total_balls : ℕ := 3
  let red_balls : ℕ := 2
  let white_balls : ℕ := 1
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  p_red > p_white :=
by
  sorry

end NUMINAMATH_CALUDE_ball_probability_comparison_l877_87751


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l877_87758

/-- The diameter of a circle with area 64π cm² is 16 cm. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 64 * π → 2 * r = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l877_87758


namespace NUMINAMATH_CALUDE_perpendicular_lines_line_through_P_l877_87798

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := 
  (m = 0) ∨ (m = -3)

-- Define point P on l₂
def P_on_l₂ (m : ℝ) : Prop := l₂ m 1 (2 * m)

-- Define line l passing through P with opposite intercepts
def line_l (x y : ℝ) : Prop := 
  (2 * x - y = 0) ∨ (x - y + 1 = 0)

-- Theorem statements
theorem perpendicular_lines (m : ℝ) : 
  (∀ x y, l₁ m x y ∧ l₂ m x y → perpendicular m) := sorry

theorem line_through_P (m : ℝ) : 
  P_on_l₂ m → (∀ x y, line_l x y) := sorry

end NUMINAMATH_CALUDE_perpendicular_lines_line_through_P_l877_87798


namespace NUMINAMATH_CALUDE_odd_product_probability_l877_87753

theorem odd_product_probability (n : ℕ) (hn : n = 1000) :
  let odd_count := (n + 1) / 2
  let total_count := n
  let p := (odd_count / total_count) * ((odd_count - 1) / (total_count - 1)) * ((odd_count - 2) / (total_count - 2))
  p < 1 / 8 := by
sorry


end NUMINAMATH_CALUDE_odd_product_probability_l877_87753


namespace NUMINAMATH_CALUDE_expression_simplification_l877_87759

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - m / (m + 3)) / ((m^2 - 9) / (m^2 + 6*m + 9)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l877_87759


namespace NUMINAMATH_CALUDE_bill_insurance_cost_l877_87770

def monthly_plan_price : ℚ := 500
def hourly_rate : ℚ := 25
def weekly_hours : ℚ := 30
def weeks_per_month : ℚ := 4
def months_per_year : ℚ := 12

def annual_income (rate : ℚ) (hours : ℚ) (weeks : ℚ) (months : ℚ) : ℚ :=
  rate * hours * weeks * months

def subsidy_rate (income : ℚ) : ℚ :=
  if income < 10000 then 0.9
  else if income ≤ 40000 then 0.5
  else 0.2

def annual_insurance_cost (plan_price : ℚ) (income : ℚ) (months : ℚ) : ℚ :=
  plan_price * (1 - subsidy_rate income) * months

theorem bill_insurance_cost :
  let income := annual_income hourly_rate weekly_hours weeks_per_month months_per_year
  annual_insurance_cost monthly_plan_price income months_per_year = 3000 :=
by sorry

end NUMINAMATH_CALUDE_bill_insurance_cost_l877_87770


namespace NUMINAMATH_CALUDE_opposite_face_is_D_l877_87782

/-- Represents the labels of the faces of a cube --/
inductive FaceLabel
  | A | B | C | D | E | F

/-- Represents the positions of faces on a cube --/
inductive Position
  | Top | Bottom | Left | Right | Front | Back

/-- Represents a cube with labeled faces --/
structure Cube where
  faces : Position → FaceLabel

/-- Defines the opposite position for each position on the cube --/
def oppositePosition : Position → Position
  | Position.Top => Position.Bottom
  | Position.Bottom => Position.Top
  | Position.Left => Position.Right
  | Position.Right => Position.Left
  | Position.Front => Position.Back
  | Position.Back => Position.Front

/-- Theorem stating that in a cube where C is on top and B is to its right, 
    the face opposite to A is labeled D --/
theorem opposite_face_is_D (cube : Cube) 
  (h1 : cube.faces Position.Top = FaceLabel.C) 
  (h2 : cube.faces Position.Right = FaceLabel.B) : 
  ∃ p : Position, cube.faces p = FaceLabel.A ∧ 
  cube.faces (oppositePosition p) = FaceLabel.D := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_D_l877_87782


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l877_87755

theorem sum_of_squares_theorem (a d : ℤ) : 
  ∃ (x y z w : ℤ), 
    a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (x*a + y*d)^2 + (z*a + w*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l877_87755


namespace NUMINAMATH_CALUDE_alexey_dowel_cost_l877_87788

theorem alexey_dowel_cost (screw_cost dowel_cost : ℚ) : 
  screw_cost = 7 →
  (0.85 * (screw_cost + dowel_cost) = screw_cost + 0.5 * dowel_cost) →
  dowel_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_alexey_dowel_cost_l877_87788


namespace NUMINAMATH_CALUDE_closest_point_sum_l877_87799

/-- The point (a, b) on the line y = -3x + 10 that is closest to (16, 8) satisfies a + b = 8.8 -/
theorem closest_point_sum (a b : ℝ) : 
  (b = -3 * a + 10) →  -- Mouse path equation
  (∀ x y : ℝ, y = -3 * x + 10 → (x - 16)^2 + (y - 8)^2 ≥ (a - 16)^2 + (b - 8)^2) →  -- (a, b) is closest to (16, 8)
  a + b = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_sum_l877_87799
