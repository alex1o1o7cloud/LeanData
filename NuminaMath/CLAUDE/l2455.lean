import Mathlib

namespace NUMINAMATH_CALUDE_river_length_problem_l2455_245585

theorem river_length_problem (straight_length crooked_length total_length : ℝ) :
  straight_length * 3 = crooked_length →
  straight_length + crooked_length = total_length →
  total_length = 80 →
  straight_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_river_length_problem_l2455_245585


namespace NUMINAMATH_CALUDE_min_value_quadratic_root_condition_l2455_245541

/-- Given a quadratic equation x^2 + ax + b - 3 = 0 with a real root in [1,2],
    the minimum value of a^2 + (b-4)^2 is 2 -/
theorem min_value_quadratic_root_condition (a b : ℝ) :
  (∃ x : ℝ, x^2 + a*x + b - 3 = 0 ∧ 1 ≤ x ∧ x ≤ 2) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x^2 + a'*x + b' - 3 = 0 ∧ 1 ≤ x ∧ x ≤ 2) →
    a^2 + (b-4)^2 ≤ a'^2 + (b'-4)^2) →
  a^2 + (b-4)^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_min_value_quadratic_root_condition_l2455_245541


namespace NUMINAMATH_CALUDE_maxim_birth_probability_l2455_245591

/-- The year Maxim starts school -/
def school_start_year : ℕ := 2014

/-- The month Maxim starts school (1-based) -/
def school_start_month : ℕ := 9

/-- The day Maxim starts school -/
def school_start_day : ℕ := 1

/-- Maxim's age when starting school -/
def school_start_age : ℕ := 6

/-- Whether the school start date is Maxim's birthday -/
def is_birthday : Prop := False

/-- The number of days from Jan 1, 2008 to Aug 31, 2008 inclusive -/
def days_in_2008 : ℕ := 244

/-- The total number of possible birth dates -/
def total_possible_days : ℕ := 365

/-- The probability that Maxim was born in 2008 -/
def prob_born_2008 : ℚ := days_in_2008 / total_possible_days

theorem maxim_birth_probability : 
  prob_born_2008 = 244 / 365 := by sorry

end NUMINAMATH_CALUDE_maxim_birth_probability_l2455_245591


namespace NUMINAMATH_CALUDE_fermat_little_theorem_general_l2455_245571

theorem fermat_little_theorem_general (p : ℕ) (m : ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, m^p - m = k * p :=
sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_general_l2455_245571


namespace NUMINAMATH_CALUDE_system_solution_existence_l2455_245533

theorem system_solution_existence (a : ℝ) : 
  (∃ b x y : ℝ, x^2 + y^2 + 2*a*(a + y - x) = 49 ∧ 
                y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)) ↔ 
  -24 ≤ a ∧ a ≤ 24 := by
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2455_245533


namespace NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l2455_245566

/-- The amount of cinnamon used in tablespoons -/
def cinnamon : ℚ := 0.6666666666666666

/-- The amount of nutmeg used in tablespoons -/
def nutmeg : ℚ := 0.5

/-- The difference between cinnamon and nutmeg amounts -/
def difference : ℚ := cinnamon - nutmeg

theorem cinnamon_nutmeg_difference : difference = 0.1666666666666666 := by sorry

end NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l2455_245566


namespace NUMINAMATH_CALUDE_particle_motion_l2455_245556

/-- A particle moves under the influence of gravity and an additional constant acceleration. -/
theorem particle_motion
  (V₀ g a t V S : ℝ)
  (hV : V = g * t + a * t + V₀)
  (hS : S = (1/2) * (g + a) * t^2 + V₀ * t) :
  t = (2 * S) / (V + V₀) :=
sorry

end NUMINAMATH_CALUDE_particle_motion_l2455_245556


namespace NUMINAMATH_CALUDE_corn_acreage_l2455_245589

theorem corn_acreage (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ) 
  (h1 : total_land = 1034)
  (h2 : ratio_beans = 5)
  (h3 : ratio_wheat = 2)
  (h4 : ratio_corn = 4) : 
  (total_land * ratio_corn) / (ratio_beans + ratio_wheat + ratio_corn) = 376 := by
  sorry

#eval (1034 * 4) / (5 + 2 + 4)

end NUMINAMATH_CALUDE_corn_acreage_l2455_245589


namespace NUMINAMATH_CALUDE_angle_relationship_l2455_245503

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α) 
  (h2 : α < 2 * β) 
  (h3 : 2 * β ≤ π / 2) 
  (h4 : 2 * Real.cos (α + β) * Real.cos β = -1 + 2 * Real.sin (α + β) * Real.sin β) : 
  α + 2 * β = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l2455_245503


namespace NUMINAMATH_CALUDE_simplify_product_of_roots_l2455_245587

theorem simplify_product_of_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 15 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_roots_l2455_245587


namespace NUMINAMATH_CALUDE_log_three_seven_l2455_245531

theorem log_three_seven (a b : ℝ) (h1 : Real.log 2 / Real.log 3 = a) (h2 : Real.log 7 / Real.log 2 = b) :
  Real.log 7 / Real.log 3 = a * b := by
  sorry

end NUMINAMATH_CALUDE_log_three_seven_l2455_245531


namespace NUMINAMATH_CALUDE_max_a_min_b_for_sin_inequality_l2455_245524

theorem max_a_min_b_for_sin_inequality 
  (h : ∀ x ∈ Set.Ioo 0 (π/2), a * x < Real.sin x ∧ Real.sin x < b * x) :
  (∀ a' : ℝ, (∀ x ∈ Set.Ioo 0 (π/2), a' * x < Real.sin x) → a' ≤ 2/π) ∧
  (∀ b' : ℝ, (∀ x ∈ Set.Ioo 0 (π/2), Real.sin x < b' * x) → b' ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_max_a_min_b_for_sin_inequality_l2455_245524


namespace NUMINAMATH_CALUDE_system_consistent_iff_k_equals_four_l2455_245546

theorem system_consistent_iff_k_equals_four 
  (x y u : ℝ) (k : ℝ) : 
  (x + y = 1 ∧ k * x + y = 2 ∧ x + k * u = 3) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_consistent_iff_k_equals_four_l2455_245546


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2455_245539

/-- Given a geometric sequence with first term a₁ and common ratio r,
    the nth term is given by aₙ = a₁ * r^(n-1) -/
def geometric_sequence (a₁ : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^(n-1)

/-- The 12th term of a geometric sequence with first term 5 and common ratio -3 is -885735 -/
theorem twelfth_term_of_specific_sequence :
  geometric_sequence 5 (-3) 12 = -885735 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2455_245539


namespace NUMINAMATH_CALUDE_time_with_family_l2455_245532

/-- Given a 24-hour day, if a person spends 1/3 of the day sleeping, 
    1/6 of the day in school, 1/12 of the day making assignments, 
    then the remaining time spent with family is 10 hours. -/
theorem time_with_family (total_hours : ℝ) 
  (sleep_fraction : ℝ) (school_fraction : ℝ) (assignment_fraction : ℝ) :
  total_hours = 24 →
  sleep_fraction = 1/3 →
  school_fraction = 1/6 →
  assignment_fraction = 1/12 →
  total_hours - (sleep_fraction + school_fraction + assignment_fraction) * total_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_time_with_family_l2455_245532


namespace NUMINAMATH_CALUDE_olivia_money_distribution_l2455_245551

/-- Prove that Olivia needs to give 2 euros to each sister for equal distribution -/
theorem olivia_money_distribution (olivia_initial : ℕ) (sister_initial : ℕ) (num_sisters : ℕ) 
  (olivia_gives : ℕ) :
  olivia_initial = 20 →
  sister_initial = 10 →
  num_sisters = 4 →
  olivia_gives = 2 →
  (olivia_initial - num_sisters * olivia_gives = 
   sister_initial + olivia_gives) ∧
  (olivia_initial + num_sisters * sister_initial = 
   (num_sisters + 1) * (sister_initial + olivia_gives)) :=
by sorry

end NUMINAMATH_CALUDE_olivia_money_distribution_l2455_245551


namespace NUMINAMATH_CALUDE_bruce_bank_savings_l2455_245509

/-- The amount of money Bruce puts in the bank -/
def money_in_bank (aunt_money grandfather_money : ℕ) : ℚ :=
  (aunt_money + grandfather_money : ℚ) / 5

/-- Theorem stating the amount Bruce put in the bank -/
theorem bruce_bank_savings :
  money_in_bank 75 150 = 45 := by sorry

end NUMINAMATH_CALUDE_bruce_bank_savings_l2455_245509


namespace NUMINAMATH_CALUDE_interest_calculation_l2455_245501

/-- Calculates the simple interest and final amount given initial principal, annual rate, and time in years -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ × ℝ :=
  let interest := principal * rate * time
  let final_amount := principal + interest
  (interest, final_amount)

theorem interest_calculation (P : ℝ) :
  let (interest, final_amount) := simple_interest P 0.06 0.25
  final_amount = 510.60 → interest = 7.54 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l2455_245501


namespace NUMINAMATH_CALUDE_compound_interest_rate_calculation_l2455_245553

theorem compound_interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (CI : ℝ)  -- Compound Interest
  (t : ℝ)  -- Time in years
  (n : ℝ)  -- Number of times interest is compounded per year
  (h1 : P = 8000)
  (h2 : CI = 484.76847061839544)
  (h3 : t = 1.5)
  (h4 : n = 2)
  : ∃ (r : ℝ), abs (r - 0.0397350993377484) < 0.0000000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_calculation_l2455_245553


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2455_245579

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 8 = 6) →
  (a 3 * a 8 = 5) →
  a 5 + a 6 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2455_245579


namespace NUMINAMATH_CALUDE_smallest_n_value_l2455_245594

-- Define the cost of purple candy
def purple_cost : ℕ := 20

-- Define the quantities of other candies
def red_quantity : ℕ := 12
def green_quantity : ℕ := 14
def blue_quantity : ℕ := 15

-- Define the theorem
theorem smallest_n_value :
  ∃ (n : ℕ), n > 0 ∧ 
  (purple_cost * n) % red_quantity = 0 ∧
  (purple_cost * n) % green_quantity = 0 ∧
  (purple_cost * n) % blue_quantity = 0 ∧
  (∀ (m : ℕ), m > 0 → 
    (purple_cost * m) % red_quantity = 0 →
    (purple_cost * m) % green_quantity = 0 →
    (purple_cost * m) % blue_quantity = 0 →
    m ≥ n) ∧
  n = 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2455_245594


namespace NUMINAMATH_CALUDE_z_coord_for_specific_line_l2455_245559

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The z-coordinate of a point on the line when its y-coordinate is given -/
def z_coord_at_y (l : Line3D) (y : ℝ) : ℝ :=
  sorry

theorem z_coord_for_specific_line :
  let l : Line3D := { point1 := (3, 3, 2), point2 := (6, 2, -1) }
  z_coord_at_y l 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_z_coord_for_specific_line_l2455_245559


namespace NUMINAMATH_CALUDE_range_of_a_l2455_245596

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 4*x + a = 0

theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) :
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2455_245596


namespace NUMINAMATH_CALUDE_jakes_total_earnings_l2455_245586

/-- Calculates Jake's total earnings from selling baby snakes --/
def jakes_earnings (viper_count cobra_count python_count anaconda_count : ℕ)
  (viper_eggs cobra_eggs python_eggs anaconda_eggs : ℕ)
  (viper_price cobra_price python_price anaconda_price : ℚ)
  (viper_discount cobra_discount python_discount anaconda_discount : ℚ) : ℚ :=
  let viper_total := viper_count * viper_eggs * (viper_price * (1 - viper_discount))
  let cobra_total := cobra_count * cobra_eggs * (cobra_price * (1 - cobra_discount))
  let python_total := python_count * python_eggs * (python_price * (1 - python_discount))
  let anaconda_total := anaconda_count * anaconda_eggs * (anaconda_price * (1 - anaconda_discount))
  viper_total + cobra_total + python_total + anaconda_total

/-- Theorem stating Jake's total earnings --/
theorem jakes_total_earnings :
  jakes_earnings 3 2 1 1 3 2 4 5 300 250 450 500 (10/100) (5/100) (75/1000) (12/100) = 7245 := by
  sorry

end NUMINAMATH_CALUDE_jakes_total_earnings_l2455_245586


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l2455_245573

/-- Given a line y = x + m intersecting the ellipse 4x^2 + y^2 = 1 and forming a chord of length 2√2/5, prove that m = ± √5/2 -/
theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    4 * x₁^2 + (x₁ + m)^2 = 1 ∧ 
    4 * x₂^2 + (x₂ + m)^2 = 1 ∧ 
    (x₂ - x₁)^2 + ((x₂ + m) - (x₁ + m))^2 = (2 * Real.sqrt 2 / 5)^2) → 
  m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2 :=
by sorry


end NUMINAMATH_CALUDE_line_ellipse_intersection_l2455_245573


namespace NUMINAMATH_CALUDE_negative_one_powers_equality_l2455_245545

theorem negative_one_powers_equality : -1^2022 - (-1)^2023 - (-1)^0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_powers_equality_l2455_245545


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l2455_245513

theorem ellipse_focus_distance (x y : ℝ) :
  x^2 / 25 + y^2 / 16 = 1 →
  ∃ (f1 f2 : ℝ × ℝ), 
    (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ 
      Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) = 3) →
    Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l2455_245513


namespace NUMINAMATH_CALUDE_martha_cookies_theorem_l2455_245561

/-- Given that Martha can make 24 cookies with 3 cups of flour, this function
    calculates how many cookies she can make with a given number of cups. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Given that Martha can make 24 cookies with 3 cups of flour, this function
    calculates how many cups of flour are needed to make a given number of cookies. -/
def flour_for_cookies (cookies : ℚ) : ℚ :=
  (3 / 24) * cookies

/-- Theorem stating that with 5 cups of flour, Martha can make 40 cookies,
    and 60 cookies require 7.5 cups of flour. -/
theorem martha_cookies_theorem :
  cookies_from_flour 5 = 40 ∧ flour_for_cookies 60 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_martha_cookies_theorem_l2455_245561


namespace NUMINAMATH_CALUDE_exponent_simplification_l2455_245538

theorem exponent_simplification :
  3^12 * 8^12 * 3^3 * 8^8 = 24^15 * 32768 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2455_245538


namespace NUMINAMATH_CALUDE_production_theorem_l2455_245554

/-- Represents the production scenario -/
structure ProductionScenario where
  women : ℕ
  hours_per_day : ℕ
  days : ℕ
  units_produced : ℚ

/-- The production function that calculates the units produced given a scenario -/
def production_function (x : ProductionScenario) (z : ProductionScenario) : ℚ :=
  (z.women * z.hours_per_day * z.days : ℚ) * x.units_produced / (x.women * x.hours_per_day * x.days : ℚ)

theorem production_theorem (x z : ProductionScenario) 
  (h : x.women = x.hours_per_day ∧ x.hours_per_day = x.days ∧ x.units_produced = x.women ^ 2) :
  production_function x z = (z.women * z.hours_per_day * z.days : ℚ) / x.women := by
  sorry

#check production_theorem

end NUMINAMATH_CALUDE_production_theorem_l2455_245554


namespace NUMINAMATH_CALUDE_part_one_part_two_l2455_245526

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x < 4 - 2*a ↔ -4 < x ∧ x < 4) → a = 0 := 
sorry

-- Part 2
theorem part_two : 
  (∀ m : ℝ, (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + m) ↔ 2 ≤ m) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2455_245526


namespace NUMINAMATH_CALUDE_missing_number_proof_l2455_245537

theorem missing_number_proof : ∃ x : ℚ, (306 / 34) * 15 + x = 405 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2455_245537


namespace NUMINAMATH_CALUDE_green_beans_count_l2455_245582

theorem green_beans_count (total : ℕ) (red_fraction : ℚ) (white_fraction : ℚ) (green_fraction : ℚ) : 
  total = 572 →
  red_fraction = 1/4 →
  white_fraction = 1/3 →
  green_fraction = 1/2 →
  ∃ (red white green : ℕ),
    red = total * red_fraction ∧
    white = (total - red) * white_fraction ∧
    green = (total - red - white) * green_fraction ∧
    green = 143 :=
by sorry

end NUMINAMATH_CALUDE_green_beans_count_l2455_245582


namespace NUMINAMATH_CALUDE_roots_of_unity_real_fifth_power_l2455_245515

theorem roots_of_unity_real_fifth_power :
  ∃ (S : Finset ℂ), 
    (S.card = 30) ∧ 
    (∀ z ∈ S, z^30 = 1) ∧
    (∃ (T : Finset ℂ), 
      (T ⊆ S) ∧ 
      (T.card = 10) ∧ 
      (∀ z ∈ T, ∃ (r : ℝ), z^5 = r) ∧
      (∀ z ∈ S \ T, ¬∃ (r : ℝ), z^5 = r)) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_unity_real_fifth_power_l2455_245515


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l2455_245580

/-- An arithmetic progression with first term 5 and 25th term 173 has common difference 7. -/
theorem arithmetic_progression_common_difference : 
  ∀ (a : ℕ → ℝ), 
    (a 1 = 5) → 
    (a 25 = 173) → 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → 
    (a 2 - a 1 = 7) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l2455_245580


namespace NUMINAMATH_CALUDE_fraction_simplification_l2455_245536

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 * x^2 - x + 1) / (x^2 - 1) - x / (x - 1) = (x - 1) / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2455_245536


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2455_245575

theorem simple_interest_problem (simple_interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  simple_interest = 4016.25 →
  rate = 0.01 →
  time = 3 →
  principal = simple_interest / (rate * time) →
  principal = 133875 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2455_245575


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l2455_245520

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l2455_245520


namespace NUMINAMATH_CALUDE_max_integer_solution_inequality_system_negative_six_satisfies_system_max_integer_solution_is_negative_six_l2455_245523

theorem max_integer_solution_inequality_system :
  ∀ x : ℤ, (x + 5 < 0 ∧ (3 * x - 1) / 2 ≥ 2 * x + 1) → x ≤ -6 :=
by
  sorry

theorem negative_six_satisfies_system :
  -6 + 5 < 0 ∧ (3 * (-6) - 1) / 2 ≥ 2 * (-6) + 1 :=
by
  sorry

theorem max_integer_solution_is_negative_six :
  ∃ x : ℤ, x + 5 < 0 ∧ (3 * x - 1) / 2 ≥ 2 * x + 1 ∧
  ∀ y : ℤ, (y + 5 < 0 ∧ (3 * y - 1) / 2 ≥ 2 * y + 1) → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_max_integer_solution_inequality_system_negative_six_satisfies_system_max_integer_solution_is_negative_six_l2455_245523


namespace NUMINAMATH_CALUDE_restaurant_gratuity_l2455_245535

/-- Calculate the gratuity for a restaurant bill -/
theorem restaurant_gratuity (price1 price2 price3 : ℕ) (tip_percentage : ℚ) : 
  price1 = 10 → price2 = 13 → price3 = 17 → tip_percentage = 1/10 →
  (price1 + price2 + price3 : ℚ) * tip_percentage = 4 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_gratuity_l2455_245535


namespace NUMINAMATH_CALUDE_fraction_to_longest_side_is_five_twelfths_l2455_245570

/-- Represents a trapezoid field with corn -/
structure CornField where
  -- Side lengths in clockwise order from a 60° angle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Angles at the non-parallel sides
  angle1 : ℝ
  angle2 : ℝ
  -- Conditions
  side1_eq : side1 = 150
  side2_eq : side2 = 150
  side3_eq : side3 = 200
  side4_eq : side4 = 200
  angle1_eq : angle1 = 60
  angle2_eq : angle2 = 120
  is_trapezoid : angle1 + angle2 = 180

/-- The fraction of the crop brought to the longest side -/
def fractionToLongestSide (field : CornField) : ℚ :=
  5/12

/-- Theorem stating that the fraction of the crop brought to the longest side is 5/12 -/
theorem fraction_to_longest_side_is_five_twelfths (field : CornField) :
  fractionToLongestSide field = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_longest_side_is_five_twelfths_l2455_245570


namespace NUMINAMATH_CALUDE_unique_number_l2455_245578

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ),
    n = 100 * x + 10 * y + z ∧
    x ≥ 1 ∧ x ≤ 9 ∧
    y ≥ 0 ∧ y ≤ 9 ∧
    z ≥ 0 ∧ z ≤ 9 ∧
    100 * z + 10 * y + x = n + 198 ∧
    100 * x + 10 * z + y = n + 9 ∧
    x^2 + y^2 + z^2 = 4 * (x + y + z) + 2

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 345 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l2455_245578


namespace NUMINAMATH_CALUDE_kanul_cash_theorem_l2455_245540

/-- The total amount of cash Kanul had -/
def total_cash : ℝ := 5714.29

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery -/
def machinery : ℝ := 1000

/-- The percentage of total cash spent -/
def percentage_spent : ℝ := 0.30

theorem kanul_cash_theorem :
  total_cash = raw_materials + machinery + percentage_spent * total_cash := by
  sorry

end NUMINAMATH_CALUDE_kanul_cash_theorem_l2455_245540


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l2455_245502

/-- Calculates the share of profit for an investor based on their investment, time period, and total profit --/
def calculate_share_of_profit (investment : ℕ) (months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * months * total_profit) / total_investment_months

theorem jose_share_of_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (total_profit : ℕ)
  (h1 : tom_investment = 30000)
  (h2 : tom_months = 12)
  (h3 : jose_investment = 45000)
  (h4 : jose_months = 10)
  (h5 : total_profit = 63000) :
  calculate_share_of_profit jose_investment jose_months (tom_investment * tom_months + jose_investment * jose_months) total_profit = 35000 :=
by sorry

end NUMINAMATH_CALUDE_jose_share_of_profit_l2455_245502


namespace NUMINAMATH_CALUDE_blue_chairs_count_l2455_245549

/-- Represents the number of chairs of each color in a classroom --/
structure Classroom where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Defines the conditions for the classroom chair problem --/
def validClassroom (c : Classroom) : Prop :=
  c.green = 3 * c.blue ∧
  c.white = c.blue + c.green - 13 ∧
  c.blue + c.green + c.white = 67

/-- Theorem stating that in a valid classroom, there are 10 blue chairs --/
theorem blue_chairs_count (c : Classroom) (h : validClassroom c) : c.blue = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_chairs_count_l2455_245549


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2455_245525

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 / (1 + Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2455_245525


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l2455_245567

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

-- State the theorem
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x < 4 ∨ x ≥ 10} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l2455_245567


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2455_245569

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 9*a + 9 = 0) → (b^2 - 9*b + 9 = 0) → a^2 + b^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2455_245569


namespace NUMINAMATH_CALUDE_weight_removed_l2455_245527

/-- Given weights of sugar and salt bags, and their combined weight after removal,
    prove the amount of weight removed. -/
theorem weight_removed (sugar_weight salt_weight new_combined_weight : ℕ)
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : new_combined_weight = 42) :
  sugar_weight + salt_weight - new_combined_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_weight_removed_l2455_245527


namespace NUMINAMATH_CALUDE_inverse_of_A_zero_matrix_if_not_invertible_l2455_245505

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 7; 2, 6]

theorem inverse_of_A :
  let inv_A := !![0.6, -0.7; -0.2, 0.4]
  A.det ≠ 0 → A * inv_A = 1 ∧ inv_A * A = 1 :=
by sorry

theorem zero_matrix_if_not_invertible :
  A.det = 0 → A⁻¹ = 0 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_A_zero_matrix_if_not_invertible_l2455_245505


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2455_245504

theorem circle_area_ratio (r R : ℝ) (h : r = R / 3) :
  (π * r^2) / (π * R^2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2455_245504


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l2455_245565

/-- The number of nonzero terms in the expansion of (x+4)(3x^3 + 2x^2 + 3x + 9) - 4(x^4 - 3x^3 + 2x^2 + 7x) -/
theorem nonzero_terms_count (x : ℝ) : 
  let expansion := (x + 4) * (3*x^3 + 2*x^2 + 3*x + 9) - 4*(x^4 - 3*x^3 + 2*x^2 + 7*x)
  ∃ (a b c d e : ℝ), 
    expansion = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l2455_245565


namespace NUMINAMATH_CALUDE_solve_equations_l2455_245516

theorem solve_equations :
  (∃ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 ∧ x = 8) ∧
  (∃ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_l2455_245516


namespace NUMINAMATH_CALUDE_double_root_k_l2455_245522

/-- A cubic equation with a double root -/
def has_double_root (k : ℝ) : Prop :=
  ∃ (r s : ℝ), (∀ x, x^3 + k*x - 128 = (x - r)^2 * (x - s))

/-- The value of k for which x^3 + kx - 128 = 0 has a double root -/
theorem double_root_k : ∃ k : ℝ, has_double_root k ∧ k = -48 := by
  sorry

end NUMINAMATH_CALUDE_double_root_k_l2455_245522


namespace NUMINAMATH_CALUDE_intersection_distance_proof_l2455_245568

/-- The distance between intersection points of y = 5 and y = 3x^2 + 2x - 2 -/
def intersection_distance : ℝ := sorry

/-- The equation y = 3x^2 + 2x - 2 -/
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem intersection_distance_proof :
  let p : ℕ := 88
  let q : ℕ := 3
  (∃ (x₁ x₂ : ℝ), 
    parabola x₁ = 5 ∧ 
    parabola x₂ = 5 ∧ 
    x₁ ≠ x₂ ∧
    intersection_distance = |x₁ - x₂|) ∧
  intersection_distance = Real.sqrt p / q ∧
  p - q^2 = 79 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_proof_l2455_245568


namespace NUMINAMATH_CALUDE_sarah_candy_duration_l2455_245514

/-- The number of days Sarah's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (friends_candy : ℕ) 
  (traded_candy : ℕ) (given_away_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  let total_received := neighbors_candy + sister_candy + friends_candy
  let total_removed := traded_candy + given_away_candy
  let remaining_candy := total_received - total_removed
  remaining_candy / daily_consumption

/-- Theorem stating that Sarah's candy will last 9 days -/
theorem sarah_candy_duration : 
  candy_duration 66 15 20 10 5 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_duration_l2455_245514


namespace NUMINAMATH_CALUDE_phi_value_l2455_245583

open Real

theorem phi_value (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = sin x * cos φ + cos x * sin φ) →
  (0 < φ) →
  (φ < π) →
  (f (2 * (π/6) + π/6) = 1/2) →
  φ = π/3 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l2455_245583


namespace NUMINAMATH_CALUDE_fold_three_to_negative_three_fold_seven_to_negative_five_fold_points_with_distance_l2455_245547

-- Define a folding operation
def fold (m : ℝ) (x : ℝ) : ℝ := 2 * m - x

-- Theorem 1
theorem fold_three_to_negative_three :
  fold 0 3 = -3 :=
sorry

-- Theorem 2
theorem fold_seven_to_negative_five :
  fold 1 7 = -5 :=
sorry

-- Theorem 3
theorem fold_points_with_distance (m : ℝ) (h : m > 0) :
  ∃ (a b : ℝ), a < b ∧ b - a = m ∧ fold ((a + b) / 2) a = b ∧ a = -(1/2) * m + 1 ∧ b = (1/2) * m + 1 :=
sorry

end NUMINAMATH_CALUDE_fold_three_to_negative_three_fold_seven_to_negative_five_fold_points_with_distance_l2455_245547


namespace NUMINAMATH_CALUDE_max_ab_value_max_ab_value_achieved_l2455_245598

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  a * b ≤ (1/4 : ℝ) := by
  sorry

theorem max_ab_value_achieved (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  (∃ a' b' : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a' * x + b'| ≤ 1) ∧ a' * b' = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_max_ab_value_achieved_l2455_245598


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2455_245529

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x / y > 1) ∧
  ∃ x y : ℝ, x / y > 1 ∧ ¬(x > y ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2455_245529


namespace NUMINAMATH_CALUDE_inequality_proof_l2455_245555

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2455_245555


namespace NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l2455_245581

/-- Converts a quinary (base 5) number to decimal (base 10) -/
def quinary_to_decimal (q : ℕ) : ℕ :=
  (q / 100) * 5^2 + ((q / 10) % 10) * 5^1 + (q % 10) * 5^0

/-- Converts a decimal (base 10) number to octal (base 8) -/
def decimal_to_octal (d : ℕ) : ℕ :=
  (d / 64) * 100 + ((d / 8) % 8) * 10 + (d % 8)

/-- Theorem stating that 444₅ is equal to 174₈ -/
theorem quinary_444_equals_octal_174 :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l2455_245581


namespace NUMINAMATH_CALUDE_power_diff_reciprocal_power_l2455_245550

theorem power_diff_reciprocal_power (x : ℂ) :
  x - (1 / x) = Complex.I * Real.sqrt 2 →
  x^2187 - (1 / x^2187) = Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_diff_reciprocal_power_l2455_245550


namespace NUMINAMATH_CALUDE_total_spent_calculation_l2455_245574

/-- Calculates the total amount spent at a restaurant given the food price, sales tax rate, and tip rate. -/
def totalSpent (foodPrice : ℝ) (salesTaxRate : ℝ) (tipRate : ℝ) : ℝ :=
  let priceWithTax := foodPrice * (1 + salesTaxRate)
  let tipAmount := priceWithTax * tipRate
  priceWithTax + tipAmount

/-- Theorem stating that the total amount spent is $184.80 given the specific conditions. -/
theorem total_spent_calculation (foodPrice : ℝ) (salesTaxRate : ℝ) (tipRate : ℝ) 
    (h1 : foodPrice = 140)
    (h2 : salesTaxRate = 0.1)
    (h3 : tipRate = 0.2) : 
  totalSpent foodPrice salesTaxRate tipRate = 184.80 := by
  sorry

#eval totalSpent 140 0.1 0.2

end NUMINAMATH_CALUDE_total_spent_calculation_l2455_245574


namespace NUMINAMATH_CALUDE_f_2007_values_l2455_245507

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1

theorem f_2007_values (f : ℕ → ℕ) (h : is_valid_f f) : 
  f 2007 ∈ Finset.range 2009 ∧ 
  ∀ k ∈ Finset.range 2009, ∃ g : ℕ → ℕ, is_valid_f g ∧ g 2007 = k :=
sorry

end NUMINAMATH_CALUDE_f_2007_values_l2455_245507


namespace NUMINAMATH_CALUDE_quadratic_sum_l2455_245599

/-- A quadratic function g(x) = dx^2 + ex + f -/
def g (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  g d e f 0 = 5 → g d e f 2 = 3 → d + e + 3 * f = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2455_245599


namespace NUMINAMATH_CALUDE_construction_delay_l2455_245508

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  daysBeforeAddingWorkers : ℕ
  totalDays : ℕ

/-- Calculates the total man-days for the project -/
def totalManDays (project : ConstructionProject) : ℕ :=
  (project.initialWorkers * project.daysBeforeAddingWorkers) +
  ((project.initialWorkers + project.additionalWorkers) * (project.totalDays - project.daysBeforeAddingWorkers))

/-- Calculates the number of days needed with only initial workers -/
def daysWithInitialWorkersOnly (project : ConstructionProject) : ℕ :=
  (totalManDays project) / project.initialWorkers

/-- Theorem stating the delay in construction without additional workers -/
theorem construction_delay (project : ConstructionProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.additionalWorkers = 100)
  (h3 : project.daysBeforeAddingWorkers = 10)
  (h4 : project.totalDays = 100) :
  daysWithInitialWorkersOnly project - project.totalDays = 90 := by
  sorry


end NUMINAMATH_CALUDE_construction_delay_l2455_245508


namespace NUMINAMATH_CALUDE_simplify_expression_l2455_245511

theorem simplify_expression (x : ℝ) : (3*x - 4)*(2*x + 10) - (x + 3)*(3*x - 2) = 3*x^2 + 15*x - 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2455_245511


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2455_245518

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 3) ↔ x ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2455_245518


namespace NUMINAMATH_CALUDE_largest_expression_l2455_245592

theorem largest_expression (x : ℝ) : 
  (x + 1/4) * (x - 1/4) ≥ (x + 1) * (x - 1) ∧
  (x + 1/4) * (x - 1/4) ≥ (x + 1/2) * (x - 1/2) ∧
  (x + 1/4) * (x - 1/4) ≥ (x + 1/3) * (x - 1/3) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l2455_245592


namespace NUMINAMATH_CALUDE_distinct_sums_lower_bound_l2455_245548

theorem distinct_sums_lower_bound (n : ℕ) (a : ℕ → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ i, 0 < a i) :
  (Finset.powerset (Finset.range n)).card ≥ n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_lower_bound_l2455_245548


namespace NUMINAMATH_CALUDE_new_person_weight_l2455_245590

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 70 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2455_245590


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_value_main_theorem_l2455_245544

def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def qin_jiushao_v3 (a b c d : ℝ) (x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

theorem qin_jiushao_v3_value :
  qin_jiushao_v3 2 5 6 23 (-4) = -49 :=
by sorry

-- The main theorem
theorem main_theorem :
  ∃ (v3 : ℝ), qin_jiushao_v3 2 5 6 23 (-4) = v3 ∧ v3 = -49 :=
by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_value_main_theorem_l2455_245544


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l2455_245500

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 6 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_one :
  velocity 1 = 6 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l2455_245500


namespace NUMINAMATH_CALUDE_angle_C_measure_l2455_245512

theorem angle_C_measure (A B C : ℝ) (h : A + B = 80) : A + B + C = 180 → C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2455_245512


namespace NUMINAMATH_CALUDE_second_train_speed_prove_second_train_speed_l2455_245560

/-- Calculates the speed of the second train given the conditions of the problem -/
theorem second_train_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (extra_distance : ℝ) : ℝ :=
  let speed_second := (3 * distance - 2 * extra_distance) / (6 * distance / speed_first - 2 * extra_distance / speed_first)
  speed_second

/-- Proves that the speed of the second train is 125/3 kmph given the problem conditions -/
theorem prove_second_train_speed :
  second_train_speed 1100 50 100 = 125/3 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_prove_second_train_speed_l2455_245560


namespace NUMINAMATH_CALUDE_johnson_potatoes_problem_l2455_245562

theorem johnson_potatoes_problem (initial_potatoes : ℕ) (remaining_potatoes : ℕ) 
  (h1 : initial_potatoes = 300)
  (h2 : remaining_potatoes = 47) :
  ∃ (gina_potatoes : ℕ),
    gina_potatoes = 69 ∧
    initial_potatoes - remaining_potatoes = 
      gina_potatoes + 2 * gina_potatoes + 2 * gina_potatoes / 3 :=
by sorry

end NUMINAMATH_CALUDE_johnson_potatoes_problem_l2455_245562


namespace NUMINAMATH_CALUDE_truck_speed_difference_l2455_245543

/-- Represents the speed difference between paved and dirt roads for a semi truck journey --/
theorem truck_speed_difference 
  (total_distance : ℝ) 
  (paved_time dirt_time : ℝ) 
  (dirt_speed : ℝ) :
  total_distance = 200 →
  paved_time = 2 →
  dirt_time = 3 →
  dirt_speed = 32 →
  (total_distance - dirt_speed * dirt_time) / paved_time - dirt_speed = 20 := by
  sorry

#check truck_speed_difference

end NUMINAMATH_CALUDE_truck_speed_difference_l2455_245543


namespace NUMINAMATH_CALUDE_louis_ate_nine_boxes_l2455_245558

/-- The number of Lemon Heads in each package -/
def lemon_heads_per_package : ℕ := 6

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := total_lemon_heads / lemon_heads_per_package

theorem louis_ate_nine_boxes : whole_boxes = 9 := by
  sorry

end NUMINAMATH_CALUDE_louis_ate_nine_boxes_l2455_245558


namespace NUMINAMATH_CALUDE_marble_problem_l2455_245530

theorem marble_problem (b : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = b)
  (h2 : brian = 3 * b)
  (h3 : caden = 4 * brian)
  (h4 : daryl = 6 * caden)
  (h5 : angela + brian + caden + daryl = 312) :
  b = 39 / 11 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l2455_245530


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_4_l2455_245542

theorem units_digit_of_3_pow_4 : (3^4 : ℕ) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_4_l2455_245542


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l2455_245572

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l2455_245572


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2455_245576

/-- Proves that given an interest rate of 5% per annum for 2 years, 
    if the difference between compound interest and simple interest is 18, 
    then the principal amount is 7200. -/
theorem interest_difference_theorem (P : ℝ) : 
  P * (1 + 0.05)^2 - P - (P * 0.05 * 2) = 18 → P = 7200 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2455_245576


namespace NUMINAMATH_CALUDE_regular_polygon_right_triangles_l2455_245521

/-- Given a regular polygon with n sides, if there are 1200 ways to choose
    three vertices that form a right triangle, then n = 50. -/
theorem regular_polygon_right_triangles (n : ℕ) : n > 0 →
  (n / 2 : ℕ) * (n - 2) = 1200 → n = 50 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_right_triangles_l2455_245521


namespace NUMINAMATH_CALUDE_total_units_is_531_l2455_245557

/-- A mixed-use development with various floor types and unit distributions -/
structure MixedUseDevelopment where
  total_floors : Nat
  regular_floors : Nat
  luxury_floors : Nat
  penthouse_floors : Nat
  commercial_floors : Nat
  other_floors : Nat
  regular_odd_units : Nat
  regular_even_units : Nat
  luxury_avg_units : Nat
  penthouse_units : Nat
  commercial_units : Nat
  amenities_uncounted_units : Nat
  other_uncounted_units : Nat

/-- Calculate the total number of units in the mixed-use development -/
def total_units (dev : MixedUseDevelopment) : Nat :=
  let regular_units := (dev.regular_floors / 2 + dev.regular_floors % 2) * dev.regular_odd_units +
                       (dev.regular_floors / 2) * dev.regular_even_units
  let luxury_units := dev.luxury_floors * dev.luxury_avg_units
  let penthouse_units := dev.penthouse_floors * dev.penthouse_units
  let commercial_units := dev.commercial_floors * dev.commercial_units
  let uncounted_units := dev.amenities_uncounted_units + dev.other_uncounted_units
  regular_units + luxury_units + penthouse_units + commercial_units + uncounted_units

/-- The mixed-use development described in the problem -/
def problem_development : MixedUseDevelopment where
  total_floors := 60
  regular_floors := 25
  luxury_floors := 20
  penthouse_floors := 10
  commercial_floors := 3
  other_floors := 2
  regular_odd_units := 14
  regular_even_units := 12
  luxury_avg_units := 8
  penthouse_units := 2
  commercial_units := 5
  amenities_uncounted_units := 4
  other_uncounted_units := 6

/-- Theorem stating that the total number of units in the problem development is 531 -/
theorem total_units_is_531 : total_units problem_development = 531 := by
  sorry


end NUMINAMATH_CALUDE_total_units_is_531_l2455_245557


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2455_245563

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 3) ^ 2 = a 1 * a 9

/-- The main theorem -/
theorem arithmetic_geometric_ratio (seq : ArithmeticSequence) :
  (seq.a 2 + seq.a 4 + seq.a 10) / (seq.a 1 + seq.a 3 + seq.a 9) = 16 / 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2455_245563


namespace NUMINAMATH_CALUDE_fraction_closest_to_longest_side_l2455_245564

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (angleA angleD : Real)
  (angleB angleC : Real)
  (lengthAB lengthBC lengthCD lengthDA : Real)

-- Define the function to calculate the area closest to DA
def areaClosestToDA (q : Quadrilateral) : Real := sorry

-- Define the function to calculate the total area of the quadrilateral
def totalArea (q : Quadrilateral) : Real := sorry

-- Theorem statement
theorem fraction_closest_to_longest_side 
  (q : Quadrilateral)
  (h1 : q.A = (0, 0))
  (h2 : q.B = (1, 2))
  (h3 : q.C = (3, 2))
  (h4 : q.D = (4, 0))
  (h5 : q.angleA = 75)
  (h6 : q.angleD = 75)
  (h7 : q.angleB = 105)
  (h8 : q.angleC = 105)
  (h9 : q.lengthAB = 100)
  (h10 : q.lengthBC = 150)
  (h11 : q.lengthCD = 100)
  (h12 : q.lengthDA = 150)
  : areaClosestToDA q / totalArea q = areaClosestToDA q / totalArea q := by
  sorry

end NUMINAMATH_CALUDE_fraction_closest_to_longest_side_l2455_245564


namespace NUMINAMATH_CALUDE_chapters_undetermined_l2455_245584

/-- Represents a book with a number of pages and chapters -/
structure Book where
  pages : ℕ
  chapters : ℕ

/-- Represents Jake's reading progress -/
structure ReadingProgress where
  initialRead : ℕ
  laterRead : ℕ
  totalRead : ℕ

/-- Given the conditions of Jake's reading and the book, 
    prove that the number of chapters cannot be determined -/
theorem chapters_undetermined (book : Book) (progress : ReadingProgress) : 
  book.pages = 95 ∧ 
  progress.initialRead = 37 ∧ 
  progress.laterRead = 25 ∧ 
  progress.totalRead = 62 →
  ¬ ∃ (n : ℕ), ∀ (b : Book), 
    b.pages = book.pages ∧ 
    b.chapters = n :=
by sorry

end NUMINAMATH_CALUDE_chapters_undetermined_l2455_245584


namespace NUMINAMATH_CALUDE_set_S_properties_l2455_245552

-- Define the set S
def S (m n : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ n}

-- Main theorem
theorem set_S_properties (m n : ℝ) (h_nonempty : Set.Nonempty (S m n))
    (h_closed : ∀ x ∈ S m n, x^2 ∈ S m n) :
  (m = -1/2 → 1/4 ≤ n ∧ n ≤ 1) ∧
  (n = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_set_S_properties_l2455_245552


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l2455_245510

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l2455_245510


namespace NUMINAMATH_CALUDE_gamma_bank_lowest_savings_l2455_245577

def initial_funds : ℝ := 150000
def total_cost : ℝ := 201200

def rebs_bank_interest : ℝ := 2720.33
def gamma_bank_interest : ℝ := 3375.00
def tisi_bank_interest : ℝ := 2349.13
def btv_bank_interest : ℝ := 2264.11

def amount_to_save (interest : ℝ) : ℝ :=
  total_cost - initial_funds - interest

theorem gamma_bank_lowest_savings :
  let rebs_savings := amount_to_save rebs_bank_interest
  let gamma_savings := amount_to_save gamma_bank_interest
  let tisi_savings := amount_to_save tisi_bank_interest
  let btv_savings := amount_to_save btv_bank_interest
  (gamma_savings ≤ rebs_savings) ∧
  (gamma_savings ≤ tisi_savings) ∧
  (gamma_savings ≤ btv_savings) :=
by sorry

end NUMINAMATH_CALUDE_gamma_bank_lowest_savings_l2455_245577


namespace NUMINAMATH_CALUDE_intersection_parallel_line_equation_specific_line_equation_l2455_245534

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line_equation (a b c d e f g h i : ℝ) :
  (∃ x y, a * x + b * y = c ∧ d * x + e * y = f) →  -- Intersection point exists
  (∀ x y, (a * x + b * y = c ∧ d * x + e * y = f) → g * x + h * y + i = 0) →  -- Line passes through intersection
  (∃ k, ∀ x y, g * x + h * y + i = k * (g * x + h * y + 0)) →  -- Parallel to g * x + h * y + 0 = 0
  ∃ k, ∀ x y, g * x + h * y + i = k * (g * x + h * y - 27) :=
by sorry

/-- The specific case for the given problem -/
theorem specific_line_equation :
  (∃ x y, x + y = 9 ∧ 2 * x - y = 18) →
  (∀ x y, (x + y = 9 ∧ 2 * x - y = 18) → 3 * x - 2 * y + i = 0) →
  (∃ k, ∀ x y, 3 * x - 2 * y + i = k * (3 * x - 2 * y + 8)) →
  i = -27 :=
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_equation_specific_line_equation_l2455_245534


namespace NUMINAMATH_CALUDE_ones_digit_of_nine_to_46_l2455_245528

theorem ones_digit_of_nine_to_46 : (9^46 : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_nine_to_46_l2455_245528


namespace NUMINAMATH_CALUDE_unique_solution_l2455_245595

/-- A polynomial that satisfies the given functional equation -/
def functional_equation (p : ℝ → ℝ) : Prop :=
  ∀ x, p (p x) = 2 * x * p x + 3 * x^2

/-- The theorem stating that p(x) = 3x is the unique solution to the functional equation -/
theorem unique_solution :
  ∃! p : ℝ → ℝ, functional_equation p ∧ ∀ x, p x = 3 * x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2455_245595


namespace NUMINAMATH_CALUDE_num_segments_collinear_points_l2455_245597

/-- The number of distinct segments formed by n collinear points -/
def num_segments (n : ℕ) : ℕ := n.choose 2

/-- Theorem: For n distinct collinear points, the number of distinct segments is n choose 2 -/
theorem num_segments_collinear_points (n : ℕ) (h : n ≥ 2) :
  num_segments n = n.choose 2 := by sorry

end NUMINAMATH_CALUDE_num_segments_collinear_points_l2455_245597


namespace NUMINAMATH_CALUDE_nickel_probability_is_5_24_l2455_245517

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 800
  | Coin.Nickel => 500
  | Coin.Penny => 300

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of randomly selecting a nickel from the jar -/
def nickelProbability : ℚ := coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_5_24 : nickelProbability = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_nickel_probability_is_5_24_l2455_245517


namespace NUMINAMATH_CALUDE_garden_perimeter_l2455_245588

-- Define the garden shape
structure Garden where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ

-- Define the conditions
def is_valid_garden (g : Garden) : Prop :=
  g.a + g.b + g.c = 3 ∧
  g.a ≥ 0 ∧ g.b ≥ 0 ∧ g.c ≥ 0 ∧ g.x ≥ 0

-- Calculate the perimeter
def perimeter (g : Garden) : ℝ :=
  3 + 5 + g.a + g.x + g.b + 4 + g.c + (4 + (5 - g.x))

-- Theorem statement
theorem garden_perimeter (g : Garden) (h : is_valid_garden g) : perimeter g = 24 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2455_245588


namespace NUMINAMATH_CALUDE_baseball_division_games_l2455_245506

theorem baseball_division_games 
  (N M : ℕ) 
  (h1 : N > 2 * M) 
  (h2 : M > 4) 
  (h3 : 2 * N + 5 * M = 82) : 
  2 * N = 52 := by
  sorry

end NUMINAMATH_CALUDE_baseball_division_games_l2455_245506


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2455_245593

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2455_245593


namespace NUMINAMATH_CALUDE_smaller_bill_value_l2455_245519

/-- The value of the smaller denomination bill -/
def x : ℕ := sorry

/-- The total number of bills Anna has -/
def total_bills : ℕ := 12

/-- The number of smaller denomination bills Anna has -/
def smaller_bills : ℕ := 4

/-- The value of a $10 bill -/
def ten_dollar : ℕ := 10

/-- The total value of all bills in dollars -/
def total_value : ℕ := 100

theorem smaller_bill_value :
  x * smaller_bills + (total_bills - smaller_bills) * ten_dollar = total_value ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_bill_value_l2455_245519
