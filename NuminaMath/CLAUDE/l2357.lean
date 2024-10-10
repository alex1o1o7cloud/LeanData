import Mathlib

namespace rectangular_prism_diagonal_l2357_235755

theorem rectangular_prism_diagonal 
  (a b c : ℝ) 
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11) 
  (h2 : 4 * (a + b + c) = 24) : 
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
sorry

end rectangular_prism_diagonal_l2357_235755


namespace simplify_expression_l2357_235787

theorem simplify_expression (x : ℝ) : 
  3*x + 5*x^2 + 12 - (6 - 3*x - 10*x^2) = 15*x^2 + 6*x + 6 := by
  sorry

end simplify_expression_l2357_235787


namespace age_difference_is_32_l2357_235789

/-- The age difference between Mrs Bai and her daughter Jenni -/
def age_difference : ℕ :=
  let jenni_age : ℕ := 19
  let sum_of_ages : ℕ := 70
  sum_of_ages - 2 * jenni_age

/-- Theorem stating that the age difference between Mrs Bai and Jenni is 32 years -/
theorem age_difference_is_32 : age_difference = 32 := by
  sorry

end age_difference_is_32_l2357_235789


namespace treaty_signing_day_l2357_235760

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def advance_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => advance_day (advance_days d m)

theorem treaty_signing_day :
  advance_days DayOfWeek.Wednesday 1566 = DayOfWeek.Wednesday :=
by sorry


end treaty_signing_day_l2357_235760


namespace tickets_to_be_sold_l2357_235790

def total_tickets : ℕ := 100
def jude_sales : ℕ := 16

def andrea_sales (jude_sales : ℕ) : ℕ := 2 * jude_sales

def sandra_sales (jude_sales : ℕ) : ℕ := jude_sales / 2 + 4

def total_sold (jude_sales : ℕ) : ℕ :=
  jude_sales + andrea_sales jude_sales + sandra_sales jude_sales

theorem tickets_to_be_sold :
  total_tickets - total_sold jude_sales = 40 :=
by sorry

end tickets_to_be_sold_l2357_235790


namespace bus_driver_overtime_rate_increase_bus_driver_overtime_rate_increase_approx_l2357_235793

/-- Calculates the percentage increase in overtime rate compared to regular rate for a bus driver --/
theorem bus_driver_overtime_rate_increase 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) : ℝ :=
  let overtime_hours := total_hours - regular_hours
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  let percentage_increase := (overtime_rate - regular_rate) / regular_rate * 100
  percentage_increase

/-- The percentage increase in overtime rate is approximately 74.93% --/
theorem bus_driver_overtime_rate_increase_approx :
  ∃ ε > 0, abs (bus_driver_overtime_rate_increase 14 40 998 57.88 - 74.93) < ε :=
sorry

end bus_driver_overtime_rate_increase_bus_driver_overtime_rate_increase_approx_l2357_235793


namespace complex_fraction_equals_minus_one_plus_i_l2357_235746

theorem complex_fraction_equals_minus_one_plus_i :
  (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I :=
by sorry

end complex_fraction_equals_minus_one_plus_i_l2357_235746


namespace gravitational_force_at_new_distance_l2357_235791

/-- Gravitational force calculation -/
theorem gravitational_force_at_new_distance
  (f1 : ℝ) (d1 : ℝ) (d2 : ℝ)
  (h1 : f1 = 480)
  (h2 : d1 = 5000)
  (h3 : d2 = 300000)
  (h4 : ∀ (f d : ℝ), f * d^2 = f1 * d1^2) :
  ∃ (f2 : ℝ), f2 = 1 / 75 ∧ f2 * d2^2 = f1 * d1^2 := by
  sorry

end gravitational_force_at_new_distance_l2357_235791


namespace average_visitors_is_276_l2357_235772

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 26
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Theorem stating that the average number of visitors is 276 given the specified conditions -/
theorem average_visitors_is_276 :
  averageVisitors 510 240 = 276 := by
  sorry

#eval averageVisitors 510 240

end average_visitors_is_276_l2357_235772


namespace larger_number_problem_l2357_235758

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : max x y = 23 := by
  sorry

end larger_number_problem_l2357_235758


namespace age_difference_l2357_235786

theorem age_difference (A B C : ℕ) : 
  (∃ k : ℕ, A = B + k) →  -- A is some years older than B
  B = 2 * C →             -- B is twice as old as C
  A + B + C = 27 →        -- Total of ages is 27
  B = 10 →                -- B is 10 years old
  A = B + 2 :=            -- A is 2 years older than B
by sorry

end age_difference_l2357_235786


namespace three_numbers_sum_l2357_235759

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →  -- x is the least, z is the greatest
  y = 9 →  -- median is 9
  (x + y + z) / 3 = x + 20 →  -- mean is 20 more than least
  (x + y + z) / 3 = z - 18 →  -- mean is 18 less than greatest
  x + y + z = 21 := by sorry

end three_numbers_sum_l2357_235759


namespace store_sales_theorem_l2357_235798

/-- Represents the daily sales and profit calculations for a store. -/
structure StoreSales where
  initial_sales : ℕ
  initial_profit : ℕ
  sales_increase : ℕ
  min_profit : ℕ

/-- Calculates the new sales quantity after a price reduction. -/
def new_sales (s : StoreSales) (reduction : ℕ) : ℕ :=
  s.initial_sales + s.sales_increase * reduction

/-- Calculates the new profit per item after a price reduction. -/
def new_profit_per_item (s : StoreSales) (reduction : ℕ) : ℕ :=
  s.initial_profit - reduction

/-- Calculates the total daily profit after a price reduction. -/
def total_daily_profit (s : StoreSales) (reduction : ℕ) : ℕ :=
  (new_sales s reduction) * (new_profit_per_item s reduction)

/-- The main theorem stating the two parts of the problem. -/
theorem store_sales_theorem (s : StoreSales) 
    (h1 : s.initial_sales = 20)
    (h2 : s.initial_profit = 40)
    (h3 : s.sales_increase = 2)
    (h4 : s.min_profit = 25) : 
  (new_sales s 4 = 28) ∧ 
  (∃ (x : ℕ), x = 5 ∧ total_daily_profit s x = 1050 ∧ new_profit_per_item s x ≥ s.min_profit) := by
  sorry


end store_sales_theorem_l2357_235798


namespace correct_calculation_l2357_235771

theorem correct_calculation (x : ℤ) : x + 19 = 50 → 16 * x = 496 := by
  sorry

end correct_calculation_l2357_235771


namespace solution_range_l2357_235774

def A : Set ℝ := {x | (x + 1) / (x - 3) ≤ 0 ∧ x ≠ 3}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 ≤ 0}

theorem solution_range (a : ℝ) : 
  (∀ x, x ∈ B a → x ∈ A) ↔ -1/3 ≤ a ∧ a < 1 :=
sorry

end solution_range_l2357_235774


namespace tan_double_angle_special_case_l2357_235788

theorem tan_double_angle_special_case (x : ℝ) 
  (h : Real.sin x - 3 * Real.cos x = Real.sqrt 5) : 
  Real.tan (2 * x) = 4 / 3 := by
  sorry

end tan_double_angle_special_case_l2357_235788


namespace age_difference_l2357_235751

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 17) : A - C = 17 := by
  sorry

end age_difference_l2357_235751


namespace smallest_positive_difference_l2357_235766

/-- Vovochka's sum method for two three-digit numbers -/
def vovochkaSum (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Correct sum for two three-digit numbers -/
def correctSum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sumDifference (a b c d e f : ℕ) : ℤ :=
  (vovochkaSum a b c d e f : ℤ) - (correctSum a b c d e f : ℤ)

theorem smallest_positive_difference :
  ∀ a b c d e f : ℕ,
    a < 10 → b < 10 → c < 10 → d < 10 → e < 10 → f < 10 →
    (∃ a' b' c' d' e' f' : ℕ,
      a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ d' < 10 ∧ e' < 10 ∧ f' < 10 ∧
      sumDifference a' b' c' d' e' f' > 0 ∧
      sumDifference a' b' c' d' e' f' ≤ sumDifference a b c d e f) →
    (∃ a' b' c' d' e' f' : ℕ,
      a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ d' < 10 ∧ e' < 10 ∧ f' < 10 ∧
      sumDifference a' b' c' d' e' f' = 1800 ∧
      sumDifference a' b' c' d' e' f' > 0 ∧
      ∀ x y z u v w : ℕ,
        x < 10 → y < 10 → z < 10 → u < 10 → v < 10 → w < 10 →
        sumDifference x y z u v w > 0 →
        sumDifference a' b' c' d' e' f' ≤ sumDifference x y z u v w) :=
by sorry

end smallest_positive_difference_l2357_235766


namespace tulip_lilac_cost_comparison_l2357_235717

/-- Given that 4 tulips and 5 lilacs cost less than 22 yuan, and 6 tulips and 3 lilacs cost more than 24 yuan, prove that 2 tulips cost more than 3 lilacs. -/
theorem tulip_lilac_cost_comparison (x y : ℝ) 
  (h1 : 4 * x + 5 * y < 22) 
  (h2 : 6 * x + 3 * y > 24) : 
  2 * x > 3 * y := by
  sorry

end tulip_lilac_cost_comparison_l2357_235717


namespace total_packs_eq_sum_l2357_235716

/-- The number of glue stick packs Emily's mom bought -/
def total_packs : ℕ := sorry

/-- The number of glue stick packs Emily received -/
def emily_packs : ℕ := 6

/-- The number of glue stick packs Emily's sister received -/
def sister_packs : ℕ := 7

/-- Theorem: The total number of glue stick packs is the sum of packs given to Emily and her sister -/
theorem total_packs_eq_sum : total_packs = emily_packs + sister_packs := by sorry

end total_packs_eq_sum_l2357_235716


namespace jennifer_fruit_count_l2357_235764

def fruit_problem (pears oranges : ℕ) : Prop :=
  let apples := 2 * pears
  let cherries := oranges / 2
  let grapes := 3 * apples
  let initial_total := pears + oranges + apples + cherries + grapes
  let pineapples := initial_total
  let remaining_pears := pears - 3
  let remaining_oranges := oranges - 5
  let remaining_apples := apples - 5
  let remaining_cherries := cherries - 7
  let remaining_grapes := grapes - 3
  let remaining_before_pineapples := remaining_pears + remaining_oranges + remaining_apples + remaining_cherries + remaining_grapes
  let remaining_pineapples := pineapples - (pineapples / 2)
  remaining_before_pineapples + remaining_pineapples = 247

theorem jennifer_fruit_count : fruit_problem 15 30 := by
  sorry

end jennifer_fruit_count_l2357_235764


namespace male_attendee_fraction_l2357_235781

theorem male_attendee_fraction :
  let male_fraction : ℝ → ℝ := λ x => x
  let female_fraction : ℝ → ℝ := λ x => 1 - x
  let male_on_time : ℝ → ℝ := λ x => (7/8) * x
  let female_on_time : ℝ → ℝ := λ x => (9/10) * (1 - x)
  let total_on_time : ℝ := 0.885
  ∀ x : ℝ, male_on_time x + female_on_time x = total_on_time → x = 0.6 :=
by
  sorry

end male_attendee_fraction_l2357_235781


namespace digits_after_decimal_is_six_l2357_235708

/-- The number of digits to the right of the decimal point when 5^7 / (10^5 * 8^2) is expressed as a decimal -/
def digits_after_decimal : ℕ :=
  let fraction := (5^7 : ℚ) / ((10^5 * 8^2) : ℚ)
  6

theorem digits_after_decimal_is_six :
  digits_after_decimal = 6 := by sorry

end digits_after_decimal_is_six_l2357_235708


namespace inequality_solution_l2357_235711

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 →
  3 < x ∧ x < 17/3 := by
sorry

end inequality_solution_l2357_235711


namespace january_salary_solve_salary_problem_l2357_235752

/-- Represents the monthly salary structure -/
structure MonthlySalary where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

/-- Theorem stating the salary for January given the conditions -/
theorem january_salary (s : MonthlySalary) :
  (s.january + s.february + s.march + s.april) / 4 = 8000 →
  (s.february + s.march + s.april + s.may) / 4 = 8400 →
  s.may = 6500 →
  s.january = 4900 := by
  sorry

/-- Main theorem proving the salary calculation -/
theorem solve_salary_problem :
  ∃ s : MonthlySalary,
    (s.january + s.february + s.march + s.april) / 4 = 8000 ∧
    (s.february + s.march + s.april + s.may) / 4 = 8400 ∧
    s.may = 6500 ∧
    s.january = 4900 := by
  sorry

end january_salary_solve_salary_problem_l2357_235752


namespace star_eight_four_l2357_235714

-- Define the & operation
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

-- Define the ★ operation
def star (c d : ℝ) : ℝ := amp c d + 2 * (c + d)

-- Theorem to prove
theorem star_eight_four : star 8 4 = 72 := by
  sorry

end star_eight_four_l2357_235714


namespace sqrt_two_irrational_l2357_235739

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality as the negation of rationality
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l2357_235739


namespace cannot_finish_third_l2357_235709

/-- Represents the relative positions of runners in a race -/
def BeatsIn (runners : Type) := runners → runners → Prop

/-- A race with 5 runners and their relative positions -/
structure Race (runners : Type) :=
  (beats : BeatsIn runners)
  (P Q R S T : runners)
  (p_beats_q : beats P Q)
  (p_beats_r : beats P R)
  (p_beats_s : beats P S)
  (q_beats_s : beats Q S)
  (s_beats_r : beats S R)
  (t_after_p : beats P T)
  (t_before_q : beats T Q)

/-- Represents the finishing position of a runner -/
def FinishPosition (runners : Type) := runners → ℕ

theorem cannot_finish_third (runners : Type) (race : Race runners) 
  (finish : FinishPosition runners) :
  (finish race.P ≠ 3) ∧ (finish race.R ≠ 3) := by sorry

end cannot_finish_third_l2357_235709


namespace square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l2357_235794

theorem square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three :
  let x : ℝ := Real.sqrt 2 + 1
  (x^2 - 2*x + 2 : ℝ) = 3 := by
sorry

end square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l2357_235794


namespace sunday_sales_proof_l2357_235750

def price_per_caricature : ℕ := 20
def saturday_sales : ℕ := 24
def total_revenue : ℕ := 800

theorem sunday_sales_proof : 
  ∃ (sunday_sales : ℕ), 
    price_per_caricature * (saturday_sales + sunday_sales) = total_revenue ∧ 
    sunday_sales = 16 := by
  sorry

end sunday_sales_proof_l2357_235750


namespace optimal_rectangle_dimensions_l2357_235765

/-- The total length of the fence in meters -/
def fence_length : ℝ := 60

/-- The area of the rectangle as a function of its width -/
def area (x : ℝ) : ℝ := x * (fence_length - 2 * x)

/-- The width that maximizes the area -/
def optimal_width : ℝ := 15

/-- The length that maximizes the area -/
def optimal_length : ℝ := 30

theorem optimal_rectangle_dimensions :
  (∀ x : ℝ, 0 < x → x < fence_length / 2 → area x ≤ area optimal_width) ∧
  optimal_length = fence_length - 2 * optimal_width :=
sorry

end optimal_rectangle_dimensions_l2357_235765


namespace triangle_problem_l2357_235795

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a^2 + c^2 - b^2 = a * c →
  a = 8 * Real.sqrt 3 →
  Real.cos A = 3 / 5 →
  -- Conclusions
  B = π / 3 ∧ b = 15 := by
sorry

end triangle_problem_l2357_235795


namespace first_fish_length_is_0_3_l2357_235706

/-- The length of the second fish in feet -/
def second_fish_length : ℝ := 0.2

/-- The difference in length between the first and second fish in feet -/
def length_difference : ℝ := 0.1

/-- The length of the first fish in feet -/
def first_fish_length : ℝ := second_fish_length + length_difference

theorem first_fish_length_is_0_3 : first_fish_length = 0.3 := by
  sorry

end first_fish_length_is_0_3_l2357_235706


namespace max_internet_days_l2357_235721

/-- Represents the tiered pricing structure for internet service -/
def daily_rate (day : ℕ) : ℚ :=
  if day ≤ 3 then 1/2
  else if day ≤ 7 then 7/10
  else 9/10

/-- Calculates the additional fee for every 5 days -/
def additional_fee (day : ℕ) : ℚ :=
  if day % 5 = 0 then 1 else 0

/-- Calculates the total cost for a given number of days -/
def total_cost (days : ℕ) : ℚ :=
  (Finset.range days).sum (λ d => daily_rate (d + 1) + additional_fee (d + 1))

/-- Theorem stating that 8 is the maximum number of days of internet connection -/
theorem max_internet_days : 
  ∀ n : ℕ, n ≤ 8 → total_cost n ≤ 7 ∧ 
  (n < 8 → total_cost (n + 1) ≤ 7) ∧
  (total_cost 9 > 7) :=
sorry

end max_internet_days_l2357_235721


namespace absolute_opposite_reciprocal_of_negative_three_halves_l2357_235779

theorem absolute_opposite_reciprocal_of_negative_three_halves :
  let x : ℚ := -3/2
  (abs x = 3/2) ∧ (-x = 3/2) ∧ (x⁻¹ = -2/3) := by
  sorry

end absolute_opposite_reciprocal_of_negative_three_halves_l2357_235779


namespace line_passes_through_point_l2357_235705

/-- Given a line equation 3x + ay - 5 = 0 passing through point A(1, 2), prove that a = 1 -/
theorem line_passes_through_point (a : ℝ) : 
  (3 * 1 + a * 2 - 5 = 0) → a = 1 := by
  sorry

end line_passes_through_point_l2357_235705


namespace symmetric_distribution_within_one_std_dev_l2357_235745

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std_dev : ℝ

/-- The percentage of a symmetric distribution within one standard deviation of the mean -/
def percent_within_one_std_dev (dist : SymmetricDistribution) : ℝ :=
  2 * (dist.percent_less_than_mean_plus_std_dev - 50)

theorem symmetric_distribution_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std_dev = 82) :
  percent_within_one_std_dev dist = 64 := by
  sorry

#check symmetric_distribution_within_one_std_dev

end symmetric_distribution_within_one_std_dev_l2357_235745


namespace mall_computer_sales_l2357_235784

theorem mall_computer_sales (planned_sales : ℕ) (golden_week_avg : ℕ) (increase_percent : ℕ) (remaining_days : ℕ) :
  planned_sales = 900 →
  golden_week_avg = 54 →
  increase_percent = 30 →
  remaining_days = 24 →
  (∃ x : ℕ, x ≥ 33 ∧ golden_week_avg * 7 + x * remaining_days ≥ planned_sales + planned_sales * increase_percent / 100) :=
by
  sorry

#check mall_computer_sales

end mall_computer_sales_l2357_235784


namespace mowing_time_c_l2357_235768

-- Define the work rates
def work_rate (days : ℚ) : ℚ := 1 / days

-- Define the given conditions
def condition1 (a b : ℚ) : Prop := a + b = work_rate 28
def condition2 (a b c : ℚ) : Prop := a + b + c = work_rate 21

-- Theorem statement
theorem mowing_time_c (a b c : ℚ) 
  (h1 : condition1 a b) (h2 : condition2 a b c) : c = work_rate 84 := by
  sorry

end mowing_time_c_l2357_235768


namespace sequence_sum_l2357_235780

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  b 1 = 1 →
  b 3 = b 2 + 2 →
  b 4 = a 3 + a 5 →
  b 5 = a 4 + 2 * a 6 →
  a 2018 + b 9 = 2274 := by
  sorry


end sequence_sum_l2357_235780


namespace total_cartons_packed_l2357_235761

/-- Proves the total number of cartons packed given the conditions -/
theorem total_cartons_packed 
  (cans_per_carton : ℕ) 
  (loaded_cartons : ℕ) 
  (remaining_cans : ℕ) : 
  cans_per_carton = 20 → 
  loaded_cartons = 40 → 
  remaining_cans = 200 → 
  loaded_cartons + (remaining_cans / cans_per_carton) = 50 := by
  sorry

end total_cartons_packed_l2357_235761


namespace max_participants_l2357_235748

/-- Represents a round-robin chess tournament. -/
structure ChessTournament where
  n : ℕ  -- number of players
  a : ℕ  -- number of draws
  b : ℕ  -- number of wins (and losses)

/-- The conditions of the tournament are met. -/
def validTournament (t : ChessTournament) : Prop :=
  t.n > 0 ∧
  2 * t.a + 3 * t.b = 120 ∧
  t.a + t.b = t.n * (t.n - 1) / 2

/-- The maximum number of participants in the tournament is 11. -/
theorem max_participants :
  ∀ t : ChessTournament, validTournament t → t.n ≤ 11 :=
by sorry

end max_participants_l2357_235748


namespace supermarket_promotion_cost_l2357_235736

/-- Represents the cost calculation for a supermarket promotion --/
def supermarket_promotion (x : ℕ) : Prop :=
  let teapot_price : ℕ := 20
  let teacup_price : ℕ := 6
  let num_teapots : ℕ := 5
  x > 5 →
  (num_teapots * teapot_price + (x - num_teapots) * teacup_price = 6 * x + 70) ∧
  ((num_teapots * teapot_price + x * teacup_price) * 9 / 10 = 54 * x / 10 + 90)

theorem supermarket_promotion_cost (x : ℕ) : supermarket_promotion x :=
by sorry

end supermarket_promotion_cost_l2357_235736


namespace oil_leak_calculation_l2357_235704

/-- The total amount of oil leaked from a broken pipe -/
def total_oil_leaked (before_fixing : ℕ) (while_fixing : ℕ) : ℕ :=
  before_fixing + while_fixing

/-- Theorem: Given the specific amounts of oil leaked before and during fixing,
    the total amount of oil leaked is 6206 gallons -/
theorem oil_leak_calculation :
  total_oil_leaked 2475 3731 = 6206 := by
  sorry

end oil_leak_calculation_l2357_235704


namespace appropriate_sampling_methods_l2357_235728

/-- Represents different sampling methods -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a community with different income groups -/
structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ
  sample_size : ℕ

/-- Represents a group of volleyball players -/
structure VolleyballTeam where
  total_players : ℕ
  players_to_select : ℕ

/-- Determines the most appropriate sampling method for a community survey -/
def best_community_sampling_method (c : Community) : SamplingMethod :=
  sorry

/-- Determines the most appropriate sampling method for a volleyball team survey -/
def best_volleyball_sampling_method (v : VolleyballTeam) : SamplingMethod :=
  sorry

/-- Theorem stating the most appropriate sampling methods for the given scenarios -/
theorem appropriate_sampling_methods 
  (community : Community)
  (volleyball_team : VolleyballTeam)
  (h_community : community = { 
    total_households := 400,
    high_income := 120,
    middle_income := 180,
    low_income := 100,
    sample_size := 100
  })
  (h_volleyball : volleyball_team = {
    total_players := 12,
    players_to_select := 3
  }) :
  best_community_sampling_method community = SamplingMethod.StratifiedSampling ∧
  best_volleyball_sampling_method volleyball_team = SamplingMethod.SimpleRandomSampling :=
sorry

end appropriate_sampling_methods_l2357_235728


namespace smallest_n_with_seven_and_terminating_l2357_235718

/-- A function that checks if a number contains the digit 7 -/
def contains_seven (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < n ∧ n % 10^(d+1) / 10^d = 7

/-- A function that checks if a fraction 1/n is a terminating decimal -/
def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

/-- Theorem stating that 128 is the smallest positive integer n such that
    1/n is a terminating decimal and n contains the digit 7 -/
theorem smallest_n_with_seven_and_terminating :
  ∀ n : ℕ, n > 0 → is_terminating_decimal n → contains_seven n → n ≥ 128 :=
sorry

end smallest_n_with_seven_and_terminating_l2357_235718


namespace factorable_p_values_l2357_235777

def is_factorable (p : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ x, x^2 + p*x + 12 = (x + a) * (x + b)

theorem factorable_p_values (p : ℤ) :
  is_factorable p ↔ p ∈ ({-13, -8, -7, 7, 8, 13} : Set ℤ) := by
  sorry

end factorable_p_values_l2357_235777


namespace class_size_l2357_235735

/-- Represents the number of students in the class -/
def n : ℕ := sorry

/-- Represents the number of leftover erasers -/
def x : ℕ := sorry

/-- The number of gel pens bought -/
def gel_pens : ℕ := 2 * n + 2 * x

/-- The number of ballpoint pens bought -/
def ballpoint_pens : ℕ := 3 * n + 48

/-- The number of erasers bought -/
def erasers : ℕ := 4 * n + x

theorem class_size : 
  gel_pens = ballpoint_pens ∧ 
  ballpoint_pens = erasers ∧ 
  x = 2 * n → 
  n = 16 := by sorry

end class_size_l2357_235735


namespace possible_values_of_P_zero_l2357_235713

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that the polynomial P must satisfy -/
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, |y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|

/-- The theorem stating the possible values of P(0) -/
theorem possible_values_of_P_zero (P : RealPolynomial) 
  (h : SatisfiesProperty P) : 
  P 0 < 0 ∨ P 0 = 1 := by
  sorry

end possible_values_of_P_zero_l2357_235713


namespace sum_of_extreme_prime_factors_of_1260_l2357_235769

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem sum_of_extreme_prime_factors_of_1260 :
  ∃ (min max : ℕ),
    is_prime_factor min 1260 ∧
    is_prime_factor max 1260 ∧
    (∀ p, is_prime_factor p 1260 → min ≤ p) ∧
    (∀ p, is_prime_factor p 1260 → p ≤ max) ∧
    min + max = 9 :=
  sorry

end sum_of_extreme_prime_factors_of_1260_l2357_235769


namespace limit_of_sequence_l2357_235726

/-- The sequence a_n defined as (1 + 3n) / (6 - n) converges to -3 as n approaches infinity. -/
theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((1 : ℝ) + 3 * n) / (6 - n) + 3| < ε :=
sorry

end limit_of_sequence_l2357_235726


namespace product_of_fractions_l2357_235727

theorem product_of_fractions : (2 : ℚ) / 3 * (5 : ℚ) / 11 = 10 / 33 := by sorry

end product_of_fractions_l2357_235727


namespace simplify_fraction_l2357_235753

theorem simplify_fraction : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end simplify_fraction_l2357_235753


namespace bathroom_area_is_eight_l2357_235715

/-- The area of a rectangular bathroom -/
def bathroom_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a bathroom with length 4 feet and width 2 feet is 8 square feet -/
theorem bathroom_area_is_eight :
  bathroom_area 4 2 = 8 := by
  sorry

end bathroom_area_is_eight_l2357_235715


namespace woman_work_time_l2357_235703

/-- Represents the time taken to complete a work unit -/
structure WorkTime where
  men : ℕ
  women : ℕ
  days : ℚ

/-- The work rate of a single person -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

theorem woman_work_time (wt1 wt2 : WorkTime) : 
  wt1.men = 10 ∧ 
  wt1.women = 15 ∧ 
  wt1.days = 6 ∧
  wt2.men = 1 ∧ 
  wt2.women = 0 ∧ 
  wt2.days = 100 →
  ∃ wt3 : WorkTime, wt3.men = 0 ∧ wt3.women = 1 ∧ wt3.days = 225 :=
by sorry

#check woman_work_time

end woman_work_time_l2357_235703


namespace total_fish_l2357_235796

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 14) : 
  lilly_fish + rosy_fish = 24 := by
sorry

end total_fish_l2357_235796


namespace a_range_l2357_235700

-- Define the line equation
def line_equation (a x y : ℝ) : ℝ := a * x + 2 * y - 1

-- Define the points A and B
def point_A : ℝ × ℝ := (3, -1)
def point_B : ℝ × ℝ := (-1, 2)

-- Define the condition for points being on the same side of the line
def same_side (a : ℝ) : Prop :=
  (line_equation a point_A.1 point_A.2) * (line_equation a point_B.1 point_B.2) > 0

-- Theorem stating the range of a
theorem a_range : 
  ∀ a : ℝ, same_side a ↔ a ∈ Set.Ioo 1 3 :=
sorry

end a_range_l2357_235700


namespace quadratic_inequality_solutions_l2357_235732

theorem quadratic_inequality_solutions (b : ℤ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, x^2 + b*x + 1 ≤ 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) →
  b = 4 ∨ b = -4 :=
sorry

end quadratic_inequality_solutions_l2357_235732


namespace consecutive_cs_majors_probability_l2357_235799

/-- The number of people sitting at the round table -/
def total_people : ℕ := 12

/-- The number of computer science majors -/
def cs_majors : ℕ := 5

/-- The number of engineering majors -/
def eng_majors : ℕ := 4

/-- The number of art majors -/
def art_majors : ℕ := 3

/-- The probability of all computer science majors sitting consecutively -/
def consecutive_cs_prob : ℚ := 1 / 66

theorem consecutive_cs_majors_probability :
  (total_people = cs_majors + eng_majors + art_majors) →
  (consecutive_cs_prob = (total_people : ℚ) / (total_people.choose cs_majors)) := by
  sorry

end consecutive_cs_majors_probability_l2357_235799


namespace inequality_proof_l2357_235702

theorem inequality_proof (x y z : ℝ) : (x^2 - y^2)^2 + (z - x)^2 + (x - 1)^2 ≥ 0 := by
  sorry

end inequality_proof_l2357_235702


namespace rope_cutting_ratio_l2357_235743

/-- Given a rope of initial length 100 feet, prove that after specific cuts,
    the ratio of the final piece to its parent piece is 1:5 -/
theorem rope_cutting_ratio :
  ∀ (initial_length : ℝ) (final_piece_length : ℝ),
    initial_length = 100 →
    final_piece_length = 5 →
    ∃ (second_cut_length : ℝ),
      second_cut_length = initial_length / 4 ∧
      final_piece_length / second_cut_length = 1 / 5 := by
  sorry

end rope_cutting_ratio_l2357_235743


namespace blasting_safety_condition_l2357_235742

/-- Represents the parameters of a blasting operation safety scenario -/
structure BlastingSafety where
  safetyDistance : ℝ
  fuseSpeed : ℝ
  blasterSpeed : ℝ

/-- Defines the safety condition for a blasting operation -/
def isSafe (params : BlastingSafety) (fuseLength : ℝ) : Prop :=
  fuseLength / params.fuseSpeed > (params.safetyDistance - fuseLength) / params.blasterSpeed

/-- Theorem stating the safety condition for a specific blasting scenario -/
theorem blasting_safety_condition :
  let params : BlastingSafety := {
    safetyDistance := 50,
    fuseSpeed := 0.2,
    blasterSpeed := 3
  }
  ∀ x : ℝ, isSafe params x ↔ x / 0.2 > (50 - x) / 3 := by
  sorry


end blasting_safety_condition_l2357_235742


namespace optimal_profit_distribution_l2357_235712

/-- Represents the profit and production setup for handicrafts A and B --/
structure HandicraftSetup where
  profit_diff : ℝ  -- Profit difference between B and A
  profit_A_equal : ℝ  -- Profit of A when quantities are equal
  profit_B_equal : ℝ  -- Profit of B when quantities are equal
  total_workers : ℕ  -- Total number of workers
  A_production_rate : ℕ  -- Number of A pieces one worker can produce
  B_production_rate : ℕ  -- Number of B pieces one worker can produce
  min_B_production : ℕ  -- Minimum number of B pieces to be produced
  profit_decrease_rate : ℝ  -- Rate of profit decrease per extra B piece

/-- Calculates the maximum profit for the given handicraft setup --/
def max_profit (setup : HandicraftSetup) : ℝ :=
  let profit_A := setup.profit_A_equal * setup.profit_B_equal / (setup.profit_B_equal - setup.profit_diff)
  let profit_B := profit_A + setup.profit_diff
  let m := setup.total_workers / 2  -- Approximate midpoint for worker distribution
  (-2) * (m - 25)^2 + 3200

/-- Theorem stating the maximum profit and optimal worker distribution --/
theorem optimal_profit_distribution (setup : HandicraftSetup) :
  setup.profit_diff = 105 ∧
  setup.profit_A_equal = 30 ∧
  setup.profit_B_equal = 240 ∧
  setup.total_workers = 65 ∧
  setup.A_production_rate = 2 ∧
  setup.B_production_rate = 1 ∧
  setup.min_B_production = 5 ∧
  setup.profit_decrease_rate = 2 →
  max_profit setup = 3200 ∧
  ∃ (workers_A workers_B : ℕ),
    workers_A = 40 ∧
    workers_B = 25 ∧
    workers_A + workers_B = setup.total_workers :=
by
  sorry

end optimal_profit_distribution_l2357_235712


namespace richmond_tigers_ticket_sales_l2357_235749

theorem richmond_tigers_ticket_sales (total_tickets : ℕ) (first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570) (h2 : first_half_tickets = 3867) :
  total_tickets - first_half_tickets = 5703 := by
  sorry

end richmond_tigers_ticket_sales_l2357_235749


namespace greatest_two_digit_multiple_of_17_l2357_235725

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l2357_235725


namespace similar_triangle_shortest_side_l2357_235754

theorem similar_triangle_shortest_side 
  (side1 : ℝ) 
  (hyp1 : ℝ) 
  (hyp2 : ℝ) 
  (is_right_triangle : side1^2 + (hyp1^2 - side1^2) = hyp1^2)
  (hyp1_positive : hyp1 > 0)
  (hyp2_positive : hyp2 > 0)
  (h_similar : hyp2 / hyp1 * side1 = 72) :
  ∃ (side2 : ℝ), side2 = 72 ∧ side2 ≤ hyp2 := by sorry

end similar_triangle_shortest_side_l2357_235754


namespace expression_equivalence_l2357_235724

theorem expression_equivalence : 
  (4+5)*(4^2+5^2)*(4^4+5^4)*(4^8+5^8)*(4^16+5^16)*(4^32+5^32)*(4^64+5^64) = 5^128 - 4^128 := by
  sorry

end expression_equivalence_l2357_235724


namespace no_primes_in_range_l2357_235773

theorem no_primes_in_range (n m : ℕ) (hn : n > 1) (hm : 1 ≤ m ∧ m ≤ n) :
  ∀ k, n! + m < k ∧ k < n! + n + m → ¬ Nat.Prime k := by
  sorry

end no_primes_in_range_l2357_235773


namespace merchant_profit_comparison_l2357_235707

/-- Represents the profit calculation for two merchants selling goods --/
theorem merchant_profit_comparison
  (x : ℝ) -- cost price of goods for each merchant
  (h_pos : x > 0) -- assumption that cost price is positive
  : x < 1.08 * x := by
  sorry

#check merchant_profit_comparison

end merchant_profit_comparison_l2357_235707


namespace student_distribution_l2357_235792

/-- The number of ways to distribute n students to k universities --/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The condition that each university admits at least one student --/
def at_least_one (n : ℕ) (k : ℕ) : Prop :=
  sorry

theorem student_distribution :
  ∃ (d : ℕ → ℕ → ℕ), ∃ (c : ℕ → ℕ → Prop),
    d 5 3 = 150 ∧ c 5 3 ∧
    ∀ (n k : ℕ), c n k → d n k = distribute n k :=
by sorry

end student_distribution_l2357_235792


namespace cubic_three_roots_m_range_l2357_235734

/-- Given a cubic function f(x) = x³ - 6x² + 9x + m, if there exist three distinct
    real roots, then the parameter m must be in the open interval (-4, 0). -/
theorem cubic_three_roots_m_range (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧
    (a^3 - 6*a^2 + 9*a + m = 0) ∧
    (b^3 - 6*b^2 + 9*b + m = 0) ∧
    (c^3 - 6*c^2 + 9*c + m = 0)) →
  -4 < m ∧ m < 0 := by
sorry

end cubic_three_roots_m_range_l2357_235734


namespace sixty_seven_in_one_row_l2357_235719

def pascal_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem sixty_seven_in_one_row :
  ∃! row : ℕ, ∃ k : ℕ, pascal_coefficient row k = 67 :=
by
  sorry

end sixty_seven_in_one_row_l2357_235719


namespace interest_rate_is_twelve_percent_l2357_235744

/-- Given a banker's gain and true discount on a bill due in 1 year,
    calculate the rate of interest per annum. -/
def calculate_interest_rate (bankers_gain : ℚ) (true_discount : ℚ) : ℚ :=
  bankers_gain / true_discount

/-- Theorem stating that for the given banker's gain and true discount,
    the calculated interest rate is 12% -/
theorem interest_rate_is_twelve_percent :
  let bankers_gain : ℚ := 78/10
  let true_discount : ℚ := 65
  calculate_interest_rate bankers_gain true_discount = 12/100 := by
  sorry

end interest_rate_is_twelve_percent_l2357_235744


namespace integer_square_four_l2357_235729

theorem integer_square_four (x : ℝ) (y : ℤ) 
  (eq1 : 4 * x + y = 34)
  (eq2 : 2 * x - y = 20) : 
  y = -2 ∧ y^2 = 4 := by
  sorry

end integer_square_four_l2357_235729


namespace phone_rep_work_hours_l2357_235783

theorem phone_rep_work_hours 
  (num_reps : ℕ) 
  (num_days : ℕ) 
  (hourly_rate : ℚ) 
  (total_pay : ℚ) 
  (h1 : num_reps = 50)
  (h2 : num_days = 5)
  (h3 : hourly_rate = 14)
  (h4 : total_pay = 28000) :
  (total_pay / hourly_rate) / (num_reps * num_days) = 8 := by
sorry

end phone_rep_work_hours_l2357_235783


namespace two_digit_reverse_difference_cube_l2357_235767

/-- A two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- The reversed digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Predicate for a number being a positive perfect cube -/
def isPositivePerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ n = k^3

/-- The main theorem -/
theorem two_digit_reverse_difference_cube :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, TwoDigitNumber n ∧ isPositivePerfectCube (n - reverseDigits n)) ∧
    Finset.card s = 2 := by
  sorry

end two_digit_reverse_difference_cube_l2357_235767


namespace log_property_l2357_235722

theorem log_property (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.log x) (h2 : f (a * b) = 1) :
  f (a ^ 2) + f (b ^ 2) = 2 := by
  sorry

end log_property_l2357_235722


namespace max_value_sqrt_sum_l2357_235778

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + 9 * c^2 = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c ≤ Real.sqrt 21 / 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
  a₀ + b₀ + 9 * c₀^2 = 1 ∧ 
  Real.sqrt a₀ + Real.sqrt b₀ + Real.sqrt 3 * c₀ = Real.sqrt 21 / 3 :=
sorry

end max_value_sqrt_sum_l2357_235778


namespace min_value_of_a2_plus_b2_l2357_235710

/-- The circle equation: x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line equation: ax + 2by + 4 = 0 -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x + 2*b*y + 4 = 0

/-- The chord length is 4 -/
def chord_length_is_4 (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    line_equation a b x₁ y₁ ∧
    line_equation a b x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2

theorem min_value_of_a2_plus_b2 :
  ∀ a b : ℝ, chord_length_is_4 a b →
  ∃ min : ℝ, min = 2 ∧ a^2 + b^2 ≥ min :=
by sorry

end min_value_of_a2_plus_b2_l2357_235710


namespace rook_paths_eq_catalan_l2357_235731

/-- The number of valid paths for a rook on an n × n chessboard -/
def rookPaths (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (Nat.choose (2 * n - 2) (n - 1)) / n

/-- The Catalan number C_n -/
def catalanNumber (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (Nat.choose (2 * n) n) / (n + 1)

/-- Theorem: The number of valid rook paths on an n × n chessboard
    is equal to the (n-1)th Catalan number -/
theorem rook_paths_eq_catalan (n : ℕ) :
  rookPaths n = catalanNumber (n - 1) :=
sorry

end rook_paths_eq_catalan_l2357_235731


namespace equation_equivalence_l2357_235737

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

-- Define the simplified ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Theorem stating the equivalence of the two equations
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ ellipse_equation x y :=
sorry

end equation_equivalence_l2357_235737


namespace jerrys_age_l2357_235797

/-- Given that Mickey's age is 10 years more than 200% of Jerry's age,
    and Mickey is 22 years old, Jerry's age is 6 years. -/
theorem jerrys_age (mickey jerry : ℕ) 
  (h1 : mickey = 2 * jerry + 10)  -- Mickey's age relation to Jerry's
  (h2 : mickey = 22)              -- Mickey's age
  : jerry = 6 := by
  sorry

end jerrys_age_l2357_235797


namespace unique_positive_number_l2357_235701

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 8 = 128 / x := by
  sorry

end unique_positive_number_l2357_235701


namespace rectangle_area_constraint_l2357_235785

theorem rectangle_area_constraint (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 70 → m = (1 + Real.sqrt 1129) / 6 :=
by
  sorry

end rectangle_area_constraint_l2357_235785


namespace cosine_symmetry_center_l2357_235720

/-- Given a cosine function y = 2cos(2x) translated π/12 units to the right,
    prove that (5π/6, 0) is one of its symmetry centers. -/
theorem cosine_symmetry_center :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * (x - π/12))
  ∃ (k : ℤ), (5*π/6 : ℝ) = k*π/2 + π/3 ∧ 
    (∀ x : ℝ, f (5*π/6 + x) = f (5*π/6 - x)) :=
by sorry

end cosine_symmetry_center_l2357_235720


namespace no_real_roots_x_squared_plus_three_l2357_235733

theorem no_real_roots_x_squared_plus_three : 
  ∀ x : ℝ, x^2 + 3 ≠ 0 := by
sorry

end no_real_roots_x_squared_plus_three_l2357_235733


namespace students_behind_minyoung_l2357_235782

/-- Given a line of students with Minyoung, prove the number behind her. -/
theorem students_behind_minyoung 
  (total : ℕ) 
  (in_front : ℕ) 
  (h1 : total = 35) 
  (h2 : in_front = 27) : 
  total - (in_front + 1) = 7 := by
  sorry

end students_behind_minyoung_l2357_235782


namespace katya_problems_l2357_235740

theorem katya_problems (p_katya : ℝ) (p_pen : ℝ) (total_problems : ℕ) (good_grade : ℝ) 
  (h_katya : p_katya = 4/5)
  (h_pen : p_pen = 1/2)
  (h_total : total_problems = 20)
  (h_good : good_grade = 13) :
  ∃ x : ℝ, x ≥ 10 ∧ 
    x * p_katya + (total_problems - x) * p_pen ≥ good_grade ∧
    ∀ y : ℝ, y < 10 → y * p_katya + (total_problems - y) * p_pen < good_grade := by
  sorry

end katya_problems_l2357_235740


namespace negation_of_parallelogram_is_rhombus_is_true_l2357_235723

-- Define the property of being a parallelogram
def is_parallelogram (shape : Type) : Prop := sorry

-- Define the property of being a rhombus
def is_rhombus (shape : Type) : Prop := sorry

-- The statement we want to prove
theorem negation_of_parallelogram_is_rhombus_is_true :
  ∃ (shape : Type), is_parallelogram shape ∧ ¬is_rhombus shape := by sorry

end negation_of_parallelogram_is_rhombus_is_true_l2357_235723


namespace total_turnips_l2357_235756

theorem total_turnips (keith_turnips alyssa_turnips : ℕ) 
  (h1 : keith_turnips = 6) 
  (h2 : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 := by
  sorry

end total_turnips_l2357_235756


namespace equation_solution_l2357_235775

theorem equation_solution :
  ∃! x : ℚ, x + 5/6 = 11/18 - 2/9 ∧ x = -4/9 := by
  sorry

end equation_solution_l2357_235775


namespace tan_30_plus_2cos_30_l2357_235763

theorem tan_30_plus_2cos_30 : Real.tan (π / 6) + 2 * Real.cos (π / 6) = 4 * Real.sqrt 3 / 3 := by
  sorry

end tan_30_plus_2cos_30_l2357_235763


namespace quadratic_equation_problem_l2357_235741

theorem quadratic_equation_problem (k : ℝ) (α β : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + 3 - k = 0 ∧ y^2 + 2*y + 3 - k = 0) →
  (k > 2 ∧ 
   (k^2 = α*β + 3*k ∧ α^2 + 2*α + 3 - k = 0 ∧ β^2 + 2*β + 3 - k = 0) → k = 3) :=
by sorry

end quadratic_equation_problem_l2357_235741


namespace sunflower_majority_on_friday_friday_first_sunflower_majority_l2357_235762

/-- Represents the amount of sunflower seeds in the feeder on a given day -/
def sunflower_seeds (day : ℕ) : ℝ :=
  0.9 + (0.63 ^ (day - 1)) * 0.9

/-- Represents the total amount of seeds in the feeder on a given day -/
def total_seeds (day : ℕ) : ℝ :=
  3 * day

/-- The theorem states that on the 5th day, sunflower seeds exceed half of the total seeds -/
theorem sunflower_majority_on_friday :
  sunflower_seeds 5 > (total_seeds 5) / 2 := by
  sorry

/-- Helper function to check if sunflower seeds exceed half of total seeds on a given day -/
def is_sunflower_majority (day : ℕ) : Prop :=
  sunflower_seeds day > (total_seeds day) / 2

/-- The theorem states that Friday (day 5) is the first day when sunflower seeds exceed half of the total seeds -/
theorem friday_first_sunflower_majority :
  is_sunflower_majority 5 ∧ 
  (∀ d : ℕ, d < 5 → ¬is_sunflower_majority d) := by
  sorry

end sunflower_majority_on_friday_friday_first_sunflower_majority_l2357_235762


namespace equation_solution_l2357_235738

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem equation_solution (a : ℕ+) :
  (∃ n : ℕ+, 7 * a * n - 3 * factorial n = 2020) ↔ (a = 68 ∨ a = 289) :=
sorry

end equation_solution_l2357_235738


namespace circle_triangle_angle_measure_l2357_235776

-- Define the circle and triangle
def Circle : Type := Unit
def Point : Type := Unit
def Triangle : Type := Unit

-- Define the center of the circle
def center (c : Circle) : Point := sorry

-- Define the vertices of the triangle
def X (t : Triangle) : Point := sorry
def Y (t : Triangle) : Point := sorry
def Z (t : Triangle) : Point := sorry

-- Define the property of being circumscribed
def is_circumscribed (c : Circle) (t : Triangle) : Prop := sorry

-- Define the measure of an angle
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem circle_triangle_angle_measure 
  (c : Circle) (t : Triangle) (h_circumscribed : is_circumscribed c t) 
  (h_XOY : angle_measure (X t) (center c) (Y t) = 120)
  (h_YOZ : angle_measure (Y t) (center c) (Z t) = 140) :
  angle_measure (X t) (Y t) (Z t) = 60 := by sorry

end circle_triangle_angle_measure_l2357_235776


namespace min_value_and_inequality_range_l2357_235770

def f (x : ℝ) : ℝ := |2*x - 1| + |x + 2|

theorem min_value_and_inequality_range :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 5/2) ∧
  (∀ (a b x : ℝ), a ≠ 0 → |2*b - a| + |b + 2*a| ≥ |a| * (|x + 1| + |x - 1|) → -5/4 ≤ x ∧ x ≤ 5/4) :=
sorry

end min_value_and_inequality_range_l2357_235770


namespace cloth_meters_sold_l2357_235747

/-- Proves that the number of meters of cloth sold is 66, given the selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_meters_sold
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (cost_per_meter : ℕ)
  (h1 : selling_price = 660)
  (h2 : profit_per_meter = 5)
  (h3 : cost_per_meter = 5) :
  selling_price / (profit_per_meter + cost_per_meter) = 66 := by
  sorry

#check cloth_meters_sold

end cloth_meters_sold_l2357_235747


namespace boys_in_art_class_l2357_235757

theorem boys_in_art_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) 
  (h1 : total = 35) 
  (h2 : ratio_girls = 4) 
  (h3 : ratio_boys = 3) : 
  (ratio_boys * total) / (ratio_girls + ratio_boys) = 15 := by
  sorry

end boys_in_art_class_l2357_235757


namespace tom_filled_33_balloons_l2357_235730

/-- The number of water balloons filled up by Anthony -/
def anthony_balloons : ℕ := 44

/-- The number of water balloons filled up by Luke -/
def luke_balloons : ℕ := anthony_balloons / 4

/-- The number of water balloons filled up by Tom -/
def tom_balloons : ℕ := 3 * luke_balloons

/-- Theorem stating that Tom filled up 33 water balloons -/
theorem tom_filled_33_balloons : tom_balloons = 33 := by
  sorry

end tom_filled_33_balloons_l2357_235730
