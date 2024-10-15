import Mathlib

namespace NUMINAMATH_CALUDE_copy_machines_output_l2516_251690

/-- The number of copies made by two machines in a given time -/
def total_copies (rate1 rate2 time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

/-- Theorem stating that two machines with rates 25 and 55 copies per minute
    make 2400 copies in 30 minutes -/
theorem copy_machines_output : total_copies 25 55 30 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_copy_machines_output_l2516_251690


namespace NUMINAMATH_CALUDE_dinner_cost_is_120_l2516_251691

/-- Calculates the cost of dinner before tip given the total cost, ticket price, number of tickets, limo hourly rate, limo hours, and tip percentage. -/
def dinner_cost (total : ℚ) (ticket_price : ℚ) (num_tickets : ℕ) (limo_rate : ℚ) (limo_hours : ℕ) (tip_percent : ℚ) : ℚ :=
  let ticket_cost := ticket_price * num_tickets
  let limo_cost := limo_rate * limo_hours
  let dinner_with_tip := total - (ticket_cost + limo_cost)
  dinner_with_tip / (1 + tip_percent)

/-- Proves that the cost of dinner before tip is $120 given the specified conditions. -/
theorem dinner_cost_is_120 :
  dinner_cost 836 100 2 80 6 (30/100) = 120 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_is_120_l2516_251691


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2516_251617

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (12*x + 2)*(12*x + 6)*(12*x + 10)*(6*x + 3) = 864 * k) ∧
  (∀ (m : ℤ), m > 864 → ∃ (y : ℤ), Odd y ∧ ¬∃ (l : ℤ), (12*y + 2)*(12*y + 6)*(12*y + 10)*(6*y + 3) = m * l) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2516_251617


namespace NUMINAMATH_CALUDE_ratio_a5_b5_l2516_251669

/-- Given two arithmetic sequences a and b, S_n and T_n represent the sum of their first n terms respectively. -/
def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ := sorry

def T_n (b : ℕ → ℚ) (n : ℕ) : ℚ := sorry

/-- The ratio of S_n to T_n is equal to 7n / (n+3) for all n. -/
axiom ratio_condition {a b : ℕ → ℚ} (n : ℕ) : S_n a n / T_n b n = (7 * n) / (n + 3)

/-- The main theorem: given the ratio condition, the ratio of a_5 to b_5 is 21/4. -/
theorem ratio_a5_b5 {a b : ℕ → ℚ} : a 5 / b 5 = 21 / 4 := sorry

end NUMINAMATH_CALUDE_ratio_a5_b5_l2516_251669


namespace NUMINAMATH_CALUDE_entropy_increase_l2516_251696

-- Define the temperature in Kelvin
def T : ℝ := 298

-- Define the enthalpy change in kJ/mol
def ΔH : ℝ := 2171

-- Define the entropy change in J/(mol·K)
def ΔS : ℝ := 635.5

-- Theorem to prove that the entropy change is positive
theorem entropy_increase : ΔS > 0 := by
  sorry

end NUMINAMATH_CALUDE_entropy_increase_l2516_251696


namespace NUMINAMATH_CALUDE_unique_zero_point_b_range_l2516_251685

-- Define the function f_n
def f_n (n : ℕ) (b c : ℝ) (x : ℝ) : ℝ := x^n + b*x + c

-- Part I
theorem unique_zero_point (n : ℕ) (h : n ≥ 2) :
  ∃! x, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ f_n n 1 (-1) x = 0 :=
sorry

-- Part II
theorem b_range (h : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1 : ℝ) 1 → x₂ ∈ Set.Icc (-1 : ℝ) 1 →
  |f_n 2 b c x₁ - f_n 2 b c x₂| ≤ 4) :
  b ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_point_b_range_l2516_251685


namespace NUMINAMATH_CALUDE_school_teachers_count_l2516_251657

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total = 2400)
  (h2 : sample_size = 150)
  (h3 : students_in_sample = 135) :
  total - (total * students_in_sample / sample_size) = 240 :=
by sorry

end NUMINAMATH_CALUDE_school_teachers_count_l2516_251657


namespace NUMINAMATH_CALUDE_wallpapering_solution_l2516_251610

/-- Represents the number of days needed to complete the wallpapering job -/
structure WallpaperingJob where
  worker1 : ℝ  -- Days needed for worker 1 to complete the job alone
  worker2 : ℝ  -- Days needed for worker 2 to complete the job alone

/-- The wallpapering job satisfies the given conditions -/
def satisfies_conditions (job : WallpaperingJob) : Prop :=
  -- Worker 1 needs 3 days more than Worker 2
  job.worker1 = job.worker2 + 3 ∧
  -- The combined work of both workers in 7 days equals the whole job
  (7 / job.worker1) + (5.5 / job.worker2) = 1

/-- The theorem stating the solution to the wallpapering problem -/
theorem wallpapering_solution :
  ∃ (job : WallpaperingJob), satisfies_conditions job ∧ job.worker1 = 14 ∧ job.worker2 = 11 := by
  sorry


end NUMINAMATH_CALUDE_wallpapering_solution_l2516_251610


namespace NUMINAMATH_CALUDE_P_inter_Q_l2516_251654

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem P_inter_Q : P ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_P_inter_Q_l2516_251654


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2516_251677

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2516_251677


namespace NUMINAMATH_CALUDE_team_selection_count_l2516_251683

def total_students : ℕ := 11
def num_girls : ℕ := 3
def num_boys : ℕ := 8
def team_size : ℕ := 5

theorem team_selection_count :
  (Nat.choose total_students team_size) - (Nat.choose num_boys team_size) = 406 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l2516_251683


namespace NUMINAMATH_CALUDE_banana_cost_l2516_251689

theorem banana_cost (num_bananas : ℕ) (num_oranges : ℕ) (orange_cost : ℚ) (total_cost : ℚ) :
  num_bananas = 5 →
  num_oranges = 10 →
  orange_cost = 3/2 →
  total_cost = 25 →
  (total_cost - num_oranges * orange_cost) / num_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_banana_cost_l2516_251689


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l2516_251698

theorem mod_23_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 50238 ≡ n [ZMOD 23] ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l2516_251698


namespace NUMINAMATH_CALUDE_factor_x12_minus_1024_l2516_251665

theorem factor_x12_minus_1024 (x : ℝ) : 
  x^12 - 1024 = (x^6 + 32) * (x^3 + 4 * Real.sqrt 2) * (x^3 - 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_x12_minus_1024_l2516_251665


namespace NUMINAMATH_CALUDE_counterexample_exists_l2516_251613

theorem counterexample_exists : ∃ a : ℝ, (|a - 1| > 1) ∧ (a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2516_251613


namespace NUMINAMATH_CALUDE_two_digit_product_4320_l2516_251695

theorem two_digit_product_4320 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4320 ∧ a = 60 ∧ b = 72 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_4320_l2516_251695


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2516_251630

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 5*x + 4 = 0 ↔ x = 4 ∨ x = 1) ∧
  (∃ x : ℝ, x^2 = 4 - 2*x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2516_251630


namespace NUMINAMATH_CALUDE_symmetric_point_of_2_5_l2516_251645

/-- Given a point P(a,b) and a line with equation x+y=0, 
    the symmetric point Q(x,y) satisfies:
    1. x + y = 0 (lies on the line)
    2. The midpoint of PQ lies on the line
    3. PQ is perpendicular to the line -/
def is_symmetric_point (a b x y : ℝ) : Prop :=
  x + y = 0 ∧
  (a + x) / 2 + (b + y) / 2 = 0 ∧
  (x - a) = (b - y)

/-- The point symmetric to P(2,5) with respect to the line x+y=0 
    has coordinates (-5,-2) -/
theorem symmetric_point_of_2_5 : 
  is_symmetric_point 2 5 (-5) (-2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_2_5_l2516_251645


namespace NUMINAMATH_CALUDE_equation_solution_l2516_251608

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.4 * x = 2.1 * x - 30) ∧ (x = -35 / 1.3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2516_251608


namespace NUMINAMATH_CALUDE_cooking_time_calculation_l2516_251656

/-- Represents the cooking time for each food item -/
structure CookingTime where
  waffles : ℕ
  steak : ℕ
  chili : ℕ
  fries : ℕ

/-- Represents the quantity of each food item to be cooked -/
structure CookingQuantity where
  waffles : ℕ
  steak : ℕ
  chili : ℕ
  fries : ℕ

/-- Calculates the total cooking time given the cooking times and quantities -/
def totalCookingTime (time : CookingTime) (quantity : CookingQuantity) : ℕ :=
  time.waffles * quantity.waffles +
  time.steak * quantity.steak +
  time.chili * quantity.chili +
  time.fries * quantity.fries

/-- Theorem: Given the specified cooking times and quantities, the total cooking time is 218 minutes -/
theorem cooking_time_calculation (time : CookingTime) (quantity : CookingQuantity)
  (hw : time.waffles = 10)
  (hs : time.steak = 6)
  (hc : time.chili = 20)
  (hf : time.fries = 15)
  (qw : quantity.waffles = 5)
  (qs : quantity.steak = 8)
  (qc : quantity.chili = 3)
  (qf : quantity.fries = 4) :
  totalCookingTime time quantity = 218 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_calculation_l2516_251656


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2516_251603

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (c x y n : ℝ),
  -- Hyperbola equation
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  -- M is on the hyperbola
  c^2 / a^2 - (b^2 / a)^2 / b^2 = 1 ∧
  -- F is a focus
  c^2 = a^2 + b^2 ∧
  -- M is center of circle
  (x - c)^2 + (y - n)^2 = (b^2 / a)^2 ∧
  -- Circle tangent to x-axis at F
  n = b^2 / a ∧
  -- Circle intersects y-axis
  c^2 + n^2 = (2 * n)^2 ∧
  -- MPQ is equilateral
  c^2 = 3 * n^2 →
  -- Eccentricity is √3
  c / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2516_251603


namespace NUMINAMATH_CALUDE_intersection_range_l2516_251675

-- Define the curve
def curve (x y : ℝ) : Prop :=
  Real.sqrt (1 - (y - 1)^2) = abs x - 1

-- Define the line
def line (k x y : ℝ) : Prop :=
  k * x - y = 2

-- Define the intersection condition
def intersect_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (k ≥ -2 ∧ k < -4/3) ∨ (k > 4/3 ∧ k ≤ 2)

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersect_at_two_points k ↔ k_range k :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l2516_251675


namespace NUMINAMATH_CALUDE_jacoby_hourly_wage_l2516_251664

/-- Proves that Jacoby's hourly wage is $19 given the conditions of his savings and expenses --/
theorem jacoby_hourly_wage :
  let total_needed : ℕ := 5000
  let hours_worked : ℕ := 10
  let cookies_sold : ℕ := 24
  let cookie_price : ℕ := 4
  let lottery_win : ℕ := 500
  let sister_gift : ℕ := 500
  let remaining_needed : ℕ := 3214
  let hourly_wage : ℕ := (total_needed - remaining_needed - (cookies_sold * cookie_price) - lottery_win - 2 * sister_gift + 10) / hours_worked
  hourly_wage = 19
  := by sorry

end NUMINAMATH_CALUDE_jacoby_hourly_wage_l2516_251664


namespace NUMINAMATH_CALUDE_prob_second_good_is_five_ninths_l2516_251604

/-- Represents the number of good transistors initially in the box -/
def initial_good : ℕ := 6

/-- Represents the number of bad transistors initially in the box -/
def initial_bad : ℕ := 4

/-- Represents the total number of transistors initially in the box -/
def initial_total : ℕ := initial_good + initial_bad

/-- Represents the probability of selecting a good transistor as the second one,
    given that the first one selected was good -/
def prob_second_good : ℚ := (initial_good - 1) / (initial_total - 1)

theorem prob_second_good_is_five_ninths :
  prob_second_good = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_second_good_is_five_ninths_l2516_251604


namespace NUMINAMATH_CALUDE_intersection_complement_equals_one_l2516_251619

universe u

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem intersection_complement_equals_one : M ∩ (U \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_one_l2516_251619


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2516_251634

theorem sum_of_xyz (x y z : ℤ) 
  (hz : z = 4)
  (hxy : x + y = 7)
  (hxz : x + z = 8) : 
  x + y + z = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2516_251634


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2516_251693

/-- Proposition p -/
def p (x : ℝ) : Prop := x^2 - x - 20 > 0

/-- Proposition q -/
def q (x : ℝ) : Prop := |x| - 2 > 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2516_251693


namespace NUMINAMATH_CALUDE_power_mod_nine_l2516_251666

theorem power_mod_nine (x : ℤ) : x = 5 → x^46655 % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_nine_l2516_251666


namespace NUMINAMATH_CALUDE_blue_parrots_count_l2516_251614

/-- The number of blue parrots on Bird Island --/
def blue_parrots : ℕ := 38

/-- The total number of parrots on Bird Island after new arrivals --/
def total_parrots : ℕ := 150

/-- The fraction of red parrots --/
def red_fraction : ℚ := 1/2

/-- The fraction of green parrots --/
def green_fraction : ℚ := 1/4

/-- The number of new parrots that arrived --/
def new_parrots : ℕ := 30

theorem blue_parrots_count :
  blue_parrots = total_parrots - (red_fraction * total_parrots).floor - (green_fraction * total_parrots).floor :=
by sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l2516_251614


namespace NUMINAMATH_CALUDE_smaller_number_in_sum_l2516_251611

theorem smaller_number_in_sum (x y : ℕ) : 
  x + y = 84 → y = 3 * x → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_sum_l2516_251611


namespace NUMINAMATH_CALUDE_first_transfer_amount_l2516_251606

/-- Proves that the amount of the first bank transfer is approximately $91.18 given the initial and final balances and service charge. -/
theorem first_transfer_amount (initial_balance : ℝ) (final_balance : ℝ) (service_charge_rate : ℝ) :
  initial_balance = 400 →
  final_balance = 307 →
  service_charge_rate = 0.02 →
  ∃ (transfer_amount : ℝ), 
    initial_balance - (transfer_amount * (1 + service_charge_rate)) = final_balance ∧
    (transfer_amount ≥ 91.17 ∧ transfer_amount ≤ 91.19) :=
by sorry

end NUMINAMATH_CALUDE_first_transfer_amount_l2516_251606


namespace NUMINAMATH_CALUDE_aprils_roses_l2516_251640

theorem aprils_roses (initial_roses : ℕ) 
  (rose_price : ℕ) 
  (total_earnings : ℕ) 
  (roses_left : ℕ) : 
  rose_price = 4 → 
  total_earnings = 36 → 
  roses_left = 4 → 
  initial_roses = 13 := by
  sorry

end NUMINAMATH_CALUDE_aprils_roses_l2516_251640


namespace NUMINAMATH_CALUDE_total_amount_is_sum_of_shares_l2516_251676

/-- Represents the time in days it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℕ

/-- Represents the share of money received by a person -/
structure Share where
  amount : ℕ

/-- Represents a worker with their individual work time and share -/
structure Worker where
  workTime : WorkTime
  share : Share

/-- Theorem: The total amount received for the work is the sum of individual shares -/
theorem total_amount_is_sum_of_shares 
  (a b c : Worker)
  (h1 : a.workTime.days = 6)
  (h2 : b.workTime.days = 8)
  (h3 : a.share.amount = 300)
  (h4 : b.share.amount = 225)
  (h5 : c.share.amount = 75) :
  a.share.amount + b.share.amount + c.share.amount = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_sum_of_shares_l2516_251676


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2516_251699

def f (x : ℝ) : ℝ := (x - 2) * (x - 1) * (x + 1)

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (-x)

def c : ℕ := 3

def d : ℕ := 2

theorem intersection_points_theorem : 10 * c + d = 32 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2516_251699


namespace NUMINAMATH_CALUDE_recliner_price_drop_l2516_251648

theorem recliner_price_drop 
  (initial_quantity : ℝ) 
  (initial_price : ℝ) 
  (quantity_increase_ratio : ℝ) 
  (revenue_increase_ratio : ℝ) 
  (h1 : quantity_increase_ratio = 1.60) 
  (h2 : revenue_increase_ratio = 1.2800000000000003) : 
  let new_quantity := initial_quantity * quantity_increase_ratio
  let new_price := initial_price * (revenue_increase_ratio / quantity_increase_ratio)
  new_price / initial_price = 0.80 := by
sorry

end NUMINAMATH_CALUDE_recliner_price_drop_l2516_251648


namespace NUMINAMATH_CALUDE_phone_inventory_and_profit_optimization_l2516_251680

/-- Represents a phone model with purchase and selling prices -/
structure PhoneModel where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the inventory and financial data of a business hall -/
structure BusinessHall where
  modelA : PhoneModel
  modelB : PhoneModel
  totalSpent : ℕ
  totalProfit : ℕ

/-- Theorem stating the correct number of units purchased and maximum profit -/
theorem phone_inventory_and_profit_optimization 
  (hall : BusinessHall) 
  (hall_data : hall.modelA.purchasePrice = 3000 ∧ 
               hall.modelA.sellingPrice = 3400 ∧
               hall.modelB.purchasePrice = 3500 ∧ 
               hall.modelB.sellingPrice = 4000 ∧
               hall.totalSpent = 32000 ∧ 
               hall.totalProfit = 4400) :
  (∃ (a b : ℕ), 
    a * hall.modelA.purchasePrice + b * hall.modelB.purchasePrice = hall.totalSpent ∧
    a * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    b * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) = hall.totalProfit ∧
    a = 6 ∧ b = 4) ∧
  (∃ (x : ℕ), 
    x ≥ 10 ∧ 30 - x ≤ 2 * x ∧
    ∀ y : ℕ, y ≥ 10 → 30 - y ≤ 2 * y → 
    x * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - x) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) ≥
    y * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - y) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) ∧
    x * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - x) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) = 14000) :=
by sorry

end NUMINAMATH_CALUDE_phone_inventory_and_profit_optimization_l2516_251680


namespace NUMINAMATH_CALUDE_equation_solutions_l2516_251621

theorem equation_solutions : 
  ∀ m n : ℕ, 20^m - 10*m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2516_251621


namespace NUMINAMATH_CALUDE_function_derivative_problem_l2516_251688

theorem function_derivative_problem (a : ℝ) :
  (∀ x, f x = (2 * x + a) ^ 2) →
  (deriv f) 2 = 20 →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_derivative_problem_l2516_251688


namespace NUMINAMATH_CALUDE_square_ratio_bounds_l2516_251622

theorem square_ratio_bounds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ∃ (m M : ℝ), 
    (0 ≤ m) ∧ 
    (M ≤ 1) ∧ 
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → m ≤ ((|z + w| / (|z| + |w|))^2) ∧ ((|z + w| / (|z| + |w|))^2) ≤ M) ∧
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ((|a + b| / (|a| + |b|))^2) = m) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ ((|c + d| / (|c| + |d|))^2) = M) ∧
    (M - m = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_bounds_l2516_251622


namespace NUMINAMATH_CALUDE_baron_munchausen_contradiction_l2516_251653

-- Define the total distance and time of the walk
variable (S : ℝ) -- Total distance
variable (T : ℝ) -- Total time

-- Define the speeds
def speed1 : ℝ := 5 -- Speed for half the distance
def speed2 : ℝ := 6 -- Speed for half the time

-- Theorem: It's impossible to satisfy both conditions
theorem baron_munchausen_contradiction :
  ¬(∃ (S T : ℝ), S > 0 ∧ T > 0 ∧
    (S / 2) / speed1 + (S / 2) / speed2 = T ∧
    (S / 2) + speed2 * (T / 2) = S) :=
sorry

end NUMINAMATH_CALUDE_baron_munchausen_contradiction_l2516_251653


namespace NUMINAMATH_CALUDE_product_mod_seven_l2516_251638

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2516_251638


namespace NUMINAMATH_CALUDE_classroom_students_count_l2516_251667

theorem classroom_students_count : ∃! n : ℕ, n < 60 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ n = 22 := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_count_l2516_251667


namespace NUMINAMATH_CALUDE_remainder_3_180_mod_5_l2516_251639

theorem remainder_3_180_mod_5 : 3^180 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_180_mod_5_l2516_251639


namespace NUMINAMATH_CALUDE_board_game_change_l2516_251612

theorem board_game_change (num_games : ℕ) (game_cost : ℕ) (payment : ℕ) (change_bill : ℕ) : 
  num_games = 8 →
  game_cost = 18 →
  payment = 200 →
  change_bill = 10 →
  (payment - num_games * game_cost) / change_bill = 5 := by
sorry

end NUMINAMATH_CALUDE_board_game_change_l2516_251612


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l2516_251642

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

def replace_greatest_with_one (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  let max_digit := max d1 (max d2 d3)
  if d1 = max_digit then
    100 + d2 * 10 + d3
  else if d2 = max_digit then
    d1 * 100 + 10 + d3
  else
    d1 * 100 + d2 * 10 + 1

theorem special_numbers_theorem :
  {n : ℕ | is_three_digit n ∧ 
           has_distinct_digits n ∧ 
           (replace_greatest_with_one n) % 30 = 0} =
  {230, 320, 560, 650, 890, 980} := by sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l2516_251642


namespace NUMINAMATH_CALUDE_student_count_l2516_251620

/-- Given a group of students where replacing one student changes the average weight,
    this theorem proves the total number of students. -/
theorem student_count
  (avg_decrease : ℝ)  -- The decrease in average weight
  (old_weight : ℝ)    -- Weight of the replaced student
  (new_weight : ℝ)    -- Weight of the new student
  (h1 : avg_decrease = 5)  -- The average weight decreases by 5 kg
  (h2 : old_weight = 86)   -- The replaced student weighs 86 kg
  (h3 : new_weight = 46)   -- The new student weighs 46 kg
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2516_251620


namespace NUMINAMATH_CALUDE_vector_a_solution_l2516_251632

theorem vector_a_solution (a b : ℝ × ℝ) : 
  b = (1, 2) → 
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  (a.1^2 + a.2^2 = 20) → 
  (a = (4, -2) ∨ a = (-4, 2)) := by
sorry

end NUMINAMATH_CALUDE_vector_a_solution_l2516_251632


namespace NUMINAMATH_CALUDE_solution_set_m_2_range_of_m_l2516_251692

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| + |2 * x + m|

-- Theorem 1: Solution set for f(x) ≤ 3 when m = 2
theorem solution_set_m_2 :
  {x : ℝ | f x 2 ≤ 3} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem 2: Range of m values for f(x) ≤ |2x - 3| with x ∈ [0, 1]
theorem range_of_m :
  {m : ℝ | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x m ≤ |2 * x - 3|} = {m : ℝ | -3 ≤ m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_m_2_range_of_m_l2516_251692


namespace NUMINAMATH_CALUDE_profit_share_calculation_l2516_251650

theorem profit_share_calculation (investment_A investment_B investment_C : ℕ)
  (profit_difference_AC : ℕ) (profit_share_B : ℕ) :
  investment_A = 6000 →
  investment_B = 8000 →
  investment_C = 10000 →
  profit_difference_AC = 500 →
  profit_share_B = 1000 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_calculation_l2516_251650


namespace NUMINAMATH_CALUDE_g_composition_of_three_l2516_251631

def g (x : ℝ) : ℝ := 7 * x - 3

theorem g_composition_of_three : g (g (g 3)) = 858 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l2516_251631


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2516_251647

/-- A rhombus with perimeter 32 has side length 8 -/
theorem rhombus_side_length (perimeter : ℝ) (h1 : perimeter = 32) : perimeter / 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2516_251647


namespace NUMINAMATH_CALUDE_typist_salary_problem_l2516_251628

theorem typist_salary_problem (x : ℝ) : 
  (x * 1.1 * 0.95 = 6270) → x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l2516_251628


namespace NUMINAMATH_CALUDE_product_mod_eight_l2516_251633

theorem product_mod_eight : (55 * 57) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eight_l2516_251633


namespace NUMINAMATH_CALUDE_cube_surface_area_l2516_251673

theorem cube_surface_area (V : ℝ) (h : V = 64) : 
  6 * (V ^ (1/3))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2516_251673


namespace NUMINAMATH_CALUDE_income_analysis_l2516_251652

/-- Represents the income status of a household -/
inductive IncomeStatus
| Above10000
| Below10000

/-- Represents a region with households -/
structure Region where
  totalHouseholds : ℕ
  aboveThreshold : ℕ

/-- Represents the sample data -/
structure SampleData where
  regionA : Region
  regionB : Region
  totalSample : ℕ

/-- The probability of selecting a household with income above 10000 from a region -/
def probAbove10000 (r : Region) : ℚ :=
  r.aboveThreshold / r.totalHouseholds

/-- The expected value of X (number of households with income > 10000 when selecting one from each region) -/
def expectedX (sd : SampleData) : ℚ :=
  (probAbove10000 sd.regionA + probAbove10000 sd.regionB) / 2

/-- The main theorem to be proved -/
theorem income_analysis (sd : SampleData)
  (h1 : sd.regionA.totalHouseholds = 300)
  (h2 : sd.regionA.aboveThreshold = 100)
  (h3 : sd.regionB.totalHouseholds = 200)
  (h4 : sd.regionB.aboveThreshold = 150)
  (h5 : sd.totalSample = 500) :
  probAbove10000 sd.regionA = 1/3 ∧ expectedX sd = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_income_analysis_l2516_251652


namespace NUMINAMATH_CALUDE_hexagon_extended_point_distance_l2516_251660

/-- Regular hexagon with side length 1 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (is_regular : ∀ (X Y : ℝ × ℝ), (X, Y) ∈ [(A, B), (B, C), (C, D), (D, E), (E, F), (F, A)] → dist X Y = 1)

/-- Point Y extended from A such that BY = 4AB -/
def extend_point (h : RegularHexagon) (Y : ℝ × ℝ) : Prop :=
  dist h.B Y = 4 * dist h.A h.B

/-- The length of segment EY is √21 -/
theorem hexagon_extended_point_distance (h : RegularHexagon) (Y : ℝ × ℝ) 
  (h_extend : extend_point h Y) : 
  dist h.E Y = Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_hexagon_extended_point_distance_l2516_251660


namespace NUMINAMATH_CALUDE_range_of_f_inequality_l2516_251605

open Real

noncomputable def f (x : ℝ) : ℝ := 2*x + sin x

theorem range_of_f_inequality (h1 : ∀ x ∈ Set.Ioo (-2) 2, HasDerivAt f (2 + cos x) x)
                              (h2 : f 0 = 0) :
  {x : ℝ | f (1 + x) + f (x - x^2) > 0} = Set.Ioo (1 - Real.sqrt 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_inequality_l2516_251605


namespace NUMINAMATH_CALUDE_rotation_exists_l2516_251672

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line
  mk :: -- Constructor

/-- Represents a point in 3D space -/
structure Point3D where
  -- Add necessary fields for a 3D point
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane
  mk :: -- Constructor

/-- Represents a rotation in 3D space -/
structure Rotation3D where
  -- Add necessary fields for a 3D rotation
  mk :: -- Constructor
  apply : Point3D → Point3D  -- Applies the rotation to a point

def are_skew (l1 l2 : Line3D) : Prop := sorry

def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

def rotation_maps (r : Rotation3D) (l1 l2 : Line3D) (p1 p2 : Point3D) : Prop := sorry

def plane_of_symmetry (p1 p2 : Point3D) : Plane3D := sorry

def plane_intersection (p1 p2 : Plane3D) : Line3D := sorry

theorem rotation_exists (a a' : Line3D) (A : Point3D) (A' : Point3D) 
  (h1 : are_skew a a')
  (h2 : point_on_line A a)
  (h3 : point_on_line A' a') :
  ∃ (l : Line3D), ∃ (r : Rotation3D), rotation_maps r a a' A A' := by
  sorry

end NUMINAMATH_CALUDE_rotation_exists_l2516_251672


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2516_251679

theorem x_plus_y_values (x y : ℝ) (hx : |x| = 3) (hy : |y| = 6) (hxy : x > y) :
  (x + y = -3 ∨ x + y = -9) ∧ ∀ z, (x + y = z → z = -3 ∨ z = -9) :=
by sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2516_251679


namespace NUMINAMATH_CALUDE_tiffany_green_buckets_l2516_251629

/-- Carnival ring toss game -/
structure CarnivalGame where
  total_money : ℕ
  cost_per_play : ℕ
  rings_per_play : ℕ
  red_bucket_points : ℕ
  green_bucket_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  total_points : ℕ

/-- Calculate the number of green buckets hit -/
def green_buckets_hit (game : CarnivalGame) : ℕ :=
  (game.total_points - game.red_buckets_hit * game.red_bucket_points) / game.green_bucket_points

/-- Theorem: Tiffany hit 10 green buckets -/
theorem tiffany_green_buckets :
  let game : CarnivalGame := {
    total_money := 3,
    cost_per_play := 1,
    rings_per_play := 5,
    red_bucket_points := 2,
    green_bucket_points := 3,
    games_played := 2,
    red_buckets_hit := 4,
    total_points := 38
  }
  green_buckets_hit game = 10 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_green_buckets_l2516_251629


namespace NUMINAMATH_CALUDE_no_absolute_winner_probability_l2516_251627

/-- Represents a player in the mini-tournament -/
inductive Player : Type
| Alyosha : Player
| Borya : Player
| Vasya : Player

/-- Represents the result of a match between two players -/
def MatchResult := Player → Player → ℝ

/-- The probability that there is no absolute winner in the mini-tournament -/
def noAbsoluteWinnerProbability (matchResult : MatchResult) : ℝ :=
  let p_AB := matchResult Player.Alyosha Player.Borya
  let p_BV := matchResult Player.Borya Player.Vasya
  0.24 * (1 - p_AB) * (1 - p_BV) + 0.36 * p_AB * (1 - p_BV)

/-- The main theorem stating that the probability of no absolute winner is 0.36 -/
theorem no_absolute_winner_probability (matchResult : MatchResult) 
  (h1 : matchResult Player.Alyosha Player.Borya = 0.6)
  (h2 : matchResult Player.Borya Player.Vasya = 0.4) :
  noAbsoluteWinnerProbability matchResult = 0.36 := by
  sorry


end NUMINAMATH_CALUDE_no_absolute_winner_probability_l2516_251627


namespace NUMINAMATH_CALUDE_travel_ratio_l2516_251663

theorem travel_ratio (george joseph patrick zack : ℕ) : 
  george = 6 →
  joseph = george / 2 →
  patrick = joseph * 3 →
  zack = 18 →
  zack / patrick = 2 :=
by sorry

end NUMINAMATH_CALUDE_travel_ratio_l2516_251663


namespace NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l2516_251637

/-- Revenue function for gadget sales -/
def R (p : ℝ) : ℝ := p * (200 - 4 * p)

/-- The price that maximizes revenue -/
def optimal_price : ℝ := 25

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℝ, p ≤ 40 → R p ≤ R optimal_price := by sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l2516_251637


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2516_251681

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 4))) = 11 / 29 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2516_251681


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_twelve_l2516_251646

-- Define the function f
def f (x : ℝ) : ℝ := x * |x| + 3 * x

-- State the theorem
theorem inverse_sum_equals_negative_twelve :
  ∃ (a b : ℝ), f a = 9 ∧ f b = -121 ∧ a + b = -12 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_twelve_l2516_251646


namespace NUMINAMATH_CALUDE_original_list_size_l2516_251671

/-- The number of integers in the original list -/
def n : ℕ := sorry

/-- The mean of the original list -/
def m : ℚ := sorry

/-- The sum of the integers in the original list -/
def original_sum : ℚ := n * m

/-- The equation representing the first condition -/
axiom first_condition : (m + 2) * (n + 1) = original_sum + 15

/-- The equation representing the second condition -/
axiom second_condition : (m + 1) * (n + 2) = original_sum + 16

theorem original_list_size : n = 4 := by sorry

end NUMINAMATH_CALUDE_original_list_size_l2516_251671


namespace NUMINAMATH_CALUDE_pierre_cake_consumption_l2516_251651

theorem pierre_cake_consumption (cake_weight : ℝ) (num_parts : ℕ) 
  (nathalie_parts : ℝ) (pierre_multiplier : ℝ) : 
  cake_weight = 400 → 
  num_parts = 8 → 
  nathalie_parts = 1 / 8 → 
  pierre_multiplier = 2 → 
  pierre_multiplier * (nathalie_parts * cake_weight) = 100 := by
  sorry

end NUMINAMATH_CALUDE_pierre_cake_consumption_l2516_251651


namespace NUMINAMATH_CALUDE_fenced_area_with_cutouts_l2516_251661

theorem fenced_area_with_cutouts (yard_length yard_width cutout1_side cutout2_side : ℝ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 15)
  (h3 : cutout1_side = 4)
  (h4 : cutout2_side = 2) :
  yard_length * yard_width - (cutout1_side * cutout1_side + cutout2_side * cutout2_side) = 280 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_with_cutouts_l2516_251661


namespace NUMINAMATH_CALUDE_calculate_expression_l2516_251697

theorem calculate_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^4 + 3*x^2 - 2*y + 2*y^2) / 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2516_251697


namespace NUMINAMATH_CALUDE_ratio_HD_HA_is_5_11_l2516_251623

/-- A triangle with sides of lengths 13, 14, and 15 -/
structure Triangle :=
  (a b c : ℝ)
  (side_a : a = 13)
  (side_b : b = 14)
  (side_c : c = 15)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The altitude from vertex A to the side of length 14 -/
def altitude_AD (t : Triangle) : ℝ := sorry

/-- The ratio of HD to HA -/
def ratio_HD_HA (t : Triangle) : ℚ := sorry

/-- Theorem: The ratio HD:HA is 5:11 -/
theorem ratio_HD_HA_is_5_11 (t : Triangle) : 
  ratio_HD_HA t = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_ratio_HD_HA_is_5_11_l2516_251623


namespace NUMINAMATH_CALUDE_intersection_equals_singleton_two_l2516_251601

def M : Set ℤ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

theorem intersection_equals_singleton_two : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_singleton_two_l2516_251601


namespace NUMINAMATH_CALUDE_remove_two_gives_eight_point_five_l2516_251602

def original_list : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def remove_number (list : List ℕ) (n : ℕ) : List ℕ :=
  list.filter (· ≠ n)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem remove_two_gives_eight_point_five :
  average (remove_number original_list 2) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_two_gives_eight_point_five_l2516_251602


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2516_251678

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z := i^2018 / (i^2019 - 1)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2516_251678


namespace NUMINAMATH_CALUDE_exactly_two_successes_out_of_three_l2516_251600

/-- The probability of making a successful shot -/
def p : ℚ := 2 / 3

/-- The number of attempts -/
def n : ℕ := 3

/-- The number of successful shots -/
def k : ℕ := 2

/-- The probability of making exactly k successful shots out of n attempts -/
def probability_k_successes : ℚ := 
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem exactly_two_successes_out_of_three : 
  probability_k_successes = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_successes_out_of_three_l2516_251600


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l2516_251616

theorem correct_quotient_proof (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 42 →  -- Dividing by 12 (incorrect divisor) yields 42
  D / 21 = 24  -- The correct quotient when dividing by 21 is 24
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l2516_251616


namespace NUMINAMATH_CALUDE_num_al_sandwiches_l2516_251659

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 6

/-- Represents whether turkey is available. -/
def has_turkey : Prop := True

/-- Represents whether roast beef is available. -/
def has_roast_beef : Prop := True

/-- Represents whether swiss cheese is available. -/
def has_swiss_cheese : Prop := True

/-- Represents whether rye bread is available. -/
def has_rye_bread : Prop := True

/-- Represents the number of sandwich combinations with turkey and swiss cheese. -/
def turkey_swiss_combos : ℕ := num_bread

/-- Represents the number of sandwich combinations with rye bread and roast beef. -/
def rye_roast_beef_combos : ℕ := num_cheese

/-- Theorem stating the number of different sandwiches Al can order. -/
theorem num_al_sandwiches : 
  num_bread * num_meat * num_cheese - turkey_swiss_combos - rye_roast_beef_combos = 199 := by
  sorry

end NUMINAMATH_CALUDE_num_al_sandwiches_l2516_251659


namespace NUMINAMATH_CALUDE_exists_n_congruence_l2516_251674

/-- ν(n) denotes the exponent of 2 in the prime factorization of n! -/
def ν (n : ℕ) : ℕ := sorry

/-- For any positive integers a and m, there exists an integer n > 1 such that ν(n) ≡ a (mod m) -/
theorem exists_n_congruence (a m : ℕ+) : ∃ n : ℕ, n > 1 ∧ ν n % m = a % m := by
  sorry

end NUMINAMATH_CALUDE_exists_n_congruence_l2516_251674


namespace NUMINAMATH_CALUDE_a_explicit_formula_l2516_251682

/-- Sequence {a_n} defined recursively --/
def a : ℕ → ℚ
  | 0 => 0
  | n + 1 => a n + (n + 1)^3

/-- Theorem stating the explicit formula for a_n --/
theorem a_explicit_formula (n : ℕ) : a n = n^2 * (n + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_a_explicit_formula_l2516_251682


namespace NUMINAMATH_CALUDE_value_is_appropriate_for_project_assessment_other_terms_not_appropriate_l2516_251607

-- Define the possible options
inductive ProjectAssessmentTerm
  | Price
  | Value
  | Cost
  | Expense

-- Define a function that determines if a term is appropriate for project assessment
def isAppropriateForProjectAssessment (term : ProjectAssessmentTerm) : Prop :=
  match term with
  | ProjectAssessmentTerm.Value => True
  | _ => False

-- Theorem stating that "Value" is the appropriate term
theorem value_is_appropriate_for_project_assessment :
  isAppropriateForProjectAssessment ProjectAssessmentTerm.Value :=
by sorry

-- Theorem stating that other terms are not appropriate
theorem other_terms_not_appropriate (term : ProjectAssessmentTerm) :
  term ≠ ProjectAssessmentTerm.Value →
  ¬(isAppropriateForProjectAssessment term) :=
by sorry

end NUMINAMATH_CALUDE_value_is_appropriate_for_project_assessment_other_terms_not_appropriate_l2516_251607


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l2516_251641

theorem factorial_ratio_simplification (N : ℕ) (h : N ≥ 2) :
  (Nat.factorial (N - 2) * (N - 1) * N) / Nat.factorial (N + 2) = 1 / ((N + 2) * (N + 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l2516_251641


namespace NUMINAMATH_CALUDE_expression_simplification_l2516_251618

theorem expression_simplification (y : ℝ) : 7*y + 8 - 2*y + 15 = 5*y + 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2516_251618


namespace NUMINAMATH_CALUDE_f_extrema_l2516_251615

def f (x : ℝ) := x^2 - 2*x

theorem f_extrema :
  ∀ x ∈ Set.Icc (-1 : ℝ) 5,
    -1 ≤ f x ∧ f x ≤ 15 ∧
    (∃ x₁ ∈ Set.Icc (-1 : ℝ) 5, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-1 : ℝ) 5, f x₂ = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l2516_251615


namespace NUMINAMATH_CALUDE_successful_hatch_percentage_l2516_251662

/-- The number of eggs laid by each turtle -/
def eggs_per_turtle : ℕ := 20

/-- The number of turtles -/
def num_turtles : ℕ := 6

/-- The number of hatchlings produced -/
def num_hatchlings : ℕ := 48

/-- The percentage of eggs that successfully hatch -/
def hatch_percentage : ℚ := 40

theorem successful_hatch_percentage :
  (eggs_per_turtle * num_turtles : ℚ) * (hatch_percentage / 100) = num_hatchlings :=
sorry

end NUMINAMATH_CALUDE_successful_hatch_percentage_l2516_251662


namespace NUMINAMATH_CALUDE_sine_cosine_roots_l2516_251649

theorem sine_cosine_roots (θ : Real) (k : Real) 
  (h1 : θ > 0 ∧ θ < 2 * Real.pi)
  (h2 : (Real.sin θ)^2 - k * (Real.sin θ) + k + 1 = 0)
  (h3 : (Real.cos θ)^2 - k * (Real.cos θ) + k + 1 = 0) :
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_l2516_251649


namespace NUMINAMATH_CALUDE_megan_country_albums_l2516_251624

theorem megan_country_albums :
  ∀ (country_albums pop_albums total_songs songs_per_album : ℕ),
    pop_albums = 8 →
    songs_per_album = 7 →
    total_songs = 70 →
    total_songs = country_albums * songs_per_album + pop_albums * songs_per_album →
    country_albums = 2 := by
  sorry

end NUMINAMATH_CALUDE_megan_country_albums_l2516_251624


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2516_251625

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_product : a 3 * a 7 = 8) : 
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2516_251625


namespace NUMINAMATH_CALUDE_tommy_steaks_l2516_251643

/-- The number of steaks needed for a family dinner -/
def steaks_needed (family_members : ℕ) (pounds_per_member : ℕ) (ounces_per_steak : ℕ) : ℕ :=
  let total_ounces := family_members * pounds_per_member * 16
  (total_ounces + ounces_per_steak - 1) / ounces_per_steak

/-- Theorem: Tommy needs to buy 4 steaks for his family -/
theorem tommy_steaks : steaks_needed 5 1 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommy_steaks_l2516_251643


namespace NUMINAMATH_CALUDE_rectangle_Q_coordinates_l2516_251670

/-- A rectangle in a 2D plane --/
structure Rectangle where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The specific rectangle from the problem --/
def problemRectangle : Rectangle where
  O := (0, 0)
  P := (0, 3)
  R := (5, 0)
  Q := (5, 3)  -- We'll prove this is correct

/-- Predicate to check if four points form a rectangle --/
def isRectangle (rect : Rectangle) : Prop :=
  -- Opposite sides are parallel and equal in length
  (rect.O.1 = rect.P.1 ∧ rect.Q.1 = rect.R.1) ∧
  (rect.O.2 = rect.R.2 ∧ rect.P.2 = rect.Q.2) ∧
  (rect.P.1 - rect.O.1)^2 + (rect.P.2 - rect.O.2)^2 =
  (rect.Q.1 - rect.R.1)^2 + (rect.Q.2 - rect.R.2)^2 ∧
  (rect.R.1 - rect.O.1)^2 + (rect.R.2 - rect.O.2)^2 =
  (rect.Q.1 - rect.P.1)^2 + (rect.Q.2 - rect.P.2)^2

theorem rectangle_Q_coordinates :
  isRectangle problemRectangle →
  problemRectangle.Q = (5, 3) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_Q_coordinates_l2516_251670


namespace NUMINAMATH_CALUDE_fifth_term_is_eight_l2516_251658

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci_like n + fibonacci_like (n + 1)

theorem fifth_term_is_eight : fibonacci_like 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_eight_l2516_251658


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2516_251626

theorem quadratic_real_roots_condition (k : ℝ) : 
  (k ≠ 0) → 
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ 
  (k ≤ 1/4 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2516_251626


namespace NUMINAMATH_CALUDE_max_white_rooks_8x8_l2516_251668

/-- Represents a chessboard configuration with black and white rooks -/
structure ChessboardConfig where
  size : Nat
  blackRooks : Nat
  whiteRooks : Nat
  differentCells : Bool
  onlyAttackOpposite : Bool

/-- Defines the maximum number of white rooks for a given configuration -/
def maxWhiteRooks (config : ChessboardConfig) : Nat :=
  sorry

/-- Theorem stating the maximum number of white rooks for the given configuration -/
theorem max_white_rooks_8x8 :
  let config : ChessboardConfig := {
    size := 8,
    blackRooks := 6,
    whiteRooks := 14,
    differentCells := true,
    onlyAttackOpposite := true
  }
  maxWhiteRooks config = 14 := by sorry

end NUMINAMATH_CALUDE_max_white_rooks_8x8_l2516_251668


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2516_251635

/-- The original plane equation -/
def plane_a (x y z : ℝ) : Prop := 2*x + 3*y + z - 1 = 0

/-- The similarity transformation with scale factor k -/
def similarity_transform (k : ℝ) (x y z : ℝ) : Prop := 2*x + 3*y + z - k = 0

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (1, 2, -1)

/-- The scale factor -/
def k : ℝ := 2

/-- Theorem stating that point A does not lie on the transformed plane -/
theorem point_not_on_transformed_plane : 
  ¬ similarity_transform k point_A.1 point_A.2.1 point_A.2.2 :=
sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2516_251635


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l2516_251609

theorem square_area_equal_perimeter (a b c s : ℝ) : 
  a = 6 → b = 8 → c = 10 → -- Triangle side lengths
  a^2 + b^2 = c^2 →        -- Right-angled triangle condition
  4 * s = a + b + c →      -- Equal perimeter condition
  s^2 = 36 :=              -- Square area
by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l2516_251609


namespace NUMINAMATH_CALUDE_krishans_money_l2516_251694

theorem krishans_money (x y : ℝ) (ram gopal krishan : ℝ) : 
  ram = 1503 →
  ram + gopal + krishan = 15000 →
  ram / (7 * x) = gopal / (17 * x) →
  ram / (7 * x) = krishan / (17 * y) →
  ∃ ε > 0, |krishan - 9845| < ε :=
by sorry

end NUMINAMATH_CALUDE_krishans_money_l2516_251694


namespace NUMINAMATH_CALUDE_H_surjective_l2516_251636

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2| + x

-- State the theorem
theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l2516_251636


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l2516_251687

theorem sum_of_distinct_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l2516_251687


namespace NUMINAMATH_CALUDE_inequalities_on_positive_reals_l2516_251684

theorem inequalities_on_positive_reals :
  ∀ x : ℝ, x > 0 →
    (Real.log x < x) ∧
    (Real.sin x < x) ∧
    (Real.exp x > x) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_on_positive_reals_l2516_251684


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l2516_251686

theorem cubic_equation_solutions_mean (x : ℝ) : 
  (x^3 + 5*x^2 - 14*x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, y^3 + 5*y^2 - 14*y = 0) ∧ 
                   (∀ z, z^3 + 5*z^2 - 14*z = 0 → z ∈ s) ∧
                   (Finset.sum s id / s.card = -5/3)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l2516_251686


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2516_251644

theorem fractional_equation_solution :
  ∃ x : ℝ, (2 * x) / (x - 3) = 1 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2516_251644


namespace NUMINAMATH_CALUDE_train_length_calculation_l2516_251655

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes to pass the man -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (pass_time : ℝ) :
  train_speed = 60 →
  man_speed = 6 →
  pass_time = 32.99736021118311 →
  ∃ (train_length : ℝ), 
    (train_length ≥ 604.99 ∧ train_length ≤ 605.01) ∧
    train_length = (train_speed + man_speed) * (5 / 18) * pass_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2516_251655
