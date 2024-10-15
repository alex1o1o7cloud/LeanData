import Mathlib

namespace NUMINAMATH_CALUDE_cooking_dishes_time_is_one_point_five_l1657_165712

/-- Represents the daily schedule of a working mom -/
structure DailySchedule where
  total_awake_time : ℝ
  work_time : ℝ
  gym_time : ℝ
  bathing_time : ℝ
  homework_bedtime : ℝ
  packing_lunches : ℝ
  cleaning_time : ℝ
  shower_leisure : ℝ

/-- Calculates the time spent on cooking and dishes -/
def cooking_dishes_time (schedule : DailySchedule) : ℝ :=
  schedule.total_awake_time - (schedule.work_time + schedule.gym_time + 
  schedule.bathing_time + schedule.homework_bedtime + schedule.packing_lunches + 
  schedule.cleaning_time + schedule.shower_leisure)

/-- Theorem stating that the cooking and dishes time for the given schedule is 1.5 hours -/
theorem cooking_dishes_time_is_one_point_five (schedule : DailySchedule) 
  (h1 : schedule.total_awake_time = 16)
  (h2 : schedule.work_time = 8)
  (h3 : schedule.gym_time = 2)
  (h4 : schedule.bathing_time = 0.5)
  (h5 : schedule.homework_bedtime = 1)
  (h6 : schedule.packing_lunches = 0.5)
  (h7 : schedule.cleaning_time = 0.5)
  (h8 : schedule.shower_leisure = 2) :
  cooking_dishes_time schedule = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cooking_dishes_time_is_one_point_five_l1657_165712


namespace NUMINAMATH_CALUDE_equation_solution_l1657_165747

def equation (x : ℝ) : Prop :=
  (45 * x)^2 = (0.45 * 1200) * 80 / (12 + 4 * 3)

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ abs (x - 0.942808153803174) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1657_165747


namespace NUMINAMATH_CALUDE_prove_triangle_cotangent_formula_l1657_165726

def triangle_cotangent_formula (A B C a b c p r S : Real) : Prop :=
  let ctg_half (x : Real) := (p - x) / r
  A + B + C = Real.pi ∧
  p = (a + b + c) / 2 ∧
  S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧
  S = p * r ∧
  ctg_half a + ctg_half b + ctg_half c = ctg_half a * ctg_half b * ctg_half c

theorem prove_triangle_cotangent_formula (A B C a b c p r S : Real) :
  triangle_cotangent_formula A B C a b c p r S := by
  sorry

end NUMINAMATH_CALUDE_prove_triangle_cotangent_formula_l1657_165726


namespace NUMINAMATH_CALUDE_truck_driver_earnings_l1657_165799

/-- Calculates the net earnings of a truck driver given specific conditions --/
theorem truck_driver_earnings
  (gas_cost : ℝ)
  (fuel_efficiency : ℝ)
  (driving_speed : ℝ)
  (payment_rate : ℝ)
  (driving_duration : ℝ)
  (h1 : gas_cost = 2)
  (h2 : fuel_efficiency = 10)
  (h3 : driving_speed = 30)
  (h4 : payment_rate = 0.5)
  (h5 : driving_duration = 10)
  : ∃ (net_earnings : ℝ), net_earnings = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_truck_driver_earnings_l1657_165799


namespace NUMINAMATH_CALUDE_train_crossing_time_l1657_165769

/-- A train problem -/
theorem train_crossing_time
  (train_speed : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_speed = 20)
  (h2 : platform_length = 300)
  (h3 : platform_crossing_time = 30) :
  let train_length := train_speed * platform_crossing_time - platform_length
  (train_length / train_speed) = 15 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1657_165769


namespace NUMINAMATH_CALUDE_jons_website_hours_l1657_165794

theorem jons_website_hours (earnings_per_visit : ℚ) (visits_per_hour : ℕ) 
  (monthly_earnings : ℚ) (days_in_month : ℕ) 
  (h1 : earnings_per_visit = 1/10) 
  (h2 : visits_per_hour = 50) 
  (h3 : monthly_earnings = 3600) 
  (h4 : days_in_month = 30) : 
  (monthly_earnings / earnings_per_visit / visits_per_hour) / days_in_month = 24 := by
  sorry

end NUMINAMATH_CALUDE_jons_website_hours_l1657_165794


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1657_165765

theorem complex_absolute_value (t : ℝ) : 
  t > 0 → Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 13 → t = 2 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1657_165765


namespace NUMINAMATH_CALUDE_plot_perimeter_l1657_165780

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ
  length_width_relation : length = width + 10
  cost_equation : fencing_cost = (2 * (length + width)) * fencing_rate

/-- The perimeter of the rectangular plot is 300 meters -/
theorem plot_perimeter (plot : RectangularPlot) 
  (h : plot.fencing_rate = 6.5 ∧ plot.fencing_cost = 1950) : 
  2 * (plot.length + plot.width) = 300 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_l1657_165780


namespace NUMINAMATH_CALUDE_marias_paper_count_l1657_165743

theorem marias_paper_count : 
  ∀ (desk_sheets backpack_sheets : ℕ),
    desk_sheets = 50 →
    backpack_sheets = 41 →
    desk_sheets + backpack_sheets = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_marias_paper_count_l1657_165743


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_in_range_l1657_165731

theorem fraction_integer_iff_p_in_range (p : ℕ+) :
  (∃ (k : ℕ+), (4 * p + 17 : ℚ) / (3 * p - 7 : ℚ) = k) ↔ 3 ≤ p ∧ p ≤ 40 := by
sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_in_range_l1657_165731


namespace NUMINAMATH_CALUDE_sum_and_powers_equality_l1657_165771

theorem sum_and_powers_equality : (3 + 7)^3 + (3^2 + 7^2 + 3^3 + 7^3) = 1428 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_powers_equality_l1657_165771


namespace NUMINAMATH_CALUDE_council_vote_change_l1657_165723

theorem council_vote_change (total_members : ℕ) 
  (initial_for initial_against : ℚ) 
  (revote_for revote_against : ℚ) : 
  total_members = 500 ∧ 
  initial_for + initial_against = total_members ∧
  initial_against > initial_for ∧
  revote_for + revote_against = total_members ∧
  revote_for > revote_against ∧
  revote_for - revote_against = (3/2) * (initial_against - initial_for) ∧
  revote_for = (11/10) * initial_against →
  revote_for - initial_for = 156.25 := by
sorry

end NUMINAMATH_CALUDE_council_vote_change_l1657_165723


namespace NUMINAMATH_CALUDE_symmetric_points_l1657_165774

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- The theorem stating that (4, -3) is symmetric to (-4, 3) with respect to the origin -/
theorem symmetric_points : symmetric_wrt_origin (4, -3) (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l1657_165774


namespace NUMINAMATH_CALUDE_bruce_fruit_shopping_l1657_165707

def grapes_quantity : ℝ := 8
def grapes_price : ℝ := 70
def mangoes_quantity : ℝ := 11
def mangoes_price : ℝ := 55
def oranges_quantity : ℝ := 5
def oranges_price : ℝ := 45
def apples_quantity : ℝ := 3
def apples_price : ℝ := 90
def cherries_quantity : ℝ := 4.5
def cherries_price : ℝ := 120

def total_cost : ℝ := grapes_quantity * grapes_price + 
                      mangoes_quantity * mangoes_price + 
                      oranges_quantity * oranges_price + 
                      apples_quantity * apples_price + 
                      cherries_quantity * cherries_price

theorem bruce_fruit_shopping : total_cost = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_shopping_l1657_165707


namespace NUMINAMATH_CALUDE_mo_drinking_difference_l1657_165709

/-- Mo's drinking habits and last week's data --/
structure MoDrinkingData where
  n : ℕ  -- Number of hot chocolate cups on rainy days
  total_cups : ℕ  -- Total cups of tea and hot chocolate last week
  rainy_days : ℕ  -- Number of rainy days last week

/-- Theorem stating the difference between tea and hot chocolate cups --/
theorem mo_drinking_difference (data : MoDrinkingData) : 
  data.n ≤ 2 ∧ 
  data.total_cups = 20 ∧ 
  data.rainy_days = 2 → 
  (7 - data.rainy_days) * 3 - data.rainy_days * data.n = 11 := by
  sorry

#check mo_drinking_difference

end NUMINAMATH_CALUDE_mo_drinking_difference_l1657_165709


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1657_165755

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l1657_165755


namespace NUMINAMATH_CALUDE_base4_division_theorem_l1657_165744

/-- Represents a number in base 4 --/
def Base4 : Type := Nat

/-- Converts a Base4 number to its decimal representation --/
def to_decimal (n : Base4) : Nat :=
  sorry

/-- Converts a decimal number to its Base4 representation --/
def to_base4 (n : Nat) : Base4 :=
  sorry

/-- Performs division in Base4 --/
def base4_div (a b : Base4) : Base4 :=
  sorry

theorem base4_division_theorem :
  base4_div (to_base4 1023) (to_base4 11) = to_base4 33 := by
  sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l1657_165744


namespace NUMINAMATH_CALUDE_managers_salary_l1657_165746

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 15 →
  avg_salary = 1800 →
  avg_increase = 150 →
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary = 4200 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l1657_165746


namespace NUMINAMATH_CALUDE_houses_with_neither_l1657_165710

theorem houses_with_neither (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ)
  (h_total : total = 70)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_both : both = 35) :
  total - (garage + pool - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_houses_with_neither_l1657_165710


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1657_165764

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) (h_sum : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1657_165764


namespace NUMINAMATH_CALUDE_equation_root_one_l1657_165716

theorem equation_root_one (k : ℝ) : 
  let a : ℝ := 13 / 2
  let b : ℝ := -4
  ∃ x : ℝ, x = 1 ∧ (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_one_l1657_165716


namespace NUMINAMATH_CALUDE_bowling_team_size_l1657_165751

theorem bowling_team_size (original_average : ℝ) (new_average : ℝ) 
  (new_player1_weight : ℝ) (new_player2_weight : ℝ) 
  (h1 : original_average = 76) 
  (h2 : new_average = 78) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) : 
  ∃ n : ℕ, n > 0 ∧ 
  (n : ℝ) * original_average + new_player1_weight + new_player2_weight = 
  (n + 2 : ℝ) * new_average := by
  sorry

#check bowling_team_size

end NUMINAMATH_CALUDE_bowling_team_size_l1657_165751


namespace NUMINAMATH_CALUDE_least_possible_difference_l1657_165763

theorem least_possible_difference (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 5 → 
  ∀ (s : ℤ), z - x ≥ s → s ≥ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1657_165763


namespace NUMINAMATH_CALUDE_two_absent_one_present_probability_l1657_165760

-- Define the probability of a student being absent
def p_absent : ℚ := 1 / 20

-- Define the probability of a student being present
def p_present : ℚ := 1 - p_absent

-- Define the number of students
def n_students : ℕ := 3

-- Define the number of absent students we're interested in
def n_absent : ℕ := 2

-- Theorem statement
theorem two_absent_one_present_probability :
  (n_students.choose n_absent : ℚ) * p_absent ^ n_absent * p_present ^ (n_students - n_absent) = 57 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_two_absent_one_present_probability_l1657_165760


namespace NUMINAMATH_CALUDE_thompson_children_ages_l1657_165713

/-- Represents a 5-digit license plate number -/
structure LicensePlate where
  digits : Fin 5 → Nat
  sum_constraint : (digits 0) + (digits 1) + (digits 2) + (digits 3) + (digits 4) = 5
  format_constraint : ∃ a b c, 
    ((digits 0 = a ∧ digits 1 = a ∧ digits 2 = b ∧ digits 3 = b ∧ digits 4 = c) ∨
     (digits 0 = a ∧ digits 1 = b ∧ digits 2 = a ∧ digits 3 = b ∧ digits 4 = c))

/-- Represents the ages of Mr. Thompson's children -/
structure ChildrenAges where
  ages : Fin 6 → Nat
  oldest_12 : ∃ i, ages i = 12
  sum_40 : (ages 0) + (ages 1) + (ages 2) + (ages 3) + (ages 4) + (ages 5) = 40

theorem thompson_children_ages 
  (plate : LicensePlate) 
  (ages : ChildrenAges) 
  (divisibility : ∀ i, plate.digits 0 * 10000 + plate.digits 1 * 1000 + 
                       plate.digits 2 * 100 + plate.digits 3 * 10 + plate.digits 4 % ages.ages i = 0) :
  ¬(∃ i, ages.ages i = 10) :=
sorry

end NUMINAMATH_CALUDE_thompson_children_ages_l1657_165713


namespace NUMINAMATH_CALUDE_ellipse_sum_l1657_165758

theorem ellipse_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →  -- Ellipse equation
  (h = 3 ∧ k = -5) →                                   -- Center at (3, -5)
  (a = 7 ∨ b = 7) →                                    -- Semi-major axis is 7
  (a = 2 ∨ b = 2) →                                    -- Semi-minor axis is 2
  (a > b) →                                            -- Ensure a is semi-major axis
  h + k + a + b = 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_l1657_165758


namespace NUMINAMATH_CALUDE_joan_attended_395_games_l1657_165784

/-- The number of baseball games Joan attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Proof that Joan attended 395 baseball games -/
theorem joan_attended_395_games (total_games night_games missed_games : ℕ) 
  (h1 : total_games = 864)
  (h2 : night_games = 128)
  (h3 : missed_games = 469) :
  games_attended total_games missed_games = 395 := by
  sorry

#eval games_attended 864 469

end NUMINAMATH_CALUDE_joan_attended_395_games_l1657_165784


namespace NUMINAMATH_CALUDE_inequality_proof_l1657_165702

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1657_165702


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1657_165745

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (3 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (2 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m) ∧ (3 ∣ m) → m ≥ n) ∧
  n = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1657_165745


namespace NUMINAMATH_CALUDE_pig_problem_l1657_165761

theorem pig_problem (x y : ℕ) : 
  (y - 100 = 100 * x) →  -- If each person contributes 100 coins, there's a surplus of 100
  (y = 90 * x) →         -- If each person contributes 90 coins, it's just enough
  (x = 10 ∧ y = 900) :=  -- Then the number of people is 10 and the price of the pig is 900
by sorry

end NUMINAMATH_CALUDE_pig_problem_l1657_165761


namespace NUMINAMATH_CALUDE_total_workers_is_214_l1657_165766

/-- Represents a workshop with its salary information -/
structure Workshop where
  avgSalary : ℕ
  techCount : ℕ
  techAvgSalary : ℕ
  otherSalary : ℕ

/-- Calculates the total number of workers in a workshop -/
def totalWorkers (w : Workshop) : ℕ :=
  let otherWorkers := (w.avgSalary * (w.techCount + 1) - w.techAvgSalary * w.techCount) / (w.avgSalary - w.otherSalary)
  w.techCount + otherWorkers

/-- The given workshops -/
def workshopA : Workshop := {
  avgSalary := 8000,
  techCount := 7,
  techAvgSalary := 20000,
  otherSalary := 6000
}

def workshopB : Workshop := {
  avgSalary := 9000,
  techCount := 10,
  techAvgSalary := 25000,
  otherSalary := 5000
}

def workshopC : Workshop := {
  avgSalary := 10000,
  techCount := 15,
  techAvgSalary := 30000,
  otherSalary := 7000
}

/-- The main theorem to prove -/
theorem total_workers_is_214 :
  totalWorkers workshopA + totalWorkers workshopB + totalWorkers workshopC = 214 := by
  sorry


end NUMINAMATH_CALUDE_total_workers_is_214_l1657_165766


namespace NUMINAMATH_CALUDE_can_detect_drum_l1657_165737

-- Define the stone type
def Stone : Type := ℕ

-- Define the set of 100 stones
def S : Finset Stone := sorry

-- Define the weight function
def weight : Stone → ℕ := sorry

-- Define the property that all stones have different weights
axiom different_weights : ∀ s₁ s₂ : Stone, s₁ ≠ s₂ → weight s₁ ≠ weight s₂

-- Define a subset of 10 stones
def Subset : Type := Finset Stone

-- Define the property that a subset has exactly 10 stones
def has_ten_stones (subset : Subset) : Prop := subset.card = 10

-- Define the ordering function (by the brownie)
def order_stones (subset : Subset) : List Stone := sorry

-- Define the potential swapping function (by the drum)
def swap_stones (ordered_stones : List Stone) : List Stone := sorry

-- Define the observation function (what Andryusha sees)
def observe (subset : Subset) : List Stone := sorry

-- The main theorem
theorem can_detect_drum :
  ∃ (f : Subset → Bool),
    (∀ subset : Subset, has_ten_stones subset →
      f subset = true ↔ observe subset ≠ order_stones subset) :=
sorry

end NUMINAMATH_CALUDE_can_detect_drum_l1657_165737


namespace NUMINAMATH_CALUDE_problem_solution_l1657_165792

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (h₁ : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (h₂ : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12)
  (h₃ : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1657_165792


namespace NUMINAMATH_CALUDE_certificate_recipients_l1657_165736

theorem certificate_recipients (total : ℕ) (difference : ℕ) (recipients : ℕ) : 
  total = 120 → 
  difference = 36 → 
  recipients = total / 2 + difference / 2 → 
  recipients = 78 := by
sorry

end NUMINAMATH_CALUDE_certificate_recipients_l1657_165736


namespace NUMINAMATH_CALUDE_teacher_work_days_l1657_165795

/-- Represents the number of days a teacher works in a month -/
def days_worked_per_month (periods_per_day : ℕ) (pay_per_period : ℕ) (months_worked : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / months_worked) / (periods_per_day * pay_per_period)

/-- Theorem stating the number of days a teacher works in a month given specific conditions -/
theorem teacher_work_days :
  days_worked_per_month 5 5 6 3600 = 24 := by
  sorry

end NUMINAMATH_CALUDE_teacher_work_days_l1657_165795


namespace NUMINAMATH_CALUDE_rancher_cows_count_l1657_165782

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →  -- The rancher raises 5 times as many cows as horses
  cows + horses = 168 →  -- The total number of animals is 168
  cows = 140 :=  -- Prove that the number of cows is 140
by sorry

end NUMINAMATH_CALUDE_rancher_cows_count_l1657_165782


namespace NUMINAMATH_CALUDE_initial_liquid_x_percentage_l1657_165778

theorem initial_liquid_x_percentage
  (initial_water_percentage : Real)
  (initial_solution_weight : Real)
  (evaporated_water : Real)
  (added_solution : Real)
  (final_liquid_x_percentage : Real)
  (h1 : initial_water_percentage = 70)
  (h2 : initial_solution_weight = 8)
  (h3 : evaporated_water = 3)
  (h4 : added_solution = 3)
  (h5 : final_liquid_x_percentage = 41.25)
  : Real := by
  sorry

#check initial_liquid_x_percentage

end NUMINAMATH_CALUDE_initial_liquid_x_percentage_l1657_165778


namespace NUMINAMATH_CALUDE_rectangles_in_35_44_grid_l1657_165753

/-- The number of rectangles in a grid -/
def count_rectangles (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Theorem: The number of rectangles in a 35 · 44 grid is 87 -/
theorem rectangles_in_35_44_grid :
  count_rectangles 35 44 = 87 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_35_44_grid_l1657_165753


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1657_165750

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n+1) - a n = d) 
  (h2 : a 1 = f (d - 1)) 
  (h3 : a 3 = f (d + 1)) :
  ∀ n, a n = 2*n + 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1657_165750


namespace NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_prime_mod_seven_l1657_165787

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the operation we're performing
def operation (n : ℕ) : ℕ := (n^2)^4

-- Theorem statement
theorem fourth_power_of_square_of_fourth_prime_mod_seven :
  operation fourth_smallest_prime % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_prime_mod_seven_l1657_165787


namespace NUMINAMATH_CALUDE_trig_identities_l1657_165789

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1657_165789


namespace NUMINAMATH_CALUDE_square_unbounded_l1657_165748

theorem square_unbounded : ∀ (M : ℝ), M > 0 → ∃ (N : ℝ), ∀ (x : ℝ), x > N → x^2 > M := by
  sorry

end NUMINAMATH_CALUDE_square_unbounded_l1657_165748


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1657_165725

/-- A line in the xy-plane is represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the xy-plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- A point lies on a line if its coordinates satisfy the line's equation. -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The problem statement as a theorem. -/
theorem parallel_line_through_point :
  let l1 : Line := { slope := -2, intercept := 3 }
  let p : Point := { x := 1, y := 2 }
  ∃ l2 : Line, parallel l1 l2 ∧ pointOnLine p l2 ∧ l2.slope = -2 ∧ l2.intercept = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1657_165725


namespace NUMINAMATH_CALUDE_division_invariance_l1657_165732

theorem division_invariance (a b : ℝ) (h : b ≠ 0) : (10 * a) / (10 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_division_invariance_l1657_165732


namespace NUMINAMATH_CALUDE_blender_sales_at_600_l1657_165759

/-- Represents the relationship between price and number of customers for blenders. -/
structure BlenderSales where
  price : ℝ
  customers : ℝ

/-- The inverse proportionality constant for blender sales. -/
def k : ℝ := 10 * 300

/-- Axiom: The number of customers is inversely proportional to the price of blenders. -/
axiom inverse_proportion (b : BlenderSales) : b.price * b.customers = k

/-- The theorem to be proved. -/
theorem blender_sales_at_600 :
  ∃ (b : BlenderSales), b.price = 600 ∧ b.customers = 5 :=
sorry

end NUMINAMATH_CALUDE_blender_sales_at_600_l1657_165759


namespace NUMINAMATH_CALUDE_min_additional_coins_l1657_165796

def friends : ℕ := 15
def initial_coins : ℕ := 100

theorem min_additional_coins :
  let required_coins := (friends * (friends + 1)) / 2
  required_coins - initial_coins = 20 := by
sorry

end NUMINAMATH_CALUDE_min_additional_coins_l1657_165796


namespace NUMINAMATH_CALUDE_max_temperature_range_l1657_165738

theorem max_temperature_range (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 50)
  (min_temp : ∃ i, temps i = 45 ∧ ∀ j, temps j ≥ 45) :
  ∃ i j, temps i - temps j ≤ 25 ∧ 
  ∀ k l, temps k - temps l ≤ temps i - temps j :=
by sorry

end NUMINAMATH_CALUDE_max_temperature_range_l1657_165738


namespace NUMINAMATH_CALUDE_unique_digit_product_equation_l1657_165701

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_digit_product_equation : 
  ∃! x : ℕ, digit_product x = x^2 - 10*x - 22 ∧ x = 12 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_product_equation_l1657_165701


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1657_165739

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (1 - 1 / (x + 1)) / (x / (x - 1)) = (x - 1) / (x + 1) ∧
  (2 - 1) / (2 + 1) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1657_165739


namespace NUMINAMATH_CALUDE_mad_hatter_win_condition_l1657_165785

/-- Represents the fraction of voters for each candidate and undecided voters -/
structure VoteDistribution where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ
  undecided : ℝ

/-- Represents the final vote count after undecided voters have voted -/
structure FinalVotes where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ

def minimum_fraction_for_mad_hatter (initial_votes : VoteDistribution) : ℝ :=
  0.7

theorem mad_hatter_win_condition (initial_votes : VoteDistribution) 
  (h1 : initial_votes.mad_hatter = 0.2)
  (h2 : initial_votes.march_hare = 0.25)
  (h3 : initial_votes.dormouse = 0.3)
  (h4 : initial_votes.undecided = 0.25)
  (h5 : initial_votes.mad_hatter + initial_votes.march_hare + initial_votes.dormouse + initial_votes.undecided = 1) :
  ∀ (final_votes : FinalVotes),
    (final_votes.mad_hatter ≥ initial_votes.mad_hatter + initial_votes.undecided * minimum_fraction_for_mad_hatter initial_votes) →
    (final_votes.march_hare ≤ initial_votes.march_hare + initial_votes.undecided * (1 - minimum_fraction_for_mad_hatter initial_votes)) →
    (final_votes.dormouse ≤ initial_votes.dormouse + initial_votes.undecided * (1 - minimum_fraction_for_mad_hatter initial_votes)) →
    (final_votes.mad_hatter + final_votes.march_hare + final_votes.dormouse = 1) →
    (final_votes.mad_hatter ≥ final_votes.march_hare ∧ final_votes.mad_hatter ≥ final_votes.dormouse) :=
  sorry

end NUMINAMATH_CALUDE_mad_hatter_win_condition_l1657_165785


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l1657_165719

/-- Number of blue flags -/
def blue_flags : ℕ := 11

/-- Number of green flags -/
def green_flags : ℕ := 10

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Number of distinguishable arrangements -/
def M : ℕ := 660

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem flag_arrangement_remainder :
  M % 1000 = 660 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l1657_165719


namespace NUMINAMATH_CALUDE_motel_pricing_solution_l1657_165797

/-- A motel pricing structure with a flat fee for the first night and a consistent nightly fee thereafter. -/
structure MotelPricing where
  flat_fee : ℝ
  nightly_fee : ℝ

/-- The total cost for a stay at the motel given the number of nights. -/
def total_cost (p : MotelPricing) (nights : ℕ) : ℝ :=
  p.flat_fee + p.nightly_fee * (nights - 1)

theorem motel_pricing_solution :
  ∃ (p : MotelPricing),
    total_cost p 4 = 215 ∧
    total_cost p 3 = 155 ∧
    p.flat_fee = 35 ∧
    p.nightly_fee = 60 := by
  sorry

end NUMINAMATH_CALUDE_motel_pricing_solution_l1657_165797


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1657_165706

theorem sum_of_roots_quadratic (b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - b*x + 20
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x * y = 20) →
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x + y = b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1657_165706


namespace NUMINAMATH_CALUDE_tennis_players_l1657_165754

theorem tennis_players (total : ℕ) (squash : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 38)
  (h2 : squash = 21)
  (h3 : neither = 10)
  (h4 : both = 12) :
  total - squash + both - neither = 19 :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_l1657_165754


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1657_165749

theorem inequality_solution_set : 
  {x : ℝ | 8*x^3 + 9*x^2 + 7*x < 6} = 
  {x : ℝ | (-6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1657_165749


namespace NUMINAMATH_CALUDE_polynomial_expansion_p_value_l1657_165767

/-- The value of p in the expansion of (x+y)^8 -/
theorem polynomial_expansion_p_value :
  ∀ (p q : ℝ),
  p > 0 →
  q > 0 →
  p + q = 1 →
  8 * p^7 * q = 28 * p^6 * q^2 →
  p = 7/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_p_value_l1657_165767


namespace NUMINAMATH_CALUDE_beach_trip_driving_time_l1657_165781

theorem beach_trip_driving_time :
  ∀ (x : ℝ),
  (2.5 * (2 * x) + 2 * x = 14) →
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_beach_trip_driving_time_l1657_165781


namespace NUMINAMATH_CALUDE_james_and_louise_ages_james_and_louise_ages_proof_l1657_165762

theorem james_and_louise_ages : ℕ → ℕ → Prop :=
  fun j l =>
    (j = l + 9) →                  -- James is nine years older than Louise
    (j + 7 = 3 * (l - 3)) →        -- Seven years from now, James will be three times as old as Louise was three years before now
    (j + l = 35)                   -- The sum of their current ages is 35

-- The proof of this theorem
theorem james_and_louise_ages_proof : ∃ j l : ℕ, james_and_louise_ages j l := by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_james_and_louise_ages_proof_l1657_165762


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l1657_165733

theorem value_added_to_numbers (n : ℕ) (original_avg new_avg x : ℝ) 
  (h1 : n = 15)
  (h2 : original_avg = 40)
  (h3 : new_avg = 53)
  (h4 : n * new_avg = n * original_avg + n * x) :
  x = 13 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l1657_165733


namespace NUMINAMATH_CALUDE_wolf_sheep_problem_l1657_165703

theorem wolf_sheep_problem (x : ℕ) : 
  (∃ y : ℕ, y = 3 * x + 2 ∧ y = 8 * x - 8) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_wolf_sheep_problem_l1657_165703


namespace NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l1657_165777

/-- The amount of additional cents Elizabeth needs to purchase a pencil -/
def additional_cents_needed (elizabeth_dollars : ℕ) (borrowed_cents : ℕ) (pencil_dollars : ℕ) : ℕ :=
  pencil_dollars * 100 - (elizabeth_dollars * 100 + borrowed_cents)

theorem elizabeth_pencil_purchase :
  additional_cents_needed 5 53 6 = 47 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_pencil_purchase_l1657_165777


namespace NUMINAMATH_CALUDE_cosine_equality_l1657_165790

theorem cosine_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (812 * π / 180) → n = 92 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l1657_165790


namespace NUMINAMATH_CALUDE_wally_bears_count_l1657_165735

def bear_price (n : ℕ) : ℚ :=
  4 - (n - 1) * (1/2)

def total_cost (num_bears : ℕ) : ℚ :=
  (num_bears : ℚ) / 2 * (2 * 4 + (num_bears - 1) * (-1/2))

theorem wally_bears_count : 
  ∃ (n : ℕ), n > 0 ∧ total_cost n = 354 :=
sorry

end NUMINAMATH_CALUDE_wally_bears_count_l1657_165735


namespace NUMINAMATH_CALUDE_bucket_weight_l1657_165768

/-- Given a bucket with the following properties:
    1. When three-fourths full, it weighs p kilograms.
    2. When one-third full, it weighs q kilograms.
    This theorem states that when the bucket is five-sixths full, 
    it weighs (6p - q) / 5 kilograms. -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let weight_three_fourths := p
  let weight_one_third := q
  let weight_five_sixths := (6 * p - q) / 5
  weight_five_sixths

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l1657_165768


namespace NUMINAMATH_CALUDE_polar_equation_graph_l1657_165708

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a graph in polar coordinates -/
inductive PolarGraph
  | Circle : PolarGraph
  | Ray : PolarGraph
  | Both : PolarGraph

/-- The equation (ρ-3)(θ-π/2)=0 with ρ≥0 -/
def polarEquation (p : PolarPoint) : Prop :=
  (p.ρ - 3) * (p.θ - Real.pi / 2) = 0 ∧ p.ρ ≥ 0

/-- The theorem stating that the equation represents a circle and a ray -/
theorem polar_equation_graph : 
  (∃ p : PolarPoint, polarEquation p) → PolarGraph.Both = PolarGraph.Both :=
sorry

end NUMINAMATH_CALUDE_polar_equation_graph_l1657_165708


namespace NUMINAMATH_CALUDE_multiple_problem_l1657_165742

theorem multiple_problem (m : ℚ) : 38 + m * 43 = 124 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l1657_165742


namespace NUMINAMATH_CALUDE_river_round_trip_time_l1657_165741

/-- The time taken for a round trip on a river with given conditions -/
theorem river_round_trip_time
  (rower_speed : ℝ)
  (river_speed : ℝ)
  (distance : ℝ)
  (h1 : rower_speed = 6)
  (h2 : river_speed = 1)
  (h3 : distance = 2.916666666666667)
  : (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1 := by
  sorry

#eval (2.916666666666667 / (6 - 1)) + (2.916666666666667 / (6 + 1))

end NUMINAMATH_CALUDE_river_round_trip_time_l1657_165741


namespace NUMINAMATH_CALUDE_f_extrema_g_negativity_l1657_165727

noncomputable def f (x : ℝ) : ℝ := -x^2 + Real.log x

noncomputable def g (a x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x - 2*a*x

def interval : Set ℝ := Set.Icc (1/Real.exp 1) (Real.exp 1)

theorem f_extrema :
  ∃ (x_min x_max : ℝ), x_min ∈ interval ∧ x_max ∈ interval ∧
  (∀ x ∈ interval, f x ≥ f x_min) ∧
  (∀ x ∈ interval, f x ≤ f x_max) ∧
  f x_min = 1 - Real.exp 2 ∧
  f x_max = -1/2 - 1/2 * Real.log 2 :=
sorry

theorem g_negativity :
  ∀ a : ℝ, (∀ x > 2, g a x < 0) ↔ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_g_negativity_l1657_165727


namespace NUMINAMATH_CALUDE_escalator_standing_time_l1657_165705

/-- Represents the time it takes Clea to ride an escalator in different scenarios -/
structure EscalatorRide where
  nonOperatingWalkTime : ℝ
  operatingWalkTime : ℝ
  standingTime : ℝ

/-- Proves that given the conditions, the standing time on the operating escalator is 80 seconds -/
theorem escalator_standing_time (ride : EscalatorRide) 
  (h1 : ride.nonOperatingWalkTime = 120)
  (h2 : ride.operatingWalkTime = 48) :
  ride.standingTime = 80 := by
  sorry

#check escalator_standing_time

end NUMINAMATH_CALUDE_escalator_standing_time_l1657_165705


namespace NUMINAMATH_CALUDE_tournament_players_count_l1657_165757

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players not in the lowest 8
  total_players : ℕ := n + 8
  points_among_n : ℕ := n * (n - 1) / 2
  points_n_vs_lowest8 : ℕ := points_among_n / 3
  points_among_lowest8 : ℕ := 28
  total_points : ℕ := 4 * points_among_n / 3 + 2 * points_among_lowest8

/-- The theorem stating that the total number of players in the tournament is 50 -/
theorem tournament_players_count (t : Tournament) : t.total_players = 50 := by
  sorry

end NUMINAMATH_CALUDE_tournament_players_count_l1657_165757


namespace NUMINAMATH_CALUDE_polynomial_coefficient_square_difference_l1657_165700

theorem polynomial_coefficient_square_difference (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_square_difference_l1657_165700


namespace NUMINAMATH_CALUDE_expression_evaluation_l1657_165734

theorem expression_evaluation (x y : ℚ) (hx : x = -1) (hy : y = -1/3) :
  (3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y) = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1657_165734


namespace NUMINAMATH_CALUDE_solve_for_a_l1657_165779

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^2

-- Theorem statement
theorem solve_for_a : ∃ (a : ℝ), star a 3 = 7 ∧ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1657_165779


namespace NUMINAMATH_CALUDE_quiz_competition_score_l1657_165722

/-- Calculates the final score in a quiz competition given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - (incorrect : ℚ) * (1 / 4)

/-- Represents the quiz competition problem -/
theorem quiz_competition_score :
  let total_questions : ℕ := 35
  let correct_answers : ℕ := 17
  let incorrect_answers : ℕ := 12
  let unanswered_questions : ℕ := 6
  correct_answers + incorrect_answers + unanswered_questions = total_questions →
  calculate_score correct_answers incorrect_answers unanswered_questions = 14 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_score_l1657_165722


namespace NUMINAMATH_CALUDE_molecular_weight_C8H10N4O6_l1657_165775

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Carbon atoms in C8H10N4O6 -/
def num_C : ℕ := 8

/-- The number of Hydrogen atoms in C8H10N4O6 -/
def num_H : ℕ := 10

/-- The number of Nitrogen atoms in C8H10N4O6 -/
def num_N : ℕ := 4

/-- The number of Oxygen atoms in C8H10N4O6 -/
def num_O : ℕ := 6

/-- The molecular weight of C8H10N4O6 in g/mol -/
def molecular_weight : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_N * atomic_weight_N +
  num_O * atomic_weight_O

theorem molecular_weight_C8H10N4O6 : molecular_weight = 258.22 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_C8H10N4O6_l1657_165775


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l1657_165740

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical :
  let x : ℝ := -5
  let y : ℝ := 0
  let z : ℝ := -8
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.pi
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 5 ∧ θ = Real.pi ∧ z = -8 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l1657_165740


namespace NUMINAMATH_CALUDE_fourth_side_length_l1657_165718

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of three sides of the quadrilateral -/
  side_length : ℝ
  /-- Assertion that the quadrilateral is a kite with two equal consecutive sides -/
  is_kite : Prop
  /-- Assertion that one diagonal is a diameter of the circle -/
  diagonal_is_diameter : Prop

/-- The theorem stating the length of the fourth side of the quadrilateral -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h1 : q.radius = 150 * Real.sqrt 2)
  (h2 : q.side_length = 150) :
  ∃ (fourth_side : ℝ), fourth_side = 150 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l1657_165718


namespace NUMINAMATH_CALUDE_overtime_hours_l1657_165783

/-- Queenie's daily wage as a part-time clerk -/
def daily_wage : ℕ := 150

/-- Queenie's overtime pay rate per hour -/
def overtime_rate : ℕ := 5

/-- Number of days Queenie worked -/
def days_worked : ℕ := 5

/-- Total amount Queenie received -/
def total_pay : ℕ := 770

/-- Calculate the number of overtime hours Queenie worked -/
theorem overtime_hours : 
  (total_pay - daily_wage * days_worked) / overtime_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_l1657_165783


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_l1657_165728

/-- The probability of selecting at least one girl from a group of 4 boys and 3 girls when choosing 2 people -/
theorem prob_at_least_one_girl (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 4 → num_girls = 3 → 
  (1 - (Nat.choose num_boys 2 : ℚ) / (Nat.choose (num_boys + num_girls) 2 : ℚ)) = 5/7 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_l1657_165728


namespace NUMINAMATH_CALUDE_square_equality_l1657_165798

theorem square_equality (n : ℕ) : (n + 3)^2 = 3*(n + 2)^2 - 3*(n + 1)^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l1657_165798


namespace NUMINAMATH_CALUDE_claire_earnings_l1657_165717

/-- Represents the total earnings from selling roses with discounts applied -/
def total_earnings (total_flowers : ℕ) (tulips : ℕ) (white_roses : ℕ) 
  (small_red_roses : ℕ) (medium_red_roses : ℕ) 
  (small_price : ℚ) (medium_price : ℚ) (large_price : ℚ) : ℚ :=
  let total_roses := total_flowers - tulips
  let red_roses := total_roses - white_roses
  let large_red_roses := red_roses - small_red_roses - medium_red_roses
  let small_sold := small_red_roses / 2
  let medium_sold := medium_red_roses / 2
  let large_sold := large_red_roses / 2
  let small_earnings := small_sold * small_price * (1 - 0.1)  -- 10% discount
  let medium_earnings := medium_sold * medium_price * (1 - 0.15)  -- 15% discount
  let large_earnings := large_sold * large_price * (1 - 0.15)  -- 15% discount
  small_earnings + medium_earnings + large_earnings

/-- Theorem stating that Claire's earnings are $92.13 -/
theorem claire_earnings : 
  total_earnings 400 120 80 40 60 0.75 1 1.25 = 92.13 := by
  sorry


end NUMINAMATH_CALUDE_claire_earnings_l1657_165717


namespace NUMINAMATH_CALUDE_hard_drives_sold_l1657_165776

/-- Represents the number of hard drives sold -/
def num_hard_drives : ℕ := 14

/-- Represents the total earnings from all items -/
def total_earnings : ℕ := 8960

/-- Theorem stating that the number of hard drives sold is 14 -/
theorem hard_drives_sold : 
  10 * 600 + 8 * 200 + 4 * 60 + num_hard_drives * 80 = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_hard_drives_sold_l1657_165776


namespace NUMINAMATH_CALUDE_toy_cost_price_l1657_165715

theorem toy_cost_price (profit_equality : 30 * (12 - C) = 20 * (15 - C)) : C = 6 :=
by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l1657_165715


namespace NUMINAMATH_CALUDE_min_value_fraction_l1657_165704

theorem min_value_fraction (x y : ℝ) (h : (x + 2)^2 + y^2 = 1) :
  ∃ k : ℝ, k = (y - 1) / (x - 2) ∧ k ≥ 0 ∧ ∀ m : ℝ, m = (y - 1) / (x - 2) → m ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1657_165704


namespace NUMINAMATH_CALUDE_parabola_tangent_ellipse_l1657_165730

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define point A on the parabola
def point_A : ℝ × ℝ := (2, 4)

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- State the theorem
theorem parabola_tangent_ellipse :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  parabola (point_A.1) = point_A.2 →
  tangent_line 1 = 0 →
  tangent_line 0 = -4 →
  ellipse a b 1 0 →
  ellipse a b 0 (-4) →
  ellipse (Real.sqrt 17) 4 1 0 ∧
  ellipse (Real.sqrt 17) 4 0 (-4) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_ellipse_l1657_165730


namespace NUMINAMATH_CALUDE_loan_duration_l1657_165770

/-- Given a loan split into two parts, prove the duration of the second part. -/
theorem loan_duration (total sum : ℕ) (second_part : ℕ) (first_rate second_rate : ℚ) (first_duration : ℕ) :
  total = 2691 →
  second_part = 1656 →
  first_rate = 3 / 100 →
  second_rate = 5 / 100 →
  first_duration = 8 →
  (total - second_part) * first_rate * first_duration = second_part * second_rate * 3 →
  3 = (total - second_part) * first_rate * first_duration / (second_part * second_rate) :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_l1657_165770


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1657_165724

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n, a (n + 1) = q * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → a n < a m

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : increasing_sequence a)
  (h3 : a 5 ^ 2 = a 10)
  (h4 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  a 5 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1657_165724


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l1657_165786

/-- Given two positive integers with specific properties, prove that the second factor of their LCM is 13 -/
theorem lcm_factor_problem (A B : ℕ+) (X : ℕ+) : 
  (Nat.gcd A B = 23) →
  (Nat.lcm A B = 23 * 12 * X) →
  (A = 299) →
  X = 13 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l1657_165786


namespace NUMINAMATH_CALUDE_song_count_proof_l1657_165793

def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

theorem song_count_proof (initial deleted added : ℕ) 
  (h1 : initial ≥ deleted) : 
  final_song_count initial deleted added = initial - deleted + added :=
by
  sorry

#eval final_song_count 34 14 44

end NUMINAMATH_CALUDE_song_count_proof_l1657_165793


namespace NUMINAMATH_CALUDE_anna_candy_store_l1657_165720

def candy_store_problem (initial_amount : ℚ) 
                        (gum_price : ℚ) (gum_quantity : ℕ)
                        (chocolate_price : ℚ) (chocolate_quantity : ℕ)
                        (candy_cane_price : ℚ) (candy_cane_quantity : ℕ) : Prop :=
  let total_spent := gum_price * gum_quantity + 
                     chocolate_price * chocolate_quantity + 
                     candy_cane_price * candy_cane_quantity
  initial_amount - total_spent = 1

theorem anna_candy_store : 
  candy_store_problem 10 1 3 1 5 (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_anna_candy_store_l1657_165720


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1657_165756

theorem quadratic_equation_solution :
  ∃! (x : ℚ), x > 0 ∧ 6 * x^2 + 9 * x - 24 = 0 ∧ x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1657_165756


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1657_165714

theorem difference_of_squares_special_case : (827 : ℤ) * 827 - 826 * 828 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1657_165714


namespace NUMINAMATH_CALUDE_saturday_ice_cream_amount_l1657_165729

/-- The amount of ice cream eaten on Saturday night, given the amount eaten on Friday and the total amount eaten over both nights. -/
def ice_cream_saturday (friday : ℝ) (total : ℝ) : ℝ :=
  total - friday

theorem saturday_ice_cream_amount :
  ice_cream_saturday 3.25 3.5 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_saturday_ice_cream_amount_l1657_165729


namespace NUMINAMATH_CALUDE_michael_truck_meet_once_l1657_165788

/-- Represents the meeting of Michael and the truck -/
structure Meeting where
  time : ℝ
  position : ℝ

/-- Represents the problem setup -/
structure Setup where
  michael_speed : ℝ
  pail_spacing : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck -/
def count_meetings (s : Setup) : ℕ :=
  sorry

/-- The main theorem stating that Michael and the truck meet exactly once -/
theorem michael_truck_meet_once (s : Setup) 
  (h1 : s.michael_speed = 6)
  (h2 : s.pail_spacing = 300)
  (h3 : s.truck_speed = 15)
  (h4 : s.truck_stop_time = 45)
  (h5 : s.initial_distance = 300) : 
  count_meetings s = 1 := by
  sorry

end NUMINAMATH_CALUDE_michael_truck_meet_once_l1657_165788


namespace NUMINAMATH_CALUDE_keith_books_l1657_165721

theorem keith_books (jason_books : ℕ) (total_books : ℕ) (h1 : jason_books = 21) (h2 : total_books = 41) :
  total_books - jason_books = 20 := by
sorry

end NUMINAMATH_CALUDE_keith_books_l1657_165721


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1657_165752

theorem square_plus_reciprocal_square (x : ℝ) (hx : x ≠ 0) 
  (h : x + 1/x = Real.sqrt 2019) : x^2 + 1/x^2 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1657_165752


namespace NUMINAMATH_CALUDE_clown_count_l1657_165772

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 357

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 842

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 300534 := by
  sorry

end NUMINAMATH_CALUDE_clown_count_l1657_165772


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l1657_165791

theorem tens_digit_of_2023_pow_2024_minus_2025 : ∃ k : ℕ, (2023^2024 - 2025) % 100 = 10 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l1657_165791


namespace NUMINAMATH_CALUDE_more_than_half_inside_l1657_165773

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The circle is inscribed in the triangle -/
  inscribed : circle ⊆ triangle

/-- A square circumscribed around a circle -/
structure CircumscribedSquare where
  /-- The square -/
  square : Set (ℝ × ℝ)
  /-- The circumscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The square is circumscribed around the circle -/
  circumscribed : circle ⊆ square

/-- The perimeter of a square -/
def squarePerimeter (s : CircumscribedSquare) : ℝ := sorry

/-- The length of the square's perimeter segments inside the triangle -/
def insidePerimeterLength (t : InscribedTriangle) (s : CircumscribedSquare) : ℝ := sorry

/-- Main theorem: More than half of the square's perimeter is inside the triangle -/
theorem more_than_half_inside (t : InscribedTriangle) (s : CircumscribedSquare) 
  (h : t.circle = s.circle) : 
  insidePerimeterLength t s > squarePerimeter s / 2 := by sorry

end NUMINAMATH_CALUDE_more_than_half_inside_l1657_165773


namespace NUMINAMATH_CALUDE_remainder_after_adding_1470_l1657_165711

theorem remainder_after_adding_1470 (n : ℤ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_1470_l1657_165711
