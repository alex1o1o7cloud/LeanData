import Mathlib

namespace NUMINAMATH_CALUDE_triangle_tangent_half_angles_sum_l1231_123110

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = π

-- State the theorem
theorem triangle_tangent_half_angles_sum (t : Triangle) :
  Real.tan (t.A / 2) * Real.tan (t.B / 2) + 
  Real.tan (t.B / 2) * Real.tan (t.C / 2) + 
  Real.tan (t.C / 2) * Real.tan (t.A / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_half_angles_sum_l1231_123110


namespace NUMINAMATH_CALUDE_problem_solution_l1231_123132

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1231_123132


namespace NUMINAMATH_CALUDE_total_practice_time_is_307_5_l1231_123122

/-- Represents Daniel's weekly practice schedule -/
structure PracticeSchedule where
  basketball_school_day : ℝ  -- Minutes of basketball practice on school days
  basketball_weekend_day : ℝ  -- Minutes of basketball practice on weekend days
  soccer_weekday : ℝ  -- Minutes of soccer practice on weekdays
  gymnastics : ℝ  -- Minutes of gymnastics practice
  soccer_saturday : ℝ  -- Minutes of soccer practice on Saturday (averaged)
  swimming_saturday : ℝ  -- Minutes of swimming practice on Saturday (averaged)

/-- Calculates the total practice time for one week -/
def total_practice_time (schedule : PracticeSchedule) : ℝ :=
  schedule.basketball_school_day * 5 +
  schedule.basketball_weekend_day * 2 +
  schedule.soccer_weekday * 3 +
  schedule.gymnastics * 2 +
  schedule.soccer_saturday +
  schedule.swimming_saturday

/-- Daniel's actual practice schedule -/
def daniel_schedule : PracticeSchedule :=
  { basketball_school_day := 15
  , basketball_weekend_day := 30
  , soccer_weekday := 20
  , gymnastics := 30
  , soccer_saturday := 22.5
  , swimming_saturday := 30 }

theorem total_practice_time_is_307_5 :
  total_practice_time daniel_schedule = 307.5 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_time_is_307_5_l1231_123122


namespace NUMINAMATH_CALUDE_rahul_work_days_l1231_123134

/-- Represents the number of days it takes Rahul to complete the work -/
def rahul_days : ℝ := 3

/-- Represents the number of days it takes Rajesh to complete the work -/
def rajesh_days : ℝ := 2

/-- Represents the total payment for the work -/
def total_payment : ℝ := 355

/-- Represents Rahul's share of the payment -/
def rahul_share : ℝ := 142

/-- Theorem stating that given the conditions, Rahul can complete the work in 3 days -/
theorem rahul_work_days :
  (rahul_share / (total_payment - rahul_share) = (1 / rahul_days) / (1 / rajesh_days)) →
  rahul_days = 3 :=
by sorry

end NUMINAMATH_CALUDE_rahul_work_days_l1231_123134


namespace NUMINAMATH_CALUDE_exponential_property_l1231_123107

theorem exponential_property (a : ℝ) :
  (∀ x > 0, a^x > 1) → a > 1 := by sorry

end NUMINAMATH_CALUDE_exponential_property_l1231_123107


namespace NUMINAMATH_CALUDE_probability_all_red_at_fourth_l1231_123175

/-- The number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- The number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- The total number of balls initially in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- The probability of drawing a specific sequence of balls -/
def sequence_probability (red_indices : List ℕ) : ℚ :=
  sorry

/-- The probability of drawing all red balls exactly at the 4th draw -/
def all_red_at_fourth_draw : ℚ :=
  sequence_probability [1, 4] + sequence_probability [2, 4] + sequence_probability [3, 4]

theorem probability_all_red_at_fourth : all_red_at_fourth_draw = 434/10000 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_at_fourth_l1231_123175


namespace NUMINAMATH_CALUDE_constant_n_value_l1231_123137

theorem constant_n_value (m n : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + m) = x^2 + n*x + 12) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_n_value_l1231_123137


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1231_123178

/-- Given an arithmetic sequence with first term 2/3 and second term 4/3,
    prove that its tenth term is 20/3. -/
theorem tenth_term_of_arithmetic_sequence :
  let a₁ : ℚ := 2/3  -- First term
  let a₂ : ℚ := 4/3  -- Second term
  let d : ℚ := a₂ - a₁  -- Common difference
  let a₁₀ : ℚ := a₁ + 9 * d  -- Tenth term
  a₁₀ = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1231_123178


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l1231_123159

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def has_form_173x5 (n : ℕ) : Prop :=
  ∃ x : ℕ, x < 10 ∧ n = 17300 + 10 * x + 5

theorem missing_digit_divisible_by_nine :
  ∃! n : ℕ, is_five_digit n ∧ has_form_173x5 n ∧ n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l1231_123159


namespace NUMINAMATH_CALUDE_green_ribbons_count_l1231_123104

theorem green_ribbons_count (total : ℕ) 
  (h_red : (1 : ℚ) / 4 * total = total / 4)
  (h_blue : (3 : ℚ) / 8 * total = 3 * total / 8)
  (h_green : (1 : ℚ) / 8 * total = total / 8)
  (h_white : total - (total / 4 + 3 * total / 8 + total / 8) = 36) :
  total / 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_green_ribbons_count_l1231_123104


namespace NUMINAMATH_CALUDE_percent_k_equal_to_125_percent_j_l1231_123118

theorem percent_k_equal_to_125_percent_j (j k l m : ℝ) 
  (h1 : 1.25 * j = (12.5 / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 3.5 * (2 * j)) :
  (12.5 / 100) * k = 1.25 * j := by
  sorry

end NUMINAMATH_CALUDE_percent_k_equal_to_125_percent_j_l1231_123118


namespace NUMINAMATH_CALUDE_f_equals_g_l1231_123186

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (m : ℝ) : ℝ := m^2 - 2*m - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1231_123186


namespace NUMINAMATH_CALUDE_nadia_playing_time_l1231_123196

/-- Represents the number of mistakes Nadia makes per 40 notes -/
def mistakes_per_40_notes : ℚ := 3

/-- Represents the number of notes Nadia can play per minute -/
def notes_per_minute : ℚ := 60

/-- Represents the total number of mistakes Nadia made -/
def total_mistakes : ℚ := 36

/-- Calculates the number of minutes Nadia played -/
def minutes_played : ℚ :=
  total_mistakes / (mistakes_per_40_notes * notes_per_minute / 40)

theorem nadia_playing_time :
  minutes_played = 8 := by sorry

end NUMINAMATH_CALUDE_nadia_playing_time_l1231_123196


namespace NUMINAMATH_CALUDE_g_of_2_eq_11_l1231_123182

/-- The function g(x) = x^3 + x^2 - 1 -/
def g (x : ℝ) : ℝ := x^3 + x^2 - 1

/-- Theorem: g(2) = 11 -/
theorem g_of_2_eq_11 : g 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_eq_11_l1231_123182


namespace NUMINAMATH_CALUDE_optimal_viewpoint_for_scenery_l1231_123191

/-- The problem setup -/
structure ScenerySetup where
  A : ℝ × ℝ
  B : ℝ × ℝ
  distance_AB : ℝ

/-- The viewing angle between two points from a given viewpoint -/
def viewing_angle (viewpoint : ℝ × ℝ) (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The optimal viewpoint maximizes the viewing angle -/
def is_optimal_viewpoint (setup : ScenerySetup) (viewpoint : ℝ × ℝ) : Prop :=
  ∀ other : ℝ × ℝ, viewing_angle viewpoint setup.A setup.B ≥ viewing_angle other setup.A setup.B

/-- The main theorem -/
theorem optimal_viewpoint_for_scenery (setup : ScenerySetup) 
    (h1 : setup.A = (Real.sqrt 2, Real.sqrt 2))
    (h2 : setup.B = (0, 2 * Real.sqrt 2))
    (h3 : setup.distance_AB = 2)
    (h4 : setup.A.2 > 0 ∧ setup.B.2 > 0) : -- Ensuring A and B are on the same side of x-axis
  is_optimal_viewpoint setup (0, 0) := by sorry

end NUMINAMATH_CALUDE_optimal_viewpoint_for_scenery_l1231_123191


namespace NUMINAMATH_CALUDE_composite_probability_six_dice_l1231_123109

/-- The number of sides on a standard die -/
def dieSize : Nat := 6

/-- The number of dice rolled -/
def numDice : Nat := 6

/-- The set of possible outcomes when rolling a die -/
def dieOutcomes : Finset Nat := Finset.range dieSize

/-- The total number of possible outcomes when rolling 6 dice -/
def totalOutcomes : Nat := dieSize ^ numDice

/-- A function that determines if a number is prime -/
def isPrime (n : Nat) : Bool := sorry

/-- A function that determines if a number is composite -/
def isComposite (n : Nat) : Bool := n > 1 ∧ ¬(isPrime n)

/-- The number of outcomes where the product is not composite -/
def nonCompositeOutcomes : Nat := 19

/-- The probability of rolling a composite product -/
def compositeProb : Rat := (totalOutcomes - nonCompositeOutcomes) / totalOutcomes

theorem composite_probability_six_dice :
  compositeProb = 46637 / 46656 := by sorry

end NUMINAMATH_CALUDE_composite_probability_six_dice_l1231_123109


namespace NUMINAMATH_CALUDE_tank_capacity_l1231_123120

theorem tank_capacity : ∀ (x : ℚ), 
  (x / 8 + 90 = x / 2) → x = 240 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1231_123120


namespace NUMINAMATH_CALUDE_sarahs_bread_shop_profit_l1231_123176

/-- Sarah's bread shop profit calculation --/
theorem sarahs_bread_shop_profit :
  ∀ (total_loaves : ℕ) 
    (cost_per_loaf morning_price afternoon_price evening_price : ℚ)
    (morning_fraction afternoon_fraction : ℚ),
  total_loaves = 60 →
  cost_per_loaf = 1 →
  morning_price = 3 →
  afternoon_price = 3/2 →
  evening_price = 1 →
  morning_fraction = 1/3 →
  afternoon_fraction = 3/4 →
  let morning_sales := (total_loaves : ℚ) * morning_fraction * morning_price
  let remaining_after_morning := total_loaves - (total_loaves : ℚ) * morning_fraction
  let afternoon_sales := remaining_after_morning * afternoon_fraction * afternoon_price
  let remaining_after_afternoon := remaining_after_morning - remaining_after_morning * afternoon_fraction
  let evening_sales := remaining_after_afternoon * evening_price
  let total_revenue := morning_sales + afternoon_sales + evening_sales
  let total_cost := (total_loaves : ℚ) * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 55 := by
sorry


end NUMINAMATH_CALUDE_sarahs_bread_shop_profit_l1231_123176


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l1231_123147

theorem at_least_one_quadratic_has_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l1231_123147


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1231_123125

theorem inequality_solution_set : 
  {x : ℝ | x * (2 - x) ≤ 0} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1231_123125


namespace NUMINAMATH_CALUDE_work_completion_time_l1231_123133

-- Define the work rates
def work_rate_A : ℚ := 1 / 9
def work_rate_B : ℚ := 1 / 18
def work_rate_combined : ℚ := 1 / 6

-- Define the completion times
def time_A : ℚ := 9
def time_B : ℚ := 18
def time_combined : ℚ := 6

-- Theorem statement
theorem work_completion_time :
  (work_rate_A + work_rate_B = work_rate_combined) →
  (1 / work_rate_A = time_A) →
  (1 / work_rate_B = time_B) →
  (1 / work_rate_combined = time_combined) →
  time_B = 18 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l1231_123133


namespace NUMINAMATH_CALUDE_u_diff_divisible_by_factorial_l1231_123153

/-- The sequence u_k defined recursively -/
def u (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | k + 1 => a ^ (u a k)

/-- Theorem stating that n! divides u_{n+1} - u_n for all n ≥ 1 -/
theorem u_diff_divisible_by_factorial (a : ℕ+) (n : ℕ) (h : n ≥ 1) :
  (n.factorial : ℤ) ∣ (u a (n + 1) : ℤ) - (u a n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_u_diff_divisible_by_factorial_l1231_123153


namespace NUMINAMATH_CALUDE_function_monotonicity_l1231_123112

theorem function_monotonicity (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) :
  (∀ x, (x^2 - 3*x + 2) * (deriv (deriv f) x) ≤ 0) →
  (∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2) :=
by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l1231_123112


namespace NUMINAMATH_CALUDE_contrapositive_product_nonzero_l1231_123142

theorem contrapositive_product_nonzero (a b : ℝ) :
  (¬(a * b ≠ 0) → ¬(a ≠ 0 ∧ b ≠ 0)) ↔ ((a = 0 ∨ b = 0) → a * b = 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_product_nonzero_l1231_123142


namespace NUMINAMATH_CALUDE_one_tricycle_l1231_123145

/-- The number of cars in the driveway -/
def num_cars : ℕ := 2

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of bikes in the driveway -/
def num_bikes : ℕ := 2

/-- The number of wheels on each bike -/
def wheels_per_bike : ℕ := 2

/-- The number of trash cans in the driveway -/
def num_trash_cans : ℕ := 1

/-- The number of wheels on each trash can -/
def wheels_per_trash_can : ℕ := 2

/-- The number of roller skates (individual skates, not pairs) -/
def num_roller_skates : ℕ := 2

/-- The number of wheels on each roller skate -/
def wheels_per_roller_skate : ℕ := 4

/-- The total number of wheels in the driveway -/
def total_wheels : ℕ := 25

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

theorem one_tricycle :
  ∃ (num_tricycles : ℕ),
    num_tricycles * wheels_per_tricycle =
      total_wheels -
      (num_cars * wheels_per_car +
       num_bikes * wheels_per_bike +
       num_trash_cans * wheels_per_trash_can +
       num_roller_skates * wheels_per_roller_skate) ∧
    num_tricycles = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_tricycle_l1231_123145


namespace NUMINAMATH_CALUDE_oldest_person_is_A_l1231_123135

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person
  | D : Person

-- Define the age relation
def olderThan : Person → Person → Prop := sorry

-- Define the statements made by each person
def statementA : Prop := olderThan Person.B Person.D
def statementB : Prop := olderThan Person.C Person.A
def statementC : Prop := olderThan Person.D Person.C
def statementD : Prop := olderThan Person.B Person.C

-- Define a function to check if a statement is true
def isTrueStatement (p : Person) : Prop :=
  match p with
  | Person.A => statementA
  | Person.B => statementB
  | Person.C => statementC
  | Person.D => statementD

-- Theorem to prove
theorem oldest_person_is_A :
  (∀ (p q : Person), p ≠ q → olderThan p q ∨ olderThan q p) →
  (∀ (p q r : Person), olderThan p q → olderThan q r → olderThan p r) →
  (∃! (p : Person), isTrueStatement p) →
  (∀ (p : Person), isTrueStatement p → ∀ (q : Person), q ≠ p → olderThan p q) →
  (∀ (p : Person), olderThan Person.A p ∨ p = Person.A) :=
sorry

end NUMINAMATH_CALUDE_oldest_person_is_A_l1231_123135


namespace NUMINAMATH_CALUDE_first_five_multiples_average_l1231_123160

theorem first_five_multiples_average (n : ℝ) : 
  (n + 2*n + 3*n + 4*n + 5*n) / 5 = 27 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_five_multiples_average_l1231_123160


namespace NUMINAMATH_CALUDE_exactly_two_statements_true_l1231_123180

def M (x : ℝ) : ℝ := 2 - 4*x
def N (x : ℝ) : ℝ := 4*x + 1

def statement1 : Prop := ¬ ∃ x : ℝ, M x + N x = 0
def statement2 : Prop := ∀ x : ℝ, ¬(M x > 0 ∧ N x > 0)
def statement3 : Prop := ∀ a : ℝ, (∀ x : ℝ, (M x + a) * N x = 1 - 16*x^2) → a = -1
def statement4 : Prop := ∀ x : ℝ, M x * N x = -3 → M x^2 + N x^2 = 11

theorem exactly_two_statements_true : 
  ∃! n : Fin 4, (n.val = 2 ∧ 
    (statement1 ∧ statement3) ∨
    (statement1 ∧ statement2) ∨
    (statement1 ∧ statement4) ∨
    (statement2 ∧ statement3) ∨
    (statement2 ∧ statement4) ∨
    (statement3 ∧ statement4)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_statements_true_l1231_123180


namespace NUMINAMATH_CALUDE_intersection_values_eq_two_eight_l1231_123106

/-- The set of positive real values k for which |z - 4| = 3|z + 4| and |z| = k have exactly one solution in ℂ -/
def intersection_values : Set ℝ :=
  {k : ℝ | k > 0 ∧ ∃! (z : ℂ), Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_values set contains exactly 2 and 8 -/
theorem intersection_values_eq_two_eight : intersection_values = {2, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_values_eq_two_eight_l1231_123106


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1231_123192

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 20) : x - y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1231_123192


namespace NUMINAMATH_CALUDE_cubic_function_property_l1231_123193

/-- Given a cubic function f(x) with certain properties, prove that f(1) has specific values -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => (1/3) * x^3 + a^2 * x^2 + a * x + b
  (f (-1) = -7/12 ∧ (λ x => x^2 + 2*a^2*x + a) (-1) = 0) → 
  (f 1 = 25/12 ∨ f 1 = 1/12) := by
sorry


end NUMINAMATH_CALUDE_cubic_function_property_l1231_123193


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1231_123183

theorem sqrt_equation_solution (x : ℝ) :
  x > 16 →
  (Real.sqrt (x - 8 * Real.sqrt (x - 16)) + 4 = Real.sqrt (x + 8 * Real.sqrt (x - 16)) - 4) ↔
  x ≥ 32 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1231_123183


namespace NUMINAMATH_CALUDE_horizontal_distance_is_three_l1231_123114

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Define the points P and Q
def P : { x : ℝ // f x = 10 } := sorry
def Q : { x : ℝ // |f x| = 2 } := sorry

-- Theorem statement
theorem horizontal_distance_is_three :
  ∃ (xp xq : ℝ), 
    f xp = 10 ∧ 
    |f xq| = 2 ∧ 
    |xp - xq| = 3 :=
sorry

end NUMINAMATH_CALUDE_horizontal_distance_is_three_l1231_123114


namespace NUMINAMATH_CALUDE_quadratic_roots_determine_c_l1231_123102

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := -3 * x^2 + c * x - 8

-- State the theorem
theorem quadratic_roots_determine_c :
  (∀ x : ℝ, f c x < 0 ↔ x < 2 ∨ x > 4) →
  (f c 2 = 0 ∧ f c 4 = 0) →
  c = 18 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_determine_c_l1231_123102


namespace NUMINAMATH_CALUDE_assignment_b_is_valid_l1231_123123

-- Define what a valid assignment statement is
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String), stmt = var ++ "=" ++ expr ∧ var.length > 0

-- Define the specific statement we're checking
def statement_to_check : String := "a=a+1"

-- Theorem to prove
theorem assignment_b_is_valid : is_valid_assignment statement_to_check := by
  sorry

end NUMINAMATH_CALUDE_assignment_b_is_valid_l1231_123123


namespace NUMINAMATH_CALUDE_grape_sales_profit_l1231_123163

/-- Profit function for grape sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 510 * x - 7500

/-- Theorem stating the properties of the profit function -/
theorem grape_sales_profit :
  let w := profit_function
  (w 28 = 1040) ∧
  (∀ x, w x ≤ w (51/2)) ∧
  (w (51/2) = 1102.5) := by
  sorry

end NUMINAMATH_CALUDE_grape_sales_profit_l1231_123163


namespace NUMINAMATH_CALUDE_circle_equation_l1231_123173

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) -/
theorem circle_equation (x y : ℝ) : 
  (∀ D E F : ℝ, (x^2 + y^2 + D*x + E*y + F = 0) → 
    (0^2 + 0^2 + D*0 + E*0 + F = 0 ∧ 
     4^2 + 0^2 + D*4 + E*0 + F = 0 ∧ 
     (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0)) →
  (x^2 + y^2 - 4*x - 6*y = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1231_123173


namespace NUMINAMATH_CALUDE_probability_proof_l1231_123144

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls

def probability_one_white_one_black : ℚ :=
  (white_balls * black_balls : ℚ) / (total_balls * (total_balls - 1) / 2)

theorem probability_proof :
  probability_one_white_one_black = 56 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_proof_l1231_123144


namespace NUMINAMATH_CALUDE_original_average_l1231_123100

theorem original_average (n : ℕ) (A : ℚ) (h1 : n = 15) (h2 : (n : ℚ) * (A + 15) = n * 55) : A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l1231_123100


namespace NUMINAMATH_CALUDE_expression_equals_forty_times_ten_to_power_l1231_123174

theorem expression_equals_forty_times_ten_to_power : 
  (3^1506 + 7^1507)^2 - (3^1506 - 7^1507)^2 = 40 * 10^1506 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_forty_times_ten_to_power_l1231_123174


namespace NUMINAMATH_CALUDE_angle_product_theorem_l1231_123149

theorem angle_product_theorem (α β : Real) (m : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ y = Real.sqrt 3 * x ∧ x < 0 ∧ y < 0) →  -- condition 1
  ((1/2)^2 + m^2 = 1) →  -- condition 2
  (Real.sin α * Real.cos β < 0) →  -- condition 3
  (Real.cos α * Real.sin β = Real.sqrt 3 / 4 ∨ Real.cos α * Real.sin β = -Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_angle_product_theorem_l1231_123149


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l1231_123148

theorem last_three_digits_of_7_to_103 : 7^103 % 1000 = 327 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l1231_123148


namespace NUMINAMATH_CALUDE_inequality_fraction_comparison_l1231_123115

theorem inequality_fraction_comparison (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  a / b > b / a := by
  sorry

end NUMINAMATH_CALUDE_inequality_fraction_comparison_l1231_123115


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l1231_123197

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number satisfying the conditions -/
theorem exists_number_with_digit_sum_decrease : 
  ∃ (n : ℕ), 
    (∃ (m : ℕ), (11 * n = 10 * m)) ∧ 
    (sum_of_digits (11 * n / 10) = (9 * sum_of_digits n) / 10) := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l1231_123197


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l1231_123172

noncomputable def f (x b a : ℝ) : ℝ := Real.log x + x^2 - b*x + a

theorem min_slope_tangent_line (b a : ℝ) (hb : b > 0) :
  ∃ m : ℝ, m = 2 ∧ ∀ x, x > 0 → (1/x + x) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l1231_123172


namespace NUMINAMATH_CALUDE_max_bottles_from_c_and_d_l1231_123198

/-- Represents a shop selling recyclable bottles -/
structure Shop where
  price : ℕ
  available : ℕ

/-- Calculates the total cost of purchasing a given number of bottles from a shop -/
def totalCost (shop : Shop) (bottles : ℕ) : ℕ :=
  shop.price * bottles

theorem max_bottles_from_c_and_d (budget : ℕ) (shopA shopB shopC shopD : Shop) 
  (bottlesA bottlesB : ℕ) :
  budget = 600 ∧
  shopA = { price := 1, available := 200 } ∧
  shopB = { price := 2, available := 150 } ∧
  shopC = { price := 3, available := 100 } ∧
  shopD = { price := 5, available := 50 } ∧
  bottlesA = 150 ∧
  bottlesB = 180 ∧
  bottlesA ≤ shopA.available ∧
  bottlesB ≤ shopB.available →
  ∃ (bottlesC bottlesD : ℕ),
    bottlesC + bottlesD = 30 ∧
    bottlesC ≤ shopC.available ∧
    bottlesD ≤ shopD.available ∧
    totalCost shopA bottlesA + totalCost shopB bottlesB + totalCost shopC bottlesC + totalCost shopD bottlesD = budget ∧
    ∀ (newBottlesC newBottlesD : ℕ),
      newBottlesC ≤ shopC.available →
      newBottlesD ≤ shopD.available →
      totalCost shopA bottlesA + totalCost shopB bottlesB + totalCost shopC newBottlesC + totalCost shopD newBottlesD ≤ budget →
      newBottlesC + newBottlesD ≤ bottlesC + bottlesD :=
by sorry

end NUMINAMATH_CALUDE_max_bottles_from_c_and_d_l1231_123198


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l1231_123116

theorem parallel_vectors_t_value (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, 3) →
  b = (3, t) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l1231_123116


namespace NUMINAMATH_CALUDE_johns_skateboarding_distance_l1231_123108

/-- The total distance John skateboarded, given his journey details -/
def total_skateboarding_distance (initial_skate : ℝ) (walk : ℝ) : ℝ :=
  2 * (initial_skate + walk) - walk

/-- Theorem stating that John's total skateboarding distance is 24 miles -/
theorem johns_skateboarding_distance :
  total_skateboarding_distance 10 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_skateboarding_distance_l1231_123108


namespace NUMINAMATH_CALUDE_find_number_l1231_123157

theorem find_number : ∃ x : ℕ, x * 9999 = 183868020 ∧ x = 18387 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1231_123157


namespace NUMINAMATH_CALUDE_plant_arrangements_eq_144_l1231_123190

/-- The number of ways to arrange 3 distinct vegetable plants and 3 distinct flower plants in a row,
    with all flower plants next to each other -/
def plant_arrangements : ℕ :=
  (Nat.factorial 4) * (Nat.factorial 3)

/-- Theorem stating that the number of plant arrangements is 144 -/
theorem plant_arrangements_eq_144 : plant_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangements_eq_144_l1231_123190


namespace NUMINAMATH_CALUDE_intersection_distance_l1231_123165

-- Define the circle centers and radius
structure CircleConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  r : ℝ

-- Define the conditions
def validConfig (config : CircleConfig) : Prop :=
  1 < config.r ∧ config.r < 2 ∧
  dist config.A config.B = 2 ∧
  dist config.B config.C = 2 ∧
  dist config.A config.C = 2

-- Define the intersection points
def B' (config : CircleConfig) : ℝ × ℝ := sorry
def C' (config : CircleConfig) : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance (config : CircleConfig) 
  (h : validConfig config) :
  dist (B' config) (C' config) = 1 + Real.sqrt (3 * (config.r^2 - 1)) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1231_123165


namespace NUMINAMATH_CALUDE_julio_orange_bottles_l1231_123146

-- Define the number of bottles for each person and soda type
def julio_grape_bottles : ℕ := 7
def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3

-- Define the volume of soda per bottle
def liters_per_bottle : ℕ := 2

-- Define the additional amount of soda Julio has compared to Mateo
def julio_extra_liters : ℕ := 14

-- Define a function to calculate the total liters of soda
def total_liters (orange_bottles grape_bottles : ℕ) : ℕ :=
  (orange_bottles + grape_bottles) * liters_per_bottle

-- State the theorem
theorem julio_orange_bottles : 
  ∃ (julio_orange_bottles : ℕ),
    total_liters julio_orange_bottles julio_grape_bottles = 
    total_liters mateo_orange_bottles mateo_grape_bottles + julio_extra_liters ∧
    julio_orange_bottles = 4 := by
  sorry

end NUMINAMATH_CALUDE_julio_orange_bottles_l1231_123146


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l1231_123156

/-- Calculates the total amount of horse food needed per day on a farm -/
def total_horse_food (num_sheep : ℕ) (sheep_ratio horse_ratio : ℕ) (food_per_horse : ℕ) : ℕ :=
  let num_horses := (num_sheep * horse_ratio) / sheep_ratio
  num_horses * food_per_horse

/-- Theorem: The Stewart farm needs 12,880 ounces of horse food per day -/
theorem stewart_farm_horse_food :
  total_horse_food 40 5 7 230 = 12880 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l1231_123156


namespace NUMINAMATH_CALUDE_sampling_probability_l1231_123181

theorem sampling_probability (m : ℕ) (h_m : m ≥ 2017) :
  let systematic_prob := (3 : ℚ) / 2017
  let stratified_prob := (3 : ℚ) / 2017
  systematic_prob = stratified_prob := by sorry

end NUMINAMATH_CALUDE_sampling_probability_l1231_123181


namespace NUMINAMATH_CALUDE_system_solution_expression_simplification_l1231_123194

-- Part 1: System of equations
theorem system_solution :
  ∃ (x y : ℝ), 2 * x + y = 3 ∧ 3 * x + y = 5 ∧ x = 2 ∧ y = -1 := by sorry

-- Part 2: Expression calculation
theorem expression_simplification (a : ℝ) (h : a ≠ 1) :
  (a^2 / (a^2 - 2*a + 1)) * ((a - 1) / a) - (1 / (a - 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_expression_simplification_l1231_123194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1231_123121

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : ∀ n, S seq n / S seq (2 * n) = (n + 1 : ℚ) / (4 * n + 2)) :
  seq.a 3 / seq.a 5 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1231_123121


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1231_123199

theorem min_sum_of_squares (x y : ℝ) (h : (x + 4) * (y - 4) = 0) :
  ∃ (min : ℝ), min = 32 ∧ ∀ (a b : ℝ), (a + 4) * (b - 4) = 0 → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l1231_123199


namespace NUMINAMATH_CALUDE_smallest_k_for_omega_inequality_l1231_123131

/-- ω(n) denotes the number of positive prime divisors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- The theorem states that 5 is the smallest positive integer k 
    such that 2^ω(n) ≤ k∙n^(1/4) for all positive integers n -/
theorem smallest_k_for_omega_inequality : 
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → (2 : ℝ)^(omega n : ℝ) ≤ k * (n : ℝ)^(1/4)) ∧ 
  (∀ k : ℕ, 0 < k → k < 5 → ∃ n : ℕ, n > 0 ∧ (2 : ℝ)^(omega n : ℝ) > k * (n : ℝ)^(1/4)) ∧
  (∀ n : ℕ, n > 0 → (2 : ℝ)^(omega n : ℝ) ≤ 5 * (n : ℝ)^(1/4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_omega_inequality_l1231_123131


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l1231_123152

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l1231_123152


namespace NUMINAMATH_CALUDE_farmer_plants_two_rows_per_bed_l1231_123179

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  radishes : ℕ
  radish_per_row : ℕ
  plant_beds : ℕ

/-- Calculates the number of rows per plant bed -/
def rows_per_bed (fp : FarmerPlanting) : ℚ :=
  let bean_rows := fp.bean_seedlings / fp.bean_per_row
  let pumpkin_rows := fp.pumpkin_seeds / fp.pumpkin_per_row
  let radish_rows := fp.radishes / fp.radish_per_row
  let total_rows := bean_rows + pumpkin_rows + radish_rows
  total_rows / fp.plant_beds

/-- Theorem stating that for the given planting scenario, the farmer plants 2 rows per bed -/
theorem farmer_plants_two_rows_per_bed (fp : FarmerPlanting)
  (h1 : fp.bean_seedlings = 64)
  (h2 : fp.bean_per_row = 8)
  (h3 : fp.pumpkin_seeds = 84)
  (h4 : fp.pumpkin_per_row = 7)
  (h5 : fp.radishes = 48)
  (h6 : fp.radish_per_row = 6)
  (h7 : fp.plant_beds = 14) :
  rows_per_bed fp = 2 := by
  sorry

#eval rows_per_bed {
  bean_seedlings := 64,
  bean_per_row := 8,
  pumpkin_seeds := 84,
  pumpkin_per_row := 7,
  radishes := 48,
  radish_per_row := 6,
  plant_beds := 14
}

end NUMINAMATH_CALUDE_farmer_plants_two_rows_per_bed_l1231_123179


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l1231_123140

theorem greatest_q_minus_r (q r : ℕ+) (h : 1001 = 17 * q + r) : 
  ∀ (q' r' : ℕ+), 1001 = 17 * q' + r' → q - r ≥ q' - r' := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l1231_123140


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1231_123166

def U : Set ℕ := {x | x > 0 ∧ x < 9}

def M : Set ℕ := {1, 2, 3}

def N : Set ℕ := {3, 4, 5, 6}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1231_123166


namespace NUMINAMATH_CALUDE_cubic_polynomial_c_value_l1231_123155

theorem cubic_polynomial_c_value 
  (g : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, g x = x^3 + a*x^2 + b*x + c) 
  (h2 : ∃ r₁ r₂ r₃ : ℕ, (∀ i, Odd (r₁ + 2*i) ∧ g (r₁ + 2*i) = 0) ∧ 
                        (r₁ < r₂) ∧ (r₂ < r₃) ∧ 
                        g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0)
  (h3 : a + b + c = -11) :
  c = -15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_c_value_l1231_123155


namespace NUMINAMATH_CALUDE_circle_equation_radius_l1231_123127

/-- The radius of a circle given its equation in standard form -/
def circle_radius (h : ℝ) (k : ℝ) (r : ℝ) : ℝ := r

theorem circle_equation_radius :
  circle_radius 1 0 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l1231_123127


namespace NUMINAMATH_CALUDE_first_sales_amount_l1231_123119

/-- Proves that the amount of the first sales is $10 million -/
theorem first_sales_amount (initial_royalties : ℝ) (subsequent_royalties : ℝ) 
  (subsequent_sales : ℝ) (royalty_rate_ratio : ℝ) :
  initial_royalties = 2 →
  subsequent_royalties = 8 →
  subsequent_sales = 100 →
  royalty_rate_ratio = 0.4 →
  ∃ (initial_sales : ℝ), initial_sales = 10 ∧ 
    (initial_royalties / initial_sales = subsequent_royalties / subsequent_sales / royalty_rate_ratio) :=
by sorry

end NUMINAMATH_CALUDE_first_sales_amount_l1231_123119


namespace NUMINAMATH_CALUDE_coffee_beans_remaining_l1231_123101

theorem coffee_beans_remaining (J B B_remaining : ℝ) 
  (h1 : J = 0.25 * (J + B))
  (h2 : J + B_remaining = 0.60 * (J + B))
  (h3 : J > 0)
  (h4 : B > 0) :
  B_remaining / B = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_coffee_beans_remaining_l1231_123101


namespace NUMINAMATH_CALUDE_equal_angle_point_exists_l1231_123139

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Three non-overlapping circles -/
structure ThreeCircles where
  c₁ : Circle
  c₂ : Circle
  c₃ : Circle
  non_overlapping : c₁.center ≠ c₂.center ∧ c₂.center ≠ c₃.center ∧ c₁.center ≠ c₃.center

/-- Distance between two points in 2D plane -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The point from which all circles are seen at the same angle -/
def equal_angle_point (circles : ThreeCircles) (R : ℝ × ℝ) : Prop :=
  let O₁ := circles.c₁.center
  let O₂ := circles.c₂.center
  let O₃ := circles.c₃.center
  let r₁ := circles.c₁.radius
  let r₂ := circles.c₂.radius
  let r₃ := circles.c₃.radius
  (distance O₁ R / distance O₂ R = r₁ / r₂) ∧
  (distance O₂ R / distance O₃ R = r₂ / r₃) ∧
  (distance O₁ R / distance O₃ R = r₁ / r₃)

theorem equal_angle_point_exists (circles : ThreeCircles) :
  ∃ R : ℝ × ℝ, equal_angle_point circles R :=
sorry

end NUMINAMATH_CALUDE_equal_angle_point_exists_l1231_123139


namespace NUMINAMATH_CALUDE_set_equality_problem_l1231_123141

theorem set_equality_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 4} →
  B = {0, 1, a} →
  A ∪ B = {0, 1, 4} →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_set_equality_problem_l1231_123141


namespace NUMINAMATH_CALUDE_normal_dist_peak_l1231_123161

/-- A normal distribution with probability 0.5 of falling within the interval (0.2, +∞) -/
structure NormalDist where
  pdf : ℝ → ℝ
  cdf : ℝ → ℝ
  right_tail_prob : cdf 0.2 = 0.5

/-- The peak of the probability density function occurs at x = 0.2 -/
theorem normal_dist_peak (d : NormalDist) : 
  ∀ x : ℝ, d.pdf x ≤ d.pdf 0.2 :=
sorry

end NUMINAMATH_CALUDE_normal_dist_peak_l1231_123161


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l1231_123117

/-- Given a function g(x) = (x^2 + 3x) / 2, prove that g(x+3) = (x^2 + 9x + 18) / 2 for all real x -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ (x^2 + 3*x) / 2
  g (x + 3) = (x^2 + 9*x + 18) / 2 := by
sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l1231_123117


namespace NUMINAMATH_CALUDE_walking_time_proof_l1231_123188

/-- Proves that walking 1.5 km at 5 km/h takes 18 minutes -/
theorem walking_time_proof (speed : ℝ) (distance : ℝ) (time_minutes : ℝ) : 
  speed = 5 → distance = 1.5 → time_minutes = (distance / speed) * 60 → time_minutes = 18 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_proof_l1231_123188


namespace NUMINAMATH_CALUDE_inequality_proof_l1231_123187

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1231_123187


namespace NUMINAMATH_CALUDE_prob_other_is_one_given_one_is_one_l1231_123171

/-- Represents the number of balls with each label -/
def ballCounts : Fin 3 → Nat
  | 0 => 1  -- number of balls labeled 0
  | 1 => 2  -- number of balls labeled 1
  | 2 => 2  -- number of balls labeled 2

/-- The total number of balls -/
def totalBalls : Nat := (ballCounts 0) + (ballCounts 1) + (ballCounts 2)

/-- The probability of drawing two balls, one of which is labeled 1 -/
def probOneIsOne : ℚ := (ballCounts 1 * (totalBalls - ballCounts 1)) / (totalBalls.choose 2)

/-- The probability of drawing two balls, both labeled 1 -/
def probBothAreOne : ℚ := ((ballCounts 1).choose 2) / (totalBalls.choose 2)

/-- The main theorem to prove -/
theorem prob_other_is_one_given_one_is_one :
  probBothAreOne / probOneIsOne = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_other_is_one_given_one_is_one_l1231_123171


namespace NUMINAMATH_CALUDE_alice_has_largest_result_l1231_123185

def initial_number : ℕ := 15

def alice_result (n : ℕ) : ℕ := n * 3 - 2 + 4

def bob_result (n : ℕ) : ℕ := n * 2 + 3 - 5

def charlie_result (n : ℕ) : ℕ := ((n + 5) / 2) * 4

theorem alice_has_largest_result :
  alice_result initial_number > bob_result initial_number ∧
  alice_result initial_number > charlie_result initial_number := by
  sorry

end NUMINAMATH_CALUDE_alice_has_largest_result_l1231_123185


namespace NUMINAMATH_CALUDE_souvenir_purchasing_plans_l1231_123136

def number_of_purchasing_plans (total_items : ℕ) (types : ℕ) (items_per_type : ℕ) : ℕ :=
  let f (x : ℕ → ℕ) := x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10
  let coefficient_of_x25 := (Nat.choose 24 3) - 4 * (Nat.choose 14 3) + 6 * (Nat.choose 4 3)
  coefficient_of_x25

theorem souvenir_purchasing_plans :
  number_of_purchasing_plans 25 4 10 = 592 :=
sorry

end NUMINAMATH_CALUDE_souvenir_purchasing_plans_l1231_123136


namespace NUMINAMATH_CALUDE_probability_diamond_or_ace_l1231_123151

def standard_deck : ℕ := 52

def diamond_count : ℕ := 13

def ace_count : ℕ := 4

def favorable_outcomes : ℕ := diamond_count + ace_count - 1

theorem probability_diamond_or_ace :
  (favorable_outcomes : ℚ) / standard_deck = 4 / 13 :=
by sorry

end NUMINAMATH_CALUDE_probability_diamond_or_ace_l1231_123151


namespace NUMINAMATH_CALUDE_roller_coaster_cars_l1231_123143

theorem roller_coaster_cars (n : ℕ) (h : n > 0) :
  (n - 1 : ℚ) / n = 1/2 ↔ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cars_l1231_123143


namespace NUMINAMATH_CALUDE_minimizing_n_is_six_l1231_123154

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  sum : ℕ → ℤ  -- The sum function
  first_fifth_sum : a 1 + a 5 = -14
  ninth_sum : sum 9 = -27

/-- The value of n that minimizes the sum of the first n terms -/
def minimizing_n (seq : ArithmeticSequence) : ℕ :=
  6

/-- Theorem stating that 6 is the value of n that minimizes S_n -/
theorem minimizing_n_is_six (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.sum n ≥ seq.sum (minimizing_n seq) :=
sorry

end NUMINAMATH_CALUDE_minimizing_n_is_six_l1231_123154


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l1231_123177

theorem stratified_sampling_sum (total_population : ℕ) (sample_size : ℕ) 
  (type_a_count : ℕ) (type_b_count : ℕ) :
  total_population = 100 →
  sample_size = 20 →
  type_a_count = 10 →
  type_b_count = 20 →
  (type_a_count * sample_size / total_population + 
   type_b_count * sample_size / total_population : ℚ) = 6 := by
  sorry

#check stratified_sampling_sum

end NUMINAMATH_CALUDE_stratified_sampling_sum_l1231_123177


namespace NUMINAMATH_CALUDE_sugar_cube_weight_l1231_123126

theorem sugar_cube_weight
  (ants1 : ℕ) (cubes1 : ℕ) (weight1 : ℝ) (hours1 : ℝ)
  (ants2 : ℕ) (cubes2 : ℕ) (hours2 : ℝ)
  (h1 : ants1 = 15)
  (h2 : cubes1 = 600)
  (h3 : weight1 = 10)
  (h4 : hours1 = 5)
  (h5 : ants2 = 20)
  (h6 : cubes2 = 960)
  (h7 : hours2 = 3)
  : ∃ weight2 : ℝ,
    weight2 = 5 ∧
    (ants1 : ℝ) * (cubes1 : ℝ) * weight1 / hours1 =
    (ants2 : ℝ) * (cubes2 : ℝ) * weight2 / hours2 :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_cube_weight_l1231_123126


namespace NUMINAMATH_CALUDE_minimum_contribution_l1231_123124

theorem minimum_contribution 
  (n : ℕ) 
  (total : ℝ) 
  (max_individual : ℝ) 
  (h1 : n = 15) 
  (h2 : total = 30) 
  (h3 : max_individual = 16) : 
  ∃ (min_contribution : ℝ), 
    (∀ (i : ℕ), i ≤ n → min_contribution ≤ max_individual) ∧ 
    (n * min_contribution ≤ total) ∧ 
    (∀ (x : ℝ), (∀ (i : ℕ), i ≤ n → x ≤ max_individual) ∧ (n * x ≤ total) → x ≤ min_contribution) ∧
    min_contribution = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_contribution_l1231_123124


namespace NUMINAMATH_CALUDE_expression_equality_l1231_123129

theorem expression_equality : -Real.sqrt 4 + |(-Real.sqrt 2 - 1)| + (π - 2013)^0 - (1/5)^0 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1231_123129


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l1231_123195

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem for part (1)
theorem solution_set_part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
by sorry

-- Theorem for part (2)
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l1231_123195


namespace NUMINAMATH_CALUDE_middle_number_is_five_l1231_123111

/-- Represents a triple of positive integers in increasing order -/
structure IncreasingTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a < b
  h2 : b < c

/-- The set of all valid triples according to the problem conditions -/
def ValidTriples : Set IncreasingTriple :=
  { t : IncreasingTriple | t.a + t.b + t.c = 16 }

/-- A triple is ambiguous if there exists another valid triple with the same middle number -/
def IsAmbiguous (t : IncreasingTriple) : Prop :=
  ∃ t' : IncreasingTriple, t' ∈ ValidTriples ∧ t' ≠ t ∧ t'.b = t.b

theorem middle_number_is_five :
  ∀ t ∈ ValidTriples, IsAmbiguous t → t.b = 5 := by sorry

end NUMINAMATH_CALUDE_middle_number_is_five_l1231_123111


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_39_l1231_123158

theorem quadratic_expression_equals_39 (x : ℝ) :
  (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 + 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_39_l1231_123158


namespace NUMINAMATH_CALUDE_sum_product_equality_l1231_123168

theorem sum_product_equality (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l1231_123168


namespace NUMINAMATH_CALUDE_fraction_equality_l1231_123167

theorem fraction_equality (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  let C : ℝ := 19 / 5
  let D : ℝ := 17 / 5
  (D * x - 17) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1231_123167


namespace NUMINAMATH_CALUDE_excess_calories_james_james_excess_calories_l1231_123184

/-- Calculates the excess calories James ate after eating Cheezits and going for a run -/
theorem excess_calories_james (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ) 
  (run_duration : ℕ) (calories_burned_per_minute : ℕ) : ℕ :=
  let total_ounces := bags * ounces_per_bag
  let total_calories_consumed := total_ounces * calories_per_ounce
  let total_calories_burned := run_duration * calories_burned_per_minute
  total_calories_consumed - total_calories_burned

/-- Proves that James ate 420 excess calories -/
theorem james_excess_calories : 
  excess_calories_james 3 2 150 40 12 = 420 := by
  sorry

end NUMINAMATH_CALUDE_excess_calories_james_james_excess_calories_l1231_123184


namespace NUMINAMATH_CALUDE_markus_more_marbles_l1231_123162

theorem markus_more_marbles (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) 
  (markus_bags : ℕ) (markus_marbles_per_bag : ℕ) 
  (h1 : mara_bags = 12) (h2 : mara_marbles_per_bag = 2) 
  (h3 : markus_bags = 2) (h4 : markus_marbles_per_bag = 13) : 
  markus_bags * markus_marbles_per_bag - mara_bags * mara_marbles_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_markus_more_marbles_l1231_123162


namespace NUMINAMATH_CALUDE_oarsmen_count_l1231_123128

theorem oarsmen_count (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 1.8 →
  old_weight = 53 →
  new_weight = 71 →
  (new_weight - old_weight) / average_increase = 10 := by
sorry

end NUMINAMATH_CALUDE_oarsmen_count_l1231_123128


namespace NUMINAMATH_CALUDE_prob_between_30_and_40_l1231_123130

/-- Represents the age groups in the population -/
inductive AgeGroup
  | LessThan20
  | Between20And30
  | Between30And40
  | MoreThan40

/-- Represents the population with their age distribution -/
structure Population where
  total : ℕ
  ageDist : AgeGroup → ℕ
  sum_eq_total : (ageDist AgeGroup.LessThan20) + (ageDist AgeGroup.Between20And30) + 
                 (ageDist AgeGroup.Between30And40) + (ageDist AgeGroup.MoreThan40) = total

/-- The probability of selecting a person from a specific age group -/
def prob (p : Population) (ag : AgeGroup) : ℚ :=
  (p.ageDist ag : ℚ) / (p.total : ℚ)

/-- The given population -/
def givenPopulation : Population where
  total := 200
  ageDist := fun
    | AgeGroup.LessThan20 => 20
    | AgeGroup.Between20And30 => 30
    | AgeGroup.Between30And40 => 70
    | AgeGroup.MoreThan40 => 80
  sum_eq_total := by sorry

theorem prob_between_30_and_40 : 
  prob givenPopulation AgeGroup.Between30And40 = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_prob_between_30_and_40_l1231_123130


namespace NUMINAMATH_CALUDE_ruth_apples_l1231_123105

theorem ruth_apples (x : ℕ) : x - 5 = 84 → x = 89 := by sorry

end NUMINAMATH_CALUDE_ruth_apples_l1231_123105


namespace NUMINAMATH_CALUDE_bears_captured_pieces_l1231_123113

theorem bears_captured_pieces (H B F : ℕ) : 
  (64 : ℕ) = H + B + F →
  H = B / 2 →
  H = F / 5 →
  (0 : ℕ) = 16 - B :=
by sorry

end NUMINAMATH_CALUDE_bears_captured_pieces_l1231_123113


namespace NUMINAMATH_CALUDE_sam_distance_l1231_123169

/-- The distance traveled by Sam given his walking speed and duration -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that Sam's traveled distance is 8 miles -/
theorem sam_distance :
  let speed := 4 -- miles per hour
  let time := 2 -- hours
  distance_traveled speed time = 8 := by sorry

end NUMINAMATH_CALUDE_sam_distance_l1231_123169


namespace NUMINAMATH_CALUDE_equilateral_triangle_circles_l1231_123189

theorem equilateral_triangle_circles (rA rB rC : ℝ) : 
  rA < rB ∧ rB < rC →  -- radii form increasing sequence
  ∃ (d : ℝ), rB = rA + d ∧ rC = rA + 2*d →  -- arithmetic sequence
  6 - (rA + rB) = 3.5 →  -- shortest distance between circles A and B
  6 - (rA + rC) = 3 →  -- shortest distance between circles A and C
  (1/6) * (π * rA^2 + π * rB^2 + π * rC^2) = 29*π/24 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circles_l1231_123189


namespace NUMINAMATH_CALUDE_josh_marbles_count_l1231_123170

/-- The number of marbles Josh has after receiving some from Jack -/
def total_marbles (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Josh's final marble count is the sum of his initial count and received marbles -/
theorem josh_marbles_count (initial : ℕ) (received : ℕ) :
  total_marbles initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_count_l1231_123170


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1231_123150

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 4 ∧ b = 5) ∨ (a = 4 ∧ c = 5) ∨ (b = 4 ∧ c = 5)) →
  c = Real.sqrt 41 ∨ (a = 3 ∨ b = 3) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1231_123150


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l1231_123164

theorem sum_of_cubes_difference (p q r : ℕ+) :
  (p + q + r : ℕ)^3 - p^3 - q^3 - r^3 = 200 →
  (p : ℕ) + q + r = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l1231_123164


namespace NUMINAMATH_CALUDE_carl_reaches_goal_in_53_days_l1231_123103

/-- Represents Carl's earnings and candy bar goal --/
structure CarlsEarnings where
  candy_bar_cost : ℚ
  weekly_trash_pay : ℚ
  biweekly_dog_pay : ℚ
  aunt_payment : ℚ
  candy_bar_goal : ℕ

/-- Calculates the number of days needed for Carl to reach his candy bar goal --/
def days_to_reach_goal (e : CarlsEarnings) : ℕ :=
  sorry

/-- Theorem stating that given Carl's specific earnings and goal, it takes 53 days to reach the goal --/
theorem carl_reaches_goal_in_53_days :
  let e : CarlsEarnings := {
    candy_bar_cost := 1/2,
    weekly_trash_pay := 3/4,
    biweekly_dog_pay := 5/4,
    aunt_payment := 5,
    candy_bar_goal := 30
  }
  days_to_reach_goal e = 53 := by
  sorry

end NUMINAMATH_CALUDE_carl_reaches_goal_in_53_days_l1231_123103


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1231_123138

def total_cost (total_bricks : ℕ) (discount1_percent : ℚ) (discount2_percent : ℚ) 
                (full_price : ℚ) (discount1_fraction : ℚ) (discount2_fraction : ℚ)
                (full_price_fraction : ℚ) (additional_cost : ℚ) : ℚ :=
  let discounted_price1 := full_price * (1 - discount1_percent)
  let discounted_price2 := full_price * (1 - discount2_percent)
  let cost1 := (total_bricks : ℚ) * discount1_fraction * discounted_price1
  let cost2 := (total_bricks : ℚ) * discount2_fraction * discounted_price2
  let cost3 := (total_bricks : ℚ) * full_price_fraction * full_price
  cost1 + cost2 + cost3 + additional_cost

theorem total_cost_calculation :
  total_cost 1000 (1/2) (1/5) (1/2) (3/10) (2/5) (3/10) 200 = 585 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1231_123138
