import Mathlib

namespace NUMINAMATH_CALUDE_xiao_ming_brother_age_l4145_414575

def has_no_repeated_digits (year : ℕ) : Prop := sorry

def is_multiple_of_19 (year : ℕ) : Prop := year % 19 = 0

theorem xiao_ming_brother_age (birth_year : ℕ) 
  (h1 : is_multiple_of_19 birth_year)
  (h2 : ∀ y : ℕ, birth_year ≤ y → y < 2013 → ¬(has_no_repeated_digits y))
  (h3 : has_no_repeated_digits 2013) :
  2013 - birth_year = 18 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_brother_age_l4145_414575


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4145_414532

theorem sum_of_squares_of_roots (a b c : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = -12) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 73/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4145_414532


namespace NUMINAMATH_CALUDE_tommy_savings_tommy_current_savings_l4145_414500

/-- Calculates the amount of money Tommy already has -/
theorem tommy_savings (num_books : ℕ) (cost_per_book : ℕ) (amount_to_save : ℕ) : ℕ :=
  num_books * cost_per_book - amount_to_save

/-- Proves that Tommy already has $13 -/
theorem tommy_current_savings : tommy_savings 8 5 27 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tommy_savings_tommy_current_savings_l4145_414500


namespace NUMINAMATH_CALUDE_cubic_function_properties_l4145_414506

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define monotonically increasing function
def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement
theorem cubic_function_properties :
  isPowerFunction f ∧ isMonotonicallyIncreasing f :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l4145_414506


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l4145_414522

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ -2 ∧ a ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l4145_414522


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_positive_l4145_414561

theorem sum_of_x_and_y_positive (x y : ℝ) (h : 2 * x + 3 * y > 2 - y + 3 - x) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_positive_l4145_414561


namespace NUMINAMATH_CALUDE_lower_limit_x_l4145_414567

/-- The function f(x) = x - 5 -/
def f (x : ℝ) : ℝ := x - 5

/-- The lower limit of x for which f(x) ≤ 8 is 13 -/
theorem lower_limit_x (x : ℝ) : f x ≤ 8 ↔ x ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_x_l4145_414567


namespace NUMINAMATH_CALUDE_min_a_value_l4145_414585

noncomputable def f (x : ℝ) : ℝ := (2 * 2023^x) / (2023^x + 1)

theorem min_a_value (a : ℝ) :
  (∀ x : ℝ, x > 0 → f (a * Real.exp x) ≥ 2 - f (Real.log a - Real.log x)) →
  a ≥ 1 / Real.exp 1 ∧ ∀ b : ℝ, (∀ x : ℝ, x > 0 → f (b * Real.exp x) ≥ 2 - f (Real.log b - Real.log x)) → b ≥ 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l4145_414585


namespace NUMINAMATH_CALUDE_part_one_part_two_l4145_414578

-- Define the inequalities p and q
def p (x a : ℝ) : Prop := x^2 - 6*a*x + 8*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Theorem for part 1
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x : ℝ, p x a → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p x a))

-- Theorem for part 2
theorem part_two :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4145_414578


namespace NUMINAMATH_CALUDE_apple_difference_l4145_414545

theorem apple_difference (martha_apples harry_apples : ℕ) 
  (h1 : martha_apples = 68)
  (h2 : harry_apples = 19)
  (h3 : ∃ tim_apples : ℕ, tim_apples = 2 * harry_apples ∧ tim_apples < martha_apples) :
  martha_apples - (2 * harry_apples) = 30 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l4145_414545


namespace NUMINAMATH_CALUDE_time_difference_between_arrivals_l4145_414582

/-- Represents the problem of calculating the time difference between arrivals of a car and a minivan --/
theorem time_difference_between_arrivals 
  (car_speed : ℝ) 
  (minivan_speed : ℝ) 
  (passing_time_before_arrival : ℝ) :
  car_speed = 40 →
  minivan_speed = 50 →
  passing_time_before_arrival = 1/6 →
  ∃ (time_difference : ℝ),
    time_difference = passing_time_before_arrival - (car_speed * passing_time_before_arrival) / minivan_speed ∧
    time_difference * 60 = 2 := by
  sorry


end NUMINAMATH_CALUDE_time_difference_between_arrivals_l4145_414582


namespace NUMINAMATH_CALUDE_fuel_in_truck_is_38_l4145_414546

/-- Calculates the amount of fuel already in a truck given the total capacity,
    amount spent, change received, and cost per liter. -/
def fuel_already_in_truck (total_capacity : ℕ) (amount_spent : ℕ) (change : ℕ) (cost_per_liter : ℕ) : ℕ :=
  total_capacity - (amount_spent - change) / cost_per_liter

/-- Proves that given the specific conditions, the amount of fuel already in the truck is 38 liters. -/
theorem fuel_in_truck_is_38 :
  fuel_already_in_truck 150 350 14 3 = 38 := by
  sorry

#eval fuel_already_in_truck 150 350 14 3

end NUMINAMATH_CALUDE_fuel_in_truck_is_38_l4145_414546


namespace NUMINAMATH_CALUDE_dave_ice_cubes_l4145_414580

theorem dave_ice_cubes (original : ℕ) (new : ℕ) (total : ℕ) : 
  original = 2 → new = 7 → total = original + new → total = 9 := by
  sorry

end NUMINAMATH_CALUDE_dave_ice_cubes_l4145_414580


namespace NUMINAMATH_CALUDE_curve_C_bound_expression_l4145_414597

theorem curve_C_bound_expression (x y : ℝ) :
  4 * x^2 + y^2 = 16 →
  -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_curve_C_bound_expression_l4145_414597


namespace NUMINAMATH_CALUDE_double_series_convergence_l4145_414544

/-- The double series ∑_{m=1}^∞ ∑_{n=1}^∞ 1/(m(m+n+2)) converges to 1 -/
theorem double_series_convergence :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * (m + n + 2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_double_series_convergence_l4145_414544


namespace NUMINAMATH_CALUDE_wilsborough_change_l4145_414510

/-- Calculates the change Mrs. Wilsborough received after buying concert tickets -/
theorem wilsborough_change : 
  let vip_price : ℕ := 120
  let regular_price : ℕ := 60
  let discount_price : ℕ := 30
  let vip_count : ℕ := 4
  let regular_count : ℕ := 5
  let discount_count : ℕ := 3
  let payment : ℕ := 1000
  let total_cost : ℕ := vip_price * vip_count + regular_price * regular_count + discount_price * discount_count
  payment - total_cost = 130 := by
  sorry

end NUMINAMATH_CALUDE_wilsborough_change_l4145_414510


namespace NUMINAMATH_CALUDE_noemi_initial_money_l4145_414571

/-- The amount of money Noemi lost on roulette -/
def roulette_loss : ℕ := 400

/-- The amount of money Noemi lost on blackjack -/
def blackjack_loss : ℕ := 500

/-- The amount of money Noemi still has in her purse -/
def remaining_money : ℕ := 800

/-- The initial amount of money Noemi had -/
def initial_money : ℕ := roulette_loss + blackjack_loss + remaining_money

theorem noemi_initial_money : 
  initial_money = roulette_loss + blackjack_loss + remaining_money := by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_money_l4145_414571


namespace NUMINAMATH_CALUDE_sin_even_translation_l4145_414538

theorem sin_even_translation (φ : Real) : 
  (∀ x, (1/2) * Real.sin (2*x + φ - π/4) = (1/2) * Real.sin (2*(-x) + φ - π/4)) → 
  ∃ k : Int, φ = π/4 + k * π :=
by sorry

end NUMINAMATH_CALUDE_sin_even_translation_l4145_414538


namespace NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l4145_414584

theorem lcm_from_product_and_gcd (a b c : ℕ+) :
  a * b * c = 1354808 ∧ Nat.gcd a (Nat.gcd b c) = 11 →
  Nat.lcm a (Nat.lcm b c) = 123164 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l4145_414584


namespace NUMINAMATH_CALUDE_toy_production_proof_l4145_414504

/-- A factory produces toys. -/
structure ToyFactory where
  weekly_production : ℕ
  working_days : ℕ
  uniform_production : Bool

/-- Calculate the daily toy production for a given factory. -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.working_days

theorem toy_production_proof (factory : ToyFactory) 
  (h1 : factory.weekly_production = 5505)
  (h2 : factory.working_days = 5)
  (h3 : factory.uniform_production = true) :
  daily_production factory = 1101 := by
  sorry

#eval daily_production { weekly_production := 5505, working_days := 5, uniform_production := true }

end NUMINAMATH_CALUDE_toy_production_proof_l4145_414504


namespace NUMINAMATH_CALUDE_rhombus_properties_l4145_414507

/-- Properties of a rhombus -/
structure Rhombus where
  /-- The diagonals of a rhombus are perpendicular to each other -/
  diagonals_perpendicular : Prop
  /-- The diagonals of a rhombus bisect each other -/
  diagonals_bisect : Prop

theorem rhombus_properties (R : Rhombus) : 
  (R.diagonals_perpendicular ∨ R.diagonals_bisect) ∧ 
  (R.diagonals_perpendicular ∧ R.diagonals_bisect) ∧ 
  ¬(¬R.diagonals_perpendicular) := by
  sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_rhombus_properties_l4145_414507


namespace NUMINAMATH_CALUDE_subway_to_bike_speed_ratio_l4145_414548

/-- The ratio of subway speed to mountain bike speed -/
theorem subway_to_bike_speed_ratio :
  ∀ (v_bike v_subway : ℝ),
  v_bike > 0 →
  v_subway > 0 →
  10 * v_bike + 40 * v_subway = 210 * v_bike →
  v_subway / v_bike = 5 := by
sorry

end NUMINAMATH_CALUDE_subway_to_bike_speed_ratio_l4145_414548


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l4145_414563

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem: "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  ¬(∀ a : ℕ → ℝ, ∀ q : ℝ, GeometricSequence a q → (q > 1 → IncreasingSequence a)) ∧
  ¬(∀ a : ℕ → ℝ, ∀ q : ℝ, GeometricSequence a q → (IncreasingSequence a → q > 1)) :=
sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l4145_414563


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_sum_of_coefficients_is_eight_l4145_414593

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 2 * (4 * x^6 + 9 * x^3 - 5) + 8 * (x^4 - 8 * x + 6)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one :
  (p 1) = 8 := by sorry

/-- The sum of coefficients of the given polynomial is 8 -/
theorem sum_of_coefficients_is_eight :
  ∃ (f : ℝ → ℝ), (∀ x, f x = p x) ∧ (f 1 = 8) := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_sum_of_coefficients_is_eight_l4145_414593


namespace NUMINAMATH_CALUDE_plan_b_more_cost_effective_l4145_414537

/-- Plan A's cost per megabyte in cents -/
def plan_a_cost_per_mb : ℚ := 12

/-- Plan B's setup fee in cents -/
def plan_b_setup_fee : ℚ := 3000

/-- Plan B's cost per megabyte in cents -/
def plan_b_cost_per_mb : ℚ := 8

/-- The minimum number of megabytes for Plan B to be more cost-effective -/
def min_mb_for_plan_b : ℕ := 751

theorem plan_b_more_cost_effective :
  (↑min_mb_for_plan_b * plan_b_cost_per_mb + plan_b_setup_fee < ↑min_mb_for_plan_b * plan_a_cost_per_mb) ∧
  ∀ m : ℕ, m < min_mb_for_plan_b →
    (↑m * plan_b_cost_per_mb + plan_b_setup_fee ≥ ↑m * plan_a_cost_per_mb) :=
by sorry

end NUMINAMATH_CALUDE_plan_b_more_cost_effective_l4145_414537


namespace NUMINAMATH_CALUDE_frank_weekly_spending_l4145_414596

theorem frank_weekly_spending (lawn_money weed_money weeks : ℕ) 
  (h1 : lawn_money = 5)
  (h2 : weed_money = 58)
  (h3 : weeks = 9) :
  (lawn_money + weed_money) / weeks = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_weekly_spending_l4145_414596


namespace NUMINAMATH_CALUDE_line_intersects_circle_l4145_414599

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - point_P.1) + point_P.2

-- Theorem statement
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_through_P m x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l4145_414599


namespace NUMINAMATH_CALUDE_jules_dog_walking_rate_l4145_414581

/-- Proves that Jules charges $1.25 per block for dog walking -/
theorem jules_dog_walking_rate :
  let vacation_cost : ℚ := 1000
  let family_members : ℕ := 5
  let start_fee : ℚ := 2
  let dogs_walked : ℕ := 20
  let total_blocks : ℕ := 128
  let individual_contribution := vacation_cost / family_members
  let total_start_fees := start_fee * dogs_walked
  let remaining_to_earn := individual_contribution - total_start_fees
  let rate_per_block := remaining_to_earn / total_blocks
  rate_per_block = 1.25 := by
sorry


end NUMINAMATH_CALUDE_jules_dog_walking_rate_l4145_414581


namespace NUMINAMATH_CALUDE_complex_modulus_l4145_414527

theorem complex_modulus (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l4145_414527


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l4145_414554

/-- The ratio of the volume of a cube with edge length 4 inches to the volume of a cube with edge length 2 feet is 1/216. -/
theorem cube_volume_ratio : 
  let small_cube_edge : ℚ := 4 / 12  -- 4 inches in feet
  let large_cube_edge : ℚ := 2       -- 2 feet
  (small_cube_edge ^ 3) / (large_cube_edge ^ 3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l4145_414554


namespace NUMINAMATH_CALUDE_range_of_a_given_p_and_q_l4145_414534

-- Define the propositions
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - Real.log x - a ≥ 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0

-- Define the range of a
def range_of_a : Set ℝ :=
  Set.Ici (-4) ∪ Set.Icc (-2) 1

-- State the theorem
theorem range_of_a_given_p_and_q :
  ∀ a : ℝ, prop_p a ∧ prop_q a ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_given_p_and_q_l4145_414534


namespace NUMINAMATH_CALUDE_correct_borrowing_process_l4145_414557

/-- Represents the steps in the book borrowing process -/
inductive BorrowingStep
  | StorageEntry
  | LocatingBook
  | Reading
  | Borrowing
  | StorageExit
  | Returning

/-- Defines the correct order of the book borrowing process -/
def correctBorrowingOrder : List BorrowingStep :=
  [BorrowingStep.StorageEntry, BorrowingStep.LocatingBook, BorrowingStep.Reading, 
   BorrowingStep.Borrowing, BorrowingStep.StorageExit, BorrowingStep.Returning]

/-- Theorem stating that the defined order is correct -/
theorem correct_borrowing_process :
  correctBorrowingOrder = [BorrowingStep.StorageEntry, BorrowingStep.LocatingBook, 
    BorrowingStep.Reading, BorrowingStep.Borrowing, BorrowingStep.StorageExit, 
    BorrowingStep.Returning] :=
by
  sorry


end NUMINAMATH_CALUDE_correct_borrowing_process_l4145_414557


namespace NUMINAMATH_CALUDE_die_roll_sequences_l4145_414533

/-- The number of sides on the die -/
def num_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 6

/-- The number of distinct sequences when rolling a die -/
def num_sequences : ℕ := num_sides ^ num_rolls

theorem die_roll_sequences :
  num_sequences = 46656 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_sequences_l4145_414533


namespace NUMINAMATH_CALUDE_martha_coffee_spending_cut_l4145_414566

def coffee_spending_cut_percentage (latte_cost : ℚ) (iced_coffee_cost : ℚ)
  (lattes_per_week : ℕ) (iced_coffees_per_week : ℕ) (weeks_per_year : ℕ)
  (savings_goal : ℚ) : ℚ :=
  let weekly_spending := latte_cost * lattes_per_week + iced_coffee_cost * iced_coffees_per_week
  let annual_spending := weekly_spending * weeks_per_year
  (savings_goal / annual_spending) * 100

theorem martha_coffee_spending_cut (latte_cost : ℚ) (iced_coffee_cost : ℚ)
  (lattes_per_week : ℕ) (iced_coffees_per_week : ℕ) (weeks_per_year : ℕ)
  (savings_goal : ℚ) :
  latte_cost = 4 →
  iced_coffee_cost = 2 →
  lattes_per_week = 5 →
  iced_coffees_per_week = 3 →
  weeks_per_year = 52 →
  savings_goal = 338 →
  coffee_spending_cut_percentage latte_cost iced_coffee_cost lattes_per_week
    iced_coffees_per_week weeks_per_year savings_goal = 25 :=
by sorry

end NUMINAMATH_CALUDE_martha_coffee_spending_cut_l4145_414566


namespace NUMINAMATH_CALUDE_expression_evaluation_l4145_414543

theorem expression_evaluation :
  Real.sqrt ((16^6 + 2^18) / (16^3 + 2^21)) = (8 * Real.sqrt 65) / Real.sqrt 513 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4145_414543


namespace NUMINAMATH_CALUDE_solution_set_solution_characterization_solution_equals_intervals_l4145_414594

theorem solution_set : Set ℝ := by
  sorry

theorem solution_characterization :
  solution_set = {x : ℝ | 2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ∧ (x - 3)^2 ≤ 16} := by
  sorry

theorem solution_equals_intervals :
  solution_set = Set.Icc (-1) 1 ∪ Set.Icc 5 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_solution_characterization_solution_equals_intervals_l4145_414594


namespace NUMINAMATH_CALUDE_trigonometric_identities_l4145_414519

theorem trigonometric_identities :
  (Real.cos (15 * π / 180))^2 - (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (π / 8) * Real.cos (π / 8) = Real.sqrt 2 / 4 ∧
  Real.tan (15 * π / 180) = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l4145_414519


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l4145_414524

theorem computer_literate_female_employees 
  (total_employees : ℕ)
  (female_percentage : ℚ)
  (male_literate_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (h_total : total_employees = 1300)
  (h_female : female_percentage = 60 / 100)
  (h_male_literate : male_literate_percentage = 50 / 100)
  (h_total_literate : total_literate_percentage = 62 / 100) :
  ↑(total_employees * female_percentage * total_literate_percentage - 
    total_employees * (1 - female_percentage) * male_literate_percentage : ℚ).num = 546 := by
  sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l4145_414524


namespace NUMINAMATH_CALUDE_divide_five_children_l4145_414502

/-- The number of ways to divide n distinguishable objects into two non-empty, 
    unordered groups, where rotations within groups and swapping of groups 
    don't create new arrangements -/
def divide_into_two_groups (n : ℕ) : ℕ :=
  sorry

/-- There are 5 children to be divided -/
def num_children : ℕ := 5

/-- The theorem stating that the number of ways to divide 5 children
    into two groups under the given conditions is 50 -/
theorem divide_five_children : 
  divide_into_two_groups num_children = 50 := by
  sorry

end NUMINAMATH_CALUDE_divide_five_children_l4145_414502


namespace NUMINAMATH_CALUDE_no_inscribable_2010_gon_l4145_414505

theorem no_inscribable_2010_gon : ¬ ∃ (sides : Fin 2010 → ℕ), 
  (∀ i : Fin 2010, 1 ≤ sides i ∧ sides i ≤ 2010) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 2010 → ∃ i : Fin 2010, sides i = n) ∧
  (∃ r : ℝ, r > 0 ∧ ∀ i : Fin 2010, 
    ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = (sides i)^2 ∧ a * b = r * (sides i)) :=
by sorry

end NUMINAMATH_CALUDE_no_inscribable_2010_gon_l4145_414505


namespace NUMINAMATH_CALUDE_input_for_output_nine_l4145_414558

theorem input_for_output_nine (x : ℝ) (y : ℝ) : 
  (x < 0 → y = (x + 1)^2) ∧
  (x ≥ 0 → y = (x - 1)^2) ∧
  (y = 9) →
  (x = -4 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_input_for_output_nine_l4145_414558


namespace NUMINAMATH_CALUDE_time_to_see_again_value_l4145_414513

/-- The time (in seconds) before Jenny and Kenny can see each other again -/
def time_to_see_again (jenny_speed : ℝ) (kenny_speed : ℝ) (path_distance : ℝ) (building_diameter : ℝ) (initial_distance : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time before Jenny and Kenny can see each other again -/
theorem time_to_see_again_value :
  time_to_see_again 2 4 300 150 300 = 48 := by
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_value_l4145_414513


namespace NUMINAMATH_CALUDE_m_function_inequality_l4145_414521

/-- An M-function is a function f: ℝ → ℝ defined on (0, +∞) that satisfies xf''(x) > f(x) for all x in (0, +∞) -/
def is_M_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x * (deriv^[2] f x) > f x

/-- Theorem: For any M-function f and positive real numbers x₁ and x₂, 
    the sum f(x₁) + f(x₂) is less than f(x₁ + x₂) -/
theorem m_function_inequality (f : ℝ → ℝ) (hf : is_M_function f) 
  (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  f x₁ + f x₂ < f (x₁ + x₂) :=
sorry

end NUMINAMATH_CALUDE_m_function_inequality_l4145_414521


namespace NUMINAMATH_CALUDE_radish_patch_area_l4145_414536

theorem radish_patch_area (pea_patch : ℝ) (radish_patch : ℝ) : 
  pea_patch = 2 * radish_patch →
  pea_patch / 6 = 5 →
  radish_patch = 15 := by
  sorry

end NUMINAMATH_CALUDE_radish_patch_area_l4145_414536


namespace NUMINAMATH_CALUDE_derivative_product_at_one_and_neg_one_l4145_414526

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + abs x)

-- State the theorem
theorem derivative_product_at_one_and_neg_one :
  (deriv f 1) * (deriv f (-1)) = 9 := by sorry

end NUMINAMATH_CALUDE_derivative_product_at_one_and_neg_one_l4145_414526


namespace NUMINAMATH_CALUDE_inequality_proof_l4145_414560

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8 / (x * y) + y^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4145_414560


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4145_414579

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4145_414579


namespace NUMINAMATH_CALUDE_functions_satisfy_equation_l4145_414528

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (x : ℝ) : ℝ := a * x^2 + b * x + c
def h (x : ℝ) : ℝ := a * x + b

theorem functions_satisfy_equation :
  ∀ (x y : ℝ), f a b c x - g a b c y = (x - y) * h a b (x + y) := by sorry

end NUMINAMATH_CALUDE_functions_satisfy_equation_l4145_414528


namespace NUMINAMATH_CALUDE_unique_perpendicular_to_skew_lines_l4145_414555

/-- A line in three-dimensional space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- A line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicular lines
  sorry

theorem unique_perpendicular_to_skew_lines 
  (p : ℝ × ℝ × ℝ) (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! l : Line3D, l.point = p ∧ is_perpendicular l l1 ∧ is_perpendicular l l2 :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_to_skew_lines_l4145_414555


namespace NUMINAMATH_CALUDE_car_distribution_l4145_414574

theorem car_distribution (total_cars : ℕ) (first_supplier : ℕ) : 
  total_cars = 5650000 →
  first_supplier = 1000000 →
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let remaining_cars := total_cars - (first_supplier + second_supplier + third_supplier)
  remaining_cars / 2 = 325000 :=
by sorry

end NUMINAMATH_CALUDE_car_distribution_l4145_414574


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4145_414556

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, (1 < x ∧ x < 2) → x * (x - 3) < 0) ∧
  (∃ x, x * (x - 3) < 0 ∧ ¬(1 < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4145_414556


namespace NUMINAMATH_CALUDE_digit_interchange_l4145_414540

theorem digit_interchange (x : ℕ) : x = 9 ↔ 32 - x = 23 := by sorry

end NUMINAMATH_CALUDE_digit_interchange_l4145_414540


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4145_414520

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (9 + 3 * x) = 15 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4145_414520


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4145_414501

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2023)
  (eq2 : x + 2023 * Real.sin y = 2022)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2023 + π / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4145_414501


namespace NUMINAMATH_CALUDE_root_range_l4145_414588

theorem root_range (k : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 + (k-5)*x + 9 = 0) ↔ 
  (-5 < k ∧ k < -3/2) := by
sorry

end NUMINAMATH_CALUDE_root_range_l4145_414588


namespace NUMINAMATH_CALUDE_number_of_boys_l4145_414587

/-- Given a school with a total of 1396 people, 315 girls, and 772 teachers,
    prove that there are 309 boys in the school. -/
theorem number_of_boys (total : ℕ) (girls : ℕ) (teachers : ℕ) 
    (h1 : total = 1396) 
    (h2 : girls = 315) 
    (h3 : teachers = 772) : 
  total - girls - teachers = 309 := by
  sorry


end NUMINAMATH_CALUDE_number_of_boys_l4145_414587


namespace NUMINAMATH_CALUDE_distribute_five_gifts_to_three_fans_l4145_414573

/-- The number of ways to distribute n identical gifts to k different fans,
    where each fan receives at least one gift -/
def distribute_gifts (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 identical gifts to 3 different fans,
    where each fan receives at least one gift, can be done in 6 ways -/
theorem distribute_five_gifts_to_three_fans :
  distribute_gifts 5 3 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_five_gifts_to_three_fans_l4145_414573


namespace NUMINAMATH_CALUDE_paula_and_olive_spend_twenty_l4145_414547

/-- The total amount spent by Paula and Olive at the kiddy gift shop -/
def total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains : ℕ)
  (olive_coloring_books olive_bracelets : ℕ) : ℕ :=
  (paula_bracelets * bracelet_price + paula_keychains * keychain_price) +
  (olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price)

/-- Theorem stating that Paula and Olive spend $20 in total -/
theorem paula_and_olive_spend_twenty :
  total_spent 4 5 3 2 1 1 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_paula_and_olive_spend_twenty_l4145_414547


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l4145_414523

theorem inequality_system_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x > 4 ∧ 3 * x + a > 0) ↔ x > 2) → 
  a ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l4145_414523


namespace NUMINAMATH_CALUDE_impossible_perpendicular_l4145_414515

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)

-- Define the theorem
theorem impossible_perpendicular 
  (α : Plane) (a b : Line) (P : Point)
  (h1 : perpendicular a α)
  (h2 : intersect a b P) :
  ¬ (perpendicular b α) := by
  sorry

end NUMINAMATH_CALUDE_impossible_perpendicular_l4145_414515


namespace NUMINAMATH_CALUDE_final_cucumber_count_l4145_414565

theorem final_cucumber_count (initial_total : ℕ) (initial_carrots : ℕ) (added_cucumbers : ℕ)
  (h1 : initial_total = 10)
  (h2 : initial_carrots = 4)
  (h3 : added_cucumbers = 2) :
  initial_total - initial_carrots + added_cucumbers = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_cucumber_count_l4145_414565


namespace NUMINAMATH_CALUDE_quintons_fruit_trees_l4145_414512

/-- Represents the width of an apple tree in feet -/
def apple_width : ℕ := 10

/-- Represents the space needed between apple trees in feet -/
def apple_space : ℕ := 12

/-- Represents the width of a peach tree in feet -/
def peach_width : ℕ := 12

/-- Represents the space needed between peach trees in feet -/
def peach_space : ℕ := 15

/-- Represents the total space available for all trees in feet -/
def total_space : ℕ := 71

/-- Calculates the total number of fruit trees Quinton can plant -/
def total_fruit_trees : ℕ :=
  let apple_trees := 2
  let apple_total_space := apple_trees * apple_width + (apple_trees - 1) * apple_space
  let peach_space_left := total_space - apple_total_space
  let peach_trees := 1 + (peach_space_left - peach_width) / (peach_width + peach_space)
  apple_trees + peach_trees

theorem quintons_fruit_trees :
  total_fruit_trees = 4 := by
  sorry

end NUMINAMATH_CALUDE_quintons_fruit_trees_l4145_414512


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l4145_414514

-- Define an isosceles triangle
structure IsoscelesTriangle where
  baseAngle : ℝ
  vertexAngle : ℝ
  isIsosceles : baseAngle ≥ 0 ∧ vertexAngle ≥ 0
  angleSum : baseAngle + baseAngle + vertexAngle = 180

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.baseAngle = 80) : 
  triangle.vertexAngle = 20 :=
by
  sorry

#check isosceles_triangle_vertex_angle

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l4145_414514


namespace NUMINAMATH_CALUDE_fraction_simplification_l4145_414576

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^10 + 15*x^5 + 125) / (x^5 + 5) = 248 + 25/62 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4145_414576


namespace NUMINAMATH_CALUDE_only_f4_decreasing_l4145_414583

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem stating that only f4 has a negative derivative for all real x
theorem only_f4_decreasing :
  (∀ x : ℝ, deriv f1 x > 0) ∧
  (∃ x : ℝ, deriv f2 x ≥ 0) ∧
  (∀ x : ℝ, deriv f3 x > 0) ∧
  (∀ x : ℝ, deriv f4 x < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_f4_decreasing_l4145_414583


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l4145_414503

theorem mean_of_added_numbers (original_numbers : List ℝ) (x y z : ℝ) :
  original_numbers.length = 12 →
  original_numbers.sum / original_numbers.length = 72 →
  (original_numbers.sum + x + y + z) / (original_numbers.length + 3) = 80 →
  (x + y + z) / 3 = 112 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l4145_414503


namespace NUMINAMATH_CALUDE_cookies_eaten_l4145_414529

theorem cookies_eaten (original : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  original = 18 → remaining = 9 → eaten = original - remaining → eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l4145_414529


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l4145_414570

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l4145_414570


namespace NUMINAMATH_CALUDE_division_remainder_l4145_414535

theorem division_remainder : 
  let a := 555
  let b := 445
  let number := 220030
  let sum := a + b
  let diff := a - b
  let quotient := 2 * diff
  number % sum = 30 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l4145_414535


namespace NUMINAMATH_CALUDE_A_equals_nine_l4145_414595

/-- Represents the positions in the diagram --/
inductive Position
| A | B | C | D | E | F | G

/-- Represents the assignment of numbers to positions --/
def Assignment := Position → Fin 10

/-- Checks if all numbers from 1 to 10 are used exactly once --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∀ n : Fin 10, ∃! p : Position, a p = n

/-- Checks if the square condition is satisfied --/
def square_condition (a : Assignment) : Prop :=
  a Position.F = |a Position.A - a Position.B|

/-- Checks if the circle condition is satisfied --/
def circle_condition (a : Assignment) : Prop :=
  a Position.G = a Position.D + a Position.E

/-- Main theorem: A equals 9 --/
theorem A_equals_nine :
  ∃ (a : Assignment),
    is_valid_assignment a ∧
    square_condition a ∧
    circle_condition a ∧
    a Position.A = 9 := by
  sorry

end NUMINAMATH_CALUDE_A_equals_nine_l4145_414595


namespace NUMINAMATH_CALUDE_train_distance_problem_l4145_414598

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 50)
  (h2 : speed2 = 40)
  (h3 : distance_diff = 100) :
  let time := distance_diff / (speed1 - speed2)
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  let total_distance := distance1 + distance2
  total_distance = 900 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l4145_414598


namespace NUMINAMATH_CALUDE_constant_slope_on_parabola_l4145_414577

/-- Given a parabola and two fixed points, prove that the slope of the line formed by 
    the intersections of lines from a moving point on the parabola to the fixed points 
    is constant. -/
theorem constant_slope_on_parabola 
  (p : ℝ) (x₀ y₀ : ℝ) (h_p : p > 0) :
  let A : ℝ × ℝ := (x₀, y₀)
  let B : ℝ × ℝ := (y₀^2 / p - x₀, y₀)
  let parabola := {P : ℝ × ℝ | P.2^2 = 2 * p * P.1}
  ∀ P ∈ parabola, ∃ C D : ℝ × ℝ,
    (C ∈ parabola ∧ D ∈ parabola) ∧
    (∃ t : ℝ, P = (2 * p * t^2, 2 * p * t)) ∧
    (C ≠ P ∧ D ≠ P) ∧
    (C.2 - A.2) / (C.1 - A.1) = (P.2 - A.2) / (P.1 - A.1) ∧
    (D.2 - B.2) / (D.1 - B.1) = (P.2 - B.2) / (P.1 - B.1) ∧
    (D.2 - C.2) / (D.1 - C.1) = p / y₀ :=
by sorry

end NUMINAMATH_CALUDE_constant_slope_on_parabola_l4145_414577


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4145_414562

theorem polynomial_factorization (a : ℝ) : a^3 + 10*a^2 + 25*a = a*(a+5)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4145_414562


namespace NUMINAMATH_CALUDE_min_distance_line_to_log_curve_l4145_414569

/-- The minimum distance between a point on y = x and a point on y = ln x is √2/2 -/
theorem min_distance_line_to_log_curve : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = P.1) → 
    (Q.2 = Real.log Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_to_log_curve_l4145_414569


namespace NUMINAMATH_CALUDE_enchiladas_ordered_l4145_414517

/-- The number of enchiladas you ordered -/
def your_enchiladas : ℕ := 3

/-- The cost of each taco in dollars -/
def taco_cost : ℚ := 9/10

/-- Your bill in dollars (without tax) -/
def your_bill : ℚ := 78/10

/-- Your friend's bill in dollars (without tax) -/
def friend_bill : ℚ := 127/10

/-- The cost of each enchilada in dollars -/
def enchilada_cost : ℚ := 2

theorem enchiladas_ordered :
  (2 * taco_cost + your_enchiladas * enchilada_cost = your_bill) ∧
  (3 * taco_cost + 5 * enchilada_cost = friend_bill) :=
by sorry

end NUMINAMATH_CALUDE_enchiladas_ordered_l4145_414517


namespace NUMINAMATH_CALUDE_trig_ratio_equality_l4145_414572

theorem trig_ratio_equality (α : Real) (h : Real.tan α = 2 * Real.tan (π / 5)) :
  Real.cos (α - 3 * π / 10) / Real.sin (α - π / 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_equality_l4145_414572


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_g_minimum_value_f_inequality_l4145_414509

noncomputable section

variable (a : ℝ) (x : ℝ)

def f (x : ℝ) := a * x * Real.log x + (-a) * x

def g (x : ℝ) := x + 1 / Real.exp (x - 1)

theorem tangent_parallel_to_x_axis (h : a ≠ 0) :
  ∃ b : ℝ, (a * Real.log 1 + a + b = 0) → f a x = a * x * Real.log x + (-a) * x :=
sorry

theorem g_minimum_value :
  x > 0 → ∀ y > 0, g x ≥ 2 :=
sorry

theorem f_inequality (h : a ≠ 0) (hx : x > 0) :
  f a x / a + 2 / (x * Real.exp (x - 1) + 1) ≥ 1 - x :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_g_minimum_value_f_inequality_l4145_414509


namespace NUMINAMATH_CALUDE_vertex_C_coordinates_l4145_414518

-- Define the coordinate type
def Coordinate := ℝ × ℝ

-- Define the line equation type
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Coordinate
  B : Coordinate
  C : Coordinate

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  t.A = (5, 1) ∧
  ∃ (eq_CM : LineEquation), eq_CM = ⟨2, -1, -5⟩ ∧
  ∃ (eq_BH : LineEquation), eq_BH = ⟨1, -2, -5⟩

-- State the theorem
theorem vertex_C_coordinates (t : Triangle) :
  problem_conditions t → t.C = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_vertex_C_coordinates_l4145_414518


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l4145_414590

def is_valid_solution (A R K : Nat) : Prop :=
  A ≠ R ∧ A ≠ K ∧ R ≠ K ∧
  A < 10 ∧ R < 10 ∧ K < 10 ∧
  1000 * A + 100 * R + 10 * K + A +
  100 * R + 10 * K + A +
  10 * K + A +
  A = 2014

theorem cryptarithm_solution :
  ∀ A R K : Nat, is_valid_solution A R K → A = 1 ∧ R = 4 ∧ K = 7 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l4145_414590


namespace NUMINAMATH_CALUDE_range_of_a_l4145_414549

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - x - 2 ≥ 0 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ x^2 - x - 2 < 0) → 
  a ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4145_414549


namespace NUMINAMATH_CALUDE_negation_and_converse_l4145_414542

def last_digit (n : ℤ) : ℕ := (n % 10).natAbs

def divisible_by_five (n : ℤ) : Prop := n % 5 = 0

def statement (n : ℤ) : Prop :=
  (last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n

theorem negation_and_converse :
  (∀ n : ℤ, ¬statement n ↔ (last_digit n = 0 ∨ last_digit n = 5) ∧ ¬(divisible_by_five n)) ∧
  (∀ n : ℤ, (¬(last_digit n = 0 ∨ last_digit n = 5) → ¬(divisible_by_five n)) →
    ((last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n)) :=
sorry

end NUMINAMATH_CALUDE_negation_and_converse_l4145_414542


namespace NUMINAMATH_CALUDE_negative_inequality_transform_l4145_414559

theorem negative_inequality_transform {a b : ℝ} (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_transform_l4145_414559


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l4145_414586

theorem similar_triangles_leg_length 
  (leg1 : ℝ) 
  (hyp1 : ℝ) 
  (hyp2 : ℝ) 
  (h1 : leg1 = 15) 
  (h2 : hyp1 = 17) 
  (h3 : hyp2 = 51) : 
  (leg1 * hyp2 / hyp1) = 45 :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l4145_414586


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l4145_414508

theorem jason_pokemon_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 676 → cards_bought = 224 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 452 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l4145_414508


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4145_414525

theorem fraction_sum_equality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  1 / (x - 2) + 2 / (x + 2) + 4 / (4 - x^2) = 3 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4145_414525


namespace NUMINAMATH_CALUDE_one_fourth_of_7_2_l4145_414539

theorem one_fourth_of_7_2 : (7.2 : ℚ) / 4 = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_7_2_l4145_414539


namespace NUMINAMATH_CALUDE_john_finishes_ahead_l4145_414550

/-- The distance John finishes ahead of Steve in a race --/
def distance_ahead (john_speed steve_speed initial_distance push_time : ℝ) : ℝ :=
  (john_speed * push_time) - (steve_speed * push_time + initial_distance)

/-- Theorem stating that John finishes 2 meters ahead of Steve --/
theorem john_finishes_ahead :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.7
  let initial_distance : ℝ := 15
  let push_time : ℝ := 34
  distance_ahead john_speed steve_speed initial_distance push_time = 2 := by
sorry


end NUMINAMATH_CALUDE_john_finishes_ahead_l4145_414550


namespace NUMINAMATH_CALUDE_club_growth_l4145_414516

def club_members (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 4 * club_members k - 12

theorem club_growth : club_members 4 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_club_growth_l4145_414516


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4145_414589

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + Real.cos α) = -1) : 
  Real.tan α = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4145_414589


namespace NUMINAMATH_CALUDE_log3_one_over_81_l4145_414511

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_one_over_81 : log3 (1/81) = -4 := by
  sorry

end NUMINAMATH_CALUDE_log3_one_over_81_l4145_414511


namespace NUMINAMATH_CALUDE_bike_rental_problem_l4145_414591

/-- Calculates the number of hours a bike was rented given the total amount paid,
    the initial charge, and the hourly rate. -/
def rental_hours (total_paid : ℚ) (initial_charge : ℚ) (hourly_rate : ℚ) : ℚ :=
  (total_paid - initial_charge) / hourly_rate

/-- Proves that given the specific rental conditions and total payment,
    the number of hours rented is 9. -/
theorem bike_rental_problem :
  let total_paid : ℚ := 80
  let initial_charge : ℚ := 17
  let hourly_rate : ℚ := 7
  rental_hours total_paid initial_charge hourly_rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_bike_rental_problem_l4145_414591


namespace NUMINAMATH_CALUDE_right_triangle_cos_B_l4145_414541

theorem right_triangle_cos_B (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c^2 = a^2 + b^2) : 
  let cos_B := b / c
  cos_B = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_cos_B_l4145_414541


namespace NUMINAMATH_CALUDE_graph_regions_count_l4145_414551

/-- A line in the coordinate plane defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The set of lines defining the graph -/
def graph_lines : Set Line := {⟨3, 0⟩, ⟨1/3, 0⟩}

/-- The number of regions created by the graph lines -/
def num_regions : ℕ := 4

/-- Theorem stating that the number of regions created by the graph lines is 4 -/
theorem graph_regions_count :
  num_regions = 4 :=
sorry

end NUMINAMATH_CALUDE_graph_regions_count_l4145_414551


namespace NUMINAMATH_CALUDE_ellipse_dot_product_l4145_414553

/-- Given an ellipse C: x²/4 + y²/3 = 1, prove that the dot product of AB⃗ and AF⃗ is 6,
    where A is the left vertex, B is the upper vertex, and F is the right focus. -/
theorem ellipse_dot_product : 
  let C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 / 3 = 1}
  let A : ℝ × ℝ := (-2, 0)  -- left vertex
  let B : ℝ × ℝ := (0, Real.sqrt 3)  -- upper vertex
  let F : ℝ × ℝ := (1, 0)  -- right focus
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AF : ℝ × ℝ := (F.1 - A.1, F.2 - A.2)
  (AB.1 * AF.1 + AB.2 * AF.2 : ℝ) = 6 := by
sorry


end NUMINAMATH_CALUDE_ellipse_dot_product_l4145_414553


namespace NUMINAMATH_CALUDE_updated_mean_calculation_l4145_414568

theorem updated_mean_calculation (n : ℕ) (original_mean : ℝ) (decrement : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : decrement = 34) :
  (n : ℝ) * original_mean - n * decrement = n * 166 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_calculation_l4145_414568


namespace NUMINAMATH_CALUDE_dilation_matrix_determinant_l4145_414592

/-- Given a 2x2 matrix E representing a dilation centered at the origin with scale factor 5, 
    its determinant is 25. -/
theorem dilation_matrix_determinant : 
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ), 
  (∀ (i j : Fin 2), E i j = if i = j then 5 else 0) →
  Matrix.det E = 25 := by
sorry

end NUMINAMATH_CALUDE_dilation_matrix_determinant_l4145_414592


namespace NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l4145_414531

/-- Calculates the simple interest rate given the principal, interest, and time. -/
def calculate_interest_rate (principal interest : ℕ) (time : ℕ) : ℚ :=
  (interest * 100 : ℚ) / (principal * time)

/-- Proves that the interest rate is 12% given the specified conditions. -/
theorem interest_rate_is_twelve_percent 
  (principal : ℕ) 
  (interest : ℕ) 
  (time : ℕ) 
  (h1 : principal = 875)
  (h2 : interest = 2100)
  (h3 : time = 20) :
  calculate_interest_rate principal interest time = 12 := by
  sorry

#eval calculate_interest_rate 875 2100 20

end NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l4145_414531


namespace NUMINAMATH_CALUDE_inequality_chain_l4145_414530

theorem inequality_chain (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) :
  x^2 > a*x ∧ a*x > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l4145_414530


namespace NUMINAMATH_CALUDE_ed_has_27_pets_l4145_414564

/-- Represents the number of pets Ed has -/
structure Pets where
  dogs : ℕ
  cats : ℕ
  fish : ℕ
  birds : ℕ
  turtles : ℕ

/-- The conditions given in the problem -/
def petConditions (p : Pets) : Prop :=
  p.dogs = 2 ∧
  p.cats = 3 ∧
  p.fish = 3 * p.birds ∧
  p.fish = 2 * (p.dogs + p.cats) ∧
  p.turtles = p.birds / 2

/-- The total number of pets -/
def totalPets (p : Pets) : ℕ :=
  p.dogs + p.cats + p.fish + p.birds + p.turtles

/-- Theorem stating that given the conditions, Ed has 27 pets in total -/
theorem ed_has_27_pets :
  ∃ p : Pets, petConditions p ∧ totalPets p = 27 := by
  sorry


end NUMINAMATH_CALUDE_ed_has_27_pets_l4145_414564


namespace NUMINAMATH_CALUDE_product_increase_by_2022_l4145_414552

theorem product_increase_by_2022 : ∃ (a b c : ℕ),
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_by_2022_l4145_414552
