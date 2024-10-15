import Mathlib

namespace NUMINAMATH_CALUDE_dinner_bill_tip_percentage_l1890_189025

theorem dinner_bill_tip_percentage 
  (total_bill : ℝ)
  (num_friends : ℕ)
  (silas_payment : ℝ)
  (one_friend_payment : ℝ)
  (h1 : total_bill = 150)
  (h2 : num_friends = 6)
  (h3 : silas_payment = total_bill / 2)
  (h4 : one_friend_payment = 18)
  : (((one_friend_payment - (total_bill - silas_payment) / (num_friends - 1)) * (num_friends - 1)) / total_bill) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_bill_tip_percentage_l1890_189025


namespace NUMINAMATH_CALUDE_max_profit_children_clothing_l1890_189089

/-- Profit function for children's clothing sales -/
def profit (x : ℝ) : ℝ :=
  (x - 30) * (-2 * x + 200) - 450

/-- Theorem: Maximum profit for children's clothing sales -/
theorem max_profit_children_clothing :
  let x_min : ℝ := 30
  let x_max : ℝ := 60
  ∀ x ∈ Set.Icc x_min x_max,
    profit x ≤ profit x_max ∧
    profit x_max = 1950 := by
  sorry

#check max_profit_children_clothing

end NUMINAMATH_CALUDE_max_profit_children_clothing_l1890_189089


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l1890_189080

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, p < 20 → is_prime p → ¬(n % p = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, is_nonprime n ∧ 
           has_no_prime_factor_less_than_20 n ∧
           (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_prime_factor_less_than_20 m)) ∧
           n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l1890_189080


namespace NUMINAMATH_CALUDE_average_after_adding_constant_specific_average_problem_l1890_189023

theorem average_after_adding_constant (n : ℕ) (original_avg : ℚ) (added_const : ℚ) :
  n > 0 →
  let new_avg := original_avg + added_const
  new_avg = (n * original_avg + n * added_const) / n := by
  sorry

theorem specific_average_problem :
  let n : ℕ := 15
  let original_avg : ℚ := 40
  let added_const : ℚ := 10
  let new_avg := original_avg + added_const
  new_avg = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_after_adding_constant_specific_average_problem_l1890_189023


namespace NUMINAMATH_CALUDE_odd_function_sum_l1890_189078

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 3 = -2) :
  f (-3) + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1890_189078


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1890_189038

theorem simplify_and_evaluate_expression :
  let a : ℝ := 3 - Real.sqrt 2
  let expression := (((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6*a + 9)))
  expression = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1890_189038


namespace NUMINAMATH_CALUDE_late_car_speed_l1890_189032

/-- Proves that given a journey of 70 km, if a car arrives on time with an average speed
    of 40 km/hr and arrives 15 minutes late with a slower speed, then the slower speed
    is 35 km/hr. -/
theorem late_car_speed (distance : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  distance = 70 →
  on_time_speed = 40 →
  late_time = 0.25 →
  let on_time_duration := distance / on_time_speed
  let late_duration := on_time_duration + late_time
  let late_speed := distance / late_duration
  late_speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_late_car_speed_l1890_189032


namespace NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l1890_189074

theorem quadratic_roots_real_distinct :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + 6*x₁ + 8 = 0) ∧ (x₂^2 + 6*x₂ + 8 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l1890_189074


namespace NUMINAMATH_CALUDE_desert_area_changes_l1890_189010

/-- Represents the desert area problem -/
structure DesertArea where
  initial_area : ℝ  -- Initial desert area in 1997
  annual_increase : ℝ  -- Annual increase in desert area
  afforestation_rate : ℝ  -- Annual reduction due to afforestation measures

/-- Calculates the desert area after a given number of years without afforestation -/
def area_after_years (d : DesertArea) (years : ℕ) : ℝ :=
  d.initial_area + d.annual_increase * years

/-- Calculates the desert area after a given number of years with afforestation -/
def area_with_afforestation (d : DesertArea) (years : ℕ) : ℝ :=
  d.initial_area + d.annual_increase * years - d.afforestation_rate * years

/-- Main theorem about desert area changes -/
theorem desert_area_changes (d : DesertArea) 
    (h1 : d.initial_area = 9e5)
    (h2 : d.annual_increase = 2000)
    (h3 : d.afforestation_rate = 8000) :
    area_after_years d 23 = 9.46e5 ∧ 
    (∃ (y : ℕ), y ≤ 19 ∧ area_with_afforestation d y < 8e5 ∧ 
                ∀ (z : ℕ), z < y → area_with_afforestation d z ≥ 8e5) :=
  sorry


end NUMINAMATH_CALUDE_desert_area_changes_l1890_189010


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l1890_189009

-- Define the conversion factor from yards to feet
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def volume_cubic_yards : ℝ := 7

-- Theorem: 7 cubic yards are equal to 189 cubic feet
theorem cubic_yards_to_cubic_feet :
  (volume_cubic_yards * yards_to_feet ^ 3 : ℝ) = 189 :=
by sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l1890_189009


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1890_189084

theorem shaded_area_theorem (square_side : ℝ) (h : square_side = 12) :
  let triangle_base : ℝ := square_side * 3 / 4
  let triangle_height : ℝ := square_side / 4
  triangle_base * triangle_height / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1890_189084


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l1890_189013

theorem largest_number_with_conditions : ∃ n : ℕ, n = 93 ∧
  n < 100 ∧
  n % 8 = 5 ∧
  n % 3 = 0 ∧
  ∀ m : ℕ, m < 100 → m % 8 = 5 → m % 3 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l1890_189013


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l1890_189021

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l1890_189021


namespace NUMINAMATH_CALUDE_cube_piercing_theorem_l1890_189031

/-- Represents a brick with dimensions 2 × 2 × 1 -/
structure Brick :=
  (x : ℕ) (y : ℕ) (z : ℕ)

/-- Represents a cube constructed from bricks -/
structure Cube :=
  (size : ℕ)
  (bricks : List Brick)

/-- Represents a line perpendicular to a face of the cube -/
structure PerpLine :=
  (x : ℕ) (y : ℕ) (face : Nat)

/-- Function to check if a line intersects a brick -/
def intersects (l : PerpLine) (b : Brick) : Prop := sorry

/-- Theorem stating that there exists a line not intersecting any brick -/
theorem cube_piercing_theorem (c : Cube) 
  (h1 : c.size = 20) 
  (h2 : c.bricks.length = 2000) 
  (h3 : ∀ b ∈ c.bricks, b.x = 2 ∧ b.y = 2 ∧ b.z = 1) :
  ∃ l : PerpLine, ∀ b ∈ c.bricks, ¬(intersects l b) := by sorry

end NUMINAMATH_CALUDE_cube_piercing_theorem_l1890_189031


namespace NUMINAMATH_CALUDE_product_digit_sum_l1890_189054

theorem product_digit_sum (k : ℕ) : k = 222 ↔ 9 * k = 2000 ∧ ∃! (n : ℕ), 9 * n = 2000 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l1890_189054


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1890_189073

/-- Given vectors a and b in R², if (a - b) is perpendicular to b, then the x-coordinate of b is either -1 or 3. -/
theorem vector_perpendicular_condition (x : ℝ) :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → x = -1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1890_189073


namespace NUMINAMATH_CALUDE_lowest_digit_change_l1890_189063

/-- The correct sum of the addition -/
def correct_sum : ℕ := 1179

/-- The first addend in the incorrect addition -/
def addend1 : ℕ := 374

/-- The second addend in the incorrect addition -/
def addend2 : ℕ := 519

/-- The third addend in the incorrect addition -/
def addend3 : ℕ := 286

/-- The incorrect sum displayed in the problem -/
def incorrect_sum : ℕ := 1229

/-- Function to check if a digit change makes the addition correct -/
def is_correct_change (digit : ℕ) (position : ℕ) : Prop :=
  ∃ (new_addend : ℕ),
    (position = 1 ∧ new_addend + addend2 + addend3 = correct_sum) ∨
    (position = 2 ∧ addend1 + new_addend + addend3 = correct_sum) ∨
    (position = 3 ∧ addend1 + addend2 + new_addend = correct_sum)

/-- The lowest digit that can be changed to make the addition correct -/
def lowest_changeable_digit : ℕ := 4

theorem lowest_digit_change :
  (∀ d : ℕ, d < lowest_changeable_digit → ¬∃ p : ℕ, is_correct_change d p) ∧
  (∃ p : ℕ, is_correct_change lowest_changeable_digit p) :=
sorry

end NUMINAMATH_CALUDE_lowest_digit_change_l1890_189063


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1890_189036

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius : ℝ := r / 2
  let height : ℝ := r * Real.sqrt 3 / 2
  (1 / 3) * Real.pi * base_radius^2 * height = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1890_189036


namespace NUMINAMATH_CALUDE_sock_selection_combinations_l1890_189093

theorem sock_selection_combinations : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_combinations_l1890_189093


namespace NUMINAMATH_CALUDE_min_value_theorem_l1890_189050

noncomputable def f (x : ℝ) : ℝ := min (3^x - 1) (-x^2 + 2*x + 1)

theorem min_value_theorem (m a b : ℝ) :
  (∀ x, f x ≤ m) ∧  -- m is the maximum value of f
  (∃ x, f x = m) ∧  -- m is attained for some x
  (a > 0) ∧ (b > 0) ∧ (a + 2*b = m) →  -- conditions on a and b
  (∀ a' b', a' > 0 → b' > 0 → a' + 2*b' = m → 
    2 / (a' + 1) + 1 / b' ≥ 8/3) ∧  -- 8/3 is the minimum value
  (∃ a' b', a' > 0 ∧ b' > 0 ∧ a' + 2*b' = m ∧ 
    2 / (a' + 1) + 1 / b' = 8/3)  -- minimum is attained for some a' and b'
  := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1890_189050


namespace NUMINAMATH_CALUDE_cages_needed_cages_needed_is_five_l1890_189067

def initial_puppies : ℕ := 45
def sold_puppies : ℕ := 11
def puppies_per_cage : ℕ := 7

theorem cages_needed : ℕ :=
  let remaining_puppies := initial_puppies - sold_puppies
  (remaining_puppies + puppies_per_cage - 1) / puppies_per_cage

theorem cages_needed_is_five : cages_needed = 5 := by
  sorry

end NUMINAMATH_CALUDE_cages_needed_cages_needed_is_five_l1890_189067


namespace NUMINAMATH_CALUDE_line_problem_l1890_189097

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_problem (a b : ℝ) :
  let l0 : Line := ⟨1, -1, 1⟩
  let l1 : Line := ⟨a, -2, 1⟩
  let l2 : Line := ⟨1, b, 3⟩
  perpendicular l0 l1 → parallel l0 l2 → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l1890_189097


namespace NUMINAMATH_CALUDE_dvds_sold_per_day_is_497_l1890_189077

/-- Represents the DVD business model -/
structure DVDBusiness where
  initialCost : ℕ
  productionCost : ℕ
  sellingPriceFactor : ℚ
  daysPerWeek : ℕ
  totalWeeks : ℕ
  totalProfit : ℕ

/-- Calculates the number of DVDs sold per day -/
def calculateDVDsSoldPerDay (business : DVDBusiness) : ℕ :=
  let sellingPrice := business.productionCost * business.sellingPriceFactor
  let profitPerDVD := sellingPrice - business.productionCost
  let totalDays := business.daysPerWeek * business.totalWeeks
  let profitPerDay := business.totalProfit / totalDays
  (profitPerDay / profitPerDVD).floor.toNat

/-- Theorem stating that the number of DVDs sold per day is 497 -/
theorem dvds_sold_per_day_is_497 (business : DVDBusiness) 
  (h1 : business.initialCost = 2000)
  (h2 : business.productionCost = 6)
  (h3 : business.sellingPriceFactor = 2.5)
  (h4 : business.daysPerWeek = 5)
  (h5 : business.totalWeeks = 20)
  (h6 : business.totalProfit = 448000) :
  calculateDVDsSoldPerDay business = 497 := by
  sorry

#eval calculateDVDsSoldPerDay {
  initialCost := 2000,
  productionCost := 6,
  sellingPriceFactor := 2.5,
  daysPerWeek := 5,
  totalWeeks := 20,
  totalProfit := 448000
}

end NUMINAMATH_CALUDE_dvds_sold_per_day_is_497_l1890_189077


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1890_189008

theorem cryptarithm_solution :
  ∃! (A B : ℕ), 
    A < 10 ∧ B < 10 ∧ A ≠ B ∧
    9 * (10 * A + B) = 100 * A + 10 * A + B ∧
    A = 2 ∧ B = 5 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1890_189008


namespace NUMINAMATH_CALUDE_project_wage_difference_l1890_189056

theorem project_wage_difference (total_pay : ℝ) (p_hours q_hours : ℝ) 
  (hp : total_pay = 420)
  (hpq : q_hours = p_hours + 10)
  (hw : p_hours * (1.5 * (total_pay / q_hours)) = total_pay) :
  1.5 * (total_pay / q_hours) - (total_pay / q_hours) = 7 := by
  sorry

end NUMINAMATH_CALUDE_project_wage_difference_l1890_189056


namespace NUMINAMATH_CALUDE_tablet_price_after_discounts_l1890_189007

theorem tablet_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 250 ∧ discount1 = 0.30 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 131.25 := by
  sorry

end NUMINAMATH_CALUDE_tablet_price_after_discounts_l1890_189007


namespace NUMINAMATH_CALUDE_condo_units_l1890_189049

/-- Calculates the total number of units in a condo building -/
def total_units (total_floors : ℕ) (regular_units : ℕ) (penthouse_units : ℕ) (penthouse_floors : ℕ) : ℕ :=
  (total_floors - penthouse_floors) * regular_units + penthouse_floors * penthouse_units

/-- Theorem stating that a condo with the given specifications has 256 units -/
theorem condo_units : total_units 23 12 2 2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_condo_units_l1890_189049


namespace NUMINAMATH_CALUDE_circle_problem_l1890_189004

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 15^2}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10^2}
def P : ℝ × ℝ := (9, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the theorem
theorem circle_problem (k : ℝ) :
  P ∈ larger_circle ∧
  S k ∈ smaller_circle ∧
  (∀ p ∈ larger_circle, ∃ q ∈ smaller_circle, ‖p - q‖ = 5) →
  k = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_problem_l1890_189004


namespace NUMINAMATH_CALUDE_harriett_found_three_dollars_l1890_189071

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The number of quarters Harriett found -/
def quarters_found : ℕ := 10

/-- The number of dimes Harriett found -/
def dimes_found : ℕ := 3

/-- The number of nickels Harriett found -/
def nickels_found : ℕ := 3

/-- The number of pennies Harriett found -/
def pennies_found : ℕ := 5

/-- The total value of the coins Harriett found -/
def total_value : ℚ := 
  quarters_found * quarter_value + 
  dimes_found * dime_value + 
  nickels_found * nickel_value + 
  pennies_found * penny_value

theorem harriett_found_three_dollars : total_value = 3 := by
  sorry

end NUMINAMATH_CALUDE_harriett_found_three_dollars_l1890_189071


namespace NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l1890_189091

theorem remainder_31_pow_31_plus_31_mod_32 : (31^31 + 31) % 32 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l1890_189091


namespace NUMINAMATH_CALUDE_running_contest_average_distance_l1890_189044

/-- The average distance run by two people given their individual distances -/
def average_distance (d1 d2 : ℕ) : ℚ :=
  (d1 + d2) / 2

theorem running_contest_average_distance :
  let block_length : ℕ := 200
  let johnny_laps : ℕ := 4
  let mickey_laps : ℕ := johnny_laps / 2
  let johnny_distance : ℕ := johnny_laps * block_length
  let mickey_distance : ℕ := mickey_laps * block_length
  average_distance johnny_distance mickey_distance = 600 := by
sorry

end NUMINAMATH_CALUDE_running_contest_average_distance_l1890_189044


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_sqrt_5_12_l1890_189096

theorem rationalize_and_simplify_sqrt_5_12 : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_sqrt_5_12_l1890_189096


namespace NUMINAMATH_CALUDE_camp_total_boys_l1890_189026

structure Camp where
  totalBoys : ℕ
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ
  schoolAScience : ℕ
  schoolAMath : ℕ
  schoolBScience : ℕ
  schoolBEnglish : ℕ

def isValidCamp (c : Camp) : Prop :=
  c.schoolA + c.schoolB + c.schoolC = c.totalBoys ∧
  c.schoolA = c.totalBoys / 5 ∧
  c.schoolB = c.totalBoys / 4 ∧
  c.schoolC = c.totalBoys - c.schoolA - c.schoolB ∧
  c.schoolAScience = c.schoolA * 3 / 10 ∧
  c.schoolAMath = c.schoolA * 2 / 5 ∧
  c.schoolBScience = c.schoolB / 2 ∧
  c.schoolBEnglish = c.schoolB / 10 ∧
  c.schoolA - c.schoolAScience = 56 ∧
  c.schoolBEnglish = 35

theorem camp_total_boys (c : Camp) (h : isValidCamp c) : c.totalBoys = 400 := by
  sorry

end NUMINAMATH_CALUDE_camp_total_boys_l1890_189026


namespace NUMINAMATH_CALUDE_constant_dot_product_implies_ratio_l1890_189085

/-- Given that O is the origin, P is any point on the line 2x + y - 2 = 0,
    a = (m, n) is a non-zero vector, and the dot product of OP and a is always constant,
    then m/n = 2. -/
theorem constant_dot_product_implies_ratio (m n : ℝ) :
  (∀ x y : ℝ, 2 * x + y - 2 = 0 →
    ∃ k : ℝ, ∀ x' y' : ℝ, 2 * x' + y' - 2 = 0 →
      m * x' + n * y' = k) →
  m ≠ 0 ∨ n ≠ 0 →
  m / n = 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_dot_product_implies_ratio_l1890_189085


namespace NUMINAMATH_CALUDE_x_value_proof_l1890_189060

theorem x_value_proof (x : ℝ) (h : (1/2 : ℝ) - (1/3 : ℝ) = 3/x) : x = 18 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1890_189060


namespace NUMINAMATH_CALUDE_function_inequality_l1890_189064

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, HasDerivAt f (f' x) x ∧ f' x < 1/3)

-- State the theorem
theorem function_inequality (x : ℝ) :
  f x < x/3 + 2/3 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l1890_189064


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1890_189047

theorem perpendicular_lines (a : ℝ) : 
  (∃ (x y : ℝ), x + a * y - a = 0 ∧ a * x - (2 * a - 3) * y - 1 = 0) →
  ((-1 : ℝ) / a) * (a / (2 * a - 3)) = -1 →
  a = 0 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1890_189047


namespace NUMINAMATH_CALUDE_prob_C_is_one_fourth_l1890_189014

/-- A game spinner with four regions A, B, C, and D -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The probability of all regions in a spinner sum to 1 -/
def valid_spinner (s : Spinner) : Prop :=
  s.probA + s.probB + s.probC + s.probD = 1

/-- Theorem: Given a valid spinner with probA = 1/4, probB = 1/3, and probD = 1/6, 
    the probability of region C is 1/4 -/
theorem prob_C_is_one_fourth (s : Spinner) 
  (h_valid : valid_spinner s)
  (h_probA : s.probA = 1/4)
  (h_probB : s.probB = 1/3)
  (h_probD : s.probD = 1/6) :
  s.probC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_C_is_one_fourth_l1890_189014


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1890_189033

def team_size : ℕ := 12
def center_capable : ℕ := 4

def starting_lineup_combinations : ℕ :=
  center_capable * (team_size - 1) * (team_size - 2) * (team_size - 3)

theorem starting_lineup_count :
  starting_lineup_combinations = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l1890_189033


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l1890_189076

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure CarFuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Represents the distance a car can travel on a full tank in miles -/
structure CarRange where
  highway : ℝ
  city : ℝ

/-- The difference between highway and city fuel efficiency in miles per gallon -/
def efficiency_difference : ℝ := 12

theorem city_fuel_efficiency 
  (car_range : CarRange)
  (car_efficiency : CarFuelEfficiency)
  (h1 : car_range.highway = 800)
  (h2 : car_range.city = 500)
  (h3 : car_efficiency.city = car_efficiency.highway - efficiency_difference)
  (h4 : car_range.highway / car_efficiency.highway = car_range.city / car_efficiency.city) :
  car_efficiency.city = 20 := by
sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l1890_189076


namespace NUMINAMATH_CALUDE_value_of_x_l1890_189040

theorem value_of_x (x y z : ℝ) 
  (h1 : x = (1/2) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : z = 80) : 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l1890_189040


namespace NUMINAMATH_CALUDE_clock_problem_l1890_189083

/-- Represents a clock with hourly chimes -/
structure Clock :=
  (current_hour : Nat)
  (total_chimes : Nat)

/-- Function to calculate the number of chimes for a given hour -/
def chimes_for_hour (h : Nat) : Nat :=
  if h = 0 then 12 else h

/-- Function to check if the hour and minute hands overlap -/
def hands_overlap (h : Nat) : Prop :=
  h ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]

/-- Theorem representing the clock problem -/
theorem clock_problem (c : Clock) (h : Nat) : 
  c.total_chimes = 12 ∧ 
  hands_overlap h ∧ 
  c.current_hour = 3 ∧
  chimes_for_hour 3 + chimes_for_hour 4 + chimes_for_hour 5 = 12 →
  h - c.current_hour = 3 := by
  sorry

#check clock_problem

end NUMINAMATH_CALUDE_clock_problem_l1890_189083


namespace NUMINAMATH_CALUDE_shells_added_correct_l1890_189045

/-- The amount of shells added to a bucket -/
def shells_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the difference between the final and initial amounts
    equals the amount of shells added -/
theorem shells_added_correct (initial final : ℕ) (h : final ≥ initial) :
  shells_added initial final = final - initial :=
by
  sorry

end NUMINAMATH_CALUDE_shells_added_correct_l1890_189045


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l1890_189066

/-- The probability of selecting at least one woman when choosing 3 people at random from a group of 5 men and 5 women -/
theorem prob_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) : 
  total_people = men + women → 
  men = 5 → 
  women = 5 → 
  selected = 3 → 
  (1 : ℚ) - (men.choose selected : ℚ) / (total_people.choose selected : ℚ) = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l1890_189066


namespace NUMINAMATH_CALUDE_cafeteria_lasagnas_l1890_189053

/-- The number of lasagnas made by the school cafeteria -/
def num_lasagnas : ℕ := sorry

/-- The amount of ground mince used for each lasagna (in pounds) -/
def mince_per_lasagna : ℕ := 2

/-- The amount of ground mince used for each cottage pie (in pounds) -/
def mince_per_cottage_pie : ℕ := 3

/-- The total amount of ground mince used (in pounds) -/
def total_mince_used : ℕ := 500

/-- The number of cottage pies made -/
def num_cottage_pies : ℕ := 100

/-- Theorem stating that the number of lasagnas made is 100 -/
theorem cafeteria_lasagnas : num_lasagnas = 100 := by sorry

end NUMINAMATH_CALUDE_cafeteria_lasagnas_l1890_189053


namespace NUMINAMATH_CALUDE_marbles_on_desk_l1890_189028

theorem marbles_on_desk (desk_marbles : ℕ) : desk_marbles + 6 = 8 → desk_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_on_desk_l1890_189028


namespace NUMINAMATH_CALUDE_polygon_with_17_diagonals_has_8_sides_l1890_189037

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 17 diagonals has 8 sides -/
theorem polygon_with_17_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 17 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_17_diagonals_has_8_sides_l1890_189037


namespace NUMINAMATH_CALUDE_remainder_theorem_l1890_189042

-- Define the polynomial q(x)
def q (x : ℝ) (D : ℝ) : ℝ := 2 * x^6 - 3 * x^4 + D * x^2 + 6

-- State the theorem
theorem remainder_theorem (D : ℝ) :
  q 2 D = 14 → q (-2) D = 158 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1890_189042


namespace NUMINAMATH_CALUDE_perimeter_ABCDEFG_l1890_189003

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (m p1 p2 : Point2D) : Prop := sorry

/-- Calculate the perimeter of a polygon given by a list of points -/
def perimeter (points : List Point2D) : ℝ := sorry

/-- The main theorem -/
theorem perimeter_ABCDEFG :
  ∀ (A B C D E F G : Point2D),
    isEquilateral ⟨A, B, C⟩ →
    isEquilateral ⟨A, D, E⟩ →
    isEquilateral ⟨E, F, G⟩ →
    isMidpoint D A C →
    isMidpoint G A E →
    distance A B = 6 →
    perimeter [A, B, C, D, E, F, G] = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDEFG_l1890_189003


namespace NUMINAMATH_CALUDE_circus_kids_l1890_189006

theorem circus_kids (total_cost : ℕ) (kid_ticket_cost : ℕ) (num_adults : ℕ) : 
  total_cost = 50 →
  kid_ticket_cost = 5 →
  num_adults = 2 →
  ∃ (num_kids : ℕ), 
    (num_kids * kid_ticket_cost + num_adults * (2 * kid_ticket_cost) = total_cost) ∧
    num_kids = 2 := by
  sorry

end NUMINAMATH_CALUDE_circus_kids_l1890_189006


namespace NUMINAMATH_CALUDE_circle_m_range_l1890_189048

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 > c.radius^2

/-- The circle equation in the form x^2 + y^2 - 2x + 1 - m = 0 -/
def circleEquation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 1 - m = 0

theorem circle_m_range :
  ∀ m : ℝ,
  (∃ c : Circle, 
    (∀ x y : ℝ, circleEquation m x y ↔ (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2) ∧
    isOutside ⟨1, 1⟩ c) →
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l1890_189048


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1890_189065

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1890_189065


namespace NUMINAMATH_CALUDE_complex_calculation_result_l1890_189075

theorem complex_calculation_result : 
  ((0.60 * 50 * 0.45 * 30) - (0.40 * 35 / (0.25 * 20))) * ((3/5 * 100) + (2/7 * 49)) = 29762.8 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_result_l1890_189075


namespace NUMINAMATH_CALUDE_train_speed_problem_l1890_189058

/-- Given two trains traveling in opposite directions, this theorem proves
    the speed of the second train given the conditions of the problem. -/
theorem train_speed_problem (v : ℝ) : v = 50 := by
  -- Define the speed of the first train
  let speed1 : ℝ := 64
  -- Define the time of travel
  let time : ℝ := 2.5
  -- Define the total distance between trains after the given time
  let total_distance : ℝ := 285
  
  -- The equation representing the problem:
  -- speed1 * time + v * time = total_distance
  have h : speed1 * time + v * time = total_distance := by sorry
  
  -- Prove that v = 50 given the above equation
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1890_189058


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l1890_189055

theorem cube_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l1890_189055


namespace NUMINAMATH_CALUDE_largest_odd_five_digit_has_2_in_hundreds_place_l1890_189046

def Digits : Finset Nat := {1, 2, 3, 5, 8}

def is_odd (n : Nat) : Prop := n % 2 = 1

def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n < 100000

def uses_all_digits (n : Nat) (digits : Finset Nat) : Prop :=
  (Finset.card digits = 5) ∧
  (∀ d ∈ digits, ∃ i : Nat, (n / (10^i)) % 10 = d) ∧
  (∀ i : Nat, i < 5 → (n / (10^i)) % 10 ∈ digits)

def largest_odd_five_digit (n : Nat) : Prop :=
  is_odd n ∧
  is_five_digit n ∧
  uses_all_digits n Digits ∧
  ∀ m : Nat, is_odd m ∧ is_five_digit m ∧ uses_all_digits m Digits → m ≤ n

theorem largest_odd_five_digit_has_2_in_hundreds_place :
  ∃ n : Nat, largest_odd_five_digit n ∧ (n / 100) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_odd_five_digit_has_2_in_hundreds_place_l1890_189046


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l1890_189087

theorem complex_multiplication_sum (a b : ℝ) (i : ℂ) : 
  (1 + i) * (2 + i) = a + b * i → i * i = -1 → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l1890_189087


namespace NUMINAMATH_CALUDE_derivative_f_derivative_g_l1890_189041

noncomputable section

open Real

-- Function 1
def f (x : ℝ) : ℝ := (1 / Real.sqrt x) * cos x

-- Function 2
def g (x : ℝ) : ℝ := 5 * x^10 * sin x - 2 * Real.sqrt x * cos x - 9

-- Theorem for the derivative of function 1
theorem derivative_f (x : ℝ) (hx : x > 0) :
  deriv f x = -(cos x + 2 * x * sin x) / (2 * x * Real.sqrt x) :=
sorry

-- Theorem for the derivative of function 2
theorem derivative_g (x : ℝ) (hx : x > 0) :
  deriv g x = 50 * x^9 * sin x + 5 * x^10 * cos x - (Real.sqrt x * cos x) / x + 2 * Real.sqrt x * sin x :=
sorry

end NUMINAMATH_CALUDE_derivative_f_derivative_g_l1890_189041


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1890_189029

/-- Given vectors a and b in ℝ², if a + b = (5, -10) and a - b = (3, 6),
    then the cosine of the angle between a and b is 2√13/13. -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (5, -10)) 
  (h2 : a - b = (3, 6)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l1890_189029


namespace NUMINAMATH_CALUDE_sum_of_first_five_terms_l1890_189092

/-- Coordinate of point P_n on y-axis -/
def a (n : ℕ+) : ℚ := 2 / n

/-- Area of triangle formed by line through P_n and P_{n+1} and coordinate axes -/
def b (n : ℕ+) : ℚ := 4 + 1 / n - 1 / (n + 1)

/-- Sum of first n terms of sequence {b_n} -/
def S (n : ℕ+) : ℚ := 4 * n + n / (n + 1)

/-- Theorem: The sum of the first 5 terms of sequence {b_n} is 125/6 -/
theorem sum_of_first_five_terms : S 5 = 125 / 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_five_terms_l1890_189092


namespace NUMINAMATH_CALUDE_quadratic_ratio_l1890_189035

/-- Given a quadratic polynomial x^2 + 1560x + 2400, prove that when written in the form (x + b)^2 + c, the ratio c/b equals -300 -/
theorem quadratic_ratio (x : ℝ) : 
  ∃ (b c : ℝ), (∀ x, x^2 + 1560*x + 2400 = (x + b)^2 + c) ∧ c/b = -300 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l1890_189035


namespace NUMINAMATH_CALUDE_xiaoying_final_score_l1890_189030

/-- Calculates the weighted sum of scores given the scores and weights -/
def weightedSum (scores : List ℝ) (weights : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) scores weights)

/-- Xiaoying's speech competition scores -/
def speechScores : List ℝ := [86, 90, 80]

/-- Weights for each category in the speech competition -/
def categoryWeights : List ℝ := [0.5, 0.4, 0.1]

/-- Theorem stating that Xiaoying's final score is 87 -/
theorem xiaoying_final_score :
  weightedSum speechScores categoryWeights = 87 := by
  sorry

end NUMINAMATH_CALUDE_xiaoying_final_score_l1890_189030


namespace NUMINAMATH_CALUDE_train_length_l1890_189002

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * (1000 / 3600) →
  crossing_time = 18.598512119030477 →
  bridge_length = 200 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1890_189002


namespace NUMINAMATH_CALUDE_pen_distribution_l1890_189017

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem pen_distribution (num_pencils : ℕ) (num_pens : ℕ) 
  (h1 : num_pencils = 1203)
  (h2 : is_prime num_pencils)
  (h3 : ∀ (students : ℕ), students > 1 → ¬(num_pencils % students = 0 ∧ num_pens % students = 0)) :
  ∃ (n : ℕ), num_pens = n :=
sorry

end NUMINAMATH_CALUDE_pen_distribution_l1890_189017


namespace NUMINAMATH_CALUDE_lenny_remaining_money_l1890_189024

-- Define the initial amount and expenses
def initial_amount : ℝ := 270
def console_price : ℝ := 149
def console_discount : ℝ := 0.15
def grocery_price : ℝ := 60
def grocery_discount : ℝ := 0.10
def lunch_price : ℝ := 30
def magazine_price : ℝ := 3.99

-- Define the function to calculate the remaining money
def remaining_money : ℝ :=
  initial_amount -
  (console_price * (1 - console_discount)) -
  (grocery_price * (1 - grocery_discount)) -
  lunch_price -
  magazine_price

-- Theorem to prove
theorem lenny_remaining_money :
  remaining_money = 55.36 := by sorry

end NUMINAMATH_CALUDE_lenny_remaining_money_l1890_189024


namespace NUMINAMATH_CALUDE_triangle_inequality_l1890_189079

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D, E, F on the sides of the triangle
structure TriangleWithPoints extends Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

-- Define the condition DC + CE = EA + AF = FB + BD
def satisfiesCondition (t : TriangleWithPoints) : Prop :=
  let distAB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let distBC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let distCA := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let distDC := Real.sqrt ((t.D.1 - t.C.1)^2 + (t.D.2 - t.C.2)^2)
  let distCE := Real.sqrt ((t.C.1 - t.E.1)^2 + (t.C.2 - t.E.2)^2)
  let distEA := Real.sqrt ((t.E.1 - t.A.1)^2 + (t.E.2 - t.A.2)^2)
  let distAF := Real.sqrt ((t.A.1 - t.F.1)^2 + (t.A.2 - t.F.2)^2)
  let distFB := Real.sqrt ((t.F.1 - t.B.1)^2 + (t.F.2 - t.B.2)^2)
  let distBD := Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2)
  distDC + distCE = distEA + distAF ∧ distEA + distAF = distFB + distBD

-- State the theorem
theorem triangle_inequality (t : TriangleWithPoints) (h : satisfiesCondition t) :
  let distDE := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let distEF := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let distFD := Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)
  let distAB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let distBC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let distCA := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  distDE + distEF + distFD ≥ (1/2) * (distAB + distBC + distCA) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1890_189079


namespace NUMINAMATH_CALUDE_odd_function_value_at_half_l1890_189022

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_half
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x < 0, f x = 1 / (x + 1)) :
  f (1/2) = -2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_at_half_l1890_189022


namespace NUMINAMATH_CALUDE_term_2005_is_334th_l1890_189001

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℕ := 7 + 6 * (n - 1)

-- State the theorem
theorem term_2005_is_334th :
  arithmeticSequence 334 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_term_2005_is_334th_l1890_189001


namespace NUMINAMATH_CALUDE_perpendicular_line_l1890_189034

/-- Given a line L1 with equation 3x + 2y - 5 = 0 and a point P(1, -2),
    we define a line L2 with equation 2x - 3y - 8 = 0.
    This theorem states that L2 passes through P and is perpendicular to L1. -/
theorem perpendicular_line (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 2 * y - 5 = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2 * x - 3 * y - 8 = 0
  let P : ℝ × ℝ := (1, -2)
  (L2 P.1 P.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 → 
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) = 
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1)) * 
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1))) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_l1890_189034


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1890_189011

theorem triangle_angle_measure (A B C : ℝ) : 
  -- Triangle ABC
  A + B + C = 180 →
  -- Angle C is triple angle B
  C = 3 * B →
  -- Angle B is 15°
  B = 15 →
  -- Then angle A is 120°
  A = 120 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1890_189011


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_1_m_range_for_inequality_l1890_189027

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part 1
theorem solution_set_for_m_eq_1 :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
sorry

-- Part 2
theorem m_range_for_inequality (m : ℝ) :
  (0 < m ∧ m < 1/4) →
  (∀ x ∈ Set.Icc m (2*m), (1/2) * (f m x) ≤ |x + 1|) →
  m ∈ Set.Ioo 0 (1/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_1_m_range_for_inequality_l1890_189027


namespace NUMINAMATH_CALUDE_extrema_sum_implies_a_range_l1890_189090

/-- Given a function f(x) = ax - x^2 - ln x, if f(x) has extrema and the sum of these extrema
    is not less than 4 + ln 2, then a ∈ [2√3, +∞). -/
theorem extrema_sum_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x > 0, a * x - x^2 - Real.log x ≤ max (a * x₁ - x₁^2 - Real.log x₁) (a * x₂ - x₂^2 - Real.log x₂)) ∧
    (a * x₁ - x₁^2 - Real.log x₁) + (a * x₂ - x₂^2 - Real.log x₂) ≥ 4 + Real.log 2) →
  a ≥ 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_extrema_sum_implies_a_range_l1890_189090


namespace NUMINAMATH_CALUDE_chip_thickness_comparison_l1890_189020

theorem chip_thickness_comparison : 
  let a : ℝ := (1/3) * Real.sin (1/2)
  let b : ℝ := (1/2) * Real.sin (1/3)
  let c : ℝ := (1/3) * Real.cos (7/8)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_chip_thickness_comparison_l1890_189020


namespace NUMINAMATH_CALUDE_farmers_income_2010_l1890_189019

/-- Farmers' income in a given year -/
structure FarmerIncome where
  wage : ℝ
  other : ℝ

/-- Calculate farmers' income after n years -/
def futureIncome (initial : FarmerIncome) (n : ℕ) : ℝ :=
  initial.wage * (1 + 0.06) ^ n + (initial.other + n * 320)

theorem farmers_income_2010 :
  let initial : FarmerIncome := { wage := 3600, other := 2700 }
  let income2010 := futureIncome initial 5
  8800 ≤ income2010 ∧ income2010 < 9200 := by
  sorry

end NUMINAMATH_CALUDE_farmers_income_2010_l1890_189019


namespace NUMINAMATH_CALUDE_money_split_l1890_189016

theorem money_split (donna_share : ℚ) (donna_amount : ℕ) (total : ℕ) : 
  donna_share = 5 / 17 →
  donna_amount = 35 →
  donna_share * total = donna_amount →
  total = 119 := by
sorry

end NUMINAMATH_CALUDE_money_split_l1890_189016


namespace NUMINAMATH_CALUDE_robotics_club_non_participants_l1890_189051

theorem robotics_club_non_participants (total students_in_electronics students_in_programming students_in_both : ℕ) 
  (h1 : total = 80)
  (h2 : students_in_electronics = 45)
  (h3 : students_in_programming = 50)
  (h4 : students_in_both = 30) :
  total - (students_in_electronics + students_in_programming - students_in_both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_non_participants_l1890_189051


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1890_189099

theorem largest_five_digit_divisible_by_8 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 8 = 0 → n ≤ 99992 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1890_189099


namespace NUMINAMATH_CALUDE_hexagonal_gcd_bound_hexagonal_gcd_achieves_bound_l1890_189000

def H (n : ℕ+) : ℕ := 2 * n.val ^ 2 - n.val

theorem hexagonal_gcd_bound (n : ℕ+) : Nat.gcd (3 * H n) (n.val + 1) ≤ 12 :=
sorry

theorem hexagonal_gcd_achieves_bound : ∃ n : ℕ+, Nat.gcd (3 * H n) (n.val + 1) = 12 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_gcd_bound_hexagonal_gcd_achieves_bound_l1890_189000


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1890_189068

/-- Given three partners a, b, and c with their capital investments and profit shares,
    prove that the total profit is 16500. -/
theorem total_profit_calculation (a b c : ℕ) (profit_b : ℕ) :
  (2 * a = 3 * b) →  -- Twice a's capital equals thrice b's capital
  (b = 4 * c) →      -- b's capital is 4 times c's capital
  (profit_b = 6000) →  -- b's share of the profit is 6000
  (∃ (total_profit : ℕ), 
    total_profit = 16500 ∧
    total_profit * 4 = profit_b * 11) := by
  sorry

#check total_profit_calculation

end NUMINAMATH_CALUDE_total_profit_calculation_l1890_189068


namespace NUMINAMATH_CALUDE_total_harvest_earnings_l1890_189018

/-- Lewis's weekly earnings during the harvest -/
def weekly_earnings : ℕ := 2

/-- Duration of the harvest in weeks -/
def harvest_duration : ℕ := 89

/-- Theorem stating the total earnings for the harvest -/
theorem total_harvest_earnings :
  weekly_earnings * harvest_duration = 178 := by sorry

end NUMINAMATH_CALUDE_total_harvest_earnings_l1890_189018


namespace NUMINAMATH_CALUDE_watch_cost_price_l1890_189061

theorem watch_cost_price (CP : ℝ) : 
  (1.04 * CP - 0.90 * CP = 280) → CP = 2000 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1890_189061


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l1890_189098

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l1890_189098


namespace NUMINAMATH_CALUDE_dennis_floor_l1890_189052

/-- Given the floor arrangements of Frank, Charlie, and Dennis, prove that Dennis lives on the 6th floor. -/
theorem dennis_floor : 
  ∀ (frank_floor charlie_floor dennis_floor : ℕ),
  frank_floor = 16 →
  charlie_floor = frank_floor / 4 →
  dennis_floor = charlie_floor + 2 →
  dennis_floor = 6 := by
sorry

end NUMINAMATH_CALUDE_dennis_floor_l1890_189052


namespace NUMINAMATH_CALUDE_fib_150_mod_5_l1890_189043

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property we want to prove
theorem fib_150_mod_5 : fib 150 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_5_l1890_189043


namespace NUMINAMATH_CALUDE_alice_walk_distance_l1890_189005

theorem alice_walk_distance (grass_miles : ℝ) : 
  (∀ (day : Fin 5), grass_miles > 0) →  -- Alice walks a positive distance through grass each weekday
  (∀ (day : Fin 5), 12 > 0) →  -- Alice walks 12 miles through forest each weekday
  (5 * grass_miles + 5 * 12 = 110) →  -- Total weekly distance is 110 miles
  grass_miles = 10 := by
  sorry

end NUMINAMATH_CALUDE_alice_walk_distance_l1890_189005


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1890_189081

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3) ↔ 3 * (x : ℝ) - m ≤ 0) → 
  (9 ≤ m ∧ m < 12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1890_189081


namespace NUMINAMATH_CALUDE_oil_bottles_volume_l1890_189094

theorem oil_bottles_volume :
  let total_bottles : ℕ := 60
  let bottles_250ml : ℕ := 20
  let bottles_300ml : ℕ := 25
  let bottles_350ml : ℕ := total_bottles - bottles_250ml - bottles_300ml
  let volume_250ml : ℕ := 250
  let volume_300ml : ℕ := 300
  let volume_350ml : ℕ := 350
  let total_volume_ml : ℕ := bottles_250ml * volume_250ml + bottles_300ml * volume_300ml + bottles_350ml * volume_350ml
  let ml_per_liter : ℕ := 1000
  total_volume_ml / ml_per_liter = (17750 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_oil_bottles_volume_l1890_189094


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_complement_union_when_a_is_two_union_equals_B_iff_l1890_189015

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Theorem 1: When a = 2, A ∩ B = {x | 1 < x ≤ 4}
theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | 1 < x ∧ x ≤ 4} := by sorry

-- Theorem 2: When a = 2, (Uᶜ A) ∪ (Uᶜ B) = {x | x ≤ 1 or x > 4}
theorem complement_union_when_a_is_two :
  (Set.univ \ A 2) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 1 ∨ x > 4} := by sorry

-- Theorem 3: A ∪ B = B if and only if a ≤ -4 or -1 ≤ a ≤ 1/2
theorem union_equals_B_iff :
  ∀ a : ℝ, A a ∪ B = B ↔ a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_complement_union_when_a_is_two_union_equals_B_iff_l1890_189015


namespace NUMINAMATH_CALUDE_more_customers_left_than_stayed_l1890_189086

theorem more_customers_left_than_stayed (initial_customers remaining_customers : ℕ) :
  initial_customers = 25 →
  remaining_customers = 7 →
  (initial_customers - remaining_customers) - remaining_customers = 11 := by
  sorry

end NUMINAMATH_CALUDE_more_customers_left_than_stayed_l1890_189086


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1890_189070

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1890_189070


namespace NUMINAMATH_CALUDE_sin_plus_cos_shift_l1890_189072

theorem sin_plus_cos_shift (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_shift_l1890_189072


namespace NUMINAMATH_CALUDE_sqrt_product_eq_180_l1890_189059

theorem sqrt_product_eq_180 : Real.sqrt 75 * Real.sqrt 48 * (27 ^ (1/3 : ℝ)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_eq_180_l1890_189059


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1890_189069

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus with slope 45°
def line (x y : ℝ) : Prop := y = x - 2

-- Define the chord length
def chord_length : ℝ := 16

-- Theorem statement
theorem parabola_chord_length :
  ∀ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line x₁ y₁ ∧ line x₂ y₂ ∧
  A ≠ B →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1890_189069


namespace NUMINAMATH_CALUDE_study_seminar_selection_l1890_189095

theorem study_seminar_selection (n m k : ℕ) (h1 : n = 10) (h2 : m = 6) (h3 : k = 2) :
  (n.choose m) - ((n - k).choose (m - k)) = 140 := by
  sorry

end NUMINAMATH_CALUDE_study_seminar_selection_l1890_189095


namespace NUMINAMATH_CALUDE_cheese_distribution_l1890_189082

theorem cheese_distribution (M : ℝ) (x y : ℝ) : 
  -- Total cheese weight
  M > 0 →
  -- White's slice is exactly one-quarter of the total
  y = M / 4 →
  -- Thin's slice weighs x
  -- Fat's slice weighs x + 20
  -- White's slice weighs y
  -- Gray's slice weighs y + 8
  x + (x + 20) + y + (y + 8) = M →
  -- Gray cuts 8 grams, Fat cuts 20 grams
  -- To achieve equal distribution, Fat and Thin should each get 14 grams
  14 = (28 : ℝ) / 2 ∧
  x + 14 = y ∧
  (x + 20) - 20 + 14 = y ∧
  (y + 8) - 8 = y :=
by
  sorry

end NUMINAMATH_CALUDE_cheese_distribution_l1890_189082


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l1890_189088

def jogger_speed : ℝ := 9 -- km/hr
def train_speed : ℝ := 45 -- km/hr
def train_length : ℝ := 210 -- meters
def passing_time : ℝ := 41 -- seconds

theorem jogger_distance_ahead (jogger_speed train_speed train_length passing_time : ℝ) :
  jogger_speed = 9 ∧ 
  train_speed = 45 ∧ 
  train_length = 210 ∧ 
  passing_time = 41 →
  (train_speed - jogger_speed) * passing_time / 3600 * 1000 - train_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l1890_189088


namespace NUMINAMATH_CALUDE_min_pizzas_to_earn_back_car_cost_l1890_189012

/-- The cost of the car John bought -/
def car_cost : ℕ := 6500

/-- The amount John receives for each pizza delivered -/
def income_per_pizza : ℕ := 12

/-- The amount John spends on gas for each pizza delivered -/
def gas_cost_per_pizza : ℕ := 4

/-- The amount John spends on maintenance for each pizza delivered -/
def maintenance_cost_per_pizza : ℕ := 1

/-- The minimum whole number of pizzas John must deliver to earn back the car cost -/
def min_pizzas : ℕ := 929

theorem min_pizzas_to_earn_back_car_cost :
  ∀ n : ℕ, n ≥ min_pizzas →
    n * (income_per_pizza - gas_cost_per_pizza - maintenance_cost_per_pizza) ≥ car_cost ∧
    ∀ m : ℕ, m < min_pizzas →
      m * (income_per_pizza - gas_cost_per_pizza - maintenance_cost_per_pizza) < car_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_to_earn_back_car_cost_l1890_189012


namespace NUMINAMATH_CALUDE_banana_milk_distribution_l1890_189039

/-- The amount of banana milk Hyeonju drinks in milliliters -/
def hyeonju_amount : ℕ := 1000

/-- The amount of banana milk Jinsol drinks in milliliters -/
def jinsol_amount : ℕ := hyeonju_amount + 200

/-- The amount of banana milk Changhyeok drinks in milliliters -/
def changhyeok_amount : ℕ := hyeonju_amount - 200

/-- The total amount of banana milk in milliliters -/
def total_amount : ℕ := 3000

theorem banana_milk_distribution :
  hyeonju_amount + jinsol_amount + changhyeok_amount = total_amount ∧
  jinsol_amount = hyeonju_amount + 200 ∧
  hyeonju_amount = changhyeok_amount + 200 := by
  sorry

end NUMINAMATH_CALUDE_banana_milk_distribution_l1890_189039


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l1890_189062

theorem inequalities_satisfied (a b c x y z : ℝ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x * y * z ≤ a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l1890_189062


namespace NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l1890_189057

theorem alternating_sum_of_coefficients : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), 
  (∀ x : ℝ, (1 + 3*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -32 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l1890_189057
