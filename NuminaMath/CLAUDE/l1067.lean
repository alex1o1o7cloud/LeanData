import Mathlib

namespace NUMINAMATH_CALUDE_part_one_part_two_l1067_106729

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem part_one : (Set.univ \ P 3) ∩ Q = {x | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : {a : ℝ | P a ⊆ Q ∧ P a ≠ Q} = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1067_106729


namespace NUMINAMATH_CALUDE_max_value_of_f_l1067_106719

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f c) ∧
  f c = 1/4 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1067_106719


namespace NUMINAMATH_CALUDE_difference_set_not_always_equal_l1067_106718

theorem difference_set_not_always_equal :
  ∃ (A B : Set α) (hA : A.Nonempty) (hB : B.Nonempty),
    (A \ B) ≠ (B \ A) :=
by sorry

end NUMINAMATH_CALUDE_difference_set_not_always_equal_l1067_106718


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1067_106764

theorem polynomial_simplification (x : ℝ) : 
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1067_106764


namespace NUMINAMATH_CALUDE_area_of_triangle_fpg_l1067_106726

/-- Given a trapezoid EFGH with bases EF and GH, and point P at the intersection of diagonals,
    this theorem states that the area of triangle FPG is 28.125 square units. -/
theorem area_of_triangle_fpg (EF GH : ℝ) (area_EFGH : ℝ) :
  EF = 15 →
  GH = 25 →
  area_EFGH = 200 →
  ∃ (area_FPG : ℝ), area_FPG = 28.125 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_fpg_l1067_106726


namespace NUMINAMATH_CALUDE_exists_xAxis_visitsAllLines_l1067_106769

/-- Represents a line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Configuration of n lines in a plane -/
structure LineConfiguration where
  n : ℕ
  lines : Fin n → Line
  not_parallel : ∀ i j, i ≠ j → (lines i).slope ≠ (lines j).slope
  not_perpendicular : ∀ i j, i ≠ j → (lines i).slope * (lines j).slope ≠ -1
  not_concurrent : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬∃ (x y : ℝ), (y = (lines i).slope * x + (lines i).intercept) ∧
                   (y = (lines j).slope * x + (lines j).intercept) ∧
                   (y = (lines k).slope * x + (lines k).intercept)

/-- A point visits all lines if it intersects with each line -/
def visitsAllLines (cfg : LineConfiguration) (xAxis : Line) : Prop :=
  ∀ i, ∃ x, xAxis.slope * x + xAxis.intercept = (cfg.lines i).slope * x + (cfg.lines i).intercept

/-- Main theorem: There exists a line that can be chosen as x-axis to visit all lines -/
theorem exists_xAxis_visitsAllLines (cfg : LineConfiguration) :
  ∃ xAxis, visitsAllLines cfg xAxis := by
  sorry

end NUMINAMATH_CALUDE_exists_xAxis_visitsAllLines_l1067_106769


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1067_106796

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (9 * a) % 35 = 1 ∧ 
    (7 * b) % 35 = 1 ∧ 
    (7 * a + 3 * b) % 35 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1067_106796


namespace NUMINAMATH_CALUDE_chicken_egg_production_l1067_106752

theorem chicken_egg_production (num_chickens : ℕ) (total_eggs : ℕ) (num_days : ℕ) 
  (h1 : num_chickens = 4)
  (h2 : total_eggs = 36)
  (h3 : num_days = 3) :
  total_eggs / (num_chickens * num_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l1067_106752


namespace NUMINAMATH_CALUDE_callie_summer_frogs_count_l1067_106755

def alster_frogs : ℚ := 2

def quinn_frogs (alster_frogs : ℚ) : ℚ := 2 * alster_frogs

def bret_frogs (quinn_frogs : ℚ) : ℚ := 3 * quinn_frogs

def callie_summer_frogs (bret_frogs : ℚ) : ℚ := (5/8) * bret_frogs

theorem callie_summer_frogs_count :
  callie_summer_frogs (bret_frogs (quinn_frogs alster_frogs)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_callie_summer_frogs_count_l1067_106755


namespace NUMINAMATH_CALUDE_james_total_socks_l1067_106702

/-- Calculates the total number of socks James has -/
def total_socks (red_pairs : ℕ) : ℕ :=
  let red := red_pairs * 2
  let black := red / 2
  let white := (red + black) * 2
  red + black + white

/-- Proves that James has 180 socks in total -/
theorem james_total_socks : total_socks 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_james_total_socks_l1067_106702


namespace NUMINAMATH_CALUDE_blue_crayons_count_l1067_106750

theorem blue_crayons_count (blue : ℕ) (red : ℕ) : 
  red = 4 * blue →  -- Condition 1: Red crayons are four times blue crayons
  blue > 0 →        -- Condition 2: There is at least one blue crayon
  blue + red = 15 → -- Condition 3: Total number of crayons is 15
  blue = 3 :=        -- Conclusion: Number of blue crayons is 3
by
  sorry

end NUMINAMATH_CALUDE_blue_crayons_count_l1067_106750


namespace NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l1067_106732

/-- The number of toothpicks in a grid with a gap -/
def toothpicks_in_grid_with_gap (length width gap_length gap_width : ℕ) : ℕ :=
  let vertical_toothpicks := (length + 1) * width
  let horizontal_toothpicks := (width + 1) * length
  let gap_vertical_toothpicks := (gap_length + 1) * gap_width
  let gap_horizontal_toothpicks := (gap_width + 1) * gap_length
  vertical_toothpicks + horizontal_toothpicks - gap_vertical_toothpicks - gap_horizontal_toothpicks

/-- Theorem stating the number of toothpicks in the specific grid described in the problem -/
theorem toothpicks_in_specific_grid :
  toothpicks_in_grid_with_gap 70 40 10 5 = 5595 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l1067_106732


namespace NUMINAMATH_CALUDE_diamond_two_seven_l1067_106754

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 3 * y

-- Theorem statement
theorem diamond_two_seven : diamond 2 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_seven_l1067_106754


namespace NUMINAMATH_CALUDE_c_range_l1067_106738

-- Define the functions
def f (c : ℝ) (x : ℝ) := x^2 - 2*c*x + 1

-- State the theorem
theorem c_range (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) :
  (((∀ x y : ℝ, x < y → c^x > c^y) ∨
    (∀ x y : ℝ, x > y → x > 1/2 → y > 1/2 → f c x > f c y)) ∧
   ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧
     (∀ x y : ℝ, x > y → x > 1/2 → y > 1/2 → f c x > f c y))) →
  (1/2 < c ∧ c < 1) :=
by sorry

end NUMINAMATH_CALUDE_c_range_l1067_106738


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1067_106771

theorem y_intercept_of_line (x y : ℝ) :
  x + 2*y - 1 = 0 → x = 0 → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1067_106771


namespace NUMINAMATH_CALUDE_toms_initial_money_l1067_106725

theorem toms_initial_money (current_money : ℕ) (weekend_earnings : ℕ) (initial_money : ℕ) :
  current_money = 86 →
  weekend_earnings = 12 →
  current_money = initial_money + weekend_earnings →
  initial_money = 74 :=
by sorry

end NUMINAMATH_CALUDE_toms_initial_money_l1067_106725


namespace NUMINAMATH_CALUDE_video_votes_l1067_106713

theorem video_votes (total_votes : ℕ) (score : ℤ) (like_percent : ℚ) (dislike_percent : ℚ) : 
  score = 120 ∧ 
  like_percent = 58 / 100 ∧ 
  dislike_percent = 30 / 100 ∧ 
  (like_percent - dislike_percent) * total_votes = score →
  total_votes = 429 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l1067_106713


namespace NUMINAMATH_CALUDE_school_location_minimizes_distance_l1067_106774

/-- Represents a town with a number of students -/
structure Town where
  name : String
  students : ℕ

/-- Calculates the total distance traveled by students -/
def totalDistance (schoolLocation : Town) (townA : Town) (townB : Town) (distance : ℕ) : ℕ :=
  if schoolLocation.name = townA.name then
    townB.students * distance
  else if schoolLocation.name = townB.name then
    townA.students * distance
  else
    (townA.students + townB.students) * distance

/-- Theorem: Building a school in the town with more students minimizes total distance -/
theorem school_location_minimizes_distance (townA townB : Town) (distance : ℕ) :
  townA.students < townB.students →
  totalDistance townB townA townB distance ≤ totalDistance townA townA townB distance :=
by
  sorry

end NUMINAMATH_CALUDE_school_location_minimizes_distance_l1067_106774


namespace NUMINAMATH_CALUDE_six_digit_square_last_three_root_l1067_106792

-- Define a function to check if a number is a six-digit square number
def isSixDigitSquare (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ ∃ k : ℕ, n = k^2

-- Define a function to get the last three digits of a number
def lastThreeDigits (n : ℕ) : ℕ :=
  n % 1000

-- Define the main theorem
theorem six_digit_square_last_three_root : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, isSixDigitSquare n ∧ lastThreeDigits n = n.sqrt) ∧ 
    s.card = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_six_digit_square_last_three_root_l1067_106792


namespace NUMINAMATH_CALUDE_tobias_driveways_l1067_106740

/-- The number of driveways Tobias shoveled -/
def num_driveways : ℕ :=
  let shoe_cost : ℕ := 95
  let months_saved : ℕ := 3
  let monthly_allowance : ℕ := 5
  let lawn_mowing_fee : ℕ := 15
  let driveway_shoveling_fee : ℕ := 7
  let change_after_purchase : ℕ := 15
  let lawns_mowed : ℕ := 4
  let total_money : ℕ := shoe_cost + change_after_purchase
  let money_from_allowance : ℕ := months_saved * monthly_allowance
  let money_from_mowing : ℕ := lawns_mowed * lawn_mowing_fee
  let money_from_shoveling : ℕ := total_money - money_from_allowance - money_from_mowing
  money_from_shoveling / driveway_shoveling_fee

theorem tobias_driveways : num_driveways = 2 := by
  sorry

end NUMINAMATH_CALUDE_tobias_driveways_l1067_106740


namespace NUMINAMATH_CALUDE_woodys_weekly_allowance_l1067_106711

/-- Woody's weekly allowance problem -/
theorem woodys_weekly_allowance 
  (console_cost : ℕ) 
  (initial_savings : ℕ) 
  (weeks_to_save : ℕ) 
  (h1 : console_cost = 282)
  (h2 : initial_savings = 42)
  (h3 : weeks_to_save = 10) :
  (console_cost - initial_savings) / weeks_to_save = 24 := by
  sorry

end NUMINAMATH_CALUDE_woodys_weekly_allowance_l1067_106711


namespace NUMINAMATH_CALUDE_sum_of_segments_is_224_l1067_106770

/-- Given seven points A, B, C, D, E, F, G on a line in that order, 
    this function calculates the sum of lengths of all segments with endpoints at these points. -/
def sumOfSegments (AG BF CE : ℝ) : ℝ :=
  6 * AG + 4 * BF + 2 * CE

/-- Theorem stating that for the given conditions, the sum of all segment lengths is 224 cm. -/
theorem sum_of_segments_is_224 (AG BF CE : ℝ) 
  (h1 : AG = 23) (h2 : BF = 17) (h3 : CE = 9) : 
  sumOfSegments AG BF CE = 224 := by
  sorry

#eval sumOfSegments 23 17 9

end NUMINAMATH_CALUDE_sum_of_segments_is_224_l1067_106770


namespace NUMINAMATH_CALUDE_function_value_at_five_l1067_106715

open Real

theorem function_value_at_five
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > -3 / x)
  (h2 : ∀ x > 0, f (f x + 3 / x) = 2) :
  f 5 = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_five_l1067_106715


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_l1067_106786

/-- Calculates the total cost of a meal at Joe's Fast Food --/
def total_cost (sandwich_price : ℚ) (soda_price : ℚ) (fries_price : ℚ) 
                (sandwich_qty : ℕ) (soda_qty : ℕ) (fries_qty : ℕ) 
                (discount : ℚ) : ℚ :=
  sandwich_price * sandwich_qty + soda_price * soda_qty + fries_price * fries_qty - discount

/-- Theorem stating the total cost of the specified meal --/
theorem joes_fast_food_cost : 
  total_cost 4 (3/2) (5/2) 4 6 3 5 = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_l1067_106786


namespace NUMINAMATH_CALUDE_remainder_2468135790_div_101_l1067_106733

theorem remainder_2468135790_div_101 : 
  2468135790 % 101 = 50 := by sorry

end NUMINAMATH_CALUDE_remainder_2468135790_div_101_l1067_106733


namespace NUMINAMATH_CALUDE_cubes_equation_solution_l1067_106736

theorem cubes_equation_solution (x y z : ℤ) (h : x^3 + 2*y^3 = 4*z^3) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubes_equation_solution_l1067_106736


namespace NUMINAMATH_CALUDE_roxanne_change_l1067_106756

/-- Calculates the change Roxanne should receive after her purchase. -/
def calculate_change : ℚ :=
  let lemonade_cost : ℚ := 2 * 2
  let sandwich_cost : ℚ := 2 * 2.5
  let watermelon_cost : ℚ := 1.25
  let chips_cost : ℚ := 1.75
  let cookie_cost : ℚ := 3 * 0.75
  let total_cost : ℚ := lemonade_cost + sandwich_cost + watermelon_cost + chips_cost + cookie_cost
  let payment : ℚ := 50
  payment - total_cost

/-- Theorem stating that Roxanne's change is $35.75. -/
theorem roxanne_change : calculate_change = 35.75 := by
  sorry

end NUMINAMATH_CALUDE_roxanne_change_l1067_106756


namespace NUMINAMATH_CALUDE_central_cell_value_l1067_106714

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

theorem central_cell_value (t : Table) :
  satisfies_conditions t → t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l1067_106714


namespace NUMINAMATH_CALUDE_janet_sculpture_weight_l1067_106746

/-- Given Janet's work details, prove the weight of the first sculpture -/
theorem janet_sculpture_weight
  (exterminator_rate : ℝ)
  (sculpture_rate : ℝ)
  (exterminator_hours : ℝ)
  (second_sculpture_weight : ℝ)
  (total_income : ℝ)
  (h1 : exterminator_rate = 70)
  (h2 : sculpture_rate = 20)
  (h3 : exterminator_hours = 20)
  (h4 : second_sculpture_weight = 7)
  (h5 : total_income = 1640)
  : ∃ (first_sculpture_weight : ℝ),
    first_sculpture_weight = 5 ∧
    total_income = exterminator_rate * exterminator_hours +
                   sculpture_rate * (first_sculpture_weight + second_sculpture_weight) :=
by sorry

end NUMINAMATH_CALUDE_janet_sculpture_weight_l1067_106746


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1067_106782

theorem min_value_sum_reciprocals (n : ℕ+) (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) ≥ 1 ∧
  ((1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1067_106782


namespace NUMINAMATH_CALUDE_star_equation_solution_l1067_106744

/-- Custom binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem stating that if 4 ⋆ x = 52, then x = 8 -/
theorem star_equation_solution (x : ℝ) (h : star 4 x = 52) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1067_106744


namespace NUMINAMATH_CALUDE_unique_n_value_l1067_106727

theorem unique_n_value : ∃! n : ℕ, 
  50 < n ∧ n < 120 ∧ 
  ∃ k : ℕ, n = 8 * k ∧
  n % 7 = 3 ∧
  n % 9 = 3 ∧
  n = 192 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_value_l1067_106727


namespace NUMINAMATH_CALUDE_opposite_of_negative_abs_two_fifths_l1067_106778

theorem opposite_of_negative_abs_two_fifths :
  -(- |2 / 5|) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_abs_two_fifths_l1067_106778


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1067_106773

/-- Proves that the average salary of all workers in a workshop is 8000, given the specified conditions. -/
theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (h1 : total_workers = 49)
  (h2 : num_technicians = 7)
  (h3 : avg_salary_technicians = 20000)
  (h4 : avg_salary_rest = 6000) :
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l1067_106773


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1067_106790

theorem subtraction_of_fractions : (5 : ℚ) / 6 - (1 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1067_106790


namespace NUMINAMATH_CALUDE_gray_area_division_l1067_106735

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

-- Define a square within the rectangle
structure InternalSquare where
  side : ℝ
  x : ℝ  -- x-coordinate of the square's top-left corner
  y : ℝ  -- y-coordinate of the square's top-left corner
  side_pos : side > 0
  within_rectangle : (r : Rectangle) → x ≥ 0 ∧ y ≥ 0 ∧ x + side ≤ r.width ∧ y + side ≤ r.height

-- Define the theorem
theorem gray_area_division (r : Rectangle) (s : InternalSquare) :
  ∃ (line : ℝ → ℝ → Prop), 
    (∀ (x y : ℝ), (x ≥ 0 ∧ x ≤ r.width ∧ y ≥ 0 ∧ y ≤ r.height) →
      (¬(x ≥ s.x ∧ x ≤ s.x + s.side ∧ y ≥ s.y ∧ y ≤ s.y + s.side) →
        (line x y ∨ ¬line x y))) ∧
    (∃ (area1 area2 : ℝ), area1 = area2 ∧
      area1 + area2 = r.width * r.height - s.side * s.side) :=
by sorry

end NUMINAMATH_CALUDE_gray_area_division_l1067_106735


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1067_106757

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the expression we're interested in
def expression : ℕ := 24^3 + 42^3

-- Theorem statement
theorem units_digit_of_expression :
  unitsDigit expression = 6 := by
  sorry


end NUMINAMATH_CALUDE_units_digit_of_expression_l1067_106757


namespace NUMINAMATH_CALUDE_lucky_number_theorem_l1067_106751

/-- A "lucky number" is a three-digit positive integer that can be expressed as m(m+3) for some positive integer m. -/
def is_lucky_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (m + 3)

/-- The largest "lucky number". -/
def largest_lucky_number : ℕ := 990

/-- The sum of all N where M and N are both "lucky numbers" and M - N = 350. -/
def sum_of_satisfying_N : ℕ := 614

theorem lucky_number_theorem :
  (∀ n : ℕ, is_lucky_number n → n ≤ largest_lucky_number) ∧
  (∀ M N : ℕ, is_lucky_number M → is_lucky_number N → M - N = 350 →
    N = 460 ∨ N = 154) ∧
  (sum_of_satisfying_N = 614) := by sorry

end NUMINAMATH_CALUDE_lucky_number_theorem_l1067_106751


namespace NUMINAMATH_CALUDE_right_pyramid_height_l1067_106753

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  base_perimeter : ℝ
  /-- The distance from the apex to each vertex of the base in inches -/
  apex_to_vertex : ℝ

/-- The height of a right pyramid from its apex to the center of its base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

theorem right_pyramid_height (p : RightPyramid) 
  (h1 : p.base_perimeter = 40)
  (h2 : p.apex_to_vertex = 12) :
  pyramid_height p = Real.sqrt 94 := by
  sorry

end NUMINAMATH_CALUDE_right_pyramid_height_l1067_106753


namespace NUMINAMATH_CALUDE_average_salary_all_employees_l1067_106737

theorem average_salary_all_employees
  (officer_avg_salary : ℕ)
  (non_officer_avg_salary : ℕ)
  (num_officers : ℕ)
  (num_non_officers : ℕ)
  (h1 : officer_avg_salary = 450)
  (h2 : non_officer_avg_salary = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 495) :
  (officer_avg_salary * num_officers + non_officer_avg_salary * num_non_officers) / (num_officers + num_non_officers) = 120 :=
by sorry

end NUMINAMATH_CALUDE_average_salary_all_employees_l1067_106737


namespace NUMINAMATH_CALUDE_berry_multiple_l1067_106716

/-- Given the number of berries for Skylar, Steve, and Stacy, and their relationships,
    prove that the multiple of Steve's berries that Stacy has 2 more than is 3. -/
theorem berry_multiple (skylar_berries : ℕ) (steve_berries : ℕ) (stacy_berries : ℕ) 
    (h1 : skylar_berries = 20)
    (h2 : steve_berries = skylar_berries / 2)
    (h3 : stacy_berries = 32)
    (h4 : ∃ m : ℕ, stacy_berries = m * steve_berries + 2) :
  ∃ m : ℕ, m = 3 ∧ stacy_berries = m * steve_berries + 2 :=
by sorry

end NUMINAMATH_CALUDE_berry_multiple_l1067_106716


namespace NUMINAMATH_CALUDE_probability_two_fours_eight_dice_l1067_106781

theorem probability_two_fours_eight_dice : 
  let n : ℕ := 8  -- number of dice
  let k : ℕ := 2  -- number of successes (showing 4)
  let p : ℚ := 1 / 6  -- probability of rolling a 4 on a single die
  Nat.choose n k * p^k * (1 - p)^(n - k) = (28 * 15625 : ℚ) / 279936 := by
sorry

end NUMINAMATH_CALUDE_probability_two_fours_eight_dice_l1067_106781


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1067_106704

/-- Simple interest rate calculation --/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (amount : ℝ)
  (time : ℝ)
  (h_principal : principal = 8000)
  (h_amount : amount = 12500)
  (h_time : time = 7)
  (h_simple_interest : amount - principal = principal * (rate / 100) * time) :
  ∃ rate : ℝ, abs (rate - 8.04) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1067_106704


namespace NUMINAMATH_CALUDE_johns_total_expenses_l1067_106703

/-- The number of days in John's original tour program -/
def original_days : ℕ := 20

/-- The number of additional days if John extends his trip -/
def additional_days : ℕ := 4

/-- The amount by which John must reduce his daily expenses if he extends his trip -/
def expense_reduction : ℕ := 3

/-- John's total expenses remain the same whether he stays for the original duration or extends his trip -/
axiom expense_equality (daily_expense : ℕ) :
  original_days * daily_expense = (original_days + additional_days) * (daily_expense - expense_reduction)

theorem johns_total_expenses : ∃ (total_expense : ℕ), total_expense = 360 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_expenses_l1067_106703


namespace NUMINAMATH_CALUDE_two_solution_range_l1067_106795

/-- 
Given a system of equations:
  y = x^2
  y = x + m
The range of m for which the system has two distinct solutions is (-1/4, +∞).
-/
theorem two_solution_range (x y m : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m) ↔ m > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_two_solution_range_l1067_106795


namespace NUMINAMATH_CALUDE_root_existence_quadratic_root_existence_l1067_106730

theorem root_existence (f : ℝ → ℝ) (h1 : f 1.1 < 0) (h2 : f 1.2 > 0) 
  (h3 : Continuous f) :
  ∃ x : ℝ, x > 1.1 ∧ x < 1.2 ∧ f x = 0 :=
sorry

def f (x : ℝ) : ℝ := x^2 + 12*x - 15

theorem quadratic_root_existence (h1 : f 1.1 < 0) (h2 : f 1.2 > 0) :
  ∃ x : ℝ, x > 1.1 ∧ x < 1.2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_existence_quadratic_root_existence_l1067_106730


namespace NUMINAMATH_CALUDE_jim_bike_shop_profit_l1067_106749

/-- Represents Jim's bike shop financials for a month -/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repair_cost : ℕ
  complex_repairs_count : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ

/-- Calculates the total profit of the bike shop -/
def total_profit (shop : BikeShop) : ℤ :=
  (shop.tire_repair_price - shop.tire_repair_cost) * shop.tire_repairs_count +
  (shop.complex_repair_price - shop.complex_repair_cost) * shop.complex_repairs_count +
  shop.retail_profit - shop.fixed_expenses

/-- Theorem stating that Jim's bike shop profit is $3000 -/
theorem jim_bike_shop_profit :
  ∃ (shop : BikeShop),
    shop.tire_repair_price = 20 ∧
    shop.tire_repair_cost = 5 ∧
    shop.tire_repairs_count = 300 ∧
    shop.complex_repair_price = 300 ∧
    shop.complex_repair_cost = 50 ∧
    shop.complex_repairs_count = 2 ∧
    shop.retail_profit = 2000 ∧
    shop.fixed_expenses = 4000 ∧
    total_profit shop = 3000 := by
  sorry

end NUMINAMATH_CALUDE_jim_bike_shop_profit_l1067_106749


namespace NUMINAMATH_CALUDE_disease_cases_2005_2015_l1067_106712

/-- Calculates the number of disease cases in a given year, assuming a linear decrease. -/
def cases_in_year (initial_year initial_cases final_year final_cases target_year : ℕ) : ℕ :=
  initial_cases - (initial_cases - final_cases) * (target_year - initial_year) / (final_year - initial_year)

/-- Theorem stating the number of disease cases in 2005 and 2015 given the conditions. -/
theorem disease_cases_2005_2015 :
  cases_in_year 1970 300000 2020 100 2005 = 90070 ∧
  cases_in_year 1970 300000 2020 100 2015 = 30090 :=
by
  sorry

#eval cases_in_year 1970 300000 2020 100 2005
#eval cases_in_year 1970 300000 2020 100 2015

end NUMINAMATH_CALUDE_disease_cases_2005_2015_l1067_106712


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1067_106775

/-- The solution set of the quadratic inequality (m^2-2m-3)x^2-(m-3)x-1<0 is ℝ if and only if -1/5 < m ≤ 3 -/
theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ (-1/5 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1067_106775


namespace NUMINAMATH_CALUDE_segment_ratio_l1067_106743

/-- Given a line segment GH with points E and F on it, where GE is 3 times EH and GF is 5 times FH,
    prove that EF is 1/12 of GH. -/
theorem segment_ratio (G E F H : ℝ) (h1 : G ≤ E) (h2 : E ≤ F) (h3 : F ≤ H)
  (h4 : E - G = 3 * (H - E)) (h5 : F - G = 5 * (H - F)) :
  (F - E) / (H - G) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_segment_ratio_l1067_106743


namespace NUMINAMATH_CALUDE_final_strawberry_count_l1067_106768

/-- The number of strawberry plants after n months of doubling, starting from an initial number. -/
def plants_after_months (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

/-- The theorem stating the final number of strawberry plants -/
theorem final_strawberry_count :
  let initial_plants : ℕ := 3
  let months_passed : ℕ := 3
  let plants_given_away : ℕ := 4
  (plants_after_months initial_plants months_passed) - plants_given_away = 20 :=
by sorry

end NUMINAMATH_CALUDE_final_strawberry_count_l1067_106768


namespace NUMINAMATH_CALUDE_base_conversion_problem_l1067_106783

theorem base_conversion_problem : ∃! (b : ℕ), b > 1 ∧ b ^ 3 ≤ 216 ∧ 216 < b ^ 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l1067_106783


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1067_106721

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 3 * x + 2 > 0) ↔ (b < x ∧ x < 1)) → 
  (a = -5 ∧ b = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1067_106721


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1067_106710

theorem complex_power_magnitude : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1067_106710


namespace NUMINAMATH_CALUDE_festival_ferry_total_l1067_106785

/-- Represents the ferry schedule and passenger count --/
structure FerrySchedule where
  startTime : Nat  -- Start time in minutes after midnight
  endTime : Nat    -- End time in minutes after midnight
  interval : Nat   -- Interval between trips in minutes
  initialPassengers : Nat  -- Number of passengers on the first trip
  passengerDecrease : Nat  -- Decrease in passengers per trip

/-- Calculates the total number of people ferried --/
def totalPeopleFerried (schedule : FerrySchedule) : Nat :=
  let numTrips := (schedule.endTime - schedule.startTime) / schedule.interval + 1
  let lastTripPassengers := schedule.initialPassengers - (numTrips - 1) * schedule.passengerDecrease
  (numTrips * (schedule.initialPassengers + lastTripPassengers)) / 2

/-- The ferry schedule for the festival --/
def festivalFerry : FerrySchedule :=
  { startTime := 9 * 60  -- 9 AM in minutes
    endTime := 16 * 60   -- 4 PM in minutes
    interval := 30
    initialPassengers := 120
    passengerDecrease := 2 }

/-- Theorem stating the total number of people ferried to the festival --/
theorem festival_ferry_total : totalPeopleFerried festivalFerry = 1590 := by
  sorry


end NUMINAMATH_CALUDE_festival_ferry_total_l1067_106785


namespace NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1067_106720

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Define acute angle condition
def acute_angle (A B C : ℝ × ℝ) : Prop :=
  let BA := (A.1 - B.1, A.2 - B.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  BA.1 * BC.1 + BA.2 * BC.2 > 0

-- Theorem 1: Collinearity condition
theorem collinearity_condition :
  ∀ m : ℝ, collinear OA OB (OC m) ↔ m = 1/2 := sorry

-- Theorem 2: Acute angle condition
theorem acute_angle_condition :
  ∀ m : ℝ, acute_angle OA OB (OC m) ↔ m ∈ Set.Ioo (-3/4 : ℝ) (1/2) ∪ Set.Ioi (1/2) := sorry

end NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1067_106720


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l1067_106762

theorem quadratic_form_h_value :
  ∃ (a k : ℝ), ∀ x, 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l1067_106762


namespace NUMINAMATH_CALUDE_volleyball_tournament_max_wins_l1067_106758

theorem volleyball_tournament_max_wins (n : ℕ) : 
  let european_teams := n + 9
  let total_matches := n.choose 2 + european_teams.choose 2 + n * european_teams
  let european_wins := 9 * (total_matches - european_teams.choose 2)
  ∃ (k : ℕ), 
    k ≤ n * european_teams ∧ 
    european_teams.choose 2 + k = european_wins ∧
    (∀ m : ℕ, m ≤ n → m - 1 + min n european_teams ≤ 11) ∧
    (∃ m : ℕ, m ≤ n ∧ m - 1 + min n european_teams = 11) :=
by sorry

end NUMINAMATH_CALUDE_volleyball_tournament_max_wins_l1067_106758


namespace NUMINAMATH_CALUDE_not_prime_5n_plus_1_l1067_106793

theorem not_prime_5n_plus_1 (n : ℕ) (x y : ℕ) 
  (h1 : x^2 = 2*n + 1) (h2 : y^2 = 3*n + 1) : 
  ¬ Nat.Prime (5*n + 1) := by
sorry

end NUMINAMATH_CALUDE_not_prime_5n_plus_1_l1067_106793


namespace NUMINAMATH_CALUDE_value_of_expression_l1067_106745

theorem value_of_expression (x : ℝ) (h : x = 5) : (3*x + 4)^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1067_106745


namespace NUMINAMATH_CALUDE_students_in_class_l1067_106705

theorem students_in_class (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n + 2 = 4 * k) ∧
  (∃ l : ℕ, n + 3 = 5 * l) ∧
  (∃ m : ℕ, n + 4 = 6 * m) ↔ 
  (n = 122 ∨ n = 182) :=
by sorry

end NUMINAMATH_CALUDE_students_in_class_l1067_106705


namespace NUMINAMATH_CALUDE_triangles_from_circle_points_l1067_106780

def points_on_circle : ℕ := 10

theorem triangles_from_circle_points :
  Nat.choose points_on_circle 3 = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangles_from_circle_points_l1067_106780


namespace NUMINAMATH_CALUDE_k_range_proof_l1067_106741

theorem k_range_proof (k : ℝ) : 
  (∀ x, x > k → 3 / (x + 1) < 1) ∧ 
  (∃ x, 3 / (x + 1) < 1 ∧ x ≤ k) ↔ 
  k ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_k_range_proof_l1067_106741


namespace NUMINAMATH_CALUDE_card_position_unique_card_position_valid_l1067_106797

/-- Represents a position in a 6x6 grid -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the magician's trick setup -/
structure MagicTrick where
  initialColumn : Fin 6
  finalColumn : Fin 6

/-- Given the initial and final column numbers, determines the unique position of the card in the final layout -/
def findCardPosition (trick : MagicTrick) : Position :=
  { row := trick.initialColumn
  , col := trick.finalColumn }

/-- Theorem stating that the card position can be uniquely determined -/
theorem card_position_unique (trick : MagicTrick) :
  ∃! pos : Position, pos = findCardPosition trick :=
sorry

/-- Theorem stating that the determined position is valid within the 6x6 grid -/
theorem card_position_valid (trick : MagicTrick) :
  let pos := findCardPosition trick
  pos.row < 6 ∧ pos.col < 6 :=
sorry

end NUMINAMATH_CALUDE_card_position_unique_card_position_valid_l1067_106797


namespace NUMINAMATH_CALUDE_five_girls_five_boys_arrangements_l1067_106748

/-- The number of ways to arrange n girls and n boys around a circular table
    such that no two people of the same gender sit next to each other -/
def alternatingArrangements (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

/-- Theorem: There are 28800 ways to arrange 5 girls and 5 boys around a circular table
    such that no two people of the same gender sit next to each other -/
theorem five_girls_five_boys_arrangements :
  alternatingArrangements 5 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_five_girls_five_boys_arrangements_l1067_106748


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1067_106760

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 3 + a 5 + a 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1067_106760


namespace NUMINAMATH_CALUDE_division_by_less_than_one_multiplication_by_greater_than_one_multiply_by_100_equals_divide_by_001_l1067_106765

-- Statement 1
theorem division_by_less_than_one (x y : ℝ) (hx : x > 0) (hy : 0 < y) (hy1 : y < 1) :
  x / y > x :=
sorry

-- Statement 2
theorem multiplication_by_greater_than_one (x y : ℝ) (hy : y > 1) :
  x * y > x :=
sorry

-- Statement 3
theorem multiply_by_100_equals_divide_by_001 (x : ℝ) :
  x * 100 = x / 0.01 :=
sorry

end NUMINAMATH_CALUDE_division_by_less_than_one_multiplication_by_greater_than_one_multiply_by_100_equals_divide_by_001_l1067_106765


namespace NUMINAMATH_CALUDE_angle_A_value_triangle_area_l1067_106784

namespace TriangleABC

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition a * sin(B) = √3 * b * cos(A) -/
def condition1 (t : Triangle) : Prop :=
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A

/-- The conditions a = 3 and b = 2c -/
def condition2 (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

/-- The theorem stating that if condition1 holds, then A = π/3 -/
theorem angle_A_value (t : Triangle) (h : condition1 t) : t.A = Real.pi / 3 := by
  sorry

/-- The theorem stating that if condition1 and condition2 hold, then the area of the triangle is (3√3)/2 -/
theorem triangle_area (t : Triangle) (h1 : condition1 t) (h2 : condition2 t) : 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end TriangleABC

end NUMINAMATH_CALUDE_angle_A_value_triangle_area_l1067_106784


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1067_106724

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 10) : x^3 + 1/x^3 = 970 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1067_106724


namespace NUMINAMATH_CALUDE_sphere_radius_equal_cylinder_surface_l1067_106791

/-- The radius of a sphere with surface area equal to the curved surface area of a right circular cylinder with height and diameter both 12 cm. -/
theorem sphere_radius_equal_cylinder_surface (h : ℝ) (d : ℝ) (r : ℝ) :
  h = 12 →
  d = 12 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h →
  r = 6 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_equal_cylinder_surface_l1067_106791


namespace NUMINAMATH_CALUDE_function_range_theorem_l1067_106794

open Real

theorem function_range_theorem (f : ℝ → ℝ) (a b m : ℝ) :
  (∀ x, x > 0 → f x = 2 - 1/x) →
  a < b →
  (∀ x, x ∈ Set.Ioo a b ↔ f x ∈ Set.Ioo (m*a) (m*b)) →
  m ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_function_range_theorem_l1067_106794


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1067_106709

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x : ℝ, 6 * x^2 + b * x + 12 * x + 18 = 0) →
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) ∧ (b₁ + b₂ = -24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1067_106709


namespace NUMINAMATH_CALUDE_train_length_l1067_106766

/-- The length of a train given its speed and the time it takes to pass a bridge -/
theorem train_length (bridge_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  bridge_length = 180 →
  train_speed_kmh = 65 →
  passing_time = 21.04615384615385 →
  ∃ train_length : ℝ, abs (train_length - 200) < 0.00001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1067_106766


namespace NUMINAMATH_CALUDE_water_volume_in_solution_l1067_106722

/-- Calculates the volume of a component in a solution given the total volume and the component's proportion -/
def component_volume (total_volume : ℝ) (proportion : ℝ) : ℝ :=
  total_volume * proportion

theorem water_volume_in_solution (total_volume : ℝ) (water_proportion : ℝ) 
  (h1 : total_volume = 1.20)
  (h2 : water_proportion = 0.50) :
  component_volume total_volume water_proportion = 0.60 := by
  sorry

#eval component_volume 1.20 0.50

end NUMINAMATH_CALUDE_water_volume_in_solution_l1067_106722


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l1067_106779

theorem complex_modulus_equation (n : ℝ) : 
  Complex.abs (6 + n * Complex.I) = 6 * Real.sqrt 5 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l1067_106779


namespace NUMINAMATH_CALUDE_max_salary_theorem_l1067_106787

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  maxTotalSalary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def maxSinglePlayerSalary (team : BaseballTeam) : ℕ :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player in the given conditions -/
theorem max_salary_theorem (team : BaseballTeam) 
  (h1 : team.players = 18)
  (h2 : team.minSalary = 20000)
  (h3 : team.maxTotalSalary = 800000) :
  maxSinglePlayerSalary team = 460000 := by
  sorry

#eval maxSinglePlayerSalary ⟨18, 20000, 800000⟩

end NUMINAMATH_CALUDE_max_salary_theorem_l1067_106787


namespace NUMINAMATH_CALUDE_simplify_fraction_l1067_106734

theorem simplify_fraction (x y z : ℚ) (hx : x = 5) (hz : z = 2) :
  (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1067_106734


namespace NUMINAMATH_CALUDE_fruit_ratio_l1067_106707

theorem fruit_ratio (initial_fruits : ℕ) (oranges_left : ℕ) : 
  initial_fruits = 150 → 
  oranges_left = 50 → 
  (oranges_left : ℚ) / (initial_fruits / 2 - oranges_left : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_ratio_l1067_106707


namespace NUMINAMATH_CALUDE_function_with_period_two_is_even_l1067_106717

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_with_period_two_is_even
  (f : ℝ → ℝ)
  (h_period : smallest_positive_period f 2)
  (h_symmetry : ∀ x, f (x + 2) = f (2 - x)) :
  is_even f :=
sorry

end NUMINAMATH_CALUDE_function_with_period_two_is_even_l1067_106717


namespace NUMINAMATH_CALUDE_cow_herd_distribution_l1067_106798

theorem cow_herd_distribution (total : ℕ) : 
  (total : ℚ) / 3 + (total : ℚ) / 6 + (total : ℚ) / 8 + 9 = total → total = 216 := by
  sorry

end NUMINAMATH_CALUDE_cow_herd_distribution_l1067_106798


namespace NUMINAMATH_CALUDE_prob_product_div_by_eight_l1067_106728

/-- The probability of rolling an odd number on a standard 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The probability of rolling a 2 on a standard 6-sided die -/
def prob_two : ℚ := 1/6

/-- The probability of rolling a 4 on a standard 6-sided die -/
def prob_four : ℚ := 1/6

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- Theorem: The probability that the product of 8 standard 6-sided dice rolls is divisible by 8 is 1651/1728 -/
theorem prob_product_div_by_eight : 
  (1 : ℚ) - (prob_odd ^ num_dice + 
    num_dice * prob_two * prob_odd ^ (num_dice - 1) + 
    (num_dice.choose 2) * prob_two^2 * prob_odd^(num_dice - 2) + 
    num_dice * prob_four * prob_odd^(num_dice - 1)) = 1651/1728 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_div_by_eight_l1067_106728


namespace NUMINAMATH_CALUDE_total_questions_to_review_l1067_106763

-- Define the given conditions
def num_classes : ℕ := 5
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10

-- State the theorem
theorem total_questions_to_review :
  num_classes * students_per_class * questions_per_exam = 1750 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_to_review_l1067_106763


namespace NUMINAMATH_CALUDE_solutions_count_l1067_106767

/-- The number of different integer solutions (x, y) for |x|+|y|=n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_count :
  (num_solutions 1 = 4) ∧
  (num_solutions 2 = 8) ∧
  (num_solutions 3 = 12) →
  ∀ n : ℕ, num_solutions n = 4 * n :=
by sorry

end NUMINAMATH_CALUDE_solutions_count_l1067_106767


namespace NUMINAMATH_CALUDE_min_value_theorem_l1067_106772

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3*x*y) :
  2*x + y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 3*x₀*y₀ ∧ 2*x₀ + y₀ = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1067_106772


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l1067_106788

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The pan of brownies -/
def pan : Rectangle := { length := 24, width := 20 }

/-- A single brownie piece -/
def piece : Rectangle := { length := 3, width := 4 }

/-- The number of brownie pieces that can be cut from the pan -/
def num_pieces : ℕ := (area pan) / (area piece)

theorem brownie_pieces_count : num_pieces = 40 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l1067_106788


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1067_106761

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x*y ≤ a*b) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x + y ≤ a + b) ∧
  x*y = 64 ∧ x + y = 18 := by
sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1067_106761


namespace NUMINAMATH_CALUDE_two_lucky_tickets_exist_l1067_106747

/-- A ticket number is a 6-digit integer -/
def TicketNumber := { n : ℕ // n ≥ 100000 ∧ n < 1000000 }

/-- Sum of the first three digits of a ticket number -/
def sumFirstThree (n : TicketNumber) : ℕ := 
  (n.val / 100000) + ((n.val / 10000) % 10) + ((n.val / 1000) % 10)

/-- Sum of the last three digits of a ticket number -/
def sumLastThree (n : TicketNumber) : ℕ := 
  ((n.val / 100) % 10) + ((n.val / 10) % 10) + (n.val % 10)

/-- A ticket is lucky if the sum of its first three digits equals the sum of its last three digits -/
def isLucky (n : TicketNumber) : Prop := sumFirstThree n = sumLastThree n

/-- There exist two lucky tickets among ten consecutive tickets -/
theorem two_lucky_tickets_exist : 
  ∃ (n : TicketNumber) (a b : ℕ), 0 ≤ a ∧ a < b ∧ b ≤ 9 ∧ 
    isLucky ⟨n.val + a, sorry⟩ ∧ isLucky ⟨n.val + b, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_two_lucky_tickets_exist_l1067_106747


namespace NUMINAMATH_CALUDE_intersection_of_given_lines_l1067_106723

/-- The intersection point of two lines in 2D space -/
def intersection_point (line1_start : ℝ × ℝ) (line1_dir : ℝ × ℝ) (line2_start : ℝ × ℝ) (line2_dir : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem intersection_of_given_lines :
  let line1_start : ℝ × ℝ := (2, -3)
  let line1_dir : ℝ × ℝ := (3, 4)
  let line2_start : ℝ × ℝ := (-1, 4)
  let line2_dir : ℝ × ℝ := (5, -1)
  intersection_point line1_start line1_dir line2_start line2_dir = (124/5, 137/5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_given_lines_l1067_106723


namespace NUMINAMATH_CALUDE_square_divisible_into_2020_elegant_triangles_l1067_106789

/-- An elegant triangle is a right-angled triangle where one leg is 10 times longer than the other. -/
def ElegantTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (a = 10*b ∨ b = 10*a)

/-- A square can be divided into n identical elegant triangles. -/
def SquareDivisibleIntoElegantTriangles (n : ℕ) : Prop :=
  ∃ (s a b c : ℝ), s > 0 ∧ ElegantTriangle a b c ∧ 
    (n : ℝ) * (1/2 * a * b) = s^2

theorem square_divisible_into_2020_elegant_triangles :
  SquareDivisibleIntoElegantTriangles 2020 := by
  sorry


end NUMINAMATH_CALUDE_square_divisible_into_2020_elegant_triangles_l1067_106789


namespace NUMINAMATH_CALUDE_club_average_age_l1067_106742

theorem club_average_age (num_females num_males num_children : ℕ)
                         (avg_age_females avg_age_males avg_age_children : ℚ) :
  num_females = 12 →
  num_males = 20 →
  num_children = 8 →
  avg_age_females = 28 →
  avg_age_males = 40 →
  avg_age_children = 10 →
  let total_sum := num_females * avg_age_females + num_males * avg_age_males + num_children * avg_age_children
  let total_people := num_females + num_males + num_children
  (total_sum / total_people : ℚ) = 30.4 := by
  sorry

end NUMINAMATH_CALUDE_club_average_age_l1067_106742


namespace NUMINAMATH_CALUDE_tracis_road_trip_l1067_106706

/-- Traci's road trip problem -/
theorem tracis_road_trip (total_distance : ℝ) (remaining_distance : ℝ) (x : ℝ) : 
  total_distance = 600 →
  remaining_distance = 300 →
  remaining_distance = total_distance - x * total_distance - (1/4) * (total_distance - x * total_distance) →
  x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tracis_road_trip_l1067_106706


namespace NUMINAMATH_CALUDE_obtuse_angle_is_in_second_quadrant_l1067_106759

/-- Definition of an obtuse angle -/
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

/-- Definition of an angle in the second quadrant -/
def is_in_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

/-- Theorem: An obtuse angle is an angle in the second quadrant -/
theorem obtuse_angle_is_in_second_quadrant (θ : ℝ) :
  is_obtuse_angle θ ↔ is_in_second_quadrant θ :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_is_in_second_quadrant_l1067_106759


namespace NUMINAMATH_CALUDE_composition_result_l1067_106739

/-- Given two functions f and g, prove that f(g(f(3))) = 119 -/
theorem composition_result :
  let f (x : ℝ) := 2 * x + 5
  let g (x : ℝ) := 5 * x + 2
  f (g (f 3)) = 119 := by sorry

end NUMINAMATH_CALUDE_composition_result_l1067_106739


namespace NUMINAMATH_CALUDE_triangle_five_sixths_nine_fourths_l1067_106701

/-- The triangle operation ∆ defined for fractions -/
def triangle (m n p q : ℚ) : ℚ := m^2 * p * (q / n)

/-- Theorem stating that (5/6) ∆ (9/4) = 150 -/
theorem triangle_five_sixths_nine_fourths : 
  triangle (5/6) (5/6) (9/4) (9/4) = 150 := by sorry

end NUMINAMATH_CALUDE_triangle_five_sixths_nine_fourths_l1067_106701


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l1067_106776

theorem sqrt_sum_equals_twelve :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l1067_106776


namespace NUMINAMATH_CALUDE_remainder_3m_mod_5_l1067_106708

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3m_mod_5_l1067_106708


namespace NUMINAMATH_CALUDE_max_player_salary_l1067_106700

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 25 →
  min_salary = 20000 →
  max_total = 900000 →
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary = 420000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_l1067_106700


namespace NUMINAMATH_CALUDE_multiple_count_l1067_106777

theorem multiple_count (n : ℕ) (h1 : n > 0) (h2 : n ≤ 400) : 
  (∃ (k : ℕ), k > 0 ∧ (∀ m : ℕ, m > 0 → m ≤ 400 → m % k = 0 → m ∈ Finset.range 401) ∧ 
  (Finset.filter (λ m => m % k = 0) (Finset.range 401)).card = 16) → 
  n = 25 :=
sorry

end NUMINAMATH_CALUDE_multiple_count_l1067_106777


namespace NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l1067_106731

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  ∀ n : ℕ, 
  n > 2 → 
  (360 : ℝ) / n = 18 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l1067_106731


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l1067_106799

-- Define the foci
def F₁ : ℝ × ℝ := (0, 2)
def F₂ : ℝ × ℝ := (8, 2)

-- Define the ellipse
def Ellipse : Set (ℝ × ℝ) :=
  {P | Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 12}

-- Define the ellipse equation parameters
noncomputable def h : ℝ := (F₁.1 + F₂.1) / 2
noncomputable def k : ℝ := (F₁.2 + F₂.2) / 2
noncomputable def a : ℝ := 6
noncomputable def b : ℝ := Real.sqrt (a^2 - ((F₂.1 - F₁.1) / 2)^2)

-- Theorem statement
theorem ellipse_parameter_sum :
  h + k + a + b = 12 + 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l1067_106799
