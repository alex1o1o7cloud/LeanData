import Mathlib

namespace decimal_multiplication_addition_l1823_182306

theorem decimal_multiplication_addition : (0.3 * 0.7) + (0.5 * 0.4) = 0.41 := by
  sorry

end decimal_multiplication_addition_l1823_182306


namespace quadrilateral_perimeter_l1823_182301

-- Define the quadrilateral ABCD and points E and F
variable (A B C D E F : Point)

-- Define the properties of the quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : Point) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- Define the intersection of two rays
def ray_intersection (P Q R S : Point) : Point := sorry

-- Define the perimeter of a triangle
def triangle_perimeter (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_perimeter 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_angle : angle_measure B A D = π / 3)
  (h_side1 : distance B C = 1)
  (h_side2 : distance A D = 1)
  (h_intersect1 : E = ray_intersection A B C D)
  (h_intersect2 : F = ray_intersection B C A D)
  (h_perimeter1 : ∃ n : ℕ, triangle_perimeter B C E = n)
  (h_perimeter2 : ∃ m : ℕ, triangle_perimeter C D F = m) :
  distance A B + distance B C + distance C D + distance D A = 38 / 7 := by
  sorry

end quadrilateral_perimeter_l1823_182301


namespace building_height_l1823_182397

/-- Prove that given a flagpole of height 18 meters casting a shadow of 45 meters,
    and a building casting a shadow of 70 meters under similar conditions,
    the height of the building is 28 meters. -/
theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_shadow : ℝ)
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_shadow = 70)
  : (flagpole_height / flagpole_shadow) * building_shadow = 28 :=
by sorry

end building_height_l1823_182397


namespace cross_number_puzzle_l1823_182380

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem cross_number_puzzle :
  ∃ (m n : ℕ),
    is_three_digit (3^m) ∧
    is_three_digit (7^n) ∧
    (3^m / 10) % 10 = (7^n / 10) % 10 ∧
    (3^m / 10) % 10 = 4 :=
by sorry

end cross_number_puzzle_l1823_182380


namespace thabo_hardcover_count_l1823_182304

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabos_books : BookCollection where
  hardcover_nonfiction := 25
  paperback_nonfiction := 45
  paperback_fiction := 90

theorem thabo_hardcover_count :
  ∀ (books : BookCollection),
    books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 160 →
    books.paperback_nonfiction = books.hardcover_nonfiction + 20 →
    books.paperback_fiction = 2 * books.paperback_nonfiction →
    books.hardcover_nonfiction = 25 := by
  sorry

#eval thabos_books.hardcover_nonfiction

end thabo_hardcover_count_l1823_182304


namespace collins_savings_l1823_182376

def cans_per_dollar : ℚ := 4

def cans_at_home : ℕ := 12
def cans_at_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_office : ℕ := 250

def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_office

def total_money : ℚ := (total_cans : ℚ) / cans_per_dollar

def savings_amount : ℚ := total_money / 2

theorem collins_savings : savings_amount = 43 := by sorry

end collins_savings_l1823_182376


namespace range_of_a_l1823_182381

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) →
  (∃ x : ℝ, x^2 + 2*a*x + (2 - a) = 0) →
  a ≤ -2 ∨ a = 1 := by
sorry

end range_of_a_l1823_182381


namespace remainder_17_power_77_mod_7_l1823_182366

theorem remainder_17_power_77_mod_7 : 17^77 % 7 = 5 := by
  sorry

end remainder_17_power_77_mod_7_l1823_182366


namespace clock_angle_at_3_40_l1823_182386

-- Define the clock and its properties
def clock_degrees : ℝ := 360
def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def time_elapsed : ℝ := 40  -- Minutes elapsed since 3:00

-- Define the positions of the hands at 3:40
def minute_hand_position : ℝ := time_elapsed * minute_hand_speed
def hour_hand_position : ℝ := 90 + time_elapsed * hour_hand_speed

-- Define the angle between the hands
def angle_between_hands : ℝ := |minute_hand_position - hour_hand_position|

-- Theorem statement
theorem clock_angle_at_3_40 : 
  min angle_between_hands (clock_degrees - angle_between_hands) = 130 :=
sorry

end clock_angle_at_3_40_l1823_182386


namespace glasses_in_five_hours_l1823_182349

/-- The number of glasses of water consumed in a given time period -/
def glasses_consumed (rate_minutes : ℕ) (time_hours : ℕ) : ℕ :=
  (time_hours * 60) / rate_minutes

/-- Theorem: Given a rate of 1 glass every 20 minutes, 
    the number of glasses consumed in 5 hours is 15 -/
theorem glasses_in_five_hours : glasses_consumed 20 5 = 15 := by
  sorry

end glasses_in_five_hours_l1823_182349


namespace base_conversion_equality_l1823_182358

theorem base_conversion_equality (k : ℕ) : k = 7 ↔ 
  5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k^1 + 4 * k^0 :=
by sorry

end base_conversion_equality_l1823_182358


namespace parrot_response_characterization_l1823_182367

def parrot_calc (n : ℤ) : ℚ :=
  (5 * n + 14) / 6 - 1

theorem parrot_response_characterization :
  ∀ n : ℤ, (∃ k : ℤ, parrot_calc n = k) ↔ ∃ m : ℤ, n = 6 * m + 2 :=
sorry

end parrot_response_characterization_l1823_182367


namespace greatest_common_divisor_problem_l1823_182342

theorem greatest_common_divisor_problem :
  Nat.gcd 105 (Nat.gcd 1001 (Nat.gcd 2436 (Nat.gcd 10202 49575))) = 7 := by
  sorry

end greatest_common_divisor_problem_l1823_182342


namespace partnership_profit_share_l1823_182375

/-- 
Given:
- A, B, and C are in a partnership
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 4400

Prove that B's share of the profit is 800
-/
theorem partnership_profit_share (c : ℝ) (total_profit : ℝ) 
  (h1 : c > 0)
  (h2 : total_profit = 4400) :
  let b := (2/3) * c
  let a := 3 * b
  let total_investment := a + b + c
  b / total_investment * total_profit = 800 := by
sorry

end partnership_profit_share_l1823_182375


namespace inequality_solution_range_l1823_182337

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℕ+, (x ≤ 4 → a * x.val + 4 ≥ 0) ∧ (x > 4 → a * x.val + 4 < 0)) → 
  -1 ≤ a ∧ a < -4/5 := by
  sorry

end inequality_solution_range_l1823_182337


namespace initial_amount_spent_l1823_182333

theorem initial_amount_spent (total_sets : ℕ) (twenty_dollar_sets : ℕ) (price_per_set : ℕ) :
  total_sets = 250 →
  twenty_dollar_sets = 178 →
  price_per_set = 20 →
  (twenty_dollar_sets * price_per_set : ℕ) = 3560 :=
by sorry

end initial_amount_spent_l1823_182333


namespace labor_union_tree_planting_l1823_182313

theorem labor_union_tree_planting (x : ℕ) : 
  (2 * x + 21 = x * 2 + 21) ∧ 
  (3 * x - 24 = x * 3 - 24) → 
  2 * x + 21 = 3 * x - 24 := by
sorry

end labor_union_tree_planting_l1823_182313


namespace binary_to_quaternary_conversion_l1823_182302

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : ℕ) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : ℕ := sorry

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal 101001110010) = 221302 := by sorry

end binary_to_quaternary_conversion_l1823_182302


namespace diamond_two_three_l1823_182382

/-- The diamond operation defined for real numbers -/
def diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

/-- Theorem stating that 2 ◇ 3 = 16 -/
theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l1823_182382


namespace bald_eagle_dive_time_l1823_182312

/-- The time it takes for the bald eagle to dive to the ground given the specified conditions -/
theorem bald_eagle_dive_time : 
  ∀ (v_eagle : ℝ) (v_falcon : ℝ) (t_falcon : ℝ) (distance : ℝ),
  v_eagle > 0 →
  v_falcon = 2 * v_eagle →
  t_falcon = 15 →
  distance > 0 →
  distance = v_eagle * (2 * t_falcon) →
  distance = v_falcon * t_falcon →
  2 * t_falcon = 30 :=
by
  sorry

end bald_eagle_dive_time_l1823_182312


namespace path_count_theorem_l1823_182357

/-- The number of paths from (0,0) to (4,3) on a 5x4 grid with exactly 7 steps -/
def number_of_paths : ℕ := 35

/-- The width of the grid -/
def grid_width : ℕ := 5

/-- The height of the grid -/
def grid_height : ℕ := 4

/-- The total number of steps in each path -/
def total_steps : ℕ := 7

/-- The number of steps to the right in each path -/
def right_steps : ℕ := 4

/-- The number of steps up in each path -/
def up_steps : ℕ := 3

theorem path_count_theorem :
  number_of_paths = Nat.choose total_steps up_steps :=
by sorry

end path_count_theorem_l1823_182357


namespace circle_area_increase_l1823_182391

theorem circle_area_increase (r : ℝ) (hr : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end circle_area_increase_l1823_182391


namespace rabbits_after_four_springs_l1823_182326

/-- Calculates the total number of rabbits after four breeding seasons --/
def totalRabbitsAfterFourSprings (initialBreedingRabbits : ℕ) 
  (spring1KittensPerRabbit spring1AdoptionRate : ℚ) (spring1Returns : ℕ)
  (spring2Kittens : ℕ) (spring2AdoptionRate : ℚ) (spring2Returns : ℕ)
  (spring3BreedingRabbits : ℕ) (spring3KittensPerRabbit : ℕ) (spring3AdoptionRate : ℚ) (spring3Returns : ℕ)
  (spring4BreedingRabbits : ℕ) (spring4KittensPerRabbit : ℕ) (spring4AdoptionRate : ℚ) (spring4Returns : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the total number of rabbits after four springs is 242 --/
theorem rabbits_after_four_springs : 
  totalRabbitsAfterFourSprings 10 10 (1/2) 5 60 (2/5) 10 12 8 (3/10) 3 12 6 (1/5) 2 = 242 :=
by sorry

end rabbits_after_four_springs_l1823_182326


namespace abcd_imag_zero_l1823_182319

open Complex

-- Define the condition for angles being equal and oppositely oriented
def anglesEqualOpposite (a b c d : ℂ) : Prop :=
  ∃ θ : ℝ, b / a = exp (θ * I) ∧ d / c = exp (-θ * I)

theorem abcd_imag_zero (a b c d : ℂ) 
  (h : anglesEqualOpposite a b c d) : 
  (a * b * c * d).im = 0 := by
  sorry

end abcd_imag_zero_l1823_182319


namespace phone_service_cost_per_minute_l1823_182314

/-- Calculates the cost per minute for a phone service given the total bill, monthly fee, and minutes used. -/
def cost_per_minute (total_bill monthly_fee : ℚ) (minutes_used : ℕ) : ℚ :=
  (total_bill - monthly_fee) / minutes_used

/-- Theorem stating that given the specific conditions, the cost per minute is $0.12. -/
theorem phone_service_cost_per_minute :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let minutes_used : ℕ := 178
  cost_per_minute total_bill monthly_fee minutes_used = 0.12 := by
  sorry

end phone_service_cost_per_minute_l1823_182314


namespace fraction_comparison_l1823_182365

theorem fraction_comparison : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 4 / 20 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 3 / 15 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 6 : ℚ) ≠ 7 / 5 := by
  sorry

end fraction_comparison_l1823_182365


namespace sales_solution_l1823_182355

def sales_problem (sale1 sale2 sale3 sale5 sale6 average : ℕ) : Prop :=
  let total_sales := 6 * average
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6
  total_sales - known_sales = 5730

theorem sales_solution :
  sales_problem 4000 6524 5689 6000 12557 7000 := by
  sorry

end sales_solution_l1823_182355


namespace max_age_on_aubrey_eighth_birthday_l1823_182359

/-- Proves that Max's age on Aubrey's 8th birthday is 6 years -/
theorem max_age_on_aubrey_eighth_birthday 
  (max_birth : ℕ) -- Max's birth year
  (luka_birth : ℕ) -- Luka's birth year
  (aubrey_birth : ℕ) -- Aubrey's birth year
  (h1 : max_birth = luka_birth + 4) -- Max born when Luka turned 4
  (h2 : luka_birth = aubrey_birth - 2) -- Luka is 2 years older than Aubrey
  (h3 : aubrey_birth + 8 = max_birth + 6) -- Aubrey's 8th birthday is when Max is 6
  : (aubrey_birth + 8) - max_birth = 6 := by
sorry

end max_age_on_aubrey_eighth_birthday_l1823_182359


namespace three_aligned_probability_l1823_182339

-- Define the grid
def Grid := Fin 3 × Fin 3

-- Define the number of markers
def num_markers : ℕ := 4

-- Define the total number of cells in the grid
def total_cells : ℕ := 9

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total number of ways to place markers
def total_arrangements : ℕ := combination total_cells num_markers

-- Define the number of ways to align 3 markers in a row, column, or diagonal
def aligned_arrangements : ℕ := 48

-- The main theorem
theorem three_aligned_probability :
  (aligned_arrangements : ℚ) / total_arrangements = 8 / 21 := by
  sorry

end three_aligned_probability_l1823_182339


namespace binomial_2024_1_l1823_182305

theorem binomial_2024_1 : Nat.choose 2024 1 = 2024 := by
  sorry

end binomial_2024_1_l1823_182305


namespace coefficient_x3y2z5_in_expansion_l1823_182309

/-- The coefficient of x³y²z⁵ in the expansion of (2x+y+z)¹⁰ -/
def coefficient : ℕ :=
  2^3 * (Nat.choose 10 3) * (Nat.choose 7 2) * (Nat.choose 5 5)

/-- Theorem stating that the coefficient of x³y²z⁵ in (2x+y+z)¹⁰ is 20160 -/
theorem coefficient_x3y2z5_in_expansion : coefficient = 20160 := by
  sorry

end coefficient_x3y2z5_in_expansion_l1823_182309


namespace complex_modulus_l1823_182369

theorem complex_modulus (z : ℂ) : z - Complex.I = 1 + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l1823_182369


namespace marble_count_l1823_182384

theorem marble_count (fabian kyle miles : ℕ) 
  (h1 : fabian = 15)
  (h2 : fabian = 3 * kyle)
  (h3 : fabian = 5 * miles) :
  kyle + miles = 8 := by
  sorry

end marble_count_l1823_182384


namespace boys_in_second_grade_l1823_182321

/-- The number of students in the 3rd grade -/
def third_grade : ℕ := 19

/-- The number of students in the 4th grade -/
def fourth_grade : ℕ := 2 * third_grade

/-- The number of girls in the 2nd grade -/
def second_grade_girls : ℕ := 19

/-- The total number of students across all three grades -/
def total_students : ℕ := 86

/-- The number of boys in the 2nd grade -/
def second_grade_boys : ℕ := total_students - fourth_grade - third_grade - second_grade_girls

theorem boys_in_second_grade : second_grade_boys = 10 := by
  sorry

end boys_in_second_grade_l1823_182321


namespace fraction_sum_l1823_182374

theorem fraction_sum (p q : ℚ) (h : p / q = 4 / 5) : 
  1 / 7 + (2 * q - p) / (2 * q + p) = 4 / 7 := by
  sorry

end fraction_sum_l1823_182374


namespace intersection_points_on_circle_l1823_182318

/-- Two parabolas with mutually perpendicular axes -/
structure PerpendicularParabolas where
  -- First parabola: x = ay² + b
  a : ℝ
  b : ℝ
  -- Second parabola: y = cx² + d
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  c_pos : 0 < c

/-- The four intersection points of two perpendicular parabolas -/
def intersectionPoints (p : PerpendicularParabolas) : Set (ℝ × ℝ) :=
  {point | point.1 = p.a * point.2^2 + p.b ∧ point.2 = p.c * point.1^2 + p.d}

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that the intersection points lie on a circle -/
theorem intersection_points_on_circle (p : PerpendicularParabolas) :
  ∃ (circle : Circle), ∀ point ∈ intersectionPoints p,
    (point.1 - circle.center.1)^2 + (point.2 - circle.center.2)^2 = circle.radius^2 := by
  sorry

end intersection_points_on_circle_l1823_182318


namespace system_of_equations_l1823_182399

theorem system_of_equations (x y z : ℝ) 
  (eq1 : y + z = 15 - 4*x)
  (eq2 : x + z = -17 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  2*x + 2*y + 2*z = 7/3 := by
sorry

end system_of_equations_l1823_182399


namespace abs_sum_minimum_l1823_182378

theorem abs_sum_minimum (x : ℝ) : 
  |x - 4| + |x - 6| ≥ 2 ∧ ∃ y : ℝ, |y - 4| + |y - 6| = 2 := by
  sorry

end abs_sum_minimum_l1823_182378


namespace marathon_distance_l1823_182353

theorem marathon_distance (marathon_miles : ℕ) (marathon_yards : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) :
  marathon_miles = 26 →
  marathon_yards = 395 →
  yards_per_mile = 1760 →
  num_marathons = 15 →
  (num_marathons * marathon_yards) % yards_per_mile = 645 := by
  sorry

#check marathon_distance

end marathon_distance_l1823_182353


namespace inscribed_circle_triangle_perimeter_l1823_182344

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of XT, where T is the tangency point on XY -/
  xt : ℝ
  /-- The length of TY, where T is the tangency point on XY -/
  ty : ℝ

/-- Calculate the perimeter of a triangle with an inscribed circle -/
def perimeter (t : InscribedCircleTriangle) : ℝ :=
  sorry

theorem inscribed_circle_triangle_perimeter
  (t : InscribedCircleTriangle)
  (h_r : t.r = 24)
  (h_xt : t.xt = 26)
  (h_ty : t.ty = 31) :
  perimeter t = 345 :=
sorry

end inscribed_circle_triangle_perimeter_l1823_182344


namespace polynomial_simplification_l1823_182356

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 5 * x^4 - 3 * Real.sqrt 2 * x^3 + 8 * x^2 + 2 * x - 6) + 
  (-5 * x^4 + Real.sqrt 2 * x^3 - 3 * x^2 + x + 10) = 
  2 * x^5 - 2 * Real.sqrt 2 * x^3 + 5 * x^2 + 3 * x + 4 := by
  sorry

end polynomial_simplification_l1823_182356


namespace pictures_deleted_l1823_182362

theorem pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (remaining_pics : ℕ) : 
  zoo_pics = 49 → museum_pics = 8 → remaining_pics = 19 →
  zoo_pics + museum_pics - remaining_pics = 38 := by
  sorry

end pictures_deleted_l1823_182362


namespace not_equivalent_fraction_l1823_182335

theorem not_equivalent_fraction : (1 : ℚ) / 20000000 ≠ (48 : ℚ) / 1000000000 := by
  sorry

end not_equivalent_fraction_l1823_182335


namespace herbert_age_next_year_l1823_182398

theorem herbert_age_next_year (kris_age : ℕ) (age_difference : ℕ) :
  kris_age = 24 →
  age_difference = 10 →
  kris_age - age_difference + 1 = 15 := by
  sorry

end herbert_age_next_year_l1823_182398


namespace cube_root_plus_abs_plus_power_equals_six_linear_function_through_two_points_l1823_182370

-- Problem 1
theorem cube_root_plus_abs_plus_power_equals_six :
  (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_two_points :
  ∀ (k b : ℝ), (∀ x y : ℝ, y = k * x + b) →
  (1 = k * 0 + b) →
  (5 = k * 2 + b) →
  (∀ x : ℝ, k * x + b = 2 * x + 1) := by sorry

end cube_root_plus_abs_plus_power_equals_six_linear_function_through_two_points_l1823_182370


namespace inequality_proof_l1823_182329

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) ∧
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
sorry

end inequality_proof_l1823_182329


namespace inequality_analysis_l1823_182389

theorem inequality_analysis (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) :
  (∀ z, x + z > y + z) ∧
  (∀ z, x - z > y - z) ∧
  (∃ z, ¬(x * z > y * z)) ∧
  (∀ z, x / z^2 > y / z^2) ∧
  (∀ z, x * z^2 > y * z^2) :=
by sorry

end inequality_analysis_l1823_182389


namespace rectangle_perimeter_l1823_182317

/-- 
Given a rectangle where:
- The long sides are three times the length of the short sides
- One short side is 80 feet long
Prove that the perimeter of the rectangle is 640 feet.
-/
theorem rectangle_perimeter (short_side : ℝ) (h1 : short_side = 80) : 
  2 * short_side + 2 * (3 * short_side) = 640 := by
  sorry

#check rectangle_perimeter

end rectangle_perimeter_l1823_182317


namespace second_number_approximation_l1823_182347

theorem second_number_approximation (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 9)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) : 
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ y = 40 + ε :=
sorry

end second_number_approximation_l1823_182347


namespace prob_at_least_three_matching_l1823_182351

/-- The number of sides on each die -/
def numSides : ℕ := 10

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of getting at least three matching dice out of five fair ten-sided dice -/
def probAtLeastThreeMatching : ℚ := 173 / 20000

/-- Theorem stating that the probability of at least three out of five fair ten-sided dice 
    showing the same value is 173/20000 -/
theorem prob_at_least_three_matching : 
  probAtLeastThreeMatching = 173 / 20000 := by
  sorry

end prob_at_least_three_matching_l1823_182351


namespace avg_problem_l1823_182396

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Theorem: The average of [2 4], [6 2], and [3 3] is 10/3 -/
theorem avg_problem : avg3 (avg2 2 4) (avg2 6 2) (avg2 3 3) = 10 / 3 := by
  sorry

end avg_problem_l1823_182396


namespace product_equals_zero_l1823_182325

theorem product_equals_zero (a : ℤ) (h : a = -1) : (a - 3) * (a - 2) * (a - 1) * a = 0 := by
  sorry

end product_equals_zero_l1823_182325


namespace average_home_runs_l1823_182310

theorem average_home_runs (players_5 players_7 players_9 players_11 players_13 : ℕ) 
  (h1 : players_5 = 3)
  (h2 : players_7 = 2)
  (h3 : players_9 = 1)
  (h4 : players_11 = 2)
  (h5 : players_13 = 1) :
  (5 * players_5 + 7 * players_7 + 9 * players_9 + 11 * players_11 + 13 * players_13) / 
  (players_5 + players_7 + players_9 + players_11 + players_13) = 73 / 9 :=
by sorry

end average_home_runs_l1823_182310


namespace circle_equation_correct_l1823_182368

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a circle -/
def Point.liesOn (p : Point) (c : Circle) : Prop :=
  c.equation p.x p.y

/-- The circle we want to prove about -/
def ourCircle : Circle :=
  { center := { x := 2, y := -1 }
  , equation := fun x y => (x - 2)^2 + (y + 1)^2 = 2 }

/-- The theorem to prove -/
theorem circle_equation_correct :
  ourCircle.center = { x := 2, y := -1 } ∧
  Point.liesOn { x := 3, y := 0 } ourCircle :=
sorry

end circle_equation_correct_l1823_182368


namespace yu_chan_walking_distance_l1823_182338

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walking_distance : 
  step_length * walking_time * steps_per_minute = 682.5 := by
  sorry

end yu_chan_walking_distance_l1823_182338


namespace equation_solution_l1823_182340

theorem equation_solution : ∃! x : ℝ, (4 : ℝ) ^ (x + 6) = 64 ^ x :=
  have h : (4 : ℝ) ^ (3 + 6) = 64 ^ 3 := by sorry
  ⟨3, h, λ y hy => by sorry⟩

#check equation_solution

end equation_solution_l1823_182340


namespace problem_solution_l1823_182379

/-- Checks if a sequence of binomial coefficients forms an arithmetic sequence -/
def is_arithmetic_sequence (n : ℕ) (j : ℕ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i < k - 1 → 2 * (n.choose (j + i + 1)) = (n.choose (j + i)) + (n.choose (j + i + 2))

/-- The value of k that satisfies the conditions of the problem -/
def k : ℕ := 4

/-- The condition (a) of the problem -/
def condition_a (k : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ∀ j : ℕ, j ≤ n - k + 1 → ¬(is_arithmetic_sequence n j k)

/-- The condition (b) of the problem -/
def condition_b (k : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ ∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_sequence n j (k - 1)

/-- The form of n that satisfies condition (b) for k = 4 -/
def valid_n (m : ℕ) : ℕ := m^2 - 2

theorem problem_solution :
  condition_a k ∧
  condition_b k ∧
  (∀ n : ℕ, n > 0 → (∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_sequence n j (k - 1))
                 ↔ (∃ m : ℕ, m ≥ 3 ∧ n = valid_n m)) :=
sorry

end problem_solution_l1823_182379


namespace age_ratio_proof_l1823_182334

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  (a = b + 2) →  -- a is two years older than b
  (a + b + c = 27) →  -- The total of the ages of a, b, and c is 27
  (b = 10) →  -- b is 10 years old
  (b : ℚ) / c = 2 / 1 :=  -- The ratio of b's age to c's age is 2:1
by sorry

end age_ratio_proof_l1823_182334


namespace gcd_9247_4567_l1823_182350

theorem gcd_9247_4567 : Nat.gcd 9247 4567 = 1 := by
  sorry

end gcd_9247_4567_l1823_182350


namespace union_M_N_intersect_N_complement_M_l1823_182360

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | x ≤ 2} := by sorry

-- Theorem for N ∩ (∁ᵤM)
theorem intersect_N_complement_M : N ∩ (U \ M) = {x | x < -2} := by sorry

end union_M_N_intersect_N_complement_M_l1823_182360


namespace triangle_perimeter_l1823_182388

theorem triangle_perimeter (a b c : ℝ) :
  |a - 2 * Real.sqrt 2| + Real.sqrt (b - 5) + (c - 3 * Real.sqrt 2)^2 = 0 →
  a + b + c = 5 + 5 * Real.sqrt 2 := by
  sorry

end triangle_perimeter_l1823_182388


namespace two_tangent_circles_l1823_182377

/-- The parabola y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The directrix of the parabola -/
def directrix : ℝ → ℝ := λ x => -2

/-- The point M -/
def point_M : ℝ × ℝ := (3, 3)

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_focus : dist center focus = radius
  passes_through_M : dist center point_M = radius
  tangent_to_directrix : abs (center.2 - directrix center.1) = radius

/-- The main theorem -/
theorem two_tangent_circles : 
  ∃! (circles : Finset TangentCircle), circles.card = 2 := by sorry

end two_tangent_circles_l1823_182377


namespace last_two_digits_sum_l1823_182387

theorem last_two_digits_sum (n : ℕ) : (6^15 + 10^15) % 100 = 0 := by
  sorry

end last_two_digits_sum_l1823_182387


namespace expression_evaluation_l1823_182324

theorem expression_evaluation : -20 + 7 * (8 - 2 / 2) = 29 := by
  sorry

end expression_evaluation_l1823_182324


namespace integer_and_mod_three_remainder_l1823_182371

theorem integer_and_mod_three_remainder (n : ℕ+) :
  ∃ k : ℤ, (n.val : ℝ)^3 + (3/2) * (n.val : ℝ)^2 + (1/2) * (n.val : ℝ) - 1 = (k : ℝ) ∧ k ≡ 2 [ZMOD 3] :=
sorry

end integer_and_mod_three_remainder_l1823_182371


namespace triangle_probability_is_2ln2_minus_1_l1823_182315

-- Define the rod breaking process
def rod_break (total_length : ℝ) : ℝ × ℝ × ℝ :=
  sorry

-- Define the condition for triangle formation
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the probability of forming a triangle
def triangle_probability : ℝ :=
  sorry

-- Theorem statement
theorem triangle_probability_is_2ln2_minus_1 :
  triangle_probability = 2 * Real.log 2 - 1 :=
sorry

end triangle_probability_is_2ln2_minus_1_l1823_182315


namespace inequality_solution_l1823_182346

theorem inequality_solution : 
  {x : ℝ | 5*x > 4*x + 2} = {x : ℝ | x > 2} := by sorry

end inequality_solution_l1823_182346


namespace count_special_numbers_eq_210_l1823_182395

/-- The number of ways to choose k elements from n elements without replacement and with order -/
def permutations (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The count of four-digit numbers with specific properties -/
def count_special_numbers : ℕ :=
  let digits := 10  -- 0 to 9
  let case1 := permutations 8 2 * permutations 2 2  -- for 0 and 8
  let case2 := permutations 7 1 * permutations 7 1 * permutations 2 2  -- for 1 and 9
  case1 + case2

theorem count_special_numbers_eq_210 :
  count_special_numbers = 210 := by sorry

end count_special_numbers_eq_210_l1823_182395


namespace jessica_expense_increase_l1823_182308

/-- Calculates the increase in Jessica's yearly expenses --/
def yearly_expense_increase (
  last_year_rent : ℕ)
  (last_year_food : ℕ)
  (last_year_insurance : ℕ)
  (rent_increase_percent : ℕ)
  (food_increase_percent : ℕ)
  (insurance_multiplier : ℕ) : ℕ :=
  let new_rent := last_year_rent + last_year_rent * rent_increase_percent / 100
  let new_food := last_year_food + last_year_food * food_increase_percent / 100
  let new_insurance := last_year_insurance * insurance_multiplier
  let last_year_total := last_year_rent + last_year_food + last_year_insurance
  let this_year_total := new_rent + new_food + new_insurance
  (this_year_total - last_year_total) * 12

theorem jessica_expense_increase :
  yearly_expense_increase 1000 200 100 30 50 3 = 7200 := by
  sorry

end jessica_expense_increase_l1823_182308


namespace integer_solutions_of_inequalities_l1823_182303

theorem integer_solutions_of_inequalities :
  let S := { x : ℤ | (4 * (1 + x) : ℚ) / 3 - 1 ≤ (5 + x : ℚ) / 2 ∧
                     (x : ℚ) - 5 ≤ (3 / 2) * ((3 * x : ℚ) - 2) }
  S = {0, 1, 2} := by
  sorry

end integer_solutions_of_inequalities_l1823_182303


namespace largest_factorial_as_consecutive_product_l1823_182330

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 0 → 
  (∃ k : ℕ, n! = (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5)) → 
  n = 0 :=
by sorry

end largest_factorial_as_consecutive_product_l1823_182330


namespace mary_final_book_count_l1823_182390

def calculate_final_books (initial_books : ℕ) (monthly_club_books : ℕ) (months : ℕ) 
  (bought_books : ℕ) (gift_books : ℕ) (removed_books : ℕ) : ℕ :=
  initial_books + monthly_club_books * months + bought_books + gift_books - removed_books

theorem mary_final_book_count : 
  calculate_final_books 72 1 12 7 5 15 = 81 := by sorry

end mary_final_book_count_l1823_182390


namespace vector_projection_l1823_182300

/-- The projection of vector a onto vector b is -√5/5 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (3, 1) → b = (-2, 4) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 5 := by
  sorry

end vector_projection_l1823_182300


namespace absolute_value_theorem_l1823_182322

theorem absolute_value_theorem (x : ℝ) (h : x < 1) : 
  |x - Real.sqrt ((x - 2)^2)| = 2 - 2*x := by
  sorry

end absolute_value_theorem_l1823_182322


namespace gcd_of_B_l1823_182336

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = 6 * x + 6}

theorem gcd_of_B : ∃ d : ℕ, d > 0 ∧ ∀ n ∈ B, d ∣ n ∧ ∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d :=
  sorry

end gcd_of_B_l1823_182336


namespace floor_length_percentage_l1823_182392

-- Define the parameters of the problem
def floor_length : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def paint_rate : ℝ := 3.00001

-- Define the theorem
theorem floor_length_percentage (l b : ℝ) (h1 : l = floor_length) 
  (h2 : l * b = total_cost / paint_rate) : 
  (l - b) / b * 100 = 200 := by
  sorry

end floor_length_percentage_l1823_182392


namespace jerry_bacon_strips_l1823_182383

/-- Represents the number of calories in Jerry's breakfast items and total breakfast --/
structure BreakfastCalories where
  pancakeCalories : ℕ
  baconCalories : ℕ
  cerealCalories : ℕ
  totalCalories : ℕ

/-- Calculates the number of bacon strips in Jerry's breakfast --/
def calculateBaconStrips (b : BreakfastCalories) : ℕ :=
  (b.totalCalories - (6 * b.pancakeCalories + b.cerealCalories)) / b.baconCalories

/-- Theorem stating that Jerry had 2 strips of bacon for breakfast --/
theorem jerry_bacon_strips :
  let b : BreakfastCalories := {
    pancakeCalories := 120,
    baconCalories := 100,
    cerealCalories := 200,
    totalCalories := 1120
  }
  calculateBaconStrips b = 2 := by
  sorry


end jerry_bacon_strips_l1823_182383


namespace contrapositive_equivalence_l1823_182341

theorem contrapositive_equivalence :
  (∀ a b : ℝ, (a + b = 3 → a^2 + b^2 ≥ 4)) ↔
  (∀ a b : ℝ, (a^2 + b^2 < 4 → a + b ≠ 3)) :=
by sorry

end contrapositive_equivalence_l1823_182341


namespace savings_amount_l1823_182348

/-- Represents the price of a single book -/
def book_price : ℝ := 45

/-- Represents the discount percentage for Promotion A -/
def promotion_a_discount : ℝ := 0.4

/-- Represents the fixed discount amount for Promotion B -/
def promotion_b_discount : ℝ := 15

/-- Represents the local tax rate -/
def tax_rate : ℝ := 0.08

/-- Calculates the total cost for Promotion A including tax -/
def total_cost_a : ℝ :=
  (book_price + book_price * (1 - promotion_a_discount)) * (1 + tax_rate)

/-- Calculates the total cost for Promotion B including tax -/
def total_cost_b : ℝ :=
  (book_price + (book_price - promotion_b_discount)) * (1 + tax_rate)

/-- Theorem stating the savings amount by choosing Promotion A over Promotion B -/
theorem savings_amount : 
  total_cost_b - total_cost_a = 3.24 := by sorry

end savings_amount_l1823_182348


namespace function_value_theorem_l1823_182363

theorem function_value_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (x + 1) = x) 
  (h2 : f a = 8) : 
  a = 9 := by
  sorry

end function_value_theorem_l1823_182363


namespace relationship_between_exponents_l1823_182352

theorem relationship_between_exponents 
  (a c e f : ℝ) 
  (x y z w : ℝ) 
  (h1 : a^(2*x) = c^(3*y)) 
  (h2 : a^(2*x) = e) 
  (h3 : c^(3*y) = e) 
  (h4 : c^(4*z) = a^(3*w)) 
  (h5 : c^(4*z) = f) 
  (h6 : a^(3*w) = f) 
  (h7 : a ≠ 0) 
  (h8 : c ≠ 0) 
  (h9 : e > 0) 
  (h10 : f > 0) : 
  2*w*z = x*y := by
sorry

end relationship_between_exponents_l1823_182352


namespace valid_numbers_characterization_l1823_182332

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (10 * n) % 7 = 0 ∧
  (n / 1000 * 10000 + (n % 1000)) % 7 = 0 ∧
  (n / 100 * 1000 + (n % 100) + (n / 1000 * 10000)) % 7 = 0 ∧
  (n / 10 * 100 + (n % 10) + (n / 100 * 10000)) % 7 = 0 ∧
  (n * 10 + (n / 1000)) % 7 = 0

theorem valid_numbers_characterization :
  {n : ℕ | is_valid_number n} = {7000, 7007, 7070, 7077, 7700, 7707, 7770, 7777} := by
  sorry

end valid_numbers_characterization_l1823_182332


namespace equation_solutions_l1823_182331

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, (x - 1)^2 = 2*(x - 1) ↔ x = 1 ∨ x = 3) := by
sorry

end equation_solutions_l1823_182331


namespace highest_score_can_be_less_than_16_l1823_182361

/-- Represents a team in the tournament -/
structure Team :=
  (id : Nat)
  (score : Nat)

/-- Represents the tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (games_played : Nat)
  (total_points : Nat)

/-- The tournament satisfies the given conditions -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 16 ∧
  t.games_played = (t.num_teams * (t.num_teams - 1)) / 2 ∧
  t.total_points = 2 * t.games_played

/-- The highest score in the tournament -/
def highest_score (t : Tournament) : Nat :=
  Finset.sup t.teams (fun team => team.score)

/-- Theorem stating that it's possible for the highest score to be less than 16 -/
theorem highest_score_can_be_less_than_16 (t : Tournament) :
  valid_tournament t → ∃ (score : Nat), highest_score t < 16 :=
by
  sorry

end highest_score_can_be_less_than_16_l1823_182361


namespace unique_multiplication_707_l1823_182311

theorem unique_multiplication_707 : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    ∃ (a b : ℕ), n = 100 * a + 70 + b ∧ 
    707 * n = 124432 := by
  sorry

end unique_multiplication_707_l1823_182311


namespace circumscribed_circle_condition_l1823_182372

/-- Two lines forming a quadrilateral with coordinate axes that has a circumscribed circle -/
def has_circumscribed_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((a + 2) * x + (1 - a) * y - 3 = 0) ∧
    ((a - 1) * x + (2 * a + 3) * y + 2 = 0) ∧
    (x ≥ 0 ∧ y ≥ 0)

/-- Theorem stating the condition for the quadrilateral to have a circumscribed circle -/
theorem circumscribed_circle_condition (a : ℝ) :
  has_circumscribed_circle a → (a = 1 ∨ a = -1) :=
by
  sorry

end circumscribed_circle_condition_l1823_182372


namespace average_age_of_women_l1823_182354

/-- The average age of four women given the following conditions:
    - There are 15 men initially.
    - The average age of 15 men is 40 years.
    - Four men of ages 26, 32, 41, and 39 years are replaced by four women.
    - The new average age increases by 2.9 years after the replacement. -/
theorem average_age_of_women (
  initial_men : ℕ)
  (initial_avg_age : ℝ)
  (replaced_men_ages : Fin 4 → ℝ)
  (new_avg_increase : ℝ)
  (h1 : initial_men = 15)
  (h2 : initial_avg_age = 40)
  (h3 : replaced_men_ages = ![26, 32, 41, 39])
  (h4 : new_avg_increase = 2.9)
  : (initial_men * initial_avg_age + 4 * new_avg_increase * initial_men - (replaced_men_ages 0 + replaced_men_ages 1 + replaced_men_ages 2 + replaced_men_ages 3)) / 4 = 45.375 := by
  sorry


end average_age_of_women_l1823_182354


namespace jump_rope_problem_l1823_182323

theorem jump_rope_problem (a : ℕ) : 
  let counts : List ℕ := [180, 182, 173, 175, a, 178, 176]
  (counts.sum / counts.length : ℚ) = 178 →
  a = 182 := by
sorry

end jump_rope_problem_l1823_182323


namespace events_mutually_exclusive_l1823_182327

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of male students
def male_students : ℕ := 3

-- Define the number of female students
def female_students : ℕ := 2

-- Define the number of students to be selected
def selected_students : ℕ := 2

-- Define the event "at least one male student is selected"
def at_least_one_male (selected : Finset (Fin total_students)) : Prop :=
  ∃ s ∈ selected, s.val < male_students

-- Define the event "all female students are selected"
def all_females (selected : Finset (Fin total_students)) : Prop :=
  ∀ s ∈ selected, s.val ≥ male_students

-- Theorem statement
theorem events_mutually_exclusive :
  ∀ selected : Finset (Fin total_students),
  selected.card = selected_students →
  ¬(at_least_one_male selected ∧ all_females selected) :=
sorry

end events_mutually_exclusive_l1823_182327


namespace max_switches_student_circle_l1823_182364

/-- 
Given n students with distinct heights arranged in a circle, 
where switches are allowed between a student and the one directly 
in front if the height difference is at least 2, the maximum number 
of possible switches before reaching a stable arrangement is ⁿC₃.
-/
theorem max_switches_student_circle (n : ℕ) : 
  ∃ (heights : Fin n → ℕ) (is_switch : Fin n → Fin n → Bool),
  (∀ i j, i ≠ j → heights i ≠ heights j) →
  (∀ i j, is_switch i j = true ↔ heights i > heights j + 1) →
  (∃ (switches : List (Fin n × Fin n)),
    (∀ (s : Fin n × Fin n), s ∈ switches → is_switch s.1 s.2 = true) ∧
    (∀ i j, is_switch i j = false) ∧
    switches.length = Nat.choose n 3) :=
by sorry

end max_switches_student_circle_l1823_182364


namespace kim_shoe_pairs_l1823_182345

/-- The number of shoes Kim has -/
def total_shoes : ℕ := 18

/-- The probability of selecting two shoes of the same color -/
def probability : ℚ := 58823529411764705 / 1000000000000000000

/-- The number of pairs of shoes Kim has -/
def num_pairs : ℕ := total_shoes / 2

theorem kim_shoe_pairs :
  (probability = 1 / (total_shoes - 1)) → num_pairs = 9 := by
  sorry

end kim_shoe_pairs_l1823_182345


namespace division_equality_l1823_182316

theorem division_equality : 250 / (5 + 12 * 3^2) = 250 / 113 := by
  sorry

end division_equality_l1823_182316


namespace coconut_grove_yield_l1823_182343

theorem coconut_grove_yield (yield_group1 yield_group2 yield_group3 : ℕ) : 
  yield_group1 = 60 →
  yield_group2 = 120 →
  (3 * yield_group1 + 2 * yield_group2 + yield_group3) / 6 = 100 →
  yield_group3 = 180 := by
sorry

end coconut_grove_yield_l1823_182343


namespace books_pages_after_move_l1823_182328

theorem books_pages_after_move (initial_books : ℕ) (pages_per_book : ℕ) (lost_books : ℕ) : 
  initial_books = 10 → pages_per_book = 100 → lost_books = 2 →
  (initial_books - lost_books) * pages_per_book = 800 := by
  sorry

end books_pages_after_move_l1823_182328


namespace betty_age_l1823_182307

/-- Given the ages of Albert, Mary, Betty, and Charlie, prove Betty's age --/
theorem betty_age (albert mary betty charlie : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14)
  (h4 : charlie = 3 * betty)
  (h5 : charlie = mary + 10) :
  betty = 7 := by
sorry

end betty_age_l1823_182307


namespace like_terms_exponents_l1823_182385

theorem like_terms_exponents (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), 3 * a^(2*m) * b^2 = k * (-1/2 * a^2 * b^(n+1))) →
  m + n = 2 := by
sorry

end like_terms_exponents_l1823_182385


namespace domain_of_g_l1823_182320

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-3) 5

-- Define the function g
def g (x : ℝ) : ℝ := f (x + 1) + f (x - 2)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc (-1) 4

-- Theorem statement
theorem domain_of_g :
  ∀ x ∈ domain_g, (x + 1 ∈ domain_f ∧ x - 2 ∈ domain_f) ∧
  ∀ x ∉ domain_g, (x + 1 ∉ domain_f ∨ x - 2 ∉ domain_f) :=
sorry

end domain_of_g_l1823_182320


namespace missing_sale_is_6088_l1823_182394

/-- Calculates the missing sale amount given the sales for five months and the average sale for six months. -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem stating that the missing sale amount is 6088 given the specific sales and average. -/
theorem missing_sale_is_6088 :
  calculate_missing_sale 5921 5468 5568 6433 5922 5900 = 6088 := by
  sorry

#eval calculate_missing_sale 5921 5468 5568 6433 5922 5900

end missing_sale_is_6088_l1823_182394


namespace complex_equation_solution_l1823_182373

theorem complex_equation_solution (b : ℝ) : 
  (2 - Complex.I) * (4 * Complex.I) = 4 - b * Complex.I → b = -8 := by
  sorry

end complex_equation_solution_l1823_182373


namespace smallest_divisor_of_930_l1823_182393

theorem smallest_divisor_of_930 : ∃ (d : ℕ), d > 1 ∧ d ∣ 930 ∧ ∀ (k : ℕ), 1 < k ∧ k < d → ¬(k ∣ 930) :=
by sorry

end smallest_divisor_of_930_l1823_182393
