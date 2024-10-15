import Mathlib

namespace NUMINAMATH_CALUDE_cable_package_savings_l3020_302063

/-- Calculates the savings from choosing a bundle package over individual subscriptions --/
theorem cable_package_savings
  (basic_cost movie_cost bundle_cost : ℕ)
  (sports_cost_diff : ℕ)
  (h1 : basic_cost = 15)
  (h2 : movie_cost = 12)
  (h3 : sports_cost_diff = 3)
  (h4 : bundle_cost = 25) :
  basic_cost + movie_cost + (movie_cost - sports_cost_diff) - bundle_cost = 11 := by
  sorry


end NUMINAMATH_CALUDE_cable_package_savings_l3020_302063


namespace NUMINAMATH_CALUDE_simplify_expression_l3020_302019

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (144 : ℝ) ^ (1/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3020_302019


namespace NUMINAMATH_CALUDE_num_constructible_heights_l3020_302037

/-- The number of bricks available --/
def num_bricks : ℕ := 25

/-- The possible height contributions of each brick after normalization and simplification --/
def height_options : List ℕ := [0, 3, 4]

/-- A function that returns the set of all possible tower heights --/
noncomputable def constructible_heights : Finset ℕ :=
  sorry

/-- The theorem stating that the number of constructible heights is 98 --/
theorem num_constructible_heights :
  Finset.card constructible_heights = 98 :=
sorry

end NUMINAMATH_CALUDE_num_constructible_heights_l3020_302037


namespace NUMINAMATH_CALUDE_isosceles_triangle_circumscribed_circle_l3020_302046

/-- Given a circle with radius 3 and an isosceles triangle circumscribed around it with a base angle of 30°, 
    this theorem proves the lengths of the sides of the triangle. -/
theorem isosceles_triangle_circumscribed_circle 
  (r : ℝ) 
  (base_angle : ℝ) 
  (h_r : r = 3) 
  (h_angle : base_angle = 30 * π / 180) : 
  ∃ (equal_side base_side : ℝ),
    equal_side = 4 * Real.sqrt 3 + 6 ∧ 
    base_side = 6 * Real.sqrt 3 + 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circumscribed_circle_l3020_302046


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3020_302042

/-- Proves that the average salary of all workers in a workshop is 8000,
    given the specified conditions. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (other_salary : ℕ)
  (h1 : total_workers = 49)
  (h2 : technicians = 7)
  (h3 : technician_salary = 20000)
  (h4 : other_salary = 6000) :
  (technicians * technician_salary + (total_workers - technicians) * other_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3020_302042


namespace NUMINAMATH_CALUDE_arctan_tan_equation_solution_l3020_302098

theorem arctan_tan_equation_solution :
  ∃! x : ℝ, -2*π/3 ≤ x ∧ x ≤ 2*π/3 ∧ Real.arctan (Real.tan x) = 3*x/4 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_equation_solution_l3020_302098


namespace NUMINAMATH_CALUDE_external_bisector_angles_theorem_l3020_302097

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angles of a triangle
def angles (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define external angle bisectors
def externalAngleBisectors (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem external_bisector_angles_theorem (t : Triangle) :
  let t' := externalAngleBisectors t
  angles t' = (40, 65, 75) → angles t = (100, 30, 50) := by
  sorry

end NUMINAMATH_CALUDE_external_bisector_angles_theorem_l3020_302097


namespace NUMINAMATH_CALUDE_budget_is_seven_seventy_l3020_302034

/-- The budget for bulbs given the number of crocus bulbs and their cost -/
def budget_for_bulbs (num_crocus : ℕ) (cost_per_crocus : ℚ) : ℚ :=
  num_crocus * cost_per_crocus

/-- Theorem stating that the budget for bulbs is $7.70 -/
theorem budget_is_seven_seventy :
  budget_for_bulbs 22 (35/100) = 77/10 := by
  sorry

end NUMINAMATH_CALUDE_budget_is_seven_seventy_l3020_302034


namespace NUMINAMATH_CALUDE_range_of_a_l3020_302018

def equation1 (a x : ℝ) : Prop := x^2 + 4*a*x - 4*a + 3 = 0

def equation2 (a x : ℝ) : Prop := x^2 + (a-1)*x + a^2 = 0

def equation3 (a x : ℝ) : Prop := x^2 + 2*a*x - 2*a = 0

def has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, equation1 a x ∨ equation2 a x ∨ equation3 a x

theorem range_of_a : ∀ a : ℝ, has_real_root a ↔ a ≥ -1 ∨ a ≤ -3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3020_302018


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3020_302068

theorem units_digit_of_product (n : ℕ) : n % 10 = (2^101 * 7^1002 * 3^1004) % 10 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3020_302068


namespace NUMINAMATH_CALUDE_color_change_probability_l3020_302011

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of 5-second intervals where a color change occurs -/
def colorChangeIntervals (cycle : TrafficLightCycle) : ℕ := 3

/-- Represents the duration of the observation interval -/
def observationInterval : ℕ := 5

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem color_change_probability (cycle : TrafficLightCycle)
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50) :
    (colorChangeIntervals cycle : ℚ) * observationInterval / (cycleDuration cycle) = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_color_change_probability_l3020_302011


namespace NUMINAMATH_CALUDE_sum_of_specific_values_is_zero_l3020_302055

-- Define an odd function f on ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+1) = -f(x-1)
def hasFunctionalProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = -f (x - 1)

-- Theorem statement
theorem sum_of_specific_values_is_zero
  (f : ℝ → ℝ)
  (h1 : isOddFunction f)
  (h2 : hasFunctionalProperty f) :
  f 0 + f 1 + f 2 + f 3 + f 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_specific_values_is_zero_l3020_302055


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l3020_302083

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes
  (m : Line) (α β : Plane)
  (h1 : parallel α β)
  (h2 : perpendicular m α) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l3020_302083


namespace NUMINAMATH_CALUDE_sum_of_ages_l3020_302033

/-- Given that Ann is 5 years older than Susan and Ann is 16 years old, 
    prove that the sum of their ages is 27 years. -/
theorem sum_of_ages (ann_age susan_age : ℕ) : 
  ann_age = 16 → 
  ann_age = susan_age + 5 → 
  ann_age + susan_age = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3020_302033


namespace NUMINAMATH_CALUDE_percentage_problem_l3020_302053

theorem percentage_problem (x : ℝ) : 
  (40 * x / 100) + (25 / 100 * 60) = 23 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3020_302053


namespace NUMINAMATH_CALUDE_bus_stop_problem_l3020_302001

/-- The number of students who got off the bus at the first stop -/
def students_who_got_off (initial_students : ℕ) (remaining_students : ℕ) : ℕ :=
  initial_students - remaining_students

theorem bus_stop_problem (initial_students remaining_students : ℕ) 
  (h1 : initial_students = 10)
  (h2 : remaining_students = 7) :
  students_who_got_off initial_students remaining_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l3020_302001


namespace NUMINAMATH_CALUDE_find_p_l3020_302081

theorem find_p (P Q : ℝ) (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l3020_302081


namespace NUMINAMATH_CALUDE_function_composition_inverse_l3020_302032

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -5 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_inverse (a b : ℝ) :
  (∀ x, h a b x = x - 9) → (a - b = 41 / 5) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_inverse_l3020_302032


namespace NUMINAMATH_CALUDE_projection_matrix_l3020_302049

def P : Matrix (Fin 2) (Fin 2) ℚ := !![965/1008, 18/41; 19/34, 23/41]

theorem projection_matrix : P * P = P := by sorry

end NUMINAMATH_CALUDE_projection_matrix_l3020_302049


namespace NUMINAMATH_CALUDE_inequality_chain_l3020_302057

theorem inequality_chain (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) :
  x^2 > a*x ∧ a*x > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3020_302057


namespace NUMINAMATH_CALUDE_problem_solution_l3020_302095

theorem problem_solution : ∃ x : ℝ, 3 * x + 3 * 14 + 3 * 15 + 11 = 152 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3020_302095


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3020_302061

theorem quadratic_equation_roots (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 12 → x^2 - 10*x - 11 = 0 ∨ y^2 - 10*y - 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3020_302061


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l3020_302076

theorem imaginary_part_of_one_minus_i_squared (i : ℂ) : 
  Complex.im ((1 - i)^2) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l3020_302076


namespace NUMINAMATH_CALUDE_students_practicing_both_sports_l3020_302072

theorem students_practicing_both_sports :
  -- Define variables
  ∀ (F B x : ℕ),
  -- Condition 1: One-fifth of footballers play basketball
  F / 5 = x →
  -- Condition 2: One-seventh of basketball players play football
  B / 7 = x →
  -- Condition 3: 110 students practice exactly one sport
  (F - x) + (B - x) = 110 →
  -- Conclusion: x (students practicing both sports) = 11
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_students_practicing_both_sports_l3020_302072


namespace NUMINAMATH_CALUDE_books_remaining_after_loans_and_returns_l3020_302002

/-- Calculates the number of books remaining in a special collection after loans and returns. -/
theorem books_remaining_after_loans_and_returns 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 45 → 
  return_rate = 4/5 → 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 66 := by
  sorry

#check books_remaining_after_loans_and_returns

end NUMINAMATH_CALUDE_books_remaining_after_loans_and_returns_l3020_302002


namespace NUMINAMATH_CALUDE_f_composition_of_3_l3020_302023

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_composition_of_3 : f (f (f (f (f 3)))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_3_l3020_302023


namespace NUMINAMATH_CALUDE_expression_evaluation_l3020_302020

theorem expression_evaluation : 
  (2024^3 - 3 * 2024^2 * 2025 + 4 * 2024 * 2025^2 - 2025^3 + 2) / (2024 * 2025) = 2025 - 1 / (2024 * 2025) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3020_302020


namespace NUMINAMATH_CALUDE_equation_solution_l3020_302030

theorem equation_solution : ∃ x : ℚ, (1/7 : ℚ) + 7/x = 15/x + (1/15 : ℚ) ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3020_302030


namespace NUMINAMATH_CALUDE_inequality_proof_l3020_302015

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b*c)) + (b^3 / (c*a)) + (c^3 / (a*b)) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3020_302015


namespace NUMINAMATH_CALUDE_tiffany_lives_l3020_302071

theorem tiffany_lives (initial_lives lost_lives gained_lives final_lives : ℕ) : 
  lost_lives = 14 →
  gained_lives = 27 →
  final_lives = 56 →
  final_lives = initial_lives - lost_lives + gained_lives →
  initial_lives = 43 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_lives_l3020_302071


namespace NUMINAMATH_CALUDE_mosaic_completion_time_l3020_302012

-- Define the start time
def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the time when 1/4 of the mosaic is completed
def quarter_time : ℕ := (12 + 12) * 60 + 45  -- 12:45 PM in minutes since midnight

-- Define the fraction of work completed
def fraction_completed : ℚ := 1/4

-- Define the duration to complete 1/4 of the mosaic
def quarter_duration : ℕ := quarter_time - start_time

-- Theorem to prove
theorem mosaic_completion_time :
  let total_duration : ℕ := (quarter_duration * 4)
  let finish_time : ℕ := (start_time + total_duration) % (24 * 60)
  finish_time = 0  -- 0 minutes past midnight (12:00 AM)
  := by sorry

end NUMINAMATH_CALUDE_mosaic_completion_time_l3020_302012


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3020_302036

theorem multiply_mixed_number : 7 * (12 + 1/4) = 85 + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3020_302036


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3020_302027

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3020_302027


namespace NUMINAMATH_CALUDE_unique_digit_product_l3020_302074

theorem unique_digit_product (A M C : ℕ) : 
  A < 10 → M < 10 → C < 10 →
  (100 * A + 10 * M + C) * (A + M + C) = 2008 →
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_product_l3020_302074


namespace NUMINAMATH_CALUDE_annulus_area_l3020_302041

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (b c a : ℝ) (h1 : b > c) (h2 : b^2 = c^2 + a^2) :
  (π * b^2 - π * c^2) = π * a^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l3020_302041


namespace NUMINAMATH_CALUDE_monthly_rent_is_400_l3020_302010

/-- Calculates the monthly rent per resident in a rental building -/
def monthly_rent_per_resident (total_units : ℕ) (occupancy_rate : ℚ) (total_annual_rent : ℕ) : ℚ :=
  let occupied_units : ℚ := total_units * occupancy_rate
  let annual_rent_per_resident : ℚ := total_annual_rent / occupied_units
  annual_rent_per_resident / 12

/-- Proves that the monthly rent per resident is $400 -/
theorem monthly_rent_is_400 :
  monthly_rent_per_resident 100 (3/4) 360000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_monthly_rent_is_400_l3020_302010


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l3020_302066

/-- The area of a circular sector with central angle 120° and radius √3 is equal to π. -/
theorem sector_area_120_deg_sqrt3_radius (π : ℝ) : 
  let angle : ℝ := 2 * π / 3  -- 120° in radians
  let radius : ℝ := Real.sqrt 3
  let sector_area : ℝ := (1 / 2) * angle * radius^2
  sector_area = π :=
by sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l3020_302066


namespace NUMINAMATH_CALUDE_base_12_remainder_l3020_302006

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The problem statement --/
theorem base_12_remainder (digits : List Nat) (h : digits = [3, 4, 7, 2]) :
  base12ToBase10 digits % 10 = 5 := by
  sorry

#eval base12ToBase10 [3, 4, 7, 2]

end NUMINAMATH_CALUDE_base_12_remainder_l3020_302006


namespace NUMINAMATH_CALUDE_cost_of_second_box_l3020_302065

/-- The cost of cards in the first box -/
def cost_box1 : ℚ := 1.25

/-- The number of cards bought from each box -/
def cards_bought : ℕ := 6

/-- The total amount spent -/
def total_spent : ℚ := 18

/-- The cost of cards in the second box -/
def cost_box2 : ℚ := (total_spent - cards_bought * cost_box1) / cards_bought

theorem cost_of_second_box : cost_box2 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_second_box_l3020_302065


namespace NUMINAMATH_CALUDE_drug_storage_temperature_range_l3020_302052

/-- Given a drug with a storage temperature of 20 ± 2 (°C), 
    the difference between the highest and lowest suitable storage temperatures is 4°C -/
theorem drug_storage_temperature_range : 
  let recommended_temp : ℝ := 20
  let tolerance : ℝ := 2
  let highest_temp := recommended_temp + tolerance
  let lowest_temp := recommended_temp - tolerance
  highest_temp - lowest_temp = 4 := by
  sorry

end NUMINAMATH_CALUDE_drug_storage_temperature_range_l3020_302052


namespace NUMINAMATH_CALUDE_jack_additional_apples_l3020_302013

/-- Represents the capacity of apple baskets and current apple counts -/
structure AppleBaskets where
  jack_capacity : ℕ
  jill_capacity : ℕ
  jack_current : ℕ

/-- The conditions of the apple picking problem -/
def apple_picking_conditions (ab : AppleBaskets) : Prop :=
  ab.jill_capacity = 2 * ab.jack_capacity ∧
  ab.jack_capacity = 12 ∧
  3 * ab.jack_current = ab.jill_capacity

/-- The theorem stating how many more apples Jack's basket can hold -/
theorem jack_additional_apples (ab : AppleBaskets) 
  (h : apple_picking_conditions ab) : 
  ab.jack_capacity - ab.jack_current = 4 := by
  sorry


end NUMINAMATH_CALUDE_jack_additional_apples_l3020_302013


namespace NUMINAMATH_CALUDE_residue_16_pow_3030_mod_23_l3020_302039

theorem residue_16_pow_3030_mod_23 : 16^3030 ≡ 1 [ZMOD 23] := by
  sorry

end NUMINAMATH_CALUDE_residue_16_pow_3030_mod_23_l3020_302039


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l3020_302086

/-- Given a 2x2 matrix A with inverse [[3, -1], [1, 1]], 
    prove that the inverse of A^3 is [[20, -12], [12, -4]] -/
theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, -1], ![1, 1]]) : 
  (A^3)⁻¹ = ![![20, -12], ![12, -4]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l3020_302086


namespace NUMINAMATH_CALUDE_wilsborough_change_l3020_302085

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

end NUMINAMATH_CALUDE_wilsborough_change_l3020_302085


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l3020_302007

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (40 * π / 180) =
  (Real.sin (50 * π / 180) * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.cos (20 * π / 180) * Real.cos (30 * π / 180))) /
  (Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l3020_302007


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3020_302089

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3020_302089


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3020_302035

/-- Given a hyperbola with the general equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    one asymptote passing through the point (2, √3),
    and one focus lying on the directrix of the parabola y² = 4√7x,
    prove that the specific equation of the hyperbola is x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_condition : b / a = Real.sqrt 3 / 2)
  (focus_condition : ∃ (x y : ℝ), x = -Real.sqrt 7 ∧ x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 = 4 ∧ b^2 = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3020_302035


namespace NUMINAMATH_CALUDE_light_flash_interval_l3020_302094

/-- Given a light that flashes 600 times in 1/6 of an hour, prove that the time between each flash is 1 second. -/
theorem light_flash_interval (flashes_per_sixth_hour : ℕ) (h : flashes_per_sixth_hour = 600) :
  (1 / 6 : ℚ) * 3600 / flashes_per_sixth_hour = 1 := by
  sorry

#check light_flash_interval

end NUMINAMATH_CALUDE_light_flash_interval_l3020_302094


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_g_minimum_value_f_inequality_l3020_302084

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

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_g_minimum_value_f_inequality_l3020_302084


namespace NUMINAMATH_CALUDE_pillar_base_side_length_l3020_302090

theorem pillar_base_side_length (string_length : ℝ) (side_length : ℝ) : 
  string_length = 78 → 
  string_length = 3 * side_length → 
  side_length = 26 := by
  sorry

#check pillar_base_side_length

end NUMINAMATH_CALUDE_pillar_base_side_length_l3020_302090


namespace NUMINAMATH_CALUDE_triangle_side_length_l3020_302082

/-- Given a triangle ABC where ∠C = 2∠A, a = 34, and c = 60, prove that b = 4352/450 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h1 : C = 2 * A) (h2 : a = 34) (h3 : c = 60) : 
  b = 4352 / 450 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3020_302082


namespace NUMINAMATH_CALUDE_oliver_used_30_tickets_l3020_302077

/-- The number of times Oliver rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Oliver rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def ticket_cost : ℕ := 3

/-- The total number of tickets Oliver used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * ticket_cost

/-- Theorem stating that Oliver used 30 tickets -/
theorem oliver_used_30_tickets : total_tickets = 30 := by
  sorry

end NUMINAMATH_CALUDE_oliver_used_30_tickets_l3020_302077


namespace NUMINAMATH_CALUDE_missing_number_is_seven_l3020_302031

def known_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

theorem missing_number_is_seven (x : ℕ) :
  (known_numbers.sum + x) / 12 = 12 →
  x = 7 := by sorry

end NUMINAMATH_CALUDE_missing_number_is_seven_l3020_302031


namespace NUMINAMATH_CALUDE_all_sheep_can_be_blue_not_all_sheep_can_be_red_or_green_l3020_302044

/-- Represents the count of sheep of each color -/
structure SheepCounts where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents a transformation of sheep colors -/
inductive SheepTransform
  | BlueRedToGreen
  | BlueGreenToRed
  | RedGreenToBlue

/-- Applies a transformation to the sheep counts -/
def applyTransform (counts : SheepCounts) (transform : SheepTransform) : SheepCounts :=
  match transform with
  | SheepTransform.BlueRedToGreen => 
      ⟨counts.blue - 1, counts.red - 1, counts.green + 2⟩
  | SheepTransform.BlueGreenToRed => 
      ⟨counts.blue - 1, counts.red + 2, counts.green - 1⟩
  | SheepTransform.RedGreenToBlue => 
      ⟨counts.blue + 2, counts.red - 1, counts.green - 1⟩

/-- The initial counts of sheep -/
def initialCounts : SheepCounts := ⟨22, 18, 15⟩

/-- Theorem stating that it's possible for all sheep to become blue -/
theorem all_sheep_can_be_blue :
  ∃ (transforms : List SheepTransform), 
    let finalCounts := transforms.foldl applyTransform initialCounts
    finalCounts.red = 0 ∧ finalCounts.green = 0 ∧ finalCounts.blue > 0 :=
sorry

/-- Theorem stating that it's impossible for all sheep to become red or green -/
theorem not_all_sheep_can_be_red_or_green :
  ¬∃ (transforms : List SheepTransform), 
    let finalCounts := transforms.foldl applyTransform initialCounts
    (finalCounts.blue = 0 ∧ finalCounts.green = 0 ∧ finalCounts.red > 0) ∨
    (finalCounts.blue = 0 ∧ finalCounts.red = 0 ∧ finalCounts.green > 0) :=
sorry

end NUMINAMATH_CALUDE_all_sheep_can_be_blue_not_all_sheep_can_be_red_or_green_l3020_302044


namespace NUMINAMATH_CALUDE_max_cos_diff_l3020_302070

theorem max_cos_diff (x y : Real) (h : Real.sin x - Real.sin y = 3/4) :
  ∃ (max_val : Real), max_val = 23/32 ∧ 
    ∀ (z w : Real), Real.sin z - Real.sin w = 3/4 → Real.cos (z - w) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_cos_diff_l3020_302070


namespace NUMINAMATH_CALUDE_short_trees_planted_l3020_302025

/-- The number of short trees planted in a park. -/
theorem short_trees_planted (current : ℕ) (final : ℕ) (planted : ℕ) : 
  current = 3 → final = 12 → planted = final - current → planted = 9 := by
sorry

end NUMINAMATH_CALUDE_short_trees_planted_l3020_302025


namespace NUMINAMATH_CALUDE_train_average_speed_l3020_302093

theorem train_average_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 80 →
  time = 8 →
  speed = distance / time →
  speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_average_speed_l3020_302093


namespace NUMINAMATH_CALUDE_fred_money_left_l3020_302087

def fred_book_problem (initial_amount : ℕ) (num_books : ℕ) (avg_cost : ℕ) : ℕ :=
  initial_amount - (num_books * avg_cost)

theorem fred_money_left :
  fred_book_problem 236 6 37 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fred_money_left_l3020_302087


namespace NUMINAMATH_CALUDE_f_2012_equals_cos_l3020_302040

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => Real.cos x
| (n + 1) => λ x => deriv (f n) x

theorem f_2012_equals_cos : f 2012 = λ x => Real.cos x := by sorry

end NUMINAMATH_CALUDE_f_2012_equals_cos_l3020_302040


namespace NUMINAMATH_CALUDE_cubic_inequality_l3020_302045

theorem cubic_inequality (a b c : ℝ) 
  (h : ∃ x₁ x₂ x₃ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    x₁^3 + a*x₁^2 + b*x₁ + c = 0 ∧
    x₂^3 + a*x₂^2 + b*x₂ + c = 0 ∧
    x₃^3 + a*x₃^2 + b*x₃ + c = 0 ∧
    x₁ + x₂ + x₃ ≤ 1) :
  a^3*(1 + a + b) - 9*c*(3 + 3*a + a^2) ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3020_302045


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3020_302016

theorem largest_n_with_unique_k : 
  (∃! (n : ℕ), n > 0 ∧ 
    (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15)) → 
  (∃! (n : ℕ), n = 72 ∧ n > 0 ∧ 
    (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3020_302016


namespace NUMINAMATH_CALUDE_system_solution_l3020_302073

theorem system_solution (x y : ℝ) 
  (eq1 : 2 * x + 3 * y = 9) 
  (eq2 : 3 * x + 2 * y = 11) : 
  x - y = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3020_302073


namespace NUMINAMATH_CALUDE_highest_possible_average_after_removing_lowest_score_l3020_302096

def number_of_tests : ℕ := 9
def original_average : ℚ := 68
def lowest_possible_score : ℚ := 0

theorem highest_possible_average_after_removing_lowest_score :
  let total_score : ℚ := number_of_tests * original_average
  let remaining_score : ℚ := total_score - lowest_possible_score
  let new_average : ℚ := remaining_score / (number_of_tests - 1)
  new_average = 76.5 := by
  sorry

end NUMINAMATH_CALUDE_highest_possible_average_after_removing_lowest_score_l3020_302096


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3020_302051

/-- An arithmetic sequence with non-zero terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n ≠ 0) ∧
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_condition : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0) :
  a 7 = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3020_302051


namespace NUMINAMATH_CALUDE_square_sum_problem_l3020_302004

theorem square_sum_problem (a b c d m n : ℕ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l3020_302004


namespace NUMINAMATH_CALUDE_greatest_x_value_l3020_302008

theorem greatest_x_value (x : ℝ) : 
  ((5*x - 20)^2 / (4*x - 5)^2 + (5*x - 20) / (4*x - 5) = 12) → 
  x ≤ 40/21 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3020_302008


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l3020_302092

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if a line is parallel to a plane -/
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is outside of a plane -/
def is_outside (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem parallel_sufficient_not_necessary
  (l : Line3D) (α : Plane3D) :
  (is_parallel l α → is_outside l α) ∧
  ∃ l', is_outside l' α ∧ ¬is_parallel l' α :=
sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l3020_302092


namespace NUMINAMATH_CALUDE_perfect_square_octal_rep_c_is_one_l3020_302047

/-- Octal representation of a number -/
structure OctalRep where
  a : ℕ
  b : ℕ
  c : ℕ
  h_a_nonzero : a ≠ 0

/-- Perfect square with specific octal representation -/
def is_perfect_square_with_octal_rep (n : ℕ) (rep : OctalRep) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ n = 8^3 * rep.a + 8^2 * rep.b + 8 * 3 + rep.c

theorem perfect_square_octal_rep_c_is_one (n : ℕ) (rep : OctalRep) :
  is_perfect_square_with_octal_rep n rep → rep.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_octal_rep_c_is_one_l3020_302047


namespace NUMINAMATH_CALUDE_differential_of_exponential_trig_function_l3020_302038

/-- The differential of y = e^x(cos 2x + 2sin 2x) is dy = 5 e^x cos 2x · dx -/
theorem differential_of_exponential_trig_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => Real.exp x * (Real.cos (2 * x) + 2 * Real.sin (2 * x))
  (deriv y) x = 5 * Real.exp x * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_differential_of_exponential_trig_function_l3020_302038


namespace NUMINAMATH_CALUDE_tan_sum_equals_one_l3020_302099

theorem tan_sum_equals_one (α β : Real) 
  (h1 : Real.tan (α + π/6) = 1/2) 
  (h2 : Real.tan (β - π/6) = 1/3) : 
  Real.tan (α + β) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_equals_one_l3020_302099


namespace NUMINAMATH_CALUDE_incenter_is_angle_bisectors_intersection_l3020_302079

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- An angle bisector of a triangle --/
def angle_bisector (t : Triangle) (vertex : Fin 3) : Set (ℝ × ℝ) := sorry

/-- The intersection point of the angle bisectors --/
def angle_bisectors_intersection (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The incenter of a triangle is the intersection point of its angle bisectors --/
theorem incenter_is_angle_bisectors_intersection (t : Triangle) :
  incenter t = angle_bisectors_intersection t := by sorry

end NUMINAMATH_CALUDE_incenter_is_angle_bisectors_intersection_l3020_302079


namespace NUMINAMATH_CALUDE_house_renovation_time_l3020_302026

theorem house_renovation_time :
  let num_bedrooms : ℕ := 3
  let bedroom_time : ℕ := 4
  let kitchen_time : ℕ := bedroom_time + bedroom_time / 2
  let bedrooms_and_kitchen_time : ℕ := num_bedrooms * bedroom_time + kitchen_time
  let living_room_time : ℕ := 2 * bedrooms_and_kitchen_time
  let total_time : ℕ := bedrooms_and_kitchen_time + living_room_time
  total_time = 54 := by sorry

end NUMINAMATH_CALUDE_house_renovation_time_l3020_302026


namespace NUMINAMATH_CALUDE_percentage_equality_l3020_302091

-- Define variables
variable (j k l m : ℝ)

-- Define the conditions
def condition1 : Prop := 1.25 * j = 0.25 * k
def condition2 : Prop := 1.5 * k = 0.5 * l
def condition3 : Prop := 0.2 * m = 7 * j

-- Theorem statement
theorem percentage_equality 
  (h1 : condition1 j k) 
  (h2 : condition2 k l) 
  (h3 : condition3 j m) : 
  1.75 * l = 0.75 * m := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l3020_302091


namespace NUMINAMATH_CALUDE_ad_cost_per_square_inch_l3020_302024

/-- Proves that the cost per square inch for advertising is $8 --/
theorem ad_cost_per_square_inch :
  let page_length : ℝ := 9
  let page_width : ℝ := 12
  let full_page_area : ℝ := page_length * page_width
  let ad_area : ℝ := full_page_area / 2
  let total_cost : ℝ := 432
  let cost_per_square_inch : ℝ := total_cost / ad_area
  cost_per_square_inch = 8 := by
  sorry

end NUMINAMATH_CALUDE_ad_cost_per_square_inch_l3020_302024


namespace NUMINAMATH_CALUDE_article_markups_l3020_302003

/-- Calculate the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
def calculateMarkup (purchasePrice : ℝ) (overheadPercentage : ℝ) (netProfit : ℝ) : ℝ :=
  let overheadCost := overheadPercentage * purchasePrice
  let totalCost := purchasePrice + overheadCost
  let sellingPrice := totalCost + netProfit
  sellingPrice - purchasePrice

/-- The markups for two articles with given purchase prices, overhead percentages, and desired net profits. -/
theorem article_markups :
  let article1Markup := calculateMarkup 48 0.35 18
  let article2Markup := calculateMarkup 60 0.40 22
  article1Markup = 34.80 ∧ article2Markup = 46 := by
  sorry

#eval calculateMarkup 48 0.35 18
#eval calculateMarkup 60 0.40 22

end NUMINAMATH_CALUDE_article_markups_l3020_302003


namespace NUMINAMATH_CALUDE_watermelon_cost_l3020_302056

/-- The problem of determining the cost of a watermelon --/
theorem watermelon_cost (total_fruits : ℕ) (total_value : ℕ) 
  (melon_capacity : ℕ) (watermelon_capacity : ℕ) :
  total_fruits = 150 →
  total_value = 24000 →
  melon_capacity = 120 →
  watermelon_capacity = 160 →
  ∃ (num_watermelons num_melons : ℕ) (watermelon_cost melon_cost : ℚ),
    num_watermelons + num_melons = total_fruits ∧
    num_watermelons * watermelon_cost = num_melons * melon_cost ∧
    num_watermelons * watermelon_cost + num_melons * melon_cost = total_value ∧
    (num_watermelons : ℚ) / watermelon_capacity + (num_melons : ℚ) / melon_capacity = 1 ∧
    watermelon_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_cost_l3020_302056


namespace NUMINAMATH_CALUDE_sufficient_condition_sum_greater_than_double_l3020_302062

theorem sufficient_condition_sum_greater_than_double (a b c : ℝ) :
  a > c ∧ b > c → a + b > 2 * c := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_sum_greater_than_double_l3020_302062


namespace NUMINAMATH_CALUDE_initial_pizza_slices_l3020_302014

-- Define the number of slices eaten at each meal and the number of slices left
def breakfast_slices : ℕ := 4
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5
def slices_left : ℕ := 2

-- Define the total number of slices eaten
def total_eaten : ℕ := breakfast_slices + lunch_slices + snack_slices + dinner_slices

-- Theorem: The initial number of pizza slices is 15
theorem initial_pizza_slices : 
  total_eaten + slices_left = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_pizza_slices_l3020_302014


namespace NUMINAMATH_CALUDE_find_a_l3020_302022

-- Define the complex numbers a, b, c
variable (a b c : ℂ)

-- Define the conditions
def condition1 : Prop := a + b + c = 5
def condition2 : Prop := a * b + b * c + c * a = 7
def condition3 : Prop := a * b * c = 6
def condition4 : Prop := a.im = 0  -- a is real

-- Theorem statement
theorem find_a (h1 : condition1 a b c) (h2 : condition2 a b c) 
                (h3 : condition3 a b c) (h4 : condition4 a) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_find_a_l3020_302022


namespace NUMINAMATH_CALUDE_smallest_x_value_l3020_302058

theorem smallest_x_value (x : ℝ) : 
  (4 * x / 10 + 1 / (4 * x) = 5 / 8) → 
  x ≥ (25 - Real.sqrt 1265) / 32 ∧ 
  ∃ y : ℝ, y = (25 - Real.sqrt 1265) / 32 ∧ 4 * y / 10 + 1 / (4 * y) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3020_302058


namespace NUMINAMATH_CALUDE_ships_required_equals_round_trip_duration_moscow_astrakhan_ships_required_l3020_302080

/-- Represents the duration of travel and stay in days -/
structure TravelDuration :=
  (moscow_to_astrakhan : ℕ)
  (stay_in_astrakhan : ℕ)
  (astrakhan_to_moscow : ℕ)
  (stay_in_moscow : ℕ)

/-- Calculates the total round trip duration -/
def round_trip_duration (t : TravelDuration) : ℕ :=
  t.moscow_to_astrakhan + t.stay_in_astrakhan + t.astrakhan_to_moscow + t.stay_in_moscow

/-- The number of ships required for continuous daily departures -/
def ships_required (t : TravelDuration) : ℕ :=
  round_trip_duration t

/-- Theorem stating that the number of ships required is equal to the round trip duration -/
theorem ships_required_equals_round_trip_duration (t : TravelDuration) :
  ships_required t = round_trip_duration t := by
  sorry

/-- The specific travel durations given in the problem -/
def moscow_astrakhan_route : TravelDuration :=
  { moscow_to_astrakhan := 4
  , stay_in_astrakhan := 2
  , astrakhan_to_moscow := 5
  , stay_in_moscow := 2 }

/-- Theorem proving that 13 ships are required for the Moscow-Astrakhan route -/
theorem moscow_astrakhan_ships_required :
  ships_required moscow_astrakhan_route = 13 := by
  sorry

end NUMINAMATH_CALUDE_ships_required_equals_round_trip_duration_moscow_astrakhan_ships_required_l3020_302080


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3020_302029

/-- Given a geometric sequence with first term a₁ = 3 and second term a₂ = -1/2,
    prove that the 7th term a₇ = 1/15552 -/
theorem geometric_sequence_seventh_term :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  let a₇ : ℚ := a₁ * r^6
  a₇ = 1/15552 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3020_302029


namespace NUMINAMATH_CALUDE_triangle_side_length_l3020_302050

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  (a = Real.sqrt 3) →
  (Real.sin B = 1 / 2) →
  (C = π / 6) →
  -- Sum of angles in a triangle is π
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusion
  b = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3020_302050


namespace NUMINAMATH_CALUDE_min_sum_of_cubes_when_sum_is_eight_l3020_302017

theorem min_sum_of_cubes_when_sum_is_eight :
  ∀ x y : ℝ, x + y = 8 →
  x^3 + y^3 ≥ 4^3 + 4^3 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_cubes_when_sum_is_eight_l3020_302017


namespace NUMINAMATH_CALUDE_investment_distribution_l3020_302009

/-- Investment problem with given conditions and amounts -/
theorem investment_distribution (total : ℝ) (bonds stocks mutual_funds : ℝ) : 
  total = 210000 ∧ 
  stocks = 2 * bonds ∧ 
  mutual_funds = 4 * stocks ∧ 
  total = bonds + stocks + mutual_funds →
  bonds = 19090.91 ∧ 
  stocks = 38181.82 ∧ 
  mutual_funds = 152727.27 := by
  sorry

end NUMINAMATH_CALUDE_investment_distribution_l3020_302009


namespace NUMINAMATH_CALUDE_product_195_205_l3020_302054

theorem product_195_205 : 195 * 205 = 39975 := by
  sorry

end NUMINAMATH_CALUDE_product_195_205_l3020_302054


namespace NUMINAMATH_CALUDE_intersection_midpoint_distance_l3020_302059

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 - (Real.sqrt 3 / 2) * t, t / 2)

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point P
def point_P : ℝ × ℝ := (Real.sqrt 3, 0)

-- Theorem statement
theorem intersection_midpoint_distance : 
  ∃ (t₁ t₂ : ℝ), 
    let A := line_l t₁
    let B := line_l t₂
    let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
    curve_C (Real.arctan (A.2 / A.1)) = A ∧     -- A is on curve C
    curve_C (Real.arctan (B.2 / B.1)) = B ∧     -- B is on curve C
    Real.sqrt ((D.1 - point_P.1)^2 + (D.2 - point_P.2)^2) = (3 + Real.sqrt 3) / 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_intersection_midpoint_distance_l3020_302059


namespace NUMINAMATH_CALUDE_special_function_at_one_third_l3020_302000

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 1 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- The main theorem -/
theorem special_function_at_one_third {g : ℝ → ℝ} (hg : special_function g) : 
  g (1/3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_third_l3020_302000


namespace NUMINAMATH_CALUDE_nine_crosses_fit_chessboard_l3020_302064

/-- Represents a cross pentomino -/
structure CrossPentomino where
  area : ℕ
  size : ℕ × ℕ

/-- Represents a chessboard -/
structure Chessboard where
  size : ℕ × ℕ
  area : ℕ

/-- Theorem: Nine cross pentominoes can fit within an 8x8 chessboard -/
theorem nine_crosses_fit_chessboard (cross : CrossPentomino) (board : Chessboard) : 
  cross.area = 5 ∧ 
  cross.size = (1, 1) ∧ 
  board.size = (8, 8) ∧ 
  board.area = 64 →
  9 * cross.area ≤ board.area :=
by sorry

end NUMINAMATH_CALUDE_nine_crosses_fit_chessboard_l3020_302064


namespace NUMINAMATH_CALUDE_clock_angle_120_elapsed_time_l3020_302005

/-- Represents the angle between clock hands at a given time --/
def clockAngle (hours minutes : ℝ) : ℝ :=
  (30 * hours + 0.5 * minutes) - (6 * minutes)

/-- Finds the time when the clock hands form a 120° angle after 6:00 PM --/
def findNextAngle120 : ℝ :=
  let f := fun t : ℝ => abs (clockAngle (6 + t / 60) (t % 60) - 120)
  sorry -- Minimize f(t) for 0 ≤ t < 60

theorem clock_angle_120_elapsed_time :
  ∃ t : ℝ, 0 < t ∧ t < 60 ∧ 
  abs (clockAngle 6 0 - 120) < 0.01 ∧
  abs (clockAngle (6 + t / 60) (t % 60) - 120) < 0.01 ∧
  abs (t - 43.64) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_clock_angle_120_elapsed_time_l3020_302005


namespace NUMINAMATH_CALUDE_hospital_nurse_count_l3020_302043

/-- Given a hospital with doctors and nurses, calculate the number of nurses -/
theorem hospital_nurse_count 
  (total : ℕ) -- Total number of doctors and nurses
  (doc_ratio : ℕ) -- Ratio part for doctors
  (nurse_ratio : ℕ) -- Ratio part for nurses
  (h_total : total = 200) -- Total is 200
  (h_ratio : doc_ratio = 4 ∧ nurse_ratio = 6) -- Ratio is 4:6
  : (nurse_ratio : ℚ) / (doc_ratio + nurse_ratio) * total = 120 := by
  sorry

end NUMINAMATH_CALUDE_hospital_nurse_count_l3020_302043


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_l3020_302069

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations for parallel and perpendicular
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeparallel : Plane → Plane → Prop)
variable (planeperpendicular : Plane → Plane → Prop)
variable (lineplaneparallel : Line → Plane → Prop)
variable (lineplaneperpendicular : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ₚ " => planeparallel
local infix:50 " ⊥ₚ " => planeperpendicular
local infix:50 " ∥ₗₚ " => lineplaneparallel
local infix:50 " ⊥ₗₚ " => lineplaneperpendicular

-- Theorem statements
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) :
  m ⊥ₗₚ α → n ⊥ₗₚ β → α ⊥ₚ β → m ⊥ n :=
sorry

theorem perpendicular_parallel 
  (m n : Line) (α β : Plane) :
  m ⊥ₗₚ α → n ∥ₗₚ β → α ∥ₚ β → m ⊥ n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_l3020_302069


namespace NUMINAMATH_CALUDE_angle_measure_l3020_302067

theorem angle_measure (x : ℝ) : 
  (180 - x = 6 * (90 - x)) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3020_302067


namespace NUMINAMATH_CALUDE_total_apples_calculation_total_apples_is_210_l3020_302048

/-- The number of apples bought by two men and three women -/
def total_apples : ℕ := by sorry

/-- The number of men -/
def num_men : ℕ := 2

/-- The number of women -/
def num_women : ℕ := 3

/-- The number of apples bought by each man -/
def apples_per_man : ℕ := 30

/-- The additional number of apples bought by each woman compared to each man -/
def additional_apples_per_woman : ℕ := 20

/-- The number of apples bought by each woman -/
def apples_per_woman : ℕ := apples_per_man + additional_apples_per_woman

theorem total_apples_calculation :
  total_apples = num_men * apples_per_man + num_women * apples_per_woman :=
by sorry

theorem total_apples_is_210 : total_apples = 210 := by sorry

end NUMINAMATH_CALUDE_total_apples_calculation_total_apples_is_210_l3020_302048


namespace NUMINAMATH_CALUDE_square_root_sum_equals_abs_sum_l3020_302088

theorem square_root_sum_equals_abs_sum (x : ℝ) : 
  Real.sqrt ((x - 3)^2) + Real.sqrt ((x + 5)^2) = |x - 3| + |x + 5| :=
by sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_abs_sum_l3020_302088


namespace NUMINAMATH_CALUDE_steel_rusting_not_LeChatelier_l3020_302021

/-- Le Chatelier's principle states that if a change in conditions is imposed on a system at equilibrium, 
    the equilibrium will shift in a direction that tends to reduce that change. -/
def LeChatelier_principle : Prop := sorry

/-- Rusting of steel in humid air -/
def steel_rusting : Prop := sorry

/-- A chemical process that can be explained by Le Chatelier's principle -/
def explainable_by_LeChatelier (process : Prop) : Prop := sorry

theorem steel_rusting_not_LeChatelier : 
  ¬(explainable_by_LeChatelier steel_rusting) := by sorry

end NUMINAMATH_CALUDE_steel_rusting_not_LeChatelier_l3020_302021


namespace NUMINAMATH_CALUDE_puzzle_solution_l3020_302078

def special_operation (a b c : Nat) : Nat :=
  (a * b) * 10000 + (a * c) * 100 + ((a + b + c) * 2)

theorem puzzle_solution :
  (special_operation 5 3 2 = 151022) →
  (special_operation 9 2 4 = 183652) →
  (special_operation 7 2 5 = 143556) := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3020_302078


namespace NUMINAMATH_CALUDE_expression_evaluation_l3020_302075

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) : 
  (3*x^2 + y)^2 - (3*x^2 - y)^2 = 144 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3020_302075


namespace NUMINAMATH_CALUDE_squares_remaining_l3020_302028

theorem squares_remaining (total : ℕ) (removed_fraction : ℚ) (result : ℕ) : 
  total = 12 →
  removed_fraction = 1/2 * 2/3 →
  result = total - (removed_fraction * total).num →
  result = 8 := by
  sorry

end NUMINAMATH_CALUDE_squares_remaining_l3020_302028


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l3020_302060

/-- Represents the compensation structure and work details of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeMultiplier : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation based on the given compensation structure --/
def calculateTotalCompensation (c : BusDriverCompensation) : ℝ :=
  c.regularRate * c.regularHours + c.regularRate * c.overtimeMultiplier * c.overtimeHours

/-- Theorem stating that the regular rate of $16 per hour satisfies the given conditions --/
theorem bus_driver_regular_rate :
  ∃ (c : BusDriverCompensation),
    c.regularRate = 16 ∧
    c.overtimeMultiplier = 1.75 ∧
    c.regularHours = 40 ∧
    c.overtimeHours = 12 ∧
    c.totalCompensation = 976 ∧
    calculateTotalCompensation c = c.totalCompensation :=
  sorry

end NUMINAMATH_CALUDE_bus_driver_regular_rate_l3020_302060
