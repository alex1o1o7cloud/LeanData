import Mathlib

namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l2366_236637

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := 4518

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := total_jelly - blueberry_jelly

theorem strawberry_jelly_amount : strawberry_jelly = 1792 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l2366_236637


namespace NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_vector_l2366_236670

/-- The equation of a line passing through point P(-1, 2) and parallel to vector {8, -4} --/
theorem line_equation_through_point_parallel_to_vector :
  let P : ℝ × ℝ := (-1, 2)
  let a : ℝ × ℝ := (8, -4)
  let line_eq (x y : ℝ) := y = -1/2 * x + 3/2
  (∀ x y : ℝ, line_eq x y ↔ 
    (∃ t : ℝ, x = P.1 + t * a.1 ∧ y = P.2 + t * a.2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_vector_l2366_236670


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2366_236610

theorem inequality_solution_set (x : ℝ) :
  (x^2 / (x + 1) ≥ 3 / (x - 2) + 9 / 4) ↔ (x < -3/4 ∨ (x > 2 ∧ x < 5)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2366_236610


namespace NUMINAMATH_CALUDE_right_handed_players_count_l2366_236605

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) 
  (left_handed_percentage : ℚ) :
  total_players = 120 →
  throwers = 58 →
  left_handed_percentage = 40 / 100 →
  (total_players - throwers : ℚ) * left_handed_percentage = 24 →
  throwers + (total_players - throwers - 24) = 96 :=
by sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l2366_236605


namespace NUMINAMATH_CALUDE_simon_received_stamps_l2366_236603

/-- The number of stamps Simon received from his friends -/
def stamps_received (initial_stamps current_stamps : ℕ) : ℕ :=
  current_stamps - initial_stamps

/-- Theorem stating that Simon received 27 stamps from his friends -/
theorem simon_received_stamps :
  stamps_received 34 61 = 27 := by
  sorry

end NUMINAMATH_CALUDE_simon_received_stamps_l2366_236603


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l2366_236678

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_3402 :
  largest_perfect_square_factor 3402 = 81 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l2366_236678


namespace NUMINAMATH_CALUDE_solution_set_l2366_236623

theorem solution_set (x : ℝ) :
  (|x^2 - x - 2| + |1/x| = |x^2 - x - 2 + 1/x|) →
  ((-1 ≤ x ∧ x < 0) ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l2366_236623


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2366_236618

/-- Represents a repeating decimal with a given numerator and denominator. -/
def repeating_decimal (numerator denominator : ℕ) : ℚ :=
  numerator / denominator

/-- The sum of the given repeating decimals is equal to 2224/9999. -/
theorem sum_of_repeating_decimals :
  repeating_decimal 2 9 + repeating_decimal 2 99 + repeating_decimal 2 9999 = 2224 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2366_236618


namespace NUMINAMATH_CALUDE_no_root_greater_than_sqrt29_div_2_l2366_236629

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 + 3 = 53
def equation2 (x : ℝ) : Prop := (3*x - 1)^2 = (x - 2)^2
def equation3 (x : ℝ) : Prop := Real.sqrt (x^2 - 9) ≥ Real.sqrt (x - 2)

-- Define a function to check if a number is a root of any equation
def is_root (x : ℝ) : Prop :=
  equation1 x ∨ equation2 x ∨ equation3 x

-- Theorem statement
theorem no_root_greater_than_sqrt29_div_2 :
  ∀ x : ℝ, is_root x → x ≤ Real.sqrt 29 / 2 :=
by sorry

end NUMINAMATH_CALUDE_no_root_greater_than_sqrt29_div_2_l2366_236629


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2366_236684

def selling_price : ℝ := 24000
def cost_price : ℝ := 20000
def potential_profit_percentage : ℝ := 8

theorem discount_percentage_calculation :
  let potential_profit := (potential_profit_percentage / 100) * cost_price
  let selling_price_with_potential_profit := cost_price + potential_profit
  let discount_amount := selling_price - selling_price_with_potential_profit
  let discount_percentage := (discount_amount / selling_price) * 100
  discount_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2366_236684


namespace NUMINAMATH_CALUDE_inequality_condition_l2366_236619

theorem inequality_condition (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x - 1 < 0) → 
  (-3 < m ∧ m < 1) ∧ 
  ¬(∀ m : ℝ, (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x - 1 < 0) → (-3 < m ∧ m < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l2366_236619


namespace NUMINAMATH_CALUDE_dave_files_left_l2366_236606

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 24

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := 2

/-- The difference between the number of files and apps left -/
def file_app_difference : ℕ := 22

theorem dave_files_left :
  files_left = apps_left + file_app_difference :=
by sorry

end NUMINAMATH_CALUDE_dave_files_left_l2366_236606


namespace NUMINAMATH_CALUDE_economics_problem_l2366_236620

def R (x : ℕ) : ℚ := x^2 + 16/x^2 + 40

def C (x : ℕ) : ℚ := 10*x + 40/x

def MC (x : ℕ) : ℚ := C (x+1) - C x

def z (x : ℕ) : ℚ := R x - C x

theorem economics_problem (x : ℕ) (h : 1 ≤ x ∧ x ≤ 10) :
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 10 → R y ≥ 72) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 9 → MC y ≤ 86/9) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 10 → z y ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_economics_problem_l2366_236620


namespace NUMINAMATH_CALUDE_evas_numbers_l2366_236655

theorem evas_numbers (a b : ℕ) (h1 : a > b) 
  (h2 : 10 ≤ a + b) (h3 : a + b < 100)
  (h4 : 10 ≤ a - b) (h5 : a - b < 100)
  (h6 : (a + b) * (a - b) = 645) : 
  a = 29 ∧ b = 14 := by
sorry

end NUMINAMATH_CALUDE_evas_numbers_l2366_236655


namespace NUMINAMATH_CALUDE_carmina_coins_count_l2366_236690

theorem carmina_coins_count :
  ∀ (n d : ℕ),
  (5 * n + 10 * d = 360) →
  (10 * n + 5 * d = 540) →
  n + d = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_carmina_coins_count_l2366_236690


namespace NUMINAMATH_CALUDE_johns_income_increase_johns_income_increase_result_l2366_236681

/-- Calculates the final percentage increase in John's net weekly income --/
theorem johns_income_increase (initial_salary : ℝ) (first_raise : ℝ) (second_raise : ℝ) (third_raise : ℝ) (freelance_income : ℝ) (tax_rate : ℝ) : ℝ :=
  let salary_after_raises := initial_salary * (1 + first_raise) * (1 + second_raise) * (1 + third_raise)
  let total_income := salary_after_raises + freelance_income
  let net_income := total_income * (1 - tax_rate)
  let initial_net_income := initial_salary * (1 - tax_rate)
  let percentage_increase := (net_income - initial_net_income) / initial_net_income * 100
  percentage_increase

/-- The final percentage increase in John's net weekly income is approximately 66.17% --/
theorem johns_income_increase_result :
  abs (johns_income_increase 30 0.1 0.15 0.05 10 0.05 - 66.17) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_johns_income_increase_johns_income_increase_result_l2366_236681


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2366_236615

theorem simplify_fraction_product : (90 : ℚ) / 150 * 35 / 21 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2366_236615


namespace NUMINAMATH_CALUDE_cone_base_area_l2366_236675

/-- Given a cylinder with radius 1 and height 1, when reshaped into a cone
    with the same volume and height, the area of the base of the cone is 3π. -/
theorem cone_base_area (cylinder_radius : ℝ) (cylinder_height : ℝ) 
  (cone_height : ℝ) (cone_base_radius : ℝ) :
  cylinder_radius = 1 →
  cylinder_height = 1 →
  cone_height = cylinder_height →
  π * cylinder_radius^2 * cylinder_height = (1/3) * π * cone_base_radius^2 * cone_height →
  π * cone_base_radius^2 = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_area_l2366_236675


namespace NUMINAMATH_CALUDE_root_sum_product_l2366_236611

theorem root_sum_product (c d : ℝ) : 
  (c^4 - 6*c^3 - 4*c - 1 = 0) → 
  (d^4 - 6*d^3 - 4*d - 1 = 0) → 
  (c ≠ d) →
  (cd + c + d = 4) := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l2366_236611


namespace NUMINAMATH_CALUDE_product_42_sum_9_l2366_236630

theorem product_42_sum_9 (a b c : ℕ+) : 
  a * b * c = 42 → a + b = 9 → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_42_sum_9_l2366_236630


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l2366_236688

theorem largest_x_floor_div : 
  ∀ x : ℝ, (↑⌊x⌋ / x = 7 / 8) → x ≤ 48 / 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l2366_236688


namespace NUMINAMATH_CALUDE_larger_integer_value_l2366_236613

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 5 / 2)
  (h_product : (a : ℕ) * (b : ℕ) = 360) :
  max a b = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2366_236613


namespace NUMINAMATH_CALUDE_angle_inequalities_l2366_236651

theorem angle_inequalities (α β : Real) (h1 : π / 2 < α) (h2 : α < β) (h3 : β < π) :
  (π < α + β ∧ α + β < 2 * π) ∧
  (-π / 2 < α - β ∧ α - β < 0) ∧
  (1 / 2 < α / β ∧ α / β < 1) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequalities_l2366_236651


namespace NUMINAMATH_CALUDE_eldest_child_age_l2366_236616

theorem eldest_child_age (y m e : ℕ) : 
  m = y + 3 →
  e = 3 * y →
  e = y + m + 2 →
  e = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_eldest_child_age_l2366_236616


namespace NUMINAMATH_CALUDE_frank_recycling_cans_l2366_236644

def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

theorem frank_recycling_cans :
  let saturday_bags := 5
  let sunday_bags := 3
  let cans_per_bag := 5
  total_cans saturday_bags sunday_bags cans_per_bag = 40 := by
sorry

end NUMINAMATH_CALUDE_frank_recycling_cans_l2366_236644


namespace NUMINAMATH_CALUDE_function_satisfying_equation_l2366_236639

theorem function_satisfying_equation (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) → (∀ x : ℝ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_equation_l2366_236639


namespace NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achieved_l2366_236668

theorem max_value_sqrt_product (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (Real.sqrt (a * b * c * d) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d))) ≤ 1 :=
by sorry

theorem max_value_achieved (a b c d : Real) :
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) →
  Real.sqrt (a * b * c * d) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achieved_l2366_236668


namespace NUMINAMATH_CALUDE_coffee_consumption_l2366_236632

theorem coffee_consumption (initial_amount : ℝ) (first_fraction : ℝ) (second_fraction : ℝ) (final_amount : ℝ) : 
  initial_amount = 12 →
  first_fraction = 1/4 →
  second_fraction = 1/2 →
  final_amount = 1 →
  initial_amount - (first_fraction * initial_amount + second_fraction * initial_amount + final_amount) = 2 := by
sorry


end NUMINAMATH_CALUDE_coffee_consumption_l2366_236632


namespace NUMINAMATH_CALUDE_octagon_proof_l2366_236662

theorem octagon_proof (n : ℕ) (h : n > 2) : 
  (n * (n - 3)) / 2 = n + 2 * (n - 2) → n = 8 := by
sorry

end NUMINAMATH_CALUDE_octagon_proof_l2366_236662


namespace NUMINAMATH_CALUDE_cow_count_l2366_236604

/-- Represents the number of cows in a farm -/
def num_cows : ℕ := 40

/-- Represents the number of bags of husk consumed by a group of cows in 40 days -/
def group_consumption : ℕ := 40

/-- Represents the number of days it takes one cow to consume one bag of husk -/
def days_per_bag : ℕ := 40

/-- Represents the number of days over which the consumption is measured -/
def total_days : ℕ := 40

theorem cow_count :
  num_cows = group_consumption * days_per_bag / total_days :=
by sorry

end NUMINAMATH_CALUDE_cow_count_l2366_236604


namespace NUMINAMATH_CALUDE_circle_tangency_count_l2366_236694

theorem circle_tangency_count : ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 120 ∧ 120 % r = 0) ∧ 
  (∀ r < 120, 120 % r = 0 → r ∈ S) ∧ 
  Finset.card S = 15 := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_count_l2366_236694


namespace NUMINAMATH_CALUDE_coloring_perfect_square_difference_l2366_236614

/-- A coloring of integers using three colors -/
def Coloring := ℤ → Fin 3

/-- Theorem: For any coloring of integers using three colors, 
    there exist two distinct integers with the same color 
    whose difference is a perfect square -/
theorem coloring_perfect_square_difference (c : Coloring) : 
  ∃ (x y k : ℤ), x ≠ y ∧ c x = c y ∧ y - x = k^2 := by
  sorry

end NUMINAMATH_CALUDE_coloring_perfect_square_difference_l2366_236614


namespace NUMINAMATH_CALUDE_geometric_sequence_floor_frac_l2366_236640

theorem geometric_sequence_floor_frac (x : ℝ) : 
  x ≠ 0 →
  let floor_x := ⌊x⌋
  let frac_x := x - floor_x
  (frac_x * floor_x = floor_x * x) →
  x = (5 + Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_floor_frac_l2366_236640


namespace NUMINAMATH_CALUDE_mission_duration_l2366_236643

theorem mission_duration (planned_duration : ℕ) (overtime_percentage : ℚ) (second_mission : ℕ) : 
  planned_duration = 5 → 
  overtime_percentage = 60 / 100 → 
  second_mission = 3 → 
  planned_duration + (planned_duration * overtime_percentage).floor + second_mission = 11 := by
  sorry

end NUMINAMATH_CALUDE_mission_duration_l2366_236643


namespace NUMINAMATH_CALUDE_campaign_fliers_l2366_236664

theorem campaign_fliers (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (9/10 : ℚ) * (3/4 : ℚ) = 1350 → 
  initial_fliers = 2000 := by
  sorry

end NUMINAMATH_CALUDE_campaign_fliers_l2366_236664


namespace NUMINAMATH_CALUDE_eight_million_factorization_roundness_of_eight_million_l2366_236677

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- 8,000,000 can be expressed as 8 × 10^6 -/
theorem eight_million_factorization : (8000000 : ℕ) = 8 * 10^6 := by sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_eight_million_factorization_roundness_of_eight_million_l2366_236677


namespace NUMINAMATH_CALUDE_negation_of_conjunction_l2366_236658

theorem negation_of_conjunction (p q : Prop) : ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_conjunction_l2366_236658


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2366_236650

theorem quadratic_solution_property (k : ℝ) : 
  (∃ a b : ℝ, a ≠ b ∧ 
   3 * a^2 + 6 * a + k = 0 ∧ 
   3 * b^2 + 6 * b + k = 0 ∧
   |a - b| = (1/2) * (a^2 + b^2)) ↔ 
  (k = 0 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2366_236650


namespace NUMINAMATH_CALUDE_original_decimal_value_l2366_236622

theorem original_decimal_value : 
  ∃ x : ℝ, (x / 100 = x - 1.782) ∧ (x = 1.8) := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_value_l2366_236622


namespace NUMINAMATH_CALUDE_incorrect_calculation_correction_l2366_236683

theorem incorrect_calculation_correction (x : ℝ) : 
  25 * x = 812 → x / 4 = 8.12 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_correction_l2366_236683


namespace NUMINAMATH_CALUDE_product_of_solution_l2366_236631

open Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

-- State the theorem
theorem product_of_solution (x y : ℝ) 
  (eq1 : (floor x : ℝ) + frac y = 1.7)
  (eq2 : frac x + (floor y : ℝ) = 3.6) :
  x * y = 5.92 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solution_l2366_236631


namespace NUMINAMATH_CALUDE_fencing_required_l2366_236685

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 20 → 
  2 * (area / uncovered_side) + uncovered_side = 88 := by
  sorry

#check fencing_required

end NUMINAMATH_CALUDE_fencing_required_l2366_236685


namespace NUMINAMATH_CALUDE_lucille_earnings_lucille_earnings_proof_l2366_236671

/-- Calculates the amount of money Lucille has left after weeding and buying a soda -/
theorem lucille_earnings (cents_per_weed : ℕ) (flower_bed_weeds : ℕ) (vegetable_patch_weeds : ℕ) 
  (grass_weeds : ℕ) (soda_cost : ℕ) : ℕ :=
  let total_weeds := flower_bed_weeds + vegetable_patch_weeds + grass_weeds / 2
  let total_earnings := total_weeds * cents_per_weed
  total_earnings - soda_cost

/-- Proves that Lucille has 147 cents left after weeding and buying a soda -/
theorem lucille_earnings_proof :
  lucille_earnings 6 11 14 32 99 = 147 := by
  sorry

end NUMINAMATH_CALUDE_lucille_earnings_lucille_earnings_proof_l2366_236671


namespace NUMINAMATH_CALUDE_circle_equation_implies_m_lt_5_l2366_236602

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The equation of a circle given by x^2 + y^2 - 4x - 2y + m = 0 --/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + m = 0

/-- Theorem: If x^2 + y^2 - 4x - 2y + m = 0 represents a circle, then m < 5 --/
theorem circle_equation_implies_m_lt_5 :
  ∀ m : ℝ, (∃ c : Circle, ∀ x y : ℝ, circle_equation x y m ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) → m < 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_implies_m_lt_5_l2366_236602


namespace NUMINAMATH_CALUDE_distance_A_to_B_is_300_l2366_236674

/-- The distance between two points A and B, given the following conditions:
    - Monkeys travel from A to B
    - A monkey departs from A every 3 minutes
    - It takes a monkey 12 minutes to travel from A to B
    - A rabbit runs from B to A
    - When the rabbit starts, a monkey has just arrived at B
    - The rabbit encounters 5 monkeys on its way to A
    - The rabbit arrives at A just as another monkey leaves A
    - The rabbit's speed is 3 km/h
-/
def distance_A_to_B : ℝ :=
  let monkey_departure_interval : ℝ := 3 -- minutes
  let monkey_travel_time : ℝ := 12 -- minutes
  let encountered_monkeys : ℕ := 5
  let rabbit_speed : ℝ := 3 * 1000 / 60 -- convert 3 km/h to m/min

  -- Define the distance based on the given conditions
  300 -- meters

theorem distance_A_to_B_is_300 :
  distance_A_to_B = 300 := by sorry

end NUMINAMATH_CALUDE_distance_A_to_B_is_300_l2366_236674


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2366_236627

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 4 * a 6 + a 5 ^ 2 = 50) :
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2366_236627


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2366_236679

def polynomial (x : ℤ) : ℤ := x^3 - 5*x^2 - 8*x + 24

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  (∀ x : ℤ, is_root x ↔ (x = -3 ∨ x = 2 ∨ x = 4)) ∨
  (∀ x : ℤ, ¬is_root x) := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2366_236679


namespace NUMINAMATH_CALUDE_negation_forall_positive_converse_product_zero_symmetry_implies_even_symmetry_shifted_functions_l2366_236648

-- Statement 1
theorem negation_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x, P x) ↔ ∃ x, ¬(P x) :=
sorry

-- Statement 2
theorem converse_product_zero (a b : ℝ) :
  (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (a = 0 ∨ b = 0 → a * b = 0) :=
sorry

-- Statement 3
theorem symmetry_implies_even (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = f (x - 1)) → (∀ x, f x = f (-x)) :=
sorry

-- Statement 4
theorem symmetry_shifted_functions (f : ℝ → ℝ) :
  ∀ x, f (x + 1) = f (-(x - 1)) :=
sorry

end NUMINAMATH_CALUDE_negation_forall_positive_converse_product_zero_symmetry_implies_even_symmetry_shifted_functions_l2366_236648


namespace NUMINAMATH_CALUDE_angle_problem_l2366_236673

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle1 = 80)
  (h4 : angle2 = 100) :
  angle4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l2366_236673


namespace NUMINAMATH_CALUDE_sum_of_coefficients_of_fifth_power_l2366_236608

theorem sum_of_coefficients_of_fifth_power (a b : ℕ) (h : (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2) : a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_of_fifth_power_l2366_236608


namespace NUMINAMATH_CALUDE_james_total_cost_l2366_236621

/-- The total cost of buying dirt bikes, off-road vehicles, and registering them all -/
def total_cost (dirt_bike_count : ℕ) (dirt_bike_price : ℕ) 
                (offroad_count : ℕ) (offroad_price : ℕ) 
                (registration_fee : ℕ) : ℕ :=
  dirt_bike_count * dirt_bike_price + 
  offroad_count * offroad_price + 
  (dirt_bike_count + offroad_count) * registration_fee

/-- Theorem stating the total cost for James' purchase -/
theorem james_total_cost : 
  total_cost 3 150 4 300 25 = 1825 := by
  sorry

end NUMINAMATH_CALUDE_james_total_cost_l2366_236621


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l2366_236626

/-- The total land owned by the farmer in acres -/
def total_land : ℝ := 6000

/-- The proportion of land cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- The proportion of cleared land planted with soybeans -/
def soybean_proportion : ℝ := 0.30

/-- The proportion of cleared land planted with wheat -/
def wheat_proportion : ℝ := 0.60

/-- The amount of cleared land planted with corn in acres -/
def corn_land : ℝ := 540

theorem farmer_land_ownership :
  total_land * cleared_proportion * (1 - soybean_proportion - wheat_proportion) = corn_land :=
by sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l2366_236626


namespace NUMINAMATH_CALUDE_craft_store_solution_l2366_236645

/-- Represents the craft store problem -/
structure CraftStore where
  markedPrice : ℝ
  costPrice : ℝ
  profitPerItem : ℝ
  discountedSales : ℕ
  discountPercentage : ℝ
  reducedPriceSales : ℕ
  priceReduction : ℝ
  dailySales : ℕ
  salesIncrease : ℕ
  priceDecreaseStep : ℝ

/-- The craft store problem statement -/
def craftStoreProblem (cs : CraftStore) : Prop :=
  -- Profit at marked price
  cs.profitPerItem = cs.markedPrice - cs.costPrice
  -- Equal profit for discounted and reduced price sales
  ∧ cs.discountedSales * (cs.markedPrice * cs.discountPercentage - cs.costPrice) =
    cs.reducedPriceSales * (cs.markedPrice - cs.priceReduction - cs.costPrice)
  -- Daily sales at marked price
  ∧ cs.dailySales * (cs.markedPrice - cs.costPrice) =
    (cs.dailySales + cs.salesIncrease) * (cs.markedPrice - cs.priceDecreaseStep - cs.costPrice)

/-- The theorem to be proved -/
theorem craft_store_solution (cs : CraftStore) 
  (h : craftStoreProblem cs) : 
  cs.costPrice = 155 
  ∧ cs.markedPrice = 200 
  ∧ (∃ optimalReduction maxProfit, 
      optimalReduction = 10 
      ∧ maxProfit = 4900 
      ∧ ∀ reduction, 
        cs.dailySales * (cs.markedPrice - reduction - cs.costPrice) 
        + (cs.salesIncrease * reduction / cs.priceDecreaseStep) 
          * (cs.markedPrice - reduction - cs.costPrice) 
        ≤ maxProfit) :=
sorry

end NUMINAMATH_CALUDE_craft_store_solution_l2366_236645


namespace NUMINAMATH_CALUDE_x0_value_l2366_236699

-- Define the function f
def f (x : ℝ) : ℝ := x^5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 5 * x^4

-- Theorem statement
theorem x0_value (x₀ : ℝ) (h : f' x₀ = 20) : x₀ = Real.sqrt 2 ∨ x₀ = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l2366_236699


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2366_236628

/-- The range of m for which the line y = kx + 1 always intersects the ellipse x²/5 + y²/m = 1 at two points -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), (∀ (x y : ℝ), (y = k*x + 1 ∧ x^2/5 + y^2/m = 1) → ∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    (p.1^2/5 + p.2^2/m = 1) ∧ (q.1^2/5 + q.2^2/m = 1) ∧ 
    p.2 = k*p.1 + 1 ∧ q.2 = k*q.1 + 1) ↔ 
  (m > 1 ∧ m < 5) ∨ m > 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2366_236628


namespace NUMINAMATH_CALUDE_impossible_to_equalize_l2366_236672

/-- Represents the circular arrangement of six numbers -/
def CircularArrangement := Fin 6 → ℕ

/-- The initial arrangement of numbers from 1 to 6 -/
def initial_arrangement : CircularArrangement :=
  fun i => i.val + 1

/-- Adds 1 to three consecutive numbers in the arrangement -/
def add_to_consecutive (a : CircularArrangement) (start : Fin 6) : CircularArrangement :=
  fun i => if i = start ∨ i = start.succ ∨ i = start.succ.succ then a i + 1 else a i

/-- Subtracts 1 from three alternating numbers in the arrangement -/
def subtract_from_alternating (a : CircularArrangement) (start : Fin 6) : CircularArrangement :=
  fun i => if i = start ∨ i = start.succ.succ ∨ i = start.succ.succ.succ.succ then a i - 1 else a i

/-- Checks if all numbers in the arrangement are equal -/
def all_equal (a : CircularArrangement) : Prop :=
  ∀ i j : Fin 6, a i = a j

/-- Main theorem: It's impossible to equalize all numbers using the given operations -/
theorem impossible_to_equalize :
  ¬ ∃ (ops : List (CircularArrangement → CircularArrangement)),
    all_equal (ops.foldl (fun acc op => op acc) initial_arrangement) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_equalize_l2366_236672


namespace NUMINAMATH_CALUDE_find_a_min_value_of_sum_l2366_236661

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (I)
theorem find_a :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∃ a : ℝ, a = 2) :=
sorry

-- Theorem for part (II)
theorem min_value_of_sum (x : ℝ) :
  ∃ m : ℝ, m = 5/3 ∧ ∀ x : ℝ, f 2 (3*x) + f 2 (x+3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_find_a_min_value_of_sum_l2366_236661


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_improvement_l2366_236624

/-- Proves the increase in travel distance after modifying a car's fuel efficiency -/
theorem car_fuel_efficiency_improvement (initial_mpg : ℝ) (tank_capacity : ℝ) 
  (fuel_reduction_factor : ℝ) (h1 : initial_mpg = 28) (h2 : tank_capacity = 15) 
  (h3 : fuel_reduction_factor = 0.8) : 
  (initial_mpg / fuel_reduction_factor - initial_mpg) * tank_capacity = 84 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_improvement_l2366_236624


namespace NUMINAMATH_CALUDE_division_problem_l2366_236617

theorem division_problem (a b : ℕ+) (q r : ℤ) 
  (h1 : (a : ℤ) * (b : ℤ) = q * ((a : ℤ) + (b : ℤ)) + r)
  (h2 : 0 ≤ r ∧ r < (a : ℤ) + (b : ℤ))
  (h3 : q^2 + r = 2011) :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ 
  ((a : ℤ) = t ∧ (b : ℤ) = t + 2012 ∨ (a : ℤ) = t + 2012 ∧ (b : ℤ) = t) :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2366_236617


namespace NUMINAMATH_CALUDE_camp_boys_percentage_l2366_236607

theorem camp_boys_percentage (total : ℕ) (added_boys : ℕ) (girl_percentage : ℚ) : 
  total = 60 →
  added_boys = 60 →
  girl_percentage = 5 / 100 →
  (girl_percentage * (total + added_boys) : ℚ) = (total - (9 * total / 10) : ℚ) →
  (9 * total / 10 : ℚ) / total = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_camp_boys_percentage_l2366_236607


namespace NUMINAMATH_CALUDE_family_ages_theorem_l2366_236659

/-- Represents the ages and birth times of a father and his two children -/
structure FamilyAges where
  fatherCurrentAge : ℝ
  sonAgeFiveYearsAgo : ℝ
  daughterAgeFiveYearsAgo : ℝ
  sonCurrentAge : ℝ
  daughterCurrentAge : ℝ
  fatherAgeAtSonBirth : ℝ
  fatherAgeAtDaughterBirth : ℝ

/-- Theorem about the ages in a family based on given conditions -/
theorem family_ages_theorem (f : FamilyAges)
    (h1 : f.fatherCurrentAge = 38)
    (h2 : f.sonAgeFiveYearsAgo = 7)
    (h3 : f.daughterAgeFiveYearsAgo = f.sonAgeFiveYearsAgo / 2)
    (h4 : f.sonCurrentAge = f.sonAgeFiveYearsAgo + 5)
    (h5 : f.daughterCurrentAge = f.daughterAgeFiveYearsAgo + 5)
    (h6 : f.fatherAgeAtSonBirth = f.fatherCurrentAge - f.sonCurrentAge)
    (h7 : f.fatherAgeAtDaughterBirth = f.fatherCurrentAge - f.daughterCurrentAge) :
    f.sonCurrentAge = 12 ∧
    f.daughterCurrentAge = 8.5 ∧
    f.fatherAgeAtSonBirth = 26 ∧
    f.fatherAgeAtDaughterBirth = 29.5 := by
  sorry


end NUMINAMATH_CALUDE_family_ages_theorem_l2366_236659


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2366_236612

theorem no_positive_integer_solutions :
  ¬∃ (x y : ℕ+), x^2 + y^2 = x^4 := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2366_236612


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2366_236689

/-- Represents the selection process for a student survey -/
structure StudentSurvey where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ
  remaining_students : ℕ

/-- The probability of a student being selected in the survey -/
def selection_probability (survey : StudentSurvey) : ℚ :=
  (survey.remaining_students : ℚ) / (survey.total_students : ℚ) *
  (survey.selected_students : ℚ) / (survey.remaining_students : ℚ)

/-- The specific survey described in the problem -/
def school_survey : StudentSurvey :=
  { total_students := 2012
  , selected_students := 50
  , eliminated_students := 12
  , remaining_students := 2000 }

/-- Theorem stating that the selection probability is equal for all students -/
theorem equal_selection_probability :
  ∀ (s1 s2 : StudentSurvey),
    s1 = school_survey → s2 = school_survey →
    selection_probability s1 = selection_probability s2 :=
by sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2366_236689


namespace NUMINAMATH_CALUDE_exponent_problem_l2366_236652

theorem exponent_problem (a m n : ℕ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l2366_236652


namespace NUMINAMATH_CALUDE_question_distribution_l2366_236676

-- Define the types for our problem
def TotalQuestions : ℕ := 100
def CorrectAnswersPerStudent : ℕ := 60

-- Define the number of students
def NumStudents : ℕ := 3

-- Define the types of questions
def EasyQuestions (x : ℕ) : Prop := x ≤ TotalQuestions
def MediumQuestions (y : ℕ) : Prop := y ≤ TotalQuestions
def DifficultQuestions (z : ℕ) : Prop := z ≤ TotalQuestions

-- State the theorem
theorem question_distribution 
  (x y z : ℕ) 
  (h1 : EasyQuestions x)
  (h2 : MediumQuestions y)
  (h3 : DifficultQuestions z)
  (h4 : x + y + z = TotalQuestions)
  (h5 : 3 * x + 2 * y + z = NumStudents * CorrectAnswersPerStudent) :
  z - x = 20 :=
sorry

end NUMINAMATH_CALUDE_question_distribution_l2366_236676


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_count_l2366_236687

/-- The number of gem stone necklaces sold by Faye -/
def gem_stone_necklaces : ℕ := 7

/-- The number of bead necklaces sold by Faye -/
def bead_necklaces : ℕ := 3

/-- The price of each necklace in dollars -/
def necklace_price : ℕ := 7

/-- The total earnings from all necklaces in dollars -/
def total_earnings : ℕ := 70

/-- Theorem stating that the number of gem stone necklaces sold is 7 -/
theorem gem_stone_necklaces_count :
  gem_stone_necklaces = (total_earnings - bead_necklaces * necklace_price) / necklace_price :=
by sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_count_l2366_236687


namespace NUMINAMATH_CALUDE_problem_solution_l2366_236660

def three_digit_number (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem problem_solution (a b : ℕ) : 
  (three_digit_number 5 b 9) - (three_digit_number 2 a 3) = 326 →
  (three_digit_number 5 6 9) % 9 = 0 →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2366_236660


namespace NUMINAMATH_CALUDE_james_pistachio_expenditure_l2366_236638

/-- Calculates the weekly expenditure on pistachios given the cost per can, ounces per can, daily consumption, and days of consumption. -/
def weekly_pistachio_expenditure (cost_per_can : ℚ) (ounces_per_can : ℚ) (ounces_consumed : ℚ) (days_consumed : ℕ) : ℚ :=
  let weekly_consumption := (7 : ℚ) / days_consumed * ounces_consumed
  let cans_needed := (weekly_consumption / ounces_per_can).ceil
  cans_needed * cost_per_can

/-- Theorem stating that James' weekly expenditure on pistachios is $90. -/
theorem james_pistachio_expenditure :
  weekly_pistachio_expenditure 10 5 30 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_james_pistachio_expenditure_l2366_236638


namespace NUMINAMATH_CALUDE_remaining_students_l2366_236666

def number_of_groups : ℕ := 3
def students_per_group : ℕ := 8
def students_who_left : ℕ := 2

theorem remaining_students :
  (number_of_groups * students_per_group) - students_who_left = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_l2366_236666


namespace NUMINAMATH_CALUDE_tent_donation_problem_l2366_236667

theorem tent_donation_problem (total_tents : ℕ) (total_value : ℕ) 
  (cost_A : ℕ) (cost_B : ℕ) :
  total_tents = 300 →
  total_value = 260000 →
  cost_A = 800 →
  cost_B = 1000 →
  ∃ (num_A num_B : ℕ),
    num_A + num_B = total_tents ∧
    num_A * cost_A + num_B * cost_B = total_value ∧
    num_A = 200 ∧
    num_B = 100 :=
by sorry

end NUMINAMATH_CALUDE_tent_donation_problem_l2366_236667


namespace NUMINAMATH_CALUDE_ellies_calculation_l2366_236625

theorem ellies_calculation (x y z : ℝ) 
  (h1 : x - (y + z) = 18) 
  (h2 : x - y - z = 6) : 
  x - y = 12 := by
sorry

end NUMINAMATH_CALUDE_ellies_calculation_l2366_236625


namespace NUMINAMATH_CALUDE_luisa_trip_cost_l2366_236642

/-- Represents a leg of Luisa's trip -/
structure TripLeg where
  distance : Float
  fuelEfficiency : Float
  gasPrice : Float

/-- Calculates the cost of gas for a single leg of the trip -/
def gasCost (leg : TripLeg) : Float :=
  (leg.distance / leg.fuelEfficiency) * leg.gasPrice

/-- Luisa's trip legs -/
def luisaTrip : List TripLeg := [
  { distance := 10, fuelEfficiency := 15, gasPrice := 3.50 },
  { distance := 6,  fuelEfficiency := 12, gasPrice := 3.60 },
  { distance := 7,  fuelEfficiency := 14, gasPrice := 3.40 },
  { distance := 5,  fuelEfficiency := 10, gasPrice := 3.55 },
  { distance := 3,  fuelEfficiency := 13, gasPrice := 3.55 },
  { distance := 9,  fuelEfficiency := 15, gasPrice := 3.50 }
]

/-- Calculates the total cost of Luisa's trip -/
def totalTripCost : Float :=
  luisaTrip.map gasCost |> List.sum

/-- Proves that the total cost of Luisa's trip is approximately $10.53 -/
theorem luisa_trip_cost : 
  (totalTripCost - 10.53).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_luisa_trip_cost_l2366_236642


namespace NUMINAMATH_CALUDE_discount_order_difference_l2366_236609

/-- Calculates the final price after applying discounts and tax -/
def final_price (initial_price : ℚ) (flat_discount : ℚ) (percent_discount : ℚ) (tax_rate : ℚ) (flat_first : Bool) : ℚ :=
  let price_after_flat := initial_price - flat_discount
  let price_after_percent := initial_price * (1 - percent_discount)
  let discounted_price := if flat_first then
    price_after_flat * (1 - percent_discount)
  else
    price_after_percent - flat_discount
  discounted_price * (1 + tax_rate)

/-- The difference in final price between two discount application orders -/
def price_difference (initial_price flat_discount percent_discount tax_rate : ℚ) : ℚ :=
  (final_price initial_price flat_discount percent_discount tax_rate true) -
  (final_price initial_price flat_discount percent_discount tax_rate false)

theorem discount_order_difference :
  price_difference 30 5 (25/100) (10/100) = 1375/1000 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_l2366_236609


namespace NUMINAMATH_CALUDE_greatest_t_value_l2366_236636

theorem greatest_t_value : 
  let f : ℝ → ℝ := λ t => (t^2 - t - 90) / (t - 8)
  let g : ℝ → ℝ := λ t => 6 / (t + 7)
  ∃ t_max : ℝ, t_max = -1 ∧ 
    (∀ t : ℝ, t ≠ 8 ∧ t ≠ -7 → f t = g t → t ≤ t_max) :=
by sorry

end NUMINAMATH_CALUDE_greatest_t_value_l2366_236636


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l2366_236663

theorem three_digit_number_theorem (x y m n : ℕ) : 
  (100 * y + x = m * (x + y)) →
  (100 * x + y = n * (x + y)) →
  n = 101 - m := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l2366_236663


namespace NUMINAMATH_CALUDE_unique_twisty_divisible_by_12_l2366_236657

/-- A function that checks if a number is twisty -/
def is_twisty (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
  n = a * 10000 + b * 1000 + a * 100 + b * 10 + a

/-- A function that checks if a number is five digits long -/
def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

/-- The main theorem -/
theorem unique_twisty_divisible_by_12 : 
  ∃! (n : ℕ), is_twisty n ∧ is_five_digit n ∧ n % 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_twisty_divisible_by_12_l2366_236657


namespace NUMINAMATH_CALUDE_pink_crayons_l2366_236641

def crayon_box (total red blue green yellow pink purple : ℕ) : Prop :=
  total = 48 ∧
  red = 12 ∧
  blue = 8 ∧
  green = (3 * blue) / 4 ∧
  yellow = (15 * total) / 100 ∧
  pink = purple ∧
  total = red + blue + green + yellow + pink + purple

theorem pink_crayons (total red blue green yellow pink purple : ℕ) :
  crayon_box total red blue green yellow pink purple → pink = 8 := by
  sorry

end NUMINAMATH_CALUDE_pink_crayons_l2366_236641


namespace NUMINAMATH_CALUDE_cubic_sum_values_l2366_236634

def M (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, b, c],
    ![b, c, a],
    ![c, a, b]]

theorem cubic_sum_values (a b c : ℂ) :
  M a b c ^ 2 = 1 →
  a * b * c = -1 →
  (a^3 + b^3 + c^3 = -2) ∨ (a^3 + b^3 + c^3 = -4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_values_l2366_236634


namespace NUMINAMATH_CALUDE_passing_marks_l2366_236600

/-- Given an exam with total marks T and passing marks P, prove that P = 240 -/
theorem passing_marks (T : ℝ) (P : ℝ) : 
  (0.30 * T = P - 60) →  -- Condition 1: 30% fails by 60 marks
  (0.45 * T = P + 30) →  -- Condition 2: 45% passes by 30 marks
  P = 240 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_l2366_236600


namespace NUMINAMATH_CALUDE_parents_john_age_ratio_l2366_236647

/-- Given information about Mark, John, and their parents' ages, prove the ratio of parents' age to John's age -/
theorem parents_john_age_ratio :
  ∀ (mark_age john_age parents_age : ℕ),
    mark_age = 18 →
    john_age = mark_age - 10 →
    parents_age = 22 + mark_age →
    parents_age / john_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_parents_john_age_ratio_l2366_236647


namespace NUMINAMATH_CALUDE_shortest_chord_length_l2366_236601

/-- Circle C with center (1,2) and radius 5 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l passing through point M(3,1) -/
def line_l (x y m : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- Point M(3,1) -/
def point_M : ℝ × ℝ := (3, 1)

/-- M is inside circle C -/
axiom M_inside_C : circle_C point_M.1 point_M.2

/-- The shortest chord theorem -/
theorem shortest_chord_length :
  ∃ (m : ℝ), line_l point_M.1 point_M.2 m →
  (∀ (x y : ℝ), line_l x y m → circle_C x y →
  ∃ (x' y' : ℝ), line_l x' y' m ∧ circle_C x' y' ∧
  ((x - x')^2 + (y - y')^2)^(1/2) ≤ 4 * 5^(1/2)) ∧
  (∃ (x y x' y' : ℝ), line_l x y m ∧ circle_C x y ∧
  line_l x' y' m ∧ circle_C x' y' ∧
  ((x - x')^2 + (y - y')^2)^(1/2) = 4 * 5^(1/2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_length_l2366_236601


namespace NUMINAMATH_CALUDE_M_greater_than_N_l2366_236698

theorem M_greater_than_N (a : ℝ) : 2*a*(a-2) + 7 > (a-2)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l2366_236698


namespace NUMINAMATH_CALUDE_car_travel_distance_l2366_236635

theorem car_travel_distance (speed1 speed2 total_distance average_speed : ℝ) 
  (h1 : speed1 = 75)
  (h2 : speed2 = 80)
  (h3 : total_distance = 320)
  (h4 : average_speed = 77.4193548387097)
  (h5 : total_distance = 2 * (total_distance / 2)) : 
  total_distance / 2 = 160 := by
  sorry

#check car_travel_distance

end NUMINAMATH_CALUDE_car_travel_distance_l2366_236635


namespace NUMINAMATH_CALUDE_pizza_slices_left_l2366_236692

theorem pizza_slices_left (total_slices : ℕ) (eaten_fraction : ℚ) (h1 : total_slices = 16) (h2 : eaten_fraction = 3/4) : 
  total_slices * (1 - eaten_fraction) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l2366_236692


namespace NUMINAMATH_CALUDE_total_money_l2366_236693

/-- Given that A and C together have 400, B and C together have 750, and C has 250,
    prove that the total amount of money A, B, and C have between them is 900. -/
theorem total_money (a b c : ℕ) 
  (h1 : a + c = 400)
  (h2 : b + c = 750)
  (h3 : c = 250) :
  a + b + c = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2366_236693


namespace NUMINAMATH_CALUDE_exists_zero_term_l2366_236691

def recursion (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 
    (a n ≥ b n → a (n + 1) = a n - b n ∧ b (n + 1) = 2 * b n) ∧
    (a n < b n → a (n + 1) = 2 * a n ∧ b (n + 1) = b n - a n)

theorem exists_zero_term (a b : ℕ → ℕ) :
  recursion a b →
  (∃ k : ℕ, a k = 0) ↔
  (∃ m : ℕ, m > 0 ∧ (a 1 + b 1) / Nat.gcd (a 1) (b 1) = 2^m) :=
sorry

end NUMINAMATH_CALUDE_exists_zero_term_l2366_236691


namespace NUMINAMATH_CALUDE_mike_yard_sale_books_l2366_236682

/-- Calculates the number of books bought at a yard sale -/
def books_bought_at_yard_sale (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem: The number of books Mike bought at the yard sale is 21 -/
theorem mike_yard_sale_books :
  let initial_books : ℕ := 35
  let final_books : ℕ := 56
  books_bought_at_yard_sale initial_books final_books = 21 := by
  sorry

end NUMINAMATH_CALUDE_mike_yard_sale_books_l2366_236682


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2366_236696

/-- The total surface area of a cylinder with height 8 and radius 5 is 130π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 8
  let r : ℝ := 5
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 130 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2366_236696


namespace NUMINAMATH_CALUDE_segment_ratio_l2366_236653

/-- Given four points A, B, C, D on a line segment, 
    if AB : BC = 1 : 2 and BC : CD = 8 : 5, 
    then AB : BD = 4 : 13 -/
theorem segment_ratio (A B C D : ℝ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (ratio1 : (B - A) / (C - B) = 1 / 2)
  (ratio2 : (C - B) / (D - C) = 8 / 5) :
  (B - A) / (D - B) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l2366_236653


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2366_236697

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2366_236697


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2366_236646

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b - 4*a*b = 0) :
  ∀ x y, x > 0 → y > 0 → x + 2*y - 4*x*y = 0 → a + 8*b ≤ x + 8*y ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b - 4*a*b = 0 ∧ a + 8*b = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2366_236646


namespace NUMINAMATH_CALUDE_cubic_function_property_l2366_236669

/-- Given a cubic function f(x) = ax³ + bx - 2 where f(2014) = 3, prove that f(-2014) = -7 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 2
  (f 2014 = 3) → (f (-2014) = -7) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2366_236669


namespace NUMINAMATH_CALUDE_selected_is_sample_size_l2366_236665

/-- Represents a statistical study -/
structure StatisticalStudy where
  population_size : ℕ
  selected_size : ℕ
  selected_size_le_population : selected_size ≤ population_size

/-- Definition of sample size -/
def sample_size (study : StatisticalStudy) : ℕ := study.selected_size

theorem selected_is_sample_size (study : StatisticalStudy) 
  (h1 : study.population_size = 3000) 
  (h2 : study.selected_size = 100) : 
  sample_size study = study.selected_size :=
by
  sorry

#check selected_is_sample_size

end NUMINAMATH_CALUDE_selected_is_sample_size_l2366_236665


namespace NUMINAMATH_CALUDE_circle_not_through_origin_l2366_236680

-- Define the curve
def curve (x y : ℝ) : Prop := 2 * x^2 - y^2 = 5

-- Define the line passing through (0, 2)
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the intersection points A and B
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    y₁ = line k x₁ ∧ y₂ = line k x₂ ∧
    x₁ ≠ x₂

-- Define the circle with diameter AB
def circle_AB (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) = 0

-- Theorem statement
theorem circle_not_through_origin (k : ℝ) :
  intersection_points k →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    y₁ = line k x₁ ∧ y₂ = line k x₂ ∧
    x₁ ≠ x₂ ∧
    ¬(circle_AB x₁ y₁ x₂ y₂ 0 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_not_through_origin_l2366_236680


namespace NUMINAMATH_CALUDE_triple_coverage_theorem_l2366_236695

/-- Represents a rectangular rug with given dimensions -/
structure Rug where
  width : ℝ
  length : ℝ

/-- Represents the arrangement of rugs in the auditorium -/
structure AuditoriumArrangement where
  auditorium_size : ℝ
  rug1 : Rug
  rug2 : Rug
  rug3 : Rug

/-- Calculates the area covered by all three rugs simultaneously -/
def triple_coverage_area (arrangement : AuditoriumArrangement) : ℝ :=
  sorry

/-- The specific arrangement in the problem -/
def problem_arrangement : AuditoriumArrangement :=
  { auditorium_size := 10
  , rug1 := { width := 6, length := 8 }
  , rug2 := { width := 6, length := 6 }
  , rug3 := { width := 5, length := 7 }
  }

theorem triple_coverage_theorem :
  triple_coverage_area problem_arrangement = 6 := by sorry

end NUMINAMATH_CALUDE_triple_coverage_theorem_l2366_236695


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l2366_236686

theorem power_of_product_equals_product_of_powers (a : ℝ) : (2 * a^3)^3 = 8 * a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l2366_236686


namespace NUMINAMATH_CALUDE_sqrt_five_squared_l2366_236654

theorem sqrt_five_squared : (Real.sqrt 5)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_l2366_236654


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2366_236633

theorem abs_inequality_equivalence (x : ℝ) : 
  |5 - 2*x| < 3 ↔ 1 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2366_236633


namespace NUMINAMATH_CALUDE_expression_evaluation_l2366_236656

theorem expression_evaluation :
  let a : ℤ := -2
  let b : ℤ := 4
  (-(-3*a)^2 + 6*a*b) - (a^2 + 3*(a - 2*a*b)) = 14 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2366_236656


namespace NUMINAMATH_CALUDE_ribbon_left_l2366_236649

theorem ribbon_left (num_gifts : ℕ) (ribbon_per_gift : ℚ) (total_ribbon : ℚ) :
  num_gifts = 8 →
  ribbon_per_gift = 3/2 →
  total_ribbon = 15 →
  total_ribbon - (num_gifts : ℚ) * ribbon_per_gift = 3 := by
sorry

end NUMINAMATH_CALUDE_ribbon_left_l2366_236649
