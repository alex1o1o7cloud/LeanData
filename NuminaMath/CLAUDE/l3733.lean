import Mathlib

namespace two_solutions_l3733_373372

/-- The number of positive integers satisfying the equation -/
def solution_count : ℕ := 2

/-- Predicate for integers satisfying the equation -/
def satisfies_equation (n : ℕ) : Prop :=
  (n + 800) / 80 = ⌊Real.sqrt n⌋

/-- Theorem stating that exactly two positive integers satisfy the equation -/
theorem two_solutions :
  (∃ (a b : ℕ), a ≠ b ∧ satisfies_equation a ∧ satisfies_equation b) ∧
  (∀ (n : ℕ), satisfies_equation n → n = a ∨ n = b) :=
sorry

end two_solutions_l3733_373372


namespace xy_value_l3733_373376

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x+y) = 16)
  (h2 : (27 : ℝ)^(x+y) / (9 : ℝ)^(4*y) = 729) : 
  x * y = 48 := by
sorry

end xy_value_l3733_373376


namespace mandy_toys_count_mandy_toys_count_proof_l3733_373342

theorem mandy_toys_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mandy anna amanda peter =>
    anna = 3 * mandy ∧
    amanda = anna + 2 ∧
    peter = 2 * anna ∧
    mandy + anna + amanda + peter = 278 →
    mandy = 21

-- The proof is omitted
theorem mandy_toys_count_proof : mandy_toys_count 21 63 65 126 := by
  sorry

end mandy_toys_count_mandy_toys_count_proof_l3733_373342


namespace unique_quadratic_root_l3733_373362

theorem unique_quadratic_root (a : ℝ) : 
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ (a = 1 ∨ a = 5/3) := by
  sorry

end unique_quadratic_root_l3733_373362


namespace range_of_negative_values_l3733_373304

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_decreasing : ∀ x y, x < y → y < 0 → f x > f y)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
sorry

end range_of_negative_values_l3733_373304


namespace complex_equation_solution_l3733_373341

theorem complex_equation_solution (z : ℂ) :
  (1 + 2*I) * z = 4 + 3*I → z = 2 - I :=
by sorry

end complex_equation_solution_l3733_373341


namespace boat_speed_in_still_water_l3733_373332

/-- Given a boat that travels 32 km along a stream and 12 km against the same stream
    in one hour each, its speed in still water is 22 km/hr. -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 32) 
  (h2 : against_stream = 12) : 
  (along_stream + against_stream) / 2 = 22 := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l3733_373332


namespace solve_equation_l3733_373331

theorem solve_equation (x : ℝ) : 3 * x = (20 - x) + 20 → x = 10 := by
  sorry

end solve_equation_l3733_373331


namespace weekdays_wearing_one_shirt_to_school_l3733_373328

def shirts_for_two_weeks : ℕ := 22

def after_school_club_days_per_week : ℕ := 3
def saturdays_per_week : ℕ := 1
def sundays_per_week : ℕ := 1
def weeks : ℕ := 2

def shirts_for_after_school_club : ℕ := after_school_club_days_per_week * weeks
def shirts_for_saturdays : ℕ := saturdays_per_week * weeks
def shirts_for_sundays : ℕ := 2 * sundays_per_week * weeks

def shirts_for_other_activities : ℕ := 
  shirts_for_after_school_club + shirts_for_saturdays + shirts_for_sundays

theorem weekdays_wearing_one_shirt_to_school : 
  (shirts_for_two_weeks - shirts_for_other_activities) / weeks = 5 := by
  sorry

end weekdays_wearing_one_shirt_to_school_l3733_373328


namespace cloth_sale_meters_l3733_373360

/-- Proves that the number of meters of cloth sold is 85 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8925)
    (h2 : profit_per_meter = 20)
    (h3 : cost_price_per_meter = 85) :
    (total_selling_price : ℚ) / ((cost_price_per_meter : ℚ) + (profit_per_meter : ℚ)) = 85 := by
  sorry

end cloth_sale_meters_l3733_373360


namespace locus_is_circle_l3733_373383

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  s : ℝ
  a : Point
  b : Point
  c : Point

/-- The sum of squares of distances from a point to the vertices of a triangle -/
def sumOfSquaredDistances (p : Point) (t : IsoscelesRightTriangle) : ℝ :=
  (p.x - t.a.x)^2 + (p.y - t.a.y)^2 +
  (p.x - t.b.x)^2 + (p.y - t.b.y)^2 +
  (p.x - t.c.x)^2 + (p.y - t.c.y)^2

/-- The locus of points P such that the sum of squares of distances from P to the vertices is less than 2s^2 -/
def locus (t : IsoscelesRightTriangle) : Set Point :=
  {p : Point | sumOfSquaredDistances p t < 2 * t.s^2}

theorem locus_is_circle (t : IsoscelesRightTriangle) :
  locus t = {p : Point | (p.x - t.s/3)^2 + (p.y - t.s/3)^2 < (2*t.s/3)^2} :=
sorry

end locus_is_circle_l3733_373383


namespace max_stamps_purchasable_l3733_373348

theorem max_stamps_purchasable (stamp_price : ℕ) (discounted_price : ℕ) (budget : ℕ) :
  stamp_price = 50 →
  discounted_price = 45 →
  budget = 5000 →
  (∀ n : ℕ, n ≤ 50 → n * stamp_price ≤ budget) →
  (∀ n : ℕ, n > 50 → 50 * stamp_price + (n - 50) * discounted_price ≤ budget) →
  (∃ n : ℕ, n = 105 ∧
    (∀ m : ℕ, m > n → 
      (m ≤ 50 → m * stamp_price > budget) ∧
      (m > 50 → 50 * stamp_price + (m - 50) * discounted_price > budget))) :=
by sorry

end max_stamps_purchasable_l3733_373348


namespace max_sum_under_constraint_l3733_373311

theorem max_sum_under_constraint (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  16 * x * y * z = (x + y)^2 * (x + z)^2 →
  x + y + z ≤ 4 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    16 * a * b * c = (a + b)^2 * (a + c)^2 ∧ a + b + c = 4 := by
  sorry

end max_sum_under_constraint_l3733_373311


namespace outer_prism_width_is_ten_l3733_373398

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The inner prism dimensions satisfy the given conditions -/
def inner_prism_conditions (d : PrismDimensions) : Prop :=
  d.length * d.width * d.height = 128 ∧
  d.width = 2 * d.length ∧
  d.width = 2 * d.height

/-- The outer prism dimensions are one unit larger in each dimension -/
def outer_prism_dimensions (d : PrismDimensions) : PrismDimensions :=
  { length := d.length + 2
  , width := d.width + 2
  , height := d.height + 2 }

/-- The width of the outer prism is 10 inches -/
theorem outer_prism_width_is_ten (d : PrismDimensions) 
  (h : inner_prism_conditions d) : 
  (outer_prism_dimensions d).width = 10 := by
  sorry

end outer_prism_width_is_ten_l3733_373398


namespace absolute_value_inequality_solution_set_l3733_373377

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| ≥ 1} = {x : ℝ | x ≤ -2 ∨ x ≥ 0} := by sorry

end absolute_value_inequality_solution_set_l3733_373377


namespace car_speed_problem_l3733_373354

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 35

/-- The time it takes Car A to catch up with Car B when traveling at 50 km/h -/
def time_1 : ℝ := 6

/-- The time it takes Car A to catch up with Car B when traveling at 80 km/h -/
def time_2 : ℝ := 2

/-- The speed of Car A in the first scenario (km/h) -/
def speed_A_1 : ℝ := 50

/-- The speed of Car A in the second scenario (km/h) -/
def speed_A_2 : ℝ := 80

theorem car_speed_problem :
  speed_B * time_1 = speed_A_1 * time_1 - (time_1 - time_2) * speed_B ∧
  speed_B * time_2 = speed_A_2 * time_2 - (time_1 - time_2) * speed_B :=
by
  sorry

#check car_speed_problem

end car_speed_problem_l3733_373354


namespace min_distance_a_c_l3733_373381

/-- Given vectors a and b in ℝ² satisfying the specified conditions,
    prove that the minimum distance between a and c is (√7 - √2) / 2 -/
theorem min_distance_a_c (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = 1)
  (h3 : a • b = 1)
  : (∀ c : ℝ × ℝ, (a - 2 • c) • (b - c) = 0 → 
    ‖a - c‖ ≥ (Real.sqrt 7 - Real.sqrt 2) / 2) ∧ 
    (∃ c : ℝ × ℝ, (a - 2 • c) • (b - c) = 0 ∧ 
    ‖a - c‖ = (Real.sqrt 7 - Real.sqrt 2) / 2) := by
  sorry


end min_distance_a_c_l3733_373381


namespace hybrid_one_headlight_percentage_l3733_373343

theorem hybrid_one_headlight_percentage
  (total_cars : ℕ)
  (hybrid_percentage : ℚ)
  (full_headlight_hybrids : ℕ)
  (h1 : total_cars = 600)
  (h2 : hybrid_percentage = 60 / 100)
  (h3 : full_headlight_hybrids = 216) :
  let total_hybrids := (total_cars : ℚ) * hybrid_percentage
  let one_headlight_hybrids := total_hybrids - (full_headlight_hybrids : ℚ)
  one_headlight_hybrids / total_hybrids = 40 / 100 := by
sorry

end hybrid_one_headlight_percentage_l3733_373343


namespace geometric_progression_condition_l3733_373369

/-- 
Given a, b, c are real numbers and k, n, p are integers,
if a, b, c are the k-th, n-th, and p-th terms respectively of a geometric progression,
then (a/b)^(k-p) = (a/c)^(k-n)
-/
theorem geometric_progression_condition 
  (a b c : ℝ) (k n p : ℤ) 
  (hk : k ≠ n) (hn : n ≠ p) (hp : p ≠ k)
  (hgp : ∃ (r : ℝ), r ≠ 0 ∧ b = a * r^(n-k) ∧ c = a * r^(p-k)) :
  (a/b)^(k-p) = (a/c)^(k-n) :=
sorry

end geometric_progression_condition_l3733_373369


namespace sequence_a_bounds_l3733_373359

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_a_bounds (n : ℕ) : 1 - 1 / (n + 3) < sequence_a (n + 1) ∧ sequence_a (n + 1) < n + 1 := by
  sorry

end sequence_a_bounds_l3733_373359


namespace strip_length_is_one_million_l3733_373327

/-- The number of meters in a kilometer -/
def meters_per_km : ℕ := 1000

/-- The number of cubic meters in a cubic kilometer -/
def cubic_meters_in_cubic_km : ℕ := meters_per_km ^ 3

/-- The length of the strip in kilometers -/
def strip_length_km : ℕ := cubic_meters_in_cubic_km / meters_per_km

theorem strip_length_is_one_million :
  strip_length_km = 1000000 := by
  sorry


end strip_length_is_one_million_l3733_373327


namespace lucy_groceries_l3733_373312

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 4

/-- The number of packs of cake Lucy bought -/
def cake : ℕ := 22

/-- The number of packs of chocolate Lucy bought -/
def chocolate : ℕ := 16

/-- The total number of packs of groceries Lucy bought -/
def total_groceries : ℕ := cookies + cake + chocolate

theorem lucy_groceries : total_groceries = 42 := by
  sorry

end lucy_groceries_l3733_373312


namespace book_selection_l3733_373374

theorem book_selection (n m k : ℕ) (hn : n = 8) (hm : m = 5) (hk : k = 1) :
  (Nat.choose (n - k) (m - k)) = 35 := by
  sorry

end book_selection_l3733_373374


namespace addition_of_decimals_l3733_373316

theorem addition_of_decimals : 7.56 + 4.29 = 11.85 := by
  sorry

end addition_of_decimals_l3733_373316


namespace max_quadratic_expression_l3733_373364

theorem max_quadratic_expression :
  ∃ (M : ℝ), M = 67 ∧ ∀ (p : ℝ), -3 * p^2 + 30 * p - 8 ≤ M :=
sorry

end max_quadratic_expression_l3733_373364


namespace equality_condition_l3733_373352

theorem equality_condition (a b c : ℝ) :
  2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c) ↔ a = 0 ∨ a + 2 * b + 1.5 * c = 0 := by
  sorry

end equality_condition_l3733_373352


namespace new_girl_weight_l3733_373334

theorem new_girl_weight (initial_total_weight : ℝ) : 
  let initial_average := initial_total_weight / 10
  let new_average := initial_average + 5
  let new_total_weight := new_average * 10
  new_total_weight = initial_total_weight - 50 + 100 := by sorry

end new_girl_weight_l3733_373334


namespace special_parallelogram_segment_lengths_l3733_373347

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ
  h_side1 : side1 = 5
  h_side2 : side2 = 13
  h_angle : angle = Real.arccos (6/13)

/-- The property of being divided into four equal quadrilaterals by perpendicular lines -/
def hasFourEqualQuadrilaterals (p : SpecialParallelogram) : Prop := sorry

/-- The lengths of segments created by the perpendicular lines -/
def segmentLengths (p : SpecialParallelogram) : ℝ × ℝ := sorry

/-- Theorem statement -/
theorem special_parallelogram_segment_lengths (p : SpecialParallelogram) 
  (h : hasFourEqualQuadrilaterals p) : 
  segmentLengths p = (3, 39/5) := by sorry


end special_parallelogram_segment_lengths_l3733_373347


namespace game_savings_ratio_l3733_373355

theorem game_savings_ratio (game_cost : ℝ) (tax_rate : ℝ) (weekly_allowance : ℝ) (weeks_to_save : ℕ) :
  game_cost = 50 →
  tax_rate = 0.1 →
  weekly_allowance = 10 →
  weeks_to_save = 11 →
  (weekly_allowance / weekly_allowance : ℝ) = 1 := by
  sorry

end game_savings_ratio_l3733_373355


namespace reciprocal_of_negative_three_l3733_373329

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by sorry

end reciprocal_of_negative_three_l3733_373329


namespace circle_center_first_quadrant_l3733_373340

theorem circle_center_first_quadrant (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x + (2*m - 2)*y + 2*m^2 = 0 →
    ∃ r : ℝ, (x - m)^2 + (y - (1 - m))^2 = r^2) →
  (m > 0 ∧ 1 - m > 0) →
  0 < m ∧ m < 1 :=
by sorry

end circle_center_first_quadrant_l3733_373340


namespace son_age_proof_l3733_373325

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end son_age_proof_l3733_373325


namespace complex_equation_solution_l3733_373301

theorem complex_equation_solution (z : ℂ) (p q : ℝ) : 
  (∃ b : ℝ, z = Complex.I * b) →  -- z is purely imaginary
  (∃ c : ℝ, (z + 2)^2 + Complex.I * 8 = Complex.I * c) →  -- (z+2)^2 + 8i is purely imaginary
  2 * (z - 1)^2 + p * (z - 1) + q = 0 →  -- z-1 is a root of 2x^2 + px + q = 0
  z = Complex.I * 2 ∧ p = 4 ∧ q = 10 := by
sorry

end complex_equation_solution_l3733_373301


namespace rice_distribution_theorem_l3733_373317

/-- Represents the amount of rice in a container after dividing the total rice equally -/
def rice_per_container (total_pounds : ℚ) (num_containers : ℕ) : ℚ :=
  (total_pounds * 16) / num_containers

/-- Theorem stating that dividing 49 and 3/4 pounds of rice equally among 7 containers 
    results in approximately 114 ounces of rice per container -/
theorem rice_distribution_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |rice_per_container (49 + 3/4) 7 - 114| < ε :=
sorry

end rice_distribution_theorem_l3733_373317


namespace third_median_length_special_triangle_third_median_l3733_373333

/-- A triangle with specific median lengths and area -/
structure SpecialTriangle where
  -- Two medians of the triangle
  median1 : ℝ
  median2 : ℝ
  -- Area of the triangle
  area : ℝ
  -- Conditions on the medians and area
  median1_length : median1 = 4
  median2_length : median2 = 8
  triangle_area : area = 4 * Real.sqrt 15

/-- The third median of the special triangle has length 7 -/
theorem third_median_length (t : SpecialTriangle) : ℝ :=
  7

/-- The theorem stating that the third median of the special triangle has length 7 -/
theorem special_triangle_third_median (t : SpecialTriangle) : 
  third_median_length t = 7 := by
  sorry

end third_median_length_special_triangle_third_median_l3733_373333


namespace james_has_more_balloons_l3733_373330

/-- The number of balloons James has -/
def james_balloons : ℕ := 232

/-- The number of balloons Amy has -/
def amy_balloons : ℕ := 101

/-- The difference in the number of balloons between James and Amy -/
def balloon_difference : ℕ := james_balloons - amy_balloons

theorem james_has_more_balloons : balloon_difference = 131 := by
  sorry

end james_has_more_balloons_l3733_373330


namespace smaller_number_proof_l3733_373397

theorem smaller_number_proof (L S : ℕ) (h1 : L - S = 2395) (h2 : L = 6 * S + 15) : S = 476 := by
  sorry

end smaller_number_proof_l3733_373397


namespace sunzi_problem_l3733_373319

theorem sunzi_problem (x y : ℚ) : 
  (x + (1/2) * y = 48 ∧ y + (2/3) * x = 48) ↔ 
  (x + (1/2) * y = 48 ∧ y + (2/3) * x = 48) :=
by sorry

end sunzi_problem_l3733_373319


namespace initial_fish_l3733_373318

def fish_bought : ℝ := 280.0
def fish_now : ℕ := 492

theorem initial_fish : ℕ := by
  sorry

#check initial_fish = 212

end initial_fish_l3733_373318


namespace max_salary_theorem_l3733_373386

/-- Represents a basketball team in a semipro league. -/
structure BasketballTeam where
  num_players : ℕ
  min_salary : ℕ
  max_total_salary : ℕ

/-- Calculates the maximum possible salary for a single player in a basketball team. -/
def max_single_player_salary (team : BasketballTeam) : ℕ :=
  team.max_total_salary - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player in the given conditions. -/
theorem max_salary_theorem (team : BasketballTeam) 
    (h1 : team.num_players = 21)
    (h2 : team.min_salary = 20000)
    (h3 : team.max_total_salary = 900000) : 
  max_single_player_salary team = 500000 := by
  sorry

#eval max_single_player_salary { num_players := 21, min_salary := 20000, max_total_salary := 900000 }

end max_salary_theorem_l3733_373386


namespace max_value_of_function_l3733_373338

theorem max_value_of_function (x : ℝ) : 
  (∀ x, -1 ≤ Real.cos x ∧ Real.cos x ≤ 1) → 
  ∃ y_max : ℝ, y_max = 4 ∧ ∀ x, 3 - Real.cos (x / 2) ≤ y_max := by
sorry

end max_value_of_function_l3733_373338


namespace computer_knowledge_competition_compositions_l3733_373378

theorem computer_knowledge_competition_compositions :
  let n : ℕ := 8  -- number of people in each group
  let k : ℕ := 4  -- number of people to be selected from each group
  Nat.choose n k * Nat.choose n k = 4900 := by
  sorry

end computer_knowledge_competition_compositions_l3733_373378


namespace angle_A_is_pi_over_two_l3733_373346

/-- 
Given a triangle ABC where:
- The sides opposite to angles A, B, C are a, b, c respectively
- b = √5
- c = 2
- cos B = 2/3

Prove that the measure of angle A is π/2
-/
theorem angle_A_is_pi_over_two (a b c : ℝ) (A B C : ℝ) : 
  b = Real.sqrt 5 → 
  c = 2 → 
  Real.cos B = 2/3 → 
  A + B + C = π → 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a * Real.sin B = b * Real.sin A → 
  b^2 = a^2 + c^2 - 2*a*c * Real.cos B → 
  A = π/2 := by
sorry

end angle_A_is_pi_over_two_l3733_373346


namespace olivia_car_rental_cost_l3733_373337

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Proves that Olivia's car rental costs $215 given the specified conditions. -/
theorem olivia_car_rental_cost :
  carRentalCost 30 (1/4) 3 500 = 215 := by
  sorry

end olivia_car_rental_cost_l3733_373337


namespace elizabeth_pen_purchase_l3733_373371

/-- Calculates the number of pens Elizabeth can buy given her budget and pencil purchase. -/
theorem elizabeth_pen_purchase 
  (total_budget : ℚ)
  (pencil_cost : ℚ)
  (pen_cost : ℚ)
  (pencil_count : ℕ)
  (h1 : total_budget = 20)
  (h2 : pencil_cost = 8/5)  -- $1.60 expressed as a rational number
  (h3 : pen_cost = 2)
  (h4 : pencil_count = 5) :
  (total_budget - pencil_cost * ↑pencil_count) / pen_cost = 6 := by
sorry

end elizabeth_pen_purchase_l3733_373371


namespace container_volume_ratio_l3733_373302

theorem container_volume_ratio :
  ∀ (A B C : ℝ),
  A > 0 → B > 0 → C > 0 →
  (3/4 * A - 5/8 * B = 7/8 * C - 1/2 * C) →
  (A / C = 4/5) :=
by
  sorry

end container_volume_ratio_l3733_373302


namespace zayne_revenue_l3733_373326

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  single_price : ℕ  -- Price of a single bracelet
  pair_price : ℕ    -- Price of a pair of bracelets
  initial_stock : ℕ -- Initial number of bracelets
  single_sale_revenue : ℕ -- Revenue from selling single bracelets

/-- Calculates the total revenue from selling bracelets -/
def total_revenue (sale : BraceletSale) : ℕ :=
  let single_bracelets_sold := sale.single_sale_revenue / sale.single_price
  let remaining_bracelets := sale.initial_stock - single_bracelets_sold
  let pairs_sold := remaining_bracelets / 2
  let pair_revenue := pairs_sold * sale.pair_price
  sale.single_sale_revenue + pair_revenue

/-- Theorem stating that Zayne's total revenue is $132 -/
theorem zayne_revenue :
  ∃ (sale : BraceletSale),
    sale.single_price = 5 ∧
    sale.pair_price = 8 ∧
    sale.initial_stock = 30 ∧
    sale.single_sale_revenue = 60 ∧
    total_revenue sale = 132 :=
by
  sorry

end zayne_revenue_l3733_373326


namespace ashley_exam_result_l3733_373380

/-- The percentage of marks Ashley secured in the exam -/
def ashley_percentage (marks_secured : ℕ) (maximum_marks : ℕ) : ℚ :=
  (marks_secured : ℚ) / (maximum_marks : ℚ) * 100

/-- Theorem stating that Ashley secured 83% in the exam -/
theorem ashley_exam_result : ashley_percentage 332 400 = 83 := by
  sorry

end ashley_exam_result_l3733_373380


namespace inequality_system_solution_l3733_373344

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (5*x - 3 < 3*x + 5 ∧ x < a) ↔ x < 4) → a ≥ 4 := by
  sorry

end inequality_system_solution_l3733_373344


namespace complex_power_sum_l3733_373322

theorem complex_power_sum (z : ℂ) (h : z = -Complex.I) : z^100 + z^50 + 1 = -Complex.I := by
  sorry

end complex_power_sum_l3733_373322


namespace root_sum_ratio_l3733_373384

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ p q : ℝ, 
    (∀ m : ℝ, m * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    p / q + q / p = 2 ∧
    (m₁ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₁ * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    (m₂ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₂ * (q^2 - 3*q) + 2*q + 7 = 0)) →
  m₁ / m₂ + m₂ / m₁ = 136 / 9 := by
sorry

end root_sum_ratio_l3733_373384


namespace initial_mean_calculation_l3733_373394

/-- Given a set of values, proves that the initial mean before correcting an error
    is equal to the expected value. -/
theorem initial_mean_calculation (n : ℕ) (correct_value incorrect_value : ℝ) 
  (correct_mean : ℝ) (expected_initial_mean : ℝ) :
  n = 30 →
  correct_value = 165 →
  incorrect_value = 135 →
  correct_mean = 251 →
  expected_initial_mean = 250 →
  (n * correct_mean - (correct_value - incorrect_value)) / n = expected_initial_mean :=
by sorry

end initial_mean_calculation_l3733_373394


namespace brick_surface_area_l3733_373391

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 2 cm rectangular prism is 136 cm² -/
theorem brick_surface_area :
  surface_area 10 4 2 = 136 := by
  sorry

end brick_surface_area_l3733_373391


namespace min_sum_squares_min_value_is_two_l3733_373393

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a^2 + b^2 ≤ x^2 + y^2 :=
by sorry

theorem min_value_is_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ m : ℝ, m = 2 ∧ a^2 + b^2 ≥ m ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ x^2 + y^2 = m :=
by sorry

end min_sum_squares_min_value_is_two_l3733_373393


namespace xiao_dong_jump_distance_l3733_373350

/-- Given a standard jump distance and a recorded result, calculate the actual jump distance. -/
def actual_jump_distance (standard : ℝ) (recorded : ℝ) : ℝ :=
  standard + recorded

/-- Theorem: For a standard jump distance of 4.00 meters and a recorded result of -0.32,
    the actual jump distance is 3.68 meters. -/
theorem xiao_dong_jump_distance :
  let standard : ℝ := 4.00
  let recorded : ℝ := -0.32
  actual_jump_distance standard recorded = 3.68 := by
  sorry

end xiao_dong_jump_distance_l3733_373350


namespace custom_op_solution_l3733_373323

/-- Custom operation "*" for positive integers -/
def custom_op (k n : ℕ+) : ℕ := (n : ℕ) * (2 * k + n - 1) / 2

/-- Theorem stating that if 3 * n = 150 using the custom operation, then n = 15 -/
theorem custom_op_solution :
  ∃ (n : ℕ+), custom_op 3 n = 150 ∧ n = 15 := by
  sorry

end custom_op_solution_l3733_373323


namespace sarah_reading_capacity_l3733_373357

/-- The number of complete books Sarah can read given her reading speed and available time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) : ℕ :=
  (pages_per_hour * hours_available) / pages_per_book

/-- Theorem: Sarah can read 2 books in 8 hours -/
theorem sarah_reading_capacity :
  books_read 120 360 8 = 2 := by
  sorry

end sarah_reading_capacity_l3733_373357


namespace triangle_inequality_l3733_373366

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 1) : a^2 * c + b^2 * a + c^2 * b < 1/8 := by
  sorry

end triangle_inequality_l3733_373366


namespace lcm_factor_problem_l3733_373390

theorem lcm_factor_problem (A B : ℕ+) (Y : ℕ+) : 
  Nat.gcd A B = 23 →
  A = 391 →
  Nat.lcm A B = 23 * 13 * Y →
  Y = 17 := by
  sorry

end lcm_factor_problem_l3733_373390


namespace largest_value_l3733_373351

theorem largest_value (a b c d : ℝ) 
  (h : a + 1 = b - 2 ∧ a + 1 = c + 3 ∧ a + 1 = d - 4) : 
  d = max a (max b (max c d)) := by
  sorry

end largest_value_l3733_373351


namespace jellybean_count_l3733_373300

def jellybean_problem (initial : ℕ) (first_removal : ℕ) (added_back : ℕ) (second_removal : ℕ) : ℕ :=
  initial - first_removal + added_back - second_removal

theorem jellybean_count : jellybean_problem 37 15 5 4 = 23 := by
  sorry

end jellybean_count_l3733_373300


namespace divisibility_condition_l3733_373320

theorem divisibility_condition (n : ℤ) : 
  (n^5 + 3) % (n^2 + 1) = 0 ↔ n ∈ ({-3, -1, 0, 1, 2} : Set ℤ) := by sorry

end divisibility_condition_l3733_373320


namespace greatest_distance_between_circle_centers_l3733_373345

/-- The greatest distance between centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 24)
  (h_height : rectangle_height = 18)
  (h_diameter : circle_diameter = 8)
  (h_nonneg_width : 0 ≤ rectangle_width)
  (h_nonneg_height : 0 ≤ rectangle_height)
  (h_nonneg_diameter : 0 ≤ circle_diameter)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ d : ℝ, d = Real.sqrt 356 ∧
    ∀ d' : ℝ, d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by
  sorry

end greatest_distance_between_circle_centers_l3733_373345


namespace expression_values_l3733_373365

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 :=
by sorry

end expression_values_l3733_373365


namespace first_four_digits_after_decimal_l3733_373335

theorem first_four_digits_after_decimal (x : ℝ) : 
  x = (5^1001 + 2)^(5/3) → 
  ∃ n : ℕ, 0 ≤ n ∧ n < 10000 ∧ (x - ⌊x⌋) * 10000 = 3333 + n / 10000 :=
sorry

end first_four_digits_after_decimal_l3733_373335


namespace haley_facebook_pictures_l3733_373361

/-- The number of pictures Haley uploaded to Facebook -/
def total_pictures : ℕ := 65

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 17

/-- The number of additional albums -/
def additional_albums : ℕ := 6

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 8

/-- Theorem stating the total number of pictures uploaded to Facebook -/
theorem haley_facebook_pictures :
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end haley_facebook_pictures_l3733_373361


namespace water_level_unchanged_l3733_373336

-- Define the densities of water and ice
variable (ρ_water ρ_ice : ℝ)

-- Define the initial volume of water taken for freezing
variable (V : ℝ)

-- Hypothesis: density of ice is less than density of water
axiom h1 : ρ_ice < ρ_water

-- Hypothesis: mass is conserved when water freezes
axiom h2 : V * ρ_water = (V * ρ_water / ρ_ice) * ρ_ice

-- Hypothesis: Archimedes' principle applies to floating ice
axiom h3 : ∀ W : ℝ, W * ρ_ice = (W * ρ_ice / ρ_water) * ρ_water

-- Theorem: The volume of water displaced by the ice is equal to the original volume of water
theorem water_level_unchanged (V : ℝ) (h1 : ρ_ice < ρ_water) 
  (h2 : V * ρ_water = (V * ρ_water / ρ_ice) * ρ_ice) 
  (h3 : ∀ W : ℝ, W * ρ_ice = (W * ρ_ice / ρ_water) * ρ_water) :
  (V * ρ_water / ρ_ice) * ρ_ice / ρ_water = V :=
by sorry

end water_level_unchanged_l3733_373336


namespace quadratic_function_value_l3733_373388

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(1) = 7, f(2) = 12, and c = 3, then f(3) = 18 -/
theorem quadratic_function_value (a b : ℝ) 
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 := by
  sorry

end quadratic_function_value_l3733_373388


namespace cubic_equation_solution_l3733_373306

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by
  sorry

end cubic_equation_solution_l3733_373306


namespace complex_number_quadrant_l3733_373349

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I) / (2 * Complex.I - 1)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_quadrant_l3733_373349


namespace range_of_m_l3733_373367

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then 2^(x - m) else (m * x) / (4 * x^2 + 16)

theorem range_of_m (m : ℝ) :
  (∀ x₁ ≥ 2, ∃ x₂ ≤ 2, f m x₁ = f m x₂) →
  m ≤ 4 := by
  sorry

end range_of_m_l3733_373367


namespace numerica_base_l3733_373387

/-- Convert a number from base r to base 10 -/
def to_base_10 (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The base r used in Numerica -/
def r : Nat := sorry

/-- The price of the gadget in base r -/
def price : List Nat := [5, 3, 0]

/-- The payment made in base r -/
def payment : List Nat := [1, 1, 0, 0]

/-- The change received in base r -/
def change : List Nat := [4, 6, 0]

theorem numerica_base :
  (to_base_10 price r + to_base_10 change r = to_base_10 payment r) ↔ r = 9 := by
  sorry

end numerica_base_l3733_373387


namespace inequality_equivalence_l3733_373358

theorem inequality_equivalence (x : ℝ) : 
  (5 * x - 1 < (x + 1)^2 ∧ (x + 1)^2 < 7 * x - 3) ↔ (2 < x ∧ x < 4) :=
by sorry

end inequality_equivalence_l3733_373358


namespace division_problem_l3733_373382

theorem division_problem (a : ℝ) : a / 0.3 = 0.6 → a = 0.18 := by
  sorry

end division_problem_l3733_373382


namespace other_triangle_area_ratio_l3733_373324

/-- Represents a right triangle with a point on its hypotenuse and parallel lines dividing it -/
structure DividedRightTriangle where
  /-- The area of one small right triangle -/
  smallTriangleArea : ℝ
  /-- The area of the rectangle -/
  rectangleArea : ℝ
  /-- The ratio of the small triangle area to the rectangle area -/
  n : ℝ
  /-- The ratio of the longer side to the shorter side of the rectangle -/
  k : ℝ
  /-- The small triangle area is n times the rectangle area -/
  area_relation : smallTriangleArea = n * rectangleArea
  /-- The sides of the rectangle are in the ratio 1:k -/
  rectangle_ratio : k > 0

/-- The ratio of the area of the other small right triangle to the area of the rectangle is n -/
theorem other_triangle_area_ratio (t : DividedRightTriangle) :
    ∃ otherTriangleArea : ℝ, otherTriangleArea / t.rectangleArea = t.n := by
  sorry

end other_triangle_area_ratio_l3733_373324


namespace room_length_calculation_l3733_373373

/-- Proves that given a room with specified width, cost per square meter for paving,
    and total paving cost, the length of the room is as calculated. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  width = 3.75 ∧ cost_per_sqm = 600 ∧ total_cost = 12375 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end room_length_calculation_l3733_373373


namespace triangle_side_length_l3733_373368

theorem triangle_side_length (A B C : Real) (R : Real) (a b c : Real) :
  R = 5/6 →
  Real.cos B = 3/5 →
  Real.cos A = 12/13 →
  c = 21/13 :=
by
  sorry

end triangle_side_length_l3733_373368


namespace square_ad_perimeter_l3733_373310

theorem square_ad_perimeter (side_length : ℝ) (h : side_length = 8) : 
  4 * side_length = 32 := by
  sorry

end square_ad_perimeter_l3733_373310


namespace als_original_portion_l3733_373313

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3*b + 3*c = 1800 →
  a = 825 :=
by sorry

end als_original_portion_l3733_373313


namespace quadratic_negative_roots_l3733_373363

theorem quadratic_negative_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m+2)*x + m + 5 = 0 → x < 0) ↔ m ≥ 4 := by
  sorry

end quadratic_negative_roots_l3733_373363


namespace polynomial_coefficient_property_l3733_373370

theorem polynomial_coefficient_property (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243 := by
  sorry

end polynomial_coefficient_property_l3733_373370


namespace quadratic_roots_difference_ratio_l3733_373356

noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := x^2 - x + 2*a
noncomputable def f₂ (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + 3
noncomputable def f₃ (a b : ℝ) (x : ℝ) : ℝ := 4*x^2 + (2*b-3)*x + 6*a + 3
noncomputable def f₄ (a b : ℝ) (x : ℝ) : ℝ := 4*x^2 + (6*b-1)*x + 9 + 2*a

noncomputable def A (a : ℝ) : ℝ := Real.sqrt (1 - 8*a)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (4*b^2 - 12)
noncomputable def C (a b : ℝ) : ℝ := (1/4) * Real.sqrt ((2*b - 3)^2 - 64*(6*a + 3))
noncomputable def D (a b : ℝ) : ℝ := (1/4) * Real.sqrt ((6*b - 1)^2 - 64*(9 + 2*a))

theorem quadratic_roots_difference_ratio (a b : ℝ) (h : A a ≠ B b) :
  (C a b^2 - D a b^2) / (A a^2 - B b^2) = 1/2 := by
  sorry

end quadratic_roots_difference_ratio_l3733_373356


namespace interest_frequency_proof_l3733_373379

/-- The nominal interest rate per annum -/
def nominal_rate : ℝ := 0.10

/-- The effective annual rate -/
def effective_annual_rate : ℝ := 0.1025

/-- The frequency of interest payment (number of compounding periods per year) -/
def frequency : ℕ := 2

/-- Theorem stating that the given frequency results in the correct effective annual rate -/
theorem interest_frequency_proof :
  (1 + nominal_rate / frequency) ^ frequency - 1 = effective_annual_rate :=
by sorry

end interest_frequency_proof_l3733_373379


namespace evaluate_expression_l3733_373399

theorem evaluate_expression : 2 * ((3^4)^3 - (4^3)^4) = -32471550 := by
  sorry

end evaluate_expression_l3733_373399


namespace regular_polygon_exterior_angle_l3733_373375

theorem regular_polygon_exterior_angle (n : ℕ) : 
  (n > 2) → (360 / n = 72) → n = 5 := by
  sorry

end regular_polygon_exterior_angle_l3733_373375


namespace unique_positive_solution_l3733_373315

theorem unique_positive_solution :
  ∃! (y : ℝ), y > 0 ∧ (y - 6) / 12 = 6 / (y - 12) ∧ y = 18 := by
  sorry

end unique_positive_solution_l3733_373315


namespace parabola_line_intersection_l3733_373339

/-- Parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {(x, y) | y = x^2}

/-- Point Q -/
def Q : ℝ × ℝ := (10, 4)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) := {(x, y) | y - 4 = m * (x - 10)}

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop := line_through_Q m ∩ P = ∅

theorem parabola_line_intersection (r s : ℝ) :
  (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 := by
  sorry

end parabola_line_intersection_l3733_373339


namespace total_count_is_2552_l3733_373353

/-- Represents the total count for a week given the number of items and the counting schedule. -/
def weeklyCount (tiles books windows chairs lightBulbs : ℕ) : ℕ :=
  let monday := tiles * 2 + books * 2 + windows * 2
  let tuesday := tiles * 3 + books * 2 + windows * 1
  let wednesday := chairs * 4 + lightBulbs * 5
  let thursday := tiles * 1 + chairs * 2 + books * 3 + windows * 4 + lightBulbs * 5
  let friday := tiles * 1 + books * 2 + chairs * 2 + windows * 3 + lightBulbs * 3
  monday + tuesday + wednesday + thursday + friday

/-- Theorem stating that the total count for the week is 2552 given the specific item counts. -/
theorem total_count_is_2552 : weeklyCount 60 120 10 80 24 = 2552 := by
  sorry

#eval weeklyCount 60 120 10 80 24

end total_count_is_2552_l3733_373353


namespace mariel_dogs_count_l3733_373303

theorem mariel_dogs_count (total_legs : ℕ) (other_dogs : ℕ) (human_legs : ℕ) (dog_legs : ℕ) :
  total_legs = 36 →
  other_dogs = 3 →
  human_legs = 2 →
  dog_legs = 4 →
  ∃ (mariel_dogs : ℕ), mariel_dogs = 5 ∧
    total_legs = 2 * human_legs + other_dogs * dog_legs + mariel_dogs * dog_legs :=
by
  sorry

end mariel_dogs_count_l3733_373303


namespace exactly_four_valid_labelings_l3733_373308

/-- Represents a truncated 3x3 chessboard with 8 squares. -/
structure TruncatedChessboard :=
  (labels : Fin 8 → Fin 8)

/-- Checks if two positions on the board are connected (share a vertex). -/
def are_connected (p1 p2 : Fin 8) : Bool :=
  sorry

/-- Checks if a labeling is valid according to the problem rules. -/
def is_valid_labeling (board : TruncatedChessboard) : Prop :=
  (∀ p1 p2 : Fin 8, p1 ≠ p2 → board.labels p1 ≠ board.labels p2) ∧
  (∀ p1 p2 : Fin 8, are_connected p1 p2 → 
    (board.labels p1).val + 1 ≠ (board.labels p2).val ∧
    (board.labels p2).val + 1 ≠ (board.labels p1).val)

/-- The main theorem stating that there are exactly 4 valid labelings. -/
theorem exactly_four_valid_labelings :
  ∃! (valid_labelings : Finset TruncatedChessboard),
    (∀ board ∈ valid_labelings, is_valid_labeling board) ∧
    valid_labelings.card = 4 :=
  sorry

end exactly_four_valid_labelings_l3733_373308


namespace function_passes_through_point_l3733_373392

/-- Given a function f(x) = 2a^x + 3, where a > 0 and a ≠ 1, prove that f(0) = 5 -/
theorem function_passes_through_point
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (f : ℝ → ℝ) 
  (h3 : ∀ x, f x = 2 * a^x + 3) : 
  f 0 = 5 := by
  sorry

end function_passes_through_point_l3733_373392


namespace complex_expression_simplification_l3733_373307

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  7 * (4 - 2*i) + 4*i*(7 - 3*i) + 2*(5 + i) = 50 + 16*i := by
  sorry

end complex_expression_simplification_l3733_373307


namespace quadratic_sum_l3733_373396

/-- Given a quadratic equation x^2 - 10x + 15 = 0 that can be rewritten as (x + b)^2 = c,
    prove that b + c = 5 -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
  sorry

end quadratic_sum_l3733_373396


namespace milk_delivery_calculation_l3733_373309

/-- Given a total amount of milk and a difference between two people's deliveries,
    calculate the amount delivered by the person delivering more milk. -/
theorem milk_delivery_calculation (total : ℕ) (difference : ℕ) (h1 : total = 2100) (h2 : difference = 200) :
  (total + difference) / 2 = 1150 := by
  sorry

end milk_delivery_calculation_l3733_373309


namespace power_of_ten_equation_l3733_373395

theorem power_of_ten_equation (x : ℕ) : (10^x) / (10^650) = 100000 ↔ x = 655 := by
  sorry

end power_of_ten_equation_l3733_373395


namespace soda_drinkers_l3733_373321

theorem soda_drinkers (total : ℕ) (wine : ℕ) (both : ℕ) (soda : ℕ) : 
  total = 31 → wine = 26 → both = 17 → soda = total + both - wine := by
  sorry

end soda_drinkers_l3733_373321


namespace solve_baseball_cards_problem_l3733_373389

/-- The number of cards Brandon, Malcom, and Ella have, and the combined remaining cards after transactions -/
def baseball_cards_problem (brandon_cards : ℕ) (malcom_extra : ℕ) (ella_less : ℕ) 
  (malcom_fraction : ℚ) (ella_fraction : ℚ) : Prop :=
  let malcom_cards := brandon_cards + malcom_extra
  let ella_cards := malcom_cards - ella_less
  let malcom_remaining := malcom_cards - Int.floor (malcom_fraction * malcom_cards)
  let ella_remaining := ella_cards - Int.floor (ella_fraction * ella_cards)
  malcom_remaining + ella_remaining = 32

/-- Theorem statement for the baseball cards problem -/
theorem solve_baseball_cards_problem : 
  baseball_cards_problem 20 12 5 (2/3) (1/4) := by sorry

end solve_baseball_cards_problem_l3733_373389


namespace two_integers_sum_l3733_373314

theorem two_integers_sum (x y : ℕ+) : x - y = 4 → x * y = 63 → x + y = 18 := by
  sorry

end two_integers_sum_l3733_373314


namespace cubic_polynomial_problem_l3733_373385

-- Define the cubic polynomial whose roots are a, b, c
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x + 3

-- Define the properties of P
def is_valid_P (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- The theorem to prove
theorem cubic_polynomial_problem :
  ∃ (P : ℝ → ℝ) (a b c : ℝ),
    is_valid_P P a b c ∧
    (∀ x, P x = -17/3 * x^3 + 68/3 * x^2 - 31/3 * x - 18) :=
by sorry

end cubic_polynomial_problem_l3733_373385


namespace perpendicular_lines_parallel_l3733_373305

-- Define a plane
class Plane where
  -- Add any necessary properties for a plane

-- Define a line
class Line where
  -- Add any necessary properties for a line

-- Define perpendicularity between a line and a plane
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of perpendicularity between a line and a plane

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallel lines

-- Theorem statement
theorem perpendicular_lines_parallel (p : Plane) (l1 l2 : Line) :
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_lines l1 l2 :=
by sorry


end perpendicular_lines_parallel_l3733_373305
