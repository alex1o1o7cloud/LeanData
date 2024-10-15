import Mathlib

namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2883_288393

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2883_288393


namespace NUMINAMATH_CALUDE_problem_solution_l2883_288302

theorem problem_solution (x y : ℝ) (h1 : x > y) 
  (h2 : x^2*y^2 + x^2 + y^2 + 2*x*y = 40) (h3 : x*y + x + y = 8) : 
  x = 3 + Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2883_288302


namespace NUMINAMATH_CALUDE_no_valid_coloring_l2883_288372

/-- Represents a coloring of a 4x4 grid -/
def Coloring := Fin 4 → Fin 4 → Fin 8

/-- Checks if two cells are adjacent in a 4x4 grid -/
def adjacent (r1 c1 r2 c2 : Fin 4) : Prop :=
  (r1 = r2 ∧ (c1 = c2 + 1 ∨ c2 = c1 + 1)) ∨
  (c1 = c2 ∧ (r1 = r2 + 1 ∨ r2 = r1 + 1))

/-- Checks if a coloring satisfies the condition that every pair of colors
    appears on adjacent cells -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ color1 color2 : Fin 8, color1 < color2 →
    ∃ r1 c1 r2 c2 : Fin 4, 
      adjacent r1 c1 r2 c2 ∧
      c r1 c1 = color1 ∧ c r2 c2 = color2

/-- The main theorem stating that no valid coloring exists -/
theorem no_valid_coloring : ¬∃ c : Coloring, valid_coloring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l2883_288372


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2883_288396

/-- The function f(x) = kx - ln x is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2883_288396


namespace NUMINAMATH_CALUDE_jeans_cost_per_pair_l2883_288341

def leonard_cost : ℕ := 250
def michael_backpack_cost : ℕ := 100
def total_spent : ℕ := 450
def jeans_pairs : ℕ := 2

theorem jeans_cost_per_pair : 
  (total_spent - leonard_cost - michael_backpack_cost) / jeans_pairs = 50 :=
by sorry

end NUMINAMATH_CALUDE_jeans_cost_per_pair_l2883_288341


namespace NUMINAMATH_CALUDE_like_terms_imply_xy_value_l2883_288366

theorem like_terms_imply_xy_value (a b : ℝ) (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ 2 * a^x * b^3 = k * (-a^2 * b^(1-y))) →
  x * y = -4 :=
sorry

end NUMINAMATH_CALUDE_like_terms_imply_xy_value_l2883_288366


namespace NUMINAMATH_CALUDE_problem_1_l2883_288355

theorem problem_1 : Real.sqrt 9 * 3⁻¹ + 2^3 / |(-2)| = 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2883_288355


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_l2883_288359

def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iic 2 \ {2}) ∪ Set.Ici 3

theorem fractional_inequality_solution :
  {x : ℝ | (x - 3) / (x - 2) ≥ 0} = {x : ℝ | solution_set x} :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_l2883_288359


namespace NUMINAMATH_CALUDE_min_value_theorem_achievable_lower_bound_l2883_288352

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + 2*n = 1) :
  (m + 1) * (n + 1) / (m * n) ≥ 8 + 4 * Real.sqrt 3 :=
by sorry

theorem achievable_lower_bound :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m + 2*n = 1 ∧ (m + 1) * (n + 1) / (m * n) = 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_achievable_lower_bound_l2883_288352


namespace NUMINAMATH_CALUDE_value_of_x_l2883_288309

theorem value_of_x (x y z : ℚ) : 
  x = (1/2) * y → 
  y = (1/5) * z → 
  z = 60 → 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2883_288309


namespace NUMINAMATH_CALUDE_min_value_of_function_l2883_288388

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 ∧
  (x - 4 + 9 / (x + 1) = 1 ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2883_288388


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_m_l2883_288358

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem 1: Solving the inequality f(x) > 3-4x
theorem solve_inequality : 
  ∀ x : ℝ, f x > 3 - 4*x ↔ x > 3/5 := by sorry

-- Theorem 2: Finding the range of m
theorem range_of_m : 
  (∀ x : ℝ, f x + |1 - x| ≥ 6*m^2 - 5*m) ↔ m ∈ Set.Icc (-1/6 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_m_l2883_288358


namespace NUMINAMATH_CALUDE_boat_round_trip_time_specific_boat_round_trip_time_l2883_288394

/-- Calculate the total time for a round trip by boat -/
theorem boat_round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  total_time

/-- The total time taken for the specific round trip is approximately 947.6923 hours -/
theorem specific_boat_round_trip_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |boat_round_trip_time 22 4 10080 - 947.6923| < ε :=
sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_specific_boat_round_trip_time_l2883_288394


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2883_288380

theorem triangle_angle_C (a b : ℝ) (A : ℝ) :
  a = 1 →
  b = Real.sqrt 2 →
  2 * Real.sin A * (Real.cos (π / 4))^2 + Real.cos A * Real.sin (π / 2) - Real.sin A = 3 / 2 →
  ∃ (C : ℝ), (C = 7 * π / 12 ∨ C = π / 12) ∧ 
  (∃ (B : ℝ), A + B + C = π ∧ Real.sin A / a = Real.sin B / b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2883_288380


namespace NUMINAMATH_CALUDE_stadium_length_conversion_l2883_288347

/-- Converts yards to feet given a conversion factor -/
def yards_to_feet (yards : ℕ) (conversion_factor : ℕ) : ℕ :=
  yards * conversion_factor

/-- Theorem: The stadium length of 62 yards is equal to 186 feet -/
theorem stadium_length_conversion :
  yards_to_feet 62 3 = 186 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_conversion_l2883_288347


namespace NUMINAMATH_CALUDE_chicken_food_consumption_l2883_288390

/-- Given Dany's farm animals and their food consumption, calculate the amount of food each chicken eats per day. -/
theorem chicken_food_consumption 
  (num_cows : ℕ) 
  (num_sheep : ℕ) 
  (num_chickens : ℕ) 
  (cow_sheep_consumption : ℕ) 
  (total_consumption : ℕ) 
  (h1 : num_cows = 4) 
  (h2 : num_sheep = 3) 
  (h3 : num_chickens = 7) 
  (h4 : cow_sheep_consumption = 2) 
  (h5 : total_consumption = 35) : 
  ((total_consumption - (num_cows + num_sheep) * cow_sheep_consumption) / num_chickens : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_food_consumption_l2883_288390


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2883_288351

theorem child_ticket_cost (adult_price : ℕ) (total_attendees : ℕ) (total_revenue : ℕ) (child_attendees : ℕ) : 
  adult_price = 60 →
  total_attendees = 280 →
  total_revenue = 14000 →
  child_attendees = 80 →
  ∃ (child_price : ℕ), child_price = 25 ∧
    total_revenue = adult_price * (total_attendees - child_attendees) + child_price * child_attendees :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2883_288351


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2883_288387

theorem complex_equation_solution : ∃ (x y : ℝ), (2*x - 1 : ℂ) + I = y - (2 - y)*I ∧ x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2883_288387


namespace NUMINAMATH_CALUDE_problem_statement_l2883_288383

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

theorem problem_statement :
  (3 ∈ A) ∧ (∀ k : ℤ, 4*k - 2 ∉ A) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2883_288383


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2883_288348

theorem min_value_expression (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem min_value_achievable :
  ∃ x > -1, 2 * x + 1 / (x + 1) = 2 * Real.sqrt 2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2883_288348


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2883_288331

theorem min_value_of_expression :
  ∃ (x : ℝ), (8 - x) * (6 - x) * (8 + x) * (6 + x) = -196 ∧
  ∀ (y : ℝ), (8 - y) * (6 - y) * (8 + y) * (6 + y) ≥ -196 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2883_288331


namespace NUMINAMATH_CALUDE_carter_to_dog_height_ratio_l2883_288334

-- Define the heights in inches
def dog_height : ℕ := 24
def betty_height_feet : ℕ := 3
def height_difference : ℕ := 12

-- Theorem to prove
theorem carter_to_dog_height_ratio :
  let betty_height_inches : ℕ := betty_height_feet * 12
  let carter_height : ℕ := betty_height_inches + height_difference
  carter_height / dog_height = 2 := by
sorry

end NUMINAMATH_CALUDE_carter_to_dog_height_ratio_l2883_288334


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l2883_288343

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x - 20
def line2 (x y : ℝ) : Prop := 3 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate : 
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l2883_288343


namespace NUMINAMATH_CALUDE_moon_carbon_percentage_l2883_288300

/-- Represents the composition and weight of a celestial body -/
structure CelestialBody where
  weight : ℝ
  iron_percent : ℝ
  carbon_percent : ℝ
  other_percent : ℝ
  other_weight : ℝ

/-- The moon's composition and weight -/
def moon : CelestialBody := {
  weight := 250,
  iron_percent := 50,
  carbon_percent := 20,  -- This is what we want to prove
  other_percent := 30,
  other_weight := 75
}

/-- Mars' composition and weight -/
def mars : CelestialBody := {
  weight := 500,
  iron_percent := 50,
  carbon_percent := 20,
  other_percent := 30,
  other_weight := 150
}

/-- Theorem stating that the moon's carbon percentage is 20% -/
theorem moon_carbon_percentage :
  moon.carbon_percent = 20 ∧
  moon.iron_percent = 50 ∧
  moon.other_percent = 100 - moon.iron_percent - moon.carbon_percent ∧
  moon.weight = 250 ∧
  mars.weight = 2 * moon.weight ∧
  mars.iron_percent = moon.iron_percent ∧
  mars.carbon_percent = moon.carbon_percent ∧
  mars.other_percent = moon.other_percent ∧
  mars.other_weight = 150 ∧
  moon.other_weight = mars.other_weight / 2 := by
  sorry


end NUMINAMATH_CALUDE_moon_carbon_percentage_l2883_288300


namespace NUMINAMATH_CALUDE_composite_number_quotient_l2883_288305

def composite_numbers : List ℕ := [4, 6, 8, 9, 10, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39]

def product_6th_to_15th : ℕ := (List.take 10 (List.drop 5 composite_numbers)).prod

def product_16th_to_25th : ℕ := (List.take 10 (List.drop 15 composite_numbers)).prod

theorem composite_number_quotient :
  (product_6th_to_15th : ℚ) / product_16th_to_25th =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_composite_number_quotient_l2883_288305


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2883_288308

def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2883_288308


namespace NUMINAMATH_CALUDE_odd_sum_probability_l2883_288398

/-- Represents a tile with a number from 1 to 12 -/
def Tile := Fin 12

/-- Represents a player's selection of 4 tiles -/
def PlayerSelection := Finset Tile

/-- The set of all possible tile selections -/
def AllSelections : Finset (PlayerSelection × PlayerSelection × PlayerSelection) :=
  sorry

/-- Checks if a player's selection sum is odd -/
def isOddSum (selection : PlayerSelection) : Bool :=
  sorry

/-- The set of selections where all players have odd sums -/
def OddSumSelections : Finset (PlayerSelection × PlayerSelection × PlayerSelection) :=
  sorry

/-- The probability of all players obtaining an odd sum -/
theorem odd_sum_probability :
  (Finset.card OddSumSelections : ℚ) / (Finset.card AllSelections : ℚ) = 16 / 385 :=
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l2883_288398


namespace NUMINAMATH_CALUDE_quadrant_I_solution_condition_l2883_288313

theorem quadrant_I_solution_condition (c : ℝ) :
  (∃ x y : ℝ, 2 * x - y = 5 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ -2 < c ∧ c < 8/5 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_condition_l2883_288313


namespace NUMINAMATH_CALUDE_valid_coloring_iff_odd_l2883_288321

/-- A valid coloring of an n-gon satisfies the given conditions --/
def ValidColoring (n : ℕ) (P : Set (Fin n)) (coloring : Fin n → Fin n → Fin n) : Prop :=
  -- P represents the vertices of the n-gon
  (∀ i j : Fin n, coloring i j < n) ∧ 
  -- For any three distinct colors, there exists a triangle with those colors
  (∀ c₁ c₂ c₃ : Fin n, c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ → 
    ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
      coloring i j = c₁ ∧ coloring j k = c₂ ∧ coloring i k = c₃)

/-- A valid coloring of an n-gon exists if and only if n is odd --/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ P : Set (Fin n), ∃ coloring : Fin n → Fin n → Fin n, ValidColoring n P coloring) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_coloring_iff_odd_l2883_288321


namespace NUMINAMATH_CALUDE_teacher_health_survey_l2883_288375

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 60)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_teacher_health_survey_l2883_288375


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_is_151200_l2883_288377

/-- The number of ways to choose 6 players from a team of 10 players for 6 distinct positions -/
def volleyball_lineup_count : ℕ := 10 * 9 * 8 * 7 * 6 * 5

/-- Theorem stating that the number of ways to choose a volleyball lineup is 151,200 -/
theorem volleyball_lineup_count_is_151200 : volleyball_lineup_count = 151200 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_is_151200_l2883_288377


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2883_288338

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 2 → b = 3 → c^2 = a^2 + b^2 → c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2883_288338


namespace NUMINAMATH_CALUDE_not_all_odd_l2883_288374

theorem not_all_odd (a b c d : ℕ) (h1 : a = b * c + d) (h2 : d < b) : 
  ¬(Odd a ∧ Odd b ∧ Odd c ∧ Odd d) := by
  sorry

end NUMINAMATH_CALUDE_not_all_odd_l2883_288374


namespace NUMINAMATH_CALUDE_amoli_driving_time_l2883_288323

-- Define the constants from the problem
def total_distance : ℝ := 369
def amoli_speed : ℝ := 42
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2
def remaining_distance : ℝ := 121

-- Define Amoli's driving time as a variable
def amoli_time : ℝ := 3

-- Theorem statement
theorem amoli_driving_time :
  amoli_speed * amoli_time + anayet_speed * anayet_time = total_distance - remaining_distance :=
by sorry

end NUMINAMATH_CALUDE_amoli_driving_time_l2883_288323


namespace NUMINAMATH_CALUDE_eraser_ratio_l2883_288306

theorem eraser_ratio (andrea_erasers : ℕ) (anya_extra_erasers : ℕ) :
  andrea_erasers = 4 →
  anya_extra_erasers = 12 →
  (andrea_erasers + anya_extra_erasers) / andrea_erasers = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_eraser_ratio_l2883_288306


namespace NUMINAMATH_CALUDE_wang_li_final_score_l2883_288327

/-- Calculates the weighted average score given individual scores and weights -/
def weightedAverage (writtenScore demonstrationScore interviewScore : ℚ) 
  (writtenWeight demonstrationWeight interviewWeight : ℚ) : ℚ :=
  (writtenScore * writtenWeight + demonstrationScore * demonstrationWeight + interviewScore * interviewWeight) /
  (writtenWeight + demonstrationWeight + interviewWeight)

/-- Theorem stating that Wang Li's final score is 94 given the specified scores and weights -/
theorem wang_li_final_score :
  weightedAverage 96 90 95 5 3 2 = 94 := by
  sorry


end NUMINAMATH_CALUDE_wang_li_final_score_l2883_288327


namespace NUMINAMATH_CALUDE_complement_of_M_in_N_l2883_288329

def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {0, 2, 3, 4, 5}

theorem complement_of_M_in_N :
  N \ M = {0, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_N_l2883_288329


namespace NUMINAMATH_CALUDE_consecutive_cube_product_divisible_by_504_l2883_288317

theorem consecutive_cube_product_divisible_by_504 (a : ℤ) : 
  504 ∣ ((a^3 - 1) * a^3 * (a^3 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_cube_product_divisible_by_504_l2883_288317


namespace NUMINAMATH_CALUDE_equation_solutions_l2883_288318

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (3 + Real.sqrt 15) / 3 ∧ x2 = (3 - Real.sqrt 15) / 3 ∧ 
    3 * x1^2 - 6 * x1 - 2 = 0 ∧ 3 * x2^2 - 6 * x2 - 2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 3 ∧ y2 = 5 ∧ 
    (y1 - 3)^2 = 2 * y1 - 6 ∧ (y2 - 3)^2 = 2 * y2 - 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2883_288318


namespace NUMINAMATH_CALUDE_angle_values_l2883_288311

/-- Given an angle α with terminal side passing through point P(-3, m) and cosα = -3/5,
    prove the values of m, sinα, and tanα. -/
theorem angle_values (α : Real) (m : Real) 
    (h1 : ∃ (x y : Real), x = -3 ∧ y = m ∧ Real.cos α * Real.sqrt (x^2 + y^2) = x)
    (h2 : Real.cos α = -3/5) :
    (m = 4 ∨ m = -4) ∧ 
    ((Real.sin α = 4/5 ∧ Real.tan α = -4/3) ∨ 
     (Real.sin α = -4/5 ∧ Real.tan α = 4/3)) := by
  sorry

end NUMINAMATH_CALUDE_angle_values_l2883_288311


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2883_288307

theorem complex_number_theorem (z : ℂ) (b : ℝ) :
  z = (Complex.I ^ 3) / (1 - Complex.I) →
  (∃ (y : ℝ), z + b = Complex.I * y) →
  b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2883_288307


namespace NUMINAMATH_CALUDE_greatest_integer_in_ratio_l2883_288384

theorem greatest_integer_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 84 →
  2 * b = 5 * a →
  7 * a = 2 * c →
  max a (max b c) = 42 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_in_ratio_l2883_288384


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2883_288346

theorem imaginary_part_of_complex_number (z : ℂ) (h : z = -1 + Complex.I) :
  z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2883_288346


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2883_288392

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem planes_parallel_if_perpendicular_to_same_line
  (m n : Line) (α β : Plane)
  (h_not_coincident_lines : m ≠ n)
  (h_not_coincident_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_m_perp_β : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2883_288392


namespace NUMINAMATH_CALUDE_two_digit_times_eleven_l2883_288371

theorem two_digit_times_eleven (A B : ℕ) (h : A + B ≥ 10) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * (A + B - 10) + B := by
  sorry

end NUMINAMATH_CALUDE_two_digit_times_eleven_l2883_288371


namespace NUMINAMATH_CALUDE_negation_of_conditional_l2883_288389

theorem negation_of_conditional (a b : ℝ) :
  ¬(a > b → 2*a > 2*b) ↔ (a ≤ b → 2*a ≤ 2*b) := by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l2883_288389


namespace NUMINAMATH_CALUDE_trent_kept_tadpoles_l2883_288361

/-- The number of tadpoles Trent initially caught -/
def initial_tadpoles : ℕ := 180

/-- The percentage of tadpoles Trent let go -/
def percent_released : ℚ := 75 / 100

/-- The number of tadpoles Trent kept -/
def tadpoles_kept : ℕ := 45

/-- Theorem stating that the number of tadpoles Trent kept is equal to 45 -/
theorem trent_kept_tadpoles : 
  (initial_tadpoles : ℚ) * (1 - percent_released) = tadpoles_kept := by sorry

end NUMINAMATH_CALUDE_trent_kept_tadpoles_l2883_288361


namespace NUMINAMATH_CALUDE_parallelogram_area_l2883_288316

/-- The area of a parallelogram with one angle measuring 100 degrees and two consecutive sides of lengths 10 inches and 18 inches is equal to 180 sin(10°) square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 18) (h3 : θ = 100 * π / 180) :
  a * b * Real.sin ((π / 2) - (θ / 2)) = 180 * Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2883_288316


namespace NUMINAMATH_CALUDE_x_to_y_ratio_l2883_288353

theorem x_to_y_ratio (x y : ℝ) (h : 3 * x = 0.12 * 250 * y) : x / y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_ratio_l2883_288353


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l2883_288332

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

theorem divisor_sum_theorem (k m : ℕ) :
  (sum_of_geometric_series 1 2 k) * (sum_of_geometric_series 1 5 m) = 930 →
  k + m = 6 := by
sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l2883_288332


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2883_288335

theorem system_of_equations_solution :
  ∃ (x y z : ℝ),
    (4*x - 3*y + z = -9) ∧
    (2*x + 5*y - 3*z = 8) ∧
    (x + y + 2*z = 5) ∧
    (x = 1 ∧ y = -1 ∧ z = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2883_288335


namespace NUMINAMATH_CALUDE_male_attendees_fraction_l2883_288379

theorem male_attendees_fraction (M F : ℝ) : 
  M + F = 1 → 
  (7/8 : ℝ) * M + (4/5 : ℝ) * F = 0.845 → 
  M = 0.6 := by
sorry

end NUMINAMATH_CALUDE_male_attendees_fraction_l2883_288379


namespace NUMINAMATH_CALUDE_orthocenter_on_altitude_ratio_HD_HA_is_zero_l2883_288360

/-- A triangle with sides 11, 12, and 13 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 11)
  (hb : b = 12)
  (hc : c = 13)

/-- The orthocenter of the triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The altitude from vertex A to side BC -/
def altitude_AD (t : Triangle) : ℝ := sorry

/-- The point D where the altitude AD intersects BC -/
def point_D (t : Triangle) : ℝ × ℝ := sorry

/-- The point A of the triangle -/
def point_A (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_on_altitude (t : Triangle) :
  let H := orthocenter t
  let D := point_D t
  distance H D = 0 := by sorry

theorem ratio_HD_HA_is_zero (t : Triangle) :
  let H := orthocenter t
  let D := point_D t
  let A := point_A t
  distance H D / distance H A = 0 := by sorry

end NUMINAMATH_CALUDE_orthocenter_on_altitude_ratio_HD_HA_is_zero_l2883_288360


namespace NUMINAMATH_CALUDE_square_garden_multiple_l2883_288356

/-- Given a square garden with perimeter 40 feet and area equal to a multiple of the perimeter plus 20, prove that the multiple is 2. -/
theorem square_garden_multiple (side : ℝ) (multiple : ℝ) : 
  side > 0 →
  4 * side = 40 →
  side^2 = multiple * 40 + 20 →
  multiple = 2 := by sorry

end NUMINAMATH_CALUDE_square_garden_multiple_l2883_288356


namespace NUMINAMATH_CALUDE_parabola_with_vertex_two_three_l2883_288364

/-- A parabola with vertex (h, k) has the general form y = a(x - h)² + k, where a ≠ 0 -/
def is_parabola (f : ℝ → ℝ) (h k a : ℝ) : Prop :=
  ∀ x, f x = a * (x - h)^2 + k ∧ a ≠ 0

/-- The vertex of a parabola f is the point (h, k) -/
def has_vertex (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∃ a : ℝ, is_parabola f h k a

theorem parabola_with_vertex_two_three (f : ℝ → ℝ) :
  has_vertex f 2 3 → (∀ x, f x = -(x - 2)^2 + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_with_vertex_two_three_l2883_288364


namespace NUMINAMATH_CALUDE_holly_weekly_pill_count_l2883_288397

/-- Calculates the total number of pills Holly takes in a week -/
def weekly_pill_count (insulin_per_day : ℕ) (blood_pressure_per_day : ℕ) : ℕ :=
  let anticonvulsants_per_day := 2 * blood_pressure_per_day
  let daily_total := insulin_per_day + blood_pressure_per_day + anticonvulsants_per_day
  7 * daily_total

/-- Proves that Holly takes 77 pills in a week given her daily requirements -/
theorem holly_weekly_pill_count : 
  weekly_pill_count 2 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_holly_weekly_pill_count_l2883_288397


namespace NUMINAMATH_CALUDE_book_distribution_l2883_288328

theorem book_distribution (x : ℕ) : 
  (∃ n : ℕ, x = 5 * n + 6) ∧ 
  (1 ≤ x - 7 * ((x - 6) / 5 - 1) ∧ x - 7 * ((x - 6) / 5 - 1) < 7) ↔ 
  (1 ≤ x - 7 * ((x - 6) / 5 - 1) ∧ x - 7 * ((x - 6) / 5 - 1) < 7) :=
sorry

end NUMINAMATH_CALUDE_book_distribution_l2883_288328


namespace NUMINAMATH_CALUDE_inequality_proof_l2883_288310

theorem inequality_proof (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ x₅) (h5 : x₅ ≥ 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 25/2 * (x₄^2 + x₅^2) ∧
  ((x₁ + x₂ + x₃ + x₄ + x₅)^2 = 25/2 * (x₄^2 + x₅^2) ↔ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2883_288310


namespace NUMINAMATH_CALUDE_inverse_propositions_l2883_288382

-- Definitions for geometric concepts
def Line : Type := sorry
def Angle : Type := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def corresponding_angles_equal (l1 l2 : Line) : Prop := sorry

-- Definition for last digit
def last_digit (n : ℕ) : ℕ := n % 10

theorem inverse_propositions :
  -- 1. If two lines are parallel, then the corresponding angles are equal
  (∀ (l1 l2 : Line), parallel l1 l2 → corresponding_angles_equal l1 l2) ∧
  -- 2. There exist a and b such that a² = b² but a ≠ b
  (∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b) ∧
  -- 3. There exists a number divisible by 5 whose last digit is not 0
  (∃ (n : ℕ), n % 5 = 0 ∧ last_digit n ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_inverse_propositions_l2883_288382


namespace NUMINAMATH_CALUDE_midpoint_square_area_l2883_288350

theorem midpoint_square_area (A : ℝ) (h : A = 144) : 
  let s := Real.sqrt A
  let midpoint_side := Real.sqrt ((s/2)^2 + (s/2)^2)
  let midpoint_area := midpoint_side^2
  midpoint_area = 72 := by sorry

end NUMINAMATH_CALUDE_midpoint_square_area_l2883_288350


namespace NUMINAMATH_CALUDE_salt_solution_problem_l2883_288363

theorem salt_solution_problem (initial_weight : ℝ) (added_salt : ℝ) (final_percentage : ℝ) :
  initial_weight = 60 →
  added_salt = 3 →
  final_percentage = 25 →
  let final_weight := initial_weight + added_salt
  let final_salt := (final_percentage / 100) * final_weight
  let initial_salt := final_salt - added_salt
  initial_salt / initial_weight * 100 = 21.25 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_problem_l2883_288363


namespace NUMINAMATH_CALUDE_combined_mpg_l2883_288326

/-- The combined miles per gallon of two cars given their individual mpg and relative distances driven -/
theorem combined_mpg (ray_mpg tom_mpg : ℝ) (h1 : ray_mpg = 48) (h2 : tom_mpg = 24) : 
  let s : ℝ := 1  -- Tom's distance (arbitrary non-zero value)
  let ray_distance := 2 * s
  let tom_distance := s
  let total_distance := ray_distance + tom_distance
  let total_fuel := ray_distance / ray_mpg + tom_distance / tom_mpg
  total_distance / total_fuel = 36 := by
sorry


end NUMINAMATH_CALUDE_combined_mpg_l2883_288326


namespace NUMINAMATH_CALUDE_rearrangement_writing_time_l2883_288315

/-- The number of distinct letters in the name --/
def name_length : ℕ := 7

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 15

/-- The total number of minutes required to write all rearrangements --/
def total_minutes : ℕ := 336

/-- Theorem stating that the total time to write all rearrangements of a 7-letter name
    at a rate of 15 rearrangements per minute is 336 minutes --/
theorem rearrangement_writing_time :
  (Nat.factorial name_length) / rearrangements_per_minute = total_minutes := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_writing_time_l2883_288315


namespace NUMINAMATH_CALUDE_remove_matches_no_rectangle_l2883_288376

-- Define the structure of the grid
def Grid := List (List Bool)

-- Define a function to check if a grid contains a rectangle
def containsRectangle (grid : Grid) : Bool := sorry

-- Define the initial 4x4 grid
def initialGrid : Grid := sorry

-- Define a function to remove matches from the grid
def removeMatches (grid : Grid) (numToRemove : Nat) : Grid := sorry

-- Theorem statement
theorem remove_matches_no_rectangle :
  ∃ (finalGrid : Grid),
    (removeMatches initialGrid 11 = finalGrid) ∧
    (containsRectangle finalGrid = false) := by
  sorry

end NUMINAMATH_CALUDE_remove_matches_no_rectangle_l2883_288376


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2883_288373

theorem no_real_roots_quadratic (b : ℝ) : ∀ x : ℝ, x^2 - b*x + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2883_288373


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l2883_288324

-- Define the problem statement
theorem composite_sum_of_powers (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a^2016 + b^2016 + c^2016 + d^2016 = m * n :=
by sorry


end NUMINAMATH_CALUDE_composite_sum_of_powers_l2883_288324


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2883_288322

theorem quadratic_root_difference (r₁ r₂ : ℝ) : 
  2 * r₁^2 - 10 * r₁ + 2 = 0 ∧
  2 * r₂^2 - 10 * r₂ + 2 = 0 ∧
  r₁^2 + r₂^2 = 23 →
  |r₁ - r₂| = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2883_288322


namespace NUMINAMATH_CALUDE_four_integers_average_l2883_288303

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (w + x + y + z : ℚ) / 4 = 5 →
  (max a (max b (max c d)) - min a (min b (min c d)) : ℤ) ≥ 
  (max w (max x (max y z)) - min w (min x (min y z)) : ℤ) →
  (a + b + c + d - max a (max b (max c d)) - min a (min b (min c d)) : ℚ) / 2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_average_l2883_288303


namespace NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l2883_288312

/-- Represents the scenario where a person is between two points -/
structure WalkRideScenario where
  /-- Distance from the person to the apartment -/
  dist_to_apartment : ℝ
  /-- Distance from the person to the library -/
  dist_to_library : ℝ
  /-- Walking speed -/
  walking_speed : ℝ
  /-- Assumption that distances and speed are positive -/
  dist_apartment_pos : 0 < dist_to_apartment
  dist_library_pos : 0 < dist_to_library
  speed_pos : 0 < walking_speed
  /-- Assumption that the person is between the apartment and library -/
  between_points : dist_to_apartment + dist_to_library > 0

/-- The theorem stating that under the given conditions, the ratio of distances is 2/3 -/
theorem distance_ratio_is_two_thirds (scenario : WalkRideScenario) :
  scenario.dist_to_library / scenario.walking_speed =
  scenario.dist_to_apartment / scenario.walking_speed +
  (scenario.dist_to_apartment + scenario.dist_to_library) / (5 * scenario.walking_speed) →
  scenario.dist_to_apartment / scenario.dist_to_library = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l2883_288312


namespace NUMINAMATH_CALUDE_product_inequality_l2883_288336

theorem product_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) ≥ (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2883_288336


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2883_288325

theorem smallest_solution_of_equation :
  let f (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x
  ∃ (smallest : ℝ), smallest = (2 - Real.sqrt 31) / 3 ∧
    f smallest = 16 ∧
    ∀ (y : ℝ), y ≠ 3 ∧ y ≠ 0 ∧ f y = 16 → y ≥ smallest :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2883_288325


namespace NUMINAMATH_CALUDE_inequality_proof_l2883_288342

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_equality : c^2 + a*b = a^2 + b^2) : 
  c^2 + a*b ≤ a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2883_288342


namespace NUMINAMATH_CALUDE_inequality_implies_a_geq_4_l2883_288340

theorem inequality_implies_a_geq_4 (a : ℝ) (h_a_pos : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) →
  a ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_geq_4_l2883_288340


namespace NUMINAMATH_CALUDE_prime_pair_sum_both_prime_prime_pair_product_l2883_288362

/-- Two prime numbers that sum to 101 -/
def prime_pair : (ℕ × ℕ) := sorry

/-- The sum of the prime pair is 101 -/
theorem prime_pair_sum : prime_pair.1 + prime_pair.2 = 101 := sorry

/-- Both numbers in the pair are prime -/
theorem both_prime : 
  Nat.Prime prime_pair.1 ∧ Nat.Prime prime_pair.2 := sorry

/-- The product of the prime pair is 194 -/
theorem prime_pair_product : 
  prime_pair.1 * prime_pair.2 = 194 := sorry

end NUMINAMATH_CALUDE_prime_pair_sum_both_prime_prime_pair_product_l2883_288362


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2883_288349

-- Problem 1
theorem problem_1 : -1^2009 + Real.rpow 27 (1/3) - |1 - Real.sqrt 2| + Real.sqrt 8 = 3 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  y / x + x / y + 2 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2883_288349


namespace NUMINAMATH_CALUDE_interior_angles_increase_l2883_288333

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 2340 → sum_interior_angles (n + 4) = 3060 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_increase_l2883_288333


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2883_288370

theorem complex_sum_problem (a b c d e f g h : ℂ) :
  b = 2 ∧ 
  g = -a - c - e ∧ 
  a + b * Complex.I + c + d * Complex.I + e + f * Complex.I + g + h * Complex.I = -3 * Complex.I →
  d + f + h = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2883_288370


namespace NUMINAMATH_CALUDE_hex_to_decimal_conversion_l2883_288391

/-- Given a hexadecimal number m02₍₆₎ that is equivalent to 146 in decimal, 
    prove that m = 4. -/
theorem hex_to_decimal_conversion (m : ℕ) : 
  (2 + m * 6^2 = 146) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_decimal_conversion_l2883_288391


namespace NUMINAMATH_CALUDE_kingfisher_pelican_fish_difference_l2883_288385

theorem kingfisher_pelican_fish_difference (pelican_fish : ℕ) (kingfisher_fish : ℕ) (fisherman_fish : ℕ) : 
  pelican_fish = 13 →
  kingfisher_fish > pelican_fish →
  fisherman_fish = 3 * (pelican_fish + kingfisher_fish) →
  fisherman_fish = pelican_fish + 86 →
  kingfisher_fish - pelican_fish = 7 := by
sorry

end NUMINAMATH_CALUDE_kingfisher_pelican_fish_difference_l2883_288385


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2883_288378

theorem repeating_decimal_to_fraction : 
  ∃ (n : ℚ), n = 7 + 123 / 999 ∧ n = 593 / 111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2883_288378


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2883_288399

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m : ℝ) : Prop :=
  3 * m + (2 * m - 1) * m = 0

/-- The condition m = -1 is sufficient for the lines to be perpendicular -/
theorem sufficient_condition (m : ℝ) :
  m = -1 → are_perpendicular m :=
by sorry

/-- The condition m = -1 is not necessary for the lines to be perpendicular -/
theorem not_necessary_condition :
  ∃ m : ℝ, m ≠ -1 ∧ are_perpendicular m :=
by sorry

/-- The condition m = -1 is sufficient but not necessary for the lines to be perpendicular -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m = -1 → are_perpendicular m) ∧
  (∃ m : ℝ, m ≠ -1 ∧ are_perpendicular m) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2883_288399


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2883_288395

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2883_288395


namespace NUMINAMATH_CALUDE_tan_alpha_values_l2883_288337

theorem tan_alpha_values (α : Real) 
  (h : 2 * Real.sin α ^ 2 + Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 7/5) : 
  Real.tan α = 2 ∨ Real.tan α = -11/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l2883_288337


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2883_288330

def number_list : List ℝ := [3, 0, -5, 0.48, -(-7), -|(-8)|, -((-4)^2)]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2883_288330


namespace NUMINAMATH_CALUDE_blueberry_baskets_l2883_288368

theorem blueberry_baskets (initial_berries : ℕ) (total_berries : ℕ) : 
  initial_berries = 20 →
  total_berries = 200 →
  (total_berries / initial_berries) - 1 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_blueberry_baskets_l2883_288368


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2883_288314

/-- Given a principal amount P and an interest rate r, 
    if P(1 + 2r) = 710 and P(1 + 7r) = 1020, then P = 586 -/
theorem simple_interest_problem (P r : ℝ) 
  (h1 : P * (1 + 2 * r) = 710)
  (h2 : P * (1 + 7 * r) = 1020) : 
  P = 586 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2883_288314


namespace NUMINAMATH_CALUDE_base_5_representation_of_89_l2883_288339

def to_base_5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: to_base_5 (n / 5)

theorem base_5_representation_of_89 :
  to_base_5 89 = [4, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_base_5_representation_of_89_l2883_288339


namespace NUMINAMATH_CALUDE_solution_in_interval_l2883_288319

theorem solution_in_interval : ∃ x₀ : ℝ, (Real.exp x₀ + x₀ = 2) ∧ (0 < x₀ ∧ x₀ < 1) := by sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2883_288319


namespace NUMINAMATH_CALUDE_triangle_area_quadrilateral_area_n_gon_area_l2883_288357

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def point (m i : ℕ) : ℝ × ℝ :=
  (fibonacci (m + 2 * i - 1), fibonacci (m + 2 * i))

def polygon_area (n m : ℕ) : ℝ :=
  let vertices := List.range n |>.map (point m)
  -- Area calculation using Shoelace formula
  sorry

theorem triangle_area (m : ℕ) :
  polygon_area 3 m = 0.5 := by sorry

theorem quadrilateral_area (m : ℕ) :
  polygon_area 4 m = 2.5 := by sorry

theorem n_gon_area (n m : ℕ) (h : n ≥ 3) :
  polygon_area n m = (fibonacci (2 * n - 2) - n + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_quadrilateral_area_n_gon_area_l2883_288357


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l2883_288381

theorem arcsin_equation_solution :
  ∀ x : ℝ, Real.arcsin x + Real.arcsin (3 * x) = π / 2 →
  x = 1 / Real.sqrt 10 ∨ x = -(1 / Real.sqrt 10) := by
sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l2883_288381


namespace NUMINAMATH_CALUDE_optimal_probability_l2883_288369

-- Define the probability of success for a single shot
variable (p : ℝ)

-- Define the number of successful shots as a random variable
def X : ℕ → ℝ
  | n => p^n * (1 - p)

-- Define the probability of making between 35 and 69 shots
def P_35_to_69 (p : ℝ) : ℝ :=
  p^35 - p^70

-- State the theorem
theorem optimal_probability :
  ∃ (p : ℝ), p > 0 ∧ p < 1 ∧
  (∀ q : ℝ, q > 0 → q < 1 → P_35_to_69 q ≤ P_35_to_69 p) ∧
  p = (1/2)^(1/35) :=
sorry

end NUMINAMATH_CALUDE_optimal_probability_l2883_288369


namespace NUMINAMATH_CALUDE_complex_multiplication_l2883_288386

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2883_288386


namespace NUMINAMATH_CALUDE_count_seven_up_to_2017_l2883_288354

/-- Count of digit 7 in a natural number -/
def count_seven (n : ℕ) : ℕ := sorry

/-- Sum of count_seven for all numbers from 1 to n -/
def sum_count_seven (n : ℕ) : ℕ := sorry

theorem count_seven_up_to_2017 : sum_count_seven 2017 = 602 := by sorry

end NUMINAMATH_CALUDE_count_seven_up_to_2017_l2883_288354


namespace NUMINAMATH_CALUDE_total_tickets_l2883_288365

/-- The number of tickets Dave used to buy toys -/
def tickets_for_toys : ℕ := 12

/-- The number of tickets Dave used to buy clothes -/
def tickets_for_clothes : ℕ := 7

/-- The difference between tickets used for toys and clothes -/
def tickets_difference : ℕ := 5

/-- Theorem: Given the conditions, Dave won 19 tickets in total -/
theorem total_tickets : 
  tickets_for_toys + tickets_for_clothes = 19 ∧
  tickets_for_toys = tickets_for_clothes + tickets_difference :=
sorry

end NUMINAMATH_CALUDE_total_tickets_l2883_288365


namespace NUMINAMATH_CALUDE_f_two_expression_l2883_288304

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (f 1 = 2) ∧ 
  (∀ x y : ℝ, f (x * y + f x + 1) = x * f y + f x)

/-- The main theorem stating that f(2) can be expressed as c + 2 -/
theorem f_two_expression 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) :
  ∃ c : ℝ, f 2 = c + 2 :=
sorry

end NUMINAMATH_CALUDE_f_two_expression_l2883_288304


namespace NUMINAMATH_CALUDE_inverse_difference_inverse_l2883_288345

theorem inverse_difference_inverse (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - z⁻¹)⁻¹ = x * z / (z - x) :=
by sorry

end NUMINAMATH_CALUDE_inverse_difference_inverse_l2883_288345


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2883_288320

/-- 
Given two infinite geometric series:
- First series with first term a₁ = 8 and second term b₁ = 2
- Second series with first term a₂ = 8 and second term b₂ = 2 + m
If the sum of the second series is three times the sum of the first series,
then m = 4.
-/
theorem geometric_series_ratio (m : ℝ) : 
  let a₁ : ℝ := 8
  let b₁ : ℝ := 2
  let a₂ : ℝ := 8
  let b₂ : ℝ := 2 + m
  let r₁ : ℝ := b₁ / a₁
  let r₂ : ℝ := b₂ / a₂
  let s₁ : ℝ := a₁ / (1 - r₁)
  let s₂ : ℝ := a₂ / (1 - r₂)
  s₂ = 3 * s₁ → m = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_series_ratio_l2883_288320


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_100000_l2883_288301

def sequence_term (n : ℕ) : ℕ := 9 + 10 * (n - 1)

def sequence_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => sequence_term (i + 1))

theorem smallest_n_exceeding_100000 : 
  (∀ k < 142, sequence_sum k ≤ 100000) ∧ 
  sequence_sum 142 > 100000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_100000_l2883_288301


namespace NUMINAMATH_CALUDE_max_sum_of_three_integers_with_product_24_l2883_288367

theorem max_sum_of_three_integers_with_product_24 :
  (∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧
    ∀ (x y z : ℕ+), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 24 →
      x + y + z ≤ a + b + c) ∧
  (∀ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 →
    a + b + c ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_integers_with_product_24_l2883_288367


namespace NUMINAMATH_CALUDE_chessboard_coloring_l2883_288344

/-- A move on the chessboard changes the color of all squares in a 2x2 area. -/
def ChessboardMove (n : ℕ) := Fin n → Fin n → Bool

/-- The initial chessboard coloring. -/
def InitialChessboard (n : ℕ) : Fin n → Fin n → Bool :=
  λ i j => (i.val + j.val) % 2 = 0

/-- A sequence of moves. -/
def MoveSequence (n : ℕ) := List (ChessboardMove n)

/-- Apply a move to the chessboard. -/
def ApplyMove (board : Fin n → Fin n → Bool) (move : ChessboardMove n) : Fin n → Fin n → Bool :=
  λ i j => board i j ≠ move i j

/-- Apply a sequence of moves to the chessboard. -/
def ApplyMoveSequence (n : ℕ) (board : Fin n → Fin n → Bool) (moves : MoveSequence n) : Fin n → Fin n → Bool :=
  moves.foldl ApplyMove board

/-- Check if all squares on the board have the same color. -/
def AllSameColor (board : Fin n → Fin n → Bool) : Prop :=
  ∀ i j k l, board i j = board k l

/-- Main theorem: There exists a finite sequence of moves that turns all squares
    the same color if and only if n is divisible by 4. -/
theorem chessboard_coloring (n : ℕ) (h : n ≥ 3) :
  (∃ (moves : MoveSequence n), AllSameColor (ApplyMoveSequence n (InitialChessboard n) moves)) ↔
  4 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_chessboard_coloring_l2883_288344
