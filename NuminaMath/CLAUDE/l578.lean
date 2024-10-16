import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l578_57817

theorem division_problem : 180 / (12 + 13 * 2) = 45 / 19 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l578_57817


namespace NUMINAMATH_CALUDE_max_sum_on_integer_circle_l578_57895

theorem max_sum_on_integer_circle : 
  ∀ x y : ℤ, x^2 + y^2 = 100 → (∀ a b : ℤ, a^2 + b^2 = 100 → x + y ≥ a + b) → x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_integer_circle_l578_57895


namespace NUMINAMATH_CALUDE_other_diagonal_length_l578_57814

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with area 330 and one diagonal 30, the other diagonal is 22 -/
theorem other_diagonal_length :
  ∀ (r : Rhombus), r.area = 330 ∧ r.d1 = 30 → r.d2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l578_57814


namespace NUMINAMATH_CALUDE_triangle_max_area_l578_57879

theorem triangle_max_area (A B C : Real) (a b c : Real) :
  -- Triangle ABC with circumradius 1
  (a / Real.sin A = 2) ∧ (b / Real.sin B = 2) ∧ (c / Real.sin C = 2) →
  -- Given condition
  (Real.tan A) / (Real.tan B) = (2 * c - b) / b →
  -- Theorem: Maximum area is √3/2
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
                S ≤ Real.sqrt 3 / 2 ∧
                (∃ (B' C' : Real), S = Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l578_57879


namespace NUMINAMATH_CALUDE_horner_method_v2_l578_57858

def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

def horner_v2 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := 2*x + 1
  v1 * x

theorem horner_method_v2 : horner_v2 2 = 10 := by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l578_57858


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l578_57816

/-- Calculates the total cost of apples for Irene and her dog for 2 weeks -/
def apple_cost (apple_weight : Real) (red_price : Real) (green_price : Real) 
  (red_increase : Real) (green_decrease : Real) : Real :=
  let apples_needed := 14 * 0.5
  let pounds_needed := apples_needed * apple_weight
  let week1_cost := pounds_needed * red_price
  let week2_cost := pounds_needed * (green_price * (1 - green_decrease))
  week1_cost + week2_cost

theorem apple_cost_calculation :
  apple_cost (1/4) 2 2.5 0.1 0.05 = 7.65625 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l578_57816


namespace NUMINAMATH_CALUDE_fair_game_l578_57860

-- Define the deck
def deck_size : ℕ := 52

-- Define the number of black aces
def black_aces : ℕ := 2

-- Define the game outcome
inductive GameOutcome
| Player1Wins
| Player2Wins
| Tie

-- Define a function to calculate the number of possible outcomes
def num_outcomes : ℕ := (deck_size * (deck_size - 1)) / 2

-- Define a function to calculate the number of tie outcomes
def num_tie_outcomes : ℕ := deck_size - 1

-- Define a function to calculate the number of winning outcomes for each player
def num_winning_outcomes : ℕ := (num_outcomes - num_tie_outcomes) / 2

-- Theorem statement
theorem fair_game :
  num_winning_outcomes = num_winning_outcomes ∧
  num_winning_outcomes + num_winning_outcomes + num_tie_outcomes = num_outcomes :=
sorry

end NUMINAMATH_CALUDE_fair_game_l578_57860


namespace NUMINAMATH_CALUDE_x_value_proof_l578_57837

theorem x_value_proof : ∃ X : ℝ, 
  X * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  abs (X - 0.6) < 0.0000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l578_57837


namespace NUMINAMATH_CALUDE_saturday_to_monday_ratio_is_two_to_one_l578_57882

/-- Represents Mona's weekly biking schedule -/
structure BikeSchedule where
  total_distance : ℕ
  monday_distance : ℕ
  wednesday_distance : ℕ
  saturday_distance : ℕ
  total_eq : total_distance = monday_distance + wednesday_distance + saturday_distance

/-- Calculates the ratio of Saturday's distance to Monday's distance -/
def saturday_to_monday_ratio (schedule : BikeSchedule) : ℚ :=
  schedule.saturday_distance / schedule.monday_distance

/-- Theorem stating that the ratio of Saturday's distance to Monday's distance is 2:1 -/
theorem saturday_to_monday_ratio_is_two_to_one (schedule : BikeSchedule)
  (h1 : schedule.total_distance = 30)
  (h2 : schedule.monday_distance = 6)
  (h3 : schedule.wednesday_distance = 12) :
  saturday_to_monday_ratio schedule = 2 := by
  sorry

#eval saturday_to_monday_ratio {
  total_distance := 30,
  monday_distance := 6,
  wednesday_distance := 12,
  saturday_distance := 12,
  total_eq := by rfl
}

end NUMINAMATH_CALUDE_saturday_to_monday_ratio_is_two_to_one_l578_57882


namespace NUMINAMATH_CALUDE_sine_shift_right_l578_57824

/-- Shifting a sine function to the right by π/6 units -/
theorem sine_shift_right (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t + π / 6)
  let g (t : ℝ) := f (t - π / 6)
  g x = Real.sin (2 * x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sine_shift_right_l578_57824


namespace NUMINAMATH_CALUDE_number_with_given_quotient_and_remainder_l578_57859

theorem number_with_given_quotient_and_remainder : ∃ n : ℕ, n = 58 ∧ n / 9 = 6 ∧ n % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_with_given_quotient_and_remainder_l578_57859


namespace NUMINAMATH_CALUDE_boxer_weight_theorem_l578_57893

def initial_weight : ℝ := 106

def weight_loss_rate_A1 : ℝ := 2
def weight_loss_rate_A2 : ℝ := 3
def weight_loss_duration_A1 : ℝ := 2
def weight_loss_duration_A2 : ℝ := 2

def weight_loss_rate_B : ℝ := 3
def weight_loss_duration_B : ℝ := 3

def weight_loss_rate_C : ℝ := 4
def weight_loss_duration_C : ℝ := 4

def final_weight_A : ℝ := initial_weight - (weight_loss_rate_A1 * weight_loss_duration_A1 + weight_loss_rate_A2 * weight_loss_duration_A2)
def final_weight_B : ℝ := initial_weight - (weight_loss_rate_B * weight_loss_duration_B)
def final_weight_C : ℝ := initial_weight - (weight_loss_rate_C * weight_loss_duration_C)

theorem boxer_weight_theorem :
  final_weight_A = 96 ∧
  final_weight_B = 97 ∧
  final_weight_C = 90 := by
  sorry

end NUMINAMATH_CALUDE_boxer_weight_theorem_l578_57893


namespace NUMINAMATH_CALUDE_fraction_simplification_l578_57839

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l578_57839


namespace NUMINAMATH_CALUDE_tan_equation_solutions_l578_57841

open Real

noncomputable def S (x : ℝ) := tan x + x

theorem tan_equation_solutions :
  let a := arctan 500
  ∃ (sols : Finset ℝ), (∀ x ∈ sols, 0 ≤ x ∧ x ≤ a ∧ tan x = tan (S x)) ∧ Finset.card sols = 160 :=
sorry

end NUMINAMATH_CALUDE_tan_equation_solutions_l578_57841


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l578_57875

theorem certain_amount_calculation (x A : ℝ) (h1 : x = 170) (h2 : 0.65 * x = 0.2 * A) : A = 552.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l578_57875


namespace NUMINAMATH_CALUDE_min_value_of_expression_exists_min_value_l578_57869

theorem min_value_of_expression (x : ℚ) : (2*x - 5)^2 + 18 ≥ 18 :=
sorry

theorem exists_min_value : ∃ x : ℚ, (2*x - 5)^2 + 18 = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_exists_min_value_l578_57869


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_iff_m_less_than_one_l578_57885

/-- A point P(x, y) is in the third quadrant if both x and y are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The x-coordinate of point P as a function of m -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P as a function of m -/
def y_coord (m : ℝ) : ℝ := 2 * m - 3

/-- Theorem stating that for point P(m-1, 2m-3) to be in the third quadrant, m must be less than 1 -/
theorem point_in_third_quadrant_iff_m_less_than_one (m : ℝ) : 
  in_third_quadrant (x_coord m) (y_coord m) ↔ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_iff_m_less_than_one_l578_57885


namespace NUMINAMATH_CALUDE_seashells_given_l578_57836

theorem seashells_given (initial_seashells current_seashells : ℕ) 
  (h1 : initial_seashells = 5)
  (h2 : current_seashells = 3) : 
  initial_seashells - current_seashells = 2 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_l578_57836


namespace NUMINAMATH_CALUDE_other_diagonal_length_l578_57854

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diagonal1 * r.diagonal2) / 2

theorem other_diagonal_length :
  ∀ r : Rhombus, r.diagonal1 = 12 ∧ r.area = 60 → r.diagonal2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l578_57854


namespace NUMINAMATH_CALUDE_sine_function_parameters_l578_57848

theorem sine_function_parameters
  (A ω m : ℝ)
  (h_A_pos : A > 0)
  (h_ω_pos : ω > 0)
  (h_max : ∀ x, A * Real.sin (ω * x + π / 6) + m ≤ 3)
  (h_min : ∀ x, A * Real.sin (ω * x + π / 6) + m ≥ -5)
  (h_max_achieved : ∃ x, A * Real.sin (ω * x + π / 6) + m = 3)
  (h_min_achieved : ∃ x, A * Real.sin (ω * x + π / 6) + m = -5)
  (h_symmetry : ω * (π / 2) = π) :
  A = 4 ∧ ω = 2 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_parameters_l578_57848


namespace NUMINAMATH_CALUDE_board_meeting_arrangement_l578_57828

/-- The number of ways to arrange 3 indistinguishable objects among 8 positions -/
def arrangement_count : ℕ := 56

/-- The total number of seats -/
def total_seats : ℕ := 10

/-- The number of stools (men) -/
def stool_count : ℕ := 5

/-- The number of rocking chairs (women) -/
def chair_count : ℕ := 5

/-- The number of positions to fill after fixing first and last seats -/
def remaining_positions : ℕ := total_seats - 2

/-- The number of remaining stools to place after fixing first and last seats -/
def remaining_stools : ℕ := stool_count - 2

theorem board_meeting_arrangement :
  arrangement_count = Nat.choose remaining_positions remaining_stools := by
  sorry

end NUMINAMATH_CALUDE_board_meeting_arrangement_l578_57828


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l578_57881

/-- The length of a diagonal in a quadrilateral with given area and offsets -/
theorem diagonal_length_of_quadrilateral (area : ℝ) (offset1 offset2 : ℝ) 
  (h_area : area = 140)
  (h_offset1 : offset1 = 8)
  (h_offset2 : offset2 = 2)
  (h_quad_area : area = (1/2) * (offset1 + offset2) * diagonal_length) :
  diagonal_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l578_57881


namespace NUMINAMATH_CALUDE_percentage_problem_l578_57864

theorem percentage_problem (n : ℝ) : (0.1 * 0.3 * 0.5 * n = 90) → n = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l578_57864


namespace NUMINAMATH_CALUDE_age_sum_problem_l578_57826

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a * b * c = 72 → a + b + c = 14 := by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l578_57826


namespace NUMINAMATH_CALUDE_sum_zero_fraction_l578_57843

theorem sum_zero_fraction (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_sum : x + y + z = 0) : 
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_fraction_l578_57843


namespace NUMINAMATH_CALUDE_mango_tree_columns_count_l578_57810

/-- The number of columns of mango trees in a garden with given dimensions -/
def mango_tree_columns (garden_length : ℕ) (tree_distance : ℕ) (boundary_distance : ℕ) : ℕ :=
  let available_length := garden_length - 2 * boundary_distance
  let spaces := available_length / tree_distance
  spaces + 1

/-- Theorem stating that the number of mango tree columns is 12 given the specified conditions -/
theorem mango_tree_columns_count :
  mango_tree_columns 32 2 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mango_tree_columns_count_l578_57810


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l578_57842

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l578_57842


namespace NUMINAMATH_CALUDE_division_problem_l578_57834

theorem division_problem (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 
  11 * y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l578_57834


namespace NUMINAMATH_CALUDE_odd_function_inequality_l578_57897

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem odd_function_inequality (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≠ 0) →
  is_decreasing_on f (-2) 0 →
  f (1 - m) + f (1 - m^2) < 0 →
  -1 ≤ m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l578_57897


namespace NUMINAMATH_CALUDE_water_consumption_calculation_l578_57812

/-- Water billing system and consumption calculation -/
theorem water_consumption_calculation 
  (base_rate : ℝ) 
  (excess_rate : ℝ) 
  (sewage_rate : ℝ) 
  (base_volume : ℝ) 
  (total_bill : ℝ) 
  (h1 : base_rate = 1.8) 
  (h2 : excess_rate = 2.3) 
  (h3 : sewage_rate = 1) 
  (h4 : base_volume = 15) 
  (h5 : total_bill = 58.5) : 
  ∃ (consumption : ℝ), 
    consumption = 20 ∧ 
    total_bill = 
      base_rate * min consumption base_volume + 
      excess_rate * max (consumption - base_volume) 0 + 
      sewage_rate * consumption :=
sorry

end NUMINAMATH_CALUDE_water_consumption_calculation_l578_57812


namespace NUMINAMATH_CALUDE_current_trees_proof_current_trees_is_25_l578_57851

/-- The number of popular trees currently in the park -/
def current_trees : ℕ := sorry

/-- The number of popular trees to be planted today -/
def trees_to_plant : ℕ := 73

/-- The total number of popular trees after planting -/
def total_trees : ℕ := 98

/-- Theorem stating that the current number of trees plus the trees to be planted equals the total trees after planting -/
theorem current_trees_proof : current_trees + trees_to_plant = total_trees := by sorry

/-- Theorem proving that the number of current trees is 25 -/
theorem current_trees_is_25 : current_trees = 25 := by sorry

end NUMINAMATH_CALUDE_current_trees_proof_current_trees_is_25_l578_57851


namespace NUMINAMATH_CALUDE_article_cost_price_l578_57830

theorem article_cost_price (C : ℝ) : C = 400 :=
  by
  have h1 : 1.05 * C - 2 = 0.95 * C * 1.10 := by sorry
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l578_57830


namespace NUMINAMATH_CALUDE_inequality_proof_l578_57892

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) ≤ a + b + c) ∧
  (a + b + c = 1 → (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * c * a) / (c + a) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l578_57892


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l578_57832

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the configuration of a room with a sloped ceiling -/
structure Room where
  p : Point3D -- Point where walls and ceiling meet
  slope : ℝ -- Slope of the ceiling (rise / run)

/-- Represents the position of a fly in the room -/
structure FlyPosition where
  distWall1 : ℝ -- Distance from first wall
  distWall2 : ℝ -- Distance from second wall
  distFromP : ℝ -- Distance from point P

/-- Calculates the distance of a fly from the sloped ceiling in a room -/
def distanceFromCeiling (r : Room) (f : FlyPosition) : ℝ :=
  sorry

/-- Theorem stating that the fly's distance from the ceiling is (3√60 - 8)/3 -/
theorem fly_distance_from_ceiling (r : Room) (f : FlyPosition) :
  r.p = Point3D.mk 0 0 0 →
  r.slope = 1/3 →
  f.distWall1 = 2 →
  f.distWall2 = 6 →
  f.distFromP = 10 →
  distanceFromCeiling r f = (3 * Real.sqrt 60 - 8) / 3 :=
sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l578_57832


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l578_57820

/-- 
Given a man and his son, where:
- The man is 28 years older than his son
- The son's present age is 26 years
Prove that the ratio of their ages in two years will be 2:1
-/
theorem man_son_age_ratio : 
  ∀ (man_age son_age : ℕ),
  man_age = son_age + 28 →
  son_age = 26 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l578_57820


namespace NUMINAMATH_CALUDE_original_data_set_properties_l578_57807

/-- Represents a data set with its average and variance -/
structure DataSet where
  average : ℝ
  variance : ℝ

/-- The transformation applied to the original data set -/
def decrease_by_80 (d : DataSet) : DataSet :=
  { average := d.average - 80, variance := d.variance }

/-- Theorem stating the relationship between the original and transformed data sets -/
theorem original_data_set_properties (transformed : DataSet)
  (h1 : transformed = decrease_by_80 { average := 81.2, variance := 4.4 })
  (h2 : transformed.average = 1.2)
  (h3 : transformed.variance = 4.4) :
  ∃ (original : DataSet), original.average = 81.2 ∧ original.variance = 4.4 :=
sorry

end NUMINAMATH_CALUDE_original_data_set_properties_l578_57807


namespace NUMINAMATH_CALUDE_sum_distances_less_than_perimeter_l578_57840

/-- A point is inside a triangle if it's in the interior of the triangle -/
def IsInsideTriangle (A B C M : ℝ × ℝ) : Prop := sorry

/-- The distance between two points in ℝ² -/
def distance (P Q : ℝ × ℝ) : ℝ := sorry

/-- The perimeter of a triangle -/
def perimeter (A B C : ℝ × ℝ) : ℝ := distance A B + distance B C + distance C A

theorem sum_distances_less_than_perimeter (A B C M : ℝ × ℝ) 
  (h : IsInsideTriangle A B C M) : 
  distance M A + distance M B + distance M C < perimeter A B C := by
  sorry

end NUMINAMATH_CALUDE_sum_distances_less_than_perimeter_l578_57840


namespace NUMINAMATH_CALUDE_train_speed_problem_l578_57872

/-- Proves that given two trains of equal length 37.5 meters, where the faster train travels
    at 46 km/hr and passes the slower train in 27 seconds, the speed of the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) 
    (h1 : train_length = 37.5)
    (h2 : faster_speed = 46)
    (h3 : passing_time = 27) :
  ∃ slower_speed : ℝ, 
    slower_speed > 0 ∧ 
    slower_speed < faster_speed ∧
    2 * train_length = (faster_speed - slower_speed) * (5/18) * passing_time ∧
    slower_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l578_57872


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l578_57846

theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 8
  let base_radius : ℝ := r * sector_fraction / 2
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * height
  volume = 4.6875 * Real.pi * Real.sqrt 21.9375 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l578_57846


namespace NUMINAMATH_CALUDE_store_pricing_strategy_l578_57803

/-- Calculates the marked price as a percentage of the list price given the purchase discount,
    selling discount, and desired profit percentage. -/
def markedPricePercentage (purchaseDiscount sellingDiscount profitPercentage : ℚ) : ℚ :=
  let costPrice := 1 - purchaseDiscount
  let markupFactor := (1 + profitPercentage) / (1 - sellingDiscount)
  costPrice * markupFactor * 100

/-- Theorem stating that under the given conditions, the marked price should be 121.⅓% of the list price -/
theorem store_pricing_strategy :
  markedPricePercentage (30/100) (25/100) (30/100) = 121 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_strategy_l578_57803


namespace NUMINAMATH_CALUDE_distance_on_parametric_line_l578_57809

/-- Given a line l with parametric equations x = a + t and y = b + t,
    prove that the distance between a point P1 on l (corresponding to parameter t1)
    and the point P(a, b) is √2|t1|. -/
theorem distance_on_parametric_line
  (a b t1 : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let P1 : ℝ × ℝ := (a + t1, b + t1)
  dist P P1 = Real.sqrt 2 * |t1| :=
by sorry

end NUMINAMATH_CALUDE_distance_on_parametric_line_l578_57809


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l578_57813

theorem smallest_prime_divisor_of_sum (p : Nat) :
  Prime p ∧ p ∣ (3^15 + 11^9) ∧ ∀ q < p, Prime q → ¬(q ∣ (3^15 + 11^9)) →
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l578_57813


namespace NUMINAMATH_CALUDE_complex_inequality_l578_57825

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l578_57825


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l578_57827

theorem max_value_sqrt_sum (x : ℝ) (h : -25 ≤ x ∧ x ≤ 25) :
  Real.sqrt (25 + x) + Real.sqrt (25 - x) ≤ 10 ∧
  (Real.sqrt (25 + x) + Real.sqrt (25 - x) = 10 ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l578_57827


namespace NUMINAMATH_CALUDE_project_budget_increase_l578_57894

/-- Proves that the annual increase in budget for project Q is $50,000 -/
theorem project_budget_increase (initial_q initial_v decrease_v : ℕ) 
  (h1 : initial_q = 540000)
  (h2 : initial_v = 780000)
  (h3 : decrease_v = 10000)
  (h4 : ∃ (increase_q : ℕ), initial_q + 4 * increase_q = initial_v - 4 * decrease_v) :
  ∃ (increase_q : ℕ), increase_q = 50000 := by
sorry


end NUMINAMATH_CALUDE_project_budget_increase_l578_57894


namespace NUMINAMATH_CALUDE_b_le_c_for_geometric_l578_57811

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Definition of b_n -/
def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (n + 1) + a (n + 2)

/-- Definition of c_n -/
def c (a : ℕ → ℝ) (n : ℕ) : ℝ := a n + a (n + 3)

/-- Theorem: For a geometric sequence a, b_n ≤ c_n for all n -/
theorem b_le_c_for_geometric (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ n : ℕ, b a n ≤ c a n := by
  sorry

end NUMINAMATH_CALUDE_b_le_c_for_geometric_l578_57811


namespace NUMINAMATH_CALUDE_sqrt_calculation_l578_57808

theorem sqrt_calculation : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l578_57808


namespace NUMINAMATH_CALUDE_sequence_sum_l578_57838

theorem sequence_sum : 
  let seq1 := [2, 13, 24, 35]
  let seq2 := [8, 18, 28, 38, 48]
  let seq3 := [4, 7]
  (seq1.sum + seq2.sum + seq3.sum) = 225 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l578_57838


namespace NUMINAMATH_CALUDE_derivative_tan_cot_l578_57849

open Real

theorem derivative_tan_cot (x : ℝ) (k : ℤ) : 
  (∀ k, x ≠ (2 * k + 1) * π / 2 → deriv tan x = 1 / (cos x)^2) ∧
  (∀ k, x ≠ k * π → deriv cot x = -(1 / (sin x)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_tan_cot_l578_57849


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l578_57853

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + 3

theorem sum_of_max_min_g :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 1 10, g x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 10, g x = max) ∧
    (∀ x ∈ Set.Icc 1 10, min ≤ g x) ∧
    (∃ x ∈ Set.Icc 1 10, g x = min) ∧
    max + min = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l578_57853


namespace NUMINAMATH_CALUDE_intersection_point_existence_l578_57874

theorem intersection_point_existence : ∃ x : ℝ, 1/2 < x ∧ x < 1 ∧ Real.exp x = -2*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_existence_l578_57874


namespace NUMINAMATH_CALUDE_max_b_cubic_function_max_b_value_l578_57833

/-- A cubic function f(x) = ax³ + bx + c -/
def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

/-- The maximum possible value of b in a cubic function f(x) = ax³ + bx + c
    where 0 ≤ f(x) ≤ 1 for all x in [0, 1] -/
theorem max_b_cubic_function :
  ∃ (b_max : ℝ),
    ∀ (a b c : ℝ),
      (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) →
      b ≤ b_max ∧
      ∃ (a' c' : ℝ), ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a' b_max c' x ∧ cubic_function a' b_max c' x ≤ 1 :=
by
  sorry

/-- The maximum possible value of b is 3√3/2 -/
theorem max_b_value : 
  ∃ (b_max : ℝ),
    b_max = 3 * Real.sqrt 3 / 2 ∧
    ∀ (a b c : ℝ),
      (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) →
      b ≤ b_max ∧
      ∃ (a' c' : ℝ), ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a' b_max c' x ∧ cubic_function a' b_max c' x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_max_b_cubic_function_max_b_value_l578_57833


namespace NUMINAMATH_CALUDE_sin_geq_tan_minus_half_tan_cubed_l578_57804

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  Real.sin x ≥ Real.tan x - (1/2) * (Real.tan x)^3 := by
  sorry

end NUMINAMATH_CALUDE_sin_geq_tan_minus_half_tan_cubed_l578_57804


namespace NUMINAMATH_CALUDE_tamtam_yellow_shells_l578_57891

/-- The number of shells Tamtam collected of each color --/
structure ShellCollection where
  total : ℕ
  purple : ℕ
  pink : ℕ
  blue : ℕ
  orange : ℕ

/-- Calculates the number of yellow shells in a collection --/
def yellowShells (s : ShellCollection) : ℕ :=
  s.total - (s.purple + s.pink + s.blue + s.orange)

/-- Tamtam's shell collection --/
def tamtamShells : ShellCollection :=
  { total := 65
    purple := 13
    pink := 8
    blue := 12
    orange := 14 }

/-- Theorem stating that Tamtam collected 18 yellow shells --/
theorem tamtam_yellow_shells : yellowShells tamtamShells = 18 := by
  sorry

end NUMINAMATH_CALUDE_tamtam_yellow_shells_l578_57891


namespace NUMINAMATH_CALUDE_recipe_reduction_recipe_reduction_mixed_numbers_l578_57898

-- Define the original recipe quantities
def flour_original : Rat := 31/4  -- 7 3/4 cups
def sugar_original : Rat := 5/2   -- 2 1/2 cups

-- Define the reduced recipe quantities
def flour_reduced : Rat := 31/12  -- 2 7/12 cups
def sugar_reduced : Rat := 5/6    -- 5/6 cups

-- Theorem to prove the correct reduced quantities
theorem recipe_reduction :
  flour_reduced = (1/3) * flour_original ∧
  sugar_reduced = (1/3) * sugar_original :=
by sorry

-- Helper function to convert rational to mixed number string representation
noncomputable def rat_to_mixed_string (r : Rat) : String :=
  let whole := Int.floor r
  let frac := r - whole
  if frac = 0 then
    s!"{whole}"
  else
    let num := (frac.num : Int)
    let den := (frac.den : Int)
    if whole = 0 then
      s!"{num}/{den}"
    else
      s!"{whole} {num}/{den}"

-- Theorem to prove the correct string representations
theorem recipe_reduction_mixed_numbers :
  rat_to_mixed_string flour_reduced = "2 7/12" ∧
  rat_to_mixed_string sugar_reduced = "5/6" :=
by sorry

end NUMINAMATH_CALUDE_recipe_reduction_recipe_reduction_mixed_numbers_l578_57898


namespace NUMINAMATH_CALUDE_john_participation_count_l578_57802

/-- Represents the possible point values in the archery competition -/
inductive ArcheryPoints
  | first : ArcheryPoints
  | second : ArcheryPoints
  | third : ArcheryPoints
  | fourth : ArcheryPoints

/-- Returns the point value for a given place -/
def pointValue (p : ArcheryPoints) : Nat :=
  match p with
  | ArcheryPoints.first => 11
  | ArcheryPoints.second => 7
  | ArcheryPoints.third => 5
  | ArcheryPoints.fourth => 2

/-- Represents John's participation in the archery competition -/
def JohnParticipation := List ArcheryPoints

/-- Calculates the product of points for a given participation list -/
def productOfPoints (participation : JohnParticipation) : Nat :=
  participation.foldl (fun acc p => acc * pointValue p) 1

/-- Theorem: John participated 7 times given the conditions -/
theorem john_participation_count :
  ∃ (participation : JohnParticipation),
    productOfPoints participation = 38500 ∧ participation.length = 7 :=
by sorry

end NUMINAMATH_CALUDE_john_participation_count_l578_57802


namespace NUMINAMATH_CALUDE_z_pure_imaginary_iff_a_eq_neg_two_l578_57886

-- Define the complex number z as a function of real number a
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

-- Define what it means for a complex number to be purely imaginary
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- Theorem statement
theorem z_pure_imaginary_iff_a_eq_neg_two :
  ∀ a : ℝ, is_pure_imaginary (z a) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_z_pure_imaginary_iff_a_eq_neg_two_l578_57886


namespace NUMINAMATH_CALUDE_bacteria_states_l578_57815

/-- Represents the state of the bacteria population -/
structure BacteriaState where
  red : ℕ
  blue : ℕ

/-- Transformation rules for bacteria interactions -/
inductive Transform : BacteriaState → BacteriaState → Prop where
  | redRed : ∀ r b, Transform ⟨r + 2, b⟩ ⟨r, b + 1⟩
  | blueBlue : ∀ r b, Transform ⟨r, b + 2⟩ ⟨r + 4, b⟩
  | redBlue : ∀ r b, Transform ⟨r + 1, b + 1⟩ ⟨r + 3, b⟩

/-- The set of possible states given an initial state -/
def possibleStates (initial : BacteriaState) : Set BacteriaState :=
  {s | ∃ n : ℕ, s.red + 2 * s.blue = initial.red + initial.blue ∧ s.blue ≤ (initial.red + initial.blue) / 2}

/-- Theorem stating that the set of possible states matches the expected form -/
theorem bacteria_states (initial : BacteriaState) :
  possibleStates initial = 
  {s | ∃ m : ℕ, s = ⟨initial.red + initial.blue - 2 * m, m⟩ ∧ m ≤ (initial.red + initial.blue) / 2} :=
sorry

end NUMINAMATH_CALUDE_bacteria_states_l578_57815


namespace NUMINAMATH_CALUDE_sarah_shopping_theorem_l578_57805

theorem sarah_shopping_theorem (toy_car1 toy_car2_orig scarf_orig beanie gloves book necklace_orig : ℚ)
  (toy_car2_discount scarf_discount beanie_tax necklace_discount : ℚ)
  (remaining : ℚ)
  (h1 : toy_car1 = 12)
  (h2 : toy_car2_orig = 15)
  (h3 : toy_car2_discount = 0.1)
  (h4 : scarf_orig = 10)
  (h5 : scarf_discount = 0.2)
  (h6 : beanie = 14)
  (h7 : beanie_tax = 0.08)
  (h8 : necklace_orig = 20)
  (h9 : necklace_discount = 0.05)
  (h10 : gloves = 12)
  (h11 : book = 15)
  (h12 : remaining = 7) :
  toy_car1 +
  (toy_car2_orig - toy_car2_orig * toy_car2_discount) +
  (scarf_orig - scarf_orig * scarf_discount) +
  (beanie + beanie * beanie_tax) +
  (necklace_orig - necklace_orig * necklace_discount) +
  gloves +
  book +
  remaining = 101.62 := by
sorry

end NUMINAMATH_CALUDE_sarah_shopping_theorem_l578_57805


namespace NUMINAMATH_CALUDE_unique_k_no_solution_l578_57862

theorem unique_k_no_solution (k : ℕ+) : 
  (k = 2) ↔ 
  ∀ m n : ℕ+, m ≠ n → 
    ¬(Nat.lcm m.val n.val - Nat.gcd m.val n.val = k.val * (m.val - n.val)) :=
by sorry

end NUMINAMATH_CALUDE_unique_k_no_solution_l578_57862


namespace NUMINAMATH_CALUDE_factorization_valid_l578_57831

theorem factorization_valid (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l578_57831


namespace NUMINAMATH_CALUDE_sqrt_nine_over_two_simplification_l578_57878

theorem sqrt_nine_over_two_simplification :
  Real.sqrt (9 / 2) = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_over_two_simplification_l578_57878


namespace NUMINAMATH_CALUDE_dining_room_tiles_l578_57866

/-- Calculates the total number of tiles needed for a rectangular room with a border --/
def total_tiles (room_length room_width border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width - 4 * border_width)
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  let inner_tiles := (inner_area + 3) / 4  -- Ceiling division by 4
  border_tiles + inner_tiles

/-- Theorem stating that a 15ft by 18ft room with a 2ft border requires 139 tiles --/
theorem dining_room_tiles : total_tiles 18 15 2 = 139 := by
  sorry

end NUMINAMATH_CALUDE_dining_room_tiles_l578_57866


namespace NUMINAMATH_CALUDE_three_km_to_meters_four_kg_to_grams_l578_57870

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def grams_per_kilogram : ℝ := 1000

-- Theorem for kilometer to meter conversion
theorem three_km_to_meters :
  3 * meters_per_kilometer = 3000 := by sorry

-- Theorem for kilogram to gram conversion
theorem four_kg_to_grams :
  4 * grams_per_kilogram = 4000 := by sorry

end NUMINAMATH_CALUDE_three_km_to_meters_four_kg_to_grams_l578_57870


namespace NUMINAMATH_CALUDE_high_correlation_implies_r_near_one_l578_57880

-- Define the correlation coefficient
def correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define what it means for a correlation to be "very high"
def is_very_high_correlation (r : ℝ) : Prop := sorry

-- Theorem statement
theorem high_correlation_implies_r_near_one 
  (x y : List ℝ) (r : ℝ) 
  (h1 : r = correlation_coefficient x y) 
  (h2 : is_very_high_correlation r) : 
  ∀ ε > 0, |r| > 1 - ε := by
  sorry

end NUMINAMATH_CALUDE_high_correlation_implies_r_near_one_l578_57880


namespace NUMINAMATH_CALUDE_haunted_mansion_paths_l578_57819

theorem haunted_mansion_paths (n : ℕ) (h : n = 8) : n * (n - 1) * (n - 2) = 336 := by
  sorry

end NUMINAMATH_CALUDE_haunted_mansion_paths_l578_57819


namespace NUMINAMATH_CALUDE_g_range_contains_pi_quarters_l578_57871

open Real

noncomputable def g (x : ℝ) : ℝ := arctan x + arctan ((x - 1) / (x + 1)) + arctan (1 / x)

theorem g_range_contains_pi_quarters :
  ∃ (x : ℝ), g x = π / 4 ∨ g x = 5 * π / 4 :=
sorry

end NUMINAMATH_CALUDE_g_range_contains_pi_quarters_l578_57871


namespace NUMINAMATH_CALUDE_greatest_x_value_l578_57847

theorem greatest_x_value (a b c d : ℤ) (x : ℝ) :
  x = (a + b * Real.sqrt c) / d →
  (7 * x) / 9 + 1 = 3 / x →
  (∀ y : ℝ, (7 * y) / 9 + 1 = 3 / y → y ≤ x) →
  a * c * d / b = -4158 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l578_57847


namespace NUMINAMATH_CALUDE_cubic_unit_circle_roots_l578_57821

theorem cubic_unit_circle_roots (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∃ w₁ w₂ w₃ : ℂ, 
    (w₁^3 + Complex.abs a * w₁^2 + Complex.abs b * w₁ + Complex.abs c = 0) ∧
    (w₂^3 + Complex.abs a * w₂^2 + Complex.abs b * w₂ + Complex.abs c = 0) ∧
    (w₃^3 + Complex.abs a * w₃^2 + Complex.abs b * w₃ + Complex.abs c = 0) ∧
    Complex.abs w₁ = 1 ∧ Complex.abs w₂ = 1 ∧ Complex.abs w₃ = 1) :=
by sorry


end NUMINAMATH_CALUDE_cubic_unit_circle_roots_l578_57821


namespace NUMINAMATH_CALUDE_pure_imaginary_z_l578_57818

theorem pure_imaginary_z (a : ℝ) : 
  let z : ℂ := a^2 + 2*a - 2 + (2*Complex.I)/(1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 ∨ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_l578_57818


namespace NUMINAMATH_CALUDE_sarah_birthday_next_monday_l578_57856

def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def days_since_reference_date (year month day : ℕ) : ℕ :=
  sorry

def day_of_week (year month day : ℕ) : ℕ :=
  (days_since_reference_date year month day) % 7

theorem sarah_birthday_next_monday (start_year : ℕ) (start_day_of_week : ℕ) :
  start_year = 2017 →
  start_day_of_week = 5 →
  day_of_week 2025 6 16 = 1 →
  ∀ y : ℕ, start_year < y → y < 2025 → day_of_week y 6 16 ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_sarah_birthday_next_monday_l578_57856


namespace NUMINAMATH_CALUDE_sphere_inscribed_in_all_shapes_l578_57855

/-- Represents a sphere with a given diameter -/
structure Sphere where
  diameter : ℝ

/-- Represents a square-based prism with a given base edge -/
structure SquarePrism where
  baseEdge : ℝ

/-- Represents a triangular prism with an isosceles triangle base -/
structure TriangularPrism where
  base : ℝ
  height : ℝ

/-- Represents a cylinder with a given base circle diameter -/
structure Cylinder where
  baseDiameter : ℝ

/-- Predicate to check if a sphere can be inscribed in a square prism -/
def inscribedInSquarePrism (s : Sphere) (p : SquarePrism) : Prop :=
  s.diameter = p.baseEdge

/-- Predicate to check if a sphere can be inscribed in a triangular prism -/
def inscribedInTriangularPrism (s : Sphere) (p : TriangularPrism) : Prop :=
  s.diameter = p.base ∧ s.diameter ≤ p.height * (Real.sqrt 3) / 2

/-- Predicate to check if a sphere can be inscribed in a cylinder -/
def inscribedInCylinder (s : Sphere) (c : Cylinder) : Prop :=
  s.diameter = c.baseDiameter

/-- Theorem stating that a sphere with diameter a can be inscribed in all three shapes -/
theorem sphere_inscribed_in_all_shapes (a : ℝ) :
  let s : Sphere := ⟨a⟩
  let sp : SquarePrism := ⟨a⟩
  let tp : TriangularPrism := ⟨a, a⟩
  let c : Cylinder := ⟨a⟩
  inscribedInSquarePrism s sp ∧
  inscribedInTriangularPrism s tp ∧
  inscribedInCylinder s c :=
by sorry

end NUMINAMATH_CALUDE_sphere_inscribed_in_all_shapes_l578_57855


namespace NUMINAMATH_CALUDE_max_car_distance_l578_57884

/-- Represents the maximum distance a car can travel with one tire swap -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  front_tire_life + min front_tire_life rear_tire_life

/-- Theorem stating the maximum distance a car can travel with given tire lifespans -/
theorem max_car_distance :
  let front_tire_life : ℕ := 24000
  let rear_tire_life : ℕ := 36000
  max_distance front_tire_life rear_tire_life = 28800 := by
  sorry

#eval max_distance 24000 36000

end NUMINAMATH_CALUDE_max_car_distance_l578_57884


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l578_57877

theorem max_value_of_sin_cos_combination :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos x
  ∃ M : ℝ, M = 5 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l578_57877


namespace NUMINAMATH_CALUDE_product_equals_fraction_l578_57835

/-- The decimal representation of the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal and 11 -/
def product : ℚ := repeating_decimal * 11

/-- Theorem stating that the product of 0.456̄ and 11 is equal to 1672/333 -/
theorem product_equals_fraction : product = 1672 / 333 := by sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l578_57835


namespace NUMINAMATH_CALUDE_sum_of_squares_l578_57823

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l578_57823


namespace NUMINAMATH_CALUDE_ratio_problem_l578_57822

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l578_57822


namespace NUMINAMATH_CALUDE_product_equality_l578_57861

theorem product_equality (h : 213 * 16 = 3408) : 16 * 21.3 = 340.8 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l578_57861


namespace NUMINAMATH_CALUDE_max_axes_of_symmetry_l578_57857

/-- A type representing a configuration of segments on a plane -/
structure SegmentConfiguration where
  k : ℕ+  -- number of segments (positive natural number)

/-- The number of axes of symmetry for a given segment configuration -/
def axesOfSymmetry (config : SegmentConfiguration) : ℕ := sorry

/-- Theorem stating that the maximum number of axes of symmetry is 2k -/
theorem max_axes_of_symmetry (config : SegmentConfiguration) :
  ∃ (arrangement : SegmentConfiguration), 
    arrangement.k = config.k ∧ 
    axesOfSymmetry arrangement = 2 * config.k.val ∧
    ∀ (other : SegmentConfiguration), 
      other.k = config.k → 
      axesOfSymmetry other ≤ axesOfSymmetry arrangement :=
by sorry

end NUMINAMATH_CALUDE_max_axes_of_symmetry_l578_57857


namespace NUMINAMATH_CALUDE_digit_245_l578_57863

/-- The decimal representation of 13/17 -/
def decimal_rep : ℚ := 13 / 17

/-- The length of the repeating sequence in the decimal representation of 13/17 -/
def repeat_length : ℕ := 16

/-- The nth digit in the decimal representation of 13/17 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_245 : nth_digit 245 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_245_l578_57863


namespace NUMINAMATH_CALUDE_f_max_value_l578_57896

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n) / ((n + 32) * (S (n + 1)))

theorem f_max_value : ∀ n : ℕ, f n ≤ 1 / 50 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l578_57896


namespace NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l578_57844

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem six_hundred_million_scientific_notation :
  toScientificNotation 600000000 = ScientificNotation.mk 6 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l578_57844


namespace NUMINAMATH_CALUDE_city_population_ratio_l578_57865

theorem city_population_ratio (pop_x pop_y pop_z : ℝ) (s : ℝ) 
  (h1 : pop_x = 6 * pop_y)
  (h2 : pop_y = s * pop_z)
  (h3 : pop_x / pop_z = 12)
  (h4 : pop_z > 0) : 
  pop_y / pop_z = 2 := by
sorry

end NUMINAMATH_CALUDE_city_population_ratio_l578_57865


namespace NUMINAMATH_CALUDE_inequality_empty_solution_set_l578_57888

theorem inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) ↔ -2 ≤ a ∧ a < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_empty_solution_set_l578_57888


namespace NUMINAMATH_CALUDE_sum_three_digit_integers_eq_385550_l578_57899

/-- The sum of all three-digit positive integers from 200 to 900 -/
def sum_three_digit_integers : ℕ :=
  let first_term := 200
  let last_term := 900
  let common_difference := 1
  let num_terms := (last_term - first_term) / common_difference + 1
  (num_terms * (first_term + last_term)) / 2

theorem sum_three_digit_integers_eq_385550 : 
  sum_three_digit_integers = 385550 := by
  sorry

#eval sum_three_digit_integers

end NUMINAMATH_CALUDE_sum_three_digit_integers_eq_385550_l578_57899


namespace NUMINAMATH_CALUDE_quadratic_properties_l578_57867

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : quadratic a b c 0 = 2)
  (h2 : quadratic a b c 1 = 2)
  (h3 : quadratic a b c (3/2) < 0)
  (h4 : ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0)
  (h5 : ∃ x : ℝ, -1/2 < x ∧ x < 0 ∧ quadratic a b c x = 0) :
  (∀ x ≤ 0, ∀ y ≤ x, quadratic a b c y ≤ quadratic a b c x) ∧
  (3 * quadratic a b c (-1) - quadratic a b c 2 < -20/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l578_57867


namespace NUMINAMATH_CALUDE_telephone_bill_proof_l578_57890

theorem telephone_bill_proof (F C : ℝ) : 
  F + C = 40 →
  F + 2*C = 76 →
  F + C = 40 := by
sorry

end NUMINAMATH_CALUDE_telephone_bill_proof_l578_57890


namespace NUMINAMATH_CALUDE_baseball_cards_total_l578_57801

def total_baseball_cards (carlos_cards matias_cards jorge_cards : ℕ) : ℕ :=
  carlos_cards + matias_cards + jorge_cards

theorem baseball_cards_total (carlos_cards matias_cards jorge_cards : ℕ) 
  (h1 : carlos_cards = 20)
  (h2 : matias_cards = carlos_cards - 6)
  (h3 : jorge_cards = matias_cards) :
  total_baseball_cards carlos_cards matias_cards jorge_cards = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_total_l578_57801


namespace NUMINAMATH_CALUDE_money_distribution_l578_57852

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 1000)
  (ac_sum : A + C = 700)
  (bc_sum : B + C = 600) :
  C = 300 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l578_57852


namespace NUMINAMATH_CALUDE_building_shadow_length_l578_57806

/-- Given a flagstaff and a building with their respective heights and the flagstaff's shadow length,
    calculate the length of the building's shadow under similar conditions. -/
theorem building_shadow_length 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagstaff_height = 17.5)
  (h2 : flagstaff_shadow = 40.25)
  (h3 : building_height = 12.5) :
  (building_height * flagstaff_shadow) / flagstaff_height = 28.75 :=
by sorry

end NUMINAMATH_CALUDE_building_shadow_length_l578_57806


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l578_57829

theorem total_books_on_shelves (x : ℕ) : 
  (x / 2 + 5 = 2 * (x / 2 - 5)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l578_57829


namespace NUMINAMATH_CALUDE_equal_probability_same_different_color_l578_57868

theorem equal_probability_same_different_color (t : ℤ) :
  let n := t * (t + 1) / 2
  let k := t * (t - 1) / 2
  let total := n + k
  total ≥ 2 →
  (n * (n - 1) + k * (k - 1)) / (total * (total - 1)) = 
  (2 * n * k) / (total * (total - 1)) := by
sorry

end NUMINAMATH_CALUDE_equal_probability_same_different_color_l578_57868


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l578_57876

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

-- State the theorem
theorem f_decreasing_interval :
  ∀ x y : ℝ, x > 0 → y > 0 → 
  (Real.log (x + y) = Real.log x + Real.log y) →
  (∀ a b : ℝ, a > 1 → b > 1 → a < b → f a > f b) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l578_57876


namespace NUMINAMATH_CALUDE_trolley_problem_l578_57873

/-- Trolley problem theorem -/
theorem trolley_problem (initial_passengers : ℕ) 
  (second_stop_off : ℕ) (second_stop_on_multiplier : ℕ)
  (third_stop_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 10 →
  second_stop_off = 3 →
  second_stop_on_multiplier = 2 →
  third_stop_off = 18 →
  final_passengers = 12 →
  (initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers - third_stop_off) + 3 = final_passengers :=
by
  sorry

#check trolley_problem

end NUMINAMATH_CALUDE_trolley_problem_l578_57873


namespace NUMINAMATH_CALUDE_team_total_score_l578_57800

def team_score (connor_initial : ℕ) (amy_initial : ℕ) (jason_initial : ℕ) 
  (connor_bonus : ℕ) (amy_bonus : ℕ) (jason_bonus : ℕ) (emily : ℕ) : ℕ :=
  connor_initial + connor_bonus + amy_initial + amy_bonus + jason_initial + jason_bonus + emily

theorem team_total_score :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let connor_bonus := 3
  let amy_bonus := 5
  let jason_bonus := 1
  let emily := 3 * (connor_initial + amy_initial + jason_initial)
  team_score connor_initial amy_initial jason_initial connor_bonus amy_bonus jason_bonus emily = 89 := by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l578_57800


namespace NUMINAMATH_CALUDE_modular_equation_solution_l578_57887

theorem modular_equation_solution (x : ℤ) : 
  (10 * x + 3 ≡ 7 [ZMOD 15]) → 
  (∃ (a m : ℕ), m ≥ 2 ∧ a < m ∧ x ≡ a [ZMOD m] ∧ a + m = 27) :=
by sorry

end NUMINAMATH_CALUDE_modular_equation_solution_l578_57887


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_B_union_C_eq_B_iff_m_lt_4_l578_57883

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x - 7) / (x + 2) > 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 3*x + 28)}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: Prove that (complement of A) ∩ B = [-2, 7)
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Icc (-2) 7 := by sorry

-- Theorem 2: Prove that B ∪ C = B if and only if m < 4
theorem B_union_C_eq_B_iff_m_lt_4 (m : ℝ) :
  B ∪ C m = B ↔ m < 4 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_B_union_C_eq_B_iff_m_lt_4_l578_57883


namespace NUMINAMATH_CALUDE_parabola_equation_l578_57850

/-- Given a parabola in the form x² = 2py where p > 0, with axis of symmetry y = -1/2,
    prove that its equation is x² = 2y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : -p/2 = -1/2) :
  ∀ x y : ℝ, x^2 = 2*p*y ↔ x^2 = 2*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l578_57850


namespace NUMINAMATH_CALUDE_sum_of_first_five_terms_l578_57845

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_five_terms
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_4th : a 4 = 11)
  (h_5th : a 5 = 15)
  (h_6th : a 6 = 19) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 35 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_five_terms_l578_57845


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l578_57889

theorem simultaneous_inequalities (x : ℝ) :
  x^2 - 12*x + 32 > 0 ∧ x^2 - 13*x + 22 < 0 → 2 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l578_57889
