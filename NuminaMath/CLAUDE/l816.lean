import Mathlib

namespace ball_probability_l816_81619

theorem ball_probability (m : ℕ) : 
  (8 : ℝ) / (8 + m) > (m : ℝ) / (8 + m) → m < 8 := by
  sorry

end ball_probability_l816_81619


namespace largest_divisor_of_expression_l816_81635

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  (∃ (k : ℤ), (10*x + 1) * (10*x + 5) * (5*x + 3) = 3 * k) ∧
  (∀ (d : ℤ), d > 3 → ∃ (y : ℤ), Even y ∧ ¬(∃ (k : ℤ), (10*y + 1) * (10*y + 5) * (5*y + 3) = d * k)) :=
by sorry

end largest_divisor_of_expression_l816_81635


namespace order_of_abc_l816_81606

theorem order_of_abc (a b c : ℝ) 
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : 
  b < a ∧ a < c :=
sorry

end order_of_abc_l816_81606


namespace triangle_perimeter_range_l816_81639

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a = 1 →
  2 * Real.cos C + c = 2 * b →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) / Real.sin ((B + C) / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) / Real.sin ((A + C) / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) / Real.sin ((A + B) / 2) →
  let p := a + b + c
  Real.sqrt 3 + 1 < p ∧ p < 3 := by
sorry

end triangle_perimeter_range_l816_81639


namespace unique_a_value_l816_81615

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 4^x else 2^(a - x)

-- State the theorem
theorem unique_a_value (a : ℝ) (h1 : a ≠ 1) :
  f a (1 - a) = f a (a - 1) → a = 1/2 := by
  sorry

end unique_a_value_l816_81615


namespace coin_division_problem_l816_81691

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) → 
  (n % 7 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 0) := by
sorry

end coin_division_problem_l816_81691


namespace hannahs_dogs_food_l816_81610

/-- The amount of food eaten by Hannah's first dog -/
def first_dog_food : ℝ := 1.5

/-- The amount of food eaten by Hannah's second dog -/
def second_dog_food : ℝ := 2 * first_dog_food

/-- The amount of food eaten by Hannah's third dog -/
def third_dog_food : ℝ := second_dog_food + 2.5

/-- The total amount of food prepared by Hannah for her three dogs -/
def total_food : ℝ := 10

theorem hannahs_dogs_food :
  first_dog_food + second_dog_food + third_dog_food = total_food :=
by sorry

end hannahs_dogs_food_l816_81610


namespace cook_remaining_potatoes_l816_81612

/-- Given a chef needs to cook potatoes with the following conditions:
  * The total number of potatoes to cook
  * The number of potatoes already cooked
  * The time it takes to cook each potato
  This function calculates the time required to cook the remaining potatoes. -/
def time_to_cook_remaining (total : ℕ) (cooked : ℕ) (time_per_potato : ℕ) : ℕ :=
  (total - cooked) * time_per_potato

/-- Theorem stating that it takes 36 minutes to cook the remaining potatoes. -/
theorem cook_remaining_potatoes :
  time_to_cook_remaining 12 6 6 = 36 := by
  sorry

end cook_remaining_potatoes_l816_81612


namespace book_discount_percentage_l816_81690

def original_price : ℝ := 60
def discounted_price : ℝ := 45
def discount : ℝ := 15
def tax_rate : ℝ := 0.1

theorem book_discount_percentage :
  (discount / original_price) * 100 = 25 := by
  sorry

end book_discount_percentage_l816_81690


namespace max_value_of_expression_l816_81662

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (2*x + y) + y / (2*y + z) + z / (2*z + x) ≤ 1 := by
  sorry

#check max_value_of_expression

end max_value_of_expression_l816_81662


namespace jeanne_needs_eight_tickets_l816_81645

/-- The number of tickets needed for the Ferris wheel -/
def ferris_wheel_tickets : ℕ := 5

/-- The number of tickets needed for the roller coaster -/
def roller_coaster_tickets : ℕ := 4

/-- The number of tickets needed for the bumper cars -/
def bumper_cars_tickets : ℕ := 4

/-- The number of tickets Jeanne already has -/
def jeanne_tickets : ℕ := 5

/-- The total number of tickets needed for all three rides -/
def total_tickets_needed : ℕ := ferris_wheel_tickets + roller_coaster_tickets + bumper_cars_tickets

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets_needed : ℕ := total_tickets_needed - jeanne_tickets

theorem jeanne_needs_eight_tickets : additional_tickets_needed = 8 := by
  sorry

end jeanne_needs_eight_tickets_l816_81645


namespace problems_per_page_l816_81693

/-- Given a homework assignment with the following conditions:
  * There are 72 total problems
  * 32 problems have been completed
  * The remaining problems are spread equally across 5 pages
  This theorem proves that there are 8 problems on each remaining page. -/
theorem problems_per_page (total : ℕ) (completed : ℕ) (pages : ℕ) : 
  total = 72 → completed = 32 → pages = 5 → (total - completed) / pages = 8 := by
  sorry

end problems_per_page_l816_81693


namespace arithmetic_sequence_150th_term_l816_81677

/-- An arithmetic sequence with first term 3 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

/-- The 150th term of the arithmetic sequence is 748 -/
theorem arithmetic_sequence_150th_term : arithmeticSequence 150 = 748 := by
  sorry

end arithmetic_sequence_150th_term_l816_81677


namespace at_least_one_composite_l816_81663

theorem at_least_one_composite (a b c k : ℕ) 
  (ha : a ≥ 3) (hb : b ≥ 3) (hc : c ≥ 3) 
  (heq : a * b * c = k^2 + 1) : 
  ¬(Nat.Prime (a - 1) ∧ Nat.Prime (b - 1) ∧ Nat.Prime (c - 1)) := by
  sorry

end at_least_one_composite_l816_81663


namespace meaningful_fraction_l816_81601

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end meaningful_fraction_l816_81601


namespace simple_interest_rate_l816_81614

/-- Simple interest calculation --/
theorem simple_interest_rate (P : ℝ) (t : ℝ) (I : ℝ) : 
  P = 450 →
  t = 8 →
  I = P - 306 →
  I = P * (4 / 100) * t :=
by sorry

end simple_interest_rate_l816_81614


namespace population_1988_l816_81647

/-- The population growth factor for a 4-year period -/
def growth_factor : ℝ := 2

/-- The number of 4-year periods between 1988 and 2008 -/
def num_periods : ℕ := 5

/-- The population of Arloe in 2008 -/
def population_2008 : ℕ := 3456

/-- The population growth function -/
def population (initial : ℕ) (periods : ℕ) : ℝ :=
  initial * growth_factor ^ periods

theorem population_1988 :
  ∃ p : ℕ, population p num_periods = population_2008 ∧ p = 108 := by
  sorry

end population_1988_l816_81647


namespace solid_with_rectangular_views_is_cuboid_l816_81605

/-- A solid is a three-dimensional geometric object. -/
structure Solid :=
  (shape : Type)

/-- A view is a two-dimensional projection of a solid. -/
inductive View
  | Front
  | Top
  | Side

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- A cuboid is a three-dimensional solid with six rectangular faces. -/
structure Cuboid :=
  (length : ℝ)
  (width : ℝ)
  (height : ℝ)

/-- The projection of a solid onto a view. -/
def projection (s : Solid) (v : View) : Type :=
  sorry

/-- Theorem: If a solid's three views are all rectangles, then the solid is a cuboid. -/
theorem solid_with_rectangular_views_is_cuboid (s : Solid) :
  (∀ v : View, projection s v = Rectangle) → (s.shape = Cuboid) :=
sorry

end solid_with_rectangular_views_is_cuboid_l816_81605


namespace min_value_sum_reciprocals_l816_81617

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  (1/x + 4/y + 9/z) ≥ 36/5 :=
by sorry

end min_value_sum_reciprocals_l816_81617


namespace rectangular_field_area_l816_81656

theorem rectangular_field_area (L W : ℝ) : 
  L = 10 →                 -- One side is 10 feet
  2 * W + L = 130 →        -- Total fencing is 130 feet
  L * W = 600 :=           -- Area of the field is 600 square feet
by sorry

end rectangular_field_area_l816_81656


namespace range_of_a_min_value_of_a_l816_81642

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Statement 1
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f a x ≤ 3) → 0 ≤ a ∧ a ≤ 4 := by sorry

-- Statement 2
theorem min_value_of_a :
  ∃ a : ℝ, a = 1/3 ∧ (∀ x : ℝ, |x - a| + |x + a| ≥ 1 - a) ∧
  (∀ b : ℝ, (∀ x : ℝ, |x - b| + |x + b| ≥ 1 - b) → a ≤ b) := by sorry

end range_of_a_min_value_of_a_l816_81642


namespace class_size_l816_81611

theorem class_size (boys girls : ℕ) : 
  boys = 3 * (boys / 3) ∧ 
  girls = 2 * (boys / 3) ∧ 
  boys = girls + 20 → 
  boys + girls = 100 := by
sorry

end class_size_l816_81611


namespace geometric_sequence_sum_l816_81637

/-- Given a geometric sequence {aₙ} where the sum of the first n terms
    is given by Sₙ = a·2^(n-1) + 1/6, prove that a = -1/3 -/
theorem geometric_sequence_sum (a : ℝ) : 
  (∀ n : ℕ, ∃ Sn : ℝ, Sn = a * 2^(n-1) + 1/6) → a = -1/3 := by
  sorry

end geometric_sequence_sum_l816_81637


namespace cube_edge_length_l816_81608

theorem cube_edge_length : ∃ s : ℝ, s > 0 ∧ s^3 = 6 * s^2 := by
  sorry

end cube_edge_length_l816_81608


namespace brick_width_calculation_l816_81631

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The volume of the wall in cubic centimeters -/
def wall_volume : ℝ := 700 * 600 * 22.5

/-- The number of bricks required -/
def num_bricks : ℕ := 5600

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

theorem brick_width_calculation : 
  wall_volume = (brick_length * brick_width * brick_height) * num_bricks :=
by sorry

end brick_width_calculation_l816_81631


namespace rose_ratio_l816_81665

theorem rose_ratio (total : ℕ) (tulips : ℕ) (carnations : ℕ) :
  total = 40 ∧ tulips = 10 ∧ carnations = 14 →
  (total - tulips - carnations : ℚ) / total = 2 / 5 := by
  sorry

end rose_ratio_l816_81665


namespace inequality_solution_l816_81640

theorem inequality_solution (k : ℝ) : 
  (∀ x : ℝ, (k + 2) * x > k + 2 ↔ x < 1) → k = -3 := by
  sorry

end inequality_solution_l816_81640


namespace cistern_length_l816_81660

/-- The length of a cistern with given dimensions and wet surface area -/
theorem cistern_length (width : ℝ) (depth : ℝ) (wet_surface_area : ℝ) 
  (h1 : width = 2)
  (h2 : depth = 1.25)
  (h3 : wet_surface_area = 23) :
  ∃ length : ℝ, 
    wet_surface_area = length * width + 2 * length * depth + 2 * width * depth ∧ 
    length = 4 := by
  sorry

end cistern_length_l816_81660


namespace tomato_plants_per_row_l816_81648

/-- Proves that the number of plants in each row is 10, given the conditions of the tomato planting problem -/
theorem tomato_plants_per_row :
  ∀ (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ),
    rows = 30 →
    yield_per_plant = 20 →
    total_yield = 6000 →
    total_yield = rows * yield_per_plant * (total_yield / (rows * yield_per_plant)) →
    total_yield / (rows * yield_per_plant) = 10 :=
by
  sorry

end tomato_plants_per_row_l816_81648


namespace free_square_positions_l816_81666

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the rectangle size
def RectangleSize := (3, 1)

-- Define the number of rectangles
def NumRectangles := 21

-- Define the possible free square positions
def FreePosns : List (Fin 8 × Fin 8) := [(3, 3), (3, 6), (6, 3), (6, 6)]

-- Theorem statement
theorem free_square_positions (board : Chessboard) (rectangles : Fin NumRectangles → Chessboard) :
  (∃! pos : Chessboard, pos ∉ (rectangles '' univ)) →
  (∃ pos ∈ FreePosns, pos ∉ (rectangles '' univ)) :=
sorry

end free_square_positions_l816_81666


namespace tim_needs_72_keys_l816_81632

/-- The number of keys Tim needs to make for his rental properties -/
def total_keys (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_apartment

/-- Theorem stating that Tim needs 72 keys for his rental properties -/
theorem tim_needs_72_keys :
  total_keys 2 12 3 = 72 := by
  sorry

end tim_needs_72_keys_l816_81632


namespace subtraction_of_decimals_l816_81653

theorem subtraction_of_decimals : (3.75 : ℝ) - (1.46 : ℝ) = 2.29 := by
  sorry

end subtraction_of_decimals_l816_81653


namespace circumcenter_property_implies_isosceles_l816_81697

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector addition and scalar multiplication
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2 ∨
  (t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2 = (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 ∨
  (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2

-- The main theorem
theorem circumcenter_property_implies_isosceles (t : Triangle) :
  let O := circumcenter t
  vec_add (vec_add (vec_add O (vec_scale (-1) t.A)) (vec_add O (vec_scale (-1) t.B))) (vec_scale (Real.sqrt 2) (vec_add O (vec_scale (-1) t.C))) = (0, 0)
  → is_isosceles t :=
sorry

end circumcenter_property_implies_isosceles_l816_81697


namespace glove_selection_theorem_l816_81687

theorem glove_selection_theorem :
  let total_pairs : ℕ := 6
  let gloves_to_select : ℕ := 4
  let same_color_pair : ℕ := 1
  let ways_to_select_pair : ℕ := total_pairs.choose same_color_pair
  let remaining_gloves : ℕ := 2 * (total_pairs - same_color_pair)
  let ways_to_select_others : ℕ := remaining_gloves.choose (gloves_to_select - 2) - (total_pairs - same_color_pair)
  ways_to_select_pair * ways_to_select_others = 240
  := by sorry

end glove_selection_theorem_l816_81687


namespace metro_earnings_l816_81613

/-- Calculates the earnings from ticket sales over a given period of time. -/
def calculate_earnings (ticket_cost : ℕ) (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  ticket_cost * tickets_per_minute * minutes

/-- Proves that the earnings from ticket sales in 6 minutes is $90,
    given the ticket cost and average tickets sold per minute. -/
theorem metro_earnings :
  calculate_earnings 3 5 6 = 90 := by
  sorry

end metro_earnings_l816_81613


namespace adults_count_is_21_l816_81685

/-- Represents the trekking group and meal information -/
structure TrekkingGroup where
  childrenCount : ℕ
  adultMealCapacity : ℕ
  childrenMealCapacity : ℕ
  remainingChildrenCapacity : ℕ
  adultsMealCount : ℕ

/-- Theorem stating that the number of adults in the trekking group is 21 -/
theorem adults_count_is_21 (group : TrekkingGroup)
  (h1 : group.childrenCount = 70)
  (h2 : group.adultMealCapacity = 70)
  (h3 : group.childrenMealCapacity = 90)
  (h4 : group.remainingChildrenCapacity = 63)
  (h5 : group.adultsMealCount = 21) :
  group.adultsMealCount = 21 := by
  sorry

#check adults_count_is_21

end adults_count_is_21_l816_81685


namespace b_minus_d_squared_l816_81681

theorem b_minus_d_squared (a b c d : ℤ) 
  (eq1 : a - b - c + d = 12)
  (eq2 : a + b - c - d = 6) : 
  (b - d)^2 = 9 := by sorry

end b_minus_d_squared_l816_81681


namespace almonds_vs_white_sugar_difference_l816_81624

-- Define the amounts of ingredients used
def brown_sugar : ℝ := 1.28
def white_sugar : ℝ := 0.75
def ground_almonds : ℝ := 1.56
def cocoa_powder : ℝ := 0.49

-- Theorem statement
theorem almonds_vs_white_sugar_difference :
  ground_almonds - white_sugar = 0.81 := by
  sorry

end almonds_vs_white_sugar_difference_l816_81624


namespace emily_quiz_score_theorem_l816_81651

def emily_scores : List ℕ := [85, 92, 88, 90, 93]
def target_mean : ℕ := 91
def num_quizzes : ℕ := 6
def sixth_score : ℕ := 98

theorem emily_quiz_score_theorem :
  let total_sum := (emily_scores.sum + sixth_score)
  total_sum / num_quizzes = target_mean :=
by sorry

end emily_quiz_score_theorem_l816_81651


namespace sequence_properties_l816_81659

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, n > 1 → a (n - 1) + a (n + 1) > 2 * a n

theorem sequence_properties (a : ℕ+ → ℝ) (h : sequence_property a) :
  (a 2 > a 1 → ∀ n : ℕ+, n > 1 → a n > a (n - 1)) ∧
  (∃ d : ℝ, ∀ n : ℕ+, a n > a 1 + (n - 1) * d) := by
  sorry

end sequence_properties_l816_81659


namespace friendship_class_theorem_l816_81683

/-- Represents the number of students in a class with specific friendship conditions. -/
structure FriendshipClass where
  boys : ℕ
  girls : ℕ

/-- Checks if the friendship conditions are satisfied for a given class. -/
def satisfiesFriendshipConditions (c : FriendshipClass) : Prop :=
  3 * c.boys = 2 * c.girls

/-- Checks if a class with the given total number of students can satisfy the friendship conditions. -/
def canHaveStudents (n : ℕ) : Prop :=
  ∃ c : FriendshipClass, c.boys + c.girls = n ∧ satisfiesFriendshipConditions c

theorem friendship_class_theorem :
  ¬(canHaveStudents 32) ∧ (canHaveStudents 30) := by sorry

end friendship_class_theorem_l816_81683


namespace orange_profit_problem_l816_81618

/-- Represents the fruit vendor's orange selling problem -/
theorem orange_profit_problem 
  (buy_quantity : ℕ) 
  (buy_price : ℚ) 
  (sell_quantity : ℕ) 
  (sell_price : ℚ) 
  (target_profit : ℚ) :
  buy_quantity = 8 →
  buy_price = 15 →
  sell_quantity = 6 →
  sell_price = 18 →
  target_profit = 150 →
  ∃ (n : ℕ), 
    n * (sell_price / sell_quantity - buy_price / buy_quantity) ≥ target_profit ∧
    ∀ (m : ℕ), m * (sell_price / sell_quantity - buy_price / buy_quantity) ≥ target_profit → m ≥ n ∧
    n = 134 :=
sorry

end orange_profit_problem_l816_81618


namespace train_speed_proof_l816_81658

/-- Proves that a train crossing a 320-meter platform in 34 seconds and passing a stationary man in 18 seconds has a speed of 72 km/h -/
theorem train_speed_proof (platform_length : ℝ) (platform_crossing_time : ℝ) (man_passing_time : ℝ) :
  platform_length = 320 →
  platform_crossing_time = 34 →
  man_passing_time = 18 →
  ∃ (train_speed : ℝ),
    train_speed * man_passing_time = train_speed * platform_crossing_time - platform_length ∧
    train_speed * 3.6 = 72 := by
  sorry

end train_speed_proof_l816_81658


namespace dust_storm_coverage_l816_81607

/-- Given a prairie and a dust storm, calculate the area covered by the storm -/
theorem dust_storm_coverage (total_prairie_area untouched_area : ℕ) 
  (h1 : total_prairie_area = 65057)
  (h2 : untouched_area = 522) :
  total_prairie_area - untouched_area = 64535 := by
  sorry

end dust_storm_coverage_l816_81607


namespace joe_haircut_time_l816_81603

/-- Represents the time taken for different types of haircuts and the number of each type performed --/
structure HaircutData where
  womenTime : ℕ  -- Time to cut a woman's hair
  menTime : ℕ    -- Time to cut a man's hair
  kidsTime : ℕ   -- Time to cut a kid's hair
  womenCount : ℕ -- Number of women's haircuts
  menCount : ℕ   -- Number of men's haircuts
  kidsCount : ℕ  -- Number of kids' haircuts

/-- Calculates the total time spent cutting hair --/
def totalHaircutTime (data : HaircutData) : ℕ :=
  data.womenTime * data.womenCount +
  data.menTime * data.menCount +
  data.kidsTime * data.kidsCount

/-- Theorem stating that Joe's total haircut time is 255 minutes --/
theorem joe_haircut_time :
  let data : HaircutData := {
    womenTime := 50,
    menTime := 15,
    kidsTime := 25,
    womenCount := 3,
    menCount := 2,
    kidsCount := 3
  }
  totalHaircutTime data = 255 := by
  sorry

end joe_haircut_time_l816_81603


namespace cube_root_simplification_l816_81646

theorem cube_root_simplification : 
  (25^3 + 30^3 + 35^3 : ℝ)^(1/3) = 5 * 684^(1/3) := by
  sorry

end cube_root_simplification_l816_81646


namespace cone_rolling_ratio_l816_81634

/-- Represents a right circular cone. -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Represents the rolling properties of the cone. -/
structure RollingCone extends RightCircularCone where
  rotations : ℕ
  no_slip : Bool

theorem cone_rolling_ratio (c : RollingCone) 
  (h_positive : c.h > 0)
  (r_positive : c.r > 0)
  (twenty_rotations : c.rotations = 20)
  (no_slip : c.no_slip = true) :
  c.h / c.r = Real.sqrt 399 :=
sorry

end cone_rolling_ratio_l816_81634


namespace solution_set_is_open_interval_l816_81636

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 3 * x - 1

-- Define the solution set
def solution_set : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (1/2 : ℝ) 1 := by sorry

end solution_set_is_open_interval_l816_81636


namespace area_midpoint_triangle_is_sqrt3_l816_81629

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- The triangle formed by connecting midpoints of three adjacent regular hexagons -/
structure MidpointTriangle :=
  (hexagon1 : RegularHexagon)
  (hexagon2 : RegularHexagon)
  (hexagon3 : RegularHexagon)
  (are_adjacent : hexagon1 ≠ hexagon2 ∧ hexagon2 ≠ hexagon3 ∧ hexagon3 ≠ hexagon1)

/-- The area of the triangle formed by connecting midpoints of three adjacent regular hexagons -/
def area_midpoint_triangle (t : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the midpoint triangle is √3 -/
theorem area_midpoint_triangle_is_sqrt3 (t : MidpointTriangle) : 
  area_midpoint_triangle t = Real.sqrt 3 :=
sorry

end area_midpoint_triangle_is_sqrt3_l816_81629


namespace john_mean_score_l816_81699

def john_scores : List ℝ := [88, 92, 94, 86, 90, 85]

theorem john_mean_score :
  (john_scores.sum / john_scores.length : ℝ) = 535 / 6 := by
  sorry

end john_mean_score_l816_81699


namespace proposition_implication_l816_81684

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 8) : 
  ¬ P 7 := by sorry

end proposition_implication_l816_81684


namespace alla_boris_meeting_l816_81696

/-- The number of lamp posts along the alley -/
def total_posts : ℕ := 400

/-- The lamp post number where Alla is observed -/
def alla_observed : ℕ := 55

/-- The lamp post number where Boris is observed -/
def boris_observed : ℕ := 321

/-- The function to calculate the meeting point of Alla and Boris -/
def meeting_point : ℕ :=
  let alla_traveled := alla_observed - 1
  let boris_traveled := total_posts - boris_observed
  let total_traveled := alla_traveled + boris_traveled
  let alla_to_meeting := 3 * alla_traveled
  1 + alla_to_meeting

/-- Theorem stating that Alla and Boris will meet at lamp post 163 -/
theorem alla_boris_meeting :
  meeting_point = 163 := by sorry

end alla_boris_meeting_l816_81696


namespace tori_classroom_trash_l816_81626

/-- Represents the number of pieces of trash picked up in various locations --/
structure TrashCount where
  total : ℕ
  outside : ℕ

/-- Calculates the number of pieces of trash picked up in the classrooms --/
def classroom_trash (t : TrashCount) : ℕ :=
  t.total - t.outside

/-- Theorem stating that for Tori's specific trash counts, the classroom trash is 344 --/
theorem tori_classroom_trash :
  let tori_trash : TrashCount := { total := 1576, outside := 1232 }
  classroom_trash tori_trash = 344 := by
  sorry

#eval classroom_trash { total := 1576, outside := 1232 }

end tori_classroom_trash_l816_81626


namespace convex_ngon_coverage_l816_81650

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  area : Real

/-- Represents a triangle in 2D space -/
structure Triangle where
  vertices : List (Real × Real)
  area : Real

/-- Checks if a polygon is covered by a triangle -/
def is_covered (p : ConvexPolygon) (t : Triangle) : Prop :=
  sorry

/-- Main theorem: A convex n-gon with area 1 (n ≥ 6) can be covered by a triangle with area ≤ 2 -/
theorem convex_ngon_coverage (p : ConvexPolygon) :
  p.is_convex ∧ p.area = 1 ∧ p.vertices.length ≥ 6 →
  ∃ t : Triangle, t.area ≤ 2 ∧ is_covered p t :=
sorry

end convex_ngon_coverage_l816_81650


namespace water_donation_difference_l816_81602

/-- The number of food items donated by five food companies to a local food bank. -/
def food_bank_donation : ℕ := 375

/-- The number of dressed chickens donated by Foster Farms. -/
def foster_farms_chickens : ℕ := 45

/-- The number of bottles of water donated by American Summits. -/
def american_summits_water : ℕ := 2 * foster_farms_chickens

/-- The number of dressed chickens donated by Hormel. -/
def hormel_chickens : ℕ := 3 * foster_farms_chickens

/-- The number of dressed chickens donated by Boudin Butchers. -/
def boudin_butchers_chickens : ℕ := hormel_chickens / 3

/-- The number of bottles of water donated by Del Monte Foods. -/
def del_monte_water : ℕ := food_bank_donation - (foster_farms_chickens + american_summits_water + hormel_chickens + boudin_butchers_chickens)

/-- Theorem stating the difference in water bottles donated between American Summits and Del Monte Foods. -/
theorem water_donation_difference :
  american_summits_water - del_monte_water = 30 := by
  sorry

end water_donation_difference_l816_81602


namespace basketball_weight_proof_l816_81609

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 16

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 24

theorem basketball_weight_proof : 
  (6 * basketball_weight = 4 * kayak_weight) ∧ 
  (3 * kayak_weight = 72) → 
  basketball_weight = 16 := by
  sorry

end basketball_weight_proof_l816_81609


namespace product_of_numbers_l816_81668

theorem product_of_numbers (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 31) : a * b = 11 := by
  sorry

end product_of_numbers_l816_81668


namespace brendan_taxes_l816_81600

/-- Calculates the taxes paid by a waiter named Brendan based on his work schedule and income. -/
theorem brendan_taxes : 
  let hourly_wage : ℚ := 6
  let shifts_8hour : ℕ := 2
  let shifts_12hour : ℕ := 1
  let hourly_tips : ℚ := 12
  let tax_rate : ℚ := 1/5
  let reported_tips_fraction : ℚ := 1/3
  
  let total_hours : ℕ := shifts_8hour * 8 + shifts_12hour * 12
  let wage_income : ℚ := hourly_wage * total_hours
  let total_tips : ℚ := hourly_tips * total_hours
  let reported_tips : ℚ := total_tips * reported_tips_fraction
  let reported_income : ℚ := wage_income + reported_tips
  let taxes_paid : ℚ := reported_income * tax_rate

  taxes_paid = 56 := by sorry

end brendan_taxes_l816_81600


namespace train_length_l816_81686

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 6 → ∃ length : ℝ, abs (length - 100.02) < 0.01 :=
by
  sorry

end train_length_l816_81686


namespace coin_problem_l816_81667

theorem coin_problem (x y : ℕ) : 
  x + y = 40 →
  2 * x + 5 * y = 125 →
  y = 15 := by sorry

end coin_problem_l816_81667


namespace imaginary_part_of_complex_fraction_l816_81633

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_fraction_l816_81633


namespace solve_equations_l816_81638

theorem solve_equations :
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, 3 * x * (x - 1) = 2 * (x - 1) ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 1 ∧ x₂ = 2/3) ∧
  (∃ y₁ y₂ : ℝ, (∀ x : ℝ, x^2 - 6*x + 6 = 0 ↔ x = y₁ ∨ x = y₂) ∧ y₁ = 3 + Real.sqrt 3 ∧ y₂ = 3 - Real.sqrt 3) :=
by sorry

end solve_equations_l816_81638


namespace fraction_difference_equals_square_difference_l816_81649

theorem fraction_difference_equals_square_difference 
  (x y z v : ℚ) (h : x / y + z / v = 1) : 
  x / y - z / v = (x / y)^2 - (z / v)^2 := by
  sorry

end fraction_difference_equals_square_difference_l816_81649


namespace fraction_product_l816_81678

theorem fraction_product : 
  (4 : ℚ) / 5 * 5 / 6 * 6 / 7 * 7 / 8 * 8 / 9 = 4 / 9 := by
  sorry

end fraction_product_l816_81678


namespace sqrt_735_simplification_l816_81621

theorem sqrt_735_simplification : Real.sqrt 735 = 7 * Real.sqrt 15 := by
  sorry

end sqrt_735_simplification_l816_81621


namespace pyramid_volume_l816_81630

/-- Given a pyramid with a square base ABCD and vertex P, prove its volume. -/
theorem pyramid_volume (base_area : ℝ) (triangle_ABP_area : ℝ) (triangle_BCP_area : ℝ) (triangle_ADP_area : ℝ)
  (h_base : base_area = 256)
  (h_ABP : triangle_ABP_area = 128)
  (h_BCP : triangle_BCP_area = 80)
  (h_ADP : triangle_ADP_area = 128) :
  ∃ (volume : ℝ), volume = (2048 * Real.sqrt 3) / 3 := by
  sorry

end pyramid_volume_l816_81630


namespace fabric_theorem_l816_81672

def fabric_problem (checkered_cost plain_cost yard_cost : ℚ) : Prop :=
  let checkered_yards := checkered_cost / yard_cost
  let plain_yards := plain_cost / yard_cost
  let total_yards := checkered_yards + plain_yards
  total_yards = 16

theorem fabric_theorem :
  fabric_problem 75 45 7.5 := by
  sorry

end fabric_theorem_l816_81672


namespace multiple_value_l816_81652

-- Define the variables
variable (x : ℝ)
variable (m : ℝ)

-- State the theorem
theorem multiple_value (h1 : m * x + 36 = 48) (h2 : x = 4) : m = 3 := by
  sorry

end multiple_value_l816_81652


namespace cubic_identity_l816_81604

theorem cubic_identity (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + b*c + c*a) := by
  sorry

end cubic_identity_l816_81604


namespace painting_cost_is_84_l816_81674

/-- Calculates the cost of painting house numbers on a street --/
def cost_of_painting (houses_per_side : ℕ) (south_start : ℕ) (north_start : ℕ) (increment : ℕ) : ℕ :=
  let south_end := south_start + increment * (houses_per_side - 1)
  let north_end := north_start + increment * (houses_per_side - 1)
  let south_cost := (houses_per_side - (south_end / 100)) + (south_end / 100)
  let north_cost := (houses_per_side - (north_end / 100)) + (north_end / 100)
  south_cost + north_cost

/-- The total cost of painting house numbers on the street is 84 dollars --/
theorem painting_cost_is_84 :
  cost_of_painting 30 5 6 6 = 84 :=
by sorry

end painting_cost_is_84_l816_81674


namespace hat_wearers_count_l816_81655

theorem hat_wearers_count (total_people adults children : ℕ)
  (adult_women adult_men : ℕ)
  (women_hat_percentage men_hat_percentage children_hat_percentage : ℚ) :
  total_people = adults + children →
  adults = adult_women + adult_men →
  adult_women = adult_men →
  women_hat_percentage = 25 / 100 →
  men_hat_percentage = 12 / 100 →
  children_hat_percentage = 10 / 100 →
  adults = 1800 →
  children = 200 →
  (adult_women * women_hat_percentage).floor +
  (adult_men * men_hat_percentage).floor +
  (children * children_hat_percentage).floor = 353 := by
sorry

end hat_wearers_count_l816_81655


namespace total_cost_rounded_to_18_l816_81670

def item1 : ℚ := 247 / 100
def item2 : ℚ := 625 / 100
def item3 : ℚ := 876 / 100
def item4 : ℚ := 149 / 100

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

def total_cost : ℚ := item1 + item2 + item3 + item4

theorem total_cost_rounded_to_18 :
  round_to_nearest_dollar total_cost = 18 := by
  sorry

end total_cost_rounded_to_18_l816_81670


namespace all_numbers_equal_l816_81654

/-- Represents a 10x10 table of real numbers -/
def Table := Fin 10 → Fin 10 → ℝ

/-- Predicate to check if a number is underlined in its row -/
def is_underlined_in_row (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, t i j ≥ t i k

/-- Predicate to check if a number is underlined in its column -/
def is_underlined_in_col (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, t i j ≤ t k j

/-- Predicate to check if a number is underlined exactly twice -/
def is_underlined_twice (t : Table) (i j : Fin 10) : Prop :=
  is_underlined_in_row t i j ∧ is_underlined_in_col t i j

theorem all_numbers_equal (t : Table) 
  (h : ∀ i j : Fin 10, is_underlined_in_row t i j ∨ is_underlined_in_col t i j → is_underlined_twice t i j) :
  ∀ i j k l : Fin 10, t i j = t k l :=
sorry

end all_numbers_equal_l816_81654


namespace store_comparison_and_best_plan_l816_81641

/- Define the prices and quantities -/
def racket_price : ℝ := 50
def ball_price : ℝ := 20
def racket_quantity : ℕ := 10
def ball_quantity : ℕ := 40

/- Define the cost functions for each store -/
def cost_store_a (x : ℝ) : ℝ := 20 * x + 300
def cost_store_b (x : ℝ) : ℝ := 16 * x + 400

/- Define the most cost-effective plan -/
def cost_effective_plan : ℝ := racket_price * racket_quantity + ball_price * (ball_quantity - racket_quantity) * 0.8

/- Theorem statement -/
theorem store_comparison_and_best_plan :
  (cost_store_b ball_quantity < cost_store_a ball_quantity) ∧
  (cost_effective_plan = 980) := by
  sorry


end store_comparison_and_best_plan_l816_81641


namespace mark_has_six_parking_tickets_l816_81692

/-- Represents the number of tickets for each person -/
structure Tickets where
  mark_parking : ℕ
  mark_speeding : ℕ
  sarah_parking : ℕ
  sarah_speeding : ℕ
  john_parking : ℕ
  john_speeding : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (t : Tickets) : Prop :=
  t.mark_parking + t.mark_speeding + t.sarah_parking + t.sarah_speeding + t.john_parking + t.john_speeding = 36 ∧
  t.mark_parking = 2 * t.sarah_parking ∧
  t.mark_speeding = t.sarah_speeding ∧
  t.john_parking * 3 = t.mark_parking ∧
  t.john_speeding = 2 * t.sarah_speeding ∧
  t.sarah_speeding = 6

/-- The theorem stating that Mark has 6 parking tickets -/
theorem mark_has_six_parking_tickets (t : Tickets) (h : satisfies_conditions t) : t.mark_parking = 6 := by
  sorry

end mark_has_six_parking_tickets_l816_81692


namespace solution_set_reciprocal_inequality_l816_81680

theorem solution_set_reciprocal_inequality (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_reciprocal_inequality_l816_81680


namespace bus_passing_theorem_l816_81682

/-- Represents the time in minutes since midnight -/
def Time := ℕ

/-- Represents the direction of the bus -/
inductive Direction
| Austin2SanAntonio
| SanAntonio2Austin

/-- Represents a bus schedule -/
structure BusSchedule where
  start : Time
  interval : ℕ
  direction : Direction

/-- Calculates the number of buses passed during a journey -/
def count_passed_buses (sa_schedule : BusSchedule) (austin_schedule : BusSchedule) (journey_time : ℕ) : ℕ :=
  sorry

/-- Converts time from hour:minute format to minutes since midnight -/
def time_to_minutes (hour : ℕ) (minute : ℕ) : Time :=
  hour * 60 + minute

theorem bus_passing_theorem (sa_schedule : BusSchedule) (austin_schedule : BusSchedule) :
  sa_schedule.start = time_to_minutes 12 15 ∧
  sa_schedule.interval = 30 ∧
  sa_schedule.direction = Direction.SanAntonio2Austin ∧
  austin_schedule.start = time_to_minutes 12 0 ∧
  austin_schedule.interval = 45 ∧
  austin_schedule.direction = Direction.Austin2SanAntonio →
  count_passed_buses sa_schedule austin_schedule (6 * 60) = 9 :=
sorry

end bus_passing_theorem_l816_81682


namespace max_value_of_a_plus_inverse_l816_81625

theorem max_value_of_a_plus_inverse (a : ℝ) (h : a < 0) : 
  ∃ (M : ℝ), M = -2 ∧ ∀ (x : ℝ), x < 0 → x + 1/x ≤ M :=
sorry

end max_value_of_a_plus_inverse_l816_81625


namespace sample_size_is_six_l816_81676

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 18 + 12 + 6

/-- Represents the sample size -/
def n : ℕ := 6

/-- Checks if a number divides the total number of teachers -/
def divides_total (k : ℕ) : Prop := k ∣ total_teachers

/-- Checks if stratified sampling works for a given sample size -/
def stratified_sampling_works (k : ℕ) : Prop :=
  k ∣ 18 ∧ k ∣ 12 ∧ k ∣ 6

/-- Checks if systematic sampling works for a given sample size -/
def systematic_sampling_works (k : ℕ) : Prop :=
  divides_total k

/-- Checks if increasing the sample size by 1 requires excluding 1 person for systematic sampling -/
def exclusion_condition (k : ℕ) : Prop :=
  ¬(divides_total (k + 1)) ∧ ((k + 1) ∣ (total_teachers - 1))

theorem sample_size_is_six :
  stratified_sampling_works n ∧
  systematic_sampling_works n ∧
  exclusion_condition n ∧
  ∀ m : ℕ, m ≠ n →
    ¬(stratified_sampling_works m ∧
      systematic_sampling_works m ∧
      exclusion_condition m) :=
sorry

end sample_size_is_six_l816_81676


namespace rain_probability_l816_81671

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end rain_probability_l816_81671


namespace sufficient_not_necessary_l816_81673

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a * b > 4) ∧
  (∃ a b, a * b > 4 ∧ ¬(a > 2 ∧ b > 2)) := by
  sorry

end sufficient_not_necessary_l816_81673


namespace marble_game_theorem_l816_81657

/-- Represents the state of marbles for each player --/
structure MarbleState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Simulates one round of the game where the loser doubles the other players' marbles --/
def playRound (state : MarbleState) (loser : ℕ) : MarbleState :=
  match loser with
  | 1 => MarbleState.mk state.a (state.b * 3) (state.c * 3)
  | 2 => MarbleState.mk (state.a * 3) state.b (state.c * 3)
  | 3 => MarbleState.mk (state.a * 3) (state.b * 3) state.c
  | _ => state

/-- The main theorem statement --/
theorem marble_game_theorem :
  let initial_state := MarbleState.mk 165 57 21
  let after_round1 := playRound initial_state 1
  let after_round2 := playRound after_round1 2
  let final_state := playRound after_round2 3
  (after_round1.c = after_round1.a + 54) ∧
  (final_state.a = final_state.b) ∧
  (final_state.b = final_state.c) := by sorry

end marble_game_theorem_l816_81657


namespace quadratic_one_solution_negative_k_value_l816_81675

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x, 9 * x^2 + k * x + 36 = 0) ↔ k = 36 ∨ k = -36 :=
by sorry

theorem negative_k_value (k : ℝ) : 
  (∃! x, 9 * x^2 + k * x + 36 = 0) → k = -36 ∨ k = 36 :=
by sorry

end quadratic_one_solution_negative_k_value_l816_81675


namespace equation_solution_l816_81620

theorem equation_solution : (25 - 7 = 3 + x) → x = 15 := by
  sorry

end equation_solution_l816_81620


namespace min_dot_product_on_hyperbola_l816_81622

theorem min_dot_product_on_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m : ℝ × ℝ := (1, Real.sqrt (a^2 + 1/a^2))
  let B : ℝ × ℝ := (b, 1/b)
  m.1 * B.1 + m.2 * B.2 ≥ 2 * Real.sqrt (Real.sqrt 2) :=
by sorry

end min_dot_product_on_hyperbola_l816_81622


namespace intersection_point_a_l816_81695

/-- A function f(x) = 4x + b where b is an integer -/
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

/-- The inverse of f -/
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 4

theorem intersection_point_a (b : ℤ) (a : ℤ) :
  f b (-4) = a ∧ f_inv b (-4) = a → a = -4 := by
  sorry

end intersection_point_a_l816_81695


namespace all_nat_gt2_as_fib_sum_l816_81664

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define a function to check if a number is in the Fibonacci sequence
def isFib (n : ℕ) : Prop :=
  ∃ k, fib k = n

-- Define a function to represent a number as a sum of distinct Fibonacci numbers
def representAsFibSum (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ x ∈ S, isFib x) ∧ (S.sum id = n)

-- The main theorem
theorem all_nat_gt2_as_fib_sum :
  ∀ n : ℕ, n > 2 → representAsFibSum n :=
by
  sorry


end all_nat_gt2_as_fib_sum_l816_81664


namespace circle_radius_is_sqrt_2_l816_81623

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def intersects_x_axis_at (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  (p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2 = c.radius^2 ∧
  (p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2 = c.radius^2 ∧
  p1.2 = 0 ∧ p2.2 = 0

def tangent_to_line (c : Circle) : Prop :=
  ∃ (x y : ℝ), (x - y + 1 = 0) ∧
  ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (|x - y + 1| / Real.sqrt 2 = c.radius)

-- State the theorem
theorem circle_radius_is_sqrt_2 (c : Circle) :
  is_in_first_quadrant c.center →
  intersects_x_axis_at c (1, 0) (3, 0) →
  tangent_to_line c →
  c.radius = Real.sqrt 2 :=
by sorry

end circle_radius_is_sqrt_2_l816_81623


namespace divisibility_by_120_l816_81627

theorem divisibility_by_120 (n : ℕ) : ∃ k : ℤ, (n ^ 7 : ℤ) - (n ^ 3 : ℤ) = 120 * k := by
  sorry

end divisibility_by_120_l816_81627


namespace intersecting_chords_area_theorem_l816_81628

/-- Represents a circle with two intersecting chords -/
structure IntersectingChordsCircle where
  radius : ℝ
  chord_length : ℝ
  intersection_distance : ℝ

/-- Represents the area of a region in the form m*π - n*√d -/
structure RegionArea where
  m : ℕ
  n : ℕ
  d : ℕ

/-- Checks if a number is square-free (not divisible by the square of any prime) -/
def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p) ∣ n → p = 1

/-- The main theorem statement -/
theorem intersecting_chords_area_theorem (circle : IntersectingChordsCircle)
  (h1 : circle.radius = 50)
  (h2 : circle.chord_length = 90)
  (h3 : circle.intersection_distance = 24) :
  ∃ (area : RegionArea), 
    (area.m > 0 ∧ area.n > 0 ∧ area.d > 0) ∧
    is_square_free area.d ∧
    ∃ (region_area : ℝ), region_area = area.m * Real.pi - area.n * Real.sqrt area.d :=
by sorry

end intersecting_chords_area_theorem_l816_81628


namespace age_difference_l816_81669

/-- Given three people A, B, and C, where C is 14 years younger than A,
    prove that the total age of A and B is 14 years more than the total age of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 14) :
  (A + B) - (B + C) = 14 := by
  sorry

end age_difference_l816_81669


namespace sin_150_degrees_l816_81688

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l816_81688


namespace consecutive_integers_around_sqrt26_l816_81616

theorem consecutive_integers_around_sqrt26 (n m : ℤ) : 
  (n + 1 = m) → (n < Real.sqrt 26) → (Real.sqrt 26 < m) → (m + n = 11) := by
  sorry

end consecutive_integers_around_sqrt26_l816_81616


namespace pure_imaginary_product_l816_81661

theorem pure_imaginary_product (a b c d : ℝ) :
  (∃ k : ℝ, (a + b * Complex.I) * (c + d * Complex.I) = k * Complex.I) →
  (a * c - b * d = 0 ∧ a * d + b * c ≠ 0) :=
by sorry

end pure_imaginary_product_l816_81661


namespace number_puzzle_l816_81644

theorem number_puzzle (x y : ℝ) : x = 265 → (x / 5) + y = 61 → y = 8 := by
  sorry

end number_puzzle_l816_81644


namespace min_a_for_log_equation_solution_l816_81698

theorem min_a_for_log_equation_solution : 
  ∃ (a : ℝ), a > 0 ∧ (∀ a' : ℝ, a' < a → ¬∃ x : ℝ, (Real.log (a' - 2^x) / Real.log (1/2) = 2 + x)) ∧
  (∃ x : ℝ, (Real.log (a - 2^x) / Real.log (1/2) = 2 + x)) :=
by sorry

end min_a_for_log_equation_solution_l816_81698


namespace student_multiplication_problem_l816_81694

theorem student_multiplication_problem (x : ℝ) : 40 * x - 150 = 130 → x = 7 := by
  sorry

end student_multiplication_problem_l816_81694


namespace sector_arc_length_l816_81689

theorem sector_arc_length (θ : Real) (r : Real) (l : Real) : 
  θ = 2 * Real.pi / 3 → r = 2 → l = θ * r → l = 4 * Real.pi / 3 := by
  sorry

#check sector_arc_length

end sector_arc_length_l816_81689


namespace smallest_bob_number_l816_81643

def alice_number : ℕ := 45

def is_valid_bob_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ alice_number → p^2 ∣ n) ∧ (p ∣ n → p ∣ alice_number)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), is_valid_bob_number bob_number ∧
    ∀ (m : ℕ), is_valid_bob_number m → bob_number ≤ m ∧ bob_number = 2025 :=
sorry

end smallest_bob_number_l816_81643


namespace max_value_of_sum_of_squares_l816_81679

theorem max_value_of_sum_of_squares (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -1/2)
  (y_ge : y ≥ -3/2)
  (z_ge : z ≥ -1) :
  Real.sqrt (3 * x + 1.5) + Real.sqrt (3 * y + 4.5) + Real.sqrt (3 * z + 3) ≤ 9 := by
sorry

end max_value_of_sum_of_squares_l816_81679
