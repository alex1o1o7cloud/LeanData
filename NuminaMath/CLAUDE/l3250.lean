import Mathlib

namespace min_cubes_for_valid_config_l3250_325089

/-- Represents a cube with two opposite sides having protruding snaps and four sides with receptacle holes. -/
structure SpecialCube where
  snaps : Fin 2 → Bool
  holes : Fin 4 → Bool

/-- A configuration of special cubes. -/
def CubeConfiguration := List SpecialCube

/-- Checks if a configuration has no visible protruding snaps and only shows receptacle holes on visible surfaces. -/
def isValidConfiguration (config : CubeConfiguration) : Bool :=
  sorry

/-- The theorem stating that 6 is the minimum number of cubes required for a valid configuration. -/
theorem min_cubes_for_valid_config :
  ∃ (config : CubeConfiguration),
    config.length = 6 ∧ isValidConfiguration config ∧
    ∀ (smallerConfig : CubeConfiguration),
      smallerConfig.length < 6 → ¬isValidConfiguration smallerConfig :=
  sorry

end min_cubes_for_valid_config_l3250_325089


namespace floor_sum_inequality_l3250_325093

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := 
  x - floor x

-- State the theorem
theorem floor_sum_inequality (x y : ℝ) : 
  (floor x + floor y ≤ floor (x + y)) ∧ 
  (floor (x + y) ≤ floor x + floor y + 1) ∧ 
  (floor x + floor y = floor (x + y) ∨ floor (x + y) = floor x + floor y + 1) :=
sorry

end floor_sum_inequality_l3250_325093


namespace triangle_angle_equality_l3250_325082

open Real

theorem triangle_angle_equality (A B : ℝ) (a b : ℝ) 
  (h1 : sin A / a = cos B / b) 
  (h2 : a = b) : 
  B = π/4 := by
  sorry

end triangle_angle_equality_l3250_325082


namespace vector_on_line_and_parallel_l3250_325028

def line_x (t : ℝ) : ℝ := 5 * t + 3
def line_y (t : ℝ) : ℝ := t + 3

def vector_a : ℝ := 18
def vector_b : ℝ := 6

def parallel_vector_x : ℝ := 3
def parallel_vector_y : ℝ := 1

theorem vector_on_line_and_parallel :
  (∃ t : ℝ, line_x t = vector_a ∧ line_y t = vector_b) ∧
  (∃ k : ℝ, vector_a = k * parallel_vector_x ∧ vector_b = k * parallel_vector_y) :=
sorry

end vector_on_line_and_parallel_l3250_325028


namespace xiaoming_pencil_theorem_l3250_325041

/-- Represents the number of pencils and amount spent in Xiaoming's purchases -/
structure PencilPurchase where
  x : ℕ  -- number of pencils in first purchase
  y : ℕ  -- amount spent in first purchase in yuan

/-- Determines if a PencilPurchase satisfies the problem conditions -/
def satisfiesConditions (p : PencilPurchase) : Prop :=
  ∃ (price : ℚ), 
    price = p.y / p.x ∧  -- initial price per pencil
    (4 : ℚ) / 5 * price * (p.x + 10) = 4  -- condition after price drop

/-- The theorem stating the possible total numbers of pencils bought -/
theorem xiaoming_pencil_theorem (p : PencilPurchase) :
  satisfiesConditions p → (p.x + (p.x + 10) = 40 ∨ p.x + (p.x + 10) = 90) :=
by
  sorry

#check xiaoming_pencil_theorem

end xiaoming_pencil_theorem_l3250_325041


namespace total_age_difference_l3250_325040

-- Define the ages of A, B, and C as natural numbers
variable (A B C : ℕ)

-- Define the condition that C is 15 years younger than A
def age_difference : Prop := C = A - 15

-- Define the difference in total ages
def age_sum_difference : ℕ := (A + B) - (B + C)

-- Theorem statement
theorem total_age_difference (h : age_difference A C) : age_sum_difference A B C = 15 := by
  sorry

end total_age_difference_l3250_325040


namespace hiker_distance_theorem_l3250_325025

/-- Calculates the total distance walked by a hiker over three days given specific conditions -/
def total_distance_walked (day1_distance : ℕ) (day1_speed : ℕ) (day2_speed_increase : ℕ) (day3_speed : ℕ) (day3_hours : ℕ) : ℕ :=
  let day1_hours : ℕ := day1_distance / day1_speed
  let day2_hours : ℕ := day1_hours - 1
  let day2_speed : ℕ := day1_speed + day2_speed_increase
  let day2_distance : ℕ := day2_speed * day2_hours
  let day3_distance : ℕ := day3_speed * day3_hours
  day1_distance + day2_distance + day3_distance

/-- Theorem stating that the total distance walked is 53 miles given the specific conditions -/
theorem hiker_distance_theorem :
  total_distance_walked 18 3 1 5 3 = 53 := by
  sorry

end hiker_distance_theorem_l3250_325025


namespace total_stuffed_animals_l3250_325056

/-- 
Given:
- x: initial number of stuffed animals
- y: additional stuffed animals from mom
- z: factor of increase from dad's gift

Prove: The total number of stuffed animals is (x + y) * (1 + z)
-/
theorem total_stuffed_animals (x y : ℕ) (z : ℝ) :
  (x + y : ℝ) * (1 + z) = x + y + z * (x + y) := by
  sorry

end total_stuffed_animals_l3250_325056


namespace female_students_count_l3250_325075

theorem female_students_count (total : ℕ) (ways : ℕ) (f : ℕ) : 
  total = 8 → 
  ways = 30 → 
  (total - f) * (total - f - 1) * f = 2 * ways → 
  f = 3 := by
sorry

end female_students_count_l3250_325075


namespace problem_statement_l3250_325021

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x

theorem problem_statement (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2) (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end problem_statement_l3250_325021


namespace total_beanie_babies_l3250_325053

theorem total_beanie_babies (lori_beanie_babies sydney_beanie_babies : ℕ) :
  lori_beanie_babies = 300 →
  lori_beanie_babies = 15 * sydney_beanie_babies →
  lori_beanie_babies + sydney_beanie_babies = 320 := by
  sorry

end total_beanie_babies_l3250_325053


namespace bruce_pizza_production_l3250_325054

/-- The number of batches of pizza dough Bruce can make in a week -/
def pizzas_per_week (batches_per_sack : ℕ) (sacks_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  batches_per_sack * sacks_per_day * days_in_week

/-- Theorem stating that Bruce can make 525 batches of pizza dough in a week -/
theorem bruce_pizza_production :
  pizzas_per_week 15 5 7 = 525 := by
  sorry

end bruce_pizza_production_l3250_325054


namespace cable_intersections_6_8_l3250_325069

/-- The number of pairwise intersections of cables connecting houses across a street -/
def cable_intersections (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose m 2

/-- Theorem stating the number of cable intersections for 6 houses on one side and 8 on the other -/
theorem cable_intersections_6_8 :
  cable_intersections 6 8 = 420 := by
  sorry

end cable_intersections_6_8_l3250_325069


namespace trees_left_unwatered_l3250_325030

theorem trees_left_unwatered :
  let total_trees : ℕ := 29
  let boys_watering : ℕ := 9
  let trees_watered_per_boy : List ℕ := [2, 3, 1, 3, 2, 4, 3, 2, 5]
  let total_watered : ℕ := trees_watered_per_boy.sum
  total_trees - total_watered = 4 := by
sorry

end trees_left_unwatered_l3250_325030


namespace socks_price_l3250_325045

/-- Given the prices of jeans, t-shirt, and socks, where:
  1. The jeans cost twice as much as the t-shirt
  2. The t-shirt costs $10 more than the socks
  3. The jeans cost $30
  Prove that the socks cost $5 -/
theorem socks_price (jeans t_shirt socks : ℕ) 
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) : 
  socks = 5 := by
sorry

end socks_price_l3250_325045


namespace total_amount_l3250_325098

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
def money_problem (d : MoneyDistribution) : Prop :=
  d.y = 45 ∧                    -- y's share is 45 rupees
  d.y = 0.45 * d.x ∧            -- y gets 0.45 rupees for each rupee x gets
  d.z = 0.50 * d.x              -- z gets 0.50 rupees for each rupee x gets

/-- The theorem to prove -/
theorem total_amount (d : MoneyDistribution) :
  money_problem d → d.x + d.y + d.z = 195 :=
by
  sorry


end total_amount_l3250_325098


namespace distance_swam_against_current_l3250_325091

/-- Calculates the distance swam against a river current -/
theorem distance_swam_against_current
  (speed_still_water : ℝ)
  (current_speed : ℝ)
  (time : ℝ)
  (h1 : speed_still_water = 5)
  (h2 : current_speed = 1.2)
  (h3 : time = 3.1578947368421053)
  : (speed_still_water - current_speed) * time = 12 := by
  sorry

end distance_swam_against_current_l3250_325091


namespace intersection_of_sets_l3250_325071

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {1, 3, 4}
  M ∩ N = {1, 3} := by
  sorry

end intersection_of_sets_l3250_325071


namespace largest_solution_sum_l3250_325020

noncomputable def f (x : ℝ) : ℝ := 
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20)

theorem largest_solution_sum (n : ℝ) (p q r : ℕ+) :
  (∀ x : ℝ, f x = x^2 - 13*x - 6 → x ≤ n) ∧
  f n = n^2 - 13*n - 6 ∧
  n = p + Real.sqrt (q + Real.sqrt r) →
  p + q + r = 309 := by sorry

end largest_solution_sum_l3250_325020


namespace compute_expression_l3250_325092

theorem compute_expression : 75 * 1313 - 25 * 1313 + 50 * 1313 = 131300 := by
  sorry

end compute_expression_l3250_325092


namespace infinite_geometric_series_first_term_l3250_325051

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/3)
  (h_S : S = 18)
  (h_sum : S = a / (1 - r))
  (h_convergence : abs r < 1) :
  a = 12 :=
sorry

end infinite_geometric_series_first_term_l3250_325051


namespace solution_set_when_a_is_one_min_value_three_iff_l3250_325052

-- Define the function f
def f (x a : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - a)

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 5} = {x : ℝ | x ≤ -2 ∨ x ≥ 4/3} := by sorry

-- Theorem for part (2)
theorem min_value_three_iff :
  (∃ x : ℝ, f x a = 3) ∧ (∀ x : ℝ, f x a ≥ 3) ↔ a = 2 ∨ a = -4 := by sorry

end solution_set_when_a_is_one_min_value_three_iff_l3250_325052


namespace parabola_max_value_l3250_325057

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Whether a parabola opens downwards -/
def opens_downwards (p : Parabola) : Prop := p.a < 0

/-- The maximum value of a parabola -/
def max_value (p : Parabola) : ℝ := sorry

theorem parabola_max_value (p : Parabola) 
  (h1 : vertex p = (-3, 2)) 
  (h2 : opens_downwards p) : 
  max_value p = 2 := by sorry

end parabola_max_value_l3250_325057


namespace min_value_of_function_l3250_325015

theorem min_value_of_function (x : ℝ) (h : x > 0) : 4 * x + 1 / x^2 ≥ 5 ∧ ∃ y > 0, 4 * y + 1 / y^2 = 5 := by
  sorry

end min_value_of_function_l3250_325015


namespace long_jump_solution_l3250_325078

/-- Represents the long jump problem with given conditions -/
def LongJumpProblem (initial_avg : ℝ) (second_jump : ℝ) (second_avg : ℝ) (final_avg : ℝ) : Prop :=
  ∃ (n : ℕ) (third_jump : ℝ),
    -- Initial condition
    initial_avg = 3.80
    -- Second jump condition
    ∧ second_jump = 3.99
    -- New average after second jump
    ∧ second_avg = 3.81
    -- Final average after third jump
    ∧ final_avg = 3.82
    -- Relationship between jumps and averages
    ∧ (initial_avg * n + second_jump) / (n + 1) = second_avg
    ∧ (initial_avg * n + second_jump + third_jump) / (n + 2) = final_avg
    -- The third jump is the solution
    ∧ third_jump = 4.01

/-- Theorem stating the solution to the long jump problem -/
theorem long_jump_solution :
  LongJumpProblem 3.80 3.99 3.81 3.82 :=
by
  sorry

#check long_jump_solution

end long_jump_solution_l3250_325078


namespace jessica_watermelons_l3250_325048

/-- The number of watermelons Jessica initially grew -/
def initial_watermelons : ℕ := 35

/-- The number of watermelons eaten by rabbits -/
def eaten_watermelons : ℕ := 27

/-- The number of carrots Jessica grew (not used in the proof, but included for completeness) -/
def carrots : ℕ := 30

theorem jessica_watermelons : initial_watermelons - eaten_watermelons = 8 := by
  sorry

end jessica_watermelons_l3250_325048


namespace distribute_8_balls_3_boxes_l3250_325009

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 128 ways to distribute 8 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_8_balls_3_boxes : distribute_balls 8 3 = 128 := by
  sorry

end distribute_8_balls_3_boxes_l3250_325009


namespace tangent_length_circle_tangent_length_l3250_325063

/-- The length of a tangent to a circle from an external point -/
theorem tangent_length (r d l : ℝ) (hr : r > 0) (hd : d > r) : 
  r = 36 → d = 85 → l = 77 → l^2 = d^2 - r^2 := by
  sorry

/-- The main theorem stating the length of the tangent -/
theorem circle_tangent_length : 
  ∃ (l : ℝ), l = 77 ∧ l^2 = 85^2 - 36^2 := by
  sorry

end tangent_length_circle_tangent_length_l3250_325063


namespace cipher_solution_l3250_325005

/-- Represents a mapping from letters to digits -/
def Cipher := Char → Nat

/-- The condition that each letter represents a unique digit -/
def is_valid_cipher (c : Cipher) : Prop :=
  ∀ x y : Char, c x = c y → x = y

/-- The value of a word under a given cipher -/
def word_value (c : Cipher) (w : String) : Nat :=
  w.foldl (λ acc d => 10 * acc + c d) 0

/-- The main theorem -/
theorem cipher_solution (c : Cipher) 
  (h1 : is_valid_cipher c)
  (h2 : word_value c "СЕКРЕТ" - word_value c "ОТКРЫТ" = 20010)
  (h3 : c 'Т' = 9) :
  word_value c "СЕК" = 392 ∧ c 'О' = 2 :=
sorry

end cipher_solution_l3250_325005


namespace nate_running_distance_l3250_325007

/-- The total distance Nate ran in meters -/
def total_distance (field_length : ℝ) (additional_distance : ℝ) : ℝ :=
  4 * field_length + additional_distance

/-- Theorem stating the total distance Nate ran -/
theorem nate_running_distance :
  let field_length : ℝ := 168
  let additional_distance : ℝ := 500
  total_distance field_length additional_distance = 1172 := by
  sorry

end nate_running_distance_l3250_325007


namespace lindas_savings_l3250_325008

theorem lindas_savings (savings : ℝ) : 
  (2 / 3 : ℝ) * savings + 250 = savings → savings = 750 := by
  sorry

end lindas_savings_l3250_325008


namespace same_root_implies_a_value_l3250_325077

theorem same_root_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x - a = 0 ∧ x^2 + a*x - 2 = 0) → (a = 1 ∨ a = -1) :=
by sorry

end same_root_implies_a_value_l3250_325077


namespace mcdonalds_fries_cost_l3250_325070

/-- The cost of one pack of fries at McDonald's -/
def fries_cost : ℝ := 2

/-- The cost of a burger at McDonald's -/
def burger_cost : ℝ := 5

/-- The cost of a salad at McDonald's -/
def salad_cost (f : ℝ) : ℝ := 3 * f

/-- The total cost of the meal at McDonald's -/
def total_cost (f : ℝ) : ℝ := salad_cost f + burger_cost + 2 * f

theorem mcdonalds_fries_cost :
  fries_cost = 2 ∧ total_cost fries_cost = 15 :=
sorry

end mcdonalds_fries_cost_l3250_325070


namespace tan_alpha_3_implies_fraction_eq_two_thirds_l3250_325049

theorem tan_alpha_3_implies_fraction_eq_two_thirds (α : Real) (h : Real.tan α = 3) :
  1 / (Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) = 2 / 3 := by
  sorry

end tan_alpha_3_implies_fraction_eq_two_thirds_l3250_325049


namespace renovation_theorem_l3250_325087

/-- Represents a city grid --/
structure CityGrid where
  rows : Nat
  cols : Nat

/-- Calculates the minimum number of buildings after renovation --/
def minBuildingsAfterRenovation (grid : CityGrid) : Nat :=
  sorry

theorem renovation_theorem :
  (minBuildingsAfterRenovation ⟨20, 20⟩ = 25) ∧
  (minBuildingsAfterRenovation ⟨50, 90⟩ = 282) := by
  sorry

end renovation_theorem_l3250_325087


namespace inequality_solution_max_value_condition_l3250_325046

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem inequality_solution (x : ℝ) :
  f 2 x > 1 ↔ x < -3/2 ∨ x > 1 := by sorry

-- Part 2
theorem max_value_condition (a : ℝ) :
  (∃ x, f a x = 17/8 ∧ ∀ y, f a y ≤ 17/8) →
  (a = -2 ∨ a = -1/8) := by sorry

end inequality_solution_max_value_condition_l3250_325046


namespace circle_c_equation_l3250_325035

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points A and B
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 1)

-- Define the line l: x - y - 1 = 0
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the circle equation
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_c_equation :
  ∃ (c : Circle),
    (circle_equation c A.1 A.2) ∧
    (line_l B.1 B.2) ∧
    (∀ (x y : ℝ), line_l x y → (x - B.1)^2 + (y - B.2)^2 ≤ c.radius^2) ∧
    (circle_equation c x y ↔ (x - 3)^2 + y^2 = 2) :=
  sorry

end circle_c_equation_l3250_325035


namespace nine_knights_among_travelers_total_travelers_is_sixteen_l3250_325058

/-- A traveler can be either a knight or a liar -/
inductive TravelerType
  | Knight
  | Liar

/-- Represents a room in the hotel -/
structure Room where
  knights : Nat
  liars : Nat

/-- Represents the hotel with three rooms -/
structure Hotel where
  room1 : Room
  room2 : Room
  room3 : Room

def total_travelers : Nat := 16

/-- Vasily, who makes contradictory statements -/
def vasily : TravelerType := TravelerType.Liar

/-- The theorem stating that there must be 9 knights among the 16 travelers -/
theorem nine_knights_among_travelers (h : Hotel) : 
  h.room1.knights + h.room2.knights + h.room3.knights = 9 :=
by
  sorry

/-- The theorem stating that the total number of travelers is 16 -/
theorem total_travelers_is_sixteen (h : Hotel) :
  h.room1.knights + h.room1.liars + 
  h.room2.knights + h.room2.liars + 
  h.room3.knights + h.room3.liars = total_travelers :=
by
  sorry

end nine_knights_among_travelers_total_travelers_is_sixteen_l3250_325058


namespace lighthouse_model_height_l3250_325018

def original_height : ℝ := 60
def original_base_height : ℝ := 12
def original_base_volume : ℝ := 150000
def model_base_volume : ℝ := 0.15

theorem lighthouse_model_height :
  let scale_factor := (model_base_volume / original_base_volume) ^ (1/3)
  let model_height := original_height * scale_factor
  model_height * 100 = 60 := by sorry

end lighthouse_model_height_l3250_325018


namespace inequality_impossibility_l3250_325001

theorem inequality_impossibility (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(a > 0) := by
  sorry

end inequality_impossibility_l3250_325001


namespace remainder_3m_mod_5_l3250_325062

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end remainder_3m_mod_5_l3250_325062


namespace intersection_complement_equality_l3250_325029

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x < -2 ∨ x > 2}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Set.compl B) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end intersection_complement_equality_l3250_325029


namespace elvis_songwriting_time_l3250_325095

theorem elvis_songwriting_time (total_songs : ℕ) (studio_time : ℕ) (recording_time : ℕ) (editing_time : ℕ)
  (h1 : total_songs = 10)
  (h2 : studio_time = 5 * 60)  -- 5 hours in minutes
  (h3 : recording_time = 12)   -- 12 minutes per song
  (h4 : editing_time = 30)     -- 30 minutes for all songs
  : (studio_time - (total_songs * recording_time + editing_time)) / total_songs = 15 := by
  sorry

end elvis_songwriting_time_l3250_325095


namespace greatest_multiple_under_1000_l3250_325097

theorem greatest_multiple_under_1000 :
  ∃ n : ℕ, n < 1000 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_multiple_under_1000_l3250_325097


namespace division_remainder_proof_l3250_325038

theorem division_remainder_proof (L S : ℕ) (h1 : L - S = 1395) (h2 : L = 1656) 
  (h3 : ∃ q r, L = S * q + r ∧ q = 6 ∧ r < S) : 
  ∃ r, L = S * 6 + r ∧ r = 90 := by
sorry

end division_remainder_proof_l3250_325038


namespace polynomial_sum_l3250_325081

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 160 := by
sorry

end polynomial_sum_l3250_325081


namespace cellphone_surveys_count_l3250_325079

/-- Represents the weekly survey data for a worker --/
structure SurveyData where
  regularRate : ℕ
  totalSurveys : ℕ
  cellphoneRateIncrease : ℚ
  totalEarnings : ℕ

/-- Calculates the number of cellphone surveys given the survey data --/
def calculateCellphoneSurveys (data : SurveyData) : ℕ :=
  sorry

/-- Theorem stating that the number of cellphone surveys is 50 for the given data --/
theorem cellphone_surveys_count (data : SurveyData) 
  (h1 : data.regularRate = 30)
  (h2 : data.totalSurveys = 100)
  (h3 : data.cellphoneRateIncrease = 1/5)
  (h4 : data.totalEarnings = 3300) :
  calculateCellphoneSurveys data = 50 := by
  sorry

end cellphone_surveys_count_l3250_325079


namespace a_value_range_l3250_325034

/-- Proposition p: For any x, ax^2 + ax + 1 > 0 always holds true -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The equation x^2 - x + a = 0 has real roots -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of values for a satisfying the given conditions -/
def a_range (a : ℝ) : Prop := a < 0 ∨ (1/4 < a ∧ a < 4)

theorem a_value_range :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a_range a :=
by sorry

end a_value_range_l3250_325034


namespace xy_plus_one_eq_x_plus_y_l3250_325023

theorem xy_plus_one_eq_x_plus_y (x y : ℝ) :
  x * y + 1 = x + y ↔ x = 1 ∨ y = 1 := by
sorry

end xy_plus_one_eq_x_plus_y_l3250_325023


namespace expression_value_l3250_325022

theorem expression_value : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end expression_value_l3250_325022


namespace student_assignment_count_l3250_325050

/-- The number of ways to assign students to internship positions -/
def assignment_count (n_students : ℕ) (n_positions : ℕ) : ℕ :=
  (n_students.choose 2) * (n_positions.factorial)

/-- Theorem: There are 36 ways to assign 4 students to 3 internship positions -/
theorem student_assignment_count :
  assignment_count 4 3 = 36 :=
by sorry

end student_assignment_count_l3250_325050


namespace a_value_l3250_325094

theorem a_value (a : ℝ) : 3 ∈ ({1, -a^2, a-1} : Set ℝ) → a = 4 := by
  sorry

end a_value_l3250_325094


namespace darcy_folded_shirts_darcy_problem_l3250_325012

theorem darcy_folded_shirts (total_shirts : ℕ) (total_shorts : ℕ) (folded_shorts : ℕ) (remaining_to_fold : ℕ) : ℕ :=
  let total_clothing := total_shirts + total_shorts
  let folded_clothing := total_clothing - folded_shorts - remaining_to_fold
  let folded_shirts := folded_clothing - folded_shorts
  folded_shirts

theorem darcy_problem :
  darcy_folded_shirts 20 8 5 11 = 7 := by
  sorry

end darcy_folded_shirts_darcy_problem_l3250_325012


namespace rectangle_area_l3250_325085

/-- A rectangle with diagonal length x and length three times its width has area (3/10) * x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w : ℝ, 
  w > 0 ∧ 
  w^2 + (3*w)^2 = x^2 ∧ 
  w * (3*w) = (3/10) * x^2 := by
sorry

end rectangle_area_l3250_325085


namespace value_of_a_minus_b_plus_c_l3250_325065

theorem value_of_a_minus_b_plus_c 
  (a b c : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hc : |c| = 6)
  (hab : |a+b| = -(a+b))
  (hac : |a+c| = a+c) :
  a - b + c = 4 ∨ a - b + c = -2 := by
sorry

end value_of_a_minus_b_plus_c_l3250_325065


namespace power_product_squared_l3250_325006

theorem power_product_squared (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := by sorry

end power_product_squared_l3250_325006


namespace range_of_slope_intersecting_line_l3250_325003

/-- Given two points P and Q, and a line l that intersects the extension of PQ,
    prove the range of values for the slope of l. -/
theorem range_of_slope_intersecting_line (P Q : ℝ × ℝ) (m : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  ∃ (x y : ℝ), x + m * y + m = 0 ∧ 
    (∃ (t : ℝ), x = -1 + 3 * t ∧ y = 1 + t) →
  -3 < m ∧ m < -2/3 :=
sorry

end range_of_slope_intersecting_line_l3250_325003


namespace intersection_implies_equality_l3250_325024

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem intersection_implies_equality (a b c d : ℝ) 
  (h1 : f a b 1 = 1) 
  (h2 : g c d 1 = 1) : 
  a^5 + d^6 = c^6 - b^5 := by
  sorry

end intersection_implies_equality_l3250_325024


namespace cross_country_winning_scores_l3250_325060

/-- Represents a cross country meet with two teams of 6 runners each. -/
structure CrossCountryMeet where
  runners_per_team : Nat
  total_runners : Nat
  min_score : Nat
  max_score : Nat

/-- Calculates the number of possible winning scores in a cross country meet. -/
def possible_winning_scores (meet : CrossCountryMeet) : Nat :=
  meet.max_score - meet.min_score + 1

/-- Theorem stating the number of possible winning scores in the given cross country meet setup. -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.runners_per_team = 6 ∧
    meet.total_runners = 12 ∧
    meet.min_score = 21 ∧
    meet.max_score = 39 ∧
    possible_winning_scores meet = 19 := by
  sorry

end cross_country_winning_scores_l3250_325060


namespace pencil_sale_l3250_325083

theorem pencil_sale (x : ℕ) : 
  (2 * x) + (6 * 3) + (2 * 1) = 24 → x = 2 := by sorry

end pencil_sale_l3250_325083


namespace correct_quadratic_not_in_options_l3250_325064

theorem correct_quadratic_not_in_options : ∀ b c : ℝ,
  (∃ x y : ℝ, x + y = 10 ∧ x * y = c) →  -- From the first student's roots
  (∃ u v : ℝ, u + v = b ∧ u * v = -10) →  -- From the second student's roots
  (b = -10 ∧ c = -10) →  -- Derived from the conditions
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x^2 - 10*x - 10 = 0) →
  (x^2 + b*x + c ≠ x^2 - 9*x + 10) ∧
  (x^2 + b*x + c ≠ x^2 + 9*x + 10) ∧
  (x^2 + b*x + c ≠ x^2 - 9*x + 12) ∧
  (x^2 + b*x + c ≠ x^2 + 10*x - 21) :=
by sorry

end correct_quadratic_not_in_options_l3250_325064


namespace orthogonal_matrix_sum_of_squares_l3250_325036

theorem orthogonal_matrix_sum_of_squares (p q r s : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]
  (B.transpose = B⁻¹) → (p = s) → p^2 + q^2 + r^2 + s^2 = 2 := by
sorry

end orthogonal_matrix_sum_of_squares_l3250_325036


namespace circle_area_from_diameter_endpoints_l3250_325011

/-- The area of a circle with diameter endpoints C(5,9) and D(13,17) is 32π square units. -/
theorem circle_area_from_diameter_endpoints :
  let c : ℝ × ℝ := (5, 9)
  let d : ℝ × ℝ := (13, 17)
  let diameter_squared := (d.1 - c.1)^2 + (d.2 - c.2)^2
  let radius_squared := diameter_squared / 4
  π * radius_squared = 32 * π := by
  sorry

end circle_area_from_diameter_endpoints_l3250_325011


namespace polynomial_complex_roots_bounds_l3250_325067

-- Define the polynomial P(x)
def P (x : ℂ) : ℂ := x^3 + x^2 - x + 2

-- State the theorem
theorem polynomial_complex_roots_bounds :
  ∀ r : ℝ, (∃ z : ℂ, z.im ≠ 0 ∧ P z = r) ↔ (3 < r ∧ r < 49/27) :=
by sorry

end polynomial_complex_roots_bounds_l3250_325067


namespace square_of_product_l3250_325096

theorem square_of_product (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by
  sorry

end square_of_product_l3250_325096


namespace specific_frustum_small_cone_altitude_l3250_325017

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude

/-- Theorem: The altitude of the small cone cut off from a specific frustum is 18 -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := { altitude := 18, lower_base_area := 400 * Real.pi, upper_base_area := 100 * Real.pi }
  small_cone_altitude f = 18 := by sorry

end specific_frustum_small_cone_altitude_l3250_325017


namespace monthly_fee_is_two_l3250_325090

/-- Represents the monthly phone bill structure -/
structure PhoneBill where
  monthlyFee : ℝ
  perMinuteRate : ℝ
  minutesUsed : ℕ
  totalBill : ℝ

/-- Proves that the monthly fee is $2 given the specified conditions -/
theorem monthly_fee_is_two (bill : PhoneBill) 
    (h1 : bill.totalBill = bill.monthlyFee + bill.perMinuteRate * bill.minutesUsed)
    (h2 : bill.perMinuteRate = 0.12)
    (h3 : bill.totalBill = 23.36)
    (h4 : bill.minutesUsed = 178) :
    bill.monthlyFee = 2 := by
  sorry


end monthly_fee_is_two_l3250_325090


namespace max_ones_in_table_l3250_325002

/-- Represents a table with rows and columns -/
structure Table :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the constraints for the table -/
structure TableConstraints :=
  (table : Table)
  (row_sum_mod_3 : ℕ)
  (col_sum_mod_3 : ℕ)

/-- The maximum number of 1's that can be placed in the table -/
def max_ones (constraints : TableConstraints) : ℕ :=
  sorry

/-- The specific constraints for our problem -/
def our_constraints : TableConstraints :=
  { table := { rows := 2005, cols := 2006 },
    row_sum_mod_3 := 0,
    col_sum_mod_3 := 0 }

/-- The theorem to be proved -/
theorem max_ones_in_table :
  max_ones our_constraints = 1336 :=
sorry

end max_ones_in_table_l3250_325002


namespace specific_figure_perimeter_l3250_325076

/-- Represents a figure composed of unit squares -/
structure UnitSquareFigure where
  rows : Nat
  columns : Nat
  extra_column : Nat

/-- Calculates the perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : Nat :=
  sorry

/-- The specific figure described in the problem -/
def specific_figure : UnitSquareFigure :=
  { rows := 3, columns := 4, extra_column := 2 }

theorem specific_figure_perimeter :
  perimeter specific_figure = 13 := by sorry

end specific_figure_perimeter_l3250_325076


namespace pirate_treasure_division_l3250_325033

theorem pirate_treasure_division (S a b c d e : ℚ) : 
  a = (S - a) / 2 →
  b = (S - b) / 3 →
  c = (S - c) / 4 →
  d = (S - d) / 5 →
  e = 90 →
  S = a + b + c + d + e →
  S = 1800 := by
  sorry

end pirate_treasure_division_l3250_325033


namespace total_books_count_l3250_325004

def initial_books : ℕ := 35
def bought_books : ℕ := 21

theorem total_books_count : initial_books + bought_books = 56 := by
  sorry

end total_books_count_l3250_325004


namespace snickers_for_nintendo_switch_l3250_325080

def snickers_needed (total_points_needed : ℕ) (chocolate_bunnies_sold : ℕ) (points_per_bunny : ℕ) (points_per_snickers : ℕ) : ℕ :=
  let points_from_bunnies := chocolate_bunnies_sold * points_per_bunny
  let remaining_points := total_points_needed - points_from_bunnies
  remaining_points / points_per_snickers

theorem snickers_for_nintendo_switch : 
  snickers_needed 2000 8 100 25 = 48 := by
  sorry

end snickers_for_nintendo_switch_l3250_325080


namespace class_average_score_l3250_325055

theorem class_average_score (total_questions : ℕ) 
  (score_3_percent : ℝ) (score_2_percent : ℝ) (score_1_percent : ℝ) (score_0_percent : ℝ) :
  total_questions = 3 →
  score_3_percent = 0.3 →
  score_2_percent = 0.5 →
  score_1_percent = 0.1 →
  score_0_percent = 0.1 →
  score_3_percent + score_2_percent + score_1_percent + score_0_percent = 1 →
  3 * score_3_percent + 2 * score_2_percent + 1 * score_1_percent + 0 * score_0_percent = 2 := by
sorry

end class_average_score_l3250_325055


namespace zero_derivative_not_always_extremum_l3250_325013

/-- A function f: ℝ → ℝ is differentiable -/
def DifferentiableFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f

/-- x₀ is an extremum point of f if it's either a local maximum or minimum -/
def IsExtremumPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  IsLocalMax f x₀ ∨ IsLocalMin f x₀

/-- The statement that if f'(x₀) = 0, then x₀ is an extremum point of f -/
def ZeroDerivativeImpliesExtremum (f : ℝ → ℝ) : Prop :=
  ∀ x₀ : ℝ, DifferentiableAt ℝ f x₀ → deriv f x₀ = 0 → IsExtremumPoint f x₀

theorem zero_derivative_not_always_extremum :
  ¬ (∀ f : ℝ → ℝ, DifferentiableFunction f → ZeroDerivativeImpliesExtremum f) :=
by sorry

end zero_derivative_not_always_extremum_l3250_325013


namespace dealer_profit_is_sixty_percent_l3250_325099

/-- Calculates the dealer's profit percentage given the purchase and sale information. -/
def dealer_profit_percentage (purchase_quantity : ℕ) (purchase_price : ℚ) 
  (sale_quantity : ℕ) (sale_price : ℚ) : ℚ :=
  let cost_price_per_article := purchase_price / purchase_quantity
  let selling_price_per_article := sale_price / sale_quantity
  let profit_per_article := selling_price_per_article - cost_price_per_article
  (profit_per_article / cost_price_per_article) * 100

/-- Theorem stating that the dealer's profit percentage is 60% given the specified conditions. -/
theorem dealer_profit_is_sixty_percent :
  dealer_profit_percentage 15 25 12 32 = 60 := by
  sorry

end dealer_profit_is_sixty_percent_l3250_325099


namespace complex_equation_solution_l3250_325044

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (z + 2) * (1 + i^3) = 2

-- Theorem statement
theorem complex_equation_solution :
  ∃ z : ℂ, equation z ∧ z = -1 + i :=
sorry

end complex_equation_solution_l3250_325044


namespace hyperbola_equation_l3250_325019

/-- Given a hyperbola passing through the point (2√2, 1) with one asymptote equation y = 1/2x,
    its standard equation is x²/4 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, x^2 / 4 - y^2 = k ∧ (2 * Real.sqrt 2)^2 / 4 - 1^2 = k) ∧
  (∃ m : ℝ, y = 1/2 * x + m) →
  x^2 / 4 - y^2 = 1 :=
by sorry

end hyperbola_equation_l3250_325019


namespace intersection_area_bound_l3250_325032

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Define a function to reflect a triangle about a point
def reflectTriangle (t : Triangle) (p : ℝ × ℝ) : Triangle := sorry

-- Define a function to calculate the area of the intersection polygon
noncomputable def intersectionArea (t1 t2 : Triangle) : ℝ := sorry

-- Theorem statement
theorem intersection_area_bound (ABC : Triangle) (P : ℝ × ℝ) :
  intersectionArea ABC (reflectTriangle ABC P) ≤ (2/3) * triangleArea ABC := by
  sorry

end intersection_area_bound_l3250_325032


namespace trigonometric_inequality_l3250_325072

theorem trigonometric_inequality : 
  let a := Real.sin (2 * Real.pi / 7)
  let b := Real.cos (12 * Real.pi / 7)
  let c := Real.tan (9 * Real.pi / 7)
  c > a ∧ a > b := by sorry

end trigonometric_inequality_l3250_325072


namespace geometric_progression_proof_l3250_325074

theorem geometric_progression_proof (x : ℝ) (r : ℝ) : 
  (((30 + x) / (10 + x) = r) ∧ ((90 + x) / (30 + x) = r)) ↔ (x = 0 ∧ r = 3) :=
by sorry

end geometric_progression_proof_l3250_325074


namespace exterior_angle_of_right_triangle_l3250_325016

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define a right triangle
structure RightTriangle extends Triangle :=
  (right_angle : C = 90)

-- Theorem statement
theorem exterior_angle_of_right_triangle (t : RightTriangle) :
  180 - t.C = 90 := by sorry

end exterior_angle_of_right_triangle_l3250_325016


namespace digging_hours_calculation_l3250_325068

/-- Calculates the initial working hours per day given the conditions of the digging problem. -/
theorem digging_hours_calculation 
  (initial_men : ℕ) 
  (initial_depth : ℝ) 
  (new_depth : ℝ) 
  (new_hours : ℝ) 
  (extra_men : ℕ) 
  (h : initial_men = 63)
  (i : initial_depth = 30)
  (n : new_depth = 50)
  (w : new_hours = 6)
  (e : extra_men = 77) :
  ∃ (initial_hours : ℝ), 
    initial_hours = 8 ∧ 
    (initial_men : ℝ) * initial_hours * initial_depth = 
    ((initial_men : ℝ) + extra_men) * new_hours * new_depth := by
  sorry

#check digging_hours_calculation

end digging_hours_calculation_l3250_325068


namespace girls_boys_acquaintance_l3250_325073

theorem girls_boys_acquaintance (n : ℕ) :
  n > 1 →
  (∃ (girls_know : Fin (n + 1) → Fin (n + 1)) (boys_know : Fin n → ℕ),
    Function.Injective girls_know ∧
    (∀ i : Fin n, boys_know i = (n + 1) / 2) ∧
    (∀ i : Fin (n + 1), girls_know i ≤ n)) →
  Odd n :=
by sorry

end girls_boys_acquaintance_l3250_325073


namespace backyard_area_l3250_325039

/-- The area of a rectangular backyard given specific walking conditions -/
theorem backyard_area (length width : ℝ) : 
  (20 * length = 800) →
  (8 * (2 * length + 2 * width) = 800) →
  (length * width = 400) := by
  sorry

end backyard_area_l3250_325039


namespace polar_to_cartesian_l3250_325037

/-- Given a curve in polar coordinates r = p * sin(5θ), 
    this theorem states its equivalent form in Cartesian coordinates. -/
theorem polar_to_cartesian (p : ℝ) (x y : ℝ) :
  (∃ (θ : ℝ), x = (p * Real.sin (5 * θ)) * Real.cos θ ∧
               y = (p * Real.sin (5 * θ)) * Real.sin θ) ↔
  x^6 - 5*p*x^4*y + 10*p*x^2*y^3 + y^6 + 3*x^4*y^2 - p*y^5 + 3*x^2*y^4 = 0 :=
by sorry

end polar_to_cartesian_l3250_325037


namespace original_number_l3250_325088

theorem original_number : ∃ x : ℚ, 213 * x = 3408 ∧ x = 16 := by
  sorry

end original_number_l3250_325088


namespace triangle_longest_side_l3250_325000

theorem triangle_longest_side :
  ∀ x : ℝ,
  let side1 := 7
  let side2 := x + 4
  let side3 := 2*x + 1
  (side1 + side2 + side3 = 36) →
  (∃ longest : ℝ, longest = max side1 (max side2 side3) ∧ longest = 17) :=
by sorry

end triangle_longest_side_l3250_325000


namespace quarters_left_l3250_325061

/-- Given that Adam started with 88 quarters and spent 9 quarters at the arcade,
    prove that he had 79 quarters left. -/
theorem quarters_left (initial_quarters spent_quarters : ℕ) 
  (h1 : initial_quarters = 88)
  (h2 : spent_quarters = 9) :
  initial_quarters - spent_quarters = 79 := by
  sorry

end quarters_left_l3250_325061


namespace carly_job_applications_l3250_325027

/-- The number of job applications Carly sent to companies in her state -/
def in_state_applications : ℕ := 200

/-- The number of job applications Carly sent to companies in other states -/
def out_state_applications : ℕ := 2 * in_state_applications

/-- The total number of job applications Carly sent -/
def total_applications : ℕ := in_state_applications + out_state_applications

theorem carly_job_applications : total_applications = 600 := by
  sorry

end carly_job_applications_l3250_325027


namespace fraction_to_decimal_l3250_325059

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l3250_325059


namespace parabola_tangent_and_intersection_l3250_325047

/-- Parabola in the first quadrant -/
structure Parabola where
  n : ℝ
  pos_n : n > 0

/-- Point on the parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = c.n * x
  first_quadrant : x > 0 ∧ y > 0

/-- Line with slope and y-intercept -/
structure Line where
  m : ℝ
  b : ℝ

/-- Theorem about parabola tangent and intersection properties -/
theorem parabola_tangent_and_intersection
  (c : Parabola)
  (p : ParabolaPoint c)
  (h1 : p.x = 2)
  (h2 : (p.x + c.n / 4)^2 + p.y^2 = (5/2)^2) -- Distance from P to focus is 5/2
  (l₂ : Line)
  (h3 : l₂.m ≠ 0) :
  -- 1. The tangent at P intersects x-axis at (-2, 0)
  ∃ (q : ℝ × ℝ), q = (-2, 0) ∧
    (∃ (k : ℝ), k * (q.1 - p.x) + p.y = q.2 ∧ 
      ∀ (x y : ℝ), y^2 = c.n * x → (y - p.y) = k * (x - p.x) → x = q.1 ∧ y = q.2) ∧
  -- 2. If slopes of PA, PE, PB form arithmetic sequence, l₂ passes through (2, 0)
  (∀ (a b e : ℝ × ℝ),
    (a.2)^2 = c.n * a.1 ∧ (b.2)^2 = c.n * b.1 ∧ -- A and B on parabola
    a.1 = l₂.m * a.2 + l₂.b ∧ b.1 = l₂.m * b.2 + l₂.b ∧ -- A and B on l₂
    e.1 = -2 ∧ e.2 = -(l₂.b + 2) / l₂.m → -- E on l₁
    (((a.2 - p.y) / (a.1 - p.x) + (b.2 - p.y) / (b.1 - p.x)) / 2 = (e.2 - p.y) / (e.1 - p.x)) →
    l₂.b = 2) :=
sorry

end parabola_tangent_and_intersection_l3250_325047


namespace squirrel_nut_distance_l3250_325043

theorem squirrel_nut_distance (total_time : ℝ) (speed_without_nut : ℝ) (speed_with_nut : ℝ) 
  (h1 : total_time = 1200)
  (h2 : speed_without_nut = 5)
  (h3 : speed_with_nut = 3) :
  ∃ x : ℝ, x = 2250 ∧ x / speed_without_nut + x / speed_with_nut = total_time :=
by sorry

end squirrel_nut_distance_l3250_325043


namespace length_A_l3250_325031

-- Define the points
def A : ℝ × ℝ := (0, 9)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (2, 8)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' on the line y = x
def A' : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry

axiom A'_on_line : line_y_eq_x A'
axiom B'_on_line : line_y_eq_x B'

-- Define the lines AA' and BB'
def line_AA' (x : ℝ) : ℝ := sorry
def line_BB' (x : ℝ) : ℝ := sorry

-- C is on both lines AA' and BB'
axiom C_on_AA' : line_AA' C.1 = C.2
axiom C_on_BB' : line_BB' C.1 = C.2

-- Theorem: The length of A'B' is 2√2
theorem length_A'B'_is_2_sqrt_2 : 
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 := by sorry

end length_A_l3250_325031


namespace quadratic_real_root_condition_l3250_325010

theorem quadratic_real_root_condition (a b c : ℝ) : 
  (∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2*(a - b + c) * x + 3 = 0) →
  (a = c ∧ a = -b) := by
sorry

end quadratic_real_root_condition_l3250_325010


namespace rectangle_measurement_error_l3250_325042

/-- Proves that the percentage in excess for the first side is 12% given the conditions of the problem -/
theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) :
  (L * (1 + x / 100) * (W * 0.95) = L * W * 1.064) →
  x = 12 := by
sorry

end rectangle_measurement_error_l3250_325042


namespace initial_bacteria_count_l3250_325026

-- Define the doubling time in seconds
def doubling_time : ℕ := 15

-- Define the total time in seconds
def total_time : ℕ := 4 * 60

-- Define the final number of bacteria
def final_bacteria : ℕ := 2097152

-- Theorem statement
theorem initial_bacteria_count :
  ∃ (initial : ℕ), 
    initial * (2 ^ (total_time / doubling_time)) = final_bacteria ∧
    initial = 32 := by
  sorry

end initial_bacteria_count_l3250_325026


namespace product_of_three_numbers_l3250_325084

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 210 →
  8 * a = b - 11 →
  8 * a = c + 11 →
  a * b * c = 4173.75 := by
sorry

end product_of_three_numbers_l3250_325084


namespace max_product_sum_1988_l3250_325086

theorem max_product_sum_1988 (sequence : List Nat) : 
  (sequence.sum = 1988) → (sequence.all (· > 0)) →
  (sequence.prod ≤ 2 * 3^662) := by
  sorry

end max_product_sum_1988_l3250_325086


namespace intersection_x_coordinate_l3250_325066

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 4 * x - 25
def line2 (x y : ℝ) : Prop := 2 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 125 / 6 := by sorry

end intersection_x_coordinate_l3250_325066


namespace lonely_island_turtles_l3250_325014

theorem lonely_island_turtles : 
  ∀ (happy_island lonely_island : ℕ),
  happy_island = 60 →
  happy_island = 2 * lonely_island + 10 →
  lonely_island = 25 := by
sorry

end lonely_island_turtles_l3250_325014
