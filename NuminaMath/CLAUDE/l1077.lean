import Mathlib

namespace function_evaluation_l1077_107782

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 for any real number a. -/
theorem function_evaluation (a : ℝ) : (fun x : ℝ => x^2 + 1) (a + 1) = a^2 + 2*a + 2 := by
  sorry

end function_evaluation_l1077_107782


namespace first_part_to_total_ratio_l1077_107792

theorem first_part_to_total_ratio (total : ℚ) (first_part : ℚ) : 
  total = 782 →
  first_part = 204 →
  ∃ (x : ℚ), (x + 2/3 + 3/4) * first_part = total →
  first_part / total = 102 / 391 := by
  sorry

end first_part_to_total_ratio_l1077_107792


namespace triangle_property_l1077_107734

open Real

theorem triangle_property (A B C : ℝ) (R : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  cos (2 * A) - 3 * cos (B + C) - 1 = 0 ∧
  R = 1 →
  A = π / 3 ∧ 
  (∃ (S : ℝ), S ≤ 3 * sqrt 3 / 4 ∧ 
    ∀ (S' : ℝ), (∃ (a b c : ℝ), 
      a = 2 * R * sin A ∧
      b = 2 * R * sin B ∧
      c = 2 * R * sin C ∧
      S' = 1 / 2 * a * b * sin C) → 
    S' ≤ S) :=
by sorry

end triangle_property_l1077_107734


namespace line_passes_through_fixed_point_l1077_107702

/-- The line (a+1)x - y - 2a + 1 = 0 passes through the point (2,3) for all real a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), (a + 1) * 2 - 3 - 2 * a + 1 = 0 := by
  sorry

end line_passes_through_fixed_point_l1077_107702


namespace sum_of_digits_of_special_number_l1077_107754

/-- A number consisting of n digits all equal to 1 -/
def allOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem sum_of_digits_of_special_number :
  let L := allOnes 2022
  sumOfDigits (9 * L^2 + 2 * L) = 4044 := by
  sorry

end sum_of_digits_of_special_number_l1077_107754


namespace starWars_earnings_l1077_107798

/-- Represents movie financial data in millions of dollars -/
structure MovieData where
  cost : ℝ
  boxOffice : ℝ
  profit : ℝ

/-- The Lion King's financial data -/
def lionKing : MovieData := {
  cost := 10,
  boxOffice := 200,
  profit := 200 - 10
}

/-- Star Wars' financial data -/
def starWars : MovieData := {
  cost := 25,
  profit := 2 * lionKing.profit,
  boxOffice := 25 + 2 * lionKing.profit
}

/-- Theorem stating that Star Wars earned 405 million at the box office -/
theorem starWars_earnings : starWars.boxOffice = 405 := by
  sorry

#eval starWars.boxOffice

end starWars_earnings_l1077_107798


namespace chords_for_full_rotation_l1077_107791

/-- The number of chords needed to complete a full rotation when drawing chords on a larger circle
    tangent to a smaller concentric circle, given that the angle between consecutive chords is 60°. -/
def numChords : ℕ := 3

theorem chords_for_full_rotation (angle : ℝ) (h : angle = 60) :
  (numChords : ℝ) * angle = 360 := by
  sorry

end chords_for_full_rotation_l1077_107791


namespace min_value_expression_l1077_107729

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 12) / Real.sqrt (x - 4) ≥ 8 ∧ ∃ y : ℝ, y > 4 ∧ (y + 12) / Real.sqrt (y - 4) = 8 :=
sorry

end min_value_expression_l1077_107729


namespace necessary_but_not_sufficient_l1077_107700

theorem necessary_but_not_sufficient (a : ℝ) :
  (a^2 < 2*a → a < 2) ∧ ¬(∀ a, a < 2 → a^2 < 2*a) :=
sorry

end necessary_but_not_sufficient_l1077_107700


namespace geometric_sequence_sum_l1077_107714

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The statement to prove -/
theorem geometric_sequence_sum :
  (S 4 = 4) → (S 8 = 12) → (S 16 = 60) := by
  sorry

end geometric_sequence_sum_l1077_107714


namespace p_less_than_negative_one_l1077_107743

theorem p_less_than_negative_one (x y p : ℝ) 
  (eq1 : 3 * x - 2 * y = 4 - p)
  (eq2 : 4 * x - 3 * y = 2 + p)
  (ineq : x > y) : 
  p < -1 := by
sorry

end p_less_than_negative_one_l1077_107743


namespace sum_simplification_l1077_107769

theorem sum_simplification (n : ℕ) : 
  (Finset.range n).sum (λ i => (n - i) * 2^i) = 2^n + 1 - n - 2 := by
  sorry

end sum_simplification_l1077_107769


namespace girls_percentage_in_class_l1077_107785

theorem girls_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (girls_ratio : ℚ) / (boys_ratio + girls_ratio) * total_students / total_students * 100 = 57.14 := by
  sorry

end girls_percentage_in_class_l1077_107785


namespace fruit_stand_problem_l1077_107747

/-- Proves that the price of each apple is $0.80 given the conditions of the fruit stand problem -/
theorem fruit_stand_problem (total_cost : ℝ) (total_fruits : ℕ) (banana_price : ℝ) 
  (h1 : total_cost = 5.60)
  (h2 : total_fruits = 9)
  (h3 : banana_price = 0.60) :
  ∃ (apple_price : ℝ) (num_apples : ℕ),
    apple_price = 0.80 ∧
    num_apples ≤ total_fruits ∧
    apple_price * num_apples + banana_price * (total_fruits - num_apples) = total_cost :=
by sorry

end fruit_stand_problem_l1077_107747


namespace min_value_of_cosine_sum_l1077_107741

theorem min_value_of_cosine_sum (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : 0 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : 0 ≤ z ∧ z ≤ Real.pi / 2) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ 1 := by
  sorry

end min_value_of_cosine_sum_l1077_107741


namespace correct_mean_after_errors_l1077_107756

theorem correct_mean_after_errors (n : ℕ) (initial_mean : ℚ) 
  (error1_actual error1_copied error2_actual error2_copied error3_actual error3_copied : ℚ) :
  n = 50 ∧ 
  initial_mean = 325 ∧
  error1_actual = 200 ∧ error1_copied = 150 ∧
  error2_actual = 175 ∧ error2_copied = 220 ∧
  error3_actual = 592 ∧ error3_copied = 530 →
  let incorrect_sum := n * initial_mean
  let correction := (error1_actual - error1_copied) + (error3_actual - error3_copied) - (error2_actual - error2_copied)
  let corrected_sum := incorrect_sum + correction
  let correct_mean := corrected_sum / n
  correct_mean = 326.34 := by
sorry


end correct_mean_after_errors_l1077_107756


namespace coefficient_x4_in_q_squared_l1077_107795

/-- Given q(x) = x^5 - 4x^2 + 3, prove that the coefficient of x^4 in (q(x))^2 is 16 -/
theorem coefficient_x4_in_q_squared (x : ℝ) : 
  let q : ℝ → ℝ := λ x => x^5 - 4*x^2 + 3
  (q x)^2 = x^10 - 8*x^7 + 16*x^4 + 6*x^5 - 24*x^2 + 9 := by
  sorry


end coefficient_x4_in_q_squared_l1077_107795


namespace population_growth_problem_l1077_107703

theorem population_growth_problem (x y z : ℕ) : 
  (3/2)^x * (128/225)^y * (5/6)^z = 2 ↔ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end population_growth_problem_l1077_107703


namespace blue_flower_percentage_l1077_107787

theorem blue_flower_percentage (total_flowers : ℕ) (green_flowers : ℕ) (yellow_flowers : ℕ)
  (h1 : total_flowers = 96)
  (h2 : green_flowers = 9)
  (h3 : yellow_flowers = 12)
  : (total_flowers - (green_flowers + 3 * green_flowers + yellow_flowers)) / total_flowers * 100 = 50 := by
  sorry

end blue_flower_percentage_l1077_107787


namespace chess_game_probability_l1077_107740

theorem chess_game_probability (prob_draw prob_B_win : ℝ) 
  (h1 : prob_draw = 1/2)
  (h2 : prob_B_win = 1/3) :
  prob_draw + prob_B_win = 5/6 := by
  sorry

end chess_game_probability_l1077_107740


namespace race_distance_proof_l1077_107774

/-- The distance of a race where:
  * A covers the distance in 36 seconds
  * B covers the distance in 45 seconds
  * A beats B by 22 meters
-/
def race_distance : ℝ := 110

theorem race_distance_proof (A_time B_time : ℝ) (beat_distance : ℝ) 
  (h1 : A_time = 36)
  (h2 : B_time = 45)
  (h3 : beat_distance = 22)
  (h4 : A_time * (race_distance / B_time) + beat_distance = race_distance) :
  race_distance = 110 := by
  sorry

end race_distance_proof_l1077_107774


namespace cricketer_average_last_four_matches_l1077_107728

/-- Calculates the average score for the last 4 matches of a cricketer given the average score for all 10 matches and the average score for the first 6 matches. -/
def average_last_four_matches (total_average : ℚ) (first_six_average : ℚ) : ℚ :=
  let total_runs := total_average * 10
  let first_six_runs := first_six_average * 6
  let last_four_runs := total_runs - first_six_runs
  last_four_runs / 4

/-- Theorem stating that given a cricketer with an average score of 38.9 runs for 10 matches
    and an average of 42 runs for the first 6 matches, the average for the last 4 matches is 34.25 runs. -/
theorem cricketer_average_last_four_matches :
  average_last_four_matches (389 / 10) 42 = 34.25 := by
  sorry

end cricketer_average_last_four_matches_l1077_107728


namespace ackermann_3_2_l1077_107713

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 29 := by sorry

end ackermann_3_2_l1077_107713


namespace convex_polygon_three_obtuse_sides_l1077_107750

/-- A convex polygon with n sides and exactly 3 obtuse angles -/
structure ConvexPolygon (n : ℕ) :=
  (sides : ℕ)
  (is_convex : Bool)
  (obtuse_angles : ℕ)
  (sides_eq : sides = n)
  (convex : is_convex = true)
  (obtuse : obtuse_angles = 3)

/-- The theorem stating that a convex polygon with exactly 3 obtuse angles can only have 5 or 6 sides -/
theorem convex_polygon_three_obtuse_sides (n : ℕ) (p : ConvexPolygon n) : n = 5 ∨ n = 6 :=
sorry

end convex_polygon_three_obtuse_sides_l1077_107750


namespace blue_pencil_length_l1077_107772

theorem blue_pencil_length (total : ℝ) (purple : ℝ) (black : ℝ) (blue : ℝ)
  (h_total : total = 4)
  (h_purple : purple = 1.5)
  (h_black : black = 0.5)
  (h_sum : total = purple + black + blue) :
  blue = 2 := by
sorry

end blue_pencil_length_l1077_107772


namespace arithmetic_seq_sum_l1077_107725

/-- An arithmetic sequence with given first and third terms -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum (a : ℕ → ℤ) :
  arithmetic_seq a → a 1 = 1 → a 3 = -3 →
  a 1 - a 2 - a 3 - a 4 - a 5 = 17 := by
  sorry

end arithmetic_seq_sum_l1077_107725


namespace segment_rectangle_configurations_l1077_107726

/-- Represents a rectangle made of segments -/
structure SegmentRectangle where
  m : ℕ  -- number of segments on one side
  n : ℕ  -- number of segments on the other side

/-- The total number of segments in a rectangle -/
def total_segments (rect : SegmentRectangle) : ℕ :=
  rect.m * (rect.n + 1) + rect.n * (rect.m + 1)

/-- Possible configurations of a rectangle with 1997 segments -/
def is_valid_configuration (rect : SegmentRectangle) : Prop :=
  total_segments rect = 1997 ∧
  (rect.m = 2 ∧ rect.n = 399) ∨
  (rect.m = 8 ∧ rect.n = 117) ∨
  (rect.m = 23 ∧ rect.n = 42)

/-- Main theorem: The only valid configurations are 2×399, 8×117, and 23×42 -/
theorem segment_rectangle_configurations :
  ∀ rect : SegmentRectangle, total_segments rect = 1997 → is_valid_configuration rect :=
by sorry

end segment_rectangle_configurations_l1077_107726


namespace yahs_to_bahs_conversion_l1077_107739

/-- Given the conversion rates between bahs, rahs, and yahs, 
    prove that 500 yahs are equal in value to 100 bahs. -/
theorem yahs_to_bahs_conversion 
  (bah_to_rah : ℚ) (rah_to_yah : ℚ)
  (h1 : bah_to_rah = 30 / 10)  -- 10 bahs = 30 rahs
  (h2 : rah_to_yah = 10 / 6)   -- 6 rahs = 10 yahs
  : 500 * (1 / rah_to_yah) * (1 / bah_to_rah) = 100 := by
  sorry

end yahs_to_bahs_conversion_l1077_107739


namespace pizza_time_is_ten_minutes_l1077_107753

/-- Represents the pizza-making scenario --/
structure PizzaScenario where
  totalTime : ℕ        -- Total time in hours
  initialFlour : ℕ     -- Initial flour in kg
  flourPerPizza : ℚ    -- Flour required per pizza in kg
  remainingPizzas : ℕ  -- Number of pizzas that can be made with remaining flour

/-- Calculates the time taken to make each pizza --/
def timeTakenPerPizza (scenario : PizzaScenario) : ℚ :=
  let totalMinutes := scenario.totalTime * 60
  let usedFlour := scenario.initialFlour - (scenario.remainingPizzas * scenario.flourPerPizza)
  let pizzasMade := usedFlour / scenario.flourPerPizza
  totalMinutes / pizzasMade

/-- Theorem stating that the time taken per pizza is 10 minutes --/
theorem pizza_time_is_ten_minutes (scenario : PizzaScenario) 
    (h1 : scenario.totalTime = 7)
    (h2 : scenario.initialFlour = 22)
    (h3 : scenario.flourPerPizza = 1/2)
    (h4 : scenario.remainingPizzas = 2) :
    timeTakenPerPizza scenario = 10 := by
  sorry


end pizza_time_is_ten_minutes_l1077_107753


namespace cafeteria_cottage_pies_l1077_107778

/-- The number of lasagnas made by the cafeteria -/
def num_lasagnas : ℕ := 100

/-- The amount of ground mince used per lasagna (in pounds) -/
def mince_per_lasagna : ℕ := 2

/-- The amount of ground mince used per cottage pie (in pounds) -/
def mince_per_cottage_pie : ℕ := 3

/-- The total amount of ground mince used (in pounds) -/
def total_mince : ℕ := 500

/-- The number of cottage pies made by the cafeteria -/
def num_cottage_pies : ℕ := (total_mince - num_lasagnas * mince_per_lasagna) / mince_per_cottage_pie

theorem cafeteria_cottage_pies :
  num_cottage_pies = 100 :=
by sorry

end cafeteria_cottage_pies_l1077_107778


namespace boat_round_trip_time_l1077_107749

theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 2)
  (h3 : distance = 7560)
  : (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 960 :=
by sorry

end boat_round_trip_time_l1077_107749


namespace assignment_count_theorem_l1077_107711

/-- The number of ways to assign 4 distinct objects to 3 distinct groups,
    where each group must contain at least one object. -/
def assignment_count : ℕ := 36

/-- The number of distinct objects to be assigned. -/
def num_objects : ℕ := 4

/-- The number of distinct groups to which objects are assigned. -/
def num_groups : ℕ := 3

theorem assignment_count_theorem :
  (∀ assignment : Fin num_objects → Fin num_groups,
    (∀ g : Fin num_groups, ∃ o : Fin num_objects, assignment o = g) →
    ∃! c : ℕ, c = assignment_count) :=
sorry

end assignment_count_theorem_l1077_107711


namespace pizza_slices_l1077_107779

theorem pizza_slices (total_slices : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) : 
  total_slices = 16 → 
  num_pizzas = 2 → 
  total_slices = num_pizzas * slices_per_pizza → 
  slices_per_pizza = 8 := by
  sorry

end pizza_slices_l1077_107779


namespace g_sum_property_l1077_107794

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 5 * x^4 + 7

theorem g_sum_property : g 10 = 15 → g 10 + g (-10) = 30 := by
  sorry

end g_sum_property_l1077_107794


namespace cuboid_height_theorem_l1077_107770

/-- Represents a cuboid (rectangular box) -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- Theorem: A cuboid with volume 315, width 9, and length 7 has height 5 -/
theorem cuboid_height_theorem (c : Cuboid) 
  (h_volume : volume c = 315)
  (h_width : c.width = 9)
  (h_length : c.length = 7) : 
  c.height = 5 := by
  sorry

end cuboid_height_theorem_l1077_107770


namespace max_value_quadratic_function_l1077_107757

/-- Given a quadratic function f(x) = ax^2 - 2x + c with x ∈ ℝ and range [0, +∞),
    the maximum value of 1/(c+1) + 4/(a+4) is 4/3 -/
theorem max_value_quadratic_function (a c : ℝ) : 
  (∀ x, a * x^2 - 2*x + c ≥ 0) →  -- Range is [0, +∞)
  (∃ x, a * x^2 - 2*x + c = 0) →  -- Minimum value is 0
  (∃ M, M = (1 / (c + 1) + 4 / (a + 4)) ∧ 
   ∀ a' c', (∀ x, a' * x^2 - 2*x + c' ≥ 0) → 
             (∃ x, a' * x^2 - 2*x + c' = 0) → 
             M ≥ (1 / (c' + 1) + 4 / (a' + 4))) →
  (1 / (c + 1) + 4 / (a + 4)) ≤ 4/3 := by
sorry

end max_value_quadratic_function_l1077_107757


namespace candy_problem_l1077_107762

/-- Represents a set of candies with three types: hard, chocolate, and gummy -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies in a set -/
def total (s : CandySet) : ℕ := s.hard + s.chocolate + s.gummy

theorem candy_problem (s1 s2 s3 : CandySet) 
  (h1 : s1.hard + s2.hard + s3.hard = s1.chocolate + s2.chocolate + s3.chocolate)
  (h2 : s1.hard + s2.hard + s3.hard = s1.gummy + s2.gummy + s3.gummy)
  (h3 : s1.chocolate = s1.gummy)
  (h4 : s1.hard = s1.chocolate + 7)
  (h5 : s2.hard = s2.chocolate)
  (h6 : s2.gummy = s2.hard - 15)
  (h7 : s3.hard = 0) : 
  total s3 = 29 := by
sorry

end candy_problem_l1077_107762


namespace integer_solution_problem_l1077_107709

theorem integer_solution_problem :
  ∀ a b c : ℤ,
  1 < a ∧ a < b ∧ b < c →
  (∃ k : ℤ, k * ((a - 1) * (b - 1) * (c - 1)) = a * b * c - 1) →
  ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end integer_solution_problem_l1077_107709


namespace first_customer_payment_l1077_107760

/-- The cost of one MP3 player -/
def mp3_cost : ℕ := sorry

/-- The cost of one set of headphones -/
def headphone_cost : ℕ := 30

/-- The total cost for the second customer -/
def second_customer_total : ℕ := 480

/-- The number of MP3 players bought by the first customer -/
def first_customer_mp3 : ℕ := 5

/-- The number of headphones bought by the first customer -/
def first_customer_headphones : ℕ := 8

/-- The number of MP3 players bought by the second customer -/
def second_customer_mp3 : ℕ := 3

/-- The number of headphones bought by the second customer -/
def second_customer_headphones : ℕ := 4

theorem first_customer_payment :
  second_customer_mp3 * mp3_cost + second_customer_headphones * headphone_cost = second_customer_total →
  first_customer_mp3 * mp3_cost + first_customer_headphones * headphone_cost = 840 :=
by sorry

end first_customer_payment_l1077_107760


namespace cougar_sleep_duration_l1077_107723

/-- Given a cougar's nightly sleep duration C and a zebra's nightly sleep duration Z,
    where Z = C + 2 and C + Z = 70, prove that C = 34. -/
theorem cougar_sleep_duration (C Z : ℕ) (h1 : Z = C + 2) (h2 : C + Z = 70) : C = 34 := by
  sorry

end cougar_sleep_duration_l1077_107723


namespace equal_perimeter_parallel_sections_l1077_107765

/-- A tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A plane that intersects a tetrahedron -/
structure IntersectingPlane where
  plane : Plane
  tetrahedron : Tetrahedron

/-- The perimeter of the intersection between a plane and a tetrahedron -/
def intersectionPerimeter (p : IntersectingPlane) : ℝ := sorry

/-- Two edges of a tetrahedron are disjoint -/
def disjointEdges (t : Tetrahedron) (e1 e2 : Segment) : Prop := sorry

/-- A plane is parallel to two edges of a tetrahedron -/
def parallelToEdges (p : IntersectingPlane) (e1 e2 : Segment) : Prop := sorry

/-- The length of a segment -/
def length (s : Segment) : ℝ := sorry

theorem equal_perimeter_parallel_sections (t : Tetrahedron) 
  (e1 e2 : Segment) (p1 p2 : IntersectingPlane) :
  disjointEdges t e1 e2 →
  length e1 = length e2 →
  parallelToEdges p1 e1 e2 →
  parallelToEdges p2 e1 e2 →
  intersectionPerimeter p1 = intersectionPerimeter p2 := by
  sorry

end equal_perimeter_parallel_sections_l1077_107765


namespace first_box_nonempty_count_l1077_107746

def total_boxes : Nat := 4
def total_balls : Nat := 3

def ways_with_first_box_nonempty : Nat :=
  total_boxes ^ total_balls - (total_boxes - 1) ^ total_balls

theorem first_box_nonempty_count :
  ways_with_first_box_nonempty = 37 := by
  sorry

end first_box_nonempty_count_l1077_107746


namespace restaurant_tax_calculation_l1077_107708

/-- Proves that the tax amount is $3 given the initial money, order costs, and change received --/
theorem restaurant_tax_calculation (lee_money : ℕ) (friend_money : ℕ) 
  (wings_cost : ℕ) (salad_cost : ℕ) (soda_cost : ℕ) (change : ℕ) : ℕ :=
by
  -- Define the given conditions
  have h1 : lee_money = 10 := by sorry
  have h2 : friend_money = 8 := by sorry
  have h3 : wings_cost = 6 := by sorry
  have h4 : salad_cost = 4 := by sorry
  have h5 : soda_cost = 1 := by sorry
  have h6 : change = 3 := by sorry

  -- Calculate the total initial money
  let total_money := lee_money + friend_money

  -- Calculate the cost before tax
  let cost_before_tax := wings_cost + salad_cost + 2 * soda_cost

  -- Calculate the total spent including tax
  let total_spent := total_money - change

  -- Calculate the tax
  let tax := total_spent - cost_before_tax

  -- Prove that the tax is 3
  exact 3

end restaurant_tax_calculation_l1077_107708


namespace greatest_integer_inequality_l1077_107732

theorem greatest_integer_inequality : 
  (∀ x : ℤ, (1 / 4 : ℚ) + (x : ℚ) / 9 < 7 / 8 → x ≤ 5) ∧ 
  ((1 / 4 : ℚ) + (5 : ℚ) / 9 < 7 / 8) := by
sorry

end greatest_integer_inequality_l1077_107732


namespace cuboid_probabilities_l1077_107796

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of unit cubes in a cuboid -/
def Cuboid.totalUnitCubes (c : Cuboid) : ℕ := c.length * c.width * c.height

/-- Calculates the number of unit cubes with no faces painted -/
def Cuboid.noPaintedFaces (c : Cuboid) : ℕ := (c.length - 2) * (c.width - 2) * (c.height - 2)

/-- Calculates the number of unit cubes with two faces painted -/
def Cuboid.twoFacesPainted (c : Cuboid) : ℕ :=
  (c.length - 2) * c.width + (c.width - 2) * c.height + (c.height - 2) * c.length

/-- Calculates the number of unit cubes with three faces painted -/
def Cuboid.threeFacesPainted (c : Cuboid) : ℕ := 8

theorem cuboid_probabilities (c : Cuboid) (h1 : c.length = 3) (h2 : c.width = 4) (h3 : c.height = 5) :
  (c.noPaintedFaces : ℚ) / c.totalUnitCubes = 1 / 10 ∧
  ((c.twoFacesPainted + c.threeFacesPainted : ℚ) / c.totalUnitCubes = 8 / 15) := by
  sorry

end cuboid_probabilities_l1077_107796


namespace least_integer_satisfying_inequality_l1077_107751

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (∀ x : ℤ, 3 * |x| - 4 < 20 → y ≤ x) → y = -7 :=
by sorry

end least_integer_satisfying_inequality_l1077_107751


namespace function_properties_l1077_107797

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a 1 > 0) →
  (∀ x, f a x > b ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3) :=
by sorry

end function_properties_l1077_107797


namespace endpoint_is_200_l1077_107744

/-- The endpoint of a range of even integers starting from 20, given that its average
    is 35 greater than the average of even integers from 10 to 140 inclusive. -/
def endpoint : ℕ :=
  let start1 := 20
  let start2 := 10
  let end2 := 140
  let diff := 35
  let avg2 := (start2 + end2) / 2
  let endpoint := 2 * (avg2 + diff) - start1
  endpoint

theorem endpoint_is_200 : endpoint = 200 := by
  sorry

end endpoint_is_200_l1077_107744


namespace total_capacity_l1077_107767

/-- Represents the capacity of boats -/
structure BoatCapacity where
  large : ℕ
  small : ℕ

/-- The capacity of different combinations of boats -/
def boat_combinations (c : BoatCapacity) : Prop :=
  c.large + 4 * c.small = 46 ∧ 2 * c.large + 3 * c.small = 57

/-- The theorem to prove -/
theorem total_capacity (c : BoatCapacity) :
  boat_combinations c → 3 * c.large + 6 * c.small = 96 := by
  sorry


end total_capacity_l1077_107767


namespace students_not_in_biology_l1077_107759

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 840)
  (h2 : biology_percentage = 35 / 100) :
  total_students - (total_students * biology_percentage).floor = 546 := by
  sorry

end students_not_in_biology_l1077_107759


namespace six_people_arrangement_l1077_107731

theorem six_people_arrangement (n : ℕ) (h : n = 6) : 
  n.factorial - 2 * (n-1).factorial - 2 * 2 * (n-1).factorial + 2 * 2 * (n-2).factorial = 96 :=
by sorry

end six_people_arrangement_l1077_107731


namespace slope_characterization_l1077_107748

/-- The set of all possible slopes for a line with y-intercept (0,3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -2/5 ∨ m ≥ 2/5}

/-- The equation of the line with slope m and y-intercept (0,3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem slope_characterization :
  ∀ m : ℝ, m ∈ possible_slopes ↔
    ∃ x : ℝ, ellipse_equation x (line_equation m x) := by sorry

end slope_characterization_l1077_107748


namespace quadratic_inequality_solution_range_l1077_107717

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 1

-- State the theorem
theorem quadratic_inequality_solution_range (b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : ∀ x, f b x > 0 ↔ x < x₁ ∨ x > x₂)
  (h2 : x₁ < 1)
  (h3 : x₂ > 1) :
  b > 2 ∧ b ∈ Set.Ioi 2 :=
by sorry

end quadratic_inequality_solution_range_l1077_107717


namespace volume_of_inscribed_sphere_l1077_107777

/-- The volume of a sphere inscribed in a cube with side length 8 inches -/
theorem volume_of_inscribed_sphere (π : ℝ) : ℝ := by
  -- Define the side length of the cube
  let cube_side : ℝ := 8

  -- Define the radius of the inscribed sphere
  let sphere_radius : ℝ := cube_side / 2

  -- Define the volume of the sphere
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3

  -- Prove that the volume equals (256/3)π cubic inches
  have : sphere_volume = (256 / 3) * π := by sorry

  -- Return the result
  exact (256 / 3) * π

end volume_of_inscribed_sphere_l1077_107777


namespace cost_of_sneakers_l1077_107773

/-- Given the costs of items John bought, prove the cost of sneakers. -/
theorem cost_of_sneakers
  (total_cost : ℕ)
  (racket_cost : ℕ)
  (outfit_cost : ℕ)
  (h1 : total_cost = 750)
  (h2 : racket_cost = 300)
  (h3 : outfit_cost = 250) :
  total_cost - racket_cost - outfit_cost = 200 := by
  sorry

end cost_of_sneakers_l1077_107773


namespace acute_angle_inequalities_l1077_107761

theorem acute_angle_inequalities (α β : Real) 
  (h_α : 0 < α ∧ α < Real.pi / 2) 
  (h_β : 0 < β ∧ β < Real.pi / 2) : 
  (Real.sin (α + β) < Real.cos α + Real.cos β) ∧ 
  (Real.sin (α - β) < Real.cos α + Real.cos β) := by
  sorry

end acute_angle_inequalities_l1077_107761


namespace sum_of_squares_lower_bound_l1077_107710

theorem sum_of_squares_lower_bound (x y z m : ℝ) (h : x + y + z = m) :
  x^2 + y^2 + z^2 ≥ m^2 / 3 := by
  sorry

end sum_of_squares_lower_bound_l1077_107710


namespace divisor_problem_l1077_107707

theorem divisor_problem (N D : ℕ) (h1 : N % D = 255) (h2 : (2 * N) % D = 112) : D = 398 := by
  sorry

end divisor_problem_l1077_107707


namespace arithmetic_sequence_sum_l1077_107704

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 5 = 3 →
  a 6 = -2 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 0 :=
by
  sorry

end arithmetic_sequence_sum_l1077_107704


namespace complex_fraction_cube_l1077_107799

theorem complex_fraction_cube (i : ℂ) (h : i^2 = -1) :
  ((1 + i) / (1 - i))^3 = -i := by sorry

end complex_fraction_cube_l1077_107799


namespace root_equation_coefficient_l1077_107768

theorem root_equation_coefficient (a : ℝ) : (2 : ℝ)^2 + a * 2 - 2 = 0 → a = -1 := by
  sorry

end root_equation_coefficient_l1077_107768


namespace jokes_increase_factor_l1077_107720

/-- The factor by which Jessy and Alan increased their jokes -/
def increase_factor (first_saturday_jokes : ℕ) (total_jokes : ℕ) : ℚ :=
  (total_jokes - first_saturday_jokes : ℚ) / first_saturday_jokes

/-- Theorem stating that the increase factor is 2 -/
theorem jokes_increase_factor : increase_factor 18 54 = 2 := by
  sorry

#eval increase_factor 18 54

end jokes_increase_factor_l1077_107720


namespace fraction_problem_l1077_107788

theorem fraction_problem (N : ℚ) (h : (1/4) * (1/3) * (2/5) * N = 30) :
  ∃ F : ℚ, F * N = 120 ∧ F = 2/15 := by sorry

end fraction_problem_l1077_107788


namespace div_mul_sqrt_three_reciprocal_result_equals_one_l1077_107781

theorem div_mul_sqrt_three_reciprocal (x : ℝ) (h : x > 0) : 3 / Real.sqrt x * (1 / Real.sqrt x) = 3 / x :=
by sorry

theorem result_equals_one : 3 / Real.sqrt 3 * (1 / Real.sqrt 3) = 1 :=
by sorry

end div_mul_sqrt_three_reciprocal_result_equals_one_l1077_107781


namespace pen_pencil_price_ratio_l1077_107776

theorem pen_pencil_price_ratio :
  ∀ (pen_price pencil_price total_price : ℚ),
    pencil_price = 8 →
    total_price = 12 →
    total_price = pen_price + pencil_price →
    pen_price / pencil_price = 1 / 2 := by
  sorry

end pen_pencil_price_ratio_l1077_107776


namespace baseball_runs_proof_l1077_107784

theorem baseball_runs_proof (sequence : Fin 6 → ℕ) 
  (h1 : ∃ i, sequence i = 1)
  (h2 : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ sequence i = 5 ∧ sequence j = 5 ∧ sequence k = 5)
  (h3 : ∃ i j, i ≠ j ∧ sequence i = sequence j)
  (h4 : (Finset.sum Finset.univ (λ i => sequence i)) / 6 = 4) :
  ∃ i j, i ≠ j ∧ sequence i = sequence j ∧ sequence i = 4 := by
  sorry

end baseball_runs_proof_l1077_107784


namespace infinite_primes_l1077_107738

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end infinite_primes_l1077_107738


namespace number_equation_l1077_107758

theorem number_equation : ∃ n : ℝ, 2 * 2 + n = 6 ∧ n = 2 := by
  sorry

end number_equation_l1077_107758


namespace badminton_racket_cost_proof_l1077_107719

/-- The cost price of a badminton racket satisfying the given conditions -/
def badminton_racket_cost : ℝ := 125

/-- The markup percentage applied to the cost price -/
def markup_percentage : ℝ := 0.4

/-- The discount percentage applied to the marked price -/
def discount_percentage : ℝ := 0.2

/-- The profit made on the sale -/
def profit : ℝ := 15

theorem badminton_racket_cost_proof :
  (badminton_racket_cost * (1 + markup_percentage) * (1 - discount_percentage) =
   badminton_racket_cost + profit) := by
  sorry

end badminton_racket_cost_proof_l1077_107719


namespace positive_integer_solutions_of_x_plus_2y_equals_5_l1077_107724

theorem positive_integer_solutions_of_x_plus_2y_equals_5 :
  {(x, y) : ℕ × ℕ | x + 2 * y = 5 ∧ x > 0 ∧ y > 0} = {(1, 2), (3, 1)} := by
  sorry

end positive_integer_solutions_of_x_plus_2y_equals_5_l1077_107724


namespace binomial_coefficient_not_always_divisible_l1077_107737

theorem binomial_coefficient_not_always_divisible :
  ∃ k : ℕ+, ∀ n : ℕ, n > 1 → ∃ i : ℕ, 1 ≤ i ∧ i ≤ n - 1 ∧ ¬(k : ℕ) ∣ Nat.choose n i := by
  sorry

end binomial_coefficient_not_always_divisible_l1077_107737


namespace translated_function_and_triangle_area_l1077_107733

/-- A linear function f(x) = 3x + b passing through (1, 4) -/
def f (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

theorem translated_function_and_triangle_area (b : ℝ) :
  f b 1 = 4 →
  b = 1 ∧
  (1 / 2 : ℝ) * (1 / 3) * 1 = 1 / 6 := by
  sorry

end translated_function_and_triangle_area_l1077_107733


namespace cricket_bat_cost_price_l1077_107790

theorem cricket_bat_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 228) : 
  ∃ (cost_price_A : ℝ), cost_price_A = 152 ∧ 
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) := by
  sorry

end cricket_bat_cost_price_l1077_107790


namespace distance_traveled_downstream_l1077_107722

/-- Calculate the distance traveled downstream by a boat -/
theorem distance_traveled_downstream 
  (boat_speed : ℝ) 
  (current_speed : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : current_speed = 5)
  (h3 : time_minutes = 24) :
  let effective_speed := boat_speed + current_speed
  let time_hours := time_minutes / 60
  effective_speed * time_hours = 10 := by
sorry

end distance_traveled_downstream_l1077_107722


namespace work_completion_time_l1077_107706

/-- Proves that if A completes a work in 10 days, and A and B together complete the work in 
    2.3076923076923075 days, then B completes the work alone in 3 days. -/
theorem work_completion_time (a_time b_time combined_time : ℝ) 
    (ha : a_time = 10)
    (hc : combined_time = 2.3076923076923075)
    (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) : 
  b_time = 3 := by
  sorry

end work_completion_time_l1077_107706


namespace negation_of_existence_negation_of_quadratic_equation_l1077_107764

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l1077_107764


namespace acid_dilution_l1077_107716

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.40 →
  water_added = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end acid_dilution_l1077_107716


namespace room_breadth_is_five_meters_l1077_107701

/-- Given a building with 5 equal-area rooms, prove that the breadth of each room is 5 meters. -/
theorem room_breadth_is_five_meters 
  (num_rooms : ℕ) 
  (room_length : ℝ) 
  (room_height : ℝ) 
  (bricks_per_sqm : ℕ) 
  (bricks_for_floor : ℕ) :
  num_rooms = 5 →
  room_length = 4 →
  room_height = 2 →
  bricks_per_sqm = 17 →
  bricks_for_floor = 340 →
  ∃ (room_breadth : ℝ), room_breadth = 5 :=
by sorry

end room_breadth_is_five_meters_l1077_107701


namespace trapezoid_median_l1077_107771

/-- Given a triangle and a trapezoid with the same altitude, prove that the median of the trapezoid is 24 inches -/
theorem trapezoid_median (h : ℝ) : 
  let triangle_base : ℝ := 24
  let trapezoid_base1 : ℝ := 12
  let trapezoid_base2 : ℝ := 36
  let triangle_area : ℝ := (1/2) * triangle_base * h
  let trapezoid_area : ℝ := (1/2) * (trapezoid_base1 + trapezoid_base2) * h
  let trapezoid_median : ℝ := (1/2) * (trapezoid_base1 + trapezoid_base2)
  triangle_area = trapezoid_area → trapezoid_median = 24 := by
  sorry

end trapezoid_median_l1077_107771


namespace total_run_time_l1077_107766

def emma_time : ℝ := 20

theorem total_run_time (fernando_time : ℝ) 
  (h1 : fernando_time = 2 * emma_time) : 
  emma_time + fernando_time = 60 := by
  sorry

end total_run_time_l1077_107766


namespace greatest_multiple_of_5_and_7_under_1000_l1077_107727

theorem greatest_multiple_of_5_and_7_under_1000 : ∃ n : ℕ, 
  (n % 5 = 0) ∧ 
  (n % 7 = 0) ∧ 
  (n < 1000) ∧ 
  (∀ m : ℕ, (m % 5 = 0) ∧ (m % 7 = 0) ∧ (m < 1000) → m ≤ n) ∧
  (n = 980) := by
sorry

end greatest_multiple_of_5_and_7_under_1000_l1077_107727


namespace probability_point_between_C_and_E_l1077_107718

/-- Given a line segment AB with points C, D, and E, where AB = 4AD, AB = 5BC, 
    and E is the midpoint of CD, the probability that a randomly selected point 
    on AB falls between C and E is 1/4. -/
theorem probability_point_between_C_and_E 
  (A B C D E : ℝ) 
  (h1 : A < C) (h2 : C < D) (h3 : D < B)
  (h4 : B - A = 4 * (D - A))
  (h5 : B - A = 5 * (C - B))
  (h6 : E = (C + D) / 2) :
  (E - C) / (B - A) = 1 / 4 := by
  sorry

#check probability_point_between_C_and_E

end probability_point_between_C_and_E_l1077_107718


namespace min_sum_pqrs_l1077_107742

theorem min_sum_pqrs (p q r s : ℕ) : 
  p > 1 → q > 1 → r > 1 → s > 1 →
  31 * (p + 1) = 37 * (q + 1) →
  41 * (r + 1) = 43 * (s + 1) →
  p + q + r + s ≥ 14 :=
by sorry

end min_sum_pqrs_l1077_107742


namespace max_profit_plan_l1077_107715

/-- Represents the production plan for cars -/
structure CarProduction where
  a : ℕ  -- number of A type cars
  b : ℕ  -- number of B type cars

/-- Calculates the total cost of production -/
def total_cost (p : CarProduction) : ℕ := 30 * p.a + 40 * p.b

/-- Calculates the total revenue from sales -/
def total_revenue (p : CarProduction) : ℕ := 35 * p.a + 50 * p.b

/-- Calculates the profit from a production plan -/
def profit (p : CarProduction) : ℤ := total_revenue p - total_cost p

/-- Theorem stating that the maximum profit is achieved with 5 A type cars and 35 B type cars -/
theorem max_profit_plan :
  ∀ p : CarProduction,
    p.a + p.b = 40 →
    total_cost p ≤ 1550 →
    profit p ≥ 365 →
    profit p ≤ profit { a := 5, b := 35 } :=
by sorry

end max_profit_plan_l1077_107715


namespace average_score_is_two_l1077_107780

/-- Represents the score distribution of a test --/
structure ScoreDistribution where
  three_points : ℝ
  two_points : ℝ
  one_point : ℝ
  zero_points : ℝ
  sum_to_one : three_points + two_points + one_point + zero_points = 1

/-- Calculates the average score given a score distribution --/
def average_score (sd : ScoreDistribution) : ℝ :=
  3 * sd.three_points + 2 * sd.two_points + 1 * sd.one_point + 0 * sd.zero_points

/-- The main theorem stating that the average score is 2 points --/
theorem average_score_is_two (sd : ScoreDistribution)
  (h1 : sd.three_points = 0.3)
  (h2 : sd.two_points = 0.5)
  (h3 : sd.one_point = 0.1)
  (h4 : sd.zero_points = 0.1) :
  average_score sd = 2 := by
  sorry

end average_score_is_two_l1077_107780


namespace kamal_age_double_son_l1077_107786

/-- The number of years after which Kamal will be twice as old as his son -/
def years_until_double_age (kamal_age : ℕ) (son_age : ℕ) : ℕ :=
  kamal_age + 8 - 2 * (son_age + 8)

/-- Kamal's current age -/
def kamal_current_age : ℕ := 40

theorem kamal_age_double_son :
  years_until_double_age kamal_current_age
    ((kamal_current_age - 8) / 4 + 8) = 8 := by
  sorry

end kamal_age_double_son_l1077_107786


namespace parabola_intersection_l1077_107789

-- Define the two parabolas
def f (x : ℝ) : ℝ := 4 * x^2 + 6 * x - 7
def g (x : ℝ) : ℝ := 2 * x^2 + 5

-- Define the intersection points
def p1 : ℝ × ℝ := (-4, 33)
def p2 : ℝ × ℝ := (1.5, 11)

-- Theorem statement
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p1 ∨ (x, y) = p2) :=
by sorry

end parabola_intersection_l1077_107789


namespace trigonometric_identity_l1077_107735

/-- Given that sin(α) / (sin(α) - cos(α)) = -1, prove:
    1. tan(α) = 1/2
    2. (sin²(α) + 2sin(α)cos(α)) / (3sin²(α) + cos²(α)) = 5/7 -/
theorem trigonometric_identity (α : ℝ) 
    (h : Real.sin α / (Real.sin α - Real.cos α) = -1) : 
    Real.tan α = 1/2 ∧ 
    (Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / 
    (3 * Real.sin α ^ 2 + Real.cos α ^ 2) = 5/7 := by
  sorry


end trigonometric_identity_l1077_107735


namespace geometric_sequence_ratio_sum_l1077_107763

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that if the difference between the third terms is 5 times the
    difference between the second terms, then the sum of the common ratios is 5. -/
theorem geometric_sequence_ratio_sum (k p r : ℝ) (hk : k ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
  sorry

end geometric_sequence_ratio_sum_l1077_107763


namespace health_codes_suitable_for_comprehensive_survey_other_options_not_suitable_for_comprehensive_survey_l1077_107783

/-- Represents a survey option --/
inductive SurveyOption
  | MovieViewing
  | SeedGermination
  | WaterQuality
  | HealthCodes

/-- Determines if a survey option is suitable for a comprehensive survey --/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.HealthCodes => true
  | _ => false

/-- Theorem stating that the health codes survey is suitable for a comprehensive survey --/
theorem health_codes_suitable_for_comprehensive_survey :
  isSuitableForComprehensiveSurvey SurveyOption.HealthCodes = true :=
sorry

/-- Theorem stating that other survey options are not suitable for a comprehensive survey --/
theorem other_options_not_suitable_for_comprehensive_survey (option : SurveyOption) :
  option ≠ SurveyOption.HealthCodes →
  isSuitableForComprehensiveSurvey option = false :=
sorry

end health_codes_suitable_for_comprehensive_survey_other_options_not_suitable_for_comprehensive_survey_l1077_107783


namespace shortest_path_general_drinking_horse_l1077_107705

-- Define the points and the line
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 4)
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- State the theorem
theorem shortest_path_general_drinking_horse :
  ∃ (P : ℝ × ℝ), l P.1 P.2 ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) =
    2 * Real.sqrt 5 :=
sorry

end shortest_path_general_drinking_horse_l1077_107705


namespace shielas_drawing_distribution_l1077_107752

/-- Represents the number of animal drawings each neighbor receives. -/
def drawings_per_neighbor (total_drawings : ℕ) (num_neighbors : ℕ) : ℕ :=
  total_drawings / num_neighbors

/-- Proves that Shiela's neighbors each receive 9 animal drawings. -/
theorem shielas_drawing_distribution :
  let total_drawings : ℕ := 54
  let num_neighbors : ℕ := 6
  drawings_per_neighbor total_drawings num_neighbors = 9 := by
  sorry

end shielas_drawing_distribution_l1077_107752


namespace large_square_area_l1077_107745

-- Define the squares
structure Square where
  side : ℕ

-- Define the problem setup
structure SquareProblem where
  small : Square
  medium : Square
  large : Square
  small_perimeter_lt_medium_side : 4 * small.side < medium.side
  exposed_area : (large.side ^ 2 - (small.side ^ 2 + medium.side ^ 2)) = 10

-- Theorem statement
theorem large_square_area (problem : SquareProblem) : problem.large.side ^ 2 = 36 := by
  sorry

end large_square_area_l1077_107745


namespace rational_operations_closure_l1077_107721

theorem rational_operations_closure (a b : ℚ) (h : b ≠ 0) :
  (∃ (x : ℚ), x = a + b) ∧
  (∃ (y : ℚ), y = a - b) ∧
  (∃ (z : ℚ), z = a * b) ∧
  (∃ (w : ℚ), w = a / b) :=
by sorry

end rational_operations_closure_l1077_107721


namespace dodecagon_enclosed_by_dodecagons_l1077_107730

/-- The number of sides of the inner regular polygon -/
def inner_sides : ℕ := 12

/-- The number of outer regular polygons -/
def num_outer_polygons : ℕ := 12

/-- The number of sides of each outer regular polygon -/
def outer_sides : ℕ := 12

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- The exterior angle of a regular polygon with n sides -/
def exterior_angle (n : ℕ) : ℚ :=
  360 / n

/-- Theorem stating that a regular 12-sided polygon can be exactly enclosed
    by 12 regular 12-sided polygons -/
theorem dodecagon_enclosed_by_dodecagons :
  2 * (exterior_angle outer_sides / 2) = 
  180 - interior_angle inner_sides :=
sorry

end dodecagon_enclosed_by_dodecagons_l1077_107730


namespace inequality_solution_l1077_107712

theorem inequality_solution (x : ℝ) : (2 - x) / 3 + 2 > x - (x - 2) / 2 → x < 2 := by
  sorry

end inequality_solution_l1077_107712


namespace intersection_with_complement_l1077_107736

open Set

universe u

def U : Set (Fin 6) := {1,2,3,4,5,6}
def A : Set (Fin 6) := {2,4,6}
def B : Set (Fin 6) := {1,2,3,5}

theorem intersection_with_complement : A ∩ (U \ B) = {4,6} := by sorry

end intersection_with_complement_l1077_107736


namespace workshop_average_salary_l1077_107775

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 42)
  (h2 : technicians = 7)
  (h3 : avg_salary_technicians = 18000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end workshop_average_salary_l1077_107775


namespace five_digit_with_eight_count_l1077_107755

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def contains_eight (n : ℕ) : Prop := ∃ (d : ℕ), d < 5 ∧ (n / 10^d) % 10 = 8

def count_five_digit : ℕ := 90000

def count_without_eight : ℕ := 52488

theorem five_digit_with_eight_count :
  (count_five_digit - count_without_eight) = 37512 :=
sorry

end five_digit_with_eight_count_l1077_107755


namespace range_of_a_l1077_107793

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2*x + 2| - |2*x - 2| ≤ a) → a ≥ 4 := by
  sorry

end range_of_a_l1077_107793
