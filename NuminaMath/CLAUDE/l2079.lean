import Mathlib

namespace triangle_theorem_l2079_207904

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) 
  (h2 : t.c = 3 * t.a) : 
  t.B = π/3 ∧ Real.sin t.A = Real.sqrt 21 / 14 := by
  sorry


end triangle_theorem_l2079_207904


namespace sufficient_not_necessary_condition_l2079_207942

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 1) ∧ (∃ x, x > 1 ∧ x ≤ a) ↔ a > 1 :=
by sorry

end sufficient_not_necessary_condition_l2079_207942


namespace square_area_with_five_equal_rectangles_l2079_207939

theorem square_area_with_five_equal_rectangles (w : ℝ) (h : w = 5) :
  ∃ (s : ℝ), s > 0 ∧ s * s = 400 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    2 * x * y = 3 * w * y ∧
    2 * x + w = s ∧
    5 * (2 * x * y) = s * s :=
by
  sorry

#check square_area_with_five_equal_rectangles

end square_area_with_five_equal_rectangles_l2079_207939


namespace min_value_a_plus_8b_l2079_207943

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x * y = 2 * x + y → a + 8 * b ≤ x + 8 * y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * y = 2 * x + y ∧ x + 8 * y = 25 / 2) :=
sorry

end min_value_a_plus_8b_l2079_207943


namespace midpoint_movement_l2079_207933

/-- Given two points A and B with midpoint M, prove the new midpoint M' and distance between M and M' after moving A and B -/
theorem midpoint_movement (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  let m' := (a + 4 + c - 15) / 2
  let n' := (b + 12 + d - 5) / 2
  (m' = m - 11 / 2 ∧ n' = n + 7 / 2) ∧
  Real.sqrt ((m' - m) ^ 2 + (n' - n) ^ 2) = Real.sqrt 42.5 := by
  sorry


end midpoint_movement_l2079_207933


namespace quadratic_function_property_l2079_207908

theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ b) : 
  let f := fun x => x^2 + a*x + b
  (f a = f b) → f 2 = 4 := by
sorry

end quadratic_function_property_l2079_207908


namespace problem_solution_l2079_207998

theorem problem_solution (a b m n x : ℝ) 
  (h1 : a * b = 1)
  (h2 : m + n = 0)
  (h3 : |x| = 1) :
  2022 * (m + n) + 2018 * x^2 - 2019 * a * b = -1 := by
  sorry

end problem_solution_l2079_207998


namespace product_equals_one_l2079_207969

theorem product_equals_one (a b : ℝ) : a * (b + 1) + b * (a + 1) = (a + 1) * (b + 1) → a * b = 1 := by
  sorry

end product_equals_one_l2079_207969


namespace base_difference_equals_174_l2079_207928

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def base9_to_decimal (n : Nat) : Nat :=
  base_to_decimal [3, 2, 4] 9

def base6_to_decimal (n : Nat) : Nat :=
  base_to_decimal [2, 3, 1] 6

theorem base_difference_equals_174 :
  base9_to_decimal 324 - base6_to_decimal 231 = 174 := by
  sorry

end base_difference_equals_174_l2079_207928


namespace sum_of_combinations_l2079_207902

theorem sum_of_combinations : Finset.sum (Finset.range 5) (fun k => Nat.choose 6 (k + 1)) = 62 := by
  sorry

end sum_of_combinations_l2079_207902


namespace cookie_cutter_problem_l2079_207991

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := sorry

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := 46

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

theorem cookie_cutter_problem :
  num_squares = 4 :=
by sorry

end cookie_cutter_problem_l2079_207991


namespace alice_above_quota_l2079_207999

def alice_sales (quota nike_price adidas_price reebok_price : ℕ) 
                (nike_sold adidas_sold reebok_sold : ℕ) : ℕ := 
  nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold

theorem alice_above_quota : 
  let quota : ℕ := 1000
  let nike_price : ℕ := 60
  let adidas_price : ℕ := 45
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let adidas_sold : ℕ := 6
  let reebok_sold : ℕ := 9
  alice_sales quota nike_price adidas_price reebok_price nike_sold adidas_sold reebok_sold - quota = 65 := by
  sorry

end alice_above_quota_l2079_207999


namespace largest_number_l2079_207949

theorem largest_number : 
  0.9989 > 0.998 ∧ 
  0.9989 > 0.9899 ∧ 
  0.9989 > 0.9 ∧ 
  0.9989 > 0.8999 :=
by sorry

end largest_number_l2079_207949


namespace smallest_integer_l2079_207996

theorem smallest_integer (a b : ℕ) (ha : a = 80) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 40) :
  ∃ (m : ℕ), m ≥ b ∧ m = 50 ∧ Nat.lcm a m / Nat.gcd a m = 40 :=
sorry

end smallest_integer_l2079_207996


namespace stadium_attendance_l2079_207960

theorem stadium_attendance (total_start : ℕ) (girls_start : ℕ) 
  (h1 : total_start = 600)
  (h2 : girls_start = 240)
  (h3 : girls_start ≤ total_start) :
  let boys_start := total_start - girls_start
  let boys_left := boys_start / 4
  let girls_left := girls_start / 8
  let remaining := total_start - boys_left - girls_left
  remaining = 480 := by sorry

end stadium_attendance_l2079_207960


namespace min_value_inequality_l2079_207964

def f (x : ℝ) : ℝ := |3*x - 1| + |x + 1|

def g (x : ℝ) : ℝ := f x + 2*|x + 1|

theorem min_value_inequality (a b : ℝ) 
  (h1 : ∀ x, g x ≥ a^2 + b^2) 
  (h2 : ∃ x, g x = a^2 + b^2) : 
  1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 3/2 := by
  sorry

end min_value_inequality_l2079_207964


namespace train_speed_l2079_207903

/-- Calculates the speed of a train given its length, time to pass a person moving in the opposite direction, and the person's speed. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) : 
  train_length = 275 →
  passing_time = 15 →
  person_speed = 6 →
  (train_length / 1000) / (passing_time / 3600) - person_speed = 60 :=
by
  sorry

#check train_speed

end train_speed_l2079_207903


namespace distance_to_reflection_l2079_207977

/-- Given a point F with coordinates (-5, 3), prove that the distance between F
    and its reflection over the y-axis is 10. -/
theorem distance_to_reflection (F : ℝ × ℝ) : 
  F = (-5, 3) → ‖F - (5, 3)‖ = 10 := by sorry

end distance_to_reflection_l2079_207977


namespace student_score_proof_l2079_207910

theorem student_score_proof (total_questions : Nat) (score : Int) 
  (h1 : total_questions = 100)
  (h2 : score = 79) : 
  ∃ (correct incorrect : Nat),
    correct + incorrect = total_questions ∧
    score = correct - 2 * incorrect ∧
    correct = 93 := by
  sorry

end student_score_proof_l2079_207910


namespace show_dog_profit_l2079_207959

/-- Calculate the total profit from breeding and selling show dogs -/
theorem show_dog_profit
  (num_dogs : ℕ)
  (cost_per_dog : ℚ)
  (num_puppies : ℕ)
  (price_per_puppy : ℚ)
  (h1 : num_dogs = 2)
  (h2 : cost_per_dog = 250)
  (h3 : num_puppies = 6)
  (h4 : price_per_puppy = 350) :
  (num_puppies : ℚ) * price_per_puppy - (num_dogs : ℚ) * cost_per_dog = 1600 :=
by sorry

end show_dog_profit_l2079_207959


namespace inequality_proof_l2079_207932

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) := by
  sorry

end inequality_proof_l2079_207932


namespace factors_of_34020_l2079_207993

/-- The number of positive factors of 34020 -/
def num_factors : ℕ := 72

/-- The prime factorization of 34020 -/
def prime_factorization : List (ℕ × ℕ) := [(3, 5), (5, 1), (2, 2), (7, 1)]

theorem factors_of_34020 : (Nat.divisors 34020).card = num_factors := by sorry

end factors_of_34020_l2079_207993


namespace alla_boris_meeting_l2079_207970

/-- The number of lampposts along the alley -/
def total_lampposts : ℕ := 400

/-- The lamppost number where Alla is observed -/
def alla_observed : ℕ := 55

/-- The lamppost number where Boris is observed -/
def boris_observed : ℕ := 321

/-- The function to calculate the meeting point of Alla and Boris -/
def meeting_point : ℕ :=
  let intervals_covered := (alla_observed - 1) + (total_lampposts - boris_observed)
  let total_intervals := total_lampposts - 1
  (intervals_covered * 3) + 1

/-- Theorem stating that Alla and Boris meet at lamppost 163 -/
theorem alla_boris_meeting :
  meeting_point = 163 := by sorry

end alla_boris_meeting_l2079_207970


namespace larger_cross_section_distance_l2079_207912

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  -- We don't need to define the full structure, just what's necessary for the problem

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

theorem larger_cross_section_distance
  (pyramid : RightOctagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_area1 : cs1.area = 256 * Real.sqrt 2)
  (h_area2 : cs2.area = 576 * Real.sqrt 2)
  (h_distance : cs2.distance_from_apex - cs1.distance_from_apex = 10)
  (h_parallel : True)  -- Assuming parallel, but not used in the proof
  (h_larger : cs2.area > cs1.area) :
  cs2.distance_from_apex = 30 := by
sorry


end larger_cross_section_distance_l2079_207912


namespace larger_cuboid_length_l2079_207911

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the smaller cuboid -/
def smallerCuboid : CuboidDimensions :=
  { length := 5, width := 4, height := 3 }

/-- The number of smaller cuboids that can be formed from the larger cuboid -/
def numberOfSmallerCuboids : ℕ := 32

/-- The width of the larger cuboid -/
def largerCuboidWidth : ℝ := 10

/-- The height of the larger cuboid -/
def largerCuboidHeight : ℝ := 12

theorem larger_cuboid_length :
  ∃ (largerLength : ℝ),
    cuboidVolume { length := largerLength, width := largerCuboidWidth, height := largerCuboidHeight } =
    (numberOfSmallerCuboids : ℝ) * cuboidVolume smallerCuboid ∧
    largerLength = 16 := by
  sorry

end larger_cuboid_length_l2079_207911


namespace greatest_integer_less_than_negative_nineteen_thirds_l2079_207957

theorem greatest_integer_less_than_negative_nineteen_thirds :
  ⌊-19/3⌋ = -7 := by sorry

end greatest_integer_less_than_negative_nineteen_thirds_l2079_207957


namespace fourth_root_of_390820584961_l2079_207922

theorem fourth_root_of_390820584961 :
  let n : ℕ := 390820584961
  let expansion : ℕ := 1 * 75^4 + 4 * 75^3 + 6 * 75^2 + 4 * 75 + 1
  n = expansion →
  (n : ℝ) ^ (1/4 : ℝ) = 76 := by
  sorry

end fourth_root_of_390820584961_l2079_207922


namespace circle_a_l2079_207923

theorem circle_a (x y : ℝ) : 
  (x - 3)^2 + (y + 2)^2 = 16 → (∃ (center : ℝ × ℝ) (radius : ℝ), center = (3, -2) ∧ radius = 4) :=
by sorry

end circle_a_l2079_207923


namespace equation_represents_two_intersecting_lines_l2079_207938

theorem equation_represents_two_intersecting_lines :
  ∃ (m₁ m₂ b₁ b₂ : ℝ), m₁ ≠ m₂ ∧
  (∀ x y : ℝ, x^3 * (2*x + 2*y + 3) = y^3 * (2*x + 2*y + 3) ↔ 
    (y = m₁ * x + b₁) ∨ (y = m₂ * x + b₂)) :=
by sorry

end equation_represents_two_intersecting_lines_l2079_207938


namespace share_proportion_l2079_207992

theorem share_proportion (c d : ℕ) (h1 : c = d + 500) (h2 : d = 1500) :
  c / d = 4 / 3 := by
  sorry

end share_proportion_l2079_207992


namespace trig_identity_l2079_207906

theorem trig_identity (α : Real) (h : Real.sin α + Real.cos α = Real.sqrt 2) :
  Real.tan α + Real.cos α / Real.sin α = 2 := by
  sorry

end trig_identity_l2079_207906


namespace matching_segment_exists_l2079_207972

/-- A 20-digit binary number -/
def BinaryNumber := Fin 20 → Bool

/-- A is a 20-digit binary number with 10 zeros and 10 ones -/
def is_valid_A (A : BinaryNumber) : Prop :=
  (Finset.filter (λ i => A i = false) Finset.univ).card = 10 ∧
  (Finset.filter (λ i => A i = true) Finset.univ).card = 10

/-- B is any 20-digit binary number -/
def B : BinaryNumber := sorry

/-- C is a 40-digit binary number formed by concatenating B with itself -/
def C : Fin 40 → Bool :=
  λ i => B (Fin.val i % 20)

/-- Count matching bits between two binary numbers -/
def count_matches (X Y : BinaryNumber) : Nat :=
  (Finset.filter (λ i => X i = Y i) Finset.univ).card

/-- Theorem: There exists a 20-bit segment of C with at least 10 matching bits with A -/
theorem matching_segment_exists (A : BinaryNumber) (h : is_valid_A A) :
  ∃ k : Fin 21, count_matches A (λ i => C (i + k)) ≥ 10 := by sorry

end matching_segment_exists_l2079_207972


namespace extraneous_root_implies_a_value_l2079_207909

/-- The equation has an extraneous root if x = 3 is a solution to the polynomial form of the equation -/
def has_extraneous_root (a : ℚ) : Prop :=
  ∃ x : ℚ, x = 3 ∧ x - 2*a = 2*(x - 3)

/-- The original equation -/
def original_equation (x a : ℚ) : Prop :=
  x / (x - 3) - 2*a / (x - 3) = 2

theorem extraneous_root_implies_a_value :
  ∀ a : ℚ, has_extraneous_root a → a = 3/2 :=
by sorry

end extraneous_root_implies_a_value_l2079_207909


namespace elizabeth_haircut_l2079_207920

theorem elizabeth_haircut (first_cut second_cut : ℝ) 
  (h1 : first_cut = 0.375)
  (h2 : second_cut = 0.5) :
  first_cut + second_cut = 0.875 := by
  sorry

end elizabeth_haircut_l2079_207920


namespace binary_110_equals_6_l2079_207951

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 110₂ -/
def binary_110 : List Bool := [false, true, true]

theorem binary_110_equals_6 :
  binary_to_decimal binary_110 = 6 := by
  sorry

end binary_110_equals_6_l2079_207951


namespace no_integer_solution_l2079_207956

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 3*x*y - 2*y^2 ≠ 122 := by
  sorry

end no_integer_solution_l2079_207956


namespace dice_game_probability_l2079_207924

/-- Represents a pair of dice rolls -/
structure DiceRoll :=
  (first : Nat) (second : Nat)

/-- The set of all possible dice rolls -/
def allRolls : Finset DiceRoll := sorry

/-- The set of dice rolls that sum to 8 -/
def rollsSum8 : Finset DiceRoll := sorry

/-- Probability of rolling a specific combination -/
def probSpecificRoll : ℚ := 1 / 36

theorem dice_game_probability : 
  (Finset.card rollsSum8 : ℚ) * probSpecificRoll = 5 / 36 := by sorry

end dice_game_probability_l2079_207924


namespace mike_work_time_l2079_207987

-- Define the basic task times for sedans (in minutes)
def wash_time : ℝ := 10
def oil_change_time : ℝ := 15
def tire_change_time : ℝ := 30
def paint_time : ℝ := 45
def engine_service_time : ℝ := 60

-- Define the number of tasks for sedans
def sedan_washes : ℕ := 9
def sedan_oil_changes : ℕ := 6
def sedan_tire_changes : ℕ := 2
def sedan_paints : ℕ := 4
def sedan_engine_services : ℕ := 2

-- Define the number of tasks for SUVs
def suv_washes : ℕ := 7
def suv_oil_changes : ℕ := 4
def suv_tire_changes : ℕ := 3
def suv_paints : ℕ := 3
def suv_engine_services : ℕ := 1

-- Define the time multiplier for SUV washing and painting
def suv_time_multiplier : ℝ := 1.5

-- Theorem statement
theorem mike_work_time : 
  let sedan_time := 
    sedan_washes * wash_time + 
    sedan_oil_changes * oil_change_time + 
    sedan_tire_changes * tire_change_time + 
    sedan_paints * paint_time + 
    sedan_engine_services * engine_service_time
  let suv_time := 
    suv_washes * (wash_time * suv_time_multiplier) + 
    suv_oil_changes * oil_change_time + 
    suv_tire_changes * tire_change_time + 
    suv_paints * (paint_time * suv_time_multiplier) + 
    suv_engine_services * engine_service_time
  let total_time := sedan_time + suv_time
  (total_time / 60) = 17.625 := by sorry

end mike_work_time_l2079_207987


namespace division_equality_l2079_207915

theorem division_equality (h : (204 : ℝ) / 12.75 = 16) : (2.04 : ℝ) / 1.275 = 1.6 := by
  sorry

end division_equality_l2079_207915


namespace ball_bounce_distance_l2079_207919

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let firstDescent := initialHeight
  let firstAscent := initialHeight * reboundFactor
  let secondDescent := firstAscent
  let secondAscent := firstAscent * reboundFactor
  let thirdDescent := secondAscent
  firstDescent + firstAscent + secondDescent + secondAscent + thirdDescent

/-- The theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistance 90 0.5 2 = 225 := by
  sorry

#eval totalDistance 90 0.5 2

end ball_bounce_distance_l2079_207919


namespace bird_population_theorem_l2079_207965

/-- 
Given a population of birds consisting of robins and bluejays, 
if 1/3 of robins are female, 2/3 of bluejays are female, 
and the overall fraction of male birds is 7/15, 
then the fraction of birds that are robins is 2/5.
-/
theorem bird_population_theorem (total_birds : ℕ) (robins : ℕ) (bluejays : ℕ) 
  (h1 : robins + bluejays = total_birds)
  (h2 : (2 : ℚ) / 3 * robins + (1 : ℚ) / 3 * bluejays = (7 : ℚ) / 15 * total_birds) :
  (robins : ℚ) / total_birds = 2 / 5 := by
  sorry

end bird_population_theorem_l2079_207965


namespace money_distribution_l2079_207947

/-- Given a sum of money distributed among four people in a specific proportion,
    where one person receives a fixed amount more than another,
    prove that a particular person's share is as stated. -/
theorem money_distribution (total : ℝ) (a b c d : ℝ) : 
  a + b + c + d = total →
  5 * b = 3 * a →
  5 * c = 2 * a →
  5 * d = 3 * a →
  a = b + 1000 →
  c = 1000 := by
sorry

end money_distribution_l2079_207947


namespace hotel_assignment_count_l2079_207997

/-- Represents a hotel with a specific number of rooms and guests -/
structure Hotel :=
  (num_rooms : ℕ)
  (num_guests : ℕ)

/-- Represents the constraints for room assignments -/
structure RoomConstraints :=
  (max_guests_regular : ℕ)
  (min_guests_deluxe : ℕ)
  (max_guests_deluxe : ℕ)

/-- Calculates the number of valid room assignments -/
def count_valid_assignments (h : Hotel) (c : RoomConstraints) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem hotel_assignment_count :
  let h : Hotel := ⟨7, 7⟩
  let c : RoomConstraints := ⟨3, 2, 3⟩
  count_valid_assignments h c = 27720 :=
sorry

end hotel_assignment_count_l2079_207997


namespace circle_intersection_perpendicularity_l2079_207974

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (center : Circle → Point)
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (tangent : Circle → Circle → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_perpendicularity
  (O O₁ O₂ : Circle) (M N S T : Point) :
  (intersect O₁ O₂ M N) →
  (tangent O O₁ S) →
  (tangent O O₂ T) →
  (perpendicular (center O) M M N ↔ collinear S N T) :=
sorry

end circle_intersection_perpendicularity_l2079_207974


namespace lindsay_daily_income_l2079_207930

/-- Represents Doctor Lindsay's work schedule and patient fees --/
structure DoctorSchedule where
  adult_patients_per_hour : ℕ
  child_patients_per_hour : ℕ
  adult_fee : ℕ
  child_fee : ℕ
  hours_per_day : ℕ

/-- Calculates Doctor Lindsay's daily income based on her schedule --/
def daily_income (schedule : DoctorSchedule) : ℕ :=
  (schedule.adult_patients_per_hour * schedule.adult_fee +
   schedule.child_patients_per_hour * schedule.child_fee) *
  schedule.hours_per_day

/-- Theorem stating Doctor Lindsay's daily income --/
theorem lindsay_daily_income :
  ∃ (schedule : DoctorSchedule),
    schedule.adult_patients_per_hour = 4 ∧
    schedule.child_patients_per_hour = 3 ∧
    schedule.adult_fee = 50 ∧
    schedule.child_fee = 25 ∧
    schedule.hours_per_day = 8 ∧
    daily_income schedule = 2200 := by
  sorry

end lindsay_daily_income_l2079_207930


namespace largest_common_divisor_528_440_l2079_207984

theorem largest_common_divisor_528_440 : Nat.gcd 528 440 = 88 := by
  sorry

end largest_common_divisor_528_440_l2079_207984


namespace mady_balls_equals_ternary_sum_l2079_207905

/-- Represents the state of a box in Mady's game -/
inductive BoxState
| Empty : BoxState
| OneBall : BoxState
| TwoBalls : BoxState

/-- Converts a natural number to its ternary (base 3) representation -/
def toTernary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then []
      else (m % 3) :: aux (m / 3)
    aux n |>.reverse

/-- Simulates Mady's ball-placing process for a given number of steps -/
def madyProcess (steps : ℕ) : List BoxState :=
  sorry

/-- Counts the number of balls in the final state -/
def countBalls (state : List BoxState) : ℕ :=
  sorry

/-- The main theorem: The number of balls after 2023 steps equals the sum of ternary digits of 2023 -/
theorem mady_balls_equals_ternary_sum :
  countBalls (madyProcess 2023) = (toTernary 2023).sum := by
  sorry

end mady_balls_equals_ternary_sum_l2079_207905


namespace ratio_of_remaining_ingredients_l2079_207967

def total_sugar : ℕ := 13
def total_flour : ℕ := 25
def total_cocoa : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa : ℕ := 15

def remaining_flour : ℕ := total_flour - added_flour
def remaining_sugar : ℕ := total_sugar - added_sugar
def remaining_cocoa : ℕ := total_cocoa - added_cocoa

theorem ratio_of_remaining_ingredients :
  (remaining_flour : ℚ) / (remaining_sugar + remaining_cocoa) = 17 / 46 := by
sorry

end ratio_of_remaining_ingredients_l2079_207967


namespace partnership_profit_l2079_207929

theorem partnership_profit (john_investment mike_investment : ℚ)
  (equal_share_ratio investment_ratio : ℚ)
  (john_extra_profit : ℚ) :
  john_investment = 700 →
  mike_investment = 300 →
  equal_share_ratio = 1/3 →
  investment_ratio = 2/3 →
  john_extra_profit = 800 →
  ∃ (total_profit : ℚ),
    total_profit * equal_share_ratio / 2 +
    total_profit * investment_ratio * (john_investment / (john_investment + mike_investment)) -
    (total_profit * equal_share_ratio / 2 +
     total_profit * investment_ratio * (mike_investment / (john_investment + mike_investment)))
    = john_extra_profit ∧
    total_profit = 3000 :=
by sorry

end partnership_profit_l2079_207929


namespace mississippi_arrangements_l2079_207954

/-- The number of unique arrangements of the letters in MISSISSIPPI -/
def mississippiArrangements : ℕ := 34650

/-- The total number of letters in MISSISSIPPI -/
def totalLetters : ℕ := 11

/-- The number of occurrences of 'I' in MISSISSIPPI -/
def countI : ℕ := 4

/-- The number of occurrences of 'S' in MISSISSIPPI -/
def countS : ℕ := 4

/-- The number of occurrences of 'P' in MISSISSIPPI -/
def countP : ℕ := 2

/-- The number of occurrences of 'M' in MISSISSIPPI -/
def countM : ℕ := 1

theorem mississippi_arrangements :
  mississippiArrangements = Nat.factorial totalLetters / (Nat.factorial countI * Nat.factorial countS * Nat.factorial countP) :=
by sorry

end mississippi_arrangements_l2079_207954


namespace minimum_value_of_S_l2079_207913

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := (3 * n^2 - 95 * n) / 2

/-- The minimum value of S(n) for positive integers n -/
def min_S : ℚ := -392

theorem minimum_value_of_S :
  ∀ n : ℕ, n > 0 → S n ≥ min_S :=
sorry

end minimum_value_of_S_l2079_207913


namespace cheryl_material_problem_l2079_207958

theorem cheryl_material_problem (x : ℝ) : 
  x > 0 ∧ 
  x + 1/3 > 0 ∧ 
  8/24 < x + 1/3 ∧ 
  x = 0.5555555555555556 → 
  x = 0.5555555555555556 := by
sorry

end cheryl_material_problem_l2079_207958


namespace product_without_zero_ending_l2079_207975

theorem product_without_zero_ending : ∃ (a b : ℤ), 
  a * b = 100000 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 := by
  sorry

end product_without_zero_ending_l2079_207975


namespace sqrt_fourth_power_eq_256_l2079_207989

theorem sqrt_fourth_power_eq_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end sqrt_fourth_power_eq_256_l2079_207989


namespace lenny_money_left_l2079_207917

/-- Calculates the amount of money Lenny has left after his expenses -/
def money_left (initial : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial - (expense1 + expense2)

/-- Proves that Lenny has $39 left after his expenses -/
theorem lenny_money_left :
  money_left 84 24 21 = 39 := by
  sorry

end lenny_money_left_l2079_207917


namespace right_triangle_one_one_sqrt_two_l2079_207981

theorem right_triangle_one_one_sqrt_two :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 2
  a^2 + b^2 = c^2 := by sorry

end right_triangle_one_one_sqrt_two_l2079_207981


namespace fractional_equation_solution_l2079_207926

theorem fractional_equation_solution :
  ∃ x : ℝ, (3 / (x + 1) = 2 / (x - 1)) ∧ x = 5 :=
by
  -- Proof goes here
  sorry

end fractional_equation_solution_l2079_207926


namespace point_four_units_from_negative_two_l2079_207918

theorem point_four_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 4) ↔ (x = 2 ∨ x = -6) := by sorry

end point_four_units_from_negative_two_l2079_207918


namespace egyptian_fraction_iff_prime_divisor_l2079_207990

theorem egyptian_fraction_iff_prime_divisor (n : ℕ) :
  (Odd n ∧ n > 0) →
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ (p : ℕ), Prime p ∧ p ∣ n ∧ p % 4 = 1 := by
sorry

end egyptian_fraction_iff_prime_divisor_l2079_207990


namespace martha_cakes_l2079_207983

theorem martha_cakes (num_children : ℕ) (cakes_per_child : ℕ) 
  (h1 : num_children = 3)
  (h2 : cakes_per_child = 6) :
  num_children * cakes_per_child = 18 := by
  sorry

end martha_cakes_l2079_207983


namespace product_uvw_l2079_207944

theorem product_uvw (a c x y : ℝ) (u v w : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(c^5 - 1)) ∧ 
  ((a^u*x - a^v)*(a^w*y - a^3) = a^5*c^5) →
  u*v*w = 6 :=
by sorry

end product_uvw_l2079_207944


namespace greatest_x_value_l2079_207948

theorem greatest_x_value : 
  (∃ (x : ℝ), ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 20) ∧ 
  (∀ (x : ℝ), ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 20 → x ≤ 36/19) ∧
  (((4*(36/19) - 16)/(3*(36/19) - 4))^2 + (4*(36/19) - 16)/(3*(36/19) - 4) = 20) :=
by sorry

end greatest_x_value_l2079_207948


namespace height_study_concepts_l2079_207978

/-- Represents a student in the study -/
structure Student where
  height : ℝ

/-- Represents the statistical study of student heights -/
structure HeightStudy where
  allStudents : Finset Student
  sampledStudents : Finset Student
  h_sampled_subset : sampledStudents ⊆ allStudents

/-- Main theorem about the statistical concepts in the height study -/
theorem height_study_concepts (study : HeightStudy) 
  (h_total : study.allStudents.card = 480)
  (h_sampled : study.sampledStudents.card = 80) :
  (∃ (population : Finset Student), population = study.allStudents) ∧
  (∃ (sample_size : ℕ), sample_size = study.sampledStudents.card) ∧
  (∃ (sample : Finset Student), sample = study.sampledStudents) ∧
  (∃ (individual : Student), individual ∈ study.allStudents) :=
sorry

end height_study_concepts_l2079_207978


namespace scatter_plot_suitable_for_linear_relationship_only_scatter_plot_suitable_for_linear_relationship_l2079_207953

/-- A type representing different types of plots --/
inductive PlotType
  | ScatterPlot
  | StemAndLeafPlot
  | FrequencyDistributionHistogram
  | FrequencyDistributionLineChart

/-- A function that determines if a plot type is suitable for identifying linear relationships --/
def isSuitableForLinearRelationship (plot : PlotType) : Prop :=
  match plot with
  | PlotType.ScatterPlot => true
  | _ => false

/-- Theorem stating that a scatter plot is suitable for identifying linear relationships --/
theorem scatter_plot_suitable_for_linear_relationship :
  isSuitableForLinearRelationship PlotType.ScatterPlot :=
sorry

/-- Theorem stating that a scatter plot is the only suitable plot type for identifying linear relationships --/
theorem only_scatter_plot_suitable_for_linear_relationship (plot : PlotType) :
  isSuitableForLinearRelationship plot → plot = PlotType.ScatterPlot :=
sorry

end scatter_plot_suitable_for_linear_relationship_only_scatter_plot_suitable_for_linear_relationship_l2079_207953


namespace seashells_count_l2079_207976

theorem seashells_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
sorry

end seashells_count_l2079_207976


namespace linear_function_inequality_l2079_207950

/-- A linear function passing through first, second, and fourth quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = a * x + b
  second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = a * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = a * x + b
  x_intercept : a * 2 + b = 0

/-- The solution set of a(x-1)-b > 0 for a LinearFunction is x < -1 -/
theorem linear_function_inequality (f : LinearFunction) :
  {x : ℝ | f.a * (x - 1) - f.b > 0} = {x : ℝ | x < -1} := by
  sorry

end linear_function_inequality_l2079_207950


namespace only_expr3_correct_l2079_207966

-- Define the expressions to be evaluated
def expr1 : Int := (-2)^3
def expr2 : Int := (-3)^2
def expr3 : Int := -3^2
def expr4 : Int := (-2)^2

-- Theorem stating that only the third expression is correct
theorem only_expr3_correct :
  expr1 ≠ 8 ∧ 
  expr2 ≠ -9 ∧ 
  expr3 = -9 ∧ 
  expr4 = 4 :=
by sorry

end only_expr3_correct_l2079_207966


namespace max_distance_a_c_theorem_l2079_207916

def max_distance_a_c (a b c : ℝ × ℝ) : Prop :=
  let norm := λ v : ℝ × ℝ => Real.sqrt (v.1^2 + v.2^2)
  let dot := λ u v : ℝ × ℝ => u.1 * v.1 + u.2 * v.2
  norm a = 2 ∧ 
  norm b = 2 ∧ 
  dot a b = 2 ∧ 
  dot c (a + 2 • b - 2 • c) = 2 →
  (∀ c', dot c' (a + 2 • b - 2 • c') = 2 → 
    norm (a - c) ≤ (Real.sqrt 3 + Real.sqrt 7) / 2) ∧
  (∃ c', dot c' (a + 2 • b - 2 • c') = 2 ∧ 
    norm (a - c') = (Real.sqrt 3 + Real.sqrt 7) / 2)

theorem max_distance_a_c_theorem (a b c : ℝ × ℝ) : 
  max_distance_a_c a b c := by sorry

end max_distance_a_c_theorem_l2079_207916


namespace elevator_weight_problem_l2079_207921

theorem elevator_weight_problem (initial_count : ℕ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_count = 6 →
  new_person_weight = 97 →
  new_average = 151 →
  ∃ (initial_average : ℝ),
    initial_average * initial_count + new_person_weight = new_average * (initial_count + 1) ∧
    initial_average = 160 := by
  sorry

end elevator_weight_problem_l2079_207921


namespace square_side_length_l2079_207955

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) : 
  perimeter / 4 = 25 := by
  sorry

end square_side_length_l2079_207955


namespace student_line_length_l2079_207962

/-- The length of a line of students, given the number of students and the distance between them. -/
def line_length (num_students : ℕ) (distance : ℝ) : ℝ :=
  (num_students - 1 : ℝ) * distance

/-- Theorem stating that the length of a line formed by 51 students with 3 meters between each adjacent pair is 150 meters. -/
theorem student_line_length : line_length 51 3 = 150 := by
  sorry

#eval line_length 51 3

end student_line_length_l2079_207962


namespace complex_arithmetic_equality_l2079_207979

theorem complex_arithmetic_equality : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end complex_arithmetic_equality_l2079_207979


namespace x_squared_mod_26_l2079_207963

theorem x_squared_mod_26 (x : ℤ) (h1 : 5 * x ≡ 9 [ZMOD 26]) (h2 : 4 * x ≡ 15 [ZMOD 26]) :
  x^2 ≡ 10 [ZMOD 26] := by
  sorry

end x_squared_mod_26_l2079_207963


namespace intersection_when_m_is_one_range_of_m_when_B_subset_A_l2079_207952

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- Theorem 1: When m = 1, A ∩ B = { x | 1 < x < 2 }
theorem intersection_when_m_is_one :
  A ∩ B 1 = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: If B ⊆ A, then m ∈ [-1, +∞)
theorem range_of_m_when_B_subset_A :
  (∀ m : ℝ, B m ⊆ A) → {m : ℝ | -1 ≤ m} = Set.Ici (-1) := by sorry

end intersection_when_m_is_one_range_of_m_when_B_subset_A_l2079_207952


namespace intersection_point_coordinates_l2079_207907

/-- Given a triangle ABC with vertices A(x₁, y₁), B(x₂, y₂), C(x₃, y₃),
    and points E on AC and F on AB such that AE:EC = n:l and AF:FB = m:l,
    prove that the intersection point P of BE and CF has coordinates
    ((lx₁ + mx₂ + nx₃)/(l + m + n), (ly₁ + my₂ + ny₃)/(l + m + n)) -/
theorem intersection_point_coordinates
  (x₁ y₁ x₂ y₂ x₃ y₃ l m n : ℝ)
  (h₁ : m ≠ -l)
  (h₂ : n ≠ -l)
  (h₃ : l + m + n ≠ 0) :
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let C := (x₃, y₃)
  let E := ((l * x₁ + n * x₃) / (l + n), (l * y₁ + n * y₃) / (l + n))
  let F := ((l * x₁ + m * x₂) / (l + m), (l * y₁ + m * y₂) / (l + m))
  let P := ((l * x₁ + m * x₂ + n * x₃) / (l + m + n), (l * y₁ + m * y₂ + n * y₃) / (l + m + n))
  ∃ (t : ℝ), (P.1 - E.1) / (B.1 - E.1) = t ∧ (P.2 - E.2) / (B.2 - E.2) = t ∧
             (P.1 - F.1) / (C.1 - F.1) = (1 - t) ∧ (P.2 - F.2) / (C.2 - F.2) = (1 - t) :=
by sorry

end intersection_point_coordinates_l2079_207907


namespace world_cup_2006_matches_l2079_207988

/-- Calculates the number of matches in a group stage -/
def groupStageMatches (numGroups : ℕ) (teamsPerGroup : ℕ) : ℕ :=
  numGroups * (teamsPerGroup.choose 2)

/-- Calculates the number of matches in a knockout stage -/
def knockoutStageMatches (numTeams : ℕ) : ℕ :=
  numTeams - 1

/-- Represents the structure of the World Cup tournament -/
structure WorldCupTournament where
  totalTeams : ℕ
  numGroups : ℕ
  teamsPerGroup : ℕ
  advancingTeams : ℕ

/-- Calculates the total number of matches in the World Cup tournament -/
def totalMatches (t : WorldCupTournament) : ℕ :=
  groupStageMatches t.numGroups t.teamsPerGroup + knockoutStageMatches t.advancingTeams

/-- Theorem stating that the total number of matches in the 2006 World Cup is 64 -/
theorem world_cup_2006_matches :
  let t : WorldCupTournament := {
    totalTeams := 32,
    numGroups := 8,
    teamsPerGroup := 4,
    advancingTeams := 16
  }
  totalMatches t = 64 := by sorry

end world_cup_2006_matches_l2079_207988


namespace pool_concrete_weight_l2079_207995

/-- Represents the dimensions and properties of a swimming pool --/
structure Pool where
  tileLength : ℝ
  wallHeight : ℝ
  wallThickness : ℝ
  perimeterUnits : ℕ
  outerCorners : ℕ
  innerCorners : ℕ
  concreteWeight : ℝ

/-- Calculates the weight of concrete used for the walls of a pool --/
def concreteWeightForWalls (p : Pool) : ℝ :=
  let adjustedPerimeter := p.perimeterUnits * p.tileLength + p.outerCorners * p.wallThickness - p.innerCorners * p.wallThickness
  let wallVolume := adjustedPerimeter * p.wallHeight * p.wallThickness
  wallVolume * p.concreteWeight

/-- The theorem to be proved --/
theorem pool_concrete_weight :
  let p : Pool := {
    tileLength := 2,
    wallHeight := 3,
    wallThickness := 0.5,
    perimeterUnits := 32,
    outerCorners := 10,
    innerCorners := 6,
    concreteWeight := 2000
  }
  concreteWeightForWalls p = 198000 := by sorry

end pool_concrete_weight_l2079_207995


namespace distance_is_seven_l2079_207968

def point : ℝ × ℝ × ℝ := (2, 4, 6)

def line_point : ℝ × ℝ × ℝ := (8, 9, 9)

def line_direction : ℝ × ℝ × ℝ := (5, 2, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_seven :
  distance_to_line point line_point line_direction = 7 :=
sorry

end distance_is_seven_l2079_207968


namespace sum_of_x_and_y_is_one_l2079_207986

theorem sum_of_x_and_y_is_one (x y : ℝ) (h : x^2 + y^2 + x*y = 12*x - 8*y + 2) : x + y = 1 := by
  sorry

end sum_of_x_and_y_is_one_l2079_207986


namespace least_k_cube_divisible_by_120_l2079_207935

theorem least_k_cube_divisible_by_120 :
  ∀ k : ℕ, k > 0 → k^3 % 120 = 0 → k ≥ 30 :=
by
  sorry

end least_k_cube_divisible_by_120_l2079_207935


namespace julia_trip_euros_l2079_207931

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Exchange rate from USD to EUR -/
def exchange_rate : ℚ := 8 / 5

theorem julia_trip_euros (d' : ℕ) : 
  (exchange_rate * d' - 80 : ℚ) = d' → sum_of_digits d' = 7 := by
  sorry

end julia_trip_euros_l2079_207931


namespace square_of_prime_quadratic_l2079_207941

def is_square_of_prime (x : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ x = p^2

theorem square_of_prime_quadratic :
  ∀ n : ℕ, (is_square_of_prime (2*n^2 + 3*n - 35)) ↔ (n = 4 ∨ n = 12) :=
sorry

end square_of_prime_quadratic_l2079_207941


namespace traces_bag_weight_is_two_l2079_207961

/-- The weight of one of Trace's shopping bags -/
def traces_bag_weight (
  trace_bags : ℕ
  ) (gordon_bags : ℕ
  ) (gordon_bag1_weight : ℕ
  ) (gordon_bag2_weight : ℕ
  ) (lola_bags : ℕ
  ) : ℕ :=
  sorry

theorem traces_bag_weight_is_two :
  ∀ (trace_bags : ℕ)
    (gordon_bags : ℕ)
    (gordon_bag1_weight : ℕ)
    (gordon_bag2_weight : ℕ)
    (lola_bags : ℕ),
  trace_bags = 5 →
  gordon_bags = 2 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  lola_bags = 4 →
  trace_bags * traces_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags = 
    gordon_bag1_weight + gordon_bag2_weight →
  (gordon_bag1_weight + gordon_bag2_weight) / (3 * lola_bags) = 
    (gordon_bag1_weight + gordon_bag2_weight) / 3 - 1 →
  traces_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags = 2 :=
by sorry

end traces_bag_weight_is_two_l2079_207961


namespace smallest_sum_m_p_l2079_207980

/-- The function f(x) = arcsin(log_m(px)) has a domain that is a closed interval of length 1/1007 -/
def domain_length (m p : ℕ) : ℚ := (m^2 - 1 : ℚ) / (m * p)

/-- The theorem statement -/
theorem smallest_sum_m_p :
  ∀ m p : ℕ,
  m > 1 ∧ 
  p > 0 ∧ 
  domain_length m p = 1 / 1007 →
  m + p ≥ 2031 :=
sorry

end smallest_sum_m_p_l2079_207980


namespace washer_dryer_cost_difference_l2079_207925

theorem washer_dryer_cost_difference (total_cost washer_cost : ℕ) : 
  total_cost = 1200 → washer_cost = 710 → 
  washer_cost - (total_cost - washer_cost) = 220 := by
  sorry

end washer_dryer_cost_difference_l2079_207925


namespace dice_roll_distinct_roots_probability_l2079_207973

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def has_distinct_roots (a b : ℕ) : Prop := a^2 > 8*b

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 9

theorem dice_roll_distinct_roots_probability :
  (∀ a b : ℕ, is_valid_roll a → is_valid_roll b →
    (has_distinct_roots a b ↔ a^2 > 8*b)) →
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end dice_roll_distinct_roots_probability_l2079_207973


namespace max_cut_length_30x30_l2079_207946

/-- Represents a square board with side length and number of pieces it's cut into -/
structure Board :=
  (side : ℕ)
  (pieces : ℕ)

/-- Calculates the maximum possible total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  let piece_area := b.side * b.side / b.pieces
  let piece_perimeter := if piece_area = 4 then 10 else 8
  (b.pieces * piece_perimeter - 4 * b.side) / 2

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 pieces -/
theorem max_cut_length_30x30 :
  max_cut_length { side := 30, pieces := 225 } = 1065 :=
sorry

end max_cut_length_30x30_l2079_207946


namespace congruence_solution_l2079_207927

theorem congruence_solution (n : ℤ) : 
  (15 * n) % 47 = 9 % 47 → n % 47 = 10 % 47 := by
sorry

end congruence_solution_l2079_207927


namespace inscribed_hexagon_area_l2079_207937

/-- The area of a regular hexagon inscribed in a circle with area 256π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 256 * Real.pi →
  hexagon_area = 384 * Real.sqrt 3 →
  (∃ (r : ℝ), 
    r > 0 ∧
    circle_area = Real.pi * r^2 ∧
    hexagon_area = 6 * ((r^2 * Real.sqrt 3) / 4)) :=
by sorry

end inscribed_hexagon_area_l2079_207937


namespace sqrt_112_between_consecutive_integers_l2079_207940

theorem sqrt_112_between_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ n^2 < 112 ∧ (n + 1)^2 > 112 ∧ n * (n + 1) = 110 :=
by sorry

end sqrt_112_between_consecutive_integers_l2079_207940


namespace zeros_after_decimal_point_l2079_207994

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the decimal representation of (1 / (2^7 * 5^6)) * (3 / 5^2) is 7. -/
theorem zeros_after_decimal_point : ∃ (n : ℕ) (r : ℚ), 
  (1 / (2^7 * 5^6 : ℚ)) * (3 / 5^2 : ℚ) = 10^(-n : ℤ) * r ∧ 
  0 < r ∧ 
  r < 1 ∧ 
  n = 7 :=
by sorry

end zeros_after_decimal_point_l2079_207994


namespace trapezoid_properties_l2079_207901

-- Define the trapezoid and its properties
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  h : ℝ
  parallel_AB_CD : AB ≠ CD

-- Define the midpoints
def midpoint_M (t : Trapezoid) : ℝ × ℝ := sorry
def midpoint_N (t : Trapezoid) : ℝ × ℝ := sorry
def midpoint_P (t : Trapezoid) : ℝ × ℝ := sorry

-- Define the length of MN
def length_MN (t : Trapezoid) : ℝ := sorry

-- Define the area of triangle MNP
def area_MNP (t : Trapezoid) : ℝ := sorry

-- Theorem statement
theorem trapezoid_properties (t : Trapezoid) 
  (h_AB : t.AB = 15) 
  (h_CD : t.CD = 24) 
  (h_h : t.h = 14) : 
  length_MN t = 4.5 ∧ area_MNP t = 15.75 := by sorry

end trapezoid_properties_l2079_207901


namespace five_integers_problem_l2079_207971

theorem five_integers_problem : 
  ∃ (a b c d e : ℤ), 
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
      {3, 8, 9, 16, 17, 17, 18, 22, 23, 31} ∧
    a * b * c * d * e = 3360 := by
  sorry

end five_integers_problem_l2079_207971


namespace complete_square_m_value_l2079_207985

/-- Given the equation x^2 + 2x - 1 = 0, prove that when completing the square,
    the resulting equation (x+m)^2 = 2 has m = 1 -/
theorem complete_square_m_value (x : ℝ) :
  x^2 + 2*x - 1 = 0 → ∃ m : ℝ, (x + m)^2 = 2 ∧ m = 1 := by
  sorry

end complete_square_m_value_l2079_207985


namespace simplify_expression_l2079_207945

theorem simplify_expression (x : ℝ) : (5 - 2*x) - (4 + 7*x) = 1 - 9*x := by sorry

end simplify_expression_l2079_207945


namespace lars_production_l2079_207934

/-- Represents the baking rates and working hours of Lars' bakeshop --/
structure BakeshopData where
  bread_rate : ℕ  -- loaves of bread per hour
  baguette_rate : ℕ  -- baguettes per 2 hours
  croissant_rate : ℕ  -- croissants per 75 minutes
  working_hours : ℕ  -- hours worked per day

/-- Calculates the daily production of baked goods --/
def daily_production (data : BakeshopData) : ℕ × ℕ × ℕ :=
  let bread := data.bread_rate * data.working_hours
  let baguettes := data.baguette_rate * (data.working_hours / 2)
  let croissants := data.croissant_rate * (data.working_hours * 60 / 75)
  (bread, baguettes, croissants)

/-- Theorem stating Lars' daily production --/
theorem lars_production :
  let data : BakeshopData := {
    bread_rate := 10,
    baguette_rate := 30,
    croissant_rate := 20,
    working_hours := 6
  }
  daily_production data = (60, 90, 80) := by
  sorry

end lars_production_l2079_207934


namespace square_bricks_count_square_bricks_count_proof_l2079_207982

theorem square_bricks_count : ℕ → Prop :=
  fun total =>
    ∃ (length width : ℕ),
      -- Condition 1: length to width ratio is 6:5
      6 * width = 5 * length ∧
      -- Condition 2: rectangle arrangement leaves 43 bricks
      length * width + 43 = total ∧
      -- Condition 3: increasing both dimensions by 1 results in 68 bricks short
      (length + 1) * (width + 1) = total - 68 ∧
      -- The total number of bricks is 3043
      total = 3043

-- The proof of the theorem
theorem square_bricks_count_proof : square_bricks_count 3043 := by
  sorry

end square_bricks_count_square_bricks_count_proof_l2079_207982


namespace max_value_polynomial_l2079_207936

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z w : ℝ), z + w = 5 ∧ 
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≥ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4) ∧
  (∀ (z w : ℝ), z + w = 5 → 
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≤ 6084/17) :=
by sorry

end max_value_polynomial_l2079_207936


namespace power_mod_28_l2079_207914

theorem power_mod_28 : 17^1801 ≡ 17 [ZMOD 28] := by sorry

end power_mod_28_l2079_207914


namespace equal_surface_area_implies_L_value_l2079_207900

/-- Given a cube with edge length 30 and a rectangular solid with edge lengths 20, 30, and L,
    if their surface areas are equal, then L = 42. -/
theorem equal_surface_area_implies_L_value (L : ℝ) : 
  (6 * 30 * 30 = 2 * 20 * 30 + 2 * 20 * L + 2 * 30 * L) → L = 42 := by
  sorry

#check equal_surface_area_implies_L_value

end equal_surface_area_implies_L_value_l2079_207900
