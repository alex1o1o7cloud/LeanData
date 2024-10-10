import Mathlib

namespace x_squared_plus_y_squared_l1943_194366

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 94) :
  x^2 + y^2 = 7540 / 81 := by
  sorry

end x_squared_plus_y_squared_l1943_194366


namespace log_inequality_l1943_194383

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.log (1/a) / Real.log 0.3 > Real.log (1/b) / Real.log 0.3 := by
  sorry

end log_inequality_l1943_194383


namespace mode_is_nine_l1943_194399

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def frequency : Nat → Nat
| 0 => 8
| 1 => 8
| 2 => 12
| 3 => 11
| 4 => 10
| 5 => 8
| 6 => 9
| 7 => 8
| 8 => 12
| 9 => 14
| _ => 0

def is_mode (x : Nat) : Prop :=
  x ∈ digits ∧ ∀ y ∈ digits, frequency x ≥ frequency y

theorem mode_is_nine : is_mode 9 := by
  sorry

end mode_is_nine_l1943_194399


namespace angle_bisectors_concurrent_l1943_194360

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral ABCD
def Quadrilateral (A B C D : Point2D) : Prop := sorry

-- Define that P is an interior point of ABCD
def InteriorPoint (P : Point2D) (A B C D : Point2D) : Prop := sorry

-- Define the angle between three points
def Angle (P Q R : Point2D) : ℝ := sorry

-- Define the angle bisector
def AngleBisector (A B C : Point2D) : Point2D → Point2D → Prop := sorry

-- Define the perpendicular bisector of a line segment
def PerpendicularBisector (A B : Point2D) : Point2D → Point2D → Prop := sorry

-- Define when three lines are concurrent
def Concurrent (L1 L2 L3 : Point2D → Point2D → Prop) : Prop := sorry

theorem angle_bisectors_concurrent 
  (A B C D P : Point2D) 
  (h1 : Quadrilateral A B C D)
  (h2 : InteriorPoint P A B C D)
  (h3 : Angle P A D / Angle P B A / Angle D P A = 1 / 2 / 3)
  (h4 : Angle C B P / Angle B A P / Angle B P C = 1 / 2 / 3) :
  Concurrent 
    (AngleBisector A D P) 
    (AngleBisector P C B) 
    (PerpendicularBisector A B) := by sorry

end angle_bisectors_concurrent_l1943_194360


namespace dog_grouping_ways_l1943_194362

def total_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 5
def group3_size : ℕ := 3

theorem dog_grouping_ways :
  let remaining_dogs := total_dogs - 2  -- Sparky and Rex are already placed
  let remaining_group1 := group1_size - 1  -- Sparky is already in group 1
  let remaining_group2 := group2_size - 1  -- Rex is already in group 2
  (Nat.choose remaining_dogs remaining_group1) *
  (Nat.choose (remaining_dogs - remaining_group1) remaining_group2) *
  (Nat.choose (remaining_dogs - remaining_group1 - remaining_group2) group3_size) = 4200 := by
sorry

end dog_grouping_ways_l1943_194362


namespace equation_satisfies_condition_l1943_194331

theorem equation_satisfies_condition (x y z : ℤ) : 
  x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end equation_satisfies_condition_l1943_194331


namespace infinite_sum_theorem_l1943_194376

theorem infinite_sum_theorem (s : ℝ) (hs : s > 0) (heq : s^3 - 3/4 * s + 2 = 0) :
  ∑' n, (n + 1) * s^(2*n + 2) = 16/9 := by
sorry

end infinite_sum_theorem_l1943_194376


namespace largest_prime_factor_of_expression_l1943_194340

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (25^3 + 15^4 - 5^6 + 20^3) ∧ 
  ∀ (q : ℕ), q.Prime → q ∣ (25^3 + 15^4 - 5^6 + 20^3) → q ≤ p ∧ p = 97 := by
  sorry

end largest_prime_factor_of_expression_l1943_194340


namespace insufficient_album_capacity_l1943_194345

/-- Represents the capacity and quantity of each album type -/
structure AlbumType where
  capacity : ℕ
  quantity : ℕ

/-- Proves that the total capacity of all available albums is less than the total number of pictures -/
theorem insufficient_album_capacity 
  (type_a : AlbumType)
  (type_b : AlbumType)
  (type_c : AlbumType)
  (total_pictures : ℕ)
  (h1 : type_a.capacity = 12)
  (h2 : type_a.quantity = 6)
  (h3 : type_b.capacity = 18)
  (h4 : type_b.quantity = 4)
  (h5 : type_c.capacity = 24)
  (h6 : type_c.quantity = 3)
  (h7 : total_pictures = 480) :
  type_a.capacity * type_a.quantity + 
  type_b.capacity * type_b.quantity + 
  type_c.capacity * type_c.quantity < total_pictures :=
by sorry

end insufficient_album_capacity_l1943_194345


namespace product_sum_geq_geometric_mean_sum_l1943_194300

theorem product_sum_geq_geometric_mean_sum {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) := by
  sorry

end product_sum_geq_geometric_mean_sum_l1943_194300


namespace terrell_weight_lifting_l1943_194309

/-- The number of times Terrell lifts the weights in his initial routine -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in Terrell's initial routine (in pounds) -/
def initial_weight : ℕ := 25

/-- The number of dumbbells Terrell uses -/
def num_dumbbells : ℕ := 3

/-- The weight of each dumbbell in Terrell's new routine (in pounds) -/
def new_weight : ℕ := 20

/-- The total weight Terrell lifts in his initial routine -/
def total_initial_weight : ℕ := num_dumbbells * initial_weight * initial_lifts

/-- The minimum number of times Terrell needs to lift the new weights to match or exceed the initial total weight -/
def min_new_lifts : ℕ := 13

theorem terrell_weight_lifting :
  num_dumbbells * new_weight * min_new_lifts ≥ total_initial_weight :=
sorry

end terrell_weight_lifting_l1943_194309


namespace closest_integer_to_two_plus_sqrt_fifteen_l1943_194370

theorem closest_integer_to_two_plus_sqrt_fifteen :
  ∀ n : ℤ, n ≠ 6 → |6 - (2 + Real.sqrt 15)| < |n - (2 + Real.sqrt 15)| := by
  sorry

end closest_integer_to_two_plus_sqrt_fifteen_l1943_194370


namespace sector_area_l1943_194321

/-- Given a circular sector with an arc length of 2 cm and a central angle of 2 radians,
    prove that the area of the sector is 1 cm². -/
theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 2) (h2 : central_angle = 2) :
  (1 / 2) * (arc_length / central_angle)^2 * central_angle = 1 := by
  sorry

end sector_area_l1943_194321


namespace jerry_aunt_money_l1943_194337

/-- The amount of money Jerry received from his aunt -/
def aunt_money : ℝ := 9.05

/-- The amount of money Jerry received from his uncle -/
def uncle_money : ℝ := aunt_money

/-- The amount of money Jerry received from his friends -/
def friends_money : ℝ := 22 + 23 + 22 + 22

/-- The amount of money Jerry received from his sister -/
def sister_money : ℝ := 7

/-- The mean of all the money Jerry received -/
def mean_money : ℝ := 16.3

/-- The number of sources Jerry received money from -/
def num_sources : ℕ := 7

theorem jerry_aunt_money :
  (friends_money + sister_money + aunt_money + uncle_money) / num_sources = mean_money :=
sorry

end jerry_aunt_money_l1943_194337


namespace miniou_circuit_nodes_l1943_194386

/-- Definition of a Miniou circuit -/
structure MiniouCircuit where
  nodes : ℕ
  wires : ℕ
  wire_connects_two_nodes : True
  at_most_one_wire_between_nodes : True
  three_wires_per_node : True

/-- Theorem: A Miniou circuit with 13788 wires has 9192 nodes -/
theorem miniou_circuit_nodes (c : MiniouCircuit) (h : c.wires = 13788) : c.nodes = 9192 := by
  sorry

end miniou_circuit_nodes_l1943_194386


namespace no_solution_for_12x4x_divisible_by_99_l1943_194342

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_form (x : ℕ) : ℕ := 12000 + 1000 * x + 40 + x

theorem no_solution_for_12x4x_divisible_by_99 :
  ¬ ∃ x : ℕ, is_single_digit x ∧ (number_form x) % 99 = 0 := by
sorry

end no_solution_for_12x4x_divisible_by_99_l1943_194342


namespace space_station_cost_share_l1943_194326

/-- Proves that if a total cost of 50 billion dollars is shared equally among 500 million people,
    then each person's share is 100 dollars. -/
theorem space_station_cost_share :
  let total_cost : ℝ := 50 * 10^9  -- 50 billion dollars
  let num_people : ℝ := 500 * 10^6  -- 500 million people
  let share_per_person : ℝ := total_cost / num_people
  share_per_person = 100 := by sorry

end space_station_cost_share_l1943_194326


namespace max_area_quadrilateral_l1943_194313

/-- Given a point P in the first quadrant and points A on the x-axis and B on the y-axis 
    such that PA = PB = 2, the maximum area of quadrilateral PAOB is 2 + 2√2. -/
theorem max_area_quadrilateral (P A B : ℝ × ℝ) : 
  (0 < P.1 ∧ 0 < P.2) →  -- P is in the first quadrant
  A.2 = 0 →  -- A is on the x-axis
  B.1 = 0 →  -- B is on the y-axis
  Real.sqrt ((P.1 - A.1)^2 + P.2^2) = 2 →  -- PA = 2
  Real.sqrt (P.1^2 + (P.2 - B.2)^2) = 2 →  -- PB = 2
  (∃ (area : ℝ), ∀ (Q : ℝ × ℝ), 
    (0 < Q.1 ∧ 0 < Q.2) →
    Real.sqrt ((Q.1 - A.1)^2 + Q.2^2) = 2 →
    Real.sqrt (Q.1^2 + (Q.2 - B.2)^2) = 2 →
    (1/2 * |A.1 * Q.1 + B.2 * Q.2| ≤ area)) ∧
  (1/2 * |A.1 * P.1 + B.2 * P.2| = 2 + 2 * Real.sqrt 2) :=
by sorry

end max_area_quadrilateral_l1943_194313


namespace lawrence_marbles_l1943_194302

theorem lawrence_marbles (total_marbles : ℕ) (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 5504) 
  (h2 : marbles_per_friend = 86) : 
  total_marbles / marbles_per_friend = 64 := by
  sorry

end lawrence_marbles_l1943_194302


namespace imaginary_part_of_z_l1943_194368

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  Complex.im z = 3 := by
sorry

end imaginary_part_of_z_l1943_194368


namespace rectangle_division_perimeter_paradox_l1943_194348

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Predicate to check if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Theorem stating that there exists a rectangle with non-integer perimeter
    that can be divided into rectangles with integer perimeters -/
theorem rectangle_division_perimeter_paradox :
  ∃ (big : Rectangle) (small1 small2 : Rectangle),
    ¬isInteger big.perimeter ∧
    isInteger small1.perimeter ∧
    isInteger small2.perimeter ∧
    big.width = small1.width ∧
    big.width = small2.width ∧
    big.height = small1.height + small2.height :=
sorry

end rectangle_division_perimeter_paradox_l1943_194348


namespace triangle_inequality_l1943_194379

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 - a*b = c^2) : 
  (a - c) * (b - c) ≤ 0 := by
sorry

end triangle_inequality_l1943_194379


namespace square_plus_reciprocal_square_l1943_194382

theorem square_plus_reciprocal_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^2 + 1/x^2 = 7 := by
  sorry

end square_plus_reciprocal_square_l1943_194382


namespace gecko_cost_is_fifteen_l1943_194334

/-- Represents the cost of feeding Harry's pets -/
structure PetFeedingCost where
  geckos : ℕ
  iguanas : ℕ
  snakes : ℕ
  snake_cost : ℕ
  iguana_cost : ℕ
  total_annual_cost : ℕ

/-- Calculates the monthly cost per gecko -/
def gecko_monthly_cost (p : PetFeedingCost) : ℚ :=
  (p.total_annual_cost / 12 - (p.snakes * p.snake_cost + p.iguanas * p.iguana_cost)) / p.geckos

/-- Theorem stating that the monthly cost per gecko is $15 -/
theorem gecko_cost_is_fifteen (p : PetFeedingCost) 
    (h1 : p.geckos = 3)
    (h2 : p.iguanas = 2)
    (h3 : p.snakes = 4)
    (h4 : p.snake_cost = 10)
    (h5 : p.iguana_cost = 5)
    (h6 : p.total_annual_cost = 1140) :
    gecko_monthly_cost p = 15 := by
  sorry

end gecko_cost_is_fifteen_l1943_194334


namespace quadratic_root_triple_l1943_194316

/-- 
For a quadratic equation ax^2 + bx + c = 0, if one root is triple the other, 
then 3b^2 = 16ac.
-/
theorem quadratic_root_triple (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 →
  a * x₁^2 + b * x₁ + c = 0 →
  a * x₂^2 + b * x₂ + c = 0 →
  x₂ = 3 * x₁ →
  3 * b^2 = 16 * a * c :=
by sorry


end quadratic_root_triple_l1943_194316


namespace least_common_denominator_sum_l1943_194378

theorem least_common_denominator_sum (a b c d e : ℕ) 
  (ha : a = 4) (hb : b = 5) (hc : c = 6) (hd : d = 7) (he : e = 8) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 840 := by
  sorry

end least_common_denominator_sum_l1943_194378


namespace intersection_complement_equals_singleton_l1943_194305

open Set

universe u

def U : Finset ℕ := {1, 2, 3, 4}

theorem intersection_complement_equals_singleton
  (A B : Finset ℕ)
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ (A ∪ B)) = {4})
  (h4 : B = {1, 2}) :
  A ∩ (U \ B) = {3} := by
  sorry

end intersection_complement_equals_singleton_l1943_194305


namespace faster_car_speed_l1943_194364

/-- Given two cars traveling in opposite directions for 5 hours, with one car
    traveling 10 mi/h faster than the other, and ending up 500 miles apart,
    prove that the speed of the faster car is 55 mi/h. -/
theorem faster_car_speed (slower_speed faster_speed : ℝ) : 
  faster_speed = slower_speed + 10 →
  5 * slower_speed + 5 * faster_speed = 500 →
  faster_speed = 55 := by sorry

end faster_car_speed_l1943_194364


namespace sqrt_difference_approximation_l1943_194350

theorem sqrt_difference_approximation : 
  |Real.sqrt (49 + 121) - Real.sqrt (64 - 36) - 7.75| < 0.01 := by
  sorry

end sqrt_difference_approximation_l1943_194350


namespace sum_of_reciprocals_of_roots_l1943_194341

theorem sum_of_reciprocals_of_roots (m n : ℝ) : 
  m^2 + 2*m - 3 = 0 → n^2 + 2*n - 3 = 0 → m ≠ 0 → n ≠ 0 → 1/m + 1/n = 2/3 := by
  sorry

end sum_of_reciprocals_of_roots_l1943_194341


namespace sum_product_solution_l1943_194332

theorem sum_product_solution (S P : ℝ) (x y : ℝ) (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end sum_product_solution_l1943_194332


namespace sum_of_squares_equals_two_l1943_194314

theorem sum_of_squares_equals_two (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) : 
  x^2 + y^2 + z^2 = 2 := by
  sorry

end sum_of_squares_equals_two_l1943_194314


namespace inequality_proof_l1943_194336

theorem inequality_proof (a b c : ℝ) (h : Real.sqrt a ≥ Real.sqrt (b * c) ∧ Real.sqrt (b * c) ≥ Real.sqrt a - c) : b * c ≥ b + c := by
  sorry

end inequality_proof_l1943_194336


namespace complex_exponential_to_rectangular_l1943_194367

theorem complex_exponential_to_rectangular : 2 * Real.sqrt 3 * Complex.exp (Complex.I * (13 * Real.pi / 6)) = 3 + Complex.I * Real.sqrt 3 := by
  sorry

end complex_exponential_to_rectangular_l1943_194367


namespace roots_shifted_polynomial_l1943_194354

theorem roots_shifted_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 4*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 23*x + 7 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end roots_shifted_polynomial_l1943_194354


namespace polynomial_divisibility_l1943_194361

theorem polynomial_divisibility (a b c : ℕ) :
  ∃ q : Polynomial ℚ, X^(3*a) + X^(3*b+1) + X^(3*c+2) = (X^2 + X + 1) * q :=
by sorry

end polynomial_divisibility_l1943_194361


namespace rectangle_area_l1943_194329

theorem rectangle_area (L B : ℝ) 
  (h1 : L - B = 23) 
  (h2 : 2 * L + 2 * B = 226) : 
  L * B = 3060 := by sorry

end rectangle_area_l1943_194329


namespace intersection_union_theorem_l1943_194356

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x + 12 = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + 3*x + 2*b = 0}
def C : Set ℝ := {2, -3}

-- State the theorem
theorem intersection_union_theorem (a b : ℝ) :
  (A a ∩ B b = {2}) →
  (A a = {2, 6}) →
  (B b = {-5, 2}) →
  ((A a ∪ B b) ∩ C = {2}) :=
by sorry

end intersection_union_theorem_l1943_194356


namespace favorite_song_not_heard_probability_l1943_194306

-- Define the number of songs
def num_songs : ℕ := 10

-- Define the duration of the shortest song (in seconds)
def shortest_song : ℕ := 40

-- Define the increment in duration for each subsequent song (in seconds)
def duration_increment : ℕ := 40

-- Define the duration of the favorite song (in seconds)
def favorite_song_duration : ℕ := 240

-- Define the total playtime we're considering (in seconds)
def total_playtime : ℕ := 300

-- Function to calculate the duration of the nth song
def song_duration (n : ℕ) : ℕ := shortest_song + (n - 1) * duration_increment

-- Theorem stating the probability of not hearing the favorite song in its entirety
theorem favorite_song_not_heard_probability :
  let favorite_song_index : ℕ := (favorite_song_duration - shortest_song) / duration_increment + 1
  (favorite_song_index ≤ num_songs) →
  (∀ n : ℕ, n < favorite_song_index → song_duration n + favorite_song_duration > total_playtime) →
  (num_songs - 1 : ℚ) / num_songs = 9 / 10 :=
by sorry

end favorite_song_not_heard_probability_l1943_194306


namespace arithmetic_sequence_sum_l1943_194387

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ

/-- Theorem: For an arithmetic sequence with S_5 = 10 and S_10 = 30, S_15 = 60 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 5 = 10) 
  (h2 : a.S 10 = 30) : 
  a.S 15 = 60 := by
  sorry

end arithmetic_sequence_sum_l1943_194387


namespace min_value_expression_l1943_194377

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 19 / 4 :=
sorry

end min_value_expression_l1943_194377


namespace percentage_problem_l1943_194369

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 = (5 / 100) * 60 + 23 → P = 65 := by
  sorry

end percentage_problem_l1943_194369


namespace discount_clinic_savings_l1943_194344

/-- Calculates the savings when using a discount clinic compared to a normal doctor visit -/
theorem discount_clinic_savings
  (normal_cost : ℝ)
  (discount_percentage : ℝ)
  (discount_visits : ℕ)
  (h1 : normal_cost = 200)
  (h2 : discount_percentage = 0.7)
  (h3 : discount_visits = 2) :
  normal_cost - discount_visits * (normal_cost * (1 - discount_percentage)) = 80 := by
  sorry

end discount_clinic_savings_l1943_194344


namespace fruit_box_problem_l1943_194394

theorem fruit_box_problem (total_fruit oranges peaches apples : ℕ) : 
  total_fruit = 56 →
  oranges = total_fruit / 4 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 := by
sorry

end fruit_box_problem_l1943_194394


namespace max_abs_z_on_line_segment_l1943_194318

theorem max_abs_z_on_line_segment (z : ℂ) :
  Complex.abs (z - 6*I) + Complex.abs (z - 5) = Real.sqrt 61 →
  Complex.abs z ≤ 6 ∧ ∃ w : ℂ, Complex.abs (w - 6*I) + Complex.abs (w - 5) = Real.sqrt 61 ∧ Complex.abs w = 6 :=
by sorry

end max_abs_z_on_line_segment_l1943_194318


namespace factor_expression_l1943_194330

theorem factor_expression (x : ℝ) : 6 * x^3 - 54 * x = 6 * x * (x + 3) * (x - 3) := by
  sorry

end factor_expression_l1943_194330


namespace add_6666_seconds_to_3pm_l1943_194335

/-- Represents a time of day in hours, minutes, and seconds -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Converts seconds to a TimeOfDay structure -/
def secondsToTime (totalSeconds : Nat) : TimeOfDay :=
  let hours := totalSeconds / 3600
  let remainingSeconds := totalSeconds % 3600
  let minutes := remainingSeconds / 60
  let seconds := remainingSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds a TimeOfDay to another TimeOfDay, handling overflow -/
def addTime (t1 t2 : TimeOfDay) : TimeOfDay :=
  let totalSeconds := (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) +
                      (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)
  secondsToTime totalSeconds

theorem add_6666_seconds_to_3pm (startTime : TimeOfDay) (elapsedSeconds : Nat) :
  startTime.hours = 15 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 ∧ 
  elapsedSeconds = 6666 →
  let endTime := addTime startTime (secondsToTime elapsedSeconds)
  endTime.hours = 16 ∧ endTime.minutes = 51 ∧ endTime.seconds = 6 := by
  sorry

end add_6666_seconds_to_3pm_l1943_194335


namespace quadratic_properties_l1943_194303

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

-- Define the interval
def interval : Set ℝ := Set.Icc 1 4

theorem quadratic_properties :
  ∃ (x_min : ℝ), x_min ∈ interval ∧ 
  (∀ (x : ℝ), x ∈ interval → f x_min ≤ f x) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1 1.5 → y ∈ Set.Icc 1 1.5 → x ≤ y → f x ≥ f y) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1.5 4 → y ∈ Set.Icc 1.5 4 → x ≤ y → f x ≤ f y) :=
by sorry

end quadratic_properties_l1943_194303


namespace last_colored_number_l1943_194325

/-- The number of columns in the table -/
def num_columns : ℕ := 8

/-- The triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of a number in the table -/
def position (n : ℕ) : ℕ := n % num_columns

/-- Predicate to check if a number is colored -/
def is_colored (n : ℕ) : Prop := ∃ k : ℕ, n = triangular_number k

/-- Predicate to check if all columns are colored up to a certain number -/
def all_columns_colored (n : ℕ) : Prop :=
  ∀ col : ℕ, col < num_columns → ∃ m : ℕ, m ≤ n ∧ is_colored m ∧ position m = col

/-- The main theorem -/
theorem last_colored_number :
  ∃ n : ℕ, n = 120 ∧ is_colored n ∧ all_columns_colored n ∧
  ∀ m : ℕ, m < n → ¬(all_columns_colored m) :=
sorry

end last_colored_number_l1943_194325


namespace cylinder_volume_ratio_l1943_194355

/-- The ratio of cylinder volumes formed from a 5x8 rectangle -/
theorem cylinder_volume_ratio : 
  ∀ (h₁ h₂ r₁ r₂ : ℝ), 
    h₁ = 8 ∧ h₂ = 5 ∧ 
    2 * Real.pi * r₁ = 5 ∧ 
    2 * Real.pi * r₂ = 8 →
    max (Real.pi * r₁^2 * h₁) (Real.pi * r₂^2 * h₂) / 
    min (Real.pi * r₁^2 * h₁) (Real.pi * r₂^2 * h₂) = 8/5 := by
  sorry


end cylinder_volume_ratio_l1943_194355


namespace pages_to_read_on_third_day_l1943_194372

/-- Given a book with 100 pages and Lance's reading progress over three days,
    prove that he needs to read 35 pages on the third day to finish the book. -/
theorem pages_to_read_on_third_day (pages_day1 pages_day2 : ℕ) 
  (h1 : pages_day1 = 35)
  (h2 : pages_day2 = pages_day1 - 5) :
  100 - (pages_day1 + pages_day2) = 35 := by
  sorry

end pages_to_read_on_third_day_l1943_194372


namespace secret_code_is_819_l1943_194307

/-- Represents a three-digit code -/
structure Code :=
  (d1 d2 d3 : Nat)
  (h1 : d1 < 10)
  (h2 : d2 < 10)
  (h3 : d3 < 10)

/-- Checks if a given digit is in the correct position in the code -/
def correctPosition (c : Code) (pos : Nat) (digit : Nat) : Prop :=
  match pos with
  | 1 => c.d1 = digit
  | 2 => c.d2 = digit
  | 3 => c.d3 = digit
  | _ => False

/-- Checks if a given digit is in the code but in the wrong position -/
def correctButWrongPosition (c : Code) (pos : Nat) (digit : Nat) : Prop :=
  (c.d1 = digit ∨ c.d2 = digit ∨ c.d3 = digit) ∧ ¬correctPosition c pos digit

/-- Represents the clues given in the problem -/
def clues (c : Code) : Prop :=
  (∃ d, (d = 0 ∨ d = 7 ∨ d = 9) ∧ (correctPosition c 1 d ∨ correctPosition c 2 d ∨ correctPosition c 3 d)) ∧
  (c.d1 ≠ 0 ∧ c.d2 ≠ 3 ∧ c.d3 ≠ 2) ∧
  (∃ d1 d2, (d1 = 1 ∨ d1 = 0 ∨ d1 = 8) ∧ (d2 = 1 ∨ d2 = 0 ∨ d2 = 8) ∧ d1 ≠ d2 ∧
    correctButWrongPosition c 1 d1 ∧ correctButWrongPosition c 2 d2) ∧
  (∃ d, (d = 9 ∨ d = 2 ∨ d = 6) ∧ correctButWrongPosition c 1 d) ∧
  (∃ d, (d = 6 ∨ d = 7 ∨ d = 8) ∧ correctButWrongPosition c 2 d)

theorem secret_code_is_819 : ∀ c : Code, clues c → c.d1 = 8 ∧ c.d2 = 1 ∧ c.d3 = 9 := by
  sorry

end secret_code_is_819_l1943_194307


namespace sum_of_roots_l1943_194315

theorem sum_of_roots (M : ℝ) : (∃ M₁ M₂ : ℝ, M₁ * (M₁ - 8) = 7 ∧ M₂ * (M₂ - 8) = 7 ∧ M₁ + M₂ = 8) := by
  sorry

end sum_of_roots_l1943_194315


namespace stratified_sample_older_45_correct_l1943_194385

/-- Calculates the number of employees older than 45 to be drawn in a stratified sample -/
def stratifiedSampleOlder45 (totalEmployees : ℕ) (employeesOlder45 : ℕ) (sampleSize : ℕ) : ℕ :=
  (employeesOlder45 * sampleSize) / totalEmployees

/-- Proves that the stratified sample for employees older than 45 is correct -/
theorem stratified_sample_older_45_correct :
  stratifiedSampleOlder45 400 160 50 = 20 := by
  sorry

#eval stratifiedSampleOlder45 400 160 50

end stratified_sample_older_45_correct_l1943_194385


namespace product_zero_implies_factor_zero_l1943_194393

theorem product_zero_implies_factor_zero (a b : ℝ) (h : a * b = 0) :
  a = 0 ∨ b = 0 := by
  sorry

end product_zero_implies_factor_zero_l1943_194393


namespace min_sum_squares_min_sum_squares_achievable_l1943_194389

theorem min_sum_squares (a b c : ℕ) (h : a + 2*b + 3*c = 73) : 
  a^2 + b^2 + c^2 ≥ 381 := by
sorry

theorem min_sum_squares_achievable : 
  ∃ (a b c : ℕ), a + 2*b + 3*c = 73 ∧ a^2 + b^2 + c^2 = 381 := by
sorry

end min_sum_squares_min_sum_squares_achievable_l1943_194389


namespace sum_upper_bound_l1943_194363

theorem sum_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := by
sorry

end sum_upper_bound_l1943_194363


namespace problem_2003_2001_l1943_194351

theorem problem_2003_2001 : 2003^3 - 2001 * 2003^2 - 2001^2 * 2003 + 2001^3 = 8 := by
  sorry

end problem_2003_2001_l1943_194351


namespace temperature_conversion_l1943_194319

theorem temperature_conversion (k : ℝ) (t : ℝ) : 
  (t = 5 / 9 * (k - 32)) → (k = 167) → (t = 75) := by
  sorry

end temperature_conversion_l1943_194319


namespace total_nail_polishes_l1943_194304

/-- The number of nail polishes each person has -/
structure NailPolishes where
  kim : ℕ
  heidi : ℕ
  karen : ℕ
  laura : ℕ
  simon : ℕ

/-- The conditions of the nail polish problem -/
def nail_polish_conditions (np : NailPolishes) : Prop :=
  np.kim = 25 ∧
  np.heidi = np.kim + 8 ∧
  np.karen = np.kim - 6 ∧
  np.laura = 2 * np.kim ∧
  np.simon = (np.kim / 2 + 10)

/-- The theorem stating the total number of nail polishes -/
theorem total_nail_polishes (np : NailPolishes) :
  nail_polish_conditions np →
  np.heidi + np.karen + np.laura + np.simon = 125 :=
by
  sorry


end total_nail_polishes_l1943_194304


namespace pentagonal_prism_vertices_l1943_194327

/-- Definition of a pentagonal prism -/
structure PentagonalPrism :=
  (bases : ℕ)
  (rectangular_faces : ℕ)
  (h_bases : bases = 2)
  (h_faces : rectangular_faces = 5)

/-- The number of vertices in a pentagonal prism -/
def num_vertices (p : PentagonalPrism) : ℕ := 10

/-- Theorem stating that a pentagonal prism has 10 vertices -/
theorem pentagonal_prism_vertices (p : PentagonalPrism) : num_vertices p = 10 := by
  sorry

end pentagonal_prism_vertices_l1943_194327


namespace sum_binary_digits_345_l1943_194373

/-- Sum of binary digits of a natural number -/
def sum_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The sum of the binary digits of 345 is 5 -/
theorem sum_binary_digits_345 : sum_binary_digits 345 = 5 := by
  sorry

end sum_binary_digits_345_l1943_194373


namespace complement_union_M_N_l1943_194388

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def M : Finset ℕ := {1,3,5,7}
def N : Finset ℕ := {5,6,7}

theorem complement_union_M_N :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end complement_union_M_N_l1943_194388


namespace prism_volume_approximation_l1943_194365

/-- Represents a right rectangular prism -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- The main theorem to prove -/
theorem prism_volume_approximation (p : RectangularPrism) 
  (h1 : p.a * p.b = 54)
  (h2 : p.b * p.c = 56)
  (h3 : p.a * p.c = 60) :
  round (volume p) = 426 := by
  sorry


end prism_volume_approximation_l1943_194365


namespace max_truck_load_is_2000_l1943_194371

/-- Represents the maximum load a truck can carry given the following conditions:
    - There are three trucks for delivery
    - Boxes come in two weights: 10 pounds and 40 pounds
    - Customer ordered equal quantities of both lighter and heavier products
    - Total number of boxes shipped is 240
-/
def max_truck_load : ℕ :=
  let total_boxes : ℕ := 240
  let num_trucks : ℕ := 3
  let light_box_weight : ℕ := 10
  let heavy_box_weight : ℕ := 40
  let boxes_per_type : ℕ := total_boxes / 2
  let total_weight : ℕ := boxes_per_type * light_box_weight + boxes_per_type * heavy_box_weight
  total_weight / num_trucks

theorem max_truck_load_is_2000 : max_truck_load = 2000 := by
  sorry

end max_truck_load_is_2000_l1943_194371


namespace shopping_total_l1943_194339

def tuesday_discount : ℝ := 0.1
def jimmy_shorts_count : ℕ := 3
def jimmy_shorts_price : ℝ := 15
def irene_shirts_count : ℕ := 5
def irene_shirts_price : ℝ := 17

theorem shopping_total : 
  let total_before_discount := jimmy_shorts_count * jimmy_shorts_price + 
                               irene_shirts_count * irene_shirts_price
  let discount := total_before_discount * tuesday_discount
  let final_amount := total_before_discount - discount
  final_amount = 117 := by sorry

end shopping_total_l1943_194339


namespace f_composition_proof_l1943_194359

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_proof : f (f (f (-1))) = Real.pi + 1 := by
  sorry

end f_composition_proof_l1943_194359


namespace milk_fraction_after_transfers_l1943_194392

/-- Represents the contents of a cup --/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the problem setup --/
def initial_setup : CupContents × CupContents :=
  ({ tea := 8, milk := 0 }, { tea := 0, milk := 8 })

/-- Transfers a fraction of tea from the first cup to the second --/
def transfer_tea (cups : CupContents × CupContents) (fraction : ℚ) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let transfer_amount := cup1.tea * fraction
  ({ tea := cup1.tea - transfer_amount, milk := cup1.milk },
   { tea := cup2.tea + transfer_amount, milk := cup2.milk })

/-- Transfers a fraction of the mixture from the second cup to the first --/
def transfer_mixture (cups : CupContents × CupContents) (fraction : ℚ) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let total2 := cup2.tea + cup2.milk
  let transfer_tea := cup2.tea * fraction
  let transfer_milk := cup2.milk * fraction
  ({ tea := cup1.tea + transfer_tea, milk := cup1.milk + transfer_milk },
   { tea := cup2.tea - transfer_tea, milk := cup2.milk - transfer_milk })

/-- Calculates the fraction of milk in a cup --/
def milk_fraction (cup : CupContents) : ℚ :=
  cup.milk / (cup.tea + cup.milk)

/-- The main theorem to prove --/
theorem milk_fraction_after_transfers :
  let cups1 := transfer_tea initial_setup (1/4)
  let cups2 := transfer_mixture cups1 (1/3)
  milk_fraction cups2.fst = 1/3 := by sorry


end milk_fraction_after_transfers_l1943_194392


namespace factorization_equality_l1943_194374

theorem factorization_equality (a b : ℝ) : a * b^2 - 3 * a = a * (b + Real.sqrt 3) * (b - Real.sqrt 3) := by
  sorry

end factorization_equality_l1943_194374


namespace square_of_real_not_always_positive_l1943_194301

theorem square_of_real_not_always_positive : 
  ¬(∀ (a : ℝ), a^2 > 0) :=
by
  sorry

end square_of_real_not_always_positive_l1943_194301


namespace square_root_sum_equality_l1943_194323

theorem square_root_sum_equality : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + 2 * Real.sqrt (16 + 8 * Real.sqrt 3) = 10 + 6 * Real.sqrt 3 := by
  sorry

end square_root_sum_equality_l1943_194323


namespace magnitude_z_equals_magnitude_iz_l1943_194317

theorem magnitude_z_equals_magnitude_iz (z : ℂ) : Complex.abs z = Complex.abs (Complex.I * z) := by
  sorry

end magnitude_z_equals_magnitude_iz_l1943_194317


namespace abes_budget_l1943_194311

/-- Abe's restaurant budget problem -/
theorem abes_budget (B : ℚ) 
  (food_expense : B / 3 = B - (B / 4 + 1250))
  (supplies_expense : B / 4 = B - (B / 3 + 1250))
  (wages_expense : 1250 = B - (B / 3 + B / 4))
  (total_expense : B = B / 3 + B / 4 + 1250) :
  B = 3000 := by
  sorry

end abes_budget_l1943_194311


namespace inequality_proof_l1943_194381

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l1943_194381


namespace function_and_inequality_solution_l1943_194333

noncomputable section

variables (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem function_and_inequality_solution 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : f 0 = 2020)
  (h3 : ∀ x, f' x = f x - 2) :
  (∀ x, f x = 2 + 2018 * Real.exp x) ∧ 
  {x : ℝ | f x + 4034 > 2 * f' x} = {x : ℝ | x < Real.log 2} := by
  sorry

end

end function_and_inequality_solution_l1943_194333


namespace digit_150_is_2_l1943_194343

/-- The sequence of digits formed by concatenating all integers from 100 down to 50 -/
def digit_sequence : List Nat := sorry

/-- The 150th digit in the sequence -/
def digit_150 : Nat := sorry

/-- Theorem stating that the 150th digit in the sequence is 2 -/
theorem digit_150_is_2 : digit_150 = 2 := by sorry

end digit_150_is_2_l1943_194343


namespace constant_product_l1943_194324

-- Define the parabolas and points
def parabola1 (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def parabola2 (q : ℝ) (x y : ℝ) : Prop := x^2 = 2*q*y ∧ q > 0

-- Define the property of being distinct points on parabola1
def distinct_points_on_parabola1 (p : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  parabola1 p x₁ y₁ ∧ parabola1 p x₂ y₂ ∧ parabola1 p x₃ y₃ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

-- Define the property of two sides being tangent to parabola2
def two_sides_tangent (q : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  ∃ (xt₁ yt₁ xt₂ yt₂ : ℝ),
    parabola2 q xt₁ yt₁ ∧ parabola2 q xt₂ yt₂ ∧
    (xt₁ - x₁) * (y₂ - y₁) = (yt₁ - y₁) * (x₂ - x₁) ∧
    (xt₁ - x₂) * (y₁ - y₂) = (yt₁ - y₂) * (x₁ - x₂) ∧
    (xt₂ - x₁) * (y₃ - y₁) = (yt₂ - y₁) * (x₃ - x₁) ∧
    (xt₂ - x₃) * (y₁ - y₃) = (yt₂ - y₃) * (x₁ - x₃)

-- Theorem statement
theorem constant_product (p q : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  distinct_points_on_parabola1 p x₁ y₁ x₂ y₂ x₃ y₃ →
  two_sides_tangent q x₁ y₁ x₂ y₂ x₃ y₃ →
  ∃ (c : ℝ), ∀ (i j : Fin 3), i ≠ j →
    let y := [y₁, y₂, y₃]
    y[i] * y[j] * (y[i] + y[j]) = c :=
by sorry

end constant_product_l1943_194324


namespace smaller_bill_denomination_l1943_194312

def total_amount : ℕ := 1000
def fraction_smaller : ℚ := 3 / 10
def larger_denomination : ℕ := 100
def total_bills : ℕ := 13

theorem smaller_bill_denomination :
  ∃ (smaller_denomination : ℕ),
    (fraction_smaller * total_amount) / smaller_denomination +
    ((1 - fraction_smaller) * total_amount) / larger_denomination = total_bills ∧
    smaller_denomination = 50 := by
  sorry

end smaller_bill_denomination_l1943_194312


namespace inequality_solution_l1943_194308

theorem inequality_solution (x : ℝ) : 
  1 / (x - 2) + 4 / (x + 5) ≥ 1 ↔ x ∈ Set.Icc (-1) (7/2) :=
by sorry

end inequality_solution_l1943_194308


namespace largest_four_digit_divisible_by_33_l1943_194320

/-- A function that checks if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is divisible by 33 -/
def divisible_by_33 (n : ℕ) : Prop :=
  n % 33 = 0

/-- Theorem stating that 9999 is the largest four-digit number divisible by 33 -/
theorem largest_four_digit_divisible_by_33 :
  is_four_digit 9999 ∧ 
  divisible_by_33 9999 ∧ 
  ∀ n : ℕ, is_four_digit n → divisible_by_33 n → n ≤ 9999 :=
by
  sorry

end largest_four_digit_divisible_by_33_l1943_194320


namespace brick_height_specific_brick_height_l1943_194328

/-- The height of a brick given its dimensions and the wall it's used to build -/
theorem brick_height (brick_length brick_width : ℝ)
                     (wall_length wall_width wall_height : ℝ)
                     (num_bricks : ℕ) : ℝ :=
  let wall_volume := wall_length * wall_width * wall_height
  let brick_volume := wall_volume / num_bricks
  brick_volume / (brick_length * brick_width)

/-- The height of the brick is 7.5 cm given the specified conditions -/
theorem specific_brick_height :
  brick_height 20 10 2500 200 75 25000 = 7.5 := by
  sorry

end brick_height_specific_brick_height_l1943_194328


namespace angela_marbles_l1943_194395

theorem angela_marbles :
  ∀ (a : ℕ), 
  (∃ (b c d : ℕ),
    b = 3 * a ∧
    c = 2 * b ∧
    d = 4 * c ∧
    a + b + c + d = 204) →
  a = 6 := by
sorry

end angela_marbles_l1943_194395


namespace quadratic_inequality_holds_for_all_x_l1943_194397

theorem quadratic_inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4)*x - k + 8 > 0) ↔ -2 < k ∧ k < 6 := by
  sorry

end quadratic_inequality_holds_for_all_x_l1943_194397


namespace spider_web_paths_l1943_194349

/-- The number of paths from (0,0) to (m,n) on a grid, moving only up and right -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The coordinates of the target point -/
def target : (ℕ × ℕ) := (4, 3)

theorem spider_web_paths : 
  gridPaths target.1 target.2 = 35 := by sorry

end spider_web_paths_l1943_194349


namespace complex_equation_solution_l1943_194380

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.mk 1 2 * a + Complex.mk b 0 = Complex.I * 2) → (a = 1 ∧ b = -1) :=
by sorry

end complex_equation_solution_l1943_194380


namespace ice_cream_flavors_l1943_194310

theorem ice_cream_flavors (total_flavors : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : 
  total_flavors = 100 →
  tried_two_years_ago = total_flavors / 4 →
  tried_last_year = 2 * tried_two_years_ago →
  total_flavors - (tried_two_years_ago + tried_last_year) = 25 :=
by
  sorry

end ice_cream_flavors_l1943_194310


namespace range_of_x_when_a_is_one_range_of_a_l1943_194390

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Theorem 1
theorem range_of_x_when_a_is_one (x : ℝ) (h1 : p x 1) (h2 : q x) :
  2 < x ∧ x < 3 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x, ¬(p x a) → ¬(q x))
  (h_not_nec : ∃ x, ¬(q x) ∧ p x a) :
  1 < a ∧ a ≤ 2 := by sorry

end range_of_x_when_a_is_one_range_of_a_l1943_194390


namespace bridge_dealing_is_systematic_sampling_l1943_194391

/-- Represents the sampling method used in card dealing --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a deck of cards --/
structure Deck :=
  (size : Nat)
  (shuffled : Bool)

/-- Represents the card dealing process in bridge --/
structure BridgeDealing :=
  (deck : Deck)
  (startingCardRandom : Bool)
  (dealInOrder : Bool)
  (playerHandSize : Nat)

/-- Determines the sampling method used in bridge card dealing --/
def determineSamplingMethod (dealing : BridgeDealing) : SamplingMethod :=
  sorry

/-- Theorem stating that bridge card dealing uses Systematic Sampling --/
theorem bridge_dealing_is_systematic_sampling 
  (dealing : BridgeDealing) 
  (h1 : dealing.deck.size = 52)
  (h2 : dealing.deck.shuffled = true)
  (h3 : dealing.startingCardRandom = true)
  (h4 : dealing.dealInOrder = true)
  (h5 : dealing.playerHandSize = 13) :
  determineSamplingMethod dealing = SamplingMethod.Systematic :=
  sorry

end bridge_dealing_is_systematic_sampling_l1943_194391


namespace min_value_problem_l1943_194384

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) :
  (x + 1 / y) * (x + 1 / z) ≥ Real.sqrt 2 := by
  sorry

end min_value_problem_l1943_194384


namespace smallest_divisible_by_18_30_50_l1943_194357

theorem smallest_divisible_by_18_30_50 : 
  ∀ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 30 ∣ n ∧ 50 ∣ n → n ≥ 450 :=
by
  sorry

#check smallest_divisible_by_18_30_50

end smallest_divisible_by_18_30_50_l1943_194357


namespace interest_rate_calculation_l1943_194352

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℚ) : ℚ :=
  principal * time * rate / 100

theorem interest_rate_calculation (principal interest time : ℚ) 
  (h_principal : principal = 800)
  (h_interest : interest = 200)
  (h_time : time = 4) :
  ∃ (rate : ℚ), simple_interest principal time rate = interest ∧ rate = 25/4 :=
by sorry

end interest_rate_calculation_l1943_194352


namespace largest_quantity_l1943_194375

def D : ℚ := 2007 / 2006 + 2007 / 2008
def E : ℚ := 2008 / 2007 + 2010 / 2007
def F : ℚ := 2009 / 2008 + 2009 / 2010

theorem largest_quantity : E > D ∧ E > F := by
  sorry

end largest_quantity_l1943_194375


namespace units_digit_product_l1943_194396

theorem units_digit_product : (17 * 59 * 23) % 10 = 9 := by
  sorry

end units_digit_product_l1943_194396


namespace roses_cost_l1943_194346

theorem roses_cost (dozen : ℕ) (price_per_rose : ℚ) (discount_rate : ℚ) : 
  dozen * 12 * price_per_rose * discount_rate = 288 :=
by
  -- Assuming dozen = 5, price_per_rose = 6, and discount_rate = 0.8
  sorry

#check roses_cost

end roses_cost_l1943_194346


namespace find_unknown_number_l1943_194353

/-- Given two positive integers with known HCF and LCM, find the unknown number -/
theorem find_unknown_number (A B : ℕ+) (h1 : A = 24) 
  (h2 : Nat.gcd A B = 12) (h3 : Nat.lcm A B = 312) : B = 156 := by
  sorry

end find_unknown_number_l1943_194353


namespace boys_to_girls_ratio_l1943_194398

theorem boys_to_girls_ratio (S : ℚ) (G : ℚ) (B : ℚ) : 
  S > 0 → G > 0 → B > 0 →
  S = G + B →
  (1 / 2) * G = (1 / 5) * S →
  B / G = 3 / 2 := by
sorry

end boys_to_girls_ratio_l1943_194398


namespace number_ratio_l1943_194358

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 9) = 75) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l1943_194358


namespace triangle_side_ratio_l1943_194322

theorem triangle_side_ratio (a b c q : ℝ) : 
  c = b * q ∧ c = a * q^2 → ((Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2) :=
sorry

end triangle_side_ratio_l1943_194322


namespace pool_capacity_l1943_194347

/-- The capacity of a swimming pool given specific valve filling rates. -/
theorem pool_capacity 
  (fill_time : ℝ) 
  (valve_a_time : ℝ) 
  (valve_b_time : ℝ) 
  (valve_c_rate_diff : ℝ) 
  (valve_b_rate_diff : ℝ) 
  (h1 : fill_time = 40) 
  (h2 : valve_a_time = 180) 
  (h3 : valve_b_time = 240) 
  (h4 : valve_c_rate_diff = 75) 
  (h5 : valve_b_rate_diff = 60) : 
  ∃ T : ℝ, T = 16200 ∧ 
    T / fill_time = T / valve_a_time + T / valve_b_time + (T / valve_a_time + valve_c_rate_diff) :=
by sorry

end pool_capacity_l1943_194347


namespace belmont_basketball_winning_percentage_l1943_194338

theorem belmont_basketball_winning_percentage 
  (X : ℕ) (Y Z : ℝ) (h1 : 0 < Y) (h2 : Y < 100) (h3 : 0 < Z) (h4 : Z < 100) :
  let G := X * ((Y / 100) - (Z / 100)) / (Z / 100 - 1)
  ∃ (G : ℝ), (Z / 100) * (X + G) = (Y / 100) * X + G :=
by sorry

end belmont_basketball_winning_percentage_l1943_194338
