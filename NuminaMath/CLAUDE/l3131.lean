import Mathlib

namespace frog_climbs_out_l3131_313127

def well_depth : ℕ := 19
def day_climb : ℕ := 3
def night_slide : ℕ := 2

def days_to_climb (depth : ℕ) (day_climb : ℕ) (night_slide : ℕ) : ℕ :=
  (depth - day_climb) / (day_climb - night_slide) + 1

theorem frog_climbs_out : days_to_climb well_depth day_climb night_slide = 17 := by
  sorry

end frog_climbs_out_l3131_313127


namespace bobby_chocolate_pieces_l3131_313109

/-- The number of chocolate pieces Bobby ate -/
def chocolate_pieces (initial_candy pieces_more_candy total_pieces : ℕ) : ℕ :=
  total_pieces - (initial_candy + pieces_more_candy)

theorem bobby_chocolate_pieces :
  chocolate_pieces 33 4 51 = 14 := by
  sorry

end bobby_chocolate_pieces_l3131_313109


namespace seminar_attendees_l3131_313116

theorem seminar_attendees (total : ℕ) (a : ℕ) (h1 : total = 185) (h2 : a = 30) : 
  total - (a + 2*a + (a + 10) + ((a + 10) - 5)) = 20 := by
  sorry

end seminar_attendees_l3131_313116


namespace proportional_from_equality_l3131_313182

/-- Two real numbers are directly proportional if their ratio is constant -/
def DirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x = k * y

/-- Given x/3 = y/4, prove that x and y are directly proportional -/
theorem proportional_from_equality (x y : ℝ) (h : x / 3 = y / 4) :
  DirectlyProportional x y := by
  sorry

end proportional_from_equality_l3131_313182


namespace x_range_l3131_313134

theorem x_range (x : Real) 
  (h1 : -Real.pi/2 ≤ x ∧ x ≤ 3*Real.pi/2) 
  (h2 : Real.sqrt (1 + Real.sin (2*x)) = Real.sin x + Real.cos x) : 
  -Real.pi/4 ≤ x ∧ x ≤ 3*Real.pi/4 := by
  sorry

end x_range_l3131_313134


namespace replaced_crew_member_weight_l3131_313118

/-- Given a crew of 10 oarsmen, if replacing one member with a new member weighing 71 kg
    increases the average weight by 1.8 kg, then the replaced member weighed 53 kg. -/
theorem replaced_crew_member_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_increase : ℝ)
  (h_crew_size : n = 10)
  (h_new_weight : new_weight = 71)
  (h_avg_increase : avg_increase = 1.8) :
  let old_total := n * (avg_increase + (new_weight - 53) / n)
  let new_total := n * (avg_increase + new_weight / n)
  new_total - old_total = n * avg_increase :=
by sorry

end replaced_crew_member_weight_l3131_313118


namespace police_catch_thief_time_l3131_313170

/-- Proves that the time taken by the police to catch the thief is 2 hours -/
theorem police_catch_thief_time
  (thief_speed : ℝ)
  (police_station_distance : ℝ)
  (police_delay : ℝ)
  (police_speed : ℝ)
  (h1 : thief_speed = 20)
  (h2 : police_station_distance = 60)
  (h3 : police_delay = 1)
  (h4 : police_speed = 40)
  : ℝ :=
by
  sorry

#check police_catch_thief_time

end police_catch_thief_time_l3131_313170


namespace cream_fraction_after_mixing_l3131_313136

/-- Represents the contents of a cup -/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the mixing process -/
def mix_and_transfer (cup1 cup2 : CupContents) : (CupContents × CupContents) :=
  sorry

theorem cream_fraction_after_mixing :
  let initial_cup1 : CupContents := { coffee := 4, cream := 0 }
  let initial_cup2 : CupContents := { coffee := 0, cream := 4 }
  let (final_cup1, _) := mix_and_transfer initial_cup1 initial_cup2
  (final_cup1.cream / (final_cup1.coffee + final_cup1.cream)) = 2/5 := by
  sorry

end cream_fraction_after_mixing_l3131_313136


namespace T_not_subset_S_l3131_313173

def S : Set ℤ := {x | ∃ n : ℤ, x = 2 * n + 1}
def T : Set ℤ := {y | ∃ k : ℤ, y = 4 * k + 1}

theorem T_not_subset_S : ¬(T ⊆ S) := by
  sorry

end T_not_subset_S_l3131_313173


namespace unique_divisible_by_11_l3131_313175

/-- A number is divisible by 11 if the alternating sum of its digits is divisible by 11 -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  (n / 100 - (n / 10 % 10) + n % 10) % 11 = 0

/-- The set of three-digit numbers with units digit 5 and hundreds digit 6 -/
def validNumbers : Set ℕ :=
  {n : ℕ | 600 ≤ n ∧ n < 700 ∧ n % 10 = 5 ∧ n / 100 = 6}

theorem unique_divisible_by_11 :
  ∃! n : ℕ, n ∈ validNumbers ∧ isDivisibleBy11 n ∧ n = 605 := by
  sorry

#check unique_divisible_by_11

end unique_divisible_by_11_l3131_313175


namespace complex_addition_l3131_313132

theorem complex_addition (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + I) (h₂ : z₂ = 2 + 3*I) :
  z₁ + z₂ = 3 + 4*I := by sorry

end complex_addition_l3131_313132


namespace sum_reciprocal_squared_plus_one_ge_one_l3131_313171

theorem sum_reciprocal_squared_plus_one_ge_one (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := by
  sorry

end sum_reciprocal_squared_plus_one_ge_one_l3131_313171


namespace sequence_a_monotonicity_l3131_313193

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def sequence_a (x y : V) (n : ℕ) : ℝ := ‖x - n • y‖

theorem sequence_a_monotonicity (x y : V) (hx : x ≠ 0) (hy : y ≠ 0) :
  (∀ n : ℕ, sequence_a V x y n < sequence_a V x y (n + 1)) ↔
  (3 * ‖y‖ > 2 * ‖x‖ * ‖y‖⁻¹ * (inner x y)) ∧
  ¬(∀ n : ℕ, sequence_a V x y (n + 1) < sequence_a V x y n) :=
sorry

end sequence_a_monotonicity_l3131_313193


namespace sqrt_plus_square_zero_implies_sum_l3131_313185

theorem sqrt_plus_square_zero_implies_sum (x y : ℝ) :
  Real.sqrt (x - 1) + (y + 2)^2 = 0 → x + y = -1 := by
  sorry

end sqrt_plus_square_zero_implies_sum_l3131_313185


namespace systems_solution_l3131_313119

theorem systems_solution : ∃ (x y : ℝ), 
  (x - y = 1 ∧ 3*x + y = 11) ∧ 
  (3*x - 2*y = 5 ∧ 2*x + 3*y = 12) ∧
  (x = 3 ∧ y = 2) := by
  sorry

end systems_solution_l3131_313119


namespace function_not_in_first_quadrant_l3131_313181

/-- The function f(x) = (1/2)^x + m does not pass through the first quadrant if and only if m ≤ -1 -/
theorem function_not_in_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/2)^x + m ≤ 0) ↔ m ≤ -1 := by
sorry

end function_not_in_first_quadrant_l3131_313181


namespace small_boxes_in_big_box_l3131_313110

theorem small_boxes_in_big_box 
  (total_big_boxes : ℕ) 
  (candles_per_small_box : ℕ) 
  (total_candles : ℕ) 
  (h1 : total_big_boxes = 50)
  (h2 : candles_per_small_box = 40)
  (h3 : total_candles = 8000) :
  (total_candles / candles_per_small_box) / total_big_boxes = 4 := by
sorry

end small_boxes_in_big_box_l3131_313110


namespace root_sum_reciprocals_l3131_313131

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end root_sum_reciprocals_l3131_313131


namespace janice_pebbles_l3131_313154

/-- The number of pebbles each friend received -/
def pebbles_per_friend : ℕ := 4

/-- The number of friends who received pebbles -/
def number_of_friends : ℕ := 9

/-- The total number of pebbles Janice gave away -/
def total_pebbles : ℕ := pebbles_per_friend * number_of_friends

theorem janice_pebbles : total_pebbles = 36 := by
  sorry

end janice_pebbles_l3131_313154


namespace total_cost_is_2250_l3131_313144

def apple_quantity : ℕ := 8
def apple_price : ℕ := 70
def mango_quantity : ℕ := 9
def mango_price : ℕ := 55
def orange_quantity : ℕ := 5
def orange_price : ℕ := 40
def banana_quantity : ℕ := 12
def banana_price : ℕ := 30
def grape_quantity : ℕ := 7
def grape_price : ℕ := 45
def cherry_quantity : ℕ := 4
def cherry_price : ℕ := 80

def total_cost : ℕ := 
  apple_quantity * apple_price + 
  mango_quantity * mango_price + 
  orange_quantity * orange_price + 
  banana_quantity * banana_price + 
  grape_quantity * grape_price + 
  cherry_quantity * cherry_price

theorem total_cost_is_2250 : total_cost = 2250 := by
  sorry

end total_cost_is_2250_l3131_313144


namespace apples_used_for_pie_l3131_313194

theorem apples_used_for_pie (initial_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : initial_apples = 19) 
  (h2 : remaining_apples = 4) : 
  initial_apples - remaining_apples = 15 := by
  sorry

end apples_used_for_pie_l3131_313194


namespace polygon_area_is_7_5_l3131_313178

/-- Calculates the area of a polygon using the Shoelace formula -/
def polygonArea (vertices : List (ℝ × ℝ)) : ℝ :=
  let n := vertices.length
  let pairs := List.zip vertices (vertices.rotate 1)
  0.5 * (pairs.foldl (fun sum (v1, v2) => sum + v1.1 * v2.2 - v1.2 * v2.1) 0)

theorem polygon_area_is_7_5 :
  let vertices := [(2, 1), (4, 3), (7, 1), (4, 6)]
  polygonArea vertices = 7.5 := by
  sorry

#eval polygonArea [(2, 1), (4, 3), (7, 1), (4, 6)]

end polygon_area_is_7_5_l3131_313178


namespace cookies_eaten_total_l3131_313107

theorem cookies_eaten_total (charlie_cookies father_cookies mother_cookies : ℕ) 
  (h1 : charlie_cookies = 15)
  (h2 : father_cookies = 10)
  (h3 : mother_cookies = 5) :
  charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end cookies_eaten_total_l3131_313107


namespace expression_value_for_2016_l3131_313174

theorem expression_value_for_2016 :
  let x : ℤ := 2016
  (x^2 - x) - (x^2 - 2*x + 1) = 2015 :=
by sorry

end expression_value_for_2016_l3131_313174


namespace harold_bought_four_coffees_l3131_313145

/-- The cost of items bought on two different days --/
structure PurchaseData where
  doughnut_cost : ℚ
  harold_total : ℚ
  harold_doughnuts : ℕ
  melinda_total : ℚ
  melinda_doughnuts : ℕ
  melinda_coffees : ℕ

/-- Calculate the number of coffees Harold bought --/
def calculate_harold_coffees (data : PurchaseData) : ℕ :=
  sorry

/-- Theorem stating that Harold bought 4 coffees --/
theorem harold_bought_four_coffees (data : PurchaseData) 
  (h1 : data.doughnut_cost = 45/100)
  (h2 : data.harold_total = 491/100)
  (h3 : data.harold_doughnuts = 3)
  (h4 : data.melinda_total = 759/100)
  (h5 : data.melinda_doughnuts = 5)
  (h6 : data.melinda_coffees = 6) :
  calculate_harold_coffees data = 4 := by
    sorry

end harold_bought_four_coffees_l3131_313145


namespace factorization_equality_l3131_313157

theorem factorization_equality (x y : ℝ) : x^2 + x*y + x = x*(x + y + 1) := by
  sorry

end factorization_equality_l3131_313157


namespace student_number_problem_l3131_313141

theorem student_number_problem (x : ℤ) : x = 48 ↔ 5 * x - 138 = 102 := by
  sorry

end student_number_problem_l3131_313141


namespace complex_simplification_l3131_313106

theorem complex_simplification :
  ((-5 - 3*Complex.I) - (2 - 7*Complex.I)) * 2 = -14 + 8*Complex.I :=
by sorry

end complex_simplification_l3131_313106


namespace bacon_count_l3131_313166

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 228

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 23

/-- The difference between students suggesting bacon and tomatoes -/
def bacon_tomato_diff : ℕ := 314

/-- The number of students who suggested bacon -/
def bacon : ℕ := tomatoes + bacon_tomato_diff

theorem bacon_count : bacon = 337 := by
  sorry

end bacon_count_l3131_313166


namespace sin_cos_seven_eighths_pi_l3131_313104

theorem sin_cos_seven_eighths_pi : 
  Real.sin (7 * π / 8) * Real.cos (7 * π / 8) = - (Real.sqrt 2) / 4 := by
  sorry

end sin_cos_seven_eighths_pi_l3131_313104


namespace museum_trip_total_people_l3131_313177

theorem museum_trip_total_people : 
  let first_bus : ℕ := 12
  let second_bus : ℕ := 2 * first_bus
  let third_bus : ℕ := second_bus - 6
  let fourth_bus : ℕ := first_bus + 9
  first_bus + second_bus + third_bus + fourth_bus = 75
  := by sorry

end museum_trip_total_people_l3131_313177


namespace cube_equality_iff_three_l3131_313105

theorem cube_equality_iff_three (x : ℝ) (hx : x ≠ 0) :
  (3 * x)^3 = (9 * x)^2 ↔ x = 3 := by
  sorry

end cube_equality_iff_three_l3131_313105


namespace cosine_largest_angle_bound_l3131_313186

/-- Represents a sequence of non-degenerate triangles -/
def TriangleSequence := ℕ → (ℝ × ℝ × ℝ)

/-- Conditions for a valid triangle sequence -/
def IsValidTriangleSequence (seq : TriangleSequence) : Prop :=
  ∀ n : ℕ, let (a, b, c) := seq n
    0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b > c

/-- Sum of the shortest sides of the triangles -/
noncomputable def SumShortestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).1

/-- Sum of the second longest sides of the triangles -/
noncomputable def SumSecondLongestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).2.1

/-- Sum of the longest sides of the triangles -/
noncomputable def SumLongestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).2.2

/-- Cosine of the largest angle of the resultant triangle -/
noncomputable def CosLargestAngle (seq : TriangleSequence) : ℝ :=
  let A := SumShortestSides seq
  let B := SumSecondLongestSides seq
  let C := SumLongestSides seq
  (A^2 + B^2 - C^2) / (2 * A * B)

/-- The main theorem stating that the cosine of the largest angle is bounded below by 1 - √2 -/
theorem cosine_largest_angle_bound (seq : TriangleSequence) 
  (h : IsValidTriangleSequence seq) : 
  CosLargestAngle seq ≥ 1 - Real.sqrt 2 := by
  sorry

end cosine_largest_angle_bound_l3131_313186


namespace pasta_preference_ratio_l3131_313117

theorem pasta_preference_ratio : 
  ∀ (total spaghetti manicotti : ℕ),
    total = 800 →
    spaghetti = 320 →
    manicotti = 160 →
    (spaghetti : ℚ) / (manicotti : ℚ) = 2 := by
  sorry

end pasta_preference_ratio_l3131_313117


namespace tangent_angle_range_l3131_313172

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 2

noncomputable def α (x : ℝ) : ℝ := Real.arctan (3 * x^2 - 1)

theorem tangent_angle_range :
  ∀ x : ℝ, α x ∈ Set.Icc 0 (Real.pi / 2) ∪ Set.Icc (3 * Real.pi / 4) Real.pi :=
by sorry

end tangent_angle_range_l3131_313172


namespace min_value_theorem_l3131_313150

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 := by
  sorry

end min_value_theorem_l3131_313150


namespace polygon_sides_l3131_313196

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) = 3 * 360 → n = 8 := by
  sorry

end polygon_sides_l3131_313196


namespace trig_power_sum_l3131_313126

theorem trig_power_sum (x : Real) 
  (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11/36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41/216 := by
  sorry

end trig_power_sum_l3131_313126


namespace solution_set_part1_value_of_a_part2_l3131_313128

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | |x - 2| ≥ 7 - |x - 1|} = {x : ℝ | x ≤ -2 ∨ x ≥ 5} :=
sorry

-- Part 2
theorem value_of_a_part2 (a : ℝ) :
  {x : ℝ | |x - a| ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} → a = 1 :=
sorry

end solution_set_part1_value_of_a_part2_l3131_313128


namespace parallel_vectors_sum_magnitude_l3131_313183

/-- Given vectors a and b that are parallel, prove that their sum has magnitude √5 -/
theorem parallel_vectors_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -4]
  (∃ (k : ℝ), a = k • b) → 
  ‖a + b‖ = Real.sqrt 5 := by sorry

end parallel_vectors_sum_magnitude_l3131_313183


namespace average_difference_l3131_313199

-- Define the number of students and teachers
def num_students : ℕ := 120
def num_teachers : ℕ := 6

-- Define the class enrollments
def class_enrollments : List ℕ := [40, 40, 20, 10, 5, 5]

-- Define t (average number of students per teacher)
def t : ℚ := (num_students : ℚ) / num_teachers

-- Define s (average number of students per student)
def s : ℚ := (List.sum (List.map (fun x => x * x) class_enrollments) : ℚ) / num_students

-- Theorem to prove
theorem average_difference : t - s = -11.25 := by
  sorry

end average_difference_l3131_313199


namespace functional_equation_solution_l3131_313162

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (k : ℝ) :
  ∀ f : RealFunction, 
    (∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) →
    (∀ x : ℝ, f x = 0) := by
  sorry

end functional_equation_solution_l3131_313162


namespace volume_of_sphere_wedge_l3131_313138

/-- The volume of a wedge of a sphere -/
theorem volume_of_sphere_wedge (c : ℝ) (h : c = 12 * Real.pi) :
  let r := c / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let wedge_volume := sphere_volume / 4
  wedge_volume = 72 * Real.pi := by
sorry


end volume_of_sphere_wedge_l3131_313138


namespace not_always_greater_slope_for_greater_angle_not_always_inclination_equals_arctan_slope_not_different_angles_for_equal_slopes_l3131_313188

-- Define a straight line
structure Line where
  slope : ℝ
  inclination_angle : ℝ

-- Statement 1
theorem not_always_greater_slope_for_greater_angle : 
  ¬ ∀ (l1 l2 : Line), l1.inclination_angle > l2.inclination_angle → l1.slope > l2.slope :=
sorry

-- Statement 2
theorem not_always_inclination_equals_arctan_slope :
  ¬ ∀ (l : Line), l.slope = Real.tan l.inclination_angle → l.inclination_angle = Real.arctan l.slope :=
sorry

-- Statement 3
theorem not_different_angles_for_equal_slopes :
  ¬ ∃ (l1 l2 : Line), l1.slope = l2.slope ∧ l1.inclination_angle ≠ l2.inclination_angle :=
sorry

end not_always_greater_slope_for_greater_angle_not_always_inclination_equals_arctan_slope_not_different_angles_for_equal_slopes_l3131_313188


namespace cube_volume_from_diagonal_l3131_313176

/-- The volume of a cube with a given space diagonal -/
theorem cube_volume_from_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 3) : 
  let s := d / Real.sqrt 3
  s ^ 3 = 125 := by sorry

end cube_volume_from_diagonal_l3131_313176


namespace abs_inequality_solution_l3131_313169

theorem abs_inequality_solution (x : ℝ) : 
  |x + 3| - |2*x - 1| < x/2 + 1 ↔ x < -2/5 ∨ x > 2 := by sorry

end abs_inequality_solution_l3131_313169


namespace partial_fraction_decomposition_l3131_313197

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ), A = 16 ∧ B = 4 ∧ C = -16 ∧
  ∀ (x : ℝ), x ≠ 2 → x ≠ 4 →
    8 * x^2 / ((x - 4) * (x - 2)^3) = A / (x - 4) + B / (x - 2) + C / (x - 2)^3 := by
  sorry

end partial_fraction_decomposition_l3131_313197


namespace camping_problem_l3131_313152

theorem camping_problem (p m s : ℝ) 
  (h1 : s + m = p + 20)  -- Peter's indirect route equation
  (h2 : s + p = m + 16)  -- Michael's indirect route equation
  : s = 18 ∧ m = p + 2 := by
  sorry

end camping_problem_l3131_313152


namespace trig_simplification_l3131_313129

theorem trig_simplification :
  1 / Real.sin (70 * π / 180) - Real.sqrt 3 / Real.cos (70 * π / 180) = -4 := by
  sorry

end trig_simplification_l3131_313129


namespace sad_girls_count_l3131_313111

theorem sad_girls_count (total_children happy_children sad_children neutral_children
                         boys girls happy_boys neutral_boys : ℕ)
                        (h1 : total_children = 60)
                        (h2 : happy_children = 30)
                        (h3 : sad_children = 10)
                        (h4 : neutral_children = 20)
                        (h5 : boys = 16)
                        (h6 : girls = 44)
                        (h7 : happy_boys = 6)
                        (h8 : neutral_boys = 4)
                        (h9 : total_children = happy_children + sad_children + neutral_children)
                        (h10 : total_children = boys + girls) :
  girls - (happy_children - happy_boys) - (neutral_children - neutral_boys) = 4 := by
  sorry

end sad_girls_count_l3131_313111


namespace soup_problem_solution_l3131_313121

/-- Represents the number of people a can of soup can feed -/
structure SoupCanCapacity where
  adults : Nat
  children : Nat

/-- Represents the problem scenario -/
structure SoupProblem where
  capacity : SoupCanCapacity
  totalCans : Nat
  childrenFed : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdultsFed (problem : SoupProblem) : Nat :=
  let cansForChildren := (problem.childrenFed + problem.capacity.children - 1) / problem.capacity.children
  let remainingCans := problem.totalCans - cansForChildren
  remainingCans * problem.capacity.adults

/-- Theorem stating the problem and its solution -/
theorem soup_problem_solution (problem : SoupProblem)
  (h1 : problem.capacity = ⟨4, 6⟩)
  (h2 : problem.totalCans = 7)
  (h3 : problem.childrenFed = 18) :
  remainingAdultsFed problem = 16 := by
  sorry

end soup_problem_solution_l3131_313121


namespace position_2018_in_spiral_100_l3131_313102

/-- Represents a position in the matrix -/
structure Position where
  i : Nat
  j : Nat

/-- Constructs a spiral matrix of size n x n -/
def spiralMatrix (n : Nat) : Matrix (Fin n) (Fin n) Nat :=
  sorry

/-- Returns the position of a given number in the spiral matrix -/
def findPosition (n : Nat) (num : Nat) : Position :=
  sorry

/-- Theorem stating that 2018 is at position (34, 95) in a 100x100 spiral matrix -/
theorem position_2018_in_spiral_100 :
  findPosition 100 2018 = Position.mk 34 95 := by
  sorry

end position_2018_in_spiral_100_l3131_313102


namespace transformation_has_integer_root_intermediate_l3131_313151

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has integer roots -/
def has_integer_root (eq : QuadraticEquation) : Prop :=
  ∃ x : ℤ, eq.a * x^2 + eq.b * x + eq.c = 0

/-- Represents a single step in the transformation process -/
inductive TransformationStep
  | IncreaseP
  | DecreaseP
  | IncreaseQ
  | DecreaseQ

/-- Applies a transformation step to a quadratic equation -/
def apply_step (eq : QuadraticEquation) (step : TransformationStep) : QuadraticEquation :=
  match step with
  | TransformationStep.IncreaseP => ⟨eq.a, eq.b + 1, eq.c⟩
  | TransformationStep.DecreaseP => ⟨eq.a, eq.b - 1, eq.c⟩
  | TransformationStep.IncreaseQ => ⟨eq.a, eq.b, eq.c + 1⟩
  | TransformationStep.DecreaseQ => ⟨eq.a, eq.b, eq.c - 1⟩

theorem transformation_has_integer_root_intermediate 
  (initial : QuadraticEquation) 
  (final : QuadraticEquation) 
  (h_initial : initial = ⟨1, -2013, -13⟩) 
  (h_final : final = ⟨1, 13, 2013⟩) :
  ∀ steps : List TransformationStep, 
    (List.foldl apply_step initial steps = final) → 
    (∃ intermediate : QuadraticEquation, 
      intermediate ∈ List.scanl apply_step initial steps ∧ 
      has_integer_root intermediate) :=
sorry

end transformation_has_integer_root_intermediate_l3131_313151


namespace acid_concentration_proof_l3131_313189

/-- Represents the composition of an acid-water mixture -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The original mixture before any additions -/
def original_mixture : Mixture :=
  { acid := 0,  -- We don't know the initial acid amount
    water := 0 } -- We don't know the initial water amount

theorem acid_concentration_proof :
  -- The total volume of the original mixture is 10 ounces
  original_mixture.acid + original_mixture.water = 10 →
  -- After adding 1 ounce of water, the acid concentration becomes 25%
  original_mixture.acid / (original_mixture.acid + original_mixture.water + 1) = 1/4 →
  -- After adding 1 ounce of acid to the water-added mixture, the concentration becomes 40%
  (original_mixture.acid + 1) / (original_mixture.acid + original_mixture.water + 2) = 2/5 →
  -- Then the original acid concentration was 27.5%
  original_mixture.acid / (original_mixture.acid + original_mixture.water) = 11/40 :=
by sorry

end acid_concentration_proof_l3131_313189


namespace jenna_max_tanning_time_l3131_313140

/-- Represents Jenna's tanning schedule and calculates the maximum tanning time in a month. -/
def jennaTanningSchedule : ℕ :=
  let minutesPerDay : ℕ := 30
  let daysPerWeek : ℕ := 2
  let weeksFirstPeriod : ℕ := 2
  let minutesLastTwoWeeks : ℕ := 80
  
  let minutesFirstTwoWeeks := minutesPerDay * daysPerWeek * weeksFirstPeriod
  minutesFirstTwoWeeks + minutesLastTwoWeeks

/-- Proves that Jenna's maximum tanning time in a month is 200 minutes. -/
theorem jenna_max_tanning_time : jennaTanningSchedule = 200 := by
  sorry

end jenna_max_tanning_time_l3131_313140


namespace two_intersections_l3131_313137

/-- A line in a plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are identical -/
def identical (l1 l2 : Line) : Prop :=
  parallel l1 l2 ∧ l1.a * l2.c = l1.c * l2.a

/-- Check if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2) ∨ identical l1 l2

/-- The number of distinct intersection points of at least two lines -/
def num_intersections (lines : List Line) : ℕ :=
  sorry

/-- The three lines from the problem -/
def line1 : Line := ⟨3, 2, 4⟩
def line2 : Line := ⟨-1, 3, 3⟩
def line3 : Line := ⟨6, -4, 8⟩

/-- The main theorem -/
theorem two_intersections :
  num_intersections [line1, line2, line3] = 2 := by sorry

end two_intersections_l3131_313137


namespace line_direction_vector_l3131_313142

/-- Given a line passing through points (-5, 0) and (-2, 2), if its direction vector
    is of the form (2, b), then b = 4/3 -/
theorem line_direction_vector (b : ℚ) : 
  let p1 : ℚ × ℚ := (-5, 0)
  let p2 : ℚ × ℚ := (-2, 2)
  let dir : ℚ × ℚ := (2, b)
  (∃ (k : ℚ), k • (p2.1 - p1.1, p2.2 - p1.2) = dir) → b = 4/3 := by
sorry

end line_direction_vector_l3131_313142


namespace negation_of_universal_proposition_l3131_313120

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 1 < 0) ↔ (∃ x : ℝ, x^2 - 1 ≥ 0) := by sorry

end negation_of_universal_proposition_l3131_313120


namespace triangle_nested_calc_l3131_313143

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a^2 - 2*b

-- State the theorem
theorem triangle_nested_calc : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end triangle_nested_calc_l3131_313143


namespace monotone_function_implies_increasing_sequence_but_not_converse_l3131_313122

theorem monotone_function_implies_increasing_sequence_but_not_converse 
  (f : ℝ → ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = f n) :
  (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y) →
  (∀ n m, 1 ≤ n ∧ n ≤ m → a n ≤ a m) ∧
  ¬ ((∀ n m, 1 ≤ n ∧ n ≤ m → a n ≤ a m) →
     (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y)) :=
by sorry

end monotone_function_implies_increasing_sequence_but_not_converse_l3131_313122


namespace sand_art_calculation_l3131_313164

/-- The amount of sand needed to fill shapes given their dimensions and sand density. -/
theorem sand_art_calculation (rectangle_length : ℝ) (rectangle_area : ℝ) (square_side : ℝ) (sand_density : ℝ) : 
  rectangle_length = 7 →
  rectangle_area = 42 →
  square_side = 5 →
  sand_density = 3 →
  rectangle_area * sand_density + square_side * square_side * sand_density = 201 := by
  sorry

#check sand_art_calculation

end sand_art_calculation_l3131_313164


namespace triangle_area_l3131_313114

theorem triangle_area (a b : ℝ) (θ : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : θ = π / 3) :
  (1 / 2) * a * b * Real.sin θ = (3 / 2) * Real.sqrt 3 := by
  sorry

end triangle_area_l3131_313114


namespace power_relations_l3131_313167

/-- Given real numbers a, b, c, d satisfying certain conditions, 
    prove statements about their powers. -/
theorem power_relations (a b c d : ℝ) 
    (sum_eq : a + b = c + d) 
    (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
    (a^5 + b^5 = c^5 + d^5) ∧ 
    ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
  sorry


end power_relations_l3131_313167


namespace fruit_draw_ways_l3131_313195

/-- The number of fruits in the basket -/
def num_fruits : ℕ := 5

/-- The number of draws -/
def num_draws : ℕ := 2

/-- The number of ways to draw a fruit twice from a basket of 5 distinct fruits, considering the order -/
def num_ways : ℕ := num_fruits * (num_fruits - 1)

theorem fruit_draw_ways :
  num_ways = 20 :=
by sorry

end fruit_draw_ways_l3131_313195


namespace possible_no_snorers_in_sample_l3131_313113

-- Define the types for our problem
def Person : Type := Unit
def HasHeartDisease (p : Person) : Prop := sorry
def Snores (p : Person) : Prop := sorry

-- Define correlation and confidence
def Correlation (A B : Person → Prop) : Prop := sorry
def ConfidenceLevel : ℝ := sorry

-- State the theorem
theorem possible_no_snorers_in_sample 
  (corr : Correlation HasHeartDisease Snores)
  (conf : ConfidenceLevel > 0.99)
  : ∃ (sample : Finset Person), 
    (Finset.card sample = 100) ∧ 
    (∀ p ∈ sample, HasHeartDisease p) ∧
    (∀ p ∈ sample, ¬Snores p) :=
sorry

end possible_no_snorers_in_sample_l3131_313113


namespace exponential_function_point_l3131_313148

theorem exponential_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a^(1 - 1) - 2 = -1 := by sorry

end exponential_function_point_l3131_313148


namespace inequality_solution_set_l3131_313133

-- Define the inequality function
def f (x : ℝ) : ℝ := -x^2 - x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end inequality_solution_set_l3131_313133


namespace inserted_numbers_in_arithmetic_sequence_l3131_313156

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  λ i => a₁ + d * i.val

theorem inserted_numbers_in_arithmetic_sequence :
  let n : ℕ := 8
  let a₁ : ℝ := 8
  let aₙ : ℝ := 36
  let d : ℝ := (aₙ - a₁) / (n - 1)
  let seq := arithmetic_sequence a₁ d n
  (seq 1 = 12) ∧
  (seq 2 = 16) ∧
  (seq 3 = 20) ∧
  (seq 4 = 24) ∧
  (seq 5 = 28) ∧
  (seq 6 = 32) :=
by sorry

end inserted_numbers_in_arithmetic_sequence_l3131_313156


namespace imaginary_part_of_z_l3131_313165

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 3 - 6 * Complex.I) : 
  z.im = -3 := by
  sorry

end imaginary_part_of_z_l3131_313165


namespace largest_sphere_in_folded_rectangle_l3131_313168

/-- Represents a rectangle ABCD folded into a tetrahedron D-ABC -/
structure FoldedRectangle where
  ab : ℝ
  bc : ℝ
  d_projects_on_ab : Bool

/-- The radius of the largest inscribed sphere in the tetrahedron formed by folding the rectangle -/
def largest_inscribed_sphere_radius (r : FoldedRectangle) : ℝ := 
  sorry

/-- Theorem stating that for a rectangle with AB = 4 and BC = 3, folded into a tetrahedron
    where D projects onto AB, the radius of the largest inscribed sphere is 3/2 -/
theorem largest_sphere_in_folded_rectangle :
  ∀ (r : FoldedRectangle), 
    r.ab = 4 ∧ r.bc = 3 ∧ r.d_projects_on_ab = true →
    largest_inscribed_sphere_radius r = 3/2 := by
  sorry

end largest_sphere_in_folded_rectangle_l3131_313168


namespace equation_solution_l3131_313179

theorem equation_solution : ∃ x : ℝ, 4 * (4^x) + Real.sqrt (16 * (16^x)) = 32 ∧ x = 1 := by
  sorry

end equation_solution_l3131_313179


namespace boy_initial_height_l3131_313112

/-- Represents the growth rates and heights of a tree and a boy -/
structure GrowthProblem where
  initialTreeHeight : ℝ
  finalTreeHeight : ℝ
  finalBoyHeight : ℝ
  treeGrowthRate : ℝ
  boyGrowthRate : ℝ

/-- Theorem stating the boy's initial height given the growth problem parameters -/
theorem boy_initial_height (p : GrowthProblem)
  (h1 : p.initialTreeHeight = 16)
  (h2 : p.finalTreeHeight = 40)
  (h3 : p.finalBoyHeight = 36)
  (h4 : p.treeGrowthRate = 2 * p.boyGrowthRate) :
  p.finalBoyHeight - (p.finalTreeHeight - p.initialTreeHeight) / 2 = 24 := by
  sorry

end boy_initial_height_l3131_313112


namespace correlation_relationships_l3131_313123

/-- Represents a relationship between two variables -/
structure Relationship where
  variable1 : String
  variable2 : String

/-- Determines if a relationship represents a correlation -/
def is_correlation (r : Relationship) : Prop :=
  match r with
  | ⟨"snowfall", "traffic accidents"⟩ => True
  | ⟨"brain capacity", "intelligence"⟩ => True
  | ⟨"age", "weight"⟩ => False
  | ⟨"rainfall", "crop yield"⟩ => True
  | _ => False

/-- The main theorem stating which relationships represent correlations -/
theorem correlation_relationships :
  let r1 : Relationship := ⟨"snowfall", "traffic accidents"⟩
  let r2 : Relationship := ⟨"brain capacity", "intelligence"⟩
  let r3 : Relationship := ⟨"age", "weight"⟩
  let r4 : Relationship := ⟨"rainfall", "crop yield"⟩
  is_correlation r1 ∧ is_correlation r2 ∧ ¬is_correlation r3 ∧ is_correlation r4 :=
by sorry


end correlation_relationships_l3131_313123


namespace point_B_coordinate_l3131_313139

/-- Given two points A and B on a number line, where A is 3 units to the left of the origin
    and the distance between A and B is 1, prove that the coordinate of B is either -4 or -2. -/
theorem point_B_coordinate (A B : ℝ) : 
  A = -3 → abs (B - A) = 1 → (B = -4 ∨ B = -2) := by sorry

end point_B_coordinate_l3131_313139


namespace intersection_of_A_and_B_l3131_313161

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l3131_313161


namespace magenta_opposite_cyan_l3131_313160

-- Define the colors
inductive Color
| Yellow
| Orange
| Blue
| Cyan
| Magenta
| Black

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the property of opposite faces
def opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  (f1.val + f2.val) % 6 = 3

-- Define the given conditions
def cube_conditions (c : Cube) : Prop :=
  ∃ (top front right : Fin 6),
    c.faces top = Color.Cyan ∧
    c.faces right = Color.Blue ∧
    (c.faces front = Color.Yellow ∨ c.faces front = Color.Orange ∨ c.faces front = Color.Black)

-- Theorem statement
theorem magenta_opposite_cyan (c : Cube) :
  cube_conditions c →
  ∃ (magenta_face cyan_face : Fin 6),
    c.faces magenta_face = Color.Magenta ∧
    c.faces cyan_face = Color.Cyan ∧
    opposite c magenta_face cyan_face :=
by sorry

end magenta_opposite_cyan_l3131_313160


namespace remaining_amount_l3131_313190

def initial_amount : ℚ := 343
def fraction_given : ℚ := 1/7
def num_recipients : ℕ := 2

theorem remaining_amount :
  initial_amount - (fraction_given * initial_amount * num_recipients) = 245 :=
by sorry

end remaining_amount_l3131_313190


namespace compound_molecular_weight_l3131_313130

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecularWeight (carbonAtoms hydrogenAtoms oxygenAtoms : ℕ) 
  (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  (carbonAtoms : ℝ) * carbonWeight + 
  (hydrogenAtoms : ℝ) * hydrogenWeight + 
  (oxygenAtoms : ℝ) * oxygenWeight

/-- The molecular weight of the given compound is approximately 58.078 g/mol -/
theorem compound_molecular_weight :
  let carbonAtoms : ℕ := 3
  let hydrogenAtoms : ℕ := 6
  let oxygenAtoms : ℕ := 1
  let carbonWeight : ℝ := 12.01
  let hydrogenWeight : ℝ := 1.008
  let oxygenWeight : ℝ := 16.00
  abs (molecularWeight carbonAtoms hydrogenAtoms oxygenAtoms 
    carbonWeight hydrogenWeight oxygenWeight - 58.078) < 0.001 := by
  sorry

end compound_molecular_weight_l3131_313130


namespace son_age_l3131_313163

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end son_age_l3131_313163


namespace company_employees_l3131_313149

theorem company_employees (december_employees : ℕ) (increase_percentage : ℚ) : 
  december_employees = 470 →
  increase_percentage = 15 / 100 →
  ∃ (january_employees : ℕ), 
    (↑december_employees : ℚ) = (1 + increase_percentage) * january_employees ∧
    january_employees = 409 := by
  sorry

end company_employees_l3131_313149


namespace bug_path_tiles_l3131_313191

-- Define the garden dimensions
def width : ℕ := 12
def length : ℕ := 18

-- Define the function to calculate the number of tiles visited
def tilesVisited (w l : ℕ) : ℕ := w + l - Nat.gcd w l

-- Theorem statement
theorem bug_path_tiles :
  tilesVisited width length = 24 :=
sorry

end bug_path_tiles_l3131_313191


namespace cheryl_craft_project_cheryl_material_ratio_l3131_313103

/-- The total amount of material used in Cheryl's craft project --/
def total_used (bought_A bought_B bought_C leftover_A leftover_B leftover_C : ℚ) : ℚ :=
  (bought_A - leftover_A) + (bought_B - leftover_B) + (bought_C - leftover_C)

/-- Theorem stating the total amount of material used in Cheryl's craft project --/
theorem cheryl_craft_project :
  let bought_A : ℚ := 5/8
  let bought_B : ℚ := 2/9
  let bought_C : ℚ := 2/5
  let leftover_A : ℚ := 1/12
  let leftover_B : ℚ := 5/36
  let leftover_C : ℚ := 1/10
  total_used bought_A bought_B bought_C leftover_A leftover_B leftover_C = 37/40 :=
by
  sorry

/-- The ratio of materials used in Cheryl's craft project --/
def material_ratio (used_A used_B used_C : ℚ) : Prop :=
  2 * used_B = used_A ∧ 3 * used_B = used_C

/-- Theorem stating the ratio of materials used in Cheryl's craft project --/
theorem cheryl_material_ratio :
  let bought_A : ℚ := 5/8
  let bought_B : ℚ := 2/9
  let bought_C : ℚ := 2/5
  let leftover_A : ℚ := 1/12
  let leftover_B : ℚ := 5/36
  let leftover_C : ℚ := 1/10
  let used_A : ℚ := bought_A - leftover_A
  let used_B : ℚ := bought_B - leftover_B
  let used_C : ℚ := bought_C - leftover_C
  material_ratio used_A used_B used_C :=
by
  sorry

end cheryl_craft_project_cheryl_material_ratio_l3131_313103


namespace urn_theorem_l3131_313155

/-- Represents the state of the urn -/
structure UrnState where
  black : ℕ
  white : ℕ

/-- Represents the four possible operations -/
inductive Operation
  | RemoveBlack
  | RemoveBlackWhite
  | RemoveBlackAddWhite
  | RemoveWhiteAddBlack

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.RemoveBlack => ⟨state.black - 1, state.white⟩
  | Operation.RemoveBlackWhite => ⟨state.black, state.white - 1⟩
  | Operation.RemoveBlackAddWhite => ⟨state.black - 1, state.white⟩
  | Operation.RemoveWhiteAddBlack => ⟨state.black + 1, state.white - 1⟩

/-- Checks if the given state is reachable from the initial state -/
def isReachable (initialState : UrnState) (targetState : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl applyOperation initialState (List.ofFn ops)) = targetState

/-- The theorem to be proven -/
theorem urn_theorem :
  let initialState : UrnState := ⟨150, 150⟩
  let targetState : UrnState := ⟨50, 50⟩
  isReachable initialState targetState :=
sorry

end urn_theorem_l3131_313155


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l3131_313146

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 5*x + 1 - 16
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 5) :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l3131_313146


namespace flax_acres_for_given_farm_l3131_313153

/-- Represents a farm with sunflowers and flax -/
structure Farm where
  total_acres : ℕ
  sunflower_excess : ℕ

/-- Calculates the number of acres of flax to be planted -/
def flax_acres (f : Farm) : ℕ :=
  (f.total_acres - f.sunflower_excess) / 2

theorem flax_acres_for_given_farm :
  let farm : Farm := { total_acres := 240, sunflower_excess := 80 }
  flax_acres farm = 80 := by
  sorry

#eval flax_acres { total_acres := 240, sunflower_excess := 80 }

end flax_acres_for_given_farm_l3131_313153


namespace system_solution_l3131_313158

theorem system_solution : ∃ (x y : ℝ), 2*x - 3*y = -7 ∧ 5*x + 4*y = -6 ∧ (x, y) = (-2, 1) := by
  sorry

end system_solution_l3131_313158


namespace uniform_price_calculation_l3131_313124

/-- Calculates the price of a uniform given the conditions of a servant's employment --/
def uniform_price (full_year_salary : ℚ) (actual_salary : ℚ) (months_worked : ℕ) : ℚ :=
  full_year_salary * (months_worked / 12) - actual_salary

theorem uniform_price_calculation :
  uniform_price 900 650 9 = 25 := by
  sorry

end uniform_price_calculation_l3131_313124


namespace smallest_five_digit_multiple_of_18_l3131_313135

theorem smallest_five_digit_multiple_of_18 : ∃ (n : ℕ), 
  (n = 10008) ∧ 
  (∃ (k : ℕ), n = 18 * k) ∧ 
  (n ≥ 10000) ∧ 
  (∀ (m : ℕ), (∃ (j : ℕ), m = 18 * j) → m ≥ 10000 → m ≥ n) :=
by sorry

end smallest_five_digit_multiple_of_18_l3131_313135


namespace card_distribution_theorem_l3131_313187

/-- Represents the state of card distribution among points -/
structure CardState (n : ℕ) where
  cards_at_A : Fin n → ℕ
  cards_at_O : ℕ

/-- Represents a move in the game -/
inductive Move (n : ℕ)
  | outer (i : Fin n) : Move n
  | inner : Move n

/-- Applies a move to a card state -/
def apply_move (n : ℕ) (state : CardState n) (move : Move n) : CardState n :=
  sorry

/-- Checks if a state is valid according to the game rules -/
def is_valid_state (n : ℕ) (state : CardState n) : Prop :=
  sorry

/-- Checks if a state is the goal state (all points have ≥ n+1 cards) -/
def is_goal_state (n : ℕ) (state : CardState n) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem card_distribution_theorem (n : ℕ) (h_n : n ≥ 3) (T : ℕ) (h_T : T ≥ n^2 + 3*n + 1)
  (initial_state : CardState n) (h_initial : is_valid_state n initial_state) :
  ∃ (moves : List (Move n)), 
    is_goal_state n (moves.foldl (apply_move n) initial_state) :=
  sorry

end card_distribution_theorem_l3131_313187


namespace students_in_both_clubs_l3131_313180

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (h1 : total_students = 400)
  (h2 : drama_club = 180)
  (h3 : science_club = 230)
  (h4 : either_club = 350) :
  drama_club + science_club - either_club = 60 :=
by sorry

end students_in_both_clubs_l3131_313180


namespace plus_signs_count_l3131_313108

theorem plus_signs_count (total : ℕ) (plus_count : ℕ) (minus_count : ℕ) :
  total = 23 →
  plus_count + minus_count = total →
  (∀ (subset : Finset ℕ), subset.card = 10 → (∃ (i : ℕ), i ∈ subset ∧ i < plus_count)) →
  (∀ (subset : Finset ℕ), subset.card = 15 → (∃ (i : ℕ), i ∈ subset ∧ plus_count ≤ i ∧ i < total)) →
  plus_count = 14 :=
by sorry

end plus_signs_count_l3131_313108


namespace tony_curl_weight_l3131_313101

/-- The weight Tony can lift in the curl exercise, in pounds -/
def curl_weight : ℝ := sorry

/-- The weight Tony can lift in the military press exercise, in pounds -/
def military_press_weight : ℝ := sorry

/-- The weight Tony can lift in the squat exercise, in pounds -/
def squat_weight : ℝ := sorry

/-- The relationship between curl weight and military press weight -/
axiom military_press_relation : military_press_weight = 2 * curl_weight

/-- The relationship between squat weight and military press weight -/
axiom squat_relation : squat_weight = 5 * military_press_weight

/-- The known weight Tony can lift in the squat exercise -/
axiom squat_known_weight : squat_weight = 900

theorem tony_curl_weight : curl_weight = 90 := by sorry

end tony_curl_weight_l3131_313101


namespace james_delivery_l3131_313198

/-- Calculates the number of bags delivered by James in a given number of days -/
def bags_delivered (bags_per_trip : ℕ) (trips_per_day : ℕ) (days : ℕ) : ℕ :=
  bags_per_trip * trips_per_day * days

/-- Theorem stating that James delivers 1000 bags in 5 days -/
theorem james_delivery : bags_delivered 10 20 5 = 1000 := by
  sorry

end james_delivery_l3131_313198


namespace p_squared_plus_26_composite_l3131_313184

theorem p_squared_plus_26_composite (p : Nat) (hp : Prime p) : 
  ∃ (a b : Nat), a > 1 ∧ b > 1 ∧ p^2 + 26 = a * b :=
sorry

end p_squared_plus_26_composite_l3131_313184


namespace heartsuit_three_eight_l3131_313159

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_three_eight_l3131_313159


namespace correct_ranking_count_l3131_313192

/-- Represents a team in the tournament -/
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

/-- Represents the outcome of a match -/
inductive MatchOutcome : Type
| Win : Team → MatchOutcome
| Draw : MatchOutcome

/-- Represents the final ranking of teams -/
def Ranking := List Team

/-- The structure of the tournament -/
structure Tournament :=
  (saturdayMatch1 : MatchOutcome)
  (saturdayMatch2 : MatchOutcome)
  (sundayMatch1Winner : Team)
  (sundayMatch2Winner : Team)

/-- Function to calculate the number of possible rankings -/
def countPossibleRankings : ℕ :=
  -- Implementation details omitted
  256

/-- Theorem stating that the number of possible rankings is 256 -/
theorem correct_ranking_count :
  countPossibleRankings = 256 := by sorry


end correct_ranking_count_l3131_313192


namespace M_bounds_l3131_313147

/-- Represents the minimum number of black points needed in an n × n square lattice
    so that every square path has at least one black point on it. -/
def M (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for M(n) in an n × n square lattice. -/
theorem M_bounds (n : ℕ) : (2 : ℝ) / 7 * (n - 1)^2 ≤ (M n : ℝ) ∧ (M n : ℝ) ≤ 2 / 7 * n^2 := by
  sorry

end M_bounds_l3131_313147


namespace parking_space_area_l3131_313125

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : ℝ
  unpainted_side : ℝ
  is_rectangular : length > 0 ∧ width > 0
  three_sides_painted : painted_sides_sum = 2 * width + length
  unpainted_is_length : unpainted_side = length

/-- The area of a parking space is equal to its length multiplied by its width -/
def area (p : ParkingSpace) : ℝ := p.length * p.width

/-- Theorem: If a rectangular parking space has an unpainted side of 9 feet
    and the sum of the painted sides is 37 feet, then its area is 126 square feet -/
theorem parking_space_area 
  (p : ParkingSpace) 
  (h1 : p.unpainted_side = 9) 
  (h2 : p.painted_sides_sum = 37) : 
  area p = 126 := by
  sorry

end parking_space_area_l3131_313125


namespace total_pizzas_ordered_l3131_313100

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of pizzas ordered for each size -/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of slices from a given order -/
def totalSlices (slices : PizzaSlices) (order : PizzaOrder) : Nat :=
  slices.small * order.small + slices.medium * order.medium + slices.large * order.large

/-- The main theorem to prove -/
theorem total_pizzas_ordered
  (slices : PizzaSlices)
  (order : PizzaOrder)
  (h1 : slices.small = 6)
  (h2 : slices.medium = 8)
  (h3 : slices.large = 12)
  (h4 : order.small = 4)
  (h5 : order.medium = 5)
  (h6 : totalSlices slices order = 136) :
  order.small + order.medium + order.large = 15 := by
  sorry


end total_pizzas_ordered_l3131_313100


namespace money_sum_is_fifty_l3131_313115

def jack_money : ℕ := 26

def ben_money (jack : ℕ) : ℕ := jack - 9

def eric_money (ben : ℕ) : ℕ := ben - 10

def total_money (jack ben eric : ℕ) : ℕ := jack + ben + eric

theorem money_sum_is_fifty :
  total_money jack_money (ben_money jack_money) (eric_money (ben_money jack_money)) = 50 := by
  sorry

end money_sum_is_fifty_l3131_313115
