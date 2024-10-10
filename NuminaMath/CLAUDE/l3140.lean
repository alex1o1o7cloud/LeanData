import Mathlib

namespace smallest_violet_balls_l3140_314040

theorem smallest_violet_balls (x : ℕ) (y : ℕ) : 
  x > 0 ∧ 
  x % 120 = 0 ∧ 
  x / 10 + x / 8 + x / 3 + (x / 10 + 9) + (x / 8 + 10) + 8 + y = x ∧
  y = x / 60 * 13 - 27 →
  y ≥ 25 :=
sorry

end smallest_violet_balls_l3140_314040


namespace weekly_allowance_calculation_l3140_314071

theorem weekly_allowance_calculation (weekly_allowance : ℚ) : 
  (4 * weekly_allowance / 2 * 3 / 4 = 15) → weekly_allowance = 10 := by
  sorry

end weekly_allowance_calculation_l3140_314071


namespace triangle_properties_l3140_314093

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Define the property that a^2 + b^2 - c^2 = ab -/
def satisfiesProperty (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = t.a * t.b

/-- Define the collinearity of vectors (2sin A, 1) and (cos C, 1/2) -/
def vectorsAreCollinear (t : Triangle) : Prop :=
  2 * Real.sin t.A * (1/2) = Real.cos t.C * 1

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesProperty t) 
  (h2 : vectorsAreCollinear t) : 
  t.C = π/3 ∧ t.B = π/2 := by
  sorry

end triangle_properties_l3140_314093


namespace combined_drying_time_l3140_314019

-- Define the driers' capacities and individual drying times
def drier1_capacity : ℚ := 1/2
def drier2_capacity : ℚ := 3/4
def drier3_capacity : ℚ := 1

def drier1_time : ℚ := 24
def drier2_time : ℚ := 2
def drier3_time : ℚ := 8

-- Define the combined drying rate
def combined_rate : ℚ := 
  drier1_capacity / drier1_time + 
  drier2_capacity / drier2_time + 
  drier3_capacity / drier3_time

-- Theorem statement
theorem combined_drying_time : 
  1 / combined_rate = 3/2 := by sorry

end combined_drying_time_l3140_314019


namespace right_triangle_inequality_l3140_314000

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c ≥ a ∧ c ≥ b) : 
  a + b ≤ c * Real.sqrt 2 ∧ (a + b = c * Real.sqrt 2 ↔ a = b) := by
  sorry

end right_triangle_inequality_l3140_314000


namespace student_excess_is_105_l3140_314053

/-- Represents the composition of a fourth-grade classroom -/
structure Classroom where
  students : Nat
  guinea_pigs : Nat
  teachers : Nat

/-- The number of fourth-grade classrooms -/
def num_classrooms : Nat := 5

/-- A fourth-grade classroom in Big Valley School -/
def big_valley_classroom : Classroom :=
  { students := 25, guinea_pigs := 3, teachers := 1 }

/-- Theorem: The number of students exceeds the total number of guinea pigs and teachers by 105 in all fourth-grade classrooms -/
theorem student_excess_is_105 : 
  (num_classrooms * big_valley_classroom.students) - 
  (num_classrooms * (big_valley_classroom.guinea_pigs + big_valley_classroom.teachers)) = 105 := by
  sorry

end student_excess_is_105_l3140_314053


namespace alyssa_picked_25_limes_l3140_314057

/-- The number of limes picked by Alyssa -/
def alyssas_limes : ℕ := 57 - 32

/-- The total number of limes picked -/
def total_limes : ℕ := 57

/-- The number of limes picked by Mike -/
def mikes_limes : ℕ := 32

theorem alyssa_picked_25_limes : alyssas_limes = 25 := by sorry

end alyssa_picked_25_limes_l3140_314057


namespace birthday_crayons_l3140_314055

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 134

/-- The number of crayons Paul lost or gave away -/
def crayons_lost : ℕ := 345

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost

theorem birthday_crayons : total_crayons = 479 := by
  sorry

end birthday_crayons_l3140_314055


namespace children_playing_both_sports_l3140_314028

/-- Given a class of children with the following properties:
  * The total number of children is 38
  * 19 children play tennis
  * 21 children play squash
  * 10 children play neither sport
  Then, the number of children who play both sports is 12 -/
theorem children_playing_both_sports
  (total : ℕ) (tennis : ℕ) (squash : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 38)
  (h2 : tennis = 19)
  (h3 : squash = 21)
  (h4 : neither = 10)
  (h5 : total = tennis + squash - both + neither) :
  both = 12 := by
sorry

end children_playing_both_sports_l3140_314028


namespace rectangle_other_vertices_y_sum_l3140_314097

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ
  vertex4 : ℝ × ℝ

/-- The property that two points are opposite vertices of a rectangle --/
def areOppositeVertices (p1 p2 : ℝ × ℝ) (r : Rectangle) : Prop :=
  (r.vertex1 = p1 ∧ r.vertex3 = p2) ∨ (r.vertex1 = p2 ∧ r.vertex3 = p1) ∨
  (r.vertex2 = p1 ∧ r.vertex4 = p2) ∨ (r.vertex2 = p2 ∧ r.vertex4 = p1)

/-- The sum of y-coordinates of two points --/
def sumYCoordinates (p1 p2 : ℝ × ℝ) : ℝ :=
  p1.2 + p2.2

theorem rectangle_other_vertices_y_sum 
  (r : Rectangle) 
  (h : areOppositeVertices (3, 17) (9, -4) r) : 
  ∃ (v1 v2 : ℝ × ℝ), 
    ((v1 = r.vertex2 ∧ v2 = r.vertex4) ∨ (v1 = r.vertex1 ∧ v2 = r.vertex3)) ∧
    sumYCoordinates v1 v2 = 13 := by
  sorry


end rectangle_other_vertices_y_sum_l3140_314097


namespace radical_simplification_l3140_314090

theorem radical_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (8 * q^3) = 6 * q^3 * Real.sqrt 10 := by
  sorry

end radical_simplification_l3140_314090


namespace greatest_a_value_l3140_314026

theorem greatest_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = 18) ∧ (a > 0) → a ≤ 17 := by
  sorry

end greatest_a_value_l3140_314026


namespace max_intersection_area_is_zero_l3140_314069

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a right prism with an equilateral triangle base -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ
  baseVertices : List Point3D

/-- Calculates the area of intersection between a plane and a right prism -/
def intersectionArea (prism : RightPrism) (plane : Plane) : ℝ :=
  sorry

/-- The theorem stating that the maximum area of intersection is 0 -/
theorem max_intersection_area_is_zero (h : ℝ) (s : ℝ) (A B C : Point3D) :
  h = 5 →
  s = 6 →
  A = ⟨3, 0, 0⟩ →
  B = ⟨-3, 0, 0⟩ →
  C = ⟨0, 3 * Real.sqrt 3, 0⟩ →
  let prism : RightPrism := ⟨h, s, [A, B, C]⟩
  let plane : Plane := ⟨2, -3, 6, 30⟩
  intersectionArea prism plane = 0 :=
by
  sorry

end max_intersection_area_is_zero_l3140_314069


namespace enhanced_computer_price_difference_l3140_314022

/-- The price difference between an enhanced computer and a basic computer -/
def price_difference (total_basic : ℝ) (price_basic : ℝ) : ℝ :=
  let price_printer := total_basic - price_basic
  let price_enhanced := 6 * price_printer
  price_enhanced - price_basic

/-- Theorem stating the price difference between enhanced and basic computers -/
theorem enhanced_computer_price_difference :
  price_difference 2500 2000 = 500 := by
  sorry

end enhanced_computer_price_difference_l3140_314022


namespace hypotenuse_length_l3140_314080

/-- Represents a right triangle with a 45° angle -/
structure RightTriangle45 where
  leg : ℝ
  hypotenuse : ℝ

/-- The hypotenuse of a right triangle with a 45° angle is √2 times the leg -/
axiom hypotenuse_formula (t : RightTriangle45) : t.hypotenuse = t.leg * Real.sqrt 2

/-- Theorem: In a right triangle with one leg of 10 inches and an opposite angle of 45°,
    the length of the hypotenuse is 10√2 inches -/
theorem hypotenuse_length : 
  let t : RightTriangle45 := { leg := 10, hypotenuse := 10 * Real.sqrt 2 }
  t.hypotenuse = 10 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l3140_314080


namespace arithmetic_sequence_ratio_l3140_314035

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Number of terms in an arithmetic sequence -/
def num_terms (a₁ : ℚ) (d : ℚ) (l : ℚ) : ℕ :=
  Nat.floor ((l - a₁) / d + 1)

theorem arithmetic_sequence_ratio :
  let n₁ := num_terms 4 2 40
  let n₂ := num_terms 5 5 75
  let sum₁ := arithmetic_sum 4 2 n₁
  let sum₂ := arithmetic_sum 5 5 n₂
  sum₁ / sum₂ = 209 / 300 := by sorry

end arithmetic_sequence_ratio_l3140_314035


namespace distance_on_line_l3140_314012

/-- The distance between two points on a line y = mx + k -/
theorem distance_on_line (m k a b c d : ℝ) 
  (h1 : b = m * a + k) 
  (h2 : d = m * c + k) : 
  Real.sqrt ((a - c)^2 + (b - d)^2) = |a - c| * Real.sqrt (1 + m^2) := by
  sorry

end distance_on_line_l3140_314012


namespace triple_sum_power_divisibility_l3140_314049

theorem triple_sum_power_divisibility (a b c : ℤ) (h : a + b + c = 0) :
  ∃ k : ℤ, a^1999 + b^1999 + c^1999 = 6 * k :=
by sorry

end triple_sum_power_divisibility_l3140_314049


namespace log_product_equation_l3140_314041

theorem log_product_equation (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 4 → x = 2401 := by
  sorry

end log_product_equation_l3140_314041


namespace first_five_terms_of_sequence_l3140_314088

def a (n : ℕ) : ℤ := (-1)^n + n

theorem first_five_terms_of_sequence :
  (List.range 5).map (fun i => a (i + 1)) = [0, 3, 2, 5, 4] := by
  sorry

end first_five_terms_of_sequence_l3140_314088


namespace special_integers_property_l3140_314050

/-- A function that reverses the hundreds and units digits of a three-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  units * 100 + tens * 10 + hundreds

/-- The theorem stating the property of the 90 special integers -/
theorem special_integers_property :
  ∃ (S : Finset ℕ), 
    Finset.card S = 90 ∧ 
    (∀ n ∈ S, 100 < n ∧ n < 1100) ∧
    (∀ n ∈ S, reverseDigits n = n + 99) := by
  sorry

#check special_integers_property

end special_integers_property_l3140_314050


namespace min_value_of_ab_min_value_is_6_plus_4sqrt2_l3140_314025

theorem min_value_of_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 2 = 2 * (a + b)) :
  ∀ x y : ℝ, x > 1 → y > 1 → x * y + 2 = 2 * (x + y) → a * b ≤ x * y :=
by sorry

theorem min_value_is_6_plus_4sqrt2 (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 2 = 2 * (a + b)) :
  a * b = 6 + 4 * Real.sqrt 2 :=
by sorry

end min_value_of_ab_min_value_is_6_plus_4sqrt2_l3140_314025


namespace quadratic_factorization_l3140_314048

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 4 * y^2 - 5 * y - 21 = (4 * y + a) * (y + b)) →
  a - b = 10 := by sorry

end quadratic_factorization_l3140_314048


namespace mod_equivalence_2023_l3140_314006

theorem mod_equivalence_2023 :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2023 [ZMOD 8] ∧ n = 1 := by
  sorry

end mod_equivalence_2023_l3140_314006


namespace k_bound_l3140_314038

/-- A sequence a_n defined as n^2 - kn for positive integers n -/
def a (k : ℝ) (n : ℕ) : ℝ := n^2 - k * n

/-- The property that a sequence is monotonically increasing -/
def MonotonicallyIncreasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) > f n

/-- Theorem: If the sequence a_n is monotonically increasing, then k < 3 -/
theorem k_bound (k : ℝ) (h : MonotonicallyIncreasing (a k)) : k < 3 := by
  sorry

end k_bound_l3140_314038


namespace mulch_cost_calculation_l3140_314084

-- Define the constants
def tons_of_mulch : ℝ := 3
def price_per_pound : ℝ := 2.5
def pounds_per_ton : ℝ := 2000

-- Define the theorem
theorem mulch_cost_calculation :
  tons_of_mulch * pounds_per_ton * price_per_pound = 15000 := by
  sorry

end mulch_cost_calculation_l3140_314084


namespace solution_interval_l3140_314004

theorem solution_interval (x₀ : ℝ) : 
  (1/2:ℝ)^x₀ = x₀^(1/3) → 1/3 < x₀ ∧ x₀ < 1/2 := by
  sorry

end solution_interval_l3140_314004


namespace grid_paths_count_l3140_314033

/-- Represents a grid of roads between two locations -/
structure Grid where
  north_paths : Nat
  east_paths : Nat

/-- Calculates the total number of paths in a grid -/
def total_paths (g : Grid) : Nat :=
  g.north_paths * g.east_paths

/-- Theorem stating that the total number of paths in the given grid is 15 -/
theorem grid_paths_count : 
  ∀ g : Grid, g.north_paths = 3 → g.east_paths = 5 → total_paths g = 15 := by
  sorry

end grid_paths_count_l3140_314033


namespace combined_grade4_percent_is_16_l3140_314047

/-- Represents the number of students in Pinegrove school -/
def pinegrove_students : ℕ := 120

/-- Represents the number of students in Maplewood school -/
def maplewood_students : ℕ := 180

/-- Represents the percentage of grade 4 students in Pinegrove school -/
def pinegrove_grade4_percent : ℚ := 10 / 100

/-- Represents the percentage of grade 4 students in Maplewood school -/
def maplewood_grade4_percent : ℚ := 20 / 100

/-- Represents the total number of students in both schools -/
def total_students : ℕ := pinegrove_students + maplewood_students

/-- Theorem stating that the percentage of grade 4 students in the combined schools is 16% -/
theorem combined_grade4_percent_is_16 : 
  (pinegrove_grade4_percent * pinegrove_students + maplewood_grade4_percent * maplewood_students) / total_students = 16 / 100 := by
  sorry

end combined_grade4_percent_is_16_l3140_314047


namespace consecutive_non_primes_l3140_314043

theorem consecutive_non_primes (n : ℕ) : ∃ (k : ℕ), ∀ (i : ℕ), i < n → ¬ Nat.Prime (k + i) := by
  sorry

end consecutive_non_primes_l3140_314043


namespace raviraj_cycling_journey_l3140_314001

/-- Raviraj's cycling journey --/
theorem raviraj_cycling_journey (initial_south distance_west_1 distance_north distance_west_2 distance_to_home : ℝ) :
  distance_west_1 = 10 ∧
  distance_north = 20 ∧
  distance_west_2 = 20 ∧
  distance_to_home = 30 ∧
  distance_west_1 + distance_west_2 = distance_to_home ∧
  initial_south + distance_north = distance_to_home →
  initial_south = 10 := by sorry

end raviraj_cycling_journey_l3140_314001


namespace ones_digit_of_8_to_40_l3140_314083

theorem ones_digit_of_8_to_40 (cycle : List Nat) (h_cycle : cycle = [8, 4, 2, 6]) :
  (8^40 : ℕ) % 10 = 6 := by
  sorry

end ones_digit_of_8_to_40_l3140_314083


namespace bowling_team_size_l3140_314060

theorem bowling_team_size (n : ℕ) (original_avg : ℝ) (new_avg : ℝ) 
  (new_player1_weight : ℝ) (new_player2_weight : ℝ) 
  (h1 : original_avg = 112)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_avg = 106)
  (h5 : n * original_avg + new_player1_weight + new_player2_weight = (n + 2) * new_avg) :
  n = 7 := by
  sorry

end bowling_team_size_l3140_314060


namespace function_zero_l3140_314072

theorem function_zero (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -f x) 
  (h2 : ∀ x, f (-x) = f x) : 
  ∀ x, f x = 0 := by
sorry

end function_zero_l3140_314072


namespace cubic_root_sum_cubes_l3140_314052

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) ∧ 
  (b^3 - 2*b^2 + 3*b - 4 = 0) ∧ 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end cubic_root_sum_cubes_l3140_314052


namespace inverse_variation_problem_l3140_314009

/-- Given that x and y are always positive, x^3 and y vary inversely, 
    and y = 8 when x = 2, prove that x = 1 / (13.5^(1/3)) when y = 1728 -/
theorem inverse_variation_problem (x y : ℝ) 
  (h_positive : x > 0 ∧ y > 0)
  (h_inverse : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (h_inverse.choose)^3)
  (h_final : y = 1728) :
  x = 1 / (13.5^(1/3)) :=
sorry

end inverse_variation_problem_l3140_314009


namespace soccer_team_starters_l3140_314054

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem soccer_team_starters : 
  (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 220 := by
  sorry

end soccer_team_starters_l3140_314054


namespace proposition_falsity_l3140_314063

theorem proposition_falsity (P : ℕ → Prop) 
  (h_induction : ∀ k : ℕ, k > 0 → P k → P (k + 1))
  (h_false_5 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end proposition_falsity_l3140_314063


namespace car_speed_problem_l3140_314007

theorem car_speed_problem (D : ℝ) (V : ℝ) : 
  D > 0 →
  (D / ((D/3)/60 + (D/3)/24 + (D/3)/V)) = 37.89473684210527 →
  V = 48 := by sorry

end car_speed_problem_l3140_314007


namespace equidistant_line_equations_l3140_314037

/-- A line passing through (1, 2) and equidistant from (0, 0) and (3, 1) -/
structure EquidistantLine where
  -- Coefficients of the line equation ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (1, 2)
  passes_through : a + 2 * b + c = 0
  -- The line is equidistant from (0, 0) and (3, 1)
  equidistant : (c^2) / (a^2 + b^2) = (3*a + b + c)^2 / (a^2 + b^2)

/-- Theorem stating the two possible equations of the equidistant line -/
theorem equidistant_line_equations : 
  ∀ (l : EquidistantLine), (l.a = 1 ∧ l.b = -3 ∧ l.c = 5) ∨ (l.a = 3 ∧ l.b = 1 ∧ l.c = -5) :=
by sorry

end equidistant_line_equations_l3140_314037


namespace imaginary_part_of_complex_fraction_l3140_314023

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i^2 = -1 → Complex.im (2 / (2 + i)) = -2/5 := by sorry

end imaginary_part_of_complex_fraction_l3140_314023


namespace flower_count_l3140_314076

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of different candles --/
def num_candles : ℕ := 4

/-- The number of candles to choose --/
def candles_to_choose : ℕ := 2

/-- The number of flowers to choose --/
def flowers_to_choose : ℕ := 8

/-- The total number of candle + flower groupings --/
def total_groupings : ℕ := 54

theorem flower_count :
  ∃ (F : ℕ), 
    F > 0 ∧
    choose num_candles candles_to_choose * choose F flowers_to_choose = total_groupings ∧
    F = 9 :=
by sorry

end flower_count_l3140_314076


namespace rental_fee_minimization_l3140_314008

/-- Represents the total number of buses to be rented -/
def total_buses : ℕ := 6

/-- Represents the rental fee for a Type A bus -/
def type_a_fee : ℕ := 450

/-- Represents the rental fee for a Type B bus -/
def type_b_fee : ℕ := 300

/-- Calculates the total rental fee based on the number of Type B buses -/
def rental_fee (x : ℕ) : ℕ := total_buses * type_a_fee - (type_a_fee - type_b_fee) * x

theorem rental_fee_minimization :
  ∀ x : ℕ, 0 < x → x < total_buses → x < total_buses - x →
  (∀ y : ℕ, 0 < y → y < total_buses → y < total_buses - y →
    rental_fee x ≤ rental_fee y) →
  x = 2 ∧ rental_fee x = 2400 := by sorry

end rental_fee_minimization_l3140_314008


namespace shoes_cost_eleven_l3140_314058

/-- The cost of shoes given initial amount, sweater cost, T-shirt cost, and remaining amount -/
def cost_of_shoes (initial_amount sweater_cost tshirt_cost remaining_amount : ℕ) : ℕ :=
  initial_amount - sweater_cost - tshirt_cost - remaining_amount

/-- Theorem stating that the cost of shoes is 11 given the problem conditions -/
theorem shoes_cost_eleven :
  cost_of_shoes 91 24 6 50 = 11 := by
  sorry

end shoes_cost_eleven_l3140_314058


namespace base5_product_l3140_314081

/-- Converts a base-5 number represented as a list of digits to a natural number. -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a natural number to its base-5 representation as a list of digits. -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The statement of the problem. -/
theorem base5_product : 
  let a := fromBase5 [1, 3, 1]
  let b := fromBase5 [1, 3]
  toBase5 (a * b) = [2, 3, 3, 3] := by sorry

end base5_product_l3140_314081


namespace determinant_2x2_matrix_l3140_314030

theorem determinant_2x2_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, 3; -1, 2]
  Matrix.det A = 17 := by
sorry

end determinant_2x2_matrix_l3140_314030


namespace inscribed_square_area_l3140_314024

/-- The parabola function f(x) = x^2 - 10x + 20 --/
def f (x : ℝ) : ℝ := x^2 - 10*x + 20

/-- A square inscribed between a parabola and the x-axis --/
structure InscribedSquare where
  center : ℝ × ℝ
  side_length : ℝ
  h1 : center.1 = 5 -- The x-coordinate of the center is at the vertex of the parabola
  h2 : center.2 = side_length / 2 -- The y-coordinate of the center is half the side length
  h3 : f (center.1 + side_length / 2) = side_length -- The top right corner lies on the parabola

/-- The theorem stating that the area of the inscribed square is 400 --/
theorem inscribed_square_area (s : InscribedSquare) : s.side_length^2 = 400 := by
  sorry


end inscribed_square_area_l3140_314024


namespace squirrel_journey_time_l3140_314045

/-- Calculates the total journey time in minutes for a squirrel gathering nuts -/
theorem squirrel_journey_time (distance_to_tree : ℝ) (speed_to_tree : ℝ) (speed_from_tree : ℝ) :
  distance_to_tree = 2 →
  speed_to_tree = 3 →
  speed_from_tree = 2 →
  (distance_to_tree / speed_to_tree + distance_to_tree / speed_from_tree) * 60 = 100 := by
  sorry

#check squirrel_journey_time

end squirrel_journey_time_l3140_314045


namespace cube_in_pyramid_volume_l3140_314003

/-- A pyramid with a square base and isosceles triangular lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_height : ℝ

/-- A cube placed inside the pyramid -/
structure InsideCube where
  side_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InsideCube) : ℝ := c.side_length ^ 3

theorem cube_in_pyramid_volume 
  (p : Pyramid) 
  (c : InsideCube) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_height = 4) 
  (h3 : c.side_length * 2 = p.lateral_height) : 
  cube_volume c = 8 := by
  sorry

#check cube_in_pyramid_volume

end cube_in_pyramid_volume_l3140_314003


namespace integral_x_squared_plus_x_minus_one_times_exp_x_over_two_l3140_314098

theorem integral_x_squared_plus_x_minus_one_times_exp_x_over_two :
  ∫ x in (0 : ℝ)..2, (x^2 + x - 1) * Real.exp (x / 2) = 2 * (3 * Real.exp 1 - 5) := by
  sorry

end integral_x_squared_plus_x_minus_one_times_exp_x_over_two_l3140_314098


namespace chemical_equilibrium_and_precipitate_l3140_314011

-- Define the chemical reaction parameters
def initial_BaCl2_concentration : ℝ := 10
def equilibrium_constant : ℝ := 5 * 10^6
def initial_volume : ℝ := 1

-- Define the molar mass of BaSO4
def molar_mass_BaSO4 : ℝ := 233.40

-- Define the theorem
theorem chemical_equilibrium_and_precipitate :
  ∃ (equilibrium_BaSO4_concentration : ℝ) (mass_BaSO4_precipitate : ℝ),
    (abs (equilibrium_BaSO4_concentration - 10) < 0.01) ∧
    (abs (mass_BaSO4_precipitate - 2334) < 0.1) :=
sorry

end chemical_equilibrium_and_precipitate_l3140_314011


namespace area_of_B_l3140_314077

-- Define set A
def A : Set ℝ := {a : ℝ | -1 ≤ a ∧ a ≤ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 + p.2 ≥ 0}

-- Theorem statement
theorem area_of_B : MeasureTheory.volume B = 7 := by
  sorry

end area_of_B_l3140_314077


namespace pages_copied_for_30_dollars_l3140_314021

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $30, 
    the maximum number of pages that can be copied is 1000. -/
theorem pages_copied_for_30_dollars : 
  max_pages_copied 3 30 = 1000 := by
  sorry

end pages_copied_for_30_dollars_l3140_314021


namespace sally_balloons_l3140_314092

/-- Given the number of blue balloons for Alyssa, Sandy, and the total,
    prove that Sally has the correct number of blue balloons. -/
theorem sally_balloons (alyssa_balloons sandy_balloons total_balloons : ℕ)
  (h1 : alyssa_balloons = 37)
  (h2 : sandy_balloons = 28)
  (h3 : total_balloons = 104) :
  total_balloons - (alyssa_balloons + sandy_balloons) = 39 := by
  sorry

#check sally_balloons

end sally_balloons_l3140_314092


namespace min_difference_when_sum_maximized_l3140_314091

theorem min_difference_when_sum_maximized :
  ∀ x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℕ+,
    x₁ < x₂ → x₂ < x₃ → x₃ < x₄ → x₄ < x₅ → x₅ < x₆ → x₆ < x₇ → x₇ < x₈ → x₈ < x₉ →
    x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 220 →
    (∀ y₁ y₂ y₃ y₄ y₅ : ℕ+,
      y₁ < y₂ → y₂ < y₃ → y₃ < y₄ → y₄ < y₅ →
      y₁ + y₂ + y₃ + y₄ + y₅ ≤ x₁ + x₂ + x₃ + x₄ + x₅) →
    x₉ - x₁ = 9 :=
by sorry

end min_difference_when_sum_maximized_l3140_314091


namespace least_subtraction_for_divisibility_l3140_314086

theorem least_subtraction_for_divisibility (n m k : ℕ) (h : n - k ≥ 0) : 
  (∃ q : ℕ, n - k = m * q) ∧ 
  (∀ j : ℕ, j < k → ¬(∃ q : ℕ, n - j = m * q)) → 
  k = n % m :=
sorry

#check least_subtraction_for_divisibility 2361 23 15

end least_subtraction_for_divisibility_l3140_314086


namespace cookie_distribution_l3140_314070

/-- Given 24 cookies, prove that 6 friends can share them if each friend receives 3 more cookies than the previous friend, with the first friend receiving at least 1 cookie. -/
theorem cookie_distribution (total_cookies : ℕ) (cookie_increment : ℕ) (n : ℕ) : 
  total_cookies = 24 →
  cookie_increment = 3 →
  (n : ℚ) * ((1 : ℚ) + (1 : ℚ) + (cookie_increment : ℚ) * ((n : ℚ) - 1)) / 2 = (total_cookies : ℚ) →
  n = 6 := by
  sorry

end cookie_distribution_l3140_314070


namespace original_number_proof_l3140_314064

theorem original_number_proof : 
  ∃! x : ℤ, ∃ y : ℤ, x + y = 859560 ∧ x % 456 = 0 :=
by
  sorry

end original_number_proof_l3140_314064


namespace election_votes_l3140_314039

theorem election_votes (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    loser_votes = (30 * total_votes) / 100 ∧
    winner_votes - loser_votes = 174) →
  total_votes = 435 := by
sorry

end election_votes_l3140_314039


namespace speedster_convertibles_count_l3140_314065

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  nonSpeedsters : ℕ
  speedsterConvertibles : ℕ

/-- Theorem stating the number of Speedster convertibles in the inventory -/
theorem speedster_convertibles_count (inv : Inventory) :
  inv.nonSpeedsters = 30 ∧
  inv.speedsters = 3 * inv.total / 4 ∧
  inv.nonSpeedsters = inv.total - inv.speedsters ∧
  inv.speedsterConvertibles = 3 * inv.speedsters / 5 →
  inv.speedsterConvertibles = 54 := by
  sorry


end speedster_convertibles_count_l3140_314065


namespace collinear_vectors_l3140_314062

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem collinear_vectors (k : ℝ) :
  (∃ t : ℝ, (a.1 + k * c.1, a.2 + k * c.2) = t • (2 * b.1 - a.1, 2 * b.2 - a.2)) →
  k = -16/13 := by
sorry

end collinear_vectors_l3140_314062


namespace monday_water_usage_l3140_314034

/-- Represents the relationship between rainfall and water usage -/
structure RainfallWaterUsage where
  rainfall : ℝ
  water_used : ℝ

/-- The constant of inverse proportionality between rainfall and water usage -/
def inverse_proportionality_constant (day : RainfallWaterUsage) : ℝ :=
  day.rainfall * day.water_used

theorem monday_water_usage 
  (sunday : RainfallWaterUsage)
  (monday_rainfall : ℝ)
  (h_sunday_rainfall : sunday.rainfall = 3)
  (h_sunday_water : sunday.water_used = 10)
  (h_monday_rainfall : monday_rainfall = 5)
  (h_inverse_prop : ∀ (day1 day2 : RainfallWaterUsage), 
    inverse_proportionality_constant day1 = inverse_proportionality_constant day2) :
  ∃ (monday : RainfallWaterUsage), 
    monday.rainfall = monday_rainfall ∧ 
    monday.water_used = 6 :=
sorry

end monday_water_usage_l3140_314034


namespace greg_marbles_l3140_314002

/-- The number of marbles Adam has -/
def adam_marbles : ℕ := 29

/-- The number of additional marbles Greg has compared to Adam -/
def greg_additional_marbles : ℕ := 14

/-- Theorem: Greg has 43 marbles -/
theorem greg_marbles : adam_marbles + greg_additional_marbles = 43 := by
  sorry

end greg_marbles_l3140_314002


namespace min_sum_inequality_min_sum_achievable_l3140_314061

theorem min_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (12 * a)) + ((a + b + c) / (5 * a * b * c)) ≥ 4 / (360 ^ (1/4 : ℝ)) :=
sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b)) + (b / (6 * c)) + (c / (12 * a)) + ((a + b + c) / (5 * a * b * c)) = 4 / (360 ^ (1/4 : ℝ)) :=
sorry

end min_sum_inequality_min_sum_achievable_l3140_314061


namespace daves_old_cards_l3140_314036

/-- Given Dave's baseball card organization, prove the number of old cards --/
theorem daves_old_cards
  (cards_per_page : ℕ)
  (new_cards : ℕ)
  (pages_used : ℕ)
  (h1 : cards_per_page = 8)
  (h2 : new_cards = 3)
  (h3 : pages_used = 2) :
  pages_used * cards_per_page - new_cards = 13 := by
  sorry

end daves_old_cards_l3140_314036


namespace triangle_side_difference_triangle_side_difference_is_12_l3140_314044

theorem triangle_side_difference : ℕ → Prop :=
  fun d =>
    ∃ (x_min x_max : ℤ),
      (∀ x : ℤ, (x > x_min ∧ x < x_max) → (x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x)) ∧
      (∀ x : ℤ, (x ≤ x_min ∨ x ≥ x_max) → ¬(x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x)) ∧
      (x_max - x_min = d + 1)

theorem triangle_side_difference_is_12 : triangle_side_difference 12 := by
  sorry

end triangle_side_difference_triangle_side_difference_is_12_l3140_314044


namespace sugar_amount_l3140_314099

/-- The number of cups of sugar in Mary's cake recipe -/
def sugar : ℕ := sorry

/-- The total amount of flour needed for the recipe in cups -/
def total_flour : ℕ := 9

/-- The amount of flour already added in cups -/
def flour_added : ℕ := 2

/-- The remaining flour to be added is 1 cup more than the amount of sugar -/
axiom remaining_flour_sugar_relation : total_flour - flour_added = sugar + 1

theorem sugar_amount : sugar = 6 := by sorry

end sugar_amount_l3140_314099


namespace T4_championship_probability_l3140_314051

/-- Represents a team in the playoffs -/
inductive Team : Type
| T1 : Team
| T2 : Team
| T3 : Team
| T4 : Team

/-- The probability of team i winning against team j -/
def winProbability (i j : Team) : ℚ :=
  match i, j with
  | Team.T1, Team.T2 => 1/3
  | Team.T1, Team.T3 => 1/4
  | Team.T1, Team.T4 => 1/5
  | Team.T2, Team.T1 => 2/3
  | Team.T2, Team.T3 => 2/5
  | Team.T2, Team.T4 => 1/3
  | Team.T3, Team.T1 => 3/4
  | Team.T3, Team.T2 => 3/5
  | Team.T3, Team.T4 => 3/7
  | Team.T4, Team.T1 => 4/5
  | Team.T4, Team.T2 => 2/3
  | Team.T4, Team.T3 => 4/7
  | _, _ => 1/2  -- This case should never occur in our scenario

/-- The probability of T4 winning the championship -/
def T4ChampionshipProbability : ℚ :=
  (winProbability Team.T4 Team.T1) * 
  ((winProbability Team.T3 Team.T2) * (winProbability Team.T4 Team.T3) +
   (winProbability Team.T2 Team.T3) * (winProbability Team.T4 Team.T2))

theorem T4_championship_probability :
  T4ChampionshipProbability = 256/525 := by
  sorry

#eval T4ChampionshipProbability

end T4_championship_probability_l3140_314051


namespace divisors_of_2_pow_48_minus_1_l3140_314056

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 ≤ a ∧ a < b ∧ b ≤ 70 ∧
  (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 ∧
  a = 63 ∧ b = 65 := by
  sorry

end divisors_of_2_pow_48_minus_1_l3140_314056


namespace product_digit_sum_theorem_l3140_314082

def is_single_digit (n : ℕ) : Prop := 1 < n ∧ n < 10

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem product_digit_sum_theorem (x y : ℕ) :
  is_single_digit x ∧ is_single_digit y ∧ x ≠ 9 ∧ y ≠ 9 ∧ digit_sum (x * y) = x →
  (x = 3 ∧ y = 4) ∨ (x = 3 ∧ y = 7) ∨ (x = 6 ∧ y = 4) ∨ (x = 6 ∧ y = 7) :=
sorry

end product_digit_sum_theorem_l3140_314082


namespace triangle_side_length_l3140_314031

-- Define the triangle PQR
structure Triangle (P Q R : ℝ) where
  angleSum : P + Q + R = Real.pi
  positive : 0 < P ∧ 0 < Q ∧ 0 < R

-- Define the side lengths
def sideLength (a b : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_length 
  (P Q R : ℝ) 
  (tri : Triangle P Q R) 
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1)
  (h2 : sideLength P Q = 5)
  (h3 : sideLength P Q + sideLength Q R + sideLength R P = 12) :
  sideLength Q R = 3.5 := by sorry

end triangle_side_length_l3140_314031


namespace quadratic_vertex_l3140_314066

/-- The quadratic function f(x) = -(x+1)^2 - 8 has vertex coordinates (-1, -8) -/
theorem quadratic_vertex (x : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ -(x + 1)^2 - 8
  (∀ x, f x ≤ f (-1)) ∧ f (-1) = -8 := by
sorry

end quadratic_vertex_l3140_314066


namespace scores_mode_l3140_314014

def scores : List ℕ := [61, 62, 71, 78, 85, 85, 92, 96]

def mode (l : List ℕ) : ℕ := sorry

theorem scores_mode : mode scores = 85 := by sorry

end scores_mode_l3140_314014


namespace rectangular_cube_height_l3140_314087

-- Define the dimensions of the rectangular cube
def length : ℝ := 3
def width : ℝ := 2

-- Define the side length of the reference cube
def cubeSide : ℝ := 2

-- Define the surface area of the rectangular cube
def surfaceArea (h : ℝ) : ℝ := 2 * length * width + 2 * length * h + 2 * width * h

-- Define the surface area of the reference cube
def cubeSurfaceArea : ℝ := 6 * cubeSide^2

-- Theorem statement
theorem rectangular_cube_height : 
  ∃ h : ℝ, surfaceArea h = cubeSurfaceArea ∧ h = 1.2 := by sorry

end rectangular_cube_height_l3140_314087


namespace stratified_sample_composition_l3140_314067

/-- Represents the number of male students in the class -/
def male_students : ℕ := 40

/-- Represents the number of female students in the class -/
def female_students : ℕ := 30

/-- Represents the total number of students in the class -/
def total_students : ℕ := male_students + female_students

/-- Represents the size of the stratified sample -/
def sample_size : ℕ := 7

/-- Calculates the number of male students in the stratified sample -/
def male_sample : ℕ := (male_students * sample_size + total_students - 1) / total_students

/-- Calculates the number of female students in the stratified sample -/
def female_sample : ℕ := sample_size - male_sample

/-- Theorem stating that the stratified sample consists of 4 male and 3 female students -/
theorem stratified_sample_composition :
  male_sample = 4 ∧ female_sample = 3 :=
sorry

end stratified_sample_composition_l3140_314067


namespace ratio_calculation_l3140_314042

theorem ratio_calculation (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.3 * P) 
  (hN : N = 0.5 * P) : 
  M / N = 6 / 25 := by
  sorry

end ratio_calculation_l3140_314042


namespace yard_area_l3140_314073

/-- Calculates the area of a rectangular yard given the length of one side and the total length of the other three sides. -/
theorem yard_area (side_length : ℝ) (other_sides : ℝ) : 
  side_length = 40 → other_sides = 50 → side_length * ((other_sides - side_length) / 2) = 200 :=
by
  sorry

#check yard_area

end yard_area_l3140_314073


namespace max_value_A_l3140_314046

/-- The function A(x, y) as defined in the problem -/
def A (x y : ℝ) : ℝ := x^4*y + x*y^4 + x^3*y + x*y^3 + x^2*y + x*y^2

/-- The theorem stating the maximum value of A(x, y) under the given constraint -/
theorem max_value_A :
  ∀ x y : ℝ, x + y = 1 → A x y ≤ 7/16 :=
by sorry

end max_value_A_l3140_314046


namespace unique_ambiguous_product_l3140_314018

def numbers : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

def is_valid_product (p : Nat) : Prop :=
  ∃ (s : Finset Nat), s ⊆ numbers ∧ s.card = 5 ∧ s.prod id = p

def parity_ambiguous (p : Nat) : Prop :=
  ∃ (s1 s2 : Finset Nat), s1 ≠ s2 ∧
    s1 ⊆ numbers ∧ s2 ⊆ numbers ∧
    s1.card = 5 ∧ s2.card = 5 ∧
    s1.prod id = p ∧ s2.prod id = p ∧
    s1.sum id % 2 ≠ s2.sum id % 2

theorem unique_ambiguous_product :
  ∃! p, is_valid_product p ∧ parity_ambiguous p ∧ p = 420 := by sorry

end unique_ambiguous_product_l3140_314018


namespace least_positive_linear_combination_of_primes_l3140_314029

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem least_positive_linear_combination_of_primes :
  ∃ (x y z w : ℕ), 
    is_prime x ∧ is_prime y ∧ is_prime z ∧ is_prime w ∧
    24*x + 16*y - 7*z + 5*w = 13 ∧
    (∀ (a b c d : ℕ), is_prime a → is_prime b → is_prime c → is_prime d →
      24*a + 16*b - 7*c + 5*d > 0 → 24*a + 16*b - 7*c + 5*d ≥ 13) :=
by sorry

end least_positive_linear_combination_of_primes_l3140_314029


namespace amusement_park_problem_l3140_314068

/-- The number of children who got on the Ferris wheel -/
def ferris_wheel_riders : ℕ := sorry

theorem amusement_park_problem :
  let total_children : ℕ := 5
  let ferris_wheel_cost : ℕ := 5
  let merry_go_round_cost : ℕ := 3
  let ice_cream_cost : ℕ := 8
  let ice_cream_per_child : ℕ := 2
  let total_spent : ℕ := 110
  ferris_wheel_riders * ferris_wheel_cost +
  total_children * merry_go_round_cost +
  total_children * ice_cream_per_child * ice_cream_cost = total_spent ∧
  ferris_wheel_riders = 3 :=
by sorry

end amusement_park_problem_l3140_314068


namespace painting_equation_proof_l3140_314089

theorem painting_equation_proof (t : ℝ) : 
  let doug_rate : ℝ := 1 / 4
  let dave_rate : ℝ := 1 / 6
  let combined_rate : ℝ := doug_rate + dave_rate
  let break_time : ℝ := 1 / 2
  (combined_rate * (t - break_time) = 1) ↔ 
  ((1 / 4 + 1 / 6) * (t - 1 / 2) = 1) :=
by sorry

end painting_equation_proof_l3140_314089


namespace inequality_proof_l3140_314010

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end inequality_proof_l3140_314010


namespace combined_fuel_efficiency_l3140_314016

/-- The combined fuel efficiency of three cars -/
theorem combined_fuel_efficiency 
  (m : ℝ) -- distance driven by each car
  (h1 : m > 0) -- ensure distance is positive
  (efficiency1 efficiency2 efficiency3 : ℝ) -- individual car efficiencies
  (h2 : efficiency1 = 35) -- Ray's car efficiency
  (h3 : efficiency2 = 25) -- Tom's car efficiency
  (h4 : efficiency3 = 20) -- Alice's car efficiency
  : (3 * m) / (m / efficiency1 + m / efficiency2 + m / efficiency3) = 2100 / 83 := by
  sorry

#eval (2100 : ℚ) / 83 -- To show the decimal approximation

end combined_fuel_efficiency_l3140_314016


namespace karthik_weight_average_l3140_314032

def karthik_weight_range (w : ℝ) : Prop :=
  55 < w ∧ w < 62 ∧ 50 < w ∧ w < 60 ∧ w < 58

theorem karthik_weight_average :
  ∃ (min max : ℝ),
    (∀ w, karthik_weight_range w → min ≤ w ∧ w ≤ max) ∧
    (∃ w₁ w₂, karthik_weight_range w₁ ∧ karthik_weight_range w₂ ∧ w₁ = min ∧ w₂ = max) ∧
    (min + max) / 2 = 56.5 :=
sorry

end karthik_weight_average_l3140_314032


namespace annas_cupcakes_l3140_314095

theorem annas_cupcakes (C : ℕ) : 
  (C : ℚ) * (1 / 5) - 3 = 9 → C = 60 := by
  sorry

end annas_cupcakes_l3140_314095


namespace greatest_of_three_consecutive_integers_l3140_314013

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 39) → max x (max y z) = 14 := by
  sorry

end greatest_of_three_consecutive_integers_l3140_314013


namespace line_properties_l3140_314027

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - m * y + 2 = 0

-- Theorem statement
theorem line_properties (m : ℝ) :
  (∀ y, line_equation (-2) y m) ∧
  (∃ x, x ≠ 0 ∧ line_equation x 0 m) :=
by sorry

end line_properties_l3140_314027


namespace rational_fraction_value_l3140_314094

theorem rational_fraction_value (x y : ℝ) :
  3 < (x - y) / (x + y) →
  (x - y) / (x + y) < 4 →
  ∃ (a b : ℤ), x / y = a / b →
  x + y = 10 →
  x / y = -2 := by
sorry

end rational_fraction_value_l3140_314094


namespace complex_alpha_value_l3140_314079

theorem complex_alpha_value (α β : ℂ) 
  (h1 : (α + 2*β).im = 0)
  (h2 : (α - Complex.I * (3*β - α)).im = 0)
  (h3 : β = 2 + 3*Complex.I) : 
  α = 6 - 6*Complex.I := by sorry

end complex_alpha_value_l3140_314079


namespace picture_book_shelves_l3140_314015

/-- Given a bookcase with the following properties:
  * Each shelf contains exactly 6 books
  * There are 5 shelves of mystery books
  * The total number of books is 54
  Prove that the number of shelves of picture books is 4 -/
theorem picture_book_shelves :
  ∀ (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ),
    books_per_shelf = 6 →
    mystery_shelves = 5 →
    total_books = 54 →
    (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 4 :=
by sorry

end picture_book_shelves_l3140_314015


namespace x_squared_plus_7x_plus_12_bounds_l3140_314020

theorem x_squared_plus_7x_plus_12_bounds (x : ℝ) (h : x^2 - 7*x + 12 < 0) :
  42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 := by
  sorry

end x_squared_plus_7x_plus_12_bounds_l3140_314020


namespace max_value_fraction_l3140_314017

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y - x*y = 0) :
  (4 / (x + y)) ≤ 4/9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 4*y - x*y = 0 ∧ 4 / (x + y) = 4/9 :=
by sorry

end max_value_fraction_l3140_314017


namespace decagon_sign_change_impossible_l3140_314096

/-- Represents a point in the decagon where a number is placed -/
structure Point where
  value : Int
  is_vertex : Bool
  is_intersection : Bool

/-- Represents the decagon configuration -/
structure Decagon where
  points : List Point
  
/-- Represents an operation that can be performed on the decagon -/
inductive Operation
  | FlipSide : Nat → Operation  -- Flip signs along the nth side
  | FlipDiagonal : Nat → Operation  -- Flip signs along the nth diagonal

/-- Applies an operation to the decagon -/
def apply_operation (d : Decagon) (op : Operation) : Decagon :=
  sorry

/-- Checks if all points in the decagon have negative values -/
def all_negative (d : Decagon) : Bool :=
  sorry

/-- Initial setup of the decagon with all +1 values -/
def initial_decagon : Decagon :=
  sorry

theorem decagon_sign_change_impossible :
  ∀ (ops : List Operation),
    ¬(all_negative (ops.foldl apply_operation initial_decagon)) :=
  sorry

end decagon_sign_change_impossible_l3140_314096


namespace m_range_for_three_roots_l3140_314075

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 3

/-- The function g(x) defined in the problem -/
def g (x m : ℝ) : ℝ := f x - m

/-- Theorem stating the range of m for which g(x) has exactly 3 real roots -/
theorem m_range_for_three_roots :
  ∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g x m = 0) → m ∈ Set.Ioo (-24) 8 :=
sorry

end m_range_for_three_roots_l3140_314075


namespace max_consecutive_sum_less_than_1000_l3140_314059

theorem max_consecutive_sum_less_than_1000 :
  ∀ n : ℕ, (n * (n + 1)) / 2 < 1000 ↔ n ≤ 44 :=
by sorry

end max_consecutive_sum_less_than_1000_l3140_314059


namespace solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l3140_314085

-- Equation 1: x^2 - 4x + 3 = 0
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0 ∧ x₁ = 1 ∧ x₂ = 3 := by
  sorry

-- Equation 2: (x + 1)(x - 2) = 4
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, (x₁ + 1)*(x₁ - 2) = 4 ∧ (x₂ + 1)*(x₂ - 2) = 4 ∧ x₁ = -2 ∧ x₂ = 3 := by
  sorry

-- Equation 3: 3x(x - 1) = 2 - 2x
theorem solve_equation_3 : 
  ∃ x₁ x₂ : ℝ, 3*x₁*(x₁ - 1) = 2 - 2*x₁ ∧ 3*x₂*(x₂ - 1) = 2 - 2*x₂ ∧ x₁ = 1 ∧ x₂ = -2/3 := by
  sorry

-- Equation 4: 2x^2 - 4x - 1 = 0
theorem solve_equation_4 : 
  ∃ x₁ x₂ : ℝ, 2*x₁^2 - 4*x₁ - 1 = 0 ∧ 2*x₂^2 - 4*x₂ - 1 = 0 ∧ 
  x₁ = (2 + Real.sqrt 6) / 2 ∧ x₂ = (2 - Real.sqrt 6) / 2 := by
  sorry

end solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l3140_314085


namespace sqrt_equation_solution_l3140_314005

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt x + Real.sqrt (x + 4) = 8 → x = 225 / 16 := by
sorry

end sqrt_equation_solution_l3140_314005


namespace double_reflection_of_D_l3140_314078

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define reflection over y-axis
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Define reflection over x-axis
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Define the composition of reflections
def doubleReflection (p : Point) : Point :=
  reflectOverXAxis (reflectOverYAxis p)

-- Theorem statement
theorem double_reflection_of_D :
  let D : Point := { x := 3, y := 3 }
  doubleReflection D = { x := -3, y := -3 } := by
  sorry

end double_reflection_of_D_l3140_314078


namespace combined_weight_leo_kendra_prove_combined_weight_l3140_314074

/-- The combined weight of Leo and Kendra given Leo's current weight and the condition of their weight relationship after Leo gains 10 pounds. -/
theorem combined_weight_leo_kendra : ℝ → ℝ → Prop :=
  fun leo_weight kendra_weight =>
    (leo_weight = 104) →
    (leo_weight + 10 = 1.5 * kendra_weight) →
    (leo_weight + kendra_weight = 180)

/-- The theorem statement -/
theorem prove_combined_weight : ∃ (leo_weight kendra_weight : ℝ),
  combined_weight_leo_kendra leo_weight kendra_weight :=
sorry

end combined_weight_leo_kendra_prove_combined_weight_l3140_314074
