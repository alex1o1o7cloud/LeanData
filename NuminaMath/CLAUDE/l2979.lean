import Mathlib

namespace library_book_sorting_l2979_297939

theorem library_book_sorting (total_removed : ℕ) (damaged : ℕ) (x : ℚ) 
  (h1 : total_removed = 69)
  (h2 : damaged = 11)
  (h3 : total_removed = damaged + (x * damaged - 8)) :
  x = 6 := by
sorry

end library_book_sorting_l2979_297939


namespace equilateral_triangle_area_l2979_297998

theorem equilateral_triangle_area (h : ℝ) (A : ℝ) : 
  h = 3 * Real.sqrt 3 → A = (Real.sqrt 3 / 4) * (2 * h / Real.sqrt 3)^2 → A = 9 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_area_l2979_297998


namespace opposite_of_negative_five_l2979_297984

theorem opposite_of_negative_five : -(-5) = 5 := by sorry

end opposite_of_negative_five_l2979_297984


namespace berries_taken_l2979_297964

theorem berries_taken (stacy_initial : ℕ) (steve_initial : ℕ) (difference : ℕ) : 
  stacy_initial = 32 →
  steve_initial = 21 →
  difference = 7 →
  ∃ (berries_taken : ℕ), 
    steve_initial + berries_taken = stacy_initial - difference ∧
    berries_taken = 4 :=
by sorry

end berries_taken_l2979_297964


namespace root_quadratic_equation_property_l2979_297965

theorem root_quadratic_equation_property (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → 2*m^2 - 4*m + 5 = 11 := by
  sorry

end root_quadratic_equation_property_l2979_297965


namespace floor_sqrt_15_plus_1_squared_l2979_297914

theorem floor_sqrt_15_plus_1_squared : (⌊Real.sqrt 15⌋ + 1)^2 = 16 := by
  sorry

end floor_sqrt_15_plus_1_squared_l2979_297914


namespace intersection_implies_a_values_union_implies_a_range_l2979_297943

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x + a^2 - 5 = 0}

-- Part 1: A ∩ B = {2} implies a = -3 or a = 1
theorem intersection_implies_a_values (a : ℝ) : 
  A ∩ B a = {2} → a = -3 ∨ a = 1 := by sorry

-- Part 2: A ∪ B = A implies a ≤ -3 or a > 7/3
theorem union_implies_a_range (a : ℝ) :
  A ∪ B a = A → a ≤ -3 ∨ a > 7/3 := by sorry

end intersection_implies_a_values_union_implies_a_range_l2979_297943


namespace garden_breadth_calculation_l2979_297911

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_calculation (garden : RectangularGarden) 
  (h1 : perimeter garden = 480)
  (h2 : garden.length = 140) :
  garden.breadth = 100 := by
  sorry

end garden_breadth_calculation_l2979_297911


namespace nathan_tokens_used_l2979_297910

/-- The total number of tokens used by Nathan at the arcade --/
def total_tokens (air_hockey_plays : ℕ) (basketball_plays : ℕ) (skee_ball_plays : ℕ)
                 (air_hockey_cost : ℕ) (basketball_cost : ℕ) (skee_ball_cost : ℕ) : ℕ :=
  air_hockey_plays * air_hockey_cost +
  basketball_plays * basketball_cost +
  skee_ball_plays * skee_ball_cost

/-- Theorem stating that Nathan used 64 tokens in total --/
theorem nathan_tokens_used :
  total_tokens 5 7 3 4 5 3 = 64 := by
  sorry

end nathan_tokens_used_l2979_297910


namespace square_root_difference_l2979_297986

theorem square_root_difference : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - 3 * Real.sqrt 3 := by
  sorry

end square_root_difference_l2979_297986


namespace largest_proportional_part_l2979_297924

theorem largest_proportional_part (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a / b = 2 ∧ a / c = 3 →
  max (total * a / (a + b + c)) (max (total * b / (a + b + c)) (total * c / (a + b + c))) = 60 := by
  sorry

end largest_proportional_part_l2979_297924


namespace increase_both_averages_l2979_297919

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem increase_both_averages :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end increase_both_averages_l2979_297919


namespace tape_length_for_circular_base_l2979_297958

/-- The length of tape needed for a circular lamp base -/
theorem tape_length_for_circular_base :
  let area : ℝ := 176
  let π_approx : ℝ := 22 / 7
  let extra_tape : ℝ := 3
  let radius : ℝ := Real.sqrt (area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let total_length : ℝ := circumference + extra_tape
  ∃ ε > 0, abs (total_length - 50.058) < ε :=
by sorry

end tape_length_for_circular_base_l2979_297958


namespace coplanar_condition_l2979_297950

open Vector

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

def isCoplanar (p₁ p₂ p₃ p₄ p₅ : V) : Prop :=
  ∃ (a b c d : ℝ), a • (p₂ - p₁) + b • (p₃ - p₁) + c • (p₄ - p₁) + d • (p₅ - p₁) = 0

theorem coplanar_condition (O E F G H I : V) (m : ℝ) :
  (4 • (E - O) - 3 • (F - O) + 6 • (G - O) + m • (H - O) - 2 • (I - O) = 0) →
  (isCoplanar E F G H I ↔ m = -5) := by
  sorry

end coplanar_condition_l2979_297950


namespace probability_score_le_6_is_13_35_l2979_297981

structure Bag where
  red_balls : ℕ
  black_balls : ℕ

def score (red : ℕ) (black : ℕ) : ℕ :=
  red + 3 * black

def probability_score_le_6 (b : Bag) : ℚ :=
  let total_balls := b.red_balls + b.black_balls
  let drawn_balls := 4
  (Nat.choose b.red_balls 4 * Nat.choose b.black_balls 0 +
   Nat.choose b.red_balls 3 * Nat.choose b.black_balls 1) /
  Nat.choose total_balls drawn_balls

theorem probability_score_le_6_is_13_35 (b : Bag) :
  b.red_balls = 4 → b.black_balls = 3 → probability_score_le_6 b = 13 / 35 := by
  sorry

end probability_score_le_6_is_13_35_l2979_297981


namespace integer_division_property_l2979_297941

theorem integer_division_property (n : ℕ+) : 
  (∃ k : ℤ, (2^(n : ℕ) + 1 : ℤ) = k * (n : ℤ)^2) ↔ n = 1 ∨ n = 3 := by
sorry

end integer_division_property_l2979_297941


namespace nancy_folders_l2979_297995

theorem nancy_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 43 → 
  deleted_files = 31 → 
  files_per_folder = 6 → 
  (initial_files - deleted_files) / files_per_folder = 2 := by
  sorry

end nancy_folders_l2979_297995


namespace geometric_sequence_property_l2979_297957

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_a4 : a 4 = 4) :
  a 2 * a 6 = a 4 * a 4 := by
sorry

end geometric_sequence_property_l2979_297957


namespace solution_set_is_closed_interval_l2979_297920

def system_solution (x : ℝ) : Prop :=
  -2 * (x - 3) > 10 ∧ x^2 + 7*x + 12 ≤ 0

theorem solution_set_is_closed_interval :
  {x : ℝ | system_solution x} = {x : ℝ | -4 ≤ x ∧ x ≤ -3} :=
by sorry

end solution_set_is_closed_interval_l2979_297920


namespace point_B_coordinates_l2979_297956

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the length of AB
def AB_length : ℝ := 4

-- Define the possible coordinates of point B
def B1 : ℝ × ℝ := (-7, 2)
def B2 : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
  (B.2 = A.2) →                        -- AB is parallel to x-axis
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = AB_length^2) →  -- Length of AB is 4
  (B = B1 ∨ B = B2) :=
by
  sorry  -- Proof is omitted as per instructions

end point_B_coordinates_l2979_297956


namespace man_speed_with_current_l2979_297951

/-- Calculates the man's speed with the current given his speed against the current and the current speed. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Proves that given a man's speed against a current of 9.6 km/hr and a current speed of 3.2 km/hr, 
    the man's speed with the current is 16.0 km/hr. -/
theorem man_speed_with_current :
  speed_with_current 9.6 3.2 = 16.0 := by
  sorry

#eval speed_with_current 9.6 3.2

end man_speed_with_current_l2979_297951


namespace percent_decrease_long_distance_call_l2979_297946

def original_cost : ℝ := 50
def new_cost : ℝ := 10

theorem percent_decrease_long_distance_call :
  (original_cost - new_cost) / original_cost * 100 = 80 := by sorry

end percent_decrease_long_distance_call_l2979_297946


namespace two_numbers_difference_l2979_297997

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 40 ∧ 
  3 * max x y - 4 * min x y = 44 → 
  |x - y| = 18 := by
sorry

end two_numbers_difference_l2979_297997


namespace anna_guessing_ratio_l2979_297982

theorem anna_guessing_ratio (c d : ℝ) 
  (h1 : c > 0 ∧ d > 0)  -- Ensure c and d are positive
  (h2 : 0.9 * c + 0.05 * d = 0.1 * c + 0.95 * d)  -- Equal number of cat and dog images
  (h3 : 0.95 * d = d - 0.05 * d)  -- 95% correct when guessing dog
  (h4 : 0.9 * c = c - 0.1 * c)  -- 90% correct when guessing cat
  : d / c = 8 / 9 := by
  sorry

end anna_guessing_ratio_l2979_297982


namespace raft_drift_time_l2979_297907

/-- The time for a raft to drift from B to A, given boat travel times -/
theorem raft_drift_time (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) 
  (h1 : distance / (boat_speed + current_speed) = 7)
  (h2 : distance / (boat_speed - current_speed) = 5)
  (h3 : boat_speed > 0)
  (h4 : current_speed > 0) :
  distance / current_speed = 35 := by
sorry

end raft_drift_time_l2979_297907


namespace scientific_notation_equality_l2979_297985

theorem scientific_notation_equality : 
  122254 = 1.22254 * (10 : ℝ) ^ 5 := by sorry

end scientific_notation_equality_l2979_297985


namespace geometric_sum_example_l2979_297972

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of the geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_example : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end geometric_sum_example_l2979_297972


namespace triangle_angle_inequality_l2979_297962

theorem triangle_angle_inequality (a b c α β γ : Real) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_sides : (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 := by
  sorry

end triangle_angle_inequality_l2979_297962


namespace average_of_combined_results_l2979_297989

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 45) (h₂ : n₂ = 25) (h₃ : avg₁ = 25) (h₄ : avg₂ = 45) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = 2250 / 70 := by
  sorry

end average_of_combined_results_l2979_297989


namespace smallest_integer_solution_three_is_solution_three_is_smallest_solution_l2979_297953

theorem smallest_integer_solution (x : ℤ) : (6 - 3 * x < 0) → x ≥ 3 :=
by sorry

theorem three_is_solution : 6 - 3 * 3 < 0 :=
by sorry

theorem three_is_smallest_solution : ∀ y : ℤ, y < 3 → 6 - 3 * y ≥ 0 :=
by sorry

end smallest_integer_solution_three_is_solution_three_is_smallest_solution_l2979_297953


namespace negation_of_proposition_l2979_297978

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x > Real.log x)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ ≤ Real.log x₀) := by
  sorry

end negation_of_proposition_l2979_297978


namespace greatest_integer_radius_l2979_297918

theorem greatest_integer_radius (A : ℝ) (h : A < 90 * Real.pi) : 
  ∀ r : ℕ, r * r * Real.pi ≤ A → r ≤ 9 :=
by sorry

end greatest_integer_radius_l2979_297918


namespace condition_necessary_not_sufficient_l2979_297930

theorem condition_necessary_not_sufficient 
  (a₁ a₂ b₁ b₂ : ℝ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ b₁ ≠ 0 ∧ b₂ ≠ 0) 
  (A : Set ℝ) 
  (hA : A = {x : ℝ | a₁ * x + b₁ > 0}) 
  (B : Set ℝ) 
  (hB : B = {x : ℝ | a₂ * x + b₂ > 0}) : 
  (∀ (A B : Set ℝ), A = B → a₁ / a₂ = b₁ / b₂) ∧ 
  ¬(∀ (A B : Set ℝ), a₁ / a₂ = b₁ / b₂ → A = B) :=
sorry

end condition_necessary_not_sufficient_l2979_297930


namespace time_difference_to_halfway_point_l2979_297933

/-- Given that Danny can reach Steve's house in 35 minutes and it takes Steve twice as long to reach Danny's house,
    prove that Steve will take 17.5 minutes longer than Danny to reach the halfway point between their houses. -/
theorem time_difference_to_halfway_point (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 35 →
  steve_time = 2 * danny_time →
  steve_time / 2 - danny_time / 2 = 17.5 := by
sorry

end time_difference_to_halfway_point_l2979_297933


namespace divisibility_condition_l2979_297968

theorem divisibility_condition (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2*y - 1) ↔ 
  ((x = 1 ∧ y > 0) ∨ 
   (∃ t : ℕ, t > 0 ∧ x = 2*t - 1 ∧ y = t) ∨ 
   (x = 3 ∧ y = 8) ∨ 
   (x = 5 ∧ y = 8)) :=
by sorry

end divisibility_condition_l2979_297968


namespace katie_packages_l2979_297999

def cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

theorem katie_packages :
  cupcake_packages 18 8 2 = 5 := by
  sorry

end katie_packages_l2979_297999


namespace complex_modulus_theorem_l2979_297921

theorem complex_modulus_theorem (t : ℝ) (i : ℂ) (h_i : i^2 = -1) :
  let z : ℂ := (1 - t*i) / (1 + i)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (Real.sqrt 3 + t*i) = 2 := by
  sorry

end complex_modulus_theorem_l2979_297921


namespace triangle_theorem_l2979_297969

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  angle_sum : A + B + C = π
  sine_rule_ab : a / (Real.sin A) = b / (Real.sin B)
  sine_rule_bc : b / (Real.sin B) = c / (Real.sin C)

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : Real.sin t.A * Real.sin t.B + Real.sin t.B * Real.sin t.C + Real.cos (2 * t.B) = 1) :
  -- Part 1: a, b, c are in arithmetic progression
  t.a + t.c = 2 * t.b ∧ 
  -- Part 2: If C = 2π/3, then a/b = 3/5
  (t.C = 2 * π / 3 → t.a / t.b = 3 / 5) := by
  sorry

end triangle_theorem_l2979_297969


namespace carly_butterfly_practice_l2979_297928

/-- The number of days Carly practices butterfly stroke per week -/
def butterfly_days : ℕ := sorry

/-- Hours of butterfly stroke practice per day -/
def butterfly_hours_per_day : ℕ := 3

/-- Days of backstroke practice per week -/
def backstroke_days_per_week : ℕ := 6

/-- Hours of backstroke practice per day -/
def backstroke_hours_per_day : ℕ := 2

/-- Total hours of swimming practice per month -/
def total_practice_hours : ℕ := 96

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

theorem carly_butterfly_practice :
  butterfly_days = 4 ∧
  butterfly_days * butterfly_hours_per_day * weeks_per_month +
  backstroke_days_per_week * backstroke_hours_per_day * weeks_per_month =
  total_practice_hours :=
sorry

end carly_butterfly_practice_l2979_297928


namespace number_of_friends_prove_number_of_friends_l2979_297983

theorem number_of_friends (original_bill : ℝ) (discount_percent : ℝ) (individual_payment : ℝ) : ℝ :=
  let discounted_bill := original_bill * (1 - discount_percent / 100)
  discounted_bill / individual_payment

theorem prove_number_of_friends :
  number_of_friends 100 6 18.8 = 5 := by
  sorry

end number_of_friends_prove_number_of_friends_l2979_297983


namespace banana_sharing_l2979_297971

theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  total_bananas = num_friends * bananas_per_friend →
  bananas_per_friend = 7 := by
  sorry

end banana_sharing_l2979_297971


namespace trigonometric_equality_l2979_297979

theorem trigonometric_equality : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (100 * π / 180)) / 
  (Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
   Real.cos (159 * π / 180) * Real.cos (99 * π / 180)) = 1 := by
  sorry

end trigonometric_equality_l2979_297979


namespace a_neg_three_sufficient_not_necessary_l2979_297905

/-- Two lines in the plane, parameterized by a real number a -/
def line1 (a : ℝ) := {(x, y) : ℝ × ℝ | x + a * y + 2 = 0}
def line2 (a : ℝ) := {(x, y) : ℝ × ℝ | a * x + (a + 2) * y + 1 = 0}

/-- The condition for two lines to be perpendicular -/
def are_perpendicular (a : ℝ) : Prop :=
  a * (a + 3) = 0

/-- The statement to be proved -/
theorem a_neg_three_sufficient_not_necessary :
  (∀ a : ℝ, a = -3 → are_perpendicular a) ∧
  ¬(∀ a : ℝ, are_perpendicular a → a = -3) :=
sorry

end a_neg_three_sufficient_not_necessary_l2979_297905


namespace angle_side_inequality_l2979_297955

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define the angle and side length functions
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
def side_length (t : Triangle) (v : Fin 3) : ℝ := sorry

-- State the theorem
theorem angle_side_inequality (t : Triangle) :
  angle t 0 > angle t 1 → side_length t 0 > side_length t 1 := by
  sorry

end angle_side_inequality_l2979_297955


namespace jasons_quarters_l2979_297903

/-- Given that Jason had 49 quarters initially and his dad gave him 25 quarters,
    prove that Jason now has 74 quarters. -/
theorem jasons_quarters (initial : ℕ) (given : ℕ) (total : ℕ) 
    (h1 : initial = 49) 
    (h2 : given = 25) 
    (h3 : total = initial + given) : 
  total = 74 := by
  sorry

end jasons_quarters_l2979_297903


namespace arithmetic_sequence_middle_term_l2979_297901

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℤ) -- a is the arithmetic sequence
  (h1 : a 0 = 3^2) -- first term is 3^2
  (h2 : a 2 = 3^4) -- third term is 3^4
  (h3 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- arithmetic sequence
  : a 1 = 45 := by
sorry

end arithmetic_sequence_middle_term_l2979_297901


namespace two_bedroom_square_footage_l2979_297993

/-- Calculates the total square footage of two bedrooms -/
def total_square_footage (martha_bedroom : ℕ) (jenny_bedroom_difference : ℕ) : ℕ :=
  martha_bedroom + (martha_bedroom + jenny_bedroom_difference)

/-- Proves that the total square footage of two bedrooms is 300 square feet -/
theorem two_bedroom_square_footage :
  total_square_footage 120 60 = 300 := by
  sorry

end two_bedroom_square_footage_l2979_297993


namespace unique_number_digit_sum_l2979_297952

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem unique_number_digit_sum :
  ∃! N : ℕ, 400 < N ∧ N < 600 ∧ N % 2 = 1 ∧ N % 5 = 0 ∧ N % 11 = 0 ∧ sumOfDigits N = 18 :=
by
  sorry

end unique_number_digit_sum_l2979_297952


namespace remainder_of_large_number_l2979_297947

theorem remainder_of_large_number (N : ℕ) (d : ℕ) (h : N = 9876543210123456789 ∧ d = 252) :
  N % d = 27 := by
  sorry

end remainder_of_large_number_l2979_297947


namespace investment_proof_l2979_297936

-- Define the interest rates
def interest_rate_x : ℚ := 23 / 100
def interest_rate_y : ℚ := 17 / 100

-- Define the investment in fund X
def investment_x : ℚ := 42000

-- Define the interest difference
def interest_difference : ℚ := 200

-- Define the total investment
def total_investment : ℚ := 100000

-- Theorem statement
theorem investment_proof :
  ∃ (investment_y : ℚ),
    investment_y * interest_rate_y = investment_x * interest_rate_x + interest_difference ∧
    investment_x + investment_y = total_investment :=
by
  sorry


end investment_proof_l2979_297936


namespace divisible_by_six_l2979_297922

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end divisible_by_six_l2979_297922


namespace square_land_side_length_l2979_297909

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 900) :
  ∃ (side : ℝ), side * side = area ∧ side = 30 := by
  sorry

end square_land_side_length_l2979_297909


namespace difference_of_ones_and_zeros_313_l2979_297926

/-- The number of zeros in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- The number of ones in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

theorem difference_of_ones_and_zeros_313 : 
  count_ones 313 - count_zeros 313 = 3 := by sorry

end difference_of_ones_and_zeros_313_l2979_297926


namespace brennan_pepper_proof_l2979_297977

/-- The amount of pepper Brennan used (in grams) -/
def pepper_used : ℝ := 0.16

/-- The amount of pepper Brennan has left (in grams) -/
def pepper_left : ℝ := 0.09

/-- The initial amount of pepper Brennan had (in grams) -/
def initial_pepper : ℝ := pepper_used + pepper_left

theorem brennan_pepper_proof : initial_pepper = 0.25 := by
  sorry

end brennan_pepper_proof_l2979_297977


namespace bill_sue_score_ratio_l2979_297945

theorem bill_sue_score_ratio :
  ∀ (john_score sue_score : ℕ),
    45 = john_score + 20 →
    45 + john_score + sue_score = 160 →
    (45 : ℚ) / sue_score = 1 / 2 := by
  sorry

end bill_sue_score_ratio_l2979_297945


namespace find_a_l2979_297942

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem find_a : 
  (∀ x : ℝ, f (1/2 * x - 1) = 2*x - 5) → 
  f (7/4) = 6 := by sorry

end find_a_l2979_297942


namespace rectangle_area_l2979_297925

theorem rectangle_area (w l : ℕ) : 
  (2 * (w + l) = 60) →  -- Perimeter is 60 units
  (l = w + 1) →         -- Length and width are consecutive integers
  (w * l = 210)         -- Area is 210 square units
:= by sorry

end rectangle_area_l2979_297925


namespace cartesian_points_proof_l2979_297966

/-- Given points P and Q in the Cartesian coordinate system, prove cos 2θ and sin(α + β) -/
theorem cartesian_points_proof (θ α β : Real) : 
  let P : Real × Real := (1/2, Real.cos θ ^ 2)
  let Q : Real × Real := (Real.sin θ ^ 2, -1)
  (P.1 * Q.1 + P.2 * Q.2 = -1/2) → 
  (Real.cos (2 * θ) = 1/3 ∧ 
   Real.sin (α + β) = -Real.sqrt 10 / 10) := by
sorry

end cartesian_points_proof_l2979_297966


namespace imaginary_part_of_complex_fraction_l2979_297934

theorem imaginary_part_of_complex_fraction :
  Complex.im (2 / (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l2979_297934


namespace total_worth_is_22800_l2979_297904

def engagement_ring_cost : ℝ := 4000
def car_cost : ℝ := 2000
def diamond_bracelet_cost : ℝ := 2 * engagement_ring_cost
def designer_gown_cost : ℝ := 0.5 * diamond_bracelet_cost
def jewelry_set_cost : ℝ := 1.2 * engagement_ring_cost

def total_worth : ℝ := engagement_ring_cost + car_cost + diamond_bracelet_cost + designer_gown_cost + jewelry_set_cost

theorem total_worth_is_22800 : total_worth = 22800 := by sorry

end total_worth_is_22800_l2979_297904


namespace unique_solution_exponential_equation_l2979_297908

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x + 2) * (4 : ℝ)^(2*x + 4) = (8 : ℝ)^(3*x + 4) ∧ x = -2 := by
  sorry

end unique_solution_exponential_equation_l2979_297908


namespace calculator_sale_loss_l2979_297917

theorem calculator_sale_loss : 
  ∀ (x y : ℝ),
    x * (1 + 0.2) = 60 →
    y * (1 - 0.2) = 60 →
    60 + 60 - (x + y) = -5 :=
by
  sorry

end calculator_sale_loss_l2979_297917


namespace intersection_condition_l2979_297960

theorem intersection_condition (a : ℝ) : 
  let M := {x : ℝ | x - a = 0}
  let N := {x : ℝ | a * x - 1 = 0}
  (M ∩ N = N) → (a = 0 ∨ a = 1 ∨ a = -1) :=
by sorry

end intersection_condition_l2979_297960


namespace max_xy_perpendicular_vectors_l2979_297915

theorem max_xy_perpendicular_vectors (x y : ℝ) :
  let a : ℝ × ℝ := (1, x - 1)
  let b : ℝ × ℝ := (y, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  ∃ (m : ℝ), (∀ (x' y' : ℝ), 
    let a' : ℝ × ℝ := (1, x' - 1)
    let b' : ℝ × ℝ := (y', 2)
    (a'.1 * b'.1 + a'.2 * b'.2 = 0) → x' * y' ≤ m) ∧
  m = (1/2 : ℝ) :=
by sorry

end max_xy_perpendicular_vectors_l2979_297915


namespace geometric_progression_properties_l2979_297976

/-- Represents a geometric progression with given properties -/
structure GeometricProgression where
  ratio : ℚ
  fourthTerm : ℚ
  sum : ℚ

/-- The number of terms in the geometric progression -/
def numTerms (gp : GeometricProgression) : ℕ := sorry

/-- Theorem stating the properties of the specific geometric progression -/
theorem geometric_progression_properties :
  ∃ (gp : GeometricProgression),
    gp.ratio = 1/3 ∧
    gp.fourthTerm = 1/54 ∧
    gp.sum = 121/162 ∧
    numTerms gp = 5 := by sorry

end geometric_progression_properties_l2979_297976


namespace total_money_sum_l2979_297902

theorem total_money_sum (J : ℕ) : 
  (3 * J = 60) → 
  (J + 3 * J + (2 * J - 7) = 113) := by
  sorry

end total_money_sum_l2979_297902


namespace sequence_properties_l2979_297973

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := 2 * n + 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℝ := 2^n

-- Define the sum of the first n terms of a_n + b_n
def S (n : ℕ) : ℝ := n^2 + 2*n + 2^(n+1) - 2

theorem sequence_properties :
  (a 2 = 5) ∧
  (a 1 + a 4 = 12) ∧
  (∀ n, b n > 0) ∧
  (∀ n, b n * b (n+1) = 2^(a n)) ∧
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b (n+1) / b n = 2) ∧
  (∀ n, S n = n^2 + 2*n + 2^(n+1) - 2) :=
by sorry


end sequence_properties_l2979_297973


namespace problem_statement_l2979_297967

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -2) :
  (1 - x) * (1 - y) = -3 := by
  sorry

end problem_statement_l2979_297967


namespace seventh_term_is_eight_l2979_297988

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, sum n = (n : ℝ) * (a 1 + a n) / 2
  third_eighth_sum : a 3 + a 8 = 13
  seventh_sum : sum 7 = 35

/-- The main theorem stating that the 7th term of the sequence is 8 -/
theorem seventh_term_is_eight (seq : ArithmeticSequence) : seq.a 7 = 8 := by
  sorry

end seventh_term_is_eight_l2979_297988


namespace parabola_directrix_l2979_297906

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -25 / 12

/-- Theorem: The directrix of the given parabola has the equation y = -25/12 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end parabola_directrix_l2979_297906


namespace profit_calculation_l2979_297948

theorem profit_calculation (x : ℝ) 
  (h1 : 20 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 16 := by
  sorry

end profit_calculation_l2979_297948


namespace log_sum_equals_zero_l2979_297937

theorem log_sum_equals_zero :
  Real.log 2 + Real.log 5 + Real.log 0.5 / Real.log 2 = 0 := by
  sorry

end log_sum_equals_zero_l2979_297937


namespace quadratic_roots_relation_l2979_297900

/-- Given a quadratic function f(x) = x^2 - ax - b with roots 2 and 3,
    prove that g(x) = bx^2 - ax - 1 has roots -1/2 and -1/3 -/
theorem quadratic_roots_relation (a b : ℝ) : 
  (∀ x, x^2 - a*x - b = 0 ↔ x = 2 ∨ x = 3) →
  (∀ x, b*x^2 - a*x - 1 = 0 ↔ x = -1/2 ∨ x = -1/3) :=
sorry

end quadratic_roots_relation_l2979_297900


namespace class_mean_calculation_l2979_297991

/-- Calculates the overall mean score for a class given two groups of students and their respective mean scores -/
theorem class_mean_calculation 
  (total_students : ℕ) 
  (group1_students : ℕ) 
  (group2_students : ℕ) 
  (group1_mean : ℚ) 
  (group2_mean : ℚ) 
  (h1 : total_students = group1_students + group2_students)
  (h2 : total_students = 32)
  (h3 : group1_students = 24)
  (h4 : group2_students = 8)
  (h5 : group1_mean = 85 / 100)
  (h6 : group2_mean = 90 / 100) :
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 8625 / 10000 := by
  sorry


end class_mean_calculation_l2979_297991


namespace simplify_and_evaluate_l2979_297927

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_and_evaluate_l2979_297927


namespace sin_2x_value_l2979_297913

theorem sin_2x_value (x : ℝ) : 
  (Real.cos (4 * π / 5) * Real.cos (7 * π / 15) - Real.sin (9 * π / 5) * Real.sin (7 * π / 15) = 
   Real.cos (x + π / 2) * Real.cos x + 2 / 3) → 
  Real.sin (2 * x) = 1 / 3 := by
sorry

end sin_2x_value_l2979_297913


namespace vanya_number_l2979_297954

theorem vanya_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  let m := n / 10
  let d := n % 10
  (10 * d + m)^2 = 4 * n :=
by
  -- The proof would go here
  sorry

end vanya_number_l2979_297954


namespace range_of_m_l2979_297959

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| ≥ m + 4) → m ≤ -4 := by
  sorry

end range_of_m_l2979_297959


namespace green_square_coincidence_l2979_297996

/-- Represents a half of the figure -/
structure HalfFigure where
  greenSquares : ℕ
  redTriangles : ℕ
  blueTriangles : ℕ

/-- Represents the folded figure -/
structure FoldedFigure where
  coincidingGreenSquares : ℕ
  coincidingRedTrianglePairs : ℕ
  coincidingBlueTrianglePairs : ℕ
  redBluePairs : ℕ

/-- The theorem to be proved -/
theorem green_square_coincidence 
  (half : HalfFigure) 
  (folded : FoldedFigure) : 
  half.greenSquares = 4 ∧ 
  half.redTriangles = 3 ∧ 
  half.blueTriangles = 6 ∧
  folded.coincidingRedTrianglePairs = 2 ∧
  folded.coincidingBlueTrianglePairs = 2 ∧
  folded.redBluePairs = 3 →
  folded.coincidingGreenSquares = half.greenSquares :=
by sorry

end green_square_coincidence_l2979_297996


namespace pablo_puzzle_pieces_l2979_297992

/-- The number of pieces Pablo can put together per hour -/
def piecesPerHour : ℕ := 100

/-- The number of 300-piece puzzles Pablo has -/
def numLargePuzzles : ℕ := 8

/-- The number of puzzles with unknown pieces Pablo has -/
def numSmallPuzzles : ℕ := 5

/-- The maximum number of hours Pablo works on puzzles per day -/
def hoursPerDay : ℕ := 7

/-- The number of days it takes Pablo to complete all puzzles -/
def totalDays : ℕ := 7

/-- The number of pieces in each of the large puzzles -/
def piecesPerLargePuzzle : ℕ := 300

/-- The number of pieces in each of the small puzzles -/
def piecesPerSmallPuzzle : ℕ := 500

theorem pablo_puzzle_pieces :
  piecesPerSmallPuzzle * numSmallPuzzles + piecesPerLargePuzzle * numLargePuzzles = 
  piecesPerHour * hoursPerDay * totalDays :=
by sorry

end pablo_puzzle_pieces_l2979_297992


namespace y_value_l2979_297961

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = 48 := by
  sorry

end y_value_l2979_297961


namespace unique_solution_cubic_linear_l2979_297935

/-- The system of equations y = x^3 and y = 4x + m has exactly one real solution if and only if m = -8 -/
theorem unique_solution_cubic_linear (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^3 = 4*p.1 + m ∧ p.2 = p.1^3) ↔ m = -8 :=
sorry

end unique_solution_cubic_linear_l2979_297935


namespace library_book_count_l2979_297944

/-- The number of books in a library after two years of purchases -/
def total_books (initial : ℕ) (last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial + last_year + multiplier * last_year

/-- Theorem: The library now has 300 books -/
theorem library_book_count : total_books 100 50 3 = 300 := by
  sorry

end library_book_count_l2979_297944


namespace dogsled_race_speed_difference_l2979_297912

/-- Proves that the difference in average speeds between two teams is 5 mph
    given specific conditions of a dogsled race. -/
theorem dogsled_race_speed_difference
  (course_length : ℝ)
  (team_r_speed : ℝ)
  (time_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : team_r_speed = 20)
  (h3 : time_difference = 3)
  : ∃ (team_a_speed : ℝ),
    team_a_speed = course_length / (course_length / team_r_speed - time_difference) ∧
    team_a_speed - team_r_speed = 5 := by
  sorry

end dogsled_race_speed_difference_l2979_297912


namespace smallest_sum_pell_equation_l2979_297974

theorem smallest_sum_pell_equation :
  ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ x^2 - 29*y^2 = 1 ∧
  ∀ (x' y' : ℕ), x' ≥ 1 → y' ≥ 1 → x'^2 - 29*y'^2 = 1 → x + y ≤ x' + y' ∧
  x + y = 11621 :=
by sorry

end smallest_sum_pell_equation_l2979_297974


namespace joggers_meeting_l2979_297940

def lap_time_cathy : ℕ := 5
def lap_time_david : ℕ := 9
def lap_time_elena : ℕ := 8

def meeting_time : ℕ := 360
def cathy_laps : ℕ := 72

theorem joggers_meeting :
  (meeting_time % lap_time_cathy = 0) ∧
  (meeting_time % lap_time_david = 0) ∧
  (meeting_time % lap_time_elena = 0) ∧
  (∀ t : ℕ, t < meeting_time →
    ¬(t % lap_time_cathy = 0 ∧ t % lap_time_david = 0 ∧ t % lap_time_elena = 0)) ∧
  (cathy_laps = meeting_time / lap_time_cathy) :=
by sorry

end joggers_meeting_l2979_297940


namespace polynomial_identity_sum_l2979_297916

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁ * x + d₁) * (x^2 + a₂ * x + d₂) * (x^2 + a₃ * x + d₃)) →
  a₁ * d₁ + a₂ * d₂ + a₃ * d₃ = 1 := by
  sorry

end polynomial_identity_sum_l2979_297916


namespace rachel_milk_consumption_l2979_297994

theorem rachel_milk_consumption (don_milk : ℚ) (rachel_fraction : ℚ) : 
  don_milk = 1/5 → rachel_fraction = 2/3 → rachel_fraction * don_milk = 2/15 := by
  sorry

end rachel_milk_consumption_l2979_297994


namespace fraction_decimal_digits_l2979_297938

theorem fraction_decimal_digits : 
  let f : ℚ := 90 / (3^2 * 2^5)
  ∃ (d : ℕ) (n : ℕ), f = d.cast / 10^n ∧ (d % 10 ≠ 0) ∧ n = 4 :=
by sorry

end fraction_decimal_digits_l2979_297938


namespace max_d_value_l2979_297923

def a (n : ℕ) : ℕ := 150 + (n + 1)^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (k : ℕ), d k = 2 ∧ ∀ (n : ℕ), d n ≤ 2 :=
sorry

end max_d_value_l2979_297923


namespace penny_excess_purchase_l2979_297970

/-- Calculates the excess pounds of honey purchased above the minimum spend -/
def excess_honey_purchased (bulk_price : ℚ) (min_spend : ℚ) (tax_per_pound : ℚ) (total_paid : ℚ) : ℚ :=
  let total_price_per_pound := bulk_price + tax_per_pound
  let pounds_purchased := total_paid / total_price_per_pound
  let min_pounds := min_spend / bulk_price
  pounds_purchased - min_pounds

/-- Theorem stating that Penny's purchase exceeded the minimum spend by 32 pounds -/
theorem penny_excess_purchase :
  excess_honey_purchased 5 40 1 240 = 32 := by
  sorry

end penny_excess_purchase_l2979_297970


namespace one_and_two_thirds_of_number_is_45_l2979_297990

theorem one_and_two_thirds_of_number_is_45 : ∃ x : ℚ, (5 / 3) * x = 45 ∧ x = 27 := by sorry

end one_and_two_thirds_of_number_is_45_l2979_297990


namespace cyclic_system_solutions_l2979_297980

def cyclicSystem (x : Fin 5 → ℝ) (y : ℝ) : Prop :=
  ∀ i : Fin 5, x i + x ((i + 2) % 5) = y * x ((i + 1) % 5)

theorem cyclic_system_solutions :
  ∀ x : Fin 5 → ℝ, ∀ y : ℝ,
    cyclicSystem x y ↔
      ((∀ i : Fin 5, x i = 0) ∨
      (y = 2 ∧ ∃ s : ℝ, ∀ i : Fin 5, x i = s) ∨
      ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
        ∃ s t : ℝ, x 0 = s ∧ x 1 = t ∧ x 2 = -s + y*t ∧ x 3 = -y*s - t ∧ x 4 = y*s - t)) :=
by
  sorry


end cyclic_system_solutions_l2979_297980


namespace quadratic_roots_square_relation_l2979_297963

theorem quadratic_roots_square_relation (q : ℝ) : 
  (∃ (a b : ℝ), a ≠ b ∧ a^2 = b ∧ a^2 - 12*a + q = 0 ∧ b^2 - 12*b + q = 0) →
  (q = 27 ∨ q = -64) :=
by sorry

end quadratic_roots_square_relation_l2979_297963


namespace right_triangle_area_l2979_297932

/-- The area of a right triangle with hypotenuse 5 and shortest side 3 is 6 -/
theorem right_triangle_area : ∀ (a b c : ℝ),
  a = 3 →
  c = 5 →
  a ≤ b →
  b ≤ c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 6 := by
  sorry

end right_triangle_area_l2979_297932


namespace nine_nines_squared_zeros_l2979_297987

/-- The number of nines in 9,999,999 -/
def n : ℕ := 7

/-- The number 9,999,999 -/
def x : ℕ := 10^n - 1

/-- The number of zeros at the end of x^2 -/
def num_zeros (x : ℕ) : ℕ := n - 1

theorem nine_nines_squared_zeros :
  num_zeros x = 6 :=
sorry

end nine_nines_squared_zeros_l2979_297987


namespace sum_of_a_and_b_l2979_297929

theorem sum_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x-a)/(x-b) > 0 ↔ x ∈ Set.Ioi 4 ∪ Set.Iic 1) → 
  a + b = 5 := by
sorry

end sum_of_a_and_b_l2979_297929


namespace cube_sum_implies_sum_l2979_297931

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cube_sum_implies_sum_l2979_297931


namespace physics_class_size_l2979_297975

theorem physics_class_size 
  (total_students : ℕ) 
  (physics_students : ℕ) 
  (math_students : ℕ) 
  (both_subjects : ℕ) :
  total_students = 100 →
  physics_students = math_students + both_subjects →
  physics_students = 2 * math_students →
  both_subjects = 10 →
  physics_students = 62 := by
sorry

end physics_class_size_l2979_297975


namespace rectangle_count_l2979_297949

theorem rectangle_count (h v : ℕ) (h_eq : h = 5) (v_eq : v = 5) :
  (Nat.choose h 2) * (Nat.choose v 2) = 100 := by
  sorry

end rectangle_count_l2979_297949
