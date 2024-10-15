import Mathlib

namespace NUMINAMATH_CALUDE_floss_leftover_and_cost_l4009_400931

/-- Represents the floss requirements for a class -/
structure ClassFlossRequirement where
  students : ℕ
  flossPerStudent : ℚ

/-- Represents the floss sale conditions -/
structure FlossSaleConditions where
  metersPerPacket : ℚ
  pricePerPacket : ℚ
  discountRate : ℚ
  discountThreshold : ℕ

def yardToMeter : ℚ := 0.9144

def classes : List ClassFlossRequirement := [
  ⟨20, 1.5⟩,
  ⟨25, 1.75⟩,
  ⟨30, 2⟩
]

def saleConditions : FlossSaleConditions := {
  metersPerPacket := 50,
  pricePerPacket := 5,
  discountRate := 0.1,
  discountThreshold := 2
}

theorem floss_leftover_and_cost 
  (classes : List ClassFlossRequirement) 
  (saleConditions : FlossSaleConditions) 
  (yardToMeter : ℚ) : 
  ∃ (cost leftover : ℚ), cost = 14.5 ∧ leftover = 27.737 := by
  sorry

end NUMINAMATH_CALUDE_floss_leftover_and_cost_l4009_400931


namespace NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l4009_400981

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 47 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l4009_400981


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_186_l4009_400955

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement
theorem f_g_f_3_equals_186 : f (g (f 3)) = 186 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_186_l4009_400955


namespace NUMINAMATH_CALUDE_jemma_price_calculation_l4009_400988

/-- The price at which Jemma sells each frame -/
def jemma_price : ℝ := 5

/-- The number of frames Jemma sold -/
def jemma_frames : ℕ := 400

/-- The total revenue made by both Jemma and Dorothy -/
def total_revenue : ℝ := 2500

theorem jemma_price_calculation :
  (jemma_price * jemma_frames : ℝ) + 
  (jemma_price / 2 * (jemma_frames / 2) : ℝ) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_jemma_price_calculation_l4009_400988


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l4009_400920

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), (∀ x ∈ S, ∃ c d : ℕ+, (Nat.gcd c d * Nat.lcm c d = 360 ∧ Nat.gcd c d = x)) ∧ 
                      (∀ y : ℕ, (∃ e f : ℕ+, (Nat.gcd e f * Nat.lcm e f = 360 ∧ Nat.gcd e f = y)) → y ∈ S) ∧
                      S.card = 12) := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l4009_400920


namespace NUMINAMATH_CALUDE_bob_puppy_savings_l4009_400984

/-- The minimum number of additional weeks Bob must win first place to buy a puppy -/
def minimum_additional_weeks (initial_weeks : ℕ) (prize_per_week : ℕ) (puppy_cost : ℕ) : ℕ :=
  let initial_earnings := initial_weeks * prize_per_week
  let remaining_cost := puppy_cost - initial_earnings
  (remaining_cost + prize_per_week - 1) / prize_per_week

theorem bob_puppy_savings : minimum_additional_weeks 2 100 1000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bob_puppy_savings_l4009_400984


namespace NUMINAMATH_CALUDE_belize_homes_count_belize_homes_count_proof_l4009_400941

theorem belize_homes_count : ℕ → Prop :=
  fun total_homes =>
    let white_homes := total_homes / 4
    let non_white_homes := total_homes - white_homes
    let non_white_homes_with_fireplace := non_white_homes / 5
    let non_white_homes_without_fireplace := non_white_homes - non_white_homes_with_fireplace
    non_white_homes_without_fireplace = 240 → total_homes = 400

-- Proof
theorem belize_homes_count_proof : ∃ (total_homes : ℕ), belize_homes_count total_homes :=
  sorry

end NUMINAMATH_CALUDE_belize_homes_count_belize_homes_count_proof_l4009_400941


namespace NUMINAMATH_CALUDE_part_one_part_two_l4009_400927

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - 3

-- Part I
theorem part_one (m : ℝ) :
  (∀ x, f m x ≥ 0 ↔ x ≤ -2 ∨ x ≥ 4) → m = 1 := by sorry

-- Part II
theorem part_two :
  ∀ t, (∃ x, f 1 x ≥ t + |2 - x|) → t ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4009_400927


namespace NUMINAMATH_CALUDE_intersection_polar_radius_l4009_400969

-- Define the line l
def line_l (x : ℝ) : ℝ := x + 1

-- Define the curve C in polar form
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 - 4 * Real.cos θ = 0 ∧ ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem intersection_polar_radius :
  ∃ (x y ρ θ : ℝ),
    y = line_l x ∧
    curve_C ρ θ ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ ∧
    ρ = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_intersection_polar_radius_l4009_400969


namespace NUMINAMATH_CALUDE_muffin_cost_is_two_l4009_400901

/-- The cost of a muffin given the conditions of Francis and Kiera's breakfast -/
def muffin_cost : ℝ :=
  let fruit_cup_cost : ℝ := 3
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let total_cost : ℝ := 17
  2

theorem muffin_cost_is_two :
  let fruit_cup_cost : ℝ := 3
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let total_cost : ℝ := 17
  muffin_cost = 2 ∧
  (francis_muffins + kiera_muffins : ℝ) * muffin_cost +
    (francis_fruit_cups + kiera_fruit_cups : ℝ) * fruit_cup_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_is_two_l4009_400901


namespace NUMINAMATH_CALUDE_scaling_transformation_result_l4009_400909

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 3 * Real.sin (2 * x)

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

-- State the theorem
theorem scaling_transformation_result :
  ∀ (x y x' y' : ℝ),
  original_curve x y →
  scaling_transformation x y x' y' →
  y' = 9 * Real.sin x' := by sorry

end NUMINAMATH_CALUDE_scaling_transformation_result_l4009_400909


namespace NUMINAMATH_CALUDE_pears_left_l4009_400902

theorem pears_left (jason_pears keith_pears mike_ate : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_ate = 12) :
  jason_pears + keith_pears - mike_ate = 81 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_l4009_400902


namespace NUMINAMATH_CALUDE_savings_distribution_l4009_400951

/-- Represents the savings and debt problem of Tamara, Nora, and Lulu -/
theorem savings_distribution (debt : ℕ) (lulu_savings : ℕ) : 
  debt = 40 →
  lulu_savings = 6 →
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  let remainder := total_savings - debt
  remainder / 3 = 2 := by sorry

end NUMINAMATH_CALUDE_savings_distribution_l4009_400951


namespace NUMINAMATH_CALUDE_no_common_solution_l4009_400992

theorem no_common_solution : ¬∃ x : ℝ, (263 - x = 108) ∧ (25 * x = 1950) ∧ (x / 15 = 64) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l4009_400992


namespace NUMINAMATH_CALUDE_max_value_theorem_l4009_400912

theorem max_value_theorem (x y : ℝ) (h : x + y = 5) :
  ∃ (max : ℝ), max = 1175 / 16 ∧
  ∀ (a b : ℝ), a + b = 5 → a^3 * b + a^2 * b + a * b + a * b^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4009_400912


namespace NUMINAMATH_CALUDE_factorization_proof_l4009_400938

theorem factorization_proof (x : ℝ) : 15 * x^2 + 10 * x - 5 = 5 * (3 * x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l4009_400938


namespace NUMINAMATH_CALUDE_midpoint_parallelogram_area_ratio_l4009_400970

/-- Given a parallelogram, the area of the parallelogram formed by joining its midpoints is 1/4 of the original area -/
theorem midpoint_parallelogram_area_ratio (P : ℝ) (h : P > 0) :
  ∃ (smaller_area : ℝ), smaller_area = P / 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_parallelogram_area_ratio_l4009_400970


namespace NUMINAMATH_CALUDE_point_on_extension_line_l4009_400919

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (O A B P : V)

-- Define the conditions
variable (h_not_collinear : ¬Collinear ℝ {O, A, B})
variable (h_vector_equation : (2 : ℝ) • (P - O) = (2 : ℝ) • (A - O) + (2 : ℝ) • (B - O))

-- Theorem statement
theorem point_on_extension_line :
  ∃ (t : ℝ), t < 0 ∧ P = A + t • (B - A) :=
sorry

end NUMINAMATH_CALUDE_point_on_extension_line_l4009_400919


namespace NUMINAMATH_CALUDE_olivia_cookies_l4009_400906

/-- The number of chocolate chip cookies Olivia has -/
def chocolate_chip_cookies (cookies_per_bag : ℕ) (oatmeal_cookies : ℕ) (baggies : ℕ) : ℕ :=
  cookies_per_bag * baggies - oatmeal_cookies

/-- Proof that Olivia has 13 chocolate chip cookies -/
theorem olivia_cookies : chocolate_chip_cookies 9 41 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_olivia_cookies_l4009_400906


namespace NUMINAMATH_CALUDE_investment_percentage_proof_l4009_400904

theorem investment_percentage_proof (total_sum P1 P2 x : ℝ) : 
  total_sum = 1600 →
  P1 + P2 = total_sum →
  P2 = 1100 →
  (P1 * x / 100) + (P2 * 5 / 100) = 85 →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_proof_l4009_400904


namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l4009_400962

/-- Given an acute triangle ABC with side lengths a = 2 and b = 3, 
    prove that the side length c satisfies √5 < c < √13 -/
theorem acute_triangle_side_range (a b c : ℝ) : 
  a = 2 → b = 3 → 
  (a^2 + b^2 > c^2) → (a^2 + c^2 > b^2) → (b^2 + c^2 > a^2) →
  Real.sqrt 5 < c ∧ c < Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l4009_400962


namespace NUMINAMATH_CALUDE_rachel_final_lives_l4009_400965

/-- Calculates the total number of lives after losing and gaining lives in a video game. -/
def totalLives (initialLives livesLost livesGained : ℕ) : ℕ :=
  initialLives - livesLost + livesGained

/-- Proves that given the initial conditions, Rachel ends up with 32 lives. -/
theorem rachel_final_lives :
  totalLives 10 4 26 = 32 := by
  sorry

#eval totalLives 10 4 26

end NUMINAMATH_CALUDE_rachel_final_lives_l4009_400965


namespace NUMINAMATH_CALUDE_expression_evaluation_l4009_400939

theorem expression_evaluation : 
  (123 - (45 * (9 - 6) - 78)) + (0 / 1994) = 66 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4009_400939


namespace NUMINAMATH_CALUDE_combinations_of_three_from_seven_l4009_400929

theorem combinations_of_three_from_seven (n k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_combinations_of_three_from_seven_l4009_400929


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l4009_400913

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 100 + 5 / 1000 + 8 / 10000 + 2 / 100000 = 0.03582 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l4009_400913


namespace NUMINAMATH_CALUDE_remainder_3_pow_1999_mod_13_l4009_400932

theorem remainder_3_pow_1999_mod_13 : 3^1999 % 13 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_3_pow_1999_mod_13_l4009_400932


namespace NUMINAMATH_CALUDE_all_T_divisible_by_4_l4009_400961

/-- The set of all numbers which are the sum of the squares of four consecutive integers
    added to the sum of the integers themselves. -/
def T : Set ℤ :=
  {x | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n-1) + n + (n+1) + (n+2)}

/-- All members of set T are divisible by 4. -/
theorem all_T_divisible_by_4 : ∀ x ∈ T, 4 ∣ x := by sorry

end NUMINAMATH_CALUDE_all_T_divisible_by_4_l4009_400961


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l4009_400945

/-- The maximum number of 27 cubic centimetre cubes that can fit in a rectangular box -/
def max_cubes (l w h : ℕ) (cube_volume : ℕ) : ℕ :=
  (l * w * h) / cube_volume

/-- Theorem: The maximum number of 27 cubic centimetre cubes that can fit in a 
    rectangular box measuring 8 cm x 9 cm x 12 cm is 32 -/
theorem max_cubes_in_box : max_cubes 8 9 12 27 = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_in_box_l4009_400945


namespace NUMINAMATH_CALUDE_focus_coincidence_l4009_400996

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (5*x^2)/3 - (5*y^2)/2 = 1

-- Define the focus of a parabola
def parabola_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the right focus of a hyperbola
def hyperbola_right_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Theorem statement
theorem focus_coincidence :
  ∀ (x y : ℝ), parabola_focus x y ↔ hyperbola_right_focus x y :=
sorry

end NUMINAMATH_CALUDE_focus_coincidence_l4009_400996


namespace NUMINAMATH_CALUDE_total_students_l4009_400975

theorem total_students (french : ℕ) (spanish : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 5)
  (h2 : spanish = 10)
  (h3 : both = 4)
  (h4 : neither = 13) :
  french + spanish + both + neither = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l4009_400975


namespace NUMINAMATH_CALUDE_student_B_more_stable_l4009_400985

/-- Represents a student's performance metrics -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if the first student has more stable performance than the second -/
def more_stable (s1 s2 : StudentPerformance) : Prop :=
  s1.average_score = s2.average_score ∧ s1.variance < s2.variance

/-- The performance metrics for student A -/
def student_A : StudentPerformance :=
  { average_score := 82
    variance := 245 }

/-- The performance metrics for student B -/
def student_B : StudentPerformance :=
  { average_score := 82
    variance := 190 }

/-- Theorem stating that student B has more stable performance than student A -/
theorem student_B_more_stable : more_stable student_B student_A := by
  sorry

end NUMINAMATH_CALUDE_student_B_more_stable_l4009_400985


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l4009_400993

/-- The system of equations -/
def system_equations (x y m : ℝ) : Prop :=
  2 * x - y = m ∧ 3 * x + 2 * y = m + 7

/-- Point in second quadrant with given distances -/
def point_conditions (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0 ∧ y = 3 ∧ x = -2

theorem solution_part1 :
  ∃ x y : ℝ, system_equations x y 0 ∧ x = 1 ∧ y = 2 := by sorry

theorem solution_part2 :
  ∀ x y : ℝ, system_equations x y (-7) ∧ point_conditions x y := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l4009_400993


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l4009_400918

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  m * x - y + 2 * m + 1 = 0

-- Theorem statement
theorem fixed_point_on_line :
  ∀ m : ℝ, line_equation m (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l4009_400918


namespace NUMINAMATH_CALUDE_punch_difference_l4009_400928

def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def total_punch : ℝ := 21

def apple_juice : ℝ := total_punch - orange_punch - cherry_punch

theorem punch_difference : cherry_punch - apple_juice = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_punch_difference_l4009_400928


namespace NUMINAMATH_CALUDE_product_expansion_sum_l4009_400958

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (2*x^2 - 3*x + 5)*(5 - x) = a*x^3 + b*x^2 + c*x + d) →
  a + b + c + d = 16 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l4009_400958


namespace NUMINAMATH_CALUDE_line_through_point_with_opposite_intercepts_l4009_400998

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has opposite intercepts
def hasOppositeIntercepts (l : Line) : Prop :=
  (l.a ≠ 0 ∧ l.b ≠ 0) ∧ (l.c / l.a) * (l.c / l.b) < 0

-- Theorem statement
theorem line_through_point_with_opposite_intercepts :
  ∀ (l : Line),
    passesThrough l {x := 2, y := 3} →
    hasOppositeIntercepts l →
    (∃ (k : ℝ), l.a = k ∧ l.b = -k ∧ l.c = k) ∨
    (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_opposite_intercepts_l4009_400998


namespace NUMINAMATH_CALUDE_num_distinguishable_triangles_eq_960_l4009_400995

/-- Represents the number of colors available for the small triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles needed to form a large triangle -/
def triangles_per_large : ℕ := 4

/-- Represents the number of corner triangles in a large triangle -/
def num_corners : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  ((num_colors + 
    num_colors * (num_colors - 1) + 
    choose num_colors num_corners) * num_colors)

/-- The main theorem stating the number of distinguishable large triangles -/
theorem num_distinguishable_triangles_eq_960 :
  num_distinguishable_triangles = 960 := by sorry

end NUMINAMATH_CALUDE_num_distinguishable_triangles_eq_960_l4009_400995


namespace NUMINAMATH_CALUDE_percentage_problem_l4009_400986

theorem percentage_problem (N P : ℝ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4009_400986


namespace NUMINAMATH_CALUDE_min_travel_time_less_than_3_9_l4009_400916

/-- Represents the problem of three people traveling with a motorcycle --/
structure TravelProblem where
  distance : ℝ
  walkSpeed : ℝ
  motorSpeed : ℝ
  motorCapacity : ℕ

/-- Calculates the minimum time for all three people to reach the destination --/
def minTravelTime (p : TravelProblem) : ℝ :=
  sorry

/-- The main theorem stating that the minimum travel time is less than 3.9 hours --/
theorem min_travel_time_less_than_3_9 :
  let p : TravelProblem := {
    distance := 135,
    walkSpeed := 6,
    motorSpeed := 90,
    motorCapacity := 2
  }
  minTravelTime p < 3.9 := by
  sorry

end NUMINAMATH_CALUDE_min_travel_time_less_than_3_9_l4009_400916


namespace NUMINAMATH_CALUDE_domain_exact_domain_contains_l4009_400968

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + x - a

-- Theorem 1: When the domain is exactly (-2, 3), a = -6
theorem domain_exact (a : ℝ) : 
  (∀ x, -2 < x ∧ x < 3 ↔ f a x > 0) → a = -6 :=
sorry

-- Theorem 2: When the domain contains (-2, 3), a ≤ -6
theorem domain_contains (a : ℝ) :
  (∀ x, -2 < x ∧ x < 3 → f a x > 0) → a ≤ -6 :=
sorry

end NUMINAMATH_CALUDE_domain_exact_domain_contains_l4009_400968


namespace NUMINAMATH_CALUDE_teachers_survey_result_l4009_400978

def teachers_survey (total : ℕ) (high_bp : ℕ) (stress : ℕ) (both : ℕ) : Prop :=
  let neither := total - (high_bp + stress - both)
  (neither : ℚ) / total * 100 = 20

theorem teachers_survey_result : teachers_survey 150 90 60 30 := by
  sorry

end NUMINAMATH_CALUDE_teachers_survey_result_l4009_400978


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_neg_six_sqrt_three_l4009_400967

theorem sqrt_difference_equals_neg_six_sqrt_three :
  Real.sqrt ((5 - 3 * Real.sqrt 3)^2) - Real.sqrt ((5 + 3 * Real.sqrt 3)^2) = -6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_neg_six_sqrt_three_l4009_400967


namespace NUMINAMATH_CALUDE_work_completion_time_l4009_400994

/-- The time (in days) it takes for person a to complete the work alone -/
def time_a : ℝ := 90

/-- The time (in days) it takes for person b to complete the work alone -/
def time_b : ℝ := 45

/-- The time (in days) it takes for persons a, b, and c working together to complete the work -/
def time_together : ℝ := 5

/-- The time (in days) it takes for person c to complete the work alone -/
def time_c : ℝ := 6

/-- The theorem stating that given the work times for a, b, and the group, 
    the time for c to complete the work alone is 6 days -/
theorem work_completion_time :
  (1 / time_a + 1 / time_b + 1 / time_c = 1 / time_together) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4009_400994


namespace NUMINAMATH_CALUDE_total_apples_picked_l4009_400903

/-- The total number of apples picked by Mike, Nancy, and Keith is 16. -/
theorem total_apples_picked (mike_apples nancy_apples keith_apples : ℕ)
  (h1 : mike_apples = 7)
  (h2 : nancy_apples = 3)
  (h3 : keith_apples = 6) :
  mike_apples + nancy_apples + keith_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_picked_l4009_400903


namespace NUMINAMATH_CALUDE_a_divides_2b_l4009_400980

theorem a_divides_2b (a b : ℕ+) 
  (h : ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧ 
    ∀ (p : ℕ+ × ℕ+), p ∈ S → 
      ∃ (r s : ℕ+), (p.1.val ^ 2 + a.val * p.2.val + b.val = r.val ^ 2) ∧ 
                    (p.2.val ^ 2 + a.val * p.1.val + b.val = s.val ^ 2)) : 
  a.val ∣ (2 * b.val) :=
sorry

end NUMINAMATH_CALUDE_a_divides_2b_l4009_400980


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l4009_400934

theorem fixed_points_of_f (f : ℝ → ℝ) (hf : ∀ x, f x = 4 * x - x^2) :
  ∃ a b : ℝ, a ≠ b ∧ f a = b ∧ f b = a ∧
    ((a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
     (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l4009_400934


namespace NUMINAMATH_CALUDE_bernie_selection_probability_l4009_400952

theorem bernie_selection_probability 
  (p_carol : ℝ) 
  (p_both : ℝ) 
  (h1 : p_carol = 4/5)
  (h2 : p_both = 0.48)
  (h3 : p_both = p_carol * p_bernie)
  : p_bernie = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_bernie_selection_probability_l4009_400952


namespace NUMINAMATH_CALUDE_triangle_area_OAB_l4009_400973

/-- Given a line passing through (0, -2) that intersects the parabola y² = 16x at points A and B,
    where the y-coordinates of A and B satisfy y₁² - y₂² = 1, 
    prove that the area of triangle OAB (where O is the origin) is 1/16. -/
theorem triangle_area_OAB : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (∃ m : ℝ, x₁ = m * y₁ + 2 * m ∧ x₂ = m * y₂ + 2 * m) → -- Line equation
    y₁^2 = 16 * x₁ →                                      -- A satisfies parabola equation
    y₂^2 = 16 * x₂ →                                      -- B satisfies parabola equation
    y₁^2 - y₂^2 = 1 →                                     -- Given condition
    (1/2 : ℝ) * |x₁ * y₂ - x₂ * y₁| = 1/16 :=             -- Area of triangle OAB
by sorry

end NUMINAMATH_CALUDE_triangle_area_OAB_l4009_400973


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l4009_400991

-- Define a scalene triangle with prime side lengths
def ScaleneTriangleWithPrimeSides (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c

-- Define a function to check if the perimeter is prime
def HasPrimePerimeter (a b c : ℕ) : Prop :=
  Nat.Prime (a + b + c)

-- Theorem statement
theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    ScaleneTriangleWithPrimeSides a b c →
    HasPrimePerimeter a b c →
    a + b + c ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l4009_400991


namespace NUMINAMATH_CALUDE_net_profit_calculation_l4009_400935

/-- Calculates the net profit given the purchase price, markup, and overhead percentage. -/
def calculate_net_profit (purchase_price markup overhead_percent : ℚ) : ℚ :=
  let overhead := purchase_price * overhead_percent
  markup - overhead

/-- Theorem stating that given the specific values in the problem, the net profit is $40.60. -/
theorem net_profit_calculation :
  let purchase_price : ℚ := 48
  let markup : ℚ := 55
  let overhead_percent : ℚ := 0.30
  calculate_net_profit purchase_price markup overhead_percent = 40.60 := by
  sorry

#eval calculate_net_profit 48 55 0.30

end NUMINAMATH_CALUDE_net_profit_calculation_l4009_400935


namespace NUMINAMATH_CALUDE_M_intersect_N_is_empty_l4009_400950

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (2*x - x^2)}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_is_empty_l4009_400950


namespace NUMINAMATH_CALUDE_circle_area_irrational_if_rational_diameter_l4009_400954

theorem circle_area_irrational_if_rational_diameter :
  ∀ d : ℚ, d > 0 → ∃ A : ℝ, A = π * (d / 2)^2 ∧ Irrational A := by
  sorry

end NUMINAMATH_CALUDE_circle_area_irrational_if_rational_diameter_l4009_400954


namespace NUMINAMATH_CALUDE_expression_simplification_l4009_400957

theorem expression_simplification 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hc : c > 0) 
  (hd : 2 * (a - b^2)^2 + (2 * b * Real.sqrt (2 * a))^2 ≠ 0) :
  (Real.sqrt 3 * (a - b^2) + Real.sqrt 3 * b * (8 * b^3)^(1/3)) / 
  Real.sqrt (2 * (a - b^2)^2 + (2 * b * Real.sqrt (2 * a))^2) * 
  (Real.sqrt (2 * a) - Real.sqrt (2 * c)) / 
  (Real.sqrt (3 / a) - Real.sqrt (3 / c)) = 
  -Real.sqrt (a * c) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4009_400957


namespace NUMINAMATH_CALUDE_special_function_sum_l4009_400960

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Main theorem -/
theorem special_function_sum (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_prop : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_l4009_400960


namespace NUMINAMATH_CALUDE_log_ratio_squared_l4009_400997

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h2 : x * y = 243) :
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l4009_400997


namespace NUMINAMATH_CALUDE_ducks_theorem_l4009_400947

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_theorem : ducks_remaining 320 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ducks_theorem_l4009_400947


namespace NUMINAMATH_CALUDE_layla_earnings_l4009_400943

/-- Calculates the total earnings from babysitting given the hourly rates and hours worked for three families. -/
def total_earnings (rate1 rate2 rate3 : ℕ) (hours1 hours2 hours3 : ℕ) : ℕ :=
  rate1 * hours1 + rate2 * hours2 + rate3 * hours3

/-- Proves that Layla's total earnings from babysitting equal $273 given the specified rates and hours. -/
theorem layla_earnings : total_earnings 15 18 20 7 6 3 = 273 := by
  sorry

#eval total_earnings 15 18 20 7 6 3

end NUMINAMATH_CALUDE_layla_earnings_l4009_400943


namespace NUMINAMATH_CALUDE_drink_composition_l4009_400936

theorem drink_composition (coke sprite mountain_dew : ℕ) 
  (h1 : coke = 2)
  (h2 : sprite = 1)
  (h3 : mountain_dew = 3)
  (h4 : (6 : ℚ) / (coke / (coke + sprite + mountain_dew)) = 18) :
  (6 : ℚ) / ((coke : ℚ) / (coke + sprite + mountain_dew)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_drink_composition_l4009_400936


namespace NUMINAMATH_CALUDE_school_election_votes_l4009_400964

theorem school_election_votes (randy_votes shaun_votes eliot_votes : ℕ) : 
  randy_votes = 16 →
  shaun_votes = 5 * randy_votes →
  eliot_votes = 2 * shaun_votes →
  eliot_votes = 160 := by
  sorry

end NUMINAMATH_CALUDE_school_election_votes_l4009_400964


namespace NUMINAMATH_CALUDE_power_and_multiplication_l4009_400905

theorem power_and_multiplication : 2 * (3^2)^4 = 13122 := by
  sorry

end NUMINAMATH_CALUDE_power_and_multiplication_l4009_400905


namespace NUMINAMATH_CALUDE_workers_wage_increase_l4009_400922

theorem workers_wage_increase (original_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) : 
  increase_percentage = 40 →
  new_wage = 35 →
  new_wage = original_wage * (1 + increase_percentage / 100) →
  original_wage = 25 := by
sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l4009_400922


namespace NUMINAMATH_CALUDE_flea_treatment_ratio_l4009_400924

theorem flea_treatment_ratio (F : ℕ) (p : ℚ) : 
  F - 14 = 210 → 
  F * (1 - p)^4 = 14 → 
  ∃ (n : ℕ), n * (F * p) = F ∧ n = 448 := by
sorry

end NUMINAMATH_CALUDE_flea_treatment_ratio_l4009_400924


namespace NUMINAMATH_CALUDE_degree_of_polynomial_power_l4009_400942

/-- The degree of the polynomial (5x^3 + 7x + 2)^10 is 30. -/
theorem degree_of_polynomial_power (x : ℝ) : 
  Polynomial.degree ((5 * X^3 + 7 * X + 2 : Polynomial ℝ)^10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_power_l4009_400942


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l4009_400930

theorem fraction_product_simplification :
  (240 : ℚ) / 18 * 7 / 210 * 9 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l4009_400930


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_geq_four_l4009_400989

theorem quadratic_always_nonnegative_implies_a_geq_four (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + a ≥ 0) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_geq_four_l4009_400989


namespace NUMINAMATH_CALUDE_odd_periodic_function_sum_l4009_400914

-- Define the properties of function f
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 6) = f x) ∧
  (f 1 = 1)

-- State the theorem
theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : 
  f 2015 + f 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_sum_l4009_400914


namespace NUMINAMATH_CALUDE_gcd_1029_1437_5649_l4009_400940

theorem gcd_1029_1437_5649 : Nat.gcd 1029 (Nat.gcd 1437 5649) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1029_1437_5649_l4009_400940


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l4009_400963

theorem rectangle_perimeter_problem : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    (a * b : ℕ) = 2 * (2 * a + 2 * b) ∧ 
    2 * (a + b) = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l4009_400963


namespace NUMINAMATH_CALUDE_farm_width_is_15km_l4009_400953

/-- A rectangular farm with given properties has a width of 15 kilometers. -/
theorem farm_width_is_15km (length width : ℝ) : 
  length > 0 →
  width > 0 →
  2 * (length + width) = 46 →
  width = length + 7 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_farm_width_is_15km_l4009_400953


namespace NUMINAMATH_CALUDE_fresh_mushroom_mass_calculation_l4009_400915

/-- The mass of fresh mushrooms in kg that, when dried, become 15 kg lighter
    and have a moisture content of 60%, given that fresh mushrooms contain 90% water. -/
def fresh_mushroom_mass : ℝ := 20

/-- The water content of fresh mushrooms as a percentage. -/
def fresh_water_content : ℝ := 90

/-- The water content of dried mushrooms as a percentage. -/
def dried_water_content : ℝ := 60

/-- The mass reduction after drying in kg. -/
def mass_reduction : ℝ := 15

theorem fresh_mushroom_mass_calculation :
  fresh_mushroom_mass * (1 - fresh_water_content / 100) =
  (fresh_mushroom_mass - mass_reduction) * (1 - dried_water_content / 100) :=
by sorry

end NUMINAMATH_CALUDE_fresh_mushroom_mass_calculation_l4009_400915


namespace NUMINAMATH_CALUDE_no_six_odd_reciprocals_sum_to_one_l4009_400946

theorem no_six_odd_reciprocals_sum_to_one :
  ¬ ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧
    1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f = 1 := by
  sorry


end NUMINAMATH_CALUDE_no_six_odd_reciprocals_sum_to_one_l4009_400946


namespace NUMINAMATH_CALUDE_f_properties_l4009_400925

def f (a x : ℝ) : ℝ := |x - 2*a| + |x + a|

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f 1 x ≥ 3) ∧ 
  (∃ x : ℝ, f 1 x = 3) ∧
  (∀ x : ℝ, a < 0 → f a x ≥ 5*a) ∧
  (∀ x : ℝ, a > 0 → (f a x ≥ 5*a ↔ (x ≤ -2*a ∨ x ≥ 3*a))) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l4009_400925


namespace NUMINAMATH_CALUDE_average_salary_after_bonuses_and_taxes_l4009_400917

def employee_salary (name : Char) : ℕ :=
  match name with
  | 'A' => 8000
  | 'B' => 5000
  | 'C' => 11000
  | 'D' => 7000
  | 'E' => 9000
  | 'F' => 6000
  | 'G' => 10000
  | _ => 0

def apply_bonus_or_tax (salary : ℕ) (rate : ℚ) (is_bonus : Bool) : ℚ :=
  if is_bonus then
    salary + salary * rate
  else
    salary - salary * rate

def final_salary (name : Char) : ℚ :=
  match name with
  | 'A' => apply_bonus_or_tax (employee_salary 'A') (1/10) true
  | 'B' => apply_bonus_or_tax (employee_salary 'B') (1/20) false
  | 'C' => employee_salary 'C'
  | 'D' => apply_bonus_or_tax (employee_salary 'D') (1/20) false
  | 'E' => apply_bonus_or_tax (employee_salary 'E') (3/100) false
  | 'F' => apply_bonus_or_tax (employee_salary 'F') (1/20) false
  | 'G' => apply_bonus_or_tax (employee_salary 'G') (3/40) true
  | _ => 0

def total_final_salaries : ℚ :=
  (final_salary 'A') + (final_salary 'B') + (final_salary 'C') +
  (final_salary 'D') + (final_salary 'E') + (final_salary 'F') +
  (final_salary 'G')

def number_of_employees : ℕ := 7

theorem average_salary_after_bonuses_and_taxes :
  (total_final_salaries / number_of_employees) = 8054.29 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_after_bonuses_and_taxes_l4009_400917


namespace NUMINAMATH_CALUDE_sqrt_nested_expression_l4009_400937

theorem sqrt_nested_expression : 
  Real.sqrt (144 * Real.sqrt (64 * Real.sqrt 36)) = 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_expression_l4009_400937


namespace NUMINAMATH_CALUDE_max_value_of_f_l4009_400983

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

theorem max_value_of_f :
  ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4009_400983


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l4009_400944

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_equals_octal_55 : 
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l4009_400944


namespace NUMINAMATH_CALUDE_no_integer_roots_l4009_400974

theorem no_integer_roots : ¬∃ (x : ℤ), x^3 - 4*x^2 - 11*x + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l4009_400974


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_sum_l4009_400990

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields here
  dummy : Unit

/-- Represents the smaller tetrahedron formed by the centers of the faces of a regular tetrahedron -/
def smaller_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- The volume ratio of the smaller tetrahedron to the original tetrahedron -/
def volume_ratio (t : RegularTetrahedron) : ℚ :=
  sorry

/-- States that m and n are relatively prime positive integers -/
def are_relatively_prime (m n : ℕ) : Prop :=
  sorry

theorem tetrahedron_volume_ratio_sum (t : RegularTetrahedron) (m n : ℕ) :
  volume_ratio t = m / n →
  are_relatively_prime m n →
  m + n = 28 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_sum_l4009_400990


namespace NUMINAMATH_CALUDE_square_sum_from_means_l4009_400911

theorem square_sum_from_means (x y : ℝ) 
  (h_am : (x + y) / 2 = 20) 
  (h_gm : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l4009_400911


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l4009_400900

theorem rectangle_dimensions :
  ∀ (w l : ℝ),
  w > 0 →
  l = 2 * w →
  2 * (l + w) = 3 * (l * w) →
  w = 1 ∧ l = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l4009_400900


namespace NUMINAMATH_CALUDE_gargamel_tire_purchase_l4009_400908

def sale_price : ℕ := 75
def total_savings : ℕ := 36
def original_price : ℕ := 84

theorem gargamel_tire_purchase :
  (total_savings / (original_price - sale_price) : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gargamel_tire_purchase_l4009_400908


namespace NUMINAMATH_CALUDE_company_average_salary_l4009_400972

/-- Calculate the average salary for a company given the number of managers and associates, and their respective average salaries. -/
theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℝ)
  (avg_salary_associates : ℝ)
  (h_managers : num_managers = 15)
  (h_associates : num_associates = 75)
  (h_salary_managers : avg_salary_managers = 90000)
  (h_salary_associates : avg_salary_associates = 30000) :
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  let total_employees := num_managers + num_associates
  total_salary / total_employees = 40000 := by
  sorry

#check company_average_salary

end NUMINAMATH_CALUDE_company_average_salary_l4009_400972


namespace NUMINAMATH_CALUDE_binomial_10_3_l4009_400977

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l4009_400977


namespace NUMINAMATH_CALUDE_rectangular_toilet_area_l4009_400910

theorem rectangular_toilet_area :
  let length : ℝ := 5
  let width : ℝ := 17 / 20
  let area := length * width
  area = 4.25 := by sorry

end NUMINAMATH_CALUDE_rectangular_toilet_area_l4009_400910


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l4009_400926

/-- The number of digits in the sequence -/
def n : ℕ := 2023

/-- Integer a consisting of n nines in base 10 -/
def a : ℕ := 10^n - 1

/-- Integer b consisting of n sixes in base 10 -/
def b : ℕ := 2 * (10^n - 1) / 3

/-- The product 9ab -/
def prod : ℕ := 9 * a * b

/-- Sum of digits function -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : sum_of_digits prod = 20235 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l4009_400926


namespace NUMINAMATH_CALUDE_abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one_l4009_400948

theorem abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one :
  (∀ x : ℝ, |x - 1| > 2 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ |x - 1| ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one_l4009_400948


namespace NUMINAMATH_CALUDE_min_tiles_for_l_shape_min_tiles_for_specific_l_shape_l4009_400921

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the number of tiles needed to cover a rectangle -/
def tilesNeeded (r : Rectangle) (tileArea : ℕ) : ℕ := 
  (area r + tileArea - 1) / tileArea

theorem min_tiles_for_l_shape (tile : Rectangle) 
  (large : Rectangle) (small : Rectangle) : ℕ :=
  let tileArea := area tile
  let largeRect := Rectangle.mk (feetToInches large.length) (feetToInches large.width)
  let smallRect := Rectangle.mk (feetToInches small.length) (feetToInches small.width)
  tilesNeeded largeRect tileArea + tilesNeeded smallRect tileArea

theorem min_tiles_for_specific_l_shape : 
  min_tiles_for_l_shape (Rectangle.mk 2 6) (Rectangle.mk 3 4) (Rectangle.mk 2 1) = 168 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_l_shape_min_tiles_for_specific_l_shape_l4009_400921


namespace NUMINAMATH_CALUDE_jumper_cost_l4009_400976

def initial_amount : ℕ := 26
def tshirt_cost : ℕ := 4
def heels_cost : ℕ := 5
def remaining_amount : ℕ := 8

theorem jumper_cost :
  initial_amount - tshirt_cost - heels_cost - remaining_amount = 9 :=
by sorry

end NUMINAMATH_CALUDE_jumper_cost_l4009_400976


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4009_400959

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 49 = 1

-- State the theorem
theorem hyperbola_vertex_distance :
  ∃ (x y : ℝ), hyperbola x y → 
    (let vertex_distance := 2 * (Real.sqrt 144);
     vertex_distance = 24) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4009_400959


namespace NUMINAMATH_CALUDE_probability_two_red_marbles_l4009_400999

/-- The probability of selecting two red marbles without replacement from a bag containing 2 red marbles and 3 green marbles. -/
theorem probability_two_red_marbles (red : ℕ) (green : ℕ) (total : ℕ) :
  red = 2 →
  green = 3 →
  total = red + green →
  (red / total) * ((red - 1) / (total - 1)) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_marbles_l4009_400999


namespace NUMINAMATH_CALUDE_unique_triple_l4009_400949

theorem unique_triple : ∃! (a b c : ℤ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b = c ∧ 
  b * c = a ∧ 
  a = -4 ∧ b = 2 ∧ c = -2 := by sorry

end NUMINAMATH_CALUDE_unique_triple_l4009_400949


namespace NUMINAMATH_CALUDE_clara_cookie_sales_l4009_400933

/-- Represents the number of cookies in a box for each type -/
def cookies_per_box : Fin 3 → ℕ
  | 0 => 12
  | 1 => 20
  | 2 => 16

/-- Represents the number of boxes sold for each type -/
def boxes_sold : Fin 3 → ℕ
  | 0 => 50
  | 1 => 80
  | 2 => 70

/-- Calculates the total number of cookies sold -/
def total_cookies_sold : ℕ :=
  (cookies_per_box 0 * boxes_sold 0) +
  (cookies_per_box 1 * boxes_sold 1) +
  (cookies_per_box 2 * boxes_sold 2)

theorem clara_cookie_sales :
  total_cookies_sold = 3320 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_l4009_400933


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4009_400971

theorem rectangle_perimeter (x y z : ℝ) : 
  x + y + z = 75 →
  x > 0 → y > 0 → z > 0 →
  2 * (x + 75) = (2 * (y + 75) + 2 * (z + 75)) / 2 →
  2 * (x + 75) = 20 * 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4009_400971


namespace NUMINAMATH_CALUDE_march_first_is_tuesday_l4009_400982

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that March 15 is a Tuesday, prove that March 1 is also a Tuesday -/
theorem march_first_is_tuesday (march15 : MarchDate) 
  (h15 : march15.day = 15 ∧ march15.dayOfWeek = DayOfWeek.Tuesday) :
  ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_march_first_is_tuesday_l4009_400982


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l4009_400979

theorem smallest_angle_in_special_right_triangle :
  ∀ (a b : ℝ), 
  0 < a ∧ 0 < b →  -- Angles are positive
  a + b = 90 →     -- Sum of acute angles in a right triangle
  a / b = 3 / 2 →  -- Ratio of angles is 3:2
  min a b = 36 :=  -- The smallest angle is 36°
by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l4009_400979


namespace NUMINAMATH_CALUDE_six_times_two_minus_three_l4009_400987

theorem six_times_two_minus_three : 6 * 2 - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_six_times_two_minus_three_l4009_400987


namespace NUMINAMATH_CALUDE_expression_value_l4009_400956

theorem expression_value : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4009_400956


namespace NUMINAMATH_CALUDE_cost_of_corn_seeds_l4009_400907

/-- The cost of corn seeds for a farmer's harvest -/
theorem cost_of_corn_seeds
  (fertilizer_pesticide_cost : ℕ)
  (labor_cost : ℕ)
  (bags_of_corn : ℕ)
  (profit_percentage : ℚ)
  (price_per_bag : ℕ)
  (h1 : fertilizer_pesticide_cost = 35)
  (h2 : labor_cost = 15)
  (h3 : bags_of_corn = 10)
  (h4 : profit_percentage = 1/10)
  (h5 : price_per_bag = 11) :
  ∃ (corn_seed_cost : ℕ),
    corn_seed_cost = 49 ∧
    (corn_seed_cost : ℚ) + fertilizer_pesticide_cost + labor_cost +
      (profit_percentage * (bags_of_corn * price_per_bag)) =
    bags_of_corn * price_per_bag :=
by sorry

end NUMINAMATH_CALUDE_cost_of_corn_seeds_l4009_400907


namespace NUMINAMATH_CALUDE_square_difference_601_597_l4009_400923

theorem square_difference_601_597 : 601^2 - 597^2 = 4792 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_601_597_l4009_400923


namespace NUMINAMATH_CALUDE_inequality_problem_l4009_400966

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  (a - d > b - c) ∧ (a / d > b / c) ∧ (a * c > b * d) ∧ ¬(a + d > b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l4009_400966
