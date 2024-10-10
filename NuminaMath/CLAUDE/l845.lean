import Mathlib

namespace f_g_f_3_equals_101_l845_84590

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 5 * x + 4

-- State the theorem
theorem f_g_f_3_equals_101 : f (g (f 3)) = 101 := by
  sorry

end f_g_f_3_equals_101_l845_84590


namespace hyperbola_theorem_l845_84585

/-- A hyperbola is defined by its equation in the form ax² + by² = c, where a, b, and c are constants and a and b have opposite signs. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  opposite_signs : a * b < 0

/-- Two hyperbolas share the same asymptotes if they have the same ratio of coefficients for x² and y². -/
def share_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

/-- A point (x, y) is on a hyperbola if it satisfies the hyperbola's equation. -/
def point_on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  h.a * x^2 + h.b * y^2 = h.c

/-- The main theorem to be proved -/
theorem hyperbola_theorem (h1 h2 : Hyperbola) :
  h1.a = 1/4 ∧ h1.b = -1 ∧ h1.c = 1 ∧
  h2.a = -1/16 ∧ h2.b = 1/4 ∧ h2.c = 1 →
  share_asymptotes h1 h2 ∧ point_on_hyperbola h2 2 (Real.sqrt 5) :=
by sorry

end hyperbola_theorem_l845_84585


namespace solve_rope_problem_l845_84573

def rope_problem (x : ℝ) : Prop :=
  let known_ropes := [8, 20, 7]
  let total_ropes := 6
  let knot_loss := 1.2
  let final_length := 35
  let num_knots := total_ropes - 1
  let total_knot_loss := num_knots * knot_loss
  final_length + total_knot_loss = (known_ropes.sum + 3 * x)

theorem solve_rope_problem :
  ∃ x : ℝ, rope_problem x ∧ x = 2 := by sorry

end solve_rope_problem_l845_84573


namespace quadratic_inequality_range_l845_84560

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + 4 * x + 1 > 0) ↔ a ≤ 4 := by sorry

end quadratic_inequality_range_l845_84560


namespace specific_grid_rhombuses_l845_84539

/-- A grid composed of equilateral triangles -/
structure TriangleGrid where
  num_triangles : ℕ
  num_rows : ℕ
  num_cols : ℕ

/-- The number of rhombuses that can be formed from two adjacent triangles in the grid -/
def count_rhombuses (grid : TriangleGrid) : ℕ :=
  sorry

/-- Theorem stating that a specific grid with 25 triangles has 30 rhombuses -/
theorem specific_grid_rhombuses :
  ∃ (grid : TriangleGrid), 
    grid.num_triangles = 25 ∧ 
    grid.num_rows = 5 ∧ 
    grid.num_cols = 5 ∧ 
    count_rhombuses grid = 30 := by
  sorry

end specific_grid_rhombuses_l845_84539


namespace max_value_of_f_l845_84532

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2

-- Define the interval
def interval : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 4 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x :=
sorry

end max_value_of_f_l845_84532


namespace boxed_divisibility_boxed_27_divisibility_l845_84522

def boxed (n : ℕ+) : ℕ := (10^n.val - 1) / 9

theorem boxed_divisibility (m : ℕ) :
  ∃ k : ℕ, boxed (3^m : ℕ+) = k * 3^m ∧ 
  ∀ l : ℕ, boxed (3^m : ℕ+) ≠ l * 3^(m+1) :=
sorry

theorem boxed_27_divisibility (n : ℕ+) :
  27 ∣ n ↔ 27 ∣ boxed n :=
sorry

end boxed_divisibility_boxed_27_divisibility_l845_84522


namespace plane_speed_l845_84509

theorem plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) :
  distance_with_wind = 400 →
  distance_against_wind = 320 →
  wind_speed = 20 →
  ∃ (still_air_speed : ℝ) (time : ℝ),
    time > 0 ∧
    distance_with_wind = (still_air_speed + wind_speed) * time ∧
    distance_against_wind = (still_air_speed - wind_speed) * time ∧
    still_air_speed = 180 :=
by sorry

end plane_speed_l845_84509


namespace power_of_six_l845_84543

theorem power_of_six : (6 : ℕ) ^ ((6 : ℕ) / 2) = 216 := by sorry

end power_of_six_l845_84543


namespace specific_courses_not_consecutive_l845_84504

-- Define the number of courses
def n : ℕ := 6

-- Define the number of specific courses we're interested in
def k : ℕ := 3

-- Theorem statement
theorem specific_courses_not_consecutive :
  (n.factorial : ℕ) - (n - k + 1).factorial * k.factorial = 576 := by
  sorry

end specific_courses_not_consecutive_l845_84504


namespace chessboard_cannot_be_tiled_l845_84502

/-- Represents a chessboard with some squares removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (removedSquares : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the properties of our specific chessboard -/
def ourChessboard : ModifiedChessboard :=
  { size := 8, removedSquares := 2 }

/-- Defines the properties of our domino -/
def ourDomino : Domino :=
  { length := 2, width := 1 }

/-- Function to check if a chessboard can be tiled with dominoes -/
def canBeTiled (board : ModifiedChessboard) (tile : Domino) : Prop :=
  ∃ (tiling : Nat), 
    (board.size * board.size - board.removedSquares) = tiling * tile.length * tile.width

/-- Theorem stating that our specific chessboard cannot be tiled with our specific dominoes -/
theorem chessboard_cannot_be_tiled : 
  ¬(canBeTiled ourChessboard ourDomino) := by
  sorry


end chessboard_cannot_be_tiled_l845_84502


namespace candidate_total_score_l845_84577

/-- Calculates the total score of a candidate based on their written test and interview scores -/
def totalScore (writtenScore : ℝ) (interviewScore : ℝ) : ℝ :=
  0.70 * writtenScore + 0.30 * interviewScore

/-- Theorem stating that the total score of a candidate with given scores is 87 -/
theorem candidate_total_score :
  let writtenScore : ℝ := 90
  let interviewScore : ℝ := 80
  totalScore writtenScore interviewScore = 87 := by
  sorry

#eval totalScore 90 80

end candidate_total_score_l845_84577


namespace race_finish_orders_l845_84547

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

theorem race_finish_orders : number_of_permutations 4 = 24 := by
  sorry

end race_finish_orders_l845_84547


namespace largest_root_cubic_bounded_l845_84540

theorem largest_root_cubic_bounded (b₂ b₁ b₀ : ℝ) 
  (h₂ : |b₂| ≤ 1) (h₁ : |b₁| ≤ 1) (h₀ : |b₀| ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b₂*r^2 + b₁*r + b₀ = 0 ∧
  (∀ s : ℝ, s > 0 ∧ s^3 + b₂*s^2 + b₁*s + b₀ = 0 → s ≤ r) ∧
  1.5 < r ∧ r < 2 :=
sorry

end largest_root_cubic_bounded_l845_84540


namespace power_17_2023_mod_26_l845_84582

theorem power_17_2023_mod_26 : 17^2023 % 26 = 7 := by
  sorry

end power_17_2023_mod_26_l845_84582


namespace smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72_l845_84569

/-- Reverses the digits of a given integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit integer -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72 :
  ∀ p : ℕ,
    isFiveDigit p →
    p % 72 = 0 →
    (reverseDigits p) % 72 = 0 →
    p % 11 = 0 →
    p ≥ 80001 :=
  sorry

end smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72_l845_84569


namespace field_area_in_acres_l845_84536

-- Define the field dimensions
def field_length : ℕ := 30
def width_plus_diagonal : ℕ := 50

-- Define the conversion rate
def square_steps_per_acre : ℕ := 240

-- Theorem statement
theorem field_area_in_acres :
  ∃ (width : ℕ),
    width^2 + field_length^2 = (width_plus_diagonal - width)^2 ∧
    (field_length * width) / square_steps_per_acre = 2 :=
by sorry

end field_area_in_acres_l845_84536


namespace intersection_and_distance_l845_84565

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the parameter a
def a : ℝ := -3

-- Define the line equations
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := x + y + a = 0
def line3 (x y : ℝ) : Prop := a * x + 2 * y + 3 = 0

-- State the theorem
theorem intersection_and_distance :
  (line1 P.1 P.2 ∧ line2 P.1 P.2) →
  (a = -3 ∧ P.2 = 2 ∧
   (|a * P.1 + 2 * P.2 + 3| / Real.sqrt (a^2 + 2^2) = 4 * Real.sqrt 13 / 13)) :=
by sorry

end intersection_and_distance_l845_84565


namespace proportion_equality_l845_84537

theorem proportion_equality (x : ℝ) (h : (3/4) / x = 7/8) : x = 6/7 := by
  sorry

end proportion_equality_l845_84537


namespace seth_candy_bars_l845_84546

theorem seth_candy_bars (max_candy_bars : ℕ) (seth_candy_bars : ℕ) : 
  max_candy_bars = 24 →
  seth_candy_bars = 3 * max_candy_bars + 6 →
  seth_candy_bars = 78 :=
by sorry

end seth_candy_bars_l845_84546


namespace existence_of_polynomials_l845_84596

-- Define the function f
def f (x y z : ℝ) : ℝ := x^2 + y^2 + z^2 + x*y*z

-- Define the theorem
theorem existence_of_polynomials :
  ∃ (a b c : ℝ → ℝ → ℝ → ℝ),
    (∀ x y z, f (a x y z) (b x y z) (c x y z) = f x y z) ∧
    (∃ x y z, (a x y z, b x y z, c x y z) ≠ (x, y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, -y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, -y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, -y, z)) :=
by
  sorry

end existence_of_polynomials_l845_84596


namespace exactly_one_positive_integer_satisfies_condition_l845_84527

theorem exactly_one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n > 12 :=
by sorry

end exactly_one_positive_integer_satisfies_condition_l845_84527


namespace mixed_div_frac_example_l845_84597

-- Define the division operation for mixed numbers and fractions
def mixedDivFrac (whole : ℤ) (num : ℕ) (den : ℕ) (frac_num : ℕ) (frac_den : ℕ) : ℚ :=
  (whole : ℚ) + (num : ℚ) / (den : ℚ) / ((frac_num : ℚ) / (frac_den : ℚ))

-- State the theorem
theorem mixed_div_frac_example : mixedDivFrac 2 1 4 3 5 = 15 / 4 := by
  sorry

end mixed_div_frac_example_l845_84597


namespace candy_distribution_l845_84501

theorem candy_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2013 →
  a = 2 * b + 10 →
  a = 3 * c + 18 →
  a = 5 * d - 55 →
  a + b + c + d = total →
  a = 990 := by
  sorry

end candy_distribution_l845_84501


namespace simplify_fraction_l845_84541

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end simplify_fraction_l845_84541


namespace value_of_x_l845_84505

theorem value_of_x (x y z : ℝ) 
  (h1 : x = y / 3) 
  (h2 : y = z / 6) 
  (h3 : z = 72) : 
  x = 4 := by
  sorry

end value_of_x_l845_84505


namespace angle_PSU_is_20_degrees_l845_84554

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the angle measure in degrees
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the foot of the perpendicular
def foot_of_perpendicular (P S Q R : ℝ × ℝ) : Prop :=
  sorry

-- Define the center of the circumscribed circle
def circumcenter (T P Q R : ℝ × ℝ) : Prop :=
  sorry

-- Define a point on the diameter opposite to another point
def opposite_on_diameter (P U T : ℝ × ℝ) : Prop :=
  sorry

theorem angle_PSU_is_20_degrees 
  (P Q R S T U : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (angle_PRQ : angle_measure P R Q = 60)
  (angle_QRP : angle_measure Q R P = 80)
  (S_perpendicular : foot_of_perpendicular P S Q R)
  (T_circumcenter : circumcenter T P Q R)
  (U_opposite : opposite_on_diameter P U T) :
  angle_measure P S U = 20 := by
  sorry

end angle_PSU_is_20_degrees_l845_84554


namespace company_research_development_l845_84591

/-- Success probability of Team A -/
def p_a : ℚ := 2/3

/-- Success probability of Team B -/
def p_b : ℚ := 3/5

/-- Profit from successful development of product A (in thousands of dollars) -/
def profit_a : ℕ := 120

/-- Profit from successful development of product B (in thousands of dollars) -/
def profit_b : ℕ := 100

/-- The probability of at least one new product being successfully developed -/
def prob_at_least_one : ℚ := 1 - (1 - p_a) * (1 - p_b)

/-- The expected profit of the company (in thousands of dollars) -/
def expected_profit : ℚ := 
  0 * (1 - p_a) * (1 - p_b) + 
  profit_a * p_a * (1 - p_b) + 
  profit_b * (1 - p_a) * p_b + 
  (profit_a + profit_b) * p_a * p_b

theorem company_research_development :
  (prob_at_least_one = 13/15) ∧ (expected_profit = 140) := by
  sorry

end company_research_development_l845_84591


namespace grocer_banana_purchase_l845_84553

/-- Proves that the grocer purchased 792 pounds of bananas given the conditions -/
theorem grocer_banana_purchase :
  ∀ (pounds : ℝ),
  (pounds / 3 * 0.50 = pounds / 4 * 1.00 - 11.00) →
  pounds = 792 := by
sorry

end grocer_banana_purchase_l845_84553


namespace tangent_slope_angle_l845_84559

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

theorem tangent_slope_angle :
  let slope := (deriv f) 1
  Real.arctan slope = 3 * π / 4 := by sorry

end tangent_slope_angle_l845_84559


namespace right_triangle_case1_right_triangle_case2_l845_84550

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- length of BC
  b : ℝ  -- length of AC
  c : ℝ  -- length of AB
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Theorem for the first scenario
theorem right_triangle_case1 (t : RightTriangle) (h1 : t.a = 7) (h2 : t.b = 24) : t.c = 25 := by
  sorry

-- Theorem for the second scenario
theorem right_triangle_case2 (t : RightTriangle) (h1 : t.a = 12) (h2 : t.c = 13) : t.b = 5 := by
  sorry

end right_triangle_case1_right_triangle_case2_l845_84550


namespace triangle_theorem_l845_84567

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def triangleCondition (t : Triangle) : Prop :=
  t.b * (Real.sin (t.C / 2))^2 + t.c * (Real.sin (t.B / 2))^2 = t.a / 2

theorem triangle_theorem (t : Triangle) (h : triangleCondition t) :
  (t.b + t.c = 2 * t.a) ∧ (t.A ≤ Real.pi / 3) := by sorry

end triangle_theorem_l845_84567


namespace sandys_number_l845_84581

theorem sandys_number : ∃! x : ℝ, (3 * x + 20)^2 = 2500 ∧ x = 10 := by
  sorry

end sandys_number_l845_84581


namespace least_integer_with_specific_divisibility_l845_84556

theorem least_integer_with_specific_divisibility : ∃ n : ℕ+,
  (∀ k : ℕ, k ≤ 28 → k ∣ n) ∧
  (31 ∣ n) ∧
  ¬(29 ∣ n) ∧
  ¬(30 ∣ n) ∧
  (∀ m : ℕ+, m < n →
    ¬((∀ k : ℕ, k ≤ 28 → k ∣ m) ∧
      (31 ∣ m) ∧
      ¬(29 ∣ m) ∧
      ¬(30 ∣ m))) ∧
  n = 477638700 := by
sorry

end least_integer_with_specific_divisibility_l845_84556


namespace black_region_area_l845_84545

/-- The area of the black region in a square-within-square configuration -/
theorem black_region_area (larger_side smaller_side : ℝ) (h1 : larger_side = 9) (h2 : smaller_side = 4) :
  larger_side ^ 2 - smaller_side ^ 2 = 65 := by
  sorry

end black_region_area_l845_84545


namespace next_occurrence_sqrt_l845_84521

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 60 * 60

/-- The time difference in seconds between two consecutive occurrences -/
def time_difference : ℕ := seconds_per_day + seconds_per_hour

theorem next_occurrence_sqrt (S : ℕ) (h : S = time_difference) : 
  Real.sqrt (S : ℝ) = 300 := by
  sorry

end next_occurrence_sqrt_l845_84521


namespace trigonometric_equation_solution_l845_84513

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.tan (5 * π / 2 + x) - 3 * Real.tan x ^ 2 = (Real.cos (2 * x) - 1) / Real.cos x ^ 2) →
  ∃ k : ℤ, x = π / 4 * (4 * ↑k - 1) :=
by sorry

end trigonometric_equation_solution_l845_84513


namespace function_property_l845_84571

/-- Given a function f(x) = 2√3 sin(3ωx + π/3) where ω > 0,
    if f(x+θ) is an even function with a period of 2π,
    then θ = 7π/6 -/
theorem function_property (ω θ : ℝ) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sqrt 3 * Real.sin (3 * ω * x + π / 3)
  (∀ x, f (x + θ) = f (-x - θ)) ∧  -- f(x+θ) is even
  (∀ x, f (x + θ) = f (x + θ + 2 * π)) →  -- f(x+θ) has period 2π
  θ = 7 * π / 6 := by
  sorry

end function_property_l845_84571


namespace sum_of_fourth_powers_l845_84500

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 15) :
  a^4 + b^4 + c^4 = 35 := by
  sorry

end sum_of_fourth_powers_l845_84500


namespace denmark_pizza_combinations_l845_84586

/-- Represents the number of topping combinations for Denmark's pizza order --/
def toppingCombinations (cheeseOptions : Nat) (meatOptions : Nat) (vegetableOptions : Nat) : Nat :=
  let totalCombinations := cheeseOptions * meatOptions * vegetableOptions
  let restrictedCombinations := cheeseOptions * 1 * 1
  totalCombinations - restrictedCombinations

/-- Theorem: Denmark has 57 different topping combinations for his pizza --/
theorem denmark_pizza_combinations :
  toppingCombinations 3 4 5 = 57 := by
  sorry

#eval toppingCombinations 3 4 5

end denmark_pizza_combinations_l845_84586


namespace quadratic_equation_solution_l845_84518

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  -- The unique solution is x = -4
  use -4
  -- Proof goes here
  sorry

end quadratic_equation_solution_l845_84518


namespace vector_dot_product_l845_84508

/-- Given vectors a and b in ℝ², prove that (2a + b) · a = 6 -/
theorem vector_dot_product (a b : ℝ × ℝ) (h1 : a = (2, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 6 := by
  sorry

end vector_dot_product_l845_84508


namespace ordering_abc_l845_84572

theorem ordering_abc (a b c : ℝ) : 
  a = -(5/4) * Real.log (4/5) →
  b = Real.exp (1/4) / 4 →
  c = 1/3 →
  a < b ∧ b < c := by
sorry

end ordering_abc_l845_84572


namespace female_democrat_ratio_l845_84530

theorem female_democrat_ratio (total_participants male_participants female_participants female_democrats : ℕ) 
  (h1 : total_participants = 720)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : male_participants / 4 = total_participants / 3 - female_democrats)
  (h4 : total_participants / 3 = 240)
  (h5 : female_democrats = 120) :
  female_democrats / female_participants = 1 / 2 := by
  sorry

end female_democrat_ratio_l845_84530


namespace quartic_real_root_condition_l845_84584

theorem quartic_real_root_condition (p q : ℝ) :
  (∃ x : ℝ, x^4 + p * x^2 + q = 0) →
  p^2 ≥ 4 * q ∧
  ¬(∀ p q : ℝ, p^2 ≥ 4 * q → ∃ x : ℝ, x^4 + p * x^2 + q = 0) :=
by sorry

end quartic_real_root_condition_l845_84584


namespace product_of_roots_l845_84580

theorem product_of_roots (y₁ y₂ : ℝ) : 
  y₁ + 16 / y₁ = 12 → 
  y₂ + 16 / y₂ = 12 → 
  y₁ * y₂ = 16 := by
sorry

end product_of_roots_l845_84580


namespace sqrt_2x_plus_3_eq_x_solution_l845_84579

theorem sqrt_2x_plus_3_eq_x_solution :
  ∃! x : ℝ, Real.sqrt (2 * x + 3) = x :=
by
  -- The unique solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the equation
    sorry
  · -- Prove that any solution must be equal to 3
    sorry

#check sqrt_2x_plus_3_eq_x_solution

end sqrt_2x_plus_3_eq_x_solution_l845_84579


namespace scavenger_hunt_items_l845_84535

theorem scavenger_hunt_items (lewis samantha tanya : ℕ) : 
  lewis = samantha + 4 →
  samantha = 4 * tanya →
  lewis = 20 →
  tanya = 4 := by sorry

end scavenger_hunt_items_l845_84535


namespace unique_four_digit_square_l845_84568

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The main theorem -/
theorem unique_four_digit_square : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ 
  is_perfect_square n ∧
  is_perfect_square (reverse n) ∧
  is_perfect_square (n / reverse n) ∧
  n = 9801 := by
sorry

end unique_four_digit_square_l845_84568


namespace minimum_value_quadratic_l845_84507

theorem minimum_value_quadratic (a : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
    (∀ (y : ℝ), y ∈ Set.Icc 0 1 → x^2 - 2*a*x + a - 1 ≤ y^2 - 2*a*y + a - 1) ∧
    x^2 - 2*a*x + a - 1 = -2) → 
  a = 2 := by
sorry

end minimum_value_quadratic_l845_84507


namespace penalty_kick_test_l845_84531

/-- The probability of scoring a single penalty kick -/
def p_score : ℚ := 2/3

/-- The probability of missing a single penalty kick -/
def p_miss : ℚ := 1 - p_score

/-- The probability of being admitted in the penalty kick test -/
def p_admitted : ℚ := 
  p_score * p_score + 
  p_miss * p_score * p_score + 
  p_miss * p_miss * p_score * p_score + 
  p_score * p_miss * p_score * p_score

/-- The expected number of goals scored in the penalty kick test -/
def expected_goals : ℚ := 
  0 * (p_miss * p_miss * p_miss) + 
  1 * (2 * p_score * p_miss * p_miss + p_miss * p_miss * p_score * p_miss) + 
  2 * (p_score * p_score + p_miss * p_score * p_score + p_miss * p_miss * p_score * p_score + p_score * p_miss * p_score * p_miss) + 
  3 * (p_score * p_miss * p_score * p_score)

theorem penalty_kick_test :
  p_admitted = 20/27 ∧ expected_goals = 50/27 := by
  sorry

end penalty_kick_test_l845_84531


namespace original_number_proof_l845_84574

theorem original_number_proof : ∃ x : ℝ, 16 * x = 3408 ∧ x = 213 := by
  sorry

end original_number_proof_l845_84574


namespace range_of_a_l845_84519

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 ≥ 0 ∧ a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0

-- Define the theorem
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (-1 ≤ a ∧ a ≤ 1) ∨ a > 3) :=
sorry

end range_of_a_l845_84519


namespace circle_satisfies_equation_l845_84599

/-- A circle passing through two points with its center on a given line -/
structure CircleWithConstraints where
  -- Center of the circle lies on the line x - 2y - 2 = 0
  center : ℝ × ℝ
  center_on_line : center.1 - 2 * center.2 - 2 = 0
  -- Circle passes through points A(0, 4) and B(4, 6)
  passes_through_A : (center.1 - 0)^2 + (center.2 - 4)^2 = (center.1 - 4)^2 + (center.2 - 6)^2

/-- The standard equation of the circle -/
def circle_equation (c : CircleWithConstraints) (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = 25

/-- Theorem stating that the circle satisfies the given equation -/
theorem circle_satisfies_equation (c : CircleWithConstraints) :
  ∀ x y, (x - c.center.1)^2 + (y - c.center.2)^2 = (c.center.1 - 0)^2 + (c.center.2 - 4)^2 →
  circle_equation c x y := by
  sorry

end circle_satisfies_equation_l845_84599


namespace determine_hidden_numbers_l845_84558

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Given two sums S1 and S2, it is possible to determine the original numbers a, b, and c -/
theorem determine_hidden_numbers (a b c : ℕ) :
  let k := num_digits (a + b + c)
  let S1 := a + b + c
  let S2 := a + b * 10^k + c * 10^(2*k)
  ∃! (a' b' c' : ℕ), S1 = a' + b' + c' ∧ S2 = a' + b' * 10^k + c' * 10^(2*k) ∧ a' = a ∧ b' = b ∧ c' = c :=
by sorry

end determine_hidden_numbers_l845_84558


namespace exists_four_digit_with_eleven_multiple_permutation_l845_84583

/-- A permutation of the digits of a number -/
def isDigitPermutation (a b : ℕ) : Prop := sorry

/-- Check if a number is between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem exists_four_digit_with_eleven_multiple_permutation :
  ∃ n : ℕ, isFourDigit n ∧ ∃ m : ℕ, isDigitPermutation n m ∧ m % 11 = 0 := by
  sorry

end exists_four_digit_with_eleven_multiple_permutation_l845_84583


namespace complex_additive_inverse_l845_84566

theorem complex_additive_inverse (m : ℝ) : 
  let z : ℂ := (1 - m * I) / (1 - 2 * I)
  (∃ (a : ℝ), z = a - a * I) → m = -3 := by
  sorry

end complex_additive_inverse_l845_84566


namespace graph_equation_two_lines_l845_84593

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 is equivalent to the union of the lines x = 0 and y = 0 -/
theorem graph_equation_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end graph_equation_two_lines_l845_84593


namespace intersection_of_sets_l845_84589

theorem intersection_of_sets : 
  let M : Set Char := {a, b, c}
  let N : Set Char := {b, c, d}
  M ∩ N = {b, c} := by
  sorry

end intersection_of_sets_l845_84589


namespace black_population_in_south_percentage_l845_84598

/-- Represents the population data for a specific ethnic group across regions -/
structure PopulationData :=
  (ne : ℕ)
  (mw : ℕ)
  (central : ℕ)
  (south : ℕ)
  (west : ℕ)

/-- The demographic data for the nation in 2020 -/
def demographicData : List PopulationData :=
  [
    ⟨50, 60, 40, 70, 45⟩,  -- White
    ⟨6, 7, 3, 23, 5⟩,      -- Black
    ⟨2, 2, 1, 2, 6⟩,       -- Asian
    ⟨2, 2, 1, 4, 5⟩        -- Other
  ]

/-- Calculates the total population for a given PopulationData -/
def totalPopulation (data : PopulationData) : ℕ :=
  data.ne + data.mw + data.central + data.south + data.west

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem black_population_in_south_percentage :
  let blackData := demographicData[1]
  let totalBlack := totalPopulation blackData
  let blackInSouth := blackData.south
  roundToNearest ((blackInSouth : ℚ) / totalBlack * 100) = 52 := by
  sorry

end black_population_in_south_percentage_l845_84598


namespace tech_company_work_hours_l845_84538

/-- Calculates the total hours worked in a day for a tech company's help desk -/
theorem tech_company_work_hours :
  let total_hours : ℝ := 24
  let software_hours : ℝ := 24
  let user_help_hours : ℝ := 17
  let maintenance_percent : ℝ := 35
  let research_dev_percent : ℝ := 27
  let marketing_percent : ℝ := 15
  let multitasking_employees : ℕ := 3
  let additional_employees : ℕ := 4
  let additional_hours : ℝ := 12
  
  (maintenance_percent + research_dev_percent + marketing_percent) / 100 * total_hours +
  software_hours + user_help_hours ≥ total_hours →
  
  (max software_hours (max user_help_hours ((maintenance_percent + research_dev_percent + marketing_percent) / 100 * total_hours))) +
  additional_hours = 36 :=
by sorry

end tech_company_work_hours_l845_84538


namespace sqrt_seven_to_sixth_l845_84587

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end sqrt_seven_to_sixth_l845_84587


namespace smallest_n_multiple_of_nine_l845_84594

theorem smallest_n_multiple_of_nine (x y a : ℤ) : 
  (∃ k₁ k₂ : ℤ, x - a = 9 * k₁ ∧ y + a = 9 * k₂) →
  (∃ n : ℕ, n > 0 ∧ ∃ k : ℤ, x^2 + x*y + y^2 + n = 9 * k ∧
    ∀ m : ℕ, m > 0 → (∃ l : ℤ, x^2 + x*y + y^2 + m = 9 * l) → m ≥ n) →
  (∃ k : ℤ, x^2 + x*y + y^2 + 6 = 9 * k) :=
by sorry

end smallest_n_multiple_of_nine_l845_84594


namespace oven_temperature_increase_l845_84528

/-- Given an oven with a current temperature and a required temperature,
    calculate the temperature increase needed. -/
def temperature_increase_needed (current_temp required_temp : ℕ) : ℕ :=
  required_temp - current_temp

/-- Theorem stating that for an oven at 150 degrees that needs to reach 546 degrees,
    the temperature increase needed is 396 degrees. -/
theorem oven_temperature_increase :
  temperature_increase_needed 150 546 = 396 := by
  sorry

end oven_temperature_increase_l845_84528


namespace resort_group_combinations_l845_84503

theorem resort_group_combinations : Nat.choose 10 4 = 210 := by
  sorry

end resort_group_combinations_l845_84503


namespace cone_central_angle_l845_84551

/-- Represents a cone with its surface areas and central angle. -/
structure Cone where
  base_area : ℝ
  total_surface_area : ℝ
  lateral_surface_area : ℝ
  central_angle : ℝ

/-- The theorem stating the relationship between the cone's surface areas and its central angle. -/
theorem cone_central_angle (c : Cone) 
  (h1 : c.total_surface_area = 3 * c.base_area)
  (h2 : c.lateral_surface_area = 2 * c.base_area)
  (h3 : c.lateral_surface_area = (c.central_angle / 360) * (2 * π * c.base_area)) :
  c.central_angle = 240 := by
  sorry


end cone_central_angle_l845_84551


namespace remainder_theorem_application_l845_84562

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^6 + E * x^4 + F * x^2 + 5
  (q 2 = 17) → (q (-2) = 17) := by
  sorry

end remainder_theorem_application_l845_84562


namespace pyramid_edge_ratio_l845_84525

/-- Represents a pyramid with a cross-section parallel to its base -/
structure Pyramid where
  base_area : ℝ
  cross_section_area : ℝ
  upper_edge_length : ℝ
  lower_edge_length : ℝ
  parallel_cross_section : cross_section_area > 0
  area_ratio : cross_section_area / base_area = 4 / 9

/-- 
Theorem: In a pyramid with a cross-section parallel to its base, 
if the ratio of the cross-sectional area to the base area is 4:9, 
then the ratio of the lengths of the upper and lower parts of the lateral edge is 2:3.
-/
theorem pyramid_edge_ratio (p : Pyramid) : 
  p.upper_edge_length / p.lower_edge_length = 2 / 3 := by
  sorry

end pyramid_edge_ratio_l845_84525


namespace bean_ratio_l845_84523

/-- Proves that the ratio of green beans to remaining beans after removing red and white beans is 1:1 --/
theorem bean_ratio (total : ℕ) (green : ℕ) : 
  total = 572 →
  green = 143 →
  (total - total / 4 - (total - total / 4) / 3 - green) = green :=
by
  sorry

end bean_ratio_l845_84523


namespace town_population_problem_l845_84514

theorem town_population_problem (initial_population : ℕ) : 
  let after_changes := initial_population + 100 - 400
  let after_year_1 := after_changes / 2
  let after_year_2 := after_year_1 / 2
  let after_year_3 := after_year_2 / 2
  let after_year_4 := after_year_3 / 2
  after_year_4 = 60 → initial_population = 780 := by
  sorry

end town_population_problem_l845_84514


namespace set_intersection_empty_iff_complement_subset_l845_84516

universe u

theorem set_intersection_empty_iff_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = ∅ ↔ ∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ :=
sorry

end set_intersection_empty_iff_complement_subset_l845_84516


namespace smartphone_customers_l845_84534

/-- Represents the relationship between number of customers and smartphone price -/
def inversely_proportional (p c : ℝ) := ∃ k : ℝ, p * c = k

theorem smartphone_customers : 
  ∀ (p₁ p₂ c₁ c₂ : ℝ),
  inversely_proportional p₁ c₁ →
  inversely_proportional p₂ c₂ →
  p₁ = 20 →
  c₁ = 200 →
  c₂ = 400 →
  p₂ = 10 :=
by sorry

end smartphone_customers_l845_84534


namespace quadratic_roots_preservation_l845_84595

theorem quadratic_roots_preservation
  (a b : ℝ) (k : ℝ)
  (h_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*a*x₁ + b = 0 ∧ x₂^2 + 2*a*x₂ + b = 0)
  (h_k_pos : k > 0) :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    (y₁^2 + 2*a*y₁ + b) + k*(y₁ + a)^2 = 0 ∧
    (y₂^2 + 2*a*y₂ + b) + k*(y₂ + a)^2 = 0 :=
by sorry

end quadratic_roots_preservation_l845_84595


namespace jane_rejection_percentage_l845_84524

theorem jane_rejection_percentage 
  (john_rejection_rate : Real) 
  (total_rejection_rate : Real) 
  (jane_inspection_ratio : Real) :
  john_rejection_rate = 0.005 →
  total_rejection_rate = 0.0075 →
  jane_inspection_ratio = 1.25 →
  ∃ jane_rejection_rate : Real,
    jane_rejection_rate = 0.0095 ∧
    john_rejection_rate * 1 + jane_rejection_rate * jane_inspection_ratio = 
      total_rejection_rate * (1 + jane_inspection_ratio) :=
by sorry

end jane_rejection_percentage_l845_84524


namespace penny_theorem_l845_84515

def penny_problem (initial_amount : ℕ) (sock_pairs : ℕ) (sock_price : ℕ) (hat_price : ℕ) : Prop :=
  let total_spent := sock_pairs * sock_price + hat_price
  initial_amount - total_spent = 5

theorem penny_theorem : 
  penny_problem 20 4 2 7 := by
  sorry

end penny_theorem_l845_84515


namespace max_a_for_inequality_l845_84542

theorem max_a_for_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 2| + |x - 8| ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 2| + |x - 8| ≥ b) → b ≤ 6) :=
by sorry

end max_a_for_inequality_l845_84542


namespace basketball_handshakes_l845_84548

/-- Represents the number of handshakes in a basketball game --/
def total_handshakes (players_per_team : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let total_players := players_per_team * num_teams
  let player_handshakes := players_per_team * players_per_team
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the given scenario --/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 72 := by
  sorry

#eval total_handshakes 6 2 3

end basketball_handshakes_l845_84548


namespace james_van_capacity_l845_84555

/-- Proves that the total capacity of James' vans is 57600 gallons --/
theorem james_van_capacity :
  let total_vans : ℕ := 6
  let large_van_capacity : ℕ := 8000
  let large_van_count : ℕ := 2
  let medium_van_capacity : ℕ := large_van_capacity * 7 / 10  -- 30% less than 8000
  let medium_van_count : ℕ := 1
  let small_van_count : ℕ := total_vans - large_van_count - medium_van_count
  let total_capacity : ℕ := 57600
  let remaining_capacity : ℕ := total_capacity - (large_van_capacity * large_van_count + medium_van_capacity * medium_van_count)
  let small_van_capacity : ℕ := remaining_capacity / small_van_count

  (large_van_capacity * large_van_count + 
   medium_van_capacity * medium_van_count + 
   small_van_capacity * small_van_count) = total_capacity := by
  sorry

end james_van_capacity_l845_84555


namespace product_pass_rate_l845_84570

-- Define the defect rates for each step
variable (a b : ℝ)

-- Assume the defect rates are between 0 and 1
variable (ha : 0 ≤ a ∧ a ≤ 1)
variable (hb : 0 ≤ b ∧ b ≤ 1)

-- Define the pass rate of the product
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem statement
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  pass_rate a b = (1 - a) * (1 - b) :=
by
  sorry

end product_pass_rate_l845_84570


namespace princess_count_proof_l845_84561

/-- Represents the number of princesses at the ball -/
def num_princesses : ℕ := 8

/-- Represents the number of knights at the ball -/
def num_knights : ℕ := 22 - num_princesses

/-- Represents the total number of people at the ball -/
def total_people : ℕ := 22

/-- Function to calculate the number of knights a princess dances with -/
def knights_danced_with (princess_index : ℕ) : ℕ := 6 + princess_index

theorem princess_count_proof :
  (num_princesses + num_knights = total_people) ∧ 
  (knights_danced_with num_princesses = num_knights) ∧
  (∀ i, i ≥ 1 → i ≤ num_princesses → knights_danced_with i ≤ num_knights) :=
sorry

end princess_count_proof_l845_84561


namespace late_fee_is_124_l845_84552

/-- Calculates the late fee per month for the second bill given the total amount owed and details of three bills. -/
def calculate_late_fee (total_owed : ℚ) (bill1_amount : ℚ) (bill1_interest_rate : ℚ) (bill1_months : ℕ)
  (bill2_amount : ℚ) (bill2_months : ℕ) (bill3_fee1 : ℚ) (bill3_fee2 : ℚ) : ℚ :=
  let bill1_total := bill1_amount + bill1_amount * bill1_interest_rate * bill1_months
  let bill3_total := bill3_fee1 + bill3_fee2
  let bill2_total := total_owed - bill1_total - bill3_total
  (bill2_total - bill2_amount) / bill2_months

/-- Theorem stating that the late fee per month for the second bill is $124. -/
theorem late_fee_is_124 :
  calculate_late_fee 1234 200 (1/10) 2 130 6 40 80 = 124 := by
  sorry

end late_fee_is_124_l845_84552


namespace point_on_y_axis_l845_84575

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on the y-axis
def onYAxis (p : Point2D) : Prop := p.x = 0

-- State the theorem
theorem point_on_y_axis (m : ℝ) :
  let p := Point2D.mk (m - 1) (m + 3)
  onYAxis p → p = Point2D.mk 0 4 := by
  sorry

end point_on_y_axis_l845_84575


namespace min_value_theorem_l845_84510

def arithmeticSequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  arithmeticSequence a →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (∃ min : ℝ, min = 3/2 ∧ ∀ p q : ℕ, 1/p + 4/q ≥ min) :=
sorry

end min_value_theorem_l845_84510


namespace fraction_equality_l845_84529

theorem fraction_equality : 
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 := by
  sorry

end fraction_equality_l845_84529


namespace angle_in_first_quadrant_l845_84549

-- Define the angle in degrees and minutes
def angle : ℚ := -999 * 360 / 360 - 30 / 60

-- Function to normalize an angle to the range [0, 360)
def normalize_angle (θ : ℚ) : ℚ :=
  θ - 360 * ⌊θ / 360⌋

-- Define the first quadrant
def is_first_quadrant (θ : ℚ) : Prop :=
  0 < normalize_angle θ ∧ normalize_angle θ < 90

-- Theorem statement
theorem angle_in_first_quadrant : is_first_quadrant angle := by
  sorry

end angle_in_first_quadrant_l845_84549


namespace line_not_in_first_quadrant_l845_84576

/-- A line y = -3x + b that does not pass through the first quadrant has b ≤ 0 -/
theorem line_not_in_first_quadrant (b : ℝ) : 
  (∀ x y : ℝ, y = -3 * x + b → ¬(x > 0 ∧ y > 0)) → b ≤ 0 := by
  sorry

end line_not_in_first_quadrant_l845_84576


namespace equation_system_solution_l845_84526

theorem equation_system_solution :
  ∃ (x y z : ℝ),
    (x / 6) * 12 = 10 ∧
    (y / 4) * 8 = x ∧
    (z / 3) * 5 + y = 20 ∧
    x = 5 ∧
    y = 5 / 2 ∧
    z = 21 / 2 := by
  sorry

end equation_system_solution_l845_84526


namespace tom_sees_jerry_l845_84592

/-- Represents the cat-and-mouse chase problem -/
structure ChaseSetup where
  wallSideLength : ℝ
  tomSpeed : ℝ
  jerrySpeed : ℝ
  restTime : ℝ

/-- Calculates the time when Tom first sees Jerry -/
noncomputable def timeToMeet (setup : ChaseSetup) : ℝ :=
  sorry

/-- The main theorem stating when Tom will first see Jerry -/
theorem tom_sees_jerry (setup : ChaseSetup) :
  setup.wallSideLength = 100 ∧
  setup.tomSpeed = 50 ∧
  setup.jerrySpeed = 30 ∧
  setup.restTime = 1 →
  timeToMeet setup = 8 := by
  sorry

end tom_sees_jerry_l845_84592


namespace product_local_abs_value_4_in_564823_l845_84557

/-- The local value of a digit in a number -/
def local_value (n : ℕ) (d : ℕ) (p : ℕ) : ℕ := d * (10 ^ p)

/-- The absolute value of a natural number -/
def abs_nat (n : ℕ) : ℕ := n

theorem product_local_abs_value_4_in_564823 :
  let n : ℕ := 564823
  let d : ℕ := 4
  let p : ℕ := 4  -- position of 4 in 564823 (0-indexed from right)
  (local_value n d p) * (abs_nat d) = 160000 := by sorry

end product_local_abs_value_4_in_564823_l845_84557


namespace fran_red_macaroons_l845_84517

/-- Represents the number of macaroons in various states --/
structure MacaroonCounts where
  green_baked : ℕ
  green_eaten : ℕ
  red_eaten : ℕ
  total_remaining : ℕ

/-- The theorem stating the number of red macaroons Fran baked --/
theorem fran_red_macaroons (m : MacaroonCounts) 
  (h1 : m.green_baked = 40)
  (h2 : m.green_eaten = 15)
  (h3 : m.red_eaten = 2 * m.green_eaten)
  (h4 : m.total_remaining = 45) :
  ∃ red_baked : ℕ, red_baked = 50 ∧ 
    red_baked = m.red_eaten + (m.total_remaining - (m.green_baked - m.green_eaten)) :=
by sorry

end fran_red_macaroons_l845_84517


namespace diophantus_problem_l845_84506

theorem diophantus_problem (x y z t : ℤ) : 
  x = 11 ∧ y = 4 ∧ z = 7 ∧ t = 9 →
  x + y + z = 22 ∧
  x + y + t = 24 ∧
  x + z + t = 27 ∧
  y + z + t = 20 := by
sorry

end diophantus_problem_l845_84506


namespace parabola_standard_equation_l845_84512

/-- A parabola with directrix x = 1 has the standard equation y² = -4x -/
theorem parabola_standard_equation (x y : ℝ) :
  (∃ (p : ℝ), p / 2 = 1 ∧ y^2 = -2 * p * x) → y^2 = -4 * x := by
  sorry

end parabola_standard_equation_l845_84512


namespace max_cube_sum_on_circle_l845_84563

theorem max_cube_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  x^3 + y^3 ≤ 4 * Real.sqrt 2 := by
sorry

end max_cube_sum_on_circle_l845_84563


namespace trig_fraction_equality_l845_84520

theorem trig_fraction_equality (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1/2) :
  Real.cos a / (Real.sin a - 1) = 1/2 := by
  sorry

end trig_fraction_equality_l845_84520


namespace arrangements_count_l845_84564

/-- The number of departments in the unit -/
def num_departments : ℕ := 3

/-- The number of people selected from each department for training -/
def people_per_department : ℕ := 2

/-- The total number of people trained -/
def total_trained : ℕ := num_departments * people_per_department

/-- The number of people returning to the unit after training -/
def returning_people : ℕ := 2

/-- Function to calculate the number of arrangements -/
def calculate_arrangements : ℕ := 
  let same_dept := num_departments * (returning_people * (returning_people - 1))
  let diff_dept := (num_departments * (num_departments - 1) / 2) * (returning_people * returning_people)
  same_dept + diff_dept

/-- Theorem stating that the number of different arrangements is 42 -/
theorem arrangements_count : calculate_arrangements = 42 := by
  sorry

end arrangements_count_l845_84564


namespace min_value_theorem_l845_84578

theorem min_value_theorem (x : ℝ) (h : x > 1) : 
  x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end min_value_theorem_l845_84578


namespace first_quarter_homework_points_l845_84588

theorem first_quarter_homework_points :
  ∀ (homework quiz test : ℕ),
    homework + quiz + test = 265 →
    test = 4 * quiz →
    quiz = homework + 5 →
    homework = 40 := by
  sorry

end first_quarter_homework_points_l845_84588


namespace max_profit_theorem_l845_84533

def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

def profit_B (x : ℝ) : ℝ := 2 * x

def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit_theorem :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 15 → total_profit x ≥ total_profit y ∧
  total_profit x = 45.6 :=
sorry

end max_profit_theorem_l845_84533


namespace arianna_daily_chores_l845_84511

def hours_in_day : ℕ := 24
def work_hours : ℕ := 6
def sleep_hours : ℕ := 13

theorem arianna_daily_chores : 
  hours_in_day - (work_hours + sleep_hours) = 5 := by
  sorry

end arianna_daily_chores_l845_84511


namespace differential_of_y_l845_84544

noncomputable def y (x : ℝ) : ℝ := 2 * x + Real.log (|Real.sin x + 2 * Real.cos x|)

theorem differential_of_y (x : ℝ) :
  deriv y x = (5 * Real.cos x) / (Real.sin x + 2 * Real.cos x) :=
by sorry

end differential_of_y_l845_84544
