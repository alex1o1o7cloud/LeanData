import Mathlib

namespace unique_prime_product_power_l2877_287791

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

theorem unique_prime_product_power :
  ∀ k : ℕ, k > 0 →
    (∃ a n : ℕ, n > 1 ∧ primeProduct k - 1 = a^n) ↔ k = 1 := by
  sorry

end unique_prime_product_power_l2877_287791


namespace tall_trees_indeterminate_l2877_287750

/-- Represents the number of trees in the park -/
structure ParkTrees where
  short_current : ℕ
  short_planted : ℕ
  short_after : ℕ
  tall : ℕ

/-- The given information about the trees in the park -/
def park_info : ParkTrees where
  short_current := 41
  short_planted := 57
  short_after := 98
  tall := 0  -- We use 0 as a placeholder since the number is unknown

/-- Theorem stating that the number of tall trees cannot be determined -/
theorem tall_trees_indeterminate (park : ParkTrees) 
    (h1 : park.short_current = park_info.short_current)
    (h2 : park.short_planted = park_info.short_planted)
    (h3 : park.short_after = park_info.short_after)
    (h4 : park.short_after = park.short_current + park.short_planted) :
    ∀ n : ℕ, ∃ p : ParkTrees, p.short_current = park.short_current ∧ 
                               p.short_planted = park.short_planted ∧ 
                               p.short_after = park.short_after ∧ 
                               p.tall = n :=
by sorry

end tall_trees_indeterminate_l2877_287750


namespace counterexample_exists_l2877_287762

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end counterexample_exists_l2877_287762


namespace train_length_l2877_287757

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 40 → time_s = 19.8 → 
  (speed_kmh * 1000 / 3600) * time_s = 220 := by sorry

end train_length_l2877_287757


namespace average_speed_calculation_l2877_287781

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2552
def driving_time_day1 : ℕ := 5
def driving_time_day2 : ℕ := 3

theorem average_speed_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_time : ℕ := driving_time_day1 + driving_time_day2
  (total_distance : ℚ) / (total_time : ℚ) = 27.5 := by sorry

end average_speed_calculation_l2877_287781


namespace min_value_of_function_l2877_287777

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ≥ 10 / 7 ∧
  ∃ y ≥ 0, (3 * y^2 + 9 * y + 20) / (7 * (2 + y)) = 10 / 7 :=
by sorry

end min_value_of_function_l2877_287777


namespace john_biking_distance_john_biking_distance_proof_l2877_287789

theorem john_biking_distance (bike_speed walking_speed : ℝ) 
  (walking_distance total_time : ℝ) : ℝ :=
  let total_biking_distance := 
    (total_time - walking_distance / walking_speed) * bike_speed + walking_distance
  total_biking_distance

#check john_biking_distance 15 4 3 (7/6) = 9.25

theorem john_biking_distance_proof :
  john_biking_distance 15 4 3 (7/6) = 9.25 := by
  sorry

end john_biking_distance_john_biking_distance_proof_l2877_287789


namespace infinite_solutions_iff_b_eq_neg_twelve_l2877_287770

/-- The equation 4(3x-b) = 3(4x+16) has infinitely many solutions x if and only if b = -12 -/
theorem infinite_solutions_iff_b_eq_neg_twelve :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by sorry

end infinite_solutions_iff_b_eq_neg_twelve_l2877_287770


namespace area_of_triangle_APB_l2877_287714

-- Define the square and point P
def Square (s : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (8, 8)
def D : ℝ × ℝ := (0, 8)
def F : ℝ × ℝ := (4, 8)

-- Define the conditions
def PointInSquare (P : ℝ × ℝ) : Prop := P ∈ Square 8

def EqualSegments (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2

def PerpendicularPC_FD (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) * (F.1 - D.1) + (P.2 - C.2) * (F.2 - D.2) = 0

def PerpendicularPB_AC (P : ℝ × ℝ) : Prop :=
  (P.1 - B.1) * (A.1 - C.1) + (P.2 - B.2) * (A.2 - C.2) = 0

-- Theorem statement
theorem area_of_triangle_APB (P : ℝ × ℝ) 
  (h1 : PointInSquare P) 
  (h2 : EqualSegments P) 
  (h3 : PerpendicularPC_FD P) 
  (h4 : PerpendicularPB_AC P) : 
  ∃ (area : ℝ), area = 32/5 ∧ 
  area = (1/2) * ((P.1 - A.1)^2 + (P.2 - A.2)^2) := by
  sorry

end area_of_triangle_APB_l2877_287714


namespace sin_plus_two_cos_for_point_l2877_287724

/-- Given a point P(-3,4) on the terminal side of angle α, prove that sin α + 2 cos α = -2/5 -/
theorem sin_plus_two_cos_for_point (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  Real.sin α = P.2 / r ∧ Real.cos α = P.1 / r →
  Real.sin α + 2 * Real.cos α = -2/5 := by
sorry


end sin_plus_two_cos_for_point_l2877_287724


namespace circle_area_ratio_l2877_287725

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (30 : ℝ) / 360 * (2 * Real.pi * r₁) = (24 : ℝ) / 360 * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
  sorry

end circle_area_ratio_l2877_287725


namespace solution_a_is_correct_l2877_287747

/-- The amount of Solution A used in milliliters -/
def solution_a : ℝ := 100

/-- The amount of Solution B used in milliliters -/
def solution_b : ℝ := solution_a + 500

/-- The alcohol percentage in Solution A -/
def alcohol_percent_a : ℝ := 0.16

/-- The alcohol percentage in Solution B -/
def alcohol_percent_b : ℝ := 0.10

/-- The total amount of pure alcohol in the resulting mixture in milliliters -/
def total_pure_alcohol : ℝ := 76

theorem solution_a_is_correct :
  solution_a * alcohol_percent_a + solution_b * alcohol_percent_b = total_pure_alcohol :=
sorry

end solution_a_is_correct_l2877_287747


namespace system_solution_l2877_287758

theorem system_solution : 
  ∃ (x y u v : ℝ), 
    (5 * x^7 + 3 * y^2 + 5 * u + 4 * v^4 = -2) ∧
    (2 * x^7 + 8 * y^2 + 7 * u + 4 * v^4 = 6^5 / (3^4 * 4^2)) ∧
    (8 * x^7 + 2 * y^2 + 3 * u + 6 * v^4 = -6) ∧
    (5 * x^7 + 7 * y^2 + 7 * u + 8 * v^4 = 8^3 / (2^6 * 4)) ∧
    ((x = -1 ∧ (y = 1 ∨ y = -1) ∧ u = 0 ∧ v = 0) ∨
     (x = 1 ∧ (y = 1 ∨ y = -1) ∧ u = 0 ∧ v = 0)) :=
by sorry


end system_solution_l2877_287758


namespace smallest_n_for_cube_T_l2877_287705

/-- Function that calculates (n+2)3^n for a positive integer n -/
def T (n : ℕ+) : ℕ := (n + 2) * 3^(n : ℕ)

/-- Predicate to check if a natural number is a perfect cube -/
def is_cube (m : ℕ) : Prop := ∃ k : ℕ, m = k^3

/-- Theorem stating that 1 is the smallest positive integer n for which T(n) is a perfect cube -/
theorem smallest_n_for_cube_T :
  (∃ n : ℕ+, is_cube (T n)) ∧ (∀ n : ℕ+, is_cube (T n) → 1 ≤ n) := by sorry

end smallest_n_for_cube_T_l2877_287705


namespace complex_expression_equals_nine_l2877_287740

theorem complex_expression_equals_nine :
  (Real.rpow 1.5 (1/3) * Real.rpow 12 (1/6))^2 + 8 * Real.rpow 1 0.75 - Real.rpow (-1/4) (-2) - 5 * Real.rpow 0.125 0 = 9 := by
  sorry

end complex_expression_equals_nine_l2877_287740


namespace llama_cost_increase_l2877_287778

/-- Proves that the percentage increase in the cost of each llama compared to each goat is 50% -/
theorem llama_cost_increase (goat_cost : ℝ) (total_cost : ℝ) : 
  goat_cost = 400 →
  total_cost = 4800 →
  let num_goats : ℕ := 3
  let num_llamas : ℕ := 2 * num_goats
  let total_goat_cost : ℝ := goat_cost * num_goats
  let total_llama_cost : ℝ := total_cost - total_goat_cost
  let llama_cost : ℝ := total_llama_cost / num_llamas
  (llama_cost - goat_cost) / goat_cost * 100 = 50 := by
sorry

end llama_cost_increase_l2877_287778


namespace equation_solution_l2877_287737

theorem equation_solution (x : ℝ) : (4 + 2*x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -2 ∨ x = 1 := by
  sorry

end equation_solution_l2877_287737


namespace right_triangle_hypotenuse_l2877_287701

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 :=
by
  sorry

end right_triangle_hypotenuse_l2877_287701


namespace exists_valid_coloring_l2877_287783

-- Define the color type
inductive Color
  | BLUE
  | GREEN
  | RED
  | YELLOW

-- Define the coloring function type
def ColoringFunction := ℤ → Color

-- Define the property that the coloring function must satisfy
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ a b c d : ℤ, f a = f b ∧ f b = f c ∧ f c = f d ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) →
    3 * a - 2 * b ≠ 2 * c - 3 * d

-- State the theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction, ValidColoring f := by
  sorry

end exists_valid_coloring_l2877_287783


namespace focus_of_parabola_is_correct_l2877_287719

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A parabola in the 2D plane -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ

/-- The given circle from the problem -/
def given_circle : Circle :=
  { center := (3, 0), radius := 4 }

/-- Checks if a point is on the given circle -/
def on_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

/-- The parabola from the problem -/
def given_parabola : Parabola :=
  { p := 1, focus := (1, 0) }

/-- Checks if a point is on the given parabola -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 2 * given_parabola.p * x

/-- The theorem to be proved -/
theorem focus_of_parabola_is_correct :
  given_parabola.focus = (1, 0) ∧
  given_parabola.p > 0 ∧
  ∃ (x y : ℝ), on_circle x y ∧ on_parabola x y :=
sorry

end focus_of_parabola_is_correct_l2877_287719


namespace complex_equation_sum_l2877_287784

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I = 2 + b * Complex.I) → a + b = 1 := by
  sorry

end complex_equation_sum_l2877_287784


namespace cube_fourth_power_difference_l2877_287755

theorem cube_fourth_power_difference (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 22) : 
  ∃ (p q : ℕ+), 
    x = p^2 ∧ 
    y = p^3 ∧ 
    z = q^3 ∧ 
    w = q^4 ∧ 
    q^3 - p^2 = 22 ∧ 
    w - y = (q^(4/3))^3 - p^3 := by
  sorry

end cube_fourth_power_difference_l2877_287755


namespace triangle_problem_l2877_287732

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_angle_correspondence : True -- This is a placeholder for the correspondence between sides and angles

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) :
  (3 * t.a = 2 * t.b) →
  ((t.B = Real.pi / 3 → Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6) ∧
   (t.b - t.c = (1 / 3) * t.a → Real.cos t.C = 17 / 27)) := by
  sorry


end triangle_problem_l2877_287732


namespace pyramid_sum_is_25_l2877_287720

/-- Calculates the sum of blocks in a pyramid with given parameters -/
def pyramidSum (levels : Nat) (firstRowBlocks : Nat) (decrease : Nat) : Nat :=
  let blockSequence := List.range levels |>.map (fun i => firstRowBlocks - i * decrease)
  blockSequence.sum

/-- The sum of blocks in a 5-level pyramid with specific parameters is 25 -/
theorem pyramid_sum_is_25 : pyramidSum 5 9 2 = 25 := by
  sorry

end pyramid_sum_is_25_l2877_287720


namespace a_gt_3_sufficient_not_necessary_for_abs_a_gt_3_l2877_287796

theorem a_gt_3_sufficient_not_necessary_for_abs_a_gt_3 :
  (∃ a : ℝ, a > 3 → |a| > 3) ∧ 
  (∃ a : ℝ, |a| > 3 ∧ ¬(a > 3)) :=
by sorry

end a_gt_3_sufficient_not_necessary_for_abs_a_gt_3_l2877_287796


namespace batsman_average_increase_l2877_287766

theorem batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) :
  total_innings = 12 →
  last_innings_score = 65 →
  final_average = 43 →
  (total_innings * final_average - last_innings_score) / (total_innings - 1) = 41 →
  final_average - (total_innings * final_average - last_innings_score) / (total_innings - 1) = 2 :=
by sorry

end batsman_average_increase_l2877_287766


namespace randy_biscuits_left_l2877_287707

/-- The number of biscuits Randy is left with after receiving and losing some -/
def biscuits_left (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (eaten_by_brother : ℕ) : ℕ :=
  initial + from_father + from_mother - eaten_by_brother

/-- Theorem stating that Randy is left with 40 biscuits -/
theorem randy_biscuits_left : biscuits_left 32 13 15 20 = 40 := by
  sorry

end randy_biscuits_left_l2877_287707


namespace problem_statement_l2877_287727

theorem problem_statement (x y z : ℝ) (hx : x = 2) (hy : y = -3) (hz : z = 1) :
  x^2 + y^2 + z^2 + 2*x*y - z^3 = 1 := by
  sorry

end problem_statement_l2877_287727


namespace fourth_grade_students_l2877_287754

def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

theorem fourth_grade_students :
  final_student_count 10 4 42 = 48 := by
  sorry

end fourth_grade_students_l2877_287754


namespace gcf_of_75_and_100_l2877_287775

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcf_of_75_and_100_l2877_287775


namespace minor_premise_identification_l2877_287728

-- Define the basic shapes
inductive Shape
| Rectangle
| Parallelogram
| Triangle

-- Define the properties of shapes
def isParallelogram : Shape → Prop
  | Shape.Rectangle => true
  | Shape.Parallelogram => true
  | Shape.Triangle => false

-- Define the syllogism structure
structure Syllogism where
  majorPremise : Prop
  minorPremise : Prop
  conclusion : Prop

-- Define our specific syllogism
def ourSyllogism : Syllogism := {
  majorPremise := isParallelogram Shape.Rectangle
  minorPremise := ¬ isParallelogram Shape.Triangle
  conclusion := Shape.Triangle ≠ Shape.Rectangle
}

-- Theorem to prove
theorem minor_premise_identification :
  ourSyllogism.minorPremise = ¬ isParallelogram Shape.Triangle :=
by sorry

end minor_premise_identification_l2877_287728


namespace sum_of_squared_coefficients_l2877_287730

def original_expression (x : ℝ) : ℝ := 3 * (x^3 - 4*x^2 + x) - 5 * (x^3 + 2*x^2 - 5*x + 3)

def simplified_expression (x : ℝ) : ℝ := -2*x^3 - 22*x^2 + 28*x - 15

def coefficients : List ℤ := [-2, -22, 28, -15]

theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 1497 :=
sorry

end sum_of_squared_coefficients_l2877_287730


namespace fermat_like_equation_power_l2877_287742

theorem fermat_like_equation_power (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ l : ℕ, n = p^l :=
by sorry

end fermat_like_equation_power_l2877_287742


namespace nested_sqrt_value_l2877_287756

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_sqrt_value_l2877_287756


namespace composite_function_equality_l2877_287771

theorem composite_function_equality (x : ℝ) (hx : x > 0) :
  Real.sin (Real.log (Real.sqrt x)) = Real.sin ((1 / 2) * Real.log x) := by
  sorry

end composite_function_equality_l2877_287771


namespace orchard_apples_count_l2877_287729

theorem orchard_apples_count (total_apples : ℕ) : 
  (40 : ℕ) * total_apples = (100 : ℕ) * (40 : ℕ) * (24 : ℕ) / ((100 : ℕ) - (70 : ℕ)) →
  total_apples = 200 := by
  sorry

end orchard_apples_count_l2877_287729


namespace new_sales_tax_percentage_l2877_287792

/-- Proves that the new sales tax percentage is 3 1/3% given the conditions --/
theorem new_sales_tax_percentage
  (market_price : ℝ)
  (original_tax_rate : ℝ)
  (savings : ℝ)
  (h1 : market_price = 10800)
  (h2 : original_tax_rate = 3.5 / 100)
  (h3 : savings = 18) :
  let original_tax := market_price * original_tax_rate
  let new_tax := original_tax - savings
  let new_tax_rate := new_tax / market_price
  new_tax_rate = 10 / 3 / 100 := by sorry

end new_sales_tax_percentage_l2877_287792


namespace binary_calculation_l2877_287746

-- Define binary numbers as natural numbers
def bin110110 : ℕ := 54  -- 110110 in base 2 is 54 in base 10
def bin101110 : ℕ := 46  -- 101110 in base 2 is 46 in base 10
def bin100 : ℕ := 4      -- 100 in base 2 is 4 in base 10
def bin11100011110 : ℕ := 1886  -- 11100011110 in base 2 is 1886 in base 10

-- State the theorem
theorem binary_calculation :
  (bin110110 / bin100) * bin101110 = bin11100011110 := by
  sorry

end binary_calculation_l2877_287746


namespace triangles_in_4x4_grid_l2877_287782

/-- Represents a triangular grid with side length n --/
def TriangularGrid (n : ℕ) := Unit

/-- Counts the number of triangles in a triangular grid --/
def countTriangles (grid : TriangularGrid 4) : ℕ := sorry

/-- Theorem: The number of triangles in a 4x4 triangular grid is 20 --/
theorem triangles_in_4x4_grid :
  ∀ (grid : TriangularGrid 4), countTriangles grid = 20 := by sorry

end triangles_in_4x4_grid_l2877_287782


namespace apple_pear_difference_l2877_287799

theorem apple_pear_difference :
  let num_apples : ℕ := 17
  let num_pears : ℕ := 9
  num_apples - num_pears = 8 := by sorry

end apple_pear_difference_l2877_287799


namespace at_least_one_geq_two_l2877_287744

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) := by
  sorry

end at_least_one_geq_two_l2877_287744


namespace points_per_question_l2877_287743

theorem points_per_question (correct_answers : ℕ) (final_score : ℕ) : 
  correct_answers = 5 → final_score = 15 → (final_score / correct_answers : ℚ) = 3 := by
  sorry

end points_per_question_l2877_287743


namespace solution_value_a_l2877_287751

theorem solution_value_a (a x y : ℝ) : 
  a * x - 3 * y = 0 ∧ x + y = 1 ∧ 2 * x + y = 0 → a = -6 := by
  sorry

end solution_value_a_l2877_287751


namespace product_difference_implies_sum_l2877_287776

theorem product_difference_implies_sum (p q : ℤ) 
  (h1 : p * q = 1764) 
  (h2 : p - q = 20) : 
  p + q = 86 := by
sorry

end product_difference_implies_sum_l2877_287776


namespace positive_integer_solution_exists_l2877_287723

theorem positive_integer_solution_exists (a : ℤ) (h : a > 2) :
  ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 := by
  sorry

end positive_integer_solution_exists_l2877_287723


namespace coconut_trips_l2877_287709

def total_coconuts : ℕ := 144
def barbie_capacity : ℕ := 4
def bruno_capacity : ℕ := 8

theorem coconut_trips : 
  (total_coconuts / (barbie_capacity + bruno_capacity) : ℕ) = 12 := by
  sorry

end coconut_trips_l2877_287709


namespace expand_expression_l2877_287708

theorem expand_expression (x y : ℝ) : 
  (6*x + 8 - 3*y) * (4*x - 5*y) = 24*x^2 - 42*x*y + 32*x - 40*y + 15*y^2 := by
  sorry

end expand_expression_l2877_287708


namespace probability_age_less_than_20_l2877_287767

theorem probability_age_less_than_20 (total : ℕ) (age_over_30 : ℕ) (age_under_20 : ℕ) :
  total = 120 →
  age_over_30 = 90 →
  age_under_20 = total - age_over_30 →
  (age_under_20 : ℚ) / (total : ℚ) = 1 / 4 :=
by sorry

end probability_age_less_than_20_l2877_287767


namespace min_value_of_f_l2877_287788

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  f x = -20 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 3 → f y ≥ f x :=
by sorry

end min_value_of_f_l2877_287788


namespace sequence_properties_l2877_287779

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := (1/2) * n^2 + (1/2) * n

/-- The general term of sequence a_n -/
def a (n : ℕ) : ℚ := n

/-- The n-th term of sequence b_n -/
def b (n : ℕ) : ℚ := a n * 2^(n-1)

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℚ := (n-1) * 2^n + 1

theorem sequence_properties (n : ℕ) :
  (∀ k, S k = (1/2) * k^2 + (1/2) * k) →
  (a n = n) ∧
  (T n = (n-1) * 2^n + 1) := by
  sorry

end sequence_properties_l2877_287779


namespace fraction_value_at_three_l2877_287703

theorem fraction_value_at_three :
  let x : ℝ := 3
  (x^8 + 8*x^4 + 16) / (x^4 - 4) = 93 := by sorry

end fraction_value_at_three_l2877_287703


namespace john_profit_is_13100_l2877_287748

/-- Calculates the profit made by John from chopping trees and selling tables -/
def john_profit : ℕ := by
  -- Define the number of trees in each group
  let trees_group1 : ℕ := 10
  let trees_group2 : ℕ := 10
  let trees_group3 : ℕ := 10

  -- Define the number of planks per tree in each group
  let planks_per_tree_group1 : ℕ := 20
  let planks_per_tree_group2 : ℕ := 25
  let planks_per_tree_group3 : ℕ := 30

  -- Define the labor cost per tree in each group
  let labor_cost_group1 : ℕ := 120
  let labor_cost_group2 : ℕ := 80
  let labor_cost_group3 : ℕ := 60

  -- Define the number of planks required to make a table
  let planks_per_table : ℕ := 15

  -- Define the selling price for each group of tables
  let price_tables_1_10 : ℕ := 350
  let price_tables_11_30 : ℕ := 325
  let price_decrease_per_5_tables : ℕ := 10

  -- Calculate the total number of planks
  let total_planks : ℕ := 
    trees_group1 * planks_per_tree_group1 +
    trees_group2 * planks_per_tree_group2 +
    trees_group3 * planks_per_tree_group3

  -- Calculate the total number of tables
  let total_tables : ℕ := total_planks / planks_per_table

  -- Calculate the total labor cost
  let total_labor_cost : ℕ := 
    trees_group1 * labor_cost_group1 +
    trees_group2 * labor_cost_group2 +
    trees_group3 * labor_cost_group3

  -- Calculate the total revenue
  let total_revenue : ℕ := 
    10 * price_tables_1_10 +
    20 * price_tables_11_30 +
    5 * (price_tables_11_30 - price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 2 * price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 3 * price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 4 * price_decrease_per_5_tables)

  -- Calculate the profit
  let profit : ℕ := total_revenue - total_labor_cost

  -- Prove that the profit is equal to 13100
  sorry

theorem john_profit_is_13100 : john_profit = 13100 := by sorry

end john_profit_is_13100_l2877_287748


namespace ratio_odd_even_divisors_l2877_287749

def N : ℕ := 18 * 52 * 75 * 98

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N) * 30 = sum_even_divisors N :=
sorry

end ratio_odd_even_divisors_l2877_287749


namespace triangle_right_angled_l2877_287765

theorem triangle_right_angled (A B C : Real) (a b c : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  a = -c * Real.cos (A + C) →
  a^2 + b^2 = c^2 :=
sorry

end triangle_right_angled_l2877_287765


namespace kennel_dogs_count_l2877_287795

theorem kennel_dogs_count (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 2 / 3 →
  cats = dogs - 6 →
  dogs = 18 := by
sorry

end kennel_dogs_count_l2877_287795


namespace sweets_neither_red_nor_green_l2877_287745

/-- Given a bowl of sweets with red, green, and other colors, calculate the number of sweets that are neither red nor green. -/
theorem sweets_neither_red_nor_green 
  (total : ℕ) 
  (red : ℕ) 
  (green : ℕ) 
  (h_total : total = 285) 
  (h_red : red = 49) 
  (h_green : green = 59) :
  total - (red + green) = 177 := by
  sorry

end sweets_neither_red_nor_green_l2877_287745


namespace min_value_w_l2877_287710

/-- The minimum value of w = 2x^2 + 3y^2 + 8x - 5y + 30 is 26.25 -/
theorem min_value_w :
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 ≥ 26.25) ∧
  (∃ x y : ℝ, 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 = 26.25) := by
  sorry

end min_value_w_l2877_287710


namespace intersection_complement_equality_l2877_287772

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by
  sorry

end intersection_complement_equality_l2877_287772


namespace fixed_point_of_exponential_function_l2877_287761

theorem fixed_point_of_exponential_function 
  (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 1
  f 2 = 0 := by sorry

end fixed_point_of_exponential_function_l2877_287761


namespace sum_of_first_six_primes_mod_seventh_prime_l2877_287780

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_of_first_six_primes_mod_seventh_prime : 
  (first_six_primes.sum) % seventh_prime = 7 := by
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l2877_287780


namespace investment_scientific_notation_l2877_287787

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The total industrial investment in yuan -/
def total_investment : ℝ := 314.86 * 10^9

theorem investment_scientific_notation :
  to_scientific_notation total_investment = ScientificNotation.mk 3.1486 10 sorry := by
  sorry

end investment_scientific_notation_l2877_287787


namespace radio_survey_female_nonlisteners_l2877_287735

theorem radio_survey_female_nonlisteners (total_surveyed : ℕ) 
  (males_listen females_dont_listen total_listen total_dont_listen : ℕ) :
  total_surveyed = total_listen + total_dont_listen →
  males_listen ≤ total_listen →
  females_dont_listen ≤ total_dont_listen →
  total_surveyed = 255 →
  males_listen = 45 →
  total_listen = 120 →
  total_dont_listen = 135 →
  females_dont_listen = 87 →
  females_dont_listen = 87 :=
by sorry

end radio_survey_female_nonlisteners_l2877_287735


namespace sphere_circular_views_l2877_287786

-- Define a type for geometric bodies
inductive GeometricBody
  | Cone
  | Sphere
  | Cylinder
  | HollowCylinder

-- Define a function to check if a view is circular
def isCircularView (body : GeometricBody) (view : String) : Prop :=
  match body, view with
  | GeometricBody.Sphere, _ => True
  | _, _ => False

-- Main theorem
theorem sphere_circular_views :
  ∀ (body : GeometricBody),
    (isCircularView body "main" ∧
     isCircularView body "left" ∧
     isCircularView body "top") →
    body = GeometricBody.Sphere :=
by sorry

end sphere_circular_views_l2877_287786


namespace xy_value_l2877_287752

theorem xy_value (x y : ℝ) (h : |x - 2*y| + (5*x - 7*y - 3)^2 = 0) : x^y = 2 := by
  sorry

end xy_value_l2877_287752


namespace otimes_four_otimes_four_four_l2877_287706

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 + 3*x*y - y

-- Theorem statement
theorem otimes_four_otimes_four_four : otimes 4 (otimes 4 4) = 1252 := by
  sorry

end otimes_four_otimes_four_four_l2877_287706


namespace complex_equation_solution_l2877_287713

theorem complex_equation_solution (a : ℝ) (i : ℂ) : 
  i * i = -1 → (a - i)^2 = 2*i → a = -1 := by sorry

end complex_equation_solution_l2877_287713


namespace sqrt_sum_inequality_l2877_287798

theorem sqrt_sum_inequality (x y α : ℝ) :
  Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α) →
  x + y ≥ 2 * α := by
  sorry

end sqrt_sum_inequality_l2877_287798


namespace right_triangles_count_l2877_287790

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if a triangle is right-angled with vertex c as the right angle -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Checks if a point is a lattice point (has integer coordinates) -/
def isLatticePoint (p : Point) : Prop := sorry

/-- Calculates the incenter of a triangle -/
def incenter (t : Triangle) : Point := sorry

/-- Counts the number of right triangles satisfying the given conditions -/
def countRightTriangles (p : ℕ) (isPrime : Nat.Prime p) : ℕ := sorry

/-- The main theorem -/
theorem right_triangles_count (p : ℕ) (isPrime : Nat.Prime p) :
  let m := Point.mk (p * 1994) (7 * p * 1994)
  countRightTriangles p isPrime =
    if p = 2 then 18
    else if p = 997 then 20
    else 36 := by
  sorry

end right_triangles_count_l2877_287790


namespace rebecca_eggs_count_l2877_287797

/-- Proves that Rebecca has 13 eggs given the problem conditions -/
theorem rebecca_eggs_count :
  ∀ (total_items : ℕ) (group_size : ℕ) (num_groups : ℕ) (num_marbles : ℕ),
    group_size = 2 →
    num_groups = 8 →
    num_marbles = 3 →
    total_items = group_size * num_groups →
    total_items = num_marbles + (total_items - num_marbles) →
    (total_items - num_marbles) = 13 :=
by
  sorry

end rebecca_eggs_count_l2877_287797


namespace candle_height_relation_l2877_287731

/-- Represents the remaining height of a burning candle -/
def remaining_height (initial_height burning_rate t : ℝ) : ℝ :=
  initial_height - burning_rate * t

/-- Theorem stating the relationship between remaining height and burning time for a specific candle -/
theorem candle_height_relation (h t : ℝ) :
  remaining_height 20 4 t = h ↔ h = 20 - 4 * t := by sorry

end candle_height_relation_l2877_287731


namespace thirtieth_term_of_sequence_l2877_287700

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 17) (h₃ : a₃ = 31) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 409 := by
  sorry

end thirtieth_term_of_sequence_l2877_287700


namespace absolute_value_inequality_l2877_287763

theorem absolute_value_inequality (x : ℝ) : 
  3 ≤ |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by sorry

end absolute_value_inequality_l2877_287763


namespace solve_for_a_l2877_287793

theorem solve_for_a : ∃ a : ℝ, (2 * 1 - a * (-1) = 3) ∧ a = 1 := by sorry

end solve_for_a_l2877_287793


namespace problem_solution_l2877_287726

theorem problem_solution (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + 2 / b = d) (h2 : b + 2 / c = d) (h3 : c + 2 / a = d) :
  d = Real.sqrt 2 ∨ d = -Real.sqrt 2 := by
  sorry

end problem_solution_l2877_287726


namespace polynomial_roots_l2877_287733

theorem polynomial_roots : ∃ (x₁ x₂ x₃ x₄ : ℝ), 
  (x₁ = 0 ∧ x₂ = 1/3 ∧ x₃ = 2 ∧ x₄ = -5) ∧
  (∀ x : ℝ, 3*x^4 + 11*x^3 - 28*x^2 + 10*x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) := by
  sorry

end polynomial_roots_l2877_287733


namespace oleg_event_guests_l2877_287721

theorem oleg_event_guests (total_guests men : ℕ) (h1 : total_guests = 80) (h2 : men = 40) :
  let women := men / 2
  let adults := men + women
  let original_children := total_guests - adults
  original_children + 10 = 30 := by
  sorry

end oleg_event_guests_l2877_287721


namespace cookie_sugar_measurement_l2877_287794

-- Define the amount of sugar needed
def sugar_needed : ℚ := 15/4

-- Define the capacity of the measuring cup
def cup_capacity : ℚ := 1/3

-- Theorem to prove
theorem cookie_sugar_measurement :
  ⌈sugar_needed / cup_capacity⌉ = 12 := by
  sorry

end cookie_sugar_measurement_l2877_287794


namespace floor_ceiling_expression_l2877_287738

theorem floor_ceiling_expression : 
  ⌊⌈(12 / 5 : ℚ)^2⌉ * 3 + 14 / 3⌋ = 22 := by
  sorry

end floor_ceiling_expression_l2877_287738


namespace prob_at_least_three_even_is_five_sixteenths_l2877_287717

/-- Probability of rolling an even number on a fair die -/
def prob_even : ℚ := 1/2

/-- Number of rolls -/
def num_rolls : ℕ := 4

/-- Probability of rolling an even number at least three times in four rolls -/
def prob_at_least_three_even : ℚ :=
  Nat.choose num_rolls 3 * prob_even^3 * (1 - prob_even) +
  Nat.choose num_rolls 4 * prob_even^4

theorem prob_at_least_three_even_is_five_sixteenths :
  prob_at_least_three_even = 5/16 := by
  sorry

end prob_at_least_three_even_is_five_sixteenths_l2877_287717


namespace friendly_snakes_not_green_l2877_287759

structure Snake where
  friendly : Bool
  green : Bool
  can_multiply : Bool
  can_divide : Bool

def Tom_snakes : Finset Snake := sorry

theorem friendly_snakes_not_green :
  ∀ s ∈ Tom_snakes,
  (s.friendly → s.can_multiply) ∧
  (s.green → ¬s.can_divide) ∧
  (¬s.can_divide → ¬s.can_multiply) →
  (s.friendly → ¬s.green) :=
by sorry

end friendly_snakes_not_green_l2877_287759


namespace homework_points_l2877_287716

theorem homework_points (total_points : ℕ) (test_quiz_ratio : ℕ) (quiz_homework_diff : ℕ)
  (h1 : total_points = 265)
  (h2 : test_quiz_ratio = 4)
  (h3 : quiz_homework_diff = 5) :
  ∃ (homework : ℕ), 
    homework + (homework + quiz_homework_diff) + test_quiz_ratio * (homework + quiz_homework_diff) = total_points ∧ 
    homework = 40 := by
  sorry

end homework_points_l2877_287716


namespace arithmetic_sequence_problem_l2877_287774

/-- Given an arithmetic sequence {a_n} with first term a_1 = 1 and common difference d = 3,
    if a_n = 2005, then n = 669. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) : 
  a 1 = 1 →                                 -- First term is 1
  (∀ k, a (k + 1) - a k = 3) →              -- Common difference is 3
  a n = 2005 →                              -- nth term is 2005
  n = 669 := by
sorry

end arithmetic_sequence_problem_l2877_287774


namespace two_fifths_of_number_l2877_287712

theorem two_fifths_of_number (n : ℝ) : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 16 → (2/5 : ℝ) * n = 192 := by
  sorry

end two_fifths_of_number_l2877_287712


namespace problem_solution_l2877_287722

def has_at_least_four_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).card ≥ 4

def divisor_differences_divide (n : ℕ) : Prop :=
  ∀ a b : ℕ, a ∣ n → b ∣ n → 1 < a → a < b → b < n → (b - a) ∣ n

def satisfies_conditions (n : ℕ) : Prop :=
  has_at_least_four_divisors n ∧ divisor_differences_divide n

theorem problem_solution : 
  {n : ℕ | satisfies_conditions n} = {6, 8, 12} := by sorry

end problem_solution_l2877_287722


namespace right_triangle_hypotenuse_l2877_287785

theorem right_triangle_hypotenuse (x : ℚ) :
  let a := 9
  let b := 3 * x + 6
  let c := x + 15
  (a + b + c = 45) →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  (max a (max b c) = 75 / 4) := by
  sorry

end right_triangle_hypotenuse_l2877_287785


namespace total_days_is_30_l2877_287702

/-- The number of days being considered -/
def total_days : ℕ := sorry

/-- The mean daily profit for all days in rupees -/
def mean_profit : ℕ := 350

/-- The mean profit for the first 15 days in rupees -/
def mean_profit_first_15 : ℕ := 275

/-- The mean profit for the last 15 days in rupees -/
def mean_profit_last_15 : ℕ := 425

/-- Theorem stating that the total number of days is 30 -/
theorem total_days_is_30 :
  total_days = 30 ∧
  total_days * mean_profit = 15 * mean_profit_first_15 + 15 * mean_profit_last_15 :=
by sorry

end total_days_is_30_l2877_287702


namespace debate_students_difference_l2877_287734

theorem debate_students_difference (s1 s2 s3 : ℕ) : 
  s1 = 2 * s2 →
  s3 = 200 →
  s1 + s2 + s3 = 920 →
  s2 - s3 = 40 := by
sorry

end debate_students_difference_l2877_287734


namespace function_with_two_symmetry_centers_decomposition_l2877_287739

/-- A function has a center of symmetry at a if f(a-x) + f(a+x) = 2f(a) for all real x -/
def HasCenterOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) + f (a + x) = 2 * f a

/-- A function is linear if f(x) = mx + b for some real m and b -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- A function is periodic if there exists a non-zero real number p such that f(x + p) = f(x) for all real x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- Main theorem: A function with at least two centers of symmetry can be written as the sum of a linear function and a periodic function -/
theorem function_with_two_symmetry_centers_decomposition (f : ℝ → ℝ) 
  (h1 : ∃ p q : ℝ, p ≠ q ∧ HasCenterOfSymmetry f p ∧ HasCenterOfSymmetry f q) :
  ∃ g h : ℝ → ℝ, IsLinear g ∧ IsPeriodic h ∧ ∀ x : ℝ, f x = g x + h x := by
  sorry


end function_with_two_symmetry_centers_decomposition_l2877_287739


namespace diamond_seven_three_l2877_287768

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem diamond_seven_three : diamond 7 3 = 22 := by
  sorry

end diamond_seven_three_l2877_287768


namespace largest_angle_in_special_triangle_l2877_287764

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 6/5 of a right angle
  a + b = 6/5 * 90 →
  -- One angle is 30° larger than the other
  b = a + 30 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 72°
  max a (max b c) = 72 := by
sorry

end largest_angle_in_special_triangle_l2877_287764


namespace lines_in_same_plane_l2877_287704

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the necessary relations
variable (lies_in : Point → Line → Prop)
variable (lies_in_plane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_in_same_plane 
  (a b : Line) 
  (α : Plane) 
  (h1 : intersect a b) 
  (h2 : lies_in_plane a α) 
  (h3 : lies_in_plane b α) :
  ∀ (c : Line), parallel c b → (∃ (p : Point), lies_in p a ∧ lies_in p c) → 
  lies_in_plane c α :=
sorry

end lines_in_same_plane_l2877_287704


namespace five_digit_palindromes_count_l2877_287718

/-- A function that counts the number of 5-digit palindromes -/
def count_5digit_palindromes : ℕ :=
  let A := 9  -- digits 1 to 9
  let B := 10 -- digits 0 to 9
  let C := 10 -- digits 0 to 9
  A * B * C

/-- Theorem stating that the number of 5-digit palindromes is 900 -/
theorem five_digit_palindromes_count : count_5digit_palindromes = 900 := by
  sorry

end five_digit_palindromes_count_l2877_287718


namespace vector_relation_l2877_287741

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem vector_relation (h : B - A = 2 • (D - C)) :
  B - D = A - C - (3/2 : ℝ) • (B - A) := by
  sorry

end vector_relation_l2877_287741


namespace pills_remaining_l2877_287769

def initial_pills : ℕ := 200
def daily_dose : ℕ := 12
def days : ℕ := 14

theorem pills_remaining : initial_pills - (daily_dose * days) = 32 := by
  sorry

end pills_remaining_l2877_287769


namespace last_number_is_25_l2877_287736

theorem last_number_is_25 (numbers : Fin 7 → ℝ) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 13 →
  (numbers 3 + numbers 4 + numbers 5 + numbers 6) / 4 = 15 →
  numbers 4 + numbers 5 + numbers 6 = 55 →
  (numbers 3) ^ 2 = numbers 6 →
  numbers 6 = 25 := by
sorry

end last_number_is_25_l2877_287736


namespace all_admissible_triangles_finite_and_generable_l2877_287753

-- Define an admissible angle
def AdmissibleAngle (n : ℕ) (m : ℕ) : ℚ := (m * 180) / n

-- Define an admissible triangle
structure AdmissibleTriangle (n : ℕ) where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  sum_180 : AdmissibleAngle n angle1 + AdmissibleAngle n angle2 + AdmissibleAngle n angle3 = 180
  angle1_pos : angle1 > 0
  angle2_pos : angle2 > 0
  angle3_pos : angle3 > 0

-- Define a function to check if two triangles are similar
def areSimilar (n : ℕ) (t1 t2 : AdmissibleTriangle n) : Prop :=
  (t1.angle1 = t2.angle1 ∧ t1.angle2 = t2.angle2 ∧ t1.angle3 = t2.angle3) ∨
  (t1.angle1 = t2.angle2 ∧ t1.angle2 = t2.angle3 ∧ t1.angle3 = t2.angle1) ∨
  (t1.angle1 = t2.angle3 ∧ t1.angle2 = t2.angle1 ∧ t1.angle3 = t2.angle2)

-- Define the set of all possible admissible triangles
def AllAdmissibleTriangles (n : ℕ) : Set (AdmissibleTriangle n) :=
  {t : AdmissibleTriangle n | True}

-- Define the process of cutting triangles
def CutTriangle (n : ℕ) (t : AdmissibleTriangle n) : 
  Option (AdmissibleTriangle n × AdmissibleTriangle n) :=
  sorry -- Implementation details omitted

-- The main theorem
theorem all_admissible_triangles_finite_and_generable 
  (n : ℕ) (h_prime : Nat.Prime n) (h_gt_3 : n > 3) :
  ∃ (S : Set (AdmissibleTriangle n)),
    Finite S ∧ 
    (∀ t ∈ AllAdmissibleTriangles n, ∃ s ∈ S, areSimilar n t s) ∧
    (∀ t ∈ S, CutTriangle n t = none) :=
  sorry


end all_admissible_triangles_finite_and_generable_l2877_287753


namespace dice_sum_not_sixteen_l2877_287715

theorem dice_sum_not_sixteen (a b c d e : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  1 ≤ e ∧ e ≤ 6 →
  a * b * c * d * e = 72 →
  a + b + c + d + e ≠ 16 := by
sorry

end dice_sum_not_sixteen_l2877_287715


namespace tan_two_implies_fraction_five_l2877_287773

theorem tan_two_implies_fraction_five (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end tan_two_implies_fraction_five_l2877_287773


namespace no_roots_around_1000_l2877_287711

/-- A quadratic trinomial -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.value (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- A list of quadratic trinomials -/
def trinomials : List QuadraticTrinomial := sorry

/-- The sum of all quadratic trinomials in the list -/
def f (x : ℝ) : ℝ :=
  (trinomials.map (fun q => q.value x)).sum

/-- All trinomials are positive at x = 1000 -/
axiom all_positive : ∀ q ∈ trinomials, q.value 1000 > 0

/-- Theorem: It's impossible for f to have one root less than 1000 and another greater than 1000 -/
theorem no_roots_around_1000 : ¬∃ (r₁ r₂ : ℝ), r₁ < 1000 ∧ r₂ > 1000 ∧ f r₁ = 0 ∧ f r₂ = 0 := by
  sorry

end no_roots_around_1000_l2877_287711


namespace circle_equation_from_diameter_l2877_287760

/-- Given two points A and B in the plane, this theorem states that the equation of the circle
    for which the segment AB is a diameter is (x-1)^2+(y+3)^2=116. -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (-4, -5) →
  B = (6, -1) →
  ∀ x y : ℝ, (x - 1)^2 + (y + 3)^2 = 116 ↔
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    x = -4 * (1 - t) + 6 * t ∧
    y = -5 * (1 - t) - 1 * t :=
by sorry

end circle_equation_from_diameter_l2877_287760
