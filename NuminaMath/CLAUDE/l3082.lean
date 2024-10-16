import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_l3082_308278

theorem fraction_equality : (2023^2 - 2016^2) / (2042^2 - 1997^2) = 7 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3082_308278


namespace NUMINAMATH_CALUDE_mary_balloon_count_l3082_308200

/-- The number of yellow balloons each person has -/
structure BalloonCount where
  fred : ℕ
  sam : ℕ
  mary : ℕ

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- The actual balloon count for Fred, Sam, and Mary -/
def actual_count : BalloonCount where
  fred := 5
  sam := 6
  mary := 7

/-- Theorem stating that Mary has 7 yellow balloons -/
theorem mary_balloon_count :
  ∀ (count : BalloonCount),
    count.fred = actual_count.fred →
    count.sam = actual_count.sam →
    count.fred + count.sam + count.mary = total_balloons →
    count.mary = actual_count.mary :=
by
  sorry

end NUMINAMATH_CALUDE_mary_balloon_count_l3082_308200


namespace NUMINAMATH_CALUDE_ninety_percent_of_nine_thousand_l3082_308292

theorem ninety_percent_of_nine_thousand (total_population : ℕ) (percentage : ℚ) : 
  total_population = 9000 → percentage = 90 / 100 → 
  (percentage * total_population : ℚ) = 8100 := by
  sorry

end NUMINAMATH_CALUDE_ninety_percent_of_nine_thousand_l3082_308292


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2019_l3082_308227

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)
def S (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- State the theorem
theorem arithmetic_sequence_sum_2019 (a₁ d : ℤ) :
  a₁ = -2017 →
  (S a₁ d 2017 / 2017 - S a₁ d 2015 / 2015 = 2) →
  S a₁ d 2019 = 2019 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2019_l3082_308227


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l3082_308277

/-- Prove that given the conditions of the cyclist problem, the speed of cyclist C is 10 mph. -/
theorem cyclist_speed_problem (c d : ℝ) : 
  d = c + 5 →  -- C travels 5 mph slower than D
  (80 - 16) / c = (80 + 16) / d →  -- Travel times are equal
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l3082_308277


namespace NUMINAMATH_CALUDE_same_function_fifth_root_power_l3082_308279

theorem same_function_fifth_root_power (x : ℝ) : x = (x^5)^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_same_function_fifth_root_power_l3082_308279


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3082_308214

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3082_308214


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3082_308282

/-- A cylinder with base area S whose lateral surface unfolds into a square has lateral surface area 4πS -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let h := 2 * Real.pi * r
  h = 2 * Real.pi * r →
  2 * Real.pi * r * h = 4 * Real.pi * S :=
by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3082_308282


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3082_308257

theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let circumference := 2 * Real.pi * r
  let height := circumference
  let lateral_area := circumference * height
  lateral_area = 4 * Real.pi * S := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3082_308257


namespace NUMINAMATH_CALUDE_binomial_coefficient_17_8_l3082_308207

theorem binomial_coefficient_17_8 (h1 : Nat.choose 15 6 = 5005) 
                                  (h2 : Nat.choose 15 7 = 6435) 
                                  (h3 : Nat.choose 15 8 = 6435) : 
  Nat.choose 17 8 = 24310 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_17_8_l3082_308207


namespace NUMINAMATH_CALUDE_hamburger_cost_l3082_308218

/-- Proves that the cost of each hamburger is $4 given the initial amount,
    the number of items purchased, the cost of milkshakes, and the remaining amount. -/
theorem hamburger_cost (initial_amount : ℕ) (num_hamburgers : ℕ) (num_milkshakes : ℕ)
                        (milkshake_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 120 →
  num_hamburgers = 8 →
  num_milkshakes = 6 →
  milkshake_cost = 3 →
  remaining_amount = 70 →
  ∃ (hamburger_cost : ℕ),
    initial_amount = num_hamburgers * hamburger_cost + num_milkshakes * milkshake_cost + remaining_amount ∧
    hamburger_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_cost_l3082_308218


namespace NUMINAMATH_CALUDE_fifi_closet_hangers_l3082_308252

theorem fifi_closet_hangers :
  let pink : ℕ := 7
  let green : ℕ := 4
  let blue : ℕ := green - 1
  let yellow : ℕ := blue - 1
  pink + green + blue + yellow = 16 :=
by sorry

end NUMINAMATH_CALUDE_fifi_closet_hangers_l3082_308252


namespace NUMINAMATH_CALUDE_existence_condition_l3082_308219

theorem existence_condition (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2*x - a ≥ 0) ↔ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_condition_l3082_308219


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3082_308239

theorem two_digit_number_problem : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n % 10 = n / 10 + 4) ∧ 
  (n * (n / 10 + n % 10) = 208) ∧
  n = 26 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l3082_308239


namespace NUMINAMATH_CALUDE_solution_set_min_value_l3082_308269

-- Define the function f
def f (x : ℝ) := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set (x : ℝ) : f x ≥ 2 ↔ x ≤ -7 ∨ x ≥ 5/3 :=
sorry

-- Theorem for the minimum value of f(x)
theorem min_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_min_value_l3082_308269


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l3082_308241

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def abc_to_num (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def bca_to_num (b c a : ℕ) : ℕ := 100 * b + 10 * c + a
def cab_to_num (c a b : ℕ) : ℕ := 100 * c + 10 * a + b

def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    n = abc_to_num a b c ∧
    is_three_digit n ∧
    2 * n = bca_to_num b c a + cab_to_num c a b

theorem three_digit_number_theorem :
  {n : ℕ | satisfies_condition n} = 
  {111, 222, 333, 370, 407, 444, 481, 518, 555, 592, 629, 666, 777, 888, 999} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l3082_308241


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3082_308259

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

-- Define the universal set U
def U : Set ℕ := A ∪ B

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3082_308259


namespace NUMINAMATH_CALUDE_A_B_symmetric_x_l3082_308281

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Define symmetry with respect to x-axis
def symmetric_x (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem A_B_symmetric_x : symmetric_x A B := by
  sorry

end NUMINAMATH_CALUDE_A_B_symmetric_x_l3082_308281


namespace NUMINAMATH_CALUDE_hexagon_pattern_triangle_area_l3082_308273

/-- The area of a triangle formed by centers of alternate hexagons in a hexagonal pattern -/
theorem hexagon_pattern_triangle_area :
  ∀ (hexagon_side_length : ℝ) (triangle_side_length : ℝ),
    hexagon_side_length = 1 →
    triangle_side_length = 3 * hexagon_side_length →
    ∃ (triangle_area : ℝ),
      triangle_area = (9 * Real.sqrt 3) / 4 ∧
      triangle_area = (Real.sqrt 3 / 4) * triangle_side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_pattern_triangle_area_l3082_308273


namespace NUMINAMATH_CALUDE_total_books_read_formula_l3082_308234

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month := 5
  let months_per_year := 12
  let books_per_student_per_year := books_per_month * months_per_year
  books_per_student_per_year * c * s

/-- Theorem stating the total number of books read by the entire student body in one year -/
theorem total_books_read_formula (c s : ℕ) :
  total_books_read c s = 60 * c * s :=
by sorry

end NUMINAMATH_CALUDE_total_books_read_formula_l3082_308234


namespace NUMINAMATH_CALUDE_johnny_red_pencils_l3082_308270

/-- The number of red pencils Johnny bought -/
def total_red_pencils (total_packs : ℕ) (regular_red_per_pack : ℕ) 
  (extra_red_packs_1 : ℕ) (extra_red_per_pack_1 : ℕ)
  (extra_red_packs_2 : ℕ) (extra_red_per_pack_2 : ℕ) : ℕ :=
  total_packs * regular_red_per_pack + 
  extra_red_packs_1 * extra_red_per_pack_1 +
  extra_red_packs_2 * extra_red_per_pack_2

/-- Theorem: Johnny bought 46 red pencils -/
theorem johnny_red_pencils : 
  total_red_pencils 25 1 5 3 6 1 = 46 := by
  sorry

end NUMINAMATH_CALUDE_johnny_red_pencils_l3082_308270


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_m_geq_one_l3082_308291

theorem set_intersection_empty_implies_m_geq_one (m : ℝ) : 
  let M : Set ℝ := {x | x ≤ 1}
  let P : Set ℝ := {x | x ≤ m}
  M ∩ (Set.univ \ P) = ∅ → m ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_m_geq_one_l3082_308291


namespace NUMINAMATH_CALUDE_min_xy_value_l3082_308242

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 8) :
  (x : ℚ) * y ≥ 128 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l3082_308242


namespace NUMINAMATH_CALUDE_sibling_product_sixteen_l3082_308221

/-- Represents a family with a given number of girls and boys -/
structure Family :=
  (girls : ℕ)
  (boys : ℕ)

/-- Calculates the product of sisters and brothers for a member of the family -/
def siblingProduct (f : Family) : ℕ :=
  (f.girls - 1) * f.boys

/-- Theorem: In a family with 5 girls and 4 boys, the product of sisters and brothers is 16 -/
theorem sibling_product_sixteen (f : Family) (h1 : f.girls = 5) (h2 : f.boys = 4) :
  siblingProduct f = 16 := by
  sorry

end NUMINAMATH_CALUDE_sibling_product_sixteen_l3082_308221


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_equals_four_l3082_308245

/-- The function f with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x^2 - a^2 * x

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * x - a^2

theorem local_minimum_implies_a_equals_four :
  ∀ a : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a x ≥ f a 1) →
  f_derivative a 1 = 0 →
  a = 4 := by
  sorry

#check local_minimum_implies_a_equals_four

end NUMINAMATH_CALUDE_local_minimum_implies_a_equals_four_l3082_308245


namespace NUMINAMATH_CALUDE_walt_age_l3082_308230

theorem walt_age (walt_age music_teacher_age : ℕ) : 
  music_teacher_age = 3 * walt_age →
  music_teacher_age + 12 = 2 * (walt_age + 12) →
  walt_age = 12 := by
sorry

end NUMINAMATH_CALUDE_walt_age_l3082_308230


namespace NUMINAMATH_CALUDE_fraction_division_difference_l3082_308204

theorem fraction_division_difference : (5 / 3) / (1 / 6) - 2 / 3 = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_difference_l3082_308204


namespace NUMINAMATH_CALUDE_curvilinearTrapezoidAreaStepsCorrect_l3082_308209

/-- The steps required to calculate the area of a curvilinear trapezoid. -/
inductive CurvilinearTrapezoidAreaStep
  | division
  | approximation
  | summation
  | takingLimit

/-- The list of steps to calculate the area of a curvilinear trapezoid. -/
def curvilinearTrapezoidAreaSteps : List CurvilinearTrapezoidAreaStep :=
  [CurvilinearTrapezoidAreaStep.division,
   CurvilinearTrapezoidAreaStep.approximation,
   CurvilinearTrapezoidAreaStep.summation,
   CurvilinearTrapezoidAreaStep.takingLimit]

/-- Theorem stating that the steps to calculate the area of a curvilinear trapezoid
    are division, approximation, summation, and taking the limit. -/
theorem curvilinearTrapezoidAreaStepsCorrect :
  curvilinearTrapezoidAreaSteps =
    [CurvilinearTrapezoidAreaStep.division,
     CurvilinearTrapezoidAreaStep.approximation,
     CurvilinearTrapezoidAreaStep.summation,
     CurvilinearTrapezoidAreaStep.takingLimit] := by
  sorry

end NUMINAMATH_CALUDE_curvilinearTrapezoidAreaStepsCorrect_l3082_308209


namespace NUMINAMATH_CALUDE_clock_synchronization_l3082_308244

/-- Represents the chiming behavior of a clock -/
structure Clock where
  strikes_per_hour : ℕ
  chime_rate : ℚ

/-- The scenario of the King's and Queen's clocks -/
def clock_scenario (h : ℕ) : Prop :=
  let king_clock : Clock := { strikes_per_hour := h, chime_rate := 3/2 }
  let queen_clock : Clock := { strikes_per_hour := h, chime_rate := 1 }
  (king_clock.chime_rate * queen_clock.strikes_per_hour : ℚ) + 2 = h

/-- The theorem stating that the synchronization occurs at 5 o'clock -/
theorem clock_synchronization : 
  clock_scenario 5 := by sorry

end NUMINAMATH_CALUDE_clock_synchronization_l3082_308244


namespace NUMINAMATH_CALUDE_zero_exponent_is_one_l3082_308265

theorem zero_exponent_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_is_one_l3082_308265


namespace NUMINAMATH_CALUDE_smallest_x_value_l3082_308240

theorem smallest_x_value (x : ℝ) : 
  (((14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x) = (7 * x - 2)) → x ≥ 4/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3082_308240


namespace NUMINAMATH_CALUDE_inverse_g_neg_43_eq_neg_2_l3082_308232

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_neg_43_eq_neg_2 : g (-2) = -43 := by sorry

end NUMINAMATH_CALUDE_inverse_g_neg_43_eq_neg_2_l3082_308232


namespace NUMINAMATH_CALUDE_expression_evaluation_l3082_308250

theorem expression_evaluation :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3082_308250


namespace NUMINAMATH_CALUDE_water_jar_problem_l3082_308271

theorem water_jar_problem (small_jar large_jar : ℝ) 
  (h1 : small_jar > 0) 
  (h2 : large_jar > 0) 
  (h3 : small_jar * (1/4) = large_jar * (1/5)) : 
  (1/5) * small_jar + (1/4) * large_jar = (1/2) * large_jar := by
  sorry

end NUMINAMATH_CALUDE_water_jar_problem_l3082_308271


namespace NUMINAMATH_CALUDE_painting_job_theorem_l3082_308294

/-- Represents the time taken to complete a job given the number of painters -/
def time_to_complete (num_painters : ℕ) : ℚ :=
  12 / num_painters

/-- The problem statement -/
theorem painting_job_theorem :
  let initial_painters : ℕ := 6
  let initial_time : ℚ := 2
  let new_painters : ℕ := 8
  (initial_painters : ℚ) * initial_time = time_to_complete new_painters * new_painters ∧
  time_to_complete new_painters = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_painting_job_theorem_l3082_308294


namespace NUMINAMATH_CALUDE_additive_inverse_solution_equal_surds_solution_l3082_308258

-- Part 1
theorem additive_inverse_solution (x : ℝ) : 
  x^2 + 3*x - 6 = -((-x + 1)) → x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6 := by sorry

-- Part 2
theorem equal_surds_solution (m : ℝ) :
  Real.sqrt (m^2 - 6) = Real.sqrt (6*m + 1) → m = 7 := by sorry

end NUMINAMATH_CALUDE_additive_inverse_solution_equal_surds_solution_l3082_308258


namespace NUMINAMATH_CALUDE_segment_ratio_l3082_308283

/-- Given a line segment GH with points E and F on it, where GE is 3 times EH and GF is 5 times FH,
    prove that EF is 1/12 of GH. -/
theorem segment_ratio (G E F H : ℝ) (h1 : G ≤ E) (h2 : E ≤ F) (h3 : F ≤ H)
  (h4 : E - G = 3 * (H - E)) (h5 : F - G = 5 * (H - F)) :
  (F - E) / (H - G) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_segment_ratio_l3082_308283


namespace NUMINAMATH_CALUDE_intersection_point_l3082_308296

/-- The linear function f(x) = 5x + 1 -/
def f (x : ℝ) : ℝ := 5 * x + 1

/-- The y-axis is the set of points with x-coordinate 0 -/
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- The graph of f is the set of points (x, f(x)) -/
def graph_f : Set (ℝ × ℝ) := {p | p.2 = f p.1}

theorem intersection_point : 
  (Set.inter graph_f y_axis) = {(0, 1)} := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3082_308296


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l3082_308243

theorem sum_of_integers_ending_in_3 :
  let first_term : ℕ := 103
  let last_term : ℕ := 493
  let common_difference : ℕ := 10
  let n : ℕ := (last_term - first_term) / common_difference + 1
  let sum : ℕ := n * (first_term + last_term) / 2
  sum = 11920 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l3082_308243


namespace NUMINAMATH_CALUDE_unique_functional_equation_solution_l3082_308251

theorem unique_functional_equation_solution :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, f (m + f n) = f m + f n + f (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_solution_l3082_308251


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3082_308266

/-- Given that px³ + qx - 10 = 2006 when x = 1, prove that px³ + qx - 10 = -2026 when x = -1 -/
theorem cubic_expression_value (p q : ℝ) 
  (h : p * 1^3 + q * 1 - 10 = 2006) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3082_308266


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l3082_308247

theorem rectangular_prism_problem (m n r : ℕ) : 
  m > 0 → n > 0 → r > 0 → m ≤ n → n ≤ r →
  (m - 2) * (n - 2) * (r - 2) - 
  2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) + 
  4 * ((m - 2) + (n - 2) + (r - 2)) = 1985 →
  ((m = 1 ∧ n = 3 ∧ r = 1987) ∨
   (m = 1 ∧ n = 7 ∧ r = 399) ∨
   (m = 3 ∧ n = 3 ∧ r = 1981) ∨
   (m = 5 ∧ n = 5 ∧ r = 1981) ∨
   (m = 5 ∧ n = 7 ∧ r = 663)) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_problem_l3082_308247


namespace NUMINAMATH_CALUDE_max_cyclic_product_permutation_l3082_308220

def cyclic_product (xs : List ℕ) : ℕ :=
  let n := xs.length
  List.sum (List.zipWith (· * ·) xs (xs.rotateLeft 1))

theorem max_cyclic_product_permutation :
  let perms := List.permutations [1, 2, 3, 4, 5]
  let max_val := perms.map cyclic_product |>.maximum?
  let max_count := (perms.filter (λ p ↦ cyclic_product p = max_val.getD 0)).length
  (max_val.getD 0 = 48) ∧ (max_count = 10) := by
  sorry

end NUMINAMATH_CALUDE_max_cyclic_product_permutation_l3082_308220


namespace NUMINAMATH_CALUDE_conditional_probability_not_first_class_l3082_308268

def total_products : ℕ := 8
def first_class_products : ℕ := 6
def selected_products : ℕ := 2

theorem conditional_probability_not_first_class 
  (h1 : total_products = 8)
  (h2 : first_class_products = 6)
  (h3 : selected_products = 2)
  (h4 : first_class_products < total_products)
  (h5 : selected_products ≤ total_products) :
  (Nat.choose first_class_products 1 * Nat.choose (total_products - first_class_products) 1) / 
  (Nat.choose total_products selected_products - Nat.choose first_class_products selected_products) = 12 / 13 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_not_first_class_l3082_308268


namespace NUMINAMATH_CALUDE_parabola_shift_l3082_308206

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift amount
def shift : ℝ := 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - shift) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l3082_308206


namespace NUMINAMATH_CALUDE_anniversary_products_l3082_308203

/-- Commemorative albums and bone china cups problem -/
theorem anniversary_products (total_cost album_cost cup_cost album_price cup_price : ℝ)
  (h1 : total_cost = 312000)
  (h2 : album_cost = 3 * cup_cost)
  (h3 : album_cost + cup_cost = total_cost)
  (h4 : album_price = 1.5 * cup_price)
  (h5 : cup_cost / cup_price - 4 * (album_cost / album_price) = 1600) :
  album_cost = 240000 ∧ cup_cost = 72000 ∧ album_price = 45 ∧ cup_price = 30 := by
sorry

end NUMINAMATH_CALUDE_anniversary_products_l3082_308203


namespace NUMINAMATH_CALUDE_final_amoeba_is_blue_l3082_308249

/-- Represents the color of an amoeba -/
inductive AmoebaCop
  | Red
  | Blue
  | Yellow

/-- Represents the state of the puddle -/
structure PuddleState where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Determines if a number is odd -/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- The initial state of the puddle -/
def initialState : PuddleState :=
  { red := 26, blue := 31, yellow := 16 }

/-- Determines the color of the final amoeba based on the initial state -/
def finalAmoeba (state : PuddleState) : AmoebaCop :=
  if isOdd (state.red - state.blue) ∧ 
     isOdd (state.blue - state.yellow) ∧ 
     ¬isOdd (state.red - state.yellow)
  then AmoebaCop.Blue
  else if isOdd (state.red - state.blue) ∧ 
          isOdd (state.red - state.yellow) ∧ 
          ¬isOdd (state.blue - state.yellow)
  then AmoebaCop.Red
  else AmoebaCop.Yellow

theorem final_amoeba_is_blue :
  finalAmoeba initialState = AmoebaCop.Blue :=
by
  sorry


end NUMINAMATH_CALUDE_final_amoeba_is_blue_l3082_308249


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l3082_308222

theorem triangle_angle_not_all_greater_60 :
  ∀ (a b c : Real),
  (a > 0) → (b > 0) → (c > 0) →
  (a + b + c = 180) →
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l3082_308222


namespace NUMINAMATH_CALUDE_circle_reflection_y_axis_l3082_308217

/-- Given a circle with equation (x+2)^2 + y^2 = 5, 
    its reflection about the y-axis has the equation (x-2)^2 + y^2 = 5 -/
theorem circle_reflection_y_axis (x y : ℝ) :
  ((x + 2)^2 + y^2 = 5) → 
  ∃ (x' y' : ℝ), ((x' - 2)^2 + y'^2 = 5 ∧ x' = -x ∧ y' = y) :=
sorry

end NUMINAMATH_CALUDE_circle_reflection_y_axis_l3082_308217


namespace NUMINAMATH_CALUDE_grace_september_earnings_775_l3082_308262

/-- Represents Grace's landscaping business earnings for September --/
def grace_september_earnings : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun small_lawn_rate large_lawn_rate small_garden_rate large_garden_rate small_mulch_rate large_mulch_rate
      small_lawn_hours large_lawn_hours small_garden_hours large_garden_hours small_mulch_hours large_mulch_hours =>
    small_lawn_rate * small_lawn_hours +
    large_lawn_rate * large_lawn_hours +
    small_garden_rate * small_garden_hours +
    large_garden_rate * large_garden_hours +
    small_mulch_rate * small_mulch_hours +
    large_mulch_rate * large_mulch_hours

/-- Theorem stating that Grace's September earnings were $775 --/
theorem grace_september_earnings_775 :
  grace_september_earnings 6 10 11 15 9 13 20 43 4 5 6 4 = 775 := by
  sorry

end NUMINAMATH_CALUDE_grace_september_earnings_775_l3082_308262


namespace NUMINAMATH_CALUDE_certain_number_proof_l3082_308254

theorem certain_number_proof (n x : ℝ) (h1 : n = -4.5) (h2 : 10 * n = x - 2 * n) : x = -54 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3082_308254


namespace NUMINAMATH_CALUDE_tens_digit_of_sum_l3082_308225

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_sum : tens_digit (2^1500 + 5^768) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_sum_l3082_308225


namespace NUMINAMATH_CALUDE_perpendicular_bisector_intersection_l3082_308202

/-- The perpendicular bisector of two points A and B intersects the line AB at a point C.
    This theorem proves that for specific points A and B, the coordinates of C satisfy a linear equation. -/
theorem perpendicular_bisector_intersection (A B C : ℝ × ℝ) :
  A = (30, 10) →
  B = (6, 3) →
  C.1 = (A.1 + B.1) / 2 →
  C.2 = (A.2 + B.2) / 2 →
  2 * C.1 - 4 * C.2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_intersection_l3082_308202


namespace NUMINAMATH_CALUDE_evaluate_F_4_f_5_l3082_308233

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 3
def F (a b : ℝ) : ℝ := b^3 + a*b

-- State the theorem
theorem evaluate_F_4_f_5 : F 4 (f 5) = 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_4_f_5_l3082_308233


namespace NUMINAMATH_CALUDE_frequency_distribution_theorem_l3082_308208

-- Define the frequency of the first group
def f1 : ℕ := 6

-- Define the frequencies of the second and third groups based on the ratio
def f2 : ℕ := 2 * f1
def f3 : ℕ := 3 * f1

-- Define the sum of frequencies for the first three groups
def sum_first_three : ℕ := f1 + f2 + f3

-- Define the total number of students
def total_students : ℕ := 48

-- Theorem statement
theorem frequency_distribution_theorem :
  sum_first_three < total_students ∧ 
  total_students - sum_first_three > 0 ∧
  total_students - sum_first_three < f3 :=
by sorry

end NUMINAMATH_CALUDE_frequency_distribution_theorem_l3082_308208


namespace NUMINAMATH_CALUDE_pages_copied_eq_500_l3082_308231

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The available amount in dollars -/
def available_dollars : ℕ := 15

/-- The number of pages that can be copied -/
def pages_copied : ℕ := available_dollars * 100 / cost_per_page

theorem pages_copied_eq_500 : pages_copied = 500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_eq_500_l3082_308231


namespace NUMINAMATH_CALUDE_paper_parts_cannot_reach_2020_can_reach_2023_l3082_308237

def paper_sequence : Nat → Nat
  | 0 => 1
  | n + 1 => paper_sequence n + 2

theorem paper_parts (n : Nat) : 
  paper_sequence n = 2 * n + 1 := by sorry

theorem cannot_reach_2020 : 
  ∀ n, paper_sequence n ≠ 2020 := by sorry

theorem can_reach_2023 : 
  ∃ n, paper_sequence n = 2023 := by sorry

end NUMINAMATH_CALUDE_paper_parts_cannot_reach_2020_can_reach_2023_l3082_308237


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l3082_308216

-- Define the angle A
def angle_A : ℝ := 36

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 54 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l3082_308216


namespace NUMINAMATH_CALUDE_percentage_calculation_l3082_308263

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 300) : 1.2 * x = 1800 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3082_308263


namespace NUMINAMATH_CALUDE_preferred_numbers_count_l3082_308212

/-- A function that counts the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- A function that counts the number of four-digit "preferred" numbers. -/
def count_preferred_numbers : ℕ :=
  -- Numbers with two 8s, not in the first position
  choose 3 2 * 8 * 9 +
  -- Numbers with two 8s, including in the first position
  choose 3 1 * 9 * 9 +
  -- Numbers with four 8s
  1

/-- Theorem stating that the count of four-digit "preferred" numbers is 460. -/
theorem preferred_numbers_count : count_preferred_numbers = 460 := by sorry

end NUMINAMATH_CALUDE_preferred_numbers_count_l3082_308212


namespace NUMINAMATH_CALUDE_inequality_solution_l3082_308299

theorem inequality_solution (x : ℝ) :
  (3*x + 4 ≠ 0) →
  (3 - 2 / (3*x + 4) < 5 ↔ x ∈ Set.Ioo (-5/3) (-4/3) ∪ Set.Ioi (-4/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3082_308299


namespace NUMINAMATH_CALUDE_arccos_cos_seven_l3082_308255

theorem arccos_cos_seven : Real.arccos (Real.cos 7) = 7 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_seven_l3082_308255


namespace NUMINAMATH_CALUDE_quadrilateral_exterior_interior_angles_equal_l3082_308289

theorem quadrilateral_exterior_interior_angles_equal :
  ∀ n : ℕ, n ≥ 3 →
  (360 : ℝ) = (n - 2) * 180 ↔ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_exterior_interior_angles_equal_l3082_308289


namespace NUMINAMATH_CALUDE_tims_takeout_cost_l3082_308275

/-- The total cost of Tim's Chinese take-out -/
def total_cost : ℝ := 50

/-- The percentage of the cost that went to entrees -/
def entree_percentage : ℝ := 0.8

/-- The number of appetizers Tim bought -/
def num_appetizers : ℕ := 2

/-- The cost of a single appetizer -/
def appetizer_cost : ℝ := 5

theorem tims_takeout_cost :
  total_cost = (num_appetizers : ℝ) * appetizer_cost / (1 - entree_percentage) :=
by sorry

end NUMINAMATH_CALUDE_tims_takeout_cost_l3082_308275


namespace NUMINAMATH_CALUDE_diego_extra_cans_l3082_308261

theorem diego_extra_cans (martha_cans : ℕ) (total_cans : ℕ) (diego_cans : ℕ) : 
  martha_cans = 90 →
  total_cans = 145 →
  diego_cans = total_cans - martha_cans →
  diego_cans - martha_cans / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_diego_extra_cans_l3082_308261


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l3082_308264

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.6)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l3082_308264


namespace NUMINAMATH_CALUDE_halloween_candy_l3082_308229

/-- The number of candy pieces Debby's sister had -/
def sister_candy : ℕ := 42

/-- The number of candy pieces eaten on the first night -/
def eaten_candy : ℕ := 35

/-- The number of candy pieces left after eating -/
def remaining_candy : ℕ := 39

/-- Debby's candy pieces -/
def debby_candy : ℕ := 32

theorem halloween_candy :
  debby_candy + sister_candy - eaten_candy = remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_l3082_308229


namespace NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l3082_308205

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

-- State the theorem
theorem solution_set_of_f_neg_x (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-x) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l3082_308205


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3082_308211

theorem rectangle_area_increase (l w : ℝ) (h_l : l > 0) (h_w : w > 0) :
  let new_area := (1.15 * l) * (1.25 * w)
  let orig_area := l * w
  (new_area - orig_area) / orig_area = 0.4375 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3082_308211


namespace NUMINAMATH_CALUDE_series_sum_l3082_308295

theorem series_sum : 1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + 10 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3082_308295


namespace NUMINAMATH_CALUDE_parabola_equation_l3082_308210

/-- Given a parabola and a line intersecting it, prove the equation of the parabola. -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y = Real.sqrt 3 * x + (A.2 - Real.sqrt 3 * A.1)) →  -- Line equation
  (∀ x y, x^2 = 2 * p * y) →  -- Parabola equation
  A.1^2 = 2 * p * A.2 →  -- Point A satisfies parabola equation
  B.1^2 = 2 * p * B.2 →  -- Point B satisfies parabola equation
  A.2 = Real.sqrt 3 * A.1 + (A.2 - Real.sqrt 3 * A.1) →  -- Point A satisfies line equation
  B.2 = Real.sqrt 3 * B.1 + (A.2 - Real.sqrt 3 * A.1) →  -- Point B satisfies line equation
  A.1 + B.1 = 3 →  -- Sum of x-coordinates
  (∀ x y, x^2 = Real.sqrt 3 * y) :=  -- Conclusion: equation of the parabola
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3082_308210


namespace NUMINAMATH_CALUDE_richard_patrick_diff_l3082_308276

/-- Bowling game results -/
def bowling_game (patrick_round1 richard_round1_diff : ℕ) : ℕ × ℕ :=
  let richard_round1 := patrick_round1 + richard_round1_diff
  let patrick_round2 := 2 * richard_round1
  let richard_round2 := patrick_round2 - 3
  let patrick_total := patrick_round1 + patrick_round2
  let richard_total := richard_round1 + richard_round2
  (patrick_total, richard_total)

/-- Theorem stating the difference in total pins knocked down -/
theorem richard_patrick_diff (patrick_round1 : ℕ) : 
  (bowling_game patrick_round1 15).2 - (bowling_game patrick_round1 15).1 = 12 :=
by sorry

end NUMINAMATH_CALUDE_richard_patrick_diff_l3082_308276


namespace NUMINAMATH_CALUDE_intersection_range_is_correct_l3082_308223

/-- Line l with parameter t -/
structure Line where
  a : ℝ
  x : ℝ → ℝ
  y : ℝ → ℝ
  h1 : ∀ t, x t = a - 2 * t * (y t)
  h2 : ∀ t, y t = -4 * t

/-- Circle C with parameter θ -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h1 : ∀ θ, x θ = 4 * Real.cos θ
  h2 : ∀ θ, y θ = 4 * Real.sin θ

/-- The range of a for which line l intersects circle C -/
def intersectionRange (l : Line) (c : Circle) : Set ℝ :=
  { a | ∃ t θ, l.x t = c.x θ ∧ l.y t = c.y θ }

theorem intersection_range_is_correct (l : Line) (c : Circle) :
  intersectionRange l c = Set.Icc (-4 * Real.sqrt 5) (4 * Real.sqrt 5) := by
  sorry

#check intersection_range_is_correct

end NUMINAMATH_CALUDE_intersection_range_is_correct_l3082_308223


namespace NUMINAMATH_CALUDE_board_cutting_theorem_l3082_308260

def is_valid_board_size (n : ℕ) : Prop :=
  ∃ m : ℕ, n * n = 5 * m ∧ n > 5

theorem board_cutting_theorem (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n * n = m + 4 * m) ↔ is_valid_board_size n :=
sorry

end NUMINAMATH_CALUDE_board_cutting_theorem_l3082_308260


namespace NUMINAMATH_CALUDE_sum_of_segments_is_224_l3082_308286

/-- Given seven points A, B, C, D, E, F, G on a line in that order, 
    this function calculates the sum of lengths of all segments with endpoints at these points. -/
def sumOfSegments (AG BF CE : ℝ) : ℝ :=
  6 * AG + 4 * BF + 2 * CE

/-- Theorem stating that for the given conditions, the sum of all segment lengths is 224 cm. -/
theorem sum_of_segments_is_224 (AG BF CE : ℝ) 
  (h1 : AG = 23) (h2 : BF = 17) (h3 : CE = 9) : 
  sumOfSegments AG BF CE = 224 := by
  sorry

#eval sumOfSegments 23 17 9

end NUMINAMATH_CALUDE_sum_of_segments_is_224_l3082_308286


namespace NUMINAMATH_CALUDE_perfect_square_minus_seven_l3082_308224

theorem perfect_square_minus_seven (k : ℕ+) : 
  ∃ (n m : ℕ+), n * 2^k.val - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_minus_seven_l3082_308224


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3082_308246

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + m*x + 16 = (a*x + b)^2) → (m = 8 ∨ m = -8) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3082_308246


namespace NUMINAMATH_CALUDE_sum_256_125_base5_l3082_308272

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_256_125_base5 :
  addBase5 (toBase5 256) (toBase5 125) = [3, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_256_125_base5_l3082_308272


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3082_308287

theorem smallest_n_congruence (n : ℕ) : ∃ (m : ℕ), m > 0 ∧ (∀ k : ℕ, 0 < k → k < m → (7^k : ℤ) % 5 ≠ (k^7 : ℤ) % 5) ∧ (7^m : ℤ) % 5 = (m^7 : ℤ) % 5 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3082_308287


namespace NUMINAMATH_CALUDE_hot_air_balloon_problem_l3082_308235

theorem hot_air_balloon_problem (initial_balloons : ℕ) 
  (h1 : initial_balloons = 200)
  (h2 : initial_balloons > 0) : 
  let first_blown_up := initial_balloons / 5
  let second_blown_up := 2 * first_blown_up
  let total_blown_up := first_blown_up + second_blown_up
  initial_balloons - total_blown_up = 80 := by
sorry

end NUMINAMATH_CALUDE_hot_air_balloon_problem_l3082_308235


namespace NUMINAMATH_CALUDE_inequality_proof_l3082_308236

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  ((x*y + y*z + z*x) / 3)^3 ≤ (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ∧
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ ((x^2 + y^2 + z^2) / 2)^3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3082_308236


namespace NUMINAMATH_CALUDE_car_owners_without_others_l3082_308298

/-- Represents the number of adults owning each type of vehicle and their intersections -/
structure VehicleOwnership where
  total : ℕ
  cars : ℕ
  motorcycles : ℕ
  bicycles : ℕ
  cars_motorcycles : ℕ
  cars_bicycles : ℕ
  motorcycles_bicycles : ℕ
  all_three : ℕ

/-- The main theorem stating the number of car owners without motorcycles or bicycles -/
theorem car_owners_without_others (v : VehicleOwnership) 
  (h_total : v.total = 500)
  (h_cars : v.cars = 450)
  (h_motorcycles : v.motorcycles = 150)
  (h_bicycles : v.bicycles = 200)
  (h_pie : v.total = v.cars + v.motorcycles + v.bicycles - v.cars_motorcycles - v.cars_bicycles - v.motorcycles_bicycles + v.all_three)
  : v.cars - (v.cars_motorcycles + v.cars_bicycles - v.all_three) = 270 := by
  sorry

/-- A lemma to ensure all adults own at least one vehicle -/
lemma all_adults_own_vehicle (v : VehicleOwnership) 
  (h_total : v.total = 500)
  (h_pie : v.total = v.cars + v.motorcycles + v.bicycles - v.cars_motorcycles - v.cars_bicycles - v.motorcycles_bicycles + v.all_three)
  : v.cars + v.motorcycles + v.bicycles ≥ v.total := by
  sorry

end NUMINAMATH_CALUDE_car_owners_without_others_l3082_308298


namespace NUMINAMATH_CALUDE_kids_played_monday_l3082_308248

theorem kids_played_monday (total : ℕ) (tuesday : ℕ) (h1 : total = 16) (h2 : tuesday = 14) :
  total - tuesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kids_played_monday_l3082_308248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3082_308201

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b 2 - b 1)
  (h_sum_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_sum_T : ∀ n, T n = n * (b 1 + b n) / 2)
  (h_ratio : ∀ n, S n / T n = (n + 3) / (2 * n + 1)) :
  a 6 / b 6 = 14 / 23 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3082_308201


namespace NUMINAMATH_CALUDE_at_least_two_equal_l3082_308290

theorem at_least_two_equal (x y z : ℝ) (h : x/y + y/z + z/x = z/y + y/x + x/z) :
  (x = y) ∨ (y = z) ∨ (z = x) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l3082_308290


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3082_308226

theorem cubic_equation_root (a b : ℚ) : 
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 45 = 0 ∧ x = -2 - 5*Real.sqrt 3) →
  a = 239/71 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3082_308226


namespace NUMINAMATH_CALUDE_youngest_child_age_l3082_308238

def mother_charge : ℝ := 5.05
def child_charge_per_year : ℝ := 0.55
def total_bill : ℝ := 11.05

def is_valid_age_combination (twin_age : ℕ) (youngest_age : ℕ) : Prop :=
  twin_age > youngest_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + youngest_age) = total_bill

theorem youngest_child_age :
  ∀ youngest_age : ℕ,
    (∃ twin_age : ℕ, is_valid_age_combination twin_age youngest_age) ↔
    (youngest_age = 1 ∨ youngest_age = 3) :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3082_308238


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3082_308267

theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + m - 1 = 0 ∧
    ∀ y : ℝ, y^2 - 4*y + m - 1 = 0 → y = x) →
  m = 5 ∧ ∃ x : ℝ, x^2 - 4*x + m - 1 = 0 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3082_308267


namespace NUMINAMATH_CALUDE_schools_count_proof_l3082_308293

def number_of_schools : ℕ := 24

theorem schools_count_proof :
  ∀ (total_students : ℕ) (andrew_rank : ℕ),
    total_students = 4 * number_of_schools →
    andrew_rank = (total_students + 1) / 2 →
    andrew_rank < 50 →
    andrew_rank > 48 →
    number_of_schools = 24 := by
  sorry

end NUMINAMATH_CALUDE_schools_count_proof_l3082_308293


namespace NUMINAMATH_CALUDE_difficult_math_problems_not_set_l3082_308256

-- Define the criteria for set elements
structure SetCriteria where
  definiteness : Bool
  distinctness : Bool
  unorderedness : Bool

-- Define a function to check if something can form a set
def canFormSet (criteria : SetCriteria) : Bool :=
  criteria.definiteness ∧ criteria.distinctness ∧ criteria.unorderedness

-- Define the characteristics of "All difficult math problems"
def difficultMathProblems : SetCriteria :=
  { definiteness := false,  -- Not definite
    distinctness := true,   -- Assumed distinct
    unorderedness := true } -- Assumed unordered

-- Theorem to prove
theorem difficult_math_problems_not_set : ¬(canFormSet difficultMathProblems) := by
  sorry

end NUMINAMATH_CALUDE_difficult_math_problems_not_set_l3082_308256


namespace NUMINAMATH_CALUDE_max_non_managers_l3082_308288

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 5 / 24 →
  non_managers ≤ 38 :=
by
  sorry

end NUMINAMATH_CALUDE_max_non_managers_l3082_308288


namespace NUMINAMATH_CALUDE_floor_area_less_than_10_l3082_308280

/-- Represents a rectangular room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The condition that each wall requires more paint than the floor -/
def more_paint_on_walls (r : Room) : Prop :=
  r.length * r.height > r.length * r.width ∧
  r.width * r.height > r.length * r.width

/-- The floor area of the room -/
def floor_area (r : Room) : ℝ :=
  r.length * r.width

/-- Theorem stating that for a room with height 3 meters and more paint required for walls than floor,
    the floor area must be less than 10 square meters -/
theorem floor_area_less_than_10 (r : Room) 
  (h1 : r.height = 3)
  (h2 : more_paint_on_walls r) : 
  floor_area r < 10 := by
  sorry


end NUMINAMATH_CALUDE_floor_area_less_than_10_l3082_308280


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_is_135_l3082_308274

/-- The exterior angle formed by a square and a regular octagon sharing a common side -/
def exterior_angle_square_octagon : ℝ := 135

/-- Theorem: The exterior angle formed by a square and a regular octagon sharing a common side is 135 degrees -/
theorem exterior_angle_square_octagon_is_135 :
  exterior_angle_square_octagon = 135 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_is_135_l3082_308274


namespace NUMINAMATH_CALUDE_blue_candy_count_l3082_308213

theorem blue_candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409)
  (h2 : red = 145)
  (h3 : blue = total - red) : blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l3082_308213


namespace NUMINAMATH_CALUDE_star_equation_solution_l3082_308284

/-- Custom binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem stating that if 4 ⋆ x = 52, then x = 8 -/
theorem star_equation_solution (x : ℝ) (h : star 4 x = 52) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3082_308284


namespace NUMINAMATH_CALUDE_plane_division_l3082_308297

/-- The maximum number of parts that n planes can divide 3D space into --/
def max_parts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2^(n+1)

theorem plane_division :
  (max_parts 1 = 2) ∧
  (max_parts 2 ≤ 4) ∧
  (max_parts 3 ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l3082_308297


namespace NUMINAMATH_CALUDE_vector_parallel_value_l3082_308215

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_parallel_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (6, 8)
  parallel a b → x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_value_l3082_308215


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3082_308228

/-- Given functions f, g, h: ℝ → ℝ satisfying the functional equation
    f(x) - g(y) = (x-y) · h(x+y) for all x, y ∈ ℝ,
    prove that there exist constants d, c ∈ ℝ such that
    f(x) = g(x) = dx² + c for all x ∈ ℝ. -/
theorem functional_equation_solution
  (f g h : ℝ → ℝ)
  (h_eq : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, ∀ x : ℝ, f x = d * x^2 + c ∧ g x = d * x^2 + c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3082_308228


namespace NUMINAMATH_CALUDE_value_of_expression_l3082_308285

theorem value_of_expression (x : ℝ) (h : x = 5) : (3*x + 4)^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3082_308285


namespace NUMINAMATH_CALUDE_tangent_point_divides_equally_l3082_308253

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateralWithInscribedCircle where
  -- Sides of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Ensure all sides are positive
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  -- The quadrilateral is cyclic (inscribed in a circle)
  is_cyclic : True
  -- The quadrilateral has an inscribed circle
  has_inscribed_circle : True

/-- Theorem: In a cyclic quadrilateral with an inscribed circle, 
    if the consecutive sides have lengths 80, 120, 100, and 140, 
    then the point of tangency of the inscribed circle on the side 
    of length 100 divides it into two equal segments. -/
theorem tangent_point_divides_equally 
  (Q : CyclicQuadrilateralWithInscribedCircle) 
  (h1 : Q.a = 80) 
  (h2 : Q.b = 120) 
  (h3 : Q.c = 100) 
  (h4 : Q.d = 140) : 
  ∃ (x y : ℝ), x + y = 100 ∧ x = y :=
sorry

end NUMINAMATH_CALUDE_tangent_point_divides_equally_l3082_308253
