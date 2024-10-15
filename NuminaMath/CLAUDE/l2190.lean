import Mathlib

namespace NUMINAMATH_CALUDE_min_beans_betty_buys_l2190_219001

/-- The minimum number of pounds of beans Betty could buy given the conditions on rice and beans -/
theorem min_beans_betty_buys (r b : ℝ) 
  (h1 : r ≥ 4 + 2 * b) 
  (h2 : r ≤ 3 * b) : 
  b ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_beans_betty_buys_l2190_219001


namespace NUMINAMATH_CALUDE_ratio_a_to_d_l2190_219061

theorem ratio_a_to_d (a b c d : ℚ) : 
  a / b = 8 / 3 →
  b / c = 1 / 5 →
  c / d = 3 / 2 →
  b = 27 →
  a / d = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_d_l2190_219061


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l2190_219035

theorem gdp_scientific_notation :
  let gdp_billion : ℝ := 32.07
  let billion : ℝ := 10^9
  let gdp : ℝ := gdp_billion * billion
  ∃ (a : ℝ) (n : ℤ), gdp = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.207 ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l2190_219035


namespace NUMINAMATH_CALUDE_g_of_f_minus_two_three_l2190_219063

/-- Transformation f that takes a pair of integers and negates the second component -/
def f (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

/-- Transformation g that takes a pair of integers and negates both components -/
def g (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

/-- Theorem stating that g[f(-2,3)] = (2,3) -/
theorem g_of_f_minus_two_three : g (f (-2, 3)) = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_g_of_f_minus_two_three_l2190_219063


namespace NUMINAMATH_CALUDE_complex_sum_on_real_axis_l2190_219067

theorem complex_sum_on_real_axis (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := 3 + a * I
  (z₁ + z₂).im = 0 → a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_on_real_axis_l2190_219067


namespace NUMINAMATH_CALUDE_cookie_chips_count_l2190_219098

/-- Calculates the number of chocolate chips per cookie given the total chips,
    number of batches, and cookies per batch. -/
def chips_per_cookie (total_chips : ℕ) (num_batches : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  total_chips / (num_batches * cookies_per_batch)

/-- Proves that there are 9 chocolate chips per cookie given the problem conditions. -/
theorem cookie_chips_count :
  let total_chips : ℕ := 81
  let num_batches : ℕ := 3
  let cookies_per_batch : ℕ := 3
  chips_per_cookie total_chips num_batches cookies_per_batch = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_chips_count_l2190_219098


namespace NUMINAMATH_CALUDE_inequality_pattern_l2190_219010

theorem inequality_pattern (x : ℝ) (a : ℝ) 
  (h_x : x > 0)
  (h1 : x + 1/x ≥ 2)
  (h2 : x + 4/x^2 ≥ 3)
  (h3 : x + 27/x^3 ≥ 4)
  (h4 : x + a/x^4 ≥ 5) :
  a = 4^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_pattern_l2190_219010


namespace NUMINAMATH_CALUDE_intersection_points_count_l2190_219014

/-- The number of points with positive x-coordinates that lie on at least two of the graphs
    y = log₂x, y = 1/log₂x, y = -log₂x, and y = -1/log₂x -/
theorem intersection_points_count : ℕ := by
  sorry

#check intersection_points_count

end NUMINAMATH_CALUDE_intersection_points_count_l2190_219014


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l2190_219005

/-- The standard equation of a circle with diameter endpoints M(2,0) and N(0,4) -/
theorem circle_equation_from_diameter (x y : ℝ) : 
  let M : ℝ × ℝ := (2, 0)
  let N : ℝ × ℝ := (0, 4)
  (x - 1)^2 + (y - 2)^2 = 5 ↔ 
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      center = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
      radius^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ∧
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l2190_219005


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2190_219044

/-- The parabola equation: x = 3y^2 + 5y - 4 -/
def parabola (x y : ℝ) : Prop := x = 3 * y^2 + 5 * y - 4

/-- The line equation: x = k -/
def line (x k : ℝ) : Prop := x = k

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! y, parabola k y ∧ line k k

theorem parabola_line_intersection :
  ∀ k : ℝ, single_intersection k ↔ k = -23/12 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2190_219044


namespace NUMINAMATH_CALUDE_vector_parallel_cosine_value_l2190_219012

theorem vector_parallel_cosine_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : (9/10, 3) = (Real.cos (θ + π/6), 2)) : 
  Real.cos θ = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_cosine_value_l2190_219012


namespace NUMINAMATH_CALUDE_div_three_sevenths_by_four_l2190_219008

theorem div_three_sevenths_by_four :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_div_three_sevenths_by_four_l2190_219008


namespace NUMINAMATH_CALUDE_middle_number_between_52_and_certain_number_l2190_219081

theorem middle_number_between_52_and_certain_number 
  (certain_number : ℕ) 
  (h1 : certain_number > 52) 
  (h2 : ∃ (n : ℕ), n ≥ 52 ∧ n < certain_number ∧ certain_number - 52 - 1 = 15) :
  (52 + certain_number) / 2 = 60 :=
sorry

end NUMINAMATH_CALUDE_middle_number_between_52_and_certain_number_l2190_219081


namespace NUMINAMATH_CALUDE_trajectory_is_circle_l2190_219075

/-- The ellipse with equation x²/7 + y²/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 7 + p.2^2 / 3 = 1}

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (2, 0)

/-- The set of all points Q obtained by extending F₁P to Q such that |PQ| = |PF₂| for all P on the ellipse -/
def TrajectoryQ (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | P ∈ Ellipse ∧ ∃ t : ℝ, t > 1 ∧ Q = (t • (P - F₁) + F₁) ∧ 
    ‖Q - P‖ = ‖P - F₂‖}

/-- The theorem stating that the trajectory of Q is a circle -/
theorem trajectory_is_circle : 
  ∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, Q ∈ TrajectoryQ P) ↔ (Q.1 + 2)^2 + Q.2^2 = 28 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_circle_l2190_219075


namespace NUMINAMATH_CALUDE_unique_solution_system_l2190_219011

theorem unique_solution_system (x y z : ℂ) :
  x + y + z = 3 ∧
  x^2 + y^2 + z^2 = 3 ∧
  x^3 + y^3 + z^3 = 3 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2190_219011


namespace NUMINAMATH_CALUDE_count_negative_rationals_l2190_219003

def rational_set : Finset ℚ := {-1/2, 5, 0, -(-3), -2, -|-25|}

theorem count_negative_rationals : 
  (rational_set.filter (λ x => x < 0)).card = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_rationals_l2190_219003


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l2190_219064

def M : Set Int := {-1, 0, 1}

def N : Set Int := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l2190_219064


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l2190_219036

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define the common logarithm (base 10) function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_one :
  2 * lg (Real.sqrt 2) + log2 5 * lg 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l2190_219036


namespace NUMINAMATH_CALUDE_city_population_multiple_l2190_219094

/- Define the populations of the cities and the multiple -/
def willowdale_population : ℕ := 2000
def sun_city_population : ℕ := 12000

/- Define the relationship between the cities' populations -/
def roseville_population (m : ℕ) : ℤ := m * willowdale_population - 500
def sun_city_relation (m : ℕ) : Prop := 
  sun_city_population = 2 * (roseville_population m) + 1000

/- State the theorem -/
theorem city_population_multiple : ∃ m : ℕ, sun_city_relation m ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_city_population_multiple_l2190_219094


namespace NUMINAMATH_CALUDE_fifty_numbers_with_negative_products_l2190_219006

theorem fifty_numbers_with_negative_products (total : Nat) (neg_products : Nat) 
  (h1 : total = 50) (h2 : neg_products = 500) : 
  ∃ (m n p : Nat), m + n + p = total ∧ m * p = neg_products ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifty_numbers_with_negative_products_l2190_219006


namespace NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l2190_219015

theorem divisible_by_2000_arrangement (nums : Vector ℕ 23) :
  ∃ (arrangement : List (Sum (Prod ℕ ℕ) ℕ)),
    (arrangement.foldl (λ acc x => match x with
      | Sum.inl (a, b) => acc * (a * b)
      | Sum.inr a => acc + a
    ) 0) % 2000 = 0 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l2190_219015


namespace NUMINAMATH_CALUDE_solutions_difference_squared_l2190_219043

theorem solutions_difference_squared (α β : ℝ) : 
  α ≠ β ∧ 
  α^2 - 3*α + 1 = 0 ∧ 
  β^2 - 3*β + 1 = 0 → 
  (α - β)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_solutions_difference_squared_l2190_219043


namespace NUMINAMATH_CALUDE_noProblemProbabilityIs377Over729_l2190_219050

/-- Recursive function to calculate the number of valid arrangements for n people --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n+2) => 3 * validArrangements (n+1) - validArrangements n

/-- The number of chairs and people --/
def numChairs : ℕ := 6

/-- The total number of possible arrangements --/
def totalArrangements : ℕ := 3^numChairs

/-- The probability of no problematic seating arrangement --/
def noProblemProbability : ℚ := validArrangements numChairs / totalArrangements

theorem noProblemProbabilityIs377Over729 : 
  noProblemProbability = 377 / 729 := by sorry

end NUMINAMATH_CALUDE_noProblemProbabilityIs377Over729_l2190_219050


namespace NUMINAMATH_CALUDE_classroom_ratio_problem_l2190_219076

theorem classroom_ratio_problem (total_students : ℕ) (girl_ratio boy_ratio : ℕ) 
  (h1 : total_students = 30)
  (h2 : girl_ratio = 1)
  (h3 : boy_ratio = 2) : 
  (total_students * boy_ratio) / (girl_ratio + boy_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_problem_l2190_219076


namespace NUMINAMATH_CALUDE_bird_nests_calculation_l2190_219092

/-- Calculates the total number of nests required for birds in a park --/
theorem bird_nests_calculation (total_birds : Nat) 
  (sparrows pigeons starlings : Nat)
  (sparrow_nests pigeon_nests starling_nests : Nat)
  (h1 : total_birds = sparrows + pigeons + starlings)
  (h2 : total_birds = 10)
  (h3 : sparrows = 4)
  (h4 : pigeons = 3)
  (h5 : starlings = 3)
  (h6 : sparrow_nests = 1)
  (h7 : pigeon_nests = 2)
  (h8 : starling_nests = 3) :
  sparrows * sparrow_nests + pigeons * pigeon_nests + starlings * starling_nests = 19 := by
  sorry

end NUMINAMATH_CALUDE_bird_nests_calculation_l2190_219092


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_675_49_l2190_219046

/-- Two circles in a 2D plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  radius1 : ℝ
  center2 : ℝ × ℝ
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionDistanceSquared (c : TwoCircles) : ℝ := sorry

/-- The specific configuration of circles from the problem -/
def problemCircles : TwoCircles :=
  { center1 := (3, -1)
  , radius1 := 5
  , center2 := (3, 6)
  , radius2 := 3 }

/-- Theorem stating that the square of the distance between intersection points
    of the given circles is 675/49 -/
theorem intersection_distance_squared_is_675_49 :
  intersectionDistanceSquared problemCircles = 675 / 49 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_is_675_49_l2190_219046


namespace NUMINAMATH_CALUDE_folded_hexagon_result_verify_interior_angle_sum_l2190_219048

/-- Represents the possible polygons resulting from folding a regular hexagon in half -/
inductive FoldedHexagonShape
  | Quadrilateral
  | Pentagon

/-- Calculates the sum of interior angles for a polygon with n sides -/
def sumOfInteriorAngles (n : ℕ) : ℕ := (n - 2) * 180

/-- Represents the result of folding a regular hexagon in half -/
structure FoldedHexagonResult where
  shape : FoldedHexagonShape
  interiorAngleSum : ℕ

/-- Theorem stating the possible results of folding a regular hexagon in half -/
theorem folded_hexagon_result :
  ∃ (result : FoldedHexagonResult),
    (result.shape = FoldedHexagonShape.Quadrilateral ∧ result.interiorAngleSum = 360) ∨
    (result.shape = FoldedHexagonShape.Pentagon ∧ result.interiorAngleSum = 540) :=
by
  sorry

/-- Verification that the sum of interior angles is correct for each shape -/
theorem verify_interior_angle_sum :
  ∀ (result : FoldedHexagonResult),
    (result.shape = FoldedHexagonShape.Quadrilateral → result.interiorAngleSum = sumOfInteriorAngles 4) ∧
    (result.shape = FoldedHexagonShape.Pentagon → result.interiorAngleSum = sumOfInteriorAngles 5) :=
by
  sorry

end NUMINAMATH_CALUDE_folded_hexagon_result_verify_interior_angle_sum_l2190_219048


namespace NUMINAMATH_CALUDE_triangle_max_area_l2190_219055

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area is √3 when (a+b)(sin A - sin B) = (c-b)sin C and a = 2 -/
theorem triangle_max_area (a b c A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) →
  (a = 2) →
  (∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2190_219055


namespace NUMINAMATH_CALUDE_unique_solution_for_2n_plus_m_l2190_219045

theorem unique_solution_for_2n_plus_m : 
  ∀ n m : ℤ, 
    (3 * n - m < 5) → 
    (n + m > 26) → 
    (3 * m - 2 * n < 46) → 
    (2 * n + m = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_2n_plus_m_l2190_219045


namespace NUMINAMATH_CALUDE_luke_played_two_rounds_l2190_219086

/-- The number of rounds Luke played in a trivia game -/
def rounds_played (total_points : ℕ) (points_per_round : ℕ) : ℕ :=
  total_points / points_per_round

/-- Theorem stating that Luke played 2 rounds -/
theorem luke_played_two_rounds :
  rounds_played 84 42 = 2 := by
  sorry

end NUMINAMATH_CALUDE_luke_played_two_rounds_l2190_219086


namespace NUMINAMATH_CALUDE_quadratic_form_value_l2190_219082

theorem quadratic_form_value (x y : ℝ) 
  (eq1 : 4 * x + y = 12) 
  (eq2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_value_l2190_219082


namespace NUMINAMATH_CALUDE_coefficient_sum_l2190_219051

theorem coefficient_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l2190_219051


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l2190_219073

theorem ratio_a_to_b (a b c d : ℚ) 
  (h1 : b / c = 7 / 9)
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) :
  a / b = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l2190_219073


namespace NUMINAMATH_CALUDE_number_of_cows_l2190_219072

/-- Represents the number of legs for each animal type -/
def legs_per_animal : (Fin 2) → ℕ
| 0 => 2  -- chickens
| 1 => 4  -- cows

/-- Represents the total number of animals -/
def total_animals : ℕ := 160

/-- Represents the total number of legs -/
def total_legs : ℕ := 400

/-- Proves that the number of cows is 40 given the conditions -/
theorem number_of_cows : 
  ∃ (chickens cows : ℕ), 
    chickens + cows = total_animals ∧ 
    chickens * legs_per_animal 0 + cows * legs_per_animal 1 = total_legs ∧
    cows = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cows_l2190_219072


namespace NUMINAMATH_CALUDE_count_elements_with_leftmost_seven_l2190_219041

/-- The set of powers of 5 up to 5000 -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 5000 ∧ n = 5^k}

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a natural number -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The count of numbers in S with 7 as the leftmost digit -/
def count_leftmost_seven (S : Set ℕ) : ℕ := sorry

theorem count_elements_with_leftmost_seven :
  num_digits (5^5000) = 3501 →
  leftmost_digit (5^5000) = 7 →
  count_leftmost_seven S = 1501 := by sorry

end NUMINAMATH_CALUDE_count_elements_with_leftmost_seven_l2190_219041


namespace NUMINAMATH_CALUDE_part_one_part_two_l2190_219088

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Part I
theorem part_one : lg 24 - lg 3 - lg 4 + lg 5 = 1 := by sorry

-- Part II
theorem part_two : (((3 : ℝ) ^ (1/3) * (2 : ℝ) ^ (1/2)) ^ 6) + 
                   (((3 : ℝ) * (3 : ℝ) ^ (1/2)) ^ (1/2)) ^ (4/3) - 
                   ((2 : ℝ) ^ (1/4)) * (8 : ℝ) ^ (1/4) - 
                   (2015 : ℝ) ^ 0 = 72 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2190_219088


namespace NUMINAMATH_CALUDE_statement_is_proposition_l2190_219058

-- Define what a proposition is
def is_proposition (statement : String) : Prop :=
  ∃ (truth_value : Bool), (statement = "true") ∨ (statement = "false")

-- Define the statement we want to prove is a proposition
def statement : String := "20-5×3=10"

-- Theorem to prove
theorem statement_is_proposition : is_proposition statement := by
  sorry

end NUMINAMATH_CALUDE_statement_is_proposition_l2190_219058


namespace NUMINAMATH_CALUDE_clothing_distribution_l2190_219057

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l2190_219057


namespace NUMINAMATH_CALUDE_alex_total_fish_is_4000_l2190_219059

/-- The number of fish Brian catches per trip -/
def brian_fish_per_trip : ℕ := 400

/-- The number of times Chris goes fishing -/
def chris_fishing_trips : ℕ := 10

/-- The number of times Alex goes fishing -/
def alex_fishing_trips : ℕ := chris_fishing_trips / 2

/-- The number of fish Alex catches per trip -/
def alex_fish_per_trip : ℕ := brian_fish_per_trip * 2

/-- The total number of fish Alex caught -/
def alex_total_fish : ℕ := alex_fishing_trips * alex_fish_per_trip

theorem alex_total_fish_is_4000 : alex_total_fish = 4000 := by
  sorry

end NUMINAMATH_CALUDE_alex_total_fish_is_4000_l2190_219059


namespace NUMINAMATH_CALUDE_min_posts_for_fence_l2190_219060

def fence_length : ℝ := 40 + 40 + 100
def post_spacing : ℝ := 10
def area_width : ℝ := 40
def area_length : ℝ := 100

theorem min_posts_for_fence : 
  ⌊fence_length / post_spacing⌋ + 1 = 19 := by sorry

end NUMINAMATH_CALUDE_min_posts_for_fence_l2190_219060


namespace NUMINAMATH_CALUDE_football_hits_ground_time_l2190_219040

def football_height (t : ℝ) : ℝ := -16 * t^2 + 18 * t + 60

theorem football_hits_ground_time :
  ∃ t : ℝ, t > 0 ∧ football_height t = 0 ∧ t = 41 / 16 := by
  sorry

end NUMINAMATH_CALUDE_football_hits_ground_time_l2190_219040


namespace NUMINAMATH_CALUDE_expression_simplification_l2190_219070

theorem expression_simplification :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2190_219070


namespace NUMINAMATH_CALUDE_expression_value_l2190_219097

/-- Given that px³ + qx + 3 = 2005 when x = 3, prove that px³ + qx + 3 = -1999 when x = -3 -/
theorem expression_value (p q : ℝ) : 
  (27 * p + 3 * q + 3 = 2005) → (-27 * p - 3 * q + 3 = -1999) := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2190_219097


namespace NUMINAMATH_CALUDE_domain_of_f_l2190_219099

def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

theorem domain_of_f : Set.univ = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_l2190_219099


namespace NUMINAMATH_CALUDE_simplify_fraction_l2190_219069

theorem simplify_fraction (b c : ℚ) (hb : b = 2) (hc : c = 3) :
  15 * b^4 * c^2 / (45 * b^3 * c) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2190_219069


namespace NUMINAMATH_CALUDE_three_numbers_problem_l2190_219049

theorem three_numbers_problem (a b c : ℝ) 
  (sum_eq : a + b + c = 15)
  (sum_minus_third : a + b - c = 10)
  (sum_minus_second : a - b + c = 8) :
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l2190_219049


namespace NUMINAMATH_CALUDE_distance_difference_l2190_219079

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
sorry

end NUMINAMATH_CALUDE_distance_difference_l2190_219079


namespace NUMINAMATH_CALUDE_weight_problem_l2190_219016

/-- Given three weights A, B, and C, prove that their average weights satisfy certain conditions -/
theorem weight_problem (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)  -- The average weight of A, B, and C is 45 kg
  (h2 : (B + C) / 2 = 43)      -- The average weight of B and C is 43 kg
  (h3 : B = 31)                -- The weight of B is 31 kg
  : (A + B) / 2 = 40 :=        -- The average weight of A and B is 40 kg
by sorry

end NUMINAMATH_CALUDE_weight_problem_l2190_219016


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2012_l2190_219095

/-- Given an angle α = 2012°, this theorem proves that the smallest positive angle 
    with the same terminal side as α is 212°. -/
theorem smallest_positive_angle_2012 (α : Real) (h : α = 2012) :
  ∃ (θ : Real), 0 < θ ∧ θ ≤ 360 ∧ θ = α % 360 ∧ θ = 212 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2012_l2190_219095


namespace NUMINAMATH_CALUDE_sqrt_xy_plus_3_l2190_219009

theorem sqrt_xy_plus_3 (x y : ℝ) (h : y = Real.sqrt (1 - 4*x) + Real.sqrt (4*x - 1) + 4) :
  Real.sqrt (x*y + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_xy_plus_3_l2190_219009


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l2190_219025

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≥ 0 then Real.sqrt y else -Real.sqrt (-y)

-- Theorem statement
theorem inverse_sum_equals_negative_six :
  f_inv 9 + f_inv (-81) = -6 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l2190_219025


namespace NUMINAMATH_CALUDE_b_21_equals_861_l2190_219054

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def b (n : ℕ) : ℕ := a (2 * n - 1)

theorem b_21_equals_861 : b 21 = 861 := by sorry

end NUMINAMATH_CALUDE_b_21_equals_861_l2190_219054


namespace NUMINAMATH_CALUDE_sum_of_max_min_on_interval_l2190_219026

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem sum_of_max_min_on_interval :
  let a : ℝ := 0
  let b : ℝ := 3
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max + min = -10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_on_interval_l2190_219026


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_m_and_n_l2190_219018

theorem sqrt_equality_implies_m_and_n (m n : ℝ) :
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt m - Real.sqrt n →
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_m_and_n_l2190_219018


namespace NUMINAMATH_CALUDE_eventually_all_zero_l2190_219020

/-- Represents a quadruple of integers -/
structure Quadruple where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Generates the next quadruple in the sequence -/
def nextQuadruple (q : Quadruple) : Quadruple := {
  a := |q.a - q.b|
  b := |q.b - q.c|
  c := |q.c - q.d|
  d := |q.d - q.a|
}

/-- Checks if all elements in a quadruple are zero -/
def isAllZero (q : Quadruple) : Prop :=
  q.a = 0 ∧ q.b = 0 ∧ q.c = 0 ∧ q.d = 0

/-- Theorem: The sequence will eventually reach all zeros -/
theorem eventually_all_zero (q₀ : Quadruple) : 
  ∃ n : ℕ, isAllZero ((nextQuadruple^[n]) q₀) :=
sorry


end NUMINAMATH_CALUDE_eventually_all_zero_l2190_219020


namespace NUMINAMATH_CALUDE_no_subset_with_unique_finite_sum_representation_l2190_219085

-- Define the set S as rational numbers in (0,1)
def S : Set ℚ := {q : ℚ | 0 < q ∧ q < 1}

-- Define the property for subset T
def has_unique_finite_sum_representation (T : Set ℚ) : Prop :=
  ∀ s ∈ S, ∃! (finite_sum : List ℚ),
    (∀ t ∈ finite_sum, t ∈ T) ∧
    (∀ t ∈ finite_sum, ∀ u ∈ finite_sum, t ≠ u → t ≠ u) ∧
    (s = finite_sum.sum)

-- Theorem statement
theorem no_subset_with_unique_finite_sum_representation :
  ¬ ∃ (T : Set ℚ), T ⊆ S ∧ has_unique_finite_sum_representation T := by
  sorry

end NUMINAMATH_CALUDE_no_subset_with_unique_finite_sum_representation_l2190_219085


namespace NUMINAMATH_CALUDE_average_weight_increase_l2190_219078

/-- Proves that replacing a 70 kg person with a 110 kg person in a group of 10 increases the average weight by 4 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 10 * initial_average
  let new_total := initial_total - 70 + 110
  let new_average := new_total / 10
  new_average - initial_average = 4 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2190_219078


namespace NUMINAMATH_CALUDE_min_distance_to_point_l2190_219056

/-- The line equation ax + by + 1 = 0 -/
def line_equation (a b x y : ℝ) : Prop := a * x + b * y + 1 = 0

/-- The circle equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line always bisects the circumference of the circle -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a b x y → circle_equation x y

/-- The theorem to be proved -/
theorem min_distance_to_point (a b : ℝ) 
  (h : line_bisects_circle a b) : 
  (∀ a' b' : ℝ, line_bisects_circle a' b' → (a-2)^2 + (b-2)^2 ≤ (a'-2)^2 + (b'-2)^2) ∧
  (a-2)^2 + (b-2)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l2190_219056


namespace NUMINAMATH_CALUDE_function_value_at_cos_15_deg_l2190_219017

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 1

theorem function_value_at_cos_15_deg :
  f (Real.cos (15 * π / 180)) = -(Real.sqrt 3 / 2) - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_value_at_cos_15_deg_l2190_219017


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2190_219000

/-- Given two parallel vectors a and b in R², prove that 3a + 2b equals (-1, -2) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  (a.1 * b.2 = a.2 * b.1) →  -- Parallel condition
  (3 * a.1 + 2 * b.1 = -1 ∧ 3 * a.2 + 2 * b.2 = -2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2190_219000


namespace NUMINAMATH_CALUDE_cube_ratio_equals_64_l2190_219024

theorem cube_ratio_equals_64 : (88888 / 22222)^3 = 64 := by
  have h : 88888 / 22222 = 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_cube_ratio_equals_64_l2190_219024


namespace NUMINAMATH_CALUDE_regular_polygon_140_degree_interior_l2190_219089

/-- A regular polygon with interior angles measuring 140° has 9 sides. -/
theorem regular_polygon_140_degree_interior : ∀ n : ℕ, 
  n > 2 → -- ensure it's a valid polygon
  (180 * (n - 2) : ℝ) = (140 * n : ℝ) → -- sum of interior angles formula
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_140_degree_interior_l2190_219089


namespace NUMINAMATH_CALUDE_tensor_properties_l2190_219093

/-- Define a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define the ⊗ operation -/
def tensor (a b : Vector2D) : ℝ :=
  a.x * b.y - b.x * a.y

/-- Define the dot product -/
def dot (a b : Vector2D) : ℝ :=
  a.x * b.x + a.y * b.y

theorem tensor_properties (m n p q : ℝ) :
  let a : Vector2D := ⟨m, n⟩
  let b : Vector2D := ⟨p, q⟩
  (tensor a a = 0) ∧
  ((tensor a b)^2 + (dot a b)^2 = (m^2 + q^2) * (n^2 + p^2)) := by
  sorry

end NUMINAMATH_CALUDE_tensor_properties_l2190_219093


namespace NUMINAMATH_CALUDE_sector_area_l2190_219038

/-- Given a circular sector with central angle 2 radians and arc length 4, its area is 4. -/
theorem sector_area (θ : ℝ) (l : ℝ) (r : ℝ) (h1 : θ = 2) (h2 : l = 4) (h3 : l = r * θ) :
  (1 / 2) * r^2 * θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2190_219038


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_negative_one_l2190_219027

theorem sum_of_powers_equals_negative_one :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 + (-1)^2014 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_negative_one_l2190_219027


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2190_219080

theorem sphere_surface_area_ratio (V₁ V₂ A₁ A₂ : ℝ) (h : V₁ / V₂ = 8 / 27) :
  A₁ / A₂ = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2190_219080


namespace NUMINAMATH_CALUDE_product_divisible_by_seven_l2190_219023

theorem product_divisible_by_seven (A B : ℕ+) 
  (hA : Nat.Prime A.val)
  (hB : Nat.Prime B.val)
  (hAminusB : Nat.Prime (A.val - B.val))
  (hAplusB : Nat.Prime (A.val + B.val)) :
  7 ∣ (A.val * B.val * (A.val - B.val) * (A.val + B.val)) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_seven_l2190_219023


namespace NUMINAMATH_CALUDE_valid_numbers_l2190_219084

def is_valid_number (n : ℕ) : Prop :=
  n % 2 = 0 ∧ (Nat.divisors n).card = n / 2

theorem valid_numbers : {n : ℕ | is_valid_number n} = {8, 12} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2190_219084


namespace NUMINAMATH_CALUDE_total_guests_calculation_l2190_219066

/-- Given the number of guests in different age groups, calculate the total number of guests served. -/
theorem total_guests_calculation (adults : ℕ) (h1 : adults = 58) : ∃ (children seniors teenagers toddlers : ℕ),
  children = adults - 35 ∧
  seniors = 2 * children ∧
  teenagers = seniors - 15 ∧
  toddlers = teenagers / 2 ∧
  adults + children + seniors + teenagers + toddlers = 173 := by
  sorry


end NUMINAMATH_CALUDE_total_guests_calculation_l2190_219066


namespace NUMINAMATH_CALUDE_redskins_win_streak_probability_l2190_219030

/-- The probability of arranging wins and losses in exactly three winning streaks -/
theorem redskins_win_streak_probability 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (h1 : total_games = wins + losses)
  (h2 : wins = 10)
  (h3 : losses = 6) :
  (Nat.choose 9 2 * Nat.choose 7 3 : ℚ) / Nat.choose total_games losses = 45 / 286 := by
sorry

end NUMINAMATH_CALUDE_redskins_win_streak_probability_l2190_219030


namespace NUMINAMATH_CALUDE_fraction_comparison_l2190_219002

theorem fraction_comparison (m : ℕ) (h : m = 23^1973) :
  (23^1873 + 1) / (23^1974 + 1) > (23^1974 + 1) / (23^1975 + 1) := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2190_219002


namespace NUMINAMATH_CALUDE_quadratic_equation_results_l2190_219047

theorem quadratic_equation_results (y : ℝ) (h : 6 * y^2 + 7 = 5 * y + 12) : 
  ((12 * y - 5)^2 = 145) ∧ 
  ((5 * y + 2)^2 = (4801 + 490 * Real.sqrt 145 + 3625) / 144 ∨
   (5 * y + 2)^2 = (4801 - 490 * Real.sqrt 145 + 3625) / 144) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_results_l2190_219047


namespace NUMINAMATH_CALUDE_total_pages_calculation_l2190_219071

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 12

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 75

/-- The total number of pages in all booklets -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem total_pages_calculation : total_pages = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_calculation_l2190_219071


namespace NUMINAMATH_CALUDE_arun_weight_average_l2190_219053

def arun_weight_lower_bound : ℝ := 66
def arun_weight_upper_bound : ℝ := 72
def brother_lower_bound : ℝ := 60
def brother_upper_bound : ℝ := 70
def mother_upper_bound : ℝ := 69

theorem arun_weight_average :
  let lower := max arun_weight_lower_bound brother_lower_bound
  let upper := min (min arun_weight_upper_bound brother_upper_bound) mother_upper_bound
  (lower + upper) / 2 = 67.5 := by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l2190_219053


namespace NUMINAMATH_CALUDE_point_on_line_point_40_161_on_line_l2190_219052

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Given three points on a line, check if a fourth point is on the same line -/
theorem point_on_line (p1 p2 p3 p4 : Point)
  (h1 : collinear p1 p2 p3) : 
  collinear p1 p2 p4 ∧ collinear p2 p3 p4 → collinear p1 p3 p4 := by sorry

/-- The main theorem to prove -/
theorem point_40_161_on_line : 
  let p1 : Point := ⟨2, 9⟩
  let p2 : Point := ⟨6, 25⟩
  let p3 : Point := ⟨10, 41⟩
  let p4 : Point := ⟨40, 161⟩
  collinear p1 p2 p3 → collinear p1 p2 p4 ∧ collinear p2 p3 p4 := by sorry

end NUMINAMATH_CALUDE_point_on_line_point_40_161_on_line_l2190_219052


namespace NUMINAMATH_CALUDE_train_speed_l2190_219019

/-- Proves that a train of length 400 meters crossing a pole in 12 seconds has a speed of 120 km/hr -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 400 ∧ time = 12 → speed = (length / 1000) / (time / 3600) → speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2190_219019


namespace NUMINAMATH_CALUDE_max_value_constraint_l2190_219068

theorem max_value_constraint (x y z : ℝ) (h : x^2 + 4*y^2 + 9*z^2 = 3) :
  ∃ (M : ℝ), M = 3 ∧ x + 2*y + 3*z ≤ M ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^2 + 4*y₀^2 + 9*z₀^2 = 3 ∧ x₀ + 2*y₀ + 3*z₀ = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2190_219068


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2190_219007

/-- The function f(x) = x^2 + a/x is increasing on (1, +∞) when 0 < a < 2 -/
theorem increasing_function_condition (a : ℝ) :
  (0 < a ∧ a < 2) →
  ∃ (f : ℝ → ℝ), (∀ x > 1, f x = x^2 + a/x) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (¬ ∀ a, (∃ (f : ℝ → ℝ), (∀ x > 1, f x = x^2 + a/x) ∧
    (∀ x y, 1 < x ∧ x < y → f x < f y)) → (0 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2190_219007


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l2190_219065

-- Define the pentagon and its angles
structure Pentagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ

-- Define the theorem
theorem pentagon_angle_sum (p : Pentagon) 
  (h1 : p.A = 40)
  (h2 : p.F = p.G) : 
  p.B + p.D = 70 := by
  sorry

#check pentagon_angle_sum

end NUMINAMATH_CALUDE_pentagon_angle_sum_l2190_219065


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2190_219042

/-- Estimates the number of fish in a pond based on a capture-recapture method. -/
def estimate_fish_population (tagged_fish : ℕ) (second_catch : ℕ) (recaptured : ℕ) : ℕ :=
  (tagged_fish * second_catch) / recaptured

/-- Theorem stating that given the specific conditions of the problem, 
    the estimated fish population is 600. -/
theorem fish_population_estimate :
  let tagged_fish : ℕ := 30
  let second_catch : ℕ := 40
  let recaptured : ℕ := 2
  estimate_fish_population tagged_fish second_catch recaptured = 600 := by
  sorry

#eval estimate_fish_population 30 40 2

end NUMINAMATH_CALUDE_fish_population_estimate_l2190_219042


namespace NUMINAMATH_CALUDE_f_properties_l2190_219022

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, x2 > x1 → f x2 > f x1) ∧
  (∀ x t : ℝ, x ∈ Set.Icc 1 2 → (f (x - t) + f (x^2 - t^2) ≥ 0 ↔ t ∈ Set.Icc (-2) 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2190_219022


namespace NUMINAMATH_CALUDE_equation_solution_l2190_219090

theorem equation_solution : 
  ∀ x : ℝ, (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2190_219090


namespace NUMINAMATH_CALUDE_shakes_undetermined_l2190_219074

/-- Represents the price of a burger -/
def burger_price : ℝ := sorry

/-- Represents the price of a shake -/
def shake_price : ℝ := sorry

/-- Represents the price of a cola -/
def cola_price : ℝ := sorry

/-- Represents the number of shakes in the second purchase -/
def num_shakes_second : ℝ := sorry

/-- The total cost of the first purchase -/
def first_purchase : Prop :=
  3 * burger_price + 7 * shake_price + cola_price = 120

/-- The total cost of the second purchase -/
def second_purchase : Prop :=
  4 * burger_price + num_shakes_second * shake_price + cola_price = 164.5

/-- Theorem stating that the number of shakes in the second purchase cannot be uniquely determined -/
theorem shakes_undetermined (h1 : first_purchase) (h2 : second_purchase) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (4 * burger_price + x * shake_price + cola_price = 164.5) ∧
    (4 * burger_price + y * shake_price + cola_price = 164.5) :=
  sorry

end NUMINAMATH_CALUDE_shakes_undetermined_l2190_219074


namespace NUMINAMATH_CALUDE_nanometers_to_meters_l2190_219029

-- Define the conversion factors
def nanometer_to_millimeter : ℝ := 1e-6
def millimeter_to_meter : ℝ := 1e-3

-- Define the given length in nanometers
def length_in_nanometers : ℝ := 3e10

-- State the theorem
theorem nanometers_to_meters :
  length_in_nanometers * nanometer_to_millimeter * millimeter_to_meter = 30 := by
  sorry

end NUMINAMATH_CALUDE_nanometers_to_meters_l2190_219029


namespace NUMINAMATH_CALUDE_least_integer_with_deletion_property_l2190_219087

theorem least_integer_with_deletion_property : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n = 17) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (m / 10 : ℚ) ≠ m / 17) ∧
  ((n / 10 : ℚ) = n / 17) := by
sorry

end NUMINAMATH_CALUDE_least_integer_with_deletion_property_l2190_219087


namespace NUMINAMATH_CALUDE_eighth_of_2_38_l2190_219077

theorem eighth_of_2_38 (x : ℕ) :
  (1 / 8 : ℝ) * (2 : ℝ)^38 = (2 : ℝ)^x → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_2_38_l2190_219077


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2190_219013

theorem base_conversion_problem : ∃ (n A B : ℕ), 
  n > 0 ∧
  n = 8 * A + B ∧
  n = 6 * B + A ∧
  A < 8 ∧
  B < 6 ∧
  n = 47 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2190_219013


namespace NUMINAMATH_CALUDE_factor_implies_root_l2190_219033

theorem factor_implies_root (a : ℝ) : 
  (∀ t : ℝ, (2*t + 1) ∣ (4*t^2 + 12*t + a)) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_root_l2190_219033


namespace NUMINAMATH_CALUDE_teresas_pencils_l2190_219083

/-- Teresa's pencil distribution problem -/
theorem teresas_pencils (colored_pencils black_pencils : ℕ) 
  (num_siblings pencils_per_sibling : ℕ) : 
  colored_pencils = 14 →
  black_pencils = 35 →
  num_siblings = 3 →
  pencils_per_sibling = 13 →
  colored_pencils + black_pencils - num_siblings * pencils_per_sibling = 10 :=
by sorry

end NUMINAMATH_CALUDE_teresas_pencils_l2190_219083


namespace NUMINAMATH_CALUDE_magic_square_property_l2190_219091

def magic_square : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![7.5, 5, 2.5],
    ![0, 5, 10],
    ![7.5, 5, 2.5]]

def row_sum (m : Matrix (Fin 3) (Fin 3) ℚ) (i : Fin 3) : ℚ :=
  m i 0 + m i 1 + m i 2

def col_sum (m : Matrix (Fin 3) (Fin 3) ℚ) (j : Fin 3) : ℚ :=
  m 0 j + m 1 j + m 2 j

def diag_sum (m : Matrix (Fin 3) (Fin 3) ℚ) : ℚ :=
  m 0 0 + m 1 1 + m 2 2

def anti_diag_sum (m : Matrix (Fin 3) (Fin 3) ℚ) : ℚ :=
  m 0 2 + m 1 1 + m 2 0

theorem magic_square_property :
  (∀ i : Fin 3, row_sum magic_square i = 15) ∧
  (∀ j : Fin 3, col_sum magic_square j = 15) ∧
  diag_sum magic_square = 15 ∧
  anti_diag_sum magic_square = 15 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_property_l2190_219091


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l2190_219032

theorem zeroth_power_of_nonzero_rational (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l2190_219032


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2190_219028

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2190_219028


namespace NUMINAMATH_CALUDE_ones_digit_of_9_to_53_l2190_219034

theorem ones_digit_of_9_to_53 : Nat.mod (9^53) 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_9_to_53_l2190_219034


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l2190_219004

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  2 * Real.sin A ^ 2 + 3 * Real.cos (B + C) = 0 →
  S = 5 * Real.sqrt 3 →
  a = Real.sqrt 21 →
  A = π / 3 ∧ b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l2190_219004


namespace NUMINAMATH_CALUDE_student_average_less_than_true_average_l2190_219062

theorem student_average_less_than_true_average 
  (x y w : ℝ) (h : x > y ∧ y > w) : 
  ((x + y) / 2 + w) / 2 < (x + y + w) / 3 := by
sorry

end NUMINAMATH_CALUDE_student_average_less_than_true_average_l2190_219062


namespace NUMINAMATH_CALUDE_equation_equivalence_l2190_219021

-- Define the original equation
def original_equation (x y : ℝ) : Prop := 2 * x - 3 * y - 4 = 0

-- Define the intercept form
def intercept_form (x y : ℝ) : Prop := x / 2 + y / (-4/3) = 1

-- Theorem stating the equivalence of the two forms
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ intercept_form x y :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2190_219021


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l2190_219037

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) :
  initial_volume = 6 →
  initial_percentage = 0.3 →
  added_alcohol = 2.4 →
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l2190_219037


namespace NUMINAMATH_CALUDE_tessa_final_debt_l2190_219039

/-- Calculates the final debt given an initial debt, a fractional repayment, and an additional loan --/
def finalDebt (initialDebt : ℚ) (repaymentFraction : ℚ) (additionalLoan : ℚ) : ℚ :=
  initialDebt - (repaymentFraction * initialDebt) + additionalLoan

/-- Proves that Tessa's final debt is $30 --/
theorem tessa_final_debt :
  finalDebt 40 (1/2) 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tessa_final_debt_l2190_219039


namespace NUMINAMATH_CALUDE_cistern_problem_l2190_219031

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_sides_area := 2 * length * depth
  let short_sides_area := 2 * width * depth
  bottom_area + long_sides_area + short_sides_area

/-- Theorem stating that for a cistern with given dimensions, the wet surface area is 88 square meters -/
theorem cistern_problem : 
  cistern_wet_surface_area 12 4 1.25 = 88 := by
  sorry

#eval cistern_wet_surface_area 12 4 1.25

end NUMINAMATH_CALUDE_cistern_problem_l2190_219031


namespace NUMINAMATH_CALUDE_curve_symmetry_condition_l2190_219096

/-- Given a curve y = (mx + n) / (tx + u) symmetric about y = x, prove m - u = 0 -/
theorem curve_symmetry_condition 
  (m n t u : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h_symmetry : ∀ x y : ℝ, y = (m * x + n) / (t * x + u) ↔ x = (m * y + n) / (t * y + u)) :
  m - u = 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_condition_l2190_219096
