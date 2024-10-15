import Mathlib

namespace NUMINAMATH_CALUDE_cone_sphere_volume_equality_implies_lateral_area_l2448_244896

/-- Given a cone with base radius 1 and a sphere with radius 1, if their volumes are equal,
    then the lateral surface area of the cone is √17π. -/
theorem cone_sphere_volume_equality_implies_lateral_area (π : ℝ) (h : ℝ) :
  (1/3 : ℝ) * π * 1^2 * h = (4/3 : ℝ) * π * 1^3 →
  π * 1 * (1^2 + h^2).sqrt = π * Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_equality_implies_lateral_area_l2448_244896


namespace NUMINAMATH_CALUDE_four_Y_three_equals_negative_eleven_l2448_244836

/-- The Y operation defined for any two real numbers -/
def Y (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2

/-- Theorem stating that 4 Y 3 equals -11 -/
theorem four_Y_three_equals_negative_eleven : Y 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_negative_eleven_l2448_244836


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2448_244880

theorem least_five_digit_square_cube : 
  (∀ n : ℕ, n < 15625 → ¬(∃ a b : ℕ, n = a^2 ∧ n = b^3 ∧ n ≥ 10000)) ∧ 
  (∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3) ∧ 
  15625 ≥ 10000 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2448_244880


namespace NUMINAMATH_CALUDE_smallest_n_sum_squares_over_n_is_square_l2448_244877

/-- Sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is a perfect square -/
def is_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- Predicate to check if the sum of squares divided by n is a square -/
def is_sum_of_squares_over_n_square (n : ℕ) : Prop :=
  is_square (sum_of_squares n / n)

theorem smallest_n_sum_squares_over_n_is_square :
  (∀ m : ℕ, m > 1 ∧ m < 337 → ¬is_sum_of_squares_over_n_square m) ∧
  is_sum_of_squares_over_n_square 337 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_sum_squares_over_n_is_square_l2448_244877


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l2448_244823

theorem square_diff_fourth_power : (7^2 - 5^2)^4 = 331776 := by sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l2448_244823


namespace NUMINAMATH_CALUDE_slope_range_of_intersecting_line_l2448_244891

/-- Given points A, B, and P, and a line l passing through P and intersecting line segment AB,
    prove that the range of the slope of line l is [0, π/4] ∪ [3π/4, π). -/
theorem slope_range_of_intersecting_line (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) 
    (hA : A = (1, -2))
    (hB : B = (2, 1))
    (hP : P = (0, -1))
    (hl : P ∈ l)
    (hintersect : ∃ Q ∈ l, Q ∈ Set.Icc A B) :
  ∃ s : Set ℝ, s = Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π ∧
    ∀ θ : ℝ, (∃ Q ∈ l, Q ≠ P ∧ Real.tan θ = (Q.2 - P.2) / (Q.1 - P.1)) → θ ∈ s :=
sorry

end NUMINAMATH_CALUDE_slope_range_of_intersecting_line_l2448_244891


namespace NUMINAMATH_CALUDE_sum_equals_three_l2448_244853

/-- The largest proper fraction with denominator 9 -/
def largest_proper_fraction : ℚ := 8/9

/-- The smallest improper fraction with denominator 9 -/
def smallest_improper_fraction : ℚ := 9/9

/-- The smallest mixed number with fractional part having denominator 9 -/
def smallest_mixed_number : ℚ := 1 + 1/9

/-- The sum of the largest proper fraction, smallest improper fraction, and smallest mixed number -/
def sum_of_fractions : ℚ := largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number

theorem sum_equals_three : sum_of_fractions = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_three_l2448_244853


namespace NUMINAMATH_CALUDE_circle_area_relationship_l2448_244833

theorem circle_area_relationship (A B : ℝ → ℝ → Prop) : 
  (∃ r : ℝ, (∀ x y : ℝ, A x y ↔ (x - r)^2 + (y - r)^2 = r^2) ∧ 
             (∀ x y : ℝ, B x y ↔ (x - 2*r)^2 + (y - 2*r)^2 = (2*r)^2)) →
  (π * r^2 = 16 * π) →
  (π * (2*r)^2 = 64 * π) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_relationship_l2448_244833


namespace NUMINAMATH_CALUDE_no_perfect_cube_solution_l2448_244878

theorem no_perfect_cube_solution : ¬∃ (n : ℕ), n > 0 ∧ ∃ (y : ℕ), 3 * n^2 + 3 * n + 7 = y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_solution_l2448_244878


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2448_244852

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |3*x - 9|

-- Define the domain
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x, domain x → g x ≤ max_g) ∧
    (∃ x, domain x ∧ g x = max_g) ∧
    (∀ x, domain x → min_g ≤ g x) ∧
    (∃ x, domain x ∧ g x = min_g) ∧
    max_g + min_g = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2448_244852


namespace NUMINAMATH_CALUDE_exactly_one_truck_congestion_at_least_two_trucks_congestion_l2448_244855

-- Define the probabilities for highways I and II
def prob_congestion_I : ℚ := 1/10
def prob_no_congestion_I : ℚ := 9/10
def prob_congestion_II : ℚ := 3/5
def prob_no_congestion_II : ℚ := 2/5

-- Define the events
def event_A : ℚ := prob_congestion_I
def event_B : ℚ := prob_congestion_I
def event_C : ℚ := prob_congestion_II

-- Theorem for the first question
theorem exactly_one_truck_congestion :
  prob_congestion_I * prob_no_congestion_I + prob_no_congestion_I * prob_congestion_I = 9/50 := by sorry

-- Theorem for the second question
theorem at_least_two_trucks_congestion :
  event_A * event_B * (1 - event_C) + 
  event_A * (1 - event_B) * event_C + 
  (1 - event_A) * event_B * event_C + 
  event_A * event_B * event_C = 59/500 := by sorry

end NUMINAMATH_CALUDE_exactly_one_truck_congestion_at_least_two_trucks_congestion_l2448_244855


namespace NUMINAMATH_CALUDE_emails_remaining_proof_l2448_244856

/-- Given an initial number of emails, calculates the number of emails remaining in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
def remaining_emails (initial : ℕ) : ℕ :=
  let after_trash := initial / 2
  let to_work_folder := (40 * after_trash) / 100
  after_trash - to_work_folder

/-- Proves that given 400 initial emails, 120 emails remain in the inbox after the operations. -/
theorem emails_remaining_proof :
  remaining_emails 400 = 120 := by
  sorry

#eval remaining_emails 400  -- Should output 120

end NUMINAMATH_CALUDE_emails_remaining_proof_l2448_244856


namespace NUMINAMATH_CALUDE_age_ratio_this_year_l2448_244847

def yoongi_age_last_year : ℕ := 6
def grandfather_age_last_year : ℕ := 62

def yoongi_age_this_year : ℕ := yoongi_age_last_year + 1
def grandfather_age_this_year : ℕ := grandfather_age_last_year + 1

theorem age_ratio_this_year :
  grandfather_age_this_year / yoongi_age_this_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_this_year_l2448_244847


namespace NUMINAMATH_CALUDE_largest_n_for_exponential_inequality_l2448_244890

theorem largest_n_for_exponential_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), Real.exp (n * x) + Real.exp (-n * x) ≥ n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), Real.exp (m * y) + Real.exp (-m * y) < m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_exponential_inequality_l2448_244890


namespace NUMINAMATH_CALUDE_modulus_of_complex_l2448_244863

def i : ℂ := Complex.I

theorem modulus_of_complex (z : ℂ) : z = (2 + i) / (1 - i) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l2448_244863


namespace NUMINAMATH_CALUDE_min_value_trig_fraction_l2448_244871

theorem min_value_trig_fraction (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 3 ≥ (2/3) * ((Real.sin x)^6 + (Real.cos x)^6 + 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_fraction_l2448_244871


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2448_244813

theorem smallest_winning_number : ∃ N : ℕ,
  N ≥ 0 ∧ N ≤ 999 ∧
  (∀ m : ℕ, m ≥ 0 ∧ m < N →
    (3*m < 1000 ∧
     3*m - 30 < 1000 ∧
     9*m - 90 < 1000 ∧
     9*m - 120 < 1000 ∧
     27*m - 360 < 1000 ∧
     27*m - 390 < 1000 ∧
     81*m - 1170 < 1000 ∧
     81*(m-1) - 1170 ≥ 1000)) ∧
  3*N < 1000 ∧
  3*N - 30 < 1000 ∧
  9*N - 90 < 1000 ∧
  9*N - 120 < 1000 ∧
  27*N - 360 < 1000 ∧
  27*N - 390 < 1000 ∧
  81*N - 1170 < 1000 ∧
  81*(N-1) - 1170 ≥ 1000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2448_244813


namespace NUMINAMATH_CALUDE_unique_solution_l2448_244815

theorem unique_solution : ∃! (x p : ℕ), 
  Prime p ∧ 
  x * (x + 1) * (x + 2) * (x + 3) = 1679^(p - 1) + 1680^(p - 1) + 1681^(p - 1) ∧
  x = 4 ∧ p = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2448_244815


namespace NUMINAMATH_CALUDE_three_pairs_probability_l2448_244848

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (cards_per_rank : Nat)
  (h1 : cards = ranks * cards_per_rank)

/-- A poker hand -/
structure PokerHand :=
  (size : Nat)

/-- The probability of drawing a specific hand -/
def probability (deck : Deck) (hand : PokerHand) (valid_hands : Nat) : Rat :=
  valid_hands / (Nat.choose deck.cards hand.size)

/-- Theorem: Probability of drawing exactly three pairs in a 6-card hand -/
theorem three_pairs_probability (d : Deck) (h : PokerHand) : 
  d.cards = 52 → d.ranks = 13 → d.cards_per_rank = 4 → h.size = 6 → 
  probability d h ((Nat.choose d.ranks 3) * (Nat.choose d.cards_per_rank 2)^3) = 154/51845 := by
  sorry


end NUMINAMATH_CALUDE_three_pairs_probability_l2448_244848


namespace NUMINAMATH_CALUDE_josie_shortage_l2448_244830

def gift_amount : ℝ := 150
def cassette_count : ℕ := 5
def cassette_price : ℝ := 18
def headphone_count : ℕ := 2
def headphone_price : ℝ := 45
def vinyl_count : ℕ := 3
def vinyl_price : ℝ := 22
def magazine_count : ℕ := 4
def magazine_price : ℝ := 7

def total_cost : ℝ :=
  cassette_count * cassette_price +
  headphone_count * headphone_price +
  vinyl_count * vinyl_price +
  magazine_count * magazine_price

theorem josie_shortage : gift_amount - total_cost = -124 := by
  sorry

end NUMINAMATH_CALUDE_josie_shortage_l2448_244830


namespace NUMINAMATH_CALUDE_nine_bounces_on_12x10_table_l2448_244850

/-- Represents a rectangular pool table -/
structure PoolTable where
  width : ℕ
  height : ℕ

/-- Represents a ball's path on the pool table -/
structure BallPath where
  start_x : ℕ
  start_y : ℕ
  slope : ℚ

/-- Calculates the number of wall bounces for a ball's path on a pool table -/
def count_wall_bounces (table : PoolTable) (path : BallPath) : ℕ :=
  sorry

/-- Theorem stating that a ball hit from (0,0) along y=x on a 12x10 table bounces 9 times -/
theorem nine_bounces_on_12x10_table :
  let table : PoolTable := { width := 12, height := 10 }
  let path : BallPath := { start_x := 0, start_y := 0, slope := 1 }
  count_wall_bounces table path = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_bounces_on_12x10_table_l2448_244850


namespace NUMINAMATH_CALUDE_final_i_is_16_l2448_244857

def update_i (i : ℕ) : ℕ :=
  let new_i := 2 * i
  if new_i > 20 then new_i - 20 else new_i

def final_i : ℕ :=
  (List.range 5).foldl (fun acc _ => update_i acc) 2

theorem final_i_is_16 : final_i = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_i_is_16_l2448_244857


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l2448_244881

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {-3+a, 2*a-1, a^2+1}

theorem intersection_and_union_of_sets :
  ∃ (a : ℝ), (A a ∩ B a = {-3}) ∧ (a = -1) ∧ (A a ∪ B a = {-4, -3, 0, 1, 2}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l2448_244881


namespace NUMINAMATH_CALUDE_geli_workout_results_l2448_244801

/-- Represents a workout routine with push-ups and runs -/
structure WorkoutRoutine where
  workoutsPerWeek : ℕ
  weeks : ℕ
  initialPushups : ℕ
  pushupIncrement : ℕ
  pushupsMileRatio : ℕ

/-- Calculates the total number of push-ups for a given workout routine -/
def totalPushups (routine : WorkoutRoutine) : ℕ :=
  let totalDays := routine.workoutsPerWeek * routine.weeks
  let lastDayPushups := routine.initialPushups + (totalDays - 1) * routine.pushupIncrement
  totalDays * (routine.initialPushups + lastDayPushups) / 2

/-- Calculates the number of one-mile runs based on the total push-ups -/
def totalRuns (routine : WorkoutRoutine) : ℕ :=
  totalPushups routine / routine.pushupsMileRatio

/-- Theorem stating the results for Geli's specific workout routine -/
theorem geli_workout_results :
  let routine : WorkoutRoutine := {
    workoutsPerWeek := 3,
    weeks := 4,
    initialPushups := 10,
    pushupIncrement := 5,
    pushupsMileRatio := 30
  }
  totalPushups routine = 450 ∧ totalRuns routine = 15 := by
  sorry


end NUMINAMATH_CALUDE_geli_workout_results_l2448_244801


namespace NUMINAMATH_CALUDE_nested_expression_value_l2448_244862

theorem nested_expression_value : (3*(3*(3*(3*(3*(3+2)+2)+2)+2)+2)+2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l2448_244862


namespace NUMINAMATH_CALUDE_y_axis_inclination_l2448_244885

-- Define the concept of an axis
def Axis : Type := ℝ → ℝ

-- Define the x-axis and y-axis
def x_axis : Axis := λ x => 0
def y_axis : Axis := λ y => y

-- Define the concept of perpendicular axes
def perpendicular (a b : Axis) : Prop := sorry

-- Define the concept of inclination angle
def inclination_angle (a : Axis) : ℝ := sorry

-- Theorem statement
theorem y_axis_inclination :
  perpendicular x_axis y_axis →
  inclination_angle y_axis = 90 :=
sorry

end NUMINAMATH_CALUDE_y_axis_inclination_l2448_244885


namespace NUMINAMATH_CALUDE_fourth_hexagon_dots_l2448_244884

/-- Calculates the number of dots in the nth hexagon of the pattern. -/
def hexagonDots (n : ℕ) : ℕ :=
  if n = 0 then 1
  else hexagonDots (n - 1) + 6 * n

/-- The number of dots in the fourth hexagon is 55. -/
theorem fourth_hexagon_dots : hexagonDots 4 = 55 := by sorry

end NUMINAMATH_CALUDE_fourth_hexagon_dots_l2448_244884


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_system_l2448_244824

theorem unique_solution_diophantine_system :
  ∀ a b c : ℕ,
  a^3 - b^3 - c^3 = 3*a*b*c →
  a^2 = 2*(b + c) →
  a = 2 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_system_l2448_244824


namespace NUMINAMATH_CALUDE_max_students_is_eight_l2448_244895

def knows (n : ℕ) : (Fin n → Fin n → Prop) → Prop :=
  λ f => ∀ (i j : Fin n), i ≠ j → f i j = f j i

def satisfies_conditions (n : ℕ) (f : Fin n → Fin n → Prop) : Prop :=
  knows n f ∧
  (∀ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    f a b ∨ f b c ∨ f a c) ∧
  (∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d → 
    (¬f a b ∧ ¬f c d) ∨ (¬f a c ∧ ¬f b d) ∨ (¬f a d ∧ ¬f b c))

theorem max_students_is_eight :
  (∃ (f : Fin 8 → Fin 8 → Prop), satisfies_conditions 8 f) ∧
  (∀ n > 8, ¬∃ (f : Fin n → Fin n → Prop), satisfies_conditions n f) :=
sorry

end NUMINAMATH_CALUDE_max_students_is_eight_l2448_244895


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l2448_244838

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_interior_sum :
  interior_sum 5 = 14 →
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l2448_244838


namespace NUMINAMATH_CALUDE_g_of_3_eq_6_l2448_244835

/-- A function satisfying the given conditions -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 2 ∧ ∀ x y : ℝ, g (x^2 + y^2) = (x + y) * (g x + g y)

/-- Theorem stating that g(3) = 6 for any function satisfying the conditions -/
theorem g_of_3_eq_6 (g : ℝ → ℝ) (h : special_function g) : g 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_6_l2448_244835


namespace NUMINAMATH_CALUDE_tangent_line_2sinx_at_pi_l2448_244807

/-- The equation of the tangent line to y = 2sin(x) at (π, 0) is y = -2x + 2π -/
theorem tangent_line_2sinx_at_pi (x y : ℝ) : 
  (y = 2 * Real.sin x) → -- curve equation
  (y = -2 * (x - Real.pi) + 0) → -- point-slope form of tangent line
  (y = -2 * x + 2 * Real.pi) -- final equation of tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_2sinx_at_pi_l2448_244807


namespace NUMINAMATH_CALUDE_base_faces_area_sum_l2448_244872

/-- A pentagonal prism with given surface area and lateral area -/
structure PentagonalPrism where
  surfaceArea : ℝ
  lateralArea : ℝ

/-- Theorem: For a pentagonal prism with surface area 30 and lateral area 25,
    the sum of the areas of the two base faces equals 5 -/
theorem base_faces_area_sum (prism : PentagonalPrism)
    (h1 : prism.surfaceArea = 30)
    (h2 : prism.lateralArea = 25) :
    prism.surfaceArea - prism.lateralArea = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_faces_area_sum_l2448_244872


namespace NUMINAMATH_CALUDE_min_value_theorem_l2448_244827

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2448_244827


namespace NUMINAMATH_CALUDE_coordinates_product_l2448_244869

/-- Given points A and M, where M is one-third of the way from A to B, 
    prove that the product of B's coordinates is -85 -/
theorem coordinates_product (A M : ℝ × ℝ) (h1 : A = (4, 2)) (h2 : M = (1, 7)) : 
  let B := (3 * M.1 - 2 * A.1, 3 * M.2 - 2 * A.2)
  B.1 * B.2 = -85 := by sorry

end NUMINAMATH_CALUDE_coordinates_product_l2448_244869


namespace NUMINAMATH_CALUDE_double_counted_is_eight_l2448_244874

/-- The number of double-counted toddlers in Bill's count -/
def double_counted : ℕ := 26 - 21 + 3

/-- Proof that the number of double-counted toddlers is 8 -/
theorem double_counted_is_eight : double_counted = 8 := by
  sorry

#eval double_counted

end NUMINAMATH_CALUDE_double_counted_is_eight_l2448_244874


namespace NUMINAMATH_CALUDE_permutation_100_2_l2448_244809

/-- The number of permutations of n distinct objects taken k at a time -/
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

/-- The permutation A₁₀₀² equals 9900 -/
theorem permutation_100_2 : permutation 100 2 = 9900 := by sorry

end NUMINAMATH_CALUDE_permutation_100_2_l2448_244809


namespace NUMINAMATH_CALUDE_prime_squared_product_l2448_244858

theorem prime_squared_product (p q : ℕ) : 
  Prime p → Prime q → Nat.totient (p^2 * q^2) = 11424 → p^2 * q^2 = 7^2 * 17^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_product_l2448_244858


namespace NUMINAMATH_CALUDE_boys_walking_speed_l2448_244864

/-- 
Given two boys walking in the same direction for 7 hours, with one boy walking at 5.5 km/h 
and ending up 10.5 km apart, prove that the speed of the other boy is 7 km/h.
-/
theorem boys_walking_speed 
  (time : ℝ) 
  (distance_apart : ℝ) 
  (speed_second_boy : ℝ) 
  (speed_first_boy : ℝ) 
  (h1 : time = 7) 
  (h2 : distance_apart = 10.5) 
  (h3 : speed_second_boy = 5.5) 
  (h4 : distance_apart = (speed_first_boy - speed_second_boy) * time) : 
  speed_first_boy = 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_walking_speed_l2448_244864


namespace NUMINAMATH_CALUDE_consecutive_numbers_probability_l2448_244883

def set_size : ℕ := 20
def selection_size : ℕ := 5

def prob_consecutive_numbers : ℚ :=
  1 - (Nat.choose (set_size - selection_size + 1) selection_size : ℚ) / (Nat.choose set_size selection_size : ℚ)

theorem consecutive_numbers_probability :
  prob_consecutive_numbers = 232 / 323 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_probability_l2448_244883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2448_244829

/-- 
An arithmetic sequence is a sequence where the difference between 
any two consecutive terms is constant.
-/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/--
Given an arithmetic sequence a_n where a_5 = 10 and a_12 = 31,
the common difference d is equal to 3.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a5 : a 5 = 10) 
  (h_a12 : a 12 = 31) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2448_244829


namespace NUMINAMATH_CALUDE_nickels_maximize_value_expected_value_is_7480_l2448_244837

/-- Represents the types of coins --/
inductive Coin
| Quarter
| Nickel
| Dime

/-- Represents the material of a coin --/
inductive Material
| Regular
| Iron

/-- The number of quarters Alice has --/
def initial_quarters : ℕ := 20

/-- The exchange rate for quarters to nickels --/
def quarters_to_nickels : ℕ := 4

/-- The exchange rate for quarters to dimes --/
def quarters_to_dimes : ℕ := 2

/-- The probability of a nickel being iron --/
def iron_nickel_prob : ℚ := 3/10

/-- The probability of a dime being iron --/
def iron_dime_prob : ℚ := 1/10

/-- The value of an iron nickel in cents --/
def iron_nickel_value : ℕ := 300

/-- The value of an iron dime in cents --/
def iron_dime_value : ℕ := 500

/-- The value of a regular nickel in cents --/
def regular_nickel_value : ℕ := 5

/-- The value of a regular dime in cents --/
def regular_dime_value : ℕ := 10

/-- Calculates the expected value of a nickel in cents --/
def expected_nickel_value : ℚ :=
  iron_nickel_prob * iron_nickel_value + (1 - iron_nickel_prob) * regular_nickel_value

/-- Calculates the expected value of a dime in cents --/
def expected_dime_value : ℚ :=
  iron_dime_prob * iron_dime_value + (1 - iron_dime_prob) * regular_dime_value

/-- Theorem stating that exchanging for nickels maximizes expected value --/
theorem nickels_maximize_value :
  expected_nickel_value * quarters_to_nickels > expected_dime_value * quarters_to_dimes :=
sorry

/-- Calculates the total number of nickels Alice can get --/
def total_nickels : ℕ := initial_quarters * quarters_to_nickels

/-- Calculates the expected total value in cents after exchanging for nickels --/
def expected_total_value : ℚ := total_nickels * expected_nickel_value

/-- Theorem stating that the expected total value is 7480 cents ($74.80) --/
theorem expected_value_is_7480 : expected_total_value = 7480 := sorry

end NUMINAMATH_CALUDE_nickels_maximize_value_expected_value_is_7480_l2448_244837


namespace NUMINAMATH_CALUDE_d_equals_square_iff_l2448_244819

/-- Move the last digit of a number to the first position -/
def moveLastToFirst (a : ℕ) : ℕ :=
  sorry

/-- Square a number -/
def square (b : ℕ) : ℕ :=
  sorry

/-- Move the first digit of a number to the end -/
def moveFirstToLast (c : ℕ) : ℕ :=
  sorry

/-- The d(a) function as described in the problem -/
def d (a : ℕ) : ℕ :=
  moveFirstToLast (square (moveLastToFirst a))

/-- Check if a number is of the form 222...21 -/
def is222_21 (a : ℕ) : Prop :=
  sorry

/-- The main theorem -/
theorem d_equals_square_iff (a : ℕ) :
  d a = a^2 ↔ a = 1 ∨ a = 2 ∨ a = 3 ∨ is222_21 a :=
sorry

end NUMINAMATH_CALUDE_d_equals_square_iff_l2448_244819


namespace NUMINAMATH_CALUDE_unique_power_of_two_plus_one_l2448_244875

theorem unique_power_of_two_plus_one : 
  ∃! (n : ℕ), ∃ (A p : ℕ), p > 1 ∧ 2^n + 1 = A^p :=
by
  sorry

end NUMINAMATH_CALUDE_unique_power_of_two_plus_one_l2448_244875


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l2448_244806

theorem average_speed_round_trip (outbound_speed inbound_speed : ℝ) 
  (h1 : outbound_speed = 130)
  (h2 : inbound_speed = 88)
  (h3 : outbound_speed > 0)
  (h4 : inbound_speed > 0) :
  (2 * outbound_speed * inbound_speed) / (outbound_speed + inbound_speed) = 105 := by
sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l2448_244806


namespace NUMINAMATH_CALUDE_next_event_occurrence_l2448_244849

/-- Represents the periodic event occurrence pattern -/
structure EventPattern where
  x : ℕ  -- Number of consecutive years the event occurs
  y : ℕ  -- Number of consecutive years of break

/-- Checks if the event occurs in a given year based on the pattern and a reference year -/
def eventOccurs (pattern : EventPattern) (referenceYear : ℕ) (year : ℕ) : Prop :=
  (year - referenceYear) % (pattern.x + pattern.y) < pattern.x

/-- The main theorem stating the next occurrence of the event after 2013 -/
theorem next_event_occurrence (pattern : EventPattern) : 
  (eventOccurs pattern 1964 1964) ∧
  (eventOccurs pattern 1964 1986) ∧
  (eventOccurs pattern 1964 1996) ∧
  (eventOccurs pattern 1964 2008) ∧
  (¬ eventOccurs pattern 1964 1976) ∧
  (¬ eventOccurs pattern 1964 1993) ∧
  (¬ eventOccurs pattern 1964 2006) ∧
  (¬ eventOccurs pattern 1964 2013) →
  ∀ year : ℕ, year > 2013 → eventOccurs pattern 1964 year → year ≥ 2018 :=
by
  sorry

#check next_event_occurrence

end NUMINAMATH_CALUDE_next_event_occurrence_l2448_244849


namespace NUMINAMATH_CALUDE_not_right_triangle_condition_l2448_244822

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The theorem to be proved -/
theorem not_right_triangle_condition (t : Triangle) 
  (h1 : t.a = 3^2)
  (h2 : t.b = 4^2)
  (h3 : t.c = 5^2) : 
  ¬ is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_condition_l2448_244822


namespace NUMINAMATH_CALUDE_earnings_calculation_l2448_244826

/-- If a person spends 10% of their earnings and is left with $405, prove their total earnings were $450. -/
theorem earnings_calculation (spent_percentage : Real) (remaining_amount : Real) (total_earnings : Real) : 
  spent_percentage = 0.1 →
  remaining_amount = 405 →
  remaining_amount = (1 - spent_percentage) * total_earnings →
  total_earnings = 450 := by
sorry

end NUMINAMATH_CALUDE_earnings_calculation_l2448_244826


namespace NUMINAMATH_CALUDE_min_value_of_product_l2448_244804

/-- Given positive real numbers x₁, x₂, x₃, x₄ such that their sum is π,
    the product of (2sin²xᵢ + 1/sin²xᵢ) for i = 1 to 4 has a minimum value of 81. -/
theorem min_value_of_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0)
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l2448_244804


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2448_244893

theorem possible_values_of_a (a b c : ℝ) 
  (eq1 : a * b + a + b = c)
  (eq2 : b * c + b + c = a)
  (eq3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2448_244893


namespace NUMINAMATH_CALUDE_horner_rule_evaluation_l2448_244879

def horner_polynomial (x : ℝ) : ℝ :=
  (((((2 * x - 0) * x - 3) * x + 2) * x + 7) * x + 6) * x + 3

theorem horner_rule_evaluation :
  horner_polynomial 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_evaluation_l2448_244879


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l2448_244870

theorem restaurant_bill_calculation (num_adults num_teenagers num_children : ℕ)
  (adult_meal_cost teenager_meal_cost child_meal_cost : ℚ)
  (soda_cost dessert_cost appetizer_cost : ℚ)
  (num_desserts num_appetizers : ℕ)
  (h1 : num_adults = 6)
  (h2 : num_teenagers = 3)
  (h3 : num_children = 1)
  (h4 : adult_meal_cost = 9)
  (h5 : teenager_meal_cost = 7)
  (h6 : child_meal_cost = 5)
  (h7 : soda_cost = 2.5)
  (h8 : dessert_cost = 4)
  (h9 : appetizer_cost = 6)
  (h10 : num_desserts = 3)
  (h11 : num_appetizers = 2) :
  (num_adults * adult_meal_cost +
   num_teenagers * teenager_meal_cost +
   num_children * child_meal_cost +
   (num_adults + num_teenagers + num_children) * soda_cost +
   num_desserts * dessert_cost +
   num_appetizers * appetizer_cost) = 129 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l2448_244870


namespace NUMINAMATH_CALUDE_tan_product_simplification_l2448_244892

theorem tan_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l2448_244892


namespace NUMINAMATH_CALUDE_victoria_work_hours_l2448_244845

/-- Calculates the number of hours worked per day given the total hours and number of weeks worked. -/
def hours_per_day (total_hours : ℕ) (weeks : ℕ) : ℚ :=
  total_hours / (weeks * 7)

/-- Theorem: Given 315 total hours worked over 5 weeks, the number of hours worked per day is 9. -/
theorem victoria_work_hours :
  hours_per_day 315 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_victoria_work_hours_l2448_244845


namespace NUMINAMATH_CALUDE_allowance_spent_at_toy_store_l2448_244860

theorem allowance_spent_at_toy_store 
  (total_allowance : ℚ)
  (arcade_fraction : ℚ)
  (remaining_after_toy_store : ℚ)
  (h1 : total_allowance = 9/4)  -- $2.25 as a fraction
  (h2 : arcade_fraction = 3/5)
  (h3 : remaining_after_toy_store = 3/5)  -- $0.60 as a fraction
  : (total_allowance - arcade_fraction * total_allowance - remaining_after_toy_store) / 
    (total_allowance - arcade_fraction * total_allowance) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_allowance_spent_at_toy_store_l2448_244860


namespace NUMINAMATH_CALUDE_function_value_symmetry_l2448_244887

/-- Given a function f(x) = ax^7 + bx - 2 where f(2008) = 10, prove that f(-2008) = -12 -/
theorem function_value_symmetry (a b : ℝ) :
  let f := λ x : ℝ => a * x^7 + b * x - 2
  f 2008 = 10 → f (-2008) = -12 := by
sorry

end NUMINAMATH_CALUDE_function_value_symmetry_l2448_244887


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l2448_244803

-- Define the cubic polynomial whose roots are a, b, c
def cubic (x : ℝ) : ℝ := x^3 + 4*x^2 + 6*x + 9

-- Define the properties of P
def P_properties (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  cubic a = 0 ∧ cubic b = 0 ∧ cubic c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- Theorem statement
theorem cubic_polynomial_theorem :
  ∀ (P : ℝ → ℝ) (a b c : ℝ),
  P_properties P a b c →
  (∀ x, P x = 16*x^3 + 64*x^2 + 90*x + 140) :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l2448_244803


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2448_244898

theorem polygon_interior_angles_sum (n : ℕ) (sum : ℝ) : 
  sum = 900 → (n - 2) * 180 = sum → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2448_244898


namespace NUMINAMATH_CALUDE_sqrt_121_equals_plus_minus_11_l2448_244839

theorem sqrt_121_equals_plus_minus_11 : ∀ (x : ℝ), x^2 = 121 ↔ x = 11 ∨ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_121_equals_plus_minus_11_l2448_244839


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l2448_244899

theorem quadratic_equation_k_value (x₁ x₂ k : ℝ) : 
  x₁^2 - 3*x₁ + k = 0 →
  x₂^2 - 3*x₂ + k = 0 →
  x₁ * x₂ + 2*x₁ + 2*x₂ = 1 →
  k = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l2448_244899


namespace NUMINAMATH_CALUDE_johns_average_speed_l2448_244882

/-- Calculates the overall average speed given two activities with their respective durations and speeds. -/
def overall_average_speed (duration1 duration2 : ℚ) (speed1 speed2 : ℚ) : ℚ :=
  (duration1 * speed1 + duration2 * speed2) / (duration1 + duration2)

/-- Proves that John's overall average speed is 11.6 mph given his scooter ride and jog. -/
theorem johns_average_speed :
  let scooter_duration : ℚ := 40 / 60  -- 40 minutes in hours
  let scooter_speed : ℚ := 20  -- 20 mph
  let jog_duration : ℚ := 60 / 60  -- 60 minutes in hours
  let jog_speed : ℚ := 6  -- 6 mph
  overall_average_speed scooter_duration jog_duration scooter_speed jog_speed = 58 / 5 := by
  sorry

#eval (58 : ℚ) / 5  -- Should evaluate to 11.6

end NUMINAMATH_CALUDE_johns_average_speed_l2448_244882


namespace NUMINAMATH_CALUDE_power_of_product_l2448_244832

theorem power_of_product (a b : ℝ) : ((-3 * a^2 * b^3)^2) = 9 * a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2448_244832


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l2448_244816

theorem fruit_drink_volume (grapefruit_percent : ℝ) (lemon_percent : ℝ) (orange_volume : ℝ) :
  grapefruit_percent = 0.25 →
  lemon_percent = 0.35 →
  orange_volume = 20 →
  ∃ total_volume : ℝ,
    total_volume = 50 ∧
    grapefruit_percent * total_volume + lemon_percent * total_volume + orange_volume = total_volume :=
by sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l2448_244816


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l2448_244814

def total_marbles : ℕ := 12
def marbles_per_color : ℕ := 3
def num_colors : ℕ := 4
def marbles_taken_each_turn : ℕ := 3

def probability_cheryl_same_color : ℚ := 2 / 55

theorem cheryl_same_color_probability :
  let total_outcomes := Nat.choose total_marbles marbles_taken_each_turn *
                        Nat.choose (total_marbles - marbles_taken_each_turn) marbles_taken_each_turn *
                        Nat.choose (total_marbles - 2 * marbles_taken_each_turn) marbles_taken_each_turn
  let favorable_outcomes := num_colors * Nat.choose (total_marbles - marbles_taken_each_turn) marbles_taken_each_turn *
                            Nat.choose (total_marbles - 2 * marbles_taken_each_turn) marbles_taken_each_turn
  (favorable_outcomes : ℚ) / total_outcomes = probability_cheryl_same_color := by
  sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l2448_244814


namespace NUMINAMATH_CALUDE_min_initial_coins_l2448_244805

/-- Represents the game state at each round -/
structure GameState where
  huanhuan : ℕ
  lele : ℕ

/-- Represents the game with initial state and two rounds -/
structure Game where
  initial : GameState
  first_round : ℕ
  second_round : ℕ

/-- Checks if the game satisfies all the given conditions -/
def valid_game (g : Game) : Prop :=
  g.initial.huanhuan = 7 * g.initial.lele ∧
  g.initial.huanhuan + g.first_round = 6 * (g.initial.lele + g.first_round) ∧
  g.initial.huanhuan + g.first_round + g.second_round = 
    5 * (g.initial.lele + g.first_round + g.second_round)

/-- Theorem stating the minimum number of gold coins Huanhuan had at the beginning -/
theorem min_initial_coins (g : Game) (h : valid_game g) : g.initial.huanhuan ≥ 70 := by
  sorry

#check min_initial_coins

end NUMINAMATH_CALUDE_min_initial_coins_l2448_244805


namespace NUMINAMATH_CALUDE_uniform_scores_smaller_variance_l2448_244866

/-- Class scores data -/
structure ClassScores where
  mean : ℝ
  variance : ℝ

/-- Uniformity of scores -/
def more_uniform (a b : ClassScores) : Prop :=
  a.variance < b.variance

/-- Theorem: Class with smaller variance has more uniform scores -/
theorem uniform_scores_smaller_variance 
  (class_a class_b : ClassScores) 
  (h_mean : class_a.mean = class_b.mean) 
  (h_var : class_a.variance > class_b.variance) : 
  more_uniform class_b class_a :=
by sorry

end NUMINAMATH_CALUDE_uniform_scores_smaller_variance_l2448_244866


namespace NUMINAMATH_CALUDE_arc_length_theorem_l2448_244817

-- Define the curve
def curve (x y : ℝ) : Prop := Real.exp (2 * y) * (Real.exp (2 * x) - 1) = Real.exp (2 * x) + 1

-- Define the arc length function
noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x) ^ 2)

-- State the theorem
theorem arc_length_theorem :
  ∃ f : ℝ → ℝ,
    (∀ x, curve x (f x)) ∧
    arcLength f 1 2 = (1 / 2) * Real.log (Real.exp 4 + 1) - 1 := by sorry

end NUMINAMATH_CALUDE_arc_length_theorem_l2448_244817


namespace NUMINAMATH_CALUDE_surface_points_is_75_l2448_244802

/-- Represents a cube with faces marked with points -/
structure Cube where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the assembled shape of cubes -/
structure AssembledShape where
  cubes : Fin 7 → Cube
  glued_pairs : Fin 9 → Fin 7 × Fin 6 × Fin 7 × Fin 6
  glued_pairs_same_points : ∀ i : Fin 9,
    let (c1, f1, c2, f2) := glued_pairs i
    (cubes c1).faces f1 = (cubes c2).faces f2

/-- The total number of points on the surface of the assembled shape -/
def surface_points (shape : AssembledShape) : Nat :=
  sorry

/-- Theorem stating that the total number of points on the surface is 75 -/
theorem surface_points_is_75 (shape : AssembledShape) :
  surface_points shape = 75 := by
  sorry

end NUMINAMATH_CALUDE_surface_points_is_75_l2448_244802


namespace NUMINAMATH_CALUDE_temporary_employee_percentage_is_32_l2448_244820

/-- Represents the composition of workers in a factory -/
structure WorkforceComposition where
  technician_ratio : ℝ
  non_technician_ratio : ℝ
  technician_permanent_ratio : ℝ
  non_technician_permanent_ratio : ℝ

/-- Calculates the percentage of temporary employees given a workforce composition -/
def temporary_employee_percentage (wc : WorkforceComposition) : ℝ :=
  100 - (wc.technician_ratio * wc.technician_permanent_ratio + 
         wc.non_technician_ratio * wc.non_technician_permanent_ratio)

/-- The main theorem stating the percentage of temporary employees -/
theorem temporary_employee_percentage_is_32 (wc : WorkforceComposition) 
  (h1 : wc.technician_ratio = 80)
  (h2 : wc.non_technician_ratio = 20)
  (h3 : wc.technician_permanent_ratio = 80)
  (h4 : wc.non_technician_permanent_ratio = 20)
  (h5 : wc.technician_ratio + wc.non_technician_ratio = 100) :
  temporary_employee_percentage wc = 32 := by
  sorry

#eval temporary_employee_percentage ⟨80, 20, 80, 20⟩

end NUMINAMATH_CALUDE_temporary_employee_percentage_is_32_l2448_244820


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2448_244818

/-- A geometric sequence with the given first three terms has its fourth term equal to -24 -/
theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (x : ℝ) 
  (h1 : a 1 = x)
  (h2 : a 2 = 3*x + 3)
  (h3 : a 3 = 6*x + 6)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n+1) / a n = a 2 / a 1) :
  a 4 = -24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2448_244818


namespace NUMINAMATH_CALUDE_inequality_proof_l2448_244851

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ x*y*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2448_244851


namespace NUMINAMATH_CALUDE_unique_determinable_score_l2448_244831

/-- The AHSME scoring system and constraints -/
structure AHSME where
  total_questions : ℕ
  score : ℕ
  correct : ℕ
  wrong : ℕ
  score_formula : score = 30 + 4 * correct - wrong
  total_answered : correct + wrong ≤ total_questions

/-- The uniqueness of the score for determining correct answers -/
def is_unique_determinable_score (s : ℕ) : Prop :=
  s > 80 ∧
  ∃! (exam : AHSME),
    exam.total_questions = 30 ∧
    exam.score = s ∧
    ∀ (s' : ℕ), 80 < s' ∧ s' < s →
      ¬∃! (exam' : AHSME),
        exam'.total_questions = 30 ∧
        exam'.score = s'

/-- The theorem stating that 119 is the unique score that satisfies the conditions -/
theorem unique_determinable_score :
  is_unique_determinable_score 119 :=
sorry

end NUMINAMATH_CALUDE_unique_determinable_score_l2448_244831


namespace NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l2448_244868

-- Define a triangle ABC
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the property of being an isosceles triangle
def isIsosceles (t : Triangle) : Prop := sorry

-- Define the property of having two equal interior angles
def hasTwoEqualAngles (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_angles (t : Triangle) :
  (¬ isIsosceles t → ¬ hasTwoEqualAngles t) ↔
  (hasTwoEqualAngles t → isIsosceles t) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l2448_244868


namespace NUMINAMATH_CALUDE_unique_solution_l2448_244865

def original_number : Nat := 20222023

theorem unique_solution (n : Nat) :
  (n ≥ 1000000000 ∧ n < 10000000000) ∧  -- 10-digit number
  (∃ (a b : Nat), n = a * 1000000000 + original_number * 10 + b) ∧  -- Formed by adding digits to left and right
  (n % 72 = 0) →  -- Divisible by 72
  n = 3202220232 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2448_244865


namespace NUMINAMATH_CALUDE_square_difference_l2448_244861

theorem square_difference (x y k c : ℝ) 
  (h1 : x * y = k) 
  (h2 : 1 / x^2 + 1 / y^2 = c) : 
  (x - y)^2 = c * k^2 - 2 * k := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2448_244861


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_points_l2448_244894

/-- Quadratic function f(x) = 3x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 4 * x + c

/-- Circle equation: x^2 + y^2 + Dx + Ey + F = 0 -/
def circle_equation (D E F x y : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

/-- Theorem: The circle passing through the intersection points of f(x) with the axes
    also passes through the fixed points (0, 1/3) and (4/3, 1/3) -/
theorem circle_passes_through_fixed_points (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 4/3) : 
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, f c x = 0 ∧ y = 0 → circle_equation D E F x y) ∧ 
    (∀ x y : ℝ, x = 0 ∧ f c 0 = y → circle_equation D E F x y) ∧
    circle_equation D E F 0 (1/3) ∧ 
    circle_equation D E F (4/3) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_points_l2448_244894


namespace NUMINAMATH_CALUDE_bisecting_line_value_l2448_244812

/-- The equation of a line that bisects the circumference of a circle. -/
def bisecting_line (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

/-- The equation of the circle. -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 2*y + 8 = 0

/-- Theorem stating that if the line y = x + b bisects the circumference of the given circle,
    then b = -5. -/
theorem bisecting_line_value (b : ℝ) :
  (∀ x y : ℝ, bisecting_line b x y ∧ circle_equation x y → 
    ∃ c_x c_y : ℝ, c_x^2 + c_y^2 - 8*c_x + 2*c_y + 8 = 0 ∧ bisecting_line b c_x c_y) →
  b = -5 :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_value_l2448_244812


namespace NUMINAMATH_CALUDE_total_students_calculation_l2448_244840

/-- Represents a high school with three years of students -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Represents a sample taken from the high school -/
structure Sample where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- The theorem stating the conditions and the conclusion about the total number of students -/
theorem total_students_calculation (school : HighSchool) (sample : Sample) :
  school.second_year = 300 →
  sample.first_year = 20 →
  sample.third_year = 10 →
  sample.first_year + sample.second_year + sample.third_year = 45 →
  (sample.first_year : ℚ) / sample.third_year = 2 →
  (sample.first_year : ℚ) / school.first_year = 
    (sample.second_year : ℚ) / school.second_year →
  (sample.second_year : ℚ) / school.second_year = 
    (sample.third_year : ℚ) / school.third_year →
  school.first_year + school.second_year + school.third_year = 900 := by
  sorry


end NUMINAMATH_CALUDE_total_students_calculation_l2448_244840


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l2448_244888

def total_players : ℕ := 15
def lineup_size : ℕ := 6
def pre_selected_players : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (total_players - pre_selected_players) (lineup_size - pre_selected_players) = 715 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l2448_244888


namespace NUMINAMATH_CALUDE_triangle_operation_result_l2448_244886

-- Define the triangle operation
def triangle (a b : ℝ) : ℝ := a^2 - 2*b

-- Theorem statement
theorem triangle_operation_result : triangle (-2) (triangle 3 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_operation_result_l2448_244886


namespace NUMINAMATH_CALUDE_average_score_is_8_1_l2448_244828

theorem average_score_is_8_1 (shooters_7 shooters_8 shooters_9 shooters_10 : ℕ)
  (h1 : shooters_7 = 4)
  (h2 : shooters_8 = 2)
  (h3 : shooters_9 = 3)
  (h4 : shooters_10 = 1) :
  let total_points := 7 * shooters_7 + 8 * shooters_8 + 9 * shooters_9 + 10 * shooters_10
  let total_shooters := shooters_7 + shooters_8 + shooters_9 + shooters_10
  (total_points : ℚ) / total_shooters = 81 / 10 :=
by sorry

end NUMINAMATH_CALUDE_average_score_is_8_1_l2448_244828


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2448_244867

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  4 * side = 40 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2448_244867


namespace NUMINAMATH_CALUDE_line_proof_l2448_244859

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y + 1 = 0
def line2 (x y : ℝ) : Prop := x - 3*y + 4 = 0
def line3 (x y : ℝ) : Prop := 3*x + 4*y - 7 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := y = (4/3)*x + (1/9)

-- Theorem statement
theorem line_proof :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧
    (result_line x₀ y₀) ∧
    (∀ (x y : ℝ), line3 x y → (y - y₀) = -(3/4) * (x - x₀)) :=
by sorry

end NUMINAMATH_CALUDE_line_proof_l2448_244859


namespace NUMINAMATH_CALUDE_irregular_shape_area_l2448_244873

/-- The area of an irregular shape consisting of a rectangle connected to a semi-circle -/
theorem irregular_shape_area (square_area : ℝ) (rect_length : ℝ) : 
  square_area = 2025 →
  rect_length = 10 →
  let circle_radius := Real.sqrt square_area
  let rect_breadth := (3 / 5) * circle_radius
  let rect_area := rect_length * rect_breadth
  let semicircle_area := (1 / 2) * Real.pi * circle_radius ^ 2
  rect_area + semicircle_area = 270 + 1012.5 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_irregular_shape_area_l2448_244873


namespace NUMINAMATH_CALUDE_simplify_expression_l2448_244808

theorem simplify_expression (x : ℝ) : 2 * x^5 * (3 * x^9) = 6 * x^14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2448_244808


namespace NUMINAMATH_CALUDE_pet_shop_guinea_pigs_l2448_244843

/-- Given a pet shop with rabbits and guinea pigs, prove the number of guinea pigs. -/
theorem pet_shop_guinea_pigs (rabbit_count : ℕ) (ratio_rabbits : ℕ) (ratio_guinea_pigs : ℕ) 
  (h_ratio : ratio_rabbits = 5 ∧ ratio_guinea_pigs = 4)
  (h_rabbits : rabbit_count = 25) :
  (rabbit_count * ratio_guinea_pigs) / ratio_rabbits = 20 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_guinea_pigs_l2448_244843


namespace NUMINAMATH_CALUDE_square_of_95_l2448_244889

theorem square_of_95 : 95^2 = 9025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_95_l2448_244889


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2448_244834

/-- Simplification of polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  (15 * x^12 + 8 * x^9 + 5 * x^7) + (3 * x^13 + 2 * x^12 + x^11 + 6 * x^9 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9) =
  3 * x^13 + 17 * x^12 + x^11 + 14 * x^9 + 8 * x^7 + 4 * x^4 + 6 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2448_244834


namespace NUMINAMATH_CALUDE_odd_function_properties_l2448_244842

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_shift : ∀ x, f (x - 2) = -f x) : 
  f 2 = 0 ∧ has_period f 4 ∧ ∀ x, f (x + 2) = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2448_244842


namespace NUMINAMATH_CALUDE_min_ear_sightings_l2448_244897

/-- Represents the direction a child is facing -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the grid of children -/
def Grid (n : Nat) := Position → Direction

/-- Counts the number of children seeing an ear in the given grid -/
def countEarSightings (n : Nat) (grid : Grid n) : Nat :=
  sorry

/-- Theorem stating the minimal number of children seeing an ear -/
theorem min_ear_sightings (n : Nat) :
  (∃ (grid : Grid n), countEarSightings n grid = n + 2) ∧
  (∀ (grid : Grid n), countEarSightings n grid ≥ n + 2) :=
sorry

end NUMINAMATH_CALUDE_min_ear_sightings_l2448_244897


namespace NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l2448_244821

theorem five_consecutive_integers_product_not_square (a : ℕ+) :
  ∃ (n : ℕ), (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) : ℕ) ≠ n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l2448_244821


namespace NUMINAMATH_CALUDE_vector_relations_l2448_244846

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define parallelism
def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

-- Define collinearity (same as parallelism for vectors)
def collinear (v w : V) : Prop := parallel v w

-- Theorem: Only "Equal vectors must be collinear" is true
theorem vector_relations :
  (∀ v w : V, parallel v w → v = w) = false ∧ 
  (∀ v w : V, v ≠ w → ¬(parallel v w)) = false ∧
  (∀ v w : V, collinear v w → v = w) = false ∧
  (∀ v w : V, v = w → collinear v w) = true :=
sorry

end NUMINAMATH_CALUDE_vector_relations_l2448_244846


namespace NUMINAMATH_CALUDE_H_upper_bound_l2448_244800

open Real

noncomputable def f (x : ℝ) : ℝ := x + log x

noncomputable def H (x m : ℝ) : ℝ := f x - log (exp x - 1)

theorem H_upper_bound {m : ℝ} (hm : m > 0) :
  ∀ x, 0 < x → x < m → H x m < m / 2 := by sorry

end NUMINAMATH_CALUDE_H_upper_bound_l2448_244800


namespace NUMINAMATH_CALUDE_product_in_base_10_l2448_244876

-- Define the binary number 11001₂
def binary_num : ℕ := 25

-- Define the ternary number 112₃
def ternary_num : ℕ := 14

-- Theorem to prove
theorem product_in_base_10 : binary_num * ternary_num = 350 := by
  sorry

end NUMINAMATH_CALUDE_product_in_base_10_l2448_244876


namespace NUMINAMATH_CALUDE_book_price_change_l2448_244810

theorem book_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.3)
  let final_price := price_after_decrease * (1 + 0.2)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = -16 := by
sorry

end NUMINAMATH_CALUDE_book_price_change_l2448_244810


namespace NUMINAMATH_CALUDE_basketball_handshakes_l2448_244841

/-- The number of handshakes in a basketball game scenario --/
def total_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let player_handshakes := team_size * team_size
  let referee_handshakes := (team_size * num_teams) * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the given scenario --/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l2448_244841


namespace NUMINAMATH_CALUDE_rearrangement_count_correct_l2448_244825

/-- The number of ways to rearrange 3 out of 8 people in a row, 
    while keeping the other 5 in their original positions. -/
def rearrangement_count : ℕ := Nat.choose 8 3 * 2

/-- Theorem stating that the number of rearrangements is correct. -/
theorem rearrangement_count_correct : 
  rearrangement_count = Nat.choose 8 3 * 2 := by sorry

end NUMINAMATH_CALUDE_rearrangement_count_correct_l2448_244825


namespace NUMINAMATH_CALUDE_base9_734_equals_base10_598_l2448_244811

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₀ * 9^0 + d₁ * 9^1 + d₂ * 9^2

/-- Theorem: The base-9 number 734 is equal to 598 in base-10 --/
theorem base9_734_equals_base10_598 : base9ToBase10 7 3 4 = 598 := by
  sorry

#eval base9ToBase10 7 3 4

end NUMINAMATH_CALUDE_base9_734_equals_base10_598_l2448_244811


namespace NUMINAMATH_CALUDE_weekend_weather_probability_l2448_244844

/-- The probability of rain on each day -/
def rain_prob : ℝ := 0.75

/-- The number of days in the weekend -/
def num_days : ℕ := 3

/-- The number of desired sunny days -/
def desired_sunny_days : ℕ := 2

/-- Theorem: The probability of having exactly two sunny days and one rainy day
    during a three-day period, where the probability of rain each day is 0.75,
    is equal to 27/64 -/
theorem weekend_weather_probability :
  (Nat.choose num_days desired_sunny_days : ℝ) *
  (1 - rain_prob) ^ desired_sunny_days *
  rain_prob ^ (num_days - desired_sunny_days) =
  27 / 64 := by sorry

end NUMINAMATH_CALUDE_weekend_weather_probability_l2448_244844


namespace NUMINAMATH_CALUDE_negation_of_cubic_inequality_l2448_244854

theorem negation_of_cubic_inequality :
  (¬ (∀ x : ℝ, x^3 - x ≥ 0)) ↔ (∃ x : ℝ, x^3 - x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_cubic_inequality_l2448_244854
