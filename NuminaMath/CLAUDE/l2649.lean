import Mathlib

namespace NUMINAMATH_CALUDE_gcd_459_357_l2649_264905

theorem gcd_459_357 : Int.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2649_264905


namespace NUMINAMATH_CALUDE_four_similar_triangle_solutions_l2649_264945

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Define a line
structure Line :=
  (m b : ℝ)

-- Function to check if a point is on a side of a triangle
def isPointOnSide (T : Triangle) (P : Point) : Prop :=
  sorry

-- Function to check if two triangles are similar
def areSimilarTriangles (T1 T2 : Triangle) : Prop :=
  sorry

-- Function to check if a line intersects a triangle
def lineIntersectsTriangle (L : Line) (T : Triangle) : Prop :=
  sorry

-- Function to get the triangle cut off by a line
def getCutOffTriangle (T : Triangle) (L : Line) : Triangle :=
  sorry

-- The main theorem
theorem four_similar_triangle_solutions 
  (T : Triangle) (P : Point) (h : isPointOnSide T P) :
  ∃ (L1 L2 L3 L4 : Line),
    (L1 ≠ L2 ∧ L1 ≠ L3 ∧ L1 ≠ L4 ∧ L2 ≠ L3 ∧ L2 ≠ L4 ∧ L3 ≠ L4) ∧
    (∀ (L : Line), 
      (lineIntersectsTriangle L T ∧ areSimilarTriangles (getCutOffTriangle T L) T) →
      (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4)) :=
sorry

end NUMINAMATH_CALUDE_four_similar_triangle_solutions_l2649_264945


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l2649_264928

/-- Represents the number of trees planted by each grade --/
structure TreePlanting where
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- The conditions of the tree planting activity --/
def treePlantingConditions (t : TreePlanting) : Prop :=
  t.fourth = 30 ∧
  t.sixth = 3 * t.fifth - 30 ∧
  t.fourth + t.fifth + t.sixth = 240

/-- The theorem stating the ratio of trees planted by 5th graders to 4th graders --/
theorem tree_planting_ratio (t : TreePlanting) :
  treePlantingConditions t → (t.fifth : ℚ) / t.fourth = 2 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_ratio_l2649_264928


namespace NUMINAMATH_CALUDE_eggs_ratio_is_one_to_one_l2649_264990

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the total number of eggs Megan initially had --/
def initial_eggs : ℕ := 2 * dozen

/-- Represents the number of eggs Megan used for cooking --/
def used_eggs : ℕ := 2 + 4

/-- Represents the number of eggs Megan plans to use for her meals --/
def planned_meals_eggs : ℕ := 3 * 3

/-- Theorem stating that the ratio of eggs Megan gave to her aunt to the eggs she kept for herself is 1:1 --/
theorem eggs_ratio_is_one_to_one : 
  (initial_eggs - used_eggs - planned_meals_eggs) = planned_meals_eggs := by
  sorry

end NUMINAMATH_CALUDE_eggs_ratio_is_one_to_one_l2649_264990


namespace NUMINAMATH_CALUDE_rectangular_field_length_l2649_264986

/-- Represents a rectangular field with a given width and area. -/
structure RectangularField where
  width : ℝ
  area : ℝ

/-- The length of a rectangular field is 10 meters more than its width. -/
def length (field : RectangularField) : ℝ := field.width + 10

/-- The theorem stating that a rectangular field with an area of 171 square meters
    and length 10 meters more than its width has a length of 19 meters. -/
theorem rectangular_field_length (field : RectangularField) 
  (h1 : field.area = 171)
  (h2 : field.area = field.width * (field.width + 10)) :
  length field = 19 := by
  sorry

#check rectangular_field_length

end NUMINAMATH_CALUDE_rectangular_field_length_l2649_264986


namespace NUMINAMATH_CALUDE_X_4_equivalence_l2649_264985

-- Define the type for a die
def Die : Type := Fin 6

-- Define the type for a pair of dice
def DicePair : Type := Die × Die

-- Define the sum of points on a pair of dice
def sum_points (pair : DicePair) : Nat :=
  pair.1.val + 1 + pair.2.val + 1

-- Define the event X = 4
def X_equals_4 (pair : DicePair) : Prop :=
  sum_points pair = 4

-- Define the event where one die shows 3 and the other shows 1
def one_3_one_1 (pair : DicePair) : Prop :=
  (pair.1.val = 2 ∧ pair.2.val = 0) ∨ (pair.1.val = 0 ∧ pair.2.val = 2)

-- Define the event where both dice show 2
def both_2 (pair : DicePair) : Prop :=
  pair.1.val = 1 ∧ pair.2.val = 1

-- Theorem: X = 4 is equivalent to (one 3 and one 1) or (both 2)
theorem X_4_equivalence (pair : DicePair) :
  X_equals_4 pair ↔ one_3_one_1 pair ∨ both_2 pair :=
sorry

end NUMINAMATH_CALUDE_X_4_equivalence_l2649_264985


namespace NUMINAMATH_CALUDE_fruit_stand_average_price_l2649_264968

theorem fruit_stand_average_price (apple_price orange_price : ℚ)
  (total_fruits : ℕ) (oranges_removed : ℕ) (kept_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : oranges_removed = 4)
  (h5 : kept_avg_price = 50/100) :
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = 54/100 :=
by sorry

end NUMINAMATH_CALUDE_fruit_stand_average_price_l2649_264968


namespace NUMINAMATH_CALUDE_parallelogram_area_18_16_l2649_264925

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 16 cm is 288 square centimeters -/
theorem parallelogram_area_18_16 : parallelogram_area 18 16 = 288 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_16_l2649_264925


namespace NUMINAMATH_CALUDE_soccer_league_games_l2649_264995

/-- The number of games played in a soccer league where each team plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 10 teams, where each team plays every other team once, 
    the total number of games played is 45 -/
theorem soccer_league_games : num_games 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l2649_264995


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l2649_264977

theorem fraction_sum_product_equality (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : a / b + c / d = (a / b) * (c / d)) : 
  b / a + d / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l2649_264977


namespace NUMINAMATH_CALUDE_equation_equivalence_l2649_264949

theorem equation_equivalence : ∀ x : ℝ, (2 * (x + 1) = x + 7) ↔ (x = 5) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2649_264949


namespace NUMINAMATH_CALUDE_max_value_of_fraction_difference_l2649_264923

theorem max_value_of_fraction_difference (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 4 * a - b ≥ 2) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y ≤ 1 / 2) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y = 1 / 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_difference_l2649_264923


namespace NUMINAMATH_CALUDE_wades_total_spend_l2649_264936

/-- Wade's purchase at a rest stop -/
def wades_purchase : ℕ → ℕ → ℕ → ℕ → ℕ := fun num_sandwiches sandwich_price num_drinks drink_price =>
  num_sandwiches * sandwich_price + num_drinks * drink_price

theorem wades_total_spend :
  wades_purchase 3 6 2 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_wades_total_spend_l2649_264936


namespace NUMINAMATH_CALUDE_expand_expression_l2649_264961

theorem expand_expression (x : ℝ) : 5 * (-3 * x^3 + 4 * x^2 - 2 * x + 7) = -15 * x^3 + 20 * x^2 - 10 * x + 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2649_264961


namespace NUMINAMATH_CALUDE_min_value_at_seven_l2649_264971

/-- The quadratic function f(x) = x^2 - 14x + 40 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 40

theorem min_value_at_seven :
  ∀ x : ℝ, f 7 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_at_seven_l2649_264971


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2649_264922

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 4 ∧ 
  n % 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2649_264922


namespace NUMINAMATH_CALUDE_binomial_variance_example_l2649_264964

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial random variable with n=10 and p=1/4 is 15/8 -/
theorem binomial_variance_example : 
  ∀ ξ : BinomialRV, ξ.n = 10 ∧ ξ.p = 1/4 → variance ξ = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l2649_264964


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2649_264910

/-- The area of a square inscribed in the ellipse (x²/4) + (y²/8) = 1, 
    with its sides parallel to the coordinate axes, is equal to 32/3. -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 4 + y^2 / 8 = 1) →
  (∃ s : ℝ, s > 0 ∧ x^2 ≤ s^2 ∧ y^2 ≤ s^2) →
  (4 * s^2 = 32 / 3) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2649_264910


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2649_264951

/-- Prove that the given expression equals 1/15 -/
theorem problem_1 : 
  (2 * (Nat.factorial 8 / Nat.factorial 3) + 7 * (Nat.factorial 8 / Nat.factorial 4)) / 
  (Nat.factorial 8 - Nat.factorial 9 / Nat.factorial 4) = 1 / 15 := by
  sorry

/-- Prove that the sum of combinations equals C(202, 4) -/
theorem problem_2 : 
  Nat.choose 200 198 + Nat.choose 200 196 + 2 * Nat.choose 200 197 = Nat.choose 202 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2649_264951


namespace NUMINAMATH_CALUDE_tire_price_proof_l2649_264927

/-- The regular price of a single tire -/
def regular_price : ℚ := 295 / 3

/-- The price of the fourth tire under the offer -/
def fourth_tire_price : ℚ := 5

/-- The total discount applied to the purchase -/
def total_discount : ℚ := 10

/-- The total amount Jane paid for four tires -/
def total_paid : ℚ := 290

/-- Theorem stating that the regular price of a tire is 295/3 given the sale conditions -/
theorem tire_price_proof :
  3 * regular_price + fourth_tire_price - total_discount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l2649_264927


namespace NUMINAMATH_CALUDE_symmetric_function_periodic_l2649_264916

/-- A function f: ℝ → ℝ is symmetric with respect to the point (a, y₀) if for all x, f(a + x) - y₀ = y₀ - f(a - x) -/
def SymmetricPoint (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f (a + x) - y₀ = y₀ - f (a - x)

/-- A function f: ℝ → ℝ is symmetric with respect to the line x = b if for all x, f(b + x) = f(b - x) -/
def SymmetricLine (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (b + x) = f (b - x)

/-- The main theorem: if f is symmetric with respect to a point (a, y₀) and a line x = b where b > a,
    then f is periodic with period 4(b-a) -/
theorem symmetric_function_periodic (f : ℝ → ℝ) (a b y₀ : ℝ) 
    (h_point : SymmetricPoint f a y₀) (h_line : SymmetricLine f b) (h_order : b > a) :
    ∀ x, f (x + 4*(b - a)) = f x := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_periodic_l2649_264916


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2649_264938

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2649_264938


namespace NUMINAMATH_CALUDE_officers_count_l2649_264940

/-- The number of ways to choose 4 distinct officers from a group of 15 people -/
def choose_officers (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3)

/-- Theorem: There are 32,760 ways to choose 4 distinct officers from a group of 15 people -/
theorem officers_count : choose_officers 15 = 32760 := by
  sorry

end NUMINAMATH_CALUDE_officers_count_l2649_264940


namespace NUMINAMATH_CALUDE_bike_price_calculation_l2649_264981

theorem bike_price_calculation (current_price : ℝ) : 
  (current_price * 1.1 = 82500) → current_price = 75000 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_calculation_l2649_264981


namespace NUMINAMATH_CALUDE_laps_run_l2649_264909

/-- Proves the number of laps run given total distance, track length, and remaining laps -/
theorem laps_run (total_distance : ℕ) (track_length : ℕ) (remaining_laps : ℕ) 
  (h1 : total_distance = 2400)
  (h2 : track_length = 150)
  (h3 : remaining_laps = 4) :
  ∃ (x : ℕ), x * track_length + remaining_laps * track_length = total_distance ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_laps_run_l2649_264909


namespace NUMINAMATH_CALUDE_race_distance_l2649_264966

/-- The race problem -/
theorem race_distance (a_time b_time : ℝ) (lead_distance : ℝ) (race_distance : ℝ) : 
  a_time = 36 →
  b_time = 45 →
  lead_distance = 30 →
  (race_distance / a_time) * b_time = race_distance + lead_distance →
  race_distance = 120 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l2649_264966


namespace NUMINAMATH_CALUDE_carson_roller_coaster_rides_l2649_264943

/-- Represents the carnival problem with given wait times and ride frequencies. -/
def carnival_problem (total_time roller_coaster_wait tilt_a_whirl_wait giant_slide_wait : ℕ)
  (tilt_a_whirl_rides giant_slide_rides : ℕ) : Prop :=
  ∃ (roller_coaster_rides : ℕ),
    roller_coaster_rides * roller_coaster_wait +
    tilt_a_whirl_rides * tilt_a_whirl_wait +
    giant_slide_rides * giant_slide_wait = total_time

/-- Theorem stating that Carson rides the roller coaster 4 times. -/
theorem carson_roller_coaster_rides :
  carnival_problem (4 * 60) 30 60 15 1 4 →
  ∃ (roller_coaster_rides : ℕ), roller_coaster_rides = 4 := by
  sorry


end NUMINAMATH_CALUDE_carson_roller_coaster_rides_l2649_264943


namespace NUMINAMATH_CALUDE_smallest_special_number_l2649_264911

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_small_prime_factor (n : ℕ) : Prop := ∀ p : ℕ, is_prime p → p < 100 → ¬(n % p = 0)

theorem smallest_special_number : 
  (∀ m : ℕ, m < 10403 → (is_prime m ∨ is_square m ∨ ¬(has_no_small_prime_factor m))) ∧
  ¬(is_prime 10403) ∧ 
  ¬(is_square 10403) ∧ 
  has_no_small_prime_factor 10403 :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l2649_264911


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l2649_264919

theorem min_value_arithmetic_sequence (a : ℝ) (m : ℕ+) :
  (∃ (S : ℕ+ → ℝ), S m = 36 ∧ 
    (∀ n : ℕ+, S n = n * a - 4 * (n * (n - 1)) / 2)) →
  ∀ a' : ℝ, (∃ m' : ℕ+, ∃ S' : ℕ+ → ℝ, 
    S' m' = 36 ∧ 
    (∀ n : ℕ+, S' n = n * a' - 4 * (n * (n - 1)) / 2)) →
  a' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l2649_264919


namespace NUMINAMATH_CALUDE_probability_theorem_l2649_264982

def standard_dice : ℕ := 6

def roll_count : ℕ := 4

def probability_at_least_three_distinct_with_six : ℚ :=
  360 / (standard_dice ^ roll_count)

theorem probability_theorem :
  probability_at_least_three_distinct_with_six = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2649_264982


namespace NUMINAMATH_CALUDE_value_of_expression_l2649_264937

theorem value_of_expression (a b : ℝ) (h : 2 * a - b = -1) : 
  4 * a - 2 * b + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l2649_264937


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_l2649_264970

theorem not_arithmetic_sequence : ¬∃ (m n k : ℤ) (a d : ℝ), 
  m < n ∧ n < k ∧ 
  1 = a + (m - 1) * d ∧ 
  Real.sqrt 3 = a + (n - 1) * d ∧ 
  2 = a + (k - 1) * d :=
sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_l2649_264970


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_inequality_l2649_264924

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 2*a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} := by sorry

-- Part II
theorem range_of_a_given_inequality (a : ℝ) :
  (∀ x : ℝ, f a x ≥ a^2 - 3*a - 3) → a ∈ Set.Icc (-1) (2 + Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_inequality_l2649_264924


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l2649_264918

theorem f_monotone_increasing (k : ℝ) (h_k : k ≥ 0) :
  ∀ x ≥ Real.sqrt (2 * k + 1), HasDerivAt (λ x => x + (2 * k + 1) / x) ((x^2 - (2 * k + 1)) / x^2) x ∧
  (x^2 - (2 * k + 1)) / x^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l2649_264918


namespace NUMINAMATH_CALUDE_inequality_proof_l2649_264988

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2649_264988


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l2649_264900

theorem dolphin_training_hours 
  (num_dolphins : ℕ) 
  (hours_per_dolphin : ℕ) 
  (num_trainers : ℕ) 
  (h1 : num_dolphins = 4)
  (h2 : hours_per_dolphin = 3)
  (h3 : num_trainers = 2)
  : (num_dolphins * hours_per_dolphin) / num_trainers = 6 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l2649_264900


namespace NUMINAMATH_CALUDE_integer_solutions_x4_minus_2y2_eq_1_l2649_264963

theorem integer_solutions_x4_minus_2y2_eq_1 :
  ∀ x y : ℤ, x^4 - 2*y^2 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_x4_minus_2y2_eq_1_l2649_264963


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l2649_264999

theorem triangle_angle_ratio (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle
  A = 20 →           -- Smallest angle
  B = 3 * A →        -- Middle angle is 3 times the smallest
  A ≤ B →            -- B is larger than or equal to A
  B ≤ C →            -- C is the largest angle
  C / A = 5 :=       -- Ratio of largest to smallest is 5:1
by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l2649_264999


namespace NUMINAMATH_CALUDE_square_area_ratio_l2649_264956

theorem square_area_ratio (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := s^3 * Real.pi^(1/3)
  let new_area := new_side^2
  original_area / new_area = s^4 * Real.pi^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2649_264956


namespace NUMINAMATH_CALUDE_solve_linear_system_l2649_264976

theorem solve_linear_system (b : ℚ) : 
  (∃ x y : ℚ, x + b * y = 0 ∧ x + y = -1 ∧ x = 1) →
  b = 1/2 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_system_l2649_264976


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2649_264962

theorem quadratic_roots_sum (m n p : ℤ) : 
  (∃ x : ℝ, 4 * x * (2 * x - 5) = -4) →
  (∃ x : ℝ, x = (m + Real.sqrt n : ℝ) / p ∧ 4 * x * (2 * x - 5) = -4) →
  (∃ x : ℝ, x = (m - Real.sqrt n : ℝ) / p ∧ 4 * x * (2 * x - 5) = -4) →
  m + n + p = 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2649_264962


namespace NUMINAMATH_CALUDE_math_team_selection_l2649_264983

theorem math_team_selection (boys girls : ℕ) (h1 : boys = 7) (h2 : girls = 10) :
  (boys.choose 4) * (girls.choose 2) = 1575 :=
by sorry

end NUMINAMATH_CALUDE_math_team_selection_l2649_264983


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l2649_264953

/-- Given a cubic equation and conditions on a polynomial P,
    prove that P has a specific form. -/
theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) :
  (a^3 + 5*a^2 + 8*a + 13 = 0) →
  (b^3 + 5*b^2 + 8*b + 13 = 0) →
  (c^3 + 5*c^2 + 8*c + 13 = 0) →
  (∀ x, ∃ p q r s, P x = p*x^3 + q*x^2 + r*x + s) →
  (P a = b + c + 2) →
  (P b = a + c + 2) →
  (P c = a + b + 2) →
  (P (a + b + c) = -22) →
  (∀ x, P x = (19*x^3 + 95*x^2 + 152*x + 247) / 52 - x - 3) :=
by sorry


end NUMINAMATH_CALUDE_cubic_polynomial_problem_l2649_264953


namespace NUMINAMATH_CALUDE_least_tiles_cover_room_l2649_264974

def room_length : ℕ := 624
def room_width : ℕ := 432

theorem least_tiles_cover_room (length : ℕ) (width : ℕ) 
  (h1 : length = room_length) (h2 : width = room_width) : 
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧ 
    length % tile_size = 0 ∧ 
    width % tile_size = 0 ∧ 
    (length / tile_size) * (width / tile_size) = 117 ∧
    ∀ (other_size : ℕ), 
      other_size > 0 → 
      length % other_size = 0 → 
      width % other_size = 0 → 
      other_size ≤ tile_size :=
by sorry

end NUMINAMATH_CALUDE_least_tiles_cover_room_l2649_264974


namespace NUMINAMATH_CALUDE_probability_two_forks_one_spoon_one_knife_l2649_264912

/-- The number of forks in the drawer -/
def num_forks : ℕ := 8

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 5

/-- The number of knives in the drawer -/
def num_knives : ℕ := 7

/-- The total number of pieces of silverware -/
def total_silverware : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be drawn -/
def num_drawn : ℕ := 4

/-- The probability of drawing 2 forks, 1 spoon, and 1 knife -/
theorem probability_two_forks_one_spoon_one_knife :
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1 : ℚ) /
  (Nat.choose total_silverware num_drawn : ℚ) = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_forks_one_spoon_one_knife_l2649_264912


namespace NUMINAMATH_CALUDE_or_not_implies_q_l2649_264941

theorem or_not_implies_q (p q : Prop) : (p ∨ q) → ¬p → q := by
  sorry

end NUMINAMATH_CALUDE_or_not_implies_q_l2649_264941


namespace NUMINAMATH_CALUDE_equation_value_l2649_264992

theorem equation_value (x y : ℝ) (h : 2*x - y = -1) : 3 + 4*x - 2*y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l2649_264992


namespace NUMINAMATH_CALUDE_cashew_price_l2649_264948

/-- Proves the price of cashew nuts given the conditions of the problem -/
theorem cashew_price (peanut_price : ℕ) (cashew_amount peanut_amount total_amount : ℕ) (total_price : ℕ) :
  peanut_price = 130 →
  cashew_amount = 3 →
  peanut_amount = 2 →
  total_amount = 5 →
  total_price = 178 →
  cashew_amount * (total_price * total_amount - peanut_price * peanut_amount) / cashew_amount = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_cashew_price_l2649_264948


namespace NUMINAMATH_CALUDE_managers_salary_l2649_264939

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) : 
  num_employees = 18 →
  avg_salary = 2000 →
  salary_increase = 200 →
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 5800 := by
sorry

end NUMINAMATH_CALUDE_managers_salary_l2649_264939


namespace NUMINAMATH_CALUDE_tangent_circle_radii_product_l2649_264907

/-- A circle passing through (3,4) and tangent to both axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_point : (center.1 - 3)^2 + (center.2 - 4)^2 = radius^2
  tangent_to_x_axis : center.2 = radius
  tangent_to_y_axis : center.1 = radius

/-- The two possible radii of tangent circles -/
def radii : ℝ × ℝ :=
  let a := TangentCircle.radius
  let equation := a^2 - 14*a + 25 = 0
  sorry

theorem tangent_circle_radii_product :
  let (r₁, r₂) := radii
  r₁ * r₂ = 25 := by sorry

end NUMINAMATH_CALUDE_tangent_circle_radii_product_l2649_264907


namespace NUMINAMATH_CALUDE_simplify_expression_l2649_264952

theorem simplify_expression (a b c d x : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  ((x + a)^4) / ((a - b)*(a - c)*(a - d)) + 
  ((x + b)^4) / ((b - a)*(b - c)*(b - d)) + 
  ((x + c)^4) / ((c - a)*(c - b)*(c - d)) + 
  ((x + d)^4) / ((d - a)*(d - b)*(d - c)) = 
  a + b + c + d + 4*x := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2649_264952


namespace NUMINAMATH_CALUDE_abs_value_inequality_iff_l2649_264906

theorem abs_value_inequality_iff (a b : ℝ) : a * |a| > b * |b| ↔ a > b := by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_iff_l2649_264906


namespace NUMINAMATH_CALUDE_inscribed_triangle_circumscribed_square_l2649_264993

theorem inscribed_triangle_circumscribed_square (r : ℝ) : 
  r > 0 → 
  let triangle_side := r * Real.sqrt 3
  let triangle_perimeter := 3 * triangle_side
  let square_side := r * Real.sqrt 2
  let square_area := square_side ^ 2
  triangle_perimeter = square_area →
  r = 3 * Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_circumscribed_square_l2649_264993


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2649_264975

/-- The range of b values for which the line y = kx + b always has two common points with the ellipse x²/9 + y²/4 = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), 
  (∀ (b : ℝ), (∃! (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + b ∧ 
    y₂ = k * x₂ + b ∧ 
    x₁^2 / 9 + y₁^2 / 4 = 1 ∧ 
    x₂^2 / 9 + y₂^2 / 4 = 1)) ↔ 
  (-2 < b ∧ b < 2) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2649_264975


namespace NUMINAMATH_CALUDE_sector_area_l2649_264954

theorem sector_area (r : ℝ) (θ : ℝ) (chord_length : ℝ) : 
  θ = 2 ∧ 
  chord_length = 2 * Real.sin 1 ∧ 
  chord_length = 2 * r * Real.sin (θ / 2) →
  (1 / 2) * r^2 * θ = 1 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l2649_264954


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l2649_264947

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ (n : ℕ), n = 888 ∧
    has_only_even_digits n ∧
    n < 1000 ∧
    n % 9 = 0 ∧
    ∀ m : ℕ, has_only_even_digits m ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l2649_264947


namespace NUMINAMATH_CALUDE_negative_two_equals_negative_abs_two_l2649_264903

theorem negative_two_equals_negative_abs_two : -2 = -|-2| := by
  sorry

end NUMINAMATH_CALUDE_negative_two_equals_negative_abs_two_l2649_264903


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l2649_264913

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.sideLength * s.sideLength

/-- The overlap configuration of two squares -/
structure SquareOverlap where
  largeSquare : Square
  smallSquare : Square
  smallSquareTouchesCenter : smallSquare.sideLength = largeSquare.sideLength / 2

/-- The area covered only by the larger square in the overlap configuration -/
def SquareOverlap.areaOnlyLarger (so : SquareOverlap) : ℝ :=
  so.largeSquare.area - so.smallSquare.area

/-- The main theorem -/
theorem overlap_area_theorem (so : SquareOverlap) 
    (h1 : so.largeSquare.sideLength = 8) 
    (h2 : so.smallSquare.sideLength = 4) : 
    so.areaOnlyLarger = 48 := by
  sorry


end NUMINAMATH_CALUDE_overlap_area_theorem_l2649_264913


namespace NUMINAMATH_CALUDE_missing_digits_sum_l2649_264960

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- The addition problem structure -/
structure AdditionProblem where
  d1 : Digit  -- First missing digit
  d2 : Digit  -- Second missing digit

/-- The addition problem is valid -/
def isValidAddition (p : AdditionProblem) : Prop :=
  708 + 10 * p.d1.val + 2182 = 86301 + 100 * p.d2.val

/-- The theorem to be proved -/
theorem missing_digits_sum (p : AdditionProblem) 
  (h : isValidAddition p) : p.d1.val + p.d2.val = 7 := by
  sorry

#check missing_digits_sum

end NUMINAMATH_CALUDE_missing_digits_sum_l2649_264960


namespace NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l2649_264942

theorem sum_of_factorization_coefficients :
  ∀ (a b c : ℤ),
  (∀ x : ℝ, x^2 + 17*x + 70 = (x + a) * (x + b)) →
  (∀ x : ℝ, x^2 - 19*x + 84 = (x - b) * (x - c)) →
  a + b + c = 29 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l2649_264942


namespace NUMINAMATH_CALUDE_min_separating_edges_l2649_264980

/-- Represents a color in the grid -/
inductive Color
| Red
| Green
| Blue

/-- Represents a cell in the grid -/
structure Cell :=
  (row : Fin 33)
  (col : Fin 33)
  (color : Color)

/-- Represents the grid -/
def Grid := Array (Array Cell)

/-- Checks if two cells are adjacent -/
def isAdjacent (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row ∧ (c1.col.val + 1 = c2.col.val ∨ c1.col.val = c2.col.val + 1)) ∨
  (c1.col = c2.col ∧ (c1.row.val + 1 = c2.row.val ∨ c1.row.val = c2.row.val + 1))

/-- Counts the number of separating edges in the grid -/
def countSeparatingEdges (grid : Grid) : Nat :=
  sorry

/-- Checks if the grid has an equal number of cells for each color -/
def hasEqualColorDistribution (grid : Grid) : Prop :=
  sorry

/-- Theorem: The minimum number of separating edges in a 33x33 grid with three equally distributed colors is 56 -/
theorem min_separating_edges (grid : Grid) 
  (h : hasEqualColorDistribution grid) : 
  countSeparatingEdges grid ≥ 56 := by
  sorry

end NUMINAMATH_CALUDE_min_separating_edges_l2649_264980


namespace NUMINAMATH_CALUDE_minimum_dimes_needed_l2649_264991

/-- The cost of the jacket in cents -/
def jacket_cost : ℕ := 4550

/-- The value of two $20 bills in cents -/
def bills_value : ℕ := 2 * 2000

/-- The value of five quarters in cents -/
def quarters_value : ℕ := 5 * 25

/-- The value of six nickels in cents -/
def nickels_value : ℕ := 6 * 5

/-- The value of one dime in cents -/
def dime_value : ℕ := 10

/-- The minimum number of dimes needed -/
def min_dimes : ℕ := 40

theorem minimum_dimes_needed :
  ∀ n : ℕ, 
    n ≥ min_dimes → 
    bills_value + quarters_value + nickels_value + n * dime_value ≥ jacket_cost ∧
    ∀ m : ℕ, m < min_dimes → 
      bills_value + quarters_value + nickels_value + m * dime_value < jacket_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_dimes_needed_l2649_264991


namespace NUMINAMATH_CALUDE_power_eight_divided_by_four_l2649_264933

theorem power_eight_divided_by_four (n : ℕ) : n = 8^2022 → n/4 = 4^3032 := by
  sorry

end NUMINAMATH_CALUDE_power_eight_divided_by_four_l2649_264933


namespace NUMINAMATH_CALUDE_complex_coordinate_proof_l2649_264987

theorem complex_coordinate_proof (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_proof_l2649_264987


namespace NUMINAMATH_CALUDE_cube_root_unity_polynomial_identity_l2649_264904

theorem cube_root_unity_polynomial_identity
  (a b c : ℂ) (n m : ℕ) :
  (∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n + 2) + b * x^(3*m + 1) + c = 0) →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_polynomial_identity_l2649_264904


namespace NUMINAMATH_CALUDE_clerical_staff_fraction_l2649_264932

theorem clerical_staff_fraction (total_employees : ℕ) (f : ℚ) : 
  total_employees = 3600 →
  (2/3 : ℚ) * (f * total_employees) = (1/4 : ℚ) * (total_employees - (1/3 : ℚ) * (f * total_employees)) →
  f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_clerical_staff_fraction_l2649_264932


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2649_264997

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 10 -/
def base9_to_base10 (n : ℕ) : ℕ := sorry

theorem base_conversion_subtraction :
  base8_to_base10 52143 - base9_to_base10 3456 = 19041 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2649_264997


namespace NUMINAMATH_CALUDE_johns_age_fraction_l2649_264934

theorem johns_age_fraction (john_age mother_age father_age : ℕ) : 
  father_age = 40 →
  father_age = mother_age + 4 →
  john_age = mother_age - 16 →
  (john_age : ℚ) / father_age = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_johns_age_fraction_l2649_264934


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l2649_264921

/-- The point that minimizes the sum of distances from two fixed points on a line --/
theorem minimize_sum_distances (A B C : ℝ × ℝ) : 
  A = (3, 6) → 
  B = (6, 2) → 
  C.2 = 0 → 
  (∀ k : ℝ, C = (k, 0) → 
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) + 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ≥ 
    Real.sqrt ((6.75 - A.1)^2 + (0 - A.2)^2) + 
    Real.sqrt ((6.75 - B.1)^2 + (0 - B.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_distances_l2649_264921


namespace NUMINAMATH_CALUDE_sheet_width_calculation_l2649_264901

/-- The width of a rectangular sheet of paper with specific properties -/
def sheet_width : ℝ := sorry

theorem sheet_width_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  abs (sheet_width - 6.6) < ε ∧
  sheet_width * 13 * 2 = 6.5 * 11 + 100 := by sorry

end NUMINAMATH_CALUDE_sheet_width_calculation_l2649_264901


namespace NUMINAMATH_CALUDE_light_glow_start_time_l2649_264967

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts a Time to total seconds -/
def Time.toSeconds (t : Time) : Nat :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Converts total seconds to a Time -/
def Time.fromSeconds (s : Nat) : Time :=
  { hours := s / 3600
  , minutes := (s % 3600) / 60
  , seconds := s % 60 }

/-- Subtracts two Times, assuming t1 ≥ t2 -/
def Time.sub (t1 t2 : Time) : Time :=
  Time.fromSeconds (t1.toSeconds - t2.toSeconds)

theorem light_glow_start_time 
  (glow_interval : Nat) 
  (glow_count : Nat) 
  (end_time : Time) : 
  glow_interval = 21 →
  glow_count = 236 →
  end_time = { hours := 3, minutes := 20, seconds := 47 } →
  Time.sub end_time (Time.fromSeconds (glow_interval * glow_count)) = 
    { hours := 1, minutes := 58, seconds := 11 } :=
by sorry

end NUMINAMATH_CALUDE_light_glow_start_time_l2649_264967


namespace NUMINAMATH_CALUDE_cubic_one_real_solution_l2649_264944

/-- The cubic equation 4x^3 + 9x^2 + kx + 4 = 0 has exactly one real solution if and only if k = 6.75 -/
theorem cubic_one_real_solution (k : ℝ) : 
  (∃! x : ℝ, 4 * x^3 + 9 * x^2 + k * x + 4 = 0) ↔ k = 27/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_one_real_solution_l2649_264944


namespace NUMINAMATH_CALUDE_power_difference_solutions_l2649_264931

theorem power_difference_solutions :
  ∀ m n : ℕ+,
  (2^(m : ℕ) - 3^(n : ℕ) = 1 ∧ 3^(n : ℕ) - 2^(m : ℕ) = 1) ↔ 
  ((m = 2 ∧ n = 1) ∨ (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_power_difference_solutions_l2649_264931


namespace NUMINAMATH_CALUDE_product_third_fourth_term_l2649_264902

/-- An arithmetic sequence with common difference 2 and eighth term 20 -/
def ArithmeticSequence (a : ℕ) : ℕ → ℕ :=
  fun n => a + (n - 1) * 2

theorem product_third_fourth_term (a : ℕ) :
  ArithmeticSequence a 8 = 20 →
  ArithmeticSequence a 3 * ArithmeticSequence a 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_third_fourth_term_l2649_264902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2649_264929

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Given an arithmetic sequence a where a₁ + a₃ + a₅ = 3, prove a₂ + a₄ = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) 
    (h2 : a 1 + a 3 + a 5 = 3) : a 2 + a 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2649_264929


namespace NUMINAMATH_CALUDE_tax_reduction_is_sixty_percent_tax_reduction_for_thousand_million_l2649_264958

/-- Represents the tax reduction process for a commodity -/
def tax_reduction (initial_tax : ℝ) : ℝ :=
  let after_first_reduction := initial_tax * (1 - 0.25)
  let after_second_reduction := after_first_reduction * (1 - 0.20)
  after_second_reduction

/-- Theorem stating that the tax reduction process results in 60% of the initial tax -/
theorem tax_reduction_is_sixty_percent (a : ℝ) :
  tax_reduction a = 0.60 * a := by
  sorry

/-- Corollary for the specific case where the initial tax is 1000 million euros -/
theorem tax_reduction_for_thousand_million :
  tax_reduction 1000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_is_sixty_percent_tax_reduction_for_thousand_million_l2649_264958


namespace NUMINAMATH_CALUDE_female_democrats_count_l2649_264973

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 780 →
  female + male = total →
  (female / 2 + male / 4 : ℚ) = total / 3 →
  female / 2 = 130 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2649_264973


namespace NUMINAMATH_CALUDE_function_properties_l2649_264950

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * x^2 + 1

-- Define the theorem
theorem function_properties (a : ℝ) :
  f a 1 = 5 →
  (a = 2 ∨ a = -2) ∧
  (∀ x y : ℝ, x < y ∧ x ≤ 0 ∧ 0 < y → f a x > f a y) ∧
  (∀ x y : ℝ, x < y ∧ 0 < x → f a x < f a y) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2649_264950


namespace NUMINAMATH_CALUDE_power_multiplication_l2649_264915

theorem power_multiplication (a b : ℕ) : (10 : ℕ) ^ 655 * (10 : ℕ) ^ 652 = (10 : ℕ) ^ (655 + 652) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2649_264915


namespace NUMINAMATH_CALUDE_mean_temperature_l2649_264984

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2649_264984


namespace NUMINAMATH_CALUDE_count_special_multiples_l2649_264996

theorem count_special_multiples : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 300 ∧ 3 ∣ n ∧ 5 ∣ n ∧ ¬(6 ∣ n) ∧ ¬(8 ∣ n)) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 300 ∧ 3 ∣ n ∧ 5 ∣ n ∧ ¬(6 ∣ n) ∧ ¬(8 ∣ n) → n ∈ S) ∧
  Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_special_multiples_l2649_264996


namespace NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_five_l2649_264959

theorem two_numbers_with_difference_and_quotient_five :
  ∀ x y : ℝ, x - y = 5 → x / y = 5 → x = 25 / 4 ∧ y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_five_l2649_264959


namespace NUMINAMATH_CALUDE_prime_rational_sum_l2649_264979

theorem prime_rational_sum (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℚ) (n : ℕ), x > 0 ∧ y > 0 ∧ x + y + p / x + p / y = 3 * n) ↔ 3 ∣ (p + 1) :=
sorry

end NUMINAMATH_CALUDE_prime_rational_sum_l2649_264979


namespace NUMINAMATH_CALUDE_sum_of_factors_l2649_264978

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60 →
  a + b + c + d + e = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2649_264978


namespace NUMINAMATH_CALUDE_two_over_x_values_l2649_264926

theorem two_over_x_values (x : ℝ) (h : 1 - 9/x + 20/x^2 = 0) :
  2/x = 1/2 ∨ 2/x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_two_over_x_values_l2649_264926


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2649_264917

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The median of a sequence is the middle value when the sequence is ordered. -/
def hasMedian (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ n : ℕ, a n = m ∧ (∀ i j : ℕ, i ≤ n ∧ n ≤ j → a i ≤ m ∧ m ≤ a j)

theorem arithmetic_sequence_first_term
  (a : ℕ → ℕ)
  (h1 : isArithmeticSequence a)
  (h2 : hasMedian a 1010)
  (h3 : ∃ n : ℕ, a n = 2015 ∧ ∀ m : ℕ, m > n → a m > 2015) :
  a 0 = 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2649_264917


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2649_264989

theorem square_of_real_not_always_positive : 
  ¬ (∀ a : ℝ, a^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2649_264989


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2649_264965

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (20^4 + 15^4 - 10^5) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (20^4 + 15^4 - 10^5) → q ≤ p ∧
    p = 59 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2649_264965


namespace NUMINAMATH_CALUDE_real_roots_sum_product_l2649_264994

theorem real_roots_sum_product (c d : ℝ) : 
  (c^4 - 6*c + 3 = 0) → 
  (d^4 - 6*d + 3 = 0) → 
  (c ≠ d) →
  (c*d + c + d = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_real_roots_sum_product_l2649_264994


namespace NUMINAMATH_CALUDE_room_tiles_theorem_l2649_264969

/-- Given a room with length and width in centimeters, 
    calculate the least number of square tiles required to cover the floor. -/
def leastNumberOfTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room of 624 cm by 432 cm, 
    the least number of square tiles required is 117. -/
theorem room_tiles_theorem :
  leastNumberOfTiles 624 432 = 117 := by
  sorry

#eval leastNumberOfTiles 624 432

end NUMINAMATH_CALUDE_room_tiles_theorem_l2649_264969


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l2649_264955

-- Define the given conditions
def total_worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

-- State the theorem
theorem remaining_problems_to_grade :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l2649_264955


namespace NUMINAMATH_CALUDE_inequality_proof_l2649_264957

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ 36 / (a - d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2649_264957


namespace NUMINAMATH_CALUDE_cos_180_degrees_l2649_264914

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l2649_264914


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2649_264998

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2649_264998


namespace NUMINAMATH_CALUDE_dave_final_tickets_l2649_264935

/-- Calculates the number of tickets Dave had left after a series of transactions at the arcade. -/
def dave_tickets_left (initial : ℕ) (won : ℕ) (spent1 : ℕ) (traded : ℕ) (spent2 : ℕ) : ℕ :=
  initial + won - spent1 + traded - spent2

/-- Proves that Dave had 57 tickets left at the end of his arcade visit. -/
theorem dave_final_tickets :
  dave_tickets_left 25 127 84 45 56 = 57 := by
  sorry

end NUMINAMATH_CALUDE_dave_final_tickets_l2649_264935


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l2649_264972

def num_packages : ℕ := 5
def num_correct : ℕ := 3

theorem correct_delivery_probability :
  (num_packages.choose num_correct * (num_correct.factorial * (num_packages - num_correct).factorial)) /
  num_packages.factorial = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l2649_264972


namespace NUMINAMATH_CALUDE_complex_coordinates_of_product_l2649_264946

theorem complex_coordinates_of_product : 
  let z : ℂ := (2 - Complex.I) * (1 + Complex.I)
  Complex.re z = 3 ∧ Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_coordinates_of_product_l2649_264946


namespace NUMINAMATH_CALUDE_laptop_sale_price_l2649_264920

def original_price : ℝ := 500
def first_discount : ℝ := 0.10
def second_discount : ℝ := 0.20
def delivery_fee : ℝ := 30

theorem laptop_sale_price :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  let final_price := price_after_second_discount + delivery_fee
  final_price = 390 := by sorry

end NUMINAMATH_CALUDE_laptop_sale_price_l2649_264920


namespace NUMINAMATH_CALUDE_chip_defect_rate_line_A_l2649_264908

theorem chip_defect_rate_line_A :
  let total_chips : ℕ := 20
  let chips_line_A : ℕ := 12
  let chips_line_B : ℕ := 8
  let defect_rate_B : ℚ := 1 / 20
  let overall_defect_rate : ℚ := 8 / 100
  let defect_rate_A : ℚ := 1 / 10
  (chips_line_A : ℚ) * defect_rate_A + (chips_line_B : ℚ) * defect_rate_B = (total_chips : ℚ) * overall_defect_rate :=
by sorry

end NUMINAMATH_CALUDE_chip_defect_rate_line_A_l2649_264908


namespace NUMINAMATH_CALUDE_distance_thirty_students_l2649_264930

/-- The distance between the first and last student in a line of students -/
def distance_between_ends (num_students : ℕ) (gap_distance : ℝ) : ℝ :=
  (num_students - 1 : ℝ) * gap_distance

/-- Theorem: For 30 students standing in a line with 3 meters between adjacent students,
    the distance between the first and last student is 87 meters. -/
theorem distance_thirty_students :
  distance_between_ends 30 3 = 87 := by
  sorry

end NUMINAMATH_CALUDE_distance_thirty_students_l2649_264930
