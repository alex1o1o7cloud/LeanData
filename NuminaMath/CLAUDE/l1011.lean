import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l1011_101194

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase : 
  (1.56 - 1) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l1011_101194


namespace NUMINAMATH_CALUDE_exponent_division_rule_l1011_101195

theorem exponent_division_rule (a b : ℝ) (m : ℤ) 
  (ha : a > 0) (hb : b ≠ 0) : 
  (b / a) ^ m = a ^ (-m) * b ^ m := by sorry

end NUMINAMATH_CALUDE_exponent_division_rule_l1011_101195


namespace NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l1011_101133

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((2*x + y) * (2*x - y) - (2*x - 3*y)^2) / (-2*y) = -6*x + 5*y := by
  sorry

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  ((2*x + y) * (2*x - y) - (2*x - 3*y)^2) / (-2*y) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_expression_evaluation_l1011_101133


namespace NUMINAMATH_CALUDE_circle_area_sum_l1011_101102

/-- The sum of the areas of an infinite sequence of circles, where the first circle
    has a radius of 3 inches and each subsequent circle's radius is 2/3 of the previous one,
    is equal to 81π/5. -/
theorem circle_area_sum : 
  let r : ℕ → ℝ := λ n => 3 * (2/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (r n)^2
  (∑' n, area n) = 81*π/5 :=
sorry

end NUMINAMATH_CALUDE_circle_area_sum_l1011_101102


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1011_101121

-- Define the repeating decimal 0.3333...
def repeating_3 : ℚ := 1 / 3

-- Define the repeating decimal 0.2121...
def repeating_21 : ℚ := 7 / 33

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_3 + repeating_21 = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1011_101121


namespace NUMINAMATH_CALUDE_quadratic_roots_integrality_l1011_101124

/-- Given two quadratic equations x^2 - px + q = 0 and x^2 - (p+1)x + q = 0,
    this theorem states that when q > 0, both equations can have integer roots,
    but when q < 0, they cannot both have integer roots simultaneously. -/
theorem quadratic_roots_integrality (p q : ℤ) :
  (q > 0 → ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) ∧
  (q < 0 → ¬∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integrality_l1011_101124


namespace NUMINAMATH_CALUDE_jameson_medal_count_l1011_101146

/-- Represents the number of medals Jameson has in each category -/
structure MedalCount where
  track : Nat
  swimming : Nat
  badminton : Nat

/-- Calculates the total number of medals -/
def totalMedals (medals : MedalCount) : Nat :=
  medals.track + medals.swimming + medals.badminton

/-- Theorem: Jameson's total medal count is 20 -/
theorem jameson_medal_count :
  ∀ (medals : MedalCount),
    medals.track = 5 →
    medals.swimming = 2 * medals.track →
    medals.badminton = 5 →
    totalMedals medals = 20 := by
  sorry


end NUMINAMATH_CALUDE_jameson_medal_count_l1011_101146


namespace NUMINAMATH_CALUDE_product_of_one_plus_tangents_l1011_101108

theorem product_of_one_plus_tangents (A B C : Real) : 
  A = π / 12 →  -- 15°
  B = π / 6 →   -- 30°
  A + B + C = π / 2 →  -- 90°
  (1 + Real.tan A) * (1 + Real.tan B) * (1 + Real.tan C) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tangents_l1011_101108


namespace NUMINAMATH_CALUDE_not_perfect_square_l1011_101131

theorem not_perfect_square : 
  (∃ a : ℕ, 1^2016 = a^2) ∧ 
  (∀ b : ℕ, 2^2017 ≠ b^2) ∧ 
  (∃ c : ℕ, 3^2018 = c^2) ∧ 
  (∃ d : ℕ, 4^2019 = d^2) ∧ 
  (∃ e : ℕ, 5^2020 = e^2) := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1011_101131


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l1011_101154

theorem fraction_multiplication_equality : 
  (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5040.000000000001 = 756.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l1011_101154


namespace NUMINAMATH_CALUDE_marys_income_percentage_l1011_101109

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.5))
  (h2 : mary = tim * (1 + 0.6)) :
  mary = juan * 0.8 := by sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l1011_101109


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1011_101157

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i + 2 / (1 + i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1011_101157


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l1011_101115

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the total number of classrooms
def total_classrooms : ℕ := 5

-- Define the number of absent rabbits
def absent_rabbits : ℕ := 1

-- Theorem statement
theorem student_rabbit_difference :
  students_per_classroom * total_classrooms - 
  (rabbits_per_classroom * total_classrooms) = 105 := by
  sorry


end NUMINAMATH_CALUDE_student_rabbit_difference_l1011_101115


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1011_101140

theorem circle_center_radius_sum : ∃ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 - 14*x + y^2 + 6*y = 25 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
  a + b + r = 4 + Real.sqrt 83 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1011_101140


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l1011_101130

theorem quadratic_expression_values (a b : ℝ) 
  (ha : a^2 = 16)
  (hb : abs b = 3)
  (hab : a * b < 0) :
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l1011_101130


namespace NUMINAMATH_CALUDE_initial_strawberry_plants_l1011_101127

/-- The number of strawberry plants after n months of doubling -/
def plants_after_months (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

/-- The theorem stating the initial number of strawberry plants -/
theorem initial_strawberry_plants : ∃ (initial : ℕ), 
  plants_after_months initial 3 - 4 = 20 ∧ initial > 0 := by
  sorry

end NUMINAMATH_CALUDE_initial_strawberry_plants_l1011_101127


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_products_l1011_101149

theorem cube_sum_greater_than_mixed_products {a b : ℝ} (ha : a > 0) (hb : b > 0) (hnq : a ≠ b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_products_l1011_101149


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1011_101152

/-- Given a quadratic equation ax^2 + 15x + c = 0 with exactly one solution,
    where a + c = 36 and a < c, prove that a = (36 - √1071) / 2 and c = (36 + √1071) / 2 -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 15 * x + c = 0) → 
  (a + c = 36) → 
  (a < c) → 
  (a = (36 - Real.sqrt 1071) / 2 ∧ c = (36 + Real.sqrt 1071) / 2) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1011_101152


namespace NUMINAMATH_CALUDE_divisibility_by_30_l1011_101104

theorem divisibility_by_30 (a m n : ℕ) (k : ℤ) 
  (h1 : m > n) (h2 : n ≥ 2) (h3 : m - n = 4 * k.natAbs) : 
  ∃ (q : ℤ), a^m - a^n = 30 * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_30_l1011_101104


namespace NUMINAMATH_CALUDE_school_run_speed_l1011_101199

theorem school_run_speed (v : ℝ) (h : v > 0) : 
  (v + 2) / v = 2.5 → (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_school_run_speed_l1011_101199


namespace NUMINAMATH_CALUDE_arrangements_count_l1011_101101

-- Define the number of people and exits
def num_people : ℕ := 5
def num_exits : ℕ := 4

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ := sorry

-- Theorem statement
theorem arrangements_count : num_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1011_101101


namespace NUMINAMATH_CALUDE_intersection_M_N_l1011_101122

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x ^ 2 ≤ 4}

def N : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1011_101122


namespace NUMINAMATH_CALUDE_cut_rectangle_pentagon_area_cut_rectangle_pentagon_area_proof_l1011_101100

/-- Pentagon formed by removing a triangle from a rectangle --/
structure CutRectanglePentagon where
  sides : Finset ℕ
  side_count : sides.card = 5
  side_values : sides = {14, 21, 22, 28, 35}

/-- Theorem stating the area of the specific pentagon --/
theorem cut_rectangle_pentagon_area (p : CutRectanglePentagon) : ℕ :=
  1176

#check cut_rectangle_pentagon_area

/-- Proof of the theorem --/
theorem cut_rectangle_pentagon_area_proof (p : CutRectanglePentagon) :
  cut_rectangle_pentagon_area p = 1176 := by
  sorry

end NUMINAMATH_CALUDE_cut_rectangle_pentagon_area_cut_rectangle_pentagon_area_proof_l1011_101100


namespace NUMINAMATH_CALUDE_ben_winning_strategy_l1011_101160

/-- Represents the state of a card (0 for letter, 1 for number) -/
inductive CardState
| Letter : CardState
| Number : CardState

/-- Represents a configuration of cards -/
def Configuration := Vector CardState 2019

/-- Represents a move in the game -/
structure Move where
  position : Fin 2019

/-- Applies a move to a configuration -/
def applyMove (config : Configuration) (move : Move) : Configuration :=
  sorry

/-- Checks if all cards are showing numbers -/
def allNumbers (config : Configuration) : Prop :=
  sorry

theorem ben_winning_strategy :
  ∀ (initial_config : Configuration),
  ∃ (moves : List Move),
    moves.length ≤ 2019 ∧
    allNumbers (moves.foldl applyMove initial_config) :=
  sorry

end NUMINAMATH_CALUDE_ben_winning_strategy_l1011_101160


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1011_101167

theorem sin_cos_identity : 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) - 
  Real.sin (69 * π / 180) * Real.cos (9 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1011_101167


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l1011_101128

theorem gcd_of_polynomial_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1428 * k) : 
  Nat.gcd (Int.natAbs (b^2 + 11*b + 30)) (Int.natAbs (b + 6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l1011_101128


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l1011_101168

/-- Given that 4 pages cost 6 cents, prove that $15 (1500 cents) will allow copying 1000 pages. -/
theorem pages_copied_for_fifteen_dollars :
  let pages_per_six_cents : ℚ := 4
  let cents_per_four_pages : ℚ := 6
  let total_cents : ℚ := 1500
  (total_cents * pages_per_six_cents) / cents_per_four_pages = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l1011_101168


namespace NUMINAMATH_CALUDE_money_division_l1011_101106

theorem money_division (p q r : ℕ) (total : ℚ) :
  p + q + r = 22 →
  3 * total / 22 = p →
  7 * total / 22 = q →
  12 * total / 22 = r →
  q - p = 4400 →
  r - q = 5500 := by
sorry

end NUMINAMATH_CALUDE_money_division_l1011_101106


namespace NUMINAMATH_CALUDE_first_liquid_volume_l1011_101179

theorem first_liquid_volume (x : ℝ) : 
  (0.75 * x + 63) / (x + 90) = 0.7263157894736842 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_liquid_volume_l1011_101179


namespace NUMINAMATH_CALUDE_potion_kit_cost_is_18_silver_l1011_101170

/-- Represents the cost of items in Harry's purchase --/
structure PurchaseCost where
  spellbookCost : ℕ
  owlCost : ℕ
  totalSilver : ℕ
  silverToGold : ℕ

/-- Calculates the cost of each potion kit in silver --/
def potionKitCost (p : PurchaseCost) : ℕ :=
  let totalGold := p.totalSilver / p.silverToGold
  let spellbooksTotalCost := 5 * p.spellbookCost
  let remainingGold := totalGold - spellbooksTotalCost - p.owlCost
  let potionKitGold := remainingGold / 3
  potionKitGold * p.silverToGold

/-- Theorem stating that each potion kit costs 18 silvers --/
theorem potion_kit_cost_is_18_silver (p : PurchaseCost) 
  (h1 : p.spellbookCost = 5)
  (h2 : p.owlCost = 28)
  (h3 : p.totalSilver = 537)
  (h4 : p.silverToGold = 9) : 
  potionKitCost p = 18 := by
  sorry


end NUMINAMATH_CALUDE_potion_kit_cost_is_18_silver_l1011_101170


namespace NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l1011_101169

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Theorem: For a parabola y^2 = 2px and a line passing through its focus
    with an inclination angle of 60°, intersecting the parabola at points A and B
    in the first and fourth quadrants respectively, the ratio |AF| / |BF| = 3 -/
theorem parabola_line_intersection_ratio 
  (para : Parabola) 
  (l : Line) 
  (A B : Point) 
  (h1 : l.point = Point.mk (para.p / 2) 0)  -- Focus of the parabola
  (h2 : l.angle = π / 3)  -- 60° in radians
  (h3 : A.x > 0 ∧ A.y > 0)  -- A in first quadrant
  (h4 : B.x > 0 ∧ B.y < 0)  -- B in fourth quadrant
  (h5 : A.y^2 = 2 * para.p * A.x)  -- A on parabola
  (h6 : B.y^2 = 2 * para.p * B.x)  -- B on parabola
  : abs (A.x - para.p / 2) / abs (B.x - para.p / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l1011_101169


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1011_101125

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 12 14 1.25 = 233 := by
  sorry

#eval total_wet_surface_area 12 14 1.25

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1011_101125


namespace NUMINAMATH_CALUDE_function_value_range_l1011_101187

theorem function_value_range (x : ℝ) :
  x ∈ Set.Icc 1 4 →
  2 ≤ x^2 - 4*x + 6 ∧ x^2 - 4*x + 6 ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_function_value_range_l1011_101187


namespace NUMINAMATH_CALUDE_dina_machine_l1011_101150

def f (x : ℚ) : ℚ := 2 * x - 3

theorem dina_machine (x : ℚ) : f (f x) = -35 → x = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_dina_machine_l1011_101150


namespace NUMINAMATH_CALUDE_emma_sandwich_combinations_l1011_101107

def num_meat : ℕ := 12
def num_cheese : ℕ := 11

def sandwich_combinations : ℕ := (num_meat.choose 2) * (num_cheese.choose 2)

theorem emma_sandwich_combinations :
  sandwich_combinations = 3630 := by sorry

end NUMINAMATH_CALUDE_emma_sandwich_combinations_l1011_101107


namespace NUMINAMATH_CALUDE_jamie_ball_collection_l1011_101155

/-- Calculates the total number of balls Jamie has after all transactions --/
def total_balls (initial_red : ℕ) (blue_multiplier : ℕ) (lost_red : ℕ) (yellow_multiplier : ℕ) : ℕ :=
  let initial_blue := initial_red * blue_multiplier
  let remaining_red := initial_red - lost_red
  let bought_yellow := lost_red * yellow_multiplier
  remaining_red + initial_blue + bought_yellow

theorem jamie_ball_collection :
  total_balls 16 2 6 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_jamie_ball_collection_l1011_101155


namespace NUMINAMATH_CALUDE_range_of_a_l1011_101111

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1011_101111


namespace NUMINAMATH_CALUDE_chloe_trivia_score_l1011_101162

/-- Chloe's trivia game score calculation -/
theorem chloe_trivia_score (first_round : ℕ) (last_round_loss : ℕ) (total_points : ℕ) 
  (h1 : first_round = 40)
  (h2 : last_round_loss = 4)
  (h3 : total_points = 86) :
  ∃ second_round : ℕ, second_round = 50 ∧ 
    first_round + second_round - last_round_loss = total_points :=
by sorry

end NUMINAMATH_CALUDE_chloe_trivia_score_l1011_101162


namespace NUMINAMATH_CALUDE_least_integer_divisible_by_four_primes_l1011_101156

theorem least_integer_divisible_by_four_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p q r s : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
   p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
   n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ p q r s : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
     p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
     m % p = 0 ∧ m % q = 0 ∧ m % r = 0 ∧ m % s = 0) → 
    m ≥ 210) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_divisible_by_four_primes_l1011_101156


namespace NUMINAMATH_CALUDE_flight_duration_sum_l1011_101137

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

theorem flight_duration_sum (departureLA : Time) (arrivalNY : Time) (h m : ℕ) :
  departureLA.hours = 9 →
  departureLA.minutes = 15 →
  arrivalNY.hours = 17 →
  arrivalNY.minutes = 40 →
  0 < m →
  m < 60 →
  timeDiffInMinutes 
    {hours := departureLA.hours + 3, minutes := departureLA.minutes, valid := sorry}
    arrivalNY = h * 60 + m →
  h + m = 30 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l1011_101137


namespace NUMINAMATH_CALUDE_calculate_3Y5_l1011_101173

-- Define the operation Y
def Y (a b : ℤ) : ℤ := b + 12 * a - a^3

-- Theorem to prove
theorem calculate_3Y5 : Y 3 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_calculate_3Y5_l1011_101173


namespace NUMINAMATH_CALUDE_class_mean_score_l1011_101163

/-- Proves that the overall mean score of a class is 76.17% given the specified conditions -/
theorem class_mean_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_avg : ℚ) (group2_avg : ℚ) :
  total_students = 48 →
  group1_students = 40 →
  group2_students = 8 →
  group1_avg = 75 / 100 →
  group2_avg = 82 / 100 →
  let overall_avg := (group1_students * group1_avg + group2_students * group2_avg) / total_students
  overall_avg = 7617 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_class_mean_score_l1011_101163


namespace NUMINAMATH_CALUDE_parabola_equation_l1011_101183

/-- Given a parabola y^2 = mx (m > 0) whose directrix is at a distance of 3 from the line x = 1,
    prove that the equation of the parabola is y^2 = 8x. -/
theorem parabola_equation (m : ℝ) (h_m_pos : m > 0) : 
  (∃ (k : ℝ), k = -m/4 ∧ |k - 1| = 3) → 
  (∀ (x y : ℝ), y^2 = m*x ↔ y^2 = 8*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1011_101183


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1011_101116

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem for part (I)
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ -6 ≤ a ∧ a ≤ -2 := by sorry

-- Theorem for part (II)
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a < -9 ∨ a > 1 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1011_101116


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l1011_101175

theorem det_2x2_matrix (x : ℝ) : 
  Matrix.det !![5, x; 4, 3] = 15 - 4 * x := by sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l1011_101175


namespace NUMINAMATH_CALUDE_B_alone_time_l1011_101147

/-- The time it takes for A and B together to complete the job -/
def time_AB : ℝ := 3

/-- The time it takes for B and C together to complete the job -/
def time_BC : ℝ := 6

/-- The time it takes for A and C together to complete the job -/
def time_AC : ℝ := 3.6

/-- The rate at which A completes the job -/
def rate_A : ℝ := sorry

/-- The rate at which B completes the job -/
def rate_B : ℝ := sorry

/-- The rate at which C completes the job -/
def rate_C : ℝ := sorry

theorem B_alone_time : 
  rate_A + rate_B = 1 / time_AB ∧ 
  rate_B + rate_C = 1 / time_BC ∧ 
  rate_A + rate_C = 1 / time_AC → 
  1 / rate_B = 9 := by sorry

end NUMINAMATH_CALUDE_B_alone_time_l1011_101147


namespace NUMINAMATH_CALUDE_max_wrong_questions_l1011_101181

theorem max_wrong_questions (total_questions : ℕ) (success_threshold : ℚ) : 
  total_questions = 50 → success_threshold = 85 / 100 → 
  ∃ max_wrong : ℕ, max_wrong = 7 ∧ 
  (↑(total_questions - max_wrong) / ↑total_questions ≥ success_threshold) ∧
  ∀ wrong : ℕ, wrong > max_wrong → 
  (↑(total_questions - wrong) / ↑total_questions < success_threshold) :=
by
  sorry

end NUMINAMATH_CALUDE_max_wrong_questions_l1011_101181


namespace NUMINAMATH_CALUDE_unused_cubes_for_5x5x5_with_9_tunnels_l1011_101132

/-- Represents a large cube made of small cubes with tunnels --/
structure LargeCube where
  size : Nat
  numTunnels : Nat

/-- Calculates the number of unused small cubes in a large cube with tunnels --/
def unusedCubes (c : LargeCube) : Nat :=
  c.size^3 - (c.numTunnels * c.size - 6)

/-- Theorem stating that for a 5x5x5 cube with 9 tunnels, 39 small cubes are unused --/
theorem unused_cubes_for_5x5x5_with_9_tunnels :
  let c : LargeCube := { size := 5, numTunnels := 9 }
  unusedCubes c = 39 := by
  sorry

#eval unusedCubes { size := 5, numTunnels := 9 }

end NUMINAMATH_CALUDE_unused_cubes_for_5x5x5_with_9_tunnels_l1011_101132


namespace NUMINAMATH_CALUDE_red_ball_probability_not_red_ball_probability_l1011_101126

/-- Represents the set of ball colors in the box -/
inductive BallColor
| Red
| White
| Black

/-- Represents the count of balls for each color -/
def ballCount : BallColor → ℕ
| BallColor.Red => 3
| BallColor.White => 5
| BallColor.Black => 7

/-- The total number of balls in the box -/
def totalBalls : ℕ := ballCount BallColor.Red + ballCount BallColor.White + ballCount BallColor.Black

/-- The probability of drawing a ball of a specific color -/
def drawProbability (color : BallColor) : ℚ :=
  ballCount color / totalBalls

theorem red_ball_probability :
  drawProbability BallColor.Red = 1 / 5 := by sorry

theorem not_red_ball_probability :
  1 - drawProbability BallColor.Red = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_red_ball_probability_not_red_ball_probability_l1011_101126


namespace NUMINAMATH_CALUDE_fraction_nonzero_digits_l1011_101118

def fraction : ℚ := 120 / (2^4 * 5^8)

def count_nonzero_decimal_digits (q : ℚ) : ℕ :=
  -- Function to count non-zero digits after the decimal point
  sorry

theorem fraction_nonzero_digits :
  count_nonzero_decimal_digits fraction = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_nonzero_digits_l1011_101118


namespace NUMINAMATH_CALUDE_binomial_square_last_term_l1011_101142

theorem binomial_square_last_term (a b : ℝ) :
  ∃ x y : ℝ, x^2 - 10*x*y + 25*y^2 = (x + y)^2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_square_last_term_l1011_101142


namespace NUMINAMATH_CALUDE_exists_m_eq_power_plus_n_l1011_101198

/-- n(m) denotes the number of factors of 2 in m! -/
def n (m : ℕ+) : ℕ := sorry

/-- Theorem: There exists a natural number m > 2006^2006 such that m = 3^2006 + n(m) -/
theorem exists_m_eq_power_plus_n : ∃ m : ℕ+, 
  (m : ℕ) > 2006^2006 ∧ (m : ℕ) = 3^2006 + n m := by
  sorry

end NUMINAMATH_CALUDE_exists_m_eq_power_plus_n_l1011_101198


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1011_101139

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, a n > 0) →  -- Positive sequence
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n) →  -- Geometric sequence
  a 3 = 3 →  -- Given condition
  a 5 = 8 * a 7 →  -- Given condition
  a 10 = 3 * Real.sqrt 2 / 128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1011_101139


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1011_101166

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) :
  1 / x + 1 / y = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1011_101166


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l1011_101113

/-- 
Given an isosceles, obtuse triangle where one angle is 75% larger than a right angle,
prove that the measure of each of the two smallest angles is 45/4 degrees.
-/
theorem isosceles_obtuse_triangle_smallest_angle 
  (α β γ : ℝ) 
  (h_isosceles : α = β)
  (h_obtuse : γ > 90)
  (h_large_angle : γ = 90 * 1.75)
  (h_angle_sum : α + β + γ = 180) : 
  α = 45 / 4 := by
sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l1011_101113


namespace NUMINAMATH_CALUDE_negation_existence_quadratic_l1011_101178

theorem negation_existence_quadratic (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + m > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_quadratic_l1011_101178


namespace NUMINAMATH_CALUDE_product_calculation_l1011_101193

theorem product_calculation : (1/2 : ℚ) * 8 * (1/8 : ℚ) * 32 * (1/32 : ℚ) * 128 * (1/128 : ℚ) * 512 * (1/512 : ℚ) * 2048 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l1011_101193


namespace NUMINAMATH_CALUDE_minimum_packaging_cost_l1011_101159

/-- Calculates the minimum cost for packaging a collection given box dimensions, cost per box, and total volume to be packaged -/
theorem minimum_packaging_cost 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (cost_per_box : ℝ) 
  (total_volume : ℝ) 
  (h_box_length : box_length = 20)
  (h_box_width : box_width = 20)
  (h_box_height : box_height = 15)
  (h_cost_per_box : cost_per_box = 0.70)
  (h_total_volume : total_volume = 3060000) :
  ⌈total_volume / (box_length * box_width * box_height)⌉ * cost_per_box = 357 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packaging_cost_l1011_101159


namespace NUMINAMATH_CALUDE_second_number_is_thirty_l1011_101114

theorem second_number_is_thirty
  (a b c : ℝ)
  (sum_eq_98 : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  b = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_thirty_l1011_101114


namespace NUMINAMATH_CALUDE_min_value_on_negative_interval_l1011_101180

/-- Given positive real numbers a and b, and a function f with maximum value 4 on [0,1],
    prove that the minimum value of f on [-1,0] is -3/2 -/
theorem min_value_on_negative_interval
  (a b : ℝ) (f : ℝ → ℝ)
  (a_pos : 0 < a) (b_pos : 0 < b)
  (f_def : ∀ x, f x = a * x^3 + b * x + 2^x)
  (max_value : ∀ x ∈ Set.Icc 0 1, f x ≤ 4)
  (max_achieved : ∃ x ∈ Set.Icc 0 1, f x = 4) :
  ∀ x ∈ Set.Icc (-1) 0, f x ≥ -3/2 ∧ ∃ y ∈ Set.Icc (-1) 0, f y = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_negative_interval_l1011_101180


namespace NUMINAMATH_CALUDE_score_mode_l1011_101158

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The list of all scores from the stem-and-leaf plot -/
def scores : List Score := [
  {stem := 5, leaf := 5}, {stem := 5, leaf := 5},
  {stem := 6, leaf := 4}, {stem := 6, leaf := 8},
  {stem := 7, leaf := 2}, {stem := 7, leaf := 6}, {stem := 7, leaf := 6}, {stem := 7, leaf := 9},
  {stem := 8, leaf := 1}, {stem := 8, leaf := 3}, {stem := 8, leaf := 3}, {stem := 8, leaf := 3}, {stem := 8, leaf := 9}, {stem := 8, leaf := 9},
  {stem := 9, leaf := 0}, {stem := 9, leaf := 5}, {stem := 9, leaf := 5}, {stem := 9, leaf := 5}, {stem := 9, leaf := 7}, {stem := 9, leaf := 8},
  {stem := 10, leaf := 2}, {stem := 10, leaf := 2}, {stem := 10, leaf := 2}, {stem := 10, leaf := 3}, {stem := 10, leaf := 3}, {stem := 10, leaf := 3}, {stem := 10, leaf := 4},
  {stem := 11, leaf := 0}, {stem := 11, leaf := 0}, {stem := 11, leaf := 1}
]

/-- Converts a Score to its numerical value -/
def scoreValue (s : Score) : Nat := s.stem * 10 + s.leaf

/-- Defines the mode of a list of scores -/
def mode (l : List Score) : Set Nat := sorry

/-- Theorem stating that the mode of the given scores is {83, 95, 102, 103} -/
theorem score_mode : mode scores = {83, 95, 102, 103} := by sorry

end NUMINAMATH_CALUDE_score_mode_l1011_101158


namespace NUMINAMATH_CALUDE_gomoku_piece_count_l1011_101117

/-- Represents the number of pieces in the Gomoku game box -/
structure GomokuBox where
  initial_black : ℕ
  initial_white : ℕ
  added_black : ℕ
  added_white : ℕ

/-- Theorem statement for the Gomoku piece counting problem -/
theorem gomoku_piece_count (box : GomokuBox) : 
  box.initial_black = box.initial_white ∧ 
  box.initial_black + box.initial_white ≤ 10 ∧
  box.added_black + box.added_white = 20 ∧
  7 * (box.initial_white + box.added_white) = 8 * (box.initial_black + box.added_black) →
  box.initial_black + box.added_black = 16 := by
  sorry

end NUMINAMATH_CALUDE_gomoku_piece_count_l1011_101117


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l1011_101171

/-- The price of company KW as a percentage of the combined assets of companies A and B -/
theorem company_kw_price_percentage (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  let p := 1.2 * a  -- Price of company KW
  let combined_assets := a + b
  p / combined_assets = 0.75 := by sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l1011_101171


namespace NUMINAMATH_CALUDE_square_area_with_inscribed_circle_l1011_101191

theorem square_area_with_inscribed_circle (r : ℝ) (h1 : r > 0) 
  (h2 : (r - 1)^2 + (r - 2)^2 = r^2) : (2*r)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_inscribed_circle_l1011_101191


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1011_101143

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.log x > 1) ↔ (∃ x : ℝ, Real.log x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1011_101143


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1011_101144

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | 1 - x > 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1011_101144


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l1011_101176

theorem right_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2) (leg_relation : a = 2*b) :
  (a + b) / c = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l1011_101176


namespace NUMINAMATH_CALUDE_regression_line_estimate_l1011_101120

/-- Represents a linear regression line y = ax + b -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on the regression line -/
def RegressionLine.evaluate (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

theorem regression_line_estimate :
  ∀ (line : RegressionLine),
    line.slope = 1.23 →
    line.evaluate 4 = 5 →
    line.evaluate 2 = 2.54 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_estimate_l1011_101120


namespace NUMINAMATH_CALUDE_books_left_over_l1011_101136

theorem books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ)
  (h1 : initial_boxes = 1575)
  (h2 : books_per_initial_box = 45)
  (h3 : books_per_new_box = 46) :
  (initial_boxes * books_per_initial_box) % books_per_new_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_left_over_l1011_101136


namespace NUMINAMATH_CALUDE_min_singing_in_shower_l1011_101153

/-- Represents the youth summer village population -/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamilies : ℕ
  workingNoFamilySinging : ℕ

/-- The minimum number of people who like to sing in the shower -/
def minSingingInShower (v : Village) : ℕ := v.workingNoFamilySinging

theorem min_singing_in_shower (v : Village) 
  (h1 : v.total = 100)
  (h2 : v.notWorking = 50)
  (h3 : v.withFamilies = 25)
  (h4 : v.workingNoFamilySinging = 50)
  (h5 : v.workingNoFamilySinging ≤ v.total - v.notWorking)
  (h6 : v.workingNoFamilySinging ≤ v.total - v.withFamilies) :
  minSingingInShower v = 50 := by
  sorry

#check min_singing_in_shower

end NUMINAMATH_CALUDE_min_singing_in_shower_l1011_101153


namespace NUMINAMATH_CALUDE_divisors_of_factorial_8_l1011_101129

theorem divisors_of_factorial_8 : (Nat.divisors (Nat.factorial 8)).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_factorial_8_l1011_101129


namespace NUMINAMATH_CALUDE_cookie_jar_remaining_l1011_101105

/-- The amount left in the cookie jar after Doris and Martha's spending -/
theorem cookie_jar_remaining (initial_amount : ℕ) (doris_spent : ℕ) (martha_spent : ℕ) : 
  initial_amount = 21 → 
  doris_spent = 6 → 
  martha_spent = doris_spent / 2 → 
  initial_amount - (doris_spent + martha_spent) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_remaining_l1011_101105


namespace NUMINAMATH_CALUDE_jordan_fourth_period_blocks_l1011_101186

/-- Represents the number of shots blocked by a hockey goalie in each period of a game --/
structure GoalieBlocks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the total number of shots blocked in a game --/
def totalBlocks (blocks : GoalieBlocks) : ℕ :=
  blocks.first + blocks.second + blocks.third + blocks.fourth

/-- Theorem: Given the conditions of Jordan's game, he blocked 4 shots in the fourth period --/
theorem jordan_fourth_period_blocks :
  ∀ (blocks : GoalieBlocks),
    blocks.first = 4 →
    blocks.second = 2 * blocks.first →
    blocks.third = blocks.second - 3 →
    totalBlocks blocks = 21 →
    blocks.fourth = 4 := by
  sorry


end NUMINAMATH_CALUDE_jordan_fourth_period_blocks_l1011_101186


namespace NUMINAMATH_CALUDE_tournament_configuration_impossible_l1011_101151

structure Tournament where
  num_teams : Nat
  games_played : Fin num_teams → Nat

def is_valid_configuration (t : Tournament) : Prop :=
  t.num_teams = 12 ∧
  (∃ i : Fin t.num_teams, t.games_played i = 11) ∧
  (∃ i j k : Fin t.num_teams, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    t.games_played i = 9 ∧ t.games_played j = 9 ∧ t.games_played k = 9) ∧
  (∃ i j : Fin t.num_teams, i ≠ j ∧ 
    t.games_played i = 6 ∧ t.games_played j = 6) ∧
  (∃ i j k l : Fin t.num_teams, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    t.games_played i = 4 ∧ t.games_played j = 4 ∧ t.games_played k = 4 ∧ t.games_played l = 4) ∧
  (∃ i j : Fin t.num_teams, i ≠ j ∧ 
    t.games_played i = 1 ∧ t.games_played j = 1)

theorem tournament_configuration_impossible :
  ¬∃ t : Tournament, is_valid_configuration t := by
  sorry

end NUMINAMATH_CALUDE_tournament_configuration_impossible_l1011_101151


namespace NUMINAMATH_CALUDE_intersection_length_theorem_l1011_101135

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 - y^2/3 = 1 ∧ x < -1

-- Define a line through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop := x = m * y - 2

-- Theorem statement
theorem intersection_length_theorem 
  (A B P Q : ℝ × ℝ) 
  (m : ℝ) 
  (h₁ : C A.1 A.2) 
  (h₂ : C B.1 B.2) 
  (h₃ : F₂ P.1 P.2) 
  (h₄ : F₂ Q.1 Q.2) 
  (h₅ : line_through_F₁ m A.1 A.2) 
  (h₆ : line_through_F₁ m B.1 B.2) 
  (h₇ : line_through_F₁ m P.1 P.2) 
  (h₈ : line_through_F₁ m Q.1 Q.2) 
  (h₉ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_length_theorem_l1011_101135


namespace NUMINAMATH_CALUDE_function_f_property_l1011_101188

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = 2 - f x) ∧
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- The theorem statement -/
theorem function_f_property (f : ℝ → ℝ) (a : ℝ) 
  (hf : FunctionF f) 
  (h : ∀ x ∈ Set.Icc 1 2, f (a * x + 2) + f 1 ≤ 2) : 
  a ∈ Set.Iic (-3) :=
sorry

end NUMINAMATH_CALUDE_function_f_property_l1011_101188


namespace NUMINAMATH_CALUDE_det_scalar_multiple_l1011_101134

theorem det_scalar_multiple {a b c d : ℝ} (h : Matrix.det !![a, b; c, d] = 5) :
  Matrix.det !![3*a, 3*b; 3*c, 3*d] = 45 := by
  sorry

end NUMINAMATH_CALUDE_det_scalar_multiple_l1011_101134


namespace NUMINAMATH_CALUDE_sequence_closed_form_l1011_101177

theorem sequence_closed_form (a : ℕ → ℤ) :
  a 1 = 0 ∧
  (∀ n : ℕ, n ≥ 2 → a n - 2 * a (n - 1) = n^2 - 3) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n + 2) - n^2 - 4*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_closed_form_l1011_101177


namespace NUMINAMATH_CALUDE_touchdown_points_l1011_101182

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
sorry

end NUMINAMATH_CALUDE_touchdown_points_l1011_101182


namespace NUMINAMATH_CALUDE_cos_300_deg_l1011_101192

theorem cos_300_deg : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_deg_l1011_101192


namespace NUMINAMATH_CALUDE_only_B_on_x_axis_l1011_101112

def point_A : ℝ × ℝ := (-2, -3)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (0, 3)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis : 
  ¬(is_on_x_axis point_A) ∧
  is_on_x_axis point_B ∧
  ¬(is_on_x_axis point_C) ∧
  ¬(is_on_x_axis point_D) :=
by sorry

end NUMINAMATH_CALUDE_only_B_on_x_axis_l1011_101112


namespace NUMINAMATH_CALUDE_cranberry_juice_can_ounces_l1011_101190

/-- Given a can of cranberry juice that sells for 84 cents with a unit cost of 7.0 cents per ounce,
    prove that the can contains 12 ounces of juice. -/
theorem cranberry_juice_can_ounces :
  ∀ (total_cost unit_cost : ℚ),
    total_cost = 84 →
    unit_cost = 7 →
    total_cost / unit_cost = 12 := by
sorry

end NUMINAMATH_CALUDE_cranberry_juice_can_ounces_l1011_101190


namespace NUMINAMATH_CALUDE_cockroach_search_l1011_101161

/-- The cockroach's search problem -/
theorem cockroach_search (D : ℝ) (h : D > 0) :
  ∃ (path : ℕ → ℝ × ℝ),
    (∀ n, dist (path n) (path (n+1)) ≤ 1) ∧
    (∀ n, dist (path (n+1)) (D, 0) < dist (path n) (D, 0) ∨
          dist (path (n+1)) (D, 0) = dist (path n) (D, 0)) ∧
    (∃ n, path n = (D, 0)) ∧
    (∃ n, path n = (D, 0) ∧ n ≤ ⌊(3/2 * D + 7)⌋) :=
sorry


end NUMINAMATH_CALUDE_cockroach_search_l1011_101161


namespace NUMINAMATH_CALUDE_union_of_sets_l1011_101174

theorem union_of_sets : 
  let M : Set ℤ := {-1, 0, 1}
  let N : Set ℤ := {0, 1, 2}
  M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1011_101174


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1011_101110

/-- The length of the major axis of the ellipse 16x^2 + 9y^2 = 144 is 8 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 16 * x^2 + 9 * y^2 = 144}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1011_101110


namespace NUMINAMATH_CALUDE_sunzi_wood_problem_l1011_101103

theorem sunzi_wood_problem (x : ℝ) : 
  (∃ rope : ℝ, rope = x + 4.5 ∧ (rope / 2) + 1 = x) → 
  (1/2 * (x + 4.5) = x - 1) := by
sorry

end NUMINAMATH_CALUDE_sunzi_wood_problem_l1011_101103


namespace NUMINAMATH_CALUDE_eighth_term_value_l1011_101189

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n², 
    prove that the 8th term a₈ = 15 -/
theorem eighth_term_value (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h : ∀ n, S n = n^2) : 
    a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1011_101189


namespace NUMINAMATH_CALUDE_photo_arrangements_l1011_101196

/-- The number of ways to arrange n distinct objects taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of students in the front row -/
def front_row : ℕ := 3

/-- The number of students in the back row -/
def back_row : ℕ := 4

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of spaces between boys -/
def spaces_between_boys : ℕ := 5

theorem photo_arrangements :
  (A total_students front_row * A back_row back_row = 5040) ∧
  (A front_row 1 * A back_row 1 * A (total_students - 2) (total_students - 2) = 1440) ∧
  (A (total_students - 2) (total_students - 2) * A 3 3 = 720) ∧
  (A num_boys num_boys * A spaces_between_boys num_girls = 1440) :=
sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1011_101196


namespace NUMINAMATH_CALUDE_one_third_of_six_y_plus_three_l1011_101197

theorem one_third_of_six_y_plus_three (y : ℝ) : (1 / 3) * (6 * y + 3) = 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_six_y_plus_three_l1011_101197


namespace NUMINAMATH_CALUDE_euclidean_algorithm_bound_l1011_101138

/-- The number of divisions performed by the Euclidean algorithm -/
def euclidean_divisions (a b : ℕ) : ℕ := sorry

/-- The number of digits of a natural number in decimal -/
def num_digits (n : ℕ) : ℕ := sorry

theorem euclidean_algorithm_bound (a b : ℕ) (h1 : a > b) (h2 : b > 0) :
  euclidean_divisions a b ≤ 5 * (num_digits b) := by sorry

end NUMINAMATH_CALUDE_euclidean_algorithm_bound_l1011_101138


namespace NUMINAMATH_CALUDE_toms_deck_cost_l1011_101145

/-- Calculate the cost of a deck of cards -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_price : ℚ) (uncommon_price : ℚ) (common_price : ℚ) : ℚ :=
  rare_count * rare_price + uncommon_count * uncommon_price + common_count * common_price

/-- The total cost of Tom's deck is $32 -/
theorem toms_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_toms_deck_cost_l1011_101145


namespace NUMINAMATH_CALUDE_average_of_numbers_divisible_by_4_l1011_101165

theorem average_of_numbers_divisible_by_4 :
  let numbers := (Finset.range 25).filter (fun n => 6 < n + 6 ∧ n + 6 ≤ 30 ∧ (n + 6) % 4 = 0)
  (numbers.sum id) / numbers.card = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_divisible_by_4_l1011_101165


namespace NUMINAMATH_CALUDE_fractional_equation_range_l1011_101164

theorem fractional_equation_range (x m : ℝ) : 
  (x / (x - 1) = m / (2 * x - 2) + 3) →
  (x ≥ 0) →
  (m ≤ 6 ∧ m ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_range_l1011_101164


namespace NUMINAMATH_CALUDE_alpha_is_two_thirds_l1011_101123

theorem alpha_is_two_thirds (α : ℚ) 
  (h1 : 0 < α) 
  (h2 : α < 1) 
  (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : 
  α = 2/3 := by
sorry

end NUMINAMATH_CALUDE_alpha_is_two_thirds_l1011_101123


namespace NUMINAMATH_CALUDE_larry_gave_candies_l1011_101141

/-- Given that Anna starts with 5 candies and ends up with 91 candies after receiving some from Larry,
    prove that Larry gave Anna 86 candies. -/
theorem larry_gave_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 5)
  (h2 : final_candies = 91) :
  final_candies - initial_candies = 86 := by
  sorry

end NUMINAMATH_CALUDE_larry_gave_candies_l1011_101141


namespace NUMINAMATH_CALUDE_only_2222_cannot_form_24_l1011_101172

/-- A hand is a list of four natural numbers representing card values. -/
def Hand := List Nat

/-- Possible arithmetic operations -/
inductive Operation
| Add
| Sub
| Mul
| Div

/-- Apply an operation to two natural numbers -/
def applyOp (op : Operation) (a b : Nat) : Option Nat :=
  match op with
  | Operation.Add => some (a + b)
  | Operation.Sub => if a ≥ b then some (a - b) else none
  | Operation.Mul => some (a * b)
  | Operation.Div => if b ≠ 0 && a % b = 0 then some (a / b) else none

/-- Check if a hand can form 24 using the given operations and rules -/
def canForm24 (hand : Hand) : Prop :=
  ∃ (op1 op2 op3 : Operation) (perm : List Nat),
    perm.length = 4 ∧
    perm.toFinset = hand.toFinset ∧
    (∃ (x y z : Nat),
      applyOp op1 perm[0]! perm[1]! = some x ∧
      applyOp op2 x perm[2]! = some y ∧
      applyOp op3 y perm[3]! = some 24)

theorem only_2222_cannot_form_24 :
  canForm24 [1, 2, 3, 3] ∧
  canForm24 [1, 5, 5, 5] ∧
  canForm24 [3, 3, 3, 3] ∧
  ¬canForm24 [2, 2, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_only_2222_cannot_form_24_l1011_101172


namespace NUMINAMATH_CALUDE_abc_value_l1011_101185

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1011_101185


namespace NUMINAMATH_CALUDE_inverse_square_inequality_l1011_101184

theorem inverse_square_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x ≤ y) :
  1 / y ^ 2 ≤ 1 / x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_inequality_l1011_101184


namespace NUMINAMATH_CALUDE_remainder_theorem_l1011_101148

def q (x : ℝ) : ℝ := 2*x^6 - 3*x^4 + 5*x^2 + 3

theorem remainder_theorem (q : ℝ → ℝ) (a : ℝ) :
  q a = (q 2) → q (-2) = 103 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1011_101148


namespace NUMINAMATH_CALUDE_probability_at_least_four_successes_l1011_101119

theorem probability_at_least_four_successes (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 3/5) :
  let binomial := fun (k : ℕ) => n.choose k * p^k * (1 - p)^(n - k)
  binomial 4 + binomial 5 = 1053/3125 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_four_successes_l1011_101119
