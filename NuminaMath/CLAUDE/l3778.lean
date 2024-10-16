import Mathlib

namespace NUMINAMATH_CALUDE_total_plums_eq_27_l3778_377827

/-- The number of plums Alyssa picked -/
def alyssas_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jasons_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := alyssas_plums + jasons_plums

theorem total_plums_eq_27 : total_plums = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_plums_eq_27_l3778_377827


namespace NUMINAMATH_CALUDE_walker_catch_up_equations_l3778_377817

theorem walker_catch_up_equations 
  (good_efficiency bad_efficiency initial_lead : ℕ) 
  (h_efficiency : good_efficiency > bad_efficiency) 
  (h_initial_lead : initial_lead > 0) : 
  ∃ (x y : ℚ), 
    x - y = initial_lead ∧ 
    x = (good_efficiency : ℚ) / bad_efficiency * y ∧ 
    x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_walker_catch_up_equations_l3778_377817


namespace NUMINAMATH_CALUDE_expected_sum_of_rook_positions_l3778_377870

/-- Represents a chessboard with 64 fields -/
def ChessboardSize : ℕ := 64

/-- Number of rooks placed on the board -/
def NumRooks : ℕ := 6

/-- Expected value of a single randomly chosen position -/
def ExpectedSinglePosition : ℚ := (ChessboardSize + 1) / 2

/-- Theorem: The expected value of the sum of positions of NumRooks rooks 
    on a chessboard of size ChessboardSize is NumRooks * ExpectedSinglePosition -/
theorem expected_sum_of_rook_positions :
  NumRooks * ExpectedSinglePosition = 195 := by sorry

end NUMINAMATH_CALUDE_expected_sum_of_rook_positions_l3778_377870


namespace NUMINAMATH_CALUDE_investment_worth_l3778_377809

def investment_problem (initial_investment : ℚ) (months : ℕ) (monthly_earnings : ℚ) : Prop :=
  let total_earnings := monthly_earnings * months
  let current_worth := initial_investment + total_earnings
  (months = 5) ∧
  (monthly_earnings = 12) ∧
  (total_earnings = 2 * initial_investment) ∧
  (current_worth = 90)

theorem investment_worth :
  ∃ (initial_investment : ℚ), investment_problem initial_investment 5 12 :=
by sorry

end NUMINAMATH_CALUDE_investment_worth_l3778_377809


namespace NUMINAMATH_CALUDE_first_cross_fraction_solution_second_cross_fraction_solution_third_cross_fraction_solution_l3778_377814

/-- Definition of a cross fraction equation -/
def is_cross_fraction_equation (m n x : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧ x + m * n / x = m + n

/-- Theorem for the first cross fraction equation -/
theorem first_cross_fraction_solution :
  ∀ x₁ x₂ : ℝ, is_cross_fraction_equation (-3) (-4) x₁ ∧ is_cross_fraction_equation (-3) (-4) x₂ →
  (x₁ = -3 ∧ x₂ = -4) ∨ (x₁ = -4 ∧ x₂ = -3) :=
sorry

/-- Theorem for the second cross fraction equation -/
theorem second_cross_fraction_solution :
  ∀ a b : ℝ, is_cross_fraction_equation a b a ∧ is_cross_fraction_equation a b b →
  b / a + a / b + 1 = -31 / 6 :=
sorry

/-- Theorem for the third cross fraction equation -/
theorem third_cross_fraction_solution :
  ∀ k x₁ x₂ : ℝ, k > 2 → x₁ > x₂ →
  is_cross_fraction_equation (2023 * k - 2022) 1 x₁ ∧ is_cross_fraction_equation (2023 * k - 2022) 1 x₂ →
  (x₁ + 4044) / x₂ = 2022 :=
sorry

end NUMINAMATH_CALUDE_first_cross_fraction_solution_second_cross_fraction_solution_third_cross_fraction_solution_l3778_377814


namespace NUMINAMATH_CALUDE_exists_special_sequence_l3778_377822

/-- A sequence of positive integers satisfying the required properties -/
def SpecialSequence : Type :=
  ℕ → ℕ+

/-- The property that a number has no square factors other than 1 -/
def HasNoSquareFactors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

/-- The main theorem stating the existence of the special sequence -/
theorem exists_special_sequence :
  ∃ (seq : SpecialSequence),
    (∀ i j : ℕ, i < j → seq i < seq j) ∧
    (∀ i j : ℕ, i ≠ j → HasNoSquareFactors ((seq i).val + (seq j).val)) := by
  sorry


end NUMINAMATH_CALUDE_exists_special_sequence_l3778_377822


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l3778_377866

/-- Given two points on the parabola y = 2x^2 with perpendicular tangents, 
    their intersection point has y-coordinate -1/2 -/
theorem intersection_of_perpendicular_tangents_on_parabola 
  (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 2 * a^2)
  let B : ℝ × ℝ := (b, 2 * b^2)
  let tangent_A (x : ℝ) := 4 * a * x - 2 * a^2
  let tangent_B (x : ℝ) := 4 * b * x - 2 * b^2
  -- Condition: A and B are on the parabola y = 2x^2
  -- Condition: Tangents at A and B are perpendicular
  4 * a * 4 * b = -1 →
  -- Conclusion: The y-coordinate of the intersection point P is -1/2
  ∃ x, tangent_A x = tangent_B x ∧ tangent_A x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l3778_377866


namespace NUMINAMATH_CALUDE_circle_equation_l3778_377890

/-- The standard equation of a circle with center (-3, 4) and radius √5 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let radius : ℝ := Real.sqrt 5
  (x + 3)^2 + (y - 4)^2 = 5 ↔
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3778_377890


namespace NUMINAMATH_CALUDE_remainder_512_210_mod_13_l3778_377828

theorem remainder_512_210_mod_13 : 512^210 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_512_210_mod_13_l3778_377828


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l3778_377892

/-- Given a rhombus with one diagonal of length 70 meters and an area of 5600 square meters,
    the other diagonal has a length of 160 meters. -/
theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 70 → area = 5600 → area = (d1 * d2) / 2 → d2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l3778_377892


namespace NUMINAMATH_CALUDE_inverse_f_negative_three_l3778_377848

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_negative_three :
  (Function.invFun f) (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_negative_three_l3778_377848


namespace NUMINAMATH_CALUDE_max_reflections_before_target_angle_max_reflections_is_optimal_l3778_377811

/-- The angle between the two reflecting lines in degrees -/
def angle_between_lines : ℝ := 5

/-- The target angle of incidence in degrees -/
def target_angle : ℝ := 85

/-- The maximum number of reflections -/
def max_reflections : ℕ := 17

theorem max_reflections_before_target_angle :
  ∀ n : ℕ, n * angle_between_lines ≤ target_angle ↔ n ≤ max_reflections :=
by sorry

theorem max_reflections_is_optimal :
  (max_reflections + 1) * angle_between_lines > target_angle :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_before_target_angle_max_reflections_is_optimal_l3778_377811


namespace NUMINAMATH_CALUDE_inequality_preservation_l3778_377898

theorem inequality_preservation (a b : ℝ) (h : a > b) : a + 1 > b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3778_377898


namespace NUMINAMATH_CALUDE_range_of_m_l3778_377885

theorem range_of_m (x y m : ℝ) (h1 : x^2 + 4*y^2*(m^2 + 3*m)*x*y = 0) (h2 : x*y ≠ 0) : -4 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3778_377885


namespace NUMINAMATH_CALUDE_share_calculation_l3778_377838

theorem share_calculation (total : ℕ) (a b c : ℕ) : 
  total = 770 →
  a = b + 40 →
  c = a + 30 →
  total = a + b + c →
  b = 220 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l3778_377838


namespace NUMINAMATH_CALUDE_finance_marketing_specialization_contradiction_l3778_377807

theorem finance_marketing_specialization_contradiction 
  (finance_percent1 : ℝ) 
  (finance_percent2 : ℝ) 
  (marketing_percent : ℝ) 
  (h1 : finance_percent1 = 88) 
  (h2 : marketing_percent = 76) 
  (h3 : finance_percent2 = 90) 
  (h4 : 0 ≤ finance_percent1 ∧ finance_percent1 ≤ 100) 
  (h5 : 0 ≤ finance_percent2 ∧ finance_percent2 ≤ 100) 
  (h6 : 0 ≤ marketing_percent ∧ marketing_percent ≤ 100) :
  finance_percent1 ≠ finance_percent2 := by
  sorry

end NUMINAMATH_CALUDE_finance_marketing_specialization_contradiction_l3778_377807


namespace NUMINAMATH_CALUDE_tangent_at_negative_one_range_of_a_l3778_377806

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the condition for the shared tangent line
def shared_tangent (a : ℝ) (x₁ : ℝ) : Prop :=
  ∃ (x₂ : ℝ), f' x₁ * (x₂ - x₁) + f x₁ = g a x₂ ∧ f' x₁ = 2 * x₂

-- Theorem for part 1
theorem tangent_at_negative_one (a : ℝ) :
  shared_tangent a (-1) → a = 3 := by sorry

-- Theorem for part 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x₁ : ℝ, shared_tangent a x₁) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_tangent_at_negative_one_range_of_a_l3778_377806


namespace NUMINAMATH_CALUDE_marbleCombinations_eq_twelve_l3778_377893

/-- The number of ways to select 4 marbles from a set of 5 indistinguishable red marbles,
    4 indistinguishable blue marbles, and 2 indistinguishable black marbles -/
def marbleCombinations : ℕ :=
  let red := 5
  let blue := 4
  let black := 2
  let totalSelect := 4
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 + t.2.1 + t.2.2 = totalSelect ∧ 
    t.1 ≤ red ∧ 
    t.2.1 ≤ blue ∧ 
    t.2.2 ≤ black
  ) (Finset.product (Finset.range (red + 1)) (Finset.product (Finset.range (blue + 1)) (Finset.range (black + 1))))).card

theorem marbleCombinations_eq_twelve : marbleCombinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_marbleCombinations_eq_twelve_l3778_377893


namespace NUMINAMATH_CALUDE_rectangle_midpoint_angle_equality_l3778_377849

-- Define the rectangle ABCD
variable (A B C D : Point)

-- Define the property of being a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the midpoint property
def is_midpoint (M A D : Point) : Prop := sorry

-- Define a point on the extension of a line segment
def on_extension (P D C : Point) : Prop := sorry

-- Define the intersection of two lines
def intersection (Q P M A C : Point) : Prop := sorry

-- Define the angle equality
def angle_eq (Q N M P : Point) : Prop := sorry

-- State the theorem
theorem rectangle_midpoint_angle_equality 
  (h_rect : is_rectangle A B C D)
  (h_midpoint_M : is_midpoint M A D)
  (h_midpoint_N : is_midpoint N B C)
  (h_extension_P : on_extension P D C)
  (h_intersection_Q : intersection Q P M A C) :
  angle_eq Q N M P :=
sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_angle_equality_l3778_377849


namespace NUMINAMATH_CALUDE_infinite_numbers_with_equal_digit_sum_l3778_377895

/-- Given a natural number, returns the sum of its digits in decimal representation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number contains the digit 0 in its decimal representation -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_numbers_with_equal_digit_sum (k : ℕ) :
  ∃ (T : Set ℕ), Set.Infinite T ∧ ∀ t ∈ T,
    ¬contains_zero t ∧ sum_of_digits t = sum_of_digits (k * t) := by
  sorry

end NUMINAMATH_CALUDE_infinite_numbers_with_equal_digit_sum_l3778_377895


namespace NUMINAMATH_CALUDE_lily_break_time_l3778_377872

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  words_per_minute : ℕ
  minutes_before_break : ℕ
  total_minutes : ℕ
  total_words : ℕ

/-- Calculates the break time in minutes for a given typing scenario -/
def calculate_break_time (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that Lily's break time is 2 minutes -/
theorem lily_break_time :
  let lily_scenario : TypingScenario := {
    words_per_minute := 15,
    minutes_before_break := 10,
    total_minutes := 19,
    total_words := 255
  }
  calculate_break_time lily_scenario = 2 :=
by sorry

end NUMINAMATH_CALUDE_lily_break_time_l3778_377872


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3778_377860

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3778_377860


namespace NUMINAMATH_CALUDE_prime_power_sum_condition_l3778_377868

theorem prime_power_sum_condition (n : ℕ) :
  Nat.Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_condition_l3778_377868


namespace NUMINAMATH_CALUDE_bucket_leak_problem_l3778_377877

/-- Converts gallons to quarts -/
def gallons_to_quarts (g : ℝ) : ℝ := 4 * g

/-- Calculates the amount of water leaked given initial and remaining amounts -/
def water_leaked (initial : ℝ) (remaining : ℝ) : ℝ := initial - remaining

theorem bucket_leak_problem (initial : ℝ) (remaining_gallons : ℝ) 
  (h1 : initial = 4) 
  (h2 : remaining_gallons = 0.33) : 
  water_leaked initial (gallons_to_quarts remaining_gallons) = 2.68 := by
  sorry

#eval water_leaked 4 (gallons_to_quarts 0.33)

end NUMINAMATH_CALUDE_bucket_leak_problem_l3778_377877


namespace NUMINAMATH_CALUDE_midpoint_barycentric_coords_l3778_377829

/-- Barycentric coordinates of a point -/
structure BarycentricCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given two points in barycentric coordinates, compute the midpoint -/
def midpoint_barycentric (M N : BarycentricCoord) : Prop :=
  let m := M.x + M.y + M.z
  let n := N.x + N.y + N.z
  ∃ (k : ℝ) (S : BarycentricCoord), k ≠ 0 ∧
    S.x = k * (M.x / (2 * m) + N.x / (2 * n)) ∧
    S.y = k * (M.y / (2 * m) + N.y / (2 * n)) ∧
    S.z = k * (M.z / (2 * m) + N.z / (2 * n))

theorem midpoint_barycentric_coords (M N : BarycentricCoord) : 
  midpoint_barycentric M N := by sorry

end NUMINAMATH_CALUDE_midpoint_barycentric_coords_l3778_377829


namespace NUMINAMATH_CALUDE_acute_angle_x_l3778_377813

theorem acute_angle_x (x : Real) (h : 0 < x ∧ x < π / 2) 
  (eq : Real.sin (3 * π / 5) * Real.cos x + Real.cos (2 * π / 5) * Real.sin x = Real.sqrt 3 / 2) : 
  x = 4 * π / 15 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_x_l3778_377813


namespace NUMINAMATH_CALUDE_opposite_of_miss_both_is_hit_at_least_once_l3778_377835

-- Define the possible outcomes of a single shot
inductive ShotOutcome
  | Hit
  | Miss

-- Define the type for a two-shot sequence
def TwoShots := (ShotOutcome × ShotOutcome)

-- Define the event of missing both shots
def MissBoth (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Miss ∧ shots.2 = ShotOutcome.Miss

-- Define the event of hitting at least once
def HitAtLeastOnce (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Hit ∨ shots.2 = ShotOutcome.Hit

-- Theorem: The opposite of missing both is hitting at least once
theorem opposite_of_miss_both_is_hit_at_least_once :
  ∀ (shots : TwoShots), ¬(MissBoth shots) ↔ HitAtLeastOnce shots :=
by sorry


end NUMINAMATH_CALUDE_opposite_of_miss_both_is_hit_at_least_once_l3778_377835


namespace NUMINAMATH_CALUDE_taxi_trip_length_l3778_377852

theorem taxi_trip_length 
  (initial_fee : ℝ) 
  (additional_charge : ℝ) 
  (segment_length : ℝ) 
  (total_charge : ℝ) : 
  initial_fee = 2.25 →
  additional_charge = 0.15 →
  segment_length = 2/5 →
  total_charge = 3.60 →
  ∃ (trip_length : ℝ), 
    trip_length = 3.6 ∧ 
    total_charge = initial_fee + (trip_length / segment_length) * additional_charge :=
by sorry

end NUMINAMATH_CALUDE_taxi_trip_length_l3778_377852


namespace NUMINAMATH_CALUDE_parallel_implies_x_half_perpendicular_implies_x_two_or_neg_two_l3778_377816

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define u and v
def u (x : ℝ) : ℝ × ℝ := a + b x
def v (x : ℝ) : ℝ × ℝ := a - b x

-- Helper function to check if two vectors are parallel
def isParallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v1.1 * k = v2.1 ∧ v1.2 * k = v2.2

-- Helper function to check if two vectors are perpendicular
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem for part I
theorem parallel_implies_x_half :
  ∀ x : ℝ, isParallel (u x) (v x) → x = 1/2 := by sorry

-- Theorem for part II
theorem perpendicular_implies_x_two_or_neg_two :
  ∀ x : ℝ, isPerpendicular (u x) (v x) → x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_x_half_perpendicular_implies_x_two_or_neg_two_l3778_377816


namespace NUMINAMATH_CALUDE_pentagon_arrangement_exists_l3778_377804

/-- Represents a pentagon arrangement of natural numbers -/
def PentagonArrangement := Fin 5 → ℕ

/-- Checks if two numbers are coprime -/
def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if two numbers have a common divisor greater than 1 -/
def have_common_divisor (a b : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

/-- Checks if the given arrangement satisfies the conditions -/
def is_valid_arrangement (arr : PentagonArrangement) : Prop :=
  (∀ i : Fin 5, are_coprime (arr i) (arr ((i + 1) % 5))) ∧
  (∀ i : Fin 5, have_common_divisor (arr i) (arr ((i + 2) % 5)))

/-- The main theorem: there exists a valid pentagon arrangement -/
theorem pentagon_arrangement_exists : ∃ arr : PentagonArrangement, is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_pentagon_arrangement_exists_l3778_377804


namespace NUMINAMATH_CALUDE_decimal_division_to_percentage_l3778_377862

theorem decimal_division_to_percentage : (0.15 / 0.005) * 100 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_to_percentage_l3778_377862


namespace NUMINAMATH_CALUDE_marge_personal_spending_l3778_377887

/-- Calculates Marge's personal spending amount after one year --/
def personal_spending_after_one_year (
  lottery_winnings : ℝ)
  (tax_rate : ℝ)
  (mortgage_rate : ℝ)
  (retirement_rate : ℝ)
  (retirement_interest : ℝ)
  (college_rate : ℝ)
  (savings : ℝ)
  (stock_investment_rate : ℝ)
  (stock_return : ℝ) : ℝ :=
  let after_tax := lottery_winnings * (1 - tax_rate)
  let after_mortgage := after_tax * (1 - mortgage_rate)
  let after_retirement := after_mortgage * (1 - retirement_rate)
  let after_college := after_retirement * (1 - college_rate)
  let retirement_growth := after_mortgage * retirement_rate * retirement_interest
  let stock_investment := savings * stock_investment_rate
  let stock_growth := stock_investment * stock_return
  after_college + (savings - stock_investment) + retirement_growth + stock_growth

/-- Theorem stating that Marge's personal spending after one year is $5,363 --/
theorem marge_personal_spending :
  personal_spending_after_one_year 50000 0.6 0.5 0.4 0.05 0.25 1500 0.6 0.07 = 5363 := by
  sorry

end NUMINAMATH_CALUDE_marge_personal_spending_l3778_377887


namespace NUMINAMATH_CALUDE_money_problem_l3778_377812

theorem money_problem (a b : ℚ) : 
  a = 80/7 ∧ b = 40/7 →
  7*a + b < 100 ∧ 4*a - b = 40 ∧ b = (1/2) * a := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3778_377812


namespace NUMINAMATH_CALUDE_expression_simplification_l3778_377858

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 3) :
  (x - 3) / (x - 2) / (x + 2 - 5 / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3778_377858


namespace NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l3778_377888

theorem smallest_multiple_of_42_and_56_not_18 : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (42 ∣ m.val ∧ 56 ∣ m.val) → 18 ∣ m.val) ∧ 
  42 ∣ n.val ∧ 56 ∣ n.val ∧ ¬(18 ∣ n.val) ∧ n.val = 168 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l3778_377888


namespace NUMINAMATH_CALUDE_championship_completion_impossible_l3778_377837

/-- Represents a chess game between two players -/
structure Game where
  player1 : Nat
  player2 : Nat
  deriving Repr

/-- Represents the state of the chess championship -/
structure ChampionshipState where
  numPlayers : Nat
  gamesPlayed : List Game
  deriving Repr

/-- Checks if the championship rules are followed -/
def rulesFollowed (state : ChampionshipState) : Prop :=
  ∀ p1 p2, p1 < state.numPlayers → p2 < state.numPlayers → p1 ≠ p2 →
    let gamesPlayedByP1 := (state.gamesPlayed.filter (λ g => g.player1 = p1 ∨ g.player2 = p1)).length
    let gamesPlayedByP2 := (state.gamesPlayed.filter (λ g => g.player1 = p2 ∨ g.player2 = p2)).length
    (gamesPlayedByP1 : Int) - gamesPlayedByP2 ≤ 1 ∧ gamesPlayedByP2 - gamesPlayedByP1 ≤ 1

/-- Checks if the championship is complete -/
def isComplete (state : ChampionshipState) : Prop :=
  state.gamesPlayed.length = state.numPlayers * (state.numPlayers - 1) / 2

/-- Theorem: There exists a championship state that follows the rules but cannot be completed -/
theorem championship_completion_impossible : ∃ (state : ChampionshipState), 
  rulesFollowed state ∧ ¬∃ (finalState : ChampionshipState), 
    finalState.numPlayers = state.numPlayers ∧ 
    state.gamesPlayed ⊆ finalState.gamesPlayed ∧ 
    rulesFollowed finalState ∧ 
    isComplete finalState :=
sorry

end NUMINAMATH_CALUDE_championship_completion_impossible_l3778_377837


namespace NUMINAMATH_CALUDE_kitten_weight_l3778_377850

/-- The weight of a kitten and two dogs satisfying certain conditions -/
structure AnimalWeights where
  kitten : ℝ
  smallDog : ℝ
  largeDog : ℝ
  total_weight : kitten + smallDog + largeDog = 36
  larger_pair : kitten + largeDog = 2 * smallDog
  smaller_pair : kitten + smallDog = largeDog

/-- The kitten's weight is 6 pounds given the conditions -/
theorem kitten_weight (w : AnimalWeights) : w.kitten = 6 := by
  sorry

end NUMINAMATH_CALUDE_kitten_weight_l3778_377850


namespace NUMINAMATH_CALUDE_sum_of_square_roots_l3778_377803

theorem sum_of_square_roots : 
  Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_l3778_377803


namespace NUMINAMATH_CALUDE_flagpole_height_l3778_377801

/-- The height of a flagpole given specific conditions --/
theorem flagpole_height :
  ∀ (wire_length : ℝ) (wire_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ),
  wire_ground_distance = 5 →
  person_distance = 3 →
  person_height = 1.8 →
  ∃ (pole_height : ℝ),
    pole_height = 4.5 ∧
    pole_height / wire_ground_distance = person_height / (wire_ground_distance - person_distance) :=
by
  sorry


end NUMINAMATH_CALUDE_flagpole_height_l3778_377801


namespace NUMINAMATH_CALUDE_feathers_per_crown_calculation_l3778_377823

/-- Given a total number of feathers and a number of crowns, 
    calculate the number of feathers per crown. -/
def feathers_per_crown (total_feathers : ℕ) (num_crowns : ℕ) : ℕ :=
  (total_feathers + num_crowns - 1) / num_crowns

/-- Theorem stating that given 6538 feathers and 934 crowns, 
    the number of feathers per crown is 7. -/
theorem feathers_per_crown_calculation :
  feathers_per_crown 6538 934 = 7 := by
  sorry


end NUMINAMATH_CALUDE_feathers_per_crown_calculation_l3778_377823


namespace NUMINAMATH_CALUDE_a_less_than_sqrt3b_l3778_377808

theorem a_less_than_sqrt3b (a b : ℤ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : (a + b) ∣ (a * b + 1)) 
  (h4 : (a - b) ∣ (a * b - 1)) : 
  a < Real.sqrt 3 * b := by
sorry

end NUMINAMATH_CALUDE_a_less_than_sqrt3b_l3778_377808


namespace NUMINAMATH_CALUDE_positive_y_solution_l3778_377800

theorem positive_y_solution (x y z : ℝ) 
  (eq1 : x * y = 4 - x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 2 * z)
  (eq3 : x * z = 40 - 5 * x - 2 * z)
  (y_pos : y > 0) :
  y = 2 := by
sorry

end NUMINAMATH_CALUDE_positive_y_solution_l3778_377800


namespace NUMINAMATH_CALUDE_rational_sqrt_property_l3778_377815

theorem rational_sqrt_property (A : Set ℝ) : 
  (∃ a b c d : ℝ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → a ≠ b → a ≠ c → b ≠ c → ∃ q : ℚ, (a^2 + b*c : ℝ) = q) →
  ∃ M : ℕ, ∀ a : ℝ, a ∈ A → ∃ q : ℚ, a * Real.sqrt M = q :=
by sorry

end NUMINAMATH_CALUDE_rational_sqrt_property_l3778_377815


namespace NUMINAMATH_CALUDE_abs_a_minus_b_l3778_377836

theorem abs_a_minus_b (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_l3778_377836


namespace NUMINAMATH_CALUDE_sum_f_91_and_neg_91_l3778_377891

/-- A polynomial function of degree 6 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

/-- Theorem: Given f(x) = ax^6 + bx^4 - cx^2 + 3 and f(91) = 1, prove f(91) + f(-91) = 2 -/
theorem sum_f_91_and_neg_91 (a b c : ℝ) (h : f a b c 91 = 1) : f a b c 91 + f a b c (-91) = 2 := by
  sorry

#check sum_f_91_and_neg_91

end NUMINAMATH_CALUDE_sum_f_91_and_neg_91_l3778_377891


namespace NUMINAMATH_CALUDE_seven_digit_divisibility_l3778_377865

theorem seven_digit_divisibility (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →
  (74 * 100000 + A * 10000 + 52 * 100 + B * 10 + 1) % 3 = 0 →
  (326 * 10000 + A * 1000 + B * 100 + 4 * 10 + C) % 3 = 0 →
  C = 1 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_divisibility_l3778_377865


namespace NUMINAMATH_CALUDE_water_addition_proof_l3778_377861

/-- Proves that adding 3 litres of water to 11 litres of 42% alcohol solution results in 33% alcohol mixture -/
theorem water_addition_proof (initial_volume : ℝ) (initial_alcohol_percent : ℝ) 
  (final_alcohol_percent : ℝ) (water_added : ℝ) : 
  initial_volume = 11 →
  initial_alcohol_percent = 0.42 →
  final_alcohol_percent = 0.33 →
  water_added = 3 →
  initial_volume * initial_alcohol_percent = 
    (initial_volume + water_added) * final_alcohol_percent := by
  sorry

#check water_addition_proof

end NUMINAMATH_CALUDE_water_addition_proof_l3778_377861


namespace NUMINAMATH_CALUDE_range_of_a_l3778_377886

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), x > 3 → x > a) ↔ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3778_377886


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3778_377889

theorem magnitude_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((1 - i) / (2 * i + 1)) = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3778_377889


namespace NUMINAMATH_CALUDE_D_value_l3778_377820

/-- The determinant of a matrix with elements |i-j| -/
def D (n : ℕ) : ℚ :=
  let M : Matrix (Fin n) (Fin n) ℚ := λ i j => |i.val - j.val|
  M.det

/-- Theorem stating the value of the determinant D_n -/
theorem D_value (n : ℕ) (h : n > 0) : D n = (-1)^(n-1) * (n-1) * 2^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_D_value_l3778_377820


namespace NUMINAMATH_CALUDE_knowledge_competition_probability_l3778_377879

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The number of preset questions in the competition -/
def total_questions : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := p_correct * p_correct * (1 - p_correct) * p_correct

theorem knowledge_competition_probability :
  prob_four_questions = 0.128 :=
sorry

end NUMINAMATH_CALUDE_knowledge_competition_probability_l3778_377879


namespace NUMINAMATH_CALUDE_t_shirt_cost_is_8_l3778_377859

/-- The cost of a t-shirt in dollars -/
def t_shirt_cost : ℝ := sorry

/-- The total amount Timothy has to spend in dollars -/
def total_budget : ℝ := 50

/-- The cost of a bag in dollars -/
def bag_cost : ℝ := 10

/-- The number of t-shirts Timothy buys -/
def num_t_shirts : ℕ := 2

/-- The number of bags Timothy buys -/
def num_bags : ℕ := 2

/-- The number of key chains Timothy buys -/
def num_key_chains : ℕ := 21

/-- The cost of 3 key chains in dollars -/
def cost_3_key_chains : ℝ := 2

theorem t_shirt_cost_is_8 :
  t_shirt_cost = 8 ∧
  total_budget = num_t_shirts * t_shirt_cost + num_bags * bag_cost +
    (num_key_chains / 3 : ℝ) * cost_3_key_chains :=
by sorry

end NUMINAMATH_CALUDE_t_shirt_cost_is_8_l3778_377859


namespace NUMINAMATH_CALUDE_circle_condition_intersection_condition_l3778_377845

-- Define the equation C
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem 1: Range of m for C to represent a circle
theorem circle_condition (m : ℝ) :
  (∃ x y, C x y m) ∧ (∀ x y, C x y m → (x - 1)^2 + (y - 2)^2 = 5 - m) →
  m < 5 :=
sorry

-- Theorem 2: Value of m when C intersects l with |MN| = 4√5/5
theorem intersection_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, C x₁ y₁ m ∧ C x₂ y₂ m ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4*Real.sqrt 5 / 5)^2) →
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_intersection_condition_l3778_377845


namespace NUMINAMATH_CALUDE_history_score_l3778_377833

theorem history_score (math_score : ℚ) (third_subject_score : ℚ) (average_score : ℚ) :
  math_score = 74 ∧ third_subject_score = 70 ∧ average_score = 75 →
  (math_score + third_subject_score + (3 * average_score - math_score - third_subject_score)) / 3 = average_score :=
by
  sorry

#eval (74 + 70 + (3 * 75 - 74 - 70)) / 3  -- Should evaluate to 75

end NUMINAMATH_CALUDE_history_score_l3778_377833


namespace NUMINAMATH_CALUDE_sample_data_properties_l3778_377825

theorem sample_data_properties (x : Fin 6 → ℝ) (h : ∀ i j : Fin 6, i ≤ j → x i ≤ x j) :
  (let median1 := (x 2 + x 3) / 2
   let median2 := (x 2 + x 3) / 2
   median1 = median2) ∧
  (x 4 - x 1 ≤ x 5 - x 0) :=
by sorry

end NUMINAMATH_CALUDE_sample_data_properties_l3778_377825


namespace NUMINAMATH_CALUDE_cow_count_is_twenty_l3778_377821

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem: In a group where the total number of legs is 40 more than twice
    the number of heads, the number of cows is 20 -/
theorem cow_count_is_twenty (count : AnimalCount) 
    (h : totalLegs count = 2 * totalHeads count + 40) : 
    count.cows = 20 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_twenty_l3778_377821


namespace NUMINAMATH_CALUDE_factor_expression_l3778_377843

theorem factor_expression (a : ℝ) : 53 * a^2 + 159 * a = 53 * a * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3778_377843


namespace NUMINAMATH_CALUDE_isabel_pop_albums_l3778_377873

/-- The number of pop albums Isabel bought -/
def number_of_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (total_songs - country_albums * songs_per_album) / songs_per_album

/-- Theorem stating that Isabel bought 5 pop albums -/
theorem isabel_pop_albums :
  number_of_pop_albums 72 4 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_isabel_pop_albums_l3778_377873


namespace NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_approx_l3778_377844

/-- The perimeter of a semicircular cubicle with radius 14 units is approximately 71.96 units. -/
theorem semicircular_cubicle_perimeter_approx : ∃ (p : ℝ), 
  (abs (p - (28 + π * 14)) < 0.01) ∧ (abs (p - 71.96) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_approx_l3778_377844


namespace NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l3778_377883

/-- The number of triangles formed with a fixed vertex from 8 points on a circle -/
theorem triangles_with_fixed_vertex (n : ℕ) (h : n = 8) : 
  (Nat.choose (n - 1) 2) = 21 := by
  sorry

#check triangles_with_fixed_vertex

end NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l3778_377883


namespace NUMINAMATH_CALUDE_river_width_calculation_l3778_377864

/-- Given a river with depth, flow rate, and volume flow per minute, calculate its width. -/
theorem river_width_calculation (depth : ℝ) (flow_rate_kmph : ℝ) (volume_flow : ℝ) :
  depth = 3 →
  flow_rate_kmph = 2 →
  volume_flow = 3600 →
  (volume_flow / (depth * (flow_rate_kmph * 1000 / 60))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_river_width_calculation_l3778_377864


namespace NUMINAMATH_CALUDE_function_inequality_condition_l3778_377855

theorem function_inequality_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x : ℝ, |x + 0.4| < b → |5 * x - 3 + 1| < a) ↔ b ≤ a / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l3778_377855


namespace NUMINAMATH_CALUDE_charles_total_money_l3778_377802

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of pennies Charles found on his way to school -/
def pennies_found : ℕ := 6

/-- The number of nickels Charles found on his way to school -/
def nickels_found : ℕ := 8

/-- The number of dimes Charles found on his way to school -/
def dimes_found : ℕ := 6

/-- The number of quarters Charles found on his way to school -/
def quarters_found : ℕ := 5

/-- The number of nickels Charles had at home -/
def nickels_at_home : ℕ := 3

/-- The number of dimes Charles had at home -/
def dimes_at_home : ℕ := 12

/-- The number of quarters Charles had at home -/
def quarters_at_home : ℕ := 7

/-- The number of half-dollars Charles had at home -/
def half_dollars_at_home : ℕ := 2

/-- The total amount of money Charles has -/
def total_money : ℚ :=
  penny_value * pennies_found +
  nickel_value * (nickels_found + nickels_at_home) +
  dime_value * (dimes_found + dimes_at_home) +
  quarter_value * (quarters_found + quarters_at_home) +
  half_dollar_value * half_dollars_at_home

theorem charles_total_money :
  total_money = 6.41 := by sorry

end NUMINAMATH_CALUDE_charles_total_money_l3778_377802


namespace NUMINAMATH_CALUDE_concatenated_digits_theorem_l3778_377842

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Concatenation of two natural numbers -/
def concatenate (a b : ℕ) : ℕ := sorry

theorem concatenated_digits_theorem :
  num_digits (concatenate (5^1971) (2^1971)) = 1972 := by sorry

end NUMINAMATH_CALUDE_concatenated_digits_theorem_l3778_377842


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3778_377894

def a : Fin 2 → ℝ := ![3, -2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ b x = k • a) → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3778_377894


namespace NUMINAMATH_CALUDE_line_equation_solution_l3778_377831

/-- The line x = k intersects y = x^2 + 4x + 4 and y = mx + b at two points 4 units apart -/
def intersectionCondition (m b k : ℝ) : Prop :=
  ∃ k, |k^2 + 4*k + 4 - (m*k + b)| = 4

/-- The line y = mx + b passes through the point (2, 8) -/
def passesThroughPoint (m b : ℝ) : Prop :=
  8 = 2*m + b

/-- b is not equal to 0 -/
def bNonZero (b : ℝ) : Prop :=
  b ≠ 0

/-- The theorem stating that given the conditions, the unique solution for the line equation is y = 8x - 8 -/
theorem line_equation_solution (m b : ℝ) :
  (∃ k, intersectionCondition m b k) →
  passesThroughPoint m b →
  bNonZero b →
  m = 8 ∧ b = -8 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_solution_l3778_377831


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3778_377856

theorem binomial_coefficient_equality (n : ℕ+) : 
  (Nat.choose n.val 2 = Nat.choose n.val 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3778_377856


namespace NUMINAMATH_CALUDE_total_candies_l3778_377834

/-- The total number of candies in a jar, given the number of red and blue candies -/
theorem total_candies (red : ℕ) (blue : ℕ) (h1 : red = 145) (h2 : blue = 3264) : 
  red + blue = 3409 :=
by sorry

end NUMINAMATH_CALUDE_total_candies_l3778_377834


namespace NUMINAMATH_CALUDE_candy_probability_l3778_377841

theorem candy_probability (p1 p2 : ℚ) : 
  (3/8 : ℚ) ≤ p1 ∧ p1 ≤ (2/5 : ℚ) ∧ 
  (3/8 : ℚ) ≤ p2 ∧ p2 ≤ (2/5 : ℚ) ∧ 
  p1 = (5/13 : ℚ) ∧ p2 = (7/18 : ℚ) →
  ((3/8 : ℚ) ≤ (5/13 : ℚ) ∧ (5/13 : ℚ) ≤ (2/5 : ℚ)) ∧
  ((3/8 : ℚ) ≤ (7/18 : ℚ) ∧ (7/18 : ℚ) ≤ (2/5 : ℚ)) ∧
  ¬((3/8 : ℚ) ≤ (17/40 : ℚ) ∧ (17/40 : ℚ) ≤ (2/5 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_l3778_377841


namespace NUMINAMATH_CALUDE_distance_to_line_l3778_377851

/-- Given two perpendicular lines and a plane, calculate the distance from a point to one of the lines -/
theorem distance_to_line (m θ ψ : ℝ) (hm : m > 0) (hθ : 0 < θ ∧ θ < π / 2) (hψ : 0 < ψ ∧ ψ < π / 2) :
  ∃ (d : ℝ), d = Real.sqrt (m^2 + (m * Real.sin θ / Real.sin ψ)^2) ∧ d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l3778_377851


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3778_377857

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 = 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 1 = 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 1, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3778_377857


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3778_377867

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3778_377867


namespace NUMINAMATH_CALUDE_expression_evaluation_l3778_377863

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1) = -13 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3778_377863


namespace NUMINAMATH_CALUDE_galia_number_transformation_l3778_377819

theorem galia_number_transformation (k : ℝ) :
  (∃ N : ℝ, ((k * N + N) / N - N = k - 100)) → (∃ N : ℝ, N = 101) :=
by sorry

end NUMINAMATH_CALUDE_galia_number_transformation_l3778_377819


namespace NUMINAMATH_CALUDE_worker_b_time_l3778_377884

/-- Given workers a, b, and c, and their work rates, prove that b alone takes 6 hours to complete the work. -/
theorem worker_b_time (a b c : ℝ) : 
  a = 1/3 →                -- a can do the work in 3 hours
  b + c = 1/3 →            -- b and c together can do the work in 3 hours
  a + c = 1/2 →            -- a and c together can do the work in 2 hours
  1/b = 6                  -- b alone takes 6 hours to do the work
:= by sorry

end NUMINAMATH_CALUDE_worker_b_time_l3778_377884


namespace NUMINAMATH_CALUDE_unique_n_existence_and_value_l3778_377881

theorem unique_n_existence_and_value : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_existence_and_value_l3778_377881


namespace NUMINAMATH_CALUDE_money_distribution_l3778_377832

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 350)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3778_377832


namespace NUMINAMATH_CALUDE_combination_sum_equality_l3778_377840

theorem combination_sum_equality (n k m : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Nat.choose n m) + 
  (Finset.sum (Finset.range k) (fun i => Nat.choose k (i + 1) * Nat.choose n (m - (i + 1)))) + 
  (Nat.choose n (m - k)) = 
  Nat.choose (n + k) m := by sorry

end NUMINAMATH_CALUDE_combination_sum_equality_l3778_377840


namespace NUMINAMATH_CALUDE_min_distance_squared_l3778_377805

/-- The line on which point P(x,y) moves --/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The distance function from point (x,y) to (2,2) --/
def distance_squared (x y : ℝ) : ℝ := (x - 2)^2 + (y - 2)^2

/-- Theorem stating the minimum value of the distance function --/
theorem min_distance_squared :
  ∃ (min : ℝ), min = 1/2 ∧ 
  (∀ x y : ℝ, line x y → distance_squared x y ≥ min) ∧
  (∃ x y : ℝ, line x y ∧ distance_squared x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l3778_377805


namespace NUMINAMATH_CALUDE_stamp_collection_duration_l3778_377869

/-- Proves the collection duration for two stamp collectors given their collection rates and total stamps --/
theorem stamp_collection_duration (total_stamps : ℕ) (rate1 rate2 : ℕ) (extra_weeks : ℕ) : 
  total_stamps = 300 →
  rate1 = 5 →
  rate2 = 3 →
  extra_weeks = 20 →
  ∃ (weeks1 weeks2 : ℕ), 
    weeks1 = 30 ∧
    weeks2 = 50 ∧
    weeks2 = weeks1 + extra_weeks ∧
    total_stamps = rate1 * weeks1 + rate2 * weeks2 :=
by sorry


end NUMINAMATH_CALUDE_stamp_collection_duration_l3778_377869


namespace NUMINAMATH_CALUDE_max_value_theorem_l3778_377875

theorem max_value_theorem (x y : ℝ) : 
  (2*x + 3*y + 5) / Real.sqrt (x^2 + 2*y^2 + 2) ≤ Real.sqrt 38 ∧ 
  ∃ (x₀ y₀ : ℝ), (2*x₀ + 3*y₀ + 5) / Real.sqrt (x₀^2 + 2*y₀^2 + 2) = Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3778_377875


namespace NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l3778_377846

/-- Given a square and a regular octagon with equal perimeters, 
    if the area of the square is 16, then the area of the octagon is 8 + 4√2 -/
theorem octagon_area_given_equal_perimeter_and_square_area (a b : ℝ) : 
  a > 0 → b > 0 → 4 * a = 8 * b → a^2 = 16 → 
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l3778_377846


namespace NUMINAMATH_CALUDE_emma_remaining_amount_l3778_377876

def calculate_remaining_amount (initial_amount furniture_cost fraction_given : ℚ) : ℚ :=
  let remaining_after_furniture := initial_amount - furniture_cost
  let amount_given := fraction_given * remaining_after_furniture
  remaining_after_furniture - amount_given

theorem emma_remaining_amount :
  calculate_remaining_amount 2000 400 (3/4) = 400 := by
  sorry

end NUMINAMATH_CALUDE_emma_remaining_amount_l3778_377876


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3778_377847

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, 0, 1}
def B : Set Int := {-1, 0, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-2, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3778_377847


namespace NUMINAMATH_CALUDE_pentagon_area_l3778_377897

/-- Given points on a coordinate plane, prove the area of the pentagon formed by these points and their intersection. -/
theorem pentagon_area (A B D E C : ℝ × ℝ) : 
  A = (9, 1) →
  B = (2, 0) →
  D = (1, 5) →
  E = (9, 7) →
  (C.1 - A.1) / (D.1 - A.1) = (C.2 - A.2) / (D.2 - A.2) →
  (C.1 - B.1) / (E.1 - B.1) = (C.2 - B.2) / (E.2 - B.2) →
  abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * E.2 + E.1 * A.2) -
       (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * E.1 + E.2 * A.1)) / 2 = 33 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_area_l3778_377897


namespace NUMINAMATH_CALUDE_midpoint_calculation_l3778_377810

/-- Given two points A and B in a 2D plane, proves that 3x - 5y = -18,
    where (x, y) is the midpoint of AB. -/
theorem midpoint_calculation (A B : ℝ × ℝ) (h1 : A = (-8, 15)) (h2 : B = (16, -3)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -18 := by
sorry

end NUMINAMATH_CALUDE_midpoint_calculation_l3778_377810


namespace NUMINAMATH_CALUDE_sum_of_digits_499849_l3778_377853

def number : Nat := 499849

def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (fun c => c.toNat - '0'.toNat)
  digits.sum

theorem sum_of_digits_499849 :
  sumOfDigits number = 43 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_499849_l3778_377853


namespace NUMINAMATH_CALUDE_inverse_of_A_l3778_377818

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -3; -2, 1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![-1/2, -3/2; -1, -2]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3778_377818


namespace NUMINAMATH_CALUDE_symmetry_and_line_equation_l3778_377899

/-- The curve on which points P and Q lie -/
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line of symmetry for points P and Q -/
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

/-- The condition satisfied by the coordinates of P and Q -/
def coordinate_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- The theorem stating the value of m and the equation of line PQ -/
theorem symmetry_and_line_equation 
  (x₁ y₁ x₂ y₂ m : ℝ) 
  (h_curve_P : curve x₁ y₁)
  (h_curve_Q : curve x₂ y₂)
  (h_symmetry : symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂)
  (h_condition : coordinate_condition x₁ y₁ x₂ y₂) :
  m = -1 ∧ ∀ (x y : ℝ), y = -x + 1 ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_line_equation_l3778_377899


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3778_377854

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem max_value_of_expression (x y z : ℕ) 
  (h_two_digit_x : is_two_digit x)
  (h_two_digit_y : is_two_digit y)
  (h_two_digit_z : is_two_digit z)
  (h_mean : (x + y + z) / 3 = 60) :
  (∀ a b c : ℕ, is_two_digit a → is_two_digit b → is_two_digit c → 
    (a + b + c) / 3 = 60 → (a + b) / c ≤ 17) ∧
  (∃ a b c : ℕ, is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ 
    (a + b + c) / 3 = 60 ∧ (a + b) / c = 17) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3778_377854


namespace NUMINAMATH_CALUDE_lindas_savings_l3778_377878

theorem lindas_savings (savings : ℝ) 
  (h1 : savings * (1/4) = 200)
  (h2 : ∃ (furniture_cost : ℝ), furniture_cost = savings * (3/4) ∧ 
        furniture_cost * 0.8 = savings * (3/4))
  : savings = 800 := by
sorry

end NUMINAMATH_CALUDE_lindas_savings_l3778_377878


namespace NUMINAMATH_CALUDE_tom_marble_pairs_l3778_377839

/-- The number of distinct pairs of marbles Tom can choose -/
def distinct_pairs : ℕ := 8

/-- The number of red marbles Tom has -/
def red_marbles : ℕ := 1

/-- The number of blue marbles Tom has -/
def blue_marbles : ℕ := 1

/-- The number of yellow marbles Tom has -/
def yellow_marbles : ℕ := 4

/-- The number of green marbles Tom has -/
def green_marbles : ℕ := 2

/-- Theorem stating that the number of distinct pairs of marbles Tom can choose is 8 -/
theorem tom_marble_pairs :
  distinct_pairs = 8 :=
by sorry

end NUMINAMATH_CALUDE_tom_marble_pairs_l3778_377839


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3778_377871

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n) →  -- geometric sequence
  (a 1)^2 - 10*(a 1) + 16 = 0 →  -- a_1 is a root
  (a 19)^2 - 10*(a 19) + 16 = 0 →  -- a_19 is a root
  a 8 * a 10 * a 12 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3778_377871


namespace NUMINAMATH_CALUDE_range_of_Z_l3778_377826

-- Define x and y as real numbers
variable (x y : ℝ)

-- Define Z as the sum of x and y
def Z : ℝ := x + y

-- Theorem stating that the range of Z is (0,4)
theorem range_of_Z : ∀ z : ℝ, (∃ x y : ℝ, Z x y = z) ↔ 0 < z ∧ z < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_Z_l3778_377826


namespace NUMINAMATH_CALUDE_triangle_segment_length_l3778_377882

structure Triangle :=
  (A B C : ℝ × ℝ)

def angleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  sorry  -- Definition of angle bisector

theorem triangle_segment_length 
  (ABC : Triangle) 
  (D : ℝ × ℝ) 
  (h_bisector : angleBisector ABC D)
  (h_AD : dist D ABC.A = 15)
  (h_DC : dist D ABC.C = 45)
  (h_DB : dist D ABC.B = 24) :
  dist ABC.A ABC.B = 39 :=
sorry

#check triangle_segment_length

end NUMINAMATH_CALUDE_triangle_segment_length_l3778_377882


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_two_second_or_fourth_quadrant_implies_a_range_l3778_377824

-- Define the complex number z
def z (a : ℝ) : ℂ := (a^2 - a - 2 : ℝ) + (a^2 - 3*a - 4 : ℝ)*Complex.I

-- Part 1: z is a pure imaginary number implies a = 2
theorem pure_imaginary_implies_a_eq_two :
  ∀ a : ℝ, (z a).re = 0 → (z a).im ≠ 0 → a = 2 := by sorry

-- Part 2: z in second or fourth quadrant implies 2 < a < 4
theorem second_or_fourth_quadrant_implies_a_range :
  ∀ a : ℝ, (z a).re * (z a).im < 0 → 2 < a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_two_second_or_fourth_quadrant_implies_a_range_l3778_377824


namespace NUMINAMATH_CALUDE_exists_perpendicular_k_line_intersects_circle_chord_length_when_k_neg_one_l3778_377896

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line l₀
def line_l0 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- Statement 1: Perpendicularity condition
theorem exists_perpendicular_k : ∃ k : ℝ, ∀ x y : ℝ, 
  line_l k x y → line_l0 x y → k * (1/2) = -1 :=
sorry

-- Statement 2: Intersection of line l and circle O
theorem line_intersects_circle : ∀ k : ℝ, ∃ x y : ℝ, 
  line_l k x y ∧ circle_O x y :=
sorry

-- Statement 3: Chord length when k = -1
theorem chord_length_when_k_neg_one : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_l (-1) x₁ y₁ ∧ line_l (-1) x₂ y₂ ∧ 
    circle_O x₁ y₁ ∧ circle_O x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 28 :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_k_line_intersects_circle_chord_length_when_k_neg_one_l3778_377896


namespace NUMINAMATH_CALUDE_N_inverse_proof_l3778_377874

def N : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, -1; 4, -3, 2; -3, 5, 0]

theorem N_inverse_proof :
  let N_inv : Matrix (Fin 3) (Fin 3) ℝ := !![5/21, 5/14, -1/21; 3/14, 1/14, 5/42; -1/21, -19/42, 11/42]
  N * N_inv = 1 ∧ N_inv * N = 1 := by sorry

end NUMINAMATH_CALUDE_N_inverse_proof_l3778_377874


namespace NUMINAMATH_CALUDE_identify_brother_l3778_377830

-- Define the brothers
inductive Brother : Type
| Tweedledum : Brother
| Tweedledee : Brother

-- Define the card suits
inductive Suit : Type
| Red : Suit
| Black : Suit

-- Define the statement made by one of the brothers
def statement (b : Brother) (s : Suit) : Prop :=
  b = Brother.Tweedledum ∨ s = Suit.Black

-- Define the rule that someone with a black card cannot make a true statement
axiom black_card_rule : ∀ (b : Brother) (s : Suit), 
  s = Suit.Black → ¬(statement b s)

-- Theorem to prove
theorem identify_brother : 
  ∃ (b : Brother) (s : Suit), statement b s ∧ b = Brother.Tweedledum ∧ s = Suit.Red :=
sorry

end NUMINAMATH_CALUDE_identify_brother_l3778_377830


namespace NUMINAMATH_CALUDE_total_length_is_24_l3778_377880

/-- Represents a geometric figure with perpendicular adjacent sides -/
structure GeometricFigure where
  bottom : ℝ
  right : ℝ
  top_left : ℝ
  top_right : ℝ
  middle_horizontal : ℝ
  middle_vertical : ℝ
  left : ℝ

/-- Calculates the total length of visible segments in the transformed figure -/
def total_length_after_transform (fig : GeometricFigure) : ℝ :=
  fig.bottom + (fig.right - 2) + (fig.top_left - 3) + fig.left

/-- Theorem stating that the total length of segments in Figure 2 is 24 units -/
theorem total_length_is_24 (fig : GeometricFigure) 
  (h1 : fig.bottom = 5)
  (h2 : fig.right = 10)
  (h3 : fig.top_left = 4)
  (h4 : fig.top_right = 4)
  (h5 : fig.middle_horizontal = 3)
  (h6 : fig.middle_vertical = 3)
  (h7 : fig.left = 10) :
  total_length_after_transform fig = 24 := by
  sorry


end NUMINAMATH_CALUDE_total_length_is_24_l3778_377880
