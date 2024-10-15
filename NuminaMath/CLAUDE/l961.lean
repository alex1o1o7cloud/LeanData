import Mathlib

namespace NUMINAMATH_CALUDE_power_product_six_three_l961_96107

theorem power_product_six_three : (6 : ℕ)^5 * (3 : ℕ)^5 = 1889568 := by
  sorry

end NUMINAMATH_CALUDE_power_product_six_three_l961_96107


namespace NUMINAMATH_CALUDE_unique_solution_for_pure_imaginary_l961_96144

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number constructed from m -/
def complex_number (m : ℝ) : ℂ :=
  ⟨m^2 - 5*m + 6, m^2 - 3*m⟩

theorem unique_solution_for_pure_imaginary :
  ∃! m : ℝ, is_pure_imaginary (complex_number m) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_pure_imaginary_l961_96144


namespace NUMINAMATH_CALUDE_remainder_of_n_mod_500_l961_96192

/-- The set S containing elements from 1 to 12 -/
def S : Finset ℕ := Finset.range 12

/-- The number of sets of two non-empty disjoint subsets of S -/
def n : ℕ := ((3^12 - 2 * 2^12 + 1) / 2 : ℕ)

/-- Theorem stating that the remainder of n divided by 500 is 125 -/
theorem remainder_of_n_mod_500 : n % 500 = 125 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_mod_500_l961_96192


namespace NUMINAMATH_CALUDE_sandbox_width_l961_96153

/-- A sandbox is a rectangle with a specific perimeter and length-width relationship -/
structure Sandbox where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 30
  length_eq : length = 2 * width

theorem sandbox_width (s : Sandbox) : s.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_width_l961_96153


namespace NUMINAMATH_CALUDE_quadratic_polynomials_intersection_l961_96193

-- Define the type for quadratic polynomials
def QuadraticPolynomial := ℝ → ℝ

-- Define a function to check if three polynomials have pairwise distinct leading coefficients
def pairwiseDistinctLeadingCoeff (f g h : QuadraticPolynomial) : Prop :=
  ∃ (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ),
    (∀ x, f x = a₁ * x^2 + b₁ * x + c₁) ∧
    (∀ x, g x = a₂ * x^2 + b₂ * x + c₂) ∧
    (∀ x, h x = a₃ * x^2 + b₃ * x + c₃) ∧
    a₁ ≠ a₂ ∧ a₂ ≠ a₃ ∧ a₁ ≠ a₃

-- Define a function to check if two polynomials intersect at exactly one point
def intersectAtOnePoint (f g : QuadraticPolynomial) : Prop :=
  ∃! x, f x = g x

-- Main theorem
theorem quadratic_polynomials_intersection
  (f g h : QuadraticPolynomial)
  (h₁ : pairwiseDistinctLeadingCoeff f g h)
  (h₂ : intersectAtOnePoint f g)
  (h₃ : intersectAtOnePoint g h)
  (h₄ : intersectAtOnePoint f h) :
  ∃! x, f x = g x ∧ g x = h x :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_intersection_l961_96193


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l961_96100

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) : 
  total_packages = 9 → total_pieces = 135 → total_pieces / total_packages = 15 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l961_96100


namespace NUMINAMATH_CALUDE_madeline_sleep_hours_madeline_sleeps_eight_hours_l961_96121

/-- Calculates the number of hours Madeline spends sleeping per day given her weekly schedule. -/
theorem madeline_sleep_hours (class_hours_per_week : ℕ) 
                              (homework_hours_per_day : ℕ) 
                              (work_hours_per_week : ℕ) 
                              (leftover_hours_per_week : ℕ) : ℕ :=
  let total_hours_per_week : ℕ := 24 * 7
  let remaining_hours : ℕ := total_hours_per_week - class_hours_per_week - 
                             (homework_hours_per_day * 7) - work_hours_per_week - 
                             leftover_hours_per_week
  remaining_hours / 7

/-- Proves that Madeline spends 8 hours per day sleeping given her schedule. -/
theorem madeline_sleeps_eight_hours : 
  madeline_sleep_hours 18 4 20 46 = 8 := by
  sorry

end NUMINAMATH_CALUDE_madeline_sleep_hours_madeline_sleeps_eight_hours_l961_96121


namespace NUMINAMATH_CALUDE_camel_moves_divisible_by_three_l961_96132

/-- Represents the color of a square --/
inductive SquareColor
| Black
| White

/-- Represents a camel's movement --/
def CamelMove := ℕ → SquareColor

/-- A camel's movement pattern that alternates between black and white squares --/
def alternatingPattern : CamelMove :=
  fun n => match n % 3 with
    | 0 => SquareColor.Black
    | 1 => SquareColor.White
    | _ => SquareColor.Black

/-- Theorem: If a camel makes n moves in an alternating pattern and returns to its starting position, then n is divisible by 3 --/
theorem camel_moves_divisible_by_three (n : ℕ) 
  (h1 : alternatingPattern n = alternatingPattern 0) : 
  3 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_camel_moves_divisible_by_three_l961_96132


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l961_96119

theorem existence_of_special_integer (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat, x > 0 ∧ (∀ p : Nat, Nat.Prime p →
    (p ∈ P ↔ ∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a ^ p + b ^ p)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l961_96119


namespace NUMINAMATH_CALUDE_larger_integer_value_l961_96134

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (b : ℚ) / (a : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  (b : ℕ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l961_96134


namespace NUMINAMATH_CALUDE_least_x_for_integer_fraction_l961_96151

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem least_x_for_integer_fraction :
  ∀ x : ℝ, (is_integer (24 / (x - 4)) ∧ x < -20) → False :=
by sorry

end NUMINAMATH_CALUDE_least_x_for_integer_fraction_l961_96151


namespace NUMINAMATH_CALUDE_xyz_congruence_l961_96145

theorem xyz_congruence (x y z : Int) : 
  x < 7 → y < 7 → z < 7 →
  (x + 3*y + 2*z) % 7 = 2 →
  (3*x + 2*y + z) % 7 = 5 →
  (2*x + y + 3*z) % 7 = 3 →
  (x * y * z) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_congruence_l961_96145


namespace NUMINAMATH_CALUDE_lottery_probability_calculation_l961_96195

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallsDrawn : ℕ := 5
def specialBallCount : ℕ := 45

def lotteryProbability : ℚ :=
  1 / (megaBallCount * (winnerBallCount.choose winnerBallsDrawn) * specialBallCount)

theorem lottery_probability_calculation :
  lotteryProbability = 1 / 2861184000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_calculation_l961_96195


namespace NUMINAMATH_CALUDE_fruit_display_ratio_l961_96187

theorem fruit_display_ratio (apples oranges bananas : ℕ) : 
  apples = 2 * oranges →
  apples + oranges + bananas = 35 →
  bananas = 5 →
  oranges = 2 * bananas :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_display_ratio_l961_96187


namespace NUMINAMATH_CALUDE_total_is_527_given_shares_inconsistent_l961_96165

/-- Represents the shares of money for three individuals --/
structure Shares :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Calculates the total amount from given shares --/
def total_amount (s : Shares) : ℕ := s.a + s.b + s.c

/-- The given shares --/
def given_shares : Shares := ⟨372, 93, 62⟩

/-- Theorem stating that the total amount is 527 --/
theorem total_is_527 : total_amount given_shares = 527 := by
  sorry

/-- Property that should hold for the shares based on the problem statement --/
def shares_property (s : Shares) : Prop :=
  s.a = (2 * s.b) / 3 ∧ s.b = s.c / 4

/-- Theorem stating that the given shares do not satisfy the problem's conditions --/
theorem given_shares_inconsistent : ¬ shares_property given_shares := by
  sorry

end NUMINAMATH_CALUDE_total_is_527_given_shares_inconsistent_l961_96165


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l961_96171

theorem partial_fraction_decomposition_product : 
  ∀ (A B C : ℚ),
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 →
    (x^2 - 23) / (x^3 - 3*x^2 - 4*x + 12) = 
    A / (x - 1) + B / (x + 3) + C / (x - 4)) →
  A * B * C = 11/36 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l961_96171


namespace NUMINAMATH_CALUDE_individual_egg_price_is_50_l961_96197

/-- The price per individual egg in cents -/
def individual_egg_price : ℕ := sorry

/-- The number of eggs in a tray -/
def eggs_per_tray : ℕ := 30

/-- The price of a tray of eggs in cents -/
def tray_price : ℕ := 1200

/-- The savings per egg when buying a tray, in cents -/
def savings_per_egg : ℕ := 10

theorem individual_egg_price_is_50 : 
  individual_egg_price = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_individual_egg_price_is_50_l961_96197


namespace NUMINAMATH_CALUDE_james_marbles_distribution_l961_96120

theorem james_marbles_distribution (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 28)
  (h2 : remaining_marbles = 21)
  (h3 : initial_marbles > remaining_marbles) :
  ∃ (num_bags : ℕ), 
    num_bags > 1 ∧ 
    (initial_marbles - remaining_marbles) * num_bags = initial_marbles ∧
    num_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_james_marbles_distribution_l961_96120


namespace NUMINAMATH_CALUDE_spade_calculation_l961_96163

-- Define the ⋄ operation
def spade (x y : ℝ) : ℝ := (x + y)^2 * (x - y)

-- Theorem statement
theorem spade_calculation : spade 2 (spade 3 6) = 14229845 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l961_96163


namespace NUMINAMATH_CALUDE_root_product_equality_l961_96173

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
  sorry


end NUMINAMATH_CALUDE_root_product_equality_l961_96173


namespace NUMINAMATH_CALUDE_age_sum_after_ten_years_l961_96169

theorem age_sum_after_ten_years 
  (kareem_age : ℕ) 
  (son_age : ℕ) 
  (h1 : kareem_age = 42) 
  (h2 : son_age = 14) 
  (h3 : kareem_age = 3 * son_age) : 
  (kareem_age + 10) + (son_age + 10) = 76 := by
sorry

end NUMINAMATH_CALUDE_age_sum_after_ten_years_l961_96169


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l961_96162

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^502 * k = 15^504 - 6^502) ∧ 
  (∀ m : ℕ, m > 502 → ¬(∃ k : ℕ, 2^m * k = 15^504 - 6^502)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l961_96162


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l961_96124

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 7 * x - 10
  let S : Set ℝ := {x | f x ≥ 0}
  S = {x | x ≥ 10/3 ∨ x ≤ -1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l961_96124


namespace NUMINAMATH_CALUDE_min_chord_length_proof_l961_96113

/-- The circle equation: x^2 + y^2 - 6x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (1, 2)

/-- The minimum length of the chord -/
def min_chord_length : ℝ := 2

theorem min_chord_length_proof :
  ∀ (x y : ℝ), circle_equation x y →
  ∀ (chord_length : ℝ),
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle_equation x1 y1 ∧ 
    circle_equation x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2 ∧
    (x1 + x2) / 2 = point.1 ∧ 
    (y1 + y2) / 2 = point.2) →
  chord_length ≥ min_chord_length :=
by sorry

#check min_chord_length_proof

end NUMINAMATH_CALUDE_min_chord_length_proof_l961_96113


namespace NUMINAMATH_CALUDE_family_member_bites_l961_96125

-- Define the number of mosquito bites Cyrus got on arms and legs
def cyrus_arms_legs_bites : ℕ := 14

-- Define the number of mosquito bites Cyrus got on his body
def cyrus_body_bites : ℕ := 10

-- Define the number of other family members
def family_members : ℕ := 6

-- Define Cyrus' total bites
def cyrus_total_bites : ℕ := cyrus_arms_legs_bites + cyrus_body_bites

-- Define the family's total bites
def family_total_bites : ℕ := cyrus_total_bites / 2

-- Theorem to prove
theorem family_member_bites :
  family_total_bites / family_members = 2 :=
by sorry

end NUMINAMATH_CALUDE_family_member_bites_l961_96125


namespace NUMINAMATH_CALUDE_chocolate_bar_theorem_l961_96136

theorem chocolate_bar_theorem (n m : ℕ) (h : n * m = 25) :
  (∃ (b w : ℕ), b + w = n * m ∧ b = w + 1 ∧ b = (25 * w) / 3) →
  n + m = 10 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_theorem_l961_96136


namespace NUMINAMATH_CALUDE_contest_result_l961_96166

/-- The number of baskets made by Alex, Sandra, Hector, and Jordan -/
def total_baskets (alex sandra hector jordan : ℕ) : ℕ :=
  alex + sandra + hector + jordan

/-- Theorem stating the total number of baskets under given conditions -/
theorem contest_result : ∃ (alex sandra hector jordan : ℕ),
  alex = 8 ∧
  sandra = 3 * alex ∧
  hector = 2 * sandra ∧
  jordan = (alex + sandra + hector) / 5 ∧
  total_baskets alex sandra hector jordan = 96 := by
  sorry

end NUMINAMATH_CALUDE_contest_result_l961_96166


namespace NUMINAMATH_CALUDE_M_on_inscribed_square_l961_96157

/-- Right triangle with squares and inscribed square -/
structure RightTriangleWithSquares where
  -- Right triangle ABC
  a : ℝ
  b : ℝ
  c : ℝ
  -- Pythagorean theorem
  pythagoras : a^2 + b^2 = c^2
  -- Positivity of sides
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  -- Inscribed square side length
  x : ℝ
  x_def : x = (a * b) / (a + b)
  -- Point M
  M : ℝ × ℝ

/-- The theorem stating that M lies on the perimeter of the inscribed square -/
theorem M_on_inscribed_square (t : RightTriangleWithSquares) :
  t.M.1 = t.x ∧ 0 ≤ t.M.2 ∧ t.M.2 ≤ t.x := by
  sorry

end NUMINAMATH_CALUDE_M_on_inscribed_square_l961_96157


namespace NUMINAMATH_CALUDE_a_7_equals_two_l961_96181

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Arithmetic sequence property -/
def is_arithmetic (b : Sequence) : Prop :=
  ∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m

theorem a_7_equals_two (a b : Sequence) 
  (h1 : ∀ n, a n ≠ 0)
  (h2 : a 4 - 2 * a 7 + a 8 = 0)
  (h3 : is_arithmetic b)
  (h4 : b 7 = a 7)
  (h5 : b 2 < b 8)
  (h6 : b 8 < b 11) :
  a 7 = 2 :=
sorry

end NUMINAMATH_CALUDE_a_7_equals_two_l961_96181


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l961_96172

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given a geometric sequence {aₙ} satisfying a₁ + a₂ = 3 and a₂ + a₃ = 6, prove that a₇ = 64 -/
theorem geometric_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_geom : IsGeometricSequence a) 
  (h_sum1 : a 1 + a 2 = 3) 
  (h_sum2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l961_96172


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l961_96133

theorem basketball_lineup_count :
  let total_players : ℕ := 12
  let point_guards : ℕ := 1
  let other_players : ℕ := 5
  Nat.choose total_players point_guards * Nat.choose (total_players - point_guards) other_players = 5544 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l961_96133


namespace NUMINAMATH_CALUDE_john_sleep_for_target_score_l961_96105

/-- Represents the relationship between sleep hours and exam score -/
structure ExamPerformance where
  sleep : ℝ
  score : ℝ

/-- The inverse relationship between sleep and score -/
def inverseRelation (e1 e2 : ExamPerformance) : Prop :=
  e1.sleep * e1.score = e2.sleep * e2.score

theorem john_sleep_for_target_score 
  (e1 : ExamPerformance) 
  (e2 : ExamPerformance) 
  (h1 : e1.sleep = 6) 
  (h2 : e1.score = 80) 
  (h3 : inverseRelation e1 e2) 
  (h4 : (e1.score + e2.score) / 2 = 85) : 
  e2.sleep = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_john_sleep_for_target_score_l961_96105


namespace NUMINAMATH_CALUDE_partition_6_5_l961_96102

/-- The number of ways to partition n into at most k non-negative integer parts -/
def num_partitions (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition 6 into at most 5 non-negative integer parts -/
theorem partition_6_5 : num_partitions 6 5 = 11 := by sorry

end NUMINAMATH_CALUDE_partition_6_5_l961_96102


namespace NUMINAMATH_CALUDE_problem_solution_l961_96183

theorem problem_solution : (-1)^2022 + |(-2)^3 + (-3)^2| - (-1/4 + 1/6) * (-24) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l961_96183


namespace NUMINAMATH_CALUDE_tyrone_quarters_l961_96110

/-- Represents the count of each type of coin or bill --/
structure CoinCount where
  dollars_1 : ℕ
  dollars_5 : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in dollars of a given coin count, excluding quarters --/
def value_without_quarters (c : CoinCount) : ℚ :=
  c.dollars_1 + 5 * c.dollars_5 + 0.1 * c.dimes + 0.05 * c.nickels + 0.01 * c.pennies

/-- The value of a quarter in dollars --/
def quarter_value : ℚ := 0.25

theorem tyrone_quarters : 
  ∀ (c : CoinCount) (total : ℚ),
    c.dollars_1 = 2 →
    c.dollars_5 = 1 →
    c.dimes = 20 →
    c.nickels = 8 →
    c.pennies = 35 →
    total = 13 →
    (total - value_without_quarters c) / quarter_value = 13 := by
  sorry

end NUMINAMATH_CALUDE_tyrone_quarters_l961_96110


namespace NUMINAMATH_CALUDE_wire_cutting_is_random_event_l961_96123

/-- An event that can occur but is not certain to occur -/
structure PossibleEvent where
  can_occur : Bool
  not_certain : Bool

/-- A random event is a possible event that exhibits regularity in repeated trials -/
structure RandomEvent extends PossibleEvent where
  exhibits_regularity : Bool

/-- The event of cutting a wire into three pieces to form a triangle -/
def wire_cutting_event (a : ℝ) : PossibleEvent :=
  { can_occur := true,
    not_certain := true }

/-- Theorem: The wire cutting event is a random event -/
theorem wire_cutting_is_random_event (a : ℝ) :
  ∃ (e : RandomEvent), (e.toPossibleEvent = wire_cutting_event a) :=
sorry

end NUMINAMATH_CALUDE_wire_cutting_is_random_event_l961_96123


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l961_96178

theorem cyclic_sum_inequality (a b c : ℝ) : 
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l961_96178


namespace NUMINAMATH_CALUDE_food_problem_l961_96137

/-- The number of days food lasts for a group of men -/
def food_duration (initial_men : ℕ) (additional_men : ℕ) (initial_days : ℕ) (additional_days : ℕ) : Prop :=
  initial_men * initial_days = 
  initial_men * 2 + (initial_men + additional_men) * additional_days

theorem food_problem : 
  ∃ (D : ℕ), food_duration 760 760 D 10 ∧ D = 22 := by
  sorry

end NUMINAMATH_CALUDE_food_problem_l961_96137


namespace NUMINAMATH_CALUDE_sqrt_real_range_l961_96128

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 6 - 2 * x) ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l961_96128


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l961_96130

theorem imaginary_unit_sum (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l961_96130


namespace NUMINAMATH_CALUDE_temple_storage_cost_l961_96186

/-- Calculates the total cost for storing items for a group of people -/
def totalCost (numPeople : ℕ) (numPeopleWithGloves : ℕ) (costPerObject : ℕ) : ℕ :=
  let numObjectsPerPerson := 2 + 2 + 1 + 1  -- 2 shoes, 2 socks, 1 mobile, 1 umbrella
  let totalObjects := numPeople * numObjectsPerPerson + numPeopleWithGloves * 2
  totalObjects * costPerObject

/-- Proves that the total cost for the given scenario is 374 dollars -/
theorem temple_storage_cost : totalCost 5 2 11 = 374 := by
  sorry

end NUMINAMATH_CALUDE_temple_storage_cost_l961_96186


namespace NUMINAMATH_CALUDE_first_number_in_sequence_l961_96158

def sequence_sum (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → n ≤ 10 → a n = a (n-1) + a (n-2) + a (n-3)

theorem first_number_in_sequence 
  (a : ℕ → ℤ) 
  (h_sum : sequence_sum a) 
  (h_8 : a 8 = 29) 
  (h_9 : a 9 = 56) 
  (h_10 : a 10 = 108) : 
  a 1 = 32 := by sorry

end NUMINAMATH_CALUDE_first_number_in_sequence_l961_96158


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l961_96114

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l961_96114


namespace NUMINAMATH_CALUDE_red_balls_count_l961_96115

theorem red_balls_count (black_balls white_balls : ℕ) (red_prob : ℝ) : 
  black_balls = 8 → white_balls = 4 → red_prob = 0.4 → 
  (black_balls + white_balls : ℝ) / (1 - red_prob) = black_balls + white_balls + 8 := by
  sorry

#check red_balls_count

end NUMINAMATH_CALUDE_red_balls_count_l961_96115


namespace NUMINAMATH_CALUDE_hyperbola_equation_l961_96189

/-- Given a hyperbola with the equation (x^2/a^2) - (y^2/b^2) = 1, where a > 0 and b > 0,
    if the eccentricity is 2 and the distance from the right focus to one of the asymptotes is √3,
    then the equation of the hyperbola is x^2 - (y^2/3) = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a →  -- eccentricity is 2
  b = Real.sqrt 3 →  -- distance from right focus to asymptote is √3
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l961_96189


namespace NUMINAMATH_CALUDE_xyz_value_l961_96139

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 195)
  (h2 : y * (z + x) = 204)
  (h3 : z * (x + y) = 213) : 
  x * y * z = 1029 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l961_96139


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l961_96138

/-- A geometric sequence with specific terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 3) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l961_96138


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_157_l961_96116

theorem first_nonzero_digit_of_one_over_157 : ∃ (n : ℕ), 
  (1000 : ℚ) / 157 > 6 ∧ (1000 : ℚ) / 157 < 7 ∧ 
  (1000 * (1 : ℚ) / 157 - 6) * 10 ≥ 3 ∧ (1000 * (1 : ℚ) / 157 - 6) * 10 < 4 := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_157_l961_96116


namespace NUMINAMATH_CALUDE_base9_to_base10_conversion_l961_96184

/-- Converts a base-9 number represented as a list of digits to its base-10 equivalent -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The base-9 representation of the number -/
def base9Number : List Nat := [7, 4, 8, 2]

theorem base9_to_base10_conversion :
  base9ToBase10 base9Number = 2149 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base10_conversion_l961_96184


namespace NUMINAMATH_CALUDE_train_crossing_time_l961_96170

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 333.33)
  (h3 : platform_crossing_time = 38)
  : ∃ (signal_pole_crossing_time : ℝ),
    signal_pole_crossing_time = train_length / ((train_length + platform_length) / platform_crossing_time) ∧
    (signal_pole_crossing_time ≥ 17.9 ∧ signal_pole_crossing_time ≤ 18.1) :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l961_96170


namespace NUMINAMATH_CALUDE_board_length_l961_96112

/-- Given a board cut into two pieces, prove that its total length is 20.0 feet. -/
theorem board_length : 
  ∀ (shorter longer : ℝ),
  shorter = 8.0 →
  2 * shorter = longer + 4 →
  shorter + longer = 20.0 := by
sorry

end NUMINAMATH_CALUDE_board_length_l961_96112


namespace NUMINAMATH_CALUDE_grandpa_grandchildren_ages_l961_96109

theorem grandpa_grandchildren_ages (grandpa_age : ℕ) (gc1_age gc2_age gc3_age : ℕ) (years : ℕ) :
  grandpa_age = 75 →
  gc1_age = 13 →
  gc2_age = 15 →
  gc3_age = 17 →
  years = 15 →
  grandpa_age + years = (gc1_age + years) + (gc2_age + years) + (gc3_age + years) :=
by sorry

end NUMINAMATH_CALUDE_grandpa_grandchildren_ages_l961_96109


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l961_96168

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U :
  (U \ M) = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l961_96168


namespace NUMINAMATH_CALUDE_square_ratio_proof_l961_96176

theorem square_ratio_proof (area_ratio : Rat) (a b c : ℕ) : 
  area_ratio = 50 / 98 →
  (a : Rat) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  (a : Rat) / c = 5 / 7 →
  a + b + c = 12 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l961_96176


namespace NUMINAMATH_CALUDE_function_inequality_l961_96161

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x : ℝ, x ≠ 1 → (x - 1) * deriv f x > 0) :
  f 0 + f 2 > 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l961_96161


namespace NUMINAMATH_CALUDE_symposium_pair_selection_l961_96185

theorem symposium_pair_selection (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 2) :
  Nat.choose n k = 435 := by
  sorry

end NUMINAMATH_CALUDE_symposium_pair_selection_l961_96185


namespace NUMINAMATH_CALUDE_men_french_percentage_l961_96103

/-- Represents the percentage of employees who are men -/
def percent_men : ℝ := 0.35

/-- Represents the percentage of employees who speak French -/
def percent_french : ℝ := 0.40

/-- Represents the percentage of women who do not speak French -/
def percent_women_not_french : ℝ := 0.7077

/-- Represents the percentage of men who speak French -/
def percent_men_french : ℝ := 0.60

theorem men_french_percentage :
  percent_men * percent_men_french + (1 - percent_men) * (1 - percent_women_not_french) = percent_french :=
sorry


end NUMINAMATH_CALUDE_men_french_percentage_l961_96103


namespace NUMINAMATH_CALUDE_same_color_probability_l961_96174

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 4

/-- The probability of drawing four marbles of the same color -/
theorem same_color_probability : 
  (Nat.choose red_marbles drawn_marbles + 
   Nat.choose white_marbles drawn_marbles + 
   Nat.choose blue_marbles drawn_marbles) / 
  Nat.choose total_marbles drawn_marbles = 8 / 399 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l961_96174


namespace NUMINAMATH_CALUDE_income_difference_is_negative_150_l961_96148

/-- Calculates the difference in income between Janet's first month as a freelancer and her current job -/
def income_difference : ℤ :=
  let current_job_weekly_hours : ℕ := 40
  let current_job_hourly_rate : ℕ := 30
  let freelance_weeks : List ℕ := [30, 35, 40, 50]
  let freelance_rates : List ℕ := [45, 40, 35, 38]
  let extra_fica_tax_weekly : ℕ := 25
  let healthcare_premium_monthly : ℕ := 400
  let increased_rent_monthly : ℕ := 750
  let business_expenses_monthly : ℕ := 150
  let weeks_per_month : ℕ := 4

  let current_job_monthly_income := current_job_weekly_hours * current_job_hourly_rate * weeks_per_month
  
  let freelance_monthly_income := (List.zip freelance_weeks freelance_rates).map (fun (h, r) => h * r) |>.sum
  
  let extra_expenses_monthly := extra_fica_tax_weekly * weeks_per_month + 
                                healthcare_premium_monthly + 
                                increased_rent_monthly + 
                                business_expenses_monthly
  
  let freelance_net_income := freelance_monthly_income - extra_expenses_monthly
  
  freelance_net_income - current_job_monthly_income

theorem income_difference_is_negative_150 : income_difference = -150 := by
  sorry

end NUMINAMATH_CALUDE_income_difference_is_negative_150_l961_96148


namespace NUMINAMATH_CALUDE_power_of_two_sum_l961_96149

theorem power_of_two_sum : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l961_96149


namespace NUMINAMATH_CALUDE_kangaroo_equality_days_l961_96175

/-- The number of days required for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos bert_kangaroos bert_daily_purchase : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_daily_purchase

/-- Theorem stating that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem kangaroo_equality_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

#eval days_to_equal_kangaroos 100 20 2

end NUMINAMATH_CALUDE_kangaroo_equality_days_l961_96175


namespace NUMINAMATH_CALUDE_ducks_joining_l961_96196

theorem ducks_joining (original : ℕ) (total : ℕ) (joined : ℕ) : 
  original = 13 → total = 33 → joined = total - original → joined = 20 := by
sorry

end NUMINAMATH_CALUDE_ducks_joining_l961_96196


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l961_96104

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

def figure_problem (counts : TriangleCounts) (pairs : CoincidingPairs) : Prop :=
  counts.red = 4 ∧
  counts.blue = 6 ∧
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 3 ∧
  pairs.white_white = 7

theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) :
  figure_problem counts pairs → pairs.white_white = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l961_96104


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l961_96159

theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (a c : ℝ), ∀ x, 16 * x^2 + 40 * x + b = (a * x + c)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l961_96159


namespace NUMINAMATH_CALUDE_expansion_coefficient_l961_96180

/-- Represents the coefficient of x^n in the expansion of (x^2 + x + 1)^k -/
def generalized_pascal (k n : ℕ) : ℕ := sorry

/-- The coefficient of x^8 in the expansion of (1+ax)(x^2+x+1)^5 -/
def coeff_x8 (a : ℝ) : ℝ := generalized_pascal 5 2 + a * generalized_pascal 5 1

theorem expansion_coefficient (a : ℝ) : coeff_x8 a = 75 → a = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l961_96180


namespace NUMINAMATH_CALUDE_number_wall_solution_l961_96117

/-- Represents a number wall with 4 elements in the bottom row -/
structure NumberWall :=
  (bottom_row : Fin 4 → ℕ)
  (second_row_right : ℕ)
  (top : ℕ)

/-- Checks if a number wall is valid according to the summing rules -/
def is_valid_wall (w : NumberWall) : Prop :=
  w.second_row_right = w.bottom_row 2 + w.bottom_row 3 ∧
  w.top = (w.bottom_row 0 + w.bottom_row 1 + w.bottom_row 2) + w.second_row_right

theorem number_wall_solution (w : NumberWall) 
  (h1 : w.bottom_row 1 = 3)
  (h2 : w.bottom_row 2 = 6)
  (h3 : w.bottom_row 3 = 5)
  (h4 : w.second_row_right = 20)
  (h5 : w.top = 57)
  (h6 : is_valid_wall w) :
  w.bottom_row 0 = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l961_96117


namespace NUMINAMATH_CALUDE_batsman_new_average_l961_96155

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the new average after an additional inning -/
def newAverage (bp : BatsmanPerformance) : Nat :=
  (bp.totalRuns + 74) / (bp.innings + 1)

/-- Theorem: The batsman's new average is 26 runs -/
theorem batsman_new_average (bp : BatsmanPerformance) 
  (h1 : bp.innings = 16)
  (h2 : newAverage bp = bp.totalRuns / bp.innings + 3)
  : newAverage bp = 26 := by
  sorry

#check batsman_new_average

end NUMINAMATH_CALUDE_batsman_new_average_l961_96155


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l961_96106

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, -2; 1, 0] →
  (B^3)⁻¹ = !![15, -14; 7, -6] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l961_96106


namespace NUMINAMATH_CALUDE_book_length_l961_96188

theorem book_length (P : ℕ) 
  (h1 : 2 * P = 3 * ((2 * P) / 3 - P / 3 + 100)) : P = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_length_l961_96188


namespace NUMINAMATH_CALUDE_simson_line_l961_96164

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the properties and relations
variable (incircle : Point → Point → Point → Point → Prop)
variable (on_circle : Point → Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Line → Prop)
variable (on_line : Point → Line → Prop)
variable (collinear : Point → Point → Point → Prop)

-- Define the theorem
theorem simson_line 
  (A B C P U V W : Point) 
  (circle : Line) 
  (BC CA AB : Line) :
  incircle A B C P →
  on_circle A B C P →
  perpendicular P U BC →
  perpendicular P V CA →
  perpendicular P W AB →
  on_line U BC →
  on_line V CA →
  on_line W AB →
  collinear U V W :=
sorry

end NUMINAMATH_CALUDE_simson_line_l961_96164


namespace NUMINAMATH_CALUDE_triangle_side_length_l961_96108

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  S = (1/2) * b * c * Real.sin A ∧
  S = Real.sqrt 3 ∧
  b = 1 ∧
  A = π/3 →
  a = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l961_96108


namespace NUMINAMATH_CALUDE_determine_c_l961_96111

/-- A function f(x) = x^2 + ax + b with domain [0, +∞) -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The solution set of f(x) < c is (m, m+6) -/
def solution_set (a b c m : ℝ) : Prop :=
  ∀ x, x ∈ Set.Ioo m (m+6) ↔ f a b x < c

theorem determine_c (a b c m : ℝ) :
  (∀ x, x ≥ 0 → f a b x = x^2 + a*x + b) →
  solution_set a b c m →
  c = 9 := by sorry

end NUMINAMATH_CALUDE_determine_c_l961_96111


namespace NUMINAMATH_CALUDE_intersection_line_equation_l961_96135

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def circle1 : Circle := { center := (-5, -3), radius := 15 }
def circle2 : Circle := { center := (4, 15), radius := 9 }

/-- The line passing through the intersection points of two circles -/
def intersectionLine (c1 c2 : Circle) : Line := sorry

theorem intersection_line_equation :
  let l := intersectionLine circle1 circle2
  l.a = 1 ∧ l.b = 1 ∧ l.c = -27/4 := by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l961_96135


namespace NUMINAMATH_CALUDE_triangle_side_length_l961_96126

-- Define the triangle ABC
def triangle_ABC (a : ℕ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
    let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    AB = 1 ∧ BC = 2007 ∧ AC = a

-- Theorem statement
theorem triangle_side_length :
  ∀ a : ℕ, triangle_ABC a → a = 2007 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l961_96126


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l961_96152

def set_A : Set ℝ := {x | x^2 + x - 2 < 0}
def set_B : Set ℝ := {x | x > 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l961_96152


namespace NUMINAMATH_CALUDE_garden_length_l961_96177

/-- A rectangular garden with length twice the width and 180 yards of fencing. -/
structure Garden where
  width : ℝ
  length : ℝ
  fencing : ℝ
  twice_width : length = 2 * width
  total_fencing : 2 * length + 2 * width = fencing

/-- The length of a garden with 180 yards of fencing is 60 yards. -/
theorem garden_length (g : Garden) (h : g.fencing = 180) : g.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l961_96177


namespace NUMINAMATH_CALUDE_fraction_equality_l961_96182

theorem fraction_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x / y + y / x = 4) : 
  x * y / (x^2 - y^2) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l961_96182


namespace NUMINAMATH_CALUDE_folded_line_length_squared_l961_96156

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  is_positive : side_length > 0

-- Define the folding operation
def fold (t : EquilateralTriangle) (fold_point : ℝ) :=
  0 < fold_point ∧ fold_point < t.side_length

-- Theorem statement
theorem folded_line_length_squared 
  (t : EquilateralTriangle) 
  (h_side : t.side_length = 10) 
  (h_fold : fold t 3) : 
  ∃ (l : ℝ), l^2 = 37/4 ∧ l > 0 := by
  sorry

end NUMINAMATH_CALUDE_folded_line_length_squared_l961_96156


namespace NUMINAMATH_CALUDE_flowers_for_maria_l961_96129

def days_until_birthday : ℕ := 22
def savings_per_day : ℚ := 2
def cost_per_flower : ℚ := 4

theorem flowers_for_maria :
  ⌊(days_until_birthday * savings_per_day) / cost_per_flower⌋ = 11 := by
  sorry

end NUMINAMATH_CALUDE_flowers_for_maria_l961_96129


namespace NUMINAMATH_CALUDE_insects_in_lab_l961_96118

/-- The number of insects in a laboratory given the total number of insect legs and legs per insect. -/
def num_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem: There are 8 insects in the laboratory. -/
theorem insects_in_lab : num_insects 48 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_insects_in_lab_l961_96118


namespace NUMINAMATH_CALUDE_vertical_translation_by_two_l961_96198

/-- For any real-valued function f and any real number x,
    f(x) + 2 is equal to a vertical translation of f(x) by 2 units upward -/
theorem vertical_translation_by_two (f : ℝ → ℝ) (x : ℝ) :
  f x + 2 = (fun y ↦ f y + 2) x :=
by sorry

end NUMINAMATH_CALUDE_vertical_translation_by_two_l961_96198


namespace NUMINAMATH_CALUDE_power_of_two_equation_l961_96147

theorem power_of_two_equation : ∃ x : ℕ, 
  8 * (32 ^ 10) = 2 ^ x ∧ x = 53 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l961_96147


namespace NUMINAMATH_CALUDE_car_rental_savings_l961_96143

def trip_distance : ℝ := 150
def first_option_cost : ℝ := 50
def second_option_cost : ℝ := 90
def gasoline_efficiency : ℝ := 15
def gasoline_cost_per_liter : ℝ := 0.9

theorem car_rental_savings : 
  let total_distance := 2 * trip_distance
  let gasoline_needed := total_distance / gasoline_efficiency
  let gasoline_cost := gasoline_needed * gasoline_cost_per_liter
  let first_option_total := first_option_cost + gasoline_cost
  second_option_cost - first_option_total = 22 := by sorry

end NUMINAMATH_CALUDE_car_rental_savings_l961_96143


namespace NUMINAMATH_CALUDE_inequality_proof_l961_96146

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l961_96146


namespace NUMINAMATH_CALUDE_angle_between_vectors_l961_96150

theorem angle_between_vectors (a b : ℝ × ℝ) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 2) →
  (Real.sqrt (a.1^2 + a.2^2) = 1) →
  (Real.sqrt (b.1^2 + b.2^2) = 2) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l961_96150


namespace NUMINAMATH_CALUDE_inverse_of_complex_l961_96101

theorem inverse_of_complex (z : ℂ) : z = 1 - 2 * I → z⁻¹ = (1 / 5 : ℂ) + (2 / 5 : ℂ) * I := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_complex_l961_96101


namespace NUMINAMATH_CALUDE_system_solution_l961_96160

theorem system_solution : ∃ (x y z : ℤ),
  (5732 * x + 2134 * y + 2134 * z = 7866) ∧
  (2134 * x + 5732 * y + 2134 * z = 670) ∧
  (2134 * x + 2134 * y + 5732 * z = 11464) ∧
  x = 1 ∧ y = -1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l961_96160


namespace NUMINAMATH_CALUDE_no_rational_sqrt_sin_cos_l961_96127

theorem no_rational_sqrt_sin_cos : 
  ¬ ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ 
    ∃ (a b c d : ℕ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
    (Real.sqrt (Real.sin θ) = a / b) ∧ 
    (Real.sqrt (Real.cos θ) = c / d) :=
by sorry

end NUMINAMATH_CALUDE_no_rational_sqrt_sin_cos_l961_96127


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l961_96141

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 25)
  (h2 : failed_english = 48)
  (h3 : failed_both = 27) :
  100 - (failed_hindi + failed_english - failed_both) = 54 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l961_96141


namespace NUMINAMATH_CALUDE_max_sphere_cone_volume_ratio_l961_96179

/-- The maximum volume ratio of a sphere inscribed in a cone to the cone itself -/
theorem max_sphere_cone_volume_ratio :
  ∃ (r m R : ℝ) (α : ℝ),
    r > 0 ∧ m > 0 ∧ R > 0 ∧ 0 < α ∧ α < π / 2 ∧
    r = m * Real.tan α ∧
    R = (m - R) * Real.sin α ∧
    ∀ (r' m' R' : ℝ) (α' : ℝ),
      r' > 0 → m' > 0 → R' > 0 → 0 < α' → α' < π / 2 →
      r' = m' * Real.tan α' →
      R' = (m' - R') * Real.sin α' →
      (4 / 3 * π * R' ^ 3) / ((1 / 3) * π * r' ^ 2 * m') ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_cone_volume_ratio_l961_96179


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_angle_l961_96131

/-- A quadrilateral with specific side lengths and angles has diagonals that intersect at a 60° angle. -/
theorem quadrilateral_diagonal_angle (a b c : ℝ) (angle_ab angle_bc : ℝ) :
  a = 4 * Real.sqrt 3 →
  b = 9 →
  c = Real.sqrt 3 →
  angle_ab = π / 6 →  -- 30° in radians
  angle_bc = π / 2 →  -- 90° in radians
  ∃ (angle_diagonals : ℝ), angle_diagonals = π / 3 :=  -- 60° in radians
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_angle_l961_96131


namespace NUMINAMATH_CALUDE_xyz_value_l961_96154

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h3 : (x + y + z)^2 = 25) :
  x * y * z = 31 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l961_96154


namespace NUMINAMATH_CALUDE_intersection_M_N_l961_96191

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l961_96191


namespace NUMINAMATH_CALUDE_number_divided_by_ten_l961_96142

theorem number_divided_by_ten : (120 : ℚ) / 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_ten_l961_96142


namespace NUMINAMATH_CALUDE_wrong_to_right_ratio_l961_96194

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) (h1 : total = 48) (h2 : correct = 16) :
  (total - correct) / correct = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_right_ratio_l961_96194


namespace NUMINAMATH_CALUDE_complement_of_union_l961_96122

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l961_96122


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l961_96190

/-- For a polynomial (x-m)(x-n), the condition for it to not contain a linear term in x is m + n = 0. -/
theorem no_linear_term_condition (x m n : ℝ) : 
  (∀ (a b c : ℝ), (x - m) * (x - n) = a * x^2 + c → b = 0) ↔ m + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l961_96190


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l961_96167

/-- The number of tiles in a square with side length n -/
def tilesInSquare (n : ℕ) : ℕ := n * n

/-- The difference in tiles between two consecutive squares in the sequence -/
def tileDifference (n : ℕ) : ℕ :=
  tilesInSquare (n + 1) - tilesInSquare n

theorem ninth_minus_eighth_square_tiles : tileDifference 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l961_96167


namespace NUMINAMATH_CALUDE_problem_solution_l961_96140

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x - 2|

-- State the theorem
theorem problem_solution (m : ℝ) (a b c x y z : ℝ) 
  (h1 : ∀ x, f m (x + 1) ≥ 0 ↔ 0 ≤ x ∧ x ≤ 1)
  (h2 : x^2 + y^2 + z^2 = a^2 + b^2 + c^2)
  (h3 : x^2 + y^2 + z^2 = m) :
  m = 1 ∧ a*x + b*y + c*z ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l961_96140


namespace NUMINAMATH_CALUDE_min_value_inequality_l961_96199

theorem min_value_inequality (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2*y = 1) :
  1 / (x + 1) + 1 / y ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l961_96199
