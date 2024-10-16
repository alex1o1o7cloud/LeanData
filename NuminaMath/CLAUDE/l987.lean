import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_problem_l987_98752

theorem divisibility_problem (a : ℤ) : 
  0 ≤ a ∧ a < 13 → (12^20 + a) % 13 = 0 → a = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l987_98752


namespace NUMINAMATH_CALUDE_sqrt_inequality_increasing_function_inequality_l987_98727

-- Part 1
theorem sqrt_inequality (x₁ x₂ : ℝ) (h1 : 0 ≤ x₁) (h2 : 0 ≤ x₂) (h3 : x₁ ≠ x₂) :
  (1/2) * (Real.sqrt x₁ + Real.sqrt x₂) < Real.sqrt ((x₁ + x₂) / 2) := by
  sorry

-- Part 2
theorem increasing_function_inequality {f : ℝ → ℝ} (h : Monotone f) 
  {a b : ℝ} (h1 : a + f a ≤ b + f b) : a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_increasing_function_inequality_l987_98727


namespace NUMINAMATH_CALUDE_billy_and_sam_money_l987_98711

/-- The amount of money Sam has -/
def sam_money : ℕ := 75

/-- The amount of money Billy has -/
def billy_money : ℕ := 2 * sam_money - 25

/-- The total amount of money Billy and Sam have together -/
def total_money : ℕ := sam_money + billy_money

theorem billy_and_sam_money : total_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_billy_and_sam_money_l987_98711


namespace NUMINAMATH_CALUDE_equation_solution_l987_98780

theorem equation_solution : ∃ x : ℚ, (5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)) ∧ (x = 492 / 11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l987_98780


namespace NUMINAMATH_CALUDE_parabola_properties_l987_98778

/-- A parabola with equation x² = 3y is symmetric with respect to the y-axis and passes through
    the intersection points of x - y = 0 and x² + y² - 6y = 0 -/
theorem parabola_properties (x y : ℝ) :
  (x^2 = 3*y) →
  (∀ (x₀ : ℝ), (x₀^2 = 3*y) ↔ ((-x₀)^2 = 3*y)) ∧
  (∃ (x₁ y₁ : ℝ), x₁ - y₁ = 0 ∧ x₁^2 + y₁^2 - 6*y₁ = 0 ∧ x₁^2 = 3*y₁) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l987_98778


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l987_98740

theorem quadratic_equation_solution :
  let x₁ : ℝ := -1 + Real.sqrt 6 / 2
  let x₂ : ℝ := -1 - Real.sqrt 6 / 2
  2 * x₁^2 + 4 * x₁ - 1 = 0 ∧ 2 * x₂^2 + 4 * x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l987_98740


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l987_98787

-- Define the original expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14)^(1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 1 + 3

-- Theorem statement
theorem simplification_and_exponent_sum (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) : 
  original_expression x y z = simplified_expression x y z ∧ 
  sum_of_exponents = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l987_98787


namespace NUMINAMATH_CALUDE_sum_equality_exists_l987_98763

theorem sum_equality_exists (a : Fin 16 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ i, a i > 0) 
  (h_bound : ∀ i, a i ≤ 100) : 
  ∃ i j k l : Fin 16, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l :=
sorry

end NUMINAMATH_CALUDE_sum_equality_exists_l987_98763


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l987_98748

theorem quadratic_equation_solution (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let x₁ : ℝ := 4*a/(3*b)
  let x₂ : ℝ := -3*b/(4*a)
  (12*a*b*x₁^2 - (16*a^2 - 9*b^2)*x₁ - 12*a*b = 0) ∧
  (12*a*b*x₂^2 - (16*a^2 - 9*b^2)*x₂ - 12*a*b = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l987_98748


namespace NUMINAMATH_CALUDE_stone_counting_l987_98728

theorem stone_counting (n : Nat) (h : n = 99) : n % 16 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stone_counting_l987_98728


namespace NUMINAMATH_CALUDE_q_sum_l987_98710

/-- Given a function q: ℝ → ℝ where q(1) = 3, prove that q(1) + q(2) = 8 -/
theorem q_sum (q : ℝ → ℝ) (h : q 1 = 3) : q 1 + q 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_q_sum_l987_98710


namespace NUMINAMATH_CALUDE_isabel_games_l987_98785

/-- The number of DS games Isabel had initially -/
def initial_games : ℕ := 90

/-- The number of DS games Isabel gave away -/
def games_given_away : ℕ := 87

/-- The number of DS games Isabel has left -/
def games_left : ℕ := 3

/-- Theorem stating that the initial number of games is equal to the sum of games given away and games left -/
theorem isabel_games : initial_games = games_given_away + games_left := by
  sorry

end NUMINAMATH_CALUDE_isabel_games_l987_98785


namespace NUMINAMATH_CALUDE_max_items_purchasable_l987_98799

theorem max_items_purchasable (available : ℚ) (cost_per_item : ℚ) (h1 : available = 9.2) (h2 : cost_per_item = 1.05) :
  ⌊available / cost_per_item⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_items_purchasable_l987_98799


namespace NUMINAMATH_CALUDE_intersection_A_B_l987_98788

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {y | ∃ x, y = Real.exp x}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l987_98788


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l987_98789

theorem angle_sum_in_circle (y : ℚ) : 
  (6 * y + 3 * y + y + 4 * y = 360) → y = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l987_98789


namespace NUMINAMATH_CALUDE_two_digit_number_is_30_l987_98725

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := { n : ℕ × ℕ // n.1 < 10 ∧ n.2 < 10 }

/-- Converts a two-digit number to its decimal representation -/
def to_decimal (n : TwoDigitNumber) : ℚ :=
  n.val.1 * 10 + n.val.2

/-- Represents a repeating decimal of the form 2.xy̅ -/
def repeating_decimal (n : TwoDigitNumber) : ℚ :=
  2 + (to_decimal n) / 99

/-- The main theorem stating that the two-digit number satisfying the equation is 30 -/
theorem two_digit_number_is_30 :
  ∃ (n : TwoDigitNumber), 
    75 * (repeating_decimal n - (2 + (to_decimal n) / 100)) = 2 ∧
    to_decimal n = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_is_30_l987_98725


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l987_98745

-- Define the sample space
inductive CoinToss
  | HH  -- Both heads
  | HT  -- Head and tail
  | TH  -- Tail and head
  | TT  -- Both tails

-- Define events A and B
def event_A (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HT ∨ outcome = CoinToss.TH

def event_B (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (∀ outcome : CoinToss, ¬(event_A outcome ∧ event_B outcome)) ∧
  (∃ outcome : CoinToss, ¬event_A outcome ∧ ¬event_B outcome) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l987_98745


namespace NUMINAMATH_CALUDE_inequality_proof_l987_98724

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l987_98724


namespace NUMINAMATH_CALUDE_pat_shark_photo_profit_l987_98716

/-- Calculates the expected profit for Pat's shark photo hunting trip. -/
theorem pat_shark_photo_profit :
  let photo_earnings : ℕ → ℚ := λ n => 15 * n
  let sharks_per_hour : ℕ := 6
  let fuel_cost_per_hour : ℚ := 50
  let hunting_hours : ℕ := 5
  let total_sharks : ℕ := sharks_per_hour * hunting_hours
  let total_earnings : ℚ := photo_earnings total_sharks
  let total_fuel_cost : ℚ := fuel_cost_per_hour * hunting_hours
  let profit : ℚ := total_earnings - total_fuel_cost
  profit = 200 := by
sorry


end NUMINAMATH_CALUDE_pat_shark_photo_profit_l987_98716


namespace NUMINAMATH_CALUDE_linear_function_proof_l987_98729

def f (x : ℝ) : ℝ := -x + 1

theorem linear_function_proof :
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) ∧
  f 0 = 1 ∧
  ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l987_98729


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_80_l987_98782

theorem closest_integer_to_cube_root_80 : 
  ∀ n : ℤ, |n - (80 : ℝ)^(1/3)| ≥ |4 - (80 : ℝ)^(1/3)| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_80_l987_98782


namespace NUMINAMATH_CALUDE_basketball_tournament_l987_98700

/-- The number of teams in the basketball tournament --/
def n : ℕ := 12

/-- The total number of matches played in the tournament --/
def total_matches (n : ℕ) : ℕ := n * (n - 1)

/-- The total number of points distributed in the tournament --/
def total_points (n : ℕ) : ℕ := 2 * total_matches n

/-- The number of teams scoring 24 points --/
def a (n : ℕ) : ℤ := n * (n - 1) - 11 * n + 33

/-- The number of teams scoring 22 points --/
def b (n : ℕ) : ℤ := -n^2 + 12 * n - 36

theorem basketball_tournament :
  (∃ (winner : ℕ) (last1 last2 : ℕ),
    winner = 26 ∧ 
    last1 = 20 ∧ 
    last2 = 20 ∧ 
    winner + last1 + last2 + 24 * (a n) + 22 * (b n) = total_points n) ∧
  a n ≥ 0 ∧
  b n ≥ 0 ∧
  a n + b n = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_basketball_tournament_l987_98700


namespace NUMINAMATH_CALUDE_ellipse_equation_l987_98771

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  e : ℝ  -- eccentricity

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given an ellipse C with specific properties, prove its equation -/
theorem ellipse_equation (C : Ellipse) (F₁ F₂ A B : Point) :
  C.center = (0, 0) →  -- center at origin
  F₁.y = 0 →  -- foci on x-axis
  F₂.y = 0 →
  C.e = Real.sqrt 2 / 2 →  -- eccentricity is √2/2
  (A.x - F₁.x)^2 + (A.y - F₁.y)^2 = (B.x - F₁.x)^2 + (B.y - F₁.y)^2 →  -- A and B on line through F₁
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) +
    Real.sqrt ((B.x - F₂.x)^2 + (B.y - F₂.y)^2) +
    Real.sqrt ((A.x - F₂.x)^2 + (A.y - F₂.y)^2) = 16 →  -- perimeter of ABF₂ is 16
  ∀ (x y : ℝ), x^2 / 64 + y^2 / 32 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 / C.a)^2 + (p.2 / C.b)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l987_98771


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l987_98741

/-- The minimum number of additional marbles needed for Lisa to distribute to her friends -/
def minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

theorem lisa_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 50) :
  minimum_additional_marbles num_friends initial_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l987_98741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l987_98781

/-- An arithmetic sequence with sum S_n and common difference d -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  d : ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Main theorem about properties of an arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
  (h : seq.S 7 > seq.S 6 ∧ seq.S 6 > seq.S 8) :
  seq.d < 0 ∧ 
  seq.S 14 < 0 ∧
  (∀ n, seq.S n ≤ seq.S 7) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l987_98781


namespace NUMINAMATH_CALUDE_even_square_diff_implies_even_sum_l987_98772

theorem even_square_diff_implies_even_sum (n m : ℤ) (h : Even (n^2 - m^2)) : Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_even_square_diff_implies_even_sum_l987_98772


namespace NUMINAMATH_CALUDE_selection_methods_count_l987_98793

def num_male_students : ℕ := 5
def num_female_students : ℕ := 4
def num_representatives : ℕ := 4
def min_female_representatives : ℕ := 2

theorem selection_methods_count :
  (Finset.sum (Finset.range (num_representatives - min_female_representatives + 1))
    (λ k => Nat.choose num_female_students (min_female_representatives + k) *
            Nat.choose num_male_students (num_representatives - (min_female_representatives + k))))
  = 81 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l987_98793


namespace NUMINAMATH_CALUDE_greatest_consecutive_nonneg_integers_sum_120_l987_98708

theorem greatest_consecutive_nonneg_integers_sum_120 :
  ∀ n : ℕ, (∃ a : ℕ, (n : ℕ) * (2 * a + n - 1) = 240) →
  n ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_nonneg_integers_sum_120_l987_98708


namespace NUMINAMATH_CALUDE_bella_pizza_consumption_l987_98759

theorem bella_pizza_consumption 
  (rachel_pizza : ℕ) 
  (total_pizza : ℕ) 
  (h1 : rachel_pizza = 598)
  (h2 : total_pizza = 952) :
  total_pizza - rachel_pizza = 354 := by
sorry

end NUMINAMATH_CALUDE_bella_pizza_consumption_l987_98759


namespace NUMINAMATH_CALUDE_calendar_sum_equality_l987_98797

/-- A calendar with dates behind letters --/
structure Calendar where
  C : ℕ
  A : ℕ
  B : ℕ
  S : ℕ

/-- The calendar satisfies the given conditions --/
def valid_calendar (cal : Calendar) : Prop :=
  cal.A = cal.C + 3 ∧
  cal.B = cal.A + 10 ∧
  cal.S = cal.C + 16

theorem calendar_sum_equality (cal : Calendar) (h : valid_calendar cal) :
  cal.C + cal.S = cal.A + cal.B :=
by sorry

end NUMINAMATH_CALUDE_calendar_sum_equality_l987_98797


namespace NUMINAMATH_CALUDE_raise_time_on_hoop_l987_98749

/-- Time required to raise an object by a certain distance when wrapped around a rotating hoop -/
theorem raise_time_on_hoop (r : ℝ) (rpm : ℝ) (distance : ℝ) : 
  r > 0 → rpm > 0 → distance > 0 → 
  (distance / (2 * π * r)) * (60 / rpm) = 15 / π := by
  sorry

end NUMINAMATH_CALUDE_raise_time_on_hoop_l987_98749


namespace NUMINAMATH_CALUDE_gideon_age_proof_l987_98702

/-- The number of years in a century -/
def century : ℕ := 100

/-- Gideon's initial number of marbles -/
def initial_marbles : ℕ := century

/-- The fraction of marbles Gideon gives to his sister -/
def fraction_given : ℚ := 3/4

/-- Gideon's current age -/
def gideon_age : ℕ := 45

theorem gideon_age_proof :
  gideon_age = initial_marbles - (fraction_given * initial_marbles).num - 5 :=
sorry

end NUMINAMATH_CALUDE_gideon_age_proof_l987_98702


namespace NUMINAMATH_CALUDE_chocolate_chip_difference_l987_98767

/-- The number of chocolate chips Viviana has exceeds the number Susana has -/
def viviana_more_chocolate (viviana_chocolate susana_chocolate : ℕ) : Prop :=
  viviana_chocolate > susana_chocolate

/-- The problem statement -/
theorem chocolate_chip_difference 
  (viviana_vanilla susana_chocolate : ℕ) 
  (h1 : viviana_vanilla = 20)
  (h2 : susana_chocolate = 25)
  (h3 : ∃ (viviana_chocolate susana_vanilla : ℕ), 
    viviana_more_chocolate viviana_chocolate susana_chocolate ∧
    susana_vanilla = 3 * viviana_vanilla / 4 ∧
    viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla = 90) :
  ∃ (viviana_chocolate : ℕ), viviana_chocolate - susana_chocolate = 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_chip_difference_l987_98767


namespace NUMINAMATH_CALUDE_sum_of_cubes_inequality_l987_98707

theorem sum_of_cubes_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_inequality_l987_98707


namespace NUMINAMATH_CALUDE_derivative_at_one_is_negative_one_l987_98705

open Real

theorem derivative_at_one_is_negative_one
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x > 0, f x = 2 * x * (deriv f 1) + log x) →
  deriv f 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_is_negative_one_l987_98705


namespace NUMINAMATH_CALUDE_jelly_bean_count_jelly_bean_theorem_l987_98798

theorem jelly_bean_count : ℕ → Prop :=
  fun total_jelly_beans =>
    let red_jelly_beans := (3 * total_jelly_beans) / 4
    let coconut_red_jelly_beans := red_jelly_beans / 4
    coconut_red_jelly_beans = 750 →
    total_jelly_beans = 4000

-- Proof
theorem jelly_bean_theorem : jelly_bean_count 4000 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_count_jelly_bean_theorem_l987_98798


namespace NUMINAMATH_CALUDE_sequence_properties_l987_98746

/-- Sequence a_n with sum S_n satisfying given conditions -/
def sequence_a (n : ℕ+) : ℚ :=
  sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ+) : ℚ :=
  sorry

/-- Sequence b_n defined as n / a_n -/
def sequence_b (n : ℕ+) : ℚ :=
  n / sequence_a n

/-- Sum of first n terms of sequence b_n -/
def T (n : ℕ+) : ℚ :=
  sorry

/-- Main theorem stating the properties of sequences a_n and b_n -/
theorem sequence_properties :
  (∀ n : ℕ+, 4 * n * S n = (n + 1)^2 * sequence_a n) ∧
  sequence_a 1 = 1 ∧
  (∀ n : ℕ+, sequence_a n = n^3) ∧
  (∀ n : ℕ+, T n < 7/4) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l987_98746


namespace NUMINAMATH_CALUDE_committee_formation_count_l987_98753

def club_size : ℕ := 30
def committee_size : ℕ := 5

def ways_to_form_committee : ℕ :=
  club_size * (Nat.choose (club_size - 1) (committee_size - 1))

theorem committee_formation_count :
  ways_to_form_committee = 712530 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l987_98753


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l987_98796

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (a + 3) * x + 3 ≤ 0}
  (a < 0 → solution_set = {x : ℝ | x ≤ 3/a ∨ x ≥ 1}) ∧
  (a = 0 → solution_set = {x : ℝ | x ≥ 1}) ∧
  (0 < a ∧ a < 3 → solution_set = {x : ℝ | 1 ≤ x ∧ x ≤ 3/a}) ∧
  (a = 3 → solution_set = {1}) ∧
  (a > 3 → solution_set = {x : ℝ | 3/a ≤ x ∧ x ≤ 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l987_98796


namespace NUMINAMATH_CALUDE_probability_no_shaded_square_l987_98790

/-- Represents a rectangular grid with shaded squares -/
structure ShadedGrid :=
  (rows : Nat)
  (cols : Nat)
  (shaded_cols : Finset Nat)

/-- Calculates the total number of rectangles in the grid -/
def total_rectangles (grid : ShadedGrid) : Nat :=
  (grid.rows * Nat.choose grid.cols 2)

/-- Calculates the number of rectangles containing a shaded square -/
def shaded_rectangles (grid : ShadedGrid) : Nat :=
  grid.rows * (grid.shaded_cols.card * (grid.cols - grid.shaded_cols.card))

/-- Theorem stating the probability of selecting a rectangle without a shaded square -/
theorem probability_no_shaded_square (grid : ShadedGrid) 
  (h1 : grid.rows = 2)
  (h2 : grid.cols = 2005)
  (h3 : grid.shaded_cols = {1003}) :
  (total_rectangles grid - shaded_rectangles grid) / total_rectangles grid = 1002 / 2005 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_shaded_square_l987_98790


namespace NUMINAMATH_CALUDE_silver_solution_percentage_second_solution_percentage_l987_98737

/-- Given two silver solutions mixed to form a new solution, 
    calculate the silver percentage in the second solution. -/
theorem silver_solution_percentage 
  (volume1 : ℝ) (percent1 : ℝ) 
  (volume2 : ℝ) (final_percent : ℝ) : ℝ :=
  let total_volume := volume1 + volume2
  let silver_volume1 := volume1 * (percent1 / 100)
  let total_silver := total_volume * (final_percent / 100)
  let silver_volume2 := total_silver - silver_volume1
  (silver_volume2 / volume2) * 100

/-- Prove that the percentage of silver in the second solution is 10% -/
theorem second_solution_percentage : 
  silver_solution_percentage 5 4 2.5 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_silver_solution_percentage_second_solution_percentage_l987_98737


namespace NUMINAMATH_CALUDE_average_expenditure_for_week_l987_98701

/-- The average expenditure for a week given the average expenditures for two parts of the week -/
theorem average_expenditure_for_week 
  (avg_first_3_days : ℝ) 
  (avg_next_4_days : ℝ) 
  (h1 : avg_first_3_days = 350)
  (h2 : avg_next_4_days = 420) :
  (3 * avg_first_3_days + 4 * avg_next_4_days) / 7 = 390 := by
  sorry

#check average_expenditure_for_week

end NUMINAMATH_CALUDE_average_expenditure_for_week_l987_98701


namespace NUMINAMATH_CALUDE_P_is_projection_l987_98712

def P : Matrix (Fin 2) (Fin 2) ℚ := !![20/49, 20/49; 29/49, 29/49]

theorem P_is_projection : P * P = P := by sorry

end NUMINAMATH_CALUDE_P_is_projection_l987_98712


namespace NUMINAMATH_CALUDE_quarters_found_l987_98751

/-- The number of quarters Alyssa found in her couch -/
def num_quarters : ℕ := sorry

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of pennies Alyssa found -/
def num_pennies : ℕ := 7

/-- The total amount Alyssa found in cents -/
def total_amount : ℕ := 307

theorem quarters_found :
  num_quarters * quarter_value + num_pennies * penny_value = total_amount ∧
  num_quarters = 12 := by sorry

end NUMINAMATH_CALUDE_quarters_found_l987_98751


namespace NUMINAMATH_CALUDE_obstacle_course_completion_time_l987_98747

/-- Represents the time taken for a part of the obstacle course -/
structure CourseTime where
  minutes : ℕ
  seconds : ℕ

/-- Calculates the total seconds from a CourseTime -/
def totalSeconds (ct : CourseTime) : ℕ := ct.minutes * 60 + ct.seconds

/-- The obstacle course completion time problem -/
theorem obstacle_course_completion_time 
  (part1 : CourseTime)
  (part2 : ℕ)
  (part3 : CourseTime)
  (h1 : part1 = ⟨7, 23⟩)
  (h2 : part2 = 73)
  (h3 : part3 = ⟨5, 58⟩) :
  totalSeconds part1 + part2 + totalSeconds part3 = 874 := by
  sorry

end NUMINAMATH_CALUDE_obstacle_course_completion_time_l987_98747


namespace NUMINAMATH_CALUDE_more_males_than_females_difference_in_population_l987_98795

theorem more_males_than_females : Int → Int → Int
  | num_males, num_females =>
    num_males - num_females

theorem difference_in_population (num_males num_females : Int) 
  (h1 : num_males = 23) 
  (h2 : num_females = 9) : 
  more_males_than_females num_males num_females = 14 := by
  sorry

end NUMINAMATH_CALUDE_more_males_than_females_difference_in_population_l987_98795


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_and_shortest_chord_l987_98722

/-- The circle M with center (3, -3) and radius 4 -/
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + (y + 3)^2 = 16

/-- The line l with parameter m -/
def line_l (m x y : ℝ) : Prop := (m + 1) * x + (m + 4) * y - 3 * m = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (4, -1)

theorem line_passes_through_fixed_point_and_shortest_chord :
  (∀ m : ℝ, line_l m (point_P.1) (point_P.2)) ∧
  (∀ m : ℝ, ∃ x y : ℝ, circle_M x y ∧ line_l m x y ∧
    ∀ x' y' : ℝ, circle_M x' y' ∧ line_l m x' y' →
      (x - x')^2 + (y - y')^2 ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_and_shortest_chord_l987_98722


namespace NUMINAMATH_CALUDE_shanghai_population_aging_l987_98730

/-- Represents a city's demographic characteristics -/
structure CityDemographics where
  location : String
  economy : String
  inMigrationRate : String
  mechanicalGrowthRate : String
  naturalGrowthRate : String

/-- Represents possible population issues -/
inductive PopulationIssue
  | SatelliteTownPopulation
  | PopulationAging
  | LargePopulationBase
  | YoungPopulationStructure

/-- Determines the most significant population issue for a given city -/
def mostSignificantIssue (city : CityDemographics) : PopulationIssue :=
  sorry

/-- Shanghai's demographic characteristics -/
def shanghai : CityDemographics := {
  location := "eastern coast of China",
  economy := "developed",
  inMigrationRate := "high",
  mechanicalGrowthRate := "high",
  naturalGrowthRate := "low"
}

/-- Theorem stating that Shanghai's most significant population issue is aging -/
theorem shanghai_population_aging :
  mostSignificantIssue shanghai = PopulationIssue.PopulationAging :=
  sorry

end NUMINAMATH_CALUDE_shanghai_population_aging_l987_98730


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l987_98720

/-- The volume of a cylinder formed by rotating a square around its vertical line of symmetry -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (volume : ℝ) :
  side_length = 20 →
  volume = π * (side_length / 2)^2 * side_length →
  volume = 2000 * π := by
sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l987_98720


namespace NUMINAMATH_CALUDE_solution_is_correct_l987_98773

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  (1, 3)  -- We define this based on the given lines, without solving the system

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, l1 x y = k * l2 x y

/-- The first given line -/
def line1 (x y : ℝ) : ℝ := 3*x - 2*y + 3

/-- The second given line -/
def line2 (x y : ℝ) : ℝ := x + y - 4

/-- The line parallel to which we need to find our solution -/
def parallel_line (x y : ℝ) : ℝ := 2*x + y - 1

/-- The proposed solution line -/
def solution_line (x y : ℝ) : ℝ := 2*x + y - 5

theorem solution_is_correct : 
  let (ix, iy) := intersection_point line1 line2
  solution_line ix iy = 0 ∧ 
  are_parallel solution_line parallel_line :=
by sorry

end NUMINAMATH_CALUDE_solution_is_correct_l987_98773


namespace NUMINAMATH_CALUDE_circle_equation_l987_98714

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 2}

-- Define the line L: 2x - 3y - 1 = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 - 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (3, 0)

theorem circle_equation :
  (∃ c : ℝ × ℝ, c ∈ L ∧ c ∈ C) ∧  -- The center of C lies on L
  A ∈ C ∧ B ∈ C →                  -- C passes through A and B
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 2} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l987_98714


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l987_98765

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l987_98765


namespace NUMINAMATH_CALUDE_second_mixture_percentage_l987_98766

/-- Represents the composition of an alcohol mixture -/
structure AlcoholMixture where
  volume : ℝ
  percentage : ℝ

/-- Proves that the second mixture has 50% alcohol content -/
theorem second_mixture_percentage
  (total_mixture : AlcoholMixture)
  (first_mixture : AlcoholMixture)
  (h_total_volume : total_mixture.volume = 10)
  (h_total_percentage : total_mixture.percentage = 45)
  (h_first_volume : first_mixture.volume = 2.5)
  (h_first_percentage : first_mixture.percentage = 30)
  : ∃ (second_mixture : AlcoholMixture),
    second_mixture.volume = total_mixture.volume - first_mixture.volume ∧
    second_mixture.percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_mixture_percentage_l987_98766


namespace NUMINAMATH_CALUDE_hedgehog_strawberries_l987_98775

theorem hedgehog_strawberries : 
  ∀ (num_hedgehogs num_baskets strawberries_per_basket : ℕ) 
    (remaining_fraction : ℚ),
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  remaining_fraction = 2 / 9 →
  ∃ (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_eaten_per_hedgehog = 1050 ∧
    (num_baskets * strawberries_per_basket) * (1 - remaining_fraction) = 
      num_hedgehogs * strawberries_eaten_per_hedgehog :=
by sorry

end NUMINAMATH_CALUDE_hedgehog_strawberries_l987_98775


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l987_98718

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^4 : Polynomial ℝ) + 3 * X^3 - 4 = (X^2 + X - 3) * q + (5 * X - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l987_98718


namespace NUMINAMATH_CALUDE_smallest_a_inequality_two_ninths_satisfies_inequality_l987_98733

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  ∀ a : ℝ, (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → a * (x^2 + y^2 + z^2) + x*y*z ≥ 1/3) →
  a ≥ 2/9 :=
by sorry

theorem two_ninths_satisfies_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  (2/9 : ℝ) * (x^2 + y^2 + z^2) + x*y*z ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_inequality_two_ninths_satisfies_inequality_l987_98733


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l987_98732

/-- Represents the number of recommendation spots for each language --/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the number of male and female candidates --/
structure Candidates :=
  (male : Nat)
  (female : Nat)

/-- Calculates the number of different recommendation plans --/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

/-- Theorem stating that the number of recommendation plans is 36 --/
theorem recommendation_plans_count :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  countRecommendationPlans spots candidates = 36 :=
sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l987_98732


namespace NUMINAMATH_CALUDE_unique_positive_number_l987_98776

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l987_98776


namespace NUMINAMATH_CALUDE_ten_cut_patterns_l987_98743

/-- Represents a grid with cells that can be cut into rectangles and squares. -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (total_cells : ℕ)
  (removed_cells : ℕ)

/-- Represents a way to cut the grid. -/
structure CutPattern :=
  (rectangles : ℕ)
  (squares : ℕ)

/-- The number of valid cut patterns for a given grid. -/
def valid_cut_patterns (g : Grid) (p : CutPattern) : ℕ := sorry

/-- The main theorem stating that there are exactly 10 ways to cut the specific grid. -/
theorem ten_cut_patterns :
  ∃ (g : Grid) (p : CutPattern),
    g.rows = 3 ∧
    g.cols = 6 ∧
    g.total_cells = 17 ∧
    g.removed_cells = 1 ∧
    p.rectangles = 8 ∧
    p.squares = 1 ∧
    valid_cut_patterns g p = 10 := by sorry

end NUMINAMATH_CALUDE_ten_cut_patterns_l987_98743


namespace NUMINAMATH_CALUDE_square_plus_eight_divisible_by_eleven_l987_98791

theorem square_plus_eight_divisible_by_eleven : 
  ∃ k : ℤ, 5^2 + 8 = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_square_plus_eight_divisible_by_eleven_l987_98791


namespace NUMINAMATH_CALUDE_equation_solutions_l987_98750

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l987_98750


namespace NUMINAMATH_CALUDE_jungkook_weight_proof_l987_98784

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

/-- Jungkook's base weight in kilograms -/
def base_weight_kg : ℕ := 54

/-- Additional weight in grams -/
def additional_weight_g : ℕ := 154

/-- Jungkook's total weight in grams -/
def jungkook_weight_g : ℕ := base_weight_kg * kg_to_g + additional_weight_g

theorem jungkook_weight_proof : jungkook_weight_g = 54154 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_weight_proof_l987_98784


namespace NUMINAMATH_CALUDE_probability_of_selecting_female_student_l987_98735

theorem probability_of_selecting_female_student :
  let total_students : ℕ := 4
  let female_students : ℕ := 3
  let male_students : ℕ := 1
  female_students + male_students = total_students →
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_female_student_l987_98735


namespace NUMINAMATH_CALUDE_cherry_tomatoes_per_jar_l987_98757

theorem cherry_tomatoes_per_jar 
  (total_tomatoes : ℕ) 
  (num_jars : ℕ) 
  (h1 : total_tomatoes = 56) 
  (h2 : num_jars = 7) : 
  total_tomatoes / num_jars = 8 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomatoes_per_jar_l987_98757


namespace NUMINAMATH_CALUDE_team_a_win_probability_l987_98723

/-- The probability of Team A winning a non-fifth set -/
def p_regular : ℚ := 2/3

/-- The probability of Team A winning the fifth set -/
def p_fifth : ℚ := 1/2

/-- The probability of Team A winning the match -/
def p_win : ℚ := 20/27

/-- Theorem stating that the probability of Team A winning the match is 20/27 -/
theorem team_a_win_probability : 
  p_win = p_regular^3 + 
          3 * p_regular^2 * (1 - p_regular) * p_regular + 
          6 * p_regular^2 * (1 - p_regular)^2 * p_fifth := by
  sorry

#check team_a_win_probability

end NUMINAMATH_CALUDE_team_a_win_probability_l987_98723


namespace NUMINAMATH_CALUDE_oldest_babysat_current_age_l987_98794

/- Define the parameters of the problem -/
def jane_start_age : ℕ := 18
def jane_current_age : ℕ := 34
def years_since_stopped : ℕ := 10

/- Define the function to calculate the maximum age of a child Jane could baby-sit at a given age -/
def max_child_age (jane_age : ℕ) : ℕ :=
  jane_age / 2

/- Theorem statement -/
theorem oldest_babysat_current_age :
  let jane_stop_age : ℕ := jane_current_age - years_since_stopped
  let max_child_age_when_stopped : ℕ := max_child_age jane_stop_age
  let oldest_babysat_age : ℕ := max_child_age_when_stopped + years_since_stopped
  oldest_babysat_age = 22 := by sorry

end NUMINAMATH_CALUDE_oldest_babysat_current_age_l987_98794


namespace NUMINAMATH_CALUDE_last_four_digits_5_power_2017_l987_98709

/-- The last four digits of 5^n, represented as an integer between 0 and 9999 -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

theorem last_four_digits_5_power_2017 :
  lastFourDigits 5 = 3125 ∧
  lastFourDigits 6 = 5625 ∧
  lastFourDigits 7 = 8125 →
  lastFourDigits 2017 = 3125 := by
sorry

end NUMINAMATH_CALUDE_last_four_digits_5_power_2017_l987_98709


namespace NUMINAMATH_CALUDE_inequality_solution_set_l987_98734

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem inequality_solution_set (x : ℝ) :
  (0 < x ∧ f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) ↔ (1/Real.exp 1 < x ∧ x < Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l987_98734


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l987_98738

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  3 * (4 - 2*i) - 2*i*(3 - 2*i) + (1 + i)*(2 + i) = 9 - 9*i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l987_98738


namespace NUMINAMATH_CALUDE_superhero_payment_l987_98715

/-- Superhero payment calculation -/
theorem superhero_payment (W : ℝ) : 
  let superman_productivity := 0.1 * W
  let flash_productivity := 2 * superman_productivity
  let combined_productivity := superman_productivity + flash_productivity
  let remaining_work := 0.9 * W
  let combined_time := remaining_work / combined_productivity
  let superman_total_time := 1 + combined_time
  let flash_total_time := combined_time
  let payment (t : ℝ) := 90 / t
  (payment superman_total_time = 22.5) ∧ (payment flash_total_time = 30) :=
by sorry

end NUMINAMATH_CALUDE_superhero_payment_l987_98715


namespace NUMINAMATH_CALUDE_equal_fractions_from_given_numbers_l987_98755

theorem equal_fractions_from_given_numbers : 
  let numbers : Finset ℕ := {2, 4, 5, 6, 12, 15}
  ∃ (a b c d e f : ℕ), 
    a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a : ℚ) / b = (c : ℚ) / d ∧ (c : ℚ) / d = (e : ℚ) / f :=
by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_from_given_numbers_l987_98755


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l987_98731

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l987_98731


namespace NUMINAMATH_CALUDE_alice_apples_l987_98770

theorem alice_apples (A : ℕ) : 
  A > 2 →
  A % 9 = 2 →
  A % 10 = 2 →
  A % 11 = 2 →
  (∀ B : ℕ, B > 2 → B % 9 = 2 → B % 10 = 2 → B % 11 = 2 → A ≤ B) →
  A = 992 := by
sorry

end NUMINAMATH_CALUDE_alice_apples_l987_98770


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_conditions_l987_98719

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

theorem equality_conditions (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2)) ↔
  (x₁ * y₁ - z₁^2 = x₂ * y₂ - z₂^2 ∧ x₁ = x₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_conditions_l987_98719


namespace NUMINAMATH_CALUDE_pizza_slices_per_child_l987_98744

/-- Calculates the number of pizza slices each child wants given the following conditions:
  * There are 2 adults and 6 children
  * Each adult wants 3 slices
  * They order 3 pizzas with 4 slices each
-/
theorem pizza_slices_per_child 
  (num_adults : Nat) 
  (num_children : Nat) 
  (slices_per_adult : Nat) 
  (num_pizzas : Nat) 
  (slices_per_pizza : Nat) 
  (h1 : num_adults = 2) 
  (h2 : num_children = 6) 
  (h3 : slices_per_adult = 3) 
  (h4 : num_pizzas = 3) 
  (h5 : slices_per_pizza = 4) : 
  (num_pizzas * slices_per_pizza - num_adults * slices_per_adult) / num_children = 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_child_l987_98744


namespace NUMINAMATH_CALUDE_sine_shift_equivalence_l987_98777

theorem sine_shift_equivalence :
  ∀ x : ℝ, Real.sin (2 * x + π / 6) = Real.sin (2 * (x + π / 4) - π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_equivalence_l987_98777


namespace NUMINAMATH_CALUDE_factorization_equality_l987_98758

theorem factorization_equality (x : ℝ) :
  (x^2 + 5*x + 2) * (x^2 + 5*x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l987_98758


namespace NUMINAMATH_CALUDE_second_oil_price_l987_98706

/-- Given two types of oil mixed together, calculate the price of the second oil -/
theorem second_oil_price (volume1 volume2 : ℝ) (price1 mixed_price : ℝ) :
  volume1 = 10 →
  volume2 = 5 →
  price1 = 50 →
  mixed_price = 55.33 →
  (volume1 * price1 + volume2 * (volume1 * price1 + volume2 * mixed_price - volume1 * price1) / volume2) / (volume1 + volume2) = mixed_price →
  (volume1 * price1 + volume2 * mixed_price - volume1 * price1) / volume2 = 65.99 := by
  sorry

#eval (10 * 50 + 5 * 55.33 * 3 - 10 * 50) / 5

end NUMINAMATH_CALUDE_second_oil_price_l987_98706


namespace NUMINAMATH_CALUDE_even_function_k_value_l987_98721

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 3

theorem even_function_k_value (k : ℝ) :
  (∀ x, f k x = f k (-x)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_k_value_l987_98721


namespace NUMINAMATH_CALUDE_unique_polynomial_mapping_l987_98786

/-- A second-degree polynomial in two variables -/
def p (x y : ℕ) : ℕ := ((x + y)^2 + 3*x + y) / 2

/-- Theorem stating the existence of a unique mapping for non-negative integers -/
theorem unique_polynomial_mapping :
  ∀ n : ℕ, ∃! (k m : ℕ), p k m = n :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_mapping_l987_98786


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l987_98703

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a) 
  (h_prod : a 4 * a 8 = 9) : 
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l987_98703


namespace NUMINAMATH_CALUDE_complex_expression_equals_half_l987_98761

theorem complex_expression_equals_half :
  |2 - Real.sqrt 2| - Real.sqrt (1/12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_half_l987_98761


namespace NUMINAMATH_CALUDE_teacher_worksheets_l987_98783

theorem teacher_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) : 
  problems_per_worksheet = 3 →
  graded_worksheets = 7 →
  remaining_problems = 24 →
  graded_worksheets + (remaining_problems / problems_per_worksheet) = 15 := by
  sorry

end NUMINAMATH_CALUDE_teacher_worksheets_l987_98783


namespace NUMINAMATH_CALUDE_no_inscribed_circle_pentagon_l987_98769

/-- A pentagon with side lengths a, b, c, d, e has an inscribed circle if and only if
    there exists a positive real number r such that
    2(a + b + c + d + e) = (a + b - c - d + e)(a - b + c - d + e)(-a + b + c - d + e)(-a - b + c + d + e)/r^2 -/
def has_inscribed_circle (a b c d e : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 2*(a + b + c + d + e) = (a + b - c - d + e)*(a - b + c - d + e)*(-a + b + c - d + e)*(-a - b + c + d + e)/(r^2)

/-- Theorem: There does not exist a pentagon with side lengths 3, 4, 9, 11, and 13 cm
    that has an inscribed circle -/
theorem no_inscribed_circle_pentagon : ¬ has_inscribed_circle 3 4 9 11 13 := by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_circle_pentagon_l987_98769


namespace NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l987_98768

theorem largest_k_for_distinct_roots : 
  ∃ (k : ℤ), k = 8 ∧ 
  (∀ (x : ℝ), x^2 - 6*x + k = 0 → (∃ (y : ℝ), x ≠ y ∧ y^2 - 6*y + k = 0)) ∧
  (∀ (m : ℤ), m > k → ¬(∀ (x : ℝ), x^2 - 6*x + m = 0 → (∃ (y : ℝ), x ≠ y ∧ y^2 - 6*y + m = 0))) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l987_98768


namespace NUMINAMATH_CALUDE_set_inclusion_condition_l987_98704

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}

def B : Set ℝ := {x : ℝ | x^2 + 2*x - 3 > 0}

def C (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem set_inclusion_condition (a : ℝ) : 
  C a ⊆ A ∩ B ↔ (1 ≤ a ∧ a ≤ 2) ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_set_inclusion_condition_l987_98704


namespace NUMINAMATH_CALUDE_max_oranges_donated_l987_98736

theorem max_oranges_donated (n : ℕ) : ∃ (q r : ℕ), n = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_oranges_donated_l987_98736


namespace NUMINAMATH_CALUDE_equation_solutions_count_l987_98792

theorem equation_solutions_count :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (m n : ℤ), (m, n) ∈ s ↔ m^4 + 8*n^2 + 425 = n^4 + 42*m^2) ∧
    s.card = 16 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l987_98792


namespace NUMINAMATH_CALUDE_elastic_collision_momentum_exchange_l987_98756

/-- Represents a particle with mass and velocity -/
structure Particle where
  mass : ℝ
  velocity : ℝ

/-- Calculates the momentum of a particle -/
def momentum (p : Particle) : ℝ := p.mass * p.velocity

/-- Represents the state of two particles before and after a collision -/
structure CollisionState where
  particle1 : Particle
  particle2 : Particle

/-- Defines an elastic head-on collision between two identical particles -/
def elasticCollision (initial : CollisionState) (final : CollisionState) : Prop :=
  initial.particle1.mass = initial.particle2.mass ∧
  initial.particle1.mass = final.particle1.mass ∧
  initial.particle2.velocity = 0 ∧
  momentum initial.particle1 + momentum initial.particle2 = momentum final.particle1 + momentum final.particle2 ∧
  (momentum initial.particle1)^2 + (momentum initial.particle2)^2 = (momentum final.particle1)^2 + (momentum final.particle2)^2

theorem elastic_collision_momentum_exchange 
  (initial final : CollisionState)
  (h_elastic : elasticCollision initial final)
  (h_initial_momentum : momentum initial.particle1 = p ∧ momentum initial.particle2 = 0) :
  momentum final.particle1 = 0 ∧ momentum final.particle2 = p := by
  sorry

end NUMINAMATH_CALUDE_elastic_collision_momentum_exchange_l987_98756


namespace NUMINAMATH_CALUDE_cupcake_count_l987_98742

theorem cupcake_count (initial : ℕ) (sold : ℕ) (additional : ℕ) : 
  initial ≥ sold → initial - sold + additional = (initial - sold) + additional := by
  sorry

end NUMINAMATH_CALUDE_cupcake_count_l987_98742


namespace NUMINAMATH_CALUDE_saturday_price_calculation_l987_98762

theorem saturday_price_calculation (original_price : ℝ) 
  (h1 : original_price = 180) 
  (sale_discount : ℝ) (h2 : sale_discount = 0.5)
  (saturday_discount : ℝ) (h3 : saturday_discount = 0.2) : 
  original_price * (1 - sale_discount) * (1 - saturday_discount) = 72 := by
  sorry

end NUMINAMATH_CALUDE_saturday_price_calculation_l987_98762


namespace NUMINAMATH_CALUDE_runner_speed_problem_l987_98774

theorem runner_speed_problem (total_distance : ℝ) (total_time : ℝ) (first_segment_distance : ℝ) (first_segment_speed : ℝ) (last_segment_distance : ℝ) :
  total_distance = 16 →
  total_time = 1.5 →
  first_segment_distance = 10 →
  first_segment_speed = 12 →
  last_segment_distance = 6 →
  (last_segment_distance / (total_time - (first_segment_distance / first_segment_speed))) = 9 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_problem_l987_98774


namespace NUMINAMATH_CALUDE_number_problem_l987_98764

theorem number_problem (x : ℝ) : 
  (15 / 100 * 40 = 25 / 100 * x + 2) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l987_98764


namespace NUMINAMATH_CALUDE_austin_robot_purchase_l987_98717

theorem austin_robot_purchase (num_robots : ℕ) (cost_per_robot tax change : ℚ) : 
  num_robots = 7 →
  cost_per_robot = 8.75 →
  tax = 7.22 →
  change = 11.53 →
  (num_robots : ℚ) * cost_per_robot + tax + change = 80 := by
  sorry

end NUMINAMATH_CALUDE_austin_robot_purchase_l987_98717


namespace NUMINAMATH_CALUDE_distinct_lunches_l987_98760

/-- The number of main course options available --/
def main_course_options : ℕ := 4

/-- The number of beverage options available --/
def beverage_options : ℕ := 3

/-- The number of snack options available --/
def snack_options : ℕ := 2

/-- The total number of distinct possible lunches --/
def total_lunches : ℕ := main_course_options * beverage_options * snack_options

/-- Theorem stating that the total number of distinct possible lunches is 24 --/
theorem distinct_lunches : total_lunches = 24 := by
  sorry

end NUMINAMATH_CALUDE_distinct_lunches_l987_98760


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l987_98713

theorem right_triangle_max_ratio :
  ∀ (a b c h : ℝ),
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 + b^2 = c^2 →
  h * c = a * b →
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l987_98713


namespace NUMINAMATH_CALUDE_number_of_players_is_16_l987_98779

def jersey_cost : ℚ := 25
def shorts_cost : ℚ := 15.20
def socks_cost : ℚ := 6.80
def total_cost : ℚ := 752

def equipment_cost_per_player : ℚ := jersey_cost + shorts_cost + socks_cost

theorem number_of_players_is_16 :
  (total_cost / equipment_cost_per_player : ℚ) = 16 := by sorry

end NUMINAMATH_CALUDE_number_of_players_is_16_l987_98779


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l987_98726

theorem geometric_sequence_property (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h_sum : a 1 + a 2 + a 3 = 18)
  (h_inv_sum : 1 / a 1 + 1 / a 2 + 1 / a 3 = 2) :
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l987_98726


namespace NUMINAMATH_CALUDE_f_properties_l987_98739

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x - 3)

theorem f_properties :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 81) ∧
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ → f x₂ < f x₁) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l987_98739


namespace NUMINAMATH_CALUDE_leadership_selection_count_l987_98754

def tribe_size : ℕ := 12
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 2

def leadership_selection_ways : ℕ :=
  tribe_size *
  (Nat.choose (tribe_size - 1) num_supporting_chiefs) *
  (Nat.choose (tribe_size - 1 - num_supporting_chiefs) (num_supporting_chiefs * num_inferior_officers_per_chief) /
   Nat.factorial num_supporting_chiefs)

theorem leadership_selection_count :
  leadership_selection_ways = 248040 :=
by sorry

end NUMINAMATH_CALUDE_leadership_selection_count_l987_98754
