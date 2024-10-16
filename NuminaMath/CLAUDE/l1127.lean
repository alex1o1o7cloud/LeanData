import Mathlib

namespace NUMINAMATH_CALUDE_technician_permanent_percentage_l1127_112721

/-- Represents the composition of workers in a factory -/
structure Factory where
  total_workers : ℕ
  technicians : ℕ
  non_technicians : ℕ
  permanent_non_technicians : ℕ
  temporary_workers : ℕ

/-- The conditions of the factory -/
def factory_conditions (f : Factory) : Prop :=
  f.technicians = f.total_workers / 2 ∧
  f.non_technicians = f.total_workers / 2 ∧
  f.permanent_non_technicians = f.non_technicians / 2 ∧
  f.temporary_workers = f.total_workers / 2

/-- The theorem to be proved -/
theorem technician_permanent_percentage (f : Factory) 
  (h : factory_conditions f) : 
  (f.technicians - (f.temporary_workers - f.permanent_non_technicians)) * 2 = f.technicians := by
  sorry

#check technician_permanent_percentage

end NUMINAMATH_CALUDE_technician_permanent_percentage_l1127_112721


namespace NUMINAMATH_CALUDE_smallest_perfect_square_with_perfect_square_factors_l1127_112722

/-- A function that returns the number of positive integer factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square_num (n : ℕ) : Prop := sorry

theorem smallest_perfect_square_with_perfect_square_factors : 
  ∀ n : ℕ, n > 1 → is_perfect_square n → is_perfect_square_num (num_factors n) → n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_with_perfect_square_factors_l1127_112722


namespace NUMINAMATH_CALUDE_distance_after_7km_l1127_112772

/-- Regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Point on the perimeter of the hexagon -/
structure PerimeterPoint (h : RegularHexagon) where
  distance_from_start : ℝ
  on_perimeter : distance_from_start ≥ 0 ∧ distance_from_start ≤ 6 * h.side_length

/-- The distance from the starting point to a point on the perimeter -/
def distance_to_start (h : RegularHexagon) (p : PerimeterPoint h) : ℝ :=
  sorry

theorem distance_after_7km (h : RegularHexagon) (p : PerimeterPoint h) 
  (h_distance : p.distance_from_start = 7) :
  distance_to_start h p = 2 :=
sorry

end NUMINAMATH_CALUDE_distance_after_7km_l1127_112772


namespace NUMINAMATH_CALUDE_value_of_b_is_two_l1127_112727

def f (x : ℝ) := x^2 - 2*x + 2

theorem value_of_b_is_two :
  ∃ b : ℝ, b > 1 ∧
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_value_of_b_is_two_l1127_112727


namespace NUMINAMATH_CALUDE_probability_of_red_from_B_mutually_exclusive_events_l1127_112750

structure Bag where
  red : ℕ
  white : ℕ
  black : ℕ

def bagA : Bag := ⟨5, 2, 3⟩
def bagB : Bag := ⟨4, 3, 3⟩

def totalBalls (bag : Bag) : ℕ := bag.red + bag.white + bag.black

def P_A1 : ℚ := bagA.red / totalBalls bagA
def P_A2 : ℚ := bagA.white / totalBalls bagA
def P_A3 : ℚ := bagA.black / totalBalls bagA

def P_B_given_A1 : ℚ := (bagB.red + 1) / (totalBalls bagB + 1)
def P_B_given_A2 : ℚ := bagB.red / (totalBalls bagB + 1)
def P_B_given_A3 : ℚ := bagB.red / (totalBalls bagB + 1)

def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

theorem probability_of_red_from_B : P_B = 9 / 22 := by sorry

theorem mutually_exclusive_events : P_A1 + P_A2 + P_A3 = 1 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_from_B_mutually_exclusive_events_l1127_112750


namespace NUMINAMATH_CALUDE_cos_beta_value_l1127_112755

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 2) (h4 : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_beta_value_l1127_112755


namespace NUMINAMATH_CALUDE_great_m_conference_teams_l1127_112782

/-- The number of games played when each team in a conference plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of teams in the GREAT M conference -/
def num_teams : ℕ := 8

theorem great_m_conference_teams :
  num_games num_teams = 28 ∧ num_teams > 0 := by
  sorry

#eval num_games num_teams -- Should output 28

end NUMINAMATH_CALUDE_great_m_conference_teams_l1127_112782


namespace NUMINAMATH_CALUDE_num_organizations_in_foundation_l1127_112774

/-- The number of organizations in a public foundation --/
def num_organizations (total_raised : ℚ) (donation_percentage : ℚ) (amount_per_org : ℚ) : ℚ :=
  (total_raised * donation_percentage) / amount_per_org

/-- Theorem stating the number of organizations in the public foundation --/
theorem num_organizations_in_foundation : 
  num_organizations 2500 0.8 250 = 8 := by
  sorry

end NUMINAMATH_CALUDE_num_organizations_in_foundation_l1127_112774


namespace NUMINAMATH_CALUDE_corn_stalks_per_row_l1127_112729

/-- Proves that given 5 rows of corn, 8 corn stalks per bushel, and a total harvest of 50 bushels,
    the number of corn stalks in each row is 80. -/
theorem corn_stalks_per_row 
  (rows : ℕ) 
  (stalks_per_bushel : ℕ) 
  (total_bushels : ℕ) 
  (h1 : rows = 5)
  (h2 : stalks_per_bushel = 8)
  (h3 : total_bushels = 50) :
  (total_bushels * stalks_per_bushel) / rows = 80 := by
  sorry

end NUMINAMATH_CALUDE_corn_stalks_per_row_l1127_112729


namespace NUMINAMATH_CALUDE_cards_eaten_ratio_l1127_112742

/-- Given that Benny bought 4 new baseball cards and has 34 cards left after his dog ate some,
    prove that the ratio of cards eaten by the dog to the total number of cards before
    the dog ate them is (X - 30) / (X + 4), where X is the number of cards Benny had
    before buying the new ones. -/
theorem cards_eaten_ratio (X : ℕ) : 
  let cards_bought : ℕ := 4
  let cards_left : ℕ := 34
  let total_before_eating : ℕ := X + cards_bought
  let cards_eaten : ℕ := total_before_eating - cards_left
  cards_eaten / total_before_eating = (X - 30) / (X + 4) :=
by sorry

end NUMINAMATH_CALUDE_cards_eaten_ratio_l1127_112742


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1127_112708

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {-1, 0, 1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1127_112708


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l1127_112757

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the concept of two circles not intersecting
def NonIntersecting (c1 c2 : Circle) : Prop := sorry

-- Define the concept of a circle being tangent to another circle
def IsTangent (c1 c2 : Circle) : Prop := sorry

-- Define the concept of a hyperbola
def Hyperbola : Set (ℝ × ℝ) := sorry

-- Define the concept of an ellipse
def Ellipse : Set (ℝ × ℝ) := sorry

-- The main theorem
theorem trajectory_of_moving_circle 
  (O₁ O₂ : Circle) 
  (h_diff : O₁.radius ≠ O₂.radius) 
  (h_non_intersect : NonIntersecting O₁ O₂) :
  ∃ (trajectory : Set (ℝ × ℝ)), 
    (∀ (O : Circle), IsTangent O O₁ ∧ IsTangent O O₂ → O.center ∈ trajectory) ∧
    ((trajectory = Hyperbola) ∨ (trajectory = Ellipse)) := 
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l1127_112757


namespace NUMINAMATH_CALUDE_division_problem_l1127_112771

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 12 → 
  divisor = 17 → 
  remainder = 7 → 
  dividend = divisor * quotient + remainder →
  quotient = 0 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1127_112771


namespace NUMINAMATH_CALUDE_smallest_integer_divisibility_l1127_112784

theorem smallest_integer_divisibility (n : ℕ) : 
  ∃ (a_n : ℤ), 
    (a_n > (Real.sqrt 3 + 1)^(2*n)) ∧ 
    (∀ (x : ℤ), x > (Real.sqrt 3 + 1)^(2*n) → a_n ≤ x) ∧ 
    (∃ (k : ℤ), a_n = 2^(n+1) * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_divisibility_l1127_112784


namespace NUMINAMATH_CALUDE_walking_equations_correct_l1127_112788

/-- Represents the speeds and distances of two people walking --/
structure WalkingScenario where
  distance : ℝ
  catchup_time : ℝ
  meet_time : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- The system of equations correctly represents the walking scenario --/
def correct_equations (s : WalkingScenario) : Prop :=
  (10 * s.speed_b - 10 * s.speed_a = s.distance) ∧
  (2 * s.speed_a + 2 * s.speed_b = s.distance)

/-- The given scenario satisfies the conditions --/
def satisfies_conditions (s : WalkingScenario) : Prop :=
  s.distance = 50 ∧
  s.catchup_time = 10 ∧
  s.meet_time = 2 ∧
  s.speed_a > 0 ∧
  s.speed_b > 0

theorem walking_equations_correct (s : WalkingScenario) 
  (h : satisfies_conditions s) : correct_equations s := by
  sorry


end NUMINAMATH_CALUDE_walking_equations_correct_l1127_112788


namespace NUMINAMATH_CALUDE_function_domain_l1127_112766

/-- The function y = √(x-1) / (x-2) is defined for x ≥ 1 and x ≠ 2 -/
theorem function_domain (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x - 1) / (x - 2)) ↔ (x ≥ 1 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_function_domain_l1127_112766


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l1127_112798

theorem geometric_sequence_value (a : ℝ) : 
  (∃ (r : ℝ), 1 * r = a ∧ a * r = (1/16 : ℝ)) → 
  (a = (1/4 : ℝ) ∨ a = -(1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l1127_112798


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l1127_112764

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def total_jelly_beans : ℕ := 4000
def red_jelly_beans : ℕ := (3 * total_jelly_beans) / 4
def coconut_flavored_jelly_beans : ℕ := 750

theorem jelly_bean_ratio :
  Ratio.mk coconut_flavored_jelly_beans red_jelly_beans = Ratio.mk 1 4 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l1127_112764


namespace NUMINAMATH_CALUDE_smallest_seven_digit_binary_l1127_112770

theorem smallest_seven_digit_binary : ∀ n : ℕ, n > 0 → (
  (Nat.log 2 n + 1 = 7) ↔ n ≥ 64 ∧ ∀ m : ℕ, m > 0 ∧ m < 64 → Nat.log 2 m + 1 < 7
) := by sorry

end NUMINAMATH_CALUDE_smallest_seven_digit_binary_l1127_112770


namespace NUMINAMATH_CALUDE_message_decoding_l1127_112793

-- Define the Russian alphabet
def RussianAlphabet : List Char := ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

-- Define the digit groups
def DigitGroups : List (List Nat) := [[1], [3, 7, 8], [0, 4, 5, 6]]

-- Define the encoding function
def encode (groups : List (List Nat)) (alphabet : List Char) : Nat → Option Char := sorry

-- Define the decoding function
def decode (groups : List (List Nat)) (alphabet : List Char) : List Nat → String := sorry

-- Theorem statement
theorem message_decoding :
  decode DigitGroups RussianAlphabet [8, 7, 3, 1, 4, 6, 5, 0, 7, 3, 8, 1] = "НАУКА" ∧
  encode DigitGroups RussianAlphabet 8 = some 'Н' ∧
  encode DigitGroups RussianAlphabet 7 = some 'А' ∧
  encode DigitGroups RussianAlphabet 3 = some 'У' ∧
  encode DigitGroups RussianAlphabet 1 = some 'К' ∧
  encode DigitGroups RussianAlphabet 4 = some 'А' ∧
  encode DigitGroups RussianAlphabet 6 = none ∧
  encode DigitGroups RussianAlphabet 5 = some 'К' ∧
  encode DigitGroups RussianAlphabet 0 = none := by
  sorry

end NUMINAMATH_CALUDE_message_decoding_l1127_112793


namespace NUMINAMATH_CALUDE_divide_by_ten_equals_two_l1127_112794

theorem divide_by_ten_equals_two : ∃ x : ℚ, x * 5 = 100 ∧ x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_ten_equals_two_l1127_112794


namespace NUMINAMATH_CALUDE_greatest_x_lcm_l1127_112712

theorem greatest_x_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 10 14) = 70) → x ≤ 70 ∧ ∃ y : ℕ, y = 70 ∧ Nat.lcm y (Nat.lcm 10 14) = 70 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_lcm_l1127_112712


namespace NUMINAMATH_CALUDE_factors_of_1320_l1127_112765

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def count_factors (factorization : List (ℕ × ℕ)) : ℕ := sorry

theorem factors_of_1320 :
  let factorization := prime_factorization 1320
  count_factors factorization = 24 := by sorry

end NUMINAMATH_CALUDE_factors_of_1320_l1127_112765


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_negative_two_satisfies_inequality_smallest_integer_is_negative_two_l1127_112792

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x^2 - 4 < 20) → x ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  3 * (-2)^2 - 4 < 20 :=
by
  sorry

theorem smallest_integer_is_negative_two :
  ∃ x : ℤ, (∀ y : ℤ, (3 * y^2 - 4 < 20) → y ≥ x) ∧ (3 * x^2 - 4 < 20) ∧ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_negative_two_satisfies_inequality_smallest_integer_is_negative_two_l1127_112792


namespace NUMINAMATH_CALUDE_women_doubles_tournament_handshakes_l1127_112787

/-- The number of handshakes in a women's doubles tennis tournament -/
def num_handshakes (num_teams : ℕ) (team_size : ℕ) : ℕ :=
  let total_players := num_teams * team_size
  let handshakes_per_player := total_players - team_size
  (total_players * handshakes_per_player) / 2

theorem women_doubles_tournament_handshakes :
  num_handshakes 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_women_doubles_tournament_handshakes_l1127_112787


namespace NUMINAMATH_CALUDE_selection_probabilities_l1127_112790

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The probability of selecting exactly 2 boys and 1 girl -/
def prob_2boys_1girl : ℚ := 3/5

/-- The probability of selecting at least 1 girl -/
def prob_at_least_1girl : ℚ := 4/5

theorem selection_probabilities :
  (Nat.choose num_boys 2 * Nat.choose num_girls 1) / Nat.choose total_people num_selected = prob_2boys_1girl ∧
  1 - (Nat.choose num_boys num_selected) / Nat.choose total_people num_selected = prob_at_least_1girl :=
by sorry

end NUMINAMATH_CALUDE_selection_probabilities_l1127_112790


namespace NUMINAMATH_CALUDE_right_triangle_sin_I_l1127_112768

theorem right_triangle_sin_I (G H I : Real) :
  -- GHI is a right triangle with ∠G = 90°
  G + H + I = Real.pi →
  G = Real.pi / 2 →
  -- sin H = 3/5
  Real.sin H = 3 / 5 →
  -- Prove: sin I = 4/5
  Real.sin I = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_I_l1127_112768


namespace NUMINAMATH_CALUDE_part_I_part_II_l1127_112720

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → m^2 - 3*m + x - 1 ≤ 0

def q (m a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ m - a*x ≤ 0

-- Part I
theorem part_I : 
  ∃ S : Set ℝ, S = {m : ℝ | (m < 1 ∨ (1 < m ∧ m ≤ 2)) ∧ 
  ((p m ∧ ¬q m 1) ∨ (¬p m ∧ q m 1))} := by sorry

-- Part II
theorem part_II : 
  ∃ S : Set ℝ, S = {a : ℝ | a ≥ 2 ∨ a ≤ -2} ∧
  ∀ m : ℝ, (p m → q m a) ∧ ¬(q m a → p m) := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1127_112720


namespace NUMINAMATH_CALUDE_max_a_value_l1127_112740

theorem max_a_value (a : ℝ) : (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l1127_112740


namespace NUMINAMATH_CALUDE_asha_money_problem_l1127_112734

/-- Asha's money problem -/
theorem asha_money_problem (brother_loan : ℕ) (father_loan : ℕ) (mother_loan : ℕ) (granny_gift : ℕ) (savings : ℕ) (spent_fraction : ℚ) :
  brother_loan = 20 →
  father_loan = 40 →
  mother_loan = 30 →
  granny_gift = 70 →
  savings = 100 →
  spent_fraction = 3 / 4 →
  ∃ (remaining : ℕ), remaining = 65 ∧ 
    remaining = (brother_loan + father_loan + mother_loan + granny_gift + savings) - 
                (spent_fraction * (brother_loan + father_loan + mother_loan + granny_gift + savings)).floor :=
by sorry

end NUMINAMATH_CALUDE_asha_money_problem_l1127_112734


namespace NUMINAMATH_CALUDE_m_range_l1127_112714

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x^2 - 4| + x^2 + m*x

/-- The condition that f has two distinct zero points in (0, 3) -/
def has_two_distinct_zeros (m : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ x < 3 ∧ 0 < y ∧ y < 3 ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0

/-- The theorem stating the range of m -/
theorem m_range (m : ℝ) :
  has_two_distinct_zeros m → -14/3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1127_112714


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l1127_112701

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  distance_sum : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def compute_ellipse_params (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem: sum of ellipse parameters equals 14 -/
theorem ellipse_param_sum (e : Ellipse) : 
  let ep := compute_ellipse_params e
  e.F₁ = (0, 2) → e.F₂ = (6, 2) → e.distance_sum = 10 →
  ep.h + ep.k + ep.a + ep.b = 14 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l1127_112701


namespace NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1127_112738

/-- Calculates the number of pages read on the fourth day given the reading pattern for a book -/
theorem pages_read_on_fourth_day
  (total_pages : ℕ)
  (day1_pages : ℕ)
  (day2_multiplier : ℕ)
  (day3_additional : ℕ)
  (h1 : total_pages = 354)
  (h2 : day1_pages = 63)
  (h3 : day2_multiplier = 2)
  (h4 : day3_additional = 10) :
  total_pages - (day1_pages + day2_multiplier * day1_pages + (day2_multiplier * day1_pages + day3_additional)) = 29 :=
by sorry

end NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1127_112738


namespace NUMINAMATH_CALUDE_bens_savings_proof_l1127_112743

/-- Ben's daily savings before his parents' contributions --/
def daily_savings : ℕ := 50 - 15

/-- The number of days Ben saved money --/
def num_days : ℕ := 7

/-- Ben's total savings after his mom doubled it --/
def doubled_savings : ℕ := 2 * (daily_savings * num_days)

/-- Ben's final amount after 7 days --/
def final_amount : ℕ := 500

/-- The additional amount Ben's dad gave him --/
def dads_contribution : ℕ := final_amount - doubled_savings

theorem bens_savings_proof :
  dads_contribution = 10 :=
sorry

end NUMINAMATH_CALUDE_bens_savings_proof_l1127_112743


namespace NUMINAMATH_CALUDE_opposite_of_three_l1127_112791

theorem opposite_of_three : -(3 : ℝ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1127_112791


namespace NUMINAMATH_CALUDE_m_range_l1127_112732

-- Define the propositions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x)

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (necessary_but_not_sufficient (p m) q) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1127_112732


namespace NUMINAMATH_CALUDE_custom_op_five_three_l1127_112759

/-- Custom binary operation " defined as m " n = n^2 - m -/
def custom_op (m n : ℤ) : ℤ := n^2 - m

/-- Theorem stating that 5 " 3 = 4 -/
theorem custom_op_five_three : custom_op 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_five_three_l1127_112759


namespace NUMINAMATH_CALUDE_inverse_composition_l1127_112704

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Condition: f⁻¹(g(x)) = 7x - 4
axiom condition : ∀ x, f_inv (g x) = 7 * x - 4

-- Theorem to prove
theorem inverse_composition :
  g_inv (f 2) = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_l1127_112704


namespace NUMINAMATH_CALUDE_unique_solution_is_two_l1127_112707

theorem unique_solution_is_two : 
  ∃! n : ℕ+, 
    (n : ℕ) ∣ (Nat.totient n)^(Nat.divisors n).card + 1 ∧ 
    ¬((Nat.divisors n).card^5 ∣ (n : ℕ)^(Nat.totient n) - 1) ∧
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_two_l1127_112707


namespace NUMINAMATH_CALUDE_empty_quadratic_set_implies_m_greater_than_one_l1127_112751

theorem empty_quadratic_set_implies_m_greater_than_one (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m ≠ 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_quadratic_set_implies_m_greater_than_one_l1127_112751


namespace NUMINAMATH_CALUDE_opposite_pairs_l1127_112706

theorem opposite_pairs : 
  ((-3)^2 = -(-3^2)) ∧ 
  ((-3)^2 ≠ -(3^2)) ∧ 
  ((-2)^3 ≠ -(-2^3)) ∧ 
  (|-2|^3 ≠ -(|-2^3|)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l1127_112706


namespace NUMINAMATH_CALUDE_pop_survey_l1127_112767

theorem pop_survey (total : ℕ) (pop_angle : ℕ) (pop_count : ℕ) : 
  total = 472 →
  pop_angle = 251 →
  (pop_count : ℝ) / total * 360 ≥ pop_angle.pred →
  (pop_count : ℝ) / total * 360 < pop_angle.succ →
  pop_count = 329 := by
sorry

end NUMINAMATH_CALUDE_pop_survey_l1127_112767


namespace NUMINAMATH_CALUDE_max_eccentricity_sum_l1127_112783

/-- Given an ellipse with eccentricity e₁ and a hyperbola with eccentricity e₂ sharing the same foci,
    if the minor axis of the ellipse is three times the length of the conjugate axis of the hyperbola,
    then the maximum value of 1/e₁ + 1/e₂ is 10/3. -/
theorem max_eccentricity_sum (e₁ e₂ : ℝ) (h_ellipse : 0 < e₁ ∧ e₁ < 1) (h_hyperbola : e₂ > 1)
  (h_foci : ∃ (c : ℝ), c > 0 ∧ c^2 * e₁^2 = c^2 * e₂^2)
  (h_axes : ∃ (b₁ b₂ : ℝ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ = 3 * b₂) :
  (∀ x y : ℝ, x > 0 ∧ y > 1 → 1/x + 1/y ≤ 10/3) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 1 ∧ 1/x + 1/y = 10/3) :=
sorry

end NUMINAMATH_CALUDE_max_eccentricity_sum_l1127_112783


namespace NUMINAMATH_CALUDE_A_completes_in_20_days_l1127_112745

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- The number of days A and B work together -/
def together_days : ℝ := 8

/-- The number of days B works alone after A quits -/
def B_alone_days : ℝ := 10

/-- The total amount of work (100% of the project) -/
def total_work : ℝ := 1

/-- Theorem stating that A can complete the project alone in 20 days -/
theorem A_completes_in_20_days :
  ∃ A_days : ℝ,
    A_days = 20 ∧
    together_days * (1 / A_days + 1 / B_days) + B_alone_days * (1 / B_days) = total_work :=
by sorry

end NUMINAMATH_CALUDE_A_completes_in_20_days_l1127_112745


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1127_112719

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Convert a line from slope-intercept form to general form -/
def toGeneralForm (l : Line) : GeneralLineEquation :=
  { a := l.slope, b := -1, c := l.y_intercept }

/-- The main theorem -/
theorem perpendicular_line_equation 
  (l : Line) 
  (h1 : l.y_intercept = 2) 
  (h2 : perpendicular l { slope := -1, y_intercept := 3 }) : 
  toGeneralForm l = { a := 1, b := -1, c := 2 } := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1127_112719


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1127_112737

theorem min_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_min_sum_reciprocals_l1127_112737


namespace NUMINAMATH_CALUDE_distance_product_range_l1127_112778

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

-- Define a line with slope 45°
def line_slope_45 (x₁ y₁ x₂ y₂ : ℝ) : Prop := y₂ - y₁ = x₂ - x₁

-- Define the property of a point being on a curve
def point_on_curve (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop := C x y

-- Define the property of a line intersecting a curve at two distinct points
def line_intersects_curve_at_two_points (C : ℝ → ℝ → Prop) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  line_slope_45 x₁ y₁ x₂ y₂ ∧ line_slope_45 x₁ y₁ x₃ y₃ ∧
  point_on_curve C x₂ y₂ ∧ point_on_curve C x₃ y₃ ∧
  (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2)

-- Define the product of distances
def distance_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  distance x₁ y₁ x₂ y₂ * distance x₁ y₁ x₃ y₃

-- Main theorem
theorem distance_product_range :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    point_on_curve C₁ x₁ y₁ →
    line_intersects_curve_at_two_points C₂ x₁ y₁ x₂ y₂ x₃ y₃ →
    4 ≤ distance_product x₁ y₁ x₂ y₂ x₃ y₃ ∧
    distance_product x₁ y₁ x₂ y₂ x₃ y₃ < 8 ∨
    8 < distance_product x₁ y₁ x₂ y₂ x₃ y₃ ∧
    distance_product x₁ y₁ x₂ y₂ x₃ y₃ ≤ 200 :=
sorry

end NUMINAMATH_CALUDE_distance_product_range_l1127_112778


namespace NUMINAMATH_CALUDE_tina_total_time_l1127_112724

def assignment_time : ℕ := 15
def total_sticky_keys : ℕ := 25
def time_per_key : ℕ := 5
def cleaned_keys : ℕ := 1

def remaining_keys : ℕ := total_sticky_keys - cleaned_keys
def cleaning_time : ℕ := remaining_keys * time_per_key
def total_time : ℕ := cleaning_time + assignment_time

theorem tina_total_time : total_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_time_l1127_112724


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1127_112731

-- Define the simple interest calculation function
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

-- State the theorem
theorem interest_rate_calculation (principal time interest : ℚ) 
  (h1 : principal = 8925)
  (h2 : time = 5)
  (h3 : interest = 4016.25)
  (h4 : simple_interest principal (9 : ℚ) time = interest) :
  ∃ (rate : ℚ), simple_interest principal rate time = interest ∧ rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l1127_112731


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1127_112716

theorem quadratic_equation_roots (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x => p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f (-1) = 0) → (f (-r * (p - q) / (p * (q - r))) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1127_112716


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1127_112789

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => (n - 1) * (totalEmployees / sampleSize) + firstSample

/-- Theorem: In a systematic sampling of 40 samples from 200 employees, 
    if the 5th sample is 22, then the 10th sample is 47 -/
theorem systematic_sampling_theorem 
  (totalEmployees : ℕ) (sampleSize : ℕ) (groupSize : ℕ) (fifthSample : ℕ) :
  totalEmployees = 200 →
  sampleSize = 40 →
  groupSize = 5 →
  fifthSample = 22 →
  systematicSample totalEmployees sampleSize (fifthSample - (5 - 1) * groupSize) 10 = 47 := by
  sorry

#check systematic_sampling_theorem

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1127_112789


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l1127_112735

def workshop_A : ℕ := 120
def workshop_B : ℕ := 80
def workshop_C : ℕ := 60

def total_production : ℕ := workshop_A + workshop_B + workshop_C

def sample_size_C : ℕ := 3

theorem stratified_sampling_size :
  (workshop_C : ℚ) / total_production = sample_size_C / (13 : ℚ) := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l1127_112735


namespace NUMINAMATH_CALUDE_equation_solutions_l1127_112710

theorem equation_solutions :
  let f (x : ℝ) := (17 * x - x^2) / (x + 2) * (x + (17 - x) / (x + 2))
  ∀ x : ℝ, f x = 56 ↔ x = 1 ∨ x = 65 ∨ x = -25 + Real.sqrt 624 ∨ x = -25 - Real.sqrt 624 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1127_112710


namespace NUMINAMATH_CALUDE_sum_pascal_triangle_rows_8_to_10_l1127_112797

-- Define a function to calculate the sum of interior numbers for a given row
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Define the sum of interior numbers for rows 8, 9, and 10
def sumRows8to10 : ℕ := sumInteriorNumbers 8 + sumInteriorNumbers 9 + sumInteriorNumbers 10

-- Theorem statement
theorem sum_pascal_triangle_rows_8_to_10 : sumRows8to10 = 890 := by
  sorry

end NUMINAMATH_CALUDE_sum_pascal_triangle_rows_8_to_10_l1127_112797


namespace NUMINAMATH_CALUDE_new_profit_percentage_approximation_l1127_112753

/-- Represents the cost distribution and increase percentages for a restaurant --/
structure RestaurantCosts where
  meat_percent : ℝ
  vegetables_percent : ℝ
  dairy_percent : ℝ
  grains_percent : ℝ
  labor_percent : ℝ
  meat_increase : ℝ
  vegetables_increase : ℝ
  dairy_increase : ℝ
  grains_increase : ℝ
  labor_increase : ℝ

/-- Calculates the new profit percentage given the initial costs and increases --/
def calculate_new_profit_percentage (costs : RestaurantCosts) : ℝ :=
  sorry

/-- Theorem stating that the new profit percentage is approximately 56.34% --/
theorem new_profit_percentage_approximation (costs : RestaurantCosts)
  (h1 : costs.meat_percent = 0.30)
  (h2 : costs.vegetables_percent = 0.25)
  (h3 : costs.dairy_percent = 0.20)
  (h4 : costs.grains_percent = 0.20)
  (h5 : costs.labor_percent = 0.05)
  (h6 : costs.meat_increase = 0.12)
  (h7 : costs.vegetables_increase = 0.10)
  (h8 : costs.dairy_increase = 0.08)
  (h9 : costs.grains_increase = 0.06)
  (h10 : costs.labor_increase = 0.05) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |calculate_new_profit_percentage costs - 0.5634| < ε :=
sorry

end NUMINAMATH_CALUDE_new_profit_percentage_approximation_l1127_112753


namespace NUMINAMATH_CALUDE_sum_leq_product_plus_two_l1127_112749

theorem sum_leq_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x * y * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_leq_product_plus_two_l1127_112749


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1127_112775

theorem trigonometric_identities (x : Real) (h : Real.tan x = 2) :
  (2/3 * Real.sin x^2 + 1/4 * Real.cos x^2 = 7/12) ∧
  (2 * Real.sin x^2 - Real.sin x * Real.cos x + Real.cos x^2 = 7/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1127_112775


namespace NUMINAMATH_CALUDE_problem_statement_l1127_112728

theorem problem_statement : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1127_112728


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1127_112741

theorem trigonometric_equation_solution (z : ℝ) : 
  2 * (Real.cos z) * (Real.sin (3 * Real.pi / 2 - z))^3 - 
  5 * (Real.sin z)^2 * (Real.cos z)^2 + 
  (Real.sin z) * (Real.cos (3 * Real.pi / 2 + z))^3 = 
  Real.cos (2 * z) → 
  ∃ (n : ℤ), z = Real.pi / 3 * (3 * ↑n + 1) ∨ z = Real.pi / 3 * (3 * ↑n - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1127_112741


namespace NUMINAMATH_CALUDE_evaluate_expression_l1127_112711

theorem evaluate_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1127_112711


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1127_112705

theorem complex_magnitude_problem (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I) :
  Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1127_112705


namespace NUMINAMATH_CALUDE_max_factors_theorem_l1127_112715

def is_valid_pair (b n : ℕ) : Prop :=
  0 < b ∧ b ≤ 20 ∧ 0 < n ∧ n ≤ 20 ∧ b ≠ n

def num_factors (m : ℕ) : ℕ := (Nat.factors m).length + 1

def max_factors : ℕ := 81

theorem max_factors_theorem :
  ∀ b n : ℕ, is_valid_pair b n →
    num_factors (b^n) ≤ max_factors :=
by sorry

end NUMINAMATH_CALUDE_max_factors_theorem_l1127_112715


namespace NUMINAMATH_CALUDE_angle_ENG_is_45_degrees_l1127_112713

-- Define the rectangle EFGH
structure Rectangle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

-- Define the properties of the rectangle
def is_valid_rectangle (rect : Rectangle) : Prop :=
  rect.E.1 = 0 ∧ rect.E.2 = 0 ∧
  rect.F.1 = 8 ∧ rect.F.2 = 0 ∧
  rect.G.1 = 8 ∧ rect.G.2 = 4 ∧
  rect.H.1 = 0 ∧ rect.H.2 = 4

-- Define point N on side EF
def N : ℝ × ℝ := (4, 0)

-- Define the property that triangle ENG is isosceles
def is_isosceles_ENG (rect : Rectangle) : Prop :=
  let EN := ((N.1 - rect.E.1)^2 + (N.2 - rect.E.2)^2).sqrt
  let NG := ((rect.G.1 - N.1)^2 + (rect.G.2 - N.2)^2).sqrt
  EN = NG

-- Theorem statement
theorem angle_ENG_is_45_degrees (rect : Rectangle) 
  (h1 : is_valid_rectangle rect) 
  (h2 : is_isosceles_ENG rect) : 
  Real.arctan 1 = 45 * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_angle_ENG_is_45_degrees_l1127_112713


namespace NUMINAMATH_CALUDE_boatsman_speed_calculation_l1127_112785

/-- The speed of the boatsman in still water -/
def boatsman_speed : ℝ := 7

/-- The speed of the river -/
def river_speed : ℝ := 3

/-- The distance between the two destinations -/
def distance : ℝ := 40

/-- The time difference between upstream and downstream travel -/
def time_difference : ℝ := 6

theorem boatsman_speed_calculation :
  (distance / (boatsman_speed - river_speed) - distance / (boatsman_speed + river_speed) = time_difference) ∧
  (boatsman_speed > river_speed) :=
sorry

end NUMINAMATH_CALUDE_boatsman_speed_calculation_l1127_112785


namespace NUMINAMATH_CALUDE_work_hours_difference_l1127_112781

def hours_week1 : ℕ := 35
def hours_week2 : ℕ := 35
def hours_week3 : ℕ := 48
def hours_week4 : ℕ := 48

theorem work_hours_difference : 
  (hours_week3 + hours_week4) - (hours_week1 + hours_week2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_difference_l1127_112781


namespace NUMINAMATH_CALUDE_dogs_wearing_neither_l1127_112709

theorem dogs_wearing_neither (total : ℕ) (tags : ℕ) (collars : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : tags = 45)
  (h3 : collars = 40)
  (h4 : both = 6) :
  total - (tags + collars - both) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dogs_wearing_neither_l1127_112709


namespace NUMINAMATH_CALUDE_obtain_11_from_1_l1127_112702

/-- Represents the allowed operations on the calculator -/
inductive Operation
  | Multiply3
  | Add3
  | Divide3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Multiply3 => n * 3
  | Operation.Add3 => n + 3
  | Operation.Divide3 => if n % 3 = 0 then n / 3 else n

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem: It's possible to obtain 11 from 1 using the given calculator operations -/
theorem obtain_11_from_1 : ∃ (ops : List Operation), applyOperations 1 ops = 11 := by
  sorry

end NUMINAMATH_CALUDE_obtain_11_from_1_l1127_112702


namespace NUMINAMATH_CALUDE_condition_relationship_l1127_112744

theorem condition_relationship (A B : Prop) 
  (h : (¬A → ¬B) ∧ ¬(¬B → ¬A)) : 
  (B → A) ∧ ¬(A → B) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l1127_112744


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1127_112736

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}
def B : Set ℝ := Set.range (λ x => 2 * x + 1)

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1127_112736


namespace NUMINAMATH_CALUDE_remaining_fabric_area_l1127_112761

/-- Calculates the remaining fabric area after cutting curtains -/
theorem remaining_fabric_area (bolt_length bolt_width living_room_length living_room_width bedroom_length bedroom_width : ℝ) 
  (h1 : bolt_length = 16)
  (h2 : bolt_width = 12)
  (h3 : living_room_length = 4)
  (h4 : living_room_width = 6)
  (h5 : bedroom_length = 2)
  (h6 : bedroom_width = 4) :
  bolt_length * bolt_width - (living_room_length * living_room_width + bedroom_length * bedroom_width) = 160 := by
  sorry

#check remaining_fabric_area

end NUMINAMATH_CALUDE_remaining_fabric_area_l1127_112761


namespace NUMINAMATH_CALUDE_marble_remainder_l1127_112780

theorem marble_remainder (n m k : ℤ) : ∃ q : ℤ, (8*n + 5) + (8*m + 3) + (8*k + 7) = 8*q + 7 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l1127_112780


namespace NUMINAMATH_CALUDE_sophies_bakery_purchase_l1127_112700

/-- Sophie's bakery purchase problem -/
theorem sophies_bakery_purchase
  (cupcake_price : ℚ)
  (cupcake_quantity : ℕ)
  (doughnut_price : ℚ)
  (doughnut_quantity : ℕ)
  (cookie_price : ℚ)
  (cookie_quantity : ℕ)
  (pie_slice_price : ℚ)
  (total_spent : ℚ)
  (h1 : cupcake_price = 2)
  (h2 : cupcake_quantity = 5)
  (h3 : doughnut_price = 1)
  (h4 : doughnut_quantity = 6)
  (h5 : cookie_price = 0.6)
  (h6 : cookie_quantity = 15)
  (h7 : pie_slice_price = 2)
  (h8 : total_spent = 33)
  : (total_spent - (cupcake_price * cupcake_quantity + doughnut_price * doughnut_quantity + cookie_price * cookie_quantity)) / pie_slice_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_sophies_bakery_purchase_l1127_112700


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1127_112739

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x > 0, x^2 - x ≤ Real.exp x - a*x - 1) →
  a ≤ Real.exp 1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1127_112739


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l1127_112777

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l1127_112777


namespace NUMINAMATH_CALUDE_polygon_interior_angles_increase_l1127_112718

theorem polygon_interior_angles_increase (n : ℕ) :
  (n + 1 - 2) * 180 - (n - 2) * 180 = 180 → n + 1 - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_increase_l1127_112718


namespace NUMINAMATH_CALUDE_total_books_is_80_l1127_112799

/-- Calculates the total number of books bought given the conditions -/
def total_books (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_bought : ℕ) : ℕ :=
  let history_books_bought := (total_price - math_book_price * math_books_bought) / history_book_price
  math_books_bought + history_books_bought

/-- Proves that the total number of books bought is 80 under the given conditions -/
theorem total_books_is_80 :
  total_books 390 4 5 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_80_l1127_112799


namespace NUMINAMATH_CALUDE_image_of_one_three_l1127_112776

/-- A set of ordered pairs of real numbers -/
def RealPair : Type := ℝ × ℝ

/-- The mapping f: A → B -/
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

/-- Theorem: The image of (1, 3) under f is (-2, 4) -/
theorem image_of_one_three :
  f (1, 3) = (-2, 4) := by
  sorry

end NUMINAMATH_CALUDE_image_of_one_three_l1127_112776


namespace NUMINAMATH_CALUDE_equality_from_quadratic_equation_l1127_112723

theorem equality_from_quadratic_equation 
  (m n p : ℝ) 
  (hm : m ≠ 0) 
  (hn : n ≠ 0) 
  (hp : p ≠ 0) 
  (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 
  2 * p = m + n := by
  sorry

end NUMINAMATH_CALUDE_equality_from_quadratic_equation_l1127_112723


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1127_112752

/-- Given three people A, B, and C with the following conditions:
    - A is two years older than B
    - The total of the ages of A, B, and C is 22
    - B is 8 years old
    Prove that the ratio of B's age to C's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 22 →
  b = 8 →
  b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1127_112752


namespace NUMINAMATH_CALUDE_solution_to_inequality_l1127_112703

theorem solution_to_inequality : 1 - 1 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_solution_to_inequality_l1127_112703


namespace NUMINAMATH_CALUDE_statue_model_ratio_l1127_112717

/-- Given a statue of height 75 feet and a model of height 5 inches,
    prove that one inch of the model represents 15 feet of the statue. -/
theorem statue_model_ratio :
  let statue_height : ℝ := 75  -- statue height in feet
  let model_height : ℝ := 5    -- model height in inches
  statue_height / model_height = 15 := by
sorry


end NUMINAMATH_CALUDE_statue_model_ratio_l1127_112717


namespace NUMINAMATH_CALUDE_no_valid_stacking_l1127_112726

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of crates -/
def numCrates : ℕ := 12

/-- The dimensions of each crate -/
def crateDims : CrateDimensions := ⟨3, 4, 5⟩

/-- The target total height -/
def targetHeight : ℕ := 50

/-- Theorem stating that there are no valid ways to stack the crates to reach the target height -/
theorem no_valid_stacking :
  ¬∃ (a b c : ℕ), a + b + c = numCrates ∧
                  a * crateDims.length + b * crateDims.width + c * crateDims.height = targetHeight :=
by sorry

end NUMINAMATH_CALUDE_no_valid_stacking_l1127_112726


namespace NUMINAMATH_CALUDE_car_trip_mpg_l1127_112754

/-- Represents the miles per gallon for a car trip -/
structure MPG where
  ab : ℝ  -- Miles per gallon from A to B
  bc : ℝ  -- Miles per gallon from B to C
  total : ℝ  -- Overall miles per gallon for the entire trip

/-- Represents the distance for a car trip -/
structure Distance where
  ab : ℝ  -- Distance from A to B
  bc : ℝ  -- Distance from B to C

theorem car_trip_mpg (d : Distance) (mpg : MPG) :
  d.bc = d.ab / 2 →  -- Distance from B to C is half of A to B
  mpg.ab = 40 →  -- MPG from A to B is 40
  mpg.total = 300 / 7 →  -- Overall MPG is 300/7 (approx. 42.857142857142854)
  d.ab > 0 →  -- Distance from A to B is positive
  mpg.bc = 100 / 9 :=  -- MPG from B to C is 100/9 (approx. 11.11)
by sorry

end NUMINAMATH_CALUDE_car_trip_mpg_l1127_112754


namespace NUMINAMATH_CALUDE_largest_circle_area_l1127_112760

/-- The area of the largest circle formed from a string with length equal to the perimeter of a 15x9 rectangle is 576/π. -/
theorem largest_circle_area (length width : ℝ) (h1 : length = 15) (h2 : width = 9) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * π)
  π * radius^2 = 576 / π :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_area_l1127_112760


namespace NUMINAMATH_CALUDE_namjoon_lowest_height_l1127_112773

/-- Heights of planks in centimeters -/
def height_A : ℝ := 2.4
def height_B : ℝ := 3.2
def height_C : ℝ := 2.8

/-- Number of planks each person stands on -/
def num_A : ℕ := 8
def num_B : ℕ := 4
def num_C : ℕ := 5

/-- Total heights for each person -/
def height_Eunji : ℝ := height_A * num_A
def height_Namjoon : ℝ := height_B * num_B
def height_Hoseok : ℝ := height_C * num_C

theorem namjoon_lowest_height :
  height_Namjoon < height_Eunji ∧ height_Namjoon < height_Hoseok :=
by sorry

end NUMINAMATH_CALUDE_namjoon_lowest_height_l1127_112773


namespace NUMINAMATH_CALUDE_karlson_candies_theorem_l1127_112756

/-- Represents the maximum number of candies Karlson can eat -/
def max_candies (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that the maximum number of candies Karlson can eat with 31 initial ones is 465 -/
theorem karlson_candies_theorem :
  max_candies 31 = 465 := by
  sorry

#eval max_candies 31

end NUMINAMATH_CALUDE_karlson_candies_theorem_l1127_112756


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1127_112733

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_roots_range (a b c : ℝ) (ha : a ≠ 0) :
  f a b c (-1) = -2 →
  f a b c (-1/2) = -1/4 →
  f a b c 0 = 1 →
  f a b c (1/2) = 7/4 →
  f a b c 1 = 2 →
  f a b c (3/2) = 7/4 →
  f a b c 2 = 1 →
  f a b c (5/2) = -1/4 →
  f a b c 3 = -2 →
  ∃ x₁ x₂ : ℝ, f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ 
    -1/2 < x₁ ∧ x₁ < 0 ∧ 2 < x₂ ∧ x₂ < 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1127_112733


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1127_112762

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 77 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1127_112762


namespace NUMINAMATH_CALUDE_candy_bar_problem_l1127_112786

theorem candy_bar_problem (F : ℕ) : 
  (∃ (J : ℕ), 
    J = 10 * (2 * F + 6) ∧ 
    (2 * F + 6) = F + (F + 6) ∧
    (40 * J) / 100 = 120) → 
  F = 12 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_problem_l1127_112786


namespace NUMINAMATH_CALUDE_min_value_of_function_l1127_112758

theorem min_value_of_function (x : ℝ) :
  -1 ≤ x ∧ x ≤ 1 →
  ∀ y : ℝ, y = x^4 + 2*x^2 - 1 →
  y ≥ -1 ∧ ∃ x₀ : ℝ, -1 ≤ x₀ ∧ x₀ ≤ 1 ∧ x₀^4 + 2*x₀^2 - 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1127_112758


namespace NUMINAMATH_CALUDE_point_movement_l1127_112763

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line by a given distance -/
def movePoint (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement :
  let A : Point := ⟨-4⟩
  let B : Point := movePoint A 6
  B.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l1127_112763


namespace NUMINAMATH_CALUDE_cupcake_packages_l1127_112796

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) :
  initial_cupcakes = 39 →
  eaten_cupcakes = 21 →
  cupcakes_per_package = 3 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 6 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_packages_l1127_112796


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_fraction_simplification_l1127_112779

-- Problem 1
theorem sqrt_expression_simplification :
  3 * Real.sqrt 2 - (Real.sqrt 3 + 2 * Real.sqrt 2) * Real.sqrt 6 = -4 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem fraction_simplification (a : ℝ) (h1 : a^2 ≠ 4) (h2 : a ≠ 2) :
  a / (a^2 - 4) + 1 / (4 - 2*a) = 1 / (2*a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_fraction_simplification_l1127_112779


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_3_l1127_112725

theorem tan_alpha_2_implies_expression_3 (α : Real) (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 - (Real.cos α) * (Real.sin α) + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_3_l1127_112725


namespace NUMINAMATH_CALUDE_half_sum_sequence_common_ratio_l1127_112769

/-- A geometric sequence where each term is half the sum of its next two terms -/
def HalfSumSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a n = (a (n + 1) + a (n + 2)) / 2

/-- The common ratio of a geometric sequence -/
def CommonRatio (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem half_sum_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) :
  HalfSumSequence a → CommonRatio a r → r = 1 := by sorry

end NUMINAMATH_CALUDE_half_sum_sequence_common_ratio_l1127_112769


namespace NUMINAMATH_CALUDE_factorization_cubic_l1127_112747

theorem factorization_cubic (a : ℝ) : a^3 + 4*a^2 + 4*a = a*(a + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_l1127_112747


namespace NUMINAMATH_CALUDE_plane_through_points_l1127_112748

/-- The equation of a plane containing three given points. -/
def plane_equation (p₁ p₂ p₃ : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ :=
  sorry

/-- Check if the given coefficients form a valid plane equation. -/
def is_valid_plane_equation (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := coeffs
  A > 0 ∧ Nat.gcd (Nat.gcd (A.natAbs) (B.natAbs)) (Nat.gcd (C.natAbs) (D.natAbs)) = 1

/-- The main theorem stating the equation of the plane. -/
theorem plane_through_points :
  let p₁ : ℝ × ℝ × ℝ := (1, 0, 2)
  let p₂ : ℝ × ℝ × ℝ := (5, 0, 4)
  let p₃ : ℝ × ℝ × ℝ := (7, -2, 3)
  let coeffs := plane_equation p₁ p₂ p₃
  coeffs = (1, 1, -1, 1) ∧ is_valid_plane_equation coeffs :=
by sorry

end NUMINAMATH_CALUDE_plane_through_points_l1127_112748


namespace NUMINAMATH_CALUDE_stones_sent_away_l1127_112795

theorem stones_sent_away (original_stones : ℕ) (stones_left : ℕ) (stones_sent : ℕ) : 
  original_stones = 78 → stones_left = 15 → stones_sent = original_stones - stones_left → stones_sent = 63 := by
  sorry

end NUMINAMATH_CALUDE_stones_sent_away_l1127_112795


namespace NUMINAMATH_CALUDE_daily_earnings_a_and_c_l1127_112730

/-- Given three workers a, b, and c, with their daily earnings, prove that a and c together earn $400 per day. -/
theorem daily_earnings_a_and_c (a b c : ℕ) : 
  a + b + c = 600 →  -- Total earnings of a, b, and c
  b + c = 300 →      -- Combined earnings of b and c
  c = 100 →          -- Earnings of c
  a + c = 400 :=     -- Combined earnings of a and c
by
  sorry

end NUMINAMATH_CALUDE_daily_earnings_a_and_c_l1127_112730


namespace NUMINAMATH_CALUDE_three_numbers_sum_to_50_l1127_112746

def number_list : List Nat := [21, 19, 30, 25, 3, 12, 9, 15, 6, 27]

theorem three_numbers_sum_to_50 :
  ∃ (a b c : Nat), a ∈ number_list ∧ b ∈ number_list ∧ c ∈ number_list ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 50 :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_to_50_l1127_112746
