import Mathlib

namespace NUMINAMATH_CALUDE_flower_shop_expenses_flower_shop_weekly_expenses_l2997_299716

/-- Weekly expenses for running a flower shop -/
theorem flower_shop_expenses (rent : ℝ) (utility_rate : ℝ) (hours_per_day : ℕ) 
  (days_per_week : ℕ) (employees_per_shift : ℕ) (hourly_wage : ℝ) : ℝ :=
  let utilities := rent * utility_rate
  let employee_hours := hours_per_day * days_per_week * employees_per_shift
  let employee_wages := employee_hours * hourly_wage
  rent + utilities + employee_wages

/-- Proof of the flower shop's weekly expenses -/
theorem flower_shop_weekly_expenses :
  flower_shop_expenses 1200 0.2 16 5 2 12.5 = 3440 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_expenses_flower_shop_weekly_expenses_l2997_299716


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2997_299702

theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℝ)  -- arithmetic sequence
  (d : ℝ)      -- common difference
  (h1 : d ≠ 0) -- d is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d)  -- definition of arithmetic sequence
  (h3 : (a 9) ^ 2 = a 5 * a 15)    -- a_5, a_9, a_15 form geometric sequence
  : (a 9) / (a 5) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2997_299702


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l2997_299783

/-- The area of an equilateral triangle with perimeter 3p is (√3/4) * p^2 -/
theorem equilateral_triangle_area (p : ℝ) (p_pos : p > 0) :
  let perimeter := 3 * p
  ∃ (area : ℝ), area = (Real.sqrt 3 / 4) * p^2 ∧
  ∀ (side : ℝ), side > 0 → 3 * side = perimeter →
  area = (Real.sqrt 3 / 4) * side^2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l2997_299783


namespace NUMINAMATH_CALUDE_vector_collinear_same_direction_l2997_299771

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- Two vectors have the same direction if their corresponding components have the same sign -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 ≥ 0) ∧ (a.2 * b.2 ≥ 0)

/-- The theorem statement -/
theorem vector_collinear_same_direction (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (-x, 2)
  collinear a b ∧ same_direction a b → x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_collinear_same_direction_l2997_299771


namespace NUMINAMATH_CALUDE_sum_x_y_equals_2700_l2997_299742

theorem sum_x_y_equals_2700 (x y : ℝ) : 
  (0.9 * 600 = 0.5 * x) → 
  (0.6 * x = 0.4 * y) → 
  x + y = 2700 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_2700_l2997_299742


namespace NUMINAMATH_CALUDE_password_guess_probabilities_l2997_299767

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts -/
def guess_probability (total_digits : ℕ) (max_attempts : ℕ) : ℚ :=
  1 / total_digits + (1 - 1 / total_digits) * (1 / (total_digits - 1))

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts, given that the last digit is even -/
def guess_probability_even (total_even_digits : ℕ) (max_attempts : ℕ) : ℚ :=
  1 / total_even_digits + (1 - 1 / total_even_digits) * (1 / (total_even_digits - 1))

theorem password_guess_probabilities :
  (guess_probability 10 2 = 1/5) ∧ (guess_probability_even 5 2 = 2/5) :=
sorry

end NUMINAMATH_CALUDE_password_guess_probabilities_l2997_299767


namespace NUMINAMATH_CALUDE_spinner_sectors_area_l2997_299725

/-- Represents a circular spinner with win and lose sectors. -/
structure Spinner :=
  (radius : ℝ)
  (win_prob : ℝ)
  (lose_prob : ℝ)

/-- Calculates the area of a circular sector given the total area and probability. -/
def sector_area (total_area : ℝ) (probability : ℝ) : ℝ :=
  total_area * probability

theorem spinner_sectors_area (s : Spinner) 
  (h1 : s.radius = 12)
  (h2 : s.win_prob = 1/3)
  (h3 : s.lose_prob = 1/2) :
  let total_area := Real.pi * s.radius^2
  sector_area total_area s.win_prob = 48 * Real.pi ∧
  sector_area total_area s.lose_prob = 72 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_spinner_sectors_area_l2997_299725


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2997_299754

theorem fraction_decomposition (x A B : ℚ) : 
  (7 * x - 18) / (3 * x^2 - 5 * x - 2) = A / (x + 1) + B / (3 * x - 2) →
  A = -4/7 ∧ B = 61/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2997_299754


namespace NUMINAMATH_CALUDE_tournament_balls_used_l2997_299714

/-- A tennis tournament with specified conditions -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  balls_per_can : Nat

/-- Calculate the total number of tennis balls used in the tournament -/
def total_balls_used (t : TennisTournament) : Nat :=
  (t.games_per_round.sum * t.cans_per_game * t.balls_per_can)

/-- Theorem stating the total number of tennis balls used in the specific tournament -/
theorem tournament_balls_used :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    balls_per_can := 3
  }
  total_balls_used t = 225 := by sorry

end NUMINAMATH_CALUDE_tournament_balls_used_l2997_299714


namespace NUMINAMATH_CALUDE_school_principal_election_l2997_299723

/-- Given that Emma received 45 votes in a school principal election,
    and these votes represent 3/7 of the total votes,
    prove that the total number of votes cast is 105. -/
theorem school_principal_election (emma_votes : ℕ) (total_votes : ℕ)
    (h1 : emma_votes = 45)
    (h2 : emma_votes = 3 * total_votes / 7) :
    total_votes = 105 := by
  sorry

end NUMINAMATH_CALUDE_school_principal_election_l2997_299723


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2997_299757

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + 3 * I = (b + I) * I → a = -1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2997_299757


namespace NUMINAMATH_CALUDE_student_card_distribution_l2997_299718

/-- Given n students (n ≥ 3) and m = (n * (n-1)) / 2 cards, prove that if m is odd
    and there exists a distribution of m distinct integers from 1 to m among n students
    such that the pairwise sums of these integers give different remainders modulo m,
    then n - 2 is a perfect square. -/
theorem student_card_distribution (n : ℕ) (h1 : n ≥ 3) :
  let m : ℕ := n * (n - 1) / 2
  ∃ (distribution : Fin n → Fin m),
    Function.Injective distribution ∧
    (∀ i j : Fin n, i ≠ j →
      ∀ k l : Fin n, k ≠ l →
        (distribution i + distribution j : ℕ) % m ≠
        (distribution k + distribution l : ℕ) % m) →
    Odd m →
    ∃ k : ℕ, n - 2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_student_card_distribution_l2997_299718


namespace NUMINAMATH_CALUDE_min_obtuse_triangle_l2997_299748

-- Define the initial angles of the triangle
def α₀ : Real := 60.001
def β₀ : Real := 60
def γ₀ : Real := 59.999

-- Define a function to calculate the nth angle
def angle (n : Nat) (initial : Real) : Real :=
  (-2)^n * (initial - 60) + 60

-- Define a predicate for an obtuse triangle
def is_obtuse (α β γ : Real) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- State the theorem
theorem min_obtuse_triangle :
  ∃ (n : Nat), (∀ k < n, ¬is_obtuse (angle k α₀) (angle k β₀) (angle k γ₀)) ∧
               is_obtuse (angle n α₀) (angle n β₀) (angle n γ₀) ∧
               n = 15 := by
  sorry


end NUMINAMATH_CALUDE_min_obtuse_triangle_l2997_299748


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l2997_299775

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l2997_299775


namespace NUMINAMATH_CALUDE_set_union_equality_l2997_299761

theorem set_union_equality (a : ℝ) : 
  let A : Set ℝ := {1, a}
  let B : Set ℝ := {a^2}
  A ∪ B = A → a = -1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_union_equality_l2997_299761


namespace NUMINAMATH_CALUDE_quadratic_radical_equivalence_l2997_299706

-- Define what it means for two quadratic radicals to be of the same type
def same_type_quadratic_radical (a b : ℝ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a = k * x^2 ∧ b = k * y^2)

-- State the theorem
theorem quadratic_radical_equivalence :
  same_type_quadratic_radical (m + 1) 8 → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equivalence_l2997_299706


namespace NUMINAMATH_CALUDE_parallelogram_area_l2997_299790

/-- The area of a parallelogram with base length 3 and height 3 is 9 square units. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 3 → 
  height = 3 → 
  area = base * height → 
  area = 9 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2997_299790


namespace NUMINAMATH_CALUDE_x_power_3a_minus_2b_l2997_299730

theorem x_power_3a_minus_2b (x a b : ℝ) (h1 : x^a = 3) (h2 : x^b = 4) :
  x^(3*a - 2*b) = 27/16 := by
sorry

end NUMINAMATH_CALUDE_x_power_3a_minus_2b_l2997_299730


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l2997_299765

theorem inequality_and_minimum_value 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hxy : x + y = 1) : 
  (a + b ≥ 2 * Real.sqrt (a * b)) ∧ 
  (∃ (min : ℝ), min = 9 ∧ ∀ (z w : ℝ), 0 < z → 0 < w → z + w = 1 → 1/z + 4/w ≥ min) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l2997_299765


namespace NUMINAMATH_CALUDE_max_y_value_l2997_299784

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y = (x - y) / (x + 3 * y)) : 
  y ≤ 1/3 ∧ ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ * (1/3) = (x₀ - 1/3) / (x₀ + 1) := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l2997_299784


namespace NUMINAMATH_CALUDE_inequality_proof_l2997_299738

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = a*b) :
  (a + 2*b ≥ 8) ∧ (2*a + b ≥ 9) ∧ (a^2 + 4*b^2 + 5*a*b ≥ 72) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2997_299738


namespace NUMINAMATH_CALUDE_domino_reconstruction_theorem_l2997_299721

/-- Represents a 2x1 domino with color information -/
inductive Domino
| WhiteWhite
| BlueBlue
| WhiteBlue
| BlueWhite

/-- Represents an 8x8 grid -/
def Grid := List (List Bool)

/-- Counts the number of blue cells in a grid -/
def countBlue (g : Grid) : Nat := sorry

/-- Divides a grid into 2x1 dominoes -/
def divideToDominoes (g : Grid) : List Domino := sorry

/-- Reconstructs an 8x8 grid from a list of dominoes -/
def reconstructGrid (dominoes : List Domino) : Grid := sorry

/-- Checks if two grids have the same blue pattern -/
def samePattern (g1 g2 : Grid) : Bool := sorry

theorem domino_reconstruction_theorem (g1 g2 : Grid) 
  (h : countBlue g1 = countBlue g2) :
  ∃ (d1 d2 : List Domino), 
    d1 = divideToDominoes g1 ∧ 
    d2 = divideToDominoes g2 ∧ 
    samePattern (reconstructGrid (d1 ++ d2)) g1 ∧
    samePattern (reconstructGrid (d1 ++ d2)) g2 := by
  sorry

end NUMINAMATH_CALUDE_domino_reconstruction_theorem_l2997_299721


namespace NUMINAMATH_CALUDE_larger_number_problem_l2997_299788

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2997_299788


namespace NUMINAMATH_CALUDE_club_members_proof_l2997_299722

theorem club_members_proof (total : ℕ) (left_handed : ℕ) (jazz_lovers : ℕ) (right_handed_jazz_dislikers : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : jazz_lovers = 18)
  (h4 : right_handed_jazz_dislikers = 2)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : ℕ, x = 5 ∧ 
    x + (left_handed - x) + (jazz_lovers - x) + right_handed_jazz_dislikers = total :=
by sorry

end NUMINAMATH_CALUDE_club_members_proof_l2997_299722


namespace NUMINAMATH_CALUDE_smallest_N_and_digit_sum_l2997_299745

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_N_and_digit_sum :
  ∃ N : ℕ, 
    (∀ k : ℕ, k < N → k * (k + 1) ≤ 10^6) ∧
    N * (N + 1) > 10^6 ∧
    N = 1000 ∧
    sum_of_digits N = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_and_digit_sum_l2997_299745


namespace NUMINAMATH_CALUDE_merchant_discount_percentage_l2997_299724

/-- Proves that if a merchant marks up goods by 60% and makes a 20% profit after offering a discount, then the discount percentage is 25%. -/
theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : profit_percentage = 20) : 
  discount_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_discount_percentage_l2997_299724


namespace NUMINAMATH_CALUDE_one_in_M_l2997_299781

def M : Set ℕ := {0, 1, 2}

theorem one_in_M : 1 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_one_in_M_l2997_299781


namespace NUMINAMATH_CALUDE_marbles_given_to_mary_l2997_299780

def initial_marbles : ℕ := 64
def remaining_marbles : ℕ := 50

theorem marbles_given_to_mary :
  initial_marbles - remaining_marbles = 14 :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_to_mary_l2997_299780


namespace NUMINAMATH_CALUDE_cornelias_asian_countries_l2997_299762

theorem cornelias_asian_countries 
  (total_countries : ℕ) 
  (european_countries : ℕ) 
  (south_american_countries : ℕ) 
  (h1 : total_countries = 42)
  (h2 : european_countries = 20)
  (h3 : south_american_countries = 10)
  (h4 : 2 * (total_countries - european_countries - south_american_countries) / 2 = 
       total_countries - european_countries - south_american_countries) :
  (total_countries - european_countries - south_american_countries) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cornelias_asian_countries_l2997_299762


namespace NUMINAMATH_CALUDE_no_1989_digit_number_sum_equals_product_l2997_299766

theorem no_1989_digit_number_sum_equals_product : ¬ ∃ (n : ℕ), 
  (n ≥ 10^1988 ∧ n < 10^1989) ∧  -- n has 1989 digits
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < 1989 ∧ d₂ < 1989 ∧ d₃ < 1989 ∧ 
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    (n / 10^d₁ % 10 = 5) ∧ (n / 10^d₂ % 10 = 5) ∧ (n / 10^d₃ % 10 = 5)) ∧  -- at least three digits are 5
  (List.sum (List.map (λ i => n / 10^i % 10) (List.range 1989)) = 
   List.prod (List.map (λ i => n / 10^i % 10) (List.range 1989))) :=  -- sum of digits equals product of digits
by sorry

end NUMINAMATH_CALUDE_no_1989_digit_number_sum_equals_product_l2997_299766


namespace NUMINAMATH_CALUDE_ott_fraction_of_total_l2997_299798

/-- Represents the amount of money each person has -/
structure Money where
  loki : ℚ
  moe : ℚ
  nick : ℚ
  ott : ℚ

/-- The initial state of money distribution -/
def initial_money : Money := {
  loki := 5,
  moe := 5,
  nick := 3,
  ott := 0
}

/-- The amount of money given to Ott -/
def money_given : ℚ := 1

/-- The state of money after giving to Ott -/
def final_money : Money := {
  loki := initial_money.loki - money_given,
  moe := initial_money.moe - money_given,
  nick := initial_money.nick - money_given,
  ott := initial_money.ott + 3 * money_given
}

/-- The theorem to prove -/
theorem ott_fraction_of_total (m : Money := final_money) :
  m.ott / (m.loki + m.moe + m.nick + m.ott) = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ott_fraction_of_total_l2997_299798


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2997_299728

theorem fraction_to_decimal :
  (3 : ℚ) / 40 = 0.075 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2997_299728


namespace NUMINAMATH_CALUDE_burger_cost_is_87_l2997_299703

/-- The cost of Uri's purchase in cents -/
def uri_cost : ℕ := 385

/-- The cost of Gen's purchase in cents -/
def gen_cost : ℕ := 360

/-- The number of burgers Uri bought -/
def uri_burgers : ℕ := 3

/-- The number of sodas Uri bought -/
def uri_sodas : ℕ := 2

/-- The number of burgers Gen bought -/
def gen_burgers : ℕ := 2

/-- The number of sodas Gen bought -/
def gen_sodas : ℕ := 3

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 87

theorem burger_cost_is_87 :
  uri_burgers * burger_cost + uri_sodas * ((uri_cost - uri_burgers * burger_cost) / uri_sodas) = uri_cost ∧
  gen_burgers * burger_cost + gen_sodas * ((uri_cost - uri_burgers * burger_cost) / uri_sodas) = gen_cost :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_is_87_l2997_299703


namespace NUMINAMATH_CALUDE_base7_digit_sum_theorem_l2997_299773

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a * base7ToBase10 b)

/-- Adds two base-7 numbers --/
def addBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a + base7ToBase10 b)

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem base7_digit_sum_theorem : 
  sumDigitsBase7 (addBase7 (multiplyBase7 36 52) 20) = 23 := by sorry

end NUMINAMATH_CALUDE_base7_digit_sum_theorem_l2997_299773


namespace NUMINAMATH_CALUDE_max_value_fraction_l2997_299779

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -2) (hy : 0 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2997_299779


namespace NUMINAMATH_CALUDE_three_card_selection_l2997_299786

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of cards to be picked -/
def cards_to_pick : ℕ := 3

/-- The number of ways to pick three different cards from a standard deck where order matters -/
def ways_to_pick_three_cards : ℕ := standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2)

theorem three_card_selection :
  ways_to_pick_three_cards = 132600 :=
sorry

end NUMINAMATH_CALUDE_three_card_selection_l2997_299786


namespace NUMINAMATH_CALUDE_article_cost_l2997_299792

theorem article_cost (decreased_price : ℝ) (decrease_percentage : ℝ) (h1 : decreased_price = 760) (h2 : decrease_percentage = 24) : 
  ∃ (original_price : ℝ), original_price * (1 - decrease_percentage / 100) = decreased_price ∧ original_price = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_article_cost_l2997_299792


namespace NUMINAMATH_CALUDE_sum_x_y_z_l2997_299719

theorem sum_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : 
  x + y + z = 14 * x := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l2997_299719


namespace NUMINAMATH_CALUDE_project_monthly_allocations_l2997_299778

/-- Proves that the number of monthly allocations is 12 given the project budget conditions -/
theorem project_monthly_allocations
  (total_budget : ℕ)
  (months_passed : ℕ)
  (amount_spent : ℕ)
  (over_budget : ℕ)
  (h1 : total_budget = 12600)
  (h2 : months_passed = 6)
  (h3 : amount_spent = 6580)
  (h4 : over_budget = 280)
  (h5 : ∃ (monthly_allocation : ℕ), total_budget = monthly_allocation * (total_budget / monthly_allocation)) :
  total_budget / ((amount_spent - over_budget) / months_passed) = 12 := by
  sorry

end NUMINAMATH_CALUDE_project_monthly_allocations_l2997_299778


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2997_299707

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2997_299707


namespace NUMINAMATH_CALUDE_salary_percentage_increase_l2997_299708

theorem salary_percentage_increase 
  (initial_salary final_salary : ℝ) 
  (h1 : initial_salary = 50)
  (h2 : final_salary = 90) : 
  (final_salary - initial_salary) / initial_salary * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_increase_l2997_299708


namespace NUMINAMATH_CALUDE_derivative_zero_not_sufficient_nor_necessary_l2997_299763

-- Define a real-valued function
variable (f : ℝ → ℝ)
-- Define a real number x
variable (x : ℝ)

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Define the statement to be proved
theorem derivative_zero_not_sufficient_nor_necessary :
  ¬(∀ f x, (deriv f x = 0 → has_extremum f x) ∧ (has_extremum f x → deriv f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_not_sufficient_nor_necessary_l2997_299763


namespace NUMINAMATH_CALUDE_fold_sum_theorem_l2997_299733

/-- Represents a point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold of a piece of graph paper -/
structure Fold where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The sum of the x and y coordinates of the fourth point in a fold -/
def fourthPointSum (f : Fold) : ℝ :=
  f.p4.x + f.p4.y

/-- Theorem stating that for the given fold, the sum of x and y coordinates of the fourth point is 13 -/
theorem fold_sum_theorem (f : Fold) 
  (h1 : f.p1 = ⟨0, 4⟩) 
  (h2 : f.p2 = ⟨5, 0⟩) 
  (h3 : f.p3 = ⟨9, 6⟩) : 
  fourthPointSum f = 13 := by
  sorry

end NUMINAMATH_CALUDE_fold_sum_theorem_l2997_299733


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2997_299740

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 3 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x ∧ 
   ∀ x' y' : ℝ, y' = 3 * x' + c ∧ y'^2 = 12 * x' → x' = x ∧ y' = y) ↔ 
  c = 3 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2997_299740


namespace NUMINAMATH_CALUDE_minimum_area_of_reported_tile_l2997_299776

/-- Represents the reported dimension of a side of a tile -/
structure ReportedDimension where
  value : ℝ
  lower_bound : ℝ := value - 0.7
  upper_bound : ℝ := value + 0.7

/-- Represents a rectangular tile with reported dimensions -/
structure ReportedTile where
  length : ReportedDimension
  width : ReportedDimension

def minimum_area (tile : ReportedTile) : ℝ :=
  tile.length.lower_bound * tile.width.lower_bound

theorem minimum_area_of_reported_tile (tile : ReportedTile) 
  (h1 : tile.length.value = 3) 
  (h2 : tile.width.value = 4) : 
  minimum_area tile = 7.59 := by
  sorry

#eval minimum_area { length := { value := 3 }, width := { value := 4 } }

end NUMINAMATH_CALUDE_minimum_area_of_reported_tile_l2997_299776


namespace NUMINAMATH_CALUDE_land_area_increase_l2997_299743

theorem land_area_increase :
  let initial_side : ℝ := 6
  let increase : ℝ := 1
  let new_side := initial_side + increase
  let initial_area := initial_side ^ 2
  let new_area := new_side ^ 2
  new_area - initial_area = 13 := by
sorry

end NUMINAMATH_CALUDE_land_area_increase_l2997_299743


namespace NUMINAMATH_CALUDE_square_increase_l2997_299737

theorem square_increase (a : ℕ) : (a + 1)^2 - a^2 = 1001 → a = 500 := by
  sorry

end NUMINAMATH_CALUDE_square_increase_l2997_299737


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2997_299797

-- Define the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Finset ℕ := {1, 2, 3}

-- Define set B
def B : Finset ℕ := {2, 3, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2997_299797


namespace NUMINAMATH_CALUDE_inequality_proof_l2997_299787

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x*y / (x^2 + y^2 + 2*z^2) + y*z / (2*x^2 + y^2 + z^2) + z*x / (x^2 + 2*y^2 + z^2) ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2997_299787


namespace NUMINAMATH_CALUDE_older_brother_age_l2997_299791

theorem older_brother_age (father_age : ℕ) (n : ℕ) (x : ℕ) : 
  father_age = 50 ∧ 
  2 * (x + n) = father_age + n ∧
  x + n ≤ father_age →
  x + n = 25 :=
by sorry

end NUMINAMATH_CALUDE_older_brother_age_l2997_299791


namespace NUMINAMATH_CALUDE_drug_price_reduction_equation_l2997_299739

/-- Represents the price reduction scenario for a drug -/
def PriceReductionScenario (initial_price final_price : ℝ) (x : ℝ) : Prop :=
  initial_price * (1 - x)^2 = final_price

/-- Theorem stating the equation for the given drug price reduction scenario -/
theorem drug_price_reduction_equation :
  PriceReductionScenario 140 35 x ↔ 140 * (1 - x)^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_drug_price_reduction_equation_l2997_299739


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2997_299705

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2997_299705


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2997_299704

def quadratic_inequality (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0

def solution_set (a b c : ℝ) := {x : ℝ | quadratic_inequality a b c x}

theorem quadratic_inequality_solution_sets
  (a b c : ℝ) (h : solution_set a b c = {x : ℝ | -2 < x ∧ x < 1}) :
  {x : ℝ | c * x^2 + a * x + b ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2997_299704


namespace NUMINAMATH_CALUDE_geometric_sequence_sine_values_l2997_299794

theorem geometric_sequence_sine_values (α β γ : Real) :
  (β = 2 * α ∧ γ = 4 * α) →  -- geometric sequence condition
  (0 ≤ α ∧ α ≤ 2 * Real.pi) →  -- α ∈ [0, 2π]
  ((Real.sin β) / (Real.sin α) = (Real.sin γ) / (Real.sin β)) →  -- sine values form geometric sequence
  ((α = 2 * Real.pi / 3 ∧ β = 4 * Real.pi / 3 ∧ γ = 8 * Real.pi / 3) ∨
   (α = 4 * Real.pi / 3 ∧ β = 8 * Real.pi / 3 ∧ γ = 16 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sine_values_l2997_299794


namespace NUMINAMATH_CALUDE_activity_popularity_order_l2997_299710

def soccer_popularity : ℚ := 13/40
def swimming_popularity : ℚ := 9/24
def baseball_popularity : ℚ := 11/30
def hiking_popularity : ℚ := 3/10

def activity_order : List String := ["Swimming", "Baseball", "Soccer", "Hiking"]

theorem activity_popularity_order :
  swimming_popularity > baseball_popularity ∧
  baseball_popularity > soccer_popularity ∧
  soccer_popularity > hiking_popularity :=
by sorry

end NUMINAMATH_CALUDE_activity_popularity_order_l2997_299710


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l2997_299729

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l2997_299729


namespace NUMINAMATH_CALUDE_min_value_of_line_through_point_l2997_299769

/-- Given a line ax + by - 1 = 0 passing through the point (1, 2),
    where a and b are positive real numbers,
    the minimum value of 1/a + 2/b is 9. -/
theorem min_value_of_line_through_point (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 1 → (1/a + 2/b) ≥ 9 := by sorry

end NUMINAMATH_CALUDE_min_value_of_line_through_point_l2997_299769


namespace NUMINAMATH_CALUDE_largest_divisor_power_l2997_299772

theorem largest_divisor_power (k : ℕ+) : 
  (∀ m : ℕ+, m ≤ k → (1991 : ℤ)^(m : ℕ) ∣ 1990^19911992 + 1992^19911990) ∧ 
  ¬((1991 : ℤ)^((k + 1) : ℕ) ∣ 1990^19911992 + 1992^19911990) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_power_l2997_299772


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l2997_299700

theorem complex_addition_simplification :
  (-5 + 3*Complex.I) + (2 - 7*Complex.I) = -3 - 4*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l2997_299700


namespace NUMINAMATH_CALUDE_product_from_sum_and_difference_l2997_299736

theorem product_from_sum_and_difference :
  ∀ x y : ℝ, x + y = 72 ∧ x - y = 20 → x * y = 1196 := by
sorry

end NUMINAMATH_CALUDE_product_from_sum_and_difference_l2997_299736


namespace NUMINAMATH_CALUDE_probability_of_sum_25_l2997_299799

/-- Represents a die with numbered and blank faces -/
structure Die where
  faces : ℕ
  numbered_faces : ℕ
  min_number : ℕ
  max_number : ℕ

/-- The first die with 18 numbered faces (1-18) and 2 blank faces -/
def die1 : Die :=
  { faces := 20
  , numbered_faces := 18
  , min_number := 1
  , max_number := 18 }

/-- The second die with 19 numbered faces (2-20) and 1 blank face -/
def die2 : Die :=
  { faces := 20
  , numbered_faces := 19
  , min_number := 2
  , max_number := 20 }

/-- Calculates the number of ways to roll a specific sum with two dice -/
def waysToRollSum (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of possible outcomes when rolling two dice -/
def totalOutcomes (d1 d2 : Die) : ℕ :=
  d1.faces * d2.faces

/-- The main theorem stating the probability of rolling a sum of 25 -/
theorem probability_of_sum_25 :
  (waysToRollSum die1 die2 25 : ℚ) / (totalOutcomes die1 die2 : ℚ) = 7 / 200 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_25_l2997_299799


namespace NUMINAMATH_CALUDE_binomial_probability_l2997_299796

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_probability (ξ : BinomialVariable) 
  (h_exp : expectedValue ξ = 7)
  (h_var : variance ξ = 6) : 
  ξ.p = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l2997_299796


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_vertex_l2997_299770

/-- Given a hyperbola with equation x²/16 - y²/9 = 1, 
    prove that the standard equation of a parabola 
    with its focus at the right vertex of this hyperbola is y² = 16x -/
theorem parabola_equation_from_hyperbola_vertex (x y : ℝ) : 
  (x^2 / 16 - y^2 / 9 = 1) → 
  ∃ (x₀ y₀ : ℝ), 
    (x₀ > 0 ∧ y₀ = 0 ∧ x₀^2 / 16 - y₀^2 / 9 = 1) ∧ 
    (∀ (x' y' : ℝ), (y' - y₀)^2 = 16 * (x' - x₀)) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_vertex_l2997_299770


namespace NUMINAMATH_CALUDE_line_through_point_l2997_299711

/-- Given a line equation bx + (b-1)y = b+3 that passes through the point (3, -7), prove that b = 4/5 -/
theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 1) * (-7) = b + 3) → b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2997_299711


namespace NUMINAMATH_CALUDE_power_two_half_equals_two_l2997_299712

theorem power_two_half_equals_two : 2^(2/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_half_equals_two_l2997_299712


namespace NUMINAMATH_CALUDE_quadratic_radical_condition_l2997_299713

theorem quadratic_radical_condition (x : ℝ) : Real.sqrt ((x - 3)^2) = x - 3 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_condition_l2997_299713


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2997_299734

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 7)) = 3 → y = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2997_299734


namespace NUMINAMATH_CALUDE_doughnuts_per_staff_member_l2997_299752

theorem doughnuts_per_staff_member 
  (total_doughnuts : ℕ) 
  (staff_members : ℕ) 
  (doughnuts_left : ℕ) 
  (h1 : total_doughnuts = 50) 
  (h2 : staff_members = 19) 
  (h3 : doughnuts_left = 12) : 
  (total_doughnuts - doughnuts_left) / staff_members = 2 :=
sorry

end NUMINAMATH_CALUDE_doughnuts_per_staff_member_l2997_299752


namespace NUMINAMATH_CALUDE_division_problem_l2997_299751

theorem division_problem : (102 / 6) / 3 = 5 + 2/3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2997_299751


namespace NUMINAMATH_CALUDE_tom_trout_catch_l2997_299760

/-- Proves that Tom's catch equals 48 trout given the conditions -/
theorem tom_trout_catch (melanie_catch : ℕ) (tom_multiplier : ℕ) 
  (h1 : melanie_catch = 16)
  (h2 : tom_multiplier = 3) : 
  melanie_catch * tom_multiplier = 48 := by
  sorry

end NUMINAMATH_CALUDE_tom_trout_catch_l2997_299760


namespace NUMINAMATH_CALUDE_set_A_equivalence_l2997_299753

theorem set_A_equivalence : 
  {x : ℚ | (x + 1) * (x - 2/3) * (x^2 - 2) = 0} = {-1, 2/3} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equivalence_l2997_299753


namespace NUMINAMATH_CALUDE_dairy_water_mixture_l2997_299764

theorem dairy_water_mixture (pure_dairy : ℝ) (profit_percentage : ℝ) 
  (h1 : pure_dairy > 0)
  (h2 : profit_percentage = 25) : 
  let total_mixture := pure_dairy * (1 + profit_percentage / 100)
  let water_added := total_mixture - pure_dairy
  (water_added / total_mixture) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dairy_water_mixture_l2997_299764


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l2997_299726

/-- A function that checks if a natural number n satisfies the conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers. -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that sticks of lengths 1 to n can form an equilateral triangle
    if and only if n satisfies the specific conditions. -/
theorem equilateral_triangle_condition (n : ℕ) :
  (sum_first_n n % 3 = 0 ∧ ∀ k < n, k > 0) ↔ can_form_equilateral_triangle n := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_condition_l2997_299726


namespace NUMINAMATH_CALUDE_two_squares_same_plus_signs_l2997_299749

/-- Represents a cell in the 8x8 table -/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the 8x8 table with plus signs -/
def Table := Cell → Bool

/-- Represents a 4x4 square within the 8x8 table -/
structure Square :=
  (top_left_row : Fin 5)
  (top_left_col : Fin 5)

/-- Counts the number of plus signs in a given 4x4 square -/
def count_plus_signs (t : Table) (s : Square) : Nat :=
  sorry

theorem two_squares_same_plus_signs (t : Table) :
  ∃ s1 s2 : Square, s1 ≠ s2 ∧ count_plus_signs t s1 = count_plus_signs t s2 :=
sorry

end NUMINAMATH_CALUDE_two_squares_same_plus_signs_l2997_299749


namespace NUMINAMATH_CALUDE_chicken_count_l2997_299747

/-- The number of chickens Colten has -/
def colten_chickens : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar_chickens : ℕ := 3 * colten_chickens - 4

/-- The number of chickens Quentin has -/
def quentin_chickens : ℕ := 2 * skylar_chickens + 25

/-- The total number of chickens -/
def total_chickens : ℕ := quentin_chickens + skylar_chickens + colten_chickens

theorem chicken_count : total_chickens = 383 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l2997_299747


namespace NUMINAMATH_CALUDE_value_of_M_l2997_299795

theorem value_of_M : ∀ M : ℝ, (0.25 * M = 0.55 * 1500) → M = 3300 := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2997_299795


namespace NUMINAMATH_CALUDE_product_of_divisors_implies_n_l2997_299744

/-- The product of all positive divisors of a natural number -/
def divisor_product (n : ℕ) : ℕ := sorry

/-- The number of positive divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := sorry

theorem product_of_divisors_implies_n (N : ℕ) :
  divisor_product N = 2^120 * 3^60 * 5^90 → N = 18000 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_implies_n_l2997_299744


namespace NUMINAMATH_CALUDE_function_value_at_nine_l2997_299782

-- Define the function f(x) = k * x^(1/2)
def f (k : ℝ) (x : ℝ) : ℝ := k * (x ^ (1/2))

-- State the theorem
theorem function_value_at_nine (k : ℝ) : 
  f k 16 = 6 → f k 9 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_nine_l2997_299782


namespace NUMINAMATH_CALUDE_exp_sum_geq_sin_cos_square_l2997_299777

theorem exp_sum_geq_sin_cos_square (x : ℝ) : Real.exp x + Real.exp (-x) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_geq_sin_cos_square_l2997_299777


namespace NUMINAMATH_CALUDE_translation_of_sine_graph_l2997_299715

open Real

theorem translation_of_sine_graph (θ φ : ℝ) : 
  (abs θ < π / 2) →
  (0 < φ) →
  (φ < π) →
  (sin θ = 1 / 2) →
  (sin (θ - 2 * φ) = 1 / 2) →
  (φ = 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_translation_of_sine_graph_l2997_299715


namespace NUMINAMATH_CALUDE_product_sum_in_base_l2997_299793

/-- Represents a number in base b --/
structure BaseNumber (b : ℕ) where
  value : ℕ

/-- Converts a base b number to its decimal representation --/
def to_decimal (b : ℕ) (n : BaseNumber b) : ℕ := sorry

/-- Converts a decimal number to its representation in base b --/
def from_decimal (b : ℕ) (n : ℕ) : BaseNumber b := sorry

/-- Multiplies two numbers in base b --/
def mul_base (b : ℕ) (x y : BaseNumber b) : BaseNumber b := sorry

/-- Adds two numbers in base b --/
def add_base (b : ℕ) (x y : BaseNumber b) : BaseNumber b := sorry

theorem product_sum_in_base (b : ℕ) 
  (h : mul_base b (mul_base b (from_decimal b 14) (from_decimal b 17)) (from_decimal b 18) = from_decimal b 6180) :
  add_base b (add_base b (from_decimal b 14) (from_decimal b 17)) (from_decimal b 18) = from_decimal b 53 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_in_base_l2997_299793


namespace NUMINAMATH_CALUDE_name_tag_paper_perimeter_l2997_299758

theorem name_tag_paper_perimeter :
  ∀ (num_students : ℕ) (tag_side_length : ℝ) (paper_width : ℝ) (unused_width : ℝ),
    num_students = 24 →
    tag_side_length = 4 →
    paper_width = 34 →
    unused_width = 2 →
    (paper_width - unused_width) / tag_side_length * tag_side_length * 
      (num_students / ((paper_width - unused_width) / tag_side_length)) = 
      paper_width - unused_width →
    2 * (paper_width + (num_students / ((paper_width - unused_width) / tag_side_length)) * tag_side_length) = 92 := by
  sorry

end NUMINAMATH_CALUDE_name_tag_paper_perimeter_l2997_299758


namespace NUMINAMATH_CALUDE_gilda_marbles_l2997_299750

theorem gilda_marbles (M : ℝ) (h : M > 0) : 
  let remaining_after_pedro : ℝ := 0.70 * M
  let remaining_after_ebony : ℝ := 0.85 * remaining_after_pedro
  let remaining_after_jimmy : ℝ := 0.80 * remaining_after_ebony
  let final_remaining : ℝ := 0.90 * remaining_after_jimmy
  final_remaining / M = 0.4284 := by
sorry

end NUMINAMATH_CALUDE_gilda_marbles_l2997_299750


namespace NUMINAMATH_CALUDE_wilsons_theorem_l2997_299789

theorem wilsons_theorem (N : ℕ) (h : N > 1) :
  (Nat.factorial (N - 1) % N = N - 1) ↔ Nat.Prime N := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l2997_299789


namespace NUMINAMATH_CALUDE_square_sum_plus_quadruple_product_l2997_299735

theorem square_sum_plus_quadruple_product (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x * y = 15) : 
  x^2 + 6*x*y + y^2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_quadruple_product_l2997_299735


namespace NUMINAMATH_CALUDE_sailboat_canvas_area_l2997_299717

/-- The total area of canvas needed for a model sailboat with three sails -/
theorem sailboat_canvas_area
  (rect_length : ℝ)
  (rect_width : ℝ)
  (tri1_base : ℝ)
  (tri1_height : ℝ)
  (tri2_base : ℝ)
  (tri2_height : ℝ)
  (h_rect_length : rect_length = 5)
  (h_rect_width : rect_width = 8)
  (h_tri1_base : tri1_base = 3)
  (h_tri1_height : tri1_height = 4)
  (h_tri2_base : tri2_base = 4)
  (h_tri2_height : tri2_height = 6) :
  rect_length * rect_width +
  (tri1_base * tri1_height) / 2 +
  (tri2_base * tri2_height) / 2 = 58 := by
sorry


end NUMINAMATH_CALUDE_sailboat_canvas_area_l2997_299717


namespace NUMINAMATH_CALUDE_lamp_probability_l2997_299701

/-- Represents the total number of outlets available -/
def total_outlets : Nat := 7

/-- Represents the number of plugs to be connected -/
def num_plugs : Nat := 3

/-- Represents the number of ways to plug 3 plugs into 7 outlets -/
def total_ways : Nat := total_outlets * (total_outlets - 1) * (total_outlets - 2)

/-- Represents the number of favorable outcomes where the lamp lights up -/
def favorable_outcomes : Nat := 78

/-- Theorem stating that the probability of the lamp lighting up is 13/35 -/
theorem lamp_probability : 
  (favorable_outcomes : ℚ) / total_ways = 13 / 35 := by sorry

end NUMINAMATH_CALUDE_lamp_probability_l2997_299701


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2997_299755

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Calculates the probability of drawing 5 red marbles -/
def probFiveRed (m : MarbleCount) : ℚ :=
  (m.red.choose 5 : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white and 4 red marbles -/
def probOneWhiteFourRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.red.choose 4) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white, 1 blue, and 3 red marbles -/
def probOneWhiteOneBlueTwoRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.red.choose 3) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white, 1 blue, 1 green, and 2 red marbles -/
def probOneWhiteOneBlueOneGreenTwoRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.green.choose 1 * m.red.choose 2) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing one marble of each color -/
def probOneEachColor (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.green.choose 1 * m.yellow.choose 1 * m.red.choose 1) : ℚ) / (totalMarbles m).choose 5

/-- Checks if all probabilities are equal -/
def allProbabilitiesEqual (m : MarbleCount) : Prop :=
  probFiveRed m = probOneWhiteFourRed m ∧
  probFiveRed m = probOneWhiteOneBlueTwoRed m ∧
  probFiveRed m = probOneWhiteOneBlueOneGreenTwoRed m ∧
  probFiveRed m = probOneEachColor m

/-- The main theorem stating that 33 is the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), totalMarbles m = 33 ∧ allProbabilitiesEqual m ∧
  ∀ (m' : MarbleCount), totalMarbles m' < 33 → ¬allProbabilitiesEqual m' :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2997_299755


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2997_299774

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let given_line : Line := { a := 3, b := 1, c := -1 }
  let parallel_line : Line := { a := 3, b := 1, c := -5 }
  let point : (ℝ × ℝ) := (1, 2)
  parallel given_line parallel_line ∧
  point_on_line point.1 point.2 parallel_line :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2997_299774


namespace NUMINAMATH_CALUDE_brie_blouses_l2997_299746

/-- The number of blouses Brie has -/
def num_blouses : ℕ := sorry

/-- The number of skirts Brie has -/
def num_skirts : ℕ := 6

/-- The number of slacks Brie has -/
def num_slacks : ℕ := 8

/-- The percentage of blouses in the hamper -/
def blouse_hamper_percent : ℚ := 75 / 100

/-- The percentage of skirts in the hamper -/
def skirt_hamper_percent : ℚ := 50 / 100

/-- The percentage of slacks in the hamper -/
def slack_hamper_percent : ℚ := 25 / 100

/-- The total number of clothes to be washed -/
def clothes_to_wash : ℕ := 14

theorem brie_blouses : 
  num_blouses = 12 := by sorry

end NUMINAMATH_CALUDE_brie_blouses_l2997_299746


namespace NUMINAMATH_CALUDE_triangle_inequality_l2997_299727

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 
   (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2997_299727


namespace NUMINAMATH_CALUDE_problem_solution_l2997_299720

theorem problem_solution (x y m n a b : ℝ) : 
  x = (Real.sqrt 3 - 1) / 2 →
  y = (Real.sqrt 3 + 1) / 2 →
  m = 1 / x - 1 / y →
  n = y / x + x / y →
  Real.sqrt a - Real.sqrt b = n + 2 →
  Real.sqrt (a * b) = m →
  m = 2 ∧ n = 4 ∧ Real.sqrt a + Real.sqrt b = 2 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2997_299720


namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l2997_299785

theorem mod_eight_equivalence :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4850 [ZMOD 8] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l2997_299785


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2997_299732

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → x₂^2 - 2*x₂ - 5 = 0 → x₁ + x₂ + 3*x₁*x₂ = -13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2997_299732


namespace NUMINAMATH_CALUDE_max_pairs_for_marcella_l2997_299741

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem: With 23 initial pairs and 9 individual shoes lost,
    the maximum number of complete pairs remaining is 14. -/
theorem max_pairs_for_marcella :
  max_pairs_remaining 23 9 = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_pairs_for_marcella_l2997_299741


namespace NUMINAMATH_CALUDE_siblings_ages_l2997_299731

/-- Represents the ages of three siblings --/
structure SiblingsAges where
  david : ℕ
  yuan : ℕ
  maria : ℕ

/-- Conditions for the siblings' ages --/
def validAges (ages : SiblingsAges) : Prop :=
  ages.yuan = ages.david + 7 ∧
  ages.yuan = 2 * ages.david ∧
  ages.maria = ages.david + 4 ∧
  2 * ages.maria = ages.yuan

theorem siblings_ages :
  ∃ (ages : SiblingsAges), validAges ages ∧ ages.david = 7 ∧ ages.maria = 11 := by
  sorry

end NUMINAMATH_CALUDE_siblings_ages_l2997_299731


namespace NUMINAMATH_CALUDE_robs_baseball_cards_l2997_299756

theorem robs_baseball_cards 
  (rob_doubles : ℕ) 
  (rob_total : ℕ) 
  (jess_doubles : ℕ) 
  (h1 : rob_doubles = rob_total / 3)
  (h2 : jess_doubles = 5 * rob_doubles)
  (h3 : jess_doubles = 40) : 
  rob_total = 24 := by
sorry

end NUMINAMATH_CALUDE_robs_baseball_cards_l2997_299756


namespace NUMINAMATH_CALUDE_rational_function_pair_l2997_299709

theorem rational_function_pair (f g : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (g x - g y) = f (g x) - y)
  (h2 : ∀ x y : ℚ, g (f x - f y) = g (f x) - y) :
  ∃ c : ℚ, (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
sorry

end NUMINAMATH_CALUDE_rational_function_pair_l2997_299709


namespace NUMINAMATH_CALUDE_inequality_solution_l2997_299759

theorem inequality_solution (x : ℝ) : 
  2 - 1 / (2 * x + 3) < 4 ↔ x < -7/4 ∨ x > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2997_299759


namespace NUMINAMATH_CALUDE_baker_pastries_sold_l2997_299768

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 78

/-- The difference between pastries and cakes sold -/
def pastry_cake_difference : ℕ := 76

/-- The number of pastries sold by the baker -/
def pastries_sold : ℕ := cakes_sold + pastry_cake_difference

theorem baker_pastries_sold : pastries_sold = 154 := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_sold_l2997_299768
