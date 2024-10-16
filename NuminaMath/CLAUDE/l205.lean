import Mathlib

namespace NUMINAMATH_CALUDE_tournament_rankings_count_l205_20505

/-- Represents a player in the tournament -/
inductive Player : Type
| P : Player
| Q : Player
| R : Player
| S : Player

/-- Represents a match between two players -/
structure Match :=
  (player1 : Player)
  (player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
  (saturday_match1 : Match)
  (saturday_match2 : Match)
  (sunday_championship : Match)
  (sunday_consolation : Match)

/-- Represents a ranking of players -/
def Ranking := List Player

/-- Returns all possible rankings for a given tournament -/
def possibleRankings (t : Tournament) : List Ranking :=
  sorry

theorem tournament_rankings_count :
  ∀ t : Tournament,
  (t.saturday_match1.player1 = Player.P ∧ t.saturday_match1.player2 = Player.Q) →
  (t.saturday_match2.player1 = Player.R ∧ t.saturday_match2.player2 = Player.S) →
  (List.length (possibleRankings t) = 16) :=
by sorry

end NUMINAMATH_CALUDE_tournament_rankings_count_l205_20505


namespace NUMINAMATH_CALUDE_inequality_proof_l205_20598

theorem inequality_proof (x y : ℝ) (hx : |x| ≤ 1) (hy : |y| ≤ 1) : |x + y| ≤ |1 + x * y| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l205_20598


namespace NUMINAMATH_CALUDE_cookie_cost_calculation_l205_20578

def cookies_per_dozen : ℕ := 12

-- Define the problem parameters
def total_dozens : ℕ := 6
def selling_price : ℚ := 3/2
def charity_share : ℚ := 45

-- Theorem to prove
theorem cookie_cost_calculation :
  let total_cookies := total_dozens * cookies_per_dozen
  let total_revenue := total_cookies * selling_price
  let total_profit := 2 * charity_share
  let total_cost := total_revenue - total_profit
  (total_cost / total_cookies : ℚ) = 1/4 := by sorry

end NUMINAMATH_CALUDE_cookie_cost_calculation_l205_20578


namespace NUMINAMATH_CALUDE_batsman_average_l205_20574

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_average * 10 = previous_total →
  (previous_total + 90) / 11 = previous_average + 5 →
  (previous_total + 90) / 11 = 40 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l205_20574


namespace NUMINAMATH_CALUDE_triangle_bisector_product_l205_20512

/-- Given a triangle ABC with sides a, b, and c, internal angle bisectors of lengths fa, fb, and fc,
    and segments of internal angle bisectors on the circumcircle ta, tb, and tc,
    the product of the squares of the sides equals the product of all bisector lengths
    and their segments on the circumcircle. -/
theorem triangle_bisector_product (a b c fa fb fc ta tb tc : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hfa : fa > 0) (hfb : fb > 0) (hfc : fc > 0)
    (hta : ta > 0) (htb : tb > 0) (htc : tc > 0) :
  a^2 * b^2 * c^2 = fa * fb * fc * ta * tb * tc := by
  sorry

end NUMINAMATH_CALUDE_triangle_bisector_product_l205_20512


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_ten_is_three_sixteenths_l205_20599

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- A fair 8-sided die -/
def eight_sided_die : Finset ℕ := Finset.range 8

/-- The product space of rolling both dice -/
def dice_product : Finset (ℕ × ℕ) := six_sided_die.product eight_sided_die

/-- The subset of outcomes where the sum is greater than 10 -/
def sum_greater_than_ten : Finset (ℕ × ℕ) :=
  dice_product.filter (fun p => p.1 + p.2 + 2 > 10)

/-- The probability of the sum being greater than 10 -/
def prob_sum_greater_than_ten : ℚ :=
  (sum_greater_than_ten.card : ℚ) / (dice_product.card : ℚ)

theorem prob_sum_greater_than_ten_is_three_sixteenths :
  prob_sum_greater_than_ten = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_ten_is_three_sixteenths_l205_20599


namespace NUMINAMATH_CALUDE_fish_pond_population_l205_20510

theorem fish_pond_population (initial_tagged : Nat) (second_catch : Nat) (tagged_in_second : Nat) :
  initial_tagged = 60 →
  second_catch = 60 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (1800 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l205_20510


namespace NUMINAMATH_CALUDE_inequality_proof_l205_20552

theorem inequality_proof (n : ℕ+) : (2*n+1)^(n:ℕ) ≥ (2*n)^(n:ℕ) + (2*n-1)^(n:ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l205_20552


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l205_20586

/-- Represents the numbering of a cube's faces -/
structure CubeNumbering where
  faces : Fin 6 → Nat
  all_different : ∀ i j, i ≠ j → faces i ≠ faces j
  range_correct : ∀ i, faces i ∈ Finset.range 6

/-- Represents a pair of opposite faces on the cube -/
inductive OppositePair
  | pair1
  | pair2
  | pair3

theorem cube_sum_theorem (c : CubeNumbering) 
  (h : ∃ p : OppositePair, 
    match p with
    | OppositePair.pair1 => c.faces 0 + c.faces 1 = 11
    | OppositePair.pair2 => c.faces 2 + c.faces 3 = 11
    | OppositePair.pair3 => c.faces 4 + c.faces 5 = 11) :
  ¬ ∃ p : OppositePair, 
    match p with
    | OppositePair.pair1 => c.faces 0 + c.faces 1 = 9
    | OppositePair.pair2 => c.faces 2 + c.faces 3 = 9
    | OppositePair.pair3 => c.faces 4 + c.faces 5 = 9 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l205_20586


namespace NUMINAMATH_CALUDE_taxes_paid_equals_135_l205_20579

/-- Calculate taxes paid given gross pay and net pay -/
def calculate_taxes (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

/-- Theorem: Taxes paid are 135 dollars given the conditions -/
theorem taxes_paid_equals_135 :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_taxes gross_pay net_pay = 135 := by
  sorry

end NUMINAMATH_CALUDE_taxes_paid_equals_135_l205_20579


namespace NUMINAMATH_CALUDE_total_budget_is_40_l205_20549

/-- The total budget for Samuel and Kevin's cinema outing -/
def total_budget : ℕ :=
  let samuel_ticket := 14
  let samuel_snacks := 6
  let kevin_ticket := 14
  let kevin_drinks := 2
  let kevin_food := 4
  samuel_ticket + samuel_snacks + kevin_ticket + kevin_drinks + kevin_food

/-- Theorem stating that the total budget for the outing is $40 -/
theorem total_budget_is_40 : total_budget = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_budget_is_40_l205_20549


namespace NUMINAMATH_CALUDE_unique_integer_sum_l205_20569

theorem unique_integer_sum (y : ℝ) : 
  y = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2) →
  ∃! (d e f : ℕ+), 
    y^100 = 2*y^98 + 18*y^96 + 15*y^94 - y^50 + (d:ℝ)*y^46 + (e:ℝ)*y^44 + (f:ℝ)*y^40 ∧
    d + e + f = 242 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sum_l205_20569


namespace NUMINAMATH_CALUDE_tan_fraction_equals_two_l205_20580

theorem tan_fraction_equals_two (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_fraction_equals_two_l205_20580


namespace NUMINAMATH_CALUDE_max_r_value_l205_20583

theorem max_r_value (p q r : ℝ) (sum_eq : p + q + r = 6) (prod_sum_eq : p * q + p * r + q * r = 8) :
  r ≤ 2 + Real.sqrt (20 / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_r_value_l205_20583


namespace NUMINAMATH_CALUDE_prob_same_color_top_three_l205_20513

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of cards of each color (red or black) in a standard deck -/
def cardsPerColor : ℕ := standardDeckSize / 2

/-- The probability of drawing three cards of the same color from the top of a randomly arranged standard deck -/
def probSameColorTopThree : ℚ :=
  (2 * cardsPerColor * (cardsPerColor - 1) * (cardsPerColor - 2)) /
  (standardDeckSize * (standardDeckSize - 1) * (standardDeckSize - 2))

theorem prob_same_color_top_three :
  probSameColorTopThree = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_top_three_l205_20513


namespace NUMINAMATH_CALUDE_distribute_5_3_l205_20543

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinguishable objects into 3 distinguishable containers is 3^5 -/
theorem distribute_5_3 : distribute 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l205_20543


namespace NUMINAMATH_CALUDE_tuesday_texts_l205_20590

/-- The number of texts sent by Sydney to each person on Monday -/
def monday_texts_per_person : ℕ := 5

/-- The total number of texts sent by Sydney over both days -/
def total_texts : ℕ := 40

/-- The number of people Sydney sent texts to -/
def num_people : ℕ := 2

/-- Theorem: The number of texts sent on Tuesday is 30 -/
theorem tuesday_texts :
  total_texts - (monday_texts_per_person * num_people) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_texts_l205_20590


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l205_20537

def correct_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean - incorrect_value + correct_value

theorem corrected_mean_calculation (n : ℕ) (original_mean incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 45 →
  (correct_mean n original_mean incorrect_value correct_value) / n = 36.44 :=
by
  sorry

#eval (correct_mean 50 36 23 45) / 50

end NUMINAMATH_CALUDE_corrected_mean_calculation_l205_20537


namespace NUMINAMATH_CALUDE_min_value_of_y_l205_20544

theorem min_value_of_y (x : ℝ) (h : x > 3) : x + 1 / (x - 3) ≥ 5 ∧ ∃ x₀ > 3, x₀ + 1 / (x₀ - 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l205_20544


namespace NUMINAMATH_CALUDE_johnny_red_pencils_l205_20587

theorem johnny_red_pencils (total_packs : ℕ) (regular_packs : ℕ) (extra_pencil_packs : ℕ) 
  (h1 : total_packs = 15)
  (h2 : extra_pencil_packs = 3)
  (h3 : regular_packs = total_packs - extra_pencil_packs) :
  total_packs + 2 * extra_pencil_packs = 21 := by
  sorry

end NUMINAMATH_CALUDE_johnny_red_pencils_l205_20587


namespace NUMINAMATH_CALUDE_potato_distribution_ratio_l205_20503

/-- Represents the number of people who were served potatoes -/
def num_people : ℕ := 3

/-- Represents the number of potatoes each person received -/
def potatoes_per_person : ℕ := 8

/-- Represents the ratio of potatoes served to each person -/
def potato_ratio : List ℕ := [1, 1, 1]

/-- Theorem stating that the ratio of potatoes served to each person is 1:1:1 -/
theorem potato_distribution_ratio :
  (List.length potato_ratio = num_people) ∧
  (∀ n ∈ potato_ratio, n = 1) ∧
  (List.sum potato_ratio * potatoes_per_person = num_people * potatoes_per_person) := by
  sorry

end NUMINAMATH_CALUDE_potato_distribution_ratio_l205_20503


namespace NUMINAMATH_CALUDE_remainder_problem_l205_20535

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (5 * n + 9) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l205_20535


namespace NUMINAMATH_CALUDE_valid_money_distribution_exists_l205_20515

/-- Represents the distribution of money among 5 people --/
structure MoneyDistribution where
  total : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Checks if a MoneyDistribution satisfies the given conditions --/
def satisfiesConditions (dist : MoneyDistribution) : Prop :=
  dist.total = 5000 ∧
  dist.a / dist.b = 3 / 2 ∧
  dist.b / dist.c = 4 / 5 ∧
  dist.d = 0.6 * dist.c ∧
  dist.e = 0.6 * dist.c ∧
  dist.a + dist.b + dist.c + dist.d + dist.e = dist.total

/-- Theorem stating the existence of a valid money distribution --/
theorem valid_money_distribution_exists : ∃ (dist : MoneyDistribution), satisfiesConditions dist := by
  sorry

#check valid_money_distribution_exists

end NUMINAMATH_CALUDE_valid_money_distribution_exists_l205_20515


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l205_20538

/-- Stirling number of the second kind -/
def stirling2 (n k : ℕ) : ℕ := sorry

/-- Number of ways to assign n volunteers to k tasks, where each task must have at least one person -/
def assignVolunteers (n k : ℕ) : ℕ := (stirling2 n k) * (Nat.factorial k)

theorem volunteer_assignment_count :
  assignVolunteers 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l205_20538


namespace NUMINAMATH_CALUDE_interest_years_satisfies_equation_l205_20546

/-- The number of years that satisfies the compound and simple interest difference equation -/
def interest_years : ℕ := 2

/-- The principal amount in rupees -/
def principal : ℚ := 3600

/-- The annual interest rate as a decimal -/
def rate : ℚ := 1/10

/-- The difference between compound and simple interest in rupees -/
def interest_difference : ℚ := 36

/-- The equation that relates the number of years to the interest difference -/
def interest_equation (n : ℕ) : Prop :=
  (1 + rate) ^ n - 1 - rate * n = interest_difference / principal

theorem interest_years_satisfies_equation : 
  interest_equation interest_years :=
sorry

end NUMINAMATH_CALUDE_interest_years_satisfies_equation_l205_20546


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l205_20563

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l205_20563


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l205_20539

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 10) :
  (1 / x + 2 / y) ≥ (3 + 2 * Real.sqrt 2) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l205_20539


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l205_20526

-- Define the total number of votes
def total_votes : ℕ := 7600

-- Define the difference in votes between the winner and loser
def vote_difference : ℕ := 2280

-- Define the percentage of votes received by the losing candidate
def losing_candidate_percentage : ℚ := 35

-- Theorem statement
theorem candidate_vote_percentage :
  (2 * losing_candidate_percentage * total_votes : ℚ) = 
  (100 * (total_votes - vote_difference) : ℚ) := by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l205_20526


namespace NUMINAMATH_CALUDE_solution_set_inequality_l205_20582

theorem solution_set_inequality (x : ℝ) :
  (x ≠ 2) → (1 / (x - 2) > -2 ↔ x ∈ Set.Iio (3/2) ∪ Set.Ioi 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l205_20582


namespace NUMINAMATH_CALUDE_overlap_area_l205_20575

-- Define the points on a 2D grid
def Point := ℕ × ℕ

-- Define the rectangle
def rectangle : List Point := [(0, 0), (3, 0), (3, 2), (0, 2)]

-- Define the triangle
def triangle : List Point := [(2, 0), (2, 2), (4, 2)]

-- Function to calculate the area of a right triangle
def rightTriangleArea (base height : ℕ) : ℚ :=
  (base * height) / 2

-- Theorem stating that the overlapping area is 1 square unit
theorem overlap_area :
  let overlapBase := 1
  let overlapHeight := 2
  rightTriangleArea overlapBase overlapHeight = 1 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_l205_20575


namespace NUMINAMATH_CALUDE_circle_radius_order_l205_20588

theorem circle_radius_order :
  let rA : ℝ := Real.sqrt 50
  let aB : ℝ := 16 * Real.pi
  let cC : ℝ := 10 * Real.pi
  let rB : ℝ := Real.sqrt (aB / Real.pi)
  let rC : ℝ := cC / (2 * Real.pi)
  rB < rC ∧ rC < rA := by sorry

end NUMINAMATH_CALUDE_circle_radius_order_l205_20588


namespace NUMINAMATH_CALUDE_rectangle_ratio_l205_20594

theorem rectangle_ratio (w : ℚ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l205_20594


namespace NUMINAMATH_CALUDE_min_value_expression_l205_20545

theorem min_value_expression (a b : ℝ) (h : a ≠ -1) :
  |a + b| + |1 / (a + 1) - b| ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l205_20545


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l205_20557

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l205_20557


namespace NUMINAMATH_CALUDE_minimum_radios_l205_20585

theorem minimum_radios (n d : ℕ) (h1 : d > 0) : 
  (3 * (d / n / 3) + (n - 3) * (d / n + 12) - d = 108) →
  (∀ m : ℕ, m < n → ¬(3 * (d / m / 3) + (m - 3) * (d / m + 12) - d = 108)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_minimum_radios_l205_20585


namespace NUMINAMATH_CALUDE_log_expression_equality_l205_20566

theorem log_expression_equality : 
  Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) + 
  (Real.log 24 / Real.log 2 - Real.log 6 / Real.log 2 + 6) ^ (2/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l205_20566


namespace NUMINAMATH_CALUDE_function_range_theorem_l205_20516

open Real

theorem function_range_theorem (a : ℝ) (h₁ : a > 0) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, 
    a * x₁ + 2 = x₀^2 - 2*x₀) → 0 < a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l205_20516


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_equals_62_l205_20518

theorem root_sum_reciprocal_equals_62 :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x^3 - 9*x^2 + 9*x = 1 ↔ x = a ∨ x = b ∨ x = 1) ∧
    a > b ∧
    a > 1 ∧
    b < 1 ∧
    a/b + b/a = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_equals_62_l205_20518


namespace NUMINAMATH_CALUDE_grid_toothpicks_l205_20597

/-- Calculates the total number of toothpicks in a grid with diagonals -/
def total_toothpicks (length width : ℕ) : ℕ :=
  let vertical := (length + 1) * width
  let horizontal := (width + 1) * length
  let diagonal := 2 * (length * width)
  vertical + horizontal + diagonal

/-- Theorem stating that a 50x20 grid with diagonals uses 4070 toothpicks -/
theorem grid_toothpicks : total_toothpicks 50 20 = 4070 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l205_20597


namespace NUMINAMATH_CALUDE_paint_calculation_l205_20525

/-- The amount of paint Joe uses given the initial amount and usage fractions -/
def paint_used (initial : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week := first_week_fraction * initial
  let remaining := initial - first_week
  let second_week := second_week_fraction * remaining
  first_week + second_week

/-- Theorem stating that given 360 gallons of paint, if 2/3 is used in the first week
    and 1/5 of the remainder is used in the second week, the total amount of paint used is 264 gallons -/
theorem paint_calculation :
  paint_used 360 (2/3) (1/5) = 264 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l205_20525


namespace NUMINAMATH_CALUDE_thirty_percent_passed_l205_20501

/-- The swim club scenario -/
structure SwimClub where
  total_members : ℕ
  not_passed_with_course : ℕ
  not_passed_without_course : ℕ

/-- Calculate the percentage of members who passed the lifesaving test -/
def passed_percentage (club : SwimClub) : ℚ :=
  1 - (club.not_passed_with_course + club.not_passed_without_course : ℚ) / club.total_members

/-- The theorem stating that 30% of members passed the test -/
theorem thirty_percent_passed (club : SwimClub) 
  (h1 : club.total_members = 50)
  (h2 : club.not_passed_with_course = 5)
  (h3 : club.not_passed_without_course = 30) : 
  passed_percentage club = 30 / 100 := by
  sorry

#eval passed_percentage ⟨50, 5, 30⟩

end NUMINAMATH_CALUDE_thirty_percent_passed_l205_20501


namespace NUMINAMATH_CALUDE_player1_can_win_l205_20592

/-- Represents a square on the game board -/
structure Square where
  x : Fin 2021
  y : Fin 2021

/-- Represents a domino placement on the game board -/
structure Domino where
  square1 : Square
  square2 : Square

/-- The game state -/
structure GameState where
  board : Fin 2021 → Fin 2021 → Bool
  dominoes : List Domino

/-- A player's strategy -/
def Strategy := GameState → Domino

/-- The game play function -/
def play (player1Strategy player2Strategy : Strategy) : GameState :=
  sorry

theorem player1_can_win :
  ∃ (player1Strategy : Strategy),
    ∀ (player2Strategy : Strategy),
      let finalState := play player1Strategy player2Strategy
      ∃ (s1 s2 : Square), s1 ≠ s2 ∧ finalState.board s1.x s1.y = false ∧ finalState.board s2.x s2.y = false :=
  sorry


end NUMINAMATH_CALUDE_player1_can_win_l205_20592


namespace NUMINAMATH_CALUDE_right_triangle_smaller_angle_l205_20542

theorem right_triangle_smaller_angle (a b c : Real) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  c = 90 →           -- One angle is 90° (right angle)
  b = 2 * a →        -- One angle is twice the other
  a = 30 :=          -- The smaller angle is 30°
by sorry

end NUMINAMATH_CALUDE_right_triangle_smaller_angle_l205_20542


namespace NUMINAMATH_CALUDE_problem_solution_l205_20567

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Define the theorem
theorem problem_solution :
  -- Part 1
  (∀ x ∈ Set.Icc 1 3, g 1 x ∈ Set.Icc 0 4) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, g a x ∈ Set.Icc 0 4) → a = 1) ∧
  
  -- Part 2
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 1 → g 1 (2^x) - k * 4^x ≥ 0) ↔ k ≤ 1/4) ∧
  
  -- Part 3
  (∀ k : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (g 1 (|2^x₁ - 1|) / |2^x₁ - 1| + k * (2 / |2^x₁ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₂ - 1|) / |2^x₂ - 1| + k * (2 / |2^x₂ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₃ - 1|) / |2^x₃ - 1| + k * (2 / |2^x₃ - 1|) - 3*k = 0)) ↔
   k > 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l205_20567


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_mixing_l205_20517

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℝ := v.capacity * v.alcoholConcentration

/-- Theorem: The alcohol concentration in vessel D after mixing is 35% -/
theorem alcohol_concentration_after_mixing 
  (vesselA : Vessel)
  (vesselB : Vessel)
  (vesselC : Vessel)
  (vesselD : Vessel)
  (h1 : vesselA.capacity = 5)
  (h2 : vesselA.alcoholConcentration = 0.25)
  (h3 : vesselB.capacity = 12)
  (h4 : vesselB.alcoholConcentration = 0.45)
  (h5 : vesselC.capacity = 7)
  (h6 : vesselC.alcoholConcentration = 0.35)
  (h7 : vesselD.capacity = 26) :
  let totalAlcohol := alcoholAmount vesselA + alcoholAmount vesselB + alcoholAmount vesselC
  let totalVolume := vesselA.capacity + vesselB.capacity + vesselC.capacity + (vesselD.capacity - (vesselA.capacity + vesselB.capacity + vesselC.capacity))
  totalAlcohol / totalVolume = 0.35 := by
    sorry

end NUMINAMATH_CALUDE_alcohol_concentration_after_mixing_l205_20517


namespace NUMINAMATH_CALUDE_f_is_even_l205_20565

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem that f is an even function
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l205_20565


namespace NUMINAMATH_CALUDE_parallelogram_angle_ratio_l205_20534

-- Define the parallelogram ABCD and point O
variable (A B C D O : Point)

-- Define the property of being a parallelogram
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define the property of O being the intersection of diagonals
def is_diagonal_intersection (A B C D O : Point) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem parallelogram_angle_ratio 
  (h_para : is_parallelogram A B C D)
  (h_diag : is_diagonal_intersection A B C D O)
  (h_cab : angle_measure C A B = 3 * angle_measure D B A)
  (h_dbc : angle_measure D B C = 3 * angle_measure D B A)
  (h_acb : ∃ r : ℝ, angle_measure A C B = r * angle_measure A O B) :
  ∃ r : ℝ, angle_measure A C B = r * angle_measure A O B ∧ r = 2 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_ratio_l205_20534


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l205_20561

/-- A line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point --/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line has equal intercepts on both axes --/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

/-- The main theorem --/
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), l.passesThrough ⟨1, 4⟩ ∧ l.hasEqualIntercepts ∧
  ((l.a = 4 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l205_20561


namespace NUMINAMATH_CALUDE_toms_age_ratio_l205_20519

theorem toms_age_ratio (T N : ℝ) : T > 0 → N > 0 → T - N = 3 * (T - 3 * N) → T / N = 4 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l205_20519


namespace NUMINAMATH_CALUDE_razorback_shop_tshirts_l205_20506

/-- The Razorback shop problem -/
theorem razorback_shop_tshirts :
  let profit_per_tshirt : ℕ := 62
  let total_tshirt_revenue : ℕ := 11346
  let num_tshirts : ℕ := total_tshirt_revenue / profit_per_tshirt
  num_tshirts = 183 := by sorry

end NUMINAMATH_CALUDE_razorback_shop_tshirts_l205_20506


namespace NUMINAMATH_CALUDE_truck_capacity_l205_20500

theorem truck_capacity (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 15.5) 
  (h2 : 5 * x + 6 * y = 35) : 
  3 * x + 5 * y = 24.5 := by
sorry

end NUMINAMATH_CALUDE_truck_capacity_l205_20500


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l205_20527

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 10*x + 3

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3*x^2 - 10

-- Theorem statement
theorem tangent_point_coordinates :
  ∀ (x y : ℝ),
    x < 0 →
    y = curve x →
    curve_derivative x = 2 →
    (x = -2 ∧ y = 15) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l205_20527


namespace NUMINAMATH_CALUDE_perfect_square_condition_l205_20550

/-- A quadratic trinomial ax^2 + bx + c is a perfect square if there exists a real number r such that ax^2 + bx + c = (√a * x + r)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

/-- The main theorem: if x^2 - mx + 16 is a perfect square trinomial, then m = 8 or m = -8 -/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l205_20550


namespace NUMINAMATH_CALUDE_leila_toy_donation_l205_20555

theorem leila_toy_donation (leila_bags : ℕ) (mohamed_bags : ℕ) (mohamed_toys_per_bag : ℕ) (extra_toys : ℕ) :
  leila_bags = 2 →
  mohamed_bags = 3 →
  mohamed_toys_per_bag = 19 →
  extra_toys = 7 →
  mohamed_bags * mohamed_toys_per_bag = leila_bags * (mohamed_bags * mohamed_toys_per_bag - extra_toys) / leila_bags →
  (mohamed_bags * mohamed_toys_per_bag - extra_toys) / leila_bags = 25 :=
by sorry

end NUMINAMATH_CALUDE_leila_toy_donation_l205_20555


namespace NUMINAMATH_CALUDE_curve_C_polar_equation_l205_20589

/-- Given a curve C with parametric equations x = 1 + cos α and y = sin α, 
    its polar equation is ρ = 2cos θ -/
theorem curve_C_polar_equation (α θ : Real) (ρ x y : Real) :
  (x = 1 + Real.cos α ∧ y = Real.sin α) →
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ρ = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_curve_C_polar_equation_l205_20589


namespace NUMINAMATH_CALUDE_journey_distance_is_25_l205_20502

/-- Represents a segment of the journey with speed and duration -/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance covered in a segment -/
def distance_covered (s : Segment) : ℝ := s.speed * s.duration

/-- The journey segments -/
def journey_segments : List Segment := [
  ⟨4, 1⟩,
  ⟨5, 0.5⟩,
  ⟨3, 0.75⟩,
  ⟨2, 0.5⟩,
  ⟨6, 0.5⟩,
  ⟨7, 0.25⟩,
  ⟨4, 1.5⟩,
  ⟨6, 0.75⟩
]

/-- The total distance covered during the journey -/
def total_distance : ℝ := (journey_segments.map distance_covered).sum

theorem journey_distance_is_25 : total_distance = 25 := by sorry

end NUMINAMATH_CALUDE_journey_distance_is_25_l205_20502


namespace NUMINAMATH_CALUDE_polar_to_cartesian_and_intersection_l205_20530

-- Define the polar equations
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - 2 * Real.pi / 3) = -Real.sqrt 3

def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the standard equations
def standard_line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

def standard_circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the theorem
theorem polar_to_cartesian_and_intersection :
  (∀ ρ θ : ℝ, line_l ρ θ ↔ ∃ x y : ℝ, standard_line_l x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ ρ θ : ℝ, circle_C ρ θ ↔ ∃ x y : ℝ, standard_circle_C x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∃ A B : ℝ × ℝ, 
    standard_line_l A.1 A.2 ∧ standard_circle_C A.1 A.2 ∧
    standard_line_l B.1 B.2 ∧ standard_circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 19) :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_and_intersection_l205_20530


namespace NUMINAMATH_CALUDE_max_coconuts_count_l205_20596

/-- Represents the trading ratios and final goat count -/
structure TradingSystem where
  coconuts_per_crab : ℕ
  crabs_per_goat : ℕ
  final_goats : ℕ

/-- Calculates the number of coconuts Max has -/
def coconuts_count (ts : TradingSystem) : ℕ :=
  ts.coconuts_per_crab * ts.crabs_per_goat * ts.final_goats

/-- Theorem stating that Max has 342 coconuts given the trading system -/
theorem max_coconuts_count :
  let ts : TradingSystem := ⟨3, 6, 19⟩
  coconuts_count ts = 342 := by
  sorry


end NUMINAMATH_CALUDE_max_coconuts_count_l205_20596


namespace NUMINAMATH_CALUDE_inequality_proof_l205_20564

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + 1/a) * (1 + 1/b) ≥ 8 / (1 + a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l205_20564


namespace NUMINAMATH_CALUDE_hardly_arrangements_l205_20554

/-- The number of letters in the word "hardly" -/
def word_length : Nat := 6

/-- The number of letters to be arranged (excluding the fixed 'd') -/
def letters_to_arrange : Nat := 5

/-- Factorial function -/
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem hardly_arrangements :
  factorial letters_to_arrange = 120 :=
by sorry

end NUMINAMATH_CALUDE_hardly_arrangements_l205_20554


namespace NUMINAMATH_CALUDE_N2O3_molecular_weight_N2O3_is_limiting_reactant_l205_20508

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of N2O3
def N2O3_formula : Nat × Nat := (2, 3)

-- Define the number of moles for each reactant
def moles_N2O3 : ℝ := 3
def moles_SO2 : ℝ := 4

-- Define the function to calculate molecular weight
def molecular_weight (n_atoms_N n_atoms_O : Nat) : ℝ :=
  (n_atoms_N : ℝ) * atomic_weight_N + (n_atoms_O : ℝ) * atomic_weight_O

-- Define the function to determine the limiting reactant
def is_limiting_reactant (moles_A moles_B : ℝ) (coeff_A coeff_B : Nat) : Prop :=
  moles_A / (coeff_A : ℝ) < moles_B / (coeff_B : ℝ)

-- Theorem statements
theorem N2O3_molecular_weight :
  molecular_weight N2O3_formula.1 N2O3_formula.2 = 76.02 := by sorry

theorem N2O3_is_limiting_reactant :
  is_limiting_reactant moles_N2O3 moles_SO2 1 1 := by sorry

end NUMINAMATH_CALUDE_N2O3_molecular_weight_N2O3_is_limiting_reactant_l205_20508


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l205_20532

theorem propositions_p_and_q :
  (¬ ∀ x : ℝ, x^2 ≥ x) ∧ (∃ x : ℝ, x^2 ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l205_20532


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l205_20522

/-- Two lines are parallel if and only if their slopes are equal and they don't coincide -/
def parallel (m : ℝ) : Prop :=
  (m / 3 = 1 / (m - 2)) ∧ (-5 ≠ -1 / (m - 2))

/-- The theorem states that if the lines are parallel, then m is either 3 or -1 -/
theorem parallel_lines_m_values (m : ℝ) :
  parallel m → m = 3 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l205_20522


namespace NUMINAMATH_CALUDE_square_difference_l205_20576

theorem square_difference (x : ℤ) (h : x^2 = 3136) : (x + 2) * (x - 2) = 3132 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l205_20576


namespace NUMINAMATH_CALUDE_hypotenuse_length_l205_20529

def right_triangle_hypotenuse (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 36 ∧  -- perimeter condition
  (1/2) * a * b = 24 ∧  -- area condition
  a^2 + b^2 = c^2  -- Pythagorean theorem

theorem hypotenuse_length :
  ∃ a b c : ℝ, right_triangle_hypotenuse a b c ∧ c = 50/3 :=
by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l205_20529


namespace NUMINAMATH_CALUDE_even_decreasing_nonpos_ordering_l205_20591

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonpos (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_nonpos_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) (h_decr : decreasing_on_nonpos f) : 
  f 1 < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_nonpos_ordering_l205_20591


namespace NUMINAMATH_CALUDE_complex_square_root_l205_20562

theorem complex_square_root (z : ℂ) : 
  z^2 = -3 - 4*I ∧ z.re < 0 ∧ z.im > 0 → z = -1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_l205_20562


namespace NUMINAMATH_CALUDE_quadratic_inequality_constant_value_theorem_constant_function_inequality_l205_20553

-- 1. Prove that for all real x, x^2 + 2x + 2 ≥ 1
theorem quadratic_inequality (x : ℝ) : x^2 + 2*x + 2 ≥ 1 := by sorry

-- 2. Prove that for a > 0 and c < 0, min(3|ax^2 + bx + c| + 2) = 2
theorem constant_value_theorem (a b c : ℝ) (ha : a > 0) (hc : c < 0) :
  ∀ x, 3 * |a * x^2 + b * x + c| + 2 ≥ 2 := by sorry

-- 3. Prove that for y = ax^2 + bx + c where b > a > 0 and y ≥ 0 for all real x,
--    if (a+b+c)/(a+b) > m for all a, b, c satisfying the conditions, then m ≤ 9/8
theorem constant_function_inequality (a b c : ℝ) (hab : b > a) (ha : a > 0) 
  (h_nonneg : ∀ x, a * x^2 + b * x + c ≥ 0) 
  (h_inequality : (a + b + c) / (a + b) > 0) :
  (a + b + c) / (a + b) ≤ 9/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_constant_value_theorem_constant_function_inequality_l205_20553


namespace NUMINAMATH_CALUDE_solution_set_l205_20572

-- Define the condition
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*a*x + a > 0

-- Define the inequality
def inequality (a t : ℝ) : Prop :=
  a^(2*t + 1) < a^(t^2 + 2*t - 3)

-- State the theorem
theorem solution_set (a : ℝ) (h : always_positive a) :
  {t : ℝ | inequality a t} = {t : ℝ | -2 < t ∧ t < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_l205_20572


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l205_20581

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 2*p^2 - p + 2 = 0) → 
  (q^3 - 2*q^2 - q + 2 = 0) → 
  (r^3 - 2*r^2 - r + 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l205_20581


namespace NUMINAMATH_CALUDE_max_prime_factors_b_l205_20570

theorem max_prime_factors_b (a b : ℕ+) 
  (h_gcd : (Nat.gcd a b).factors.length = 10)
  (h_lcm : (Nat.lcm a b).factors.length = 25)
  (h_fewer : (b.val.factors.length : ℤ) < a.val.factors.length) :
  b.val.factors.length ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_max_prime_factors_b_l205_20570


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l205_20504

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem point_in_second_quadrant : 
  let z : ℂ := (complex_number 1 2) / (complex_number 1 (-1))
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l205_20504


namespace NUMINAMATH_CALUDE_trivia_team_points_per_member_l205_20511

theorem trivia_team_points_per_member 
  (total_members : ℕ) 
  (absent_members : ℕ) 
  (total_points : ℕ) 
  (h1 : total_members = 9) 
  (h2 : absent_members = 3) 
  (h3 : total_points = 12) : 
  (total_points / (total_members - absent_members) : ℚ) = 2 := by
sorry

#eval (12 : ℚ) / 6  -- This should evaluate to 2

end NUMINAMATH_CALUDE_trivia_team_points_per_member_l205_20511


namespace NUMINAMATH_CALUDE_complex_multiplication_l205_20548

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l205_20548


namespace NUMINAMATH_CALUDE_wang_processing_time_l205_20595

/-- Given that Master Wang processes 92 parts in 4 days, 
    prove that it takes 9 days to process 207 parts using proportion. -/
theorem wang_processing_time 
  (parts_per_four_days : ℕ) 
  (h_parts : parts_per_four_days = 92) 
  (new_parts : ℕ) 
  (h_new_parts : new_parts = 207) : 
  (4 : ℚ) * new_parts / parts_per_four_days = 9 := by
  sorry

end NUMINAMATH_CALUDE_wang_processing_time_l205_20595


namespace NUMINAMATH_CALUDE_stars_per_student_l205_20584

theorem stars_per_student (total_students : ℕ) (total_stars : ℕ) 
  (h1 : total_students = 124) 
  (h2 : total_stars = 372) : 
  total_stars / total_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_stars_per_student_l205_20584


namespace NUMINAMATH_CALUDE_triangle_properties_l205_20520

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The given triangle satisfies the specified conditions -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (2 * t.c - t.b) * Real.cos t.A ∧
  t.a = Real.sqrt 7 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = π / 3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l205_20520


namespace NUMINAMATH_CALUDE_sampling_inspection_correct_for_yeast_l205_20536

/-- Represents a biological experimental technique --/
inductive BiologicalTechnique
| YeastCounting
| SoilAnimalRichness
| OnionRootMitosis
| FatIdentification

/-- Represents a method used in biological experiments --/
inductive ExperimentalMethod
| SamplingInspection
| MarkRecapture
| RinsingForDye
| HydrochloricAcidWashing

/-- Function that returns the correct method for a given technique --/
def correct_method (technique : BiologicalTechnique) : ExperimentalMethod :=
  match technique with
  | BiologicalTechnique.YeastCounting => ExperimentalMethod.SamplingInspection
  | _ => ExperimentalMethod.SamplingInspection  -- Placeholder for other techniques

/-- Theorem stating that the sampling inspection method is correct for yeast counting --/
theorem sampling_inspection_correct_for_yeast :
  correct_method BiologicalTechnique.YeastCounting = ExperimentalMethod.SamplingInspection :=
by sorry

end NUMINAMATH_CALUDE_sampling_inspection_correct_for_yeast_l205_20536


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l205_20531

/-- Proves that given a boat with a speed of 20 km/hr in still water,
if it travels 26 km downstream and 14 km upstream in the same time,
then the speed of the stream is 6 km/hr. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 20 →
  downstream_distance = 26 →
  upstream_distance = 14 →
  (downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) →
  x = 6 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l205_20531


namespace NUMINAMATH_CALUDE_probability_of_sum_15_is_correct_l205_20528

/-- Represents a standard 52-card deck -/
def standardDeck : Nat := 52

/-- Represents the number of cards for each value in a standard deck -/
def cardsPerValue : Nat := 4

/-- Represents the probability of drawing two number cards (2 through 10) 
    from a standard 52-card deck that total 15 -/
def probabilityOfSum15 : ℚ := 28 / 221

theorem probability_of_sum_15_is_correct : 
  probabilityOfSum15 = (
    -- Probability of drawing a 5, 6, 7, 8, or 9 first, then completing the pair
    (5 * cardsPerValue * 4 * cardsPerValue) / (standardDeck * (standardDeck - 1)) +
    -- Probability of drawing a 10 first, then a 5
    (cardsPerValue * cardsPerValue) / (standardDeck * (standardDeck - 1))
  ) := by sorry

end NUMINAMATH_CALUDE_probability_of_sum_15_is_correct_l205_20528


namespace NUMINAMATH_CALUDE_two_composites_in_sequence_l205_20556

/-- A sequence where each term is formed by appending a digit (other than 9) to the preceding term -/
def AppendDigitSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ d, 0 ≤ d ∧ d ≤ 8 ∧ a (n + 1) = 10 * a n + d

/-- Proposition: In an infinite sequence where each term is formed by appending a digit (other than 9)
    to the preceding term, and the first term is any two-digit number, 
    there are at least two composite numbers in the sequence. -/
theorem two_composites_in_sequence (a : ℕ → ℕ) 
    (h_seq : AppendDigitSequence a) 
    (h_start : 10 ≤ a 0 ∧ a 0 < 100) : 
  ∃ i j, i ≠ j ∧ ¬ Nat.Prime (a i) ∧ ¬ Nat.Prime (a j) := by
  sorry

end NUMINAMATH_CALUDE_two_composites_in_sequence_l205_20556


namespace NUMINAMATH_CALUDE_opposite_hands_at_343_l205_20573

/-- Represents the number of degrees the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- Represents the number of degrees the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- Represents the number of minutes past 3:00 -/
def minutes_past_three : ℝ := 43

/-- The position of the minute hand after 5 minutes -/
def minute_hand_position (t : ℝ) : ℝ :=
  minute_hand_speed * (t + 5)

/-- The position of the hour hand 4 minutes ago -/
def hour_hand_position (t : ℝ) : ℝ :=
  90 + hour_hand_speed * (t - 4)

/-- Two angles are opposite if their absolute difference is 180 degrees -/
def are_opposite (a b : ℝ) : Prop :=
  abs (a - b) = 180

theorem opposite_hands_at_343 :
  are_opposite 
    (minute_hand_position minutes_past_three) 
    (hour_hand_position minutes_past_three) := by
  sorry

end NUMINAMATH_CALUDE_opposite_hands_at_343_l205_20573


namespace NUMINAMATH_CALUDE_log_simplification_l205_20593

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_simplification : (log10 2)^2 + log10 2 * log10 5 + log10 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l205_20593


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l205_20507

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 0],
    ![0, 1, 2],
    ![2, 0, 1]]

theorem matrix_equation_solution :
  ∃! (p q r : ℤ), B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ p = -3 ∧ q = 3 ∧ r = -9 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l205_20507


namespace NUMINAMATH_CALUDE_sixth_root_of_107918163081_l205_20558

theorem sixth_root_of_107918163081 :
  let n : ℕ := 107918163081
  let expansion : ℕ := 1 * 101^6 + 6 * 101^5 + 15 * 101^4 + 20 * 101^3 + 15 * 101^2 + 6 * 101 + 1
  n = expansion → (n : ℝ) ^ (1/6 : ℝ) = 102 := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_107918163081_l205_20558


namespace NUMINAMATH_CALUDE_inequality_proof_l205_20547

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a + b)^n - a^n - b^n ≥ (2^n - 2) / 2^(n - 2) * a * b * (a + b)^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l205_20547


namespace NUMINAMATH_CALUDE_roberto_salary_l205_20540

/-- Calculates the final salary after two consecutive percentage increases -/
def final_salary (starting_salary : ℝ) (first_increase : ℝ) (second_increase : ℝ) : ℝ :=
  starting_salary * (1 + first_increase) * (1 + second_increase)

/-- Theorem: Roberto's final salary calculation -/
theorem roberto_salary : 
  final_salary 80000 0.4 0.2 = 134400 := by
  sorry

#eval final_salary 80000 0.4 0.2

end NUMINAMATH_CALUDE_roberto_salary_l205_20540


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l205_20509

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧
  ∀ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1 →
  x + y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l205_20509


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l205_20568

/-- Calculates the total cost of a fruit purchase with a quantity-based discount --/
def fruitPurchaseCost (lemonPrice papayaPrice mangoPrice : ℕ) 
                      (lemonQty papayaQty mangoQty : ℕ) 
                      (fruitPerDiscount : ℕ) (discountAmount : ℕ) : ℕ :=
  let totalCost := lemonPrice * lemonQty + papayaPrice * papayaQty + mangoPrice * mangoQty
  let totalFruits := lemonQty + papayaQty + mangoQty
  let discountQty := totalFruits / fruitPerDiscount
  totalCost - discountQty * discountAmount

/-- Theorem: Tom's fruit purchase costs $21 --/
theorem toms_fruit_purchase_cost : 
  fruitPurchaseCost 2 1 4 6 4 2 4 1 = 21 := by
  sorry

#eval fruitPurchaseCost 2 1 4 6 4 2 4 1

end NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l205_20568


namespace NUMINAMATH_CALUDE_circle_areas_sum_l205_20559

/-- The sum of the areas of an infinite series of circles with radii following
    the geometric sequence 1/√(2^(n-1)) is equal to 2π. -/
theorem circle_areas_sum : 
  let radius (n : ℕ) := (1 : ℝ) / Real.sqrt (2 ^ (n - 1))
  let area (n : ℕ) := Real.pi * (radius n) ^ 2
  (∑' n, area n) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_sum_l205_20559


namespace NUMINAMATH_CALUDE_angle_on_diagonal_line_l205_20533

/-- If the terminal side of angle α lies on the line y = x, then α = kπ + π/4 for some integer k. -/
theorem angle_on_diagonal_line (α : ℝ) :
  (∃ (x y : ℝ), x = y ∧ x = Real.cos α ∧ y = Real.sin α) →
  ∃ (k : ℤ), α = k * π + π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_on_diagonal_line_l205_20533


namespace NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l205_20571

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l205_20571


namespace NUMINAMATH_CALUDE_system_solution_l205_20560

theorem system_solution (x y z a : ℝ) 
  (eq1 : x + y + z = a)
  (eq2 : x^2 + y^2 + z^2 = a^2)
  (eq3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l205_20560


namespace NUMINAMATH_CALUDE_parentheses_removal_l205_20551

theorem parentheses_removal (x y : ℝ) : x - 2 * (y - 1) = x - 2 * y + 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l205_20551


namespace NUMINAMATH_CALUDE_min_value_of_x2_plus_y2_l205_20523

theorem min_value_of_x2_plus_y2 (x y : ℝ) : 
  (x - 1)^2 + y^2 = 16 → ∃ (m : ℝ), (∀ (a b : ℝ), (a - 1)^2 + b^2 = 16 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x2_plus_y2_l205_20523


namespace NUMINAMATH_CALUDE_chocolate_difference_l205_20541

theorem chocolate_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 12) 
  (h2 : nickel_chocolates = 3) : 
  robert_chocolates - nickel_chocolates = 9 := by
sorry

end NUMINAMATH_CALUDE_chocolate_difference_l205_20541


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l205_20514

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the focus to the asymptote is equal to the length of the real axis,
    then the eccentricity of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focus_to_asymptote := (b * c) / Real.sqrt (a^2 + b^2)
  focus_to_asymptote = 2 * a →
  c / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l205_20514


namespace NUMINAMATH_CALUDE_dance_off_ratio_l205_20577

-- Define the dancing times and break time
def john_first_dance : ℕ := 3
def john_break : ℕ := 1
def john_second_dance : ℕ := 5
def combined_dance_time : ℕ := 20

-- Define John's total dancing and resting time
def john_total_time : ℕ := john_first_dance + john_break + john_second_dance

-- Define John's dancing time
def john_dance_time : ℕ := john_first_dance + john_second_dance

-- Define James' dancing time
def james_dance_time : ℕ := combined_dance_time - john_dance_time

-- Define James' additional dancing time
def james_additional_time : ℕ := james_dance_time - john_dance_time

-- Theorem to prove
theorem dance_off_ratio : 
  (james_additional_time : ℚ) / john_total_time = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_dance_off_ratio_l205_20577


namespace NUMINAMATH_CALUDE_intersection_distance_implies_omega_l205_20521

/-- Given a function f(x) = 2sin(ωx + φ) where ω > 0, if the curve y = f(x) intersects
    the line y = √3 and the distance between two adjacent intersection points is π/6,
    then ω = 2 or ω = 10. -/
theorem intersection_distance_implies_omega (ω φ : ℝ) (h1 : ω > 0) :
  (∃ x1 x2 : ℝ, x2 - x1 = π / 6 ∧
    2 * Real.sin (ω * x1 + φ) = Real.sqrt 3 ∧
    2 * Real.sin (ω * x2 + φ) = Real.sqrt 3) →
  ω = 2 ∨ ω = 10 := by
  sorry


end NUMINAMATH_CALUDE_intersection_distance_implies_omega_l205_20521


namespace NUMINAMATH_CALUDE_inequality_proof_l205_20524

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hq_geq_p : p ≤ q)
  (ha : p ≤ a ∧ a ≤ q)
  (hb : p ≤ b ∧ b ≤ q)
  (hc : p ≤ c ∧ c ≤ q)
  (hd : p ≤ d ∧ d ≤ q)
  (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l205_20524
