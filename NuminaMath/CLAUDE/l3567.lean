import Mathlib

namespace NUMINAMATH_CALUDE_line_parameterization_l3567_356744

/-- Given a line y = -2x + 7 parameterized by (x, y) = (p, 3) + t(6, l),
    prove that p = 2 and l = -12 -/
theorem line_parameterization (x y p l t : ℝ) : 
  (y = -2 * x + 7) →
  (x = p + 6 * t ∧ y = 3 + l * t) →
  (p = 2 ∧ l = -12) := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3567_356744


namespace NUMINAMATH_CALUDE_max_positive_condition_l3567_356720

theorem max_positive_condition (a : ℝ) :
  (∀ x : ℝ, max (x^3 + 3*x + a - 9) (a + 2^(5-x) - 3^(x-1)) > 0) ↔ a > -5 := by
  sorry

end NUMINAMATH_CALUDE_max_positive_condition_l3567_356720


namespace NUMINAMATH_CALUDE_min_value_at_three_l3567_356768

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 28 + Real.sqrt (9 - x^2)

theorem min_value_at_three :
  ∀ x : ℝ, 9 - x^2 ≥ 0 → f 3 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_three_l3567_356768


namespace NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l3567_356763

theorem positive_integer_pairs_satisfying_equation :
  ∀ a b : ℕ+, 
    (a.val * b.val - a.val - b.val = 12) ↔ ((a = 2 ∧ b = 14) ∨ (a = 14 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l3567_356763


namespace NUMINAMATH_CALUDE_min_max_inequality_l3567_356789

theorem min_max_inequality {a b x₁ x₂ x₃ x₄ : ℝ} 
  (ha : 0 < a) (hab : a < b) 
  (hx₁ : a ≤ x₁ ∧ x₁ ≤ b) (hx₂ : a ≤ x₂ ∧ x₂ ≤ b) 
  (hx₃ : a ≤ x₃ ∧ x₃ ≤ b) (hx₄ : a ≤ x₄ ∧ x₄ ≤ b) :
  1 ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ b/a + a/b - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_max_inequality_l3567_356789


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt5_l3567_356780

theorem complex_modulus_sqrt5 (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (a - 2 * i) * i = b - i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt5_l3567_356780


namespace NUMINAMATH_CALUDE_function_properties_l3567_356776

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

theorem function_properties :
  (∀ x : ℝ, x ≠ -1 → f x = 2 * x / (x + 1)) ∧
  f 1 = 1 ∧
  f (-2) = 4 ∧
  (∃ c : ℝ, ∀ x : ℝ, x ≠ -1 → f x + f (c - x) = 4) ∧
  (∀ x m : ℝ, x ∈ Set.Icc 1 2 → 2 < m → m ≤ 4 → f x ≤ 2 * m / ((x + 1) * |x - m|)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3567_356776


namespace NUMINAMATH_CALUDE_largest_crate_dimension_l3567_356798

def crate_width : ℝ := 5
def crate_length : ℝ := 8
def pillar_radius : ℝ := 5

theorem largest_crate_dimension (height : ℝ) :
  height ≥ 2 * pillar_radius →
  crate_width ≥ 2 * pillar_radius →
  crate_length ≥ 2 * pillar_radius →
  (∃ (max_dim : ℝ), max_dim = max height (max crate_width crate_length) ∧ max_dim = 2 * pillar_radius) :=
by sorry

end NUMINAMATH_CALUDE_largest_crate_dimension_l3567_356798


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l3567_356700

/-- The probability of picking two red balls from a bag -/
theorem prob_two_red_balls (total_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 12) (h2 : red_balls = 5) : 
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 5 / 33 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l3567_356700


namespace NUMINAMATH_CALUDE_gift_purchase_cost_l3567_356711

def total_cost (items : List (ℕ × ℚ)) (sales_tax_rate : ℚ) (credit_card_rebate : ℚ)
  (book_discount_rate : ℚ) (sneaker_discount_rate : ℚ) : ℚ :=
  sorry

theorem gift_purchase_cost :
  let items : List (ℕ × ℚ) := [
    (3, 26), (2, 83), (1, 90), (4, 7), (3, 15), (2, 22), (5, 8), (1, 65)
  ]
  let sales_tax_rate : ℚ := 6.5 / 100
  let credit_card_rebate : ℚ := 12
  let book_discount_rate : ℚ := 10 / 100
  let sneaker_discount_rate : ℚ := 15 / 100
  total_cost items sales_tax_rate credit_card_rebate book_discount_rate sneaker_discount_rate = 564.96 :=
by sorry

end NUMINAMATH_CALUDE_gift_purchase_cost_l3567_356711


namespace NUMINAMATH_CALUDE_average_pen_price_is_correct_l3567_356743

/-- The average price of a pen before discount given the following conditions:
  * 30 pens and 75 pencils were purchased
  * The total cost after a 10% discount is $510
  * The average price of a pencil before discount is $2.00
-/
def averagePenPrice (numPens : ℕ) (numPencils : ℕ) (totalCostAfterDiscount : ℚ) 
  (pencilPrice : ℚ) (discountRate : ℚ) : ℚ :=
  let totalCostBeforeDiscount : ℚ := totalCostAfterDiscount / (1 - discountRate)
  let totalPencilCost : ℚ := numPencils * pencilPrice
  let totalPenCost : ℚ := totalCostBeforeDiscount - totalPencilCost
  totalPenCost / numPens

theorem average_pen_price_is_correct : 
  averagePenPrice 30 75 510 2 (1/10) = 13.89 := by
  sorry

end NUMINAMATH_CALUDE_average_pen_price_is_correct_l3567_356743


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3567_356765

/-- Given a line with equation 5x - 4y = 20, the slope of the perpendicular line is -4/5 -/
theorem perpendicular_slope (x y : ℝ) : 
  (5 * x - 4 * y = 20) → 
  (∃ m : ℝ, m = -4/5 ∧ m * (5/4) = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3567_356765


namespace NUMINAMATH_CALUDE_evening_ticket_price_is_seven_l3567_356740

/-- Represents the earnings of a movie theater on a single day. -/
structure TheaterEarnings where
  matineePrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  openingNightCustomers : ℕ
  totalEarnings : ℕ

/-- Calculates the evening ticket price based on the theater's earnings. -/
def eveningTicketPrice (e : TheaterEarnings) : ℕ :=
  let totalCustomers := e.matineeCustomers + e.eveningCustomers + e.openingNightCustomers
  let popcornEarnings := (totalCustomers / 2) * e.popcornPrice
  let knownEarnings := e.matineeCustomers * e.matineePrice + 
                       e.openingNightCustomers * e.openingNightPrice + 
                       popcornEarnings
  (e.totalEarnings - knownEarnings) / e.eveningCustomers

/-- Theorem stating that the evening ticket price is 7 dollars given the specific conditions. -/
theorem evening_ticket_price_is_seven :
  let e : TheaterEarnings := {
    matineePrice := 5,
    openingNightPrice := 10,
    popcornPrice := 10,
    matineeCustomers := 32,
    eveningCustomers := 40,
    openingNightCustomers := 58,
    totalEarnings := 1670
  }
  eveningTicketPrice e = 7 := by sorry

end NUMINAMATH_CALUDE_evening_ticket_price_is_seven_l3567_356740


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3567_356797

/-- The probability of rolling a number other than 1 on a standard die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability that (a-1)(b-1)(c-1)(d-1) ≠ 0 when four standard dice are tossed -/
def prob_product_nonzero : ℚ := prob_not_one ^ num_dice

theorem dice_roll_probability :
  prob_product_nonzero = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3567_356797


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3567_356792

/-- 
Given a point P with coordinates (-5, 3) in the Cartesian coordinate system,
prove that its coordinates with respect to the origin are (-5, 3).
-/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-5, 3)
  P = (-5, 3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3567_356792


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3567_356718

theorem opposite_of_negative_two :
  -((-2 : ℤ)) = (2 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3567_356718


namespace NUMINAMATH_CALUDE_reseating_women_circular_l3567_356785

-- Define the recurrence relation for reseating women
def R : ℕ → ℕ
  | 0 => 0  -- We define R(0) as 0 for completeness
  | 1 => 1
  | 2 => 2
  | (n + 3) => R (n + 2) + R (n + 1)

-- Theorem statement
theorem reseating_women_circular (n : ℕ) : R 15 = 987 := by
  sorry

-- You can also add additional lemmas to help prove the main theorem
lemma R_recurrence (n : ℕ) : n ≥ 3 → R n = R (n - 1) + R (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_reseating_women_circular_l3567_356785


namespace NUMINAMATH_CALUDE_exists_password_with_twenty_permutations_l3567_356759

/-- Represents a password as a list of characters -/
def Password := List Char

/-- Counts the number of unique permutations of a password -/
def countUniquePermutations (p : Password) : Nat :=
  sorry

/-- Theorem: There exists a 5-character password with exactly 20 different permutations -/
theorem exists_password_with_twenty_permutations :
  ∃ (p : Password), p.length = 5 ∧ countUniquePermutations p = 20 := by
  sorry

end NUMINAMATH_CALUDE_exists_password_with_twenty_permutations_l3567_356759


namespace NUMINAMATH_CALUDE_socks_purchase_problem_l3567_356788

theorem socks_purchase_problem :
  ∃ (a b c : ℕ), 
    a + b + c = 15 ∧
    2 * a + 3 * b + 5 * c = 40 ∧
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
    (a = 7 ∨ a = 9 ∨ a = 11) :=
by sorry

end NUMINAMATH_CALUDE_socks_purchase_problem_l3567_356788


namespace NUMINAMATH_CALUDE_charlies_coins_l3567_356786

theorem charlies_coins (total_coins : ℕ) (pennies nickels : ℕ) : 
  total_coins = 17 →
  pennies + nickels = total_coins →
  pennies = nickels + 2 →
  pennies * 1 + nickels * 5 = 44 :=
by sorry

end NUMINAMATH_CALUDE_charlies_coins_l3567_356786


namespace NUMINAMATH_CALUDE_constant_ratio_theorem_l3567_356762

theorem constant_ratio_theorem (x₁ x₂ : ℚ) (y₁ y₂ : ℚ) (k : ℚ) :
  (2 * x₁ - 5) / (y₁ + 10) = k →
  (2 * x₂ - 5) / (y₂ + 10) = k →
  x₁ = 5 →
  y₁ = 4 →
  y₂ = 8 →
  x₂ = 40 / 7 := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_theorem_l3567_356762


namespace NUMINAMATH_CALUDE_car_cleaning_time_l3567_356705

theorem car_cleaning_time (outside_time : ℕ) (inside_time : ℕ) : 
  outside_time = 80 →
  inside_time = outside_time / 4 →
  outside_time + inside_time = 100 :=
by sorry

end NUMINAMATH_CALUDE_car_cleaning_time_l3567_356705


namespace NUMINAMATH_CALUDE_climb_out_of_well_l3567_356752

/-- The number of days it takes for a man to climb out of a well -/
def daysToClimbWell (wellDepth : ℕ) (climbUp : ℕ) (slipDown : ℕ) : ℕ :=
  let dailyProgress := climbUp - slipDown
  let daysForMostOfWell := (wellDepth - 1) / dailyProgress
  let remainingDistance := (wellDepth - 1) % dailyProgress
  if remainingDistance = 0 then
    daysForMostOfWell + 1
  else
    daysForMostOfWell + 2

/-- Theorem stating that it takes 30 days to climb out of a 30-meter well 
    when climbing 4 meters up and slipping 3 meters down each day -/
theorem climb_out_of_well : daysToClimbWell 30 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_climb_out_of_well_l3567_356752


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3567_356725

/-- The circle equation x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation ax - by + 2 = 0 -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x - b*y + 2 = 0

/-- The chord length is 4 -/
def chord_length (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation a b x₁ y₁ ∧ line_equation a b x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  chord_length a b → (1/a + 1/b ≥ 3/2 + Real.sqrt 2) ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ chord_length a₀ b₀ ∧ 1/a₀ + 1/b₀ = 3/2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3567_356725


namespace NUMINAMATH_CALUDE_inequalities_problem_l3567_356781

theorem inequalities_problem (a b c d : ℝ) 
  (ha : a > 0) 
  (hb1 : 0 > b) 
  (hb2 : b > -a) 
  (hc : c < d) 
  (hd : d < 0) : 
  (a / b + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end NUMINAMATH_CALUDE_inequalities_problem_l3567_356781


namespace NUMINAMATH_CALUDE_fraction_equality_l3567_356747

theorem fraction_equality : (1 : ℚ) / 4 - (1 : ℚ) / 6 + (1 : ℚ) / 8 = (5 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3567_356747


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3567_356739

theorem cube_volume_from_surface_area :
  ∀ (surface_area : ℝ) (volume : ℝ),
    surface_area = 384 →
    volume = (surface_area / 6) ^ (3/2) →
    volume = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3567_356739


namespace NUMINAMATH_CALUDE_coronene_bond_arrangements_l3567_356737

/-- Represents a molecule with carbon and hydrogen atoms -/
structure Molecule where
  carbon_count : ℕ
  hydrogen_count : ℕ

/-- Represents the bonding requirements for atoms -/
structure BondRequirement where
  carbon_bonds : ℕ
  hydrogen_bonds : ℕ

/-- Counts the number of valid bond arrangements for a given molecule -/
def count_bond_arrangements (m : Molecule) (req : BondRequirement) : ℕ :=
  sorry

/-- Coronene molecule -/
def coronene : Molecule :=
  { carbon_count := 24, hydrogen_count := 12 }

/-- Standard bonding requirement -/
def standard_requirement : BondRequirement :=
  { carbon_bonds := 4, hydrogen_bonds := 1 }

theorem coronene_bond_arrangements :
  count_bond_arrangements coronene standard_requirement = 20 :=
by sorry

end NUMINAMATH_CALUDE_coronene_bond_arrangements_l3567_356737


namespace NUMINAMATH_CALUDE_apple_probability_l3567_356731

theorem apple_probability (p_less_200 p_not_less_350 : ℝ) 
  (h1 : p_less_200 = 0.25) 
  (h2 : p_not_less_350 = 0.22) : 
  1 - p_less_200 - p_not_less_350 = 0.53 := by
  sorry

end NUMINAMATH_CALUDE_apple_probability_l3567_356731


namespace NUMINAMATH_CALUDE_smallest_d_is_one_l3567_356727

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def are_distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts three digits to a three-digit number -/
def to_number (a b c : Digit) : ℕ :=
  100 * a.val + 10 * b.val + c.val

/-- Converts four digits to a four-digit number -/
def to_four_digit (d c b d' : Digit) : ℕ :=
  1000 * d.val + 100 * c.val + 10 * b.val + d'.val

theorem smallest_d_is_one :
  ∃ (a b c d : Digit),
    are_distinct a b c d ∧
    to_number a b c * b.val = to_four_digit d c b d ∧
    ∀ (a' b' c' d' : Digit),
      are_distinct a' b' c' d' →
      to_number a' b' c' * b'.val = to_four_digit d' c' b' d' →
      d.val ≤ d'.val :=
sorry

end NUMINAMATH_CALUDE_smallest_d_is_one_l3567_356727


namespace NUMINAMATH_CALUDE_addends_satisfy_conditions_l3567_356704

/-- Represents the correct sum of two addends -/
def correct_sum : Nat := 982

/-- Represents the incorrect sum when one addend is missing a 0 in the units place -/
def incorrect_sum : Nat := 577

/-- Represents the first addend -/
def addend1 : Nat := 450

/-- Represents the second addend -/
def addend2 : Nat := 532

/-- Theorem stating that the two addends satisfy the problem conditions -/
theorem addends_satisfy_conditions : 
  (addend1 + addend2 = correct_sum) ∧ 
  (addend1 + (addend2 - 50) = incorrect_sum) := by
  sorry

#check addends_satisfy_conditions

end NUMINAMATH_CALUDE_addends_satisfy_conditions_l3567_356704


namespace NUMINAMATH_CALUDE_leap_year_1996_not_others_l3567_356754

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

theorem leap_year_1996_not_others : 
  is_leap_year 1996 ∧ 
  ¬is_leap_year 1998 ∧ 
  ¬is_leap_year 2010 ∧ 
  ¬is_leap_year 2100 :=
by sorry

end NUMINAMATH_CALUDE_leap_year_1996_not_others_l3567_356754


namespace NUMINAMATH_CALUDE_marys_remaining_money_l3567_356757

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  50 - (4 * p + 2.5 * p + 2 * 4 * p)

/-- Theorem stating that Mary's remaining money is 50 - 14.5p dollars -/
theorem marys_remaining_money (p : ℝ) : money_left p = 50 - 14.5 * p := by
  sorry

end NUMINAMATH_CALUDE_marys_remaining_money_l3567_356757


namespace NUMINAMATH_CALUDE_debate_team_selection_l3567_356795

def total_students : ℕ := 9
def students_to_select : ℕ := 4
def specific_students : ℕ := 2

def select_with_condition (n k m : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - m) k

theorem debate_team_selection :
  select_with_condition total_students students_to_select specific_students = 91 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_selection_l3567_356795


namespace NUMINAMATH_CALUDE_highest_validity_rate_is_91_percent_l3567_356703

/-- Represents the voting results for three candidates -/
structure VotingResult where
  total_ballots : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Calculates the highest possible validity rate for a given voting result -/
def highest_validity_rate (result : VotingResult) : ℚ :=
  let total_votes := result.votes_a + result.votes_b + result.votes_c
  let invalid_votes := total_votes - 2 * result.total_ballots
  (result.total_ballots - invalid_votes : ℚ) / result.total_ballots

/-- The main theorem stating the highest possible validity rate -/
theorem highest_validity_rate_is_91_percent (result : VotingResult) :
  result.total_ballots = 100 ∧
  result.votes_a = 88 ∧
  result.votes_b = 75 ∧
  result.votes_c = 46 →
  highest_validity_rate result = 91 / 100 :=
by sorry

#eval highest_validity_rate ⟨100, 88, 75, 46⟩

end NUMINAMATH_CALUDE_highest_validity_rate_is_91_percent_l3567_356703


namespace NUMINAMATH_CALUDE_expression_evaluation_l3567_356769

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℤ := -4
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -11 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3567_356769


namespace NUMINAMATH_CALUDE_knight_2008_winner_condition_l3567_356730

/-- Represents the game where n knights sit at a round table, count 1, 2, 3 clockwise,
    and those who say 2 or 3 are eliminated. -/
def KnightGame (n : ℕ) := True

/-- Predicate indicating whether a knight wins the game -/
def IsWinner (game : KnightGame n) (knight : ℕ) : Prop := True

theorem knight_2008_winner_condition (n : ℕ) :
  (∃ (game : KnightGame n), IsWinner game 2008) ↔
  (∃ (k : ℕ), k ≥ 6 ∧ (n = 1338 + 3^k ∨ n = 1338 + 2 * 3^k)) :=
sorry

end NUMINAMATH_CALUDE_knight_2008_winner_condition_l3567_356730


namespace NUMINAMATH_CALUDE_odd_prime_property_l3567_356772

/-- P(x) is the smallest prime factor of x^2 + 1 -/
noncomputable def P (x : ℕ) : ℕ := Nat.minFac (x^2 + 1)

theorem odd_prime_property (p a : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (ha : a < p) (ha_cong : a^2 + 1 ≡ 0 [MOD p]) : 
  a ≠ p - a ∧ P a = p ∧ P (p - a) = p :=
sorry

end NUMINAMATH_CALUDE_odd_prime_property_l3567_356772


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3567_356735

theorem sin_315_degrees :
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3567_356735


namespace NUMINAMATH_CALUDE_probability_not_snowing_l3567_356706

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 2/7) : 
  1 - p_snow = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snowing_l3567_356706


namespace NUMINAMATH_CALUDE_special_line_unique_l3567_356753

/-- A line that satisfies the given conditions -/
structure SpecialLine where
  m : ℝ
  b : ℝ
  b_nonzero : b ≠ 0
  passes_through : m * 2 + b = 7

/-- The condition for the intersection points -/
def intersectionCondition (l : SpecialLine) (k : ℝ) : Prop :=
  |k^2 + 4*k + 3 - (l.m * k + l.b)| = 4

/-- The main theorem -/
theorem special_line_unique (l : SpecialLine) :
  (∃! k, intersectionCondition l k) → l.m = 10 ∧ l.b = -13 := by
  sorry

end NUMINAMATH_CALUDE_special_line_unique_l3567_356753


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3567_356793

theorem complex_fraction_equality (a b : ℂ) 
  (h1 : (a + b) / (a - b) - (a - b) / (a + b) = 2)
  (h2 : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3567_356793


namespace NUMINAMATH_CALUDE_gerbil_revenue_calculation_l3567_356790

/-- Calculates the total revenue from gerbil sales given the initial stock, percentage sold, original price, and discount rate. -/
def gerbil_revenue (initial_stock : ℕ) (percent_sold : ℚ) (original_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let sold := ⌊initial_stock * percent_sold⌋
  let remaining := initial_stock - sold
  let discounted_price := original_price * (1 - discount_rate)
  sold * original_price + remaining * discounted_price

/-- Theorem stating that the total revenue from gerbil sales is $4696.80 given the specified conditions. -/
theorem gerbil_revenue_calculation :
  gerbil_revenue 450 (35/100) 12 (20/100) = 4696.80 := by
  sorry

end NUMINAMATH_CALUDE_gerbil_revenue_calculation_l3567_356790


namespace NUMINAMATH_CALUDE_star_polygon_forms_pyramid_net_iff_l3567_356764

/-- A structure representing two concentric circles with a star-shaped polygon construction -/
structure ConcentricCirclesWithStarPolygon where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  (r_positive : r > 0)
  (R_greater : R > r)

/-- The condition for the star-shaped polygon to form the net of a pyramid -/
def forms_pyramid_net (c : ConcentricCirclesWithStarPolygon) : Prop :=
  c.R > 2 * c.r

/-- Theorem stating the necessary and sufficient condition for the star-shaped polygon
    to form the net of a pyramid -/
theorem star_polygon_forms_pyramid_net_iff (c : ConcentricCirclesWithStarPolygon) :
  forms_pyramid_net c ↔ c.R > 2 * c.r :=
sorry

end NUMINAMATH_CALUDE_star_polygon_forms_pyramid_net_iff_l3567_356764


namespace NUMINAMATH_CALUDE_yellow_pill_cost_l3567_356702

theorem yellow_pill_cost (weeks : ℕ) (daily_yellow : ℕ) (daily_blue : ℕ) 
  (yellow_blue_diff : ℚ) (total_cost : ℚ) :
  weeks = 3 →
  daily_yellow = 1 →
  daily_blue = 1 →
  yellow_blue_diff = 2 →
  total_cost = 903 →
  ∃ (yellow_cost : ℚ), 
    yellow_cost = 22.5 ∧ 
    (weeks * 7 * (yellow_cost + (yellow_cost - yellow_blue_diff)) = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_yellow_pill_cost_l3567_356702


namespace NUMINAMATH_CALUDE_georges_calculation_l3567_356717

theorem georges_calculation (y : ℝ) : y / 7 = 30 → y + 70 = 280 := by
  sorry

end NUMINAMATH_CALUDE_georges_calculation_l3567_356717


namespace NUMINAMATH_CALUDE_intersection_point_l3567_356714

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ : ℝ) : Prop := ρ^2 - 4*ρ*Real.cos θ + 3 = 0 ∧ 0 ≤ θ ∧ θ ≤ 2*Real.pi

/-- Curve C2 in parametric form -/
def C2 (x y t : ℝ) : Prop := x = t * Real.cos (Real.pi/6) ∧ y = t * Real.sin (Real.pi/6)

/-- The intersection point of C1 and C2 has polar coordinates (√3, π/6) -/
theorem intersection_point : 
  ∃ (ρ θ : ℝ), C1 ρ θ ∧ (∃ (x y t : ℝ), C2 x y t ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
  ρ = Real.sqrt 3 ∧ θ = Real.pi/6 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3567_356714


namespace NUMINAMATH_CALUDE_go_game_probability_l3567_356741

theorem go_game_probability (P : ℝ) (h1 : P > 1/2) 
  (h2 : P^2 + (1-P)^2 = 5/8) : P = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_go_game_probability_l3567_356741


namespace NUMINAMATH_CALUDE_ball_fall_time_l3567_356738

/-- The time for a ball to fall from 60 meters to 30 meters under gravity -/
theorem ball_fall_time (g : Real) (h₀ h₁ : Real) (t : Real) :
  g = 9.8 →
  h₀ = 60 →
  h₁ = 30 →
  h₁ = h₀ - (1/2) * g * t^2 →
  t = Real.sqrt (2 * (h₀ - h₁) / g) :=
by sorry

end NUMINAMATH_CALUDE_ball_fall_time_l3567_356738


namespace NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l3567_356708

/-- Calculates the number of seashells in Jar A after n weeks -/
def shellsInJarA (n : ℕ) : ℕ := 50 + 20 * n

/-- Calculates the number of seashells in Jar B after n weeks -/
def shellsInJarB (n : ℕ) : ℕ := 30 * (2 ^ n)

/-- The total number of seashells in both jars after n weeks -/
def totalShells (n : ℕ) : ℕ := shellsInJarA n + shellsInJarB n

theorem seashell_count_after_six_weeks :
  totalShells 6 = 1110 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l3567_356708


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3567_356712

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (1 - a) * x > 1 - a ↔ x < 1) → a > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3567_356712


namespace NUMINAMATH_CALUDE_repeating_decimal_568_l3567_356761

/-- The repeating decimal 0.568568568... is equal to the fraction 568/999 -/
theorem repeating_decimal_568 : 
  (∑' n : ℕ, (568 : ℚ) / 1000^(n+1)) = 568 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_568_l3567_356761


namespace NUMINAMATH_CALUDE_max_value_trigonometric_expression_l3567_356734

theorem max_value_trigonometric_expression :
  ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), 3 * Real.cos x + 4 * Real.sin x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_expression_l3567_356734


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3567_356760

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  3 * a 1 - (1/2) * a 3 = (1/2) * a 3 - 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a → ArithmeticCondition a →
  ∀ n : ℕ, (a (n + 3) + a (n + 2)) / (a (n + 1) + a n) = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3567_356760


namespace NUMINAMATH_CALUDE_marching_band_weight_l3567_356756

/-- The total weight carried by the Oprah Winfrey High School marching band -/
def total_weight : ℕ :=
  let trumpet_weight := 5 + 3
  let clarinet_weight := 5 + 3
  let trombone_weight := 10 + 4
  let tuba_weight := 20 + 5
  let drummer_weight := 15 + 6
  let percussionist_weight := 8 + 3
  let trumpet_count := 6
  let clarinet_count := 9
  let trombone_count := 8
  let tuba_count := 3
  let drummer_count := 2
  let percussionist_count := 4
  trumpet_count * trumpet_weight +
  clarinet_count * clarinet_weight +
  trombone_count * trombone_weight +
  tuba_count * tuba_weight +
  drummer_count * drummer_weight +
  percussionist_count * percussionist_weight

theorem marching_band_weight : total_weight = 393 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_weight_l3567_356756


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l3567_356779

/-- Calculates the total money earned by Katrina and her friends in the recycling program -/
def total_money_earned (initial_signup_bonus : ℕ) (referral_bonus : ℕ) (friends_day1 : ℕ) (friends_week : ℕ) : ℕ :=
  let katrina_earnings := initial_signup_bonus + referral_bonus * (friends_day1 + friends_week)
  let friends_earnings := referral_bonus * (friends_day1 + friends_week)
  katrina_earnings + friends_earnings

/-- Proves that the total money earned by Katrina and her friends is $125.00 -/
theorem recycling_program_earnings :
  total_money_earned 5 5 5 7 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_recycling_program_earnings_l3567_356779


namespace NUMINAMATH_CALUDE_three_digit_sum_24_count_l3567_356729

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_24_count :
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_three_digit n ∧ digit_sum n = 24) ∧ s.card = 4 ∧
    ∀ n, is_three_digit n → digit_sum n = 24 → n ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_24_count_l3567_356729


namespace NUMINAMATH_CALUDE_solution_l3567_356710

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- Define the set M
def M : Set ℝ := {x | f x = 0}

-- Theorem statement
theorem solution : {1, 3} ∪ {2, 3} = M := by sorry

end NUMINAMATH_CALUDE_solution_l3567_356710


namespace NUMINAMATH_CALUDE_quadratic_with_irrational_root_l3567_356709

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_irrational_root_l3567_356709


namespace NUMINAMATH_CALUDE_f_properties_l3567_356755

noncomputable def f (x : Real) : Real := Real.sqrt 3 * (Real.sin x) ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  let a := π / 2
  let b := π
  ∃ (max_value min_value : Real),
    (∀ x ∈ Set.Icc a b, f x ≤ max_value) ∧
    (∀ x ∈ Set.Icc a b, f x ≥ min_value) ∧
    (f (5 * π / 6) = 0) ∧
    (f π = 0) ∧
    (max_value = Real.sqrt 3) ∧
    (f (π / 2) = max_value) ∧
    (min_value = -1 + Real.sqrt 3 / 2) ∧
    (f (11 * π / 12) = min_value) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3567_356755


namespace NUMINAMATH_CALUDE_weight_of_second_person_l3567_356707

/-- Proves that the weight of the second person who joined the group is 78 kg -/
theorem weight_of_second_person
  (initial_average : ℝ)
  (final_average : ℝ)
  (initial_members : ℕ)
  (weight_first_person : ℝ)
  (h_initial_average : initial_average = 48)
  (h_final_average : final_average = 51)
  (h_initial_members : initial_members = 23)
  (h_weight_first_person : weight_first_person = 93)
  : ∃ (weight_second_person : ℝ),
    weight_second_person = 78 ∧
    (initial_members : ℝ) * final_average =
      initial_members * initial_average + weight_first_person + weight_second_person :=
by sorry

end NUMINAMATH_CALUDE_weight_of_second_person_l3567_356707


namespace NUMINAMATH_CALUDE_max_nickels_l3567_356782

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter

-- Define the wallet as a function from Coin to ℕ (number of each coin type)
def Wallet := Coin → ℕ

-- Define the value of each coin in cents
def coinValue : Coin → ℕ
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10
| Coin.Quarter => 25

-- Function to calculate the total value of coins in the wallet
def totalValue (w : Wallet) : ℕ :=
  (w Coin.Penny) * (coinValue Coin.Penny) +
  (w Coin.Nickel) * (coinValue Coin.Nickel) +
  (w Coin.Dime) * (coinValue Coin.Dime) +
  (w Coin.Quarter) * (coinValue Coin.Quarter)

-- Function to count the total number of coins in the wallet
def coinCount (w : Wallet) : ℕ :=
  (w Coin.Penny) + (w Coin.Nickel) + (w Coin.Dime) + (w Coin.Quarter)

-- Theorem statement
theorem max_nickels (w : Wallet) :
  (totalValue w = 15 * coinCount w) →
  (totalValue w + coinValue Coin.Dime = 16 * (coinCount w + 1)) →
  (w Coin.Nickel = 2) := by
  sorry


end NUMINAMATH_CALUDE_max_nickels_l3567_356782


namespace NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l3567_356732

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if a triangle has perimeter 8 -/
def has_perimeter_8 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 8

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The main theorem to be proved -/
theorem unique_triangle_with_perimeter_8 :
  ∃! t : IntTriangle, has_perimeter_8 t ∧
  (∀ t' : IntTriangle, has_perimeter_8 t' → are_congruent t t') :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l3567_356732


namespace NUMINAMATH_CALUDE_combination_equality_implies_n_18_l3567_356750

theorem combination_equality_implies_n_18 (n : ℕ) :
  (Nat.choose n 14 = Nat.choose n 4) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_n_18_l3567_356750


namespace NUMINAMATH_CALUDE_onion_basket_change_l3567_356766

/-- The net change in the number of onions in Sara's basket -/
def net_change (sara_added : ℤ) (sally_removed : ℤ) (fred_added : ℤ) : ℤ :=
  sara_added - sally_removed + fred_added

/-- Theorem stating that the net change in onions is 8 -/
theorem onion_basket_change :
  net_change 4 5 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_onion_basket_change_l3567_356766


namespace NUMINAMATH_CALUDE_widget_earnings_proof_l3567_356724

/-- Calculates the earnings per widget given the hourly wage, required widgets per week,
    work hours per week, and total weekly earnings. -/
def earnings_per_widget (hourly_wage : ℚ) (widgets_per_week : ℕ) (hours_per_week : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - (hourly_wage * hours_per_week)) / widgets_per_week

/-- Proves that the earnings per widget is $0.16 given the specific conditions. -/
theorem widget_earnings_proof :
  let hourly_wage : ℚ := 25/2  -- $12.50
  let widgets_per_week : ℕ := 1250
  let hours_per_week : ℕ := 40
  let total_earnings : ℚ := 700
  earnings_per_widget hourly_wage widgets_per_week hours_per_week total_earnings = 4/25  -- $0.16
  := by sorry

end NUMINAMATH_CALUDE_widget_earnings_proof_l3567_356724


namespace NUMINAMATH_CALUDE_f_solutions_l3567_356773

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem f_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 12 ∧ f x₂ = 12 ∧ 
  (∀ x : ℝ, f x = 12 → x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_f_solutions_l3567_356773


namespace NUMINAMATH_CALUDE_peach_difference_l3567_356721

def red_peaches : ℕ := 5
def green_peaches : ℕ := 11

theorem peach_difference : green_peaches - red_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3567_356721


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3567_356745

/-- The complex number -2i/(1+i) is equal to -1-i -/
theorem complex_fraction_equality : ((-2 * Complex.I) / (1 + Complex.I)) = (-1 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3567_356745


namespace NUMINAMATH_CALUDE_first_thrilling_thursday_l3567_356794

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to determine if a given date is a Thursday -/
def isThursday (d : Date) : Bool := sorry

/-- Function to determine if a given date is a Thrilling Thursday -/
def isThrillingThursday (d : Date) : Bool := sorry

/-- The number of days in November -/
def novemberDays : Nat := 30

/-- The start date of the school -/
def schoolStartDate : Date := ⟨2, 11⟩

/-- Theorem stating that the first Thrilling Thursday after school starts is November 30 -/
theorem first_thrilling_thursday :
  let firstThrillingThursday := Date.mk 30 11
  isThursday schoolStartDate ∧
  isThrillingThursday firstThrillingThursday ∧
  (∀ d : Date, schoolStartDate.day ≤ d.day ∧ d.day < firstThrillingThursday.day →
    ¬isThrillingThursday d) := by
  sorry

end NUMINAMATH_CALUDE_first_thrilling_thursday_l3567_356794


namespace NUMINAMATH_CALUDE_tuna_cost_theorem_l3567_356716

/-- Calculates the cost of a single can of tuna in cents -/
def tuna_cost_cents (num_cans : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) 
                    (amount_paid : ℕ) (change_received : ℕ) : ℕ :=
  let total_paid := amount_paid - change_received
  let coupon_discount := num_coupons * coupon_value
  let total_cost := total_paid * 100 + coupon_discount
  total_cost / num_cans

theorem tuna_cost_theorem : 
  tuna_cost_cents 9 5 25 2000 550 = 175 := by
  sorry

end NUMINAMATH_CALUDE_tuna_cost_theorem_l3567_356716


namespace NUMINAMATH_CALUDE_proportion_problem_l3567_356749

theorem proportion_problem (hours_per_day : ℕ) (h : hours_per_day = 24) :
  ∃ x : ℕ, (36 : ℚ) / 3 = x / (24 * hours_per_day) ∧ x = 6912 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l3567_356749


namespace NUMINAMATH_CALUDE_video_game_price_l3567_356748

theorem video_game_price (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) : 
  total_games = 16 → 
  non_working_games = 8 → 
  total_earnings = 56 → 
  total_earnings / (total_games - non_working_games) = 7 := by
sorry

end NUMINAMATH_CALUDE_video_game_price_l3567_356748


namespace NUMINAMATH_CALUDE_baseball_team_average_l3567_356758

theorem baseball_team_average (total_points : ℕ) (total_players : ℕ) 
  (high_scorers : ℕ) (high_scorer_average : ℕ) (remaining_average : ℕ) : 
  total_points = 270 → 
  total_players = 9 → 
  high_scorers = 5 → 
  high_scorer_average = 50 → 
  remaining_average = 5 → 
  total_points = high_scorers * high_scorer_average + (total_players - high_scorers) * remaining_average :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_team_average_l3567_356758


namespace NUMINAMATH_CALUDE_min_value_F_l3567_356799

/-- The function F as defined in the problem -/
def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

/-- Theorem stating that the minimum value of F(m,n) is 9/32 -/
theorem min_value_F :
  ∀ m n : ℝ, F m n ≥ 9/32 := by
  sorry

end NUMINAMATH_CALUDE_min_value_F_l3567_356799


namespace NUMINAMATH_CALUDE_spherical_distance_60N_l3567_356742

/-- The spherical distance between two points on a latitude circle --/
def spherical_distance (R : ℝ) (latitude : ℝ) (arc_length : ℝ) : ℝ :=
  sorry

/-- Theorem: Spherical distance between two points on 60°N latitude --/
theorem spherical_distance_60N (R : ℝ) (h : R > 0) :
  spherical_distance R (π / 3) (π * R / 2) = π * R / 3 :=
sorry

end NUMINAMATH_CALUDE_spherical_distance_60N_l3567_356742


namespace NUMINAMATH_CALUDE_club_size_l3567_356746

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 6

/-- The cost of a helmet in dollars -/
def helmet_cost : ℕ := glove_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := 3

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (glove_cost + helmet_cost) + cap_cost

/-- The total expenditure for all players' equipment in dollars -/
def total_expenditure : ℕ := 2968

/-- The number of players in the club -/
def num_players : ℕ := total_expenditure / player_cost

theorem club_size : num_players = 72 := by
  sorry

end NUMINAMATH_CALUDE_club_size_l3567_356746


namespace NUMINAMATH_CALUDE_f_greatest_lower_bound_l3567_356787

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem f_greatest_lower_bound :
  ∃ (k : ℝ), k = -Real.exp 2 ∧
  (∀ x > 2, f x > k) ∧
  (∀ ε > 0, ∃ x > 2, f x < k + ε) :=
sorry

end NUMINAMATH_CALUDE_f_greatest_lower_bound_l3567_356787


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3567_356701

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), 2 * (-1/2) - m * (-3) + 1 - 3*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3567_356701


namespace NUMINAMATH_CALUDE_chef_nut_purchase_l3567_356777

/-- The weight of almonds bought by the chef in kilograms -/
def almond_weight : ℝ := 0.14

/-- The weight of pecans bought by the chef in kilograms -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms -/
def total_weight : ℝ := almond_weight + pecan_weight

/-- Theorem stating that the total weight of nuts bought by the chef is 0.52 kilograms -/
theorem chef_nut_purchase : total_weight = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_chef_nut_purchase_l3567_356777


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3567_356728

/-- The eccentricity of an ellipse with equation x^2 + ky^2 = 3k (k > 0) 
    that shares a focus with the parabola y^2 = 12x is √3/2 -/
theorem ellipse_eccentricity (k : ℝ) (hk : k > 0) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 3*k}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let ellipse_focus : ℝ × ℝ := (3, 0)  -- Focus of the parabola
  ellipse_focus ∈ ellipse → -- Assuming the focus is on the ellipse
  let a := Real.sqrt (3*k)  -- Semi-major axis
  let c := 3  -- Distance from center to focus
  let e := c / a  -- Eccentricity
  e = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_ellipse_eccentricity_l3567_356728


namespace NUMINAMATH_CALUDE_zit_difference_l3567_356715

def swanson_avg : ℕ := 5
def swanson_kids : ℕ := 25
def jones_avg : ℕ := 6
def jones_kids : ℕ := 32

theorem zit_difference : 
  jones_avg * jones_kids - swanson_avg * swanson_kids = 67 := by
  sorry

end NUMINAMATH_CALUDE_zit_difference_l3567_356715


namespace NUMINAMATH_CALUDE_quadratic_intercept_l3567_356778

/-- A quadratic function with vertex (5,10) and one x-intercept at (0,0) has its other x-intercept at x = 10 -/
theorem quadratic_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 10 - a * (x - 5)^2) →  -- vertex form with vertex (5,10)
  (0^2 * a + 0 * b + c = 0) →                        -- (0,0) is an x-intercept
  (∃ x, x ≠ 0 ∧ x^2 * a + x * b + c = 0 ∧ x = 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_intercept_l3567_356778


namespace NUMINAMATH_CALUDE_cos_36_degrees_l3567_356713

theorem cos_36_degrees : Real.cos (36 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l3567_356713


namespace NUMINAMATH_CALUDE_tomato_yield_per_plant_l3567_356774

theorem tomato_yield_per_plant 
  (rows : ℕ) 
  (plants_per_row : ℕ) 
  (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : plants_per_row = 10)
  (h3 : total_yield = 6000) :
  total_yield / (rows * plants_per_row) = 20 := by
sorry

end NUMINAMATH_CALUDE_tomato_yield_per_plant_l3567_356774


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l3567_356775

def f (x : ℝ) : ℝ := x + x^3

theorem f_increasing_and_odd :
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l3567_356775


namespace NUMINAMATH_CALUDE_roe_savings_l3567_356726

def savings_problem (x : ℝ) : Prop :=
  let jan_to_jul := 7 * x
  let aug_to_nov := 4 * 15
  let december := 20
  jan_to_jul + aug_to_nov + december = 150

theorem roe_savings : ∃ x : ℝ, savings_problem x ∧ x = 10 :=
  sorry

end NUMINAMATH_CALUDE_roe_savings_l3567_356726


namespace NUMINAMATH_CALUDE_cos_double_angle_when_tan_is_one_l3567_356719

theorem cos_double_angle_when_tan_is_one (θ : Real) (h : Real.tan θ = 1) : 
  Real.cos (2 * θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_when_tan_is_one_l3567_356719


namespace NUMINAMATH_CALUDE_resort_cost_theorem_l3567_356784

def resort_problem (swimming_pool_cost : ℝ) : Prop :=
  let first_cabin_cost := swimming_pool_cost
  let second_cabin_cost := first_cabin_cost / 2
  let third_cabin_cost := second_cabin_cost / 3
  let land_cost := 4 * swimming_pool_cost
  swimming_pool_cost + first_cabin_cost + second_cabin_cost + third_cabin_cost + land_cost = 150000

theorem resort_cost_theorem :
  ∃ (swimming_pool_cost : ℝ), resort_problem swimming_pool_cost :=
sorry

end NUMINAMATH_CALUDE_resort_cost_theorem_l3567_356784


namespace NUMINAMATH_CALUDE_force_at_200000_l3567_356733

/-- Represents the gravitational force at a given distance -/
def gravitational_force (d : ℝ) : ℝ := sorry

/-- The gravitational force follows the inverse square law -/
axiom inverse_square_law (d₁ d₂ : ℝ) :
  gravitational_force d₁ * d₁^2 = gravitational_force d₂ * d₂^2

/-- The gravitational force at 5,000 miles is 500 Newtons -/
axiom force_at_5000 : gravitational_force 5000 = 500

/-- Theorem: The gravitational force at 200,000 miles is 5/16 Newtons -/
theorem force_at_200000 : gravitational_force 200000 = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_force_at_200000_l3567_356733


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3567_356796

/-- Given a geometric sequence where the fourth term is 54 and the fifth term is 162,
    prove that the first term of the sequence is 2. -/
theorem geometric_sequence_first_term
  (a : ℝ)  -- First term of the sequence
  (r : ℝ)  -- Common ratio of the sequence
  (h1 : a * r^3 = 54)  -- Fourth term is 54
  (h2 : a * r^4 = 162)  -- Fifth term is 162
  : a = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3567_356796


namespace NUMINAMATH_CALUDE_problem_statement_l3567_356722

theorem problem_statement (x y : ℝ) : 
  (|x - y| > x) → (x + y > 0) → (x > 0 ∧ y > 0) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3567_356722


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3567_356770

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * I
  let z₂ : ℂ := 4 - 7 * I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3567_356770


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3567_356751

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point through which the line passes
def P : ℝ × ℝ := (2, 0)

-- Define the possible tangent lines
def line1 (x y : ℝ) : Prop := y = 0
def line2 (x y : ℝ) : Prop := 27*x - y - 54 = 0

theorem tangent_line_equation :
  ∃ (m : ℝ), 
    (∀ x y : ℝ, y = m*(x - P.1) + P.2 → 
      (∃ t : ℝ, x = t ∧ y = f t ∧ 
        (∀ s : ℝ, s ≠ t → y - f t < m*(s - t)))) ↔ 
    (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3567_356751


namespace NUMINAMATH_CALUDE_equation_equivalence_l3567_356783

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 4 / y = 1 / 3) ↔ (9 * y / (y - 12) = x) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3567_356783


namespace NUMINAMATH_CALUDE_sqrt_588_simplification_l3567_356723

theorem sqrt_588_simplification : Real.sqrt 588 = 14 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_588_simplification_l3567_356723


namespace NUMINAMATH_CALUDE_marys_average_speed_l3567_356771

/-- Mary's round trip walk problem -/
theorem marys_average_speed (uphill_distance : ℝ) (downhill_distance : ℝ)
  (uphill_time : ℝ) (downhill_time : ℝ)
  (h1 : uphill_distance = 1.5)
  (h2 : downhill_distance = 1.5)
  (h3 : uphill_time = 45 / 60)  -- Convert 45 minutes to hours
  (h4 : downhill_time = 15 / 60)  -- Convert 15 minutes to hours
  : (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 3 := by
  sorry

#check marys_average_speed

end NUMINAMATH_CALUDE_marys_average_speed_l3567_356771


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l3567_356791

theorem chicken_wings_distribution (num_friends : ℕ) (pre_cooked : ℕ) (additional_cooked : ℕ) :
  num_friends = 3 →
  pre_cooked = 8 →
  additional_cooked = 10 →
  (pre_cooked + additional_cooked) / num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l3567_356791


namespace NUMINAMATH_CALUDE_banana_profit_calculation_grocer_profit_is_eight_dollars_l3567_356736

/-- Calculates the profit for a grocer selling bananas -/
theorem banana_profit_calculation (purchase_price : ℚ) (purchase_weight : ℚ) 
  (sale_price : ℚ) (sale_weight : ℚ) (total_weight : ℚ) : ℚ :=
  let cost_per_pound := purchase_price / purchase_weight
  let revenue_per_pound := sale_price / sale_weight
  let total_cost := cost_per_pound * total_weight
  let total_revenue := revenue_per_pound * total_weight
  let profit := total_revenue - total_cost
  profit

/-- Proves that the grocer's profit is $8.00 given the specified conditions -/
theorem grocer_profit_is_eight_dollars : 
  banana_profit_calculation (1/2) 3 1 4 96 = 8 := by
  sorry

end NUMINAMATH_CALUDE_banana_profit_calculation_grocer_profit_is_eight_dollars_l3567_356736


namespace NUMINAMATH_CALUDE_expected_digits_is_one_point_six_l3567_356767

/-- A fair 20-sided die -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The probability of rolling any specific number on the die -/
def prob (n : ℕ) : ℚ := if n ∈ icosahedralDie then 1 / 20 else 0

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum fun n => prob n * numDigits n)

theorem expected_digits_is_one_point_six :
  expectedDigits = 8/5 := by sorry

end NUMINAMATH_CALUDE_expected_digits_is_one_point_six_l3567_356767
