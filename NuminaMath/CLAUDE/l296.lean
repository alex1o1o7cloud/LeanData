import Mathlib

namespace tan_sum_product_fifteen_thirty_l296_29649

theorem tan_sum_product_fifteen_thirty : 
  Real.tan (15 * π / 180) + Real.tan (30 * π / 180) + Real.tan (15 * π / 180) * Real.tan (30 * π / 180) = 1 := by
  sorry

end tan_sum_product_fifteen_thirty_l296_29649


namespace museum_pictures_l296_29655

theorem museum_pictures (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ) (museum_pics : ℕ) : 
  zoo_pics = 15 → 
  deleted_pics = 31 → 
  remaining_pics = 2 → 
  zoo_pics + museum_pics - deleted_pics = remaining_pics → 
  museum_pics = 18 := by
sorry

end museum_pictures_l296_29655


namespace cost_per_watt_hour_is_020_l296_29612

/-- Calculates the cost per watt-hour given the number of bulbs, wattage per bulb,
    number of days, and total monthly expense. -/
def cost_per_watt_hour (num_bulbs : ℕ) (watts_per_bulb : ℕ) (days : ℕ) (total_expense : ℚ) : ℚ :=
  total_expense / (num_bulbs * watts_per_bulb * days : ℚ)

/-- Theorem stating that the cost per watt-hour is $0.20 under the given conditions. -/
theorem cost_per_watt_hour_is_020 :
  cost_per_watt_hour 40 60 30 14400 = 1/5 := by sorry

end cost_per_watt_hour_is_020_l296_29612


namespace units_digit_factorial_sum_7_l296_29638

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_7 : 
  units_digit (factorial_sum 7) = 3 := by sorry

end units_digit_factorial_sum_7_l296_29638


namespace tamika_always_wins_l296_29625

def tamika_set : Finset ℕ := {11, 12, 13}
def carlos_set : Finset ℕ := {4, 6, 7}

theorem tamika_always_wins :
  ∀ (a b : ℕ) (c d : ℕ),
    a ∈ tamika_set → b ∈ tamika_set → a ≠ b →
    c ∈ carlos_set → d ∈ carlos_set → c ≠ d →
    a * b > c * d := by
  sorry

#check tamika_always_wins

end tamika_always_wins_l296_29625


namespace cookies_milk_ratio_l296_29697

-- Define the constants from the problem
def cookies_for_recipe : ℕ := 18
def quarts_for_recipe : ℕ := 3
def pints_per_quart : ℕ := 2
def cookies_to_bake : ℕ := 9

-- Define the function to calculate pints needed
def pints_needed (cookies : ℕ) : ℚ :=
  (cookies : ℚ) * (quarts_for_recipe * pints_per_quart : ℚ) / (cookies_for_recipe : ℚ)

-- Theorem statement
theorem cookies_milk_ratio :
  pints_needed cookies_to_bake = 3 := by
  sorry

end cookies_milk_ratio_l296_29697


namespace birds_on_fence_l296_29656

theorem birds_on_fence (initial_birds : ℕ) (storks_joined : ℕ) (stork_bird_difference : ℕ) :
  initial_birds = 3 →
  storks_joined = 6 →
  stork_bird_difference = 1 →
  ∃ (birds_joined : ℕ), birds_joined = 2 ∧
    storks_joined = initial_birds + birds_joined + stork_bird_difference :=
by
  sorry

end birds_on_fence_l296_29656


namespace exists_A_all_A_digit_numbers_A_minus_1_expressible_l296_29620

/-- Represents the concatenation operation -/
def concatenate (a b : ℕ) : ℕ := sorry

/-- Checks if a number is m-expressible -/
def is_m_expressible (n m : ℕ) : Prop := sorry

/-- The main theorem to be proved -/
theorem exists_A_all_A_digit_numbers_A_minus_1_expressible :
  ∃ A : ℕ, ∀ n : ℕ, (10^(A-1) ≤ n ∧ n < 10^A) → is_m_expressible n (A-1) := by
  sorry

end exists_A_all_A_digit_numbers_A_minus_1_expressible_l296_29620


namespace square_diagonals_sum_l296_29614

theorem square_diagonals_sum (x y : ℝ) (h1 : x^2 + y^2 = 145) (h2 : x^2 - y^2 = 85) :
  x * Real.sqrt 2 + y * Real.sqrt 2 = Real.sqrt 230 + Real.sqrt 60 := by
  sorry

#check square_diagonals_sum

end square_diagonals_sum_l296_29614


namespace quadratic_function_properties_l296_29626

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c 0 = 1 →
  (∃ m : ℝ, 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c x ≥ 2 * x + m) ∧
    (∀ m' : ℝ, m' > m → ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a b c x < 2 * x + m')) →
  (∀ x : ℝ, f a b c x = x^2 - x + 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c x ≥ 2 * x + (-1)) :=
by sorry


end quadratic_function_properties_l296_29626


namespace min_packs_for_120_cans_l296_29667

/-- Represents the available pack sizes for soda cans -/
def PackSizes : List Nat := [6, 12, 24, 30]

/-- The total number of cans needed -/
def TotalCans : Nat := 120

/-- A function that checks if a combination of packs can exactly make the total number of cans -/
def canMakeTotalCans (packs : List Nat) : Bool :=
  (packs.map (fun size => size * (packs.count size))).sum = TotalCans

/-- Theorem stating that the minimum number of packs needed to buy exactly 120 cans is 4 -/
theorem min_packs_for_120_cans :
  ∃ (packs : List Nat),
    packs.all (PackSizes.contains ·) ∧
    canMakeTotalCans packs ∧
    packs.length = 4 ∧
    (∀ (other_packs : List Nat),
      other_packs.all (PackSizes.contains ·) →
      canMakeTotalCans other_packs →
      other_packs.length ≥ 4) :=
by sorry

end min_packs_for_120_cans_l296_29667


namespace max_m_value_l296_29646

def f (x : ℝ) : ℝ := |2*x + 1| + |3*x - 2|

theorem max_m_value (h : Set.Icc (-4/5 : ℝ) (6/5) = {x : ℝ | f x ≤ 5}) :
  ∃ m : ℝ, m = 2 ∧ 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m^2 - 3*m + 5) ∧
  (∀ m' : ℝ, m' > m → ∃ x : ℝ, |x - 1| + |x + 2| < m'^2 - 3*m' + 5) :=
sorry

end max_m_value_l296_29646


namespace doubly_underlined_count_l296_29666

def count_doubly_underlined (n : ℕ) : ℕ :=
  let multiples_of_6_not_4 := (n / 6 + 1) / 2
  let multiples_of_4_not_3 := 2 * (n / 4 + 1) / 3
  multiples_of_6_not_4 + multiples_of_4_not_3

theorem doubly_underlined_count :
  count_doubly_underlined 2016 = 504 := by
  sorry

end doubly_underlined_count_l296_29666


namespace remainder_problem_l296_29668

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 5 = 4)
  (h2 : n^3 % 5 = 2) : 
  n % 5 = 3 := by
  sorry

end remainder_problem_l296_29668


namespace vector_AB_after_translation_l296_29601

def point_A : ℝ × ℝ := (3, 7)
def point_B : ℝ × ℝ := (5, 2)
def vector_a : ℝ × ℝ := (1, 2)

def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

theorem vector_AB_after_translation :
  vector_AB = (2, -5) := by sorry

end vector_AB_after_translation_l296_29601


namespace wilsons_theorem_l296_29629

theorem wilsons_theorem (p : ℕ) (hp : Prime p) : (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end wilsons_theorem_l296_29629


namespace intersection_when_a_half_range_of_a_when_disjoint_l296_29643

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_when_a_half :
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

theorem range_of_a_when_disjoint (a : ℝ) :
  (A a).Nonempty → (A a ∩ B = ∅) →
  (-2 < a ∧ a ≤ -1/2) ∨ (a ≥ 2) := by sorry

end intersection_when_a_half_range_of_a_when_disjoint_l296_29643


namespace sum_of_repeating_decimals_l296_29675

def repeating_decimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  repeating_decimal 7 + repeating_decimal 5 - repeating_decimal 6 = 2 / 3 := by
  sorry

end sum_of_repeating_decimals_l296_29675


namespace min_cost_halloween_bags_l296_29607

/-- Represents the cost calculation for Halloween goodie bags --/
def halloween_bags_cost (total_students : ℕ) (vampire_count : ℕ) (pumpkin_count : ℕ) 
  (pack_size : ℕ) (pack_cost : ℕ) (individual_cost : ℕ) : ℕ := 
  let vampire_packs := vampire_count / pack_size
  let vampire_individuals := vampire_count % pack_size
  let pumpkin_packs := pumpkin_count / pack_size
  let pumpkin_individuals := pumpkin_count % pack_size
  vampire_packs * pack_cost + vampire_individuals * individual_cost +
  pumpkin_packs * pack_cost + pumpkin_individuals * individual_cost

/-- Theorem stating the minimum cost for Halloween goodie bags --/
theorem min_cost_halloween_bags : 
  halloween_bags_cost 25 11 14 5 3 1 = 17 := by
  sorry

end min_cost_halloween_bags_l296_29607


namespace jacket_cost_l296_29615

theorem jacket_cost (shorts_cost total_cost : ℚ) 
  (shorts_eq : shorts_cost = 14.28)
  (total_eq : total_cost = 19.02) :
  total_cost - shorts_cost = 4.74 := by
  sorry

end jacket_cost_l296_29615


namespace parallel_vectors_imply_sum_l296_29621

-- Define the vectors a and b
def a (m n : ℝ) : Fin 3 → ℝ := ![2*m - 3, n + 2, 3]
def b (m n : ℝ) : Fin 3 → ℝ := ![2*m + 1, 3*n - 2, 6]

-- Define parallel vectors
def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, v i = k * u i)

-- Theorem statement
theorem parallel_vectors_imply_sum (m n : ℝ) :
  parallel (a m n) (b m n) → 2*m + n = 13 := by
  sorry

end parallel_vectors_imply_sum_l296_29621


namespace disk_difference_l296_29636

/-- Given a bag of disks with blue, yellow, and green colors, prove the difference between green and blue disks -/
theorem disk_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) : 
  total = 126 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  (green_ratio * (total / (blue_ratio + yellow_ratio + green_ratio))) - 
  (blue_ratio * (total / (blue_ratio + yellow_ratio + green_ratio))) = 35 := by
  sorry

end disk_difference_l296_29636


namespace a_squared_gt_b_squared_necessity_not_sufficiency_l296_29623

theorem a_squared_gt_b_squared_necessity_not_sufficiency (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) :=
sorry

end a_squared_gt_b_squared_necessity_not_sufficiency_l296_29623


namespace ticket_sales_total_l296_29699

/-- Calculates the total amount of money collected from ticket sales -/
def totalAmountCollected (adultPrice studentPrice : ℚ) (totalTickets studentTickets : ℕ) : ℚ :=
  let adultTickets := totalTickets - studentTickets
  adultPrice * adultTickets + studentPrice * studentTickets

/-- Theorem stating that the total amount collected is $222.50 given the problem conditions -/
theorem ticket_sales_total : 
  totalAmountCollected 4 (5/2) 59 9 = 445/2 := by
  sorry

#eval totalAmountCollected 4 (5/2) 59 9

end ticket_sales_total_l296_29699


namespace remaining_watch_time_l296_29652

/-- Represents a time duration in hours and minutes -/
structure Duration where
  hours : ℕ
  minutes : ℕ

/-- Converts a Duration to minutes -/
def Duration.toMinutes (d : Duration) : ℕ :=
  d.hours * 60 + d.minutes

/-- The total duration of the series -/
def seriesDuration : Duration := { hours := 6, minutes := 0 }

/-- The durations of Hannah's watching periods -/
def watchingPeriods : List Duration := [
  { hours := 2, minutes := 24 },
  { hours := 1, minutes := 25 },
  { hours := 0, minutes := 55 }
]

/-- Theorem stating the remaining time to watch the series -/
theorem remaining_watch_time :
  seriesDuration.toMinutes - (watchingPeriods.map Duration.toMinutes).sum = 76 := by
  sorry

end remaining_watch_time_l296_29652


namespace parallel_lines_m_equal_intercepts_equal_intercept_equations_l296_29642

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def l₂ (m x y : ℝ) : Prop := x - m * y + 1 - 3 * m = 0

-- Part 1: Parallel lines
theorem parallel_lines_m (m : ℝ) : 
  (∀ x y : ℝ, l₁ x y ↔ l₂ m x y) → m = 1/2 :=
sorry

-- Part 2: Equal intercepts
theorem equal_intercepts :
  ∃ m : ℝ, m ≠ 0 ∧ 
  ((∃ y : ℝ, l₂ m 0 y) ∧ (∃ x : ℝ, l₂ m x 0)) ∧
  (∀ y : ℝ, l₂ m 0 y → y = 3 * m - 1) ∧
  (∀ x : ℝ, l₂ m x 0 → x = 3 * m - 1) →
  (m = -1 ∨ m = 1/3) :=
sorry

-- Final equations for l₂ with equal intercepts
theorem equal_intercept_equations (x y : ℝ) :
  (x + y + 4 = 0 ∨ 3 * x - y = 0) ↔
  (l₂ (-1) x y ∨ l₂ (1/3) x y) :=
sorry

end parallel_lines_m_equal_intercepts_equal_intercept_equations_l296_29642


namespace equal_roots_iff_discriminant_zero_equal_roots_h_l296_29610

/-- For a quadratic equation ax² + bx + c = 0, the discriminant is b² - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has equal roots if and only if its discriminant is zero -/
theorem equal_roots_iff_discriminant_zero (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x, a*x^2 + b*x + c = 0 ∧ (∀ y, a*y^2 + b*y + c = 0 → y = x) ↔ discriminant a b c = 0 :=
sorry

/-- The value of h for which the equation 3x² - 4x + h/3 = 0 has equal roots -/
theorem equal_roots_h : ∃! h : ℝ, discriminant 3 (-4) (h/3) = 0 ∧ h = 4 := by
  sorry

end equal_roots_iff_discriminant_zero_equal_roots_h_l296_29610


namespace cards_13_and_38_lowest_probability_l296_29677

/-- Represents the probability that a card is red side up after flips -/
def probability_red_up (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The total number of cards -/
def total_cards : ℕ := 50

/-- Theorem stating that cards 13 and 38 have the lowest probability of being red side up -/
theorem cards_13_and_38_lowest_probability :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ total_cards →
    probability_red_up 13 ≤ probability_red_up k ∧
    probability_red_up 38 ≤ probability_red_up k :=
sorry

end cards_13_and_38_lowest_probability_l296_29677


namespace acute_angle_vector_range_l296_29604

def a (x : ℝ) : ℝ × ℝ := (x, 2)
def b : ℝ × ℝ := (-3, 6)

theorem acute_angle_vector_range (x : ℝ) :
  (∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ Real.cos θ = (a x).1 * b.1 + (a x).2 * b.2 / (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  x < 4 ∧ x ≠ -1 :=
by sorry

end acute_angle_vector_range_l296_29604


namespace profit_share_difference_theorem_l296_29637

/-- Calculates the difference between profit shares of two partners given their investments and a known profit share of the third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_investment := invest_a + invest_b + invest_c
  let total_profit := b_profit * total_investment / invest_b
  let a_profit := total_profit * invest_a / total_investment
  let c_profit := total_profit * invest_c / total_investment
  c_profit - a_profit

/-- The difference between profit shares of a and c is 600 given their investments and b's profit share. -/
theorem profit_share_difference_theorem :
  profit_share_difference 8000 10000 12000 1500 = 600 := by
  sorry

end profit_share_difference_theorem_l296_29637


namespace correct_final_bill_amount_l296_29628

/-- Calculates the final bill amount after applying two late fees -/
def final_bill_amount (original_bill : ℝ) (first_fee_rate : ℝ) (second_fee_rate : ℝ) : ℝ :=
  let after_first_fee := original_bill * (1 + first_fee_rate)
  after_first_fee * (1 + second_fee_rate)

/-- Theorem stating that the final bill amount is correct -/
theorem correct_final_bill_amount :
  final_bill_amount 250 0.02 0.03 = 262.65 := by
  sorry

#eval final_bill_amount 250 0.02 0.03

end correct_final_bill_amount_l296_29628


namespace max_adjacent_squares_l296_29632

/-- A square with side length 1 -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- Two squares are adjacent if they share at least one point on their boundaries -/
def Adjacent (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ (frontier s1) ∩ (frontier s2)

/-- Two squares are non-overlapping if their interiors are disjoint -/
def NonOverlapping (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  interior s1 ∩ interior s2 = ∅

/-- A configuration of squares adjacent to a given square -/
def AdjacentSquares (n : ℕ) : Prop :=
  ∃ (squares : Fin n → Set (ℝ × ℝ)),
    (∀ i, squares i = UnitSquare) ∧
    (∀ i, Adjacent (squares i) UnitSquare) ∧
    (∀ i j, i ≠ j → NonOverlapping (squares i) (squares j))

/-- The maximum number of non-overlapping unit squares that can be placed adjacent to a given unit square is 8 -/
theorem max_adjacent_squares :
  (∀ n, AdjacentSquares n → n ≤ 8) ∧ AdjacentSquares 8 := by sorry

end max_adjacent_squares_l296_29632


namespace starters_combination_l296_29698

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def required_quadruplets : ℕ := 3

-- Define the function to calculate the number of ways to choose the starters
def choose_starters (total : ℕ) (quad : ℕ) (starters : ℕ) (req_quad : ℕ) : ℕ :=
  (Nat.choose quad req_quad) * (Nat.choose (total - quad) (starters - req_quad))

-- Theorem statement
theorem starters_combination : 
  choose_starters total_players num_quadruplets num_starters required_quadruplets = 4004 := by
  sorry

end starters_combination_l296_29698


namespace guess_two_digit_number_l296_29644

/-- A two-digit number is between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem guess_two_digit_number (x : ℕ) (h : TwoDigitNumber x) :
  (2 * x + 5) * 5 = 715 → x = 69 := by
  sorry

end guess_two_digit_number_l296_29644


namespace bottles_ratio_l296_29630

/-- The number of bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of bottles Donald drinks per day -/
def donald_bottles : ℕ := 9

/-- Donald drinks more than twice the number of bottles Paul drinks -/
axiom donald_drinks_more : donald_bottles > 2 * paul_bottles

/-- The ratio of bottles Donald drinks to bottles Paul drinks is 3:1 -/
theorem bottles_ratio : (donald_bottles : ℚ) / paul_bottles = 3 := by
  sorry

end bottles_ratio_l296_29630


namespace quadratic_solution_l296_29639

/-- Given nonzero real numbers c and d such that 2x^2 + cx + d = 0 has solutions 2c and 2d,
    prove that c = 1/2 and d = -5/8 -/
theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : ∀ x, 2 * x^2 + c * x + d = 0 ↔ x = 2 * c ∨ x = 2 * d) :
  c = 1/2 ∧ d = -5/8 := by sorry

end quadratic_solution_l296_29639


namespace halfway_fraction_l296_29634

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by sorry

end halfway_fraction_l296_29634


namespace unique_number_satisfying_condition_l296_29603

theorem unique_number_satisfying_condition : ∃! x : ℚ, ((x / 3) * 24) - 7 = 41 := by
  sorry

end unique_number_satisfying_condition_l296_29603


namespace find_x_value_l296_29648

theorem find_x_value (x y z : ℝ) 
  (h1 : x ≠ 0)
  (h2 : x / 3 = z + 2 * y^2)
  (h3 : x / 6 = 3 * z - y) :
  x = 168 := by
  sorry

end find_x_value_l296_29648


namespace remainder_zero_mod_eight_l296_29654

theorem remainder_zero_mod_eight :
  (71^7 - 73^10) * (73^5 + 71^3) ≡ 0 [ZMOD 8] := by
sorry

end remainder_zero_mod_eight_l296_29654


namespace line_plane_perpendicular_parallel_l296_29653

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_perpendicular_parallel 
  (m n : Line) (α β : Plane) : 
  (perpendicularLP m α ∧ perpendicularLP n β ∧ perpendicular m n → perpendicularPP α β) ∧
  (perpendicularLP m α ∧ parallelLP n β ∧ parallel m n → perpendicularPP α β) :=
sorry

end line_plane_perpendicular_parallel_l296_29653


namespace initial_number_proof_l296_29692

theorem initial_number_proof (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end initial_number_proof_l296_29692


namespace slope_range_l296_29689

theorem slope_range (α : Real) (h : π/3 < α ∧ α < 5*π/6) :
  ∃ k : Real, (k < -Real.sqrt 3 / 3 ∨ k > Real.sqrt 3) ∧ k = Real.tan α :=
by
  sorry

end slope_range_l296_29689


namespace expression_evaluation_l296_29640

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(x - y) = -11/4 := by
  sorry

end expression_evaluation_l296_29640


namespace second_strategy_more_economical_l296_29650

/-- Proves that the second purchasing strategy (constant money spent) is more economical than
    the first strategy (constant quantity purchased) for two purchases of the same item. -/
theorem second_strategy_more_economical (p₁ p₂ x y : ℝ) 
    (hp₁ : p₁ > 0) (hp₂ : p₂ > 0) (hx : x > 0) (hy : y > 0) :
  (2 * p₁ * p₂) / (p₁ + p₂) ≤ (p₁ + p₂) / 2 := by
  sorry

#check second_strategy_more_economical

end second_strategy_more_economical_l296_29650


namespace soda_count_l296_29661

/-- Proves that given 2 sandwiches at $2.49 each and some sodas at $1.87 each,
    if the total cost is $12.46, then the number of sodas purchased is 4. -/
theorem soda_count (sandwich_cost soda_cost total_cost : ℚ) (sandwich_count : ℕ) :
  sandwich_cost = 249/100 →
  soda_cost = 187/100 →
  total_cost = 1246/100 →
  sandwich_count = 2 →
  ∃ (soda_count : ℕ), soda_count = 4 ∧
    sandwich_count * sandwich_cost + soda_count * soda_cost = total_cost :=
by sorry

end soda_count_l296_29661


namespace sack_of_rice_weight_l296_29657

theorem sack_of_rice_weight (cost : ℝ) (price_per_kg : ℝ) (profit : ℝ) (weight : ℝ) : 
  cost = 50 → 
  price_per_kg = 1.20 → 
  profit = 10 → 
  price_per_kg * weight = cost + profit → 
  weight = 50 := by
sorry

end sack_of_rice_weight_l296_29657


namespace vector_inequality_l296_29662

variable (V : Type*) [NormedAddCommGroup V]

theorem vector_inequality (v w : V) : 
  ‖v‖ + ‖w‖ ≤ ‖v + w‖ + ‖v - w‖ := by
  sorry

end vector_inequality_l296_29662


namespace series_sum_l296_29687

theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series_term (n : ℕ) := 1 / (((n - 1) * a - (n - 2) * b) * (n * a - (n - 1) * b))
  let series_sum := ∑' n, series_term n
  series_sum = 1 / ((a - b) * b) := by sorry

end series_sum_l296_29687


namespace combined_sticker_count_l296_29693

theorem combined_sticker_count 
  (june_initial : ℕ) 
  (bonnie_initial : ℕ) 
  (birthday_gift : ℕ) : 
  june_initial + bonnie_initial + 2 * birthday_gift = 
    (june_initial + birthday_gift) + (bonnie_initial + birthday_gift) := by
  sorry

end combined_sticker_count_l296_29693


namespace arithmetic_sequence_property_l296_29611

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_property_l296_29611


namespace stating_wrapping_paper_area_theorem_l296_29606

/-- Represents a rectangular box. -/
structure Box where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- Calculates the area of the square wrapping paper needed for a given box. -/
def wrappingPaperArea (box : Box) : ℝ :=
  (box.a + 2 * box.h) ^ 2

/-- 
Theorem stating that the area of the square wrapping paper for a rectangular box
with base dimensions a × b and height h, wrapped as described in the problem,
is (a + 2h)².
-/
theorem wrapping_paper_area_theorem (box : Box) :
  wrappingPaperArea box = (box.a + 2 * box.h) ^ 2 := by
  sorry

end stating_wrapping_paper_area_theorem_l296_29606


namespace expand_product_l296_29669

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 7) = 2 * x^2 + 8 * x - 42 := by
  sorry

end expand_product_l296_29669


namespace two_digit_multiplication_l296_29616

theorem two_digit_multiplication (a b c : ℕ) : 
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c := by
  sorry

end two_digit_multiplication_l296_29616


namespace geometry_propositions_l296_29622

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelPL : Plane → Line → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (((intersect α β = m) ∧ (contains α n) ∧ (perpendicular n m)) → (perpendicularPP α β)) ∧
  ((perpendicularPL α m) ∧ (perpendicularPL β m) → (parallelPP α β)) ∧
  ((perpendicularPL α m) ∧ (perpendicularPL β n) ∧ (perpendicular m n) → (perpendicularPP α β)) ∧
  (∃ (m n : Line) (α β : Plane), (parallelPL α m) ∧ (parallelPL β n) ∧ (parallel m n) ∧ ¬(parallelPP α β)) :=
by sorry

end geometry_propositions_l296_29622


namespace plane_points_distance_l296_29684

theorem plane_points_distance (n : ℕ) (P : Fin n → ℝ × ℝ) (Q : ℝ × ℝ) 
  (h_n : n ≥ 12)
  (h_distinct : ∀ i j, i ≠ j → P i ≠ P j ∧ P i ≠ Q) :
  ∃ i : Fin n, ∃ S : Finset (Fin n), 
    S.card ≥ (n / 6 : ℕ) - 1 ∧ 
    (∀ j ∈ S, j ≠ i → dist (P j) (P i) < dist (P i) Q) :=
by sorry

end plane_points_distance_l296_29684


namespace circles_tangent_radii_product_l296_29624

/-- Given two circles in a plane with radii r₁ and r₂, and distance d between their centers,
    if their common external tangent has length 2017 and their common internal tangent has length 2009,
    then the product of their radii is 8052. -/
theorem circles_tangent_radii_product (r₁ r₂ d : ℝ) 
  (h_external : d^2 - (r₁ + r₂)^2 = 2017^2)
  (h_internal : d^2 - (r₁ - r₂)^2 = 2009^2) :
  r₁ * r₂ = 8052 := by
  sorry

end circles_tangent_radii_product_l296_29624


namespace bricklayer_wage_is_44_l296_29686

/-- Represents the hourly wage of a worker -/
structure HourlyWage where
  amount : ℝ
  nonneg : amount ≥ 0

/-- Represents the total hours worked by both workers -/
def total_hours : ℝ := 90

/-- Represents the hourly wage of the electrician -/
def electrician_wage : HourlyWage := ⟨16, by norm_num⟩

/-- Represents the total payment for both workers -/
def total_payment : ℝ := 1350

/-- Represents the hours worked by each worker -/
def individual_hours : ℝ := 22.5

/-- Theorem stating that the bricklayer's hourly wage is $44 -/
theorem bricklayer_wage_is_44 :
  ∃ (bricklayer_wage : HourlyWage),
    bricklayer_wage.amount = 44 ∧
    individual_hours * (bricklayer_wage.amount + electrician_wage.amount) = total_payment ∧
    2 * individual_hours = total_hours :=
by sorry


end bricklayer_wage_is_44_l296_29686


namespace exponent_calculation_l296_29688

theorem exponent_calculation (a : ℝ) : a^3 * (a^3)^2 = a^9 := by
  sorry

end exponent_calculation_l296_29688


namespace tan_alpha_minus_beta_equals_one_l296_29613

theorem tan_alpha_minus_beta_equals_one (α β : Real) 
  (h : Real.tan β = (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α)) : 
  Real.tan (α - β) = 1 := by
  sorry

end tan_alpha_minus_beta_equals_one_l296_29613


namespace transform_f_to_g_l296_29618

def f (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4
def g (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4

theorem transform_f_to_g : 
  ∀ x : ℝ, g x = f (x + 6) - 8 := by sorry

end transform_f_to_g_l296_29618


namespace unique_prime_squared_plus_minus_six_prime_l296_29679

theorem unique_prime_squared_plus_minus_six_prime :
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (p^2 - 6) ∧ Nat.Prime (p^2 + 6) :=
by
  -- The proof goes here
  sorry

end unique_prime_squared_plus_minus_six_prime_l296_29679


namespace exists_valid_matrix_l296_29678

def is_valid_matrix (M : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  (∀ i j, M i j ≠ 0) ∧
  (∀ i j, i + 1 < 4 → j + 1 < 4 →
    M i j + M (i + 1) j + M i (j + 1) + M (i + 1) (j + 1) = 0) ∧
  (∀ i j, i + 2 < 4 → j + 2 < 4 →
    M i j + M (i + 2) j + M i (j + 2) + M (i + 2) (j + 2) = 0) ∧
  (M 0 0 + M 0 3 + M 3 0 + M 3 3 = 0)

theorem exists_valid_matrix : ∃ M : Matrix (Fin 4) (Fin 4) ℤ, is_valid_matrix M := by
  sorry

end exists_valid_matrix_l296_29678


namespace ladder_problem_l296_29671

theorem ladder_problem (ladder_length height_on_wall : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height_on_wall = 12) :
  ∃ (distance_from_wall : ℝ), 
    distance_from_wall ^ 2 + height_on_wall ^ 2 = ladder_length ^ 2 ∧ 
    distance_from_wall = 5 := by
  sorry

end ladder_problem_l296_29671


namespace prime_representation_l296_29633

theorem prime_representation (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  (p % 8 = 1 ↔ ∃ x y : ℤ, p = x^2 + 16*y^2) ∧
  (p % 8 = 5 ↔ ∃ x y : ℤ, p = 4*x^2 + 4*x*y + 5*y^2) :=
by sorry

end prime_representation_l296_29633


namespace min_value_expression_l296_29608

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 8*x*y + 4*y^2 + 4*z^2 ≥ 384 := by
sorry

end min_value_expression_l296_29608


namespace hillary_saturday_reading_l296_29635

/-- Calculates the number of minutes read on Saturday given the total assignment time and time read on Friday and Sunday. -/
def minutes_read_saturday (total_assignment : ℕ) (friday_reading : ℕ) (sunday_reading : ℕ) : ℕ :=
  total_assignment - (friday_reading + sunday_reading)

/-- Theorem stating that given the specific conditions of Hillary's reading assignment, she read for 28 minutes on Saturday. -/
theorem hillary_saturday_reading :
  minutes_read_saturday 60 16 16 = 28 := by
  sorry

end hillary_saturday_reading_l296_29635


namespace binomial_coefficient_equality_l296_29694

theorem binomial_coefficient_equality (k : ℕ) : 
  (Nat.choose 18 k = Nat.choose 18 (2 * k - 3)) ↔ (k = 3 ∨ k = 7) := by
  sorry

end binomial_coefficient_equality_l296_29694


namespace sum_remainder_mod_nine_l296_29690

theorem sum_remainder_mod_nine : (8243 + 8244 + 8245 + 8246) % 9 = 7 := by
  sorry

end sum_remainder_mod_nine_l296_29690


namespace f_at_7_equals_3_l296_29670

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x + 5

-- State the theorem
theorem f_at_7_equals_3 (p q b : ℝ) :
  (f p q (-7) = Real.sqrt 2 * b + 1) →
  f p q 7 = 3 := by
  sorry

end f_at_7_equals_3_l296_29670


namespace spinster_count_l296_29695

theorem spinster_count : 
  ∀ (spinsters cats : ℕ), 
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 →
    cats = spinsters + 35 →
    spinsters = 14 := by
  sorry

end spinster_count_l296_29695


namespace smallest_valid_seating_l296_29631

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if a seating arrangement is valid -/
def is_valid_seating (s : CircularSeating) : Prop :=
  s.seated_people > 0 ∧ 
  s.seated_people ≤ s.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < s.total_chairs → 
    ∃ (adjacent : ℕ), adjacent < s.total_chairs ∧ 
      (new_seat + 1) % s.total_chairs = adjacent ∨ 
      (new_seat + s.total_chairs - 1) % s.total_chairs = adjacent

/-- The main theorem to prove -/
theorem smallest_valid_seating :
  ∀ (s : CircularSeating), 
    s.total_chairs = 72 → 
    (is_valid_seating s ↔ s.seated_people ≥ 18) :=
by sorry

end smallest_valid_seating_l296_29631


namespace alligator_walking_time_l296_29660

/-- The combined walking time of alligators given Paul's initial journey time and additional return time -/
theorem alligator_walking_time (initial_time return_additional_time : ℕ) :
  initial_time = 4 ∧ return_additional_time = 2 →
  initial_time + (initial_time + return_additional_time) = 10 := by
  sorry

#check alligator_walking_time

end alligator_walking_time_l296_29660


namespace toms_journey_ratio_l296_29658

/-- Proves that the ratio of running time to swimming time is 1:2 given the conditions of Tom's journey --/
theorem toms_journey_ratio (swim_speed swim_time run_speed total_distance : ℝ) : 
  swim_speed = 2 →
  swim_time = 2 →
  run_speed = 4 * swim_speed →
  total_distance = 12 →
  total_distance = swim_speed * swim_time + run_speed * (total_distance - swim_speed * swim_time) / run_speed →
  (total_distance - swim_speed * swim_time) / run_speed / swim_time = 1 / 2 := by
sorry


end toms_journey_ratio_l296_29658


namespace smallest_nonzero_place_12000_l296_29651

/-- The smallest place value with a non-zero digit in 12000 is the hundreds place -/
theorem smallest_nonzero_place_12000 : 
  ∀ n : ℕ, n > 0 ∧ n < 1000 → (12000 / 10^n) % 10 = 0 :=
by sorry

end smallest_nonzero_place_12000_l296_29651


namespace valid_a_values_l296_29673

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem valid_a_values (a : ℝ) : 
  (A a ⊇ B a) ↔ (a = -1 ∨ a = 2) :=
sorry

end valid_a_values_l296_29673


namespace arccos_sin_eight_l296_29674

-- Define the problem statement
theorem arccos_sin_eight : 
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 := by
  sorry

end arccos_sin_eight_l296_29674


namespace distance_representation_l296_29627

theorem distance_representation (A B : ℝ) (hA : A = 3) (hB : B = -2) :
  |A - B| = |3 - (-2)| := by sorry

end distance_representation_l296_29627


namespace x_equals_one_l296_29696

theorem x_equals_one (y : ℝ) (a : ℝ) (x : ℝ) 
  (h1 : x + a * y = 10) 
  (h2 : y = 3) : 
  x = 1 := by
sorry

end x_equals_one_l296_29696


namespace quadratic_inequality_solution_set_l296_29602

theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x, x^2 - (a+1)*x + b ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3) → a + b = -14 :=
by sorry

end quadratic_inequality_solution_set_l296_29602


namespace carries_box_capacity_l296_29619

/-- Represents a rectangular box with height, width, and length -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box -/
def Box.volume (b : Box) : ℝ := b.height * b.width * b.length

/-- Represents the number of jellybeans a box can hold -/
def jellybeanCapacity (b : Box) (density : ℝ) : ℝ := b.volume * density

/-- Theorem: Carrie's box capacity given Bert's box capacity -/
theorem carries_box_capacity
  (bert_box : Box)
  (bert_capacity : ℝ)
  (density : ℝ)
  (h1 : jellybeanCapacity bert_box density = bert_capacity)
  (h2 : bert_capacity = 150)
  (carrie_box : Box)
  (h3 : carrie_box.height = 3 * bert_box.height)
  (h4 : carrie_box.width = 2 * bert_box.width)
  (h5 : carrie_box.length = 4 * bert_box.length) :
  jellybeanCapacity carrie_box density = 3600 := by
  sorry

end carries_box_capacity_l296_29619


namespace complement_A_complement_A_intersect_B_l296_29641

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | 2 * x + 4 < 0}

-- Define set B
def B : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- Theorem for the complement of A with respect to U
theorem complement_A : (U \ A) = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the complement of (A ∩ B) with respect to U
theorem complement_A_intersect_B : (U \ (A ∩ B)) = {x : ℝ | x < -3 ∨ (-2 ≤ x ∧ x ≤ 4)} := by sorry

end complement_A_complement_A_intersect_B_l296_29641


namespace solve_for_y_l296_29691

theorem solve_for_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 := by
  sorry

end solve_for_y_l296_29691


namespace subtracted_value_l296_29683

theorem subtracted_value (chosen_number : ℕ) (final_result : ℕ) : 
  chosen_number = 63 → final_result = 110 → 
  (chosen_number * 4 - final_result) = 142 := by
sorry

end subtracted_value_l296_29683


namespace perpendicular_line_correct_parallel_lines_correct_l296_29609

-- Define the given line l
def line_l (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 2)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + 2 * y - 7 = 0

-- Define the parallel lines
def parallel_line_1 (x y : ℝ) : Prop := 2 * x - y + 6 = 0
def parallel_line_2 (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Theorem for the perpendicular line
theorem perpendicular_line_correct :
  (perp_line point_A.1 point_A.2) ∧
  (∀ x y : ℝ, line_l x y → (x - point_A.1) * 1 + (y - point_A.2) * 2 = 0) :=
sorry

-- Theorem for the parallel lines
theorem parallel_lines_correct :
  (∀ x y : ℝ, (parallel_line_1 x y ∨ parallel_line_2 x y) →
    (abs (6 - 1) / Real.sqrt (2^2 + 1) = Real.sqrt 5 ∨
     abs (-4 - 1) / Real.sqrt (2^2 + 1) = Real.sqrt 5)) ∧
  (∀ x y : ℝ, line_l x y → (2 * 1 + 1 * 1 = 2 * 1 + 1 * 1)) :=
sorry

end perpendicular_line_correct_parallel_lines_correct_l296_29609


namespace original_paint_intensity_l296_29664

/-- 
Given a paint mixture where 20% of the original paint is replaced with a 25% solution,
resulting in a mixture with 45% intensity, prove that the original paint intensity was 50%.
-/
theorem original_paint_intensity 
  (original_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (replacement_solution_intensity : ℝ) 
  (final_intensity : ℝ) : 
  replaced_fraction = 0.2 →
  replacement_solution_intensity = 25 →
  final_intensity = 45 →
  (1 - replaced_fraction) * original_intensity + 
    replaced_fraction * replacement_solution_intensity = final_intensity →
  original_intensity = 50 := by
sorry

end original_paint_intensity_l296_29664


namespace chord_length_line_circle_intersection_l296_29645

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection : 
  ∃ (A B : ℝ × ℝ),
    (A.1 + A.2 = 2) ∧ 
    (B.1 + B.2 = 2) ∧ 
    (A.1^2 + A.2^2 = 4) ∧ 
    (B.1^2 + B.2^2 = 4) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) := by
  sorry

end chord_length_line_circle_intersection_l296_29645


namespace system_solution_l296_29680

theorem system_solution (a b c : ℝ) : 
  (∀ x y, a * x + b * y = 2 ∧ c * x - 7 * y = 8) →
  (a * 3 + b * (-2) = 2 ∧ c * 3 - 7 * (-2) = 8) →
  (a * (-2) + b * 2 = 2) →
  (a = 4 ∧ b = 5 ∧ c = -2) := by
sorry

end system_solution_l296_29680


namespace olympic_volunteer_allocation_l296_29676

theorem olympic_volunteer_allocation :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 4  -- number of projects
  let allocations : ℕ := (n.choose 2) * (k.factorial)
  allocations = 240 :=
by sorry

end olympic_volunteer_allocation_l296_29676


namespace paint_area_is_127_l296_29681

/-- Calculates the area to be painted on a wall with two windows. -/
def areaToPaint (wallHeight wallLength window1Height window1Width window2Height window2Width : ℝ) : ℝ :=
  wallHeight * wallLength - (window1Height * window1Width + window2Height * window2Width)

/-- Proves that the area to be painted is 127 square feet given the specified dimensions. -/
theorem paint_area_is_127 :
  areaToPaint 10 15 3 5 2 4 = 127 := by
  sorry

#eval areaToPaint 10 15 3 5 2 4

end paint_area_is_127_l296_29681


namespace solve_equation_l296_29663

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end solve_equation_l296_29663


namespace simplify_complex_fraction_l296_29685

theorem simplify_complex_fraction : 
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40) := by
  sorry

end simplify_complex_fraction_l296_29685


namespace det_B_squared_minus_3B_l296_29682

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det ((B ^ 2) - 3 • B) = 88 := by sorry

end det_B_squared_minus_3B_l296_29682


namespace moray_eel_eats_twenty_l296_29600

/-- The number of guppies eaten by a moray eel per day, given the total number of guppies needed,
    the number of betta fish, and the number of guppies eaten by each betta fish per day. -/
def moray_eel_guppies (total_guppies : ℕ) (num_betta : ℕ) (betta_guppies : ℕ) : ℕ :=
  total_guppies - (num_betta * betta_guppies)

/-- Theorem stating that the number of guppies eaten by the moray eel is 20,
    given the conditions in the problem. -/
theorem moray_eel_eats_twenty :
  moray_eel_guppies 55 5 7 = 20 := by
  sorry

end moray_eel_eats_twenty_l296_29600


namespace geometric_sequence_problem_l296_29672

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 5 * a 7 = 2)
  (h_sum : a 2 + a 10 = 3) :
  a 12 / a 4 = 2 ∨ a 12 / a 4 = 1/2 :=
sorry

end geometric_sequence_problem_l296_29672


namespace ellipse_chord_slopes_product_l296_29647

/-- Theorem: Product of slopes for chord through center of ellipse -/
theorem ellipse_chord_slopes_product (a b x₀ y₀ x₁ y₁ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hP : x₀^2 / a^2 + y₀^2 / b^2 = 1)  -- P is on the ellipse
  (hP₁ : x₁^2 / a^2 + y₁^2 / b^2 = 1)  -- P₁ is on the ellipse
  (hP₂ : (-x₁)^2 / a^2 + (-y₁)^2 / b^2 = 1)  -- P₂ is on the ellipse
  (k₁ : ℝ) (hk₁ : k₁ = (y₀ - y₁) / (x₀ - x₁))  -- Slope of PP₁
  (k₂ : ℝ) (hk₂ : k₂ = (y₀ - (-y₁)) / (x₀ - (-x₁)))  -- Slope of PP₂
  : k₁ * k₂ = -b^2 / a^2 := by
  sorry

end ellipse_chord_slopes_product_l296_29647


namespace stratified_sample_probability_l296_29665

/-- Represents the number of classes selected from each grade -/
structure GradeSelection where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- The probability of selecting two classes from the same grade in a stratified sample -/
def probability_same_grade (selection : GradeSelection) : Rat :=
  let total_combinations := (selection.grade1 + selection.grade2 + selection.grade3).choose 2
  let same_grade_combinations := selection.grade1.choose 2
  same_grade_combinations / total_combinations

theorem stratified_sample_probability 
  (selection : GradeSelection)
  (h_ratio : selection.grade1 = 3 ∧ selection.grade2 = 2 ∧ selection.grade3 = 1) :
  probability_same_grade selection = 1/5 := by
  sorry

end stratified_sample_probability_l296_29665


namespace sam_march_aug_earnings_l296_29605

/-- Represents Sam's work and financial situation --/
structure SamFinances where
  hourly_rate : ℝ
  march_aug_hours : ℕ := 23
  sept_feb_hours : ℕ := 8
  additional_hours : ℕ := 16
  console_cost : ℕ := 600
  car_repair_cost : ℕ := 340

/-- Theorem stating Sam's earnings from March to August --/
theorem sam_march_aug_earnings (sam : SamFinances) :
  sam.hourly_rate * sam.march_aug_hours = 460 :=
by
  have total_needed : ℝ := sam.console_cost + sam.car_repair_cost
  have total_hours : ℕ := sam.march_aug_hours + sam.sept_feb_hours + sam.additional_hours
  have : sam.hourly_rate * total_hours = total_needed :=
    sorry
  sorry

#check sam_march_aug_earnings

end sam_march_aug_earnings_l296_29605


namespace alice_bracelet_profit_l296_29659

/-- Alice's bracelet sale profit calculation -/
theorem alice_bracelet_profit :
  ∀ (total_bracelets : ℕ) 
    (material_cost given_away price : ℚ),
  total_bracelets = 52 →
  material_cost = 3 →
  given_away = 8 →
  price = 1/4 →
  (total_bracelets - given_away : ℚ) * price - material_cost = 8 :=
by
  sorry

end alice_bracelet_profit_l296_29659


namespace flower_beds_fraction_l296_29617

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure YardWithFlowerBeds where
  /-- Length of the shorter parallel side of the trapezoid -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoid -/
  long_side : ℝ
  /-- Assumption that the short side is 20 meters -/
  short_side_eq : short_side = 20
  /-- Assumption that the long side is 30 meters -/
  long_side_eq : long_side = 30

/-- The fraction of the yard occupied by the flower beds is 1/6 -/
theorem flower_beds_fraction (yard : YardWithFlowerBeds) : 
  (yard.long_side - yard.short_side)^2 / (4 * yard.long_side * (yard.long_side - yard.short_side)) = 1/6 := by
  sorry

end flower_beds_fraction_l296_29617
