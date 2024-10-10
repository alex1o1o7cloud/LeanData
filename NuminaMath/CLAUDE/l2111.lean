import Mathlib

namespace clippings_per_friend_l2111_211174

theorem clippings_per_friend 
  (num_friends : ℕ) 
  (total_glue_drops : ℕ) 
  (glue_drops_per_clipping : ℕ) 
  (h1 : num_friends = 7)
  (h2 : total_glue_drops = 126)
  (h3 : glue_drops_per_clipping = 6) :
  (total_glue_drops / glue_drops_per_clipping) / num_friends = 3 :=
by sorry

end clippings_per_friend_l2111_211174


namespace halloween_candy_problem_l2111_211165

theorem halloween_candy_problem (eaten : ℕ) (pile_size : ℕ) (num_piles : ℕ) :
  eaten = 30 →
  pile_size = 8 →
  num_piles = 6 →
  eaten + (pile_size * num_piles) = 78 :=
by sorry

end halloween_candy_problem_l2111_211165


namespace boat_speed_in_still_water_l2111_211172

/-- The speed of a boat in still water, given its downstream travel time and distance, and the stream's speed. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 7)
  (h3 : downstream_distance = 147) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 16 :=
by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l2111_211172


namespace square_to_circle_area_ratio_l2111_211185

theorem square_to_circle_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / (π * s^2) = 1 / π :=
by sorry

end square_to_circle_area_ratio_l2111_211185


namespace sqrt_ratio_implies_sum_ratio_l2111_211159

theorem sqrt_ratio_implies_sum_ratio (x y : ℝ) (h : x > 0) (k : y > 0) :
  (Real.sqrt x / Real.sqrt y = 5) → ((x + y) / (2 * y) = 13) := by
  sorry

end sqrt_ratio_implies_sum_ratio_l2111_211159


namespace white_go_stones_l2111_211177

theorem white_go_stones (total : ℕ) (difference : ℕ) (white : ℕ) (black : ℕ) : 
  total = 120 →
  difference = 36 →
  white = black + difference →
  total = white + black →
  white = 78 := by
sorry

end white_go_stones_l2111_211177


namespace rod_cutting_l2111_211158

/-- Given a rod of length 38.25 meters that can be cut into 45 pieces,
    prove that each piece is 85 centimeters long. -/
theorem rod_cutting (rod_length : Real) (num_pieces : Nat) :
  rod_length = 38.25 ∧ num_pieces = 45 →
  (rod_length / num_pieces) * 100 = 85 := by
  sorry

end rod_cutting_l2111_211158


namespace cos_alpha_value_l2111_211187

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.cos (α + Real.pi / 3) = -2/3) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 := by
  sorry

end cos_alpha_value_l2111_211187


namespace table_people_count_l2111_211110

/-- The number of seeds taken by n people in the first round -/
def first_round_seeds (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of seeds taken by n people in the second round -/
def second_round_seeds (n : ℕ) : ℕ := first_round_seeds n + n^2

/-- The difference in seeds taken between the second and first rounds -/
def seed_difference (n : ℕ) : ℕ := second_round_seeds n - first_round_seeds n

theorem table_people_count : 
  ∃ n : ℕ, n > 0 ∧ seed_difference n = 100 ∧ 
  (∀ m : ℕ, m > 0 → seed_difference m = 100 → m = n) :=
sorry

end table_people_count_l2111_211110


namespace difference_divisible_by_99_l2111_211131

/-- Given a natural number, returns the number of its digits -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Given a natural number, returns the number formed by reversing its digits -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number with an odd number of digits, 
    the difference between the number and its reverse is divisible by 99 -/
theorem difference_divisible_by_99 (n : ℕ) (h : Odd (numDigits n)) :
  99 ∣ (n - reverseDigits n) := by sorry

end difference_divisible_by_99_l2111_211131


namespace area_of_extended_quadrilateral_l2111_211180

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EFext : ℝ
  FGext : ℝ
  GHext : ℝ
  HEext : ℝ
  area : ℝ

/-- The area of quadrilateral E'F'G'H' is 57 -/
theorem area_of_extended_quadrilateral (q : ExtendedQuadrilateral) 
  (h1 : q.EF = 5)
  (h2 : q.EFext = 5)
  (h3 : q.FG = 6)
  (h4 : q.FGext = 6)
  (h5 : q.GH = 7)
  (h6 : q.GHext = 7)
  (h7 : q.HE = 10)
  (h8 : q.HEext = 10)
  (h9 : q.area = 15)
  (h10 : q.EF = q.EFext) -- Isosceles triangle condition
  : (q.area + 2 * q.area + 12 : ℝ) = 57 := by
  sorry

end area_of_extended_quadrilateral_l2111_211180


namespace inequality_implication_condition_l2111_211121

theorem inequality_implication_condition :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  ¬(∀ x : ℝ, |x - 1| < 2 → x * (x - 3) < 0) :=
by sorry

end inequality_implication_condition_l2111_211121


namespace max_value_implies_a_equals_four_l2111_211102

theorem max_value_implies_a_equals_four :
  ∀ a b c : ℕ,
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (∀ x y z : ℕ, 
    x ∈ ({1, 2, 4} : Set ℕ) →
    y ∈ ({1, 2, 4} : Set ℕ) →
    z ∈ ({1, 2, 4} : Set ℕ) →
    x ≠ y → y ≠ z → x ≠ z →
    (a / 2 : ℚ) / (b / c : ℚ) ≥ (x / 2 : ℚ) / (y / z : ℚ)) →
  (a / 2 : ℚ) / (b / c : ℚ) = 4 →
  a = 4 := by
  sorry

end max_value_implies_a_equals_four_l2111_211102


namespace exists_recurrence_sequence_l2111_211171

-- Define the sequence type
def RecurrenceSequence (x y : ℝ) := ℕ → ℝ

-- Define the recurrence relation property
def SatisfiesRecurrence (a : RecurrenceSequence x y) : Prop :=
  ∀ n : ℕ, a (n + 2) = x * a (n + 1) + y * a n

-- Define the boundedness property
def SatisfiesBoundedness (a : RecurrenceSequence x y) : Prop :=
  ∀ r : ℝ, r > 0 → ∃ i j : ℕ, i > 0 ∧ j > 0 ∧ |a i| < r ∧ r < |a j|

-- Define the non-zero property
def IsNonZero (a : RecurrenceSequence x y) : Prop :=
  ∀ n : ℕ, a n ≠ 0

-- Main theorem
theorem exists_recurrence_sequence :
  ∃ x y : ℝ, ∃ a : RecurrenceSequence x y,
    SatisfiesRecurrence a ∧ SatisfiesBoundedness a ∧ IsNonZero a := by
  sorry

end exists_recurrence_sequence_l2111_211171


namespace total_amount_is_117_l2111_211150

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ  -- Share of x in rupees
  y : ℝ  -- Share of y in rupees
  z : ℝ  -- Share of z in rupees

/-- The conditions of the money distribution problem -/
def satisfies_conditions (d : MoneyDistribution) : Prop :=
  d.y = 27 ∧                  -- y's share is 27 rupees
  d.y = 0.45 * d.x ∧          -- y gets 45 paisa for each rupee x gets
  d.z = 0.5 * d.x             -- z gets 50 paisa for each rupee x gets

/-- The theorem stating the total amount shared -/
theorem total_amount_is_117 (d : MoneyDistribution) 
  (h : satisfies_conditions d) : 
  d.x + d.y + d.z = 117 := by
  sorry


end total_amount_is_117_l2111_211150


namespace num_divisors_2310_l2111_211148

/-- The number of positive divisors of a positive integer n -/
def numPositiveDivisors (n : ℕ+) : ℕ := sorry

/-- 2310 as a positive integer -/
def n : ℕ+ := 2310

/-- Theorem: The number of positive divisors of 2310 is 32 -/
theorem num_divisors_2310 : numPositiveDivisors n = 32 := by sorry

end num_divisors_2310_l2111_211148


namespace robin_afternoon_bottles_l2111_211147

/-- The number of bottles Robin drank in the morning -/
def morning_bottles : ℕ := 7

/-- The total number of bottles Robin drank -/
def total_bottles : ℕ := 14

/-- The number of bottles Robin drank in the afternoon -/
def afternoon_bottles : ℕ := total_bottles - morning_bottles

theorem robin_afternoon_bottles : afternoon_bottles = 7 := by
  sorry

end robin_afternoon_bottles_l2111_211147


namespace converse_correct_l2111_211170

/-- The original statement -/
def original_statement (x : ℝ) : Prop := x^2 = 1 → x = 1

/-- The converse statement -/
def converse_statement (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

/-- Theorem stating that the converse_statement is indeed the converse of the original_statement -/
theorem converse_correct :
  converse_statement = (fun x => ¬(original_statement x)) := by sorry

end converse_correct_l2111_211170


namespace min_value_theorem_l2111_211186

-- Define the lines
def l₁ (m n x y : ℝ) : Prop := m * x + y + n = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0
def l₃ (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Theorem statement
theorem min_value_theorem (m n : ℝ) 
  (h1 : ∃ x y : ℝ, l₁ m n x y ∧ l₂ x y ∧ l₃ x y) 
  (h2 : m * n > 0) :
  (1 / m + 2 / n) ≥ 8 := by
  sorry

end min_value_theorem_l2111_211186


namespace class_size_l2111_211125

def is_valid_total_score (n : ℕ) : Prop :=
  n ≥ 4460 ∧ n < 4470 ∧ n % 100 = 64 ∧ n % 8 = 0 ∧ n % 9 = 0

theorem class_size (total_score : ℕ) (h1 : is_valid_total_score total_score) 
  (h2 : (total_score : ℚ) / 72 = 62) : 
  ∃ (num_students : ℕ), (num_students : ℚ) = total_score / 72 := by
sorry

end class_size_l2111_211125


namespace overlap_length_l2111_211132

/-- Given information about overlapping segments, prove the length of each overlap --/
theorem overlap_length (total_length : ℝ) (measured_length : ℝ) (num_overlaps : ℕ) 
  (h1 : total_length = 98)
  (h2 : measured_length = 83)
  (h3 : num_overlaps = 6) :
  ∃ x : ℝ, x = 2.5 ∧ total_length = measured_length + num_overlaps * x :=
by
  sorry

end overlap_length_l2111_211132


namespace complex_simplification_l2111_211142

theorem complex_simplification :
  (4 - 2*Complex.I) - (7 - 2*Complex.I) + (6 - 3*Complex.I) = 3 - 3*Complex.I :=
by sorry

end complex_simplification_l2111_211142


namespace power_zero_eq_one_negative_two_power_zero_l2111_211168

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem negative_two_power_zero : (-2 : ℝ)^0 = 1 := by sorry

end power_zero_eq_one_negative_two_power_zero_l2111_211168


namespace max_consecutive_non_palindromic_l2111_211111

/-- A year is palindromic if it reads the same backward and forward -/
def isPalindromic (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year ≤ 9999 ∧ 
  (year / 1000 = year % 10) ∧ ((year / 100) % 10 = (year / 10) % 10)

/-- The maximum number of consecutive non-palindromic years between 1000 and 9999 -/
def maxConsecutiveNonPalindromic : ℕ := 109

theorem max_consecutive_non_palindromic :
  ∀ (start : ℕ) (len : ℕ),
    start ≥ 1000 → start + len ≤ 9999 →
    (∀ y : ℕ, y ≥ start ∧ y < start + len → ¬isPalindromic y) →
    len ≤ maxConsecutiveNonPalindromic :=
by sorry

end max_consecutive_non_palindromic_l2111_211111


namespace unique_integer_satisfying_conditions_l2111_211169

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4) :
  x = 1 := by
sorry

end unique_integer_satisfying_conditions_l2111_211169


namespace intersection_of_M_and_N_l2111_211120

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_M_and_N : 
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l2111_211120


namespace combined_score_is_78_l2111_211195

/-- Represents the score of a player in either football or basketball -/
structure PlayerScore where
  name : String
  score : ℕ

/-- Calculates the total score of a list of players -/
def totalScore (players : List PlayerScore) : ℕ :=
  players.foldl (fun acc p => acc + p.score) 0

/-- The combined score of football and basketball games -/
theorem combined_score_is_78 (bruce michael jack sarah andy lily : PlayerScore) :
  bruce.name = "Bruce" ∧ bruce.score = 4 ∧
  michael.name = "Michael" ∧ michael.score = 2 * bruce.score ∧
  jack.name = "Jack" ∧ jack.score = bruce.score - 1 ∧
  sarah.name = "Sarah" ∧ sarah.score = jack.score / 2 ∧
  andy.name = "Andy" ∧ andy.score = 22 ∧
  lily.name = "Lily" ∧ lily.score = andy.score + 18 →
  totalScore [bruce, michael, jack, sarah, andy, lily] = 78 := by
sorry

#eval totalScore [
  {name := "Bruce", score := 4},
  {name := "Michael", score := 8},
  {name := "Jack", score := 3},
  {name := "Sarah", score := 1},
  {name := "Andy", score := 22},
  {name := "Lily", score := 40}
]

end combined_score_is_78_l2111_211195


namespace line_equation_l2111_211113

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 5 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - 1

-- Define the intersection points
def intersection (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola x y ∧ line k x y

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection k x₁ y₁ ∧
    intersection k x₂ y₂ ∧
    (x₁ + x₂) / 2 = -2/3

theorem line_equation :
  ∀ k : ℝ, midpoint_condition k → k = 1 :=
by sorry

end line_equation_l2111_211113


namespace triangle_transformation_exists_l2111_211145

/-- Triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Line represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflect a point over a line -/
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Translate a point by a vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Apply reflection and translation to a triangle -/
def transformTriangle (t : Triangle) (l : Line) (v : ℝ × ℝ) : Triangle :=
  { A := translate (reflect t.A l) v
  , B := translate (reflect t.B l) v
  , C := translate (reflect t.C l) v }

theorem triangle_transformation_exists :
  ∃ (l : Line) (v : ℝ × ℝ),
    let t1 : Triangle := { A := (0, 0), B := (15, 0), C := (0, 5) }
    let t2 : Triangle := { A := (17.2, 19.6), B := (26.2, 6.6), C := (22, 21) }
    transformTriangle t2 l v = t1 := by
  sorry

end triangle_transformation_exists_l2111_211145


namespace vertical_angles_are_equal_converse_is_false_l2111_211149

-- Define what it means for angles to be vertical
def are_vertical_angles (α β : Real) : Prop := sorry

-- Define what it means for angles to be equal
def are_equal_angles (α β : Real) : Prop := α = β

-- Theorem stating that vertical angles are equal
theorem vertical_angles_are_equal (α β : Real) : 
  are_vertical_angles α β → are_equal_angles α β := by sorry

-- Theorem stating that the converse is false
theorem converse_is_false : 
  ¬(∀ α β : Real, are_equal_angles α β → are_vertical_angles α β) := by sorry

end vertical_angles_are_equal_converse_is_false_l2111_211149


namespace optimal_store_strategy_l2111_211155

/-- Represents the store's inventory and pricing strategy -/
structure Store where
  total_balls : Nat
  budget : Nat
  basketball_cost : Nat
  volleyball_cost : Nat
  basketball_price_ratio : Rat
  school_basketball_revenue : Nat
  school_volleyball_revenue : Nat
  school_volleyball_count_diff : Int

/-- Represents the store's pricing and purchase strategy -/
structure Strategy where
  basketball_price : Nat
  volleyball_price : Nat
  basketball_count : Nat
  volleyball_count : Nat

/-- Checks if the strategy satisfies all constraints -/
def is_valid_strategy (store : Store) (strategy : Strategy) : Prop :=
  strategy.basketball_count + strategy.volleyball_count = store.total_balls ∧
  strategy.basketball_count * store.basketball_cost + strategy.volleyball_count * store.volleyball_cost ≤ store.budget ∧
  strategy.basketball_price = (strategy.volleyball_price : Rat) * store.basketball_price_ratio ∧
  (store.school_basketball_revenue : Rat) / strategy.basketball_price - 
    (store.school_volleyball_revenue : Rat) / strategy.volleyball_price = store.school_volleyball_count_diff

/-- Calculates the profit after price reduction -/
def profit_after_reduction (store : Store) (strategy : Strategy) : Int :=
  (strategy.basketball_price - 3 - store.basketball_cost) * strategy.basketball_count +
  (strategy.volleyball_price - 2 - store.volleyball_cost) * strategy.volleyball_count

/-- Main theorem: Proves the optimal strategy for the store -/
theorem optimal_store_strategy (store : Store) 
    (h_store : store.total_balls = 200 ∧ 
               store.budget = 5000 ∧ 
               store.basketball_cost = 30 ∧ 
               store.volleyball_cost = 24 ∧ 
               store.basketball_price_ratio = 3/2 ∧
               store.school_basketball_revenue = 1800 ∧
               store.school_volleyball_revenue = 1500 ∧
               store.school_volleyball_count_diff = 10) :
  ∃ (strategy : Strategy),
    is_valid_strategy store strategy ∧
    strategy.basketball_price = 45 ∧
    strategy.volleyball_price = 30 ∧
    strategy.basketball_count = 33 ∧
    strategy.volleyball_count = 167 ∧
    ∀ (other_strategy : Strategy),
      is_valid_strategy store other_strategy →
      profit_after_reduction store strategy ≥ profit_after_reduction store other_strategy :=
by
  sorry


end optimal_store_strategy_l2111_211155


namespace total_games_played_l2111_211105

/-- Proves that a team with the given win percentages played 175 games in total -/
theorem total_games_played (first_100_win_rate : ℝ) (remaining_win_rate : ℝ) (total_win_rate : ℝ) 
  (h1 : first_100_win_rate = 0.85)
  (h2 : remaining_win_rate = 0.5)
  (h3 : total_win_rate = 0.7) :
  ∃ (total_games : ℕ), 
    total_games = 175 ∧ 
    (first_100_win_rate * 100 + remaining_win_rate * (total_games - 100 : ℝ)) / total_games = total_win_rate :=
by sorry

end total_games_played_l2111_211105


namespace p_necessary_not_sufficient_for_q_l2111_211151

-- Define the conditions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 2| + |x + 2| > m

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 4 > 0

-- Define the relationship between p and q
theorem p_necessary_not_sufficient_for_q :
  (∃ m : ℝ, p m ∧ ¬q m) ∧ (∀ m : ℝ, q m → p m) := by sorry

end p_necessary_not_sufficient_for_q_l2111_211151


namespace rabbit_cat_age_ratio_l2111_211167

/-- Given the ages of a cat, dog, and rabbit, prove the ratio of rabbit's age to cat's age --/
theorem rabbit_cat_age_ratio 
  (cat_age : ℕ) 
  (dog_age : ℕ) 
  (rabbit_age : ℕ) 
  (h1 : cat_age = 8) 
  (h2 : dog_age = 12) 
  (h3 : rabbit_age * 3 = dog_age) : 
  (rabbit_age : ℚ) / cat_age = 1 / 2 := by
  sorry

end rabbit_cat_age_ratio_l2111_211167


namespace koala_fiber_intake_l2111_211176

/-- Given that koalas absorb 40% of the fiber they eat and a particular koala
    absorbed 16 ounces of fiber in one day, prove that it ate 40 ounces of fiber. -/
theorem koala_fiber_intake (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_intake : ℝ) :
  absorption_rate = 0.40 →
  absorbed_amount = 16 →
  absorbed_amount = absorption_rate * total_intake →
  total_intake = 40 := by
  sorry

end koala_fiber_intake_l2111_211176


namespace triangle_perimeter_l2111_211190

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 11*x + 30 = 0 →
  2 + x > 4 ∧ 4 + x > 2 ∧ 2 + 4 > x →
  2 + 4 + x = 11 :=
by
  sorry

end triangle_perimeter_l2111_211190


namespace closest_point_on_line_l2111_211156

/-- The point on the line y = 2x + 3 that is closest to (2, -1) is (-6/5, 3/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 2 * x + 3 →  -- line equation
  (x + 6/5)^2 + (y - 3/5)^2 ≤ (x - 2)^2 + (y + 1)^2 :=
by sorry

end closest_point_on_line_l2111_211156


namespace product_closest_to_127_l2111_211130

def product : ℝ := 2.5 * (50.5 + 0.25)

def options : List ℝ := [120, 125, 127, 130, 140]

theorem product_closest_to_127 :
  ∀ x ∈ options, x ≠ 127 → |product - 127| < |product - x| :=
by sorry

end product_closest_to_127_l2111_211130


namespace profit_maximization_and_cost_l2111_211163

/-- Represents the relationship between selling price and daily sales volume -/
def sales_volume (x : ℝ) : ℝ := -30 * x + 1500

/-- Calculates the daily sales profit -/
def sales_profit (x : ℝ) : ℝ := sales_volume x * (x - 30)

/-- Calculates the daily profit including additional cost a -/
def total_profit (x a : ℝ) : ℝ := sales_volume x * (x - 30 - a)

theorem profit_maximization_and_cost (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 10) :
  (∀ x, sales_profit x ≤ sales_profit 40) ∧
  (∃ x, 40 ≤ x ∧ x ≤ 45 ∧ total_profit x a = 2430) → a = 2 :=
by sorry

end profit_maximization_and_cost_l2111_211163


namespace parabola_tangent_to_line_l2111_211152

/-- A parabola of the form y = ax^2 + 6 is tangent to the line y = 2x + 4 if and only if a = 1/2 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, ax^2 + 6 = 2*x + 4 ∧ 
   ∀ y : ℝ, y ≠ x → ay^2 + 6 ≠ 2*y + 4) ↔ 
  a = 1/2 := by
sorry

end parabola_tangent_to_line_l2111_211152


namespace sqrt_sum_difference_product_l2111_211194

theorem sqrt_sum_difference_product (a b c d : ℝ) :
  Real.sqrt 75 + Real.sqrt 27 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 8 * Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end sqrt_sum_difference_product_l2111_211194


namespace no_valid_solutions_l2111_211146

theorem no_valid_solutions : ¬∃ (a b : ℝ), ∀ x, (a * x + b)^2 = 4 * x^2 + 4 * x + 4 := by sorry

end no_valid_solutions_l2111_211146


namespace A_intersect_B_l2111_211115

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 1| ≥ 2}

theorem A_intersect_B : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ -1} := by sorry

end A_intersect_B_l2111_211115


namespace inequality_solution_set_l2111_211140

theorem inequality_solution_set (x : ℝ) : 
  (8 - x^2 > 2*x) ↔ (-4 < x ∧ x < 2) := by
sorry

end inequality_solution_set_l2111_211140


namespace simplify_sum_of_roots_l2111_211133

theorem simplify_sum_of_roots : 
  Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) + Real.sqrt (1 + 2 + 3 + 4 + 5) + 2 = 
  Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10 + Real.sqrt 15 + 2 := by sorry

end simplify_sum_of_roots_l2111_211133


namespace jelly_cost_l2111_211141

/-- The cost of jelly for sandwiches --/
theorem jelly_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 378 → 
  (N * J * 7 : ℚ) / 100 = 294 / 100 := by
  sorry

end jelly_cost_l2111_211141


namespace bowling_ball_weight_l2111_211100

/-- Given that five identical bowling balls weigh the same as four identical canoes,
    and two canoes weigh 80 pounds, prove that one bowling ball weighs 32 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℕ),
    5 * bowling_ball_weight = 4 * canoe_weight →
    2 * canoe_weight = 80 →
    bowling_ball_weight = 32 := by
  sorry

end bowling_ball_weight_l2111_211100


namespace rectangle_side_increase_l2111_211191

theorem rectangle_side_increase (increase_factor : Real) :
  increase_factor > 0 →
  (1 + increase_factor)^2 = 1.8225 →
  increase_factor = 0.35 := by
sorry

end rectangle_side_increase_l2111_211191


namespace probability_different_families_l2111_211154

/-- The number of families -/
def num_families : ℕ := 6

/-- The number of members in each family -/
def members_per_family : ℕ := 3

/-- The total number of people -/
def total_people : ℕ := num_families * members_per_family

/-- The size of each group in the game -/
def group_size : ℕ := 3

/-- The probability of selecting 3 people from different families -/
theorem probability_different_families : 
  (Nat.choose num_families group_size * (members_per_family ^ group_size)) / 
  (Nat.choose total_people group_size) = 45 / 68 := by sorry

end probability_different_families_l2111_211154


namespace product_of_roots_zero_l2111_211117

theorem product_of_roots_zero (x₁ x₂ : ℝ) : 
  ((-x₁^2 + 3*x₁ = 0) ∧ (-x₂^2 + 3*x₂ = 0)) → x₁ * x₂ = 0 := by
  sorry

end product_of_roots_zero_l2111_211117


namespace dispatch_plans_eq_28_l2111_211164

/-- Given a set of athletes with the following properties:
  * There are 9 athletes in total
  * 5 athletes can play basketball
  * 6 athletes can play soccer
This function calculates the number of ways to select one athlete for basketball
and one for soccer. -/
def dispatch_plans (total : Nat) (basketball : Nat) (soccer : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of dispatch plans for the given conditions is 28. -/
theorem dispatch_plans_eq_28 : dispatch_plans 9 5 6 = 28 := by
  sorry

end dispatch_plans_eq_28_l2111_211164


namespace absolute_sum_of_roots_greater_than_four_l2111_211184

theorem absolute_sum_of_roots_greater_than_four 
  (p : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : x₁^2 + p*x₁ + 4 = 0) 
  (h3 : x₂^2 + p*x₂ + 4 = 0) : 
  |x₁ + x₂| > 4 := by
sorry

end absolute_sum_of_roots_greater_than_four_l2111_211184


namespace coefficient_x3y5_in_expansion_l2111_211135

theorem coefficient_x3y5_in_expansion : ∀ (x y : ℝ),
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℝ) * x^(8 - k) * y^k) =
  56 * x^3 * y^5 + (Finset.range 9).sum (fun k => if k ≠ 5 then (Nat.choose 8 k : ℝ) * x^(8 - k) * y^k else 0) :=
by sorry

end coefficient_x3y5_in_expansion_l2111_211135


namespace vasya_drove_two_fifths_l2111_211123

/-- Represents the fraction of total distance driven by each person -/
structure DistanceFractions where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- Conditions of the driving problem -/
def driving_conditions (d : DistanceFractions) : Prop :=
  d.anton = d.vasya / 2 ∧
  d.sasha = d.anton + d.dima ∧
  d.dima = 1 / 10 ∧
  d.anton + d.vasya + d.sasha + d.dima = 1

/-- Theorem stating that Vasya drove 2/5 of the total distance -/
theorem vasya_drove_two_fifths :
  ∀ d : DistanceFractions, driving_conditions d → d.vasya = 2 / 5 := by
  sorry


end vasya_drove_two_fifths_l2111_211123


namespace polynomial_equality_sum_l2111_211189

theorem polynomial_equality_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a₀ + a₁*(x + 3) + a₂*(x + 3)^2 + a₃*(x + 3)^3 + 
    a₄*(x + 3)^4 + a₅*(x + 3)^5 + a₆*(x + 3)^6 + a₇*(x + 3)^7 + a₈*(x + 3)^8 + 
    a₉*(x + 3)^9 + a₁₀*(x + 3)^10) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 9 :=
by sorry

end polynomial_equality_sum_l2111_211189


namespace empty_solution_set_implies_b_greater_than_nine_l2111_211136

/-- If the solution set of the inequality |x-4|-|x+5| ≥ b about x is empty, then b > 9 -/
theorem empty_solution_set_implies_b_greater_than_nine (b : ℝ) :
  (∀ x : ℝ, |x - 4| - |x + 5| < b) → b > 9 := by
  sorry

end empty_solution_set_implies_b_greater_than_nine_l2111_211136


namespace g_determinant_identity_g_1002_1004_minus_1003_squared_l2111_211107

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

-- Define the sequence G
def G : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => G (n + 1) - G n

-- State the theorem
theorem g_determinant_identity (n : ℕ) :
  A ^ n = !![G (n + 1), G n; G n, G (n - 1)] →
  G n * G (n + 2) - G (n + 1) ^ 2 = 1 := by
  sorry

-- The specific case for n = 1003
theorem g_1002_1004_minus_1003_squared :
  G 1002 * G 1004 - G 1003 ^ 2 = 1 := by
  sorry

end g_determinant_identity_g_1002_1004_minus_1003_squared_l2111_211107


namespace abc_mod_five_l2111_211104

theorem abc_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 →
  (a + 2*b + 3*c) % 5 = 0 →
  (2*a + 3*b + c) % 5 = 2 →
  (3*a + b + 2*c) % 5 = 3 →
  (a*b*c) % 5 = 3 := by
sorry

end abc_mod_five_l2111_211104


namespace book_words_per_page_l2111_211178

theorem book_words_per_page 
  (total_pages : ℕ) 
  (max_words_per_page : ℕ) 
  (total_words_mod : ℕ) 
  (modulus : ℕ) 
  (h1 : total_pages = 180) 
  (h2 : max_words_per_page = 150) 
  (h3 : total_words_mod = 203) 
  (h4 : modulus = 229) 
  (h5 : ∃ (words_per_page : ℕ), 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % modulus = total_words_mod) :
  ∃ (words_per_page : ℕ), words_per_page = 94 ∧ 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % modulus = total_words_mod :=
sorry

end book_words_per_page_l2111_211178


namespace sin_cos_rational_implies_natural_combination_l2111_211143

theorem sin_cos_rational_implies_natural_combination 
  (x y : ℝ) 
  (h1 : ∃ (a : ℚ), a > 0 ∧ Real.sin x + Real.cos y = a)
  (h2 : ∃ (b : ℚ), b > 0 ∧ Real.sin y + Real.cos x = b) :
  ∃ (m n : ℕ), ∃ (k : ℕ), m * Real.sin x + n * Real.cos x = k := by
  sorry

end sin_cos_rational_implies_natural_combination_l2111_211143


namespace current_velocity_proof_l2111_211162

/-- The velocity of the current in a river, given the following conditions:
  1. A man can row at 5 kmph in still water.
  2. It takes him 1 hour to row to a place and come back.
  3. The place is 2.4 km away. -/
def current_velocity : ℝ := 1

/-- The man's rowing speed in still water (in kmph) -/
def rowing_speed : ℝ := 5

/-- The distance to the destination (in km) -/
def distance : ℝ := 2.4

/-- The total time for the round trip (in hours) -/
def total_time : ℝ := 1

theorem current_velocity_proof :
  (distance / (rowing_speed + current_velocity) +
   distance / (rowing_speed - current_velocity) = total_time) ∧
  (current_velocity > 0) ∧
  (current_velocity < rowing_speed) := by
  sorry

end current_velocity_proof_l2111_211162


namespace shooter_prob_below_8_l2111_211101

-- Define the probabilities
def prob_10_ring : ℝ := 0.20
def prob_9_ring : ℝ := 0.30
def prob_8_ring : ℝ := 0.10

-- Define the probability of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10_ring + prob_9_ring + prob_8_ring)

-- Theorem statement
theorem shooter_prob_below_8 : prob_below_8 = 0.40 := by
  sorry

end shooter_prob_below_8_l2111_211101


namespace systematic_sample_count_l2111_211139

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the number of sampled individuals within a given interval -/
def count_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) / (s.total_population / s.sample_size))

/-- Theorem stating that for the given parameters, 13 individuals are sampled from the interval -/
theorem systematic_sample_count (s : SystematicSample) 
  (h1 : s.total_population = 840)
  (h2 : s.sample_size = 42)
  (h3 : s.interval_start = 461)
  (h4 : s.interval_end = 720) :
  count_in_interval s = 13 := by
  sorry

end systematic_sample_count_l2111_211139


namespace wheel_probability_l2111_211122

theorem wheel_probability (P_A P_B P_C P_D : ℚ) : 
  P_A = 1/4 → P_B = 1/3 → P_C = 1/6 → P_A + P_B + P_C + P_D = 1 → P_D = 1/4 := by
  sorry

end wheel_probability_l2111_211122


namespace monica_savings_l2111_211188

theorem monica_savings (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_per_cycle : ℚ) 
  (h1 : weeks_per_cycle = 60)
  (h2 : num_cycles = 5)
  (h3 : total_per_cycle = 4500) :
  (num_cycles * total_per_cycle) / (num_cycles * weeks_per_cycle) = 75 := by
  sorry

end monica_savings_l2111_211188


namespace tangent_slope_xe_pow_x_l2111_211196

open Real

theorem tangent_slope_xe_pow_x (e : ℝ) (h : e = exp 1) :
  let f : ℝ → ℝ := λ x ↦ x * exp x
  let df : ℝ → ℝ := λ x ↦ (1 + x) * exp x
  df 1 = 2 * e :=
by sorry

end tangent_slope_xe_pow_x_l2111_211196


namespace gcd_polynomial_and_b_l2111_211160

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 528 * k) :
  Nat.gcd (2 * b^4 + b^3 + 5 * b^2 + 6 * b + 132).natAbs b.natAbs = 132 := by
  sorry

end gcd_polynomial_and_b_l2111_211160


namespace quadratic_ratio_l2111_211153

/-- Given a quadratic function f(x) = ax² + bx + c where a > 0,
    if the solution set of f(x) > 0 is (-∞, -2) ∪ (-1, +∞),
    then the ratio a:b:c is 1:3:2 -/
theorem quadratic_ratio (a b c : ℝ) : 
  a > 0 → 
  (∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > -1) → 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k ∧ b = 3*k ∧ c = 2*k :=
sorry

end quadratic_ratio_l2111_211153


namespace value_of_a_minus_b_l2111_211112

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2028)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = -3 := by
sorry

end value_of_a_minus_b_l2111_211112


namespace min_difference_is_1747_l2111_211138

/-- Represents a valid digit assignment for the problem -/
structure DigitAssignment where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                  d ≠ e ∧ d ≠ f ∧
                  e ≠ f
  all_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0
  sum_constraint : 1000 * a + 100 * b + 10 * c + d + 10 * e + f = 2010

/-- The main theorem stating the minimum difference -/
theorem min_difference_is_1747 : 
  ∀ (assign : DigitAssignment), 
    1000 * assign.a + 100 * assign.b + 10 * assign.c + assign.d - (10 * assign.e + assign.f) = 1747 := by
  sorry

end min_difference_is_1747_l2111_211138


namespace arithmetic_sequence_properties_l2111_211137

/-- An arithmetic sequence with first term -1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  2 * n - 17

/-- The sum of the first n terms of the arithmetic sequence -/
def sequence_sum (n : ℕ) : ℤ :=
  n^2 - 6*n

theorem arithmetic_sequence_properties :
  ∀ n : ℕ,
  (n > 0) →
  (arithmetic_sequence n = 2 * n - 17) ∧
  (sequence_sum n = n^2 - 6*n) ∧
  (∀ t : ℝ, (∀ k : ℕ, k > 0 → sequence_sum k > t) ↔ t < -6) :=
sorry

end arithmetic_sequence_properties_l2111_211137


namespace rotate90_neg4_plus_2i_l2111_211198

def rotate90(z : ℂ) : ℂ := z * Complex.I

theorem rotate90_neg4_plus_2i :
  rotate90 (-4 + 2 * Complex.I) = -2 - 4 * Complex.I := by
  sorry

end rotate90_neg4_plus_2i_l2111_211198


namespace z_shaped_area_l2111_211134

/-- The area of a Z-shaped region formed by subtracting two squares from a rectangle -/
theorem z_shaped_area (rectangle_length rectangle_width square1_side square2_side : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 4)
  (h3 : square1_side = 2)
  (h4 : square2_side = 1) :
  rectangle_length * rectangle_width - (square1_side^2 + square2_side^2) = 19 := by
  sorry

end z_shaped_area_l2111_211134


namespace inserted_numbers_sum_l2111_211183

theorem inserted_numbers_sum : ∃ (a b : ℝ), 
  4 < a ∧ a < b ∧ b < 16 ∧ 
  (b - a = a - 4) ∧
  (b * b = a * 16) ∧
  a + b = 20 := by
  sorry

end inserted_numbers_sum_l2111_211183


namespace exists_multiple_sum_of_digits_divides_l2111_211175

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For every positive integer n, there exists a multiple of n whose sum of digits divides it -/
theorem exists_multiple_sum_of_digits_divides (n : ℕ+) : 
  ∃ k : ℕ+, (sum_of_digits (k * n) ∣ (k * n)) := by sorry

end exists_multiple_sum_of_digits_divides_l2111_211175


namespace derangement_of_five_l2111_211109

/-- Calculates the number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k + 2 => (k + 1) * (derangement (k + 1) + derangement k)

/-- The number of derangements for 5 elements is 44 -/
theorem derangement_of_five : derangement 5 = 44 := by
  sorry

end derangement_of_five_l2111_211109


namespace sum_of_fractions_integer_l2111_211129

theorem sum_of_fractions_integer (a b : ℤ) :
  (a ≠ 0 ∧ b ≠ 0) →
  (∃ k : ℤ, (a : ℚ) / b + (b : ℚ) / a = k) ↔ (a = b ∨ a = -b) :=
by sorry

end sum_of_fractions_integer_l2111_211129


namespace parallelogram_bisector_theorem_l2111_211118

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := sorry

/-- The length between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a line defined by two other points -/
def isOnLine (p1 p2 p : Point) : Prop := sorry

/-- Check if a line is an angle bisector -/
def isAngleBisector (vertex p1 p2 : Point) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_bisector_theorem (ABCD : Parallelogram) (E F : Point) :
  perimeter ABCD = 32 →
  isAngleBisector ABCD.C ABCD.D ABCD.B →
  isOnLine ABCD.A ABCD.D E →
  isOnLine ABCD.A ABCD.B F →
  distance ABCD.A E = 2 →
  (distance ABCD.B F = 7 ∨ distance ABCD.B F = 9) := by sorry

end parallelogram_bisector_theorem_l2111_211118


namespace largest_prime_factors_difference_l2111_211181

theorem largest_prime_factors_difference (n : Nat) (h : n = 184437) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p > q ∧
  (∀ r : Nat, Prime r → r ∣ n → r ≤ p) ∧
  (p ∣ n) ∧ (q ∣ n) ∧ (p - q = 8776) := by
  sorry

end largest_prime_factors_difference_l2111_211181


namespace triangle_area_sum_properties_l2111_211106

/-- Represents a rectangular prism with dimensions 1, 2, and 3 -/
structure RectangularPrism where
  length : ℝ := 1
  width : ℝ := 2
  height : ℝ := 3

/-- Represents the sum of areas of all triangles with vertices on the prism -/
def triangleAreaSum (prism : RectangularPrism) : ℝ := sorry

/-- Theorem stating the properties of the triangle area sum -/
theorem triangle_area_sum_properties (prism : RectangularPrism) :
  ∃ (m a n : ℤ), 
    triangleAreaSum prism = m + a * Real.sqrt n ∧ 
    m + n + a = 49 := by
  sorry

end triangle_area_sum_properties_l2111_211106


namespace sum_of_integers_l2111_211144

theorem sum_of_integers (a b c : ℤ) :
  a = (b + c) / 3 →
  b = (a + c) / 5 →
  c = 35 →
  a + b + c = 60 := by
sorry

end sum_of_integers_l2111_211144


namespace max_non_managers_proof_l2111_211157

/-- Represents a department in the company -/
structure Department where
  managers : ℕ
  nonManagers : ℕ

/-- The ratio condition for managers to non-managers -/
def validRatio (d : Department) : Prop :=
  (d.managers : ℚ) / d.nonManagers > 7 / 24

/-- The maximum number of non-managers allowed -/
def maxNonManagers : ℕ := 27

/-- The minimum number of managers required -/
def minManagers : ℕ := 8

theorem max_non_managers_proof (d : Department) 
    (h1 : d.managers ≥ minManagers) 
    (h2 : validRatio d) 
    (h3 : d.nonManagers ≤ maxNonManagers) :
    d.nonManagers = maxNonManagers :=
  sorry

#check max_non_managers_proof

end max_non_managers_proof_l2111_211157


namespace corrected_mean_problem_l2111_211193

/-- Given a set of observations with an incorrect mean due to a misrecorded value,
    calculate the corrected mean. -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / (n : ℚ)

/-- Theorem stating that the corrected mean for the given problem is 45.45 -/
theorem corrected_mean_problem :
  corrected_mean 100 45 20 65 = 45.45 := by
  sorry

end corrected_mean_problem_l2111_211193


namespace balloon_distribution_l2111_211114

theorem balloon_distribution (red white green chartreuse : ℕ) (friends : ℕ) : 
  red = 22 → white = 40 → green = 70 → chartreuse = 90 → friends = 10 →
  (red + white + green + chartreuse) % friends = 2 :=
by sorry

end balloon_distribution_l2111_211114


namespace nested_subtraction_simplification_l2111_211127

theorem nested_subtraction_simplification : 2 - (-2 - 2) - (-2 - (-2 - 2)) = 4 := by
  sorry

end nested_subtraction_simplification_l2111_211127


namespace regular_polygon_150_degree_angles_l2111_211161

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) → n = 12 :=
by
  sorry

end regular_polygon_150_degree_angles_l2111_211161


namespace qing_dynasty_problem_l2111_211116

/-- Represents the price of animals in ancient Chinese currency (taels) --/
structure AnimalPrices where
  horse : ℝ
  cattle : ℝ

/-- Represents a combination of horses and cattle --/
structure AnimalCombination where
  horses : ℕ
  cattle : ℕ

/-- Calculates the total cost of a combination of animals given their prices --/
def totalCost (prices : AnimalPrices) (combo : AnimalCombination) : ℝ :=
  prices.horse * combo.horses + prices.cattle * combo.cattle

/-- The theorem representing the original problem --/
theorem qing_dynasty_problem (prices : AnimalPrices) : 
  totalCost prices ⟨4, 6⟩ = 48 ∧ 
  totalCost prices ⟨2, 5⟩ = 38 ↔ 
  4 * prices.horse + 6 * prices.cattle = 48 ∧
  2 * prices.horse + 5 * prices.cattle = 38 := by
  sorry


end qing_dynasty_problem_l2111_211116


namespace ellipse_parabola_configuration_eccentricity_is_half_l2111_211199

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focal length c -/
structure Parabola where
  c : ℝ
  h_pos : 0 < c

/-- Configuration of the ellipse and parabola -/
structure Configuration where
  C₁ : Ellipse
  C₂ : Parabola
  h_focus : C₁.a * C₁.a - C₁.b * C₁.b = C₂.c * C₂.c  -- Right focus of C₁ coincides with focus of C₂
  h_center : True  -- Center of C₁ coincides with vertex of C₂ (implied by other conditions)
  h_chord_ratio : (2 * C₂.c) = 4/3 * (2 * C₁.b * C₁.b / C₁.a)  -- |CD| = 4/3 * |AB|
  h_vertices_sum : 2 * C₁.a + C₂.c = 12  -- Sum of distances from vertices of C₁ to directrix of C₂

/-- Main theorem statement -/
theorem ellipse_parabola_configuration (cfg : Configuration) :
  cfg.C₁.a * cfg.C₁.a = 16 ∧ 
  cfg.C₁.b * cfg.C₁.b = 12 ∧ 
  cfg.C₂.c = 2 :=
by sorry

/-- Corollary: Eccentricity of C₁ is 1/2 -/
theorem eccentricity_is_half (cfg : Configuration) :
  Real.sqrt (cfg.C₁.a * cfg.C₁.a - cfg.C₁.b * cfg.C₁.b) / cfg.C₁.a = 1/2 :=
by sorry

end ellipse_parabola_configuration_eccentricity_is_half_l2111_211199


namespace inequality_solution_l2111_211124

theorem inequality_solution :
  let f (x : ℝ) := x^3 - 3*x - 3/x + 1/x^3 + 5
  ∀ x : ℝ, (202 * Real.sqrt (f x) ≤ 0) ↔
    (x = (-1 - Real.sqrt 21 + Real.sqrt (2 * Real.sqrt 21 + 6)) / 4 ∨
     x = (-1 - Real.sqrt 21 - Real.sqrt (2 * Real.sqrt 21 + 6)) / 4) := by
  sorry

end inequality_solution_l2111_211124


namespace parametric_equation_solution_l2111_211192

theorem parametric_equation_solution (a b : ℝ) (h1 : a ≠ 2 * b) (h2 : a ≠ -3 * b) :
  ∃! x : ℝ, (a * x - 3) / (b * x + 1) = 2 :=
by
  use 5 / (a - 2 * b)
  sorry

end parametric_equation_solution_l2111_211192


namespace trinomial_square_l2111_211179

theorem trinomial_square (c : ℚ) : 
  (∃ b y : ℚ, ∀ x : ℚ, 9*x^2 - 21*x + c = (3*x + b + y)^2) → c = 49/4 := by
sorry

end trinomial_square_l2111_211179


namespace integer_sum_of_powers_l2111_211108

theorem integer_sum_of_powers (a b c : ℤ) 
  (h : (a - b)^10 + (a - c)^10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
  sorry

end integer_sum_of_powers_l2111_211108


namespace infinitely_many_square_sum_square_no_zero_l2111_211103

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Check if a number contains no zero digits -/
def no_zero_digits (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem infinitely_many_square_sum_square_no_zero :
  ∃ f : ℕ → ℕ, 
    (∀ m : ℕ, ∃ k : ℕ, f m = k^2) ∧ 
    (∀ m : ℕ, ∃ l : ℕ, S (f m) = l^2) ∧
    (∀ m : ℕ, no_zero_digits (f m)) ∧
    Function.Injective f :=
sorry

end infinitely_many_square_sum_square_no_zero_l2111_211103


namespace village_population_l2111_211126

theorem village_population (P : ℝ) : 
  (P * (1 - 0.1) * (1 - 0.2) = 3312) → P = 4600 := by sorry

end village_population_l2111_211126


namespace yellow_score_mixture_l2111_211128

theorem yellow_score_mixture (white_ratio black_ratio total_yellow : ℕ) 
  (h1 : white_ratio = 7)
  (h2 : black_ratio = 6)
  (h3 : total_yellow = 78) :
  (2 : ℚ) / 3 * (white_ratio - black_ratio) / (white_ratio + black_ratio) * total_yellow = 4 := by
  sorry

end yellow_score_mixture_l2111_211128


namespace exists_carmichael_number_l2111_211166

theorem exists_carmichael_number : 
  ∃ n : ℕ, 
    n > 1 ∧ 
    ¬(Nat.Prime n) ∧ 
    ∀ a : ℤ, (a^n) % n = a % n :=
by sorry

end exists_carmichael_number_l2111_211166


namespace concept_laws_theorem_l2111_211197

/-- The probability that exactly M laws are included in the Concept -/
def prob_M_laws_included (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws_included (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

theorem concept_laws_theorem (K N M : ℕ) (p : ℝ) 
  (hK : K > 0) (hN : N > 0) (hM : M ≤ K) (hp : 0 ≤ p ∧ p ≤ 1) :
  (prob_M_laws_included K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws_included K N p = K * (1 - (1 - p)^N)) := by
  sorry

end concept_laws_theorem_l2111_211197


namespace last_remaining_card_l2111_211182

/-- The largest power of 2 less than or equal to n -/
def largestPowerOf2 (n : ℕ) : ℕ :=
  (Nat.log2 n).succ

/-- The process of eliminating cards -/
def cardElimination (n : ℕ) : ℕ :=
  let L := largestPowerOf2 n
  2 * (n - 2^L) + 1

theorem last_remaining_card (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), k ≤ n ∧ cardElimination n = k :=
sorry

end last_remaining_card_l2111_211182


namespace fraction_sum_equality_l2111_211173

theorem fraction_sum_equality (p q r s : ℝ) 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := by
  sorry

end fraction_sum_equality_l2111_211173


namespace min_value_of_sum_l2111_211119

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 2) : 
  x + y ≥ 9/2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 2 ∧ x + y = 9/2 :=
by
  sorry

#check min_value_of_sum

end min_value_of_sum_l2111_211119
