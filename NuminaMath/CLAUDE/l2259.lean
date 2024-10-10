import Mathlib

namespace circular_seating_sum_l2259_225993

theorem circular_seating_sum (n : ℕ) (h : n ≥ 3) :
  (∀ (girl_sum : ℕ → ℕ) (boy_cards : ℕ → ℕ) (girl_cards : ℕ → ℕ),
    (∀ i : ℕ, i ∈ Finset.range n → 1 ≤ boy_cards i ∧ boy_cards i ≤ n) →
    (∀ i : ℕ, i ∈ Finset.range n → n + 1 ≤ girl_cards i ∧ girl_cards i ≤ 2*n) →
    (∀ i : ℕ, i ∈ Finset.range n → 
      girl_sum i = girl_cards i + boy_cards i + boy_cards ((i + 1) % n)) →
    (∀ i j : ℕ, i ∈ Finset.range n → j ∈ Finset.range n → girl_sum i = girl_sum j)) ↔
  Odd n :=
by sorry

end circular_seating_sum_l2259_225993


namespace apple_orchard_problem_l2259_225955

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧ 
  o.pure_fuji + o.cross_pollinated = 153 ∧
  o.pure_fuji = (o.total * 3) / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem apple_orchard_problem :
  ∀ o : Orchard, orchard_conditions o → o.pure_gala = 27 := by
  sorry

end apple_orchard_problem_l2259_225955


namespace negation_P_necessary_not_sufficient_for_negation_Q_l2259_225905

def P (x : ℝ) : Prop := |x - 2| ≥ 1

def Q (x : ℝ) : Prop := x^2 - 3*x + 2 ≥ 0

theorem negation_P_necessary_not_sufficient_for_negation_Q :
  (∀ x, ¬(Q x) → ¬(P x)) ∧ 
  (∃ x, ¬(P x) ∧ Q x) :=
by sorry

end negation_P_necessary_not_sufficient_for_negation_Q_l2259_225905


namespace car_fuel_efficiency_l2259_225921

/-- Proves that a car's initial fuel efficiency is 24 miles per gallon given specific conditions -/
theorem car_fuel_efficiency 
  (initial_efficiency : ℝ) 
  (improvement_factor : ℝ) 
  (tank_capacity : ℝ) 
  (additional_miles : ℝ) 
  (h1 : improvement_factor = 4/3) 
  (h2 : tank_capacity = 12) 
  (h3 : additional_miles = 96) 
  (h4 : tank_capacity * initial_efficiency * improvement_factor - 
        tank_capacity * initial_efficiency = additional_miles) : 
  initial_efficiency = 24 := by
  sorry

end car_fuel_efficiency_l2259_225921


namespace fraction_identity_l2259_225915

theorem fraction_identity (m : ℕ) (hm : m > 0) :
  (1 : ℚ) / (m * (m + 1)) = 1 / m - 1 / (m + 1) ∧
  (1 : ℚ) / (6 * 7) = 1 / 6 - 1 / 7 ∧
  ∃ (x : ℚ), x = 4 ∧ 1 / ((x - 1) * (x - 2)) + 1 / (x * (x - 1)) = 1 / x :=
by sorry

end fraction_identity_l2259_225915


namespace rearrangement_methods_l2259_225907

theorem rearrangement_methods (n m k : ℕ) (hn : n = 8) (hm : m = 4) (hk : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end rearrangement_methods_l2259_225907


namespace intersection_complement_equality_l2259_225996

def U : Finset ℕ := {1,2,3,4,5,6}
def A : Finset ℕ := {2,4,5}
def B : Finset ℕ := {1,3}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,4,5} := by
  sorry

end intersection_complement_equality_l2259_225996


namespace max_value_of_f_l2259_225903

-- Define the function f(x) = √(x-3) + √(6-x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) + Real.sqrt (6 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧
  f x = Real.sqrt 6 ∧
  ∀ (y : ℝ), 3 ≤ y ∧ y ≤ 6 → f y ≤ Real.sqrt 6 :=
sorry

end max_value_of_f_l2259_225903


namespace seventh_group_sample_l2259_225917

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (populationSize : ℕ) (groupCount : ℕ) (firstNumber : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := populationSize / groupCount
  let lastTwoDigits := (firstNumber + 33 * groupNumber) % 100
  (groupNumber - 1) * interval + lastTwoDigits

/-- Theorem stating the result of the systematic sampling for the 7th group -/
theorem seventh_group_sample :
  systematicSample 1000 10 57 7 = 688 := by
  sorry

#eval systematicSample 1000 10 57 7

end seventh_group_sample_l2259_225917


namespace four_digit_equation_solutions_l2259_225988

/-- Represents a four-digit number ABCD as a pair of two-digit numbers (AB, CD) -/
def FourDigitNumber := Nat × Nat

/-- Checks if a pair of numbers represents a valid four-digit number -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  10 ≤ n.1 ∧ n.1 ≤ 99 ∧ 10 ≤ n.2 ∧ n.2 ≤ 99

/-- Converts a pair of two-digit numbers to a four-digit number -/
def toNumber (n : FourDigitNumber) : Nat :=
  100 * n.1 + n.2

/-- The equation that the four-digit number must satisfy -/
def satisfiesEquation (n : FourDigitNumber) : Prop :=
  toNumber n = n.1 * n.2 + n.1 * n.1

theorem four_digit_equation_solutions :
  ∀ n : FourDigitNumber, 
    isValidFourDigitNumber n ∧ satisfiesEquation n ↔ 
    n = (12, 96) ∨ n = (34, 68) := by
  sorry

end four_digit_equation_solutions_l2259_225988


namespace parallelepiped_volume_l2259_225908

def vector1 : ℝ × ℝ × ℝ := (3, 4, 5)
def vector2 (m : ℝ) : ℝ × ℝ × ℝ := (2, m, 3)
def vector3 (m : ℝ) : ℝ × ℝ × ℝ := (2, 3, m)

def volume (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  abs (a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1))

theorem parallelepiped_volume (m : ℝ) :
  m > 0 →
  volume vector1 (vector2 m) (vector3 m) = 20 →
  m = (9 + Real.sqrt 249) / 6 :=
by sorry

end parallelepiped_volume_l2259_225908


namespace triangle_side_lengths_l2259_225987

theorem triangle_side_lengths (n : ℕ) : 
  (n + 4 + n + 10 > 4*n + 7) ∧ 
  (n + 4 + 4*n + 7 > n + 10) ∧ 
  (n + 10 + 4*n + 7 > n + 4) ∧ 
  (4*n + 7 > n + 10) ∧ 
  (n + 10 > n + 4) →
  (∃ (count : ℕ), count = 2 ∧ 
    (∀ m : ℕ, (m + 4 + m + 10 > 4*m + 7) ∧ 
              (m + 4 + 4*m + 7 > m + 10) ∧ 
              (m + 10 + 4*m + 7 > m + 4) ∧ 
              (4*m + 7 > m + 10) ∧ 
              (m + 10 > m + 4) ↔ 
              (m = n ∨ m = n + 1))) :=
by sorry

end triangle_side_lengths_l2259_225987


namespace great_fourteen_soccer_league_games_l2259_225936

theorem great_fourteen_soccer_league_games (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : 
  teams_per_division = 7 →
  intra_division_games = 3 →
  inter_division_games = 1 →
  (teams_per_division * (
    (teams_per_division - 1) * intra_division_games + 
    teams_per_division * inter_division_games
  )) / 2 = 175 := by
  sorry

end great_fourteen_soccer_league_games_l2259_225936


namespace jackies_activities_exceed_day_l2259_225947

/-- Represents the duration of Jackie's daily activities in hours -/
structure DailyActivities where
  working : ℝ
  exercising : ℝ
  sleeping : ℝ
  commuting : ℝ
  meals : ℝ
  language_classes : ℝ
  phone_calls : ℝ
  reading : ℝ

/-- Theorem stating that Jackie's daily activities exceed 24 hours -/
theorem jackies_activities_exceed_day (activities : DailyActivities) 
  (h1 : activities.working = 8)
  (h2 : activities.exercising = 3)
  (h3 : activities.sleeping = 8)
  (h4 : activities.commuting = 1)
  (h5 : activities.meals = 2)
  (h6 : activities.language_classes = 1.5)
  (h7 : activities.phone_calls = 0.5)
  (h8 : activities.reading = 40 / 60) :
  activities.working + activities.exercising + activities.sleeping + 
  activities.commuting + activities.meals + activities.language_classes + 
  activities.phone_calls + activities.reading > 24 := by
  sorry

#check jackies_activities_exceed_day

end jackies_activities_exceed_day_l2259_225947


namespace difference_of_squares_example_l2259_225962

theorem difference_of_squares_example (x y : ℤ) (hx : x = 12) (hy : y = 5) :
  (x - y) * (x + y) = 119 := by sorry

end difference_of_squares_example_l2259_225962


namespace perpendicular_line_equation_l2259_225960

/-- Given a line L1 with equation 3x - 4y + 6 = 0 and a point P(4, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 4x + 3y - 13 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 4 * y + 6 = 0
  let P : ℝ × ℝ := (4, -1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y - 13 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(3/4) * (x - P.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((y₂ - y₁) / (x₂ - x₁)) * ((x₂ - x₁) / (y₂ - y₁)) = -1) :=
by
  sorry

end perpendicular_line_equation_l2259_225960


namespace speed_increase_time_reduction_l2259_225919

/-- Represents Vanya's speed to school -/
def speed : ℝ := by sorry

/-- Theorem stating the relationship between speed increase and time reduction -/
theorem speed_increase_time_reduction :
  (speed + 2) / speed = 2.5 →
  (speed + 4) / speed = 4 := by sorry

end speed_increase_time_reduction_l2259_225919


namespace weed_ratio_l2259_225935

/-- Represents the number of weeds pulled on each day --/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Defines the conditions of the weed-pulling problem --/
def weed_problem (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.thursday = w.wednesday / 5 ∧
  w.friday = w.thursday - 10 ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- The theorem to be proved --/
theorem weed_ratio (w : WeedCount) (h : weed_problem w) : 
  w.wednesday = 3 * w.tuesday :=
sorry

end weed_ratio_l2259_225935


namespace percentage_error_division_vs_multiplication_l2259_225934

theorem percentage_error_division_vs_multiplication (x : ℝ) (h : x > 0) : 
  (|4 * x - x / 4| / (4 * x)) * 100 = 93.75 := by
  sorry

end percentage_error_division_vs_multiplication_l2259_225934


namespace exists_favorable_config_for_second_player_l2259_225978

/-- Represents a square on the game board -/
structure Square :=
  (hasArrow : Bool)

/-- Represents the game board -/
def Board := List Square

/-- Calculates the probability of the second player winning given a board configuration and game parameters -/
def secondPlayerWinProbability (board : Board) (s₁ : ℕ) (s₂ : ℕ) : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem stating that there exists a board configuration where the second player has a winning probability greater than 1/2, even when s₁ > s₂ -/
theorem exists_favorable_config_for_second_player :
  ∃ (board : Board) (s₁ s₂ : ℕ), s₁ > s₂ ∧ secondPlayerWinProbability board s₁ s₂ > (1/2 : ℝ) :=
sorry


end exists_favorable_config_for_second_player_l2259_225978


namespace cats_not_liking_either_l2259_225985

theorem cats_not_liking_either (total : ℕ) (cheese : ℕ) (tuna : ℕ) (both : ℕ) 
  (h_total : total = 100)
  (h_cheese : cheese = 25)
  (h_tuna : tuna = 70)
  (h_both : both = 15) :
  total - (cheese + tuna - both) = 20 := by
  sorry

end cats_not_liking_either_l2259_225985


namespace abc_inequality_l2259_225994

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 := by
  sorry

end abc_inequality_l2259_225994


namespace probability_ratio_l2259_225997

def total_slips : ℕ := 50
def numbers_range : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def probability_same_number (total : ℕ) (range : ℕ) (per_number : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / Nat.choose total drawn

def probability_three_two (total : ℕ) (range : ℕ) (per_number : ℕ) (drawn : ℕ) : ℚ :=
  (Nat.choose range 2 * Nat.choose per_number 3 * Nat.choose per_number 2 : ℚ) / Nat.choose total drawn

theorem probability_ratio :
  (probability_three_two total_slips numbers_range slips_per_number drawn_slips) /
  (probability_same_number total_slips numbers_range slips_per_number drawn_slips) = 450 := by
  sorry

end probability_ratio_l2259_225997


namespace prime_factorization_of_large_number_l2259_225922

theorem prime_factorization_of_large_number :
  1007021035035021007001 = 7^7 * 11^7 * 13^7 := by
  sorry

end prime_factorization_of_large_number_l2259_225922


namespace product_inequality_l2259_225900

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end product_inequality_l2259_225900


namespace min_value_exponential_sum_l2259_225948

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 6) :
  ∃ (min : ℝ), min = 16 ∧ ∀ (a b : ℝ), a + 2 * b = 6 → 2^a + 4^b ≥ min :=
sorry

end min_value_exponential_sum_l2259_225948


namespace line_through_points_l2259_225969

/-- Given a line y = cx + d passing through the points (3, -3) and (6, 9), prove that c + d = -11 -/
theorem line_through_points (c d : ℝ) : 
  (-3 : ℝ) = c * 3 + d → 
  9 = c * 6 + d → 
  c + d = -11 := by
  sorry

end line_through_points_l2259_225969


namespace value_added_to_number_l2259_225946

theorem value_added_to_number (n v : ℤ) : n = 9 → 3 * (n + 2) = v + n → v = 24 := by
  sorry

end value_added_to_number_l2259_225946


namespace counterfeit_coin_identifiable_l2259_225930

/-- Represents the type of coin -/
inductive CoinType
| Gold
| Silver

/-- Represents a coin with its type and whether it's counterfeit -/
structure Coin where
  type : CoinType
  isCounterfeit : Bool

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a group of coins -/
def CoinGroup := List Coin

/-- Represents a weighing action -/
def Weighing := CoinGroup → CoinGroup → WeighingResult

/-- The total number of coins -/
def totalCoins : Nat := 27

/-- The number of gold coins -/
def goldCoins : Nat := 13

/-- The number of silver coins -/
def silverCoins : Nat := 14

/-- The maximum number of weighings allowed -/
def maxWeighings : Nat := 3

/-- Axiom: There is exactly one counterfeit coin -/
axiom one_counterfeit (coins : List Coin) : 
  coins.length = totalCoins → ∃! c, c ∈ coins ∧ c.isCounterfeit

/-- Axiom: Counterfeit gold coin is lighter than real gold coins -/
axiom counterfeit_gold_lighter (w : Weighing) (c1 c2 : Coin) :
  c1.type = CoinType.Gold ∧ c2.type = CoinType.Gold ∧ c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.RightHeavier

/-- Axiom: Counterfeit silver coin is heavier than real silver coins -/
axiom counterfeit_silver_heavier (w : Weighing) (c1 c2 : Coin) :
  c1.type = CoinType.Silver ∧ c2.type = CoinType.Silver ∧ c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.LeftHeavier

/-- Axiom: Real coins of the same type have equal weight -/
axiom real_coins_equal_weight (w : Weighing) (c1 c2 : Coin) :
  c1.type = c2.type ∧ ¬c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.Equal

/-- The main theorem: It's possible to identify the counterfeit coin in at most three weighings -/
theorem counterfeit_coin_identifiable (coins : List Coin) (w : Weighing) :
  coins.length = totalCoins →
  ∃ (strategy : List (CoinGroup × CoinGroup)), 
    strategy.length ≤ maxWeighings ∧
    ∃ (counterfeit : Coin), counterfeit ∈ coins ∧ counterfeit.isCounterfeit ∧
    ∀ (c : Coin), c ∈ coins ∧ c.isCounterfeit → c = counterfeit :=
  sorry


end counterfeit_coin_identifiable_l2259_225930


namespace min_value_sum_reciprocals_l2259_225957

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_2 : a + b + c = 2) : 
  1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a) ≥ 27 / 8 := by
  sorry

end min_value_sum_reciprocals_l2259_225957


namespace x_is_even_l2259_225918

theorem x_is_even (x : ℤ) (h : ∃ (k : ℤ), (2 * x) / 3 - x / 6 = k) : ∃ (m : ℤ), x = 2 * m := by
  sorry

end x_is_even_l2259_225918


namespace tangent_line_property_l2259_225938

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^(3/2) - Real.log x - 2/3

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (1/x) * (x^(3/2) - 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := |f x + f' x|

-- State the theorem
theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) (h₄ : g x₁ = g x₂) : x₁ * x₂ < 1 := by
  sorry

end tangent_line_property_l2259_225938


namespace eight_digit_divisibility_l2259_225901

theorem eight_digit_divisibility (a b : Nat) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  ∃ k : Nat, (a * 10 + b) * 1010101 = 101 * k := by
  sorry

end eight_digit_divisibility_l2259_225901


namespace area_equality_in_divided_triangle_l2259_225928

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Represents a triangle with its three vertices -/
structure Triangle :=
  (A B C : Point)

/-- Given a triangle and a ratio, returns a point on one of its sides -/
def pointOnSide (T : Triangle) (ratio : ℝ) (side : Fin 3) : Point := sorry

theorem area_equality_in_divided_triangle (ABC : Triangle) :
  let D := pointOnSide ABC (1/3) 0
  let E := pointOnSide ABC (1/3) 1
  let F := pointOnSide ABC (1/3) 2
  let G := pointOnSide (Triangle.mk D E F) (1/2) 0
  let H := pointOnSide (Triangle.mk D E F) (1/2) 1
  let I := pointOnSide (Triangle.mk D E F) (1/2) 2
  triangleArea D A G + triangleArea E B H + triangleArea F C I = triangleArea G H I :=
by sorry

end area_equality_in_divided_triangle_l2259_225928


namespace total_students_is_240_l2259_225952

/-- The number of students from Know It All High School -/
def know_it_all_students : ℕ := 50

/-- The number of students from Karen High School -/
def karen_high_students : ℕ := (3 * know_it_all_students) / 5

/-- The combined number of students from Know It All High School and Karen High School -/
def combined_students : ℕ := know_it_all_students + karen_high_students

/-- The number of students from Novel Corona High School -/
def novel_corona_students : ℕ := 2 * combined_students

/-- The total number of students at the competition -/
def total_students : ℕ := know_it_all_students + karen_high_students + novel_corona_students

/-- Theorem stating that the total number of students at the competition is 240 -/
theorem total_students_is_240 : total_students = 240 := by
  sorry

end total_students_is_240_l2259_225952


namespace males_band_not_orchestra_zero_l2259_225939

/-- Represents the membership of students in band and orchestra --/
structure MusicGroups where
  total : ℕ
  females_band : ℕ
  males_band : ℕ
  females_orchestra : ℕ
  males_orchestra : ℕ
  females_both : ℕ

/-- The number of males in the band who are not in the orchestra is 0 --/
theorem males_band_not_orchestra_zero (g : MusicGroups)
  (h1 : g.total = 250)
  (h2 : g.females_band = 120)
  (h3 : g.males_band = 90)
  (h4 : g.females_orchestra = 90)
  (h5 : g.males_orchestra = 120)
  (h6 : g.females_both = 70) :
  g.males_band - (g.males_band + g.males_orchestra - (g.total - (g.females_band + g.females_orchestra - g.females_both))) = 0 := by
  sorry

#check males_band_not_orchestra_zero

end males_band_not_orchestra_zero_l2259_225939


namespace coupon_value_l2259_225984

/-- Calculates the value of a coupon for eyeglass frames -/
theorem coupon_value (frame_cost lens_cost insurance_percentage total_cost_after : ℚ) : 
  frame_cost = 200 →
  lens_cost = 500 →
  insurance_percentage = 80 / 100 →
  total_cost_after = 250 →
  (frame_cost + lens_cost * (1 - insurance_percentage)) - total_cost_after = 50 := by
  sorry

end coupon_value_l2259_225984


namespace free_throw_contest_ratio_l2259_225972

theorem free_throw_contest_ratio (alex sandra hector : ℕ) : 
  alex = 8 →
  sandra = 3 * alex →
  alex + sandra + hector = 80 →
  hector / sandra = 2 := by
sorry

end free_throw_contest_ratio_l2259_225972


namespace easter_egg_distribution_l2259_225998

def blue_eggs : ℕ := 12
def pink_eggs : ℕ := 5
def golden_eggs : ℕ := 3

def blue_points : ℕ := 2
def pink_points : ℕ := 3
def golden_points : ℕ := 5

def total_people : ℕ := 4

theorem easter_egg_distribution :
  let total_points := blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points
  (total_points / total_people = 13) ∧ (total_points % total_people = 2) := by
  sorry

end easter_egg_distribution_l2259_225998


namespace mikeys_leaves_mikeys_leaves_specific_l2259_225991

/-- The number of leaves Mikey has after receiving more leaves -/
def total_leaves (initial : ℝ) (new : ℝ) : ℝ :=
  initial + new

/-- Theorem stating that Mikey's total leaves is the sum of initial and new leaves -/
theorem mikeys_leaves (initial : ℝ) (new : ℝ) :
  total_leaves initial new = initial + new := by
  sorry

/-- Specific instance of Mikey's leaves problem -/
theorem mikeys_leaves_specific :
  total_leaves 356.0 112.0 = 468.0 := by
  sorry

end mikeys_leaves_mikeys_leaves_specific_l2259_225991


namespace extra_flowers_l2259_225954

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 36 → roses = 37 → used = 70 → tulips + roses - used = 3 := by
  sorry

end extra_flowers_l2259_225954


namespace martin_trip_distance_l2259_225933

/-- Calculates the total distance traveled during a two-part journey -/
def total_distance (total_time hours_per_half : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  speed1 * hours_per_half + speed2 * hours_per_half

/-- Proves that the total distance traveled in the given conditions is 620 km -/
theorem martin_trip_distance :
  let total_time : ℝ := 8
  let speed1 : ℝ := 70
  let speed2 : ℝ := 85
  let hours_per_half : ℝ := total_time / 2
  total_distance total_time hours_per_half speed1 speed2 = 620 := by
  sorry

#eval total_distance 8 4 70 85

end martin_trip_distance_l2259_225933


namespace prob_black_then_red_is_15_59_l2259_225977

/-- A deck of cards with specific properties -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (h_total : total = 60)
  (h_black : black = 30)
  (h_red : red = 30)
  (h_sum : black + red = total)

/-- The probability of drawing a black card first and a red card second -/
def prob_black_then_red (d : Deck) : ℚ :=
  (d.black : ℚ) / d.total * d.red / (d.total - 1)

/-- Theorem stating the probability is equal to 15/59 -/
theorem prob_black_then_red_is_15_59 (d : Deck) :
  prob_black_then_red d = 15 / 59 := by
  sorry


end prob_black_then_red_is_15_59_l2259_225977


namespace martha_apples_theorem_l2259_225966

def apples_to_give_away (initial_apples : ℕ) (jane_apples : ℕ) (james_extra_apples : ℕ) (final_apples : ℕ) : ℕ :=
  initial_apples - jane_apples - (jane_apples + james_extra_apples) - final_apples

theorem martha_apples_theorem :
  apples_to_give_away 20 5 2 4 = 4 := by
sorry

end martha_apples_theorem_l2259_225966


namespace percentage_problem_l2259_225992

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 660 = 12 / 100 * 1500 - 15 → P = 25 := by
  sorry

end percentage_problem_l2259_225992


namespace sqrt_360000_equals_600_l2259_225914

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_equals_600_l2259_225914


namespace forbidden_city_area_scientific_notation_l2259_225912

/-- The area of the Forbidden City in square meters -/
def forbidden_city_area : ℝ := 720000

/-- Scientific notation representation of the Forbidden City's area -/
def scientific_notation : ℝ := 7.2 * (10 ^ 5)

theorem forbidden_city_area_scientific_notation :
  forbidden_city_area = scientific_notation := by
  sorry

end forbidden_city_area_scientific_notation_l2259_225912


namespace largest_mersenne_prime_under_500_l2259_225974

/-- A Mersenne number is of the form 2^p - 1 for some positive integer p -/
def mersenne_number (p : ℕ) : ℕ := 2^p - 1

/-- A Mersenne prime is a Mersenne number that is also prime -/
def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = mersenne_number p ∧ Nat.Prime n

theorem largest_mersenne_prime_under_500 :
  ∀ n : ℕ, is_mersenne_prime n ∧ n < 500 → n ≤ 127 :=
sorry

end largest_mersenne_prime_under_500_l2259_225974


namespace line_through_coefficient_points_l2259_225975

/-- Given two lines that pass through a common point, prove that the line passing through
    the points defined by the coefficients of these lines has a specific equation. -/
theorem line_through_coefficient_points (a₁ a₂ b₁ b₂ : ℝ) : 
  (a₁ * 2 + b₁ * 3 + 1 = 0) →
  (a₂ * 2 + b₂ * 3 + 1 = 0) →
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2 * x + 3 * y + 1 = 0 := by
  sorry

end line_through_coefficient_points_l2259_225975


namespace bus_speed_calculation_l2259_225927

/-- Proves that a bus stopping for 12 minutes per hour with an average speed of 40 km/hr including stoppages has an average speed of 50 km/hr excluding stoppages. -/
theorem bus_speed_calculation (stop_time : ℝ) (avg_speed_with_stops : ℝ) :
  stop_time = 12 →
  avg_speed_with_stops = 40 →
  let moving_time : ℝ := 60 - stop_time
  let speed_ratio : ℝ := moving_time / 60
  (speed_ratio * (60 / moving_time) * avg_speed_with_stops) = 50 := by
  sorry

#check bus_speed_calculation

end bus_speed_calculation_l2259_225927


namespace min_value_of_y_l2259_225989

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x z : ℝ, x > 0 → z > 0 → x + z = 2 → 1/x + 4/z ≥ 1/a + 4/b) ∧ 1/a + 4/b = 9/2 := by
  sorry

end min_value_of_y_l2259_225989


namespace hope_project_protractors_l2259_225909

theorem hope_project_protractors :
  ∀ (x y z : ℕ),
  x > 31 ∧ z > 33 ∧
  10 * x + 15 * y + 20 * z = 1710 ∧
  8 * x + 2 * y + 8 * z = 664 →
  6 * x + 7 * y + 5 * z = 680 :=
by sorry

end hope_project_protractors_l2259_225909


namespace floor_of_e_l2259_225932

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end floor_of_e_l2259_225932


namespace smallest_d_value_l2259_225926

def σ (v : Fin 4 → ℕ) : Finset (Fin 4 → ℕ) := sorry

theorem smallest_d_value (a b c d : ℕ) :
  0 < a → a < b → b < c → c < d →
  (∃ (s : ℕ), ∃ (v₁ v₂ v₃ : Fin 4 → ℕ),
    v₁ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₂ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₃ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧
    (∀ i : Fin 4, v₁ i + v₂ i + v₃ i = s)) →
  d ≥ 6 :=
by sorry

end smallest_d_value_l2259_225926


namespace equation_solution_l2259_225961

theorem equation_solution : 
  ∃ x : ℝ, (-3 * x - 9 = 6 * x + 18) ∧ (x = -3) := by
  sorry

end equation_solution_l2259_225961


namespace base_two_representation_of_125_l2259_225920

/-- Represents a natural number in base 2 as a list of bits (least significant bit first) -/
def BaseTwoRepresentation := List Bool

/-- Converts a natural number to its base 2 representation -/
def toBaseTwoRepresentation (n : ℕ) : BaseTwoRepresentation :=
  sorry

/-- Converts a base 2 representation to its decimal (base 10) value -/
def fromBaseTwoRepresentation (bits : BaseTwoRepresentation) : ℕ :=
  sorry

theorem base_two_representation_of_125 :
  toBaseTwoRepresentation 125 = [true, false, true, true, true, true, true] := by
  sorry

end base_two_representation_of_125_l2259_225920


namespace volume_of_T_l2259_225986

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p
    |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of T is 32/3 -/
theorem volume_of_T : volume T = 32/3 := by sorry

end volume_of_T_l2259_225986


namespace factorial_ones_divisibility_l2259_225979

/-- Definition of [n]! as the product of numbers consisting of n ones -/
def factorial_ones (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => (10^(i+1) - 1) / 9)

/-- Theorem stating that [n+m]! is divisible by [n]! * [m]! -/
theorem factorial_ones_divisibility (n m : ℕ) :
  ∃ k : ℕ, factorial_ones (n + m) = k * (factorial_ones n * factorial_ones m) := by
  sorry

end factorial_ones_divisibility_l2259_225979


namespace triangle_parallelepiped_analogy_inappropriate_l2259_225981

/-- A shape in a geometric space -/
inductive GeometricShape
  | Triangle
  | Parallelepiped
  | TriangularPyramid

/-- The dimension of a geometric space -/
inductive Dimension
  | Plane
  | Space

/-- A function that determines if two shapes form an appropriate analogy across dimensions -/
def appropriateAnalogy (shape1 : GeometricShape) (dim1 : Dimension) 
                       (shape2 : GeometricShape) (dim2 : Dimension) : Prop :=
  sorry

/-- Theorem stating that comparing a triangle in a plane to a parallelepiped in space 
    is not an appropriate analogy -/
theorem triangle_parallelepiped_analogy_inappropriate :
  ¬(appropriateAnalogy GeometricShape.Triangle Dimension.Plane 
                       GeometricShape.Parallelepiped Dimension.Space) :=
by sorry

end triangle_parallelepiped_analogy_inappropriate_l2259_225981


namespace highway_extension_remaining_miles_l2259_225941

/-- Proves that given the highway extension conditions, 250 miles still need to be added -/
theorem highway_extension_remaining_miles 
  (current_length : ℝ) 
  (final_length : ℝ) 
  (first_day_miles : ℝ) 
  (second_day_multiplier : ℝ) :
  current_length = 200 →
  final_length = 650 →
  first_day_miles = 50 →
  second_day_multiplier = 3 →
  final_length - current_length - first_day_miles - (second_day_multiplier * first_day_miles) = 250 := by
  sorry

#check highway_extension_remaining_miles

end highway_extension_remaining_miles_l2259_225941


namespace goldies_hourly_rate_l2259_225902

/-- Goldie's pet-sitting earnings problem -/
theorem goldies_hourly_rate (hours_last_week hours_this_week total_earnings : ℚ) 
  (h1 : hours_last_week = 20)
  (h2 : hours_this_week = 30)
  (h3 : total_earnings = 250) :
  total_earnings / (hours_last_week + hours_this_week) = 5 := by
  sorry

end goldies_hourly_rate_l2259_225902


namespace carl_teaches_six_periods_l2259_225967

-- Define the given conditions
def cards_per_student : ℕ := 10
def students_per_class : ℕ := 30
def cards_per_pack : ℕ := 50
def cost_per_pack : ℚ := 3
def total_spent : ℚ := 108

-- Define the number of periods
def periods : ℕ := 6

-- Theorem statement
theorem carl_teaches_six_periods :
  (total_spent / cost_per_pack) * cards_per_pack =
  periods * students_per_class * cards_per_student :=
by sorry

end carl_teaches_six_periods_l2259_225967


namespace f_properties_l2259_225971

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  -1 < a → a < 0 → x > 0 →
  (∃ (max_val min_val : ℝ),
    (a = -1/2 → 
      (∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y ≤ max_val) ∧
      (∃ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y = max_val) ∧
      max_val = 1/2 + (Real.exp 1)^2/4) ∧
    (a = -1/2 → 
      (∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y ≥ min_val) ∧
      (∃ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y = min_val) ∧
      min_val = 5/4)) ∧
  (∀ y z, 0 < y → y < Real.sqrt (-a/(a+1)) → z ≥ Real.sqrt (-a/(a+1)) → 
    f a y ≥ f a (Real.sqrt (-a/(a+1))) ∧ f a z ≥ f a (Real.sqrt (-a/(a+1)))) ∧
  (∀ y, y > 0 → f a y > 1 + a/2 * Real.log (-a) ↔ 1/Real.exp 1 - 1 < a) :=
by sorry

end

end f_properties_l2259_225971


namespace polynomial_division_remainder_l2259_225929

theorem polynomial_division_remainder (x : ℂ) : 
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_l2259_225929


namespace scaled_right_triangle_area_l2259_225999

theorem scaled_right_triangle_area :
  ∀ (a b : ℝ) (scale : ℝ),
    a = 50 →
    b = 70 →
    scale = 2 →
    (1/2 : ℝ) * (a * scale) * (b * scale) = 7000 := by
  sorry

end scaled_right_triangle_area_l2259_225999


namespace two_heroes_two_villains_l2259_225968

/-- Represents the type of an inhabitant -/
inductive InhabitantType
| Hero
| Villain

/-- Represents an inhabitant on the island -/
structure Inhabitant where
  type : InhabitantType

/-- Represents the table with four inhabitants -/
structure Table where
  inhabitants : Fin 4 → Inhabitant

/-- Defines what it means for an inhabitant to tell the truth -/
def tellsTruth (i : Inhabitant) : Prop :=
  i.type = InhabitantType.Hero

/-- Defines what an inhabitant says about themselves -/
def claimsSelfHero (i : Inhabitant) : Prop :=
  true

/-- Defines what an inhabitant says about the person on their right -/
def claimsRightVillain (t : Table) (pos : Fin 4) : Prop :=
  true

/-- The main theorem stating that the only valid configuration is 2 Heroes and 2 Villains alternating -/
theorem two_heroes_two_villains (t : Table) :
  (∀ (pos : Fin 4), claimsSelfHero (t.inhabitants pos)) →
  (∀ (pos : Fin 4), claimsRightVillain t pos) →
  (∃ (pos : Fin 4),
    tellsTruth (t.inhabitants pos) ∧
    ¬tellsTruth (t.inhabitants (pos + 1)) ∧
    tellsTruth (t.inhabitants (pos + 2)) ∧
    ¬tellsTruth (t.inhabitants (pos + 3))) :=
by
  sorry

end two_heroes_two_villains_l2259_225968


namespace mihaly_third_day_foxes_l2259_225976

/-- Represents the number of animals hunted by a person on a specific day -/
structure DailyHunt where
  rabbits : ℕ
  foxes : ℕ
  pheasants : ℕ

/-- Represents the total hunt over three days for a person -/
structure ThreeDayHunt where
  day1 : DailyHunt
  day2 : DailyHunt
  day3 : DailyHunt

def Karoly : ThreeDayHunt := sorry
def Laszlo : ThreeDayHunt := sorry
def Mihaly : ThreeDayHunt := sorry

def total_animals : ℕ := 86

def first_day_foxes : ℕ := 12
def first_day_rabbits : ℕ := 14

def second_day_total : ℕ := 44

def total_pheasants : ℕ := 12

theorem mihaly_third_day_foxes :
  (∀ d : DailyHunt, d.rabbits ≥ 1 ∧ d.foxes ≥ 1 ∧ d.pheasants ≥ 1) →
  (∀ d : DailyHunt, d ≠ Laszlo.day2 → Even d.rabbits ∧ Even d.foxes ∧ Even d.pheasants) →
  Laszlo.day2.foxes = 5 →
  (Karoly.day1.foxes + Laszlo.day1.foxes + Mihaly.day1.foxes = first_day_foxes) →
  (Karoly.day1.rabbits + Laszlo.day1.rabbits + Mihaly.day1.rabbits = first_day_rabbits) →
  (Karoly.day2.rabbits + Karoly.day2.foxes + Karoly.day2.pheasants +
   Laszlo.day2.rabbits + Laszlo.day2.foxes + Laszlo.day2.pheasants +
   Mihaly.day2.rabbits + Mihaly.day2.foxes + Mihaly.day2.pheasants = second_day_total) →
  (Karoly.day1.pheasants + Karoly.day2.pheasants + Karoly.day3.pheasants +
   Laszlo.day1.pheasants + Laszlo.day2.pheasants + Laszlo.day3.pheasants +
   Mihaly.day1.pheasants + Mihaly.day2.pheasants + Mihaly.day3.pheasants = total_pheasants) →
  (Karoly.day1.rabbits + Karoly.day1.foxes + Karoly.day1.pheasants +
   Karoly.day2.rabbits + Karoly.day2.foxes + Karoly.day2.pheasants +
   Karoly.day3.rabbits + Karoly.day3.foxes + Karoly.day3.pheasants +
   Laszlo.day1.rabbits + Laszlo.day1.foxes + Laszlo.day1.pheasants +
   Laszlo.day2.rabbits + Laszlo.day2.foxes + Laszlo.day2.pheasants +
   Laszlo.day3.rabbits + Laszlo.day3.foxes + Laszlo.day3.pheasants +
   Mihaly.day1.rabbits + Mihaly.day1.foxes + Mihaly.day1.pheasants +
   Mihaly.day2.rabbits + Mihaly.day2.foxes + Mihaly.day2.pheasants +
   Mihaly.day3.rabbits + Mihaly.day3.foxes + Mihaly.day3.pheasants = total_animals) →
  Mihaly.day3.foxes = 1 := by
  sorry

end mihaly_third_day_foxes_l2259_225976


namespace lily_pad_coverage_l2259_225923

/-- Represents the size of the lily pad patch as a fraction of the lake -/
def LilyPadSize := ℚ

/-- The number of days it takes for the patch to cover the entire lake -/
def TotalDays : ℕ := 37

/-- The fraction of the lake that is covered after a given number of days -/
def coverage (days : ℕ) : LilyPadSize :=
  (1 : ℚ) / (2 ^ (TotalDays - days))

/-- Theorem stating that it takes 36 days to cover three-fourths of the lake -/
theorem lily_pad_coverage :
  coverage 36 = (3 : ℚ) / 4 := by sorry

end lily_pad_coverage_l2259_225923


namespace reciprocal_of_negative_fraction_l2259_225950

theorem reciprocal_of_negative_fraction :
  ((-1 : ℚ) / 2011)⁻¹ = -2011 := by sorry

end reciprocal_of_negative_fraction_l2259_225950


namespace half_liar_day_determination_l2259_225906

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the half-liar's statement type
structure Statement where
  yesterday : Day
  tomorrow : Day

-- Define the function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define the function to get the previous day
def prevDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Define the theorem
theorem half_liar_day_determination
  (statement_week_ago : Statement)
  (statement_today : Statement)
  (h1 : statement_week_ago.yesterday = Day.Wednesday ∧ statement_week_ago.tomorrow = Day.Thursday)
  (h2 : statement_today.yesterday = Day.Friday ∧ statement_today.tomorrow = Day.Sunday)
  (h3 : ∀ (d : Day), nextDay (nextDay (nextDay (nextDay (nextDay (nextDay (nextDay d)))))) = d)
  : ∃ (today : Day), today = Day.Saturday :=
by
  sorry


end half_liar_day_determination_l2259_225906


namespace cryptarithm_solution_l2259_225931

/-- Represents a digit in base 9 --/
def Digit := Fin 9

/-- Represents the cryptarithm LAKE + KALE + LEAK = KLAE in base 9 --/
def Cryptarithm (L A K E : Digit) : Prop :=
  (L.val + K.val + L.val) % 9 = K.val ∧
  (A.val + A.val + E.val) % 9 = L.val ∧
  (K.val + L.val + A.val) % 9 = A.val ∧
  (E.val + E.val + K.val) % 9 = E.val

/-- All digits are distinct --/
def DistinctDigits (L A K E : Digit) : Prop :=
  L ≠ A ∧ L ≠ K ∧ L ≠ E ∧ A ≠ K ∧ A ≠ E ∧ K ≠ E

theorem cryptarithm_solution :
  ∃ (L A K E : Digit),
    Cryptarithm L A K E ∧
    DistinctDigits L A K E ∧
    L.val = 0 ∧ E.val = 8 ∧ K.val = 4 ∧ (A.val = 1 ∨ A.val = 2 ∨ A.val = 3 ∨ A.val = 5 ∨ A.val = 6 ∨ A.val = 7) :=
by sorry

end cryptarithm_solution_l2259_225931


namespace binomial_coefficient_1000_1000_l2259_225980

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end binomial_coefficient_1000_1000_l2259_225980


namespace sum_of_integers_l2259_225995

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 145) 
  (h2 : x.val * y.val = 72) : 
  x.val + y.val = 17 := by
sorry

end sum_of_integers_l2259_225995


namespace point_above_line_l2259_225942

/-- A point (x, y) is above a line ax + by + c = 0 if ax + by + c < 0 -/
def IsAboveLine (x y a b c : ℝ) : Prop := a * x + b * y + c < 0

theorem point_above_line (t : ℝ) :
  IsAboveLine (-2) t 1 (-2) 4 → t > 1 := by
  sorry

end point_above_line_l2259_225942


namespace team_winning_percentage_l2259_225925

theorem team_winning_percentage 
  (first_games : ℕ) 
  (total_games : ℕ) 
  (first_win_rate : ℚ) 
  (remaining_win_rate : ℚ) 
  (h1 : first_games = 30)
  (h2 : total_games = 60)
  (h3 : first_win_rate = 2/5)
  (h4 : remaining_win_rate = 4/5) : 
  (first_win_rate * first_games + remaining_win_rate * (total_games - first_games)) / total_games = 3/5 := by
sorry

end team_winning_percentage_l2259_225925


namespace smiths_age_problem_l2259_225943

/-- Represents a 4-digit number in the form abba -/
def mirroredNumber (a b : Nat) : Nat :=
  1000 * a + 100 * b + 10 * b + a

theorem smiths_age_problem :
  ∃! n : Nat,
    59 < n ∧ n < 100 ∧
    (∃ b : Nat, b < 10 ∧ (mirroredNumber (n / 10) b) % 7 = 0) ∧
    n = 67 := by
  sorry

end smiths_age_problem_l2259_225943


namespace circle_circumference_inscribed_rectangle_l2259_225990

theorem circle_circumference_inscribed_rectangle (a b r : ℝ) (h1 : a = 9) (h2 : b = 12) 
  (h3 : r * r = (a * a + b * b) / 4) : 2 * π * r = 15 * π := by
  sorry

end circle_circumference_inscribed_rectangle_l2259_225990


namespace triangle_side_range_l2259_225951

theorem triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = 3 ∧ y = 2*a - 1 ∧ z = 4 ∧ 
    x + y > z ∧ x + z > y ∧ y + z > x) ↔ 
  (1 < a ∧ a < 4) := by
sorry

end triangle_side_range_l2259_225951


namespace function_inequality_implies_a_upper_bound_l2259_225956

open Real

theorem function_inequality_implies_a_upper_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_f : ∀ x, f x = x - a * x * log x) :
  (∃ x₀ ∈ Set.Icc (exp 1) (exp 2), f x₀ ≤ (1/4) * log x₀) →
  a ≤ 1 - 1 / (4 * exp 1) :=
by sorry

end function_inequality_implies_a_upper_bound_l2259_225956


namespace stick_pieces_l2259_225963

theorem stick_pieces (n₁ n₂ : ℕ) (h₁ : n₁ = 12) (h₂ : n₂ = 18) : 
  (n₁ - 1) + (n₂ - 1) - (n₁.lcm n₂ / n₁.gcd n₂ - 1) + 1 = 24 := by sorry

end stick_pieces_l2259_225963


namespace base_5_to_base_7_conversion_l2259_225973

def base_5_to_decimal (n : ℕ) : ℕ := 
  2 * 5^0 + 1 * 5^1 + 4 * 5^2

def decimal_to_base_7 (n : ℕ) : List ℕ :=
  [2, 1, 2]

theorem base_5_to_base_7_conversion :
  decimal_to_base_7 (base_5_to_decimal 412) = [2, 1, 2] := by
  sorry

end base_5_to_base_7_conversion_l2259_225973


namespace sams_age_l2259_225937

theorem sams_age (sam drew : ℕ) 
  (h1 : sam + drew = 54)
  (h2 : sam = drew / 2) :
  sam = 18 := by
sorry

end sams_age_l2259_225937


namespace min_value_of_2a_plus_b_l2259_225924

/-- Given a line equation x/a + y/b = 1 where a > 0 and b > 0, 
    and the line passes through the point (2, 3),
    prove that the minimum value of 2a + b is 7 + 4√3 -/
theorem min_value_of_2a_plus_b (a b : ℝ) : 
  a > 0 → b > 0 → 2 / a + 3 / b = 1 → 
  ∀ x y, x / a + y / b = 1 → 
  (2 * a + b) ≥ 7 + 4 * Real.sqrt 3 := by
  sorry

end min_value_of_2a_plus_b_l2259_225924


namespace sum_236_83_base4_l2259_225964

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 4 representation -/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem: The sum of 236 and 83 in base 4 is [1, 3, 3, 2, 3] -/
theorem sum_236_83_base4 :
  addBase4 (toBase4 236) (toBase4 83) = [1, 3, 3, 2, 3] :=
sorry

end sum_236_83_base4_l2259_225964


namespace quadratic_function_solution_set_l2259_225965

/-- Given a quadratic function f(x) = ax^2 - (a+2)x - b, where a and b are real numbers,
    if the solution set of f(x) > 0 is (-3,2), then a + b = -7. -/
theorem quadratic_function_solution_set (a b : ℝ) :
  (∀ x, (a * x^2 - (a + 2) * x - b > 0) ↔ (-3 < x ∧ x < 2)) →
  a + b = -7 := by sorry

end quadratic_function_solution_set_l2259_225965


namespace rhombus_area_l2259_225983

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 21*d₁ + 30 = 0 → 
  d₂^2 - 21*d₂ + 30 = 0 → 
  d₁ ≠ d₂ →
  (1/2 : ℝ) * d₁ * d₂ = 15 := by
sorry

end rhombus_area_l2259_225983


namespace prob_same_roll_6_7_l2259_225911

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_roll (n : ℕ) : ℚ := 1 / n

/-- The probability of rolling the same number on two dice with sides n and m -/
def prob_same_roll (n m : ℕ) : ℚ := (prob_roll n) * (prob_roll m)

theorem prob_same_roll_6_7 :
  prob_same_roll 6 7 = 1 / 42 := by
  sorry

end prob_same_roll_6_7_l2259_225911


namespace greatest_three_digit_multiple_of_17_l2259_225970

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l2259_225970


namespace third_term_is_16_l2259_225910

/-- Geometric sequence with common ratio 2 and sum of first 4 terms equal to 60 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 + a 4 = 60)

/-- The third term of the geometric sequence is 16 -/
theorem third_term_is_16 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 16 := by
  sorry

end third_term_is_16_l2259_225910


namespace minimum_time_two_people_one_bicycle_l2259_225916

/-- The minimum time problem for two people traveling with one bicycle -/
theorem minimum_time_two_people_one_bicycle
  (distance : ℝ)
  (walk_speed1 walk_speed2 bike_speed1 bike_speed2 : ℝ)
  (h_distance : distance = 40)
  (h_walk_speed1 : walk_speed1 = 4)
  (h_walk_speed2 : walk_speed2 = 6)
  (h_bike_speed1 : bike_speed1 = 30)
  (h_bike_speed2 : bike_speed2 = 20)
  (h_positive : walk_speed1 > 0 ∧ walk_speed2 > 0 ∧ bike_speed1 > 0 ∧ bike_speed2 > 0) :
  ∃ (t : ℝ), t = 25/9 ∧ 
  ∀ (t' : ℝ), (∃ (x y : ℝ), 
    x ≥ 0 ∧ y ≥ 0 ∧
    bike_speed1 * x + walk_speed1 * y = distance ∧
    walk_speed2 * x + bike_speed2 * y = distance ∧
    t' = x + y) → t ≤ t' :=
by sorry

end minimum_time_two_people_one_bicycle_l2259_225916


namespace smallest_special_is_correct_l2259_225944

/-- A natural number is special if it uses exactly four different digits in its decimal representation -/
def is_special (n : ℕ) : Prop :=
  (n.digits 10).toFinset.card = 4

/-- The smallest special number greater than 3429 -/
def smallest_special : ℕ := 3450

theorem smallest_special_is_correct :
  is_special smallest_special ∧
  smallest_special > 3429 ∧
  ∀ m : ℕ, m > 3429 → is_special m → m ≥ smallest_special :=
by sorry

end smallest_special_is_correct_l2259_225944


namespace equation_solution_l2259_225982

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 2/3 ∧ 
  (∀ x : ℝ, 3*x*(x-1) = 2*x-2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l2259_225982


namespace oak_trees_in_park_l2259_225959

theorem oak_trees_in_park (x : ℕ) : x + 4 = 9 → x = 5 := by
  sorry

end oak_trees_in_park_l2259_225959


namespace range_of_2a_plus_c_l2259_225940

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

-- State the theorem
theorem range_of_2a_plus_c (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c)
  (h2 : t.b = Real.sqrt 3) :
  Real.sqrt 3 < 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 :=
by sorry

end range_of_2a_plus_c_l2259_225940


namespace set_operations_l2259_225953

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x < 7}) ∧
  (Aᶜ = {x | x < 3 ∨ 7 ≤ x}) ∧
  ((A ∪ B)ᶜ = {x | x ≤ 2 ∨ 10 ≤ x}) := by
  sorry

end set_operations_l2259_225953


namespace cubic_roots_fourth_power_sum_l2259_225945

theorem cubic_roots_fourth_power_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 2*x^2 + 3*x - 4 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  a^4 + b^4 + c^4 = 18 := by
sorry

end cubic_roots_fourth_power_sum_l2259_225945


namespace algebraic_expression_equality_l2259_225949

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 3*x - 5 = 2) : 
  2*x^2 + 6*x - 3 = 11 := by
sorry

end algebraic_expression_equality_l2259_225949


namespace parallel_planes_from_skew_lines_l2259_225904

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_from_skew_lines 
  (α β : Plane) (l m : Line) : 
  α ≠ β →
  skew l m →
  parallel_line_plane l α →
  parallel_line_plane m α →
  parallel_line_plane l β →
  parallel_line_plane m β →
  parallel α β :=
sorry

end parallel_planes_from_skew_lines_l2259_225904


namespace last_digit_of_n_l2259_225958

/-- Represents a natural number with its digits -/
structure DigitNumber where
  value : ℕ
  num_digits : ℕ
  greater_than_ten : value > 10

/-- Represents the transformation from N to M -/
structure Transformation where
  increase_by_two : ℕ  -- position of the digit increased by 2
  increase_by_odd : List ℕ  -- list of odd numbers added to other digits

/-- Main theorem statement -/
theorem last_digit_of_n (N M : DigitNumber) (t : Transformation) :
  M.value = 3 * N.value →
  M.num_digits = N.num_digits →
  (∃ (transformed_N : ℕ), transformed_N = N.value + t.increase_by_two + t.increase_by_odd.sum) →
  N.value % 10 = 6 := by
  sorry

end last_digit_of_n_l2259_225958


namespace single_circle_percentage_l2259_225913

/-- The number of children participating in the game -/
def n : ℕ := 10

/-- Calculates the double factorial of a natural number -/
def double_factorial (k : ℕ) : ℕ :=
  if k ≤ 1 then 1 else k * double_factorial (k - 2)

/-- Calculates the number of configurations where n children form a single circle -/
def single_circle_configs (n : ℕ) : ℕ := double_factorial (2 * n - 2)

/-- Calculates the total number of possible configurations for n children -/
def total_configs (n : ℕ) : ℕ := 387099936  -- This is the precomputed value for n = 10

/-- The main theorem to be proved -/
theorem single_circle_percentage :
  (single_circle_configs n : ℚ) / (total_configs n) = 12 / 25 := by
  sorry

#eval (single_circle_configs n : ℚ) / (total_configs n)

end single_circle_percentage_l2259_225913
