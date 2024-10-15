import Mathlib

namespace NUMINAMATH_CALUDE_sandy_correct_sums_l292_29239

theorem sandy_correct_sums (total_sums : ℕ) (correct_marks : ℕ) (incorrect_marks : ℕ) (total_marks : ℤ) :
  total_sums = 30 →
  correct_marks = 3 →
  incorrect_marks = 2 →
  total_marks = 45 →
  ∃ (correct_sums : ℕ),
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧
    correct_sums = 21 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l292_29239


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l292_29250

theorem complex_fraction_sum : 
  (Complex.I : ℂ) ^ 2 = -1 → 
  (7 + 3 * Complex.I) / (7 - 3 * Complex.I) + (7 - 3 * Complex.I) / (7 + 3 * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l292_29250


namespace NUMINAMATH_CALUDE_gumball_packages_l292_29223

theorem gumball_packages (package_size : ℕ) (total_consumed : ℕ) 
  (h1 : package_size = 5)
  (h2 : total_consumed = 20) :
  (total_consumed / package_size = 4) ∧ (total_consumed % package_size = 0) := by
  sorry

end NUMINAMATH_CALUDE_gumball_packages_l292_29223


namespace NUMINAMATH_CALUDE_vodka_alcohol_percentage_l292_29222

/-- Calculates the percentage of pure alcohol in vodka -/
theorem vodka_alcohol_percentage
  (total_shots : ℕ)
  (ounces_per_shot : ℚ)
  (pure_alcohol_consumed : ℚ) :
  total_shots = 8 →
  ounces_per_shot = 3/2 →
  pure_alcohol_consumed = 3 →
  (pure_alcohol_consumed / (total_shots * ounces_per_shot)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vodka_alcohol_percentage_l292_29222


namespace NUMINAMATH_CALUDE_speed_calculation_l292_29240

/-- Given a speed v and time t, if increasing the speed by 12 miles per hour
    reduces the time by 1/4, then v = 36 miles per hour. -/
theorem speed_calculation (v t : ℝ) (h : v * t = (v + 12) * (3/4 * t)) : v = 36 :=
sorry

end NUMINAMATH_CALUDE_speed_calculation_l292_29240


namespace NUMINAMATH_CALUDE_debate_team_girls_l292_29295

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) :
  boys = 28 →
  groups = 8 →
  group_size = 4 →
  total = groups * group_size →
  total - boys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_debate_team_girls_l292_29295


namespace NUMINAMATH_CALUDE_coin_collection_problem_l292_29288

/-- Represents the types of coins in the collection --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the collection --/
def CoinCollection.totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters

/-- The total value of the collection in cents --/
def CoinCollection.totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue Coin.Penny +
  c.nickels * coinValue Coin.Nickel +
  c.dimes * coinValue Coin.Dime +
  c.quarters * coinValue Coin.Quarter

theorem coin_collection_problem :
  ∀ c : CoinCollection,
    c.totalCoins = 10 ∧
    c.totalValue = 110 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 2
    →
    c.dimes = 5 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_problem_l292_29288


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2013_l292_29259

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_eq_1 : a 1 = 1
  d : ℝ
  d_neq_0 : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 2)^2 = a 1 * a 5

/-- The 2013th term of the arithmetic sequence is 4025 -/
theorem arithmetic_sequence_2013 (seq : ArithmeticSequence) : seq.a 2013 = 4025 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2013_l292_29259


namespace NUMINAMATH_CALUDE_expression_simplification_l292_29245

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/2) (hy : y = 2) : 
  6 * (x^2 - (1/3) * x * y) - 3 * (x^2 - x * y) - 2 * x^2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l292_29245


namespace NUMINAMATH_CALUDE_special_square_area_l292_29248

/-- A square with special points and segments -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The distance BS
  bs : ℝ
  -- The distance PS
  ps : ℝ
  -- Assumption that BS = 8
  bs_eq : bs = 8
  -- Assumption that PS = 9
  ps_eq : ps = 9
  -- Assumption that BP and DQ intersect perpendicularly
  perpendicular : True

/-- The area of a SpecialSquare is 136 -/
theorem special_square_area (sq : SpecialSquare) : sq.side ^ 2 = 136 := by
  sorry

#check special_square_area

end NUMINAMATH_CALUDE_special_square_area_l292_29248


namespace NUMINAMATH_CALUDE_tyler_sanctuary_species_l292_29235

/-- The number of pairs of birds per species in Tyler's sanctuary -/
def pairs_per_species : ℕ := 7

/-- The total number of pairs of birds in Tyler's sanctuary -/
def total_pairs : ℕ := 203

/-- The number of endangered bird species in Tyler's sanctuary -/
def num_species : ℕ := total_pairs / pairs_per_species

theorem tyler_sanctuary_species :
  num_species = 29 :=
sorry

end NUMINAMATH_CALUDE_tyler_sanctuary_species_l292_29235


namespace NUMINAMATH_CALUDE_max_m_value_inequality_solution_l292_29290

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for the maximum value of m
theorem max_m_value : 
  (∃ m : ℝ, ∀ x : ℝ, f x - m ≥ 0 ∧ ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f x - m' ≥ 0) → 
  (∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, f x - m ≥ 0 ∧ ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f x - m' ≥ 0) :=
sorry

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | |x - 3| - 2*x ≤ 4} = {x : ℝ | x ≥ -1/3} :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_solution_l292_29290


namespace NUMINAMATH_CALUDE_investment_of_c_is_120000_l292_29256

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℕ
  profitShare : ℕ

/-- Calculates the investment of partner C given the investments and profit shares of A and B -/
def calculateInvestmentC (a : Partner) (b : Partner) (profitShareDiffAC : ℕ) : ℕ :=
  let profitShareA := a.investment * b.profitShare / b.investment
  let profitShareC := profitShareA + profitShareDiffAC
  profitShareC * b.investment / b.profitShare

/-- Theorem stating that given the problem conditions, C's investment is 120000 -/
theorem investment_of_c_is_120000 : 
  let a : Partner := ⟨8000, 0⟩
  let b : Partner := ⟨10000, 1700⟩
  let profitShareDiffAC := 680
  calculateInvestmentC a b profitShareDiffAC = 120000 := by
  sorry

#eval calculateInvestmentC ⟨8000, 0⟩ ⟨10000, 1700⟩ 680

end NUMINAMATH_CALUDE_investment_of_c_is_120000_l292_29256


namespace NUMINAMATH_CALUDE_pick_two_different_colors_custom_deck_l292_29280

/-- A custom deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)
  (cards_per_suit : ℕ)

/-- The number of ways to pick two different cards of different colors -/
def pick_two_different_colors (d : Deck) : ℕ :=
  d.total_cards * (d.cards_per_suit * d.red_suits)

/-- Theorem stating the number of ways to pick two different cards of different colors -/
theorem pick_two_different_colors_custom_deck :
  ∃ (d : Deck), 
    d.total_cards = 60 ∧
    d.num_suits = 4 ∧
    d.red_suits = 2 ∧
    d.black_suits = 2 ∧
    d.cards_per_suit = 15 ∧
    pick_two_different_colors d = 1800 := by
  sorry

end NUMINAMATH_CALUDE_pick_two_different_colors_custom_deck_l292_29280


namespace NUMINAMATH_CALUDE_sibling_pair_implies_a_gt_one_l292_29249

/-- A point pair (x₁, y₁) and (x₂, y₂) is a "sibling point pair" for a function f
    if they both lie on the graph of f and are symmetric about the origin. -/
def is_sibling_point_pair (f : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  f x₁ = y₁ ∧ f x₂ = y₂ ∧ x₁ = -x₂ ∧ y₁ = -y₂

/-- The function f(x) = a^x - x - a has only one sibling point pair. -/
def has_unique_sibling_pair (a : ℝ) : Prop :=
  ∃! (x₁ y₁ x₂ y₂ : ℝ), is_sibling_point_pair (fun x => a^x - x - a) x₁ y₁ x₂ y₂

theorem sibling_pair_implies_a_gt_one (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) 
    (h₃ : has_unique_sibling_pair a) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_sibling_pair_implies_a_gt_one_l292_29249


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l292_29246

/-- A conic section with foci F₁ and F₂ -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Point on a conic section -/
def PointOnConic (Γ : ConicSection) := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of a conic section -/
def eccentricity (Γ : ConicSection) : ℝ := sorry

/-- Theorem: The eccentricity of a conic section with the given property is either 1/2 or 3/2 -/
theorem conic_section_eccentricity (Γ : ConicSection) 
  (P : PointOnConic Γ)
  (h : ∃ (k : ℝ), k > 0 ∧ 
       distance P Γ.F₁ = 4 * k ∧ 
       distance Γ.F₁ Γ.F₂ = 3 * k ∧ 
       distance P Γ.F₂ = 2 * k) :
  eccentricity Γ = 1/2 ∨ eccentricity Γ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l292_29246


namespace NUMINAMATH_CALUDE_arrangement_count_is_7200_l292_29255

/-- The number of consonants in the word "ИНТЕГРАЛ" -/
def num_consonants : ℕ := 5

/-- The number of vowels in the word "ИНТЕГРАЛ" -/
def num_vowels : ℕ := 3

/-- The total number of letters in the word "ИНТЕГРАЛ" -/
def total_letters : ℕ := num_consonants + num_vowels

/-- The number of positions that must be occupied by consonants -/
def required_consonant_positions : ℕ := 3

/-- The number of remaining positions after placing consonants in required positions -/
def remaining_positions : ℕ := total_letters - required_consonant_positions

/-- The number of ways to arrange the letters in "ИНТЕГРАЛ" with consonants in specific positions -/
def arrangement_count : ℕ := 
  (num_consonants.factorial / (num_consonants - required_consonant_positions).factorial) * 
  remaining_positions.factorial

theorem arrangement_count_is_7200 : arrangement_count = 7200 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_7200_l292_29255


namespace NUMINAMATH_CALUDE_fraction_calculation_l292_29209

theorem fraction_calculation : (17/5) + (-23/8) - (-28/5) - (1/8) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l292_29209


namespace NUMINAMATH_CALUDE_cricket_run_rate_l292_29271

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (initial_overs : ℕ) (initial_run_rate : ℚ) (target_runs : ℕ) : ℚ :=
  let remaining_overs := total_overs - initial_overs
  let initial_runs := initial_run_rate * initial_overs
  let remaining_runs := target_runs - initial_runs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario. -/
theorem cricket_run_rate : required_run_rate 60 10 (32/10) 282 = 5 := by
  sorry


end NUMINAMATH_CALUDE_cricket_run_rate_l292_29271


namespace NUMINAMATH_CALUDE_angle_and_complement_differ_by_20_l292_29275

theorem angle_and_complement_differ_by_20 (α : ℝ) : 
  α - (90 - α) = 20 → α = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_and_complement_differ_by_20_l292_29275


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l292_29253

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  x * y = 1 ∧ ∀ z : ℚ, x * z = 1 → z = y :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l292_29253


namespace NUMINAMATH_CALUDE_total_gas_used_l292_29229

def gas_usage : List Float := [0.02, 0.015, 0.01, 0.03, 0.005, 0.025, 0.008, 0.018, 0.012, 0.005, 0.014, 0.01]

theorem total_gas_used (gas_usage : List Float) :
  gas_usage.sum = 0.172 := by
  sorry

#eval gas_usage.sum

end NUMINAMATH_CALUDE_total_gas_used_l292_29229


namespace NUMINAMATH_CALUDE_algorithm_characteristic_is_determinacy_l292_29202

-- Define the concept of an algorithm step
structure AlgorithmStep where
  definite : Bool
  executable : Bool
  yieldsDefiniteResult : Bool

-- Define the characteristic of determinacy
def isDeterminacy (step : AlgorithmStep) : Prop :=
  step.definite ∧ step.executable ∧ step.yieldsDefiniteResult

-- Theorem statement
theorem algorithm_characteristic_is_determinacy (step : AlgorithmStep) :
  step.definite ∧ step.executable ∧ step.yieldsDefiniteResult → isDeterminacy step :=
by
  sorry

#check algorithm_characteristic_is_determinacy

end NUMINAMATH_CALUDE_algorithm_characteristic_is_determinacy_l292_29202


namespace NUMINAMATH_CALUDE_scheme_probability_l292_29296

theorem scheme_probability (p_both : ℝ) (h1 : p_both = 0.3) :
  1 - (1 - p_both) * (1 - p_both) = 0.51 := by
sorry

end NUMINAMATH_CALUDE_scheme_probability_l292_29296


namespace NUMINAMATH_CALUDE_mrs_hilt_pennies_l292_29281

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" + 
  nickels * coin_value "nickel" + 
  dimes * coin_value "dime"

theorem mrs_hilt_pennies : 
  ∃ (p : ℕ), 
    total_value p 2 2 - total_value 4 1 1 = 13 ∧ 
    p = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pennies_l292_29281


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l292_29274

theorem roots_quadratic_equation (m : ℝ) (a b : ℝ) (s t : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b^2)^2 - s*(a^2 + 1/b^2) + t = 0) →
  ((b^2 + 1/a^2)^2 - s*(b^2 + 1/a^2) + t = 0) →
  t = 100/9 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l292_29274


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l292_29215

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 - 12)/p = (q^3 - 12)/q ∧ (q^3 - 12)/q = (r^3 - 12)/r) : 
  p^3 + q^3 + r^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l292_29215


namespace NUMINAMATH_CALUDE_sin_transformation_l292_29273

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l292_29273


namespace NUMINAMATH_CALUDE_hazel_caught_24_salmons_l292_29258

/-- Represents the number of salmons caught by Hazel and her father -/
structure FishingTrip where
  total : ℕ
  father : ℕ

/-- Calculates the number of salmons Hazel caught -/
def hazel_catch (trip : FishingTrip) : ℕ :=
  trip.total - trip.father

/-- Theorem: Given the conditions of the fishing trip, prove that Hazel caught 24 salmons -/
theorem hazel_caught_24_salmons (trip : FishingTrip)
  (h1 : trip.total = 51)
  (h2 : trip.father = 27) :
  hazel_catch trip = 24 := by
sorry

end NUMINAMATH_CALUDE_hazel_caught_24_salmons_l292_29258


namespace NUMINAMATH_CALUDE_rob_pennies_l292_29251

/-- The number of pennies Rob has -/
def num_pennies : ℕ := 12

/-- The number of quarters Rob has -/
def num_quarters : ℕ := 7

/-- The number of dimes Rob has -/
def num_dimes : ℕ := 3

/-- The number of nickels Rob has -/
def num_nickels : ℕ := 5

/-- The total amount Rob has in cents -/
def total_amount : ℕ := 242

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

theorem rob_pennies :
  num_quarters * quarter_value + num_dimes * dime_value + num_nickels * nickel_value + num_pennies * penny_value = total_amount :=
by sorry

end NUMINAMATH_CALUDE_rob_pennies_l292_29251


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l292_29266

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*k*x + 3*k^2 + 2*k = 0) ↔ -1 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l292_29266


namespace NUMINAMATH_CALUDE_dodge_truck_count_l292_29279

/-- The number of vehicles in the Taco Castle parking lot -/
structure VehicleCount where
  dodge : ℕ
  ford : ℕ
  toyota : ℕ
  volkswagen : ℕ
  honda : ℕ
  chevrolet : ℕ

/-- The relationships between different vehicle types in the parking lot -/
def valid_count (v : VehicleCount) : Prop :=
  v.ford = v.dodge / 3 ∧
  v.ford = 2 * v.toyota ∧
  v.volkswagen = v.toyota / 2 ∧
  v.honda = (3 * v.ford) / 4 ∧
  v.chevrolet = (2 * v.honda) / 3 ∧
  v.volkswagen = 5

theorem dodge_truck_count (v : VehicleCount) (h : valid_count v) : v.dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_dodge_truck_count_l292_29279


namespace NUMINAMATH_CALUDE_line_through_points_l292_29252

/-- Given a line x = 6y + 5 passing through points (m, n) and (m + 2, n + p), prove p = 1/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 6 * n + 5) ∧ (m + 2 = 6 * (n + p) + 5) → p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l292_29252


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l292_29213

theorem quadratic_equation_root (m : ℝ) : 
  ((-1 : ℝ)^2 + m * (-1) - 4 = 0) → 
  ∃ (x : ℝ), x ≠ -1 ∧ x^2 + m*x - 4 = 0 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l292_29213


namespace NUMINAMATH_CALUDE_not_divisible_by_67_l292_29298

theorem not_divisible_by_67 (x y : ℕ) 
  (h1 : ¬ 67 ∣ x) 
  (h2 : ¬ 67 ∣ y) 
  (h3 : 67 ∣ (7 * x + 32 * y)) : 
  ¬ 67 ∣ (10 * x + 17 * y + 1) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_67_l292_29298


namespace NUMINAMATH_CALUDE_middle_part_value_l292_29219

theorem middle_part_value (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ a * x + b * x + c * x = total ∧ b * x = 40 :=
by sorry

end NUMINAMATH_CALUDE_middle_part_value_l292_29219


namespace NUMINAMATH_CALUDE_base5_123_equals_38_l292_29269

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Theorem: The base-5 number 123₍₅₎ is equal to the decimal number 38 --/
theorem base5_123_equals_38 : base5ToDecimal 1 2 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_base5_123_equals_38_l292_29269


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l292_29264

/-- Given a parabola y = ax^2 + 4 that is tangent to the line y = 3x + 1, prove that a = 3/4 -/
theorem parabola_tangent_to_line (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 = 3 * x + 1 ∧ 
   ∀ y : ℝ, y ≠ x → a * y^2 + 4 ≠ 3 * y + 1) →
  a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l292_29264


namespace NUMINAMATH_CALUDE_remainder_104_pow_2006_mod_29_l292_29206

theorem remainder_104_pow_2006_mod_29 : 104^2006 % 29 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_104_pow_2006_mod_29_l292_29206


namespace NUMINAMATH_CALUDE_chess_game_probability_l292_29218

theorem chess_game_probability (p_not_lose p_draw : ℝ) 
  (h1 : p_not_lose = 0.8) 
  (h2 : p_draw = 0.5) : 
  p_not_lose - p_draw = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l292_29218


namespace NUMINAMATH_CALUDE_combinatorial_equality_l292_29204

theorem combinatorial_equality (n : ℕ) : 
  (n.choose 2) * 2 = 42 → n.choose 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_equality_l292_29204


namespace NUMINAMATH_CALUDE_circle_radius_in_square_l292_29241

theorem circle_radius_in_square (side_length : ℝ) (l_shape_ratio : ℝ) : 
  side_length = 144 →
  l_shape_ratio = 5/18 →
  let total_area := side_length^2
  let l_shape_area := l_shape_ratio * total_area
  let center_square_area := total_area - 4 * l_shape_area
  let center_square_side := Real.sqrt center_square_area
  let radius := center_square_side / 2
  radius = 61.2 := by sorry

end NUMINAMATH_CALUDE_circle_radius_in_square_l292_29241


namespace NUMINAMATH_CALUDE_max_children_to_movies_l292_29292

def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_budget : ℕ := 35

theorem max_children_to_movies :
  (total_budget - adult_ticket_cost) / child_ticket_cost = 9 :=
sorry

end NUMINAMATH_CALUDE_max_children_to_movies_l292_29292


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l292_29244

theorem sin_2alpha_value (α : Real) 
  (h1 : α > 0 ∧ α < Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 4 * Real.cos α + 1 = 0) : 
  Real.sin (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l292_29244


namespace NUMINAMATH_CALUDE_population_growth_rate_l292_29294

/-- Given that a population increases by 90 persons in 30 minutes,
    prove that it takes 20 seconds for one person to be added. -/
theorem population_growth_rate (increase : ℕ) (time_minutes : ℕ) (time_seconds : ℕ) :
  increase = 90 →
  time_minutes = 30 →
  time_seconds = time_minutes * 60 →
  time_seconds / increase = 20 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l292_29294


namespace NUMINAMATH_CALUDE_shopping_mall_entrances_exits_l292_29287

theorem shopping_mall_entrances_exits (n : ℕ) (h : n = 4) :
  (n * (n - 1) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_shopping_mall_entrances_exits_l292_29287


namespace NUMINAMATH_CALUDE_fraction_equals_875_l292_29216

theorem fraction_equals_875 (a : ℕ+) (h : (a : ℚ) / ((a : ℚ) + 35) = 875 / 1000) : 
  a = 245 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_875_l292_29216


namespace NUMINAMATH_CALUDE_tom_payment_tom_paid_1908_l292_29236

/-- Calculates the total amount Tom paid to the shopkeeper after discount -/
theorem tom_payment (apple_kg : ℕ) (apple_rate : ℕ) (mango_kg : ℕ) (mango_rate : ℕ) 
                    (grape_kg : ℕ) (grape_rate : ℕ) (discount_percent : ℕ) : ℕ :=
  let total_cost := apple_kg * apple_rate + mango_kg * mango_rate + grape_kg * grape_rate
  let discount := total_cost * discount_percent / 100
  total_cost - discount

/-- Proves that Tom paid 1908 to the shopkeeper -/
theorem tom_paid_1908 : 
  tom_payment 8 70 9 90 5 150 10 = 1908 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_tom_paid_1908_l292_29236


namespace NUMINAMATH_CALUDE_circuit_board_count_l292_29208

/-- The number of circuit boards that fail verification -/
def failed_boards : ℕ := 64

/-- The fraction of boards that pass verification but are faulty -/
def faulty_fraction : ℚ := 1 / 8

/-- The total number of faulty boards -/
def total_faulty : ℕ := 456

/-- The total number of circuit boards in the group -/
def total_boards : ℕ := 3200

theorem circuit_board_count :
  (failed_boards : ℚ) + faulty_fraction * (total_boards - failed_boards : ℚ) = total_faulty ∧
  total_boards = failed_boards + (total_faulty - failed_boards) / faulty_fraction := by
  sorry

end NUMINAMATH_CALUDE_circuit_board_count_l292_29208


namespace NUMINAMATH_CALUDE_seating_arrangements_l292_29277

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 10 people in a row. -/
def totalArrangements : ℕ := factorial 10

/-- The number of arrangements with 3 specific people in consecutive seats. -/
def threeConsecutive : ℕ := factorial 8 * factorial 3

/-- The number of arrangements with 2 specific people next to each other. -/
def twoTogether : ℕ := factorial 9 * factorial 2

/-- The number of arrangements satisfying both conditions. -/
def bothConditions : ℕ := factorial 7 * factorial 3 * factorial 2

/-- The number of valid seating arrangements. -/
def validArrangements : ℕ := totalArrangements - threeConsecutive - twoTogether + bothConditions

theorem seating_arrangements :
  validArrangements = 2685600 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l292_29277


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l292_29272

theorem garage_sale_pricing (total_items : ℕ) (n : ℕ) : 
  total_items = 34 →
  n = (total_items - 20) →
  n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l292_29272


namespace NUMINAMATH_CALUDE_transaction_outcome_l292_29232

theorem transaction_outcome : 
  let house_sell := 15000
  let store_sell := 14000
  let vehicle_sell := 18000
  let house_loss_percent := 25
  let store_gain_percent := 16.67
  let vehicle_gain_percent := 12.5
  
  let house_cost := house_sell / (1 - house_loss_percent / 100)
  let store_cost := store_sell / (1 + store_gain_percent / 100)
  let vehicle_cost := vehicle_sell / (1 + vehicle_gain_percent / 100)
  
  let total_cost := house_cost + store_cost + vehicle_cost
  let total_sell := house_sell + store_sell + vehicle_sell
  
  total_cost - total_sell = 1000 := by sorry

end NUMINAMATH_CALUDE_transaction_outcome_l292_29232


namespace NUMINAMATH_CALUDE_count_paths_l292_29210

/-- The number of paths on a 6x5 grid from A to B with specific conditions -/
def num_paths : ℕ := 252

/-- The width of the grid -/
def grid_width : ℕ := 6

/-- The height of the grid -/
def grid_height : ℕ := 5

/-- The total number of moves required -/
def total_moves : ℕ := 11

/-- Theorem stating the number of paths under given conditions -/
theorem count_paths :
  num_paths = Nat.choose (total_moves - 1) grid_height :=
sorry

end NUMINAMATH_CALUDE_count_paths_l292_29210


namespace NUMINAMATH_CALUDE_bag_price_with_discount_l292_29254

theorem bag_price_with_discount (selling_price : ℝ) (discount_percentage : ℝ) 
  (h1 : selling_price = 120)
  (h2 : discount_percentage = 4) : 
  selling_price / (1 - discount_percentage / 100) = 125 := by
  sorry

end NUMINAMATH_CALUDE_bag_price_with_discount_l292_29254


namespace NUMINAMATH_CALUDE_odd_function_implies_m_zero_l292_29201

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + 3mx^2 + nx + m^2 -/
def f (m n : ℝ) (x : ℝ) : ℝ :=
  x^3 + 3*m*x^2 + n*x + m^2

theorem odd_function_implies_m_zero (m n : ℝ) :
  IsOdd (f m n) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_zero_l292_29201


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l292_29203

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; -3, -10]

theorem matrix_sum_proof :
  A + B = !![(-2 : ℤ), 5; -1, -5] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l292_29203


namespace NUMINAMATH_CALUDE_smallest_n_for_square_root_96n_l292_29262

theorem smallest_n_for_square_root_96n (n : ℕ) : 
  (∃ k : ℕ, k * k = 96 * n) → n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_root_96n_l292_29262


namespace NUMINAMATH_CALUDE_kids_to_adult_ticket_ratio_l292_29243

def admission_price : ℝ := 30
def group_size : ℕ := 10
def num_children : ℕ := 4
def num_adults : ℕ := group_size - num_children
def discount_rate : ℝ := 0.2
def soda_price : ℝ := 5
def total_paid : ℝ := 197

def adult_ticket_price : ℝ := admission_price

theorem kids_to_adult_ticket_ratio :
  ∃ (kids_ticket_price : ℝ),
    kids_ticket_price > 0 ∧
    adult_ticket_price > 0 ∧
    (1 - discount_rate) * (num_adults * adult_ticket_price + num_children * kids_ticket_price) + soda_price = total_paid ∧
    kids_ticket_price / adult_ticket_price = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_kids_to_adult_ticket_ratio_l292_29243


namespace NUMINAMATH_CALUDE_dave_total_rides_l292_29297

/-- The number of rides Dave took on the first day -/
def first_day_rides : ℕ := 4

/-- The number of rides Dave took on the second day -/
def second_day_rides : ℕ := 3

/-- The total number of rides Dave took over two days -/
def total_rides : ℕ := first_day_rides + second_day_rides

theorem dave_total_rides : total_rides = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_total_rides_l292_29297


namespace NUMINAMATH_CALUDE_abc_positive_l292_29214

/-- A quadratic function y = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  opens_upwards : a > 0
  has_two_real_roots : b^2 > 4*a*c
  right_root_larger : ∃ (r₁ r₂ : ℝ), r₁ < 0 ∧ r₂ > 0 ∧ |r₂| > |r₁| ∧
    a*r₁^2 + b*r₁ + c = 0 ∧ a*r₂^2 + b*r₂ + c = 0

/-- Theorem: For a quadratic function with the given properties, abc > 0 -/
theorem abc_positive (f : QuadraticFunction) : f.a * f.b * f.c > 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_positive_l292_29214


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l292_29285

theorem trigonometric_equation_solution (x : ℝ) :
  (∃ (k : ℤ), x = 2 * π * k / 3) ∨ 
  (∃ (n : ℤ), x = π * (4 * n + 1) / 6) ↔ 
  (Real.cos (3 * x / 2) ≠ 0 ∧ 
   Real.sin ((3 * x - 7 * π) / 2) * Real.cos ((π - 3 * x) / 2) = 
   Real.arccos (3 * x / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l292_29285


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l292_29270

theorem similar_triangles_leg_length (x : ℝ) : x > 0 →
  (12 : ℝ) / x = 9 / 7 → x = 84 / 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l292_29270


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l292_29260

/-- A parabola defined by y = 3x² - 6x + c -/
def parabola (x y c : ℝ) : Prop := y = 3 * x^2 - 6 * x + c

/-- Three points on the parabola -/
def point_A (y₁ c : ℝ) : Prop := parabola (-3) y₁ c
def point_B (y₂ c : ℝ) : Prop := parabola (-1) y₂ c
def point_C (y₃ c : ℝ) : Prop := parabola 5 y₃ c

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_point_relationship (y₁ y₂ y₃ c : ℝ) 
  (hA : point_A y₁ c) (hB : point_B y₂ c) (hC : point_C y₃ c) :
  y₁ = y₃ ∧ y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l292_29260


namespace NUMINAMATH_CALUDE_inequality_equivalence_l292_29231

theorem inequality_equivalence (x : ℝ) : (x - 3) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l292_29231


namespace NUMINAMATH_CALUDE_bobs_cycling_wins_l292_29247

/-- The minimum number of additional weeks Bob must win first place to afford a puppy -/
theorem bobs_cycling_wins (puppy_cost : ℕ) (initial_wins : ℕ) (prize_money : ℕ) 
  (h1 : puppy_cost = 1000)
  (h2 : initial_wins = 2)
  (h3 : prize_money = 100) : 
  (puppy_cost - initial_wins * prize_money) / prize_money = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobs_cycling_wins_l292_29247


namespace NUMINAMATH_CALUDE_average_of_pqrs_l292_29263

theorem average_of_pqrs (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_pqrs_l292_29263


namespace NUMINAMATH_CALUDE_medical_team_formation_plans_l292_29230

theorem medical_team_formation_plans (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 5)
  (h2 : female_doctors = 4) :
  (Nat.choose male_doctors 1 * Nat.choose female_doctors 2) +
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_formation_plans_l292_29230


namespace NUMINAMATH_CALUDE_expression_evaluation_l292_29278

theorem expression_evaluation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

#eval 14 + 19 + 29

end NUMINAMATH_CALUDE_expression_evaluation_l292_29278


namespace NUMINAMATH_CALUDE_decimal_437_equals_fraction_l292_29234

/-- The decimal representation of 0.4̄37 as a rational number -/
def decimal_437 : ℚ := 437/990 - 4/990

/-- The fraction 43693/99900 -/
def fraction_43693_99900 : ℚ := 43693/99900

theorem decimal_437_equals_fraction : 
  decimal_437 = fraction_43693_99900 ∧ 
  (∀ n d : ℕ, n ≠ 0 ∧ d ≠ 0 → fraction_43693_99900 = n / d → n = 43693 ∧ d = 99900) := by
  sorry

#check decimal_437_equals_fraction

end NUMINAMATH_CALUDE_decimal_437_equals_fraction_l292_29234


namespace NUMINAMATH_CALUDE_pencil_cost_l292_29233

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (cost_per_pencil : ℕ) : 
  total_money = 50 → 
  num_pencils = 10 → 
  total_money = num_pencils * cost_per_pencil → 
  cost_per_pencil = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l292_29233


namespace NUMINAMATH_CALUDE_teagan_savings_proof_l292_29228

def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def penny_value : ℚ := 0.01

def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330
def total_savings : ℚ := 40

def teagan_pennies : ℕ := 200

theorem teagan_savings_proof :
  (rex_nickels : ℚ) * nickel_value + (toni_dimes : ℚ) * dime_value + (teagan_pennies : ℚ) * penny_value = total_savings :=
by sorry

end NUMINAMATH_CALUDE_teagan_savings_proof_l292_29228


namespace NUMINAMATH_CALUDE_bobby_adult_jumps_per_second_l292_29289

/-- Bobby's jumping ability as a child and adult -/
def bobby_jumping (child_jumps_per_minute : ℕ) (additional_jumps_per_minute : ℕ) : Prop :=
  let adult_jumps_per_minute := child_jumps_per_minute + additional_jumps_per_minute
  let adult_jumps_per_second := adult_jumps_per_minute / 60
  adult_jumps_per_second = 1

/-- Theorem: Bobby can jump 1 time per second as an adult -/
theorem bobby_adult_jumps_per_second :
  bobby_jumping 30 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_adult_jumps_per_second_l292_29289


namespace NUMINAMATH_CALUDE_smith_B_students_l292_29282

/-- The number of students who received a "B" in Mrs. Smith's class -/
def students_with_B_smith (
  jacob_total : ℕ
  ) (jacob_B : ℕ
  ) (smith_total : ℕ
  ) : ℕ :=
  (smith_total * jacob_B) / jacob_total

theorem smith_B_students (
  jacob_total : ℕ
  ) (jacob_B : ℕ
  ) (smith_total : ℕ
  ) (h1 : jacob_total = 20
  ) (h2 : jacob_B = 8
  ) (h3 : smith_total = 30
  ) : students_with_B_smith jacob_total jacob_B smith_total = 12 := by
  sorry

end NUMINAMATH_CALUDE_smith_B_students_l292_29282


namespace NUMINAMATH_CALUDE_potato_bag_weight_l292_29293

theorem potato_bag_weight (current_weight : ℝ) (h : current_weight = 12) :
  ∃ (original_weight : ℝ), original_weight / 2 = current_weight ∧ original_weight = 24 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l292_29293


namespace NUMINAMATH_CALUDE_last_remaining_100_l292_29268

def last_remaining (n : ℕ) : ℕ :=
  if n ≤ 1 then n else
  let m := n / 2
  2 * (if m % 2 = 0 then last_remaining m else m + 1 - last_remaining m)

theorem last_remaining_100 : last_remaining 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_100_l292_29268


namespace NUMINAMATH_CALUDE_fold_cut_unfold_result_l292_29283

/-- Represents a square sheet of paper with two sides --/
structure Sheet :=
  (side_length : ℝ)
  (white_side : Bool)
  (gray_side : Bool)

/-- Represents a fold on the sheet --/
inductive Fold
  | Vertical
  | Horizontal

/-- Represents a cut on the folded sheet --/
structure Cut :=
  (size : ℝ)

/-- The result of unfolding the sheet after folding and cutting --/
structure UnfoldedResult :=
  (num_cutouts : ℕ)
  (symmetric : Bool)

/-- Function to fold the sheet --/
def fold_sheet (s : Sheet) (f : Fold) : Sheet :=
  sorry

/-- Function to cut the folded sheet --/
def cut_sheet (s : Sheet) (c : Cut) : Sheet :=
  sorry

/-- Function to unfold the sheet --/
def unfold_sheet (s : Sheet) : UnfoldedResult :=
  sorry

/-- Theorem stating the result of folding twice, cutting, and unfolding --/
theorem fold_cut_unfold_result (s : Sheet) (f1 f2 : Fold) (c : Cut) :
  let folded := fold_sheet (fold_sheet s f1) f2
  let cut := cut_sheet folded c
  let result := unfold_sheet cut
  result.num_cutouts = 4 ∧ result.symmetric = true :=
sorry

end NUMINAMATH_CALUDE_fold_cut_unfold_result_l292_29283


namespace NUMINAMATH_CALUDE_cubic_equation_product_l292_29205

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2006) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2006) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2006) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1003 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l292_29205


namespace NUMINAMATH_CALUDE_existence_of_rationals_l292_29220

theorem existence_of_rationals (a b c d m n : ℤ) (ε : ℝ) 
  (h : a * d - b * c ≠ 0) (hε : ε > 0) :
  ∃ x y : ℚ, 0 < |a * x + b * y - m| ∧ |a * x + b * y - m| < ε ∧
           0 < |c * x + d * y - n| ∧ |c * x + d * y - n| < ε :=
by sorry


end NUMINAMATH_CALUDE_existence_of_rationals_l292_29220


namespace NUMINAMATH_CALUDE_range_of_a_l292_29207

-- Define the statements p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x > a^y
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x + a > 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | (0 < a ∧ a ≤ 1/4) ∨ (a ≥ 1)}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ valid_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l292_29207


namespace NUMINAMATH_CALUDE_andre_flowers_l292_29267

/-- The number of flowers Andre gave to Rosa -/
def flowers_given : ℕ := 90 - 67

/-- Rosa's initial number of flowers -/
def initial_flowers : ℕ := 67

/-- Rosa's final number of flowers -/
def final_flowers : ℕ := 90

theorem andre_flowers : flowers_given = final_flowers - initial_flowers := by
  sorry

end NUMINAMATH_CALUDE_andre_flowers_l292_29267


namespace NUMINAMATH_CALUDE_tire_company_cost_per_batch_l292_29211

/-- A tire company's production and sales model -/
structure TireCompany where
  cost_per_batch : ℝ
  cost_per_tire : ℝ
  selling_price : ℝ
  batch_size : ℕ
  profit_per_tire : ℝ

/-- The cost per batch for the tire company -/
def cost_per_batch (company : TireCompany) : ℝ :=
  company.cost_per_batch

/-- Theorem stating the cost per batch for the given scenario -/
theorem tire_company_cost_per_batch :
  ∀ (company : TireCompany),
    company.cost_per_tire = 8 →
    company.selling_price = 20 →
    company.batch_size = 15000 →
    company.profit_per_tire = 10.5 →
    cost_per_batch company = 22500 := by
  sorry

end NUMINAMATH_CALUDE_tire_company_cost_per_batch_l292_29211


namespace NUMINAMATH_CALUDE_not_parabola_l292_29257

/-- The equation x² + y²cos(θ) = 1, where θ is any real number, cannot represent a parabola -/
theorem not_parabola (θ : ℝ) : 
  ¬ (∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    (x^2 + y^2 * Real.cos θ = 1) ↔ (a*x^2 + b*x*y + c*y^2 + d*x + e*y = 1 ∧ b^2 = 4*a*c)) :=
by sorry

end NUMINAMATH_CALUDE_not_parabola_l292_29257


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l292_29265

theorem number_with_specific_remainders : ∃ (N : ℕ), N % 13 = 11 ∧ N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l292_29265


namespace NUMINAMATH_CALUDE_production_problem_l292_29284

def initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) : ℕ :=
  ((n + 1) * new_average - today_production) / n

theorem production_problem :
  let n : ℕ := 3
  let today_production : ℕ := 90
  let new_average : ℕ := 75
  initial_average_production n today_production new_average = 70 := by
  sorry

end NUMINAMATH_CALUDE_production_problem_l292_29284


namespace NUMINAMATH_CALUDE_mike_final_cards_l292_29238

def mike_cards (initial : ℕ) (received : ℕ) (traded : ℕ) : ℕ :=
  initial + received - traded

theorem mike_final_cards :
  mike_cards 64 18 20 = 62 := by
  sorry

end NUMINAMATH_CALUDE_mike_final_cards_l292_29238


namespace NUMINAMATH_CALUDE_trail_mix_packs_needed_l292_29212

def total_people : ℕ := 18
def pouches_per_pack : ℕ := 6

theorem trail_mix_packs_needed :
  ∃ (packs : ℕ), packs * pouches_per_pack ≥ total_people ∧
  ∀ (x : ℕ), x * pouches_per_pack ≥ total_people → x ≥ packs :=
by sorry

end NUMINAMATH_CALUDE_trail_mix_packs_needed_l292_29212


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_999_l292_29291

/-- Sum of digits function for a single number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of all digits in numbers from 0 to n -/
def sumOfAllDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in decimal representations of integers from 0 to 999 is 13500 -/
theorem sum_of_digits_up_to_999 : sumOfAllDigits 999 = 13500 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_999_l292_29291


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l292_29200

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 220 ∧ x = 220 / 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l292_29200


namespace NUMINAMATH_CALUDE_jimin_candies_l292_29226

/-- The number of candies Jimin gave to Yuna -/
def candies_given : ℕ := 25

/-- The number of candies left over -/
def candies_left : ℕ := 13

/-- The total number of candies Jimin had at the start -/
def total_candies : ℕ := candies_given + candies_left

theorem jimin_candies : total_candies = 38 := by
  sorry

end NUMINAMATH_CALUDE_jimin_candies_l292_29226


namespace NUMINAMATH_CALUDE_next_roll_for_average_three_l292_29286

def rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

theorem next_roll_for_average_three :
  let n : Nat := rolls.length
  let sum : Nat := rolls.sum
  let target_average : Rat := 3
  let next_roll : Nat := 2
  (sum + next_roll : Rat) / (n + 1) = target_average := by sorry

end NUMINAMATH_CALUDE_next_roll_for_average_three_l292_29286


namespace NUMINAMATH_CALUDE_series_relationship_l292_29221

-- Define the sequence of exponents
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => a n + a (n + 1)

-- Define the series
def series (n : ℕ) : ℕ := 2^(a n)

-- Theorem statement
theorem series_relationship (n : ℕ) :
  series n * series (n + 1) = series (n + 2) := by
  sorry


end NUMINAMATH_CALUDE_series_relationship_l292_29221


namespace NUMINAMATH_CALUDE_card_difference_l292_29237

/-- The number of cards each person has -/
structure CardCounts where
  ann : ℕ
  anton : ℕ
  heike : ℕ

/-- The conditions of the problem -/
def card_problem (c : CardCounts) : Prop :=
  c.ann = 60 ∧
  c.ann = 6 * c.heike ∧
  c.anton = c.heike

/-- The theorem to prove -/
theorem card_difference (c : CardCounts) (h : card_problem c) : 
  c.ann - c.anton = 50 := by
  sorry

end NUMINAMATH_CALUDE_card_difference_l292_29237


namespace NUMINAMATH_CALUDE_jerry_shelf_theorem_l292_29225

/-- The number of action figures and books on Jerry's shelf -/
def shelf_contents : ℕ × ℕ := (5, 9)

/-- The number of action figures added later -/
def added_figures : ℕ := 7

/-- The final difference between action figures and books -/
def figure_book_difference : ℤ :=
  (shelf_contents.1 + added_figures : ℤ) - shelf_contents.2

theorem jerry_shelf_theorem :
  figure_book_difference = 3 := by sorry

end NUMINAMATH_CALUDE_jerry_shelf_theorem_l292_29225


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l292_29261

theorem greatest_prime_factor_of_154 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 154 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 154 → q ≤ p ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l292_29261


namespace NUMINAMATH_CALUDE_stable_performance_comparison_l292_29242

/-- Represents a student's performance in standing long jumps --/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if a student's performance is more stable --/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students with the same average score, 
    the one with lower variance has more stable performance --/
theorem stable_performance_comparison 
  (student_a student_b : StudentPerformance)
  (h_same_average : student_a.average_score = student_b.average_score)
  (h_a_variance : student_a.variance = 0.48)
  (h_b_variance : student_b.variance = 0.53) :
  more_stable student_a student_b :=
by
  sorry

end NUMINAMATH_CALUDE_stable_performance_comparison_l292_29242


namespace NUMINAMATH_CALUDE_ab_is_zero_l292_29217

theorem ab_is_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_is_zero_l292_29217


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l292_29299

theorem lcm_factor_proof (A B : ℕ+) (h_hcf : Nat.gcd A B = 25) 
  (h_lcm : ∃ X : ℕ+, Nat.lcm A B = 25 * X * 14) (h_A : A = 350) (h_order : A > B) : 
  ∃ X : ℕ+, Nat.lcm A B = 25 * X * 14 ∧ X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l292_29299


namespace NUMINAMATH_CALUDE_trisha_chicken_expense_l292_29224

/-- Given Trisha's shopping expenses and initial amount, prove that she spent $22 on chicken -/
theorem trisha_chicken_expense (meat_cost veggies_cost eggs_cost dog_food_cost initial_amount remaining_amount : ℕ) 
  (h1 : meat_cost = 17)
  (h2 : veggies_cost = 43)
  (h3 : eggs_cost = 5)
  (h4 : dog_food_cost = 45)
  (h5 : initial_amount = 167)
  (h6 : remaining_amount = 35) :
  initial_amount - remaining_amount - (meat_cost + veggies_cost + eggs_cost + dog_food_cost) = 22 := by
  sorry

end NUMINAMATH_CALUDE_trisha_chicken_expense_l292_29224


namespace NUMINAMATH_CALUDE_eight_circle_times_three_l292_29276

-- Define the new operation ⨳
def circle_times (a b : ℤ) : ℤ := 4 * a + 6 * b

-- The theorem to prove
theorem eight_circle_times_three : circle_times 8 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_eight_circle_times_three_l292_29276


namespace NUMINAMATH_CALUDE_point_on_linear_function_l292_29227

theorem point_on_linear_function (m : ℝ) : 
  (3 : ℝ) = 2 * m + 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_function_l292_29227
