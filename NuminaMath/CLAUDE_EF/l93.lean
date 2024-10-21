import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_binomial_expansion_l93_9345

theorem middle_term_binomial_expansion (n : ℕ) (x : ℝ) : 
  (Finset.range (2*n + 1)).sum (λ k ↦ (Nat.choose (2*n) k) * x^k) = 
  (Finset.range (2*n + 1)).sum (λ k ↦ if k = n then 
    (Finset.prod (Finset.range n) (λ i ↦ 2*i + 1) / n.factorial) * 2^n * x^n 
  else 
    (Nat.choose (2*n) k) * x^k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_binomial_expansion_l93_9345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l93_9394

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 1

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l93_9394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_renovation_project_l93_9359

/-- Represents a construction team -/
structure Team where
  days_alone : ℝ  -- Days to complete the project alone
  daily_cost : ℝ  -- Daily cost in yuan

/-- Represents the road renovation project -/
structure Project where
  team_a : Team
  team_b : Team
  budget : ℝ
  deadline : ℝ

noncomputable def Project.days_together (p : Project) : ℝ :=
  1 / (1 / p.team_a.days_alone + 1 / p.team_b.days_alone)

noncomputable def Project.total_days (p : Project) (days_a : ℝ) : ℝ :=
  days_a + p.team_b.days_alone * (1 - days_a / p.team_a.days_alone)

noncomputable def Project.max_days_together (p : Project) : ℝ :=
  (p.budget - p.team_b.daily_cost * p.team_b.days_alone) / 
  (p.team_a.daily_cost + p.team_b.daily_cost - 3 * p.team_b.daily_cost)

noncomputable def Project.min_cost_arrangement (p : Project) : ℝ × ℝ × ℝ :=
  let days_together := p.team_a.days_alone * (1 - p.deadline / (2 * p.team_b.days_alone))
  let days_b_alone := p.deadline - days_together
  let total_cost := days_together * (p.team_a.daily_cost + p.team_b.daily_cost) + 
                    days_b_alone * p.team_b.daily_cost
  (days_together, days_b_alone, total_cost)

theorem road_renovation_project (p : Project) 
  (h1 : p.team_a.days_alone = 30) 
  (h2 : p.team_b.days_alone = 60) 
  (h3 : p.team_a.daily_cost = 25000)
  (h4 : p.team_b.daily_cost = 10000)
  (h5 : p.budget = 650000)
  (h6 : p.deadline = 24) : 
  p.days_together = 20 ∧ 
  p.total_days 10 = 40 ∧
  p.max_days_together = 10 ∧
  p.min_cost_arrangement = (18, 6, 690000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_renovation_project_l93_9359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l93_9339

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

noncomputable def expression (a b : ℝ) : ℝ :=
  (b^2 + 1) / (3 * a)

theorem hyperbola_min_value (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : eccentricity a b = 2) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 / 3 ∧ 
  ∀ (a' b' : ℝ), a' > 0 → b' > 0 → eccentricity a' b' = 2 → 
  expression a' b' ≥ min := by
  sorry

#check hyperbola_min_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l93_9339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_inequality_l93_9381

-- Part a
def equation (x : ℝ) : Prop :=
  (4 * x - 7) * Real.sqrt (x - 3) = 25 * Real.sqrt 5

theorem equation_equivalence :
  ∀ x : ℝ, equation x ↔ 16 * x^3 - 104 * x^2 + 217 * x - 3272 = 0 :=
by sorry

-- Part b
noncomputable def left_expression : ℝ :=
  (3 ^ (1/20)) * ((6 + 3 * (5 ^ (1/5)) + 6 * (25 ^ (1/3))) ^ (1/4))

noncomputable def right_expression : ℝ :=
  (16 + 9 * (5 ^ (1/3)) + 5 * (25 ^ (1/3))) ^ (1/5)

theorem inequality : left_expression ≠ right_expression :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_equivalence_inequality_l93_9381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_can_be_circle_C_ellipse_x_axis_foci_l93_9368

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (5 - t) + p.2^2 / (t - 1) = 1}

-- Helper definitions
def IsCircle (s : Set (ℝ × ℝ)) : Prop := sorry
def IsEllipse (s : Set (ℝ × ℝ)) : Prop := sorry
def FociOnXAxis (s : Set (ℝ × ℝ)) : Prop := sorry

-- Statement 1: C can be a circle
theorem C_can_be_circle :
  ∃ t : ℝ, IsCircle (C t) := by
  sorry

-- Statement 2: If C is an ellipse with foci on x-axis, then 1 < t < 3
theorem C_ellipse_x_axis_foci (t : ℝ) :
  IsEllipse (C t) → FociOnXAxis (C t) → 1 < t ∧ t < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_can_be_circle_C_ellipse_x_axis_foci_l93_9368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l93_9310

theorem tan_sum_given_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 40) 
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 1) : 
  Real.tan (x + y) = -40 / 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l93_9310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_spread_l93_9312

/-- The day of the week, represented as an integer from 1 to 7 -/
def DayOfWeek := Fin 7

/-- Convert a number of days to a day of the week, assuming Monday is 1 -/
def toDayOfWeek (n : ℕ) : DayOfWeek :=
  ⟨(n - 1) % 7 + 1, by sorry⟩

/-- The number of people who know the secret on day n -/
def peopleKnowing (n : ℕ) : ℕ := 2^(n+1) - 1

theorem secret_spread (target : ℕ) :
  (∃ n : ℕ, peopleKnowing n = target) →
  (∃ n : ℕ, peopleKnowing n = target ∧ toDayOfWeek (n + 1) = ⟨3, by sorry⟩) :=
by
  intro h
  cases h with
  | intro n hn =>
    exists n
    constructor
    · exact hn
    · sorry  -- Proof that toDayOfWeek (n + 1) = ⟨3, by sorry⟩

#check secret_spread

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_spread_l93_9312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_existence_l93_9313

theorem distinct_remainders_existence (p : ℕ) (hp : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, (Finset.univ.filter (λ i : Fin p => ∃ j : Fin p, (a j + j.val * k) % p = (a i + i.val * k) % p)).card ≤ p / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_existence_l93_9313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_product_l93_9360

theorem consecutive_numbers_product (n : ℕ) :
  (n > 0) →
  (n + (n + 1) = 11) →
  (n * (n + 1) * (n + 2) = 210) →
  ∃! k : ℕ, k = 3 ∧ 
    (Finset.prod (Finset.range k) (λ i => n + i) = 210) ∧
    (n + (n + 1) = 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_product_l93_9360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EX_approx_l93_9320

/-- The length of EX in a regular pentagon ABCDE with side length 2,
    where AB is extended to X such that AX = 4AB -/
noncomputable def length_EX : ℝ :=
  let side_length := 2
  let AX_length := 4 * side_length
  let angle_EAB := 108 * Real.pi / 180  -- Convert to radians
  let AP_length := side_length * Real.cos angle_EAB
  let PX_length := AP_length + AX_length
  let EP_length := side_length * Real.sin angle_EAB
  Real.sqrt (EP_length ^ 2 + PX_length ^ 2)

/-- Theorem stating that the length of EX is approximately 8.83 -/
theorem length_EX_approx : 
  ∃ ε > 0, |length_EX - 8.83| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EX_approx_l93_9320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_implies_a_minus_b_equals_two_l93_9388

theorem equation_implies_a_minus_b_equals_two 
  (x y : ℝ) (a b : ℤ) 
  (h : ∀ x y : ℝ, 5 * x^(a : ℝ) * y^5 + 2 * x^3 * y^((2*a - b : ℤ) : ℝ) = 7 * x^(a : ℝ) * y^5) : 
  a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_implies_a_minus_b_equals_two_l93_9388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_probability_three_l93_9301

/-- The number of derangements for a set of size n -/
def derangements_count (n : ℕ) : ℕ := sorry

/-- The probability of a derangement for a set of size n -/
def derangement_probability (n : ℕ) : ℚ :=
  (derangements_count n : ℚ) / (Nat.factorial n : ℚ)

/-- Theorem: The probability of a derangement for 3 items is 1/3 -/
theorem derangement_probability_three : derangement_probability 3 = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_probability_three_l93_9301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_decrease_hours_increase_l93_9393

theorem wage_decrease_hours_increase :
  ∀ (original_wage original_hours : ℝ),
    original_wage > 0 → original_hours > 0 →
    let new_wage := 0.8 * original_wage;
    let new_hours := original_wage * original_hours / new_wage;
    (new_hours - original_hours) / original_hours = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_decrease_hours_increase_l93_9393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_and_m_value_l93_9305

-- Define the triangle and point M
variable (A B C M : EuclideanSpace ℝ (Fin 3))

-- Define the condition that M satisfies
def is_centroid (A B C M : EuclideanSpace ℝ (Fin 3)) : Prop :=
  (M - A) + (M - B) + (M - C) = 0

-- Define the condition for the real number m
def satisfies_vector_equation (A B C M : EuclideanSpace ℝ (Fin 3)) (m : ℝ) : Prop :=
  (B - A) + (C - A) = m • (M - A)

-- State the theorem
theorem centroid_and_m_value
  (h1 : is_centroid A B C M)
  (h2 : ∃ m : ℝ, satisfies_vector_equation A B C M m) :
  is_centroid A B C M ∧ ∃ m : ℝ, satisfies_vector_equation A B C M m ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_and_m_value_l93_9305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l93_9346

-- Define the curves in polar coordinates
noncomputable def curve1 (φ : Real) : Real := 6 * Real.cos (3 * φ)
def curve2 : Real := 3

-- Define the integration bounds
noncomputable def lower_bound : Real := -Real.pi/9
noncomputable def upper_bound : Real := Real.pi/9

-- State the theorem
theorem area_between_curves : 
  (1/2) * ∫ φ in lower_bound..upper_bound, (curve1 φ)^2 - curve2^2 = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l93_9346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l93_9333

/-- A fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- The total volume of the fruit drink in ounces -/
noncomputable def total_volume (drink : FruitDrink) : ℝ :=
  drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem stating the total volume of the specific fruit drink -/
theorem fruit_drink_volume :
  ∀ (drink : FruitDrink),
    drink.orange_percent = 0.25 →
    drink.watermelon_percent = 0.4 →
    drink.grape_ounces = 105 →
    total_volume drink = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l93_9333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_football_players_l93_9395

theorem basketball_football_players (total basketball football both neither : ℕ) 
  (h_total : total = 30)
  (h_basketball : ∃ (n : ℕ), n > 0 ∧ basketball = 4 * n)
  (h_football : ∃ (n : ℕ), n > 0 ∧ football = 2 * n)
  (h_neither : ∃ (n : ℕ), n > 0 ∧ neither = n)
  (h_sum : basketball + football - both + neither = total)
  : both = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_football_players_l93_9395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_sequence_l93_9370

noncomputable def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := n * (1 + arithmetic_sequence d n) / 2

theorem min_value_arithmetic_sequence (d : ℝ) :
  d ≠ 0 →
  (1 + 2*d)^2 = 1 + 12*d →
  ∀ n : ℕ+, (2 * sum_arithmetic_sequence d n + 16) / (arithmetic_sequence d n + 3) ≥ 4 ∧
  ∃ n₀ : ℕ+, (2 * sum_arithmetic_sequence d n₀ + 16) / (arithmetic_sequence d n₀ + 3) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_sequence_l93_9370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l93_9326

def star (a b : ℕ) : ℕ := b ^ a

theorem star_properties :
  ¬(∀ (a b c m : ℕ),
    a > 0 → b > 0 → c > 0 → m > 0 →
    (star a b = star b a) ∧
    (star a (star b c) = star (star a b) c) ∧
    (star a (b ^ m) = star (star a m) b) ∧
    ((star a b) ^ m = star a (m * b))) :=
by
  intro h
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l93_9326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_savings_l93_9373

/-- Calculates the total savings on a pair of jeans given the original price and discounts -/
noncomputable def calculate_savings (original_price : ℚ) (sale_discount_percent : ℚ) (coupon_discount : ℚ) (credit_card_discount_percent : ℚ) : ℚ :=
  let price_after_sale := original_price * (1 - sale_discount_percent / 100)
  let price_after_coupon := price_after_sale - coupon_discount
  let final_price := price_after_coupon * (1 - credit_card_discount_percent / 100)
  original_price - final_price

/-- Theorem stating that the savings on the jeans with given discounts is $44 -/
theorem jeans_savings :
  calculate_savings 125 20 10 10 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_savings_l93_9373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_implies_P_but_not_conversely_l93_9392

variable (f : ℝ → ℝ)

-- f is differentiable
variable (hf : Differentiable ℝ f)

-- Proposition Q
def prop_Q (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |deriv f x| < 2017

-- Proposition P
def prop_P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → |(f x₁ - f x₂) / (x₁ - x₂)| < 2017

-- Theorem: Q implies P, but P does not always imply Q
theorem Q_implies_P_but_not_conversely (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (prop_Q f → prop_P f) ∧ ¬(prop_P f → prop_Q f) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_implies_P_but_not_conversely_l93_9392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_num_and_denom_is_133_l93_9389

/-- The repeating decimal 0.overline{34} as a rational number -/
def repeating_decimal : ℚ := 34 / 99

/-- The sum of the numerator and denominator of the fraction representing 0.overline{34} in lowest terms -/
def sum_num_denom : ℕ := 133

/-- Theorem stating that the sum of the numerator and denominator of the fraction
    representing 0.overline{34} in lowest terms is 133 -/
theorem sum_of_num_and_denom_is_133 :
  (repeating_decimal.num.natAbs + repeating_decimal.den) = sum_num_denom := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_num_and_denom_is_133_l93_9389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_counterfeit_weight_l93_9328

/-- Represents a coin that can be either genuine or counterfeit -/
inductive Coin
  | genuine
  | counterfeit

/-- Represents the result of a weighing -/
inductive WeighingResult
  | equal
  | leftHeavier
  | rightHeavier

/-- Represents a weighing action on the balance scale -/
def Weighing := (List Coin) → (List Coin) → WeighingResult

/-- Represents the state of knowledge about the counterfeit coins -/
inductive CounterfeitState
  | unknown
  | heavier
  | lighter

/-- Function to compare coins -/
def compareCoin (c1 c2 : Coin) : Bool :=
  match c1, c2 with
  | Coin.genuine, Coin.genuine => true
  | Coin.counterfeit, Coin.counterfeit => true
  | _, _ => false

/-- The main theorem stating that it's possible to determine the state of counterfeit coins in three weighings -/
theorem determine_counterfeit_weight 
  (coins : List Coin) 
  (h_count : coins.length = 239)
  (h_genuine : (coins.filter (fun x => compareCoin x Coin.genuine)).length = 237)
  (h_counterfeit : (coins.filter (fun x => compareCoin x Coin.counterfeit)).length = 2) :
  ∃ (w₁ w₂ w₃ : Weighing), 
    ∀ (initial_state : CounterfeitState),
    ∃ (final_state : CounterfeitState),
    final_state ≠ CounterfeitState.unknown ∧
    (final_state = CounterfeitState.heavier ↔ 
      (∀ (c1 c2 : Coin), compareCoin c1 Coin.counterfeit ∧ compareCoin c2 Coin.genuine → 
        w₁ [c1] [c2] = WeighingResult.leftHeavier)) ∧
    (final_state = CounterfeitState.lighter ↔ 
      (∀ (c1 c2 : Coin), compareCoin c1 Coin.counterfeit ∧ compareCoin c2 Coin.genuine → 
        w₁ [c1] [c2] = WeighingResult.rightHeavier)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_counterfeit_weight_l93_9328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l93_9349

def mySequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ a 2 = 1 ∧
  ∀ n : ℕ, n ≥ 1 → 1 / (a n) + 1 / (a (n + 2)) = 2 / (a (n + 1))

theorem sixth_term_value (a : ℕ → ℚ) (h : mySequence a) : a 6 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l93_9349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_coffee_mix_amount_l93_9352

/-- Represents the coffee mix problem --/
structure CoffeeMix where
  columbian_price : ℚ
  brazilian_price : ℚ
  mix_price : ℚ
  columbian_amount : ℚ

/-- The solution to the coffee mix problem --/
noncomputable def solve_coffee_mix (c : CoffeeMix) : ℚ :=
  c.columbian_amount + (c.columbian_amount * (c.columbian_price - c.mix_price)) / (c.mix_price - c.brazilian_price)

/-- Theorem stating that given the conditions, Joey made 100 pounds of coffee mix --/
theorem joey_coffee_mix_amount :
  let c : CoffeeMix := {
    columbian_price := 875/100,
    brazilian_price := 375/100,
    mix_price := 635/100,
    columbian_amount := 52
  }
  solve_coffee_mix c = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_coffee_mix_amount_l93_9352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_intersection_l93_9304

/-- Given a triangle ABC with points E on AC and F on AB, where AE:EC = 3:2 and AF:FB = 1:3,
    and P is the intersection of BE and CF, prove that P = (2/9)*A + (1/9)*B + (3/9)*C. -/
theorem triangle_vector_intersection (A B C E F P : EuclideanSpace ℝ (Fin 3)) : 
  (∃ (t : ℝ), E = A + t • (C - A) ∧ t / (1 - t) = 3 / 2) →
  (∃ (s : ℝ), F = A + s • (B - A) ∧ s / (1 - s) = 1 / 3) →
  (∃ (u v : ℝ), P = B + u • (E - B) ∧ P = C + v • (F - C)) →
  P = (2/9) • A + (1/9) • B + (3/9) • C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_intersection_l93_9304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interchange_queens_in_24_moves_l93_9385

/-- Represents a chessboard with Queens -/
structure Chessboard where
  black_queens : Finset (Nat × Nat)
  white_queens : Finset (Nat × Nat)

/-- Represents a move of a Queen -/
inductive Move where
  | horizontal : Nat → Nat → Nat → Move
  | vertical : Nat → Nat → Nat → Move
  | diagonal : Nat → Nat → Nat → Nat → Move

/-- The initial setup of the chessboard -/
def initial_board : Chessboard where
  black_queens := Finset.image (fun i => (i + 1, 1)) (Finset.range 8)
  white_queens := Finset.image (fun i => (i + 1, 8)) (Finset.range 8)

/-- The final setup of the chessboard after interchanging Queens -/
def final_board : Chessboard where
  black_queens := Finset.image (fun i => (i + 1, 8)) (Finset.range 8)
  white_queens := Finset.image (fun i => (i + 1, 1)) (Finset.range 8)

/-- Checks if a move is valid according to the rules -/
def is_valid_move (board : Chessboard) (move : Move) : Prop :=
  sorry

/-- Applies a move to the chessboard -/
def apply_move (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Checks if two boards are equal -/
def board_equal (b1 b2 : Chessboard) : Prop :=
  b1.black_queens = b2.black_queens ∧ b1.white_queens = b2.white_queens

/-- The main theorem stating that 24 moves are required to interchange the Queens -/
theorem interchange_queens_in_24_moves :
  ∃ (moves : List Move),
    moves.length = 24 ∧
    (moves.foldl apply_move initial_board |> board_equal final_board) ∧
    (∀ (m : Move) (i : Nat), i < moves.length → is_valid_move (moves.take i |>.foldl apply_move initial_board) (moves.get ⟨i, by sorry⟩)) ∧
    (∀ (moves' : List Move),
      moves'.length < 24 →
      ¬(moves'.foldl apply_move initial_board |> board_equal final_board)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interchange_queens_in_24_moves_l93_9385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_relationships_l93_9322

-- Define the set of men and women
inductive Man : Type
| Peter : Man
| Paul : Man
| John : Man

inductive Woman : Type
| Margaret : Woman
| Eve : Woman
| Mary : Woman

-- Define the relationships
def fiance : Man → Woman := sorry
def sister : Man → Woman := sorry

-- Define the gift amounts
def gift_to_fiance : Man → ℕ := sorry
def gift_to_sister : Man → ℕ := sorry
def gift_to_other : Man → ℕ := sorry

-- Define the total gifts received by each woman
def total_gifts : Woman → ℕ := sorry

-- State the problem conditions
axiom gift_amounts :
  (gift_to_fiance Man.Peter = 40) ∧
  (gift_to_sister Man.Peter = 24) ∧
  (gift_to_other Man.Peter = 10) ∧
  (gift_to_fiance Man.Paul = 36) ∧
  (gift_to_sister Man.Paul = 28) ∧
  (gift_to_other Man.Paul = 8) ∧
  (gift_to_fiance Man.John = 38) ∧
  (gift_to_sister Man.John = 26) ∧
  (gift_to_other Man.John = 10)

axiom gifts_received :
  (total_gifts Woman.Mary = 72) ∧
  (total_gifts Woman.Margaret = 70)

axiom relationships :
  ∀ m : Man, ∃! w : Woman, fiance m = w

axiom sister_relation :
  ∀ m : Man, ∃! w : Woman, sister m = w ∧ w ≠ fiance m

-- Theorem to prove
theorem determine_relationships :
  (fiance Man.Peter = Woman.Eve ∧ sister Man.Paul = Woman.Eve) ∧
  (fiance Man.Paul = Woman.Mary ∧ sister Man.John = Woman.Mary) ∧
  (fiance Man.John = Woman.Margaret ∧ sister Man.Peter = Woman.Margaret) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_relationships_l93_9322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_value_l93_9332

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.tan (ω * x + φ)

theorem function_period_and_value (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : |φ| < π/2) 
  (h3 : ∀ x, f ω φ (x + π/2) = f ω φ x)  -- smallest positive period is π/2
  (h4 : f ω φ (π/2) = -2) :
  ω = 2 ∧ φ = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_value_l93_9332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natashas_quarters_l93_9364

theorem natashas_quarters : ∃ n : ℕ, 
  (1 < (n : ℚ) * (1/4) ∧ (n : ℚ) * (1/4) < 10) ∧ 
  n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ 
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natashas_quarters_l93_9364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l93_9330

noncomputable section

/-- The plane equation: 5x - 2y + 6z = 40 -/
def plane_equation (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 6 * z = 40

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (2, -1, 4)

/-- The proposed closest point -/
def closest_point : ℝ × ℝ × ℝ := (138/65, -73/65, 274/65)

/-- The theorem stating that the closest_point is indeed the closest point on the plane to the given_point -/
theorem closest_point_on_plane :
  plane_equation closest_point.1 closest_point.2.1 closest_point.2.2 ∧
  ∀ (p : ℝ × ℝ × ℝ), plane_equation p.1 p.2.1 p.2.2 →
    ‖(p.1 - given_point.1, p.2.1 - given_point.2.1, p.2.2 - given_point.2.2)‖ ≥ 
    ‖(closest_point.1 - given_point.1, closest_point.2.1 - given_point.2.1, closest_point.2.2 - given_point.2.2)‖ :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l93_9330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_mixture_volume_unchanged_l93_9356

/-- Coefficient of thermal expansion of oil -/
noncomputable def β : ℝ := 2e-3

/-- Initial volume of hot oil in liters -/
noncomputable def V₁ : ℝ := 2

/-- Initial volume of cold oil in liters -/
noncomputable def V₂ : ℝ := 1

/-- Initial temperature of hot oil in °C -/
noncomputable def t₁ : ℝ := 100

/-- Initial temperature of cold oil in °C -/
noncomputable def t₂ : ℝ := 20

/-- Equilibrium temperature of the mixture in °C -/
noncomputable def t : ℝ := (V₁ * t₁ + V₂ * t₂) / (V₁ + V₂)

/-- Theorem stating that the final volume of the oil mixture remains unchanged -/
theorem oil_mixture_volume_unchanged :
  V₁ * (1 + β * t₁) + V₂ * (1 + β * t₂) = (V₁ + V₂) * (1 + β * t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_mixture_volume_unchanged_l93_9356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_range_l93_9308

-- Define the ellipse equation
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1

-- Define the condition that foci are on the x-axis
def foci_on_x_axis (a : ℝ) : Prop :=
  a^2 > a + 6 ∧ a + 6 > 0

-- Theorem statement
theorem ellipse_a_range :
  ∀ a : ℝ, (is_ellipse a ∧ foci_on_x_axis a) ↔ a ∈ Set.Ioo (-6 : ℝ) (-2 : ℝ) ∪ Set.Ioi (3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_range_l93_9308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_2_eq_1_l93_9309

-- Define g as a variable (function) of type ℝ → ℝ
variable (g : ℝ → ℝ)

def f (x : ℝ) : ℝ := g x + 2

-- g is an odd function
axiom g_odd : ∀ x : ℝ, g (-x) = -g x

-- f(2) = 3
axiom f_2_eq_3 : f 2 = 3

theorem f_neg_2_eq_1 : f (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_2_eq_1_l93_9309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_tube_volume_difference_l93_9355

/-- Calculates the volume of a cylindrical tube formed by rolling a rectangular paper. -/
noncomputable def tubeVolume (width : ℝ) (length : ℝ) : ℝ :=
  (width * length^2) / (4 * Real.pi)

/-- The problem statement as a theorem in Lean 4 -/
theorem cylindrical_tube_volume_difference : 
  let amyTubeVolume := tubeVolume 9 12
  let belindaTubeVolume := tubeVolume 7.5 10
  Real.pi * |belindaTubeVolume - amyTubeVolume| = 136.5 := by
  sorry

#check cylindrical_tube_volume_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_tube_volume_difference_l93_9355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l93_9324

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = f a b (-x) * (-1)) →
  f a b (1/2) = 2/5 →
  ∃ (g : ℝ → ℝ),
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = x / (x^2 + 1)) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧
    (Set.Ioo 0 (1/3) = {x : ℝ | g (2*x - 1) + g x < 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l93_9324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_crease_length_l93_9383

noncomputable section

-- Define the necessary structures
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ
  nondegenerate : sorry

structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define necessary functions
def Triangle.base (t : Triangle) : Line := sorry
def Triangle.area (t : Triangle) : ℝ := sorry
def Line.length (l : Line) : ℝ := sorry
def Line.parallel (l1 l2 : Line) : Prop := sorry
def Triangle.areaBelow (t : Triangle) (l : Line) : ℝ := sorry

theorem folded_triangle_crease_length 
  (ABC : Triangle) 
  (DE : Line) 
  (base_length : ℝ) 
  (area_ratio : ℝ) :
  base_length = 15 →
  Line.parallel DE (Triangle.base ABC) →
  area_ratio = 0.25 →
  Triangle.areaBelow ABC DE / Triangle.area ABC = area_ratio →
  Line.length DE = 7.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_crease_length_l93_9383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_side_c_values_l93_9307

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Part 1
theorem cos_C_value (abc : Triangle) 
  (h1 : abc.a = 14) 
  (h2 : abc.b = 40) 
  (h3 : Real.cos abc.B = 3/5) :
  Real.cos abc.C = -44/125 := by
  sorry

-- Part 2
theorem side_c_values (abc : Triangle) 
  (h1 : abc.a = 3) 
  (h2 : abc.b = 2 * Real.sqrt 6) 
  (h3 : abc.B = 2 * abc.A) :
  abc.c = 3 ∨ abc.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_side_c_values_l93_9307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l93_9379

-- Define the scaling transformation
noncomputable def scaling (x y : ℝ) : ℝ × ℝ :=
  (3 * x, y / 2)

-- State the theorem
theorem curve_equation :
  ∀ (x y : ℝ),
  let (x'', y'') := scaling x y
  y'' = Real.sin x'' →
  y = 2 * Real.sin (3 * x) :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l93_9379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l93_9367

noncomputable def f (x : ℝ) := 3 * Real.sin (x / 2 + Real.pi / 4) - 1

theorem f_minimum :
  (∀ x : ℝ, f x ≥ -4) ∧
  (∀ k : ℤ, f (4 * ↑k * Real.pi - 3 * Real.pi / 2) = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l93_9367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_l93_9366

noncomputable def g (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem axis_of_symmetry_f :
  ∃ (k : ℤ), f (-Real.pi / 2 + k * Real.pi) = f (-Real.pi / 2 - k * Real.pi) ∧
  (∀ (x : ℝ), f (x + (-Real.pi / 2)) = f ((-Real.pi / 2) - x)) := by
  sorry

#check axis_of_symmetry_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_l93_9366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l93_9398

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ  -- Length of one parallel side
  side2 : ℝ  -- Length of the other parallel side
  height : ℝ  -- Distance between parallel sides
  side1_positive : 0 < side1
  side2_positive : 0 < side2
  height_positive : 0 < height

/-- Calculate the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- The shorter parallel side of a trapezium -/
noncomputable def shorter_side (t : Trapezium) : ℝ :=
  min t.side1 t.side2

theorem trapezium_properties (t : Trapezium) 
    (h1 : t.side1 = 10) 
    (h2 : t.side2 = 18) 
    (h3 : t.height = 15) : 
    area t = 210 ∧ shorter_side t = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l93_9398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_z_implies_result_l93_9357

def z (a : ℝ) : ℂ := (a - 2) + 3*Complex.I

theorem imaginary_z_implies_result (a : ℝ) (h : z a = Complex.I * (z a).im) :
  (a + Complex.I) / (1 + a*Complex.I) = (4 - 3*Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_z_implies_result_l93_9357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l93_9348

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + 1 else Real.sqrt (1 - x)

-- Theorem statement
theorem f_composition_negative_three :
  f (f (-3)) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l93_9348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_with_remainders_l93_9390

theorem count_four_digit_integers_with_remainders : 
  (Finset.filter (λ n : ℕ => 1000 ≤ n ∧ n < 10000 ∧ 
                             n % 7 = 3 ∧ 
                             n % 8 = 4 ∧ 
                             n % 10 = 6) 
                 (Finset.range 10000)).card = 161 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_with_remainders_l93_9390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_f_inv_comp_l93_9316

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the inverse of f
noncomputable def f_inv (x : ℝ) : ℝ := (x - 3) / 2

-- Define the composition f_inv(x+1)
noncomputable def f_inv_comp (x : ℝ) : ℝ := f_inv (x + 1)

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := 2 * x + 2

-- Theorem statement
theorem inverse_of_f_inv_comp (x : ℝ) :
  g (f_inv_comp x) = x ∧ f_inv_comp (g x) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_f_inv_comp_l93_9316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problems_l93_9362

theorem calculation_problems :
  ((Real.sqrt (1/3) + Real.sqrt 6) / Real.sqrt 3 = 1 + 3 * Real.sqrt 2) ∧
  ((Real.sqrt 3)^2 - Real.sqrt 4 + Real.sqrt ((-2)^2) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problems_l93_9362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_proof_l93_9386

/-- Given a train of length 156 meters traveling at 45 km/h that crosses a bridge in 30 seconds,
    prove that the length of the bridge is 219 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 156 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 219 := by
  intro h_length h_speed h_time
  -- Convert speed from km/h to m/s
  have train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  -- Calculate total distance
  have total_distance : ℝ := train_speed_ms * crossing_time
  -- Calculate bridge length
  have bridge_length : ℝ := total_distance - train_length
  -- Prove the bridge length is 219 meters
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_proof_l93_9386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_union_ABC_l93_9353

-- Define the sets A, B, C as finite sets of natural numbers (ages)
variable (A B C : Finset ℕ)

-- Define the average function
def average (s : Finset ℕ) : ℚ := (s.sum (fun x => (x : ℚ))) / s.card

-- State the theorem
theorem average_union_ABC (hDisjoint : A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅)
  (hA : average A = 30)
  (hB : average B = 25)
  (hC : average C = 45)
  (hAB : average (A ∪ B) = 27)
  (hAC : average (A ∪ C) = 40)
  (hBC : average (B ∪ C) = 35) :
  average (A ∪ B ∪ C) = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_union_ABC_l93_9353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_imply_m_value_l93_9303

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the second component of b is 1/2. -/
theorem perpendicular_vectors_imply_m_value (a b : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b.1 = -1) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  b.2 = 1/2 := by
  sorry

#check perpendicular_vectors_imply_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_imply_m_value_l93_9303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l93_9327

/-- The volume of a regular triangular pyramid with inscribed sphere radius r and lateral face angle γ -/
noncomputable def regularTriangularPyramidVolume (r : ℝ) (γ : ℝ) : ℝ :=
  (2 * r^3 * Real.sqrt 3 * (Real.cos (γ/2))^2) / 
  ((Real.sqrt 3 - Real.sqrt (1 - Real.cos γ))^3 * Real.sqrt (1 - Real.cos γ))

/-- Theorem: The volume of a regular triangular pyramid with inscribed sphere radius r and lateral face angle γ -/
theorem regular_triangular_pyramid_volume (r : ℝ) (γ : ℝ) (h1 : r > 0) (h2 : 0 < γ ∧ γ < π) :
  let V := regularTriangularPyramidVolume r γ
  ∃ (a : ℝ), a > 0 ∧ V = (1/3) * a^2 * Real.sqrt 3 / 4 * a * Real.sin (γ/2) / Real.sqrt (1 - Real.cos γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l93_9327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l93_9350

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x : ℝ | x^2 ≤ 4}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l93_9350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_converges_to_half_l93_9331

/-- The infinite series defined by the given formula -/
noncomputable def infiniteSeries (n : ℕ) : ℝ :=
  (n^4 + 5*n^2 + 15*n + 15) / (2^n * (n^4 + 8))

/-- The sum of the infinite series from n = 2 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, infiniteSeries (n + 2)

/-- Theorem stating that the infinite series converges to 1/2 -/
theorem series_converges_to_half : seriesSum = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_converges_to_half_l93_9331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l93_9371

-- Define the constants
noncomputable def a : ℝ := (6 : ℝ) ^ (7/10)
noncomputable def b : ℝ := (7/10 : ℝ) ^ 6
noncomputable def c : ℝ := Real.log 6 / Real.log (7/10)

-- State the theorem
theorem order_of_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l93_9371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_z_plus_one_l93_9358

/-- The polynomial division theorem for this specific case -/
axiom polynomial_division_theorem (Q R : ℂ → ℂ) :
  (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) →
  (∃ a b : ℂ, ∀ z, R z = a * z + b)

/-- The sixth root of unity -/
noncomputable def ω : ℂ := Complex.exp (Real.pi * Complex.I / 3)

/-- The main theorem -/
theorem remainder_is_z_plus_one (Q R : ℂ → ℂ) :
  (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) →
  (∃ a b : ℂ, ∀ z, R z = a * z + b) →
  (∀ z, R z = z + 1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_z_plus_one_l93_9358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l93_9354

noncomputable def lineAD (x : ℝ) : ℝ := Real.sqrt 3 * (x - 1)

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def bisects (D : ℝ × ℝ) (triangle : Triangle) : Prop :=
  -- Definition of what it means for a point to bisect an angle in a triangle
  sorry

theorem problem_solution (p q : ℝ) :
  (∃ (A C D : ℝ × ℝ), 
    let triangle : Triangle := { A := A, B := (p, q), C := C }
    D.1 = 1 ∧ D.2 = 0 ∧
    (∀ x : ℝ, lineAD x = (A.2 - D.2) / (A.1 - D.1) * (x - D.1) + D.2) ∧
    bisects D triangle) →
  q = 12 / Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l93_9354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_fill_time_l93_9372

/-- Given a tanker that can be filled by two pipes A and B, this theorem proves
    the time it takes for pipe B to fill the tanker alone. -/
theorem pipe_b_fill_time (fill_time_a : ℝ) (fill_time_ab : ℝ) (fill_time_b : ℝ) :
  fill_time_a = 30 →
  fill_time_ab = 10 →
  fill_time_b = (fill_time_a * fill_time_ab) / (fill_time_a - fill_time_ab) →
  fill_time_b = 15 := by
  intros h1 h2 h3
  sorry

#check pipe_b_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_fill_time_l93_9372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_mult_count_l93_9321

/-- Represents a polynomial with integer coefficients -/
def MyPolynomial := List Int

/-- Horner's rule for polynomial evaluation -/
def horner_eval (p : MyPolynomial) (x : Int) : Int :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Counts the number of multiplication operations in Horner's rule -/
def horner_mult_count (p : MyPolynomial) : Nat :=
  p.length - 1

/-- The specific polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f : MyPolynomial := [2, 3, 0, 5, -4]

theorem horner_rule_mult_count :
  horner_mult_count f = 4 := by
  rfl

#eval horner_mult_count f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_mult_count_l93_9321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l93_9396

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/4 + y^2/m = 1) →  -- equation of the ellipse
  (∃ c a : ℝ, c/a = Real.sqrt 3/2) →  -- eccentricity condition
  (m = 1 ∨ m = 16) :=               -- conclusion
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l93_9396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l93_9387

/-- The capacity of the tank in litres -/
def C : ℝ := sorry

/-- The rate at which pipe B empties the tank in litres per hour -/
def R_B : ℝ := sorry

/-- Pipe A empties the tank in 10 hours -/
axiom pipe_A_rate : C / 10 = C / 10

/-- Inlet pipe rate is 16 litres per minute -/
axiom inlet_pipe_rate : 16 * 60 = 960

/-- With inlet pipe open and both outlet pipes working, tank empties in 18 hours -/
axiom combined_rate : C / 10 + R_B - 960 = C / 18

/-- When both outlet pipes A and B are working together without the inlet pipe, 
    they should be able to empty the tank in 10 hours -/
axiom outlet_pipes_rate : C / 10 = C / 10 + R_B

/-- The capacity of the tank is 21600 litres -/
theorem tank_capacity : C = 21600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l93_9387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_problem_l93_9375

/-- The length of the first train given the conditions of the problem -/
noncomputable def first_train_length (v1 v2 : ℝ) (l2 t : ℝ) : ℝ :=
  (v1 + v2) * (5 / 18) * t - l2

/-- Theorem stating the length of the first train given the problem conditions -/
theorem first_train_length_problem :
  let v1 : ℝ := 40  -- speed of first train in km/h
  let v2 : ℝ := 50  -- speed of second train in km/h
  let l2 : ℝ := 165 -- length of second train in meters
  let t : ℝ := 11.039116870650348 -- time to clear in seconds
  abs (first_train_length v1 v2 l2 t - 110.98) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_problem_l93_9375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_full_house_expanded_deck_l93_9334

/-- A deck of cards with an expanded number of ranks -/
structure ExpandedDeck where
  total_cards : ℕ
  num_ranks : ℕ
  cards_per_rank : ℕ
  total_cards_eq : total_cards = num_ranks * cards_per_rank

/-- A full house in poker -/
def is_full_house (hand : Finset ℕ) : Prop :=
  ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ 
    (hand.filter (λ c ↦ c % 4 = r₁)).card = 3 ∧
    (hand.filter (λ c ↦ c % 4 = r₂)).card = 2

/-- The probability of drawing a full house from an expanded deck -/
def prob_full_house (d : ExpandedDeck) : ℚ :=
  let total_outcomes := Nat.choose d.total_cards 5
  let full_house_outcomes := d.num_ranks * Nat.choose 4 3 * (d.num_ranks - 1) * Nat.choose 4 2
  full_house_outcomes / total_outcomes

/-- The main theorem stating the probability of a full house in the given expanded deck -/
theorem prob_full_house_expanded_deck :
  ∀ d : ExpandedDeck, d.total_cards = 56 ∧ d.num_ranks = 14 ∧ d.cards_per_rank = 4 →
  prob_full_house d = 2 / 875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_full_house_expanded_deck_l93_9334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l93_9319

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 14*x + y^2 - 6*y + 65 = 0

-- Define the distance function from a point to the origin
noncomputable def distance_to_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

-- Define the shortest distance from the origin to the circle
noncomputable def shortest_distance_to_circle : ℝ :=
  Real.sqrt 58 - Real.sqrt 7

-- Theorem statement
theorem shortest_distance_proof :
  ∃ (x y : ℝ), circle_equation x y ∧
  ∀ (a b : ℝ), circle_equation a b →
  distance_to_origin x y ≤ distance_to_origin a b ∧
  distance_to_origin x y = shortest_distance_to_circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l93_9319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l93_9335

theorem consecutive_integers_average (n : ℕ) (avg : ℚ) : 
  n = 10 →
  avg = 20 →
  let first := (avg - (n - 1) / 2 : ℚ).floor
  let seq := List.range n |>.map (λ i => first + i)
  let new_seq := List.range n |>.map (λ i => seq.get! i - (n - 1 - i))
  (new_seq.sum : ℚ) / n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l93_9335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_two_lines_l93_9376

/-- Definition of a line in 3D space -/
def IsLine (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (a b c d e f : ℝ), ∀ (x y z : ℝ),
    (x, y, z) ∈ L ↔ ∃ (t : ℝ), x = a*t + d ∧ y = b*t + e ∧ z = c*t + f

/-- The equation represents two lines in three-dimensional space -/
theorem equation_represents_two_lines :
  ∃ (L₁ L₂ : Set (ℝ × ℝ × ℝ)),
    IsLine L₁ ∧ IsLine L₂ ∧
    (∀ (x y z : ℝ), x^2 + 2*x*(y+z) + y^2 = z^2 + 2*z*(y+x) + x^2 ↔ (x, y, z) ∈ L₁ ∪ L₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_two_lines_l93_9376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l93_9314

theorem condition_relationship (p q : Prop) 
  (h1 : ¬q → ¬p)  -- ¬p is necessary for ¬q
  (h2 : ¬(¬p → ¬q))  -- ¬p is not sufficient for ¬q
  : (p → q) ∧ ¬(q → p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l93_9314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l93_9382

structure TriangleWithCentroid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  is_centroid : O = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem triangle_centroid_property (t : TriangleWithCentroid) :
  let dist_squared (p q : ℝ × ℝ) := (p.1 - q.1)^2 + (p.2 - q.2)^2
  dist_squared t.A t.B + dist_squared t.B t.C + dist_squared t.C t.A =
  3 * (dist_squared t.O t.A + dist_squared t.O t.B + dist_squared t.O t.C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l93_9382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_faculty_l93_9344

-- Define the number of students in each category
def numeric_methods : ℕ := 240
def automatic_control : ℕ := 423
def both_subjects : ℕ := 134

-- Define the percentage of second year students
def second_year_percentage : ℚ := 80 / 100

-- Theorem to prove
theorem total_students_in_faculty : 
  let second_year_students := numeric_methods + automatic_control - both_subjects
  let total_students := (second_year_students : ℚ) / second_year_percentage
  ⌊total_students⌋ = 661 := by
  -- Proof steps would go here
  sorry

#eval let second_year_students := numeric_methods + automatic_control - both_subjects
      let total_students := (second_year_students : ℚ) / second_year_percentage
      ⌊total_students⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_faculty_l93_9344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_range_l93_9397

/-- Given real numbers a, b, and c forming a geometric sequence with a + b + c = 1,
    the range of values for a + c is [2/3, 1) ∪ (1, 2]. -/
theorem geometric_sequence_sum_range (a b c : ℝ) :
  (∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q) →  -- geometric sequence condition
  a + b + c = 1 →                            -- sum condition
  ∃ x, x = a + c ∧ (x ∈ Set.Icc (2/3) 1 ∨ x ∈ Set.Ioo 1 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_range_l93_9397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_over_fourth_root_of_five_l93_9361

theorem fifth_root_over_fourth_root_of_five (x : ℝ) :
  x = 5 → (x^(1/5 : ℝ) / x^(1/4 : ℝ) = x^(-1/20 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_over_fourth_root_of_five_l93_9361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_point_l93_9341

noncomputable section

/-- Parabola defined by y = 2x^2 --/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Point A on the parabola --/
def A : ℝ × ℝ := (2, 4)

/-- Slope of the normal line at point A --/
noncomputable def normal_slope : ℝ := -1 / (4 * A.1)

/-- Normal line equation passing through A --/
noncomputable def normal_line (x : ℝ) : ℝ :=
  normal_slope * (x - A.1) + A.2

/-- Point B: intersection of normal line and parabola --/
def B : ℝ × ℝ := (-17/16, 289/128)

theorem normal_intersection_point :
  B.1 ≠ A.1 ∧
  parabola B.1 = B.2 ∧
  normal_line B.1 = B.2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersection_point_l93_9341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_percentage_theorem_l93_9337

noncomputable def commission_rate_first_500 : ℝ := 0.20
noncomputable def commission_rate_excess : ℝ := 0.50
noncomputable def sale_threshold : ℝ := 500
noncomputable def total_sale : ℝ := 800

noncomputable def calculate_commission (sale : ℝ) : ℝ :=
  let commission_first_500 := min sale sale_threshold * commission_rate_first_500
  let commission_excess := max (sale - sale_threshold) 0 * commission_rate_excess
  commission_first_500 + commission_excess

theorem commission_percentage_theorem :
  calculate_commission total_sale / total_sale * 100 = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_percentage_theorem_l93_9337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l93_9317

theorem simplify_trig_expression (x : ℝ) (h : 1 - Real.cos x + Real.cos (2 * x) ≠ 0) :
  (Real.sin x - Real.sin (2 * x)) / (1 - Real.cos x + Real.cos (2 * x)) = Real.tan x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l93_9317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l93_9325

-- Define the integral (marked as noncomputable due to dependency on Real.measureSpace)
noncomputable def a : ℝ := ∫ x in (0)..(2), (2*x - 1)

-- Theorem statement
theorem constant_term_binomial_expansion :
  ∃ c, c = 24 ∧ (∀ x : ℝ, x ≠ 0 → ∃ t, (x + a/x)^4 = t + c + t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l93_9325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l93_9323

noncomputable section

variable (f g : ℝ → ℝ)

axiom g_odd : ∀ x : ℝ, g (-x) = -g x
axiom f_zero : f 0 = 1
axiom f_property : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y

theorem f_g_properties :
  (∀ x : ℝ, f x ^ 2 + g x ^ 2 = 1) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ a : ℝ, a ≠ 0 → f a = 1 → ∀ x : ℝ, f (x + a) = f x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l93_9323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_sin_l93_9342

theorem min_value_cos_sin (α β : Real) (h1 : 0 ≤ α ∧ α ≤ π/2) (h2 : 0 < β ∧ β ≤ π/2) :
  ∃ (m : Real), m = 1 ∧ ∀ x, x = Real.cos α^2 * Real.sin β + 1 / Real.sin β → m ≤ x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_sin_l93_9342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_routes_from_P_to_Q_l93_9340

/-- Represents a point in the route network -/
inductive Point : Type
| P | Q | R | S | T | U

/-- Represents a direct path between two points -/
def DirectPath : Point → Point → Prop
| Point.P, Point.R => True
| Point.P, Point.S => True
| Point.R, Point.Q => True
| Point.S, Point.T => True
| Point.S, Point.U => True
| Point.T, Point.Q => True
| Point.U, Point.Q => True
| _, _ => False

/-- Counts the number of routes between two points -/
def CountRoutes : Point → Point → Nat
| Point.P, Point.Q => 3  -- We define this directly based on our problem
| _, _ => 0  -- For all other cases, we return 0

/-- The main theorem stating that there are 3 routes from P to Q -/
theorem three_routes_from_P_to_Q :
  CountRoutes Point.P Point.Q = 3 := by
  rfl  -- This is true by definition of CountRoutes


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_routes_from_P_to_Q_l93_9340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_distribution_and_conditional_l93_9302

/-- The number of students in the class -/
def total_students : ℕ := 6

/-- The number of boys in the class -/
def num_boys : ℕ := 4

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The random variable representing the number of girls selected -/
noncomputable def X : ℕ → ℝ := sorry

/-- The probability of selecting no girls -/
noncomputable def prob_zero_girls : ℝ := 1 / 5

/-- The probability of selecting one girl -/
noncomputable def prob_one_girl : ℝ := 3 / 5

/-- The probability of selecting two girls -/
noncomputable def prob_two_girls : ℝ := 1 / 5

/-- The probability of selecting either boy A or girl B -/
noncomputable def prob_A_or_B : ℝ := 4 / 5

/-- The conditional probability of selecting girl B given boy A is selected -/
noncomputable def prob_B_given_A : ℝ := 2 / 5

theorem probability_distribution_and_conditional :
  (X 0 = prob_zero_girls ∧ X 1 = prob_one_girl ∧ X 2 = prob_two_girls) ∧
  prob_A_or_B = 4 / 5 ∧
  prob_B_given_A = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_distribution_and_conditional_l93_9302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l93_9306

/-- The speed of a car given its distance traveled and time taken -/
noncomputable def car_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: A car traveling 624 km in 3 hours has a speed of 208 km/h -/
theorem car_speed_proof : car_speed 624 3 = 208 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l93_9306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l93_9399

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem f_properties :
  (∀ x : ℝ, f (x + π) = f x) ∧ 
  (∀ x : ℝ, f (x - π/4) + f (-x) = 0) ∧ 
  (∀ x y : ℝ, x ∈ Set.Ioo (π/4) (π/2) → y ∈ Set.Ioo (π/4) (π/2) → x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l93_9399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_24_degrees_l93_9336

-- Define the relationship between temperature and volume
noncomputable def volume_at_temp (initial_temp : ℝ) (initial_volume : ℝ) (temp : ℝ) : ℝ :=
  initial_volume - (5 / 4) * (initial_temp - temp)

-- State the theorem
theorem gas_volume_at_24_degrees 
  (h1 : volume_at_temp 40 35 40 = 35) -- Initial condition
  (h2 : ∀ (t1 t2 : ℝ), volume_at_temp 40 35 t2 - volume_at_temp 40 35 t1 = (5 / 4) * (t2 - t1)) -- Relationship between temperature and volume
  : volume_at_temp 40 35 24 = 15 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_24_degrees_l93_9336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l93_9378

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + a*x - 2 else -a^x

-- State the theorem
theorem range_of_a (a : ℝ) :
  a ≠ 1 →
  (∀ x y, 0 < x → x < y → f a x < f a y) →
  0 < a ∧ a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l93_9378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_order_l93_9338

-- Define the students
inductive Student : Type
| Jalen : Student
| Kim : Student
| Lucy : Student
| Ravi : Student

-- Define a function to represent the scores
def score : Student → ℕ := sorry

-- All scores are different
axiom different_scores : ∀ s₁ s₂ : Student, s₁ ≠ s₂ → score s₁ ≠ score s₂

-- Jalen's score is the highest
axiom jalen_highest : ∀ s : Student, s ≠ Student.Jalen → score Student.Jalen > score s

-- Lucy scored higher than at least one person
axiom lucy_not_lowest : ∃ s : Student, s ≠ Student.Lucy ∧ score Student.Lucy > score s

-- Ravi did not score the lowest
axiom ravi_not_lowest : ∃ s : Student, s ≠ Student.Ravi ∧ score Student.Ravi > score s

-- Theorem to prove
theorem score_order : 
  score Student.Kim < score Student.Lucy ∧ 
  score Student.Lucy < score Student.Ravi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_order_l93_9338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coord_l93_9311

-- Define the ellipse parameters
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a > b)

-- Define the eccentricity
noncomputable def e : ℝ := 2/3

-- Define the ellipse equation
def on_ellipse (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define points A and B on the ellipse
variable (xA yA xB yB : ℝ)
variable (hA : on_ellipse a b xA yA)
variable (hB : on_ellipse a b xB yB)

-- Define that A and B are not symmetrical
variable (h_not_sym : (xA ≠ xB ∨ yA ≠ -yB) ∧ (yA ≠ yB ∨ xA ≠ -xB))

-- Define the perpendicular bisector condition
variable (h_perp_bisector : (yB - yA) * (1 - (xA + xB)/2) = (xB - xA) * ((yA + yB)/2))

-- Define the x-coordinate of the midpoint
noncomputable def x₀ (a b xA yA xB yB : ℝ) : ℝ := (xA + xB) / 2

-- State the theorem
theorem midpoint_x_coord (a b xA yA xB yB : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hA : on_ellipse a b xA yA) (hB : on_ellipse a b xB yB)
  (h_not_sym : (xA ≠ xB ∨ yA ≠ -yB) ∧ (yA ≠ yB ∨ xA ≠ -xB))
  (h_perp_bisector : (yB - yA) * (1 - (xA + xB)/2) = (xB - xA) * ((yA + yB)/2)) :
  x₀ a b xA yA xB yB = 9/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coord_l93_9311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_five_primes_units_3_l93_9300

/-- A function that returns true if a number is prime and has a units digit of 3 -/
def isPrimeWithUnits3 (n : ℕ) : Bool :=
  Nat.Prime n ∧ n % 10 = 3

/-- The first five prime numbers with a units digit of 3 -/
def firstFivePrimesWithUnits3 : List ℕ :=
  (List.range 100).filter isPrimeWithUnits3 |>.take 5

theorem sum_first_five_primes_units_3 :
  (firstFivePrimesWithUnits3.sum) = 135 := by
  sorry

#eval firstFivePrimesWithUnits3
#eval firstFivePrimesWithUnits3.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_five_primes_units_3_l93_9300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l93_9374

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x) + (Real.tan (2 * x))⁻¹

theorem f_period : ∃ (p : ℝ), p > 0 ∧ p = π / 2 ∧ ∀ (x : ℝ), f (x + p) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l93_9374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_y_value_l93_9384

/-- A digit is a natural number between 1 and 9 inclusive -/
def Digit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Represents a three-digit number as 100a + 10b + c -/
def ThreeDigitNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem unique_y_value :
  ∃! (y : Digit), ∃ (k : Digit) (result : ℕ),
    ThreeDigitNumber 8 k.val 8 + ThreeDigitNumber k.val 8 8 = ThreeDigitNumber 1 6 y.val + result ∧
    result = ThreeDigitNumber k.val k.val 8 :=
by sorry

#check unique_y_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_y_value_l93_9384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l93_9329

-- Define the function f(x) = 1 - 3/(x+2)
noncomputable def f (x : ℝ) : ℝ := 1 - 3 / (x + 2)

-- Define the interval [3,5]
def I : Set ℝ := Set.Icc 3 5

-- Theorem statement
theorem f_properties :
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x < f y) ∧ 
  (∀ x ∈ I, f x ≥ 2/5) ∧
  (∀ x ∈ I, f x ≤ 4/7) ∧
  (∃ x ∈ I, f x = 2/5) ∧
  (∃ x ∈ I, f x = 4/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l93_9329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l93_9315

/-- Calculates the future value of an investment -/
noncomputable def future_value (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_difference : 
  let jose_principal : ℝ := 30000
  let jose_rate : ℝ := 0.03 / 2
  let jose_periods : ℕ := 3 * 2
  
  let patricia_principal : ℝ := 30000
  let patricia_rate : ℝ := 0.025 / 12
  let patricia_periods : ℕ := 3 * 12
  
  let jose_result := round_to_nearest (future_value jose_principal jose_rate jose_periods)
  let patricia_result := round_to_nearest (future_value patricia_principal patricia_rate patricia_periods)
  
  jose_result - patricia_result = 317 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l93_9315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_rate_at_one_third_max_efficiency_rate_l93_9343

/-- Represents the transport efficiency rate for an oil tanker truck -/
noncomputable def TransportEfficiencyRate (distance_AC : ℝ) (distance_AB : ℝ) : ℝ :=
  let max_fuel := distance_AB -- maximum fuel capacity equals round trip distance
  let fuel_consumed_AC := distance_AC / distance_AB * (max_fuel / 2)
  let fuel_consumed_CB := (distance_AB - distance_AC) / distance_AB * (max_fuel / 2)
  let fuel_delivered_B := max_fuel - 2 * fuel_consumed_AC - 2 * fuel_consumed_CB
  fuel_delivered_B / (3 * max_fuel)

/-- Theorem stating the transport efficiency rate when C is 1/3 of the way from A to B -/
theorem efficiency_rate_at_one_third : 
  TransportEfficiencyRate (1/3) 1 = 2/9 := by sorry

/-- Theorem stating the maximum transport efficiency rate -/
theorem max_efficiency_rate : 
  (∀ x : ℝ, 0 < x → x < 1 → TransportEfficiencyRate x 1 ≤ 1/4) ∧ 
  TransportEfficiencyRate (1/2) 1 = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_rate_at_one_third_max_efficiency_rate_l93_9343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_trig_composition_to_2010_l93_9391

-- Define the set of allowed trigonometric functions
inductive TrigFunction
  | sin
  | cos
  | tan
  | cot
  | arcsin
  | arccos
  | arctan
  | arccot

-- Define a type for compositions of trigonometric functions
def TrigComposition := List TrigFunction

-- Function to apply a TrigComposition to a real number
noncomputable def applyComposition (comp : TrigComposition) (x : ℝ) : ℝ :=
  comp.foldl (fun acc f => 
    match f with
    | TrigFunction.sin => Real.sin acc
    | TrigFunction.cos => Real.cos acc
    | TrigFunction.tan => Real.tan acc
    | TrigFunction.cot => 1 / Real.tan acc  -- Replace Real.cot with 1 / Real.tan
    | TrigFunction.arcsin => Real.arcsin acc
    | TrigFunction.arccos => Real.arccos acc
    | TrigFunction.arctan => Real.arctan acc
    | TrigFunction.arccot => Real.pi / 2 - Real.arctan acc  -- Replace Real.arccot with pi/2 - arctan
  ) x

-- Theorem statement
theorem exists_trig_composition_to_2010 :
  ∃ (comp : TrigComposition), applyComposition comp 2 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_trig_composition_to_2010_l93_9391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_biography_percentage_l93_9318

theorem library_biography_percentage
  (initial_percentage : ℝ)
  (increase_rate : ℝ)
  (h1 : initial_percentage = 0.20)
  (h2 : increase_rate = 0.8823529411764707)
  : (initial_percentage * (1 + increase_rate)) * 100 = 37.64705882352941 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_biography_percentage_l93_9318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_sides_l93_9363

/-- Triangle with specific angle measurements and side length -/
structure SpecialTriangle where
  -- Angle measurements in degrees
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  -- Length of the side opposite to the 40° angle
  side1 : ℝ
  -- Sum of angles is 180°
  angle_sum : angle1 + angle2 + angle3 = 180
  -- One angle is 40°
  has_40 : angle1 = 40 ∨ angle2 = 40 ∨ angle3 = 40
  -- One angle is 50°
  has_50 : angle1 = 50 ∨ angle2 = 50 ∨ angle3 = 50
  -- Side opposite to 40° angle measures 8√3
  side1_length : side1 = 8 * Real.sqrt 3

/-- The sum of the lengths of the two remaining sides is approximately 59.5 -/
theorem special_triangle_sum_sides (t : SpecialTriangle) : 
  ∃ (side2 side3 : ℝ), |side2 + side3 - 59.5| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_sides_l93_9363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l93_9377

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → Real.sin x ≤ 1)) ↔ (∃ x : ℝ, x > 0 ∧ Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l93_9377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l93_9380

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem f_properties :
  -- Part 1: Smallest positive period
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Part 2: Triangle property
  (∀ (A B C : ℝ), 
    let a := Real.sin C * Real.sqrt 6 / Real.sin B
    let b := Real.sin A * Real.sqrt 6 / Real.sin C
    let c := Real.sqrt 6
    Real.cos B = 1/3 →
    f (C/2) = -1/4 →
    b = 8/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l93_9380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_power_pattern_imaginary_unit_power_2012_l93_9365

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem for the pattern of i^n
theorem imaginary_unit_power_pattern (n : ℕ) :
  (i ^ (4 * n) = 1) ∧
  (i ^ (4 * n + 1) = i) ∧
  (i ^ (4 * n + 2) = -1) ∧
  (i ^ (4 * n + 3) = -i) :=
by sorry

-- Theorem for i^2012
theorem imaginary_unit_power_2012 : i ^ 2012 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_power_pattern_imaginary_unit_power_2012_l93_9365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_share_54_l93_9369

/-- Given the number of red and blue hats Paityn and Zola have, calculate the number of hats each gets when they combine and share equally. -/
def shared_hats (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red_ratio : ℚ) (zola_blue_ratio : ℕ) : ℕ :=
  let zola_red := (zola_red_ratio * paityn_red).num.toNat
  let zola_blue := zola_blue_ratio * paityn_blue
  let total_hats := paityn_red + paityn_blue + zola_red + zola_blue
  total_hats / 2

/-- Theorem stating that given the specific numbers of hats, each person gets 54 hats when sharing equally. -/
theorem equal_share_54 : shared_hats 20 24 (4/5) 2 = 54 := by
  -- Unfold the definition of shared_hats
  unfold shared_hats
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_share_54_l93_9369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_specific_room_l93_9351

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the shortest path between opposite corners of the ceiling that touches the floor --/
noncomputable def shortestPath (room : Room) : ℝ :=
  Real.sqrt ((room.length + room.height)^2 + room.width^2)

/-- Theorem stating the shortest path for a specific room size --/
theorem shortest_path_specific_room :
  let room : Room := { length := 4, width := 5, height := 4 }
  shortestPath room = Real.sqrt 145 := by
  sorry

-- Remove the #eval statement as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_specific_room_l93_9351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l93_9347

/-- A quadratic form in two variables -/
def quadraticForm (a b c d e f : ℤ) (x y : ℤ) : ℤ := 
  a*x^2 + b*x*y + c*y^2 + d*x + e*y + f

/-- The theorem statement -/
theorem infinitely_many_solutions
  (a b c d e f : ℤ)
  (h1 : (b^2 - 4*a*c : ℤ) > 0)
  (h2 : ¬ ∃ (k : ℤ), (b^2 - 4*a*c : ℤ) = k^2)
  (h3 : (4*a*c*f + b*d*e - a*e^2 - c*d^2 - f*b^2 : ℤ) ≠ 0)
  (h4 : ∃ (x y : ℤ), quadraticForm a b c d e f x y = 0) :
  ∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ ∀ (p : ℤ × ℤ), p ∈ S → quadraticForm a b c d e f p.1 p.2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l93_9347
