import Mathlib

namespace inequality_chain_l1792_179262

theorem inequality_chain (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b := by
  sorry

end inequality_chain_l1792_179262


namespace arithmetic_sequence_sum_l1792_179252

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a₁ : ℝ  -- First term
  d : ℝ   -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Condition for symmetry of intersection points -/
def symmetricIntersectionPoints (seq : ArithmeticSequence) : Prop := sorry

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  symmetricIntersectionPoints seq →
  sumFirstNTerms seq n = -n^2 + 2*n := by sorry

end arithmetic_sequence_sum_l1792_179252


namespace common_tangent_range_l1792_179293

/-- The range of a for which y = ln x and y = ax² have a common tangent line -/
theorem common_tangent_range (a : ℝ) : 
  (a > 0 ∧ ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
    (1 / x₁ = 2 * a * x₂) ∧ 
    (Real.log x₁ - 1 = -a * x₂^2)) ↔ 
  a ≥ 1 / (2 * Real.exp 1) :=
sorry

end common_tangent_range_l1792_179293


namespace dog_cat_sum_l1792_179294

/-- Represents a three-digit number composed of digits D, O, and G -/
def DOG (D O G : Nat) : Nat := 100 * D + 10 * O + G

/-- Represents a three-digit number composed of digits C, A, and T -/
def CAT (C A T : Nat) : Nat := 100 * C + 10 * A + T

/-- Theorem stating that if DOG + CAT = 1000 for different digits, then the sum of all digits is 28 -/
theorem dog_cat_sum (D O G C A T : Nat) 
  (h1 : D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ A ∧ D ≠ T ∧ 
        O ≠ G ∧ O ≠ C ∧ O ≠ A ∧ O ≠ T ∧ 
        G ≠ C ∧ G ≠ A ∧ G ≠ T ∧ 
        C ≠ A ∧ C ≠ T ∧ 
        A ≠ T)
  (h2 : D < 10 ∧ O < 10 ∧ G < 10 ∧ C < 10 ∧ A < 10 ∧ T < 10)
  (h3 : DOG D O G + CAT C A T = 1000) :
  D + O + G + C + A + T = 28 := by
  sorry

end dog_cat_sum_l1792_179294


namespace base_thirteen_unique_l1792_179297

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Theorem stating that 13 is the unique base for which the equation holds -/
theorem base_thirteen_unique :
  ∃! b : Nat, b > 1 ∧ 
    toDecimal [5, 3, 2, 4] b + toDecimal [6, 4, 7, 3] b = toDecimal [1, 2, 5, 3, 2] b :=
by sorry

end base_thirteen_unique_l1792_179297


namespace total_tuition_correct_l1792_179292

/-- The total tuition fee that Bran needs to pay -/
def total_tuition : ℝ := 90

/-- Bran's monthly earnings from his part-time job -/
def monthly_earnings : ℝ := 15

/-- The percentage of tuition covered by Bran's scholarship -/
def scholarship_percentage : ℝ := 0.3

/-- The number of months Bran has to pay his tuition -/
def payment_period : ℕ := 3

/-- The amount Bran still needs to pay after scholarship and earnings -/
def remaining_payment : ℝ := 18

/-- Theorem stating that the total tuition is correct given the conditions -/
theorem total_tuition_correct :
  (1 - scholarship_percentage) * total_tuition - 
  (monthly_earnings * payment_period) = remaining_payment :=
by sorry

end total_tuition_correct_l1792_179292


namespace quadratic_extrema_l1792_179221

-- Define the function
def f (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

theorem quadratic_extrema :
  ∃ (min max : ℝ), 
    (∀ x ∈ domain, f x ≥ min) ∧
    (∃ x ∈ domain, f x = min) ∧
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    min = -1 ∧ max = 3 := by
  sorry

end quadratic_extrema_l1792_179221


namespace share_face_value_l1792_179206

/-- Given shares with a certain dividend rate and market value, 
    calculate the face value that yields a desired interest rate. -/
theorem share_face_value 
  (dividend_rate : ℚ) 
  (desired_interest_rate : ℚ) 
  (market_value : ℚ) 
  (h1 : dividend_rate = 9 / 100)
  (h2 : desired_interest_rate = 12 / 100)
  (h3 : market_value = 45) : 
  ∃ (face_value : ℚ), 
    face_value * dividend_rate = market_value * desired_interest_rate ∧ 
    face_value = 60 := by
  sorry

#check share_face_value

end share_face_value_l1792_179206


namespace f_neg_two_l1792_179282

def f (x : ℝ) : ℝ := x^2 + 3*x - 5

theorem f_neg_two : f (-2) = -7 := by
  sorry

end f_neg_two_l1792_179282


namespace largest_constant_inequality_l1792_179266

theorem largest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C*(x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end largest_constant_inequality_l1792_179266


namespace sequence_sum_l1792_179218

theorem sequence_sum (A B C D E F G H I J : ℤ) : 
  E = 8 ∧ 
  A + B + C = 27 ∧ 
  B + C + D = 27 ∧ 
  C + D + E = 27 ∧ 
  D + E + F = 27 ∧ 
  E + F + G = 27 ∧ 
  F + G + H = 27 ∧ 
  G + H + I = 27 ∧ 
  H + I + J = 27 
  → A + J = -27 := by
sorry

end sequence_sum_l1792_179218


namespace intersection_of_A_and_B_l1792_179281

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 3}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {(2, 5)} := by sorry

end intersection_of_A_and_B_l1792_179281


namespace quadratic_root_property_l1792_179256

theorem quadratic_root_property (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 → β^2 - 3*β - 4 = 0 → α^2 + α*β - 3*α = 0 := by
  sorry

end quadratic_root_property_l1792_179256


namespace meeting_time_proof_l1792_179201

/-- 
Given two people traveling towards each other on a 600 km route, 
one at 70 km/hr and the other at 80 km/hr, prove that they meet 
after traveling for 4 hours.
-/
theorem meeting_time_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (t : ℝ) : 
  total_distance = 600 →
  speed1 = 70 →
  speed2 = 80 →
  speed1 * t + speed2 * t = total_distance →
  t = 4 := by
sorry

end meeting_time_proof_l1792_179201


namespace fantasia_license_plates_l1792_179251

/-- Represents the number of available letters in the alphabet. -/
def num_letters : ℕ := 26

/-- Represents the number of available digits. -/
def num_digits : ℕ := 10

/-- Calculates the number of valid license plates in Fantasia. -/
def count_license_plates : ℕ :=
  num_letters * num_letters * num_letters * num_digits * (num_digits - 1) * (num_digits - 2)

/-- Theorem stating that the number of valid license plates in Fantasia is 15,818,400. -/
theorem fantasia_license_plates :
  count_license_plates = 15818400 :=
by sorry

end fantasia_license_plates_l1792_179251


namespace f_at_two_l1792_179240

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom monotonic_increasing : Monotone f
axiom functional_equation : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1

-- State the theorem
theorem f_at_two (f : ℝ → ℝ) (h1 : Monotone f) (h2 : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  f 2 = Real.exp 2 + 1 := by
  sorry

end f_at_two_l1792_179240


namespace equation_solutions_l1792_179287

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = -9 ∧ x₁^2 + 12*x₁ + 27 = 0 ∧ x₂^2 + 12*x₂ + 27 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = (-5 + Real.sqrt 10) / 3 ∧ x₂ = (-5 - Real.sqrt 10) / 3 ∧
    3*x₁^2 + 10*x₁ + 5 = 0 ∧ 3*x₂^2 + 10*x₂ + 5 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧ 3*x₁*(x₁ - 1) = 2 - 2*x₁ ∧ 3*x₂*(x₂ - 1) = 2 - 2*x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4/3 ∧ x₂ = 2/3 ∧ (3*x₁ + 1)^2 - 9 = 0 ∧ (3*x₂ + 1)^2 - 9 = 0) :=
by sorry

end equation_solutions_l1792_179287


namespace total_rowing_campers_l1792_179237

def morning_rowing : ℕ := 13
def afternoon_rowing : ℕ := 21
def morning_hiking : ℕ := 59

theorem total_rowing_campers :
  morning_rowing + afternoon_rowing = 34 := by sorry

end total_rowing_campers_l1792_179237


namespace relative_errors_equal_l1792_179269

theorem relative_errors_equal (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 25)
  (h2 : length2 = 150)
  (h3 : error1 = 0.05)
  (h4 : error2 = 0.3) : 
  (error1 / length1) = (error2 / length2) := by
  sorry

end relative_errors_equal_l1792_179269


namespace sum_maximum_l1792_179288

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  relation : 8 * a 5 = 13 * a 11

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map seq.a |>.sum

/-- The theorem stating when the sum reaches its maximum -/
theorem sum_maximum (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), sum_n seq n ≥ sum_n seq m ∧ n = 20 := by
  sorry

end sum_maximum_l1792_179288


namespace x_percentage_of_y_pay_l1792_179264

/-- The percentage of Y's pay that X is paid, given the total pay and Y's pay -/
theorem x_percentage_of_y_pay (total_pay y_pay : ℝ) (h1 : total_pay = 700) (h2 : y_pay = 318.1818181818182) :
  (total_pay - y_pay) / y_pay * 100 = 120 := by
  sorry

end x_percentage_of_y_pay_l1792_179264


namespace foreign_exchange_earnings_equation_l1792_179298

/-- Represents the monthly decline rate as a real number between 0 and 1 -/
def monthly_decline_rate : ℝ := sorry

/-- Initial foreign exchange earnings in July (in millions of USD) -/
def initial_earnings : ℝ := 200

/-- Foreign exchange earnings in September (in millions of USD) -/
def final_earnings : ℝ := 98

/-- The number of months between July and September -/
def months_elapsed : ℕ := 2

theorem foreign_exchange_earnings_equation :
  initial_earnings * (1 - monthly_decline_rate) ^ months_elapsed = final_earnings :=
sorry

end foreign_exchange_earnings_equation_l1792_179298


namespace parallel_tangents_condition_l1792_179296

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2 - 1
def curve2 (x : ℝ) : ℝ := 1 - x^3

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 2 * x
def curve2_derivative (x : ℝ) : ℝ := -3 * x^2

-- Theorem statement
theorem parallel_tangents_condition (x₀ : ℝ) :
  curve1_derivative x₀ = curve2_derivative x₀ ↔ x₀ = 0 ∨ x₀ = -2/3 := by
  sorry

end parallel_tangents_condition_l1792_179296


namespace car_wash_goal_remaining_l1792_179283

def car_wash_fundraiser (goal : ℕ) (high_donors : ℕ) (high_donation : ℕ) (low_donors : ℕ) (low_donation : ℕ) : ℕ :=
  goal - (high_donors * high_donation + low_donors * low_donation)

theorem car_wash_goal_remaining :
  car_wash_fundraiser 150 3 10 15 5 = 45 := by sorry

end car_wash_goal_remaining_l1792_179283


namespace circle_ratio_after_increase_l1792_179289

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end circle_ratio_after_increase_l1792_179289


namespace ryan_chinese_hours_l1792_179277

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  english_more_than_chinese : ℕ

/-- Ryan's actual study schedule satisfying the given conditions -/
def ryan_schedule : StudySchedule :=
  { english_hours := 6,
    chinese_hours := 2,
    english_more_than_chinese := 4 }

/-- Theorem stating that Ryan's schedule satisfies the given conditions -/
theorem ryan_chinese_hours :
  ryan_schedule.english_hours = ryan_schedule.chinese_hours + ryan_schedule.english_more_than_chinese :=
by sorry

end ryan_chinese_hours_l1792_179277


namespace penelope_savings_l1792_179290

theorem penelope_savings (daily_savings : ℕ) (total_savings : ℕ) (savings_period : ℕ) :
  daily_savings = 24 →
  total_savings = 8760 →
  savings_period * daily_savings = total_savings →
  savings_period = 365 := by
sorry

end penelope_savings_l1792_179290


namespace polynomial_coefficient_sum_l1792_179224

theorem polynomial_coefficient_sum (a₄ a₃ a₂ a₁ a₀ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ - a₃ + a₂ - a₁ = 15 := by
sorry

end polynomial_coefficient_sum_l1792_179224


namespace marble_exchange_problem_l1792_179222

/-- Represents the marble exchange problem with Woong, Youngsoo, and Hyogeun --/
theorem marble_exchange_problem (W Y H : ℕ) : 
  (W + 2 = 20) →  -- Woong's final marbles
  (Y - 5 = 20) →  -- Youngsoo's final marbles
  (H + 3 = 20) →  -- Hyogeun's final marbles
  W = 18 :=        -- Woong's initial marbles
by sorry

end marble_exchange_problem_l1792_179222


namespace unique_pair_satisfying_conditions_l1792_179278

theorem unique_pair_satisfying_conditions :
  ∃! (a b : ℕ), 
    b > a ∧ 
    a > 1 ∧ 
    a ≤ 20 ∧ 
    b ≤ 20 ∧
    (∀ (x y : ℕ), y > x ∧ x > 1 ∧ x ≤ 20 ∧ y ≤ 20 ∧ x + y = a + b →
      ∃ (p q r s : ℕ), p ≠ r ∧ q ≠ s ∧ q > p ∧ p > 1 ∧ s > r ∧ r > 1 ∧ x * y = p * q ∧ x * y = r * s) ∧
    (∀ (p q : ℕ), q > p ∧ p > 1 ∧ a * b = p * q → a = p ∧ b = q) :=
sorry

end unique_pair_satisfying_conditions_l1792_179278


namespace zeros_order_l1792_179204

open Real

noncomputable def f (x : ℝ) := x + log x
noncomputable def g (x : ℝ) := x * log x - 1
noncomputable def h (x : ℝ) := 1 - 1/x + x/2 + x^2/3

theorem zeros_order (a b c : ℝ) 
  (ha : a > 0 ∧ f a = 0)
  (hb : b > 0 ∧ g b = 0)
  (hc : c > 0 ∧ h c = 0)
  (hf : ∀ x, x > 0 → x ≠ a → f x ≠ 0)
  (hg : ∀ x, x > 0 → x ≠ b → g x ≠ 0)
  (hh : ∀ x, x > 0 → x ≠ c → h x ≠ 0) :
  b > c ∧ c > a :=
sorry

end zeros_order_l1792_179204


namespace tileIV_in_rectangleD_l1792_179226

-- Define the structure for a tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the tiles
def tileI : Tile := ⟨3, 1, 4, 2⟩
def tileII : Tile := ⟨2, 3, 1, 5⟩
def tileIII : Tile := ⟨4, 0, 3, 1⟩
def tileIV : Tile := ⟨5, 4, 2, 0⟩

-- Define the set of all tiles
def allTiles : Set Tile := {tileI, tileII, tileIII, tileIV}

-- Define a function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) : Bool :=
  (t1.right = t2.left) ∨ (t1.left = t2.right) ∨ (t1.top = t2.bottom) ∨ (t1.bottom = t2.top)

-- Theorem: Tile IV must be placed in Rectangle D
theorem tileIV_in_rectangleD :
  ∀ (t : Tile), t ∈ allTiles → t ≠ tileIV →
    ∃ (t' : Tile), t' ∈ allTiles ∧ t' ≠ t ∧ t' ≠ tileIV ∧ canBeAdjacent t t' = true →
      ¬∃ (t'' : Tile), t'' ∈ allTiles ∧ t'' ≠ tileIV ∧ canBeAdjacent tileIV t'' = true :=
sorry

end tileIV_in_rectangleD_l1792_179226


namespace complex_number_in_first_quadrant_l1792_179279

theorem complex_number_in_first_quadrant :
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_first_quadrant_l1792_179279


namespace random_number_table_sampling_sequence_l1792_179215

/-- Represents the steps in the sampling process -/
inductive SamplingStep
  | AssignNumbers
  | ObtainSamples
  | SelectStartingNumber

/-- Represents a sequence of sampling steps -/
def SamplingSequence := List SamplingStep

/-- The correct sampling sequence -/
def correctSequence : SamplingSequence :=
  [SamplingStep.AssignNumbers, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSamples]

/-- Checks if a given sequence is valid for random number table sampling -/
def isValidSequence (seq : SamplingSequence) : Prop :=
  seq = correctSequence

theorem random_number_table_sampling_sequence :
  isValidSequence correctSequence :=
sorry

end random_number_table_sampling_sequence_l1792_179215


namespace sufficient_not_necessary_l1792_179249

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 3 → ∃ x : ℝ, x^2 + a*x + 1 < 0) ∧
  (∃ a, (∃ x : ℝ, x^2 + a*x + 1 < 0) ∧ ¬(a > 3)) :=
sorry

end sufficient_not_necessary_l1792_179249


namespace muffin_count_arthur_muffins_l1792_179213

/-- The total number of muffins Arthur wants to have -/
def total_muffins (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the total number of muffins is the sum of initial and additional muffins -/
theorem muffin_count (initial : ℕ) (additional : ℕ) :
  total_muffins initial additional = initial + additional :=
by sorry

/-- Theorem proving the specific case in the problem -/
theorem arthur_muffins :
  total_muffins 35 48 = 83 :=
by sorry

end muffin_count_arthur_muffins_l1792_179213


namespace winter_clothes_cost_theorem_l1792_179295

/-- Represents the cost calculation for winter clothes with a discount --/
def winter_clothes_cost (total_children : ℕ) (toddlers : ℕ) 
  (toddler_cost school_cost preteen_cost teen_cost : ℕ) 
  (discount_percent : ℕ) : ℕ :=
  let school_age := 2 * toddlers
  let preteens := school_age / 2
  let teens := 4 * toddlers + toddlers
  let total_cost := toddler_cost * toddlers + 
                    school_cost * school_age + 
                    preteen_cost * preteens + 
                    teen_cost * teens
  let discount := preteen_cost * preteens * discount_percent / 100
  total_cost - discount

/-- Theorem stating the total cost of winter clothes after discount --/
theorem winter_clothes_cost_theorem :
  winter_clothes_cost 60 6 35 45 55 65 30 = 2931 := by
  sorry

#eval winter_clothes_cost 60 6 35 45 55 65 30

end winter_clothes_cost_theorem_l1792_179295


namespace sum_of_X_and_Y_l1792_179205

/-- X is defined as 2 groups of 10 plus 6 units -/
def X : ℕ := 2 * 10 + 6

/-- Y is defined as 4 groups of 10 plus 1 unit -/
def Y : ℕ := 4 * 10 + 1

/-- The sum of X and Y is 67 -/
theorem sum_of_X_and_Y : X + Y = 67 := by
  sorry

end sum_of_X_and_Y_l1792_179205


namespace symmetry_implies_p_plus_r_zero_l1792_179273

/-- Represents a curve of the form y = (px + 2q) / (rx + 2s) -/
structure Curve where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  r_nonzero : r ≠ 0
  s_nonzero : s ≠ 0

/-- The property of y = 2x being an axis of symmetry for the curve -/
def is_axis_of_symmetry (c : Curve) : Prop :=
  ∀ x y : ℝ, y = (c.p * x + 2 * c.q) / (c.r * x + 2 * c.s) →
    y = (c.p * (y / 2) + 2 * c.q) / (c.r * (y / 2) + 2 * c.s)

/-- The main theorem stating that if y = 2x is an axis of symmetry, then p + r = 0 -/
theorem symmetry_implies_p_plus_r_zero (c : Curve) :
  is_axis_of_symmetry c → c.p + c.r = 0 := by sorry

end symmetry_implies_p_plus_r_zero_l1792_179273


namespace rational_equation_result_l1792_179280

theorem rational_equation_result (x y : ℚ) 
  (h : |x + 2017| + (y - 2017)^2 = 0) : 
  (x / y)^2017 = -1 := by
  sorry

end rational_equation_result_l1792_179280


namespace fence_perimeter_is_262_l1792_179275

/-- Calculates the outer perimeter of a rectangular fence with given specifications -/
def calculate_fence_perimeter (total_posts : ℕ) (post_width : ℚ) (post_spacing : ℕ) 
  (aspect_ratio : ℚ) : ℚ :=
  let width_posts := total_posts / (3 : ℕ)
  let length_posts := 2 * width_posts
  let width := (width_posts - 1) * post_spacing + width_posts * post_width
  let length := (length_posts - 1) * post_spacing + length_posts * post_width
  2 * (width + length)

/-- The outer perimeter of the fence with given specifications is 262 feet -/
theorem fence_perimeter_is_262 : 
  calculate_fence_perimeter 32 (1/2) 6 2 = 262 := by
  sorry

end fence_perimeter_is_262_l1792_179275


namespace abs_eq_neg_implies_nonpositive_l1792_179235

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end abs_eq_neg_implies_nonpositive_l1792_179235


namespace sum_with_divisibility_conditions_l1792_179241

theorem sum_with_divisibility_conditions : 
  ∃ (a b : ℕ), a + b = 316 ∧ a % 13 = 0 ∧ b % 11 = 0 := by
sorry

end sum_with_divisibility_conditions_l1792_179241


namespace remaining_numbers_count_l1792_179286

theorem remaining_numbers_count (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 5 ∧ 
  total_avg = 8 ∧ 
  subset = 3 ∧ 
  subset_avg = 4 ∧ 
  remaining_avg = 14 →
  (total - subset = 2 ∧ 
   (total * total_avg - subset * subset_avg) / (total - subset) = remaining_avg) :=
by sorry

end remaining_numbers_count_l1792_179286


namespace unique_pair_solution_l1792_179225

theorem unique_pair_solution : 
  ∃! (p n : ℕ), 
    n > p ∧ 
    p.Prime ∧ 
    (∃ k : ℕ, k > 0 ∧ n^(n - p) = k^n) ∧ 
    p = 2 ∧ 
    n = 4 := by
sorry

end unique_pair_solution_l1792_179225


namespace train_distance_l1792_179220

theorem train_distance (v_ab v_ba : ℝ) (t_diff : ℝ) (h1 : v_ab = 160)
    (h2 : v_ba = 120) (h3 : t_diff = 1) : ∃ D : ℝ,
  D / v_ba = D / v_ab + t_diff ∧ D = 480 := by
  sorry

end train_distance_l1792_179220


namespace least_integer_greater_than_sqrt_500_l1792_179239

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n := by
  sorry

end least_integer_greater_than_sqrt_500_l1792_179239


namespace forty_second_card_l1792_179272

def card_sequence : ℕ → ℕ :=
  fun n => (n - 1) % 13 + 1

theorem forty_second_card :
  card_sequence 42 = 3 := by
  sorry

end forty_second_card_l1792_179272


namespace vector_sum_zero_l1792_179250

variable {V : Type*} [AddCommGroup V]

def vector (A B : V) : V := B - A

theorem vector_sum_zero (M B O A C D : V) : 
  (vector M B + vector B O + vector O M = 0) ∧
  (vector O B + vector O C + vector B O + vector C O = 0) ∧
  (vector A B - vector A C + vector B D - vector C D = 0) := by
  sorry

end vector_sum_zero_l1792_179250


namespace boat_trip_theorem_l1792_179200

/-- Represents the boat trip scenario -/
structure BoatTrip where
  total_time : ℝ
  stream_velocity : ℝ
  boat_speed : ℝ
  distance : ℝ

/-- The specific boat trip instance from the problem -/
def problem_trip : BoatTrip where
  total_time := 38
  stream_velocity := 4
  boat_speed := 14
  distance := 360

/-- Theorem stating that the given boat trip satisfies the problem conditions -/
theorem boat_trip_theorem (trip : BoatTrip) : 
  trip.total_time = 38 ∧ 
  trip.stream_velocity = 4 ∧ 
  trip.boat_speed = 14 ∧
  trip.distance / (trip.boat_speed + trip.stream_velocity) + 
    (trip.distance / 2) / (trip.boat_speed - trip.stream_velocity) = trip.total_time →
  trip.distance = 360 := by
  sorry

#check boat_trip_theorem problem_trip

end boat_trip_theorem_l1792_179200


namespace geometric_sequence_fourth_term_l1792_179209

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum : a 1 + a 2 = -1)
  (h_diff : a 1 - a 3 = -3) :
  a 4 = -8 := by
sorry

end geometric_sequence_fourth_term_l1792_179209


namespace farmers_animal_purchase_l1792_179219

/-- The farmer's animal purchase problem -/
theorem farmers_animal_purchase
  (total : ℕ) (goat_pig_sheep : ℕ) (cow_pig_sheep : ℕ) (goat_pig : ℕ)
  (h1 : total = 1325)
  (h2 : goat_pig_sheep = 425)
  (h3 : cow_pig_sheep = 1225)
  (h4 : goat_pig = 275) :
  ∃ (cow goat sheep pig : ℕ),
    cow + goat + sheep + pig = total ∧
    goat + sheep + pig = goat_pig_sheep ∧
    cow + sheep + pig = cow_pig_sheep ∧
    goat + pig = goat_pig ∧
    cow = 900 ∧ goat = 100 ∧ sheep = 150 ∧ pig = 175 := by
  sorry


end farmers_animal_purchase_l1792_179219


namespace frogs_on_lily_pads_l1792_179247

/-- Given the total number of frogs in a pond, the number of frogs on logs, and the number of baby frogs on a rock,
    calculate the number of frogs on lily pads. -/
theorem frogs_on_lily_pads (total : ℕ) (on_logs : ℕ) (on_rock : ℕ) 
    (h1 : total = 32) 
    (h2 : on_logs = 3) 
    (h3 : on_rock = 24) : 
  total - on_logs - on_rock = 5 := by
  sorry

end frogs_on_lily_pads_l1792_179247


namespace mrs_hilt_bug_count_l1792_179202

/-- The number of bugs Mrs. Hilt saw -/
def num_bugs : ℕ := 3

/-- The number of flowers each bug eats -/
def flowers_per_bug : ℕ := 2

/-- The total number of flowers eaten -/
def total_flowers : ℕ := 6

/-- Theorem: The number of bugs is correct given the conditions -/
theorem mrs_hilt_bug_count : 
  num_bugs * flowers_per_bug = total_flowers :=
by sorry

end mrs_hilt_bug_count_l1792_179202


namespace square_pattern_l1792_179254

theorem square_pattern (n : ℕ) : (n - 1) * (n + 1) + 1 = n^2 := by
  sorry

end square_pattern_l1792_179254


namespace sphere_to_cone_height_l1792_179230

/-- Given a sphere with diameter 6 cm and a cone with base diameter 12 cm,
    if their volumes are equal, then the height of the cone is 3 cm. -/
theorem sphere_to_cone_height (sphere_diameter : ℝ) (cone_base_diameter : ℝ) (cone_height : ℝ) :
  sphere_diameter = 6 →
  cone_base_diameter = 12 →
  (4 / 3) * Real.pi * (sphere_diameter / 2) ^ 3 = (1 / 3) * Real.pi * (cone_base_diameter / 2) ^ 2 * cone_height →
  cone_height = 3 := by
  sorry

#check sphere_to_cone_height

end sphere_to_cone_height_l1792_179230


namespace common_difference_is_five_l1792_179284

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

/-- Theorem: The common difference is 5 given the conditions -/
theorem common_difference_is_five (seq : ArithmeticSequence)
  (h1 : seq.S 17 = 255)
  (h2 : seq.a 10 = 20) :
  commonDifference seq = 5 := by
  sorry

#check common_difference_is_five

end common_difference_is_five_l1792_179284


namespace last_triangle_perimeter_l1792_179259

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Constructs the next triangle in the sequence if it exists -/
def nextTriangle (T : Triangle) : Option Triangle := sorry

/-- The sequence of triangles starting from T₁ -/
def triangleSequence : ℕ → Option Triangle
  | 0 => some ⟨20, 21, 29, sorry, sorry⟩
  | n + 1 => match triangleSequence n with
    | none => none
    | some T => nextTriangle T

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Finds the last valid triangle in the sequence -/
def lastTriangle : Option Triangle := sorry

theorem last_triangle_perimeter :
  ∀ T, lastTriangle = some T → perimeter T = 35 := by sorry

end last_triangle_perimeter_l1792_179259


namespace a_can_be_any_real_l1792_179216

theorem a_can_be_any_real : ∀ (a b c d e : ℝ), 
  bd ≠ 0 → e ≠ 0 → (a / b + e < -(c / d)) → 
  (∃ (a_pos a_neg a_zero : ℝ), 
    (a_pos > 0 ∧ a_pos / b + e < -(c / d)) ∧
    (a_neg < 0 ∧ a_neg / b + e < -(c / d)) ∧
    (a_zero = 0 ∧ a_zero / b + e < -(c / d))) :=
by sorry

end a_can_be_any_real_l1792_179216


namespace greatest_integer_radius_l1792_179211

theorem greatest_integer_radius (A : ℝ) (h1 : 50 * Real.pi < A) (h2 : A < 75 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ r ≤ 8 ∧ ∀ (s : ℕ), s * s * Real.pi = A → s ≤ r :=
sorry

end greatest_integer_radius_l1792_179211


namespace petyas_addition_mistake_l1792_179258

theorem petyas_addition_mistake :
  ∃ (x y : ℕ) (c : Fin 10),
    x + y = 12345 ∧
    (10 * x + c.val) + y = 44444 ∧
    x = 3566 ∧
    y = 8779 := by
  sorry

end petyas_addition_mistake_l1792_179258


namespace megan_water_consumption_l1792_179265

/-- The number of glasses of water Megan drinks in a given time period -/
def glasses_of_water (minutes : ℕ) : ℕ :=
  minutes / 20

theorem megan_water_consumption : glasses_of_water 220 = 11 := by
  sorry

end megan_water_consumption_l1792_179265


namespace orange_grape_ratio_l1792_179234

/-- Given the number of orange and grape sweets, and the number of sweets per tray,
    calculate the ratio of orange to grape sweets in each tray. -/
def sweetRatio (orange : Nat) (grape : Nat) (perTray : Nat) : Rat :=
  (orange / perTray) / (grape / perTray)

/-- Theorem stating that for 36 orange sweets and 44 grape sweets,
    when divided into trays of 4, the ratio is 9/11. -/
theorem orange_grape_ratio :
  sweetRatio 36 44 4 = 9 / 11 := by
  sorry

end orange_grape_ratio_l1792_179234


namespace sum_base4_equals_l1792_179274

/-- Converts a base 4 number represented as a list of digits to a natural number -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation as a list of digits -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem sum_base4_equals :
  base4ToNat [2, 1, 2] + base4ToNat [1, 0, 3] + base4ToNat [3, 2, 1] =
  base4ToNat [1, 0, 1, 2] := by
  sorry

end sum_base4_equals_l1792_179274


namespace werewolf_victims_l1792_179231

/-- Given a village with a certain population, a vampire's weekly victim count, 
    and a time period, calculate the werewolf's weekly victim count. -/
theorem werewolf_victims (village_population : ℕ) (vampire_victims_per_week : ℕ) (weeks : ℕ) 
  (h1 : village_population = 72)
  (h2 : vampire_victims_per_week = 3)
  (h3 : weeks = 9) :
  ∃ (werewolf_victims_per_week : ℕ), 
    werewolf_victims_per_week * weeks + vampire_victims_per_week * weeks = village_population ∧ 
    werewolf_victims_per_week = 5 :=
by sorry

end werewolf_victims_l1792_179231


namespace chili_beans_cans_l1792_179255

-- Define the ratio of tomato soup cans to chili beans cans
def soup_to_beans_ratio : ℚ := 1 / 2

-- Define the total number of cans
def total_cans : ℕ := 12

-- Theorem to prove
theorem chili_beans_cans (t c : ℕ) 
  (h1 : t + c = total_cans) 
  (h2 : c = 2 * t) : c = 8 := by
  sorry

end chili_beans_cans_l1792_179255


namespace sqrt_plus_reciprocal_sqrt_l1792_179260

theorem sqrt_plus_reciprocal_sqrt (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 98) :
  Real.sqrt x + 1 / Real.sqrt x = 10 := by
  sorry

end sqrt_plus_reciprocal_sqrt_l1792_179260


namespace jan_2022_is_saturday_l1792_179263

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

/-- Theorem: If January 2021 has exactly five Fridays, five Saturdays, and five Sundays,
    then January 1, 2022 falls on a Saturday -/
theorem jan_2022_is_saturday
  (h : ∃ (first_day : DayOfWeek),
       (advanceDay first_day 0 = DayOfWeek.Friday ∧
        advanceDay first_day 1 = DayOfWeek.Saturday ∧
        advanceDay first_day 2 = DayOfWeek.Sunday) ∧
       (∀ (n : Nat), n < 31 → 
        (advanceDay first_day n = DayOfWeek.Friday ∨
         advanceDay first_day n = DayOfWeek.Saturday ∨
         advanceDay first_day n = DayOfWeek.Sunday) →
        (advanceDay first_day (n + 7) = advanceDay first_day n))) :
  advanceDay DayOfWeek.Friday 365 = DayOfWeek.Saturday := by
  sorry


end jan_2022_is_saturday_l1792_179263


namespace complex_equation_sum_l1792_179217

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a + 2 * i) / i = b + i) : a + b = 1 := by
  sorry

end complex_equation_sum_l1792_179217


namespace rectangular_solid_edge_sum_l1792_179299

theorem rectangular_solid_edge_sum (a r : ℝ) : 
  a > 0 ∧ r > 0 →
  (a / r) * a * (a * r) = 512 →
  2 * ((a^2 / r) + (a^2 * r) + a^2) = 320 →
  4 * (a / r + a + a * r) = 56 + 12 * Real.sqrt 5 := by
  sorry

end rectangular_solid_edge_sum_l1792_179299


namespace bacteria_growth_proof_l1792_179212

/-- The increase in bacteria population given initial and final counts -/
def bacteria_increase (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating the increase in bacteria population for the given scenario -/
theorem bacteria_growth_proof (initial final : ℕ) 
  (h1 : initial = 600) 
  (h2 : final = 8917) : 
  bacteria_increase initial final = 8317 := by
  sorry

end bacteria_growth_proof_l1792_179212


namespace no_solution_sqrt_equation_l1792_179243

theorem no_solution_sqrt_equation :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → Real.sqrt (x + 1) + Real.sqrt (3 - x) < 17 :=
by sorry

end no_solution_sqrt_equation_l1792_179243


namespace square_difference_601_597_l1792_179242

theorem square_difference_601_597 : (601 : ℤ)^2 - (597 : ℤ)^2 = 4792 := by
  sorry

end square_difference_601_597_l1792_179242


namespace digit_40000_is_1_l1792_179261

/-- The sequence of digits formed by concatenating natural numbers -/
def digit_sequence : ℕ → ℕ := sorry

/-- The 40,000th digit in the sequence -/
def digit_40000 : ℕ := digit_sequence 40000

/-- Theorem: The 40,000th digit in the sequence is 1 -/
theorem digit_40000_is_1 : digit_40000 = 1 := by sorry

end digit_40000_is_1_l1792_179261


namespace largest_prime_factor_of_10201_l1792_179236

theorem largest_prime_factor_of_10201 : 
  (Nat.factors 10201).maximum? = some 37 := by sorry

end largest_prime_factor_of_10201_l1792_179236


namespace complement_B_intersect_A_l1792_179245

open Set

universe u

def U : Set ℝ := univ
def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | x > -1/2}

theorem complement_B_intersect_A :
  (U \ B) ∩ A = {x : ℝ | -1 < x ∧ x ≤ -1/2} := by sorry

end complement_B_intersect_A_l1792_179245


namespace sum_of_cubes_zero_l1792_179207

theorem sum_of_cubes_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end sum_of_cubes_zero_l1792_179207


namespace linear_function_not_in_fourth_quadrant_l1792_179268

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Theorem statement
theorem linear_function_not_in_fourth_quadrant (k : ℝ) (h1 : k ≠ 2) :
  (∀ x > 0, f k x ≥ 0) → k > 2 := by
  sorry


end linear_function_not_in_fourth_quadrant_l1792_179268


namespace probability_of_third_draw_l1792_179214

/-- Represents the outcome of a single draw -/
inductive Ball : Type
| Hui : Ball
| Zhou : Ball
| Mei : Ball
| Li : Ball

/-- Represents the result of three draws -/
structure ThreeDraw :=
  (first : Ball)
  (second : Ball)
  (third : Ball)

/-- Checks if a ThreeDraw result meets the conditions -/
def isValidDraw (draw : ThreeDraw) : Prop :=
  ((draw.first = Ball.Hui ∨ draw.first = Ball.Zhou) ∧
   (draw.second ≠ Ball.Hui ∧ draw.second ≠ Ball.Zhou)) ∨
  ((draw.first ≠ Ball.Hui ∧ draw.first ≠ Ball.Zhou) ∧
   (draw.second = Ball.Hui ∨ draw.second = Ball.Zhou)) ∧
  (draw.third = Ball.Hui ∨ draw.third = Ball.Zhou)

/-- The total number of trials in the experiment -/
def totalTrials : Nat := 16

/-- The number of successful outcomes in the experiment -/
def successfulTrials : Nat := 2

/-- Theorem stating the probability of drawing both "惠" and "州" exactly on the third draw -/
theorem probability_of_third_draw :
  (successfulTrials : ℚ) / totalTrials = 1 / 8 :=
sorry

end probability_of_third_draw_l1792_179214


namespace logarithm_equation_l1792_179291

theorem logarithm_equation : 
  (((1 - Real.log 3 / Real.log 6) ^ 2 + 
    (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) / 
   (Real.log 4 / Real.log 6)) = 1 := by
  sorry

end logarithm_equation_l1792_179291


namespace exponential_equation_solution_l1792_179267

theorem exponential_equation_solution :
  ∃ y : ℝ, (20 : ℝ)^y * 200^(3*y) = 8000^7 ∧ y = 3 :=
by sorry

end exponential_equation_solution_l1792_179267


namespace sculpture_surface_area_l1792_179208

/-- Represents the structure of the sculpture --/
structure Sculpture :=
  (num_cubes : ℕ)
  (edge_length : ℝ)
  (top_layer : ℕ)
  (middle_layer : ℕ)
  (bottom_layer : ℕ)

/-- Calculates the exposed surface area of the sculpture --/
def exposed_surface_area (s : Sculpture) : ℝ :=
  let top_area := s.top_layer * (5 * s.edge_length^2 + s.edge_length^2)
  let middle_area := s.middle_layer * s.edge_length^2 + 8 * s.edge_length^2
  let bottom_area := s.bottom_layer * s.edge_length^2
  top_area + middle_area + bottom_area

/-- The main theorem to be proved --/
theorem sculpture_surface_area :
  ∀ s : Sculpture,
    s.num_cubes = 14 ∧
    s.edge_length = 1 ∧
    s.top_layer = 1 ∧
    s.middle_layer = 4 ∧
    s.bottom_layer = 9 →
    exposed_surface_area s = 33 := by
  sorry


end sculpture_surface_area_l1792_179208


namespace inscribed_quadrilateral_sup_ratio_l1792_179223

/-- A quadrilateral inscribed in a unit circle with two parallel sides -/
structure InscribedQuadrilateral where
  /-- The difference between the lengths of the parallel sides -/
  d : ℝ
  /-- The distance from the intersection of the diagonals to the center of the circle -/
  h : ℝ
  /-- The difference d is positive -/
  d_pos : d > 0
  /-- The quadrilateral is inscribed in a unit circle -/
  h_bound : h ≤ 1

/-- The supremum of d/h for inscribed quadrilaterals is 2 -/
theorem inscribed_quadrilateral_sup_ratio :
  ∀ ε > 0, ∃ q : InscribedQuadrilateral, q.d / q.h > 2 - ε ∧ ∀ q' : InscribedQuadrilateral, q'.d / q'.h ≤ 2 :=
sorry

end inscribed_quadrilateral_sup_ratio_l1792_179223


namespace statement_A_statement_B_statement_C_statement_D_l1792_179276

-- Define the curve C
def C (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

-- Statement A
theorem statement_A (m n : ℝ) (h1 : n > m) (h2 : m > 0) :
  ¬ (∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, C m n x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∃ c : ℝ, c > 0 ∧ a^2 = b^2 + c^2)) :=
sorry

-- Statement B
theorem statement_B (n : ℝ) (h : n > 0) :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ ∀ x : ℝ, C 0 n x y1 ∧ C 0 n x y2 :=
sorry

-- Statement C
theorem statement_C (m n : ℝ) (h : m * n < 0) :
  ∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, C m n x y →
    (y - k * x) * (y + k * x) ≤ 0 ∧ k^2 = -m/n :=
sorry

-- Statement D
theorem statement_D (n : ℝ) (h : n > 0) :
  ¬ (∀ x y : ℝ, C n n x y ↔ x^2 + y^2 = n) :=
sorry

end statement_A_statement_B_statement_C_statement_D_l1792_179276


namespace equation_solution_sum_l1792_179253

theorem equation_solution_sum : ∃ x₁ x₂ : ℝ, 
  (6 * x₁) / 30 = 7 / x₁ ∧
  (6 * x₂) / 30 = 7 / x₂ ∧
  x₁ + x₂ = 0 := by
  sorry

end equation_solution_sum_l1792_179253


namespace difference_of_squares_l1792_179257

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l1792_179257


namespace max_value_quadratic_l1792_179210

theorem max_value_quadratic (x : ℝ) :
  let f : ℝ → ℝ := fun x => 10 * x - 2 * x^2
  ∃ (max_val : ℝ), max_val = 12.5 ∧ ∀ y : ℝ, f y ≤ max_val :=
by
  sorry

end max_value_quadratic_l1792_179210


namespace complex_equation_solution_l1792_179233

theorem complex_equation_solution (a : ℂ) :
  (1 + a * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I → a = 5 + Complex.I := by
  sorry

end complex_equation_solution_l1792_179233


namespace order_of_products_l1792_179285

theorem order_of_products (m n : ℝ) (hm : m < 0) (hn : -1 < n ∧ n < 0) :
  m < m * n^2 ∧ m * n^2 < m * n := by sorry

end order_of_products_l1792_179285


namespace arithmetic_calculation_l1792_179229

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end arithmetic_calculation_l1792_179229


namespace tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth_l1792_179228

theorem tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth
  (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin (π/2 + α) + Real.sin (π - α)) /
  (Real.cos (3*π/2 + α) + 2 * Real.cos (π + α)) = 1/4 := by
  sorry

end tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth_l1792_179228


namespace trigonometric_system_solution_l1792_179271

theorem trigonometric_system_solution (x y : ℝ) :
  (Real.sin x + Real.cos y = 0) ∧ (Real.sin x ^ 2 + Real.cos y ^ 2 = 1/2) →
  (∃ (k n : ℤ), 
    ((x = (-1)^(k+1) * Real.pi/6 + Real.pi * k ∧ y = Real.pi/3 + 2*Real.pi*n) ∨
     (x = (-1)^(k+1) * Real.pi/6 + Real.pi * k ∧ y = -Real.pi/3 + 2*Real.pi*n)) ∨
    ((x = (-1)^k * Real.pi/6 + Real.pi * k ∧ y = 2*Real.pi/3 + 2*Real.pi*n) ∨
     (x = (-1)^k * Real.pi/6 + Real.pi * k ∧ y = -2*Real.pi/3 + 2*Real.pi*n))) :=
by sorry

end trigonometric_system_solution_l1792_179271


namespace toys_per_rabbit_l1792_179227

-- Define the number of rabbits
def num_rabbits : ℕ := 16

-- Define the number of toys bought on Monday
def monday_toys : ℕ := 6

-- Define the number of toys bought on Wednesday
def wednesday_toys : ℕ := 2 * monday_toys

-- Define the number of toys bought on Friday
def friday_toys : ℕ := 4 * monday_toys

-- Define the number of toys bought on Saturday
def saturday_toys : ℕ := wednesday_toys / 2

-- Define the total number of toys
def total_toys : ℕ := monday_toys + wednesday_toys + friday_toys + saturday_toys

-- Theorem statement
theorem toys_per_rabbit : total_toys / num_rabbits = 3 := by
  sorry

end toys_per_rabbit_l1792_179227


namespace volume_of_specific_pyramid_l1792_179270

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)
  (shorter_diagonal : ℝ)

/-- Represents a pyramid -/
structure Pyramid :=
  (base : Quadrilateral)
  (lateral_face_angle : ℝ)

/-- Calculate the volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem volume_of_specific_pyramid :
  let base := Quadrilateral.mk 5 5 10 10 (4 * Real.sqrt 5)
  let pyr := Pyramid.mk base (π / 4)  -- 45° in radians
  pyramid_volume pyr = 500 / 9 := by sorry

end volume_of_specific_pyramid_l1792_179270


namespace intersection_of_A_and_B_l1792_179246

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 3}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1792_179246


namespace histogram_classes_l1792_179248

def max_value : ℝ := 169
def min_value : ℝ := 143
def class_interval : ℝ := 3

theorem histogram_classes : 
  ∃ (n : ℕ), n = ⌈(max_value - min_value) / class_interval⌉ ∧ n = 9 := by
  sorry

end histogram_classes_l1792_179248


namespace emerie_quarters_l1792_179232

/-- Represents the number of coins of a specific type --/
structure CoinCount where
  dimes : Nat
  nickels : Nat
  quarters : Nat

/-- The total number of coins --/
def totalCoins (c : CoinCount) : Nat :=
  c.dimes + c.nickels + c.quarters

/-- Emerie's coin count --/
def emerie : CoinCount :=
  { dimes := 7, nickels := 5, quarters := 0 }

/-- Zain's coin count --/
def zain (e : CoinCount) : CoinCount :=
  { dimes := e.dimes + 10, nickels := e.nickels + 10, quarters := e.quarters + 10 }

theorem emerie_quarters : 
  totalCoins (zain emerie) = 48 → emerie.quarters = 6 := by
  sorry

end emerie_quarters_l1792_179232


namespace no_all_power_of_five_l1792_179244

theorem no_all_power_of_five : ¬∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018, 
  ∃ k : ℕ, (a i)^2018 + a (i.succ) = 5^k := by
  sorry

end no_all_power_of_five_l1792_179244


namespace correct_stratified_sampling_l1792_179238

/-- Represents the number of students sampled from a grade -/
structure SampledStudents :=
  (freshmen : ℕ)
  (sophomores : ℕ)
  (juniors : ℕ)

/-- Calculates the stratified sample size for a grade -/
def stratifiedSampleSize (gradeTotal : ℕ) (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (gradeTotal * sampleSize) / totalStudents

/-- Theorem stating the correct stratified sampling for the given school -/
theorem correct_stratified_sampling :
  let totalStudents : ℕ := 2700
  let freshmenTotal : ℕ := 900
  let sophomoresTotal : ℕ := 1200
  let juniorsTotal : ℕ := 600
  let sampleSize : ℕ := 135
  let result : SampledStudents := {
    freshmen := stratifiedSampleSize freshmenTotal totalStudents sampleSize,
    sophomores := stratifiedSampleSize sophomoresTotal totalStudents sampleSize,
    juniors := stratifiedSampleSize juniorsTotal totalStudents sampleSize
  }
  result.freshmen = 45 ∧ result.sophomores = 60 ∧ result.juniors = 30 :=
by sorry

end correct_stratified_sampling_l1792_179238


namespace apartments_can_decrease_l1792_179203

/-- Represents a building configuration -/
structure Building where
  entrances : ℕ
  floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in a building -/
def total_apartments (b : Building) : ℕ :=
  b.entrances * b.floors * b.apartments_per_floor

/-- Represents the modifications made to a building -/
structure Modification where
  entrances_removed : ℕ
  floors_added : ℕ

/-- Applies a modification to a building -/
def apply_modification (b : Building) (m : Modification) : Building :=
  { entrances := b.entrances - m.entrances_removed,
    floors := b.floors + m.floors_added,
    apartments_per_floor := b.apartments_per_floor }

/-- Theorem: It's possible for the number of apartments to decrease after modifications -/
theorem apartments_can_decrease (initial : Building) (mod1 mod2 : Modification) :
  ∃ (final : Building),
    final = apply_modification (apply_modification initial mod1) mod2 ∧
    total_apartments final < total_apartments initial :=
  sorry


end apartments_can_decrease_l1792_179203
